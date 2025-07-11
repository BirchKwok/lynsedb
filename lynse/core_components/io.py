import struct
from functools import lru_cache, wraps
import numpy as np
import filelock
import os
import time
import mmap
from typing import Dict
import threading
from contextlib import contextmanager

# 导入NumPack适配器
from .numpack_adapter import (
    save_nnp as save_nnp_numpack,
    load_nnp as load_nnp_numpack,
    replace_arrays as replace_arrays_numpack,
    drop_arrays as drop_arrays_numpack,
    get_array_info,
    migrate_nnp_to_numpack,
    NnpFileSavingError as NumPackError
)

# 全局缓冲区大小（默认8MB）
BUFFER_SIZE = 8 * 1024 * 1024

# 全局缓冲区管理器
class BufferManager:
    def __init__(self):
        self.write_buffers: Dict[str, bytearray] = {}
        self.buffer_positions: Dict[str, int] = {}
        self.locks: Dict[str, threading.Lock] = {}
        self._cleanup_lock = threading.Lock()

    @contextmanager
    def get_buffer(self, filename: str):
        """获取指定文件的缓冲区"""
        if filename not in self.locks:
            with self._cleanup_lock:
                if filename not in self.locks:
                    self.locks[filename] = threading.Lock()
                    self.write_buffers[filename] = bytearray(BUFFER_SIZE)
                    self.buffer_positions[filename] = 0

        with self.locks[filename]:
            yield self.write_buffers[filename], self.buffer_positions[filename]

    def flush_buffer(self, filename: str, file_obj):
        """刷新指定文件的缓冲区"""
        with self.locks.get(filename, threading.Lock()):
            if filename in self.write_buffers:
                pos = self.buffer_positions[filename]
                if pos > 0:
                    file_obj.write(self.write_buffers[filename][:pos])
                    self.buffer_positions[filename] = 0

    def remove_buffer(self, filename: str):
        """移除指定文件的缓冲区"""
        with self._cleanup_lock:
            self.write_buffers.pop(filename, None)
            self.buffer_positions.pop(filename, None)
            self.locks.pop(filename, None)

# 创建全局缓冲区管理器实例
buffer_manager = BufferManager()

class NnpFileSavingError(Exception):
    pass

@lru_cache(maxsize=None)
def get_dtype(dtype_str):
    """获取numpy dtype"""
    return np.dtype(dtype_str)

class FileHandle:
    """文件句柄管理类，支持内存映射和缓冲区管理"""
    def __init__(self, filename: str, mode: str = 'r'):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.mmap = None
        self._lock = threading.Lock()

    def __enter__(self):
        self.file = open(self.filename, f"{self.mode}b")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mmap is not None:
            self.mmap.close()
        if self.file is not None:
            if self.mode == 'w':
                buffer_manager.flush_buffer(self.filename, self.file)
            self.file.close()

    def get_mmap(self, access=mmap.ACCESS_READ):
        """获取内存映射对象"""
        if self.mmap is None:
            with self._lock:
                if self.mmap is None:
                    self.mmap = mmap.mmap(self.file.fileno(), 0, access=access)
        return self.mmap

    def write(self, data: bytes):
        """写入数据，使用缓冲区"""
        with buffer_manager.get_buffer(self.filename) as (buffer, pos):
            if len(data) + pos > BUFFER_SIZE:
                self.file.write(buffer[:pos])
                self.file.write(data)
                buffer_manager.buffer_positions[self.filename] = 0
            else:
                buffer[pos:pos + len(data)] = data
                buffer_manager.buffer_positions[self.filename] = pos + len(data)

    def flush(self):
        """刷新缓冲区"""
        buffer_manager.flush_buffer(self.filename, self.file)
        self.file.flush()

def _write_array_header(f: FileHandle, name: str, rows: int, dtype_str: str, shape: int) -> int:
    """写入数组头信息"""
    try:
        name_bytes = name.encode('utf-8')
        if len(name_bytes) > 64:
            raise NnpFileSavingError("Array name too long (max 64 bytes)")

        dtype_bytes = dtype_str.encode('utf-8')
        if len(dtype_bytes) > 30:
            raise NnpFileSavingError("Dtype string too long (max 30 bytes)")

        header = struct.pack('<64sI30sI',
                           name_bytes.ljust(64, b' '),
                           rows,
                           dtype_bytes.ljust(30, b' '),
                           shape)
        f.write(header)
        return len(header)
    except UnicodeEncodeError as e:
        raise NnpFileSavingError(f"Error encoding array name or dtype: {str(e)}")

def _read_array_header(f: FileHandle) -> tuple:
    """读取数组头信息"""
    try:
        mmap_obj = f.get_mmap()
        header_start = mmap_obj.tell()

        # 读取名称
        name_bytes = mmap_obj.read(64)
        if len(name_bytes) != 64:
            raise NnpFileSavingError(f"Incomplete header: expected 64 bytes for name, got {len(name_bytes)}")

        name_bytes = name_bytes.rstrip(b' ')
        if not name_bytes:
            raise NnpFileSavingError("Empty array name")
        try:
            name = name_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise NnpFileSavingError("Invalid UTF-8 sequence in array name")

        # 读取行数
        rows_bytes = mmap_obj.read(4)
        if len(rows_bytes) != 4:
            raise NnpFileSavingError(f"Incomplete header: expected 4 bytes for rows, got {len(rows_bytes)}")
        rows = struct.unpack('<I', rows_bytes)[0]
        if rows == 0:
            raise NnpFileSavingError("Array cannot have zero rows")

        # 读取数据类型
        dtype_bytes = mmap_obj.read(30)
        if len(dtype_bytes) != 30:
            raise NnpFileSavingError(f"Incomplete header: expected 30 bytes for dtype, got {len(dtype_bytes)}")

        dtype_bytes = dtype_bytes.rstrip(b' ')
        if not dtype_bytes:
            raise NnpFileSavingError("Empty dtype string")
        try:
            dtype_str = dtype_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raise NnpFileSavingError("Invalid UTF-8 sequence in dtype string")

        try:
            get_dtype(dtype_str)
        except TypeError:
            raise NnpFileSavingError(f"Invalid numpy dtype: {dtype_str}")

        # 读取形状
        shape_bytes = mmap_obj.read(4)
        if len(shape_bytes) != 4:
            raise NnpFileSavingError(f"Incomplete header: expected 4 bytes for shape, got {len(shape_bytes)}")
        shape = struct.unpack('<I', shape_bytes)[0]
        if shape == 0:
            raise NnpFileSavingError("Array cannot have zero columns")

        return name, rows, dtype_str, shape
    except struct.error as e:
        raise NnpFileSavingError(f"Error reading array header at position {header_start}: {str(e)}")
    except Exception as e:
        raise NnpFileSavingError(f"Error reading array header at position {header_start}: {str(e)}")

# 公共API函数 - 使用NumPack作为默认后端
def save_nnp(filename, **arrays):
    """
    保存一个或多个命名数组到nnp文件。

    参数:
        filename (str or PathLike): 输出文件路径
        **arrays: 命名数组，格式为 key=value
    """
    return save_nnp_numpack(filename, **arrays)


def load_nnp(filename, array_names=None, mmap_mode=False, parallel=True, cache_size=None):
    """
    从nnp文件加载一个或多个命名数组。

    参数:
        filename (str or PathLike): 输入文件路径
        array_names (list): 要加载的数组名称列表，如果为None则加载所有数组
        mmap_mode (bool): 是否使用内存映射模式
        parallel (bool): 是否使用并行处理
        cache_size (int): 缓存大小（字节）

    返回:
        dict: 包含加载的数组的字典
    """
    return load_nnp_numpack(filename, array_names, mmap_mode, parallel, cache_size)


def replace_arrays(filename, arrays, indexes, array_name=None):
    """
    替换nnp文件中指定数组的指定索引位置的数据。

    参数:
        filename (str or PathLike): 目标文件路径
        arrays (dict or np.ndarray): 要替换的数组数据
        indexes (list): 要替换的索引列表
        array_name (str): 数组名称（如果arrays是单个数组）
    """
    return replace_arrays_numpack(filename, arrays, indexes, array_name)


def drop_arrays(filename, indexes, array_names=None):
    """
    从nnp文件中删除指定数组的指定索引位置的数据。

    参数:
        filename (str or PathLike): 目标文件路径
        indexes (list): 要删除的索引列表
        array_names (list): 要操作的数组名称列表，如果为None则操作所有数组
    """
    return drop_arrays_numpack(filename, indexes, array_names)


def save_nnp_legacy(filename, arrays, array_name=None, append=False, batch_size=1000):
    """
    保存一个或多个命名数组到nnp文件。

    Parameters:
        filename (str or PathLike): nnp文件路径
        arrays (dict or np.ndarray): 如果是字典，键为数组名称，值为numpy数组；
                                   如果是numpy数组，需要提供array_name参数
        array_name (str, optional): 当arrays为单个numpy数组时的数组名称
        append (bool): 是否追加到现有文件中的指定数组
        batch_size (int): 批量写入的行数，用于优化大数组的写入性能

    Returns:
        None
    """
    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        # 将单个数组转换为字典格式
        if isinstance(arrays, np.ndarray):
            if array_name is None:
                raise NnpFileSavingError("Must provide array_name when saving single array")
            arrays = {array_name: arrays}

        # 验证数组名称
        for name in arrays:
            if not isinstance(name, str):
                raise NnpFileSavingError(f"Array name must be string, got {type(name)}")
            if not name:
                raise NnpFileSavingError("Array name cannot be empty")
            try:
                name.encode('utf-8')
            except UnicodeEncodeError:
                raise NnpFileSavingError(f"Array name '{name}' contains invalid characters")

        if append and os.path.exists(filename):
            # 读取现有文件的元数据
            metadata = {}
            array_headers = []

            with FileHandle(filename) as f:
                try:
                    # 读取数组数量
                    count_bytes = f.get_mmap().read(4)
                    if len(count_bytes) != 4:
                        raise NnpFileSavingError("Invalid file format: incomplete array count")
                    array_count = struct.unpack('<I', count_bytes)[0]
                    if array_count == 0:
                        raise NnpFileSavingError("Invalid file format: array count is zero")

                    # 读取所有头部信息
                    for _ in range(array_count):
                        header_start = f.get_mmap().tell()
                        name, rows, dtype_str, shape = _read_array_header(f)

                        # 保存元数据
                        metadata[name] = {
                            'header_pos': header_start,
                            'data_pos': f.get_mmap().tell(),
                            'rows': rows,
                            'dtype': dtype_str,
                            'shape': shape
                        }

                        # 跳过数据部分
                        data_size = rows * shape * get_dtype(dtype_str).itemsize
                        f.get_mmap().seek(data_size, 1)

                except Exception as e:
                    raise NnpFileSavingError(f"Error reading file metadata: {str(e)}")

            # 验证要追加的数组存在于文件中
            missing = set(arrays.keys()) - set(metadata.keys())
            if missing:
                raise NnpFileSavingError(f"Arrays not found in file: {missing}")

            # 创建临时文件进行修改
            temp_filename = f"{filename}.temp"
            try:
                with FileHandle(filename) as src, FileHandle(temp_filename, 'w') as dst:
                    # 写入数组数量
                    dst.write(struct.pack('<I', array_count))

                    # 处理每个数组
                    for name, meta in metadata.items():
                        if name in arrays:
                            array = arrays[name]
                            if array.ndim == 1:
                                array = array.reshape(-1, 1)

                            if array.shape[1] != meta['shape']:
                                raise NnpFileSavingError(f"Shape mismatch for array '{name}'")
                            if str(array.dtype) != meta['dtype']:
                                raise NnpFileSavingError(f"Dtype mismatch for array '{name}'")

                            new_rows = meta['rows'] + array.shape[0]
                            if new_rows > 10000000:
                                raise NnpFileSavingError(f"Array '{name}' exceeds maximum rows limit")

                            # 写入更新后的头信息
                            _write_array_header(dst, name, new_rows, meta['dtype'], meta['shape'])

                            # 读��并写入原始数据
                            src.get_mmap().seek(meta['data_pos'])
                            remaining_size = meta['rows'] * meta['shape'] * get_dtype(meta['dtype']).itemsize

                            # 分批读取和写入原始数据
                            while remaining_size > 0:
                                chunk_size = min(remaining_size, batch_size * meta['shape'] * get_dtype(meta['dtype']).itemsize)
                                chunk = src.get_mmap().read(chunk_size)
                                dst.write(chunk)
                                remaining_size -= chunk_size

                            # 分批写入新数据
                            array_bytes = array.tobytes()
                            for i in range(0, len(array_bytes), batch_size * meta['shape'] * get_dtype(meta['dtype']).itemsize):
                                chunk = array_bytes[i:i + batch_size * meta['shape'] * get_dtype(meta['dtype']).itemsize]
                                dst.write(chunk)
                        else:
                            # 复制原有头信息和数据
                            _write_array_header(dst, name, meta['rows'], meta['dtype'], meta['shape'])
                            src.get_mmap().seek(meta['data_pos'])
                            remaining_size = meta['rows'] * meta['shape'] * get_dtype(meta['dtype']).itemsize

                            # 分批复制数据
                            while remaining_size > 0:
                                chunk_size = min(remaining_size, batch_size * meta['shape'] * get_dtype(meta['dtype']).itemsize)
                                chunk = src.get_mmap().read(chunk_size)
                                dst.write(chunk)
                                remaining_size -= chunk_size

                # 替换原文件
                os.replace(temp_filename, filename)

            except Exception as e:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise NnpFileSavingError(f"Error during file update: {str(e)}")
        else:
            # 创建新文件
            temp_filename = f"{filename}.temp"
            try:
                with FileHandle(temp_filename, 'w') as f:
                    # 写入数组数量
                    f.write(struct.pack('<I', len(arrays)))

                    # 写入每个数组的头信息和数据
                    for name, array in arrays.items():
                        if array.ndim == 1:
                            array = array.reshape(-1, 1)

                        if array.shape[0] > 10000000:
                            raise NnpFileSavingError(f"Array '{name}' exceeds maximum rows limit")

                        _write_array_header(f, name, array.shape[0], str(array.dtype), array.shape[1])

                        # 分批写入数据
                        array_bytes = array.tobytes()
                        for i in range(0, len(array_bytes), batch_size * array.shape[1] * array.dtype.itemsize):
                            chunk = array_bytes[i:i + batch_size * array.shape[1] * array.dtype.itemsize]
                            f.write(chunk)

                # 替换原文件（如果存在）
                os.replace(temp_filename, filename)

            except Exception as e:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                raise NnpFileSavingError(f"Error during file creation: {str(e)}")

def load_nnp_legacy(filename, array_names=None, mmap_mode=False, parallel=True, cache_size=None):
    """
    从nnp文件加载一个或多个命名数组。（Legacy版本）

    Parameters:
        filename (str or PathLike): nnp文件路径
        array_names (str or list, optional): 要加载的数组名称。如果为None，加载所有数组
        mmap_mode (bool): 是否使用内存映射模式加载
        parallel (bool): 是否使用并行加载（仅在不使用mmap_mode时有效）
        cache_size (int, optional): 缓存大小（字节）。如果为None，不使用缓存

    Returns:
        dict: 数组名称到numpy数组的映射
    """
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
        # 验证数组名称
        for name in array_names:
            if not isinstance(name, str):
                raise NnpFileSavingError(f"Array name must be string, got {type(name)}")
            if not name:
                raise NnpFileSavingError("Array name cannot be empty")

    # 创建缓存
    if cache_size is not None:
        from functools import lru_cache
        @lru_cache(maxsize=cache_size)
        def cached_load(name, offset, shape, dtype):
            with FileHandle(filename) as f:
                f.get_mmap().seek(offset)
                data = f.get_mmap().read(shape[0] * shape[1] * get_dtype(dtype).itemsize)
                return np.frombuffer(data, dtype=get_dtype(dtype)).reshape(shape)

    with FileHandle(filename) as f:
        try:
            # 读取文件头
            count_bytes = f.get_mmap().read(4)
            if len(count_bytes) != 4:
                raise NnpFileSavingError("Invalid file format: incomplete array count")
            array_count = struct.unpack('<I', count_bytes)[0]
            if array_count == 0:
                raise NnpFileSavingError("Invalid file format: array count is zero")

            # 验证文件大小
            f.get_mmap().seek(0, 2)  # 移动到文件末尾
            file_size = f.get_mmap().tell()
            f.get_mmap().seek(4)  # 回到数组数据开始处

            # 收集所有需要加载的数组信息
            arrays_to_load = []
            for i in range(array_count):
                try:
                    header_start = f.get_mmap().tell()
                    if header_start >= file_size:
                        raise NnpFileSavingError(f"Unexpected end of file while reading array {i+1}/{array_count}")

                    name, rows, dtype_str, shape = _read_array_header(f)
                    if array_names is None or name in array_names:
                        dtype = get_dtype(dtype_str)
                        final_shape = (rows, shape) if shape > 1 else (rows,)
                        data_pos = f.get_mmap().tell()
                        expected_data_size = rows * shape * dtype.itemsize

                        if data_pos + expected_data_size > file_size:
                            raise NnpFileSavingError(f"Incomplete data for array '{name}'")

                        arrays_to_load.append({
                            'name': name,
                            'offset': data_pos,
                            'shape': final_shape,
                            'dtype': dtype_str,
                            'size': expected_data_size
                        })
                        f.get_mmap().seek(expected_data_size, 1)
                    else:
                        # 跳过不需要的数组数据
                        f.get_mmap().seek(rows * shape * get_dtype(dtype_str).itemsize, 1)
                except (struct.error, UnicodeDecodeError) as e:
                    raise NnpFileSavingError(f"Error reading array at position {header_start}: {str(e)}")

            # 加载数组
            arrays = {}

            if mmap_mode:
                # 内存映射模式
                for info in arrays_to_load:
                    arrays[info['name']] = np.memmap(
                        filename,
                        mode='r',
                        dtype=get_dtype(info['dtype']),
                        offset=info['offset'],
                        shape=info['shape']
                    )
            elif parallel and len(arrays_to_load) > 1:
                # 并行加载模式
                import concurrent.futures

                def load_array(info):
                    if cache_size is not None:
                        return info['name'], cached_load(
                            info['name'],
                            info['offset'],
                            info['shape'],
                            info['dtype']
                        )
                    else:
                        with FileHandle(filename) as f:
                            f.get_mmap().seek(info['offset'])
                            data = f.get_mmap().read(info['size'])
                            return info['name'], np.frombuffer(
                                data,
                                dtype=get_dtype(info['dtype'])
                            ).reshape(info['shape'])

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(load_array, info)
                        for info in arrays_to_load
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        name, array = future.result()
                        arrays[name] = array
            else:
                # 串行加载模式
                for info in arrays_to_load:
                    if cache_size is not None:
                        arrays[info['name']] = cached_load(
                            info['name'],
                            info['offset'],
                            info['shape'],
                            info['dtype']
                        )
                    else:
                        f.get_mmap().seek(info['offset'])
                        data = f.get_mmap().read(info['size'])
                        arrays[info['name']] = np.frombuffer(
                            data,
                            dtype=get_dtype(info['dtype'])
                        ).reshape(info['shape'])

        except struct.error as e:
            raise NnpFileSavingError(f"Invalid file format: {str(e)}")

    if array_names is not None:
        missing = set(array_names) - set(arrays.keys())
        if missing:
            raise KeyError(f"Arrays not found in file: {missing}")

    return arrays

def replace_arrays_legacy(filename, arrays, indexes, array_name=None):
    """
    替换nnp文件中指定数组的指定索引位置的数据。（Legacy版本）

    Parameters:
        filename (str or PathLike): nnp文件路径
        arrays (dict or np.ndarray): 要替换的新数据，格式同save_nnp
        indexes (np.ndarray): 要替换的索引位置
        array_name (str, optional): 当arrays为单个numpy数组时的数组名称

    Returns:
        None
    """
    if isinstance(arrays, np.ndarray):
        if array_name is None:
            raise NnpFileSavingError("Must provide array_name when replacing single array")
        arrays = {array_name: arrays}

    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        # 读取文件元数据
        metadata = {}
        array_headers = []

        with FileHandle(filename) as f:
            try:
                # 读取数组数量
                count_bytes = f.get_mmap().read(4)
                if len(count_bytes) != 4:
                    raise NnpFileSavingError("Invalid file format: incomplete array count")
                array_count = struct.unpack('<I', count_bytes)[0]
                if array_count == 0:
                    raise NnpFileSavingError("Invalid file format: array count is zero")

                # 读取所有头部信息
                for _ in range(array_count):
                    header_start = f.get_mmap().tell()
                    header_bytes = f.get_mmap().read(102)  # 64 + 4 + 30 + 4 = 102 bytes
                    if len(header_bytes) != 102:
                        raise NnpFileSavingError(f"Incomplete header at position {header_start}")

                    # 解析头部信息
                    name_bytes = header_bytes[:64].rstrip(b' ')
                    rows_bytes = header_bytes[64:68]
                    dtype_bytes = header_bytes[68:98].rstrip(b' ')
                    shape_bytes = header_bytes[98:]

                    try:
                        name = name_bytes.decode('utf-8')
                        rows = struct.unpack('<I', rows_bytes)[0]
                        dtype_str = dtype_bytes.decode('utf-8')
                        shape = struct.unpack('<I', shape_bytes)[0]
                    except (UnicodeDecodeError, struct.error) as e:
                        raise NnpFileSavingError(f"Error parsing header at position {header_start}: {str(e)}")

                    # 保存元数据
                    metadata[name] = {
                        'header_pos': header_start,
                        'data_pos': f.get_mmap().tell(),
                        'rows': rows,
                        'dtype': dtype_str,
                        'shape': shape
                    }
                    array_headers.append({
                        'name': name,
                        'header_bytes': header_bytes,
                        'rows': rows,
                        'shape': shape,
                        'dtype_str': dtype_str,
                        'data_pos': f.get_mmap().tell()
                    })

                    # 跳过数据部分
                    data_size = rows * shape * get_dtype(dtype_str).itemsize
                    f.get_mmap().seek(data_size, 1)

            except Exception as e:
                raise NnpFileSavingError(f"Error reading file metadata: {str(e)}")

        # 验证要替换的数组存在
        missing = set(arrays.keys()) - set(metadata.keys())
        if missing:
            raise NnpFileSavingError(f"Arrays not found in file: {missing}")

        # 创建临时文件
        temp_filename = f"{filename}.temp"
        try:
            with FileHandle(filename) as src, FileHandle(temp_filename, 'w') as dst:
                # 写入数组数量
                dst.write(struct.pack('<I', array_count))

                # 处理每个数组
                for header in array_headers:
                    name = header['name']
                    if name in arrays:
                        array = arrays[name]
                        if array.ndim == 1:
                            array = array.reshape(-1, 1)

                        if array.shape[1] != header['shape']:
                            raise NnpFileSavingError(f"Shape mismatch for array '{name}'")
                        if str(array.dtype) != header['dtype_str']:
                            raise NnpFileSavingError(f"Dtype mismatch for array '{name}'")
                        if array.shape[0] != len(indexes):
                            raise NnpFileSavingError(f"Number of rows in replacement data doesn't match index count for array '{name}'")
                        if np.any(indexes >= header['rows']):
                            raise NnpFileSavingError(f"Indexes out of range for array '{name}'")

                        # 写入头信息
                        dst.write(header['header_bytes'])

                        # 读取原始数据到内存并创建可写副本
                        src.get_mmap().seek(header['data_pos'])
                        data_size = header['rows'] * header['shape'] * get_dtype(header['dtype_str']).itemsize
                        original_data = np.frombuffer(
                            src.get_mmap().read(data_size),
                            dtype=get_dtype(header['dtype_str'])
                        ).reshape(header['rows'], header['shape']).copy()  # 创建可写副本

                        # 替换指定位置的数据
                        original_data[indexes] = array

                        # 写入更新后的数据
                        dst.write(original_data.tobytes())
                    else:
                        # 复制原有头信息和数据
                        dst.write(header['header_bytes'])
                        src.get_mmap().seek(header['data_pos'])
                        data_size = header['rows'] * header['shape'] * get_dtype(header['dtype_str']).itemsize
                        data = src.get_mmap().read(data_size)
                        if len(data) != data_size:
                            raise NnpFileSavingError(f"Incomplete data for array '{name}'")
                        dst.write(data)

            # 替换原文件
            os.replace(temp_filename, filename)

        except Exception as e:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise NnpFileSavingError(f"Error during array replacement: {str(e)}")

def drop_arrays_legacy(filename, indexes, array_names=None):
    """
    从nnp文件中删除指定数组的指定索引位置的数据。（Legacy版本）

    Parameters:
        filename (str or PathLike): nnp文件路径
        indexes (np.ndarray): 要删除的索引位置
        array_names (str or list, optional): 要操作的数组名称。如果为None，对所有数组进行操作

    Returns:
        None
    """
    if isinstance(array_names, str):
        array_names = [array_names]

    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        # 读取文件元数据
        with FileHandle(filename) as f:
            try:
                array_count = struct.unpack('<I', f.get_mmap().read(4))[0]
                arrays_meta = []

                for _ in range(array_count):
                    current_pos = f.get_mmap().tell()
                    name, rows, dtype_str, shape = _read_array_header(f)
                    if array_names is None or name in array_names:
                        if np.any(indexes >= rows):
                            raise NnpFileSavingError(f"Indexes out of range for array '{name}'")
                    arrays_meta.append({
                        'name': name,
                        'rows': rows,
                        'dtype': dtype_str,
                        'shape': shape,
                        'offset': f.get_mmap().tell(),
                        'should_process': array_names is None or name in array_names
                    })
                    f.get_mmap().seek(rows * shape * get_dtype(dtype_str).itemsize, 1)
            except (struct.error, UnicodeDecodeError) as e:
                raise NnpFileSavingError(f"Error reading file metadata: {str(e)}")

        # 创建临时文件
        temp_filename = f"{filename}.temp"
        try:
            with FileHandle(filename) as src, FileHandle(temp_filename, 'w') as dst:
                # 写入数组数量
                dst.write(struct.pack('<I', array_count))

                # 处理每个数组
                for meta in arrays_meta:
                    keep_indexes = None
                    if meta['should_process']:
                        # 计算保留的索引
                        keep_indexes = np.setdiff1d(np.arange(meta['rows']), indexes)
                        new_rows = len(keep_indexes)
                    else:
                        new_rows = meta['rows']

                    # 写入新的数组
                    _write_array_header(dst, meta['name'], new_rows, meta['dtype'], meta['shape'])

                    # 复制数据
                    src.get_mmap().seek(meta['offset'])
                    if meta['should_process']:
                        # 读取原始数据
                        dtype = get_dtype(meta['dtype'])
                        row_size = meta['shape'] * dtype.itemsize
                        for idx in keep_indexes:
                            src.get_mmap().seek(meta['offset'] + idx * row_size)
                            dst.write(src.get_mmap().read(row_size))
                    else:
                        # 直接复制整个数组数据
                        data_size = meta['rows'] * meta['shape'] * get_dtype(meta['dtype']).itemsize
                        dst.write(src.get_mmap().read(data_size))

            # 替换原文件
            os.replace(temp_filename, filename)

        except Exception as e:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise NnpFileSavingError(f"Error during array deletion: {str(e)}")

#########################################################
# test functions
#########################################################
def clean_file_when_finished(filename):
    """
    A decorator that cleans up the file when the function is finished.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                os.remove(filename)
        return wrapper
    return decorator

# 测试函数
@clean_file_when_finished('test.nnp')
def test_save_and_load_multiple_arrays():
    """测试保存和加载多个数组"""
    print("\n=== 测试保存和加载多个数组 ===")

    # 创建测试数据
    arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(500, 5),
        'array3': np.random.rand(2000)  # 一维数组
    }

    print("正在保存数组...")
    save_nnp('test.nnp', arrays)

    print("正在加载所有数组...")
    loaded_arrays = load_nnp('test.nnp')
    print(f"加载的数组数量正确: {len(loaded_arrays) == len(arrays)}")
    for name, array in arrays.items():
        loaded = loaded_arrays[name]
        print(f"数组 '{name}' 载正确: {np.allclose(array, loaded)}")

    print("正在测试选择性加载...")
    selected = load_nnp('test.nnp', array_names=['array1', 'array3'])
    print(f"选择性加载正确: {len(selected) == 2 and 'array2' not in selected}")


@clean_file_when_finished('test.nnp')
def test_append_to_named_array():
    """测试追加到命名数组"""
    print("\n=== 测试追加到命名数组 ===")

    # 初始数据
    print("正在创建初始数据...")
    arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(500, 5)
    }

    print("正在保存初始数据...")
    save_nnp('test.nnp', arrays)

    print("正在验证初始数据...")
    initial_loaded = load_nnp('test.nnp')
    for name, array in arrays.items():
        loaded = initial_loaded[name]
        print(f"初始数组 '{name}' 正确: {np.allclose(array, loaded)}")
        print(f"数组 '{name}' 的形状: {array.shape}")
        print(f"数组 '{name}' 的类型: {array.dtype}")

    # 追加数据
    print("\n正在准备追加数据...")
    append_data = {
        'array1': np.random.rand(500, 10)
    }
    print(f"追加数据的形状: {append_data['array1'].shape}")
    print(f"追加数据的类型: {append_data['array1'].dtype}")

    print("正在追加数据...")
    save_nnp('test.nnp', append_data, append=True)

    print("正在验证追加结果...")
    loaded = load_nnp('test.nnp')
    print(f"array1新行数正确: {loaded['array1'].shape[0] == 1500}")
    print(f"array2保持不变: {loaded['array2'].shape[0] == 500}")

    # 验证数据完整性
    original_part = loaded['array1'][:1000]
    appended_part = loaded['array1'][1000:]

    print(f"原始数据保持正确: {np.allclose(arrays['array1'], original_part)}")
    print(f"追加数据正确: {np.allclose(append_data['array1'], appended_part)}")
    print(f"最终数组形状: {loaded['array1'].shape}")
    print(f"最终数组类型: {loaded['array1'].dtype}")


@clean_file_when_finished('test.nnp')
def test_replace_in_named_arrays():
    """测试替换命名数组中的数据"""
    print("\n=== 测试替换命名数组中的数据 ===")

    # 初始数据
    print("正在创建初始数据...")
    arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(500, 5)
    }

    print("正在保存初始数据...")
    save_nnp('test.nnp', arrays)

    # 替换数据
    print("正在准备替换数据...")
    indexes = np.array([0, 10, 20])
    replace_data = {
        'array1': np.random.rand(3, 10)
    }

    print("正在执行替换操作...")
    replace_arrays('test.nnp', replace_data, indexes)

    print("正在验证替换结果...")
    loaded = load_nnp('test.nnp')

    # 验证替换的数据
    replaced_rows = loaded['array1'][indexes]
    print(f"替换的数据正确: {np.allclose(replace_data['array1'], replaced_rows)}")

    # 验证未替换的数据
    mask = np.ones(1000, dtype=bool)
    mask[indexes] = False
    original_data = arrays['array1'][mask]
    loaded_data = loaded['array1'][mask]
    print(f"未替换的数据保持不变: {np.allclose(original_data, loaded_data)}")

    # 验证其他数组
    print(f"其他数组保持不变: {np.allclose(arrays['array2'], loaded['array2'])}")

    print(f"最终数组形状: {loaded['array1'].shape}")
    print(f"最终数组类型: {loaded['array1'].dtype}")


@clean_file_when_finished('test.nnp')
def test_drop_from_named_arrays():
    """测试从命名数组中删除数据"""
    print("\n=== 测试从命名数组中删除数据 ===")

    # 初始数据
    print("正在创建初始数据...")
    arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(500, 5)
    }

    print("正在保存初始数据...")
    save_nnp('test.nnp', arrays)

    # 删除数据
    print("正在执行删除操作...")
    indexes = np.array([0, 10, 20])
    drop_arrays('test.nnp', indexes, array_names=['array1'])

    print("正在验证删除结果...")
    loaded = load_nnp('test.nnp')
    print(f"array1行数正确: {loaded['array1'].shape[0] == 997}")
    print(f"array2保持不变: {loaded['array2'].shape[0] == 500}")

    # 验证未被删除的数据保持正确
    keep_indexes = np.setdiff1d(np.arange(1000), indexes)
    print(f"保留的数据正确: {np.allclose(arrays['array1'][keep_indexes], loaded['array1'])}")


@clean_file_when_finished('test.nnp')
def test_mmap_mode():
    """测试内存映射模式"""
    print("\n=== 测试内存映射模式 ===")

    print("正在创建测试数据...")
    arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(500, 5)
    }

    print("正在保存数据...")
    save_nnp('test.nnp', arrays)

    print("正在使用mmap模式加载...")
    loaded = load_nnp('test.nnp', mmap_mode=True)
    print(f"是否为memmap对象: {isinstance(loaded['array1'], np.memmap)}")
    print(f"数据正确性: {np.allclose(arrays['array1'], loaded['array1'])}")


@clean_file_when_finished('test.nnp')
def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")

    # 测试无效的数组名称
    print("测试无效的数组名称...")
    try:
        save_nnp('test.nnp', {123: np.random.rand(10, 10)})
        print("错误：应该抛出异常")
    except NnpFileSavingError as e:
        print(f"正确捕获异常: {str(e)}")

    # 测试空数组名称
    print("\n测试空数组名称...")
    try:
        save_nnp('test.nnp', {'': np.random.rand(10, 10)})
        print("错误：应该抛出异常")
    except NnpFileSavingError as e:
        print(f"正确捕获异常: {str(e)}")

    # 测试超出行数限制
    print("\n测试超出行数限制...")
    try:
        save_nnp('test.nnp', {'test': np.random.rand(10000001, 10)})
        print("错误：应该抛出异常")
    except NnpFileSavingError as e:
        print(f"正确捕获异常: {str(e)}")

    # 测试追加不存在的数组
    print("\n测试追加不存在的数组...")
    save_nnp('test.nnp', {'array1': np.random.rand(10, 10)})
    try:
        save_nnp('test.nnp', {'array2': np.random.rand(10, 10)}, append=True)
        print("错误：应该抛出异常")
    except NnpFileSavingError as e:
        print(f"正确捕获异常: {str(e)}")


@clean_file_when_finished('test.nnp')
@clean_file_when_finished('test.npz')
@clean_file_when_finished('test_1.npy')
@clean_file_when_finished('test_2.npy')
def test_performance():
    """测试性能对比"""
    print("\n=== 性能测试 ===")

    # 准备不同规模的测试数据
    print("正在准备测试数据...")
    small_arrays = {
        'small1': np.random.rand(1000, 10),      # 小数组
        'small2': np.random.rand(2000, 5)        # 小数组
    }

    medium_arrays = {
        'medium1': np.random.rand(100000, 100),  # 中等数组
        'medium2': np.random.rand(200000, 50)    # 中等数组
    }

    large_arrays = {
        'large1': np.random.rand(1000000, 50),   # 大数组
        'large2': np.random.rand(2000000, 25)    # 大数组
    }

    def test_size_performance(arrays, size_label):
        """测试特定大小数据集的性能"""
        print(f"\n=== 测试{size_label}数据集性能 ===")
        total_size = sum(arr.nbytes for arr in arrays.values()) / (1024 * 1024)  # MB
        print(f"数据大小: {total_size:.2f} MB")

        # 保存测试数据
        save_nnp('test.nnp', arrays)
        np.savez('test.npz', **arrays)
        # 保存为独立的npy文件
        for i, (name, arr) in enumerate(arrays.items(), 1):
            np.save(f'test_{i}.npy', arr)

        # 测试完整读取
        print("\n完整读取测试:")

        start_time = time.time()
        _ = load_nnp('test.nnp')
        nnp_read_time = time.time() - start_time
        print(f"NNP完整读取时间: {nnp_read_time:.3f}秒")

        start_time = time.time()
        with np.load('test.npz') as data:
            _ = {name: data[name] for name in data.files}
        npz_read_time = time.time() - start_time
        print(f"NPZ完整读取时间: {npz_read_time:.3f}秒")

        start_time = time.time()
        npy_arrays = {}
        for i, name in enumerate(arrays.keys(), 1):
            npy_arrays[name] = np.load(f'test_{i}.npy')
        npy_read_time = time.time() - start_time
        print(f"NPY完整读取时间: {npy_read_time:.3f}秒")

        # 测试选择性读取
        print("\n选择性读取测试:")
        first_array_name = list(arrays.keys())[0]

        start_time = time.time()
        _ = load_nnp('test.nnp', array_names=[first_array_name])
        nnp_partial_read_time = time.time() - start_time
        print(f"NNP选择性读取时间: {nnp_partial_read_time:.3f}秒")

        start_time = time.time()
        with np.load('test.npz') as data:
            _ = data[first_array_name]
        npz_partial_read_time = time.time() - start_time
        print(f"NPZ选择性读取时间: {npz_partial_read_time:.3f}秒")

        start_time = time.time()
        _ = np.load('test_1.npy')
        npy_partial_read_time = time.time() - start_time
        print(f"NPY选择性读取时间: {npy_partial_read_time:.3f}秒")

        # 测试内存映射
        print("\n内存映射测试:")

        start_time = time.time()
        _ = load_nnp('test.nnp', mmap_mode=True)
        nnp_mmap_time = time.time() - start_time
        print(f"NNP内存映射时间: {nnp_mmap_time:.3f}秒")

        start_time = time.time()
        with np.load('test.npz', mmap_mode='r') as data:
            _ = {name: data[name] for name in data.files}
        npz_mmap_time = time.time() - start_time
        print(f"NPZ内存映射时间: {npz_mmap_time:.3f}秒")

        start_time = time.time()
        npy_mmap_arrays = {}
        for i, name in enumerate(arrays.keys(), 1):
            npy_mmap_arrays[name] = np.load(f'test_{i}.npy', mmap_mode='r')
        npy_mmap_time = time.time() - start_time
        print(f"NPY内存映射时间: {npy_mmap_time:.3f}秒")

        # 实际访问数据的性能测试
        print("\n数据访问测试:")

        # 常规加载后访问
        arrays_nnp = load_nnp('test.nnp')
        arrays_npz = np.load('test.npz')
        arrays_npy = {name: np.load(f'test_{i}.npy')
                     for i, name in enumerate(arrays.keys(), 1)}

        start_time = time.time()
        for name, arr in arrays_nnp.items():
            _ = arr.sum()
        nnp_access_time = time.time() - start_time
        print(f"NNP数据访问时间: {nnp_access_time:.3f}秒")

        start_time = time.time()
        for name in arrays_npz.files:
            _ = arrays_npz[name].sum()
        npz_access_time = time.time() - start_time
        print(f"NPZ数据访问时间: {npz_access_time:.3f}秒")

        start_time = time.time()
        for arr in arrays_npy.values():
            _ = arr.sum()
        npy_access_time = time.time() - start_time
        print(f"NPY数据访问时间: {npy_access_time:.3f}秒")

        # 内存映射模式下的访问
        arrays_nnp_mmap = load_nnp('test.nnp', mmap_mode=True)
        arrays_npz_mmap = np.load('test.npz', mmap_mode='r')
        arrays_npy_mmap = {name: np.load(f'test_{i}.npy', mmap_mode='r')
                          for i, name in enumerate(arrays.keys(), 1)}

        start_time = time.time()
        for name, arr in arrays_nnp_mmap.items():
            _ = arr.sum()
        nnp_mmap_access_time = time.time() - start_time
        print(f"NNP内存映射访问时间: {nnp_mmap_access_time:.3f}秒")

        start_time = time.time()
        for name in arrays_npz_mmap.files:
            _ = arrays_npz_mmap[name].sum()
        npz_mmap_access_time = time.time() - start_time
        print(f"NPZ内存映射访问时间: {npz_mmap_access_time:.3f}秒")

        start_time = time.time()
        for arr in arrays_npy_mmap.values():
            _ = arr.sum()
        npy_mmap_access_time = time.time() - start_time
        print(f"NPY内存映射访问时间: {npy_mmap_access_time:.3f}秒")

        return {
            'read': (nnp_read_time, npz_read_time, npy_read_time),
            'partial': (nnp_partial_read_time, npz_partial_read_time, npy_partial_read_time),
            'mmap': (nnp_mmap_time, npz_mmap_time, npy_mmap_time),
            'access': (nnp_access_time, npz_access_time, npy_access_time),
            'mmap_access': (nnp_mmap_access_time, npz_mmap_access_time, npy_mmap_access_time)
        }

    # 测试不同大小的数据集
    small_results = test_size_performance(small_arrays, "小型")
    medium_results = test_size_performance(medium_arrays, "中型")
    large_results = test_size_performance(large_arrays, "大型")

    # 总结比较
    print("\n=== 性能总结 ===")

    def print_comparison(operation, small, medium, large):
        print(f"\n{operation}性能比较:")
        print(f"小型数据 - NNP vs NPZ vs NPY: {small[0]:.3f}s vs {small[1]:.3f}s vs {small[2]:.3f}s")
        print(f"中型数据 - NNP vs NPZ vs NPY: {medium[0]:.3f}s vs {medium[1]:.3f}s vs {medium[2]:.3f}s")
        print(f"大型数据 - NNP vs NPZ vs NPY: {large[0]:.3f}s vs {large[1]:.3f}s vs {large[2]:.3f}s")

    print_comparison("完整读取", small_results['read'], medium_results['read'], large_results['read'])
    print_comparison("选择性读取", small_results['partial'], medium_results['partial'], large_results['partial'])
    print_comparison("内存映射", small_results['mmap'], medium_results['mmap'], large_results['mmap'])
    print_comparison("数据访问", small_results['access'], medium_results['access'], large_results['access'])
    print_comparison("内存映射访问", small_results['mmap_access'], medium_results['mmap_access'], large_results['mmap_access'])


@clean_file_when_finished('test.nnp')
def test_operations_after_append():
    """测试追加数据后各种操作"""
    print("\n=== 测试追加后的操作 ===")

    # 第一步：创建初始数据并保存
    print("步骤1: 创建并保存初始数据...")
    initial_arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(500, 5),
        'array3': np.random.rand(200, 3)
    }
    save_nnp('test.nnp', initial_arrays)

    # 第二步：追加数据
    print("\n步骤2: 追加数据...")
    append_arrays = {
        'array1': np.random.rand(500, 10),
        'array2': np.random.rand(300, 5)
    }
    save_nnp('test.nnp', append_arrays, append=True)

    # 第三步：验证追加��的数据
    print("\n步骤3: 验证追加后的数据...")
    loaded_after_append = load_nnp('test.nnp')
    print(f"array1新大小正确: {loaded_after_append['array1'].shape == (1500, 10)}")
    print(f"array2新大小正确: {loaded_after_append['array2'].shape == (800, 5)}")
    print(f"array3保持不变: {loaded_after_append['array3'].shape == (200, 3)}")

    # 第四步：替换部分数据
    print("\n步骤4: 测试替换操作...")
    # 替换一些原始数据和追加的数据中的行
    replace_indexes = np.array([5, 505, 1005])  # 分别从原始和追加的数据中选择
    replace_data = {
        'array1': np.random.rand(3, 10)
    }
    replace_arrays('test.nnp', replace_data, replace_indexes)

    # 验证替换结果
    loaded_after_replace = load_nnp('test.nnp')
    print(f"替换后总行数保持不变: {loaded_after_replace['array1'].shape == (1500, 10)}")
    print(f"替换的数据正确: {np.allclose(loaded_after_replace['array1'][replace_indexes], replace_data['array1'])}")

    # 第五步：删除部分数据
    print("\n步骤5: 测试删除操作...")
    # 删除包括原始和追加数据中的行
    delete_indexes = np.array([10, 510, 1010])
    drop_arrays('test.nnp', delete_indexes, array_names=['array1'])

    # 验证删除结果
    loaded_after_drop = load_nnp('test.nnp')
    print(f"删除后行数正确: {loaded_after_drop['array1'].shape == (1497, 10)}")

    # 第六步：测试选择性加载
    print("\n步骤6: 测试选择性加载...")
    selected_arrays = load_nnp('test.nnp', array_names=['array1', 'array3'])
    print(f"选择性加载正确: {len(selected_arrays) == 2 and 'array2' not in selected_arrays}")

    # 第七步：测试内存映射模式
    print("\n步骤7: 测试内存映射模式...")
    mmap_arrays = load_nnp('test.nnp', mmap_mode=True)
    print(f"内存映射加载成功: {isinstance(mmap_arrays['array1'], np.memmap)}")

    # 第八步：再次追加并验证所有操作的累积效果
    print("\n步骤8: 测试再次追加...")
    second_append = {
        'array1': np.random.rand(200, 10),
        'array2': np.random.rand(100, 5)
    }
    save_nnp('test.nnp', second_append, append=True)

    final_arrays = load_nnp('test.nnp')
    print(f"最终array1大小正确: {final_arrays['array1'].shape == (1697, 10)}")
    print(f"最终array2大小正确: {final_arrays['array2'].shape == (900, 5)}")
    print(f"array3仍然保持不变: {final_arrays['array3'].shape == (200, 3)}")


@clean_file_when_finished('test.nnp')
def test_concurrent_array_access():
    """测试并发访问数组时的数据一致性"""
    print("\n=== 测试并发数组访问 ===")

    import threading
    import queue
    import time

    # 第一步：创建测试数据
    print("步骤1: 创建测试数据...")
    test_arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(800, 5),
        'array3': np.random.rand(600, 3),
        'array4': np.random.rand(400, 8)
    }
    save_nnp('test.nnp', test_arrays)

    # 用于存储测试结果的队列
    results_queue = queue.Queue()

    def reader_thread(array_names, thread_id):
        """读取指定数组的线程函数"""
        try:
            start_time = time.time()
            arrays = load_nnp('test.nnp', array_names=array_names)
            end_time = time.time()

            # 验证数据正确性
            is_correct = all(
                np.allclose(arrays[name], test_arrays[name])
                for name in array_names
            )

            results_queue.put({
                'thread_id': thread_id,
                'array_names': array_names,
                'success': True,
                'is_correct': is_correct,
                'time': end_time - start_time
            })
        except Exception as e:
            results_queue.put({
                'thread_id': thread_id,
                'array_names': array_names,
                'success': False,
                'error': str(e)
            })

    # 第二步：创建多个并发读取场景
    print("\n步骤2: 测试并发读取...")

    # 测试场景1：多个线程读取不同的数组
    threads1 = [
        threading.Thread(target=reader_thread, args=(['array1'], 1)),
        threading.Thread(target=reader_thread, args=(['array2'], 2)),
        threading.Thread(target=reader_thread, args=(['array3'], 3)),
        threading.Thread(target=reader_thread, args=(['array4'], 4))
    ]

    # 测试场景2：多个线程读取相同的数组
    threads2 = [
        threading.Thread(target=reader_thread, args=(['array1'], 5)),
        threading.Thread(target=reader_thread, args=(['array1'], 6)),
        threading.Thread(target=reader_thread, args=(['array1'], 7)),
        threading.Thread(target=reader_thread, args=(['array1'], 8))
    ]

    # 测试场景3：多个线程读取重叠的数组集合
    threads3 = [
        threading.Thread(target=reader_thread, args=(['array1', 'array2'], 9)),
        threading.Thread(target=reader_thread, args=(['array2', 'array3'], 10)),
        threading.Thread(target=reader_thread, args=(['array3', 'array4'], 11)),
        threading.Thread(target=reader_thread, args=(['array1', 'array4'], 12))
    ]

    # 运行所有测试场景
    all_threads = threads1 + threads2 + threads3

    print("运行12个并发线程...")
    for thread in all_threads:
        thread.start()

    for thread in all_threads:
        thread.join()

    # 收集并分析结果
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # 按线程ID排序结果
    results.sort(key=lambda x: x['thread_id'])

    # 输出结果
    print("\n步骤3: 测试结果分析...")

    # 场景1结果
    print("\n场景1 - 不同数组的并发读取:")
    for result in results[:4]:
        if result['success']:
            print(f"线程 {result['thread_id']} ({result['array_names']}): "
                  f"数据正确: {result['is_correct']}, "
                  f"耗时: {result['time']:.3f}秒")
        else:
            print(f"线程 {result['thread_id']} 失败: {result['error']}")

    # 场景2结果
    print("\n场景2 - 相同数组的并发读取:")
    for result in results[4:8]:
        if result['success']:
            print(f"线程 {result['thread_id']} ({result['array_names']}): "
                  f"数据正确: {result['is_correct']}, "
                  f"耗时: {result['time']:.3f}秒")
        else:
            print(f"线程 {result['thread_id']} 失败: {result['error']}")

    # 场景3结果
    print("\n场景3 - 重叠数组集合的并发读取:")
    for result in results[8:]:
        if result['success']:
            print(f"线程 {result['thread_id']} ({result['array_names']}): "
                  f"数据正确: {result['is_correct']}, "
                  f"耗时: {result['time']:.3f}秒")
        else:
            print(f"线程 {result['thread_id']} 失败: {result['error']}")

    # 总结
    success_count = sum(1 for r in results if r['success'])
    correct_count = sum(1 for r in results if r.get('is_correct', False))

    print("\n总结:")
    print(f"总线程数: {len(all_threads)}")
    print(f"成功线程数: {success_count}")
    print(f"数据正确的线程数: {correct_count}")
    print(f"所有数据正确: {correct_count == len(all_threads)}")


# 添加全局变量用于多进程测试
_test_arrays = None
_results_queue = None

def _reader_process(array_names, process_id, use_mmap, test_arrays, results_queue):
    """读取指定数组的进程函数"""
    try:
        start_time = time.time()
        arrays = load_nnp('test.nnp', array_names=array_names, mmap_mode=use_mmap)
        end_time = time.time()

        # 验证数据正确性
        start_time = time.time()
        is_correct = all(
            np.allclose(arrays[name], test_arrays[name])
            for name in array_names
        )
        verify_time = time.time() - start_time

        results_queue.put({
            'process_id': process_id,
            'array_names': array_names,
            'success': True,
            'is_correct': is_correct,
            'load_time': end_time - start_time,
            'verify_time': verify_time,
            'use_mmap': use_mmap
        })
    except Exception as e:
        results_queue.put({
            'process_id': process_id,
            'array_names': array_names,
            'success': False,
            'error': str(e),
            'use_mmap': use_mmap
        })


@clean_file_when_finished('test.nnp')
def test_multiprocess_array_access():
    """测试多进程访问数组时的数据一致性"""
    print("\n=== 测试多进程数组访问 ===")

    import multiprocessing as mp
    from multiprocessing import Process, Queue
    import time

    # 第一步：创建测试数据
    print("步骤1: 创建测试数据...")
    test_arrays = {
        'array1': np.random.rand(1000, 10),
        'array2': np.random.rand(800, 5),
        'array3': np.random.rand(600, 3),
        'array4': np.random.rand(400, 8)
    }
    save_nnp('test.nnp', test_arrays)

    # 用于存储测试结果的队列
    results_queue = Queue()

    # 第二步：创建多个测试场景
    print("\n步骤2: 准备测试场景...")

    # 场景1：常规模式下的多进程读取
    processes1 = [
        # 不同数组的并发读取
        Process(target=_reader_process, args=(['array1'], 1, False, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array2'], 2, False, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array3'], 3, False, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array4'], 4, False, test_arrays, results_queue)),
        # 相同数组的并发读取
        Process(target=_reader_process, args=(['array1'], 5, False, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array1'], 6, False, test_arrays, results_queue)),
        # 多数组的并发读取
        Process(target=_reader_process, args=(['array1', 'array2'], 7, False, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array3', 'array4'], 8, False, test_arrays, results_queue))
    ]

    # 场景2：内存映射模式下的多进程读取
    processes2 = [
        # 不同数组的并发读取
        Process(target=_reader_process, args=(['array1'], 9, True, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array2'], 10, True, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array3'], 11, True, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array4'], 12, True, test_arrays, results_queue)),
        # 相同数组的并发读取
        Process(target=_reader_process, args=(['array1'], 13, True, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array1'], 14, True, test_arrays, results_queue)),
        # 多数组的并发读取
        Process(target=_reader_process, args=(['array1', 'array2'], 15, True, test_arrays, results_queue)),
        Process(target=_reader_process, args=(['array3', 'array4'], 16, True, test_arrays, results_queue))
    ]

    # 运行测试场景
    for p in processes1:
        p.start()
    for p in processes1:
        p.join()

    for p in processes2:
        p.start()
    for p in processes2:
        p.join()

    # 收集并分析结果
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    # 按进程ID排序结果
    results.sort(key=lambda x: x['process_id'])

    # 输出结果
    print("\n步骤3: 测试结果分析...")

    def print_scenario_results(scenario_results, scenario_name):
        print(f"\n{scenario_name}:")
        for result in scenario_results:
            if result['success']:
                print(f"进程 {result['process_id']} ({result['array_names']}): "
                      f"数据正确: {result['is_correct']}, "
                      f"加载耗时: {result['load_time']:.3f}秒, "
                      f"验证耗时: {result['verify_time']:.3f}秒")
            else:
                print(f"进程 {result['process_id']} 失败: {result['error']}")

    # 常规模式结果
    regular_results = [r for r in results if not r['use_mmap']]
    print_scenario_results(regular_results, "常规模式结果")

    # 内存映射模式结果
    mmap_results = [r for r in results if r['use_mmap']]
    print_scenario_results(mmap_results, "内存映射模式结果")

    # 性能统计
    def calculate_stats(results_subset):
        if not results_subset:
            return 0, 0, 0, 0
        success_count = sum(1 for r in results_subset if r['success'])
        correct_count = sum(1 for r in results_subset if r.get('is_correct', False))
        if success_count == 0:
            return success_count, correct_count, 0, 0
        avg_load_time = np.mean([r['load_time'] for r in results_subset if r['success']])
        avg_verify_time = np.mean([r['verify_time'] for r in results_subset if r['success']])
        return success_count, correct_count, avg_load_time, avg_verify_time

    reg_stats = calculate_stats(regular_results)
    mmap_stats = calculate_stats(mmap_results)

    print("\n性能统计:")
    print("常规模式:")
    print(f"成功进程数: {reg_stats[0]}/{len(regular_results)}")
    print(f"数据正确的进程数: {reg_stats[1]}/{len(regular_results)}")
    print(f"平均加载时间: {reg_stats[2]:.3f}秒")
    print(f"平均验证时间: {reg_stats[3]:.3f}秒")

    print("\n内存映射模式:")
    print(f"成功进程数: {mmap_stats[0]}/{len(mmap_results)}")
    print(f"数据正确的进程数: {mmap_stats[1]}/{len(mmap_results)}")
    print(f"平均加载时间: {mmap_stats[2]:.3f}秒")
    print(f"平均验证时间: {mmap_stats[3]:.3f}秒")

    # 清理全局变量
    global _test_arrays, _results_queue
    _test_arrays = None
    _results_queue = None


@clean_file_when_finished('test.nnp')
def test_buffer_and_batch_performance():
    """测试缓冲区和批量写入性能"""
    print("\n=== 测试缓冲区和批量写入性能 ===")

    # 准备不同大小的测试数据
    sizes = [
        (1000, 10),      # 小数据集
        (100000, 100),   # 中等数据集
        (1000000, 50)    # 大数据集
    ]

    for rows, cols in sizes:
        print(f"\n测试数据大小: {rows}行 x {cols}列")
        test_array = np.random.rand(rows, cols)
        total_size_mb = test_array.nbytes / (1024 * 1024)
        print(f"数据总大小: {total_size_mb:.2f}MB")

        # 测试不同批量大小的写入性能
        batch_sizes = [100, 1000, 10000]
        for batch_size in batch_sizes:
            print(f"\n批量大小: {batch_size}")

            # 测试写入性能
            start_time = time.time()
            save_nnp('test.nnp', {'test_array': test_array}, batch_size=batch_size)
            write_time = time.time() - start_time
            print(f"写入时间: {write_time:.3f}秒")
            print(f"写入速度: {total_size_mb/write_time:.2f}MB/s")

            # 验证数据正确性
            loaded_array = load_nnp('test.nnp')['test_array']
            assert np.allclose(test_array, loaded_array)


@clean_file_when_finished('test.nnp')
def test_parallel_loading_performance():
    """测试并行加载性能"""
    print("\n=== 测试并行加载性能 ===")

    # 创建多个大型数组
    arrays = {
        f'array_{i}': np.random.rand(100000, 50)
        for i in range(10)
    }

    # 保存数组
    print("保存测试数据...")
    save_nnp('test.nnp', arrays)

    # 测试不同加载模式
    print("\n测试不同加载模式:")

    # 1. 串行加载
    start_time = time.time()
    _ = load_nnp('test.nnp', parallel=False)
    serial_time = time.time() - start_time
    print(f"串行加载时间: {serial_time:.3f}秒")

    # 2. 并行加载
    start_time = time.time()
    _ = load_nnp('test.nnp', parallel=True)
    parallel_time = time.time() - start_time
    print(f"并行加载时间: {parallel_time:.3f}秒")
    print(f"加速比: {serial_time/parallel_time:.2f}x")

    # 3. 内存映射模式
    start_time = time.time()
    _ = load_nnp('test.nnp', mmap_mode=True)
    mmap_time = time.time() - start_time
    print(f"内存映射加载时间: {mmap_time:.3f}秒")

    # 4. 选择性加载
    array_names = [f'array_{i}' for i in range(5)]

    start_time = time.time()
    _ = load_nnp('test.nnp', array_names=array_names, parallel=False)
    partial_serial_time = time.time() - start_time
    print(f"串行选择性加载时间: {partial_serial_time:.3f}秒")

    start_time = time.time()
    _ = load_nnp('test.nnp', array_names=array_names, parallel=True)
    partial_parallel_time = time.time() - start_time
    print(f"并行选择性加载时间: {partial_parallel_time:.3f}秒")
    print(f"选择性加载加速比: {partial_serial_time/partial_parallel_time:.2f}x")


@clean_file_when_finished('test.nnp')
def test_cache_performance():
    """测试缓存性能"""
    print("\n=== 测试缓存性能 ===")

    # 创建测试数据
    arrays = {
        'array1': np.random.rand(50000, 50),
        'array2': np.random.rand(30000, 30),
        'array3': np.random.rand(20000, 20)
    }

    # 保存数组
    print("保存测试数据...")
    save_nnp('test.nnp', arrays)

    # 测试不同缓存大小
    cache_sizes = [None, 1024*1024, 10*1024*1024]  # None, 1MB, 10MB

    for cache_size in cache_sizes:
        print(f"\n缓存大小: {cache_size/1024/1024:.1f}MB" if cache_size else "\n不使用缓存")

        # 重复加载测试
        start_time = time.time()
        for _ in range(5):
            _ = load_nnp('test.nnp', array_names=['array1'], cache_size=cache_size)
            _ = load_nnp('test.nnp', array_names=['array2'], cache_size=cache_size)
            _ = load_nnp('test.nnp', array_names=['array3'], cache_size=cache_size)
        total_time = time.time() - start_time
        print(f"重复加载时间: {total_time:.3f}秒")


# 将进程函数移到全局作用域
def _concurrent_reader_process(args):
    """并发读取进程函数"""
    array_names, use_mmap, use_cache = args
    start_time = time.time()
    arrays = load_nnp('test.nnp',
                     array_names=array_names,
                     mmap_mode=use_mmap,
                     cache_size=1024*1024 if use_cache else None)
    end_time = time.time()
    return end_time - start_time

def _concurrent_writer_process(args):
    """并发写入进程函数"""
    array_name, rows, cols = args
    new_array = np.random.rand(rows, cols)
    start_time = time.time()
    save_nnp('test.nnp', {array_name: new_array}, append=True)
    end_time = time.time()
    return end_time - start_time

@clean_file_when_finished('test.nnp')
def test_concurrent_operations():
    """测试并发操作性能"""
    print("\n=== 测试并发操作性能 ===")

    # 创建初始数据
    initial_arrays = {
        'array1': np.random.rand(100000, 50),
        'array2': np.random.rand(80000, 40),
        'array3': np.random.rand(60000, 30)
    }

    print("保存初始数据...")
    save_nnp('test.nnp', initial_arrays)

    # 创建进程池
    import multiprocessing as mp
    with mp.Pool(4) as pool:
        # 测试并发读取
        print("\n测试并发读取性能:")
        read_tasks = [
            pool.apply_async(_concurrent_reader_process, [(['array1'], False, False)]),
            pool.apply_async(_concurrent_reader_process, [(['array2'], True, False)]),  # 使用mmap
            pool.apply_async(_concurrent_reader_process, [(['array3'], False, True)]),  # 使用缓存
            pool.apply_async(_concurrent_reader_process, [(['array1', 'array2'], False, False)]),
            pool.apply_async(_concurrent_reader_process, [(['array1', 'array2', 'array3'], False, False)])
        ]
        read_times = [task.get() for task in read_tasks]
        print(f"普通读取时间: {read_times[0]:.3f}秒")
        print(f"内存映射读取时间: {read_times[1]:.3f}秒")
        print(f"缓存读取时间: {read_times[2]:.3f}秒")
        print(f"双数组读取时间: {read_times[3]:.3f}秒")
        print(f"三数组读取时间: {read_times[4]:.3f}秒")

        # 测试并发写入
        print("\n测试并发写入性能:")
        write_tasks = [
            pool.apply_async(_concurrent_writer_process, [('array1', 1000, 50)]),
            pool.apply_async(_concurrent_writer_process, [('array2', 800, 40)]),
            pool.apply_async(_concurrent_writer_process, [('array3', 600, 30)])
        ]
        write_times = [task.get() for task in write_tasks]
        print(f"并发写入时间: {max(write_times):.3f}秒")

        # 测试混合读写
        print("\n测试混合读写性能:")
        mixed_tasks = [
            pool.apply_async(_concurrent_reader_process, [(['array1'], False, False)]),
            pool.apply_async(_concurrent_writer_process, [('array2', 800, 40)]),
            pool.apply_async(_concurrent_reader_process, [(['array3'], True, False)]),
            pool.apply_async(_concurrent_writer_process, [('array1', 1000, 50)])
        ]
        mixed_times = [task.get() for task in mixed_tasks]
        print(f"混合操作最大耗时: {max(mixed_times):.3f}秒")
        print(f"混合操作平均耗时: {sum(mixed_times)/len(mixed_times):.3f}秒")


if __name__ == '__main__':
    print("开始运行测试...")
    test_save_and_load_multiple_arrays()
    test_append_to_named_array()
    test_replace_in_named_arrays()
    test_drop_from_named_arrays()
    test_mmap_mode()
    test_error_handling()
    test_performance()
    test_operations_after_append()
    test_concurrent_array_access()
    test_multiprocess_array_access()

    # 运行新的性能测试
    test_buffer_and_batch_performance()
    test_parallel_loading_performance()
    test_cache_performance()
    test_concurrent_operations()

    print("\n所有测试完成！")
