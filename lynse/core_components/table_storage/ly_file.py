from io import BytesIO
import json
import mmap
import struct
import threading
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from fsst import FSST
import queue

from lynse.computational_layer.engines import cosine


class MMapReader:
    """LyFile的内存映射读取器"""
    def __init__(self, filepath: Union[str, Path], thread_count: int = 4):
        self.filepath = Path(filepath)
        self.thread_count = thread_count
        self._mmap: Optional[mmap.mmap] = None
        self._column_info: Dict[str, List[Tuple[int, int, int, int]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=thread_count)
        self.n_rows = 0

    def __enter__(self):
        self._init_mmap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        self._executor.shutdown(wait=True)

    def _init_mmap(self) -> None:
        """初始化内存映射"""
        with open(self.filepath, 'rb') as f:
            self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # 读取文件头
            magic, version, n_cols, self.n_rows, index_pos = struct.unpack(
                LyFile.HEADER_FORMAT, self._mmap.read(28))
            assert magic == LyFile.MAGIC, "Invalid file format"

            if index_pos > 0:
                # 从块索引读取列信息
                self._mmap.seek(index_pos)
                for _ in range(n_cols):
                    name_bytes = self._mmap.read(256)
                    name = name_bytes.decode('utf-8').rstrip('\0')
                    n_blocks = struct.unpack('<I', self._mmap.read(4))[0]

                    blocks = []
                    for _ in range(n_blocks):
                        offset, length, block_rows = struct.unpack(
                            LyFile.BLOCK_INDEX_ENTRY, self._mmap.read(16))

                        # 读取类型信息
                        current_pos = self._mmap.tell()
                        self._mmap.seek(offset)  # 直接使用偏移量
                        type_id, _, _ = struct.unpack(
                            LyFile.COLUMN_HEADER_FORMAT, self._mmap.read(264))
                        self._mmap.seek(current_pos)

                        blocks.append((type_id, length, offset, block_rows))

                    self._column_info[name] = blocks

    def read(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """并行读取数据

        Args:
            columns: 要读取的列名列表。如果为None，则读取所有列

        Returns:
            包含指定列的DataFrame
        """
        if not self.filepath.exists():
            return pd.DataFrame()

        # 如果没有指定列，读取所有列
        if columns is None:
            columns = list(self._column_info.keys())
        else:
            # 验证所有请求的列是否存在
            for col in columns:
                if col not in self._column_info:
                    raise KeyError(f"列 {col} 不存在")

        futures = {}
        for name in columns:
            col_futures = []
            for i, (type_id, length, offset, block_rows) in enumerate(self._column_info[name]):
                col_futures.append((i, self._executor.submit(
                    self._read_block, type_id, offset + 264, length)))  # 加上264跳过列头
            futures[name] = col_futures

        # 收集结果
        data = {}
        for name, col_futures in futures.items():
            # 按块的序号排序
            sorted_futures = sorted(col_futures, key=lambda x: x[0])
            column_data = []
            for _, future in sorted_futures:
                try:
                    block_data = future.result()
                    column_data.append(block_data)
                except Exception as e:
                    print(f"Error reading column {name}: {e}")
                    raise

            # 合并数据块
            try:
                # 如果是二维数组（向量），使用vstack
                if len(column_data) > 0 and len(column_data[0].shape) > 1:
                    data[name] = np.vstack(column_data)
                else:
                    # 对于一维数组，使用concatenate
                    data[name] = np.concatenate(column_data)
            except Exception as e:
                print(f"Error concatenating column {name}: {e}")
                print(f"Block shapes: {[block.shape for block in column_data]}")
                raise

        return pd.DataFrame(data)

    def read_latest(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """只读取最新的数据块"""
        if not columns:
            columns = list(self._column_info.keys())

        data = {}
        for col in columns:
            if col not in self._column_info:
                raise KeyError(f"Column {col} not found")

            # 获取最后一个数据块的信息
            type_id, length, offset, block_rows = self._column_info[col][-1]

            try:
                # 读取数据块
                block_data = self._read_block(type_id, offset + 264, length)  # 加上264跳过列头
                data[col] = block_data
            except Exception as e:
                print(f"Error reading latest block for column {col}: {e}")
                print(f"Block info: type_id={type_id}, length={length}, offset={offset}, rows={block_rows}")
                raise

        return pd.DataFrame(data)

    def _read_block(self, type_id: int, offset: int, length: int) -> np.ndarray:
        """读取单个数据块（线程安全）"""
        compressor = FSST()  # 每个线程创建自己的压缩器实例

        try:
            # 读取压缩数据
            with threading.Lock():  # 保护 mmap 读取
                self._mmap.seek(offset)
                compressed = self._mmap.read(length)

            # 解压数据
            try:
                raw_data = compressor.decompress(compressed)
            except ValueError as e:
                if "Unknown compression method" in str(e):
                    # 如果数据未压缩，直接使用原始数据
                    raw_data = compressed
                else:
                    raise

            # 解析数据
            return LyFile._parse_block_data(type_id, raw_data)

        except Exception as e:
            print(f"Error reading block at offset {offset}, length {length}: {e}")
            print(f"Type ID: {type_id}")
            print(f"Compressed data length: {len(compressed)}")
            raise

    def execute_along_column(self, column: str, func: callable) -> np.ndarray:
        """对指定列的所有数据执行函数"""
        if column not in self._column_info:
            raise KeyError(f"列 {column} 不存在")

        # 并行读取数据块
        futures = []
        for i, (type_id, length, offset, _) in enumerate(self._column_info[column]):
            future = self._executor.submit(
                self._process_block, type_id, offset + 264, length, func)
            futures.append((i, future))

        # 收集并合并结果
        try:
            sorted_futures = sorted(futures, key=lambda x: x[0])
            results = []
            for _, future in sorted_futures:
                result = future.result()
                if isinstance(result, np.ndarray):
                    if result.size == 1:  # 如果是单个元素的数组
                        result = result.item()  # 转换为标量
                results.append(result)

            # 根据结果类型选择合适的合并方式
            if len(results) == 1:
                return results[0]
            elif all(isinstance(r, (int, float, bool)) for r in results):
                # 如果所有结果都是标量，返回它们的平均值
                return np.mean(results)
            elif isinstance(results[0], np.ndarray):
                # 如果结果是数组，使用concatenate合并
                return np.concatenate(results)
            else:
                # 其他情况返回数组
                return np.array(results)

        except Exception as e:
            print(f"Encountered error executing function: {e}")
            raise

    def _process_block(self, type_id: int, offset: int, length: int,
                      func: callable) -> Union[np.ndarray, float, int]:
        """读取并处理单个数据块"""
        try:
            block_data = self._read_block(type_id, offset, length)
            result = func(block_data)

            # 确保标量结果被正确处理
            if isinstance(result, np.ndarray) and result.size == 1:
                return result.item()
            return result
        except Exception as e:
            print(f"Encountered error processing block: {e}")
            print(f"Type ID: {type_id}, Offset: {offset}, Length: {length}")
            raise


class LyFile:
    """列式存储文件格式实现"""

    MAGIC = b'LYFILE03'

    # 格式常量
    HEADER_FORMAT = '<8sIIIQ'  # magic(8) + version(4) + n_cols(4) + n_rows(4) + index_pos(8)
    CHUNK_HEADER_FORMAT = '<III'  # chunk_id(4) + n_columns(4) + n_rows(4)
    COLUMN_HEADER_FORMAT = '<II256s'  # type_id(4) + length(4) + name(256)
    METADATA_HEADER_FORMAT = '<QQQ'  # manifest_pos(8) + index_pos(8) + secondary_index_pos(8)
    BLOCK_INDEX_ENTRY = '<QII'  # offset(8) + length(4) + block_rows(4)

    # 默认配置
    DEFAULT_CHUNK_SIZE = 100_000
    DEFAULT_THREAD_COUNT = 4

    # 数据类型常量
    TYPE_INT32 = 1
    TYPE_INT64 = 2
    TYPE_FLOAT32 = 3
    TYPE_FLOAT64 = 4
    TYPE_STRING = 5
    TYPE_BLOB = 6
    TYPE_VECTOR = 7
    TYPE_NUMPY = 8

    def __init__(self, filepath: Union[str, Path],
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 thread_count: int = DEFAULT_THREAD_COUNT):
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.thread_count = thread_count
        self._manifest = {}
        self._index = {}
        self._secondary_index = {}
        self._block_index = {}

    def _convert_input_data(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]) -> Dict[str, np.ndarray]:
        """转换输入数据为列式存储格式"""
        if isinstance(data, pd.DataFrame):
            return {name: self._convert_column(values)
                   for name, values in data.items()}
        elif isinstance(data, pd.Series):
            return {data.name or 'value': self._convert_column(data)}
        elif isinstance(data, pa.Table):
            return {name: self._convert_column(data.column(name).to_numpy())
                   for name in data.column_names}
        else:  # List[Dict]
            df = pd.DataFrame(data)
            return {name: self._convert_column(values)
                   for name, values in df.items()}

    def _convert_column(self, values: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """转换单列数据为numpy数组"""
        if isinstance(values, pd.Series):
            values = values.to_numpy()

        # 检查是否为向量列
        if (values.dtype == 'object' and len(values) > 0 and
            isinstance(values[0], np.ndarray)):
            # 确保所有向量维度一致
            vector_dim = len(values[0])
            for i, v in enumerate(values):
                if not isinstance(v, np.ndarray):
                    raise ValueError(f"第 {i} 个元素不是numpy数组")
                if len(v) != vector_dim:
                    raise ValueError(f"向量维度不一致: 位置 {i}, 期望 {vector_dim}, 实际 {len(v)}")
            return values

        return values

    def write(self, data: Union[List[Dict], pd.DataFrame]):
        columns = self._convert_input_data(data)
        n_rows = len(next(iter(columns.values())))

        with open(self.filepath, 'wb') as f:
            # 写入文件头
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC,           # magic number
                              1,                    # version
                              len(columns),         # n_cols
                              n_rows,               # n_rows
                              0))                   # index_pos (临时占位)

            # 写入数据块
            self._block_index = {}  # 重置块索引
            for name in columns:
                self._block_index[name] = []  # 初始化每列的块列表

            chunk_positions = []
            for chunk_id, i in enumerate(range(0, n_rows, self.chunk_size)):
                chunk_pos = f.tell()
                chunk_positions.append(chunk_pos)

                # 准备块数据
                chunk_data = {}
                for name, values in columns.items():
                    chunk_data[name] = values[i:i + self.chunk_size]

                # 写入块头
                chunk_rows = len(next(iter(chunk_data.values())))
                f.write(struct.pack(self.CHUNK_HEADER_FORMAT,
                                  chunk_id,
                                  len(chunk_data),  # n_columns
                                  chunk_rows))      # n_rows

                # 写入列数据
                for name, values in chunk_data.items():
                    col_pos = f.tell()

                    # 压缩并写入数据
                    _, compressed, type_id, _ = self._compress_column(name, values)

                    # 写入列头
                    name_bytes = name.encode('utf-8').ljust(256, b'\0')
                    f.write(struct.pack(self.COLUMN_HEADER_FORMAT,
                                      type_id,          # type_id
                                      len(compressed),   # length
                                      name_bytes))       # name

                    # 写入压缩数据
                    f.write(compressed)

                    # 记录块信息
                    self._block_index[name].append((col_pos, len(compressed), chunk_rows))

            # 写入块索引
            index_pos = f.tell()
            for name, blocks in self._block_index.items():
                # 写入列名
                f.write(name.encode('utf-8').ljust(256, b'\0'))
                # 写入块数量
                f.write(struct.pack('<I', len(blocks)))
                # 写入块信息
                for offset, length, block_rows in blocks:
                    f.write(struct.pack(self.BLOCK_INDEX_ENTRY,
                                      offset, length, block_rows))

            # 更新文件头
            f.seek(0)
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC,
                              1,                # version
                              len(columns),     # n_cols
                              n_rows,           # n_rows
                              index_pos))       # index_pos

    def _write_manifest(self, f):
        """写入文件清单"""
        manifest = {
            'version': 3,
            'chunk_size': self.chunk_size,
            'columns': list(self._index[0].keys()),
            'n_chunks': len(self._index)
        }
        f.write(json.dumps(manifest).encode())

    def _write_index(self, f):
        """写入索引数据"""
        f.write(json.dumps(self._index).encode())

    def _compress_column(self, name: str, values: np.ndarray) -> Tuple[str, bytes, int, int]:
        """并行压缩单列数据"""
        compressor = FSST()  # 创建新的压缩器实例
        raw_data = self._prepare_column_data(values)
        compressed = compressor.compress(raw_data)
        type_id = self._get_type_id(values)
        return name, compressed, type_id, len(values)

    def _prepare_column_data(self, values: np.ndarray) -> bytes:
        """准备列数据"""
        type_id = self._get_type_id(values)

        if type_id == self.TYPE_VECTOR:

            # 获取向量维度
            vector_dim = len(values[0])
            n_vectors = len(values)


            # 将所有向量堆叠成一个连续数组
            try:
                stacked = np.vstack([v.astype(np.float64) for v in values])
            except Exception as e:
                raise

            # 写入维度信息向量数量
            header = struct.pack('<II', vector_dim, n_vectors)
            result = header + stacked.tobytes()

            return result

        elif type_id in (self.TYPE_INT32, self.TYPE_INT64,
                        self.TYPE_FLOAT32, self.TYPE_FLOAT64):
            return values.tobytes()
        elif type_id == self.TYPE_STRING:
            return '\0'.join(str(v) for v in values).encode('utf-8') + b'\0'
        elif type_id == self.TYPE_BLOB:
            return b'\0'.join(v if isinstance(v, bytes) else v.encode()
                            for v in values) + b'\0'
        else:  # TYPE_NUMPY
            buffer = BytesIO()
            np.save(buffer, values, allow_pickle=False)
            return buffer.getvalue()

    def append(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]) -> None:
        """并追加数据到文件末尾"""
        # 转换输入数据
        columns = self._convert_input_data(data)

        if not self.filepath.exists():
            return self.write(data)

        # 初始化块索引（如果还没有）
        if not self._block_index:
            self._init_block_index()

        # 并行压缩所有列
        futures = []
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            for name, values in columns.items():
                futures.append(executor.submit(
                    self._compress_column, name, values))

            # 收集压缩结果
            new_blocks: Dict[str, Tuple[int, bytes, int, int]] = {}  # name -> (n_rows, compressed, type_id, length)
            for future in concurrent.futures.as_completed(futures):
                name, compressed, type_id, n_rows = future.result()
                new_blocks[name] = (n_rows, compressed, type_id, len(compressed))

        # 追加数据并更新块索引
        with open(self.filepath, 'rb+') as f:
            # 获取文件当前大小
            f.seek(0, 2)
            file_size = f.tell()

            # 写入新的数据块
            current_pos = file_size
            for col_name, (n_rows, compressed, type_id, length) in new_blocks.items():
                if col_name not in self._block_index:
                    self._block_index[col_name] = []

                # 写入列头（包含类型信息）
                name_bytes = col_name.encode('utf-8').ljust(256, b'\0')
                header = struct.pack(self.COLUMN_HEADER_FORMAT,
                                   type_id,          # type_id
                                   length,           # length
                                   name_bytes)       # name

                f.seek(current_pos)
                f.write(header)
                current_pos += 264  # 列头长度

                # 写入数据
                f.write(compressed)

                # 记录新块的位置信息
                block_info = (current_pos - 264, length, n_rows)  # 使用列头开始位置作为偏移量
                self._block_index[col_name].append(block_info)
                current_pos += length

            # 写入更新后的块索引
            index_pos = current_pos
            for col_name in columns.keys():  # 使用原始列名
                # 写入列名
                f.write(col_name.encode('utf-8').ljust(256, b'\0'))
                # 写入块数量
                f.write(struct.pack('<I', len(self._block_index[col_name])))
                # 写入每个块的信息
                for offset, length, block_rows in self._block_index[col_name]:
                    f.write(struct.pack(self.BLOCK_INDEX_ENTRY,
                                      offset, length, block_rows))

            # 更新文件头
            total_rows = sum(block[2] for blocks in self._block_index.values()
                            for block in blocks)
            f.seek(0)
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC, 1, len(self._block_index),
                              total_rows, index_pos))

    def read(self) -> pd.DataFrame:
        """并行读取所有数据"""
        if not self.filepath.exists():
            return pd.DataFrame()

        if not self._block_index:
            self._init_block_index()

        # 并行读取所有列
        futures = {}
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            for name, blocks in self._block_index.items():
                col_futures = []
                for i, (offset, length, block_rows) in enumerate(blocks):
                    # 读取类型信息
                    with open(self.filepath, 'rb') as f:
                        f.seek(offset)
                        type_id, _, _ = struct.unpack(
                            self.COLUMN_HEADER_FORMAT, f.read(264))

                    col_futures.append((i, executor.submit(
                        self._read_block_with_type, offset + 264, length, type_id)))
                futures[name] = col_futures

        # 收集结果
        data = {}
        expected_length = None
        for name, col_futures in futures.items():
            # 按块的序号排序
            sorted_futures = sorted(col_futures, key=lambda x: x[0])
            column_data = []

            # 获取第一个块来确定数据类型
            first_block = sorted_futures[0][1].result()
            is_vector = len(first_block.shape) > 1

            # 收集所有块
            for _, future in sorted_futures:
                try:
                    block_data = future.result()
                    column_data.append(block_data)
                except Exception as e:
                    print(f"Error reading column {name}: {e}")
                    raise

            try:
                if is_vector:
                    # 对于向量数据，使用vstack并转换为列表
                    stacked = np.vstack(column_data)
                    # 将每个向量转换为列表，这样pandas可以处理
                    data[name] = [vec.tolist() for vec in stacked]
                else:
                    # 对于标量数据，使用concatenate
                    data[name] = np.concatenate(column_data)

                # 验证长度
                if expected_length is None:
                    expected_length = len(data[name])
                elif len(data[name]) != expected_length:
                    raise ValueError(
                        f"列 {name} 的长度 ({len(data[name])}) "
                        f"与预期长度 ({expected_length}) 不一致")

            except Exception as e:
                print(f"Error concatenating column {name}: {e}")
                print(f"Block shapes: {[block.shape for block in column_data]}")
                raise

        return pd.DataFrame(data)

    def _read_block_with_type(self, offset: int, length: int, type_id: int) -> np.ndarray:
        """读取单个数据块"""
        compressor = FSST()
        with open(self.filepath, 'rb') as f:
            f.seek(offset)
            compressed = f.read(length)
        raw_data = compressor.decompress(compressed)
        return self._parse_block_data(type_id, raw_data)

    def mmap_reader(self) -> 'MMapReader':
        """创建内存映射读取器"""
        return MMapReader(self.filepath, thread_count=self.thread_count)

    @property
    def shape(self) -> Tuple[int, int]:
        """返回文件的形状 (行数, 列数)"""
        if not self._block_index:
            self._init_block_index()

        # 计算实际的总行数（所有块的行数之和）
        total_rows = 0
        if self._block_index:
            # 取第一列的所有块的行数之和
            first_col = next(iter(self._block_index.values()))
            total_rows = sum(block[2] for block in first_col)

        return total_rows, len(self._block_index)

    def _get_type_id(self, values: np.ndarray) -> int:
        """获取数据类型ID"""
        if values.dtype == np.int32:
            return self.TYPE_INT32
        elif values.dtype == np.int64:
            return self.TYPE_INT64
        elif values.dtype == np.float32:
            return self.TYPE_FLOAT32
        elif values.dtype == np.float64:
            return self.TYPE_FLOAT64
        elif values.dtype.kind == 'O':
            if isinstance(values[0], (str, np.str_)):
                return self.TYPE_STRING
            elif isinstance(values[0], (bytes, np.bytes_)):
                return self.TYPE_BLOB
            elif isinstance(values[0], np.ndarray):
                return self.TYPE_VECTOR
        return self.TYPE_NUMPY

    def _init_block_index(self) -> None:
        """初始化块索引"""
        self._block_index = {}  # 重置块索引

        with open(self.filepath, 'rb') as f:
            # 读取文件头
            magic, version, n_cols, n_rows, index_pos = struct.unpack(
                self.HEADER_FORMAT, f.read(28))
            assert magic == self.MAGIC, "Invalid file format"

            if index_pos > 0:
                f.seek(index_pos)
                for _ in range(n_cols):
                    # 读取列名
                    name_bytes = f.read(256)
                    name = name_bytes.decode('utf-8').rstrip('\0')

                    # 读取块数量
                    n_blocks = struct.unpack('<I', f.read(4))[0]

                    # 初始化该列的块列表
                    blocks = []

                    # 读取每个块的信息
                    for _ in range(n_blocks):
                        # 读取块信息
                        block_info = struct.unpack(self.BLOCK_INDEX_ENTRY, f.read(16))
                        blocks.append(block_info)

                    # 将块列表存储到索引中
                    self._block_index[name] = blocks

    @staticmethod
    def _parse_block_data(type_id: int, raw_data: bytes) -> np.ndarray:
        """解析数据块"""
        try:
            if type_id == LyFile.TYPE_VECTOR:
                # 解析向量数据
                vector_dim, n_vectors = struct.unpack('<II', raw_data[:8])
                vectors_data = raw_data[8:]

                # 将字节数据转换为浮点数数组
                flat_array = np.frombuffer(vectors_data, dtype=np.float64)

                # 重塑为正确的形状并返回
                return flat_array.reshape(n_vectors, vector_dim)

            elif type_id in (LyFile.TYPE_INT32, LyFile.TYPE_INT64,
                          LyFile.TYPE_FLOAT32, LyFile.TYPE_FLOAT64):
                dtype = {
                    LyFile.TYPE_INT32: np.int32,
                    LyFile.TYPE_INT64: np.int64,
                    LyFile.TYPE_FLOAT32: np.float32,
                    LyFile.TYPE_FLOAT64: np.float64
                }[type_id]
                return np.frombuffer(raw_data, dtype=dtype)

            elif type_id == LyFile.TYPE_STRING:
                strings = raw_data.decode('utf-8').split('\0')[:-1]
                return np.array(strings, dtype=object)

            elif type_id == LyFile.TYPE_BLOB:
                blobs = raw_data.split(b'\0')[:-1]
                return np.array(blobs, dtype=object)

            else:  # TYPE_NUMPY
                with BytesIO(raw_data) as buffer:
                    return np.load(buffer, allow_pickle=False)

        except Exception as e:
            print(f"Error parsing block data: {e}")
            print(f"Type ID: {type_id}")
            print(f"Raw data length: {len(raw_data)}")
            raise


if __name__ == '__main__':
    # 创建文件
    file = LyFile("data.ly", thread_count=8)

    # 写入数据
    data = pd.DataFrame({
        'id': np.arange(1000),
        'name': [f'name_{i}' for i in range(1000)],
        'value': np.random.random(1000)
    })
    file.write(data)
    print("写入数据完成")

    # 追加数据
    new_data = pd.DataFrame({
        'id': np.arange(1000, 2000),
        'name': [f'name_{i}' for i in range(1000, 2000)],
        'value': np.random.random(1000)
    })
    file.append(new_data)
    print("追加数据完成")

    # 读取数据
    df = file.read()
    print("全量读取: ")
    print(df)
    print("\n验证id列是否有序:")
    print(df['id'].is_monotonic_increasing)

    # 使用mmap方式读取特定列
    with file.mmap_reader() as reader:
        # 只读取id列
        df_id = reader.read(['id'])
        print("\n只读取id列: ")
        print(df_id)
        print("\n验证id列是否有序:")
        print(df_id['id'].is_monotonic_increasing)

        # 读取多列
        df_partial = reader.read(['id', 'name'])
        print("\n读取id和name列: ")
        print(df_partial)
        print("\n验证id列是否有序:")
        print(df_partial['id'].is_monotonic_increasing)

        # 读取最新数据
        df_latest = reader.read_latest(['id', 'name'])
        print("\n读取最新数据块: ")
        print(df_latest)
        print("\n验证id列是否有序:")
        print(df_latest['id'].is_monotonic_increasing)

    # 显示文件信息
    print(f"\n文件形状: {file.shape}")

    file.append(new_data)
    print("追加据完成")
    print("文件形状: ", file.shape)

    print("\n=======测试性能=======")

    import time
    import psutil
    import os

    def format_size(size):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f}{unit}"
            size /= 1024
        return f"{size:.2f}TB"

    def get_memory_usage():
        """获取当前进程的内存使用"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def run_benchmark(rows: int, thread_count: int):
        print(f"\n=== 性能测试: {rows:,} 行, {thread_count} 线程 ===")

        # 准备测试数据
        test_data = pd.DataFrame({
            'id': np.arange(rows),
            'name': [f'name_{i}' for i in range(rows)],
            'value': np.random.random(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
            'amount': np.random.normal(1000, 100, rows),
            'flag': np.random.choice([True, False], rows),
            'score': np.random.randint(0, 100, rows),
            'text': [f'This is a longer text for testing compression performance {i}' for i in range(rows)],
            'vector': [np.random.random(128) for _ in range(rows)]
        })

        file_path = f"benchmark_{rows}_{thread_count}.ly"
        file = LyFile(file_path, thread_count=thread_count)

        # 测试写入性能
        start_mem = get_memory_usage()
        start_time = time.time()
        file.write(test_data)
        write_time = time.time() - start_time
        file_size = os.path.getsize(file_path)
        mem_used = get_memory_usage() - start_mem

        print(f"\n写入性能:")
        print(f"耗时: {write_time:.2f}秒")
        print(f"速度: {rows/write_time:,.0f} 行/秒")
        print(f"文件大小: {format_size(file_size)}")
        print(f"压缩比: {file_size/(test_data.memory_usage(deep=True).sum()):.2%}")
        print(f"内存使用: {format_size(mem_used)}")

        # 测试全量读取性能
        start_mem = get_memory_usage()
        start_time = time.time()
        df = file.read()
        read_time = time.time() - start_time
        mem_used = get_memory_usage() - start_mem

        print(f"\n全量读取性能:")
        print(f"耗时: {read_time:.2f}秒")
        print(f"速: {rows/read_time:,.0f} 行/秒")
        print(f"内存使用: {format_size(mem_used)}")

        # 测试列式读取性能
        with file.mmap_reader() as reader:
            # 测试单列读取
            start_time = time.time()
            df_single = reader.read(['id'])
            single_col_time = time.time() - start_time

            print(f"\n单列读取性能:")
            print(f"耗时: {single_col_time:.2f}秒")
            print(f"速度: {rows/single_col_time:,.0f} 行/秒")

            # 测试多列读取
            start_time = time.time()
            df_multi = reader.read(['id', 'name', 'value'])
            multi_col_time = time.time() - start_time

            print(f"\n多列读取性能:")
            print(f"耗时: {multi_col_time:.2f}秒")
            print(f"速度: {rows/multi_col_time:,.0f} 行/秒")

            # 测试最新数据块读取
            start_time = time.time()
            df_latest = reader.read_latest(['id', 'name', 'value'])
            latest_time = time.time() - start_time

            print(f"\n最新数据块读取性能:")
            print(f"耗时: {latest_time:.2f}秒")
            print(f"速度: {len(df_latest)/latest_time:,.0f} 行/秒")

            # 测试列处理性能
        print("\n=== 测试列处理性能 ===")
        with file.mmap_reader() as reader:
            # 测试1: 数值计算
            print("\n1. 数值列处理:")
            start_time = time.time()
            mean = float(reader.execute_along_column('value', np.mean))
            max_val = float(reader.execute_along_column('value', np.max))
            min_val = float(reader.execute_along_column('value', np.min))
            calc_time = time.time() - start_time

            print(f"数值统计耗时: {calc_time:.3f}秒")
            print(f"平均值: {mean:.3f}")
            print(f"最大值: {max_val:.3f}")
            print(f"最小值: {min_val:.3f}")

            # 验证结果
            np.testing.assert_almost_equal(
                mean,
                test_data['value'].mean(),
                decimal=3
            )

            # 测试2: 字符串处理
            print("\n2. 字符串列理:")
            start_time = time.time()
            # 计字符串平均长度
            avg_len = float(reader.execute_along_column(
                'text',
                lambda x: np.mean([len(s) for s in x])
            ))
            str_time = time.time() - start_time

            print(f"字符串处理耗时: {str_time:.3f}秒")
            print(f"文本平均长度: {avg_len:.1f}")

            # 测试3: 向量处理
            print("\n3. 向量处理:")
            start_time = time.time()

            # 计算向量余弦相似度
            query = np.random.random((1, 128))
            def calc_cosine_similarity(vectors):
                nonlocal query
                vectors_array = np.vstack(vectors).squeeze()
                return cosine(vectors_array, query, 1)[1]

            norms = reader.execute_along_column('vector', calc_cosine_similarity)
            vector_time = time.time() - start_time

            print(f"向量处理耗时: {vector_time:.3f}秒")
            print(f"向量范数示例 (前5个):")
            for i in range(min(5, len(norms))):
                print(f"  向量 {i}: {norms[i]:.3f}")

            # 测试4: 复杂计算
            print("\n4. 复杂计算:")
            start_time = time.time()

            # 计算数值列的移动平均
            def moving_average(data, window=5):
                return np.convolve(data, np.ones(window)/window, mode='valid')

            ma = reader.execute_along_column('value',
                                        lambda x: moving_average(x))
            complex_time = time.time() - start_time

            print(f"复杂计算耗时: {complex_time:.3f}秒")
            print(f"移动平均示例 (前5个): {ma[:5]}")

            # 性能总结
            print("\n处理性能总结:")
            print(f"数值计算速度: {rows/calc_time:,.0f} 行/秒")
            print(f"字符串处理速度: {rows/str_time:,.0f} 行/秒")
            print(f"向量处理速度: {rows/vector_time:,.0f} 行/秒")
            print(f"复杂计算速度: {rows/complex_time:,.0f} 行/秒")

        # 测试追加性能
        append_data = test_data.copy()
        append_data['id'] += len(test_data)

        start_time = time.time()
        file.append(append_data)
        append_time = time.time() - start_time

        print(f"\n追加性能:")
        print(f"耗时: {append_time:.2f}秒")
        print(f"速度: {rows/append_time:,.0f} 行/秒")

        # 清理测试文件
        os.remove(file_path)

    # 运行不同规模的测试
    test_configs = [
        (10_000, 4),    # 小数据量
        (100_000, 4),   # 中等数据量
        (1_000_000, 4), # 大数据量
    ]

    # 测试不同线程数
    thread_configs = [
        (100_000, 1),   # 单线程
        (100_000, 2),   # 2线程
        (100_000, 4),   # 4线程
        (100_000, 8),   # 8线程
    ]

    # 运行数据量测试
    print("\n=== 测试不同数据量 ===")
    for rows, threads in test_configs:
        run_benchmark(rows, threads)

    # 运行线程数测试
    print("\n=== 测试不同线程数 ===")
    for rows, threads in thread_configs:
        run_benchmark(rows, threads)
