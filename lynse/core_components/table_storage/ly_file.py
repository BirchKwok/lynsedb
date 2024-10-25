from io import BytesIO
import mmap
import struct
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import BinaryIO, Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from .fsst import FSST


class MMapReader:
    """LyFile的内存映射读取器"""
    def __init__(self, filepath: Union[str, Path], thread_count: int = 4):
        self.filepath = Path(filepath)
        self.compressor = FSST(thread_count=thread_count)
        self._mmap: Optional[mmap.mmap] = None
        self._column_info: Dict[str, List[Tuple[int, int, int, int]]] = {}  # 列名 -> [(类型ID, 长度, 偏移量, 行数)]
        self.n_rows = 0

    def __enter__(self):
        self._init_mmap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

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
                        self._mmap.seek(offset - 264)
                        type_id, _, _ = struct.unpack(
                            LyFile.COLUMN_HEADER_FORMAT, self._mmap.read(264))
                        self._mmap.seek(current_pos)

                        blocks.append((type_id, length, offset, block_rows))

                    self._column_info[name] = blocks

    def read(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """读取指定列的数据"""
        if not columns:
            columns = list(self._column_info.keys())

        data = {}
        for col in columns:
            if col not in self._column_info:
                raise KeyError(f"Column {col} not found")

            # 转换块信息格式以匹配 LyFile._read_column_blocks 的参数
            blocks = [(offset, length, block_rows)
                     for _, length, offset, block_rows in self._column_info[col]]
            data[col] = LyFile._read_column_blocks(self._mmap, blocks)  # 修改这里

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

            # 读取数据块
            block_data = LyFile._read_block(self._mmap, type_id, offset, length)  # 修改这里
            data[col] = block_data

        return pd.DataFrame(data)


class LyFile:
    """列式存储文件格式实现"""

    MAGIC = b'LYFILE02'  # 更新版本号
    HEADER_FORMAT = '<8sIIIQ'  # 增加块索引区域的位置信息
    COLUMN_HEADER_FORMAT = '<II256s'  # 列头格式: 类型ID(4) + 数据长度(4) + 列名(256)
    BLOCK_INDEX_ENTRY = '<QII'  # 块索引项格式：偏移量(8) + 长度(4) + 行数(4)

    # 支持的数据类型
    TYPE_INT32 = 1
    TYPE_INT64 = 2
    TYPE_FLOAT32 = 3
    TYPE_FLOAT64 = 4
    TYPE_STRING = 5
    TYPE_BLOB = 6
    TYPE_NUMPY = 7

    def __init__(self, filepath: Union[str, Path], thread_count: int = 4):
        self.filepath = Path(filepath)
        self.columns: Dict[str, Any] = {}
        self.n_rows = 0
        self.compressor = FSST(thread_count=thread_count)  # 初始化 FSST 压缩器
        self._block_index: Dict[str, List[Tuple[int, int, int]]] = {}  # 列名 -> [(偏移量, 长度, 行数)]

    @staticmethod
    def _read_block(f: Union[mmap.mmap, BinaryIO], type_id: int, offset: int, length: int) -> np.ndarray:
        """读取单个数据块"""
        # 读取压缩数据
        f.seek(offset)
        compressed = f.read(length)
        # 创建一个新的压缩器实例
        compressor = FSST()
        raw_data = compressor.decompress(compressed)

        # 根据类型解析数据
        if type_id in (LyFile.TYPE_INT32, LyFile.TYPE_INT64,
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
            return np.array(strings)
        elif type_id == LyFile.TYPE_BLOB:
            blobs = raw_data.split(b'\0')[:-1]
            return np.array(blobs)
        else:  # TYPE_NUMPY
            return np.load(BytesIO(raw_data), allow_pickle=False)

    @staticmethod
    def _read_column_blocks(f: Union[mmap.mmap, BinaryIO], blocks: List[Tuple[int, int, int]]) -> np.ndarray:
        """读取一列的所有数据块"""
        column_data = []
        total_rows = 0

        for offset, length, block_rows in blocks:
            # 读取类型信息
            f.seek(offset - 264)
            type_id, _, _ = struct.unpack(LyFile.COLUMN_HEADER_FORMAT, f.read(264))

            # 读取数据块
            block_data = LyFile._read_block(f, type_id, offset, length)
            column_data.append(block_data)
            total_rows += block_rows

        # 合并所有数据块
        result = np.concatenate(column_data)
        assert len(result) == total_rows, f"Column has incorrect number of rows"
        return result

    def write(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]) -> None:
        """写入数据"""
        # 转换输入数据为统一格式
        if isinstance(data, pd.DataFrame):
            columns = {name: data[name].values for name in data.columns}
        elif isinstance(data, pd.Series):
            columns = {data.name or 'values': data.values}
        elif isinstance(data, pa.Table):
            columns = {field.name: data[field.name].to_numpy()
                      for field in data.schema}
        else:  # List[Dict]
            df = pd.DataFrame(data)
            columns = {name: df[name].values for name in df.columns}

        # 预计算所有列的偏移量和压缩数据
        current_pos = 28  # 文件头长度
        blocks_info = {}
        compressed_data = {}

        for name, values in columns.items():
            # 准备数据
            raw_data = self._prepare_column_data(values)
            compressed = self.compressor.compress(raw_data)

            # 获取类型ID
            type_id = self._get_type_id(values)

            # 准备列头
            name_bytes = name.encode('utf-8').ljust(256, b'\0')
            header = struct.pack(self.COLUMN_HEADER_FORMAT, type_id, len(compressed), name_bytes)

            # 记录位置信息
            blocks_info[name] = [(current_pos + 264, len(compressed), len(values))]
            compressed_data[name] = (header, compressed)

            # 更新位置
            current_pos += 264 + len(compressed)  # 列头 + 数据长度

        # 写入文件
        with open(self.filepath, 'wb') as f:
            # 写入文件头
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC, 1, len(columns),
                              len(next(iter(columns.values()))), current_pos))

            # 写入列数据
            for name in columns:
                header, data = compressed_data[name]
                f.write(header)
                f.write(data)

            # 写入块索引
            for name, blocks in blocks_info.items():
                # 写入列名
                f.write(name.encode('utf-8').ljust(256, b'\0'))
                # 写入块数量
                f.write(struct.pack('<I', len(blocks)))
                # 写入块信息
                for offset, length, block_rows in blocks:
                    f.write(struct.pack(self.BLOCK_INDEX_ENTRY, offset, length, block_rows))

        # 更新内存中的块索引
        self._block_index = blocks_info

    def append(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]) -> None:
        """追加数据到文件末尾"""
        # 转换输入数据为统一格式
        if isinstance(data, pd.DataFrame):
            columns = {name: data[name].values for name in data.columns}
        elif isinstance(data, pd.Series):
            columns = {data.name or 'values': data.values}
        elif isinstance(data, pa.Table):
            columns = {field.name: data[field.name].to_numpy()
                      for field in data.schema}
        else:  # List[Dict]
            df = pd.DataFrame(data)
            columns = {name: df[name].values for name in df.columns}

        if not self.filepath.exists():
            return self.write(data)

        # 初始化块索引（如果还没有）
        if not self._block_index:
            self._init_block_index()

        # 预计算新数据块的大小和压缩数据
        new_blocks: Dict[str, Tuple[int, bytes]] = {}
        for name, values in columns.items():
            raw_data = self._prepare_column_data(values)
            compressed = self.compressor.compress(raw_data)
            new_blocks[name] = (len(values), compressed)

        # 追加数据并更新块索引
        with open(self.filepath, 'rb+') as f:
            # 获取文件当前大小
            f.seek(0, 2)
            file_size = f.tell()

            # 写入新的数据块
            current_pos = file_size
            for name, (n_rows, compressed) in new_blocks.items():
                if name not in self._block_index:
                    self._block_index[name] = []

                # 获取数据类型ID
                type_id = self._get_type_id(columns[name])

                # 写入列头（包含类型信息）
                name_bytes = name.encode('utf-8').ljust(256, b'\0')
                header = struct.pack(self.COLUMN_HEADER_FORMAT, type_id, len(compressed), name_bytes)

                f.seek(current_pos)
                f.write(header)
                current_pos += 264  # 列头长度

                # 写入数据
                f.write(compressed)

                # 记录新块的位置信息
                self._block_index[name].append((current_pos, len(compressed), n_rows))
                current_pos += len(compressed)

            # 写入更新后的块索引
            index_pos = current_pos
            for name, blocks in self._block_index.items():
                # 写入列名
                f.write(name.encode('utf-8').ljust(256, b'\0'))
                # 写入块数量
                f.write(struct.pack('<I', len(blocks)))
                # 写入每个块的信息
                for offset, length, block_rows in blocks:
                    f.write(struct.pack(self.BLOCK_INDEX_ENTRY, offset, length, block_rows))

            # 更新文件头时计算实际的总行数
            total_rows = sum(block[2] for blocks in self._block_index.values()
                            for block in blocks)
            f.seek(0)
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC, 1, len(self._block_index), total_rows, index_pos))

    def read(self) -> pd.DataFrame:
        """读取所有数据"""
        if not self.filepath.exists():
            return pd.DataFrame()

        if not self._block_index:
            self._init_block_index()

        data = {}
        with open(self.filepath, 'rb') as f:
            for name, blocks in self._block_index.items():
                data[name] = self._read_column_blocks(f, blocks)

        return pd.DataFrame(data)

    def mmap_reader(self) -> 'MMapReader':
        """创建内存映射读取器"""
        return MMapReader(self.filepath, thread_count=self.compressor.thread_count)

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
        """确定数据类型ID"""
        if np.issubdtype(values.dtype, np.integer):
            return self.TYPE_INT64 if values.dtype in (np.int64, np.uint64) else self.TYPE_INT32
        elif np.issubdtype(values.dtype, np.floating):
            return self.TYPE_FLOAT64 if values.dtype == np.float64 else self.TYPE_FLOAT32
        elif values.dtype.kind in ('U', 'O'):
            return self.TYPE_STRING if not isinstance(values[0], bytes) else self.TYPE_BLOB
        return self.TYPE_NUMPY

    def _prepare_column_data(self, values: np.ndarray) -> bytes:
        """准备列数据"""
        if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating):
            return values.tobytes()
        elif values.dtype.kind in ('U', 'O'):
            if isinstance(values[0], bytes):
                return b'\0'.join(values) + b'\0'
            return '\0'.join(str(v) for v in values).encode('utf-8') + b'\0'
        else:
            buffer = BytesIO()
            np.save(buffer, values, allow_pickle=False)
            return buffer.getvalue()

    def _init_block_index(self) -> None:
        """初始化块索引"""
        with open(self.filepath, 'rb') as f:
            magic, version, n_cols, n_rows, index_pos = struct.unpack(
                self.HEADER_FORMAT, f.read(28))
            assert magic == self.MAGIC, "Invalid file format"

            if index_pos > 0:
                f.seek(index_pos)
                for _ in range(n_cols):
                    name_bytes = f.read(256)
                    name = name_bytes.decode('utf-8').rstrip('\0')
                    n_blocks = struct.unpack('<I', f.read(4))[0]

                    blocks = []
                    for _ in range(n_blocks):
                        offset, length, block_rows = struct.unpack(
                            self.BLOCK_INDEX_ENTRY, f.read(16))
                        blocks.append((offset, length, block_rows))
                    self._block_index[name] = blocks


if __name__ == '__main__':
    # 创建文件
    file = LyFile("data.ly")

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

    # 使用mmap方式读取特定列
    with file.mmap_reader() as reader:
        # 只读取id列
        df_id = reader.read(['id'])
        print("\n只读取id列: ")
        print(df_id)

        # 读取多列
        df_partial = reader.read(['id', 'name'])
        print("\n读取id和name列: ")
        print(df_partial)

        # 读取最新数据
        df_latest = reader.read_latest(['id', 'name'])
        print("\n读取最新数据块: ")
        print(df_latest)

    # 显示文件信息
    print(f"\n文件形状: {file.shape}")

    file.append(new_data)
    print("追加数据完成")
    print("文件形状: ", file.shape)
