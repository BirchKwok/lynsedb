from io import BytesIO
import struct
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Dict, List, Union, Any
from pathlib import Path
from fsst import FSST  # 导入 FSST 压缩器


class LyFile:
    """列式存储文件格式实现"""

    MAGIC = b'LYFILE01'  # 文件魔数
    HEADER_FORMAT = '<8sIII'  # 文件头格式: 魔数(8) + 版本(4) + 列数(4) + 行数(4)
    COLUMN_HEADER_FORMAT = '<II256s'  # 列头格式: 类型ID(4) + 数据长度(4) + 列名(256)

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

    def write(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]) -> None:
        """写入数据"""
        # 转换输入数据为统一格式
        if isinstance(data, pd.DataFrame):
            columns = {name: data[name].values for name in data.columns}
        elif isinstance(data, pd.Series):
            columns = {data.name or 'values': data.values}
        elif isinstance(data, pa.Table):
            columns = {field.name: data.column(field.name).to_numpy()
                      for field in data.schema}
        else:  # List[Dict]
            columns = {}
            for row in data:
                for k, v in row.items():
                    if k not in columns:
                        columns[k] = []
                    columns[k].append(v)
            columns = {k: np.array(v) for k, v in columns.items()}

        # 写入文件
        with open(self.filepath, 'wb') as f:
            # 写入文件头
            f.write(struct.pack(self.HEADER_FORMAT,
                              self.MAGIC, 1, len(columns), len(next(iter(columns.values())))))

            # 写入每一列
            for name, values in columns.items():
                self._write_column(f, name, values)

    def append(self, data: Union[List[Dict], pd.DataFrame, pd.Series, pa.Table]) -> None:
        """追加数据"""
        if not self.filepath.exists():
            return self.write(data)

        # 读取现有数据
        existing = self.read()

        # 转换新数据
        if isinstance(data, pd.DataFrame):
            new_data = data
        elif isinstance(data, pd.Series):
            new_data = pd.DataFrame({data.name or 'values': data})
        elif isinstance(data, pa.Table):
            new_data = data.to_pandas()
        else:  # List[Dict]
            new_data = pd.DataFrame(data)

        # 合并数据
        merged = pd.concat([existing, new_data], ignore_index=True)
        self.write(merged)

    def read(self) -> pd.DataFrame:
        """读取数据"""
        if not self.filepath.exists():
            return pd.DataFrame()

        with open(self.filepath, 'rb') as f:
            # 读取文件头
            magic, version, n_cols, n_rows = struct.unpack(self.HEADER_FORMAT, f.read(20))
            assert magic == self.MAGIC, "Invalid file format"

            # 读取每一列
            data = {}
            for _ in range(n_cols):
                type_id, length, name_bytes = struct.unpack(self.COLUMN_HEADER_FORMAT, f.read(264))
                name = name_bytes.decode('utf-8').rstrip('\0')

                # 读取压缩数据
                compressed = f.read(length)
                raw_data = self.compressor.decompress(compressed)  # 使用 FSST 解压

                # 根据类型解析数据
                if type_id in (self.TYPE_INT32, self.TYPE_INT64,
                             self.TYPE_FLOAT32, self.TYPE_FLOAT64):
                    dtype = {
                        self.TYPE_INT32: np.int32,
                        self.TYPE_INT64: np.int64,
                        self.TYPE_FLOAT32: np.float32,
                        self.TYPE_FLOAT64: np.float64
                    }[type_id]
                    data[name] = np.frombuffer(raw_data, dtype=dtype)
                elif type_id == self.TYPE_STRING:
                    strings = raw_data.decode('utf-8').split('\0')[:-1]
                    data[name] = np.array(strings)
                elif type_id in (self.TYPE_BLOB, self.TYPE_NUMPY):
                    data[name] = np.load(BytesIO(raw_data))

        return pd.DataFrame(data)

    def _write_column(self, f, name: str, values: np.ndarray) -> None:
        """写入单列数据"""
        # 确定数据类型
        if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating):
            type_map = {
                np.dtype('int32'): self.TYPE_INT32,
                np.dtype('int64'): self.TYPE_INT64,
                np.dtype('float32'): self.TYPE_FLOAT32,
                np.dtype('float64'): self.TYPE_FLOAT64
            }
            type_id = type_map.get(values.dtype)
            if type_id is None:
                # 如果不是精确匹配的类型，根据类型种类选择合适的类型
                if np.issubdtype(values.dtype, np.integer):
                    type_id = self.TYPE_INT64
                    values = values.astype(np.int64)
                else:
                    type_id = self.TYPE_FLOAT64
                    values = values.astype(np.float64)
            raw_data = values.tobytes()
        elif values.dtype.kind == 'U' or values.dtype.kind == 'O':
            type_id = self.TYPE_STRING
            raw_data = '\0'.join(str(v) for v in values).encode('utf-8') + b'\0'
        else:
            type_id = self.TYPE_NUMPY
            buffer = BytesIO()
            np.save(buffer, values)
            raw_data = buffer.getvalue()

        # 使用 FSST 压缩数据
        compressed = self.compressor.compress(raw_data)

        # 写入列头
        name_bytes = name.encode('utf-8')
        name_bytes = name_bytes.ljust(256, b'\0')
        f.write(struct.pack(self.COLUMN_HEADER_FORMAT, type_id, len(compressed), name_bytes))

        # 写入压缩数据
        f.write(compressed)


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

    # 追加数据
    new_data = pd.DataFrame({
        'id': np.arange(1000, 2000),
        'name': [f'name_{i}' for i in range(1000, 2000)],
        'value': np.random.random(1000)
    })
    file.append(new_data)

    # 读取数据
    df = file.read()
    print(df)
