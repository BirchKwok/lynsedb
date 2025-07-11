import numpy as np
import os
import struct
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Iterator, Union
import mmap
from threading import RLock
import msgpack

from spinesUtils.timer import Timer


class WALBuffer:
    def __init__(self):
        self.data = []
        self.fields = []
        self.size = 0

    def append(self, data: np.ndarray, fields: List[Dict]):
        self.data.append(data)
        self.fields.extend(fields)
        self.size += len(fields)

    def clear(self):
        self.data = []
        self.fields = []
        self.size = 0

    def get_concatenated(self):
        if not self.data:
            return None, None, []
        data = np.concatenate(self.data, axis=0) if len(self.data) > 1 else self.data[0]
        return data, self.fields


class WALStorage:
    HEADER_FORMAT = '<QQdQ'  # version(8), chunk_size(8), flush_interval(8), count_rows(8)
    SEGMENT_HEADER_FORMAT = '<QQBQ'  # data_size(8), record_count(8), status(1), data_dim(8)
    VERSION = 1
    MAX_WAL_SIZE = 1024 * 1024 * 1024  # 1GB
    BUFFER_FLUSH_SIZE = 10000  # 缓冲区刷新阈值
    WRITE_BUFFER_SIZE = 8 * 1024 * 1024  # 8MB写入缓冲区

    def __init__(self, collection_name: str, chunk_size: int, storage_path: Union[str, Path], flush_interval: float = 5):
        self.storage_path = Path(storage_path) / "wal"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.current_wal_id = self._get_latest_wal_id()
        self.wal_file = self._get_wal_path(self.current_wal_id)

        self.chunk_size = chunk_size
        self.flush_interval = flush_interval
        self.lock = RLock()
        self.buffer = WALBuffer()

        self.timer = Timer()
        self.timer.start()

        self.alive = True
        self._mmap = None
        self._initialized = False
        self._write_buffer = bytearray(self.WRITE_BUFFER_SIZE)
        self._write_buffer_pos = 0
        self._current_file = None
        self._pending_row_count = 0

    def _get_wal_path(self, wal_id: int) -> Path:
        """获取指定ID的WAL文件路径"""
        return self.storage_path / f"{self.collection_name}.{wal_id:06d}.wal"

    def _get_latest_wal_id(self) -> int:
        """获取最新的WAL文件ID"""
        pattern = str(self.storage_path / f"{self.collection_name}.*.wal")
        files = glob.glob(pattern)
        if not files:
            return 0
        return max(int(Path(f).absolute().stem.split('.')[-2]) for f in files)

    def _get_file_row_count(self, file_path: Path) -> int:
        """获取WAL文件中的实际行数"""
        try:
            with open(file_path, 'rb') as f:
                # 读取文件头中的count_rows
                f.seek(struct.calcsize('<QQd'))  # 跳过version, chunk_size, flush_interval
                count_rows = struct.unpack('<Q', f.read(8))[0]
                return count_rows
        except Exception:
            return 0

    def _rotate_wal_file(self):
        """当WAL文件达到大小限制时轮转到新文件"""
        with self.lock:
            # 检查当前WAL文件大小
            if not self.wal_file.exists() or self.wal_file.stat().st_size < self.MAX_WAL_SIZE:
                return

            # 创建新的WAL文件
            self.current_wal_id += 1
            self.wal_file = self._get_wal_path(self.current_wal_id)
            self._initialize_wal_file()  # 确保新文件被正确初始化

    def cleanup(self):
        """清理所有WAL文件"""
        with self.lock:
            pattern = str(self.storage_path / f"{self.collection_name}.*.wal")
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Error cleaning up WAL file {f}: {e}")

    def reincarnate(self):
        """清理所有WAL文件并重置状态"""
        with self.lock:
            self._close_current_file()
            self.cleanup()
            self.current_wal_id = 0
            self.wal_file = self._get_wal_path(self.current_wal_id)
            self._initialized = False
            self.buffer.clear()
            self._write_buffer_pos = 0
            self._pending_row_count = 0

            # 重置计时器
            self.timer = Timer()
            self.timer.start()

    @property
    def log_dir(self) -> Path:
        """兼容性属性，返回WAL文件所在目录"""
        return self.storage_path

    def has_uncommitted_data(self) -> bool:
        """检查是否存在未提交的数据"""
        pattern = str(self.storage_path / f"{self.collection_name}.*.wal")
        files = glob.glob(pattern)

        if not files:
            return False

        if len(files) == 1:
            # 如果只有一个文件，检查其实际行数
            return self._get_file_row_count(Path(files[0])) > 0

        return True  # 如果有多个文件，说明一定有未提交数据

    def _initialize_wal_file(self):
        """延迟始化WAL文件"""
        if not self._initialized:
            with self.lock:
                if not self._initialized:  # 双重检查锁定
                    if not self.wal_file.exists():
                        with open(self.wal_file, 'wb') as f:
                            # Write header with count_rows=0
                            f.write(struct.pack(self.HEADER_FORMAT,
                                                self.VERSION, self.chunk_size, self.flush_interval, 0))
                    self._initialized = True

    def _update_row_count(self, count: int):
        """更新WAL文件头中的行数"""
        self._pending_row_count += count
        if self._pending_row_count >= self.BUFFER_FLUSH_SIZE:
            self._flush_row_count()

    def _flush_row_count(self):
        """刷新行数到文件头"""
        if self._pending_row_count > 0:
            with open(self.wal_file, 'r+b') as f:
                current_count = self._get_file_row_count(self.wal_file)
                new_count = current_count + self._pending_row_count
                f.seek(struct.calcsize('<QQd'))
                f.write(struct.pack('<Q', new_count))
                f.flush()
                os.fsync(f.fileno())
            self._pending_row_count = 0

    def write_log_data(self, data: Union[np.ndarray, List[np.ndarray]], fields: Union[List[Dict], Dict]):
        if not isinstance(data, np.ndarray):
            data = np.vstack(data) if len(data) > 1 else data[0]

        if isinstance(fields, dict):
            fields = [fields]
        if not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields):
            raise ValueError("fields should be a list of dictionaries")

        with self.lock:
            # 在实际写入时初始化WAL文件
            self._initialize_wal_file()

            self.buffer.append(data, fields)

            if self.buffer.size >= self.BUFFER_FLUSH_SIZE:
                self._flush_buffer_to_disk()
            elif self.timer.last_timestamp_diff() >= self.flush_interval:
                self._flush_buffer_to_disk()
                self.timer.middle_point()

    def _flush_buffer_to_disk(self):
        if self.buffer.size == 0:
            return

        with self.lock:
            try:
                data, fields = self.buffer.get_concatenated()
                if data is None:
                    return

                data_dim = data.shape[1]

                # 检查是否需要轮转WAL文件
                self._rotate_wal_file()

                # 使用msgpack序列化fields
                fields_bytes = msgpack.packb(fields, use_bin_type=True)
                data_bytes = data.tobytes()

                total_size = len(data_bytes) + len(fields_bytes)
                record_count = len(fields)

                # 写入段头
                header = struct.pack(self.SEGMENT_HEADER_FORMAT, total_size, record_count, 1, data_dim)
                self._write_to_buffer(header)

                # 写入数据长度和数据
                self._write_to_buffer(struct.pack('<Q', len(data_bytes)))
                self._write_to_buffer(data_bytes)

                # 写入字段长度和字段
                self._write_to_buffer(struct.pack('<Q', len(fields_bytes)))
                self._write_to_buffer(fields_bytes)

                # 立即刷新写入缓冲区
                self._flush_write_buffer()

                # 更新行数并立即刷新
                self._update_row_count(record_count)
                self._flush_row_count()

                # 清空buffer
                self.buffer.clear()

            except Exception as e:
                print(f"Error during flush: {e}")
                return

    def get_file_iterator(self) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """使用内存映射方式读取WAL文件"""
        with self.lock:
            # 确保所有数据都写入到磁盘
            self._flush_buffer_to_disk()
            self._flush_write_buffer()
            self._flush_row_count()

            if self._current_file is not None:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())

            # 获取所有WAL文件并按ID排序
            pattern = str(self.storage_path / f"{self.collection_name}.*.wal")
            wal_files = sorted(glob.glob(pattern))

            for wal_file in wal_files:
                try:
                    with open(wal_file, 'rb') as f:
                        # 获取文件大小
                        f.seek(0, os.SEEK_END)
                        file_size = f.tell()
                        f.seek(0)

                        # 创建内存映射
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            # 跳过文件头
                            header_size = struct.calcsize(self.HEADER_FORMAT)
                            if file_size < header_size:
                                print(f"File {wal_file} is too small to contain a valid header")
                                continue

                            mm.seek(header_size)

                            while mm.tell() < mm.size():
                                try:
                                    current_pos = mm.tell()
                                    remaining_size = mm.size() - current_pos

                                    # 确保剩余空间足够读取段头
                                    if remaining_size < struct.calcsize(self.SEGMENT_HEADER_FORMAT):
                                        break

                                    # 读取段头
                                    header_data = mm.read(struct.calcsize(self.SEGMENT_HEADER_FORMAT))
                                    if not header_data:
                                        break

                                    total_size, record_count, status, data_dim = struct.unpack(
                                        self.SEGMENT_HEADER_FORMAT, header_data)

                                    # 验证数据大小的合理性
                                    if total_size <= 0 or total_size > self.MAX_WAL_SIZE:
                                        print(f"Invalid segment size {total_size} at position {current_pos}")
                                        break

                                    if status != 1:  # 跳过未完成的段
                                        mm.seek(total_size, os.SEEK_CUR)
                                        continue

                                    # 确保剩余空间足够读取整个段
                                    remaining_size = mm.size() - mm.tell()
                                    if remaining_size < total_size:
                                        print(f"Incomplete segment at position {current_pos}: "
                                              f"remaining_size={remaining_size}, required_size={total_size}")
                                        break

                                    try:
                                        # 读取数据部分
                                        data_size = struct.unpack('<Q', mm.read(8))[0]
                                        if data_size <= 0 or data_size > total_size:
                                            print(f"Invalid data size {data_size} at position {mm.tell()}")
                                            break

                                        data_bytes = mm.read(data_size)
                                        data = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, data_dim)

                                        # 读取字段
                                        fields_size = struct.unpack('<Q', mm.read(8))[0]
                                        fields_bytes = mm.read(fields_size)
                                        fields = msgpack.unpackb(fields_bytes, raw=False)

                                        yield data, fields

                                    except Exception as e:
                                        print(f"Error reading segment data at position {mm.tell()}: {e}")
                                        break

                                except Exception as e:
                                    print(f"Error reading segment at position {mm.tell()} in {wal_file}: {e}")
                                    break

                except Exception as e:
                    print(f"Error opening file {wal_file}: {e}")
                    continue

    @property
    def chunk_number(self) -> int:
        """获取所有WAL文件中的chunk总数量，根据所有文件的count_rows总和计算"""
        total_rows = 0
        pattern = str(self.storage_path / f"{self.collection_name}.*.wal")

        # 先刷新缓冲区，确保所有数据都写入到文件中
        self._flush_buffer_to_disk()

        # 遍历所有WAL文件，累加count_rows
        for wal_file in glob.glob(pattern):
            try:
                total_rows += self._get_file_row_count(Path(wal_file))
            except Exception as e:
                print(f"Error reading row count in {wal_file}: {e}")
                continue

        # 根据总行数计算chunk数量（向上取整）
        return (total_rows + self.chunk_size - 1) // self.chunk_size

    def stop(self):
        """停止WAL服务"""
        self.alive = False
        with self.lock:
            self._flush_buffer_to_disk()
            self._flush_write_buffer()
            self._flush_row_count()
            self._close_current_file()

    def _open_current_file(self):
        """打开当前WAL文件用于写入"""
        if self._current_file is None:
            self._current_file = open(self.wal_file, 'ab', buffering=self.WRITE_BUFFER_SIZE)

    def _close_current_file(self):
        """关闭当前WAL文件"""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def _write_to_buffer(self, data: bytes):
        """写入数据到缓冲区"""
        if len(data) + self._write_buffer_pos > self.WRITE_BUFFER_SIZE:
            self._flush_write_buffer()
        self._write_buffer[self._write_buffer_pos:self._write_buffer_pos + len(data)] = data
        self._write_buffer_pos += len(data)

    def _flush_write_buffer(self):
        """刷新写入缓冲区到文件"""
        if self._write_buffer_pos > 0:
            self._open_current_file()
            self._current_file.write(self._write_buffer[:self._write_buffer_pos])
            self._current_file.flush()  # 确保数据写入到操作系统缓冲区
            os.fsync(self._current_file.fileno())  # 确保数据写入到磁盘
            self._write_buffer_pos = 0
