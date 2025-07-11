import time
import uuid
from pathlib import Path
import json
from typing import Union

import numpy as np
from spinesUtils.asserts import raise_if

from ..core_components.id_mapper import IDMapper
from ..core_components.io import save_nnp, load_nnp
from ..core_components.limited_dict import LimitedDict
from ..core_components.locks import ThreadLock
from ..utils.utils import SafeMmapReader
from ..core_components.bitset import BitSet


class PersistentFileStorage:
    """The worker class for reading and writing data to the files."""

    def __init__(self, collection_path, dimension, chunk_size, warm_up=False, cache_chunks=20):
        self.collection_path = Path(collection_path)
        self.collection_name = self.collection_path.name
        self.collection_chunk_path = self.collection_path / 'chunk_data'

        self.fingerprint_path = self.collection_path / 'fingerprint'

        self.collection_chunk_path.mkdir(parents=True, exist_ok=True)

        self.id_mapper = IDMapper()

        if (self.collection_path / 'id_mapper.bin').exists():
            self.id_mapper.load(self.collection_path / 'id_mapper.bin')

        self.dimension = dimension
        self.chunk_size = chunk_size

        self.lock = ThreadLock()

        self.dataloader = DataLoader(dimension, collection_path, cache_chunks=cache_chunks, warm_up=warm_up)
        self.initialize_fingerprint()

        if warm_up:
            self.dataloader.warm_up()

    def initialize_fingerprint(self):
        with self.lock:
            if self.fingerprint_path.exists():
                with open(self.fingerprint_path, 'r') as f:
                    self.fingerprint = f.readlines()[-1].strip()
            else:
                self.fingerprint: Union[str, None] = None

    def file_exists(self):
        """Check if the file exists."""
        # 检查是否存在 chunk_0.npk 或 chunk_0 文件
        npk_file = self.collection_chunk_path / 'chunk_0.npk'
        legacy_file = self.collection_chunk_path / 'chunk_0'
        return npk_file.exists() or legacy_file.exists()

    def _return_if_in_memory(self, filename):
        return self.dataloader.return_if_in_memory(filename)

    def _write_to_memory(self, filename, data):
        self.dataloader.write_to_memory(filename, data)

    def get_all_files(self, separate=False):
        return self.dataloader.get_all_files(separate=separate)

    def read(self, filename, return_memory=True):
        """Read data from the specified filename if it exists."""
        return self.dataloader.read(filename, return_memory=return_memory)

    def mmap_read(self, filename):
        # 确保文件路径有正确的扩展名
        file_path = self.collection_chunk_path / filename

        # 检查 .npk 文件是否存在
        npk_file = file_path.with_suffix('.npk')
        if npk_file.exists():
            return self.dataloader.mmap_read(npk_file)
        # 检查无扩展名的文件（向后兼容）
        elif file_path.exists():
            return self.dataloader.mmap_read(file_path)
        else:
            raise FileNotFoundError(f"File not found: {filename} (checked both .npk and no extension)")

    def get_last_file_id(self):
        ids = []
        for i in self.collection_chunk_path.glob('chunk_*'):
            # 只处理 .npk 文件和无扩展名的文件
            if i.suffix in ['.npk', '']:
                stem = i.stem  # 去掉扩展名，例如 chunk_0.npk -> chunk_0
                if '_' in stem:
                    try:
                        file_id = int(stem.split('_')[-1])  # 获取最后一部分作为ID
                        ids.append(file_id)
                    except ValueError:
                        # 跳过无法解析为整数的文件
                        continue

        if len(ids) > 0:
            return max(ids)

        return -1

    def update_fingerprint(self):
        with self.lock:
            self.fingerprint = uuid.uuid4().hex
            with open(self.fingerprint_path, 'a') as f:
                f.write(self.fingerprint + '\n')

    def _generate_bitset(self, data_length):
        """
        为新添加的数据生成一个新的bitset

        参数:
            data_length (int): 新添加数据的长度

        返回:
            BitSet: 新生成的bitset
        """
        current_ts = int(time.time())
        bitset = BitSet(data_length)
        bitset.set_all()  # 将所有位设置为1，表示新数据在当前时间点是有效的

        # 保存bitset和对应的时间戳
        bitset_filename = f"bitset_{current_ts}.pkl"
        bitset.save_to_file(self.collection_path / bitset_filename)

        return bitset

    def _time_travel_filter(self, ts: int):
        """
        实现时间轴过滤查询

        参数:
            ts (int): 目标时间戳

        返回:
            BitSet: 过滤后的bitset
        """
        # 获取所有bitset文件
        bitset_files = sorted(self.collection_path.glob("bitset_*.pkl"),
                              key=lambda x: int(x.stem.split("_")[1]))

        result_bitset = None

        for bitset_file in bitset_files:
            file_ts = int(bitset_file.stem.split("_")[1])
            if file_ts <= ts:
                current_bitset = BitSet.load_from_file(bitset_file)
                if result_bitset is None:
                    result_bitset = current_bitset
                else:
                    result_bitset |= current_bitset  # 使用OR操作合并bitset
            else:
                break

        return result_bitset

    @staticmethod
    def _write_to_disk(data, data_path, filename, append=False):
        """写入数据到磁盘文件"""
        # 确保文件有 .npk 扩展名
        file_path = data_path / filename
        npk_file_path = file_path.with_suffix('.npk')

        if append and npk_file_path.exists():
            # 追加模式：先加载现有数据，然后合并
            try:
                existing_data = load_nnp(npk_file_path)
                # 假设文件中只有一个数组，使用第一个数组
                if existing_data:
                    array_name = list(existing_data.keys())[0]
                    existing_array = existing_data[array_name]
                    # 合并数据
                    combined_data = np.vstack([existing_array, data])
                    save_nnp(npk_file_path, **{array_name: combined_data})
                else:
                    # 如果现有文件为空，则直接保存新数据
                    save_nnp(npk_file_path, data=data)
            except Exception:
                # 如果加载失败，直接保存新数据
                save_nnp(npk_file_path, data=data)
        elif append and file_path.exists():
            # 检查是否存在旧格式的文件（向后兼容）
            try:
                existing_data = load_nnp(file_path)
                # 假设文件中只有一个数组，使用第一个数组
                if existing_data:
                    array_name = list(existing_data.keys())[0]
                    existing_array = existing_data[array_name]
                    # 合并数据
                    combined_data = np.vstack([existing_array, data])
                    save_nnp(npk_file_path, **{array_name: combined_data})
                else:
                    # 如果现有文件为空，则直接保存新数据
                    save_nnp(npk_file_path, data=data)
            except Exception:
                # 如果加载失败，直接保存新数据
                save_nnp(npk_file_path, data=data)
        else:
            # 新文件或覆盖模式
            save_nnp(npk_file_path, data=data)

    def update_id_map(self):
        """
        Update the ID mapper for the collection.
        """
        start_id = 0
        end_id = -1
        for filename in self.get_all_files():
            data_dict = self.dataloader.mmap_read(filename)
            # 从字典中获取实际的数组（假设使用 "data" 键或第一个键）
            if isinstance(data_dict, dict):
                if "data" in data_dict:
                    data_array = data_dict["data"]
                else:
                    # 如果没有 "data" 键，使用第一个数组
                    data_array = list(data_dict.values())[0]
            else:
                # 向后兼容：如果直接返回数组
                data_array = data_dict

            end_id += data_array.shape[0]
            self.id_mapper.add_entry(filename, start_id, end_id)
            start_id = end_id
        self.id_mapper.save(self.collection_path / 'id_mapper.bin')

    def _write(self, data):
        if not isinstance(data, np.ndarray):
            data = np.vstack(data)

        collection_subfile_path = self.collection_chunk_path
        file_prefix = 'chunk'

        last_file_id = self.get_last_file_id()

        # read info file to get the shape of the data
        # file shape
        # save the total shape of the data
        if not (self.collection_path / 'info.json').exists():
            total_shape = [0, self.dimension]
            with open(self.collection_path / 'info.json', 'w') as f:
                json.dump({"total_shape": total_shape}, f)
        else:
            with open(self.collection_path / 'info.json', 'r') as f:
                total_shape = json.load(f)['total_shape']

        data_shape = len(data)

        # new file
        if total_shape[0] % self.chunk_size == 0 or last_file_id == -1:
            while len(data) != 0:
                last_file_id = self.get_last_file_id()

                temp_data = data[:self.chunk_size]
                data = data[self.chunk_size:]

                filename = f'{file_prefix}_{last_file_id + 1}'
                # save data
                self._write_to_disk(temp_data, collection_subfile_path, filename, append=False)

                self._write_to_memory(filename, temp_data)
        # append data to the last file
        else:
            already_stack = False
            while len(data) != 0:
                last_file_id = self.get_last_file_id()
                # run once
                if not already_stack:
                    temp_index = self.chunk_size - (total_shape[0] % self.chunk_size)
                    temp_data = data[:temp_index]

                    data = data[temp_index:]
                    already_stack = True

                    # save data
                    filename = f'{file_prefix}_{last_file_id}'
                    self._write_to_disk(temp_data, collection_subfile_path, filename, append=True)

                    # 加载数据并从字典中提取数组
                    file_path = collection_subfile_path / filename
                    npk_file_path = file_path.with_suffix('.npk')

                    # 检查 .npk 文件是否存在
                    if npk_file_path.exists():
                        loaded_data = load_nnp(npk_file_path)
                    elif file_path.exists():
                        loaded_data = load_nnp(file_path)
                    else:
                        raise FileNotFoundError(f"File not found: {filename}")

                    if isinstance(loaded_data, dict):
                        if "data" in loaded_data:
                            temp_data = loaded_data["data"]
                        else:
                            # 如果没有 "data" 键，使用第一个数组
                            temp_data = list(loaded_data.values())[0]
                    else:
                        # 向后兼容：如果直接返回数组
                        temp_data = loaded_data

                    self._write_to_memory(filename, temp_data)
                else:
                    temp_index = min(self.chunk_size, len(data))
                    temp_data = data[:temp_index]

                    data = data[temp_index:]

                    filename = f'{file_prefix}_{last_file_id + 1}'
                    self._write_to_disk(temp_data, collection_subfile_path, filename, append=False)

                    self._write_to_memory(filename, temp_data)

        with open(self.collection_path / 'info.json', 'w') as f:
            total_shape[0] += data_shape
            json.dump({"total_shape": total_shape}, f)

    def write(self, data=None):
        """Write the data to the file."""
        with self.lock:
            self._write(data)

            # update the fingerprint
            self.update_fingerprint()
            self.update_id_map()

    def get_shape(self, read_type='all'):
        """Get the shape of the data.
        parameters:
            read_type (str): The type of data to read. Must be 'chunk' or 'all'.
        """
        return self.dataloader.get_shape(read_type=read_type)

    def clear_cache(self):
        self.dataloader.clear_cache()


class DataLoader:
    def __init__(self, dimension, collection_path, cache_chunks=20, warm_up=False):
        self.mmap_reader = SafeMmapReader()

        self.dimension = dimension
        self.collection_path = Path(collection_path)
        self.collection_chunk_path = self.collection_path / 'chunk_data'
        self.cache = LimitedDict(cache_chunks) if (cache_chunks > 0 or cache_chunks == -1) else None
        self.lock = ThreadLock()
        if warm_up:
            self.warm_up()

    def file_exists(self):
        """Check if the file exists."""
        # 检查是否存在 chunk_0.npk 或 chunk_0 文件
        npk_file = self.collection_chunk_path / 'chunk_0.npk'
        legacy_file = self.collection_chunk_path / 'chunk_0'
        return npk_file.exists() or legacy_file.exists()

    def mmap_read(self, filename):
        # 确保文件路径有正确的扩展名
        file_path = self.collection_chunk_path / filename

        # 检查 .npk 文件是否存在
        npk_file = file_path.with_suffix('.npk')
        if npk_file.exists():
            return self.mmap_reader.load_nnp(npk_file)
        # 检查无扩展名的文件（向后兼容）
        elif file_path.exists():
            return self.mmap_reader.load_nnp(file_path)
        else:
            raise FileNotFoundError(f"File not found: {filename} (checked both .npk and no extension)")

    def warm_up(self):
        """Load the data from the file to the memory."""
        with self.lock:
            if not self.file_exists():
                return

            if self.cache is not None:
                filenames = self.get_all_files()

                for idx, filename in enumerate(filenames):
                    if not self.cache.is_reached_max_size:
                        self.read(filename, return_memory=False)

    def write_to_memory(self, filename, data):
        with self.lock:
            if self.cache is not None:
                if not self.cache.is_reached_max_size:
                    self.cache[filename] = data

    def return_if_in_memory(self, filename):
        if self.cache is None:
            return None

        return self.cache.get(filename, None)

    def load_data(self, filename, data_path, update_memory=True):
        # 确保文件路径有正确的扩展名
        file_path = data_path / filename

        # 检查 .npk 文件是否存在
        npk_file = file_path.with_suffix('.npk')
        if npk_file.exists():
            loaded_data = load_nnp(npk_file)
        # 检查无扩展名的文件（向后兼容）
        elif file_path.exists():
            loaded_data = load_nnp(file_path)
        else:
            raise FileNotFoundError(f"File not found: {filename} (checked both .npk and no extension)")

        # 从字典中提取实际的数组
        if isinstance(loaded_data, dict):
            if "data" in loaded_data:
                data = loaded_data["data"]
            else:
                # 如果没有 "data" 键，使用第一个数组
                data = list(loaded_data.values())[0]
        else:
            # 向后兼容：如果直接返回数组
            data = loaded_data

        if update_memory:
            self.write_to_memory(filename, data)

        return data

    def get_all_files(self, separate=False):
        with self.lock:
            # 收集有效的文件名，只处理 .npk 文件和无扩展名的文件
            valid_files = []
            for x in self.collection_chunk_path.glob('chunk_*'):
                if x.suffix in ['.npk', ''] and x.suffix != '.lock':
                    stem = x.stem
                    if '_' in stem:
                        try:
                            # 验证文件名格式是否正确
                            int(stem.split('_')[-1])
                            valid_files.append(stem)
                        except ValueError:
                            # 跳过无法解析的文件
                            continue

            # 按文件ID排序
            filenames = sorted(valid_files, key=lambda x: int(x.split('_')[-1]))

            if separate:
                if self.cache:
                    return [filename for filename in filenames if filename in self.cache], \
                        [filename for filename in filenames if filename not in self.cache]
                return [], filenames

            return filenames

    def read(self, filename, return_memory=True, use_mmap=True):
        """Read data from the specified filename if it exists."""
        if not self.file_exists():
            return

        if not return_memory:
            data = self.mmap_read(filename) if use_mmap else self.load_data(filename, self.collection_chunk_path)
        else:
            data = self.return_if_in_memory(filename)
            if data is None:
                data = self.mmap_read(filename) if use_mmap else self.load_data(filename, self.collection_chunk_path)
        return data

    def get_shape(self, read_type='all'):
        """Get the shape of the data.
        parameters:
            read_type (str): The type of data to read. Must be 'chunk' or 'all'.
        """
        raise_if(ValueError, read_type not in ['chunk', 'all'], 'read_type must be "chunk" or "all".')

        with self.lock:
            if read_type == 'chunk':
                shape = [0, self.dimension]
                filenames = self.get_all_files()
                if filenames:
                    for filename in filenames:
                        data = self.read(filename)

                        # 处理可能返回字典的情况
                        if isinstance(data, dict):
                            if "data" in data:
                                array_data = data["data"]
                            else:
                                # 如果没有 "data" 键，使用第一个数组
                                array_data = list(data.values())[0]
                        else:
                            # 向后兼容：如果直接返回数组
                            array_data = data

                        shape[0] += len(array_data)

                return shape

            else:
                if not (self.collection_path / 'info.json').exists():
                    return [self.get_shape('chunk')[0], self.dimension]

                with open(self.collection_path / 'info.json', 'r') as f:
                    return json.load(f)['total_shape']

    def clear_cache(self):
        with self.lock:
            if self.cache is not None:
                self.cache.clear()

    def __del__(self):
        try:
            self.mmap_reader.close()
        except:
            # 在析构函数中，忽略所有错误
            pass
