import os
import shutil
import tempfile
from pathlib import Path
import json

import numpy as np
from spinesUtils.asserts import raise_if

from lynse.core_components.limited_array import LimitedArray
from lynse.core_components.locks import ThreadLock
from lynse.core_components.limited_dict import LimitedDict
from lynse.utils.utils import load_chunk_file


class TemporaryFileStorage:
    """Class for handling temporary storage of data and indices with specified chunk sizes,
    ensuring paired data and indices with a sequential file ID."""

    def __init__(self, chunk_size):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chunk_size = chunk_size
        self.file_id = 0  # File ID to maintain the naming order
        self.lock = ThreadLock()

    def write_temp_data(self, data, indices, prefix='temp'):
        with self.lock:
            if not isinstance(data, np.ndarray):
                data = np.vstack(data)

            if not isinstance(indices, np.ndarray):
                indices = np.array(indices)

            num_rows = data.shape[0]
            start = 0
            file_paths = []

            while start < num_rows:
                end = min(start + self.chunk_size, num_rows)
                chunk_data = data[start:end]
                chunk_indices = indices[start:end]
                file_id = self.file_id
                self.file_id += 1  # Increment file ID for each new chunk

                data_file_path = Path(self.temp_dir.name) / f"{prefix}_data_{file_id}.npy"
                indices_file_path = Path(self.temp_dir.name) / f"{prefix}_indices_{file_id}.npy"

                with open(data_file_path, 'wb') as df, open(indices_file_path, 'wb') as inf:
                    np.save(df, chunk_data)
                    np.save(inf, chunk_indices)

                file_paths.append((data_file_path, indices_file_path))
                start += self.chunk_size

            return file_paths

    @staticmethod
    def read_temp_data(data_filepath, indices_filepath):
        return np.load(data_filepath), np.load(indices_filepath)

    def get_file_iterator(self):
        """Generate an iterator to read data and indices files sequentially."""
        with self.lock:
            file_ids = sorted([int(i.stem.split('_')[-1]) for i in Path(self.temp_dir.name).glob('temp_data_*.npy')])
            for file_id in file_ids:
                data_path = Path(self.temp_dir.name) / f"temp_data_{file_id}.npy"
                indices_path = Path(self.temp_dir.name) / f"temp_indices_{file_id}.npy"
                yield self.read_temp_data(data_path, indices_path), data_path, indices_path

    def cleanup(self):
        with self.lock:
            try:
                if os.path.exists(self.temp_dir.name):
                    shutil.rmtree(self.temp_dir.name)
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def reincarnate(self):
        with self.lock:
            self.cleanup()
            self.temp_dir = tempfile.TemporaryDirectory()
            self.file_id = 0

    def __del__(self):
        self.cleanup()


class PersistentFileStorage:
    """The worker class for reading and writing data to the files."""

    def __init__(self, collection_path, dimension, chunk_size, warm_up=False, buffer_size=20):
        self.collection_path = Path(collection_path)
        self.collection_chunk_path = self.collection_path / 'chunk_data'
        self.collection_chunk_indices_path = self.collection_path / 'chunk_indices_data'

        self.collection_chunk_path.mkdir(parents=True, exist_ok=True)
        self.collection_chunk_indices_path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.chunk_size = chunk_size

        self.quantizer = None
        self.lock = ThreadLock()

        self.dataloader = DataLoader(dimension, collection_path, buffer_size=buffer_size)

        if warm_up:
            self.dataloader.warm_up()

    def update_quantizer(self, quantizer):
        with self.lock:
            self.quantizer = quantizer

    def file_exists(self):
        """Check if the file exists."""
        return (self.collection_chunk_path / 'chunk_0').exists()

    def _return_if_in_memory(self, filename):
        return self.dataloader.return_if_in_memory(filename)

    def _write_to_memory(self, filename, data, indices):
        self.dataloader.write_to_memory(filename, data, indices)

    def _load_data(self, filename, data_path, indices_path, update_memory=True, is_mmap=True):
        return self.dataloader.load_data(filename, data_path, indices_path,
                                         update_memory=update_memory, is_mmap=is_mmap)

    def get_all_files(self, separate=True):
        return self.dataloader.get_all_files(separate=separate)

    def read(self, filename, is_mmap=True):
        """Read data from the specified filename if it exists."""
        return self.dataloader.read(filename, is_mmap=is_mmap)

    def get_last_id(self):
        ids = [int(i.stem.split('_')[-1])
               for i in self.collection_chunk_path.glob('chunk_*')]

        if len(ids) > 0:
            return max(ids)

        return -1

    @staticmethod
    def _write_to_disk(data, indices, data_path, indices_path, filename):
        with open(data_path / filename, 'wb') as f:
            np.save(f, data)

        with open(indices_path / filename, 'wb') as f:
            np.save(f, indices)

    def _incremental_write(self, data, indices):
        if not isinstance(data, np.ndarray):
            data = np.vstack(data)

        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)

        collection_subfile_path = self.collection_chunk_path
        collection_indices_path = self.collection_chunk_indices_path
        file_prefix = 'chunk'

        last_file_id = self.get_last_id()

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

        # after quantization, the disk data and memory data will be changed to quantized data
        if self.quantizer is not None:
            data = self.quantizer.fit_transform(data)

        data_shape = len(data)

        # new file
        if total_shape[0] % self.chunk_size == 0 or last_file_id == -1:
            while len(data) != 0:
                last_file_id = self.get_last_id()

                temp_data = data[:self.chunk_size]
                temp_indices = indices[:self.chunk_size]

                data = data[self.chunk_size:]
                indices = indices[self.chunk_size:]

                filename = f'{file_prefix}_{last_file_id + 1}'
                # save data
                self._write_to_disk(temp_data, temp_indices, collection_subfile_path, collection_indices_path,
                                    filename)

                self._write_to_memory(filename, temp_data, temp_indices)
        # append data to the last file
        else:
            already_stack = False
            while len(data) != 0:
                last_file_id = self.get_last_id()
                # run once
                if not already_stack:
                    temp_index = self.chunk_size - (total_shape[0] % self.chunk_size)
                    temp_data = data[:temp_index]
                    temp_indices = indices[:temp_index]

                    data = data[temp_index:]
                    indices = indices[temp_index:]
                    already_stack = True

                    # save data
                    old_data = np.load(collection_subfile_path / f'{file_prefix}_{last_file_id}')
                    temp_data = np.vstack((old_data, temp_data))
                    # save indices
                    old_indices = np.load(collection_indices_path / f'{file_prefix}_{last_file_id}')
                    temp_indices = np.concatenate((old_indices, temp_indices))

                    filename = f'{file_prefix}_{last_file_id}'
                    self._write_to_disk(temp_data, temp_indices, collection_subfile_path,
                                        collection_indices_path, filename)

                    self._write_to_memory(filename, temp_data, temp_indices)
                else:
                    temp_index = min(self.chunk_size, len(data))
                    temp_data = data[:temp_index]
                    temp_indices = indices[:temp_index]

                    data = data[temp_index:]
                    indices = indices[temp_index:]

                    filename = f'{file_prefix}_{last_file_id + 1}'
                    self._write_to_disk(temp_data, temp_indices, collection_subfile_path,
                                        collection_indices_path, filename)

                    self._write_to_memory(filename, temp_data, temp_indices)

        with open(self.collection_path / 'info.json', 'w') as f:
            total_shape[0] += data_shape
            json.dump({"total_shape": total_shape}, f)

    def _overwrite(self, filenames):
        """fit the quantizer and overwrite the data in the file."""
        if self.quantizer is not None:
            datas = []
            all_indices = {}
            limit_shape = 0
            for filename in filenames:
                _data, _idx = self.read(filename, is_mmap=False)
                limit_shape += len(_data)
                all_indices[filename] = ([limit_shape - len(_data), limit_shape], _idx)
                datas.append(_data)
                if limit_shape >= 200000:
                    break

            fitted_data = self.quantizer.fit_transform(np.vstack(datas))

            if self.dataloader.cache is not None:
                self.dataloader.clear_cache()

            for filename in filenames:
                if filename in all_indices:
                    data, indices = (fitted_data[all_indices[filename][0][0]: all_indices[filename][0][1]],
                                     all_indices[filename][1])
                else:
                    data, indices = self.read(filename, is_mmap=False)
                    data = self.quantizer.encode(data)

                self._write_to_disk(data, indices,
                                    self.collection_chunk_path, self.collection_chunk_indices_path, filename)
                self._write_to_memory(filename, data, indices)

    def read_by_idx(self, filename, idx=None):
        """Read the data from the file as a memory-mapped file."""
        return self.dataloader.read_by_idx(filename, idx=idx)

    def write(self, data=None, indices=None, filenames=None):
        """Write the data to the file."""
        with self.lock:
            if filenames:
                self._overwrite(filenames)
            else:
                self._incremental_write(data, indices=indices)

    def get_shape(self, read_type='all'):
        """Get the shape of the data.
        parameters:
            read_type (str): The type of data to read. Must be 'chunk' or 'all'.
        """
        return self.dataloader.get_shape(read_type=read_type)

    def clear_cache(self):
        self.dataloader.clear_cache()


class DataLoader:
    def __init__(self, dimension, collection_path, buffer_size=20, warm_up=False):
        self.dimension = dimension
        self.collection_path = Path(collection_path)
        self.collection_chunk_path = self.collection_path / 'chunk_data'
        self.collection_chunk_indices_path = self.collection_path / 'chunk_indices_data'
        self.cache_filenames = []
        self.cache = LimitedArray(dimension, buffer_size) if buffer_size > 0 or buffer_size == -1 else None
        self.lock = ThreadLock()
        if warm_up:
            self.warm_up()

    def file_exists(self):
        """Check if the file exists."""
        return (self.collection_chunk_path / 'chunk_0').exists()

    def warm_up(self):
        """Load the data from the file to the memory."""
        if not self.file_exists():
            return

        if self.cache is not None:
            filenames = self.get_all_files()

            for idx, filename in enumerate(filenames):
                if not self.cache.is_reached_max_size:
                    self.read(filename)

    def write_to_memory(self, filename, data, indices):
        if self.cache is not None:
            if not self.cache.is_reached_max_size:
                self.cache.add(filename, data, indices)

    def return_if_in_memory(self, filename):
        if self.cache is None:
            return None

        res = self.cache.get(filename)

        if res is None:
            return None

        return res  # data, indices

    def load_data(self, filename, data_path, indices_path, update_memory=True, is_mmap=True):
        with self.lock:
            if not is_mmap:
                data, indices = np.load(data_path / filename), np.load(indices_path / filename)
            else:
                data, indices = load_chunk_file(data_path / filename), load_chunk_file(indices_path / filename)

        if update_memory:
            self.write_to_memory(filename, np.asarray(data), np.asarray(indices))

        return data, indices

    def get_all_files(self, separate=False):
        filenames = sorted([x.stem for x in self.collection_chunk_path.glob('chunk_*')],
                           key=lambda x: int(x.split('_')[-1]))
        if separate:
            cache_filenames = []
            local_filenames = []
            for filename in filenames:
                if self.cache and filename in self.cache:
                    cache_filenames.append(filename)
                else:
                    local_filenames.append(filename)
            return [cache_filenames[0]] + local_filenames  # cache_filenames only need the first one
        return filenames

    def read(self, filename, is_mmap=True):
        """Read data from the specified filename if it exists."""
        if not self.file_exists():
            return

        return self.return_if_in_memory(filename) or self.load_data(filename, self.collection_chunk_path,
                                                                    self.collection_chunk_indices_path,
                                                                    is_mmap=is_mmap)

    def read_by_idx(self, filename, idx=None):
        """Read the data from the file as a memory-mapped file."""
        idx_filter = lambda x: np.isin(indices, idx, assume_unique=True) if idx is not None else None

        data, indices = self.read(filename, is_mmap=True)
        if not isinstance(indices, np.ndarray):
            indices = np.asarray(indices)

        inter_idx = idx_filter(indices)

        if inter_idx is None or (not np.any(inter_idx)):
            return np.array([]), np.array([])

        data = np.asarray(data[inter_idx])
        indices = indices[inter_idx]

        return data, indices

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
                        data, _ = self.read(filename)
                        shape[0] += len(data)

                return shape

            else:
                if not (self.collection_path / 'info.json').exists():
                    return [self.get_shape('chunk')[0], self.dimension]

                with open(self.collection_path / 'info.json', 'r') as f:
                    return json.load(f)['total_shape']

    def clear_cache(self):
        if self.cache is not None:
            self.cache.clear()