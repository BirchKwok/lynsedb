import uuid
from pathlib import Path
import json
from typing import Union

import numpy as np
from spinesUtils.asserts import raise_if

from ..core_components.limited_dict import LimitedDict
from ..core_components.locks import ThreadLock
from ..utils.utils import safe_mmap_reader


class PersistentFileStorage:
    """The worker class for reading and writing data to the files."""

    def __init__(self, collection_path, dimension, chunk_size, warm_up=False, cache_chunks=20):
        self.collection_path = Path(collection_path)
        self.collection_name = self.collection_path.name
        self.collection_chunk_path = self.collection_path / 'chunk_data'
        self.collection_chunk_indices_path = self.collection_path / 'chunk_ids'

        # # rename the ids folder, the version upgrade has resulted in compatibility modifications
        if (self.collection_path / 'chunk_indices_data').exists():
            (self.collection_path / 'chunk_indices_data').rename(self.collection_chunk_indices_path)

        self.fingerprint_path = self.collection_path / 'fingerprint'

        self.collection_chunk_path.mkdir(parents=True, exist_ok=True)
        self.collection_chunk_indices_path.mkdir(parents=True, exist_ok=True)

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
        return self.dataloader.file_exists()

    def _return_if_in_memory(self, filename):
        return self.dataloader.return_if_in_memory(filename)

    def _write_to_memory(self, filename, data, indices):
        self.dataloader.write_to_memory(filename, data, indices)

    def get_all_files(self, separate=False):
        return self.dataloader.get_all_files(separate=separate)

    def read(self, filename, return_memory=True):
        """Read data from the specified filename if it exists."""
        return self.dataloader.read(filename, return_memory=return_memory)

    def mmap_read(self, filename):
        return self.dataloader.mmap_read(filename)

    def get_last_id(self):
        ids = [int(i.stem.split('_')[-1])
               for i in self.collection_chunk_path.glob('chunk_*')]

        if len(ids) > 0:
            return max(ids)

        return -1

    def update_fingerprint(self):
        with self.lock:
            self.fingerprint = uuid.uuid4().hex
            with open(self.fingerprint_path, 'a') as f:
                f.write(self.fingerprint + '\n')

    @staticmethod
    def _write_to_disk(data, indices, data_path, indices_path, filename):
        with open(data_path / filename, 'wb') as f:
            np.save(f, data)

        with open(indices_path / filename, 'wb') as f:
            np.save(f, indices)

    def _write(self, data, indices):
        with self.lock:
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
                        with open(collection_subfile_path / f'{file_prefix}_{last_file_id}', 'rb') as f:
                            old_data = np.load(f)
                        temp_data = np.vstack((old_data, temp_data))
                        # save indices
                        with open(collection_indices_path / f'{file_prefix}_{last_file_id}', 'rb') as f:
                            old_indices = np.load(f)
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

    def read_by_id(self, filename, id=None):
        """Read the data from the file as a memory-mapped file."""
        return self.dataloader.read_by_id(filename, id=id)

    def read_by_only_id(self, id: Union[int, list]):
        """
        Read the data from the file by only the id.

        Parameters:
            id (Union[int, list]): The id or list of ids to read.

        Returns:
            Tuple: The data and the indices.
        """
        return self.dataloader.read_by_only_id(id=id)

    def write(self, data=None, indices=None):
        """Write the data to the file."""
        with self.lock:
            self._write(data, indices=indices)

            # update the fingerprint
            self.update_fingerprint()

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
        self.dimension = dimension
        self.collection_path = Path(collection_path)
        self.collection_chunk_path = self.collection_path / 'chunk_data'
        self.collection_chunk_indices_path = self.collection_path / 'chunk_ids'
        self.cache = LimitedDict(cache_chunks) if (cache_chunks > 0 or cache_chunks == -1) else None
        self.lock = ThreadLock()
        if warm_up:
            self.warm_up()

    def file_exists(self):
        """Check if the file exists."""
        return (self.collection_chunk_path / 'chunk_0').exists()

    def mmap_read(self, filename):
        return safe_mmap_reader(self.collection_chunk_path / filename), \
               safe_mmap_reader(self.collection_chunk_indices_path / filename)

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

    def write_to_memory(self, filename, data, indices):
        with self.lock:
            if self.cache is not None:
                if not self.cache.is_reached_max_size:
                    self.cache[filename] = (data, indices)

    def return_if_in_memory(self, filename):
        with self.lock:
            if self.cache is None:
                return None

        res = self.cache.get(filename)

        if res is None or (len(res) == 2 and res[0] is None):
            return None

        return res

    def load_data(self, filename, data_path, indices_path, update_memory=True):
        with self.lock:
            with open(data_path / filename, 'rb') as f:
                data = np.load(f)
            with open(indices_path / filename, 'rb') as f:
                indices = np.load(f)

        if update_memory:
            self.write_to_memory(filename, data, indices)

        return data, indices

    def get_all_files(self, separate=False):
        with self.lock:
            filenames = sorted([x.stem for x in self.collection_chunk_path.glob('chunk_*')],
                               key=lambda x: int(x.split('_')[-1]))
            if separate:
                if self.cache:
                    return [filename for filename in filenames if filename in self.cache], \
                        [filename for filename in filenames if filename not in self.cache]
                return [], filenames

            return filenames

    def read(self, filename, return_memory=True):
        """Read data from the specified filename if it exists."""
        with self.lock:
            if not self.file_exists():
                return

            if not return_memory:
                return self.load_data(filename, self.collection_chunk_path, self.collection_chunk_indices_path)

            return self.return_if_in_memory(filename) or self.load_data(filename, self.collection_chunk_path,
                                                                        self.collection_chunk_indices_path)

    def read_by_id(self, filename, id=None):
        """Read the data from the file as a memory-mapped file."""
        with self.lock:
            idx_filter = lambda x: np.isin(indices, id, assume_unique=True) if id is not None else None

            data, indices = self.read(filename)
            if not isinstance(indices, np.ndarray):
                indices = np.asarray(indices)

            inter_idx = idx_filter(indices)

            if inter_idx is None or (not np.any(inter_idx)):
                return np.array([]), np.array([])

            data = np.asarray(data[inter_idx])
            indices = indices[inter_idx]

            return data, indices

    def read_by_only_id(self, id: Union[int, list]):
        with self.lock:
            if isinstance(id, int):
                id = [id]

            filenames = self.get_all_files()
            data = []
            indices = []
            for filename in filenames:
                _data, _indices = self.read_by_id(filename, id)
                if len(_data) > 0:
                    data.append(_data)
                    indices.append(_indices)

            return np.vstack(data), np.concatenate(indices)

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
        with self.lock:
            if self.cache is not None:
                self.cache.clear()
