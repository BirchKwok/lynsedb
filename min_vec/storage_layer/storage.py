from pathlib import Path
import json

import numpy as np
from spinesUtils.asserts import raise_if

from min_vec.computational_layer.engines import to_normalize
from min_vec.configs.config import config
from min_vec.structures.limited_dict import LimitedDict


class StorageWorker:
    """The worker class for reading and writing data to the files."""

    def __init__(self, collection_path, dimension, chunk_size, quantizer=None, warm_up=False):
        self.collection_path = Path(collection_path)
        self.collection_chunk_path = self.collection_path / 'chunk_data'
        self.collection_chunk_indices_path = self.collection_path / 'chunk_indices_data'

        self.collection_cluster_path = self.collection_path / 'cluster_data'
        self.collection_cluster_indices_path = self.collection_path / 'cluster_indices_data'

        for path in [self.collection_chunk_path, self.collection_chunk_indices_path,
                     self.collection_cluster_path, self.collection_cluster_indices_path]:
            path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.chunk_size = chunk_size

        self.cluster_last_file_shape = {}

        self.cache = LimitedDict(max_size=config.MVDB_DATALOADER_BUFFER_SIZE)

        self.quantizer = quantizer

        if warm_up:
            self.warm_up()

    def file_exists(self):
        return ((self.collection_chunk_path / 'chunk_0').exists()
                or (self.collection_cluster_path / 'cluster_0_0').exists())

    def warm_up(self):
        """Load the data from the file to the memory."""
        if not self.file_exists():
            return

        filenames = self.get_all_files(read_type='all')

        for filename in filenames:
            data, indices = self.read(filename)
            data, indices = self.read(filename)

    def _return_if_in_memory(self, filename):
        res = self.cache.get(filename, None)

        if res is None:
            return None

        return res

    def _write_to_memory(self, filename, data, indices):
        self.cache[filename] = (data, indices)

    def _load_data(self, filename, data_path, indices_path, update_memory=True):
        # 文件读取逻辑
        data = np.load(data_path / filename)
        indices = np.load(indices_path / filename)

        if update_memory:
            self._write_to_memory(filename, data, indices)

        return data, indices

    def get_all_files(self, read_type='chunk', cluster_id=None):
        if read_type == 'chunk':
            filenames = sorted([x.stem for x in self.collection_chunk_path.glob('chunk_*')],
                               key=lambda x: int(x.split('_')[-1]))
        elif read_type == 'cluster':
            if cluster_id is None:
                filenames = sorted([x.stem for x in self.collection_cluster_path.glob('cluster_*')],
                                   key=lambda x: int(x.split('_')[-1]))
            else:
                filenames = sorted([x.stem for x in self.collection_cluster_path.glob(f'cluster_{cluster_id}_*')],
                                   key=lambda x: int(x.split('_')[-1]))
        elif read_type == 'all':
            filenames = self.get_all_files('chunk') + self.get_all_files('cluster', None)
        else:
            raise ValueError('read_type must be "chunk" or "cluster" or "all"')

        return filenames

    def _read(self, filename):
        if not self.file_exists():
            return

        if 'chunk' in filename:
            data_path = self.collection_chunk_path
            indices_path = self.collection_chunk_indices_path
        else:
            data_path = self.collection_cluster_path
            indices_path = self.collection_cluster_indices_path

        if filename in self.cache:
            return self._return_if_in_memory(filename)
        else:
            return self._load_data(filename, data_path, indices_path)

    def get_last_id(self, contains='chunk', cluster_id=None):
        if contains == 'chunk':
            ids = [int(str(i).split('_')[-1])
                   for i in self.collection_chunk_path.glob('chunk_*')]
        else:
            ids = [int(str(i).split('_')[-1])
                   for i in self.collection_cluster_path.glob(f'cluster_{cluster_id}_*')]

        if len(ids) > 0:
            return max(ids)

        return -1

    @staticmethod
    def _write_to_disk(data, indices, data_path, indices_path, filename):
        with open(data_path / filename, 'wb') as f:
            np.save(f, data)

        with open(indices_path / filename, 'wb') as f:
            np.save(f, indices)

    def _write(self, data, indices, write_type='chunk', cluster_id=None, normalize=False):
        if normalize:
            data = to_normalize(np.vstack(data))

        if write_type == 'chunk':
            collection_subfile_path = self.collection_chunk_path
            collection_indices_path = self.collection_chunk_indices_path
            file_prefix = 'chunk'
        else:
            collection_subfile_path = self.collection_cluster_path
            collection_indices_path = self.collection_cluster_indices_path

            file_prefix = f'cluster_{cluster_id}'

        last_file_id = self.get_last_id(contains=write_type, cluster_id=cluster_id)
        # read info file to get the shape of the data
        # file shape
        if write_type == 'chunk':
            if not (self.collection_path / 'info.json').exists():
                total_shape = [0, self.dimension]
                with open(self.collection_path / 'info.json', 'w') as f:
                    json.dump({"total_shape": total_shape}, f)
            else:
                with open(self.collection_path / 'info.json', 'r') as f:
                    total_shape = json.load(f)['total_shape']

            # in this class, we only use the quantizer in chunk_write, after quantization,
            # the disk data and memory data will be changed to quantized data
            if self.quantizer is not None:
                data = self.quantizer.fit_transform(data)

        else:
            if self.cluster_last_file_shape.get(cluster_id) is None:
                if last_file_id == -1:
                    total_shape = [0, self.dimension]
                else:
                    total_shape = [len(np.load(collection_subfile_path / f'{file_prefix}_{last_file_id}.npy')),
                                   self.dimension]
            else:
                total_shape = self.cluster_last_file_shape[cluster_id]

        data_shape = len(data)
        # 新文件
        if total_shape[0] % self.chunk_size == 0 or last_file_id == -1:
            while len(data) != 0:
                last_file_id = self.get_last_id(contains=write_type, cluster_id=cluster_id)

                temp_data = np.vstack(data[:self.chunk_size])
                temp_indices = np.array(indices[:self.chunk_size])

                data = data[self.chunk_size:]
                indices = indices[self.chunk_size:]

                filename = f'{file_prefix}_{last_file_id + 1}'
                # save data
                self._write_to_disk(temp_data, temp_indices, collection_subfile_path, collection_indices_path,
                                    filename)

                self._write_to_memory(filename, temp_data, temp_indices)
        # 存在未满chunk_size的新文件
        else:
            data_shape = len(data)
            already_stack = False
            while len(data) != 0:
                last_file_id = self.get_last_id(contains=write_type, cluster_id=cluster_id)
                # run once
                if not already_stack:
                    temp_index = self.chunk_size - (total_shape[0] % self.chunk_size)
                    temp_data = np.vstack(data[:temp_index])
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
                    temp_data = np.vstack(data[:temp_index])
                    temp_indices = np.array(indices[:temp_index])

                    data = data[temp_index:]
                    indices = indices[temp_index:]

                    filename = f'{file_prefix}_{last_file_id + 1}'
                    self._write_to_disk(temp_data, temp_indices, collection_subfile_path,
                                        collection_indices_path, filename)

                    self._write_to_memory(filename, temp_data, temp_indices)

        if write_type == 'chunk':
            with open(self.collection_path / 'info.json', 'w') as f:
                total_shape[0] += data_shape
                json.dump({"total_shape": total_shape}, f)
        else:
            self.cluster_last_file_shape[cluster_id] = [data_shape + total_shape[0], self.dimension]

    def cluster_write(self, cluster_id, data, indices, normalize=False):
        """Write the data to the file."""
        self._write(data, indices, write_type='cluster', cluster_id=cluster_id, normalize=normalize)

    def chunk_write(self, data, indices, normalize=False):
        """Write the data to the file."""
        self._write(data, indices, write_type='chunk', normalize=normalize)

    def read(self, filename):
        """Read the data from the file."""
        return self._read(filename=filename)

    def write(self, data, indices, write_type='chunk', cluster_id=None, normalize=False):
        """Write the data to the file."""
        if write_type == 'chunk':
            self.chunk_write(data, indices, normalize=normalize)
        elif write_type == 'cluster':
            raise_if(ValueError, not isinstance(cluster_id, str) and not cluster_id.isdigit(),
                     "cluster_id must be string-type integer.")
            self.cluster_write(cluster_id, data, indices, normalize=normalize)
        else:
            raise ValueError('write_type must be "chunk" or "cluster"')

    def get_shape(self, read_type='all'):
        """Get the shape of the data.
        parameters:
            read_type (str): The type of data to read. Must be 'chunk' or 'cluster' or 'all'.
        """
        if read_type == 'chunk':
            shape = [0, self.dimension]
            filenames = self.get_all_files(read_type='chunk')
            if filenames:
                for filename in filenames:
                    data, _ = self.read(filename)
                    shape[0] += len(data)

            return shape

        elif read_type == 'cluster':
            shape = [0, self.dimension]

            filenames = self.get_all_files(read_type='cluster')
            if filenames:
                for filename in filenames:
                    data, _ = self.read(filename)
                    shape[0] += len(data)

            return shape

        elif read_type == 'all':
            if not (self.collection_path / 'info.json').exists():
                return [self.get_shape('chunk')[0] + self.get_shape('cluster')[0], self.dimension]

            with open(self.collection_path / 'info.json', 'r') as f:
                return json.load(f)['total_shape']

    def delete_chunk(self):
        """Delete the chunk files."""
        for file in self.collection_chunk_path.glob('*'):
            file.unlink()
        for file in self.collection_chunk_indices_path.glob('*'):
            file.unlink()

    def get_cluster_dataset_num(self):
        return len(list(self.collection_cluster_path.glob('cluster_*')))

    def get_dataset_by_cluster_id(self, cluster_id):
        results = []

        filenames = self.get_all_files(read_type='cluster', cluster_id=cluster_id)

        for filename in filenames:
            results.append(self.read(filename))

        return results

    def get_chunk_dataset(self):
        results = []

        filenames = self.get_all_files(read_type='chunk')
        for filename in filenames:
            results.append(self.read(filename))

        return results

    def _modify_data(self, data, by_indices, read_type='chunk', cluster_id=None):
        raise_if(ValueError, not data.ndim == 1, 'data must be 1d array.')

        if read_type == 'chunk':
            paths = [i.name for i in self.collection_chunk_path.glob('chunk_*')]
        else:
            raise_if(ValueError, cluster_id is None, 'cluster_id must be provided when read_type is cluster.')
            paths = [i.name for i in self.collection_cluster_path.glob(f'cluster_{cluster_id}_*')]

        if read_type == 'cluster':
            paths = [i.name for i in self.collection_cluster_path.glob('cluster_*')]

        for path in paths:
            _data, indices = self._load_data(
                path,
                self.collection_chunk_path if read_type == 'chunk' else self.collection_cluster_path,
                self.collection_chunk_indices_path if read_type == 'chunk' else self.collection_cluster_indices_path,
                update_memory=False
            )

            if by_indices not in indices:
                continue

            index = np.where(indices == by_indices)[0][0]
            _data[index] = data

            with open(self.collection_chunk_path / path if read_type == 'chunk'
                      else self.collection_cluster_path / path, 'wb') as f:
                np.save(f, _data)

            # delete cache
            self.cache.pop(path, None)

            break

        return

    def modify_cluster_data(self, cluster_id, data, by_indices):
        self._modify_data(data, by_indices, read_type='cluster', cluster_id=cluster_id)

    def modify_chunk_data(self, data, by_indices):
        self._modify_data(data, by_indices, read_type='chunk')

    def clear_cache(self):
        self.cache.clear()
