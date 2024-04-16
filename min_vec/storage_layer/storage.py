from pathlib import Path
import json

import numpy as np
from spinesUtils.asserts import raise_if

from min_vec.computational_layer.engines import to_normalize
from min_vec.configs.config import config
from min_vec.data_structures.limited_dict import LimitedDict


class StorageWorker:
    """The worker class for reading and writing data to the files."""

    def __init__(self, database_path, dimension, chunk_size, quantizer=None, warm_up=False):
        self.database_path = Path(database_path)
        self.database_chunk_path = self.database_path / 'chunk_data'
        self.database_chunk_indices_path = self.database_path / 'chunk_indices_data'
        self.database_chunk_fields_path = self.database_path / 'chunk_fields_data'

        self.database_cluster_path = self.database_path / 'cluster_data'
        self.database_cluster_indices_path = self.database_path / 'cluster_indices_data'
        self.database_cluster_fields_path = self.database_path / 'cluster_fields_data'

        for path in [self.database_chunk_path, self.database_chunk_indices_path,
                     self.database_chunk_fields_path, self.database_cluster_path,
                     self.database_cluster_indices_path, self.database_cluster_fields_path]:
            path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.chunk_size = chunk_size

        self.cluster_last_file_shape = {}

        self.cache = LimitedDict(max_size=config.MVDB_DATALOADER_BUFFER_SIZE)

        self.quantizer = quantizer

        if warm_up:
            self.warm_up()

    def file_exists(self):
        return ((self.database_chunk_path / 'chunk_0').exists()
                or (self.database_cluster_path / 'cluster_0_0').exists())

    def warm_up(self):
        """Load the data from the file to the memory."""
        if not self.file_exists():
            return

        for data, indices, fields in self.read():
            pass

    def _return_if_in_memory(self, filename, reverse=False):
        res = self.cache.get(filename, None)

        if res is None:
            return None

        if reverse:
            return res[0][::-1], res[1][::-1], res[2][::-1]
        return res

    def _write_to_memory(self, filename, data, indices, fields):
        self.cache[filename] = (data, indices, fields)

    def _load_data(self, filename, data_path, indices_path, fields_path, reverse, update_memory=True):
        # 文件读取逻辑
        data = np.load(data_path / filename)
        indices = np.load(indices_path / filename)
        fields = np.load(fields_path / filename)

        if update_memory:
            self._write_to_memory(filename, data, indices, fields)

        if reverse:
            return data[::-1], indices[::-1], fields[::-1]
        else:
            return data, indices, fields

    def _read(self, reverse=False, read_type='chunk', cluster_id=None):
        if not self.file_exists():
            return

        reverse_sign = -1 if reverse else 1

        if read_type == 'chunk':
            filenames = sorted([x.stem for x in self.database_chunk_path.glob('chunk_*')],
                               key=lambda x: reverse_sign * int(x.split('_')[-1]))

            for filename in filenames:
                if filename in self.cache:
                    yield self._return_if_in_memory(filename, reverse=reverse)
                else:

                    yield self._load_data(filename, self.database_chunk_path,
                                          self.database_chunk_indices_path,
                                          self.database_chunk_fields_path, reverse)
        elif read_type == 'cluster':
            if cluster_id is None:
                cluster_ids = [int(path.stem.split('_')[-2]) for path in
                               self.database_cluster_path.glob('cluster_*')]
            else:
                cluster_ids = [cluster_id]

            for cluster_id in cluster_ids:
                filenames = sorted([x.stem for x in self.database_cluster_path.glob(f'cluster_{cluster_id}_*')],
                                   key=lambda x: reverse_sign * int(x.split('_')[-1]))
                for filename in filenames:
                    if filename in self.cache:
                        yield self._return_if_in_memory(filename, reverse=reverse)
                    else:
                        yield self._load_data(filename, self.database_cluster_path,
                                              self.database_cluster_indices_path,
                                              self.database_cluster_fields_path, reverse)
        else:
            raise ValueError('read_type must be "chunk" or "cluster"')

    def chunk_read(self, reverse=False):
        """Read the data from the file."""
        yield from self._read(reverse=reverse, read_type='chunk')

    def cluster_read(self, cluster_id, reverse=False):
        """Read the data from the file."""
        yield from self._read(reverse=reverse, read_type='cluster', cluster_id=cluster_id)

    def get_last_id(self, contains='chunk', cluster_id=None):
        if contains == 'chunk':
            ids = [int(str(i).split('.mvdb')[0].split('_')[-1])
                   for i in self.database_chunk_path.glob('chunk_*')]
        else:
            ids = [int(str(i).split('.mvdb')[0].split('_')[-1])
                   for i in self.database_cluster_path.glob(f'cluster_{cluster_id}_*')]

        if len(ids) > 0:
            return max(ids)

        return -1

    @staticmethod
    def _write_to_disk(data, indices, fields, data_path, indices_path, fields_path, filename):
        with open(data_path / filename, 'wb') as f:
            np.save(f, data)

        with open(indices_path / filename, 'wb') as f:
            np.save(f, indices)

        with open(fields_path / filename, 'wb') as f:
            np.save(f, fields)

    def _write(self, data, indices, fields, write_type='chunk', cluster_id=None, normalize=False):
        if normalize:
            data = to_normalize(np.vstack(data))

        if write_type == 'chunk':
            database_subfile_path = self.database_chunk_path
            database_indices_path = self.database_chunk_indices_path
            database_fields_path = self.database_chunk_fields_path
            file_prefix = 'chunk'
        else:
            database_subfile_path = self.database_cluster_path
            database_indices_path = self.database_cluster_indices_path
            database_fields_path = self.database_cluster_fields_path

            file_prefix = f'cluster_{cluster_id}'

        last_file_id = self.get_last_id(contains=write_type, cluster_id=cluster_id)
        # read info file to get the shape of the data
        # file shape
        if write_type == 'chunk':
            if not (self.database_path / 'info.json').exists():
                total_shape = [0, self.dimension]
                with open(self.database_path / 'info.json', 'w') as f:
                    json.dump({"total_shape": total_shape}, f)
            else:
                with open(self.database_path / 'info.json', 'r') as f:
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
                    total_shape = [len(np.load(database_subfile_path / f'{file_prefix}_{last_file_id}.npy')),
                                   self.dimension]
            else:
                total_shape = self.cluster_last_file_shape[cluster_id]

        data_shape = len(data)
        # 新文件
        if total_shape[0] % self.chunk_size == 0 or last_file_id == -1:
            while len(data) != 0:
                last_file_id = self.get_last_id(contains=write_type, cluster_id=cluster_id)

                temp_data = data[:self.chunk_size]
                temp_indices = np.array(indices[:self.chunk_size])
                temp_fields = np.array(fields[:self.chunk_size])

                data = data[self.chunk_size:]
                indices = indices[self.chunk_size:]
                fields = fields[self.chunk_size:]

                filename = f'{file_prefix}_{last_file_id + 1}'
                # save data
                self._write_to_disk(temp_data, temp_indices, temp_fields, database_subfile_path, database_indices_path,
                                    database_fields_path, filename)

                self._write_to_memory(filename, temp_data, temp_indices, temp_fields)
        # 存在未满chunk_size的新文件
        else:
            data_shape = len(data)
            already_stack = False
            while len(data) != 0:
                last_file_id = self.get_last_id(contains=write_type, cluster_id=cluster_id)
                # run once
                if not already_stack:
                    temp_index = self.chunk_size - (total_shape[0] % self.chunk_size)
                    temp_data = data[:temp_index]
                    temp_indices = indices[:temp_index]
                    temp_fields = fields[:temp_index]

                    data = data[temp_index:]
                    indices = indices[temp_index:]
                    fields = fields[temp_index:]
                    already_stack = True

                    # save data
                    old_data = np.load(database_subfile_path / f'{file_prefix}_{last_file_id}')
                    temp_data = np.vstack((old_data, temp_data))
                    # save indices
                    old_indices = np.load(database_indices_path / f'{file_prefix}_{last_file_id}')
                    temp_indices = np.concatenate((old_indices, temp_indices))
                    # save fields
                    old_fields = np.load(database_fields_path / f'{file_prefix}_{last_file_id}')
                    temp_fields = np.concatenate((old_fields, temp_fields))

                    filename = f'{file_prefix}_{last_file_id}'
                    self._write_to_disk(temp_data, temp_indices, temp_fields, database_subfile_path,
                                        database_indices_path, database_fields_path, filename)

                    self._write_to_memory(filename, temp_data, temp_indices, temp_fields)
                else:
                    temp_index = min(self.chunk_size, len(data))
                    temp_data = data[:temp_index]
                    temp_indices = np.array(indices[:temp_index])
                    temp_fields = np.array(fields[:temp_index])

                    data = data[temp_index:]
                    indices = indices[temp_index:]
                    fields = fields[temp_index:]

                    filename = f'{file_prefix}_{last_file_id + 1}'
                    self._write_to_disk(temp_data, temp_indices, temp_fields, database_subfile_path,
                                        database_indices_path,
                                        database_fields_path, filename)

                    self._write_to_memory(filename, temp_data, temp_indices, temp_fields)

        if write_type == 'chunk':
            with open(self.database_path / 'info.json', 'w') as f:
                total_shape[0] += data_shape
                json.dump({"total_shape": total_shape}, f)
        else:
            self.cluster_last_file_shape[cluster_id] = [data_shape + total_shape[0], self.dimension]

    def cluster_write(self, cluster_id, data, indices, fields, normalize=False):
        """Write the data to the file."""
        self._write(data, indices, fields, write_type='cluster', cluster_id=cluster_id, normalize=normalize)

    def chunk_write(self, data, indices, fields, normalize=False):
        """Write the data to the file."""
        self._write(data, indices, fields, write_type='chunk', normalize=normalize)

    def read(self, read_type='chunk', cluster_id=None, reverse=False):
        """Read the data from the file."""
        if read_type == 'chunk':
            yield from self.chunk_read(reverse=reverse)
        elif read_type == 'cluster':
            yield from self.cluster_read(cluster_id, reverse=reverse)
        elif read_type == 'all':
            yield from self.chunk_read(reverse=reverse)
            yield from self.cluster_read(cluster_id, reverse=reverse)
        else:
            raise ValueError('read_type must be "chunk" or "cluster"')

    def write(self, data, indices, fields, write_type='chunk', cluster_id=None, normalize=False):
        """Write the data to the file."""
        if write_type == 'chunk':
            self.chunk_write(data, indices, fields, normalize=normalize)
        elif write_type == 'cluster':
            raise_if(ValueError, not isinstance(cluster_id, str) and not cluster_id.isdigit(),
                     "cluster_id must be string-type integer.")
            self.cluster_write(cluster_id, data, indices, fields, normalize=normalize)
        else:
            raise ValueError('write_type must be "chunk" or "cluster"')

    def get_shape(self, read_type='all'):
        """Get the shape of the data.
        parameters:
            read_type (str): The type of data to read. Must be 'chunk' or 'cluster' or 'all'.
        """
        if read_type == 'chunk':
            shape = [0, self.dimension]
            for data, _, _ in self.chunk_read():
                shape[0] += len(data)
            return shape

        elif read_type == 'cluster':
            shape = [0, self.dimension]
            for data, _, _ in self.cluster_read(cluster_id=None):
                shape[0] += len(data)
            return shape

        elif read_type == 'all':
            with open(self.database_path / 'info.json', 'r') as f:
                return json.load(f)['total_shape']

    def write_file_attributes(self, attributes):
        """Write the attributes to the file."""
        # use json to save the attributes
        with open(self.database_path / 'attributes.json', 'w') as f:
            json.dump(attributes, f)

    def read_file_attributes(self):
        """Read the attributes from the file."""
        # use json to read the attributes
        with open(self.database_path / 'attributes.json', 'r') as f:
            return json.load(f)

    def delete_chunk(self):
        """Delete the chunk files."""
        for file in self.database_chunk_path.glob('*'):
            file.unlink()
        for file in self.database_chunk_indices_path.glob('*'):
            file.unlink()
        for file in self.database_chunk_fields_path.glob('*'):
            file.unlink()

    def get_cluster_dataset_num(self):
        return len(list(self.database_cluster_path.glob('cluster_*')))

    def get_dataset_by_cluster_id(self, cluster_id):
        results = []

        for data, indices, fields in self.cluster_read(cluster_id):
            results.append((data, indices, fields))

        return results

    def get_chunk_dataset(self):
        results = []

        for data, indices, fields in self.chunk_read():
            results.append((data, indices, fields))

        return results

    def _modify_data(self, data, by_indices, read_type='chunk', cluster_id=None):
        raise_if(ValueError, not data.ndim == 1, 'data must be 1d array.')

        if read_type == 'chunk':
            paths = [i.name for i in self.database_chunk_path.glob('chunk_*')]
        else:
            raise_if(ValueError, cluster_id is None, 'cluster_id must be provided when read_type is cluster.')
            paths = [i.name for i in self.database_cluster_path.glob(f'cluster_{cluster_id}_*')]

        if read_type == 'cluster':
            paths = [i.name for i in self.database_cluster_path.glob('cluster_*')]

        for path in paths:
            _data, indices, fields = self._load_data(
                path,
                self.database_chunk_path if read_type == 'chunk' else self.database_cluster_path,
                self.database_chunk_indices_path if read_type == 'chunk' else self.database_cluster_indices_path,
                self.database_chunk_fields_path if read_type == 'chunk' else self.database_cluster_fields_path,
                reverse=False, update_memory=False
            )

            if by_indices not in indices:
                continue

            index = np.where(indices == by_indices)[0][0]
            _data[index] = data

            with open(self.database_chunk_path / path if read_type == 'chunk'
                      else self.database_cluster_path / path, 'wb') as f:
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
