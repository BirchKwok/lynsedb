import h5py
import numpy as np
from spinesUtils.asserts import raise_if


class StorageWorker:
    """A class to read and write data to a file, with optimized file handle management."""

    def __init__(self, database_path, dimension, chunk_size):
        self.database_path = database_path
        self.dimension = dimension
        self.chunk_size = chunk_size
        self._open_file_handler = None

    def file_handle(self, mode='a'):
        """Context manager to open and close the file."""
        # None or closed
        if self._open_file_handler is None or not self._open_file_handler.id.valid:
            self._open_file_handler = h5py.File(self.database_path, mode=mode)

        return self._open_file_handler

    def get_partition_shape(self, partition_name):
        """Get the shape of the data."""
        with self.file_handle() as f:
            return f[partition_name].shape

    def chunk_read(self, reverse=False):
        """Read the data from the file."""
        with self.file_handle() as file:
            if 'chunk_data' not in file:
                return

            for key in sorted(file['chunk_data'].keys(), key=lambda s: int(s) * (-1 if reverse else 1)):
                data = file[f'chunk_data/{key}']
                indices = file[f'chunk_indices/{key}']
                fields = file[f'chunk_fields/{key}']
                yield data[...], indices[...], fields[...]

    def _ensure_dataset(self, file, path, shape, maxshape, dtype, data=None, resize=False):
        """Ensure that a dataset exists and is of the correct size, creating or resizing it as necessary."""
        if path in file:
            if resize:
                file[path].resize((file[path].shape[0] + shape[0]), axis=0)
                if data is not None:
                    file[path][-shape[0]:] = data
        else:
            file.create_dataset(path, data=data, shape=shape, dtype=dtype, maxshape=maxshape)

    def _write_data(self, file, data, indices, fields, data_dtype, base_path, num, resize=False, is_cluster=False):
        """Write data, indices, and fields to the file."""
        file_name1 = f'{base_path}_data/{num}' if not is_cluster else f'{base_path[0]}/{num}'
        file_name2 = f'{base_path}_indices/{num}' if not is_cluster else f'{base_path[1]}/{num}'
        file_name3 = f'{base_path}_fields/{num}' if not is_cluster else f'{base_path[2]}/{num}'

        self._ensure_dataset(
            file, file_name1, (len(data), self.dimension), (None, self.dimension),
            data_dtype, data=data, resize=resize
        )
        self._ensure_dataset(
            file, file_name2, (len(indices),), (None,), np.int64, data=indices, resize=resize
        )
        self._ensure_dataset(
            file, file_name3, (len(fields),), (None,), np.int64, data=fields, resize=resize
        )

    def chunk_write(self, data, indices, fields, data_dtype):
        """Write the data to the file, filling up the last chunk before creating a new one."""
        with self.file_handle('a') as file:
            if 'chunk_data' not in file:
                file.create_group('chunk_data')
                file.create_group('chunk_indices')
                file.create_group('chunk_fields')

            chunk_num = len(file['chunk_data'])
            if chunk_num > 0:
                last_chunk_size = file[f'chunk_data/{chunk_num}'].shape[0]
                if last_chunk_size < self.chunk_size:
                    resize = True
                else:
                    chunk_num += 1
                    resize = False
            else:
                resize = False  # 新块，不需要resize

            while len(data) > 0:
                remaining_space = self.chunk_size - last_chunk_size if resize else self.chunk_size
                data_chunk = data[:remaining_space]
                indices_chunk = indices[:remaining_space]
                fields_chunk = fields[:remaining_space]

                self._write_data(file, data_chunk, indices_chunk, fields_chunk, data_dtype, 'chunk', chunk_num, resize)

                data = data[remaining_space:]
                indices = indices[remaining_space:]
                fields = fields[remaining_space:]

                chunk_num += 1
                resize = False  # 新创建的块之后不需要resize
                last_chunk_size = 0  # 重置为新块的初始大小

    def cluster_read(self, cluster_id=None, reverse=False):
        """Read the data from the file."""
        with self.file_handle() as file:
            if 'cluster_data' not in file:
                return

            cluster_ids = [cluster_id] if cluster_id is not None else file['cluster_data'].keys()
            if reverse:
                cluster_ids = reversed(list(cluster_ids))

            for cid in cluster_ids:
                for key in sorted(file[f'cluster_data/{cid}'].keys(), key=lambda s: int(s) * (-1 if reverse else 1)):
                    data = file[f'cluster_data/{cid}/{key}']
                    indices = file[f'cluster_indices/{cid}/{key}']
                    fields = file[f'cluster_fields/{cid}/{key}']
                    yield data[...], indices[...], fields[...]

    def cluster_write(self, cluster_id, data, indices, fields, data_dtype):
        """Write the data to the file, filling up the last cluster before creating a new one."""
        with self.file_handle('a') as file:
            base_data_path = f'cluster_data/{cluster_id}'
            base_indices_path = f'cluster_indices/{cluster_id}'
            base_fields_path = f'cluster_fields/{cluster_id}'

            file.require_group(base_data_path)
            file.require_group(base_indices_path)
            file.require_group(base_fields_path)

            cluster_num = len(file[base_data_path])
            if cluster_num > 0:
                last_cluster_size = file[f'{base_data_path}/{cluster_num}'].shape[0]
                if last_cluster_size < self.chunk_size:
                    resize = True
                else:
                    cluster_num += 1
                    resize = False
            else:
                resize = False  # 新的cluster，不需要resize

            while len(data) > 0:
                remaining_space = self.chunk_size - last_cluster_size if resize else self.chunk_size
                data_chunk = data[:remaining_space]
                indices_chunk = indices[:remaining_space]
                fields_chunk = fields[:remaining_space]

                self._write_data(file, data_chunk, indices_chunk, fields_chunk, data_dtype,
                                 [base_data_path, base_indices_path, base_fields_path],
                                 cluster_num, resize, is_cluster=True)

                data = data[remaining_space:]
                indices = indices[remaining_space:]
                fields = fields[remaining_space:]

                cluster_num += 1
                resize = False  # 新创建的cluster之后不需要resize
                last_cluster_size = 0  # 重置为新cluster的初始大小

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

    def write(self, data, indices, fields, write_type='chunk', cluster_id=None, data_dtype=np.uint8):
        """Write the data to the file."""
        if write_type == 'chunk':
            self.chunk_write(data, indices, fields, data_dtype)
        elif write_type == 'cluster':
            raise_if(ValueError, not isinstance(cluster_id, int), "cluster_id must be int.")
            self.cluster_write(cluster_id, data, indices, fields, data_dtype)
        else:
            raise ValueError('write_type must be "chunk" or "cluster"')

    def write_file_attributes(self, attributes):
        """Write the attributes to the file."""
        with self.file_handle('a') as file:
            for key, value in attributes.items():
                file.attrs[key] = value

    def read_file_attributes(self):
        """Read the attributes from the file."""
        with self.file_handle() as file:
            return dict(file.attrs)

    def delete_chunk(self):
        """Delete the chunk from the file."""
        with self.file_handle('a') as file:
            del file['chunk_data']
            del file['chunk_indices']
            del file['chunk_fields']

    def get_cluster_dataset_num(self):
        """Get the number of datasets in the cluster."""
        with self.file_handle() as file:
            if 'cluster_data' not in file:
                return 0
            return len(file['cluster_data'])

    def get_shape(self, read_type='chunk'):
        """Get the shape of the data.

        parameters:
            read_type (str): The type of data to read. Must be 'chunk' or 'cluster' or 'all'.
        """

        dim = self.dimension
        total = 0
        with self.file_handle() as f:
            if read_type == 'chunk':
                if 'chunk_data' in f:
                    for _ in f['chunk_data']:
                        total += f['chunk_data'][_].shape[0]
            elif read_type == 'cluster':
                if 'cluster_data' in f:
                    for cluster_id in f['cluster_data']:
                        for _ in f['cluster_data'][cluster_id]:
                            total += f['cluster_data'][cluster_id][_].shape[0]
            else:
                if 'chunk_data' in f:
                    for _ in f['chunk_data']:
                        total += f['chunk_data'][_].shape[0]

                if 'cluster_data' in f:
                    for cluster_id in f['cluster_data']:
                        for _ in f['cluster_data'][cluster_id]:
                            total += f['cluster_data'][cluster_id][_].shape[0]

        return total, dim

    def print_file_structure(self):
        """Print the file structure."""

        def print_name(name):
            print(name)

        with self.file_handle() as f:
            print(f'File structure of {self.database_path}')
            print(f'Attributes: {f.attrs}')
            print(f'Keys: {list(f.keys())}')
            if 'chunk_data' in f:
                print(f'Chunk data: {list(f["chunk_data"].keys())}')
            if 'cluster_data' in f:
                print(f'Cluster data: {list(f["cluster_data"].keys())}')
                for cluster_id in f['cluster_data']:
                    print(f'Cluster {cluster_id} keys: {list(f["cluster_data"][cluster_id].keys())}')

            print("-" * 20)
            print(f'File structure of {self.database_path} start')
            f.visit(print_name)
            print("-" * 20)
            print(f'File structure of {self.database_path} end')
            print()
