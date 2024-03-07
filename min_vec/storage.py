from contextlib import contextmanager

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

    @contextmanager
    def file_handle(self, mode='r'):
        """Context manager to open and close the file."""
        # None or closed
        if self._open_file_handler is None or not self._open_file_handler.id.valid:
            open_file_handler = h5py.File(self.database_path, mode)
        else:
            # Use the already opened file handler
            open_file_handler = self._open_file_handler

        try:
            yield open_file_handler
        finally:
            # Check if the file should be closed after use (e.g., not reused)
            if open_file_handler is not self._open_file_handler:
                open_file_handler.close()

    def get_partition_shape(self, partition_name):
        """Get the shape of the data."""
        with self.file_handle() as f:
            return f[partition_name].shape

    def chunk_read(self, reverse=False):
        """Read the data from the file."""
        with self.file_handle('a') as file:
            if 'chunk_data' not in file or 'chunk_indices' not in file or 'chunk_fields' not in file:
                return

            data_len = len(file['chunk_data'])
            for t in range((data_len // self.chunk_size) + int(data_len % self.chunk_size != 0)):
                start_scope = t * self.chunk_size if not reverse else data_len - ((t + 1) * self.chunk_size)
                end_scope = (t + 1) * self.chunk_size if not reverse else data_len - (t * self.chunk_size)
                data = file['chunk_data'][start_scope: end_scope][...]
                indices = file['chunk_indices'][start_scope: end_scope][...]
                fields = file['chunk_fields'][start_scope: end_scope][...]

                if not reverse:
                    yield data, indices, fields
                else:
                    yield data[::-1], indices[::-1], fields[::-1]

    def cluster_read(self, cluster_id, reverse=False):
        """Read the data from the file."""
        with self.file_handle('a') as file:
            if 'cluster_data' not in file or (cluster_id is not None and cluster_id not in file['cluster_data']):
                return

            if cluster_id is not None:
                cluster_ids = [cluster_id]
            else:
                cluster_ids = list(file['cluster_data'].keys())

            data_len = len(file['cluster_data'][cluster_ids[0]])
            for cluster_id in cluster_ids:
                for t in range((len(file['cluster_data'][cluster_id]) // self.chunk_size) +
                               int(len(file['cluster_data'][cluster_id]) % self.chunk_size != 0)):
                    start_scope = t * self.chunk_size if not reverse else data_len - ((t + 1) * self.chunk_size)
                    end_scope = (t + 1) * self.chunk_size if not reverse else data_len - (t * self.chunk_size)
                    # yield data, indices, fields, length must be equal to chunk_size
                    data = file['cluster_data'][cluster_id][start_scope: end_scope][...]
                    indices = file['cluster_indices'][cluster_id][start_scope: end_scope][...]
                    fields = file['cluster_fields'][cluster_id][start_scope: end_scope][...]

                    if not reverse:
                        yield data, indices, fields
                    else:
                        yield data[::-1], indices[::-1], fields[::-1]

    def chunk_write(self, data, indices, fields, data_dtype):
        """Write the data to the file."""
        with self.file_handle('a') as file:
            if 'chunk_data' not in file:
                file.create_dataset('chunk_data', (len(data), self.dimension), maxshape=(None, self.dimension),
                                    dtype=data_dtype, chunks=(self.chunk_size, self.dimension), data=np.vstack(data))
                file.create_dataset('chunk_indices', (len(data),), maxshape=(None,), dtype=np.int64,
                                    chunks=(self.chunk_size,), data=indices)
                file.create_dataset('chunk_fields', (len(data),), maxshape=(None,), dtype=np.int64,
                                    chunks=(self.chunk_size,), data=fields)
            else:
                chunk_data = file['chunk_data']
                chunk_indices = file['chunk_indices']
                chunk_fields = file['chunk_fields']

                current_len = len(chunk_data)
                new_len = current_len + len(data)
                chunk_data.resize(new_len, axis=0)
                chunk_indices.resize(new_len, axis=0)
                chunk_fields.resize(new_len, axis=0)

                chunk_data[current_len: new_len] = np.vstack(data)
                chunk_indices[current_len: new_len] = indices
                chunk_fields[current_len: new_len] = fields

    def cluster_write(self, cluster_id, data, indices, fields, data_dtype):
        """Write the data to the file."""
        with self.file_handle('a') as file:
            if 'cluster_data' not in file:
                file.create_group('cluster_data')
                file.create_group('cluster_indices')
                file.create_group('cluster_fields')

            if cluster_id not in file['cluster_data']:
                file['cluster_data'].create_dataset(cluster_id, (len(data), self.dimension),
                                                    maxshape=(None, self.dimension), dtype=data_dtype,
                                                    chunks=(self.chunk_size, self.dimension), data=np.vstack(data))
                file['cluster_indices'].create_dataset(cluster_id, (len(data),), maxshape=(None,), dtype=np.int64,
                                                       chunks=(self.chunk_size,), data=indices)
                file['cluster_fields'].create_dataset(cluster_id, (len(data),), maxshape=(None,), dtype=np.int64,
                                                      chunks=(self.chunk_size,), data=fields)
            else:
                cluster_data = file['cluster_data'][cluster_id]
                cluster_indices = file['cluster_indices'][cluster_id]
                cluster_fields = file['cluster_fields'][cluster_id]

                current_len = len(cluster_data)
                new_len = current_len + len(data)
                cluster_data.resize(new_len, axis=0)
                cluster_indices.resize(new_len, axis=0)
                cluster_fields.resize(new_len, axis=0)

                cluster_data[current_len: new_len] = np.vstack(data)
                cluster_indices[current_len: new_len] = indices
                cluster_fields[current_len: new_len] = fields

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
            raise_if(ValueError, not isinstance(cluster_id, str) and not cluster_id.isdigit(),
                     "cluster_id must be string-type integer.")
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
                    total += f['chunk_data'].shape[0]
            elif read_type == 'cluster':
                if 'cluster_data' in f:
                    for cluster_id in f['cluster_data']:
                        total += f['cluster_data'][cluster_id].shape[0]
            else:
                if 'chunk_data' in f:
                    total += f['chunk_data'].shape[0]

                if 'cluster_data' in f:
                    for cluster_id in f['cluster_data']:
                        total += f['cluster_data'][cluster_id].shape[0]

        return total, dim

    def print_file_structure(self):
        """Print the file structure."""

        def print_name(name):
            # print the name and the dataset shape
            if isinstance(self.file_handle()[name], h5py.Dataset):
                print(name, "dataset shape: ", self.file_handle()[name].shape)

        with self.file_handle() as f:
            print(f'File structure of {self.database_path}')
            print(f'Attributes: {f.attrs}')
            print(f'Keys: {list(f.keys())}')

            print("-" * 20)
            print(f'File structure of {self.database_path} start')
            f.visit(print_name)
            print("-" * 20)
            print(f'File structure of {self.database_path} end')
            print()
