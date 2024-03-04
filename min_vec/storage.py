import h5py
import numpy as np
from spinesUtils.asserts import raise_if


class StorageWorker:
    """A class to read and write data to a file."""

    def __init__(self, database_path, dimension, chunk_size):
        self.database_path = database_path
        self.dimension = dimension
        self.chunk_size = chunk_size

    def get_partition_shape(self, partition_name):
        """Get the shape of the data."""
        with h5py.File(self.database_path, 'r') as f:
            return f[partition_name].shape

    def chunk_read(self, reverse=False):
        """Read the data from the file."""
        with h5py.File(self.database_path, 'r') as file:
            if 'chunk_data' not in file:
                return
            # sort the keys to ensure the order of the data/indices/fields
            for key in sorted(
                file['chunk_data'].keys(), key=lambda s: int(s) * (-1 if reverse else 1)
            ):
                data = file[f'chunk_data/{key}']
                indices = file[f'chunk_indices/{key}']
                fields = file[f'chunk_fields/{key}']
                yield data, indices, fields

    def _write_chunk(self, file, data, indices, fields, data_dtype, chunk_num, resize=False):
        """Write the chunk to the file."""
        if resize:
            file[f'chunk_data/{chunk_num}'].resize((file[f'chunk_data/{chunk_num}'].shape[0] + len(data)), axis=0)
            file[f'chunk_indices/{chunk_num}'].resize((file[f'chunk_indices/{chunk_num}'].shape[0] + len(indices)),
                                                      axis=0)
            file[f'chunk_fields/{chunk_num}'].resize((file[f'chunk_fields/{chunk_num}'].shape[0] + len(fields)), axis=0)

        file.create_dataset(f'chunk_data/{chunk_num}',
                            data=data, shape=(len(data), self.dimension), dtype=data_dtype,
                            maxshape=(None, self.dimension))
        file.create_dataset(f'chunk_indices/{chunk_num}',
                            data=indices, shape=(len(indices),), dtype=np.int64, maxshape=(None,))
        file.create_dataset(f'chunk_fields/{chunk_num}',
                            data=fields, shape=(len(fields),), dtype=np.int64, maxshape=(None,))

    def chunk_write(self, data, indices, fields, data_dtype):
        """Write the data to the file."""
        with (h5py.File(self.database_path, 'a') as file):
            if 'chunk_data' not in file:
                file.create_group('chunk_data')
                file.create_group('chunk_indices')
                file.create_group('chunk_fields')

            while len(data) > 0:
                if 'chunk_data' in file:
                    if len(file['chunk_data']) == 0:
                        last_chunk_shape = (0, self.dimension)
                    else:
                        last_chunk_shape = file[f'chunk_data/{len(file["chunk_data"])}'].shape
                else:
                    last_chunk_shape = (0, self.dimension)

                if last_chunk_shape[0] >= self.chunk_size:
                    chunk_num = len(file['chunk_data']) + 1
                    resize = True
                else:
                    chunk_num = len(file['chunk_data'])
                    resize = False

                data_chunk = data[:self.chunk_size]
                indices_chunk = indices[:self.chunk_size]
                fields_chunk = fields[:self.chunk_size]
                data = data[self.chunk_size:]
                indices = indices[self.chunk_size:]
                fields = fields[self.chunk_size:]
                self._write_chunk(file=file, data=data_chunk, indices=indices_chunk, fields=fields_chunk,
                                  data_dtype=data_dtype, chunk_num=chunk_num, resize=resize)

    def cluster_read(self, cluster_id=None, reverse=False):
        """Read the data from the file."""
        with h5py.File(self.database_path, 'r') as file:
            if 'cluster_data' not in file:
                return

            if cluster_id is None:
                cluster_ids = file['cluster_data'].keys() if not reverse else reversed(file['cluster_data'].keys())
            else:
                cluster_ids = [cluster_id]
            # sort the keys to ensure the order of the data/indices/fields
            for cluster_id in cluster_ids:
                for key in sorted(
                    file['cluster_data'][cluster_id].keys(), key=lambda s:
                    int(s) * (-1 if reverse else 1)
                ):
                    data = file[f'cluster_data/{cluster_id}/{key}']
                    indices = file[f'cluster_indices/{cluster_id}/{key}']
                    fields = file[f'cluster_fields/{cluster_id}/{key}']
                    yield data, indices, fields

    def _write_cluster(self, file, cluster_id, data, indices, fields, data_dtype, cluster_num, resize=False):
        """Write the cluster to the file."""
        if resize:
            file[f'cluster_data/{cluster_id}/{cluster_num}'].resize(
                (file[f'cluster_data/{cluster_id}/{cluster_num}'].shape[0] + len(data)), axis=0
            )
            file[f'cluster_indices/{cluster_id}/{cluster_num}'].resize(
                (file[f'cluster_indices/{cluster_id}/{cluster_num}'].shape[0] + len(indices)), axis=0
            )
            file[f'cluster_fields/{cluster_id}/{cluster_num}'].resize(
                (file[f'cluster_fields/{cluster_id}/{cluster_num}'].shape[0] + len(fields)), axis=0
            )

        file.create_dataset(f'cluster_data/{cluster_id}/{cluster_num}',
                            data=data, shape=(len(data), self.dimension), dtype=data_dtype,
                            maxshape=(None, self.dimension))
        file.create_dataset(f'cluster_indices/{cluster_id}/{cluster_num}',
                            data=indices, shape=(len(indices),), dtype=np.int64, maxshape=(None,))
        file.create_dataset(f'cluster_fields/{cluster_id}/{cluster_num}',
                            data=fields, shape=(len(fields),), dtype=np.int64, maxshape=(None,))

    def cluster_write(self, cluster_id, data, indices, fields, data_dtype):
        """Write the data to the file."""
        with h5py.File(self.database_path, 'a') as file:
            if 'cluster_data' not in file:
                file.create_group('cluster_data')
                file.create_group('cluster_indices')
                file.create_group('cluster_fields')

            while len(data) > 0:
                if f'cluster_data/{cluster_id}' in file:
                    if len(file[f'cluster_data/{cluster_id}']) == 0:
                        last_cluster_shape = (0, self.dimension)
                    else:
                        last_cluster_shape = file[(f'cluster_data/{cluster_id}/'
                                                   f'{len(file["cluster_data"][cluster_id])}')].shape
                else:
                    last_cluster_shape = (0, self.dimension)

                if last_cluster_shape[0] >= self.chunk_size:
                    cluster_num = len(file[f'cluster_data/{cluster_id}']) + 1
                    resize = True
                else:
                    cluster_num = len(file[f'cluster_data/{cluster_id}'])
                    resize = False

                data_chunk = data[:self.chunk_size]
                indices_chunk = indices[:self.chunk_size]
                fields_chunk = fields[:self.chunk_size]
                data = data[self.chunk_size:]
                indices = indices[self.chunk_size:]
                fields = fields[self.chunk_size:]
                self._write_cluster(file=file, cluster_id=cluster_id, data=data_chunk, indices=indices_chunk,
                                    fields=fields_chunk, data_dtype=data_dtype, cluster_num=cluster_num, resize=resize)

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
        with h5py.File(self.database_path, 'a') as file:
            for key, value in attributes.items():
                file.attrs[key] = value

    def read_file_attributes(self):
        """Read the attributes from the file."""
        with h5py.File(self.database_path, 'r') as file:
            return dict(file.attrs)

    def delete_chunk(self):
        """Delete the chunk from the file."""
        with h5py.File(self.database_path, 'a') as file:
            del file['chunk_data']
            del file['chunk_indices']
            del file['chunk_fields']

    def get_cluster_dataset_num(self):
        """Get the number of datasets in the cluster."""
        with h5py.File(self.database_path, 'r') as file:
            if 'cluster_data' not in file:
                return 0
            return len(file['cluster_data'])

    def get_shape(self, read_type='chunk'):
        """Get the shape of the data."""
        dim = self.dimension
        total = 0

        if not self.database_path.exists():
            return total, dim

        with h5py.File(self.database_path, 'r') as f:
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
        with h5py.File(self.database_path, 'r') as f:
            print(f'File structure of {self.database_path}')
            print(f'Attributes: {f.attrs}')
            print(f'Keys: {list(f.keys())}')
            if 'chunk_data' in f:
                print(f'Chunk data: {list(f["chunk_data"].keys())}')
            if 'cluster_data' in f:
                print(f'Cluster data: {list(f["cluster_data"].keys())}')
                for cluster_id in f['cluster_data']:
                    print(f'Cluster {cluster_id} keys: {list(f["cluster_data"][cluster_id].keys())}')
            print(f'File structure of {self.database_path} end')
            print()

