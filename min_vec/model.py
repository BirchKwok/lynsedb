from pathlib import Path
import os
import shutil

import numpy as np
import torch
from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert
from concurrent.futures import ThreadPoolExecutor

from min_vec.session import DatabaseSession
from min_vec.engine import cosine_distance, euclidean_distance, to_normalize, get_device
from min_vec.filter import BloomTrie
from min_vec.utils import io_assert


class MinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.

    Attributes:
        dim (int): Dimension of the vectors.
        database_path (str): Path to the database file.
        chunk_size (int): The size of each data chunk.
        dtypes (data-type): Data type of the vectors.
        distance (str): Method for calculating vector distance. Options are 'cosine' or 'L2' for Euclidean distance.
        bloom_filter_size (int): The size of the bloom filter.
        device (str): The device to use for vector operations. Options are 'auto', 'cpu', 'mps', or 'cuda'.
    """

    @ParameterTypeAssert({
        'dim': int, 'database_path': str, 'chunk_size': int, 'bloom_filter_size': int, 'device': str
    }, func_name='MinVectorDB')
    @ParameterValuesAssert({
        'database_path': lambda s: s.endswith('.mvdb'),
        'distance': ('cosine', 'L2'),
    }, func_name='MinVectorDB')
    def __init__(self, dim, database_path, chunk_size=100_000, dtypes=np.float32, distance='cosine',
                 bloom_filter_size=100_000_000, device='auto') -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            chunk_size (int): The size of each data chunk. Default is 100_000.
            dtypes (str): Data type of the vectors.
                Default is 'float32'. Options are 'float32', 'float64', 'float16', 'int32', 'int64', 'int16', 'int8'.
            distance (str): Method for calculating vector distance.
                Options are 'cosine' or 'L2' for Euclidean distance. Default is 'cosine'.
            bloom_filter_size (int): The size of the bloom filter. Default is 100_000_000.
            device (str): The device to use for vector operations.
                Options are 'auto', 'cpu', 'mps', or 'cuda'. Default is 'auto'.

        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self.dim = dim
        self.dtypes = dtypes

        self.distance = distance
        self._EMPTY_DATABASE = np.zeros((1, self.dim), dtype=self.dtypes)
        self.device = get_device(device)

        # list for add item
        self._tmp_vectors = []
        self._tmp_indices = []
        self._tmp_fields = []

        self.chunk_size = chunk_size
        self.database_path_parent = Path(database_path).parent.absolute() / Path(
            '.mvdb'.join(Path(database_path).absolute().name.split('.mvdb')[:-1]))
        self.database_name_prefix = '.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])

        if not self.database_path_parent.exists():
            self.database_path_parent.mkdir(parents=True)

        # parameters for reorganize the temporary files
        self.temp_file_target = '.temp'

        # ============== Loading or create one empty database ==============
        # first of all, initialize a database
        self.reset_database()

        # If they exist, iterate through all .mvdb files.
        self._database_chunk_path = []

        self._bloom_filter_path = self.database_path_parent / 'id_filter'

        for i in os.listdir(self.database_path_parent):
            # If it meets the naming convention, add it to the chunk list.
            if i.startswith(self.database_name_prefix) and Path(i).name.split('.')[0].split('_')[-1].isdigit():
                self._add_path_to_chunk_list(self.database_path_parent / i)

        self._id_filter = BloomTrie(bloom_size=bloom_filter_size, bloom_hash_count=5)

        # If the database is not empty, load the database and index.
        self._initialize()

        self._COMMIT_FLAG = True
        self._NEVER_COMMIT_FLAG = True

    def _initialize(self, insert=False):
        # If database_chunk_path is not empty, define the loading conditions for the database and index.
        if len(self._database_chunk_path) > 0:
            self.chunk_id = max([int(Path(i).name.split('.')[0].split('_')[-1]) for i in self._database_chunk_path])

            if self._bloom_filter_path.exists():
                self._id_filter.from_file(self._bloom_filter_path)

                if Path(self._database_chunk_path[-1]).exists():
                    last_file_path = self._database_chunk_path[-1]
                else:
                    last_file_path = Path(str(self._database_chunk_path[-1]) + self.temp_file_target)

                with open(last_file_path, 'rb') as f:
                    self.database = np.load(f, allow_pickle=True)
                    self.indices = np.load(f, allow_pickle=True).tolist()
                    self.fields = np.load(f, allow_pickle=True).tolist()
                    self.last_id = self.indices[-1]
            else:
                for chunk_id, (chunk_data, index, chunk_field) in enumerate(self._data_loader()):
                    self.database = chunk_data
                    self.indices = index.tolist()
                    for idx in index:
                        self._id_filter.add(idx)
                        self.last_id = idx
                    self.fields = chunk_field.tolist()

                # save id filter
                self._id_filter.to_file(self._bloom_filter_path)

            # last chunk file
            if self.database.shape[0] == self.chunk_size:
                self.reset_database()
                self.chunk_id += 1
            else:
                if insert:
                    # rename the last file
                    os.rename(str(self._database_chunk_path[-1]),
                              str(self._database_chunk_path[-1]) + self.temp_file_target)
        else:
            # If database_chunk_path is empty, define a new database and index.
            self.chunk_id = 0
            self.last_id = 0
            self.reset_database()

    def _check_commit(self):
        if not self._COMMIT_FLAG:
            raise ValueError("Did you forget to run `commit()` function ? Try to run `commit()` first.")

    def reset_database(self):
        """Reset the database to its initial state with zeros."""
        self.database = np.zeros((1, self.dim), dtype=self.dtypes)
        self.indices = []
        self.fields = []

    @io_assert
    def _data_loader(self, paths=None):
        """
        Generator that yields database chunks and corresponding indices.

        Parameters:
            paths (list, optional): List of paths to the database chunks.

        Yields:
            Tuple containing database chunk, indices in the chunk, and field information.
        """
        # By default, save the checkpoint first.
        paths = paths if paths is not None else self._database_chunk_path
        for i in paths:
            with open(i, 'rb') as f:
                database = np.load(f)
                index = np.load(f, allow_pickle=True)
                field = np.load(f, allow_pickle=True)

            yield database, index, field


    def _add_path_to_chunk_list(self, path):
        """
        Add a new chunk's path to the list of database chunks.

        Parameters:
            path (Path): Path of the new chunk.
        """
        path = str(path)
        if path not in self._database_chunk_path:
            self._database_chunk_path.append(path)
            self._database_chunk_path = sorted(self._database_chunk_path,
                                               key=lambda s: int(Path(s).name.split('.')[0].split('_')[-1]))

    def _is_database_reset(self):
        """
        Check if the database is in its reset state (filled with zeros).

        Returns:
            bool: True if the database is reset, False otherwise.
        """
        return (self._EMPTY_DATABASE == self.database).all()

    def _length_checker(self):
        """
        Check if the length of the database and index are equal.

        Raises:
            ValueError: If the lengths of the database, indices, and fields are not the same.
        """
        if (np.mean([self.database.shape[0], len(self.indices), len(self.fields)]) != self.database.shape[0]
                and not self._is_database_reset()):
            raise ValueError('The database, index length and field length not the same.')

    @io_assert
    def _save(self):
        """
        Save the current state of the database to a .mvdb file.

        Returns:
            Path: The path of the saved database file.
        """
        self._length_checker()
        database_name = (self.database_path_parent /
                         f'{self.database_name_prefix}_{self.chunk_id}.mvdb{self.temp_file_target}')

        with open(database_name, 'wb') as f:
            np.save(f, self.database[:self.chunk_size, :], allow_pickle=True)
            np.save(f, self.indices[:self.chunk_size], allow_pickle=True)
            np.save(f, self.fields[:self.chunk_size], allow_pickle=True)

        return database_name

    def _auto_save(self):
        """
        Automatically save the database and index to a chunk file when the current chunk is full.
        """
        while self.database.shape[0] >= self.chunk_size:
            database_name = self._save()

            if self.database.shape[0] == self.chunk_size:
                self.reset_database()
            else:
                self.database = self.database[self.chunk_size:, :]
                self.indices = self.indices[self.chunk_size:]
                self.fields = self.fields[self.chunk_size:]

            self.chunk_id += 1

            # Storage path for .mvdb file chunks.
            self._add_path_to_chunk_list(
                Path(self.temp_file_target.join(str(database_name).split(self.temp_file_target)[:-1]))
            )

    def save_checkpoint(self):
        """
        Save the current state of the database as a checkpoint.
        """
        self._auto_save()

        if not self._is_database_reset():
            # After auto-saving, if the database in memory does not meet the chunk length, it still needs to be saved.

            database_name = self._save()

            self._add_path_to_chunk_list(
                Path(self.temp_file_target.join(str(database_name).split(self.temp_file_target)[:-1]))
            )

            self.reset_database()

    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        # If this method is called, the part that meets the chunk size will be saved first,
        #  and the part that does not meet the chunk size will be directly saved as the last chunk.
        if not self._COMMIT_FLAG:
            if self._tmp_vectors:
                if self._is_database_reset():
                    self.database = np.vstack(self._tmp_vectors)
                else:
                    self.database = np.vstack((self.database, np.vstack(self._tmp_vectors)))

                self.indices.extend(self._tmp_indices)
                self.fields.extend(self._tmp_fields)
                for item in self.indices:
                    self._id_filter.add(item)

                self.save_checkpoint()

            self._tmp_vectors = []  # reset tmp_vectors
            self._tmp_indices = []  # reset tmp_indices
            self._tmp_fields = []  # reset tmp_fields

            # Initialize the database.
            if not self._is_database_reset():
                self.reset_database()

            for fp in self._database_chunk_path:
                if Path(fp + self.temp_file_target).exists():
                    os.rename(str(fp) + self.temp_file_target, fp)

            # save bloom filter
            self._id_filter.to_file(self._bloom_filter_path)

            self._COMMIT_FLAG = True
            self._NEVER_COMMIT_FLAG = False

    def _generate_new_id(self):
        """
        Generate a new ID for the vector.
        """
        while True:
            self.last_id += 1
            if self.last_id not in self._id_filter:
                break

        return self.last_id

    def _process_vector_item(self, vector, id, field, normalize):
        if id in self._id_filter:
            raise ValueError(f'id {id} already exists')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        vector = vector.astype(self.dtypes)
        if normalize:
            vector = to_normalize(vector)

        return vector, id, field

    @ParameterTypeAssert({'vectors': (tuple, list), 'normalize': bool})
    @ParameterValuesAssert({'vectors': lambda s: all(1 <= len(i) <= 3 and isinstance(i, tuple) for i in s)})
    def bulk_add_items(self, vectors, normalize: bool = False, save_immediately=False):
        """
        Bulk add vectors to the database in batches.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (vector, id, field).
            normalize (bool): Whether to normalize the input vector.
            save_immediately (bool): Whether to save the database immediately.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        if not self._NEVER_COMMIT_FLAG:
            self._initialize(insert=True)
            self._NEVER_COMMIT_FLAG = True

        batch_size = 10000
        new_ids = []

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            temp_vectors, temp_indices, temp_fields = [], [], []

            for sample in batch:
                if len(sample) == 1:
                    vector = sample[0]
                    id = None
                    field = None
                elif len(sample) == 2:
                    vector, id = sample
                    field = None
                else:
                    vector, id, field = sample

                vector, id, field = self._process_vector_item(vector, id, field, normalize)

                temp_vectors.append(vector)
                temp_indices.append(id if id is not None else self._generate_new_id())
                temp_fields.append(field)

            # Convert lists to numpy arrays
            temp_vectors = np.array(temp_vectors, dtype=self.dtypes)

            if save_immediately:
                if self._is_database_reset():
                    self.database = temp_vectors
                else:
                    self.database = np.vstack((self.database, temp_vectors))

                self.indices.extend(temp_indices)
                self.fields.extend(temp_fields)
                for item in temp_indices:
                    self._id_filter.add(item)

                self.save_checkpoint()
            else:
                self._tmp_vectors.extend(temp_vectors)
                self._tmp_indices.extend(temp_indices)
                self._tmp_fields.extend(temp_fields)

            new_ids.extend(temp_indices)

        if self._COMMIT_FLAG:
            self._COMMIT_FLAG = False

        del temp_vectors, temp_indices, temp_fields

        return new_ids

    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({
        'vector': np.ndarray, 'id': (int, None), 'field': (str, None), 'normalize': bool, 'save_immediately': bool
    })
    def add_item(self, vector, id: int = None, field: str = None, normalize: bool = False,
                 save_immediately=False) -> int:
        """
        Add a single vector to the database.

        Parameters:
            vector (np.ndarray): The vector to be added.
            id (int, optional): Optional ID for the vector.
            field (str, optional): Optional field for the vector.
            normalize (bool): Whether to normalize the input vector.
            save_immediately (bool): Whether to save the database immediately.
        Returns:
            int: The ID of the added vector.

        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """
        if not self._NEVER_COMMIT_FLAG:
            self._initialize(insert=True)
            self._NEVER_COMMIT_FLAG = True

        if id in self._id_filter:
            raise ValueError(f'id {id} already exists')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        id = self._generate_new_id() if id is None else id

        vector, id, field = self._process_vector_item(vector, id, field, normalize)

        # Add the id to then bloom filter.
        self._id_filter.add(id)

        if save_immediately:
            if self._is_database_reset():
                self.database[0, :] = vector
            else:
                self.database = np.vstack((self.database, vector))

            self.indices.append(id)
            self.fields.append(field)

            self.save_checkpoint()
        else:
            self._tmp_vectors.append(vector)
            self._tmp_indices.append(id)
            self._tmp_fields.append(field)

            if len(self._tmp_vectors) >= self.chunk_size:
                if self._is_database_reset():
                    self.database = np.vstack(self._tmp_vectors)
                else:
                    self.database = np.vstack((self.database, np.vstack(self._tmp_vectors)))

                self.indices.extend(self._tmp_indices)
                self.fields.extend(self._tmp_fields)
                self.save_checkpoint()

                self._tmp_vectors = []  # reset tmp_vectors
                self._tmp_indices = []  # reset tmp_indices
                self._tmp_fields = []  # reset tmp_fields

        if self._COMMIT_FLAG:
            self._COMMIT_FLAG = False

        return id

    def _query_chunk(self, database_chunk, index_chunk, vector, field, subset_indices, vector_field):
        """
        Query a single database chunk for the vectors most similar to the given vector.

        Parameters:
            database_chunk (np.ndarray): The database chunk to be queried.
            index_chunk (np.ndarray): The indices of the vectors in the database chunk.
            vector (np.ndarray): The query vector.
            field (str or list, optional): The target field for filtering the vectors.
            subset_indices (list, optional): The subset of indices to query.

        Returns:
            Tuple: The indices and similarity scores of the nearest vectors in the chunk.
        """
        if field is not None:
            if isinstance(field, str):
                field = [field]

            database_chunk = database_chunk[np.isin(vector_field, field)]
            index_chunk = index_chunk[np.isin(vector_field, field)]

        if subset_indices is not None:
            subset_indices = list(set(subset_indices))
            database_chunk = database_chunk[np.isin(index_chunk, subset_indices)]
            index_chunk = index_chunk[np.isin(index_chunk, subset_indices)]

        if len(index_chunk) == 0:
            return [], []

        # Distance calculation core code
        if self.distance == 'cosine':
            scores = cosine_distance(database_chunk, vector, device=self.device).squeeze()
        else:
            scores = euclidean_distance(database_chunk, vector, device=self.device).squeeze()

        if scores.ndim == 0:
            scores = [scores]

        return index_chunk, scores

    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({
        'vector': np.ndarray, 'k': int, 'field': (None, str, list),
        'normalize': bool, 'subset_indices': (None, list)
    })
    def query(self, vector, k: int = 12, field: str | list = None, normalize: bool = False, subset_indices=None):
        """
        Query the database for the vectors most similar to the given vector in batches.

        Parameters:
            vector (np.ndarray): The query vector.
            k (int): The number of nearest vectors to return.
            field (str or list, optional): The target of the vector.
            normalize (bool): Whether to normalize the input vector.
            subset_indices (list, optional): The subset of indices to query.

        Returns:
            Tuple: The indices and similarity scores of the top k nearest vectors.

        Raises:
            ValueError: If the database is empty.
        """
        self._check_commit()

        if len(self._database_chunk_path) == 0:
            raise ValueError('database is empty.')

        vector = vector.astype(self.dtypes)

        vector = to_normalize(vector) if normalize else vector
        vector = vector.reshape(-1, 1)

        all_scores = []
        all_index = []

        # 设定合适的批次大小
        batch_size = 10 if self.chunk_size > 100000 else 50

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for i in range(0, len(self._database_chunk_path), batch_size):
                batch_paths = self._database_chunk_path[i:i + batch_size]
                futures = [
                    executor.submit(self._query_chunk, database, index, vector, field, subset_indices, vector_field)
                    for database, index, vector_field in self._data_loader(batch_paths)]

                for future in futures:
                    index, scores = future.result()
                    all_scores.extend(scores)
                    all_index.extend(index)

                del batch_paths, futures

        all_scores = np.asarray(all_scores)
        all_index = np.asarray(all_index)
        top_k_indices = np.argsort(-all_scores)[:k]
        return all_index[top_k_indices], all_scores[top_k_indices]

    @property
    def shape(self):
        """
        Return the shape of the entire database.

        Returns:
            tuple: The number of vectors and the dimension of each vector in the database.
        """
        self._check_commit()

        if len(self._database_chunk_path) == 0:
            return 0, 0
        else:
            length = 0
            for database, *_ in self._data_loader():
                length += database.shape[0]

            return length, self.dim

    @ParameterTypeAssert({'n': int})
    def head(self, n=10):
        """Return the first n vectors in the database.

        Parameters:
            n (int): The number of vectors to return.

        Returns:
            The first n vectors in the database.
        """
        self._check_commit()

        if len(self._database_chunk_path) == 0:
            return None

        _database = None
        for database, *_ in self._data_loader():
            if _database is None:
                _database = database
            else:
                _database = np.vstack((_database, database))

            if _database.shape[0] >= n:
                break
        return _database[:n, :]

    def delete(self):
        """Delete all .mvdb files in the database_chunk_path list and reset the database."""
        if len(self._database_chunk_path) == 0:
            return None

        try:
            shutil.rmtree(self.database_path_parent)
        except FileNotFoundError:
            pass

        self.reset_database()

    def insert_session(self):
        """
        Create a session to insert data, which will automatically commit the data when the session ends.
        """
        return DatabaseSession(self)
