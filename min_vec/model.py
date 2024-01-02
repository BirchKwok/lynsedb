from pathlib import Path
import os
import shutil

import numpy as np
from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert

from min_vec.utils import to_normalize
from min_vec.engine import cosine_distance, euclidean_distance


class MinVectorDB:
    """
    A class for managing a vector database stored in .mvdb files and computing vectors similarity.

    Attributes:
        dim (int): Dimension of the vectors in the database.
        dtypes (data-type): Data type of the vectors (default: np.float32).
        distance (str): Method for calculating vector distance ('cosine' or 'L2').
        database_path_parent (Path): Path to the parent directory of the database.
        database_name_prefix (str): Prefix for naming the database files.
        chunk_size (int): Size of each data chunk in the database.
        all_indices (set): A set of all indices in the database.
        last_id (int): Last ID used in the database.
        temp_file_target (str): Suffix for temporary files during operations.
        _database_chunk_path (list): Paths to the database chunks.
        chunk_id (int): Identifier for the current data chunk.
        database (np.array): The numpy array representing the database.
        indices (list): List of indices in the current database chunk.
        fields (list): List of fields associated with each vector in the chunk.
    """

    @ParameterTypeAssert({'dim': int, 'database_path': str, 'chunk_size': int}, func_name='MinVectorDB')
    @ParameterValuesAssert({
        'database_path': lambda s: s.endswith('.mvdb'),
        'distance': ('cosine', 'L2')
    }, func_name='MinVectorDB')
    def __init__(self, dim, database_path, chunk_size=10000, dtypes=np.float32, distance='cosine') -> None:
        """
        Initialize the vector database.

        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            chunk_size (int): The size of each data chunk.
            dtypes (data-type): Data type of the vectors.
            distance (str): Method for calculating vector distance. Options are 'cosine' or 'L2' for Euclidean distance.
        
        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self.dim = dim
        self.dtypes = dtypes
        self.distance = distance
        self._EMPTY_DATABASE = np.zeros((1, self.dim), dtype=self.dtypes)

        self.chunk_size = chunk_size
        self.database_path_parent = Path(database_path).parent.absolute() / Path(
            '.mvdb'.join(Path(database_path).absolute().name.split('.mvdb')[:-1]))
        self.database_name_prefix = '.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])

        if not self.database_path_parent.exists():
            self.database_path_parent.mkdir(parents=True)

        self.all_indices = set()
        # parameters for reorganize the temporary files
        self.temp_file_target = '.temp'

        # ============== Loading or create one empty database ==============
        # first of all, initialize a database
        self.reset_database()

        # If they exist, iterate through all .mvdb files.
        self._database_chunk_path = []

        for i in os.listdir(self.database_path_parent):
            # If it meets the naming convention, add it to the chunk list.
            if i.startswith(self.database_name_prefix) and Path(i).name.split('.')[0].split('_')[-1].isdigit():
                self._add_path_to_chunk_list(self.database_path_parent / i)

        # If database_chunk_path is not empty, define the loading conditions for the database and index.
        if len(self._database_chunk_path) > 0:
            self.chunk_id = max([int(Path(i).name.split('.')[0].split('_')[-1]) for i in self._database_chunk_path])

            for chunk_id, (chunk_data, index, chunk_field) in enumerate(self._data_loader()):
                self.database = chunk_data
                self.indices = index
                self.fields = chunk_field
                for idx in index:
                    self.all_indices.add(idx)
                    self.last_id = idx

            if self.database.shape[0] == self.chunk_size:
                self.reset_database()
                self.chunk_id += 1
        else:
            # If database_chunk_path is empty, define a new database and index.
            self.chunk_id = 0
            self.last_id = 0
            self.reset_database()

    def reset_database(self):
        """Reset the database to its initial state with zeros."""
        self.database = np.zeros((1, self.dim), dtype=self.dtypes)
        self.indices = []
        self.fields = []

    def _data_loader(self):
        """
        Generator that yields database chunks and corresponding indices.

        Yields:
            Tuple containing database chunk, indices in the chunk, and field information.
        """
        # By default, save the checkpoint first.
        try:
            for i in self._database_chunk_path:
                with open(i, 'rb') as f:
                    database = np.load(f)
                    index = np.load(f, allow_pickle=True)
                    field = np.load(f, allow_pickle=True)
                yield database, index, field
        except FileNotFoundError:
            raise FileNotFoundError("Did you forget to run `commit()` function ? Try to run `commit()` first.")

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
            np.save(f, self.database, allow_pickle=True)
            np.save(f, self.indices, allow_pickle=True)
            np.save(f, self.fields, allow_pickle=True)

        return database_name

    def _auto_save(self):
        """
        Automatically save the database and index to a chunk file when the current chunk is full.
        """
        if self.database.shape[0] == self.chunk_size:
            database_name = self._save()

            self.chunk_id += 1

            # Storage path for .mvdb file chunks.
            self._add_path_to_chunk_list(
                Path(self.temp_file_target.join(str(database_name).split(self.temp_file_target)[:-1]))
            )
            # Reset the database.
            self.reset_database()


    def save_checkpoint(self):
        """
        Save the current state of the database as a checkpoint.
        """
        self._auto_save()

        # After auto-saving, if the database in memory does not meet the chunk length, it still needs to be saved.
        if not self._is_database_reset():
            database_name = self._save()

            self._add_path_to_chunk_list(
                Path(self.temp_file_target.join(str(database_name).split(self.temp_file_target)[:-1]))
            )

    def commit(self):
        """
        Save the database, ensuring that all data is written to disk.
        This method is required to be called after saving vectors to query them.
        """
        # If this method is called, the part that meets the chunk size will be saved first, 
        #  and the part that does not meet the chunk size will be directly saved as the last chunk.
        self.save_checkpoint()

        # Initialize the database.
        if not self._is_database_reset():
            self.reset_database()

        for fp in self._database_chunk_path:
            os.rename(str(fp) + self.temp_file_target, fp)

    @ParameterTypeAssert({'vectors': (tuple, list)})
    @ParameterValuesAssert({'vectors': lambda s: all(1 <= len(i) <= 3 and isinstance(i, tuple) for i in s)})
    def bulk_add_items(self, vectors, normalize: bool = False):
        """
        Bulk add vectors to the database.

        Parameters:
            vectors (list or tuple): A list or tuple of vectors to be saved. Each vector can be a tuple of (vector, id, field).
            normalize (bool): Whether to normalize the input vector.

        Returns:
            list: A list of indices where the vectors are stored.
        """
        new_vectors = []
        new_ids = []
        new_fields = []

        for sample in vectors:
            if len(sample) == 1:
                vector = sample[0]
                id = None
                field = None
            elif len(sample) == 2:
                vector, id = sample
                field = None
            else:
                vector, id, field = sample

            if id is not None and id in self.all_indices:
                raise ValueError(f'id {id} already exists')

            if len(vector) != self.dim:
                raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

            if normalize:
                vector = to_normalize(vector)

            new_vectors.append(vector)
            new_ids.append(id if id is not None else self._generate_new_id())
            new_fields.append(field)

        # Convert lists to numpy arrays
        new_vectors = np.array(new_vectors, dtype=self.dtypes)

        # Update database, indices, and fields
        if self._is_database_reset():
            self.database = new_vectors
        else:
            self.database = np.vstack((self.database, new_vectors))

        self.indices.extend(new_ids)
        self.fields.extend(new_fields)
        self.all_indices.update(new_ids)

        self.save_checkpoint()

        return new_ids

    def _generate_new_id(self):
        """
        Generate a new ID for the vector.
        """
        if self.last_id is None:
            if self.all_indices:
                self.last_id = max(self.all_indices)
                self.last_id += 1
            else:
                self.last_id = 0

        return self.last_id

    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({'vector': np.ndarray, 'id': (int, None), 'field': (str, None)})
    def add_item(self, vector, id: int = None, field: str = None, normalize: bool = False) -> int:
        """
        Add a single vector to the database.

        Parameters:
            vector (np.ndarray): The vector to be added.
            id (int, optional): Optional ID for the vector.
            field (str, optional): Optional field for the vector.
            normalize (bool): Whether to normalize the input vector.

        Returns:
            int: The ID of the added vector.
        
        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """

        if id in self.all_indices:
            raise ValueError(f'id {id} already exists')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        id = self._generate_new_id() if id is None else id

        vector = to_normalize(vector) if normalize else vector

        if self._is_database_reset():
            self.database[0, :] = vector
        else:
            self.database = np.vstack((self.database, vector))

        self.indices.append(id)
        self.fields.append(field)
        self.all_indices.add(id)
        self.last_id = id

        self.save_checkpoint()

        return id

    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({
        'vector': np.ndarray, 'k': int, 'field': (None, str, list),
        'normalize': bool, 'subset_indices': (None, list)
    })
    def query(self, vector, k: int = 12, field: str | list = None, normalize: bool = False, subset_indices=None):
        """
        Query the database for the vectors most similar to the given vector.

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
        if len(self._database_chunk_path) == 0:
            raise ValueError('database is empty.')

        vector = to_normalize(vector) if normalize else vector
        vector = vector.reshape(-1, 1)

        all_scores = []
        all_index = []

        if isinstance(field, str):
            field = [field]

        for database, index, vector_field in self._data_loader():
            if field is not None:
                database = database[np.isin(vector_field, field)]
                index = index[np.isin(vector_field, field)]

            if subset_indices is not None:
                subset_indices = list(set(subset_indices))
                database = database[np.isin(index, subset_indices)]
                index = index[np.isin(index, subset_indices)]

            if len(index) == 0:
                continue

            # Distance calculation core code
            if self.distance == 'cosine':
                batch_scores = cosine_distance(database, vector).squeeze()
            else:
                batch_scores = euclidean_distance(database, vector).squeeze()

            if batch_scores.ndim == 0:
                batch_scores = [batch_scores]

            all_scores.extend(list(batch_scores))
            all_index.extend(list(index))

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

        shutil.rmtree(self.database_path_parent)

        self.reset_database()

