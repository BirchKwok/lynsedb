from pathlib import Path
import os

import numpy as np
from spinesUtils.asserts import ParameterValuesAssert, ParameterTypeAssert

from min_vec.utils import to_normalize


class MinVectorDB:
    """A class for managing a vector database stored in .npy files and computing cosine similarity."""

    @ParameterTypeAssert({'dim': int, 'database_path': str, 'chunk_size': int}, func_name='MinVectorDB')
    @ParameterValuesAssert({'database_path': lambda s: s.endswith('.npy')}, func_name='MinVectorDB')
    def __init__(self, dim, database_path, chunk_size=10000, dtypes=np.float32) -> None:
        """Initialize the vector database.
        
        Parameters:
            dim (int): Dimension of the vectors.
            database_path (str): Path to the database file.
            chunk_size (int): The size of each data chunk.
            dtypes: Data type of the vectors.
        
        Raises:
            ValueError: If `chunk_size` is less than or equal to 1.
        """
        if chunk_size <= 1:
            raise ValueError('chunk_size must be greater than 1')

        self.dim = dim
        self.dtypes = dtypes

        self.chunk_size = chunk_size
        self.database_path_parent = Path(database_path).parent
        self.database_name_prefix = '.npy'.join(Path(database_path).name.split('.npy')[:-1])
        self.all_indices = set()
        self.last_id = None

        # If they exist, iterate through all .npy files.
        self.database_chunk_path = []
        for i in os.listdir(self.database_path_parent):
            # If it meets the naming convention, add it to the chunk list.
            if i.startswith(self.database_name_prefix) and Path(i).name.split('.')[0].split('_')[-1].isdigit():
                self._added_path_to_chunk_list(self.database_path_parent / i)

        # If database_chunk_path is not empty, define the loading conditions for the database and index.
        if len(self.database_chunk_path) > 0:
            self.chunk_id = max([int(Path(i).name.split('.')[0].split('_')[-1]) for i in self.database_chunk_path])
            max_chunk_path = [i for i in self.database_chunk_path
                              if int(Path(i).name.split('.')[0].split('_')[-1]) == self.chunk_id][0]

            with open(max_chunk_path, 'rb') as f:
                self.database = np.load(f, allow_pickle=True)
                self.index = list(np.load(f, allow_pickle=True))
                self.field = list(np.load(f, allow_pickle=True))

            if self.database.shape[0] == self.chunk_size:
                self.reset_database()
                self.chunk_id += 1

            for _, index, _ in self._data_loader():
                for idx in index:
                    self.all_indices.add(idx)
        else:
            # If database_chunk_path is empty, define a new database and index.
            self.chunk_id = 0
            self.reset_database()

    def reset_database(self):
        """Reset the database to its initial state with zeros."""
        self.database = np.zeros((1, self.dim), dtype=self.dtypes)
        self.index = []
        self.field = []

    def _data_loader(self):
        """Generator that yields database chunks and corresponding indices."""
        # By default, save the checkpoint first.
        self.save_checkpoint()

        for i in self.database_chunk_path:
            with open(i, 'rb') as f:
                database = np.load(f, allow_pickle=True)
                index = np.load(f, allow_pickle=True)
                field = np.load(f, allow_pickle=True)
            yield database, index, field

    def _added_path_to_chunk_list(self, path):
        """Add a new chunk's path to the list of database chunks.
        
        Parameters:
            path: Path of the new chunk.
        """
        path = str(path)
        if path not in self.database_chunk_path:
            self.database_chunk_path.append(path)
            self.database_chunk_path = sorted(self.database_chunk_path,
                                              key=lambda s: int(Path(s).name.split('.')[0].split('_')[-1]))

    def _is_database_reset(self):
        """Check if the database is in its reset state (filled with zeros)."""
        return (np.zeros((1, self.dim), dtype=self.dtypes) == self.database).all()

    def _length_checker(self):
        """Check if the length of the database and index are equal."""
        if np.mean([self.database.shape[0], len(self.index), len(self.field)]) != self.database.shape[
            0] and not self._is_database_reset():
            raise ValueError('The database, index length and field length not the same.')

    def _save(self):
        self._length_checker()
        database_name = self.database_path_parent / f'{self.database_name_prefix}_{self.chunk_id}.npy'

        with open(database_name, 'wb') as f:
            np.save(f, self.database)
            np.save(f, self.index)
            np.save(f, self.field)

        return database_name

    def _auto_save(self):
        """Automatically save the database and index to a chunk file when the current chunk is full."""
        if self.database.shape[0] == self.chunk_size:
            database_name = self._save()

            self.chunk_id += 1

            # Storage path for .npy file chunks.
            self._added_path_to_chunk_list(database_name)
            # Reset the database.
            self.reset_database()

    def save_checkpoint(self):
        """Save the current state of the database as a checkpoint."""
        self._auto_save()

        # After auto-saving, if the database in memory does not meet the chunk length, it still needs to be saved.
        if not self._is_database_reset():
            database_name = self._save()

            self._added_path_to_chunk_list(database_name)

    def commit(self):
        """Save the database, ensuring that all data is written to disk."""
        # If this method is called, the part that meets the chunk size will be saved first, 
        #  and the part that does not meet the chunk size will be directly saved as the last chunk.
        self.save_checkpoint()

        # Initialize the database.
        if not self._is_database_reset():
            self.reset_database()

    @ParameterTypeAssert({'vectors': (tuple, list)})
    @ParameterValuesAssert({'vectors': lambda s: all(1 <= len(i) <= 3 and isinstance(i, tuple) for i in s)})
    def bulk_add_items(self, vectors, normalize: bool = False):
        """Bulk add vectors to the database.
        
        Parameters:
            vectors: A list or tuple of vectors to be saved.
            normalize (bool): whether to to_normalize the input vector
        
        Returns:
            A list of indices where the vectors are stored.
        """
        indices = []
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

            indices.append(self.add_item(vector, id, field, normalize=normalize))

        return indices

    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({'vector': np.ndarray, 'id': (int, None), 'field': (str, None)})
    def add_item(self, vector, id: int = None, field: str = None, normalize: bool = False) -> int:
        """Add a single vector to the database.
        
        Parameters:
            vector: The vector to be added.
            id (int): Optional ID for the vector.
            field (str): Optional target for the vector
            normalize (bool): whether to to_normalize the input vector
        
        Returns:
            The ID of the added vector.
        
        Raises:
            ValueError: If the vector dimensions don't match or the ID already exists.
        """

        if id in self.all_indices:
            raise ValueError(f'id {id} already exists')

        if len(vector) != self.dim:
            raise ValueError(f'vector dim error, expect {self.dim}, got {len(vector)}')

        if id is None:
            if self.last_id is None:
                id = -10000
                if self.database_chunk_path == []:
                    id = max(id, len(self.index), 0 if len(self.index) == 0 else max(self.index))
                else:
                    for _, index in self._data_loader():
                        id = max(id, max(index))

                id += 1
            else:
                id = self.last_id + 1

        vector = to_normalize(vector) if normalize else vector

        if self._is_database_reset():
            self.database[0, :] = vector
        else:
            self.database = np.vstack((self.database, vector))

        self.index.append(id)
        self.field.append(field)
        self.all_indices.add(id)
        self.last_id = id

        self.save_checkpoint()

        return id

    @ParameterValuesAssert({'vector': lambda s: s.ndim == 1})
    @ParameterTypeAssert({'vector': np.ndarray, 'k': int, 'field': (None, str, list)})
    def query(self, vector, k: int = 12, field: str | list = None, normalize: bool = False):
        """Query the database for the vectors most similar to the given vector.
        
        Parameters:
            vector: The query vector.
            k (int): The number of nearest vectors to return.
            field (str or list): The target of the vector
            normalize (bool): whether to to_normalize the input vector
        
        Returns:
            The indices and similarity scores of the top k nearest vectors.
        
        Raises:
            ValueError: If the database is empty.
        """
        self.save_checkpoint()

        if len(self.database_chunk_path) == 0:
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

            batch_scores = np.dot(database, vector).squeeze()
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
        """Return the shape of the entire database."""
        self.save_checkpoint()

        if len(self.database_chunk_path) == 0:
            return None

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
        self.save_checkpoint()

        if len(self.database_chunk_path) == 0:
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
        """Delete all .npy files in the database_chunk_path list and reset the database."""
        if len(self.database_chunk_path) == 0:
            return None

        for fp in self.database_chunk_path:
            os.remove(fp)

        self.reset_database()
