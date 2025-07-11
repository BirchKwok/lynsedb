from itertools import chain

import numpy as np

from lynse.core_components.io import save_nnp, load_nnp


class IVFIndex:
    """
    Inverted File Index for storing the mapping between external IDs and file names.
    """

    def __init__(self, filepath):
        """
        Initialize the IVFIndex.

        Parameters:
            filepath (str or Pathlike): The file path.
        """
        self.filepath = filepath
        self._load()

    def add_entry(self, filename, cluster_results: np.ndarray):
        """
        Add an entry to the specified cluster.

        Parameters:
            filename (str): The file name.
            cluster_results (np.ndarray): The dictionary of cluster results.

        Returns:
            None
        """
        if filename not in self.index:
            self.index[filename] = cluster_results
        else:
            return  # Entry already exists

    def get_entries(self, cluster_id, filename):
        """
        Get all entries in the specified cluster.

        Parameters:
            cluster_id (int or list): The cluster ID.
            filename (str): The file name.

        Returns:
            Dict: The dictionary of entries.
        """
        return np.isin(self.index[filename], cluster_id)

    def remove_entry(self, filename):
        """
        Remove the specified entry from the cluster.

        Parameters:
            filename (str): The file name.

        Returns:
            bool: Whether the entry is successfully removed.
        """
        isin = self.index.get(filename)

        if isin:
            del self.index[filename]
            return True

        return False

    def update_entry(self, filename, cluster_results: np.ndarray):
        """
        Update the specified entry in the cluster.

        Parameters:
            filename (str): The file name.
            cluster_results (np.ndarray): The dictionary of cluster results.

        Returns:
            None

        """
        self.index[filename] = cluster_results

    def save(self, filepath):
        """
        Save the IVFIndex to a file.

        Parameters:
            filepath (str or Pathlike): The file path.

        Returns:
            None
        """
        for filename, cluster_results in self.index.items():
            save_nnp(filepath / (filename + '.ivf'), cluster_results=cluster_results)

    def _load(self):
        """
        Load the IVFIndex from a file.
        """
        self.index = {}
        for filename in self.filepath.iterdir():
            if filename.suffix == '.ivf':
                self.index[filename] = load_nnp(filename, mmap_mode=False)

        return self

    def clear(self):
        """
        Remove all files in the IVFIndex.

        Returns:
            None
        """
        self.index = {}
        for filename in self.filepath.iterdir():
            if filename.suffix == '.ivf':
                filename.unlink()

    def all_external_ids(self):
        """
        Get all external IDs in the IVFIndex.

        Returns:
            List: The list of external IDs.
        """
        return list(chain.from_iterable(self.index.values()))

    def __len__(self):
        length = 0

        for filename, cluster_results in self.index.items():
            length += cluster_results.shape[0]

        return length
