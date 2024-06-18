from itertools import chain

import numpy as np
import msgpack
import zlib


class IVFIndex:
    """
    Inverted File Index for storing the mapping between external IDs and file names.
    """

    def __init__(self):
        """
        Initialize the IVFIndex.
        """
        self.file_names = {}
        self.file_name_to_id = {}
        self.index = {}
        self.file_counter = 0
        self.last_external_id = 0

    def _get_file_id(self, file_name):
        """
        Get the file ID by file name.

        Parameters:
            file_name (str): The file name.

        Returns:
            int: The file ID.
        """
        if file_name in self.file_name_to_id:
            return self.file_name_to_id[file_name]

        self.file_counter += 1
        self.file_names[self.file_counter] = file_name
        self.file_name_to_id[file_name] = self.file_counter
        return self.file_counter

    def add_entry(self, cluster_id, file_name, external_id):
        """
        Add an entry to the specified cluster.

        Parameters:
            cluster_id (int): The cluster ID.
            file_name (str): The file name.
            external_id (int): The external ID.

        Returns:
            None
        """
        cluster_id = int(cluster_id)
        external_id = int(external_id)

        file_id = self._get_file_id(file_name)

        if cluster_id not in self.index:
            self.index[cluster_id] = {}

        if file_id not in self.index[cluster_id]:
            self.index[cluster_id][file_id] = set()

        # Check if the entry already exists to avoid duplicates
        if external_id in self.index[cluster_id][file_id]:
            return  # Entry already exists, do nothing

        self.index[cluster_id][file_id].add(external_id)
        self.last_external_id = external_id

    def get_entries(self, cluster_id):
        """
        Get all entries in the specified cluster.

        Parameters:
            cluster_id (int): The cluster ID.

        Returns:
            Dict: The dictionary of entries.
        """
        if cluster_id not in self.index:
            return {}

        file_entries = {}
        for file_id, external_ids in self.index[cluster_id].items():
            file_name = self.file_names[file_id]
            file_entries[file_name] = np.array(list(external_ids))

        return file_entries

    def remove_entry(self, cluster_id, external_id):
        """
        Remove the specified entry from the cluster.

        Parameters:
            cluster_id (int): The cluster ID.
            external_id (int): The external ID.

        Returns:
            bool: Whether the entry is successfully removed.
        """
        if cluster_id in self.index:
            for file_id, external_ids in self.index[cluster_id].items():
                if external_id in external_ids:
                    external_ids.remove(external_id)
                    if not external_ids:
                        del self.index[cluster_id][file_id]
                    return True
        return False

    def update_entry(self, cluster_id, external_id, new_file_name=None):
        """
        Update the specified entry in the cluster.

        Parameters:
            cluster_id (int): The cluster ID.
            external_id (int): The external ID.
            new_file_name (str): The new file name.

        Returns:
            bool: Whether the entry is successfully updated.
        """
        cluster_id = int(cluster_id)
        external_id = int(external_id)

        if cluster_id in self.index:
            for file_id, external_ids in self.index[cluster_id].items():
                if external_id in external_ids:
                    if new_file_name is not None:
                        new_file_id = self._get_file_id(new_file_name)
                        external_ids.remove(external_id)
                        if not external_ids:
                            del self.index[cluster_id][file_id]
                        if new_file_id not in self.index[cluster_id]:
                            self.index[cluster_id][new_file_id] = set()
                        self.index[cluster_id][new_file_id].add(external_id)
                    return True
        return False

    def clear_cluster(self, cluster_id):
        """
        Clear all entries in the specified cluster.

        Parameters:
            cluster_id (int): The cluster ID.

        Returns:
            None
        """
        if cluster_id in self.index:
            del self.index[cluster_id]

    def clear_all(self):
        """
        Clear all entries in the IVFIndex.
        """
        self.index.clear()
        self.file_names.clear()
        self.file_name_to_id.clear()
        self.file_counter = 0

    def save(self, file_path):
        """
        Save the IVFIndex to a file.

        Parameters:
            file_path (str or Pathlike): The file path.

        Returns:
            None
        """
        data = {
            'file_names': self.file_names,
            'index': {cluster_id: {file_id: list(external_ids) for file_id, external_ids in file_dict.items()}
                      for cluster_id, file_dict in self.index.items()},
            'file_counter': self.file_counter
        }
        packed_data = msgpack.packb(data)
        compressed_data = zlib.compress(packed_data)
        with open(file_path, 'wb') as f:
            f.write(compressed_data)

    def load(self, file_path):
        """
        Load the IVFIndex from a file.

        Parameters:
            file_path (str or Pathlike): The file path.

        Returns:
            IVFIndex: The IVFIndex object.
        """
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
            packed_data = zlib.decompress(compressed_data)
            data = msgpack.unpackb(packed_data, strict_map_key=False)
            self.file_names = data['file_names']
            self.index = {cluster_id: {file_id: set(external_ids) for file_id, external_ids in file_dict.items()}
                          for cluster_id, file_dict in data['index'].items()}
            self.file_counter = data['file_counter']

        # Rebuild the reverse mapping
        self.file_name_to_id = {name: id for id, name in self.file_names.items()}

        return self

    def all_external_ids(self):
        """
        Get all external IDs in the IVFIndex.

        Returns:
            List: The list of external IDs.
        """
        return tuple(
            chain.from_iterable(
                external_ids for file_dict in self.index.values() for external_ids in file_dict.values()
            )
        )

    def __len__(self):
        return len(self.all_external_ids())
