from .base import BaseIndex
from ..computational_layer.engines import inner_product as ip, l2sq, cosine as cos


class _IndexFlat(BaseIndex):
    def __init__(self):
        """
        Initialize the flat index.
        """
        super().__init__()

    def encode(self, vectors):
        """
        Encode the input data.

        Parameters:
            vectors (np.ndarray): The input data.

        Returns:
            np.ndarray: The encoded data.
        """
        return vectors

    def fit_transform(self, vectors, ids):
        """
        Fit the model and transform the data.

        Parameters:
            vectors (np.ndarray): The input data.
            ids (np.ndarray): The input ids.

        Returns:
            np.ndarray: The encoded data.
        """
        return self.encode(vectors)

    def save(self, filepath):
        """
        Save the model to a file.

        Parameters:
            filepath (str or Pathlike): The name of the file to save the model to.
        """
        ...

    def load(self, filepath):
        """
        Load the model from a file.

        Parameters:
            filepath (str or Pathlike): The name of the file to load the model from.

        Returns:
            _IndexSQ: The loaded model.
        """
        return self


class IndexFlatL2sq(_IndexFlat):
    name = 'IndexFlatL2'

    def __init__(self):
        super().__init__()
        self._register_distance("L2")

    def search(self, original_vec=None, encoded_vec=None, original_data=None,
               encoded_data=None, top_k=10):
        """
        Search for the nearest neighbors of the input data.

        Parameters:
            original_vec (np.ndarray): The original vector.
            encoded_vec (np.ndarray): The encoded vector.
            original_data (np.ndarray): The original data.
            encoded_data (np.ndarray): The encoded data.
            top_k (int): The number of nearest neighbors to return.

        Returns:
            np.ndarray: The indices of the nearest neighbors.
        """
        encoded_vec, encoded_data = super().check_and_encode(original_vec=original_vec, encoded_vec=encoded_vec,
                                                             original_data=original_data, encoded_data=encoded_data)

        return l2sq(encoded_vec, encoded_data, top_k, use_simd=False)


class IndexFlatIP(_IndexFlat):
    name = 'IndexFlatIP'

    def __init__(self):
        super().__init__()
        self._register_distance("IP")

    def search(self, original_vec=None, encoded_vec=None, original_data=None,
               encoded_data=None, top_k=10):
        """
        Search for the nearest neighbors of the input data.

        Parameters:
            original_vec (np.ndarray): The original vector.
            encoded_vec (np.ndarray): The encoded vector.
            original_data (np.ndarray): The original data.
            encoded_data (np.ndarray): The encoded data.
            top_k (int): The number of nearest neighbors to return.

        Returns:
            np.ndarray: The indices of the nearest neighbors.
        """
        encoded_vec, encoded_data = super().check_and_encode(original_vec=original_vec, encoded_vec=encoded_vec,
                                                             original_data=original_data, encoded_data=encoded_data)

        return ip(encoded_vec, encoded_data, top_k, use_simd=False)


class IndexFlatCos(_IndexFlat):
    name = 'IndexFlatCos'

    def __init__(self):
        super().__init__()
        self._register_distance("Cos")

    def search(self, original_vec=None, encoded_vec=None, original_data=None,
               encoded_data=None, top_k=10):
        """
        Search for the nearest neighbors of the input data.

        Parameters:
            original_vec (np.ndarray): The original vector.
            encoded_vec (np.ndarray): The encoded vector.
            original_data (np.ndarray): The original data.
            encoded_data (np.ndarray): The encoded data.
            top_k (int): The number of nearest neighbors to return.

        Returns:
            np.ndarray: The indices of the nearest neighbors.
        """
        encoded_vec, encoded_data = super().check_and_encode(original_vec=original_vec, encoded_vec=encoded_vec,
                                                             original_data=original_data, encoded_data=encoded_data)

        return cos(encoded_vec, encoded_data, top_k, use_simd=False)
