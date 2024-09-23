import cloudpickle
import numpy as np

from .base import BaseIndex
from ..computational_layer.engines import jaccard, hamming, inner_product


class _IndexBinary(BaseIndex):
    name = 'IndexBinary'

    def __init__(self):
        """
        Initialize the binary scalar quantizer.

        """
        super().__init__()

        self.means = None
        self.data = None
        self.ids = None

    @staticmethod
    def _convert_input(vectors):
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors)
        return vectors

    def fit(self, vectors):
        """
        Fit the model to the data.

        Parameters:
            vectors (np.ndarray): The input data.

        Returns:
            None
        """
        vectors = self._convert_input(vectors)
        self.means = np.mean(vectors, axis=0)

    def encode(self, vectors):
        """
        Encode the input data to binary vectors.

        Parameters:
            vectors (np.ndarray): The input data.

        Returns:
            np.ndarray: The binary vectors.
        """
        vectors = self._convert_input(vectors)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        binary_vectors = np.where(vectors > self.means, 1, 0)
        packed_bits = np.packbits(binary_vectors, axis=1)
        return packed_bits

    def fit_transform(self, vectors, ids):
        """
        Fit the model to the data and encode the input data to binary vectors.

        Parameters:
            vectors (np.ndarray): The input data.
            ids (np.ndarray): The input ids.

        Returns:
            np.ndarray: The binary vectors.
        """
        if self.means is None:
            self.fit(vectors)

        data = self.encode(vectors)
        self.data = data if self.data is None else np.vstack([self.data, data])
        self.ids = ids if self.ids is None else np.hstack([self.ids, ids])

    def _generate_encoded_data(self, encoded_data):
        if encoded_data is None:
            encoded_data = self.data
        return encoded_data

    def rescore(self, original_vec, top_k, decoded_sq_data, ids):
        original_vec = self._convert_input(original_vec)

        if original_vec.dtype != decoded_sq_data.dtype:
            query = original_vec.astype(decoded_sq_data.dtype)
        else:
            query = original_vec

        ip_ids, ip_score = inner_product(query, decoded_sq_data, n=top_k, use_simd=True)

        return ids[ip_ids], ip_score

    def _search(self, original_vec, original_data, encoded_vec, encoded_data, top_k,
                rescore, rescore_multiplier, subset_indices, distance):
        original_vec = self._convert_input(original_vec)
        encoded_data = self._generate_encoded_data(encoded_data)

        ids = self.ids

        if subset_indices is not None:
            filtered_indices = np.isin(self.ids, subset_indices, assume_unique=True)
            encoded_data = encoded_data[filtered_indices]
            ids = ids[filtered_indices]

        encoded_vec, encoded_data = super().check_and_encode(original_vec=original_vec, encoded_vec=encoded_vec,
                                                             original_data=original_data, encoded_data=encoded_data)

        if distance == 'Jaccard':
            distance_func = jaccard
        elif distance == 'Hamming':
            distance_func = hamming
        else:
            raise ValueError(f"Invalid distance: {distance}")

        sort_topk = top_k * rescore_multiplier if rescore else top_k
        _ids, scores = distance_func(encoded_vec, encoded_data, n=sort_topk, use_simd=True)

        if rescore:
            if subset_indices is not None:
                sq8_data = self.sq8_data[_ids]
            else:
                sq8_data = self.sq8_data

            decoded_sq_data = self.sq8_decode(sq8_data[_ids])
            return self.rescore(original_vec, top_k, decoded_sq_data, ids[_ids])

        return ids[_ids], scores if distance == 'Jaccard' else scores / original_vec.squeeze().shape[0]

    def save(self, filepath):
        """
        Save the model to a file.

        Parameters:
            filepath (str or Pathlike): The name of the file to save the model to.
        """
        static_vars = {
            'means': self.means
        }
        with open(filepath, 'wb') as f:
            cloudpickle.dump(static_vars, f)

    def save_data(self, filepath1, filepath2):
        """
        Save the data to a file.

        Parameters:
            filepath1 (str or Pathlike): The name of the file to save the data to.
            filepath2 (str or Pathlike): The name of the file to save the ids to.
        """
        if self.data is None:
            return
        with open(filepath1, 'wb') as f:
            np.save(f, self.data)
        with open(filepath2, 'wb') as f:
            np.save(f, self.ids)

    def load(self, filepath):
        """
        Load the model from a file.

        Parameters:
            filepath (str or Pathlike): The name of the file to load the model from.

        Returns:
            The loaded model.
        """
        try:
            with open(filepath, 'rb') as file:
                means = cloudpickle.load(file)
            self.means = means['means']

            return self
        except IOError as e:
            print(f"Error loading model: {e}")
            return None

    def __del__(self):
        if self.means is not None:
            del self.means

        if self.data is not None:
            del self.data

        if self.ids is not None:
            del self.ids

        if hasattr(self, 'sq8_data'):
            if self.sq8_data is not None:
                del self.sq8_data


class IndexBinaryJaccard(_IndexBinary):
    name = 'IndexBinaryJaccard'

    def __init__(self):
        """
        Initialize the binary scalar quantizer.

        """
        super().__init__()
        self._register_distance("Jaccard")

    def search(self, original_vec=None, encoded_vec=None, original_data=None,
               encoded_data=None, top_k=10, rescore=False, rescore_multiplier=2,
               subset_indices=None):
        """
        Search for the nearest neighbors of the input data.

        Parameters:
            original_vec (np.ndarray): The input data.
            encoded_vec (np.ndarray): The encoded data.
            original_data (np.ndarray): The input data.
            encoded_data (np.ndarray): The encoded data.
            top_k (int): The number of top elements to return.
            rescore (bool): Whether to rescore the results. If True, the distance will be set to InnerProduct.
            rescore_multiplier (int): The multiplier for the rescore.
            subset_indices (np.ndarray): The indices of the subset to search.

        Returns:
            (np.ndarray, np.ndarray): The top k values and their indices.
        """
        return self._search(original_vec, original_data, encoded_vec, encoded_data, top_k, rescore,
                            rescore_multiplier, subset_indices, 'Jaccard')


class IndexBinaryHamming(_IndexBinary):
    name = 'IndexBinaryHamming'

    def __init__(self):
        """
        Initialize the binary scalar quantizer.

        """
        super().__init__()
        self._register_distance("Hamming")

    def search(self, original_vec=None, encoded_vec=None, original_data=None,
               encoded_data=None, top_k=10, rescore=False, rescore_multiplier=2, subset_indices=None):
        """
        Search for the nearest neighbors of the input data.

        Parameters:
            original_vec (np.ndarray): The input data.
            encoded_vec (np.ndarray): The encoded data.
            original_data (np.ndarray): The input data.
            encoded_data (np.ndarray): The encoded data.
            top_k (int): The number of top elements to return.
            rescore (bool): Whether to rescore the results. If True, the distance will be set to InnerProduct.
            rescore_multiplier (int): The multiplier for the rescore.
            subset_indices (np.ndarray): The indices of the subset to search.

        Returns:
            (np.ndarray, np.ndarray): The top k values and their indices.
        """
        return self._search(original_vec, original_data, encoded_vec, encoded_data, top_k, rescore,
                            rescore_multiplier, subset_indices, 'Hamming')
