import numpy as np
import cloudpickle
from spinesUtils.asserts import raise_if

from .base import BaseIndex
from ..computational_layer.engines import inner_product as ip, l2sq, cosine


class _IndexSQ(BaseIndex):
    def __init__(self):
        """
        Initialize the scalar quantizer.
        """
        super().__init__()
        self.bits = np.int8
        self.n_levels = 2 ** 7
        self.range_vals = None
        self.min_vals = None
        self.max_vals = None
        self.fitted = False

        self.data = None
        self.ids = None

    def fit(self, vectors):
        """
        Fit the model with the entire dataset to determine the global minimum and maximum range values.

        Parameters:
            vectors (np.ndarray): The input data.
        """
        self.min_vals = np.min(vectors, axis=0)
        self.max_vals = np.max(vectors, axis=0)
        self.range_vals = self.max_vals - self.min_vals
        self.fitted = True

    def encode(self, vectors):
        raise_if(ValueError, not self.fitted, 'The model must be fitted before encoding.')

        epsilon = 1e-9
        n_levels_minus_1 = self.n_levels - 1
        range_vals = self.range_vals
        min_vals = self.min_vals

        quantized = (vectors - min_vals) / (range_vals + epsilon) * n_levels_minus_1
        quantized = np.clip(quantized, -n_levels_minus_1, n_levels_minus_1).astype(self.bits)

        return quantized

    def fit_transform(self, vectors, ids):
        """
        Fit the model to the data and encode the input data to quantized vectors.

        Parameters:
            vectors (np.ndarray): The input data.
            ids (np.ndarray): The input ids.

        Returns:
            np.ndarray: The quantized vectors.
        """
        if not self.fitted:
            self.fit(vectors)

        data = self.encode(vectors)
        self.data = data if self.data is None else np.vstack((self.data, data))
        self.ids = ids if self.ids is None else np.concatenate((self.ids, ids))

    def decode(self, quantized_vectors):
        """
        Decode the quantized vectors to the original data.

        Parameters:
            quantized_vectors (np.ndarray): The quantized vectors.

        Returns:
            np.ndarray: The original data.
        """
        raise_if(ValueError, not self.fitted, 'The model must be fitted before decoding.')
        n_levels_minus_1 = self.n_levels - 1
        range_vals = self.range_vals
        min_vals = self.min_vals

        decoded = (quantized_vectors.astype(np.float32) / n_levels_minus_1) * range_vals + min_vals

        return decoded

    def _pre_select_topk(self, vec, data, ids, topk):
        """
        Pre-select the top-k indices and values.
        """
        encoded_vec = self.encode(vec)
        pre_select_ids = cosine(encoded_vec, data, n=topk, use_simd=False)[0]
        return self.decode(data[pre_select_ids]), ids[pre_select_ids]

    @staticmethod
    def _input_dtype_convert(original_vec, decoded_data):
        if original_vec.dtype != decoded_data.dtype:
            original_vec = original_vec.astype(decoded_data.dtype)
        return original_vec

    def compute_l2_distance(self, vec, mat, topk):
        """
        Compute the L2 distance between the query vector and the quantized matrix.

        Parameters:
            vec (np.ndarray): The query vector.
            mat (np.ndarray): The quantized matrix.
            topk (int): The number of nearest neighbors to return.


        Returns:
            np.ndarray: The L2 distance between the query vector and the quantized matrix.
        """
        return l2sq(vec, mat, n=topk, use_simd=True)

    def compute_inner_product(self, vec, mat, topk):
        """
        Compute the inner product between the query vector and the quantized matrix.

        Parameters:
            vec (np.ndarray): The query vector.
            mat (np.ndarray): The quantized matrix.
            topk (int): The number of nearest neighbors to return.

        Returns:
            np.ndarray: The inner product between the query vector and the quantized matrix.
        """
        return ip(self._input_dtype_convert(vec, mat), mat, n=topk, use_simd=False)

    def compute_cosine_similarity(self, vec, mat, topk):
        """
        Compute the cosine similarity between the query vector and the quantized matrix.

        Parameters:
            vec (np.ndarray): The query vector.
            mat (np.ndarray): The quantized matrix.
            topk (int): The number of nearest neighbors to return.

        Returns:
            np.ndarray: The cosine similarity between the query vector and the quantized matrix.
        """
        return cosine(self._input_dtype_convert(vec, mat), mat, n=topk, use_simd=False)

    def save(self, filepath):
        """
        Save the model to a file.

        Parameters:
            filepath (str or Pathlike): The name of the file to save the model to.
        """
        raise_if(ValueError, not self.fitted, 'The model must be fitted before saving.')
        static_vars = {
            'range_vals': self.range_vals,
            'min_vals': self.min_vals,
            'max_vals': self.max_vals,
            'fitted': self.fitted
        }
        try:
            with open(filepath, 'wb') as file:
                cloudpickle.dump(static_vars, file)
        except IOError as e:
            print(f"Error saving model: {e}")

    def save_data(self, filepath1, filepath2):
        """
        Save the data to a file.

        Parameters:
            filepath1 (str or Pathlike): The name of the file to save the data to.
            filepath2 (str or Pathlike): The name of the file to save the ids to.
        """
        if self.data is None:
            return
        try:
            with open(filepath1, 'wb') as file:
                np.save(file, self.data)
            with open(filepath2, 'wb') as file:
                np.save(file, self.ids)
        except IOError as e:
            print(f"Error saving data: {e}")

    def load(self, filepath):
        """
        Load the model from a file.

        Parameters:
            filepath (str or Pathlike): The name of the file to load the model from.

        Returns:
            _IndexSQ: The loaded model.
        """
        try:
            with open(filepath, 'rb') as file:
                static_vars = cloudpickle.load(file)

            self.fitted = static_vars['fitted']
            self.range_vals = static_vars['range_vals']
            self.min_vals = static_vars['min_vals']
            self.max_vals = static_vars['max_vals']
            return self

        except IOError as e:
            print(f"Error loading model: {e}")
            return None


class IndexSQL2sq(_IndexSQ):
    name = 'IndexSQL2sq'

    def __init__(self):
        """
        Initialize the scalar quantizer.
        """
        super().__init__()
        self._register_distance("ScalarQuantizerL2sq")

    def search(self, original_vec=None, top_k=10, rescore_multiplier=2, subset_indices=None):
        """
        Search for the nearest neighbors of the input data.

        Parameters:
            original_vec (np.ndarray): The original vector.
            top_k (int): The number of nearest neighbors to return.
            rescore_multiplier (int): The rescore multiplier.
            subset_indices (np.ndarray): The subset indices.

        Returns:
            (np.ndarray, np.ndarray): The top k indices and their distances.
        """
        encoded_vec = self.encode(original_vec)

        encoded_data = self.data
        ids = self.ids
        if subset_indices is not None:
            filtered_indices = np.isin(ids, subset_indices, assume_unique=True)
            encoded_data = encoded_data[filtered_indices]
            ids = ids[filtered_indices]

        _ids1, _ = self.compute_l2_distance(encoded_vec, encoded_data, topk=top_k * rescore_multiplier)

        decoded_data = self.decode(encoded_data[_ids1])

        _ids2, scores = self.compute_l2_distance(original_vec, decoded_data, topk=top_k)
        return ids[_ids1][_ids2], scores


class IndexSQIP(_IndexSQ):
    name = 'IndexSQIP'

    def __init__(self):
        """
        Initialize the scalar quantizer.
        """
        super().__init__()
        self._register_distance("ScalarQuantizerIP")

    def search(self, original_vec=None, top_k=10, rescore_multiplier=2, subset_indices=None):
        """
        Search for the nearest neighbors of the input data.
        """
        encoded_data = self.data
        ids = self.ids
        if subset_indices is not None:
            filtered_indices = np.isin(ids, subset_indices, assume_unique=True)
            encoded_data = encoded_data[filtered_indices]
            ids = ids[filtered_indices]


        decoded_data, ids = self._pre_select_topk(original_vec, encoded_data, ids,
                                                  top_k * rescore_multiplier)
        _ids, scores = self.compute_inner_product(original_vec, decoded_data, topk=top_k)
        return ids[_ids], scores * -1


class IndexSQCos(_IndexSQ):
    name = 'IndexSQCos'

    def __init__(self):
        """
        Initialize the scalar quantizer.
        """
        super().__init__()
        self._register_distance("ScalarQuantizerCosine")

    def search(self, original_vec=None, top_k=10, subset_indices=None):
        """
        Search for the nearest neighbors of the input data.
        """
        encoded_data = self.data
        ids = self.ids
        if subset_indices is not None:
            filtered_indices = np.isin(ids, subset_indices, assume_unique=True)
            encoded_data = encoded_data[filtered_indices]
            ids = ids[filtered_indices]

        encoded_vec = self.encode(original_vec)
        _ids, scores = self.compute_cosine_similarity(encoded_vec, encoded_data, topk=top_k)
        return ids[_ids], scores
