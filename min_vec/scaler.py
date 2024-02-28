"""scaler.py - Scalar Quantization module"""


import numpy as np
import numexpr as ne
import cloudpickle

from spinesUtils.asserts import raise_if


class ScalarQuantization:
    bits_map = {8: np.uint8, 16: np.uint16, 32: np.uint32}

    def __init__(self, bits=8, decode_dtype=np.float32):
        self.range_vals = None
        if bits not in ScalarQuantization.bits_map:
            raise ValueError(f"Unsupported bits: {bits}. "
                             f"Supported bits are: {list(ScalarQuantization.bits_map.keys())}.")
        self.bits = ScalarQuantization.bits_map[bits]
        self.decode_dtype = decode_dtype
        self.n_levels = 2 ** bits
        self.min_vals = None
        self.max_vals = None

        self.fitted = False

    def partial_fit(self, vectors):
        if self.min_vals is None or self.max_vals is None:
            self.min_vals = np.min(vectors, axis=0)
            self.max_vals = np.max(vectors, axis=0)
        else:
            self.min_vals = np.minimum(self.min_vals, np.min(vectors, axis=0))
            self.max_vals = np.maximum(self.max_vals, np.max(vectors, axis=0))

        self.range_vals = self.max_vals - self.min_vals
        self.fitted = True

    def encode(self, vectors):
        epsilon = 1e-9
        n_levels_minus_1 = self.n_levels - 1
        min_vals = self.min_vals
        max_vals = self.max_vals

        # normalized = ne.evaluate("(vectors - min_vals) / (max_vals - min_vals + epsilon)")

        quantized = ne.evaluate(
            "((vectors - min_vals) / (max_vals - min_vals + epsilon)) * n_levels_minus_1",
            local_dict={'vectors': vectors, 'min_vals': min_vals, 'max_vals': max_vals,
                        'n_levels_minus_1': n_levels_minus_1, 'epsilon': epsilon}
        ).astype(self.bits)

        return quantized

    def decode(self, quantized_vectors):
        raise_if(ValueError, not self.fitted, 'The model must be fitted before decoding.')

        n_levels_minus_1 = self.n_levels - 1
        range_vals = self.range_vals  # 确保已经在类初始化或其他地方计算了 range_vals = self.max_vals - self.min_vals
        min_vals = self.min_vals

        decoded = ne.evaluate("(quantized_vectors / n_levels_minus_1) * range_vals + min_vals",
                              local_dict={'quantized_vectors': quantized_vectors, 'n_levels_minus_1': n_levels_minus_1,
                                          'range_vals': range_vals, 'min_vals': min_vals})

        if decoded.dtype != self.decode_dtype:
            decoded = decoded.astype(self.decode_dtype)

        return decoded

    def save(self, filepath):
        raise_if(ValueError, not self.fitted, 'The model must be fitted before saving.')
        try:
            with open(filepath, 'wb') as file:
                cloudpickle.dump(self, file)
        except IOError as e:
            print(f"Error saving model: {e}")

    @staticmethod
    def load(filepath):
        try:
            with open(filepath, 'rb') as file:
                return cloudpickle.load(file)
        except IOError as e:
            print(f"Error loading model: {e}")
            return None
