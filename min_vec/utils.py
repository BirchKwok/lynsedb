import numpy as np


def to_normalize(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

