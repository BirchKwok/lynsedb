import numpy as np
import jax.numpy as jnp


def to_normalize(vec: np.ndarray):
    if vec.ndim == 1:
        return vec / np.linalg.norm(vec)
    else:
        return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


def cosine_distance(vec1, vec2):
    return jnp.dot(vec2, vec1).block_until_ready()


def euclidean_distance(vec1, vec2):
    return jnp.linalg.norm(vec1 - vec2, axis=1).block_until_ready()


def argsort_topk(arr, k):
    """
    Argsort the array and return the top k indices.

    Parameters
    ----------
    arr : np.ndarray
        The array to be sorted.
    k : int
        The number of top indices to return.

    Returns
    -------
    np.ndarray
        The top k indices.
    """
    if k >= arr.size:
        return np.argsort(arr)

    indices = np.argpartition(arr, k)[:k]
    sorted_indices = indices[np.argsort(arr[indices])]
    return sorted_indices
