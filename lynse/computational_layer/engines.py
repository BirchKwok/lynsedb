import numpy as np
import jax.numpy as jnp
from jax import jit

import simsimd
from usearch.compiled import exact_search
from usearch.index import MetricKind

from ..core_components.fast_sort import FastSort


threads = 0


def to_normalize(vec: np.ndarray):
    """
    Normalize the input vector.

    Parameters:
        vec (np.ndarray): The input vector.

    Returns:
        np.ndarray: The normalized vector.
    """
    if vec.ndim == 1:
        return vec / np.linalg.norm(vec)
    else:
        return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


def inner_product_distance_np(vec1, vec2, n=None):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.
        n (int | None): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store inner products.
    """
    res = np.dot(vec2, vec1.squeeze())
    if n is not None:
        topk = FastSort(res, backend='numpy')
        return topk.topk(n, ascending=False)
    return np.arange(res.shape[0]), res


@jit
def inner_product_distance_jax(vec1, vec2, n=None):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.
        n (int | None): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store inner products.
    """
    res = jnp.dot(vec2, vec1)
    if n is not None:
        topk = FastSort(res, backend='jax')
        return topk.topk(n, ascending=False)
    return jnp.arange(res.shape[0]), res


def inner_product_distance(vec1, vec2, n=None):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.
        n (int | None): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store inner products.
    """
    if n is not None:
        return inner(vec1, vec2, n)

    if vec2.shape[0] <= 10_0000:
        return inner_product_distance_np(vec1, vec2, n=n)
    return inner_product_distance_jax(vec1, vec2, n=n)


def cosine_distance(vec1, vec2, n=None):
    """
    Calculate the cosine distance between two vectors.

    Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        n (int | None): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store cosine distances.
    """
    res = simsimd.cosine(vec1, vec2)
    if n is not None:
        topk = FastSort(res, backend='numpy')
        return topk.topk(n, ascending=True)
    
    return np.arange(res.shape[0]), res


@jit
def euclidean_distance_square_jax(vec1, vec2, n=None):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        n (int | None): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store Euclidean distances.
    """
    res = jnp.sum((vec1 - vec2) ** 2, axis=1)
    if n is not None:
        topk = FastSort(res, backend='jax')
        return topk.topk(n, ascending=True)
    return jnp.arange(res.shape[0]), res


def euclidean_distance_square(vec1, vec2, n=None):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        n (int | None): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store Euclidean distances.
    """
    if vec2.shape[0] <= 10_0000:
        res = np.sum((vec1 - vec2) ** 2, axis=1)
        if n is not None:
            topk = FastSort(res, backend='numpy')
            return topk.topk(n, ascending=True)
        return np.arange(res.shape[0]), res
    return euclidean_distance_square_jax(vec1, vec2)


def _check_first_input(vec1, vec2):
    if vec1.dtype != vec2.dtype:
        vec1 = vec1.astype(vec2.dtype)

    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)

    return vec1


def _wrap_results(ids, distance):
    return ids.squeeze(), distance.squeeze()


def l2sq(vec1, vec2, n):
    if vec2.dtype == np.int8:
        ids, distance = euclidean_distance_square(vec1, vec2, n)
    else:
        vec1 = _check_first_input(vec1, vec2)
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.L2sq, threads=threads)
    return _wrap_results(ids, distance)


def cosine(vec1, vec2, n):
    if vec2.dtype == np.int8:
        ids, distance = cosine_distance(vec1, vec2, n)
    else:
        vec1 = _check_first_input(vec1, vec2)
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.Cos, threads=threads)
    return _wrap_results(ids, distance)


def inner(vec1, vec2, n):
    if vec2.dtype == np.int8:
        ids, distance = inner_product_distance(vec1, vec2)
    else:
        vec1 = _check_first_input(vec1, vec2)
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.IP, threads=threads)
    return _wrap_results(ids, distance)
