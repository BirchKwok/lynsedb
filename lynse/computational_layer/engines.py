import numpy as np
import simsimd

from usearch.compiled import exact_search
from usearch.index import MetricKind

from ..core_components.fast_sort import FastSort


def inner_product(vec1, vec2, n, use_simd=False):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.
        n (int): The number of vectors to return.

    Returns:
        (np.ndarray, np.ndarray): Indices of the top vectors and the result vector to store inner products.
    """
    vec1 = _check_first_input(vec1, vec2)
    if not use_simd and vec1.dtype != np.int8:
        vec1 = _check_first_input(vec1, vec2)
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.IP)
    else:
        dis = simsimd.cdist(vec1, vec2, "inner")
        topk = FastSort(dis)
        ids, distance = topk.topk(n, ascending=False)
    return _wrap_results(ids, distance)


def _check_first_input(vec1, vec2):
    if vec1.dtype != vec2.dtype:
        vec1 = vec1.astype(vec2.dtype)

    return np.atleast_2d(vec1)


def _wrap_results(ids, distance):
    return ids.squeeze(), distance.squeeze()


def l2sq(vec1, vec2, n, use_simd=True):
    vec1 = _check_first_input(vec1, vec2)
    if not use_simd:
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.L2sq)
    else:
        dis = simsimd.sqeuclidean(vec1, vec2)
        topk = FastSort(dis)
        ids, distance = topk.topk(n, ascending=True)

    return _wrap_results(ids, distance)


def cosine(vec1, vec2, n, use_simd=False):
    vec1 = _check_first_input(vec1, vec2)
    if not use_simd:
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.Cos)
    else:
        dis = simsimd.cdist(vec1, vec2, "cosine")
        topk = FastSort(dis)
        ids, distance = topk.topk(n, ascending=True)

    return _wrap_results(ids, distance)


def hamming(vec1, vec2, n, use_simd=False):
    vec1 = _check_first_input(vec1, vec2)
    if not use_simd:
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.Hamming)
    else:
        dis = simsimd.cdist(vec1, vec2, "hamming")
        topk = FastSort(dis)
        ids, distance = topk.topk(n, ascending=True)

    return _wrap_results(ids, distance)


def jaccard(vec1, vec2, n, use_simd=False):
    vec1 = _check_first_input(vec1, vec2)
    if not use_simd:
        ids, distance, *_ = exact_search(vec2, vec1, n, metric_kind=MetricKind.Jaccard)
    else:
        dis = simsimd.cdist(vec1, vec2, "jaccard")
        topk = FastSort(dis)
        ids, distance = topk.topk(n, ascending=True)

    return _wrap_results(ids, distance)
