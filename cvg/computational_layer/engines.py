import numpy as np
import jax.numpy as jnp
from jax import jit
# import taichi as ti
#
#
# ti.init(arch=ti.cpu)


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


# @ti.kernel
# def _inner_product_distance(vec1: ti.types.ndarray(), vec2: ti.types.ndarray(), result: ti.types.ndarray()):
#     """
#     Calculate the inner product between a vector and each row of a 2D matrix.
#
#     Parameters:
#         vec1 (np.ndarray): The vector.
#         vec2 (np.ndarray): The 2D matrix.
#         result (np.ndarray): The result vector to store inner products.
#     """
#     for i in range(vec2.shape[0]):
#         sum = 0.0
#         for j in range(vec2.shape[1]):
#             sum += vec2[i, j] * vec1[j]
#         result[i] = sum


def inner_product_distance_ti(vec1, vec2):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.

    Returns:
        np.ndarray: The result vector to store inner products.
    """
    ...
    # result = np.empty(vec2.shape[0], dtype=np.float32)
    # _inner_product_distance(vec1, vec2, result)
    # return result


def inner_product_distance_np(vec1, vec2):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.

    Returns:
        np.ndarray: The result vector to store inner products.
    """
    return np.dot(vec2, vec1)


@jit
def inner_product_distance_jax(vec1, vec2):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.

    Returns:
        np.ndarray: The result vector to store inner products.
    """
    return jnp.dot(vec2, vec1)


def inner_product_distance(vec1, vec2, use='jax'):
    """
    Calculate the inner product between a vector and each row of a 2D matrix.

    Parameters:
        vec1 (np.ndarray): The vector.
        vec2 (np.ndarray): The 2D matrix.
        use (str): The engine to use, must be one of ['jax', 'ti', 'np'].

    Returns:
        np.ndarray: The result vector to store inner products.
    """
    if use == 'jax':
        return inner_product_distance_jax(vec1, vec2)
    elif use == 'ti':
        return inner_product_distance_ti(vec1, vec2)
    return inner_product_distance_np(vec1, vec2)


def cosine_distance(vec1, vec2, use='jax'):
    """
    Calculate the cosine distance between two vectors.

    Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        use (str): The engine to use, must be one of ['jax', 'ti', 'np'].

    Returns:
        np.ndarray: The cosine distance between the two vectors.
    """
    if use == 'jax':
        return inner_product_distance_jax(to_normalize(vec1), to_normalize(vec2))
    elif use == 'ti':
        return inner_product_distance_ti(to_normalize(vec1), to_normalize(vec2))
    return inner_product_distance_np(to_normalize(vec1), to_normalize(vec2))


@jit
def euclidean_distance_jax(vec1, vec2):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        np.ndarray: The Euclidean distance between the two vectors.
    """
    return jnp.linalg.norm(vec1 - vec2, axis=1)


def euclidean_distance(vec1, vec2, use_jax=False):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        use_jax (bool): Whether to use JAX.

    Returns:
        np.ndarray: The Euclidean distance between the two vectors.
    """
    if use_jax:
        return euclidean_distance_jax(vec1, vec2)
    return np.linalg.norm(vec1 - vec2, axis=1)
