import numpy as np
import torch


def to_normalize(vec: np.ndarray):
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        if (norm == 0).all():
            return vec
        return vec / norm
    elif vec.ndim == 2:
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        if (norm == 0).all():
            return vec
        return vec / norm
    else:
        raise ValueError("vec must be 1d or 2d array")


def cosine_distance(vec1, vec2, device=torch.device('cpu')):
    # 确保vec1和vec2是二维的
    if vec1.ndim == 2:
        vec1 = vec1.squeeze()
    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    vec2 = vec2.T
    if vec1.dtype != vec2.dtype:
        vec1 = vec1.astype(vec2.dtype)

    if device.type != 'cpu':
        return torch.matmul(torch.from_numpy(vec1).to(device),
                            torch.from_numpy(vec2).to(device)).cpu().numpy()

    if vec1.dtype != np.float32:
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)

    return torch.matmul(torch.from_numpy(vec1),
                        torch.from_numpy(vec2)).numpy()


def euclidean_distance(vec1, vec2, device=torch.device('cpu')):
    if vec1.ndim == 1:
        vec1 = vec1[np.newaxis, :]

    if vec2.ndim == 1:
        vec2 = vec2[np.newaxis, :]

    if vec1.dtype != vec2.dtype:
        vec1 = vec1.astype(vec2.dtype)

    if device.type != 'cpu':
        return torch.norm(torch.from_numpy(vec1).to(device) -
                          torch.from_numpy(vec2).to(device), dim=1).cpu().numpy()

    if vec1.dtype != np.float32:
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)
    return torch.norm(torch.from_numpy(vec1) -
                      torch.from_numpy(vec2), dim=1).numpy()


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
    if k >= len(arr):
        return np.argsort(arr)

    indices = np.argpartition(arr, k)[:k]
    sorted_indices = indices[np.argsort(arr[indices])]
    return sorted_indices
