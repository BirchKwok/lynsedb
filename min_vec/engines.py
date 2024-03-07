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
    if vec1.ndim == 2:
        vec1 = vec1.squeeze()

    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    if device.type != 'cpu':
        return torch.matmul(torch.from_numpy(vec1).to(device),
                            torch.from_numpy(vec2).to(device)).cpu().numpy()

    if vec1.dtype == np.float16:
        vec1 = vec1.astype(np.float32)
    if vec2.dtype == np.float16:
        vec2 = vec2.astype(np.float32)

    return torch.matmul(torch.from_numpy(vec1),
                        torch.from_numpy(vec2)).numpy()


def euclidean_distance(vec1, vec2, device=torch.device('cpu')):
    if vec1.ndim == 2:
        vec1 = vec1.squeeze()

    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    if device.type != 'cpu':
        return torch.norm(torch.from_numpy(vec1).to(device) -
                          torch.from_numpy(vec2).to(device), dim=1).cpu().numpy()

    if vec1.dtype == np.float16:
        vec1 = vec1.astype(np.float32)
    if vec2.dtype == np.float16:
        vec2 = vec2.astype(np.float32)

    return torch.norm(torch.from_numpy(vec1) -
                      torch.from_numpy(vec2), dim=1).numpy()
