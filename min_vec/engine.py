import numpy as np
import torch


def get_device(device='auto'):
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu"))
    return torch.device(device)


def to_normalize(vec: np.ndarray):
    norm_func = np.linalg.norm

    if vec.ndim == 1:
        norm = norm_func(vec)
        if norm == 0:
            return vec
        return vec / norm
    elif vec.ndim == 2:
        norm = norm_func(vec, axis=1, keepdims=True)
        if (norm == 0).any():
            return vec
        return vec / norm
    else:
        raise ValueError("vec must be 1d or 2d array")


def cosine_distance(vec1, vec2, device='auto'):
    device = get_device(device)

    if vec1.ndim == 2:
        vec1 = vec1.squeeze()

    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    if device.type != 'cpu':
        return torch.matmul(torch.tensor(vec1).to(device),
                            torch.tensor(vec2).to(device)).cpu().numpy()

    return torch.matmul(torch.tensor(vec1),
                            torch.tensor(vec2)).numpy()


def euclidean_distance(vec1, vec2, device='auto'):
    device = get_device(device)
    if vec1.ndim == 2:
        vec1 = vec1.squeeze()

    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    if device.type != 'cpu':
        return torch.norm(torch.tensor(vec1).to(device) -
                            torch.tensor(vec2).to(device), dim=1).cpu().numpy()

    return torch.norm(torch.tensor(vec1) -
                            torch.tensor(vec2), dim=1).numpy()
