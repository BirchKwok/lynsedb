import numpy as np
import torch


def get_device(device='auto'):
    if device == 'auto':
        return torch.device("cuda" if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else "cpu"))
    return torch.device(device)


def to_normalize(vec: np.ndarray, device='auto'):
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    elif vec.ndim == 2:
        device = get_device(device)
        vec = torch.from_numpy(vec).to(device)
        norm = torch.norm(vec, dim=1, keepdim=True)
        return (vec / norm).detach().cpu().numpy()
    else:
        raise ValueError("vec must be 1d or 2d array")


def cosine_distance(vec1, vec2, device='auto'):
    device = get_device(device)
    vec1 = torch.from_numpy(vec1).to(device).squeeze()
    vec2 = torch.from_numpy(vec2).to(device).squeeze()
    return torch.matmul(vec1, vec2).detach().cpu().numpy()


def euclidean_distance(vec1, vec2, device='auto'):
    device = get_device(device)
    vec1 = torch.from_numpy(vec1).to(device).squeeze()
    vec2 = torch.from_numpy(vec2).to(device).squeeze()
    return torch.norm(vec1 - vec2).detach().cpu().numpy()
