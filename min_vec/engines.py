import numpy as np
import torch


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


def cosine_distance(vec1, vec2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vec1.ndim == 2:
        vec1 = vec1.squeeze()

    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    if device.type != 'cpu':
        return torch.matmul(torch.tensor(vec1).to(device),
                            torch.tensor(vec2).to(device)).cpu().numpy()

    return torch.matmul(torch.tensor(vec1),
                        torch.tensor(vec2)).numpy()


def euclidean_distance(vec1, vec2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if vec1.ndim == 2:
        vec1 = vec1.squeeze()

    if vec2.ndim == 2:
        vec2 = vec2.squeeze()

    if device.type != 'cpu':
        return torch.norm(torch.tensor(vec1).to(device) -
                          torch.tensor(vec2).to(device), dim=1).cpu().numpy()

    return torch.norm(torch.tensor(vec1) -
                      torch.tensor(vec2), dim=1).numpy()


def silhouette_score(X, labels, metric='cosine'):
    X = np.array(X)
    labels = np.array(labels)

    silhouette_scores = []

    for label in np.unique(labels):
        intra_cluster_mask = labels == label
        intra_cluster_data = X[intra_cluster_mask]

        if metric == 'euclidean':
            intra_distances = euclidean_distance(intra_cluster_data, intra_cluster_data.T)
        else:
            intra_distances = cosine_distance(intra_cluster_data, intra_cluster_data.T)
        a = np.mean(intra_distances, axis=1)

        inter_cluster_data = X[~intra_cluster_mask]

        if metric == 'euclidean':
            inter_distances = euclidean_distance(intra_cluster_data, inter_cluster_data.T)
        else:
            inter_distances = cosine_distance(intra_cluster_data, inter_cluster_data.T)

        b = np.min(np.mean(inter_distances, axis=1), axis=0)

        silhouette = (b - a) / np.maximum(a, b)
        silhouette_scores.append(silhouette)

    all_silhouettes = np.concatenate(silhouette_scores)
    return np.mean(all_silhouettes)