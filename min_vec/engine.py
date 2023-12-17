import numpy as np


def cosine_distance(vec1, vec2):
    return np.dot(vec1, vec2)


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
