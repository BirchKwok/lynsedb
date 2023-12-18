import numpy as np


def squeeze(func):
    def wrapper(vec1, vec2):
        vec1 = vec1.squeeze()
        vec2 = vec2.squeeze()

        return func(vec1, vec2)
    return wrapper


@squeeze
def cosine_distance(vec1, vec2):
    return np.dot(vec1, vec2)


@squeeze
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
