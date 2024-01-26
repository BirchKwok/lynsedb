from functools import wraps
from pathlib import Path
import numpy as np

from spinesUtils.asserts import raise_if

from min_vec.engine import cosine_distance, euclidean_distance


class UnKnownError(Exception):
    pass


def io_checker(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except FileNotFoundError as e:
            raise_if(FileNotFoundError, True,
                     f"No such file or directory: '{Path(e.filename).absolute()}'")
        except PermissionError as e:
            raise_if(PermissionError, True,
                     f"No permission to read or write the '{Path(e.filename).absolute()}' file.")
        except IOError as e:
            raise_if(IOError, True, f"Encounter IOError "
                                    f"when read or write the '{Path(e.filename).absolute()}' file.")
        except Exception as e:
            raise_if(UnKnownError, True, f"Encounter Unknown Error "
                                      f"when read or write the file.")

    return wrapper


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


def get_env_variable(name, default=None, default_type=str):
    import os

    def type_cast(value):
        if default_type == str:
            return value  # include None
        else:
            try:
                return default_type(value)  # will raise Exception if None
            except Exception:
                return default  # include None

    if default is None:
        return type_cast(os.environ.get(name))
    else:
        return type_cast(os.environ.get(name, default))
