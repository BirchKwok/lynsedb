import os

import torch


def get_env_variable(name, default=None, default_type=str):

    def type_cast(value):
        if value == default:
            if default is not None:
                return default_type(default)
            else:
                return None

        if default_type == bool and isinstance(name, str):
            return value.lower() == 'true'
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


MVDB_LOG_LEVEL = get_env_variable('MVDB_LOG_LEVEL', 'INFO', str)
MVDB_LOG_PATH = get_env_variable('MVDB_LOG_PATH', None, str)
MVDB_TRUNCATE_LOG = get_env_variable('MVDB_TRUNCATE_LOG', True, bool)
MVDB_LOG_WITH_TIME = get_env_variable('MVDB_LOG_WITH_TIME', False, bool)
MVDB_KMEANS_EPOCHS = get_env_variable('MVDB_KMEANS_EPOCHS', 100, int)
MVDB_BULK_ADD_BATCH_SIZE = get_env_variable('MVDB_BULK_ADD_BATCH_SIZE', 100000, int)
MVDB_CACHE_SIZE = get_env_variable('MVDB_CACHE_SIZE', 10000, int)

MVDB_COSINE_SIMILARITY_THRESHOLD = os.environ.get('MVDB_COSINE_SIMILARITY_THRESHOLD', 0.8)
if MVDB_COSINE_SIMILARITY_THRESHOLD == 'None':
    MVDB_COSINE_SIMILARITY_THRESHOLD = None
else:
    MVDB_COSINE_SIMILARITY_THRESHOLD = float(MVDB_COSINE_SIMILARITY_THRESHOLD)

MVDB_COMPUTE_DEVICE = get_env_variable('MVDB_COMPUTE_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu', str)

MVDB_USER_MESSAGE_PATH = get_env_variable('MVDB_USER_MESSAGE_PATH', None, str)


def get_all_configs():
    return {
        'MVDB_LOG_LEVEL': MVDB_LOG_LEVEL,
        'MVDB_LOG_PATH': MVDB_LOG_PATH,
        'MVDB_TRUNCATE_LOG': MVDB_TRUNCATE_LOG,
        'MVDB_LOG_WITH_TIME': MVDB_LOG_WITH_TIME,
        'MVDB_KMEANS_EPOCHS': MVDB_KMEANS_EPOCHS,
        'MVDB_BULK_ADD_BATCH_SIZE': MVDB_BULK_ADD_BATCH_SIZE,
        'MVDB_CACHE_SIZE': MVDB_CACHE_SIZE,
        'MVDB_COSINE_SIMILARITY_THRESHOLD': MVDB_COSINE_SIMILARITY_THRESHOLD,
        'MVDB_COMPUTE_DEVICE': MVDB_COMPUTE_DEVICE
    }

