import os
from pathlib import Path

from spinesUtils.asserts import raise_if, augmented_isinstance


class Config:
    @staticmethod
    def get_env_variable(name, default=None, default_type=str, type_allow_list=None):

        def type_cast(value):
            if value == 'None':
                return None

            if value == default:
                return default

            if default_type == bool and isinstance(name, str):
                return value.lower() == 'true'

            if default_type == str:
                return value
            else:
                try:
                    return default_type(value)  # will raise Exception if None
                except Exception:
                    return default  # include None

        if default is None:
            value = type_cast(os.environ.get(name))
            if type_allow_list is not None:
                raise_if(ValueError, not augmented_isinstance(value, tuple(type_allow_list)),
                         f"{name} must be in {type_allow_list}")
            return value
        else:
            return type_cast(os.environ.get(name, default))

    @property
    def LYNSE_LOG_LEVEL(self):
        return self.get_env_variable('LYNSE_LOG_LEVEL', 'INFO', str, [str])

    @property
    def LYNSE_LOG_PATH(self):
        return self.get_env_variable('LYNSE_LOG_PATH', None, str, [str, None])

    @property
    def LYNSE_TRUNCATE_LOG(self):
        return self.get_env_variable('LYNSE_TRUNCATE_LOG', True, bool, [bool])

    @property
    def LYNSE_LOG_WITH_TIME(self):
        return self.get_env_variable('LYNSE_LOG_WITH_TIME', True, bool, [bool])

    @property
    def LYNSE_KMEANS_EPOCHS(self):
        return self.get_env_variable('LYNSE_KMEANS_EPOCHS', 100, int, [int])

    @property
    def LYNSE_SEARCH_CACHE_SIZE(self):
        return self.get_env_variable('LYNSE_SEARCH_CACHE_SIZE', 10000, int, [int])

    @property
    def LYNSE_DEFAULT_ROOT_PATH(self):
        return self.get_env_variable('LYNSE_DEFAULT_ROOT_PATH',
                                     Path(os.path.expanduser('~/.Convergence/data/')), Path, [str, None, Path])

    def get_all_configs(self):
        return {
            'LYNSE_LOG_LEVEL': self.LYNSE_LOG_LEVEL,
            'LYNSE_LOG_PATH': self.LYNSE_LOG_PATH,
            'LYNSE_TRUNCATE_LOG': self.LYNSE_TRUNCATE_LOG,
            'LYNSE_LOG_WITH_TIME': self.LYNSE_LOG_WITH_TIME,
            'LYNSE_KMEANS_EPOCHS': self.LYNSE_KMEANS_EPOCHS,
            'LYNSE_SEARCH_CACHE_SIZE': self.LYNSE_SEARCH_CACHE_SIZE,
        }


config = Config()

get_all_configs = config.get_all_configs
