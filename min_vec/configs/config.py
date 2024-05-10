import os

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
    def MVDB_LOG_LEVEL(self):
        return self.get_env_variable('MVDB_LOG_LEVEL', 'INFO', str, [str])

    @property
    def MVDB_LOG_PATH(self):
        return self.get_env_variable('MVDB_LOG_PATH', None, str, [str, None])

    @property
    def MVDB_TRUNCATE_LOG(self):
        return self.get_env_variable('MVDB_TRUNCATE_LOG', True, bool, [bool])

    @property
    def MVDB_LOG_WITH_TIME(self):
        return self.get_env_variable('MVDB_LOG_WITH_TIME', True, bool, [bool])

    @property
    def MVDB_KMEANS_EPOCHS(self):
        return self.get_env_variable('MVDB_KMEANS_EPOCHS', 100, int, [int])

    @property
    def MVDB_QUERY_CACHE_SIZE(self):
        return self.get_env_variable('MVDB_QUERY_CACHE_SIZE', 10000, int, [int])

    @property
    def MVDB_DATALOADER_BUFFER_SIZE(self):
        size = self.get_env_variable('MVDB_DATALOADER_BUFFER_SIZE', 20, int, [int, None])
        if size is None or size < 1:
            return 1
        return size

    @property
    def MVDB_DELAY_NUM(self):
        return self.get_env_variable('MVDB_DELAY_NUM', 1000, int, [int])

    def get_all_configs(self):
        return {
            'MVDB_LOG_LEVEL': self.MVDB_LOG_LEVEL,
            'MVDB_LOG_PATH': self.MVDB_LOG_PATH,
            'MVDB_TRUNCATE_LOG': self.MVDB_TRUNCATE_LOG,
            'MVDB_LOG_WITH_TIME': self.MVDB_LOG_WITH_TIME,
            'MVDB_KMEANS_EPOCHS': self.MVDB_KMEANS_EPOCHS,
            'MVDB_QUERY_CACHE_SIZE': self.MVDB_QUERY_CACHE_SIZE,
            'MVDB_DATALOADER_BUFFER_SIZE': self.MVDB_DATALOADER_BUFFER_SIZE
        }


config = Config()

get_all_configs = config.get_all_configs
