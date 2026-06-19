import configparser
import os
from copy import deepcopy
from pathlib import Path

from ..utils.asserts import raise_if, augmented_isinstance


class Config:
    def __init__(self):
        self._LYNSE_LOG_LEVEL = self.get_env_variable('LYNSE_LOG_LEVEL', 'INFO', str, [str])
        self._LYNSE_LOG_PATH = self.get_env_variable('LYNSE_LOG_PATH', None, str, [str, None])
        self._LYNSE_TRUNCATE_LOG = self.get_env_variable('LYNSE_TRUNCATE_LOG', True, bool, [bool])
        self._LYNSE_LOG_WITH_TIME = self.get_env_variable('LYNSE_LOG_WITH_TIME', True, bool, [bool])
        self._LYNSE_KMEANS_EPOCHS = self.get_env_variable('LYNSE_KMEANS_EPOCHS', 100, int, [int])
        self._LYNSE_SEARCH_CACHE_SIZE = self.get_env_variable('LYNSE_SEARCH_CACHE_SIZE', 10000, int, [int])
        self._LYNSE_DEFAULT_ROOT_PATH = self.get_env_variable(
            'LYNSE_DEFAULT_ROOT_PATH',
            Path(os.path.expanduser('~/.LynseDB/databases/')),
            Path, [str, None, Path]
        )
        self._LYNSE_SEARCH_CACHE_EXPIRE_SECONDS = self.get_env_variable('LYNSE_SEARCH_CACHE_EXPIRE_SECONDS',
                                                                        3600, int, [int])


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
        """Log level"""
        return self._LYNSE_LOG_LEVEL

    @LYNSE_LOG_LEVEL.setter
    def LYNSE_LOG_LEVEL(self, value):
        self._LYNSE_LOG_LEVEL = value

    @property
    def LYNSE_LOG_PATH(self):
        """Log path"""
        return self._LYNSE_LOG_PATH

    @LYNSE_LOG_PATH.setter
    def LYNSE_LOG_PATH(self, value):
        self._LYNSE_LOG_PATH = value

    @property
    def LYNSE_TRUNCATE_LOG(self):
        """Whether to truncate log"""
        return self._LYNSE_TRUNCATE_LOG

    @LYNSE_TRUNCATE_LOG.setter
    def LYNSE_TRUNCATE_LOG(self, value):
        self._LYNSE_TRUNCATE_LOG = value

    @property
    def LYNSE_LOG_WITH_TIME(self):
        """Whether to include time in log"""
        return self._LYNSE_LOG_WITH_TIME

    @LYNSE_LOG_WITH_TIME.setter
    def LYNSE_LOG_WITH_TIME(self, value):
        self._LYNSE_LOG_WITH_TIME = value

    @property
    def LYNSE_KMEANS_EPOCHS(self):
        """Number of KMeans epochs"""
        return self._LYNSE_KMEANS_EPOCHS

    @LYNSE_KMEANS_EPOCHS.setter
    def LYNSE_KMEANS_EPOCHS(self, value):
        self._LYNSE_KMEANS_EPOCHS = value

    @property
    def LYNSE_SEARCH_CACHE_SIZE(self):
        """Search cache size"""
        return self._LYNSE_SEARCH_CACHE_SIZE

    @LYNSE_SEARCH_CACHE_SIZE.setter
    def LYNSE_SEARCH_CACHE_SIZE(self, value):
        self._LYNSE_SEARCH_CACHE_SIZE = value

    @property
    def LYNSE_SEARCH_CACHE_EXPIRE_SECONDS(self):
        """Search cache expire time in seconds"""
        return self._LYNSE_SEARCH_CACHE_EXPIRE_SECONDS

    @LYNSE_SEARCH_CACHE_EXPIRE_SECONDS.setter
    def LYNSE_SEARCH_CACHE_EXPIRE_SECONDS(self, value):
        self._LYNSE_SEARCH_CACHE_EXPIRE_SECONDS = value

    @property
    def LYNSE_DEFAULT_ROOT_PATH(self):
        """Default root path"""
        return self._LYNSE_DEFAULT_ROOT_PATH

    @LYNSE_DEFAULT_ROOT_PATH.setter
    def LYNSE_DEFAULT_ROOT_PATH(self, value):
        self._LYNSE_DEFAULT_ROOT_PATH = value

    def get_all_configs(self):
        return {
            'LYNSE_LOG_LEVEL': self.LYNSE_LOG_LEVEL,
            'LYNSE_LOG_PATH': self.LYNSE_LOG_PATH,
            'LYNSE_TRUNCATE_LOG': self.LYNSE_TRUNCATE_LOG,
            'LYNSE_LOG_WITH_TIME': self.LYNSE_LOG_WITH_TIME,
            'LYNSE_KMEANS_EPOCHS': self.LYNSE_KMEANS_EPOCHS,
            'LYNSE_SEARCH_CACHE_SIZE': self.LYNSE_SEARCH_CACHE_SIZE,
            'LYNSE_DEFAULT_ROOT_PATH': str(self.LYNSE_DEFAULT_ROOT_PATH),
            'LYNSE_SEARCH_CACHE_EXPIRE_SECONDS': self.LYNSE_SEARCH_CACHE_EXPIRE_SECONDS,
        }


def _config_path() -> Path:
    return Path(os.path.expanduser('~')) / '.lynsedb_configs.ini'


def _parse_config_value(value, default):
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in {'none', 'null', '~'}:
            return None
        if isinstance(default, bool):
            return lowered in {'1', 'true', 'yes', 'on'}
        if isinstance(default, Path):
            return Path(text)
        if isinstance(default, int) and not isinstance(default, bool):
            return int(text)
        return text
    return value


def _dump_ini_config(values, comments):
    lines = ['[lynse]']
    for key, value in values.items():
        comment = comments.get(key)
        if comment:
            lines.append(f'# {comment}')
        lines.append(f'{key} = {value}')
    return '\n'.join(lines) + '\n'


def generate_config_file(regenerate=False):
    config = Config()
    config_path = _config_path()

    config_dict = deepcopy(config.get_all_configs())
    comments = {}

    for k, v in config_dict.items():
        if isinstance(v, Path):
            v = str(v)
            config_dict[k] = v
        # Get the property's docstring and add it as a comment
        docstring = getattr(config.__class__, k).fget.__doc__
        comments[k] = docstring or ""

    if config_path.exists() and not regenerate:
        insert = False
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(config_path, encoding='utf-8')
        saved_config_dict = dict(parser['lynse']) if parser.has_section('lynse') else {}
        for k, v in config_dict.items():
            if k not in saved_config_dict:
                saved_config_dict[k] = v
                insert = True

        # If the config file is missing some keys, add them
        if insert:
            with open(config_path, 'w', encoding='utf-8') as file:
                file.write(_dump_ini_config(saved_config_dict, comments))
    else:
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(_dump_ini_config(config_dict, comments))


def load_config_file(path=None):
    _config = Config()
    if path is None:
        path = _config_path()
    if not isinstance(path, Path):
        raise ValueError('path must be a Path object')

    if not path.exists():
        generate_config_file(regenerate=False)

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding='utf-8')
    config_dict = dict(parser['lynse']) if parser.has_section('lynse') else {}
    defaults = _config.get_all_configs()

    for key, value in config_dict.items():
        if key not in defaults:
            continue
        value = _parse_config_value(value, defaults[key])
        setattr(_config, f'_{key}', value)

    return _config


generate_config_file(regenerate=False)
config = load_config_file()
get_all_configs = config.get_all_configs


# Collection namespace, which will be used to store shared variables
# or information between different components of the collection
class CollectionNamespace:
    def __init__(self, name):
        self.name = name
        self.namespace = {}

    def get(self, name):
        return self.namespace.get(name)

    def set(self, name, value):
        self.namespace[name] = value

    def delete(self, name):
        del self.namespace[name]


class NamespaceManager:
    def __init__(self):
        self.collections_namespace = {}

    def add_namespace(self, name, namespace: CollectionNamespace):
        self.collections_namespace[name] = namespace

    def get_namespace(self, name):
        return self.collections_namespace.get(name)

    def delete_namespace(self, name):
        del self.collections_namespace[name]


collections_namespace = NamespaceManager()
