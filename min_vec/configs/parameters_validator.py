import json
from functools import wraps
from pathlib import Path

from spinesUtils.asserts import raise_if, generate_function_kwargs
from spinesUtils.logging import Logger


class ParametersValidator:
    """Database configuration, define once and can not be changed."""
    def __init__(self, update_configs: list, logger: Logger):
        raise_if(TypeError, not isinstance(update_configs, list), "update_configs must be a list.")

        self.update_configs_list = update_configs
        self.logger = logger

    def check_configs(self, configs: dict, update_configs_dict: dict):
        """Check if the configurations are set."""
        for key, value in configs.items():
            if key not in update_configs_dict:
                continue
            else:
                if update_configs_dict[key] != value:
                    self.logger.warning(f"The database configuration {key} has been set to {value}, "
                                        f"the new value {update_configs_dict[key]} will be ignored.")

        for key, value in update_configs_dict.items():
            if key not in configs:
                self.logger.warning(f"The database configuration {key} is not set, "
                                    f"and it will be set to {value}.")
                configs[key] = value

        return configs

    def load_configs(self, configs_json: Path):
        """Load the configurations."""
        try:
            with open(configs_json, 'r') as f:
                configs = json.load(f)
            return configs
        except (FileNotFoundError, json.JSONDecodeError, PermissionError, OSError) as e:
            self.logger.error(f"Failed to load the MinVectorDB configurations.")
            raise e

    def save_configs(self, configs_json: Path, configs: dict):
        """Save the configurations."""
        try:
            with open(configs_json, 'w') as f:
                json.dump(configs, f)

            return configs
        except (PermissionError, OSError) as e:
            self.logger.error(f"Failed to save the MinVectorDB configurations.")
            raise e

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # first parameter is self
            self_instance = args[0]

            kwargs = generate_function_kwargs(func, *args, **kwargs)

            dir_path = Path(kwargs.get("database_path"))

            update_configs_dict = {}
            for key in self.update_configs_list:
                update_configs_dict[key] = kwargs.get(key)

            self.database_path_parent = Path(dir_path).parent.absolute() / Path(
                '.mvdb'.join(Path(dir_path).absolute().name.split('.mvdb')[:-1]))

            if not self.database_path_parent.exists():
                self.database_path_parent.mkdir(parents=True)
                first_create = True
            else:
                first_create = False

            configs_json = self.database_path_parent / Path('configs.json')

            if first_create or not configs_json.exists():
                final_configs = self.save_configs(configs_json, update_configs_dict)
            else:
                existing_configs = self.load_configs(configs_json)
                final_configs = self.check_configs(existing_configs, update_configs_dict)

            return func(self_instance, **final_configs)

        return wrapper
