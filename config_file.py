from typing import Union

import yaml


class ConfigFile:
    """
    This class is used to load the config file and create a config object.

        The config file is a YAML file that contains all the parameters that are used in the project.

        The config object is a nested object that contains all the parameters from the config file.
        Each section in the config file is represented by a nested object in the config object.
    """
    def __init__(self, config):
        self._create_attributes(config)

    @staticmethod
    def load(config: Union[str, dict]):
        if isinstance(config, str):
            with open(config) as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = config
        return ConfigFile(config)

    def _create_attributes(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigFile(value))
            else:
                setattr(self, key, value)

    def save_config(self, output_path):
        with open(output_path, 'w') as f:
            yaml.dump(self.as_dict(), f)

    def as_dict(self):
        config_dict = {}
        for key, value in vars(self).items():
            if isinstance(value, ConfigFile):
                config_dict[key] = value.as_dict()
            else:
                config_dict[key] = value
        return config_dict

    @staticmethod
    def _create_nested_config(data):
        config = ConfigFile({})
        config._create_attributes(data)
        return config
