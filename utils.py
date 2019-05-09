import yaml
from easydict import EasyDict
from io import open


def get_yaml_config(path):
    with open(path, encoding="utf-8") as h:
        config = EasyDict(yaml.safe_load(h))
    return config
