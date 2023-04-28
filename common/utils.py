import yaml
from yaml import SafeLoader


def load_config(path_yaml):
    with open(path_yaml) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data