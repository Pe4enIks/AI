import yaml


def parse_config(file):
    with open(file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config
