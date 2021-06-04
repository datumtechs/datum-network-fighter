import yaml


def load_cfg(file):
    with open(file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
