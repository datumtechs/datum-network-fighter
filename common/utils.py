import yaml
from grpc import _common


def load_cfg(file):
    with open(file) as f:
        if hasattr(yaml, 'FullLoader'):
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            cfg = yaml.load(f)
    return cfg


def dump_yaml(data, file_handle):
    yaml.dump(data, file_handle, default_flow_style=False)


def check_grpc_channel_state(channel, try_to_connect=True):
    result = channel._channel.check_connectivity_state(try_to_connect)
    return _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY[result]
