import yaml
import time
import logging
from grpc import _common

log = logging.getLogger(__name__)
GateWayConsulServiceAddressKey = "fighter/icegrid_ip_port"
ViaNodeConsulServiceExternalAddressKey = "fighter/via_ip_port"


def load_cfg(file):
    with open(file) as f:
        if hasattr(yaml, 'FullLoader'):
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            cfg = yaml.load(f)
    return cfg


def dump_yaml(data, file_handle):
    yaml.dump(data, file_handle, default_flow_style=False)


def get_schedule_svc(cfg, consul_client, pipe):
    while True:
        flag, result = consul_client.query_service_info_by_filter('carrier in Tags')
        kv = consul_client.get_kv(GateWayConsulServiceAddressKey)
        if (kv is None) or len(kv) == 0:
            log.info(f'get_kv result kv is:{kv}')
            cfg['ice_grid'] = ''
        else:
            cfg['ice_grid'] = kv.replace(' ', '').replace('_', ':')
        if flag:
            result, *_ = list(result.values())
            schedule_svc = f'{result["Address"]}:{result["Port"]}'
            if cfg['schedule_svc'] != schedule_svc:
                log.info(f'get new schedule svc: {schedule_svc}')
                cfg['schedule_svc'] = schedule_svc
                # pipe.send(schedule_svc)
        elif not result:
            pass
        else:
            raise Exception(result)
        time.sleep(3)


def process_recv_address(cfg, pipe):
    while True:
        cfg['schedule_svc'] = pipe.recv()
        time.sleep(3)


def check_grpc_channel_state(channel, try_to_connect=True):
    result = channel._channel.check_connectivity_state(try_to_connect)
    return _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY[result]

def check_input_param_type(**kargs):
    for key,value in kargs.items():
        assert isinstance(value[0], value[1]), f'{key} must be type({value[1]}), not {type(value[0])}'