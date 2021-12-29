import yaml
import time


def load_cfg(file):
    with open(file) as f:
        if hasattr(yaml, 'FullLoader'):
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            cfg = yaml.load(f)
    return cfg


def dump_yaml(data, file_handle):
    yaml.dump(data, file_handle, default_flow_style=False)


def get_schedule_svc(cfg, consul_client)
    flag, result = consul_client.query_service_info_by_filter('carrier in Tags')
    if flag:
        result, _ = list(result.values())
        schedule_svc = f'{result["Address"]}:{result["Port"]}'
        if cfg['schedule_svc'] != schedule_svc:
            cfg['schedule_svc'] = schedule_svc
    elif not result:
        pass
    else:
        raise Exception(result)
