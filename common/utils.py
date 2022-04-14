import re
import os
import yaml
import time
import logging
from grpc import _common


log = logging.getLogger(__name__)


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
        if flag:
            result, *_ = list(result.values())
            schedule_svc = f'{result["Address"]}:{result["Port"]}'
            if cfg['schedule_svc'] != schedule_svc:
                log.info(f'get new schedule svc: {schedule_svc}')
                cfg['schedule_svc'] = schedule_svc
                pipe.send(schedule_svc)
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


PAT_FILE_SEPARATOR = re.compile(r'##### (.*?)\n')


def compose(files, start_dir):
    if len(files) == 0:
        return ''
    cont = []
    for i, f in enumerate(files):
        if i != 0:
            cont.append('\n'*3)
        rel = os.path.relpath(f, start_dir)
        cont.append(f'##### {rel}\n')
        with open(f) as fp:
            lines = fp.readlines()
        cont.extend(lines)
    return ''.join(cont)


def decompose(cont: str, to_dir):
    files = []
    seg = []
    for m in PAT_FILE_SEPARATOR.finditer(cont):
        seg.extend(m.span())
        files.append(m.group(1))
    if len(seg) == 0:
        files.append('main.py')
        seg.append(0)
    else:
        seg.pop(0)
    seg.append(-1)

    os.makedirs(to_dir, exist_ok=True)
    for i, f in enumerate(files):
        start = seg[2*i]
        end = seg[2*i+1]
        fn = os.path.join(to_dir, f)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'w') as fp:
            fp.write(cont[start:end].rstrip())
            fp.write('\n')
    
    return files


def merge_options(d, opts):
    for k, v in opts.items():
        if isinstance(v, dict):
            merge_options(d[k], v)
        else:
            d[k] = v
