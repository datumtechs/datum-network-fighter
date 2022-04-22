import re
import os
import yaml
import time
import logging
from collections import namedtuple
from grpc import _common

Import = namedtuple("Import", ["module", "name", "alias"])

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


def get_imports(path):
    with open(path) as fh:
        lines = fh.readlines()
    pat1 = re.compile(r'^\s*from\s+(\S+)\s+import\s+(\S+)')
    pat2 = re.compile(r'^\s*import\s+(\S+)')
    pat3 = re.compile(r'^\s*from\s+(\S+)\s+import\s+(\S+)\s+as\s+(\S+)')
    pat4 = re.compile(r'^\s*import\s+(\S+)\s+as\s+(\S+)')
    imports = []
    for line in lines:
        m = pat1.match(line)
        if m:
            imports.append(Import(m.group(1).split('.'), [m.group(2)], None))
        m = pat2.match(line)
        if m:
            imports.append(Import([], [m.group(1)], None))
        m = pat3.match(line)
        if m:
            imports.append(Import(m.group(1).split('.'), [m.group(2)], m.group(3)))
        m = pat4.match(line)
        if m:
            imports.append(Import([], [m.group(1)], m.group(2)))
    return imports


def get_imports_recursive(entry_path):
    dir_ = os.path.dirname(entry_path)
    all_mods = {}
    for root, dirs, files in os.walk(dir_):
        for f in files:
            if f.endswith('.py'):
                m = os.path.splitext(os.path.basename(f))[0]
                all_mods[m] = os.path.join(root, f)

    def get_imports_from_file(start, all_mods):
        user_mods = set()
        imports = get_imports(start)
        for i in imports:
            if len(i.module) == 0 and len(i.name) == 1:
                user_mods.add(i.name[0])
            elif len(i.module) == 1:
                user_mods.add(i.module[0])
            else:
                pass
        user_mods = user_mods.intersection(all_mods.keys())
        more_user_mods = set()
        for m in user_mods:
            m = get_imports_from_file(all_mods[m], all_mods)
            more_user_mods.update(m)
        return user_mods.union(more_user_mods)

    user_mods = get_imports_from_file(entry_path, all_mods)
    user_mods.add(os.path.splitext(os.path.basename(entry_path))[0])
    return all_mods, user_mods


def install_pkg(pkg_name: str, version: str = None, whl_file: str = None, index_url: str = None):
    """
    install the package if it is not installed.
    """
    import pkg_resources
    installed_pkgs = pkg_resources.working_set
    for i in installed_pkgs:
        if i.project_name == pkg_name:
            if version is None:
                return True
            i_ver = tuple(map(int, (i.split('.'))))
            pkg_ver = tuple(map(int, (version.split('.'))))
            if i_ver >= pkg_ver:
                return True
            return False
    import subprocess
    import sys
    ob = pkg_name if whl_file is None else whl_file
    cmd = f'{sys.executable} -m pip install {ob}'
    if index_url is not None:
        cmd += f' --index-url {index_url}'
        if index_url.startswith('http://'):
            ip = index_url.split('//')[1].split('/')[0].split(':')[0]
            cmd += ' --trusted-host ' + ip
    log.info(cmd)
    subprocess.run(cmd, shell=True)
    return True
