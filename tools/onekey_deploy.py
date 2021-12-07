import argparse
import json
import os
import re
import sys
import tempfile

import paramiko
from scp import SCPClient

cur_dir = os.path.abspath(os.path.dirname(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
sys.path.insert(0, par_dir)
from common.utils import load_cfg, dump_yaml


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


def parse_cfg(json_file):
    with open(json_file) as f:
        cfg = json.loads(f.read())
    return cfg


def pack_src():
    src_zip_name = 'fighter.tar.gz'
    cmd = f'tar -czPf {src_zip_name} common protos data_svc compute_svc third_party/via_svc tests console gateway'
    print(cmd)
    ret = os.system(cmd)
    succ = not bool(ret)
    print(f'pack src {succ}')
    return src_zip_name if succ else None


def tranfer_file(scp, a_file, to_dir):
    print(f'transfering {a_file}')
    scp.put(f'{a_file}', to_dir)
    print('transfer done.')


def unzip(ssh, a_file, to_dir):
    _, _, stderr = ssh.exec_command(f'tar -xzPf {a_file} -C {to_dir}')
    ret = stderr.read()
    ret = 'failed' if ret else 'succ'
    print(f'unzip {a_file} {ret}')


def update_svc_cfg(scp, remote_dir, cfg, svc_type):
    key_align = {'rpc_port': 'port', 'via_svc': 'via_svc', 'schedule_svc': 'schedule_svc',
                 'pass_via': 'pass_via', 'data_dir': 'data_root', 'code_dir': 'code_root_dir',
                 'results_dir': 'results_root_dir', 'host': 'bind_ip'}
    cfg = {k: v for k, v in cfg.items() if k in key_align.keys()}
    cfg = {key_align[k]: v for k, v in cfg.items()}
    cfg_tmpl = f'{svc_type}/config.yaml'
    # if svc_type in ('data_svc', 'compute_svc'):
    #     cfg_tmpl = f'{svc_type}/config.yaml'
    # else:
    #     print(f'nothing to do for {svc_type}')
    #     return
    new_cfg = load_cfg(cfg_tmpl)
    new_cfg.update(cfg)
    port = cfg['port']
    target = f'{remote_dir}/{svc_type}/config_{port}.yaml'
    _dump_yaml_to_remote(new_cfg, target)

    dir_key = ['data_root', 'code_root_dir', 'results_root_dir']
    for k in dir_key:
        if k not in new_cfg.keys():
            continue
        new_dir = new_cfg[k]
        if not new_dir.startswith('/'):  # relative directory
            new_dir = f'{remote_dir}/{svc_type}/{new_dir}'
        ssh.exec_command(f'mkdir -p {new_dir}')
        print(f'mkdir {new_dir}')


def modify_start_sh(scp, remote_dir, cfg, svc_type, py_home):
    sh = f'{svc_type}/start_svc.sh'
    with open(sh) as f:
        c = f.read()
        c = re.sub(r'(.*PYTHONPATH=.*) python\d* (.*)', f'\\1 {py_home}/bin/python \\2', c)
    target = f'{remote_dir}/{sh}'
    with tempfile.TemporaryFile('w+t') as f:
        f.write(c)
        f.seek(0)
        scp.putfo(f, target)
        print(f'update {target}')


def update_via_cfg(scp, remote_dir, cfg):
    if not cfg.get('pass_via'):
        return
    ip, port = cfg['via_svc'].split(':')
    cfg = {'public_ip': ip, 'port': int(port)}
    cfg_tmpl = 'via_svc/config.yaml'
    new_cfg = load_cfg(cfg_tmpl)
    new_cfg.update(cfg)
    target = f'{remote_dir}/{cfg_tmpl}'
    _dump_yaml_to_remote(new_cfg, target)


def _dump_yaml_to_remote(dict_, target):
    with tempfile.TemporaryFile('w+t') as f:
        dump_yaml(dict_, f)
        f.seek(0)
        scp.putfo(f, target)
    print(f'update {target}')


def start_svc(ssh, remote_dir, svc_type, cfg_file):
    print(f'start {svc_type} {cfg_file}')
    _, stdout, stderr = ssh.exec_command(f'cd {remote_dir}/{svc_type}; ./start_svc.sh {cfg_file};')
    print(stdout.read().decode())
    print(stderr.read().decode())

def start_via_svc(ssh, remote_dir, port, use_ssl):
    print(f'start via')
    _, stdout, stderr = ssh.exec_command(f'cd {remote_dir}/third_party/via_svc; ./start_svc.sh {port} {use_ssl};')
    print(stdout.read().decode())
    print(stderr.read().decode())

def kill_svc(ssh):
    print(f'kill all svc')
    _, stdout, _ = ssh.exec_command(r'ps -ef | grep "[p]ython -u main.py config.*.yaml"')
    lines = stdout.read().decode()
    print(lines)
    pids = []
    child_pids = []
    for row in lines.split('\n'):
        if not row:
            continue
        fields = row.split()
        if len(fields) < 3:
            continue
        if fields[2] == '1':  # ppid
            pids.append(fields[1])
        else:
            child_pids.append(fields[1])
    print("pids:", pids)
    if pids:
        pids = ' '.join(pids)
        cmd = f'kill -9 {pids}'
        print(cmd)
        ssh.exec_command(cmd)
    print("child_pids:", child_pids)
    if child_pids:
        child_pids = ' '.join(child_pids)
        cmd = f'kill -9 {child_pids}'
        print(cmd)
        ssh.exec_command(cmd)

    cmd = 'pkill via-go'
    print(cmd)
    ssh.exec_command(cmd)

def kill_all(node_cfg):
    node_cfg = parse_cfg(node_cfg)
    one_time_ops = {cfg['host']: False for cfg in node_cfg}
    for cfg in node_cfg:
        server, port = cfg['host'], cfg['port']
        user, password = cfg['user'], cfg['passwd']
        print(server, port, user, password)

        with createSSHClient(server, port, user, password) as ssh:
            with SCPClient(ssh.get_transport()) as scp:
                if not one_time_ops[server]:
                    one_time_ops[server] = True
                    kill_svc(ssh)


def start_all(node_cfg, use_ssl):
    node_cfg = parse_cfg(node_cfg)
    one_time_ops = {cfg['host']: False for cfg in node_cfg}
    for cfg in node_cfg:
        server, port = cfg['host'], cfg['port']
        user, password = cfg['user'], cfg['passwd']
        print(server, port, user, password)

        with createSSHClient(server, port, user, password) as ssh:
            with SCPClient(ssh.get_transport()) as scp:
                if not one_time_ops[server]:
                    one_time_ops[server] = True
                    ip, port = cfg['via_svc'].split(':')
                    start_via_svc(ssh, remote_dir, port, use_ssl)

                svc_type = cfg['svc_type']
                rpc_port = cfg['rpc_port']
                start_svc(ssh, remote_dir, svc_type, f'config_{rpc_port}.yaml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('node_config', type=str)
    parser.add_argument('--remote_dir', type=str, default='~/fighter')
    parser.add_argument('--py_env_zip', type=str)
    parser.add_argument('--src_zip', type=str)
    parser.add_argument('--start_all', action='store_true')
    parser.add_argument('--kill_all', action='store_true')
    parser.add_argument('--repack_src', action='store_true')
    parser.add_argument('--do_not_overwrite_config', action='store_true')
    parser.add_argument('--py_home', type=str, default='../python37')
    parser.add_argument('--certs_zip', type=str)
    parser.add_argument('--use_ssl', type=int, default=0)  # 0: not use ssl, 1: use ssl

    args = parser.parse_args()

    node_cfg = args.node_config
    remote_dir = args.remote_dir
    env_zip = args.py_env_zip
    py_home = args.py_home
    certs_zip = args.certs_zip
    use_ssl = args.use_ssl

    if args.start_all:
        start_all(node_cfg, use_ssl)
        sys.exit(0)
    if args.kill_all:
        kill_all(node_cfg)
        sys.exit(0)

    if args.repack_src:
        src_zip = pack_src()
    else:
        src_zip = args.src_zip
    print('src_zip:', src_zip)

    if env_zip and env_zip.endswith('.tar.gz'):
        l = len('.tar.gz')
        py_home = f'../{env_zip[:-l]}'
    else:
        py_home = py_home.rstrip('/')

    node_cfg = parse_cfg(node_cfg)
    one_time_ops = {cfg['host']: False for cfg in node_cfg}
    for cfg in node_cfg:
        server, port = cfg['host'], cfg['port']
        user, password = cfg['user'], cfg['passwd']
        print(server, port, user, password)

        with createSSHClient(server, port, user, password) as ssh:
            with SCPClient(ssh.get_transport()) as scp:
                ssh.exec_command(f'mkdir -p {remote_dir}')

                if not one_time_ops[server]:
                    one_time_ops[server] = True
                    if env_zip:
                        tranfer_file(scp, env_zip, remote_dir)
                        unzip(ssh, f'{remote_dir}/{os.path.basename(env_zip)}', remote_dir)
                    if src_zip:
                        tranfer_file(scp, src_zip, remote_dir)
                        unzip(ssh, f'{remote_dir}/{os.path.basename(src_zip)}', remote_dir)
                    if certs_zip:
                        tranfer_file(scp, certs_zip, remote_dir)
                        unzip(ssh, f'{remote_dir}/{os.path.basename(certs_zip)}', remote_dir)

                if not args.do_not_overwrite_config:
                    svc_type = cfg['svc_type']
                    modify_start_sh(scp, remote_dir, cfg, svc_type, py_home)
                    update_svc_cfg(scp, remote_dir, cfg, svc_type)
