import argparse
import json
import os
import tempfile

import paramiko
from scp import SCPClient

from common.utils import load_cfg, dump_yaml

src_zip = 'fighter.tar.gz'


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
    cmd = f'tar -czf {src_zip} common protos data_svc compute_svc via_svc tests gateway'
    print(cmd)
    ret = os.system(cmd)
    print(f'pack src {not bool(ret)}')


def tranfer_file(scp, a_file, to_dir):
    print(f'transfer {a_file}')
    scp.put(f'{a_file}', to_dir)
    print('transfer done.')


def unzip(ssh, a_file, to_dir):
    _, _, stderr = ssh.exec_command(f'tar -xzf {a_file} -C {to_dir}')
    ret = stderr.read()
    ret = 'failed' if ret else 'succ'
    print(f'unzip {a_file} {ret}')


def update_svc_cfg(scp, remote_dir, cfg):
    svc_type = cfg['svc_type']
    key_align = {'rpc_port': 'port', 'via_svc': 'via_svc', 'schedule_svc': 'schedule_svc'}
    cfg = {k: v for k, v in cfg.items() if k in key_align.keys()}
    cfg = {key_align[k]: v for k, v in cfg.items()}
    if svc_type in ('data_svc', 'compute_svc'):
        cfg_tmpl = f'{svc_type}/config.yaml'
    else:
        print(f'nothing to do for {svc_type}')
        return
    new_cfg = load_cfg(cfg_tmpl)
    new_cfg.update(cfg)
    target = f'{remote_dir}/{cfg_tmpl}'
    with tempfile.TemporaryFile('w+t') as f:
        dump_yaml(new_cfg, f)
        f.seek(0)
        scp.putfo(f, target)
    print(f'update {target}')

    dir_key = ['data_root', 'code_root_dir', 'results_root_dir']
    for k in dir_key:
        if k not in new_cfg.keys():
            continue
        new_dir = new_cfg[k]
        if not new_dir.startswith('/'):  # relative directory
            new_dir = f'{remote_dir}/{svc_type}/{new_dir}'
        ssh.exec_command(f'mkdir -p {new_dir}')
        print(f'mkdir {new_dir}')


def modify_start_sh(scp, remote_dir, cfg):
    svc_type = cfg['svc_type']
    sh = f'{svc_type}/start_svc.sh'
    with open(sh) as f:
        c = f.read()
        c = c.replace('python37', '../python37/bin/python3')
    target = f'{remote_dir}/{sh}'
    with tempfile.TemporaryFile('w+t') as f:
        f.write(c)
        f.seek(0)
        scp.putfo(f, target)
        print(f'update {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('node_config', type=str)
    parser.add_argument('--remote_dir', type=str, default='fighter')
    parser.add_argument('--py_env_zip', type=str, default='python37.tar.gz')
    parser.add_argument('--src_zip', type=str)

    args = parser.parse_args()

    node_cfg = args.node_config
    remote_dir = args.remote_dir
    env_zip = args.py_env_zip

    if not args.src_zip:
        pack_src()
    else:
        src_zip = args.src_zip

    node_cfg = parse_cfg(node_cfg)
    for cfg in node_cfg:
        server, port = cfg['host'], cfg['port']
        user, password = cfg['user'], cfg['passwd']
        print(server, port, user, password)

        with createSSHClient(server, port, user, password) as ssh:
            with SCPClient(ssh.get_transport()) as scp:
                ssh.exec_command(f'mkdir {remote_dir}')

                tranfer_file(scp, env_zip, remote_dir)
                unzip(ssh, f'{remote_dir}/{env_zip}', remote_dir)

                tranfer_file(scp, src_zip, remote_dir)
                unzip(ssh, f'{remote_dir}/{src_zip}', remote_dir)

                update_svc_cfg(scp, remote_dir, cfg)
                modify_start_sh(scp, remote_dir, cfg)
