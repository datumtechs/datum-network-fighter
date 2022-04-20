import logging
import os
from glob import glob
from time import sleep, time
from datetime import datetime
from typing import Sequence, Callable, Tuple
import base64
import zlib
import codecs
import json

log = logging.getLogger(__name__)


def install_pkg(pkg_name: str, pkg_version: str = None, whl_file: str = None):
    """
    install the package if it is not installed.
    """
    import pkg_resources
    installed_pkgs = pkg_resources.working_set
    for i in installed_pkgs:
        if i.project_name == pkg_name:
            if pkg_version is None:
                return True
            i_ver = tuple(map(int, (i.split('.'))))
            pkg_ver = tuple(map(int, (pkg_version.split('.'))))
            if i_ver >= pkg_ver:
                return True
            return False
    import subprocess
    ob = pkg_name if whl_file is None else whl_file
    cmd = f'pip install {ob}'
    subprocess.run(cmd, shell=True)
    return True


def get_game_data_filenames(rc):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % ("*", "*"))
    files = list(sorted(glob(pattern)))
    return files


def get_next_generation_model_dirs(rc):
    dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs


def write_game_data_to_file(path, data):
    try:
        with open(path, "wt") as f:
            json.dump(data, f)
    except Exception as e:
        print(e)


def read_game_data_from_file(path):
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        print(e)

def len_str(dat_len: int) -> str:
    """return hex string of len of data, always 8 chars"""
    lb = dat_len.to_bytes(4, byteorder='big')
    return codecs.encode(lb, 'hex').decode()


def recv_sth(io_channel, remote_nodeid) -> Tuple[str, bytes]:
    recv_data = io_channel.Recv(remote_nodeid, 8)
    if recv_data == '\x00'*8:
        # log.info(f'maybe peer {remote_nodeid} has quit or cannot connect to.')
        return remote_nodeid, None
    data_len = int(recv_data, 16)
    recv_data = io_channel.Recv(remote_nodeid, data_len)
    h = ''.join('{:02x}'.format(ord(c)) for c in recv_data)
    recv_data = bytes.fromhex(h)
    log.info(f'recv {data_len} bytes data from {remote_nodeid}, in fact len: {len(recv_data)}, {recv_data[:20]}')
    # assert data_len == len(recv_data)  # sometimes assert failed cause by encoding, weird
    return remote_nodeid, recv_data


def send_sth(io_channel, remote_nodeid, data: str) -> None:
    lens = len_str(len(data))
    io_channel.Send(remote_nodeid, lens)
    io_channel.Send(remote_nodeid, data)
    log.info(f'send {len(data)} to {remote_nodeid}, {lens}, {data[:20]}')


def upload_data(data, cfg, io_channel, prefix):
    party_id = cfg.entry.party_id
    data_party = cfg.entry.data_party
    result_party = cfg.entry.result_party

    if party_id in result_party:  # as server
        pass
    elif party_id not in data_party:  # compute node as client
        remote_nodeid = result_party[0]  # select first result party
        log.info(f'upload {len(data)} {type(data)} to {remote_nodeid}')
        send_sth(io_channel, remote_nodeid, prefix + data)


def read_content(path, text=False):
    try:
        flag = 'r' if text else 'rb'
        with open(path, flag) as f:
            return f.read()
    except Exception as e:
        log.info(e)


def write_content(path: str, data: bytes):
    try:
        with open(path, 'wb') as f:
            f.write(data)
    except Exception as e:
        log.info(e)


class EarlyStop(Exception):
    def __str__(self):
        return self.__class__.__name__


def run_transac(ops: Sequence[Callable], max_retries=10, *args, **kwargs):
    """transac func return True if it want to go to next step 
    else retry until to max retries, which return False"""

    for op in ops:
        for i in range(max_retries):
            try:
                if op(*args, **kwargs):
                    break
            except EarlyStop as e:
                log.info(f'{e} at {op.__name__}')
                return True
            except Exception as e:
                log.info(f'{e} at {op.__name__}')
        else:
            return False
    return True


def zip_and_b64encode(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data)).decode()


def b64decode_and_unzip(data: str) -> bytes:
    return zlib.decompress(base64.b64decode(data))


def download_model_data(cfg, io_channel):
    party_id = cfg.entry.party_id
    data_party = cfg.entry.data_party
    result_party = cfg.entry.result_party

    if party_id in data_party:  # as server
        pass
    elif party_id not in result_party:  # compute node as client
        if len(data_party) == 0:
            return False
        server_party_id = data_party[0]  # actually, just one data party
        log.info(f'client {party_id} is waiting for data from {server_party_id}')

        expected_feedback = None

        def query_best():
            nonlocal expected_feedback
            if expected_feedback is None:
                expected_feedback = 'query_best_model'
                send_sth(io_channel, server_party_id, 'query_best_model')
            if expected_feedback == 'query_best_model':
                _, data = recv_sth(io_channel, server_party_id)
                log.info(f'latest best model: {data}')
                if data is None:
                    raise ValueError(f'query failed, recv nothing')
                expected_feedback = None
                model_weight_path = cfg.resource.model_best_weight_path
                if not os.path.exists(model_weight_path):
                    return True
                mt = os.path.getmtime(model_weight_path)
                log.info(f'file mtime on server: {datetime.fromtimestamp(float(data)).strftime("%Y-%m-%d %H:%M:%S")}'
                         f', on client: {datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")}'
                         f', {mt < float(data)}, {model_weight_path}')
                behind_server = mt < float(data)
                if not behind_server:
                    raise EarlyStop
                return True
            return False

        def download_model_cfg():
            nonlocal expected_feedback

            if expected_feedback is None:
                expected_feedback = 'download_model_cfg'
                send_sth(io_channel, server_party_id, 'download_model_cfg')
            if expected_feedback == 'download_model_cfg':
                _, data = recv_sth(io_channel, server_party_id)
                if data is None:
                    raise ValueError(f'download cfg failed, recv nothing')
                model_config_path = cfg.resource.model_best_config_path
                write_content(model_config_path, data)
                expected_feedback = None
                return True
            return False

        def download_model_weight():
            nonlocal expected_feedback

            if expected_feedback is None:
                expected_feedback = 'download_model_weight'
                send_sth(io_channel, server_party_id, 'download_model_weight')
            if expected_feedback == 'download_model_weight':
                _, data = recv_sth(io_channel, server_party_id)
                if data is None:
                    raise ValueError(f'download weight failed, recv nothing')
                data = b64decode_and_unzip(data)
                model_weight_path = cfg.resource.model_best_weight_path
                write_content(model_weight_path, data)
                expected_feedback = None
                return True
            return False

        ops = [query_best, download_model_cfg, download_model_weight]
        r = run_transac(ops)
        log.info(f'end with {r}')
        return r
