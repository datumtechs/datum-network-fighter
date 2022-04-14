import os
from glob import glob
from time import sleep, time
from logging import getLogger

import channel_sdk.pyio as chsdkio

logger = getLogger(__name__)


def get_game_data_filenames(rc):
    pattern = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files


def get_next_generation_model_dirs(rc):
    dir_pattern = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs


def recv_sth(io_channel, remote_nodeid):
    recv_data = io_channel.Recv(remote_nodeid, 4).encode()
    if recv_data == b'\x00\x00\x00\x00':
        logger.info(f'maybe peer {remote_nodeid} has quit or cannot connect to.')
        return remote_nodeid, None
    data_len = int.from_bytes(recv_data, byteorder='big')
    recv_data = io_channel.Recv(remote_nodeid, data_len)
    logger.info(f'recv {data_len} bytes data from {remote_nodeid}, in fact len: {len(recv_data)}, {recv_data}')
    # assert data_len == len(recv_data)  # sometimes assert failed cause by encoding, weird
    return remote_nodeid, recv_data


def send_sth(io_channel, remote_nodeid, data: bytes):
    len_bytes = len(data).to_bytes(4, byteorder='big')
    io_channel.Send(remote_nodeid, len_bytes)
    io_channel.Send(remote_nodeid, data)
    logger.info(f'send {len(data)} to {remote_nodeid}, len_bytes: {len_bytes}')


def upload_data(path, data, cfg):
    io_channel = chsdkio.APIManager()

    logger.info("start create channel")
    channel = io_channel.create_channel(cfg.party_id, cfg.channel_config)
    if cfg.party_id in cfg.result_party:  # as server
        pass
    elif cfg.party_id not in cfg.data_party:  # compute node as client
        remote_nodeid = cfg.result_party[0]  # select first result party
        send_sth(io_channel, remote_nodeid, data)


def read_content(path):
    try:
        with open(path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(e)


def write_content(path, data):
    try:
        with open(path, 'wb') as f:
            f.write(data)
    except Exception as e:
        print(e)


def download_model_data(cfg):
    io_channel = chsdkio.APIManager()
    logger.info("start create channel")
    channel = io_channel.create_channel(cfg.party_id, cfg.channel_config)
    if cfg.party_id in cfg.data_party:  # as server
        pass
    elif cfg.party_id not in cfg.result_party:  # compute node as client
        server_party_id = cfg.data_party[0]  # actually, just one data party

        best_model_name = None
        last_update_time = None
        best_model_cfg = None
        best_model_weight = None
        expected_feedback = None

        def query_best():
            UPDATE_INTERVAL = 5 * 60
            if best_model_name is not None and last_update_time is not None:
                elapse = time() - last_update_time
                if elapse < UPDATE_INTERVAL:
                    return False
            if expected_feedback is None:
                expected_feedback = 'query_best_model'
                send_sth(io_channel, server_party_id, 'query_best_model')
            if expected_feedback == 'query_best_model':
                _, data = recv_sth(io_channel, server_party_id)
                logger.info(f'latest best model: {data}')
                if data is not None:
                    best_model_name = data
                    last_update_time = time()
                    expected_feedback = None
                    model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
                    mt = os.path.getmtime(model_weight_path)
                    return mt < float(data)
            return True

        def download_model_cfg():
            if best_model_name is None:
                return False
            if best_model_cfg is None:
                if expected_feedback is None:
                    expected_feedback = 'download_model_cfg'
                    send_sth(io_channel, server_party_id, 'download_model_cfg')
                if expected_feedback == 'download_model_cfg':
                    _, data = recv_sth(io_channel, server_party_id)
                    print(f'download model_cfg: {data}')
                    if data is None:
                        return False
                    best_model_cfg = data
                    model_config_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_config_path)
                    write_content(model_config_path, best_model_cfg)
                    expected_feedback = None
                    return True
            return False

        def download_model_weight():
            if best_model_name is None or best_model_cfg is None:
                return False
            if best_model_weight is None:
                if expected_feedback is None:
                    expected_feedback = 'download_model_weight'
                    send_sth(io_channel, server_party_id, 'download_model_weight')
                if expected_feedback == 'download_model_weight':
                    _, data = recv_sth(io_channel, server_party_id)
                    print(f'download model_weight: {data}')
                    if data is None:
                        return False
                    best_model_weight = data
                    model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
                    write_content(model_weight_path, best_model_weight)
                    expected_feedback = None
                    return True
            return False

        max_retries = 10
        for i in range(max_retries):
            try:
                updated = query_best()
                if not updated:
                    logger.info(f'no new best model, retry {i}')
                    return
                ok = download_model_cfg()
                if not ok:
                    continue
                ok = download_model_weight()
                if not ok:
                    continue

                logger.info(f'update model succ at {i} times')
                break
            except Exception as e:
                print(f'retry {i}', e)
                sleep(1)
