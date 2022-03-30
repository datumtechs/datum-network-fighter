import json
import os
from datetime import datetime
from glob import glob
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


def upload_data(path, data, cfg):
    io_channel = chsdkio.APIManager()

    logger.info("start create channel")
    channel = io_channel.create_channel(cfg.party_id, cfg.channel_config)
    if cfg.party_id in cfg.result_party:  # as server
        remote_nodeid = cfg.party_id
        recv_data = io_channel.Recv(remote_nodeid, 4)  # data length, 4 bytes
        data_len = int.from_bytes(recv_data, byteorder="big")
        recv_data = io_channel.Recv(remote_nodeid, data_len)
        write_content(path, recv_data)
    elif cfg.party_id not in cfg.data_party:  # compute node as client
        remote_nodeid = cfg.result_party[0]  # select first result party
        data_len = len(data)
        io_channel.Send(remote_nodeid, (data_len).to_bytes(4, byteorder="big"))
        io_channel.Send(remote_nodeid, data)


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
        remote_nodeid = cfg.party_id

        model_config_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_config_path)
        data = read_content(model_config_path)
        data_len = len(data)
        io_channel.Send(remote_nodeid, (data_len).to_bytes(4, byteorder="big"))
        io_channel.Send(remote_nodeid, data)

        model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
        data = read_content(model_weight_path)
        data_len = len(data)
        io_channel.Send(remote_nodeid, (data_len).to_bytes(4, byteorder="big"))
        io_channel.Send(remote_nodeid, data)

    elif cfg.party_id not in cfg.result_party:  # compute node as client
        remote_nodeid = cfg.data_party[0]  # actually, just one data party

        recv_data = io_channel.Recv(remote_nodeid, 4)  # data length, 4 bytes
        data_len = int.from_bytes(recv_data, byteorder="big")
        recv_data = io_channel.Recv(remote_nodeid, data_len)

        model_config_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_config_path)
        write_content(model_config_path, recv_data)

        recv_data = io_channel.Recv(remote_nodeid, 4)  # data length, 4 bytes
        data_len = int.from_bytes(recv_data, byteorder="big")
        recv_data = io_channel.Recv(remote_nodeid, data_len)

        model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
        write_content(model_weight_path, recv_data)
