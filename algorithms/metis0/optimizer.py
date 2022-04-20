# coding:utf-8

import faulthandler
import logging
import os
from datetime import datetime
import json
import channel_sdk.pyio as chsdkio
import numpy as np
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from common.utils import load_cfg, merge_options
from agent_helper import flip_ucci_labels
from data_helper import read_content, recv_sth, send_sth, write_content, zip_and_b64encode, b64decode_and_unzip, install_pkg

faulthandler.enable()

np.set_printoptions(suppress=True)
log = logging.getLogger(__name__)


class Optimizer:
    def __init__(self,
                 channel_config: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_dir: str,
                 upper_args: dict):
        log.info(f'channel_config:{channel_config}')
        log.info(f'cfg_dict:{cfg_dict}')
        log.info(f'data_party:{data_party}, result_party:{result_party}, results_dir:{results_dir}')

        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.compute_parties = self._get_compute_parties()
        self.results_dir = results_dir

        self.party_id = cfg_dict['party_id']
        self.all_cfg = self.setup_cfg(cfg_dict)

        self.output_file = os.path.join(results_dir, 'model')

        self.io_channel = None
        self._channel_just_for_keep_ref_count = None
        self.executor = None

    def run(self):
        self.create_channel()  # for each party
        if self.party_id in self.data_party:
            log.info(f'party:{self.party_id} is data_party')
            self.data_party_run(self.all_cfg)
        elif self.party_id in self.result_party:
            log.info(f'party:{self.party_id} is result_party')
            self.result_party_run(self.all_cfg)
        else:  # compute node for training
            import optimize
            optimize.start(self.all_cfg, self.io_channel)

    def setup_cfg(self, user_cfg):
        from easydict import EasyDict as edict
        from xiangqi import get_iccs_action_space

        cfg = edict({'entry': {}})
        cfg.entry.channel_config = self.channel_config  # it is a json-like str
        cfg.entry.data_party = self.data_party
        cfg.entry.result_party = self.result_party
        cfg.entry.party_id = self.party_id

        file_path = os.path.split(os.path.realpath(__file__))[0]
        config_file = os.path.join(file_path, 'config.yaml')
        metis0_default_cfg = load_cfg(config_file)
        # log.info(f'metis0_default_cfg:{metis0_default_cfg}')
        cfg.update(metis0_default_cfg)

        merge_options(cfg, user_cfg['dynamic_parameter'])
        user_cfg.pop('party_id')
        user_cfg.pop('dynamic_parameter')
        merge_options(cfg, user_cfg)

        # TODO: substitute variable

        log.info(cfg)
        cfg.labels = get_iccs_action_space()
        cfg.n_labels = len(cfg.labels)
        flipped = flip_ucci_labels(cfg.labels)
        cfg.unflipped_index = [cfg.labels.index(x) for x in flipped]

        log.info(f'cur work dir: {os.getcwd()}, code dir: {file_path}, create new model? {cfg.opts.new}')
        # os.makedirs(os.path.join(file_path, cfg.resource.data_dir), exist_ok=True)
        os.makedirs(os.path.join(file_path, cfg.resource.model_dir), exist_ok=True)
        os.makedirs(os.path.join(file_path, cfg.resource.next_generation_model_dir), exist_ok=True)
        # os.makedirs(os.path.join(self.results_dir, cfg.resource.play_data_dir), exist_ok=True)

        return cfg

    def create_channel(self):
        self.io_channel = chsdkio.APIManager()
        self._channel_just_for_keep_ref_count = self.io_channel.create_channel(self.party_id, self.channel_config)

    def _get_compute_parties(self):
        ch_cfg = json.loads(self.channel_config)
        return list(ch_cfg['COMPUTATION_NODES'].keys())

    def data_party_run(self, cfg):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=3)

        futures = {self.executor.submit(lambda t: recv_sth(*t), (self.io_channel, c)) for c in self.compute_parties}
        while True:
            try:
                done, not_done = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            except TimeoutError as e:
                log.info(f'TimeoutError: {e}')
                continue
            futures = not_done
            for f in done:
                new_future = self.handle_data_party(*f.result(), cfg)
                if new_future:
                    futures.add(new_future)

    def handle_data_party(self, remote_nodeid, recved: bytes, cfg):
        try:
            if recved is None:
                pass
            elif recved == b'query_best_model':
                model_weight_path = cfg.resource.model_best_weight_path
                try:
                    m_time = os.path.getmtime(model_weight_path)
                except FileNotFoundError as e:
                    log.info(f'FileNotFoundError: {e}, {os.getcwd()}, {os.path.abspath(model_weight_path)}')
                    m_time = 0
                send_sth(self.io_channel, remote_nodeid, str(m_time))
            elif recved == b'download_model_cfg':
                model_config_path = cfg.resource.model_best_config_path
                data = read_content(model_config_path, text=True)
                send_sth(self.io_channel, remote_nodeid, data)
            elif recved == b'download_model_weight':
                model_weight_path = cfg.resource.model_best_weight_path
                data = read_content(model_weight_path, text=False)
                data = zip_and_b64encode(data)
                send_sth(self.io_channel, remote_nodeid, data)
        except Exception as e:
            log.warn(f'exception {e} when send to {remote_nodeid}')

        return self.executor.submit(lambda t: recv_sth(*t), (self.io_channel, remote_nodeid))

    def result_party_run(self, cfg):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=3)

        futures = {self.executor.submit(lambda t: recv_sth(*t), (self.io_channel, c)) for c in self.compute_parties}
        while True:
            try:
                done, not_done = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            except TimeoutError as e:
                log.info(f'TimeoutError: {e}')
                continue
            futures = not_done
            for f in done:
                new_future = self.handle_result_party(*f.result(), cfg)
                if new_future:
                    futures.add(new_future)

    def handle_result_party(self, remote_nodeid, recved: bytes, cfg):
        class recv_nothing(Exception):
            pass

        class unkonwn_data(Exception):
            pass
        accepted_cmds = [b'upload_model_cfg', b'upload_model_weight']
        cmds_len = [len(c) for c in accepted_cmds]
        try:
            if recved is None:
                raise recv_nothing

            for l, c in zip(cmds_len, accepted_cmds):
                if recved[:l] == c:      # start with cmd
                    recved = recved[l:]  # then real data
                    break
            else:
                raise unkonwn_data
            if c == b'upload_model_cfg':
                sep = recved.find(b' ')
                assert sep != -1
                dir_ = recved[:sep]
                recved = recved[sep + 1:]
                model_dir = os.path.join(cfg.resource.next_generation_model_dir,  dir_.decode())
                os.makedirs(model_dir, exist_ok=True)              
                path = os.path.join(model_dir, cfg.resource.next_generation_model_config_filename)
                log.info(f'write model config to {path}, {os.path.abspath(path)}')
                write_content(path, recved)
            elif c == b'upload_model_weight':
                sep = recved.find(b' ')
                assert sep != -1
                dir_ = recved[:sep]
                recved = recved[sep + 1:]
                recved = b64decode_and_unzip(recved)
                model_dir = os.path.join(cfg.resource.next_generation_model_dir,  dir_.decode())
                os.makedirs(model_dir, exist_ok=True)              
                path = os.path.join(model_dir, cfg.resource.next_generation_model_weight_filename)
                log.info(f'write model weight to {path}, {os.path.abspath(path)}')
                write_content(path, recved)
        except recv_nothing:
            pass
        except unkonwn_data:
            log.warn(f'unknown data from {remote_nodeid}, {recved[:20]}')
        except Exception as e:
            log.warn(f'exception {e} when send to {remote_nodeid}')

        return self.executor.submit(lambda t: recv_sth(*t), (self.io_channel, remote_nodeid))


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str, **kwargs):
    install_pkg('xiangqi')
    install_pkg('easydict')
    opt = Optimizer(channel_config, cfg_dict, data_party, result_party, results_dir, kwargs)
    opt.run()
