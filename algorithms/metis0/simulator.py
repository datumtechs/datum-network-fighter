# coding:utf-8

import faulthandler
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
import numpy as np

import channel_sdk.pyio as chsdkio

from common.utils import load_cfg, merge_options
from agent_helper import flip_ucci_labels
from data_helper import read_content, recv_sth, send_sth, write_content

faulthandler.enable()

np.set_printoptions(suppress=True)
log = logging.getLogger(__name__)


class Simulator:
    def __init__(self,
                 channel_config: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_dir: str):
        log.info(f'channel_config:{channel_config}')
        log.info(f'cfg_dict:{cfg_dict}')
        log.info(f'data_party:{data_party}, result_party:{result_party}, results_dir:{results_dir}')
        assert isinstance(channel_config, str), 'type of channel_config must be str'
        assert isinstance(cfg_dict, dict), 'type of cfg_dict must be dict'
        assert isinstance(data_party, (list, tuple)), 'type of data_party must be list or tuple'
        assert isinstance(result_party, (list, tuple)), 'type of result_party must be list or tuple'
        assert isinstance(results_dir, str), 'type of results_dir must be str'

        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.compute_parties = self._get_compute_parties()
        self.party_id = cfg_dict['party_id']

        self.dynamic_parameter = cfg_dict['dynamic_parameter']

        self.output_file = os.path.join(results_dir, 'model')

        self.check_parameters()

        self.io_channel = None
        self._channel_just_for_keep_ref_count = None
        self.executor = None

    def check_parameters(self):
        log.info(f'check parameter start.')
        log.info(f'check parameter finish.')

    def run(self):
        self.create_channel()  # for each party
        cfg = self.setup_cfg()

        if self.party_id in self.data_party:
            log.info(f'party:{self.party_id} is data_party')
            self.data_party_run(cfg)
        elif self.party_id in self.result_party:
            log.info(f'party:{self.party_id} is result_party')
            self.result_party_run(cfg)
        else:  # just compute node to simulate
            import simulate
            simulate.start(cfg)

    def setup_cfg(self):
        from easydict import EasyDict as edict
        from xiangqi import get_iccs_action_space

        cfg = edict({'entry': {}})
        cfg.entry.channel_config = json.loads(self.channel_config)
        cfg.entry.data_party = self.data_party
        cfg.entry.result_party = self.result_party
        cfg.entry.party_id = self.party_id

        file_path = os.path.split(os.path.realpath(__file__))[0]
        config_file = os.path.join(file_path, 'config.yaml')
        metis0_default_cfg = load_cfg(config_file)
        log.info(f'metis0_default_cfg:{metis0_default_cfg}')
        cfg.update(metis0_default_cfg)

        merge_options(cfg, self.dynamic_parameter)
        log.info(cfg)
        cfg.labels = get_iccs_action_space()
        cfg.n_labels = len(cfg.labels)
        flipped = flip_ucci_labels(cfg.labels)
        cfg.unflipped_index = [cfg.labels.index(x) for x in flipped]

        log.info(f'cur work dir: {os.getcwd()}, code dir: {file_path}, create new model? {cfg.opts.new}')
        os.makedirs(os.path.join(file_path, cfg.resource.data_dir), exist_ok=True)
        os.makedirs(os.path.join(file_path, cfg.resource.model_dir), exist_ok=True)
        os.makedirs(os.path.join(file_path, cfg.resource.next_generation_model_dir), exist_ok=True)

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

    def handle_data_party(self, remote_nodeid, recved, cfg):
        try:
            if recved is None:
                pass
            elif recved == 'query_best_model':
                model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
                try:
                    m_time = os.path.getmtime(model_weight_path)
                except FileNotFoundError as e:
                    m_time = 0
                send_sth(self.io_channel, remote_nodeid, str(m_time).encode())
            elif recved == 'download_model_cfg':
                model_config_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_config_path)
                data = read_content(model_config_path)
                send_sth(self.io_channel, remote_nodeid, data)
            elif recved == 'download_model_weight':
                model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
                data = read_content(model_weight_path)
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

    def handle_result_party(self, remote_nodeid, recved, cfg):
        try:
            if recved is None:
                pass
            elif recved[:11] == b'upload_data':  # the first 11 bytes is 'upload_data', the rest is the data
                game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
                path = os.path.join(cfg.resource.play_data_dir, cfg.resource.play_data_filename_tmpl % game_id)
                write_content(path, recved[11:])
        except Exception as e:
            log.warn(f'exception {e} when send to {remote_nodeid}')

        return self.executor.submit(lambda t: recv_sth(*t), (self.io_channel, remote_nodeid))


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


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    """
    This is the entrance to this module
    """
    install_pkg('xiangqi')
    install_pkg('easydict')

    sim = Simulator(channel_config, cfg_dict, data_party, result_party, results_dir)
    sim.run()
