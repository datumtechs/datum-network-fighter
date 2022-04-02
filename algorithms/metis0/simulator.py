# coding:utf-8

import faulthandler
import logging
import os
from io import StringIO
from multiprocessing.pool import ThreadPool

import channel_sdk.pyio as chsdkio
import numpy as np
from common.utils import load_cfg
from agent_helper import flip_ucci_labels


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
        log.info(f"channel_config:{channel_config}")
        log.info(f"cfg_dict:{cfg_dict}")
        log.info(f"data_party:{data_party}, result_party:{result_party}, results_dir:{results_dir}")
        assert isinstance(channel_config, str), "type of channel_config must be str"
        assert isinstance(cfg_dict, dict), "type of cfg_dict must be dict"
        assert isinstance(data_party, (list, tuple)), "type of data_party must be list or tuple"
        assert isinstance(result_party, (list, tuple)), "type of result_party must be list or tuple"
        assert isinstance(results_dir, str), "type of results_dir must be str"

        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.party_id = cfg_dict["party_id"]

        self.dynamic_parameter = cfg_dict["dynamic_parameter"]

        self.output_file = os.path.join(results_dir, "model")

        self.check_parameters()

    def check_parameters(self):
        log.info(f"check parameter start.")
        log.info(f"check parameter finish.")

    def run(self):
        self.create_set_channel()  # for each party
        cfg = self.setup_cfg()

        if self.party_id in self.data_party:
            log.info(f"party:{self.party_id} is data_party")
            self.data_party_run(cfg)
        elif self.party_id in self.result_party:
            log.info(f"party:{self.party_id} is result_party")
            self.result_party_run(cfg)
        else:  # just compute node to simulate
            import simulate
            simulate.start(cfg)

    def setup_cfg(self):
        from easydict import EasyDict as edict
        import yaml
        from xiangqi import get_iccs_action_space

        cfg = edict({'entry': {}})
        cfg.entry.channel_config = self.channel_config
        cfg.entry.data_party = self.data_party
        cfg.entry.result_party = self.result_party
        cfg.entry.party_id = self.party_id

        cfg.update(self.dynamic_parameter)
        cfg.labels = get_iccs_action_space()
        cfg.n_labels = len(cfg.labels)
        flipped = flip_ucci_labels(cfg.labels)
        cfg.unflipped_index = [cfg.labels.index(x) for x in flipped]

        return cfg

    def create_set_channel(self):
        '''
        create and set channel.
        '''
        self.io_channel = chsdkio.APIManager()

        log.info("start create channel")
        self.io_channel.create_channel(self.party_id, self.channel_config)

    def _get_compute_parties(self):
        return list(self.channel_config['COMPUTATION_NODES'].keys())

    def data_party_run(self, cfg):
        from data_helper import read_content

        def send_sth(io_channel, remote_nodeid, data):
            io_channel.Send(remote_nodeid, len(data).to_bytes(4, byteorder="big"))
            io_channel.Send(remote_nodeid, data)

        comp_parties = self._get_compute_parties()
        pool = ThreadPool(processes=len(comp_parties))

        model_config_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_config_path)
        data = read_content(model_config_path)        
        pool.map_async(send_sth, [(self.io_channel, x, data) for x in comp_parties])

        model_weight_path = os.path.join(cfg.resource.model_dir, cfg.resource.model_best_weight_path)
        data = read_content(model_weight_path)
        pool.map_async(send_sth, [(self.io_channel, x, data) for x in comp_parties])

    def result_party_run(self, cfg):
        def recv_sth(io_channel, remote_nodeid):
            recv_data = io_channel.Recv(remote_nodeid, 4)
            data_len = int.from_bytes(recv_data, byteorder="big")
            recv_data = io_channel.Recv(remote_nodeid, data_len)
            return remote_nodeid, recv_data

        comp_parties = self._get_compute_parties()        
        pool = ThreadPool(processes=len(comp_parties))
        results = pool.map_async(recv_sth, [(self.io_channel, x) for x in comp_parties])
        for x in comp_parties:
            result = pool.apply_async(recv_sth, (self.io_channel, x))
            data = result.get(timeout=1)


def install_pkg(pkg_name: str, pkg_version: str=None, whl_file: str=None):
    '''
    install the package if it is not installed.
    '''
    import pkg_resources
    installed_pkgs = pkg_resources.working_set
    for i in installed_pkgs:
        if i.project_name == pkg_name:
            if pkg_version is None:
                return True
            i_ver = tuple(map(int, (i.split("."))))
            pkg_ver = tuple(map(int, (pkg_version.split("."))))
            if i_ver >= pkg_ver:
                return True
            return False
    import subprocess
    ob = pkg_name if whl_file is None else whl_file
    cmd = f"pip install {ob}"
    subprocess.run(cmd, shell=True)
    return True    


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    install_pkg("xiangqi")
    install_pkg("easydict")

    sim = Simulator(channel_config, cfg_dict, data_party, result_party, results_dir)
    sim.run()
