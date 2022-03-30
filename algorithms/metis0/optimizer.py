# coding:utf-8

import faulthandler
import logging
import os

import channel_sdk.pyio as io
import numpy as np

faulthandler.enable()

np.set_printoptions(suppress=True)
log = logging.getLogger(__name__)


class Optimizer:
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

        dynamic_parameter = cfg_dict["dynamic_parameter"]

        algorithm_parameter = dynamic_parameter["algorithm_parameter"]
        self.epochs = algorithm_parameter.get("epochs", 10)
        self.batch_size = algorithm_parameter.get("batch_size", 256)
        self.learning_rate = algorithm_parameter.get("learning_rate", 0.001)

        self.output_file = os.path.join(results_dir, "model")

        self.check_parameters()

    def check_parameters(self):
        log.info(f"check parameter start.")
        assert isinstance(self.epochs, int) and self.epochs > 0, "epochs must be type(int) and greater 0"
        assert isinstance(self.batch_size, int) and self.batch_size > 0, "batch_size must be type(int) and greater 0"
        assert isinstance(self.learning_rate,
                          float) and self.learning_rate > 0, "learning rate must be type(float) and greater 0"

        log.info(f"check parameter finish.")

    def run(self):
        self.create_set_channel()

    def create_set_channel(self):
        '''
        create and set channel.
        '''
        io_channel = io.APIManager()

        log.info("start create channel")
        channel = io_channel.create_channel(self.party_id, self.channel_config)


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    opt = Optimizer(channel_config, cfg_dict, data_party, result_party, results_dir)
    opt.run()
