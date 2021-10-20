# coding:utf-8

import sys
sys.path.append("..")
import os
import math
import json
import time
import logging
import shutil
import numpy as np
import pandas as pd
import latticex.psi as psi
import channel_sdk


log = logging.getLogger(__name__)

class PrivateSetIntersection(object):
    '''
    private set intersection.
    '''

    def __init__(self,
                 channel_config: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_dir: str):
        log.info(f"channel_config:{channel_config}, cfg_dict:{cfg_dict}, data_party:{data_party}, "
                 f"result_party:{result_party}, results_dir:{results_dir}")
        assert isinstance(channel_config, str), "type of channel_config must be string"
        assert isinstance(cfg_dict, dict), "type of cfg_dict must be dict"
        assert isinstance(data_party, (list, tuple)), "type of data_party must be list or tuple"
        assert isinstance(result_party, (list, tuple)), "type of result_party must be list or tuple"
        assert isinstance(results_dir, str), "type of results_dir must be str"
        
        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.party_id = cfg_dict["party_id"]
        self.input_file = cfg_dict["data_party"].get("input_file")
        self.id_column_name = cfg_dict["data_party"].get("key_column")
        self.feature_column_name = cfg_dict["data_party"].get("selected_columns")
        
        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.psi_type = dynamic_parameter.get("psi_type")  # default 'T_V1_Basic_GLS254'
        # self.receiveParty = dynamic_parameter.get("receiveParty")   # 0:P0, 1:P1, 2: P0 and P1

        self.output_file = os.path.join(results_dir, "psi_result.csv")
                
    def run(self):
        '''
        run psi
        '''

        psihandler = psi.PSIHandler()
        log.info("start set log.")
        psihandler.log_to_stdout(True)
        psihandler.set_loglevel(psi.LogLevel.Info)
        log.info("start set recv party.")
        psihandler.set_recv_party(2, "")

        log.info("start create and set channel.")
        self.create_set_channel()
        log.info("start activate.")
        psihandler.activate(self.psi_type, "")
        log.info("finish activate.")

        log.info("extract id.")
        input_file = self.extract_id()
        log.info("start prepare data.")
        psihandler.prepare(input_file, taskid="")
        log.info("start run.")
        psihandler.run(input_file, self.output_file, taskid="")
        log.info("finish run.")
        run_stats = psihandler.get_perf_stats(True, "")
        run_stats = run_stats.replace('\n', '').replace(' ', '')
        log.info(f"run stats: {run_stats}")
        log.info("start deactivate.")
        psihandler.deactivate("")
        log.info("remove temp dir.")
        self.remove_temp_dir()
        log.info("psi finish.")
    
    def create_set_channel(self):
        '''
        create and set channel.
        '''
        iohandler = psi.IOHandler()
        io_channel = channel_sdk.grpc.APIManager()
        log.info("start create channel.")
        channel = io_channel.create_channel(self.party_id, self.channel_config)
        log.info("start set channel.")
        iohandler.set_channel("", channel)
        log.info("set channel success.")   
    
    def extract_id(self):
        '''
        Extract feature columns or label column from input file,
        and then divide them into train set and validation set.
        '''
        train_x = ""
        train_y = ""
        val_x = ""
        val_y = ""
        temp_dir = self.get_temp_dir()
        input_file = os.path.join(temp_dir, f"input_{self.party_id}.csv")
        
        # if self.party_id in self.data_party:
        usecol = [self.id_column_name]
        input_data = pd.read_csv(self.input_file, usecols=usecol, dtype="str")
        input_data.to_csv(input_file, header=True, index=False)
        return input_file
    
    def get_temp_dir(self):
        '''
        Get the directory for temporarily saving files
        '''
        temp_dir = os.path.join(os.path.dirname(self.output_file), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def remove_temp_dir(self):
        '''
        Delete all files in the temporary directory, these files are some temporary data.
        Only delete temp file.
        '''
        temp_dir = self.get_temp_dir()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def remove_output_dir(self):
        '''
        Delete all files in the temporary directory, these files are some temporary data.
        This is used to delete all output files of the non-resulting party
        '''
        temp_dir = os.path.dirname(self.output_file)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    psi = PrivateSetIntersection(channel_config, cfg_dict, data_party, result_party, results_dir)
    psi.run()
