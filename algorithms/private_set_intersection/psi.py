# coding:utf-8

import os
import sys
import math
import json
import time
import logging
import shutil
import numpy as np
import pandas as pd
import latticex.psi as psi
import channel_sdk.pyio as io


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
        self.key_column = cfg_dict["data_party"].get("key_column")
        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.psi_type = dynamic_parameter.get("psi_type", "T_V1_Basic_GLS254")  # default 'T_V1_Basic_GLS254'

        self._check_parameters()
        self.result_type = self._get_result_type()        
        self.output_file = os.path.join(results_dir, "psi_result.csv")
        self.sdk_log_level = 3  # Trace=0, Debug=1, Audit=2, Info=3, Warn=4, Error=5, Fatal=6, Off=7
                
    def _check_parameters(self):
        assert len(self.data_party) == 2, f"length of data_party must be 2, not {len(self.data_party)}."
        assert len(self.result_party) in [1, 2], f"length of result_party must be 1 or 2, not {len(self.result_party)}."
        if len(self.result_party) == 2:
            assert self.result_party[0] == self.data_party[0] and self.result_party[1] == self.data_party[1], \
                    f"result_party:{self.result_party} not equal to data_party:{self.data_party}"
        else:
            assert self.result_party[0] in self.data_party, \
                    f"result_party:{self.result_party} not in data_party:{self.data_party}"
        self._check_input_file()

    def _check_input_file(self):
        self.input_file = self.input_file.strip()
        if os.path.exists(self.input_file):
            file_suffix = os.path.splitext(self.input_file)[-1]
            assert file_suffix == ".csv", f"input_file must csv file, not {file_suffix}"
            assert self.key_column, f"key_column can not empty. key_column={self.key_column}"
            input_columns = pd.read_csv(self.input_file, nrows=0)
            input_columns = list(input_columns.columns)
            assert self.key_column in input_columns, f"key_column:{self.key_column} not in input_file"
        else:
            raise Exception(f"input_file is not exist. input_file={self.input_file}")
    
    def _get_result_type(self):
        '''
        assume data_party = [P0, P1]:
            if result_party = [P0, P1], then result_type = 2
            if result_party = [P0], then result_type = 0
            if result_party = [P1], then result_type = 1
        '''
        if len(self.result_party) == 2:
            result_type = 2
        else:
            if self.result_party[0] == self.data_party[0]:
                result_type = 0
            else:
                result_type = 1
        return result_type

    def run(self):
        log.info("start extract key_column.")
        temp_input_file = self._extract_key_column()
        log.info("start get temp_output file name.")
        temp_dir = self._get_temp_dir()
        temp_output_file = os.path.join(temp_dir, "psi_sdk_output.csv")

        log.info("start run psi sdk.")
        self._run_psi_sdk(temp_input_file, temp_output_file)

        log.info("start process result.")
        self._process_result(temp_output_file)
        log.info("start remove temp dir.")
        self._remove_temp_dir()
        log.info("psi all success.")
    
    def _extract_key_column(self):
        '''
        Extract key column from input file,
        and then write to a new file.
        '''
        train_x = ""
        train_y = ""
        val_x = ""
        val_y = ""
        temp_dir = self._get_temp_dir()
        key_col_file = os.path.join(temp_dir, f"key_column_{self.party_id}.csv")
        log.info("read input file and write key_column to new file.")
        key_col = pd.read_csv(self.input_file, usecols=[self.key_column], dtype="str")
        key_col.to_csv(key_col_file, header=True, index=False)
        return key_col_file
    
    def _run_psi_sdk(self, input_file, output_file):
        '''
        run psi sdk
        '''
        log.info("start create psihandler.")
        psihandler = psi.PSIHandler()
        log.info("start set log.")
        psihandler.log_to_stdout(True)
        psihandler.set_loglevel(self.sdk_log_level)
        log.info("start set recv party.")
        psihandler.set_recv_party(self.result_type, "")

        log.info("start create and set channel.")
        self._create_set_channel()
        log.info("start activate.")
        psihandler.activate(self.psi_type, "")
        log.info("finish activate.")

        log.info("start prepare data.")
        psihandler.prepare(input_file, taskid="")
        log.info("start run.")
        psihandler.run(input_file, output_file, taskid="")
        log.info("finish run.")
        run_stats = psihandler.get_perf_stats(True, "")
        run_stats = run_stats.replace('\n', '').replace(' ', '')
        log.info(f"run stats: {run_stats}")
        log.info("start deactivate.")
        psihandler.deactivate("")
        log.info("finish deactivate.")
    
    def _process_result(self, file_name):
        '''
        for the result_party, add the col name to the beginning of the file, 
        and sort the values.
        '''
        if self.party_id in self.result_party:
            log.info(f"result party performs post-processing on the result.")
            key_col_name = "INTERSECTION"
            if os.path.exists(file_name):
                psi_result = pd.read_csv(file_name, header=None)
                psi_result = pd.DataFrame(psi_result.values, columns=[key_col_name])
                psi_result.sort_values(by=[key_col_name], ascending=True, inplace=True)
                psi_result.to_csv(self.output_file, index=False, header=True)
                log.info(f"psi_result shape: {psi_result.shape}")
            else:
                with open(self.output_file, 'w') as output_f:
                    output_f.write(key_col_name+"\n")
                log.info(f"psi_result file is Empty, only have Column name: {key_col_name}")
        else:
            log.info(f"{self.party_id} is not result party, no result.")

    def _create_set_channel(self):
        '''
        create and set channel.
        '''
        log.info("start create iohandler.")
        iohandler = psi.IOHandler()
        io_channel = io.APIManager()
        log.info("start create channel.")
        channel = io_channel.create_channel(self.party_id, self.channel_config)
        log.info("start set channel.")
        iohandler.set_channel("", channel)
        log.info("set channel success.")
    
    def _get_temp_dir(self):
        '''
        Get the directory for temporarily saving files
        '''
        temp_dir = os.path.join(os.path.dirname(self.output_file), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def _remove_temp_dir(self):
        '''
        Delete all files in the temporary directory, these files are some temporary data.
        Only delete temp file.
        '''
        temp_dir = self._get_temp_dir()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def _remove_output_dir(self):
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
    log.info("start main function.")
    psi = PrivateSetIntersection(channel_config, cfg_dict, data_party, result_party, results_dir)
    psi.run()
    log.info("finish main function.")
