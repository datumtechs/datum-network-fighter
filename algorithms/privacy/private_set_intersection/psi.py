# coding:utf-8

import os
import sys
import math
import json
import time
import logging
import shutil
import traceback
import numpy as np
import pandas as pd
import latticex.psi as psi
from functools import wraps


class LogWithStage():
    def __init__(self, name):
        self.run_stage = 'init log.'
        self.logger = logging.getLogger(name)

    def info(self, content):
        self.run_stage = content
        self.logger.info(content)
    
    def debug(self, content):
        self.logger.debug(content)
log = LogWithStage(__name__)


class ErrorTraceback():
    def __init__(self, algo_type):
        self.algo_type = algo_type
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                log.info(f"start {func.__name__} function. algo: {self.algo_type}.")
                result = func(*args, **kwargs)
                log.info(f"finish {func.__name__} function. algo: {self.algo_type}.")
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                all_error = traceback.extract_tb(exc_traceback)
                error_algo_file = all_error[0].filename
                error_filename = os.path.split(error_algo_file)[1]
                error_lineno, error_function = [], []
                for one_error in all_error:
                    if one_error.filename == error_algo_file:  # only report the algo file error
                        error_lineno.append(one_error.lineno)
                        error_function.append(one_error.name)
                error_msg = repr(e)
                raise Exception(f"<ALGO>:{self.algo_type}. <RUN_STAGE>:{log.run_stage} "
                                f"<ERROR>: {error_filename},{error_lineno},{error_function},{error_msg}")
            return result
        return wrapper

class BaseAlgorithm(object):
    def __init__(self,
                 io_channel,
                 cfg_dict: dict,
                 data_party: list,
                 compute_party: list,
                 result_party: list,
                 results_dir: str):
        log.info(f"cfg_dict:{cfg_dict}")
        log.info(f"data_party:{data_party}, compute_party:{compute_party}, result_party:{result_party}, results_dir:{results_dir}")
        self.check_params_type(cfg_dict=(cfg_dict, dict), data_party=(data_party, list), compute_party=(compute_party, list), 
                                result_party=(result_party, list), results_dir=(results_dir, str))        
        log.info(f"start get input parameter.")
        self.io_channel = io_channel
        self.data_party = list(data_party)
        self.compute_party = list(compute_party)
        self.result_party = list(result_party)
        self.results_dir = results_dir
        self.parse_algo_cfg(cfg_dict)
        self.check_parameters()
        self.temp_dir = self.get_temp_dir()
        log.info("finish get input parameter.")
    
    def check_params_type(self, **kargs):
        for key,value in kargs.items():
            assert isinstance(value[0], value[1]), f'{key} must be type({value[1]}), not {type(value[0])}'
    
    def parse_algo_cfg(self, cfg_dict):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} fuction is not implemented.')
    
    def check_parameters(self):
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} fuction is not implemented.')
        
    def get_temp_dir(self):
        '''
        Get the directory for temporarily saving files
        '''
        temp_dir = os.path.join(self.results_dir, 'temp')
        self.mkdir(temp_dir)
        return temp_dir
    
    def remove_temp_dir(self):
        '''
        for result party, only delete the temp dir.
        for non-result party, that is data and compute party, delete the all results
        '''
        if self.party_id in self.result_party:
            temp_dir = self.temp_dir
        else:
            temp_dir = self.results_dir
        self.remove_dir(temp_dir)
    
    def mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def remove_dir(self, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)


class PrivateSetIntersection(BaseAlgorithm):
    '''
    private set intersection.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.output_file = os.path.join(self.results_dir, "psi_result.csv")
        self.sdk_log_level = 3  # Trace=0, Debug=1, Audit=2, Info=3, Warn=4, Error=5, Fatal=6, Off=7

    def parse_algo_cfg(self, cfg_dict):
        '''
        cfg_dict:
        {
            "self_cfg_params": {
                "party_id": "data1",
                "input_data": [
                    {
                        "input_type": 1,
                        "data_type": 1,
                        "data_path": "path/to/data",
                        "key_column": "col1",
                        "selected_columns": ["col2", "col3"]
                    }
                ]
            },
            "algorithm_dynamic_params": {
                "use_alignment": true,
                "label_owner": "data1",
                "label_column": "diagnosis",
                "psi_type": "T_V1_Basic_GLS254",
                "data_flow_restrict": {
                    "data1": ["compute1"],
                    "data2": ["compute2"],
                    "compute1": ["result1"],
                    "compute2": ["result2"]
                }
            }
        }
        '''
        self.party_id = cfg_dict["self_cfg_params"]["party_id"]
        input_data = cfg_dict["self_cfg_params"]["input_data"]
        if self.party_id in self.data_party:
            for data in input_data:
                input_type = data["input_type"]
                data_type = data["data_type"]
                if input_type == 1:
                    self.input_file = data["data_path"]
                    self.key_column = data.get("key_column")
                    self.selected_columns = data.get("selected_columns")
                else:
                    raise Exception("paramter error. input_type only support 1")
        
        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]
        self.use_alignment = dynamic_parameter["use_alignment"]
        self.label_owner = dynamic_parameter.get("label_owner")
        self.label_column = dynamic_parameter.get("label_column")
        if self.use_alignment and (self.party_id == self.label_owner):
            self.data_with_label = True
        else:
            self.data_with_label = False
        if not self.use_alignment:
            self.selected_columns = []
        self.psi_type = dynamic_parameter.get("psi_type", "T_V1_Basic_GLS254")  # default 'T_V1_Basic_GLS254'
        self.data_flow_restrict = dynamic_parameter.get("data_flow_restrict")

    def check_parameters(self):
        assert len(self.data_party) == 2, f"length of data_party must be 2, not {len(self.data_party)}."
        assert len(self.result_party) in [1, 2], f"length of result_party must be 1 or 2, not {len(self.result_party)}."
        self._check_input_data()
        self.check_params_type(data_flow_restrict=(self.data_flow_restrict, dict))

    def _check_input_data(self):
        if self.party_id in self.data_party:
            self.check_params_type(data_path=(self.input_file, str))
            self.input_file = self.input_file.strip()
            if os.path.exists(self.input_file):
                file_suffix = os.path.splitext(self.input_file)[-1][1:]
                assert file_suffix == "csv", f"input_file must csv file, not {file_suffix}"
                assert self.key_column, f"key_column can not empty. key_column={self.key_column}"
                if self.use_alignment:
                    assert self.selected_columns, f"selected_columns can not empty. selected_columns={self.selected_columns}"
                input_columns = pd.read_csv(self.input_file, nrows=0)
                input_columns = list(input_columns.columns)
                assert self.key_column in input_columns, f"key_column:{self.key_column} not in input_file"
                error_col = []
                for col in self.selected_columns:
                    if col not in input_columns:
                        error_col.append(col)   
                assert not error_col, f"selected_columns:{error_col} not in input_file"
                assert self.key_column not in self.selected_columns, f"key_column:{self.key_column} can not in selected_columns"
                if self.data_with_label:
                    assert self.label_column in input_columns, f"label_column:{self.label_column} not in input_file"
                    assert self.label_column not in self.selected_columns, f"label_column:{self.label_column} can not in selected_columns"
            else:
                raise Exception(f"input_file is not exist. input_file={self.input_file}")

    def run(self):
        log.info("start create and set channel.")
        log.info("start extract data.")
        usecols_file = self._extract_data_column()
        log.info("start send_data_to_compute_party.")
        self._send_data_to_compute_party(usecols_file)
        psi_output_file = os.path.join(self.temp_dir, "psi_sdk_output.csv")
        alignment_output_file = self.output_file
        if self.party_id in self.compute_party:
            log.info("start extract key column.")
            key_col_file, key_col_name, usecols_data = self._extract_key_column(usecols_file)
            log.info("start run psi sdk.")
            self._run_psi_sdk(key_col_file, psi_output_file)
            log.info("start alignment result.")
            self._alignment_result(psi_output_file, usecols_data, alignment_output_file, key_col_name)
        log.info("start send data to result party.")
        self._send_data_to_result_party(alignment_output_file)
        log.info("finish send data to result party.")
        result_path, result_type = '', ''
        if self.party_id in self.result_party:
            result_path = alignment_output_file
            result_type = 'csv'
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("psi all success.")
        return result_path, result_type
    
    def _send_data_to_compute_party(self, data_path):
        if self.party_id in self.data_party:
            compute_party = self.data_flow_restrict[self.party_id][0]
            self.io_channel.send_data_to_other_party(compute_party, data_path)
        elif self.party_id in self.compute_party:
            for party in self.data_party:
                if self.party_id == self.data_flow_restrict[party][0]:
                    self.io_channel.recv_data_from_other_party(party, data_path)
        else:
            pass
    
    def _send_data_to_result_party(self, data_path):
        if self.party_id in self.compute_party:
            result_party = self.data_flow_restrict[self.party_id][0]
            self.io_channel.send_data_to_other_party(result_party, data_path)
        elif self.party_id in self.result_party:
            for party in self.compute_party:
                if self.party_id == self.data_flow_restrict[party][0]:
                    self.io_channel.recv_data_from_other_party(party, data_path)
        else:
            pass

    def _extract_data_column(self):
        '''
        Extract data column from input file,
        and then write to a new file.
        '''
        usecols_file = os.path.join(self.temp_dir, f"usecols_{self.party_id}.csv")

        if self.party_id in self.data_party:
            use_cols = [self.key_column] + self.selected_columns
            if self.data_with_label:
                use_cols += [self.label_column]
            log.info("read input file and write to new file.")
            usecols_data = pd.read_csv(self.input_file, usecols=use_cols, dtype="str")
            usecols_data = usecols_data[use_cols]
            usecols_data.to_csv(usecols_file, header=True, index=False)
        return usecols_file
    
    def _extract_key_column(self, usecols_file):
        usecols_data = pd.read_csv(usecols_file, header=0, dtype="str")
        usecols = list(usecols_data.columns)
        key_col_name = usecols[0]
        if self.use_alignment:
            key_data = usecols_data[key_col_name]
            key_col_file = os.path.join(self.temp_dir, f"key_col_{self.party_id}.csv")
            key_data.to_csv(key_col_file, header=True, index=False)
        else:
            key_col_file = usecols_file
        return key_col_file, key_col_name, usecols_data

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
        psihandler.set_recv_party(2, "")

        log.info("start create iohandler.")
        iohandler = psi.IOHandler()
        log.info("start set channel.")
        iohandler.set_channel("", self.io_channel.channel)
        log.info("start activate.")
        psihandler.activate(self.psi_type, "")
        log.info("finish activate.")
        log.info("start psihandler prepare data.")
        psihandler.prepare(input_file, taskid="")
        log.info("start psihandler run.")
        psihandler.run(input_file, output_file, taskid="")
        log.info("finish psihandler run.")
        run_stats = psihandler.get_perf_stats(True, "")
        run_stats = run_stats.replace('\n', '').replace(' ', '')
        log.info(f"run stats: {run_stats}")
        log.info("start deactivate.")
        psihandler.deactivate("")
        log.info("finish deactivate.")
    
    def _alignment_result(self, psi_output_file, usecols_data, alignment_output_file, key_col_name):
        '''
        for the compute_party, sort the key_col values and alignment the select_columns.
        '''
        if os.path.exists(psi_output_file):
            psi_result = pd.read_csv(psi_output_file, header=None, dtype="str")
            psi_result = pd.DataFrame(psi_result.values, columns=[key_col_name])
            psi_result.sort_values(by=[key_col_name], ascending=True, inplace=True)
            if self.use_alignment:
                alignment_result = pd.merge(psi_result, usecols_data, on=key_col_name)
            else:
                alignment_result = psi_result
            alignment_result.to_csv(alignment_output_file, index=False, header=True)
            log.info(f"alignment_result shape: {alignment_result.shape}")
        else:
            use_cols = list(usecols_data.columns)
            log.info(f"psi_result is Empty, only have Column name: {use_cols}")
            with open(alignment_output_file, 'w') as output_f:
                output_f.write(','.join(use_cols)+"\n")


@ErrorTraceback("psi")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    psi = PrivateSetIntersection(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type = psi.run()
    return result_path, result_type
