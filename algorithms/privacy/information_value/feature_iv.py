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
import latticex.rosetta as rtt
from functools import wraps


np.set_printoptions(suppress=True)
rtt.set_backend_loglevel(3)  # All(0), Trace(1), Debug(2), Info(3), Warn(4), Error(5), Fatal(6)
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
        self.data_party = ["data1", "data2"]
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


class FeatureIV(BaseAlgorithm):
    '''
    private feature information value.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.output_file = os.path.join(self.results_dir, "feature_iv_result.csv")

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
                "label_owner": "data1",
                "label_column": "Y",
                "hyperparams": {
                    "binning_type": 1,  # 1:frequency, 2:distance
                    "num_bin": 5,
                    "postive_value": 1.0,
                    "negative_value": 0.0
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
        self.label_owner = dynamic_parameter["label_owner"]
        self.label_column = dynamic_parameter["label_column"]
        if self.party_id == self.label_owner:
            self.data_with_label = True
        else:
            self.data_with_label = False
        self.binning_type = dynamic_parameter.get("binning_type", 1)
        self.num_bin = dynamic_parameter.get("num_bin", 2)
        self.postive_value = dynamic_parameter.get("postive_value", 1.0)
        self.negative_value = dynamic_parameter.get("negative_value", 0.0)
        self.calc_iv_columns = dynamic_parameter["calc_iv_columns"]  # must have iv值

    def check_parameters(self):
        self._check_input_data()
        self.check_params_type(binning_type=(self.binning_type, int),
                               num_bin=(self.num_bin, int),
                               postive_value=(self.postive_value, (int, float)),
                               negative_value=(self.negative_value, (int, float)),
                               calc_iv_columns=(self.calc_iv_columns, dict))
        assert self.binning_type in [1, 2], f"binning_type only support 1,2. not {self.binning_type}"
        assert self.num_bin > 1, f"num_bin must be greater 1, not {self.num_bin}"
        assert self.calc_iv_columns, f"calc_iv_columns can not be empty, calc_iv_columns={self.calc_iv_columns}"

    def _check_input_data(self):
        if self.party_id in self.data_party:
            self.check_params_type(data_path=(self.input_file, str))
            self.input_file = self.input_file.strip()
            if os.path.exists(self.input_file):
                file_suffix = os.path.splitext(self.input_file)[-1][1:]
                assert file_suffix == "csv", f"input_file must csv file, not {file_suffix}"
                assert self.key_column, f"key_column can not empty. key_column={self.key_column}"
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
        log.info("start extract data.")
        x_file, y_file = self._extract_feature_or_label()
        log.info("start run secure feature iv.")
        result_path, result_type, extra = self._run_secure_feature_iv(x_file, y_file)
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("feature iv all success.")
        return result_path, result_type, extra 

    def _run_secure_feature_iv(self, x_file, y_file):
        '''
        run secure feature information value
        '''
        log.info("start set channel.")
        rtt.set_channel("", self.io_channel.channel)
        log.info("waiting other party connect...")
        rtt.activate("SecureNN")
        # sharing data
        log.info(f"start sharing train data. data_owner={self.data_party}, label_owner={self.label_owner}")
        # the order of the columns of shard_x is sorted(self.data_party), according to ASCII
        shard_x, shard_y = rtt.PrivateDataset(data_owner=self.data_party, label_owner=self.label_owner).load_data(x_file, y_file, header=0)
        log.info("finish sharing train data.")
        feature_num = shard_x.shape[1]
        log.info(f"feature_num = {feature_num}.")

        if self.party_id not in self.data_party:
            if self.binning_type == 1:
                binning_type_name = "Binning-frequency"
            else:
                binning_type_name = "Binning-distance"
            feature_index = list(range(feature_num))
            # feature_index = [0]
            log.info("create iv handler")
            iv_handler = rtt.SecureFeatureIV(binning_type_name, 
                                num_of_bin=self.num_bin, 
                                bin_type=self.binning_type, 
                                good_value=self.postive_value, 
                                bad_value=self.negative_value,
                                feature_index=feature_index)
            log.info("start iv fit.")
            siv = iv_handler.Fit(shard_x, shard_y)
            log.info(f"***************: {siv}")
            log.info("iv fit success.")
            iv = iv_handler.Reveal(siv, self.result_party)
            log.info(f"iv reveal success.")
        else:
            log.info("computing, please waiting for compute finish...")
        rtt.deactivate()
        log.info("finish deactivate.")
        result_path, result_type, feature_iv_with_columns = '', '', ''
        if self.party_id in self.result_party:
            feature_iv = iv.astype('float').reshape(-1,)
            iv_columns_name = self._get_iv_columns_name()
            feature_iv_with_columns = {k:v for k,v in zip(iv_columns_name, feature_iv)}
            feature_iv_with_columns = json.dumps(feature_iv_with_columns)
            df_feature_iv = {"feature_name": iv_columns_name, "information_value": feature_iv}
            iv_result = pd.DataFrame(df_feature_iv)
            iv_result.to_csv(self.output_file, header=True, index=False)
            result_path = self.output_file
            result_type = 'csv'
        return result_path, result_type, feature_iv_with_columns
    
    def _extract_feature_or_label(self):
        '''
        Extract feature columns or label column from input file,
        and then divide them into train set and validation set.
        '''
        x_file = ""
        y_file = ""
        temp_dir = self.get_temp_dir()
        if self.party_id in self.data_party:
            usecols = [self.key_column] + self.selected_columns
            if self.data_with_label:
                usecols += [self.label_column]
            
            input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str") # usecols not ensure the order of columns
            input_data = input_data[usecols]  # use for ensure the order of columns
            assert input_data.shape[0] > 0, 'input file is no data.'
            
            if self.data_with_label:
                y_data = input_data[self.label_column]
                y_file = os.path.join(temp_dir, f"y_file_{self.party_id}.csv")
                y_data.to_csv(y_file, header=True, index=False)            
            x_data = input_data[self.selected_columns]
            x_file = os.path.join(temp_dir, f"x_file_{self.party_id}.csv")
            x_data.to_csv(x_file, header=True, index=False)

        return x_file, y_file
    
    def _get_iv_columns_name(self):
        data_party = sorted(self.data_party) # because rtt.PrivateDataset().load_data has sorted self.data_party, according to ASCII
        iv_columns_name = []
        for party in data_party:
            iv_columns_name.extend(self.calc_iv_columns[party])
        return iv_columns_name
        


@ErrorTraceback("feature_iv")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    feature_iv = FeatureIV(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type, extra = feature_iv.run()
    return result_path, result_type, extra
