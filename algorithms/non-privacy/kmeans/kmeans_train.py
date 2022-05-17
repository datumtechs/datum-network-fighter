# coding:utf-8

import os
import sys
import math
import json
import time
import logging
import traceback
import numpy as np
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
from functools import wraps


np.set_printoptions(suppress=True)
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
        self.check_params_type(cfg_dict=(cfg_dict, dict), 
                                data_party=(data_party, list), compute_party=(compute_party, list), 
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
    

class KmeansTrain(BaseAlgorithm):
    '''
    Plaintext Kmeans train.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir_name = "model"
        self.model_file_name = "model.pkl"
        self.output_dir = self._get_output_dir()
        self.output_file = os.path.join(self.output_dir, self.model_file_name)
        self.model_describe_file = os.path.join(self.output_dir, "describe.json")
        self.set_random_seed(self.random_seed)
    
    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)  

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
                "hyperparams": {
                    "n_clusters": 8,
                    "init_method": "k-means++", 
                    "n_init": 10,
                    "max_iter": 300,
                    "tol": 0.0001,
                    "random_seed": null
                },
                "data_flow_restrict": {
                    "data1": ["compute1"],
                    "compute1": ["result1"]
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
                    raise Exception(f"paramter error. input_type only support 1, not {input_type}")
        
        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]               
        hyperparams = dynamic_parameter["hyperparams"]
        self.n_clusters = hyperparams.get("n_clusters", 8)
        self.init_method = hyperparams.get("init_method", "k-means++")  # k-means++, random
        self.n_init = hyperparams.get("n_init", 10)
        self.max_iter = hyperparams.get("max_iter", 300)
        self.tol = hyperparams.get("tol", 0.0001)
        self.random_seed = hyperparams.get("random_seed", None)
        self.data_flow_restrict = dynamic_parameter["data_flow_restrict"]

    def check_parameters(self):
        log.info(f"check parameter start.")
        self._check_input_data()            
        self.check_params_type(n_clusters=(self.n_clusters, int),
                               init_method=(self.init_method, str),
                               n_init=(self.n_init, int),
                               max_iter=(self.max_iter, int),
                               tol=(self.tol, float),
                               random_seed=(self.random_seed, (int, type(None))),
                               data_flow_restrict=(self.data_flow_restrict, dict))
        assert self.n_clusters > 1, f"n_clusters must be greater 1, not {self.n_clusters}"
        assert self.init_method in ["k-means++", "random"], f"init_method only support k-means++,random. not {self.init_method}"
        assert self.n_init > 0, f"n_init must be greater 0, not {self.n_init}"
        assert self.max_iter > 0, f"max_iter must be greater 0, not {self.max_iter}"
        if self.random_seed:
            assert 0 <= self.random_seed <= 2**32 - 1, f"random_seed must be between [0,2^32-1], not {self.random_seed}"
        log.info(f"check parameter finish.")
    
    def _check_input_data(self):
        if self.party_id in self.data_party:
            self.check_params_type(data_path=(self.input_file, str), 
                                   key_column=(self.key_column, str),
                                   selected_columns=(self.selected_columns, list))
            self.input_file = self.input_file.strip()
            if os.path.exists(self.input_file):
                file_suffix = os.path.splitext(self.input_file)[-1][1:]
                assert file_suffix == "csv", f"input_file must csv file, not {file_suffix}"
                input_columns = pd.read_csv(self.input_file, nrows=0)
                input_columns = list(input_columns.columns)
                assert self.key_column in input_columns, f"key_column:{self.key_column} not in input_file"
                error_col = []
                for col in self.selected_columns:
                    if col not in input_columns:
                        error_col.append(col)   
                assert not error_col, f"selected_columns:{error_col} not in input_file"
                assert self.key_column not in self.selected_columns, f"key_column:{self.key_column} can not in selected_columns"
            else:
                raise Exception(f"input_file is not exist. input_file={self.input_file}")
                        
        
    def train(self):
        '''
        Logistic regression training algorithm implementation function
        '''
        log.info("start data party extract data column.")
        usecols_file = self._extract_data_column()
        log.info("start data party send data to compute party.")
        self._send_data_to_compute_party(usecols_file)
        evaluate_result = ""
        if self.party_id in self.compute_party:
            log.info("compute party start  compute.")
            evaluate_result = self.compute(usecols_file)
        log.info("start compute party send data to result party.")
        evaluate_result = self._send_data_to_result_party(self.output_dir, evaluate_result)
        result_path, result_type = '', ''
        if self.party_id in self.result_party:
            result_path = self.output_dir
            result_type = 'dir'
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("train success all.")
        return result_path, result_type, evaluate_result

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
    
    def _send_data_to_result_party(self, data_path, evaluate_result):
        if self.party_id in self.compute_party:
            if os.path.isdir(data_path):
                temp_model_dir = os.path.join(self.temp_dir, self.model_dir_name)
                data_path = shutil.make_archive(base_name=temp_model_dir, format='zip', root_dir=data_path)
            result_party = self.data_flow_restrict[self.party_id][0]
            self.io_channel.send_data_to_other_party(result_party, data_path)
            self.io_channel.send_sth(result_party, evaluate_result)
        elif self.party_id in self.result_party:
            for party in self.compute_party:
                if self.party_id == self.data_flow_restrict[party][0]:
                    temp_model_dir = os.path.join(self.temp_dir, f'{self.model_dir_name}.zip')
                    self.io_channel.recv_data_from_other_party(party, temp_model_dir)
                    shutil.unpack_archive(temp_model_dir, self.output_dir)
                    evaluate_result = self.io_channel.recv_sth(party)
                    evaluate_result = evaluate_result.decode()
                    log.info(f'evaluate_result: {evaluate_result}')
        else:
            pass
        return evaluate_result

    def _extract_data_column(self):
        '''
        Extract data column from input file,
        and then write to a new file.
        '''
        usecols_file = os.path.join(self.temp_dir, f"usecols_{self.party_id}.csv")

        if self.party_id in self.data_party:
            use_cols = self.selected_columns
            log.info("read input file and write to new file.")
            usecols_data = pd.read_csv(self.input_file, usecols=use_cols, dtype="str")
            assert usecols_data.shape[0] > 0, 'no data after select columns.'
            usecols_data = usecols_data[use_cols]
            usecols_data.to_csv(usecols_file, header=True, index=False)
        return usecols_file

    def _read_data(self, usecols_file):
        '''
        Extract feature columns or label column from input file,
        and then divide them into train set and validation set.
        '''
        input_data = pd.read_csv(usecols_file)
        return input_data
    
    def save_model_describe(self, feature_num, feature_name, model):
        '''save model description for prediction'''
        cluster_centers = {k:v for k,v in enumerate(model.cluster_centers_.tolist())}
        model_desc = {
            "model_file_name": self.model_file_name,
            "feature_num": feature_num,
            "n_clusters": self.n_clusters,
            "tol": self.tol,
            "feature_name": feature_name,
            "cluster_centers": cluster_centers
        }
        log.info(f"model_desc: {model_desc}")
        with open(self.model_describe_file, 'w') as f:
            json.dump(model_desc, f, indent=4)

    def compute(self, usecols_file):
        log.info("extract feature or label.")
        train_x = self._read_data(usecols_file)
        feature_num = train_x.shape[1]
        feature_name = list(train_x.columns)
        train_x = train_x.values

        log.info("train start.")
        train_start_time = time.time()
        model = KMeans(n_clusters=self.n_clusters, 
                       init=self.init_method,
                       n_init=self.n_init, 
                       max_iter=self.max_iter, 
                       tol=self.tol, 
                       random_state=self.random_seed)
        model.fit(train_x)
        log.info(f"model save to: {self.output_file}")
        joblib.dump(model, self.output_file)
        log.info(f"save model describe")
        self.save_model_describe(feature_num, feature_name, model)
        train_use_time = round(time.time()-train_start_time, 3)
        log.info(f"save model success. train_use_time={train_use_time}s")
        evaluate_result = evaluate_score(train_x, model)
        log.info(f"evaluate_result = {evaluate_result}")
        return evaluate_result
    
    def _get_output_dir(self):
        output_dir = os.path.join(self.results_dir, self.model_dir_name)
        self.mkdir(output_dir)
        return output_dir

    
def evaluate_score(X, model):
    '''
    score = (distanceMeanOut - distanceMeanIn) / max(distanceMeanOut, distanceMeanIn)
    so that -1 <= score <= 1. when score -> 1 is good, score -> -1 is bad.
    '''
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X, model.labels_, metric='euclidean')
    evaluate_result = {
        "silhouette_score": score
    }
    evaluate_result = json.dumps(evaluate_result)
    log.info("evaluate success.")
    return evaluate_result
    


@ErrorTraceback("non-privacy_kmeans_train")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    kmeans = KmeansTrain(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type, extra = kmeans.train()
    return result_path, result_type, extra
