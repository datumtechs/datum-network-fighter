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
import codecs
import shutil
import tensorflow as tf
from functools import wraps


np.set_printoptions(suppress=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    

class LRPredict(BaseAlgorithm):
    '''
    Plaintext logistic regression predict.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_file = os.path.join(self.results_dir, "result_predict.csv")       
    
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
                "model_restore_party": "model1",
                "data_flow_restrict": {
                    "data1": ["compute1"],
                    "model1": ["compute1"],
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
                elif input_type == 2:
                    self.model_path = data["data_path"]
                else:
                    raise Exception(f"paramter error. input_type only support 1/2, not {input_type}")
        
        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]
        self.model_restore_party = dynamic_parameter["model_restore_party"]
        self.data_flow_restrict = dynamic_parameter["data_flow_restrict"]
        self.data_party.remove(self.model_restore_party)  # except restore party

    def check_parameters(self):
        log.info(f"check parameter start.")
        self._check_input_data()
        self.check_params_type(model_restore_party=(self.model_restore_party, str),
                               data_flow_restrict=(self.data_flow_restrict, dict))
        if self.party_id == self.model_restore_party:
            assert os.path.exists(self.model_path), f"model_path is not exists. model_path={self.model_path}"
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
                        
        
    def predict(self):
        '''
        Logistic regression predict algorithm implementation function
        '''
        log.info("start data party extract data column.")
        usecols_file = self._extract_data_column()
        log.info("start data party send data to compute party.")
        self._send_data_to_compute_party(usecols_file)
        temp_model_path = os.path.join(self.temp_dir, 'model')
        log.info("start model party send model to compute party.")
        self._send_model_to_compute_party(temp_model_path)
        if self.party_id in self.compute_party:
            log.info("compute party start  compute.")
            self.compute(usecols_file, temp_model_path)
        log.info("start compute party send data to result party.")
        self._send_data_to_result_party(self.output_file)
        result_path, result_type = '', ''
        if self.party_id in self.result_party:
            result_path = self.output_file
            result_type = 'csv'
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("predict success all.")
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
    
    def _send_model_to_compute_party(self, data_path):
        if self.party_id == self.model_restore_party:
            temp_model_dir = data_path
            data_path = shutil.make_archive(base_name=temp_model_dir, format='zip', root_dir=self.model_path)
            compute_party = self.data_flow_restrict[self.party_id][0]
            self.io_channel.send_data_to_other_party(compute_party, data_path)
        elif self.party_id in self.compute_party:
            party = self.model_restore_party
            if self.party_id == self.data_flow_restrict[party][0]:
                temp_model_dir = data_path + '.zip'
                self.io_channel.recv_data_from_other_party(party, temp_model_dir)
                shutil.unpack_archive(temp_model_dir, data_path)
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
        x_data = pd.read_csv(usecols_file)
        return x_data.values
    
    def load_model_desc(self, model_path):
        model_desc_file = os.path.join(model_path, 'describe.json')
        assert os.path.exists(model_desc_file), f"model_desc_file is not exist. model_desc_file={model_desc_file}"
        with open(model_desc_file, 'r') as f:
            model_desc = json.load(f)
        log.info(f"model_desc: {model_desc}")
        self.model_file_prefix = model_desc["model_file_prefix"]
        self.train_feature_num = model_desc["feature_num"]
        self.class_num = model_desc["class_num"]
        self.activation = model_desc["activation"]
        self.use_intercept = model_desc["use_intercept"]
        self.check_params_type(model_file_prefix=(self.model_file_prefix, str),
                               train_feature_num=(self.train_feature_num, int),
                               class_num=(self.class_num, int),
                               activation=(self.activation, str),
                               use_intercept=(self.use_intercept, bool))
        assert self.train_feature_num >=1, f"train_feature_num must be greater or equal to 1, not {self.train_feature_num}"
        assert self.activation in ["sigmoid", "softmax"], f"activation support sigmoid,softmax. not {self.activation}"
        if self.activation == "sigmoid":
            assert self.class_num == 2, f"when activation=sigmoid, class_num must be equal to 2, not {self.class_num}"
            self.class_num = 1
            self.predict_threshold = model_desc["predict_threshold"]
            assert isinstance(self.predict_threshold, float), f"predict_threshold must be type(float), not {type(self.predict_threshold)}"
            assert 0 <= self.predict_threshold <= 1, f"predict threshold must be between [0,1], not {self.predict_threshold}"
        else:
            assert self.class_num >= 2, f"when activation=softmax, class_num must be greater or equal to 2, not {self.class_num}"

    def compute(self, usecols_file, model_path):
        log.info("load model desc.")
        self.load_model_desc(model_path)
        log.info("extract feature or label.")
        x_data = self._read_data(usecols_file)
        feature_num = x_data.shape[1]
        assert feature_num == self.train_feature_num, \
            f"the total number of features used in prediction is not the same as that used in train, {feature_num} != {self.train_feature_num}"

        log.info("start build the model structure.")
        X = tf.placeholder(tf.float64, [None, feature_num], name='X')
        Y = tf.placeholder(tf.float64, [None, self.class_num], name='Y')
        W = tf.Variable(tf.zeros([feature_num, self.class_num], dtype=tf.float64), name='W')
        logits = tf.matmul(X, W)
        if self.use_intercept:
            b = tf.Variable(tf.zeros([self.class_num], dtype=tf.float64), name='b')
            logits = logits + b
        if self.activation == 'sigmoid':
            pred_Y = tf.sigmoid(logits)
        else:
            pred_Y = tf.nn.softmax(logits)
        saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
        init = tf.global_variables_initializer()
        log.info("finish build the model structure.")

        with tf.Session() as sess:
            log.info("session init.")
            sess.run(init)
            log.info("start restore model.")
            if os.path.exists(os.path.join(model_path, "checkpoint")):
                model_file = os.path.join(model_path, self.model_file_prefix)
                log.info(f"model restore from: {model_file}")
                saver.restore(sess, model_file)
            else:
                raise Exception("model not found or model damaged")
            log.info("predict start.")
            predict_start_time = time.time()
            Y_pred = sess.run(pred_Y, feed_dict={X: x_data})
            predict_use_time = round(time.time()-predict_start_time, 3)
            log.info(f"predict success. predict_use_time={predict_use_time}s")
            if self.activation == 'sigmoid':
                Y_prob = pd.DataFrame(Y_pred, columns=["Y_pred_prob"])
                Y_class = (Y_pred > self.predict_threshold) * 1
                Y_class = pd.DataFrame(Y_class, columns=[f"Y_pred_class(>{self.predict_threshold})"])
                Y_result = pd.concat([Y_prob, Y_class], axis=1)
            else:
                Y_class = np.argmax(Y_pred, axis=1)
                Y_result = pd.DataFrame(Y_class, columns=["Y_pred_class"])
            Y_result.to_csv(self.output_file, header=True, index=False, float_format = '%.6f')

    

@ErrorTraceback("non-privacy_lr_predict")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    lr = LRPredict(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type = lr.predict()
    return result_path, result_type
