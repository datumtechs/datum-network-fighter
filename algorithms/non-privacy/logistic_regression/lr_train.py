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
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
    

class LRTrain(BaseAlgorithm):
    '''
    Plaintext logistic regression train.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = self._get_output_dir()
        self.output_file = os.path.join(self.output_dir, "model")
    
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
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.1,
                    "use_validation_set": true,
                    "validation_set_rate": 0.2,
                    "predict_threshold": 0.5
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
        self.label_owner = dynamic_parameter["label_owner"]
        self.label_column = dynamic_parameter["label_column"]
        if self.party_id == self.label_owner:
            self.data_with_label = True
        else:
            self.data_with_label = False                  
        hyperparams = dynamic_parameter["hyperparams"]
        self.epochs = hyperparams.get("epochs", 10)
        self.batch_size = hyperparams.get("batch_size", 256)
        self.learning_rate = hyperparams.get("learning_rate", 0.001)
        self.use_validation_set = hyperparams.get("use_validation_set", True)
        self.validation_set_rate = hyperparams.get("validation_set_rate", 0.2)
        if not self.use_validation_set:
            self.validation_set_rate = 0
        self.predict_threshold = hyperparams.get("predict_threshold", 0.5)
        self.data_flow_restrict = dynamic_parameter["data_flow_restrict"]

    def check_parameters(self):
        log.info(f"check parameter start.")
        self._check_input_data()     
        self.check_params_type(epochs=(self.epochs, int),
                               batch_size=(self.batch_size, int),
                               learning_rate=(self.learning_rate, float),
                               use_validation_set=(self.use_validation_set, bool),
                               validation_set_rate=(self.validation_set_rate, float),
                               predict_threshold=(self.predict_threshold, float),
                               data_flow_restrict=(self.data_flow_restrict, dict))
        assert self.epochs > 0, f"epochs must be greater 0, not {self.epochs}"
        assert self.batch_size > 0, f"batch_size must be greater 0, not {self.batch_size}"
        assert self.learning_rate > 0, f"learning rate must be greater 0, not {self.learning_rate}"
        assert 0 < self.validation_set_rate < 1, f"validattion_set_rate must be between (0,1), not {self.validation_set_rate}"
        assert 0 <= self.predict_threshold <= 1, f"predict threshold must be between [0,1], not {self.predict_threshold}"
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
                if self.data_with_label:
                    assert self.label_column in input_columns, f"label_column:{self.label_column} not in input_file"
                    assert self.label_column not in self.selected_columns, f"label_column:{self.label_column} can not in selected_columns"
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
                temp_model_dir = os.path.join(self.temp_dir, 'model')
                data_path = shutil.make_archive(base_name=temp_model_dir, format='zip', root_dir=data_path)
            result_party = self.data_flow_restrict[self.party_id][0]
            self.io_channel.send_data_to_other_party(result_party, data_path)
            self.io_channel.send_sth(result_party, evaluate_result)
        elif self.party_id in self.result_party:
            for party in self.compute_party:
                if self.party_id == self.data_flow_restrict[party][0]:
                    temp_model_dir = os.path.join(self.temp_dir, 'model.zip')
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
            if self.data_with_label:
                use_cols += [self.label_column]
            log.info("read input file and write to new file.")
            usecols_data = pd.read_csv(self.input_file, usecols=use_cols, dtype="str")
            assert usecols_data.shape[0] > 0, 'no data after select columns.'
            usecols_data = usecols_data[use_cols]
            usecols_data.to_csv(usecols_file, header=True, index=False)
        return usecols_file

    def _read_and_split_data(self, usecols_file):
        '''
        Extract feature columns or label column from input file,
        and then divide them into train set and validation set.
        '''
        input_data = pd.read_csv(usecols_file)
        y_data = input_data[self.label_column]
        del input_data[self.label_column]
        x_data = input_data
        train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, stratify=y_data, test_size=self.validation_set_rate)
        train_x, val_x, train_y, val_y = train_x.values, train_y.values.reshape(-1, 1), val_x.values, val_y.values
        return train_x, val_x, train_y, val_y
    
    def compute(self, usecols_file):
        log.info("extract feature or label.")
        train_x, train_y, val_x, val_y = self._read_and_split_data(usecols_file)
        column_total_num = train_x.shape[1]

        log.info("start build the model structure.")
        X = tf.placeholder(tf.float64, [None, column_total_num])
        Y = tf.placeholder(tf.float64, [None, 1])
        W = tf.Variable(tf.zeros([column_total_num, 1], dtype=tf.float64))
        b = tf.Variable(tf.zeros([1], dtype=tf.float64))
        logits = tf.matmul(X, W) + b
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
        loss = tf.reduce_mean(loss)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
        
        pred_Y = tf.sigmoid(logits)
        log.info("finish build the model structure.")

        with tf.Session() as sess:
            log.info("session init.")
            sess.run(init)
            # train
            log.info("train start.")
            train_start_time = time.time()
            batch_num = math.ceil(len(train_x) / self.batch_size)
            for e in range(self.epochs):
                for i in range(batch_num):
                    bX = train_x[(i * self.batch_size): (i + 1) * self.batch_size]
                    bY = train_y[(i * self.batch_size): (i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={X: bX, Y: bY})
                    if (i % 50 == 0) or (i == batch_num - 1):
                        log.info(f"epoch:{e + 1}/{self.epochs}, batch:{i + 1}/{batch_num}")
            log.info(f"model save to: {self.output_file}")
            saver.save(sess, self.output_file)
            train_use_time = round(time.time()-train_start_time, 3)
            log.info(f"save model success. train_use_time={train_use_time}s")
        
            if self.use_validation_set:
                pred_y = sess.run(pred_Y, feed_dict={X: val_x})
                evaluate_result = self.evaluate(val_y, pred_y)
            else:
                evaluate_result = ""
        return evaluate_result
    
    def evaluate(self, Y_true, Y_pred):
        '''
        only support binary class
        '''
        log.info("start model evaluation.")
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
        log.info("start evaluate auc score.")
        auc_score = roc_auc_score(Y_true, Y_pred)
        Y_pred_class = (Y_pred > self.predict_threshold).astype('int64')  # default threshold=0.5
        log.info("start evaluate accuracy score.")
        accuracy = accuracy_score(Y_true, Y_pred_class)
        log.info("start evaluate f1_score.")
        f1_score = f1_score(Y_true, Y_pred_class)
        log.info("start evaluate precision score.")
        precision = precision_score(Y_true, Y_pred_class)
        log.info("start evaluate recall score.")
        recall = recall_score(Y_true, Y_pred_class)
        auc_score = round(auc_score, 6)
        accuracy = round(accuracy, 6)
        f1_score = round(f1_score, 6)
        precision = round(precision, 6)
        recall = round(recall, 6)
        evaluate_result = {
            "AUC": auc_score,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall
        }
        log.info(f"evaluate_result = {evaluate_result}")
        evaluate_result = json.dumps(evaluate_result)
        log.info("evaluate success.")
        return evaluate_result
    
    def _get_output_dir(self):
        output_dir = os.path.join(self.results_dir, 'model')
        self.mkdir(output_dir)
        return output_dir


@ErrorTraceback("non-privacy_lr_train")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    lr = LRTrain(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type, extra = lr.train()
    return result_path, result_type, extra
