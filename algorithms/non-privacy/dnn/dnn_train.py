# coding:utf-8

import os
import sys
import math
import json
import time
import copy
import logging
import shutil
import random
import traceback
import numpy as np
import pandas as pd
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

class DnnTrain(BaseAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir_name = "model"
        self.model_file_prefix = "model"
        self.output_dir = self._get_output_dir()
        self.output_file = os.path.join(self.output_dir, self.model_file_prefix)
        self.model_describe_file = os.path.join(self.output_dir, "describe.json")
        self.set_random_seed(self.random_seed)
        self.output_layer_activation = self.layer_activation[-1]
        self.task_type = self.get_task_type()
    
    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

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
                "label_owner": "p1",
                "label_column": "Y",
                "hyperparams": {
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.1,
                    "layer_units": [32, 1],     # hidden layer and output layer units
                    "layer_activation": ["sigmoid", "sigmoid"],   # hidden layer and output layer activation
                    "init_method": "xavier_uniform",   # weight and bias init method
                    "use_intercept": true,     # whether use bias
                    "optimizer": "sgd",
                    "dropout_prob": 0,
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
                    raise Exception("paramter error. input_type only support 1, not {input_type}")
        
        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]
        self.label_owner = dynamic_parameter["label_owner"]
        self.label_column = dynamic_parameter["label_column"]
        if self.party_id == self.label_owner:
            self.data_with_label = True
        else:
            self.data_with_label = False              
        hyperparams = dynamic_parameter["hyperparams"]
        self.epochs = hyperparams.get("epochs", 50)
        self.batch_size = hyperparams.get("batch_size", 256)
        self.learning_rate = hyperparams.get("learning_rate", 0.1)
        self.layer_units = hyperparams.get("layer_units", [32, 2])
        self.layer_activation = hyperparams.get("layer_activation", ["sigmoid", "softmax"])
        self.init_method = hyperparams.get("init_method", "xavier_uniform")
        self.use_intercept = hyperparams.get("use_intercept", True)
        self.optimizer = hyperparams.get("optimizer", "Adam")
        self.dropout_prob = hyperparams.get("dropout_prob", 0.0)
        self.use_validation_set = hyperparams.get("use_validation_set", True)
        self.validation_set_rate = hyperparams.get("validation_set_rate", 0.2)
        self.predict_threshold = hyperparams.get("predict_threshold", 0.5)
        self.random_seed = hyperparams.get("random_seed", None)
        self.data_flow_restrict = dynamic_parameter["data_flow_restrict"]

    def check_parameters(self):
        log.info(f"check parameter start.")
        self._check_input_data()
        self.check_params_type(epochs=(self.epochs, int),
                               batch_size=(self.batch_size, int),
                               learning_rate=(self.learning_rate, float),
                               layer_units=(self.layer_units, list),
                               layer_activation=(self.layer_activation, list),
                               init_method=(self.init_method, str),
                               use_intercept=(self.use_intercept, bool),
                               optimizer=(self.optimizer, str),
                               dropout_prob=(self.dropout_prob, float),
                               use_validation_set=(self.use_validation_set, bool),
                               validation_set_rate=(self.validation_set_rate, float),
                               predict_threshold=(self.predict_threshold, float),
                               random_seed=(self.random_seed, (int, type(None))),
                               data_flow_restrict=(self.data_flow_restrict, dict))
        assert self.epochs > 0, f"epochs must be greater 0, not {self.epochs}"
        assert self.batch_size > 0, f"batch_size must be greater 0, not {self.batch_size}"
        assert self.learning_rate > 0, f"learning rate must be greater 0, not {self.learning_rate}"
        if self.use_validation_set:
            assert 0 < self.validation_set_rate < 1, f"validattion_set_rate must be between (0,1), not {self.validation_set_rate}"
        if self.random_seed:
            assert 0 <= self.random_seed <= 2**32 - 1, f"random_seed must be between [0,2^32-1], not {self.random_seed}"
        assert 0 <= self.predict_threshold <= 1, f"predict threshold must be between [0,1], not {self.predict_threshold}"
        assert 0 <= self.dropout_prob <= 1, f"dropout_prob must be between [0,1], not {self.dropout_prob}"
        assert self.layer_units, f"layer_units must not empty, not {self.layer_units}"
        assert self.layer_activation, f"layer_activation must not empty, not {self.layer_activation}"
        assert len(self.layer_units) == len(self.layer_activation), \
                f"the length of layer_units:{len(self.layer_units)} and layer_activation:{len(self.layer_activation)} not same"
        for i in self.layer_units:
            assert isinstance(i, int) and i > 0, f"layer_units'element can only be type(int) and greater 0, not {i}"
        for i in self.layer_activation:
            if i not in ["", "sigmoid", "relu", "tanh", "softmax"]:
                raise Exception(f'layer_activation can only be "",sigmoid,relu,tanh,softmax. not {i}')
        if self.layer_activation[-1] != 'softmax':
            # if not multi-class classification, output layer units must be 1
            if self.layer_units[-1] != 1:
                raise Exception(f"when the last layer activation is {self.layer_activation[-1]}, the last layer units must be 1, not {self.layer_units[-1]}")
        if self.init_method == 'random_uniform':
            self.init_method = tf.random_uniform
        elif self.init_method == 'random_normal':
            self.init_method = tf.random_normal
        elif self.init_method == 'truncated_normal':
            self.init_method = tf.truncated_normal
        elif self.init_method == 'zeros':
            self.init_method = tf.zeros
        elif self.init_method == 'ones':
            self.init_method = tf.ones
        elif self.init_method == 'xavier_uniform':
            self.init_method = tf.contrib.layers.xavier_initializer()
        elif self.init_method == 'xavier_normal':
            self.init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        else:
            raise Exception(f"init_method support random_uniform,random_normal,truncated_normal,zeros,ones,xavier_uniform,xavier_normal. not {self.init_method}")
        if self.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer
        elif self.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.optimizer == 'RMSProp':
            self.optimizer = tf.train.RMSPropOptimizer
        elif self.optimizer == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer
        elif self.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer
        elif self.optimizer == 'Ftrl':
            self.optimizer = tf.train.FtrlOptimizer
        else:
            raise Exception(f"optimizer support SGD,Adam,RMSprop,Momentum,Adadelta,Adagrad,Ftrl. not {self.optimizer}")
        log.info(f"check parameter finish.")

    def get_layer_activation(self, logits, activation):
        if not activation:
            one_layer = logits
        elif activation == 'sigmoid':
            one_layer = tf.sigmoid(logits)
        elif activation == 'relu':
            one_layer = tf.nn.relu(logits)
        elif activation == 'tanh':
            one_layer = tf.tanh(logits)
        elif activation == 'softmax':
            one_layer = tf.nn.softmax(logits)
        else:
            raise Exception(f'not support {activation} activation.')
        return one_layer
    
    def get_task_type(self):
        '''
        have 3 task type:
            0: binary classification
            1: multi-class classification
            2: regression
        '''
        if self.output_layer_activation == 'sigmoid':
            task_type = 0
        elif self.output_layer_activation == 'softmax':
            task_type = 1
        else:
            task_type = 2
        return task_type
    
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
        training algorithm implementation function
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
            if self.data_with_label:
                use_cols += [self.label_column]
            log.info("read input file and write to new file.")
            usecols_data = pd.read_csv(self.input_file, usecols=use_cols, dtype="str")
            assert usecols_data.shape[0] > 0, 'no data after select columns.'
            if self.data_with_label:
                y_data = usecols_data[self.label_column]
                if self.layer_activation[-1] == 'sigmoid':
                    class_num = y_data.unique().shape[0]
                    assert class_num == 2, f"label column has {class_num} class, but the last layer of the network is {self.layer_activation[-1]}."
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
        if self.use_validation_set:
            train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, test_size=self.validation_set_rate, random_state=self.random_seed)
        else:
            # val_x, val_y is invalid.
            train_x, val_x, train_y, val_y = x_data, x_data, y_data, y_data
        return train_x, val_x, train_y, val_y
    
    def save_model_describe(self, feature_num, feature_name, label_name, evaluate_result):
        '''save model description for prediction'''
        model_desc = {
            "model_file_prefix": self.model_file_prefix,
            "feature_num": feature_num,
            "layer_units": self.layer_units,
            "layer_activation": self.layer_activation,
            "task_type": self.task_type,
            "use_intercept": self.use_intercept,
            "feature_name": feature_name, 
            "label_name": label_name,
            "evaluate_result": evaluate_result
        }
        if self.task_type == 0:
            model_desc["predict_threshold"] = self.predict_threshold
        log.info(f"model_desc: {model_desc}")
        with open(self.model_describe_file, 'w') as f:
            json.dump(model_desc, f, indent=4)
    
    def compute(self, usecols_file):
        log.info("extract feature or label.")
        train_x, val_x, train_y, val_y = self._read_and_split_data(usecols_file)
        feature_num = train_x.shape[1]
        feature_name = list(train_x.columns)
        label_name = train_y.name
        train_x, val_x, train_y, val_y = train_x.values, val_x.values, train_y.values, val_y.values
        if self.task_type == 0:
            class_num = np.unique(train_y).shape[0]
            assert class_num <= 2, f"the number of class in train set, as {class_num}, is greater than 2, please let the last layer activation to softmax, not sigmoid."
            assert class_num == 2, f"when the last layer activation is sigmoid, the number of class in train set must be 2, not {class_num}"
            train_y = train_y.reshape(-1, 1)
            val_y = val_y.reshape(-1, 1)
        elif self.task_type == 1:
            class_num = np.unique(train_y).shape[0]
            assert class_num == self.layer_units[-1], f"{class_num} != {self.layer_units[-1]}. when the last layer activation is softmax, the number of class in train set must be equal to the output layer units"
            train_y = np.eye(class_num)[train_y]
            val_y = np.eye(class_num)[val_y]
        else:
            train_y = train_y.reshape(-1, 1)
            val_y = val_y.reshape(-1, 1)

        log.info("start build the model structure.")
        X = tf.placeholder(tf.float64, [None, feature_num], name='X')
        Y = tf.placeholder(tf.float64, [None, self.layer_units[-1]], name='Y')
        output = self.dnn(X, feature_num)
        with tf.name_scope('output'):
            pred_Y = self.get_layer_activation(output, self.output_layer_activation)
        with tf.name_scope('loss'):
            if self.task_type == 0:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)
                loss = tf.reduce_mean(loss)
            elif self.task_type == 1:
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output)
                loss = tf.reduce_mean(loss)
            else:
                loss = tf.square(Y - pred_Y)
                loss = tf.reduce_mean(loss)
        # optimizer
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=None, max_to_keep=5, name='saver')
        log.info("finish build the model structure.")

        with tf.Session() as sess:
            log.info("session init.")
            sess.run(init)
            # summary_writer = tf.summary.FileWriter(self.get_temp_dir(), sess.graph)
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
                evaluate = Evaluate(val_y, pred_y)
                if self.task_type == 0:
                    evaluate_result = evaluate.binary_classify(self.predict_threshold)
                elif self.task_type == 1:
                    evaluate_result = evaluate.multiclass_classify()
                else:
                    evaluate_result = evaluate.regression()
            else:
                evaluate_result = ""
        log.info(f"evaluate_result = {evaluate_result}")
        self.save_model_describe(feature_num, feature_name, label_name, evaluate_result)
        evaluate_result = json.dumps(evaluate_result)
        return evaluate_result
    
    def layer(self, input_tensor, input_dim, output_dim, activation, layer_name='Dense'):
        with tf.name_scope(layer_name):
            W = tf.Variable(self.init_method([input_dim, output_dim], dtype=tf.float64), name='W')
            if self.use_intercept:
                b = tf.Variable(self.init_method([output_dim], dtype=tf.float64), name='b')
                with tf.name_scope('logits'):
                    logits = tf.matmul(input_tensor, W) + b
            else:
                with tf.name_scope('logits'):
                    logits = tf.matmul(input_tensor, W)
            logits = tf.nn.dropout(logits, 1 - self.dropout_prob)
            one_layer = self.get_layer_activation(logits, activation)
            return one_layer
    
    def dnn(self, input_X, input_dim):
        layer_activation = copy.deepcopy(self.layer_activation[:-1])
        layer_activation.append("")
        for i in range(len(self.layer_units)):
            if i == 0:
                input_units = input_dim
                previous_output = input_X
            else:
                input_units = self.layer_units[i-1]
                previous_output = output
            output = self.layer(previous_output, 
                                input_units, 
                                self.layer_units[i], 
                                layer_activation[i], 
                                layer_name=f"Dense_{i}")
        return output
    
    def show_train_history(self, train_history, val_history, epochs, name='loss'):
        log.info("start show_train_history")
        assert all([isinstance(ele, float) for ele in train_history]), 'element of train_history must be float.'
        import matplotlib.pyplot as plt
        plt.figure()
        y_min = min(train_history)
        y_max = max(train_history)
        y_ticks = np.linspace(y_min, y_max, 10)
        plt.scatter(list(range(1, epochs+1)), train_history, label='train')
        if self.use_validation_set:
            plt.scatter(list(range(1, epochs+1)), val_history, label='val')
        plt.xlabel('epochs') 
        plt.ylabel(name)
        plt.yticks(y_ticks)
        plt.title(f'{name} with epochs')
        plt.legend()
        figure_path = os.path.join(self.results_dir, f'{name}.jpg')
        plt.savefig(figure_path)
    
    def _get_output_dir(self):
        output_dir = os.path.join(self.results_dir, self.model_dir_name)
        self.mkdir(output_dir)
        return output_dir

class BaseEvaluate():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def binary_classify(self, *args, **kwargs):
        '''binary class classification'''
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} fuction is not implemented.')
    
    def multiclass_classify(self, *args, **kwargs):
        '''multi-class classification'''
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} fuction is not implemented.')
    
    def regression(self, *args, **kwargs):
        '''regression evaluation'''
        raise NotImplementedError(f'{sys._getframe().f_code.co_name} fuction is not implemented.')

class Evaluate(BaseEvaluate):    
    def binary_classify(self, threshold):
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
        log.info("start evaluate auc score.")
        y_true = self.y_true.reshape(-1,)
        y_pred = self.y_pred.reshape(-1,)
        auc_score = roc_auc_score(y_true, y_pred)
        y_pred_class = (y_pred > threshold).astype('int64')
        log.info("start evaluate accuracy score.")
        accuracy = accuracy_score(y_true, y_pred_class)
        log.info("start evaluate f1_score.")
        f1_score = f1_score(y_true, y_pred_class)
        log.info("start evaluate precision score.")
        precision = precision_score(y_true, y_pred_class)
        log.info("start evaluate recall score.")
        recall = recall_score(y_true, y_pred_class)
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
        log.info("evaluate success.")
        return evaluate_result
    
    def multiclass_classify(self):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        y_true_index = np.argmax(self.y_true, axis=1)
        y_pred_index = np.argmax(self.y_pred, axis=1)
        accuracy = accuracy_score(y_true_index, y_pred_index)
        f1_score_micro = f1_score(y_true_index, y_pred_index, average='micro')
        precision_micro = precision_score(y_true_index, y_pred_index, average='micro')
        recall_micro = recall_score(y_true_index, y_pred_index, average='micro')
        f1_score_macro = f1_score(y_true_index, y_pred_index, average='macro')
        precision_macro = precision_score(y_true_index, y_pred_index, average='macro')
        recall_macro = recall_score(y_true_index, y_pred_index, average='macro')
        accuracy = round(accuracy, 6)
        f1_score_micro = round(f1_score_micro, 6)
        precision_micro = round(precision_micro, 6)
        recall_micro = round(recall_micro, 6)
        f1_score_macro = round(f1_score_macro, 6)
        precision_macro = round(precision_macro, 6)
        recall_macro = round(recall_macro, 6)
        evaluate_result = {
            "accuracy": accuracy,
            "f1_score_micro": f1_score_micro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_score_macro": f1_score_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro
        }
        log.info("evaluate success.")
        return evaluate_result
    
    def regression(self):
        log.info('start regression evaluate.')
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        y_true = self.y_true.reshape(-1,)
        y_pred = self.y_pred.reshape(-1,)
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred, squared=True)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = round(r2, 6)
        rmse = round(rmse, 6)
        mse = round(mse, 6)
        mae = round(mae, 6)
        evaluate_result = {
            "R2-score": r2,
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae
        }
        log.info("evaluate success.")
        return evaluate_result


@ErrorTraceback("non-privacy_dnn_train")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    privacy_dnn = DnnTrain(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type, extra = privacy_dnn.train()
    return result_path, result_type, extra
