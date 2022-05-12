# coding:utf-8

import os
import sys
import math
import json
import time
import copy
import logging
import shutil
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
                "label_owner": "p1",
                "label_column": "Y",
                "hyperparams": {
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.1,
                    "layer_units": [32, 1],     # hidden layer and output layer units
                    "layer_activation": ["sigmoid", "sigmoid"],   # hidden layer and output layer activation
                    "init_method": "random_normal",   # weight and bias init method
                    "use_intercept": true,     # whether use bias
                    "optimizer": "sgd",
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
        self.layer_units = hyperparams.get("layer_units", [32, 1])
        self.layer_activation = hyperparams.get("layer_activation", ["sigmoid", "sigmoid"])
        self.init_method = hyperparams.get("init_method", "random_normal")  # 'random_normal', 'random_uniform', 'zeros', 'ones'
        self.use_intercept = hyperparams.get("use_intercept", True)  # True: use bias, False: not use bias
        self.optimizer = hyperparams.get("optimizer", "sgd")
        self.use_validation_set = hyperparams.get("use_validation_set", True)
        self.validation_set_rate = hyperparams.get("validation_set_rate", 0.2)
        self.predict_threshold = hyperparams.get("predict_threshold", 0.5)
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
                               use_validation_set=(self.use_validation_set, bool),
                               validation_set_rate=(self.validation_set_rate, float),
                               predict_threshold=(self.predict_threshold, float),
                               data_flow_restrict=(self.data_flow_restrict, dict))
        assert self.epochs > 0, f"epochs must be greater 0, not {self.epochs}"
        assert self.batch_size > 0, f"batch_size must be greater 0, not {self.batch_size}"
        assert self.learning_rate > 0, f"learning rate must be greater 0, not {self.learning_rate}"
        assert 0 < self.validation_set_rate < 1, f"validattion_set_rate must be between (0,1), not {self.validation_set_rate}"
        assert 0 <= self.predict_threshold <= 1, f"predict threshold must be between [0,1], not {self.predict_threshold}"
        assert self.layer_units, f"layer_units must not empty, not {self.layer_units}"
        assert self.layer_activation, f"layer_activation must not empty, not {self.layer_activation}"
        assert len(self.layer_units) == len(self.layer_activation), \
                f"the length of layer_units:{len(self.layer_units)} and layer_activation:{len(self.layer_activation)} not same"
        for i in self.layer_units:
            assert isinstance(i, int) and i > 0, f"layer_units'element can only be type(int) and greater 0, not {i}"
        for i in self.layer_activation:
            if i not in ["", "sigmoid", "relu", None]:
                raise Exception(f'layer_activation can only be ""/"sigmoid"/"relu"/None, not {i}')
        if self.layer_activation[-1] == 'sigmoid':
            if self.layer_units[-1] != 1:
                raise Exception(f"when output layer activation is sigmoid, output layer units must be 1, not {self.layer_units[-1]}")
        if self.init_method == 'random_normal':
            self.init_method = tf.random_normal
        elif self.init_method == 'random_uniform':
            self.init_method = tf.random_uniform
        elif self.init_method == 'zeros':  # if len(self.layer_units) != 1, init_method not use zeros, because it can not work well.
            self.init_method = tf.zeros
        elif self.init_method == 'ones':
            self.init_method = tf.ones
        else:
            raise Exception(f"init_method only can be random_normal/random_uniform/zeros/ones, not {self.init_method}")
        if self.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer
        else:
            raise Exception(f"optimizer only can be sgd, not {self.optimizer}")
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
        if self.layer_activation[-1] == 'sigmoid':
            train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, stratify=y_data, test_size=self.validation_set_rate)
        else:
            train_x, val_x, train_y, val_y = train_test_split(x_data, y_data, test_size=self.validation_set_rate)
        train_x, val_x, train_y, val_y = train_x.values, train_y.values.reshape(-1, 1), val_x.values, val_y.values
        return train_x, val_x, train_y, val_y
    
    def compute(self, usecols_file):
        log.info("extract feature or label.")
        train_x, train_y, val_x, val_y = self._read_and_split_data(usecols_file)
        column_total_num = train_x.shape[1]

        log.info("start build the model structure.")
        X = tf.placeholder(tf.float64, [None, column_total_num], name='X')
        Y = tf.placeholder(tf.float64, [None, self.layer_units[-1]], name='Y')
        output = self.dnn(X, column_total_num)
        output_layer_activation = self.layer_activation[-1]
        with tf.name_scope('output'):
            if not output_layer_activation:
                pred_Y = output
            elif output_layer_activation == 'sigmoid':
                pred_Y = tf.sigmoid(output)
            elif output_layer_activation == 'relu':
                pred_Y = tf.nn.relu(output)
            else:
                raise Exception('output layer not support {output_layer_activation} activation.')
        with tf.name_scope('loss'):
            if (not output_layer_activation) or (output_layer_activation == 'relu'):
                loss = tf.square(Y - pred_Y)
                loss = tf.reduce_mean(loss)
            elif output_layer_activation == 'sigmoid':
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output)
                loss = tf.reduce_mean(loss)
            else:
                raise Exception('output layer not support {output_layer_activation} activation.')
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
                evaluate_result = self.evaluate(val_y, pred_y, output_layer_activation)
            else:
                evaluate_result = ""
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
            if not activation:
                one_layer = logits
            elif activation == 'sigmoid':
                one_layer = tf.sigmoid(logits)
            elif activation == 'relu':
                one_layer = tf.nn.relu(logits)
            else:
                raise Exception(f'not support {activation} activation.')
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
    
    def evaluate(self, Y_true, Y_pred, output_layer_activation):
        if (not output_layer_activation) or (output_layer_activation == 'relu'):
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            r2 = r2_score(Y_true, Y_pred)
            rmse = mean_squared_error(Y_true, Y_pred, squared=False)
            mse = mean_squared_error(Y_true, Y_pred, squared=True)
            mae = mean_absolute_error(Y_true, Y_pred)
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
        elif output_layer_activation == 'sigmoid':
            from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
            auc_score = roc_auc_score(Y_true, Y_pred)
            Y_pred_class = (Y_pred > self.predict_threshold).astype('int64')  # default threshold=0.5
            accuracy = accuracy_score(Y_true, Y_pred_class)
            f1_score = f1_score(Y_true, Y_pred_class)
            precision = precision_score(Y_true, Y_pred_class)
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
        else:
            raise Exception('output layer not support {output_layer_activation} activation.')
        log.info(f"evaluate_result = {evaluate_result}")
        evaluate_result = json.dumps(evaluate_result)
        log.info("evaluate success.")
        return evaluate_result
    
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
        output_dir = os.path.join(self.results_dir, 'model')
        self.mkdir(output_dir)
        return output_dir


@ErrorTraceback("non-privacy_dnn_train")
def main(io_channel, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    privacy_dnn = DnnTrain(io_channel, cfg_dict, data_party, compute_party, result_party, results_dir)
    result_path, result_type, extra = privacy_dnn.train()
    return result_path, result_type, extra
