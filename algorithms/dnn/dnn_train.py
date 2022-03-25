# coding:utf-8

import os
import sys
import math
import json
import time
import copy
import logging
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import latticex.rosetta as rtt
import channel_sdk.pyio as io


np.set_printoptions(suppress=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rtt.set_backend_loglevel(3)  # All(0), Trace(1), Debug(2), Info(3), Warn(4), Error(5), Fatal(6)
logger = logging.getLogger(__name__)
class LogWithStage():
    def __init__(self):
        self.run_stage = 'init log.'
    
    def info(self, content):
        self.run_stage = content
        logger.info(content)
    
    def debug(self, content):
        logger.debug(content)

log = LogWithStage()

class PrivacyDnnTrain(object):
    '''
    Privacy DNN train base on rosetta.
    '''

    def __init__(self,
                 channel_config: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_dir: str):
        '''
        cfg_dict:
        {
            "party_id": "p1",
            "data_party": {
                "input_file": "path/to/file",
                "key_column": "col1",
                "selected_columns": ["col2", "col3"],
                "use_psi": true,
                "psi_result_file": "path/to/file"
            },
            "dynamic_parameter": {
                "label_owner": "p1",
                "label_column": "Y",
                "algorithm_parameter": {
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.1,
                    "layer_units": [32, 128, 32, 1],     # hidden layer and output layer units
                    "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],   # hidden layer and output layer activation
                    "init_method": "random_normal",   # weight and bias init method
                    "use_intercept": true,     # whether use bias
                    "optimizer": "sgd",
                    "use_validation_set": true,
                    "validation_set_rate": 0.2,
                    "predict_threshold": 0.5
                }
            }
        }
        '''
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
        self.results_dir = results_dir
        self.party_id = cfg_dict["party_id"]
        self.input_file = cfg_dict["data_party"].get("input_file")
        self.key_column = cfg_dict["data_party"].get("key_column")
        self.selected_columns = cfg_dict["data_party"].get("selected_columns")
        self.use_psi = cfg_dict["data_party"].get("use_psi", True)
        self.psi_result_file = cfg_dict["data_party"].get("psi_result_file")

        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.label_owner = dynamic_parameter.get("label_owner")
        if self.party_id == self.label_owner:
            self.label_column = dynamic_parameter.get("label_column")
            self.data_with_label = True
        else:
            self.label_column = ""
            self.data_with_label = False
                        
        algorithm_parameter = dynamic_parameter["algorithm_parameter"]
        self.epochs = algorithm_parameter.get("epochs", 50)
        self.batch_size = algorithm_parameter.get("batch_size", 256)
        self.learning_rate = algorithm_parameter.get("learning_rate", 0.1)
        self.layer_units = algorithm_parameter.get("layer_units", [32, 128, 32, 1])
        self.layer_activation = algorithm_parameter.get("layer_activation", ["sigmoid", "sigmoid", "sigmoid", "sigmoid"])
        self.init_method = algorithm_parameter.get("init_method", "random_normal")  # 'random_normal', 'random_uniform', 'zeros', 'ones'
        self.use_intercept = algorithm_parameter.get("use_intercept", True)  # True: use bias, False: not use bias
        self.optimizer = algorithm_parameter.get("optimizer", "sgd")
        self.use_validation_set = algorithm_parameter.get("use_validation_set", True)
        self.validation_set_rate = algorithm_parameter.get("validation_set_rate", 0.2)
        self.predict_threshold = algorithm_parameter.get("predict_threshold", 0.5)
        self.output_file = os.path.join(self.results_dir, "model")
        
        self.check_parameters()

    def check_parameters(self):
        log.info(f"check parameter start.")        
        assert isinstance(self.epochs, int) and self.epochs > 0, "epochs must be type(int) and greater 0"
        assert isinstance(self.batch_size, int) and self.batch_size > 0, "batch_size must be type(int) and greater 0"
        assert isinstance(self.learning_rate, float) and self.learning_rate > 0, "learning rate must be type(float) and greater 0"
        assert isinstance(self.layer_units, list) and self.layer_units, "layer_units must be type(list) and not empty"
        assert isinstance(self.layer_activation, list) and self.layer_activation, "layer_activation must be type(list) and not empty"
        assert len(self.layer_units) == len(self.layer_activation), "the length of layer_units and layer_activation must be the same"
        for i in self.layer_units:
            assert isinstance(i, int) and i > 0, f'layer_units can only be type(int) and greater 0'
        for i in self.layer_activation:
            if i not in ["", "sigmoid", "relu", None]:
                raise Exception(f'layer_activation can only be ""/"sigmoid"/"relu"/None, not {i}')
        if self.layer_activation[-1] == 'sigmoid':
            if self.layer_units[-1] != 1:
                raise Exception(f"when output layer activation is sigmoid, output layer units must be 1, not {self.layer_units[-1]}")
        
        assert isinstance(self.init_method, str), "init_method must be type(str)"
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
        assert isinstance(self.optimizer, str), "optimizer must be type(str)"
        if self.optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer
        else:
            raise Exception(f"optimizer only can be sgd, not {self.optimizer}")
        assert isinstance(self.use_intercept, bool), "use_intercept must be type(bool), true or false"
        assert isinstance(self.use_validation_set, bool), "use_validation_set must be type(bool), true or false"
        assert 0 < self.validation_set_rate < 1, "validattion set rate must be between (0,1)"
        assert 0 <= self.predict_threshold <= 1, "predict threshold must be between [0,1]"
        
        if self.party_id in self.data_party:
            assert isinstance(self.use_psi, bool), "use_psi must be type(bool), true or false"
            if self.use_psi:
                assert isinstance(self.psi_result_file, str), "psi_result_file must be type(string)" 
                self.psi_result_file = self.psi_result_file.strip()
                if os.path.exists(self.psi_result_file):
                    file_suffix = os.path.splitext(self.psi_result_file)[-1]
                    assert file_suffix == ".csv", f"psi_result_file must csv file, not {file_suffix}"
                else:
                    raise Exception(f"psi_result_file is not exist. psi_result_file={self.psi_result_file}")
            
            assert isinstance(self.input_file, str), "input_file must be type(string)"
            assert isinstance(self.key_column, str), "key_column must be type(string)"
            assert isinstance(self.selected_columns, list), "selected_columns must be type(list)" 
            self.input_file = self.input_file.strip()
            if os.path.exists(self.input_file):
                file_suffix = os.path.splitext(self.input_file)[-1]
                assert file_suffix == ".csv", f"input_file must csv file, not {file_suffix}"
                input_columns = pd.read_csv(self.input_file, nrows=0)
                input_columns = list(input_columns.columns)
                assert self.key_column in input_columns, f"key_column:{self.key_column} not in input_file"
                error_col = []
                for col in self.selected_columns:
                    if col not in input_columns:
                        error_col.append(col)   
                assert not error_col, f"selected_columns:{error_col} not in input_file"
                assert self.key_column not in self.selected_columns, f"key_column:{self.key_column} can not in selected_columns"
                if self.label_column:
                    assert self.label_column in input_columns, f"label_column:{self.label_column} not in input_file"
                    assert self.label_column not in self.selected_columns, f"label_column:{self.label_column} can not in selected_columns"
            else:
                raise Exception(f"input_file is not exist. input_file={self.input_file}")
        log.info(f"check parameter finish.")
                        
        
    def train(self):
        '''
        training algorithm implementation function
        '''

        log.info("extract feature or label.")
        train_x, train_y, val_x, val_y = self.extract_feature_or_label(with_label=self.data_with_label)
        
        log.info("start create and set channel.")
        self.create_set_channel()
        log.info("waiting other party connect...")
        rtt.activate("SecureNN")
        log.info("protocol has been activated.")
        
        log.info(f"start set save model. save to party: {self.result_party}")
        rtt.set_saver_model(False, plain_model=self.result_party)
        # sharing data
        log.info(f"start sharing train data. data_owner={self.data_party}, label_owner={self.label_owner}")
        shard_x, shard_y = rtt.PrivateDataset(data_owner=self.data_party, 
                                              label_owner=self.label_owner)\
                                .load_data(train_x, train_y, header=0)
        log.info("finish sharing train data.")
        column_total_num = shard_x.shape[1]
        log.info(f"column_total_num = {column_total_num}.")
        
        if self.use_validation_set:
            log.info("start sharing validation data.")
            shard_x_val, shard_y_val = rtt.PrivateDataset(data_owner=self.data_party, 
                                                          label_owner=self.label_owner)\
                                            .load_data(val_x, val_y, header=0)
            log.info("finish sharing validation data.")

        if self.party_id not in self.data_party:  
            # mean the compute party and result party
            log.info("compute start.")
            X = tf.placeholder(tf.float64, [None, column_total_num], name='X')
            Y = tf.placeholder(tf.float64, [None, self.layer_units[-1]], name='Y')
            val_Y = tf.placeholder(tf.float64, [None, self.layer_units[-1]], name='val_Y')
                        
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
                        
            reveal_loss = rtt.SecureReveal(loss) # only reveal to the result party
            reveal_Y = rtt.SecureReveal(pred_Y)  # only reveal to the result party
            reveal_val_Y = rtt.SecureReveal(val_Y) # only reveal to the result party

            with tf.Session() as sess:
                log.info("session init.")
                sess.run(init)
                summary_writer = tf.summary.FileWriter(self.get_temp_dir(), sess.graph)
                # train
                log.info("train start.")
                train_start_time = time.time()
                batch_num = math.ceil(len(shard_x) / self.batch_size)
                loss_history_train, loss_history_val = [], []
                for e in range(self.epochs):
                    for i in range(batch_num):
                        bX = shard_x[(i * self.batch_size): (i + 1) * self.batch_size]
                        bY = shard_y[(i * self.batch_size): (i + 1) * self.batch_size]
                        sess.run(optimizer, feed_dict={X: bX, Y: bY})
                        if (i % 50 == 0) or (i == batch_num - 1):
                            log.info(f"epoch:{e + 1}/{self.epochs}, batch:{i + 1}/{batch_num}")
                    train_loss = sess.run(reveal_loss, feed_dict={X: shard_x, Y: shard_y})
                    # collect loss
                    loss_history_train.append(float(train_loss))
                    if self.use_validation_set:
                        val_loss = sess.run(reveal_loss, feed_dict={X: shard_x_val, Y: shard_y_val})
                        loss_history_val.append(float(val_loss))
                log.info(f"model save to: {self.output_file}")
                saver.save(sess, self.output_file)
                train_use_time = round(time.time()-train_start_time, 3)
                log.info(f"save model success. train_use_time={train_use_time}s")
                if self.use_validation_set:
                    Y_pred = sess.run(reveal_Y, feed_dict={X: shard_x_val})
                    log.info(f"Y_pred:\n {Y_pred[:10]}")
                    Y_actual = sess.run(reveal_val_Y, feed_dict={val_Y: shard_y_val})
                    log.info(f"Y_actual:\n {Y_actual[:10]}")
        
            running_stats = str(rtt.get_perf_stats(True)).replace('\n', '').replace(' ', '')
            log.info(f"running stats: {running_stats}")
        else:
            log.info("computing, please waiting for compute finish...")
        rtt.deactivate()
     
        log.info("remove temp dir.")
        if self.party_id in (self.data_party + self.result_party):
            # self.remove_temp_dir()
            pass
        else:
            # delete the model in the compute party.
            self.remove_output_dir()
        
        if self.party_id in self.result_party:
            log.info(f"result_party evaluation the model.")
            if self.use_validation_set:
                log.info("result_party evaluate model.")
                Y_pred = Y_pred.astype("float").reshape([-1, ])
                Y_true = Y_actual.astype("float").reshape([-1, ])
                self.model_evaluation(Y_true, Y_pred, output_layer_activation)
            # self.show_train_history(loss_history_train, loss_history_val, self.epochs)
            # log.info(f"result_party show train history finish.")
        log.info("train success all.")
    
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
    
    def model_evaluation(self, Y_true, Y_pred, output_layer_activation):
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
            evaluation_result = {
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
            evaluation_result = {
                "AUC": auc_score,
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall
            }
        else:
            raise Exception('output layer not support {output_layer_activation} activation.')
        log.info(f"evaluation_result = {evaluation_result}")
        log.info("evaluation result write to file.")
        result_file = os.path.join(self.results_dir, "evaluation_result.json")
        with open(result_file, "w") as f:
            json.dump(evaluation_result, f, indent=4)
        log.info("evaluation success.")
    
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
        
    def create_set_channel(self):
        '''
        create and set channel.
        '''
        io_channel = io.APIManager()
        log.info("start create channel.")
        channel = io_channel.create_channel(self.party_id, self.channel_config)
        log.info("start set channel.")
        rtt.set_channel("", channel)
        log.info("set channel success.")
    
    def extract_feature_or_label(self, with_label: bool=False):
        '''
        Extract feature columns or label column from input file,
        and then divide them into train set and validation set.
        '''
        train_x = ""
        train_y = ""
        val_x = ""
        val_y = ""
        temp_dir = self.get_temp_dir()
        if self.party_id in self.data_party:
            usecols = [self.key_column] + self.selected_columns
            if with_label:
                usecols += [self.label_column]
            
            input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
            input_data = input_data[usecols]
            assert input_data.shape[0] > 0, 'input file is no data.'
            if self.use_psi:
                psi_result = pd.read_csv(self.psi_result_file, dtype="str")
                psi_result.name = self.key_column
                input_data = pd.merge(psi_result, input_data, on=self.key_column, how='inner')
                assert input_data.shape[0] > 0, 'input data is empty. because no intersection with psi result.'
            # only if self.validation_set_rate==0, split_point==input_data.shape[0]
            split_point = int(input_data.shape[0] * (1 - self.validation_set_rate))
            assert split_point > 0, f"train set is empty, because validation_set_rate:{self.validation_set_rate} is too big"
            
            if with_label:
                y_data = input_data[self.label_column]
                train_y_data = y_data.iloc[:split_point]
                train_class_num = train_y_data.unique().shape[0]
                if self.layer_activation[-1] == 'sigmoid':
                    assert train_class_num == 2, f"train set must be 2 class, not {train_class_num} class."
                train_y = os.path.join(temp_dir, f"train_y_{self.party_id}.csv")
                train_y_data.to_csv(train_y, header=True, index=False)
                if self.use_validation_set:
                    assert split_point < input_data.shape[0], \
                        f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small"
                    val_y_data = y_data.iloc[split_point:]
                    val_class_num = val_y_data.unique().shape[0]
                    if self.layer_activation[-1] == 'sigmoid':
                        assert val_class_num == 2, f"validation set must be 2 class, not {val_class_num} class."
                    val_y = os.path.join(temp_dir, f"val_y_{self.party_id}.csv")
                    val_y_data.to_csv(val_y, header=True, index=False)
            
            x_data = input_data[self.selected_columns]
            train_x = os.path.join(temp_dir, f"train_x_{self.party_id}.csv")
            x_data.iloc[:split_point].to_csv(train_x, header=True, index=False)
            if self.use_validation_set:
                assert split_point < input_data.shape[0], \
                        f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small."
                val_x = os.path.join(temp_dir, f"val_x_{self.party_id}.csv")
                x_data.iloc[split_point:].to_csv(val_x, header=True, index=False)

        return train_x, train_y, val_x, val_y
    
    def get_temp_dir(self):
        '''
        Get the directory for temporarily saving files
        '''
        temp_dir = os.path.join(self.results_dir, 'temp')
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
        path = self.results_dir
        if os.path.exists(path):
            shutil.rmtree(path)


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    log.info("start main function.")
    try:
        privacy_dnn = PrivacyDnnTrain(channel_config, cfg_dict, data_party, result_party, results_dir)
        privacy_dnn.train()
    except Exception as e:
        raise Exception(f"<RUN_STAGE>: {log.run_stage} <ERROR>: {str(e)}")
    log.info("finish main function.")
