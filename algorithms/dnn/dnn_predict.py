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
log = logging.getLogger(__name__)

class PrivacyDnnPredict(object):
    '''
    Privacy Dnn predict base on rosetta.
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
                "selected_columns": ["col2", "col3"]
            },
            "dynamic_parameter": {
                "model_restore_party": "p3",
                "model_path": "/absoulte_path/to/model_dir",
                "algorithm_parameter": {
                    "layer_units": [32, 128, 32, 1],
                    "layer_activation": ["sigmoid", "sigmoid", "sigmoid", "sigmoid"],
                    "use_intercept": true,
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
        self.party_id = cfg_dict["party_id"]
        self.input_file = cfg_dict["data_party"].get("input_file")
        self.key_column = cfg_dict["data_party"].get("key_column")
        self.selected_columns = cfg_dict["data_party"].get("selected_columns")
        self.use_psi = cfg_dict["data_party"].get("use_psi", True)
        self.psi_result_file = cfg_dict["data_party"].get("psi_result_file")

        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.model_restore_party = dynamic_parameter.get("model_restore_party")
        self.model_path = dynamic_parameter.get("model_path")
        self.model_file = os.path.join(self.model_path, "model")
        algorithm_parameter = dynamic_parameter["algorithm_parameter"]
        self.layer_units = algorithm_parameter.get("layer_units", [32, 128, 32, 1])
        self.layer_activation = algorithm_parameter.get("layer_activation", ["sigmoid", "sigmoid", "sigmoid", "sigmoid"])
        self.use_intercept = algorithm_parameter.get("use_intercept", True)  # True: use b, False: not use b        
        self.predict_threshold = algorithm_parameter.get("predict_threshold", 0.5)
        self.results_dir = results_dir
        self.output_file = os.path.join(self.results_dir, "result")
        self.data_party.remove(self.model_restore_party)  # except restore party
        self.check_parameters()

    def check_parameters(self):
        log.info(f"check parameter start.")
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
                raise Exception(f"output layer activation is sigmoid, output layer units must be 1, not {self.layer_units[-1]}")
        assert isinstance(self.use_intercept, bool), "use_intercept must be type(bool), true or false"     
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
            else:
                raise Exception(f"input_file is not exist. input_file={self.input_file}")
        if self.party_id == self.model_restore_party:
            assert os.path.exists(self.model_path), f"model path not found. model_path={self.model_path}"
        log.info(f"check parameter finish.")
       

    def predict(self):
        '''
        predict algorithm implementation function
        '''

        log.info("extract feature or id.")
        file_x, id_col = self.extract_feature_or_index()
        
        log.info("start create and set channel.")
        self.create_set_channel()
        log.info("waiting other party connect...")
        rtt.activate("SecureNN")
        log.info("protocol has been activated.")
        
        log.info(f"start set restore model. restore party={self.model_restore_party}")
        rtt.set_restore_model(False, plain_model=self.model_restore_party)
        # sharing data
        log.info(f"start sharing data. data_owner={self.data_party}")
        shard_x = rtt.PrivateDataset(data_owner=self.data_party).load_X(file_x, header=0)
        log.info("finish sharing data .")
        column_total_num = shard_x.shape[1]
        log.info(f"column_total_num = {column_total_num}.")

        if self.party_id not in self.data_party:  
            # mean the compute party and result party
            log.info("compute start.")
            X = tf.placeholder(tf.float64, [None, column_total_num], name='X')
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
            saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
            init = tf.global_variables_initializer()
            reveal_Y = rtt.SecureReveal(pred_Y)  # only reveal to result party

            with tf.Session() as sess:
                log.info("session init.")
                sess.run(init)
                log.info("start restore model.")
                if self.party_id == self.model_restore_party:
                    if os.path.exists(os.path.join(self.model_path, "checkpoint")):
                        log.info(f"model restore from: {self.model_file}.")
                        saver.restore(sess, self.model_file)
                    else:
                        raise Exception("model not found or model damaged")
                else:
                    log.info("restore model...")
                    temp_file = os.path.join(self.get_temp_dir(), 'ckpt_temp_file')
                    with open(temp_file, "w") as f:
                        pass
                    saver.restore(sess, temp_file)
                log.info("finish restore model.")
                
                # predict
                log.info("predict start.")
                predict_start_time = time.time()
                Y_pred = sess.run(reveal_Y, feed_dict={X: shard_x})
                log.debug(f"Y_pred:\n {Y_pred[:10]}")
                predict_use_time = round(time.time() - predict_start_time, 3)
                log.info(f"predict success. predict_use_time={predict_use_time}s")
        else:
            log.info("computing, please waiting for compute finish...")
        rtt.deactivate()
        log.info("rtt deactivate finish.")
        
        if self.party_id in self.result_party:
            log.info("predict result write to file.")
            output_file_predict_prob = os.path.splitext(self.output_file)[0] + "_predict.csv"
            Y_pred = Y_pred.astype("float")
            if (not output_layer_activation) or (output_layer_activation == 'relu'):
                Y_result = pd.DataFrame(Y_pred, columns=["Y_pred"])
            elif output_layer_activation == 'sigmoid':
                Y_prob = pd.DataFrame(Y_pred, columns=["Y_prob"])
                Y_class = (Y_pred > self.predict_threshold) * 1
                Y_class = pd.DataFrame(Y_class, columns=[f"Y_class(>{self.predict_threshold})"])
                Y_result = pd.concat([Y_prob, Y_class], axis=1)
            Y_result.to_csv(output_file_predict_prob, header=True, index=False)
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("predict finish.")

    def layer(self, input_tensor, input_dim, output_dim, activation, layer_name='Dense'):
        with tf.name_scope(layer_name):
            W = tf.Variable(tf.random_normal([input_dim, output_dim], dtype=tf.float64), name='W')
            if self.use_intercept:
                b = tf.Variable(tf.random_normal([output_dim], dtype=tf.float64), name='b')
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
    
    def create_set_channel(self):
        '''
        create and set channel.
        '''
        io_channel = io.APIManager()
        log.info("start create channel")
        channel = io_channel.create_channel(self.party_id, self.channel_config)
        log.info("start set channel")
        rtt.set_channel("", channel)
        log.info("set channel success.")
        
    def extract_feature_or_index(self):
        '''
        Extract feature columns or index column from input file.
        '''
        file_x = ""
        id_col = None
        temp_dir = self.get_temp_dir()
        if self.party_id in self.data_party:
            if self.input_file:
                usecols = [self.key_column] + self.selected_columns
                input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
                input_data = input_data[usecols]
                assert input_data.shape[0] > 0, 'input file is no data.'
                if self.use_psi:
                    psi_result = pd.read_csv(self.psi_result_file, dtype="str")
                    psi_result.name = self.key_column
                    input_data = pd.merge(psi_result, input_data, on=self.key_column, how='inner')
                    assert input_data.shape[0] > 0, 'input data is empty. bacause no intersection with psi result.'
                
                id_col = input_data[self.key_column]
                file_x = os.path.join(temp_dir, f"file_x_{self.party_id}.csv")
                x_data = input_data.drop(labels=self.key_column, axis=1)
                x_data.to_csv(file_x, header=True, index=False)
            else:
                raise Exception(f"data_party:{self.party_id} not have data. input_file:{self.input_file}")
        return file_x, id_col
    
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


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    log.info("start main function.")
    privacy_dnn = PrivacyDnnPredict(channel_config, cfg_dict, data_party, result_party, results_dir)
    privacy_dnn.predict()
    log.info("finish main function.")
