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

class PrivacyLRPredict(object):
    '''
    Privacy logistic regression predict base on rosetta.
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
            "self_cfg_params": {
                "party_id": "data1",
                "input_data": [
                    {
                        "input_type": 1,
                        "data_type": 1,
                        "data_path": "path/to/data",
                        "key_column": "col1",
                        "selected_columns": ["col2", "col3"]
                    },
                    {
                        "input_type": 2,
                        "data_type": 1,
                        "data_path": "path/to/data1/psi_result.csv",
                        "key_column": "",
                        "selected_columns": []
                    }
                ]
            },
            "algorithm_dynamic_params": {
                "use_psi": true,
                "model_restore_party": "model1",
                "predict_threshold": 0.5
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
        
        log.info(f"start get input parameter.")
        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.results_dir = results_dir
        self.output_file = os.path.join(results_dir, "result")
        self._parse_algo_cfg(cfg_dict)
        self.data_party.remove(self.model_restore_party)  # except restore party
        self._check_parameters()
    
    def _parse_algo_cfg(self, cfg_dict):
        self.party_id = cfg_dict["self_cfg_params"]["party_id"]
        input_data = cfg_dict["self_cfg_params"]["input_data"]
        self.psi_result_data = None
        if self.party_id in self.data_party:
            for data in input_data:
                input_type = data["input_type"]
                data_type = data["data_type"]
                if input_type == 1:
                    self.input_file = data["data_path"]
                    self.key_column = data.get("key_column")
                    self.selected_columns = data.get("selected_columns")
                elif input_type == 2:
                    self.psi_result_data = data["data_path"]
                elif input_type == 3:
                    self.model_path = data["data_path"]
                    self.model_file = os.path.join(self.model_path, "model")
                else:
                    raise Exception("paramter error. input_type only support 1/2/3")

        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]
        self.use_psi = dynamic_parameter.get("use_psi", True)
        self.model_restore_party = dynamic_parameter.get("model_restore_party")
        self.predict_threshold = dynamic_parameter.get("predict_threshold", 0.5)        

    def _check_parameters(self):
        log.info(f"check parameter start.")        
        assert 0 <= self.predict_threshold <= 1, "predict threshold must be between [0,1]"
        
        if self.party_id in self.data_party:
            assert isinstance(self.use_psi, bool), "use_psi must be type(bool), true or false"
            if self.use_psi:
                assert isinstance(self.psi_result_data, str), f"psi_result_data must be type(string), not {self.psi_result_data}"
                self.psi_result_data = self.psi_result_data.strip()
                if os.path.exists(self.psi_result_data):
                    file_suffix = os.path.splitext(self.psi_result_data)[-1][1:]
                    assert file_suffix == "csv", f"psi_result_data must csv file, not {file_suffix}"
                else:
                    raise Exception(f"psi_result_data is not exist. psi_result_data={self.psi_result_data}")
            
            assert isinstance(self.input_file, str), "origin input_data must be type(string)"
            assert isinstance(self.key_column, str), "key_column must be type(string)"
            assert isinstance(self.selected_columns, list), "selected_columns must be type(list)" 
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
        if self.party_id == self.model_restore_party:
            assert os.path.exists(self.model_path), f"model path not found. model_path={self.model_path}"
        log.info(f"check parameter finish.")
       

    def predict(self):
        '''
        Logistic regression predict algorithm implementation function
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

        X = tf.placeholder(tf.float64, [None, column_total_num])
        W = tf.Variable(tf.zeros([column_total_num, 1], dtype=tf.float64))
        b = tf.Variable(tf.zeros([1], dtype=tf.float64))
        saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
        init = tf.global_variables_initializer()
        # predict
        pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
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
            Y_pred_prob = sess.run(reveal_Y, feed_dict={X: shard_x})
            log.debug(f"Y_pred_prob:\n {Y_pred_prob[:10]}")
            predict_use_time = round(time.time() - predict_start_time, 3)
            log.info(f"predict finish. predict_use_time={predict_use_time}s")
        rtt.deactivate()
        log.info("rtt deactivate finish.")
        
        if self.party_id in self.result_party:
            log.info("predict result write to file.")
            output_file_predict_prob = os.path.splitext(self.output_file)[0] + "_predict.csv"
            Y_pred_prob = Y_pred_prob.astype("float")
            Y_prob = pd.DataFrame(Y_pred_prob, columns=["Y_prob"])
            Y_class = (Y_pred_prob > self.predict_threshold) * 1
            Y_class = pd.DataFrame(Y_class, columns=["Y_class"])
            Y_result = pd.concat([Y_prob, Y_class], axis=1)
            Y_result.to_csv(output_file_predict_prob, header=True, index=False)
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("predict success all.")

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
        
    def extract_feature_or_index(self):
        '''
        Extract feature columns or index column from input file.
        '''
        file_x = ""
        id_col = None
        temp_dir = self.get_temp_dir()
        if self.party_id in self.data_party:
            usecols = [self.key_column] + self.selected_columns
            input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
            input_data = input_data[usecols]
            assert input_data.shape[0] > 0, 'input file is no data.'
            if self.use_psi:
                psi_result = pd.read_csv(self.psi_result_data, dtype="str")
                psi_result.name = self.key_column
                input_data = pd.merge(psi_result, input_data, on=self.key_column, how='inner')
                assert input_data.shape[0] > 0, 'input data is empty. because no intersection with psi result.'

            id_col = input_data[self.key_column]
            file_x = os.path.join(temp_dir, f"file_x_{self.party_id}.csv")
            x_data = input_data.drop(labels=self.key_column, axis=1)
            x_data.to_csv(file_x, header=True, index=False)
        
        return file_x, id_col
    
    def get_temp_dir(self):
        '''
        Get the directory for temporarily saving files
        '''
        temp_dir = os.path.join(os.path.dirname(self.output_file), 'temp')
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


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    log.info("start main function. logistic regression predict.")
    try:
        privacy_lr = PrivacyLRPredict(channel_config, cfg_dict, data_party, result_party, results_dir)
        privacy_lr.predict()
    except Exception as e:
        et, ev, tb = sys.exc_info()
        error_filename = traceback.extract_tb(tb)[1].filename
        error_filename = os.path.split(error_filename)[1]
        error_lineno = traceback.extract_tb(tb)[1].lineno
        error_function = traceback.extract_tb(tb)[1].name
        error_msg = str(e)
        raise Exception(f"<ALGO>:lr_predict. <RUN_STAGE>:{log.run_stage} <ERROR>: {error_filename},{error_lineno},{error_function},{error_msg}")
    log.info("finish main function. logistic regression predict.")
