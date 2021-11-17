# coding:utf-8

import sys
sys.path.append("..")
import os
import math
import json
import time
import logging
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import latticex.rosetta as rtt


np.set_printoptions(suppress=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rtt.set_backend_loglevel(5)  # All(0), Trace(1), Debug(2), Info(3), Warn(4), Error(5), Fatal(6)
log = logging.getLogger(__name__)

class PrivacyLRPredict(object):
    '''
    Privacy logistic regression predict base on rosetta.
    '''

    def __init__(self,
                 task_id: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_root_dir: str):
        log.info(f"task_id:{task_id}, cfg_dict:{cfg_dict}, data_party:{data_party},"
                 f"result_party:{result_party}, results_root_dir:{results_root_dir}")
        assert isinstance(task_id, str), "type of task_id must be string"
        assert isinstance(cfg_dict, dict), "type of cfg_dict must be dict"
        assert isinstance(data_party, (list, tuple)), "type of data_party must be list or tuple"
        assert isinstance(result_party, (list, tuple)), "type of result_party must be list or tuple"
        assert isinstance(results_root_dir, str), "type of results_root_dir must be str"
        
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.party_id = cfg_dict["party_id"]
        self.input_file = cfg_dict["data_party"].get("input_file")
        self.id_column_name = cfg_dict["data_party"].get("key_column")
        self.feature_column_name = cfg_dict["data_party"].get("selected_columns")
        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.model_restore_party = dynamic_parameter.get("model_restore_party")
        train_task_id = dynamic_parameter.get("train_task_id")
        self.model_file = os.path.join(results_root_dir, f"{train_task_id}/model")
        self.predict_threshold = dynamic_parameter.get("predict_threshold", 0.5)
        assert 0 <= self.predict_threshold <= 1, "predict threshold must be between [0,1]"
        
        output_path = os.path.join(results_root_dir, f"{task_id}/{self.party_id}")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        self.output_file = os.path.join(output_path, "result")
        
        self.data_party.remove(self.model_restore_party)  # except restore party
       

    def predict(self):
        '''
        Logistic regression predict algorithm implementation function
        '''

        log.info("extract feature or id.")
        file_x, id_col = self.extract_feature_or_index()
        log.info("waiting other party...")
        rtt.activate("SecureNN")
        log.info("protocol has been activated.")
        rtt.set_restore_model(False, plain_model=self.model_restore_party)
        log.info("finish set restore model.")
        # sharing data
        log.info("start sharing data .")
        shard_x = rtt.PrivateDataset(data_owner=self.data_party).load_X(file_x, header=0)
        log.info("finish sharing data .")
        column_total_num = shard_x.shape[1]

        X = tf.placeholder(tf.float64, [None, column_total_num])
        W = tf.Variable(tf.zeros([column_total_num, 1], dtype=tf.float64))
        b = tf.Variable(tf.zeros([1], dtype=tf.float64))
        saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
        init = tf.global_variables_initializer()
        # predict
        pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
        reveal_Y = rtt.SecureReveal(pred_Y)  # only reveal to result party

        with tf.Session() as sess:
            sess.run(init)
            if self.party_id == self.model_restore_party:
                if os.path.exists(os.path.join(os.path.dirname(self.model_file), "checkpoint")):
                    log.info(f"model restore from: {self.model_file}.")
                    saver.restore(sess, self.model_file)
                else:
                    raise Exception("model not found or model damaged")
            else:
                log.info("restore model")
                temp_file = os.path.join(self.get_temp_dir(), 'ckpt_temp_file')
                with open(temp_file, "w") as f:
                    pass
                saver.restore(sess, temp_file)
            
            # predict
            log.info("predict start.")
            predict_start_time = time.time()
            Y_pred_prob = sess.run(reveal_Y, feed_dict={X: shard_x})
            Y_pred_prob = Y_pred_prob.astype("float")
            log.debug(f"Y_pred_prob:\n {Y_pred_prob[:10]}")

            if self.party_id in self.result_party:
                log.info("predict result write to file.")
                output_file_predict_prob = os.path.splitext(self.output_file)[0] + "_predict.csv"
                Y_prob = pd.DataFrame(Y_pred_prob, columns=["Y_prob"])
                Y_class = (Y_pred_prob > self.predict_threshold) * 1
                Y_class = pd.DataFrame(Y_class, columns=["Y_class"])
                Y_result = pd.concat([Y_prob, Y_class], axis=1)
                Y_result.to_csv(output_file_predict_prob, header=True, index=False)
            predict_use_time = round(time.time() - predict_start_time, 3)
            log.info(f"predict success. predict_use_time={predict_use_time}s")
        rtt.deactivate()
        self.remove_temp_dir()
        log.info("predict finish.")

    def extract_feature_or_index(self):
        '''
        Extract feature columns or index column from input file.
        '''
        file_x = ""
        id_col = None
        temp_dir = self.get_temp_dir()
        if self.party_id in self.data_party:
            if self.input_file:
                usecols = [self.id_column_name] + self.feature_column_name
                input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
                input_data = input_data[usecols]
                id_col = input_data[self.id_column_name]
                file_x = os.path.join(temp_dir, f"file_x_{self.party_id}.csv")
                x_data = input_data.drop(labels=self.id_column_name, axis=1)
                x_data.to_csv(file_x, header=True, index=False)
            else:
                raise Exception(f"data_party:{self.party_id} not have data. input_file:{self.input_file}")
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


def main(task_id: str, cfg_dict: dict, data_party: list, result_party: list, results_root_dir: str):
    '''
    This is the entrance to this module
    '''
    privacy_lr = PrivacyLRPredict(task_id, cfg_dict, data_party, result_party, results_root_dir)
    privacy_lr.predict()
