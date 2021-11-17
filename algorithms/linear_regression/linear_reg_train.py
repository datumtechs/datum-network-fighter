# coding:utf-8

import os
import sys
import math
import json
import time
import logging
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import latticex.rosetta as rtt
import channel_sdk


np.set_printoptions(suppress=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rtt.set_backend_loglevel(5)  # All(0), Trace(1), Debug(2), Info(3), Warn(4), Error(5), Fatal(6)
log = logging.getLogger(__name__)

class PrivacyLinearRegTrain(object):
    '''
    Privacy linear regression train base on rosetta.
    '''

    def __init__(self,
                 channel_config: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_dir: str):
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

        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.label_owner = dynamic_parameter.get("label_owner")
        if self.party_id == self.label_owner:
            self.label_column = dynamic_parameter.get("label_column")
            self.data_with_label = True
        else:
            self.label_column = ""
            self.data_with_label = False
                        
        algorithm_parameter = dynamic_parameter["algorithm_parameter"]
        self.epochs = algorithm_parameter.get("epochs", 10)
        self.batch_size = algorithm_parameter.get("batch_size", 256)
        self.learning_rate = algorithm_parameter.get("learning_rate", 0.001)
        self.use_validation_set = algorithm_parameter.get("use_validation_set", True)
        self.validation_set_rate = algorithm_parameter.get("validation_set_rate", 0.2)
        self.predict_threshold = algorithm_parameter.get("predict_threshold", 0.5)

        self.output_file = os.path.join(results_dir, "model")
        
        self.check_parameters()

    def check_parameters(self):
        log.info(f"check parameter start.")        
        assert self.epochs > 0, "epochs must be greater 0"
        assert self.batch_size > 0, "batch size must be greater 0"
        assert self.learning_rate > 0, "learning rate must be greater 0"
        assert 0 < self.validation_set_rate < 1, "validattion set rate must be between (0,1)"
        assert 0 <= self.predict_threshold <= 1, "predict threshold must be between [0,1]"
        
        if self.input_file:
            self.input_file = self.input_file.strip()
        if self.party_id in self.data_party:
            if os.path.exists(self.input_file):
                input_columns = pd.read_csv(self.input_file, nrows=0)
                input_columns = list(input_columns.columns)
                if self.key_column:
                    assert self.key_column in input_columns, f"key_column:{self.key_column} not in input_file"
                if self.selected_columns:
                    error_col = []
                    for col in self.selected_columns:
                        if col not in input_columns:
                            error_col.append(col)   
                    assert not error_col, f"selected_columns:{error_col} not in input_file"
                if self.label_column:
                    assert self.label_column in input_columns, f"label_column:{self.label_column} not in input_file"
            else:
                raise Exception(f"input_file is not exist. input_file={self.input_file}")
        log.info(f"check parameter finish.")
                        
        
    def train(self):
        '''
        Linear regression training algorithm implementation function
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
        shard_x, shard_y = rtt.PrivateDataset(data_owner=self.data_party, label_owner=self.label_owner).load_data(train_x, train_y, header=0)
        log.info("finish sharing train data.")
        column_total_num = shard_x.shape[1]
        log.info(f"column_total_num = {column_total_num}.")
        
        if self.use_validation_set:
            log.info("start sharing validation data.")
            shard_x_val, shard_y_val = rtt.PrivateDataset(data_owner=self.data_party, label_owner=self.label_owner).load_data(val_x, val_y, header=0)
            log.info("finish sharing validation data.")

        if self.party_id not in self.data_party:  
            # mean the compute party and result party
            log.info("compute start.")
            X = tf.placeholder(tf.float64, [None, column_total_num])
            Y = tf.placeholder(tf.float64, [None, 1])
            W = tf.Variable(tf.zeros([column_total_num, 1], dtype=tf.float64))
            b = tf.Variable(tf.zeros([1], dtype=tf.float64))
            pred_Y = tf.matmul(X, W) + b
            loss = tf.square(Y - pred_Y)
            loss = tf.reduce_mean(loss)
            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
            
            reveal_Y = rtt.SecureReveal(pred_Y)
            actual_Y = tf.placeholder(tf.float64, [None, 1])
            reveal_Y_actual = rtt.SecureReveal(actual_Y)

            with tf.Session() as sess:
                log.info("session init.")
                sess.run(init)
                # train
                log.info("train start.")
                train_start_time = time.time()
                batch_num = math.ceil(len(shard_x) / self.batch_size)
                for e in range(self.epochs):
                    for i in range(batch_num):
                        bX = shard_x[(i * self.batch_size): (i + 1) * self.batch_size]
                        bY = shard_y[(i * self.batch_size): (i + 1) * self.batch_size]
                        sess.run(optimizer, feed_dict={X: bX, Y: bY})
                        if (i % 50 == 0) or (i == batch_num - 1):
                            log.info(f"epoch:{e + 1}/{self.epochs}, batch:{i + 1}/{batch_num}")
                log.info(f"model save to: {self.output_file}")
                saver.save(sess, self.output_file)
                train_use_time = round(time.time()-train_start_time, 3)
                log.info(f"save model success. train_use_time={train_use_time}s")
                
                if self.use_validation_set:
                    Y_pred = sess.run(reveal_Y, feed_dict={X: shard_x_val})
                    log.info(f"Y_pred:\n {Y_pred[:10]}")
                    Y_actual = sess.run(reveal_Y_actual, feed_dict={actual_Y: shard_y_val})
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
        
        if (self.party_id in self.result_party) and self.use_validation_set:
            log.info("result_party evaluate model.")
            from sklearn.metrics import r2_score, mean_squared_error
            Y_pred = Y_pred.astype("float").reshape([-1, ])
            Y_true = Y_actual.astype("float").reshape([-1, ])
            r2 = r2_score(Y_true, Y_pred)
            rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
            log.info("********************")
            log.info(f"R Squared: {round(r2, 6)}")
            log.info(f"RMSE: {round(rmse, 6)}")
            log.info("********************")
        log.info("train finish.")
    
    def create_set_channel(self):
        '''
        create and set channel.
        '''
        io_channel = channel_sdk.grpc.APIManager()
        log.info("start create channel")
        channel = io_channel.create_channel(self.party_id, self.channel_config)
        log.info("start set channel")
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
            if self.input_file:
                if with_label:
                    usecols = self.selected_columns + [self.label_column]
                else:
                    usecols = self.selected_columns
                
                input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
                input_data = input_data[usecols]
                # only if self.validation_set_rate==0, split_point==input_data.shape[0]
                split_point = int(input_data.shape[0] * (1 - self.validation_set_rate))
                assert split_point > 0, f"train set is empty, because validation_set_rate:{self.validation_set_rate} is too big"
                
                if with_label:
                    y_data = input_data[self.label_column]
                    train_y_data = y_data.iloc[:split_point]
                    train_y = os.path.join(temp_dir, f"train_y_{self.party_id}.csv")
                    train_y_data.to_csv(train_y, header=True, index=False)
                    if self.use_validation_set:
                        assert split_point < input_data.shape[0], \
                            f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small"
                        val_y_data = y_data.iloc[split_point:]
                        val_y = os.path.join(temp_dir, f"val_y_{self.party_id}.csv")
                        val_y_data.to_csv(val_y, header=True, index=False)
                    del input_data[self.label_column]
                
                x_data = input_data
                train_x = os.path.join(temp_dir, f"train_x_{self.party_id}.csv")
                x_data.iloc[:split_point].to_csv(train_x, header=True, index=False)
                if self.use_validation_set:
                    assert split_point < input_data.shape[0], \
                            f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small."
                    val_x = os.path.join(temp_dir, f"val_x_{self.party_id}.csv")
                    x_data.iloc[split_point:].to_csv(val_x, header=True, index=False)
            else:
                raise Exception(f"data_node {self.party_id} not have data. input_file:{self.input_file}")
        return train_x, train_y, val_x, val_y
    
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
    
    def remove_output_dir(self):
        '''
        Delete all files in the temporary directory, these files are some temporary data.
        This is used to delete all output files of the non-resulting party
        '''
        temp_dir = os.path.dirname(self.output_file)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    privacy_linear_reg = PrivacyLinearRegTrain(channel_config, cfg_dict, data_party, result_party, results_dir)
    privacy_linear_reg.train()
