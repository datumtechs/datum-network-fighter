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

class PrivacyLRTrain(object):
    '''
    Privacy logistic regression train base on rosetta.
    '''

    def __init__(self,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_dir: str):
        log.info(f"cfg_dict:{cfg_dict}, data_party:{data_party}, "
                 f"result_party:{result_party}, results_dir:{results_dir}")
        assert isinstance(cfg_dict, dict), "type of cfg_dict must be dict"
        assert isinstance(data_party, (list, tuple)), "type of data_party must be list or tuple"
        assert isinstance(result_party, (list, tuple)), "type of result_party must be list or tuple"
        assert isinstance(results_dir, str), "type of results_dir must be str"
        
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.party_id = cfg_dict["party_id"]
        self.input_file = cfg_dict["data_party"].get("input_file")
        self.id_column_name = cfg_dict["data_party"].get("key_column")
        self.feature_column_name = cfg_dict["data_party"].get("selected_columns")

        dynamic_parameter = cfg_dict["dynamic_parameter"]
        self.label_owner = dynamic_parameter.get("label_owner")
        if self.party_id == self.label_owner:
            self.label_column_name = dynamic_parameter.get("label_column_name")
            self.data_with_label = True
        else:
            self.label_column_name = ""
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
        assert self.epochs > 0, "epochs must be greater 0"
        assert self.batch_size > 0, "batch size must be greater 0"
        assert self.learning_rate > 0, "learning rate must be greater 0"
        assert 0 < self.validation_set_rate < 1, "validattion set rate must be between (0,1)"
        assert 0 <= self.predict_threshold <= 1, "predict threshold must be between [0,1]"
        
    def train(self):
        '''
        Logistic regression training algorithm implementation function
        '''

        log.info("extract feature or label.")
        train_x, train_y, val_x, val_y = self.extract_feature_or_label(with_label=self.data_with_label)
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
            logits = tf.matmul(X, W) + b
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
            loss = tf.reduce_mean(loss)
            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
            
            pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
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
                    log.debug(f"Y_pred:\n {Y_pred[:10]}")
                    Y_actual = sess.run(reveal_Y_actual, feed_dict={actual_Y: shard_y_val})
                    log.debug(f"Y_actual:\n {Y_actual[:10]}")
        
            running_stats = str(rtt.get_perf_stats(True)).replace('\n', '').replace(' ', '')
            log.info(f"running stats: {running_stats}")
        else:
            log.info("computing, please waiting for compute finish...")
        rtt.deactivate()
     
        log.info("remove temp dir.")
        if self.party_id in (self.data_party + self.result_party):
            self.remove_temp_dir()
        else:
            # delete the model in the compute party.
            self.remove_output_dir()
        
        if (self.party_id in self.result_party) and self.use_validation_set:
            log.info("result_party evaluate model.")
            from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
            Y_pred_prob = Y_pred.astype("float").reshape([-1, ])
            Y_true = Y_actual.astype("float").reshape([-1, ])
            auc_score = roc_auc_score(Y_true, Y_pred_prob)
            Y_pred_class = (Y_pred_prob > self.predict_threshold).astype('int64')  # default threshold=0.5
            accuracy = accuracy_score(Y_true, Y_pred_class)
            f1_score = f1_score(Y_true, Y_pred_class)
            precision = precision_score(Y_true, Y_pred_class)
            recall = recall_score(Y_true, Y_pred_class)
            log.info("********************")
            log.info(f"AUC: {round(auc_score, 6)}")
            log.info(f"ACCURACY: {round(accuracy, 6)}")
            log.info(f"F1_SCORE: {round(f1_score, 6)}")
            log.info(f"PRECISION: {round(precision, 6)}")
            log.info(f"RECALL: {round(recall, 6)}")
            log.info("********************")
        log.info("train finish.")
    
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
                    usecols = self.feature_column_name + [self.label_column_name]
                else:
                    usecols = self.feature_column_name
                
                input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
                input_data = input_data[usecols]
                # only if self.validation_set_rate==0, split_point==input_data.shape[0]
                split_point = int(input_data.shape[0] * (1 - self.validation_set_rate))
                assert split_point > 0, f"train set is empty, because validation_set_rate:{self.validation_set_rate} is too big"
                
                if with_label:
                    y_data = input_data[self.label_column_name]
                    train_y = os.path.join(temp_dir, f"train_y_{self.party_id}.csv")
                    y_data.iloc[:split_point].to_csv(train_y, header=True, index=False)
                    if self.use_validation_set:
                        assert split_point < input_data.shape[0], \
                            f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small"
                        val_y = os.path.join(temp_dir, f"val_y_{self.party_id}.csv")
                        y_data.iloc[split_point:].to_csv(val_y, header=True, index=False)
                    del input_data[self.label_column_name]
                
                x_data = input_data
                train_x = os.path.join(temp_dir, f"train_x_{self.party_id}.csv")
                x_data.iloc[:split_point].to_csv(train_x, header=True, index=False)
                if self.use_validation_set:
                    assert split_point < input_data.shape[0], \
                            f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small"
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


def main(cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    privacy_lr = PrivacyLRTrain(cfg_dict, data_party, result_party, results_dir)
    privacy_lr.train()
