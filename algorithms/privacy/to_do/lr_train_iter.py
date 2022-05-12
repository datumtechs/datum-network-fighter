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
                 task_id: str,
                 cfg_dict: dict,
                 data_party: list,
                 result_party: list,
                 results_root_dir: str):
        log.info(f"task_id:{task_id}, cfg_dict:{cfg_dict}, data_party:{data_party}, "
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
        self.use_validation_set = algorithm_parameter.get("use_validation_set", False)
        self.validation_set_rate = algorithm_parameter.get("validation_set_rate", 0.2)
        self.predict_threshold = algorithm_parameter.get("predict_threshold", 0.5)

        output_path = os.path.join(results_root_dir, f'{task_id}/{self.party_id}')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        self.output_file = os.path.join(output_path, "model")
        
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

        try:
            log.info("extract feature or label.")
            train_x, train_y, val_x, val_y = self.extract_feature_or_label(with_label=self.data_with_label)

            log.info("waiting other party...")
            rtt.activate("SecureNN")
            log.info("protocol has been activated.")
            rtt.set_saver_model(False, plain_model=self.result_party)
            log.info("finish set save model.")
            # sharing data
            log.info("start sharing train data .")
            # shard_x, shard_y = rtt.PrivateDataset(data_owner=self.data_party, label_owner=self.label_owner).load_data(train_x, train_y, header=0)
            dataset_x0 = rtt.PrivateTextLineDataset(
                train_x, data_owner='p0')  # P0 hold the file_x data
            dataset_x1 = rtt.PrivateTextLineDataset(
                train_x, data_owner='p1')  # P1 hold the file_x data
            dataset_y = rtt.PrivateTextLineDataset(
                train_y, data_owner='p0')  # P0 hold the file_y data
            log.info("finish sharing train data .")
            # column_total_num = shard_x.shape[1]
            
            # dataset decode
            def decode_p0(line):
                fields = tf.string_split([line], ',').values
                fields = rtt.PrivateInput(fields, data_owner='p0')
                return fields


            def decode_p1(line):
                fields = tf.string_split([line], ',').values
                fields = rtt.PrivateInput(fields, data_owner='p1')
                return fields


            cache_temp_dir = os.path.join(self.get_temp_dir(), 'cache')
            if not os.path.exists(cache_temp_dir):
                os.makedirs(cache_temp_dir, exist_ok=True)
            # dataset pipeline
            ### Here we using repeat so that multi-epoch can be used.
            dataset_x0 = dataset_x0 \
                .skip(1).map(decode_p0)\
                .batch(self.batch_size).repeat()

            dataset_x1 = dataset_x1 \
                .skip(1).map(decode_p1)\
                .batch(self.batch_size).repeat()

            dataset_y = dataset_y \
                .skip(1).map(decode_p0)\
                .batch(self.batch_size).repeat()

            # make iterator
            iter_x0 = dataset_x0.make_initializable_iterator()
            X0 = iter_x0.get_next()

            iter_x1 = dataset_x1.make_initializable_iterator()
            X1 = iter_x1.get_next()

            iter_y = dataset_y.make_initializable_iterator()
            Y = iter_y.get_next()

            # Join input X of P0 and P1, features splitted dataset
            X = tf.concat([X0, X1], axis=1)

            DIM_NUM = 46 #X.shape[1]
            ROW_NUM = 45036
            
            # if self.use_validation_set:
            #     log.info("start sharing validation data .")
            #     shard_x_val, shard_y_val = rtt.PrivateDataset(data_owner=self.data_party, label_owner=self.label_owner).load_data(val_x, val_y, header=0)
            #     log.info("finish sharing validation data .")
    
            log.info("compute start.")
            # initialize W & b
            W = tf.Variable(tf.zeros([DIM_NUM, 1], dtype=tf.float64))
            b = tf.Variable(tf.zeros([1], dtype=tf.float64))


            # build lr model
            pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
            dy = pred_Y - Y
            dw = tf.matmul(X, dy, transpose_a=True) * (1.0 / self.batch_size)
            db = tf.reduce_sum(dy, axis=0) * (1.0 / self.batch_size)
            delta_w = dw * self.learning_rate
            delta_b = db * self.learning_rate
            update_w = W - delta_w
            update_b = b - delta_b


            # update variables
            assign_update_w = tf.assign(W, update_w)
            assign_update_b = tf.assign(b, update_b)

            reveal_W = rtt.SecureReveal(W)
            reveal_b = rtt.SecureReveal(b)

            saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                log.info("session init.")
                sess.run(init)
                sess.run([iter_x0.initializer, iter_x1.initializer, iter_y.initializer])
                # train
                log.info("train start.")
                train_start_time = time.time()
                batch_num = math.ceil(ROW_NUM / self.batch_size)
                for e in range(self.epochs):
                    for i in range(batch_num):
                        sess.run([assign_update_w, assign_update_b])
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
        
            log.info(f"running stats:\n {rtt.get_perf_stats(True)}")
            
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
        except Exception as e:
            raise
        finally:
            rtt.deactivate()
            log.info("remove temp dir.")
            if self.party_id in (self.data_party + self.result_party):
                self.remove_temp_dir()
            else:
                # delete the model in the compute party.
                self.remove_output_dir()
    
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


def main(task_id: str, cfg_dict: dict, data_party: list, result_party: list, results_root_dir: str):
    '''
    This is the entrance to this module
    '''
    privacy_lr = PrivacyLRTrain(task_id, cfg_dict, data_party, result_party, results_root_dir)
    privacy_lr.train()
