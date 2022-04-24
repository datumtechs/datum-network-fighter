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

class PrivacyLRTrain(object):
    '''
    Privacy logistic regression train base on rosetta.
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
                "label_owner": "data1",
                "label_column": "Y",
                "hyperparams": {
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.1,
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
        
        log.info(f"start get input parameter.")
        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.result_party = list(result_party)
        self.results_dir = results_dir
        self.output_file = os.path.join(results_dir, "model")
        self._parse_algo_cfg(cfg_dict)
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
                else:
                    raise Exception("paramter error. input_type only support 1/2")
        
        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]
        self.use_psi = dynamic_parameter.get("use_psi", True)            
        self.label_owner = dynamic_parameter.get("label_owner")
        if self.party_id == self.label_owner:
            self.label_column = dynamic_parameter.get("label_column")
            self.data_with_label = True
        else:
            self.label_column = ""
            self.data_with_label = False
                        
        hyperparams = dynamic_parameter["hyperparams"]
        self.epochs = hyperparams.get("epochs", 10)
        self.batch_size = hyperparams.get("batch_size", 256)
        self.learning_rate = hyperparams.get("learning_rate", 0.001)
        self.use_validation_set = hyperparams.get("use_validation_set", True)
        self.validation_set_rate = hyperparams.get("validation_set_rate", 0.2)
        self.predict_threshold = hyperparams.get("predict_threshold", 0.5)

    def _check_parameters(self):
        log.info(f"check parameter start.")       
        assert isinstance(self.epochs, int) and self.epochs > 0, "epochs must be type(int) and greater 0"
        assert isinstance(self.batch_size, int) and self.batch_size > 0, "batch_size must be type(int) and greater 0"
        assert isinstance(self.learning_rate, float) and self.learning_rate > 0, "learning rate must be type(float) and greater 0"
        assert isinstance(self.use_validation_set, bool), "use_validation_set must be type(bool), true or false"
        assert 0 < self.validation_set_rate < 1, "validattion set rate must be between (0,1)"
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
            
            assert isinstance(self.input_file, str), "origin data_path must be type(string)"
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
                if self.label_column:
                    assert self.label_column in input_columns, f"label_column:{self.label_column} not in input_file"
                    assert self.label_column not in self.selected_columns, f"label_column:{self.label_column} can not in selected_columns"
            else:
                raise Exception(f"input_file is not exist. input_file={self.input_file}")
        log.info(f"check parameter finish.")
                        
        
    def train(self):
        '''
        Logistic regression training algorithm implementation function
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
            
            pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
            reveal_Y = rtt.SecureReveal(pred_Y)
            actual_Y = tf.placeholder(tf.float64, [None, 1])
            reveal_Y_actual = rtt.SecureReveal(actual_Y)
            log.info("finish build the model structure.")

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
            Y_pred = Y_pred.astype("float").reshape([-1, ])
            Y_true = Y_actual.astype("float").reshape([-1, ])
            evaluation_result = self.model_evaluation(Y_true, Y_pred)
        else:
            evaluation_result = ""
        log.info("train success all.")
        return evaluation_result
    
    def model_evaluation(self, Y_true, Y_pred):
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
        evaluation_result = {
            "AUC": auc_score,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall
        }
        log.info(f"evaluation_result = {evaluation_result}")
        evaluation_result = json.dumps(evaluation_result)
        log.info("evaluation success.")
        return evaluation_result
    
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
            
            input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str") # usecols not ensure the order of columns
            input_data = input_data[usecols]  # use for ensure the order of columns
            assert input_data.shape[0] > 0, 'input file is no data.'
            if self.use_psi:
                psi_result = pd.read_csv(self.psi_result_data, dtype="str")
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
                assert train_class_num == 2, f"train set must be 2 class, not {train_class_num} class."
                train_y = os.path.join(temp_dir, f"train_y_{self.party_id}.csv")
                train_y_data.to_csv(train_y, header=True, index=False)
                if self.use_validation_set:
                    assert split_point < input_data.shape[0], \
                        f"validation set is empty, because validation_set_rate:{self.validation_set_rate} is too small"
                    val_y_data = y_data.iloc[split_point:]
                    val_class_num = val_y_data.unique().shape[0]
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


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    log.info("start main function. logistic regression train.")
    try:
        privacy_lr = PrivacyLRTrain(channel_config, cfg_dict, data_party, result_party, results_dir)
        privacy_lr.train()
    except Exception as e:
        et, ev, tb = sys.exc_info()
        error_filename = traceback.extract_tb(tb)[1].filename
        error_filename = os.path.split(error_filename)[1]
        error_lineno = traceback.extract_tb(tb)[1].lineno
        error_function = traceback.extract_tb(tb)[1].name
        error_msg = str(e)
        raise Exception(f"<ALGO>:lr_train. <RUN_STAGE>:{log.run_stage} <ERROR>: {error_filename},{error_lineno},{error_function},{error_msg}")
    log.info("finish main function. logistic regression train.")
