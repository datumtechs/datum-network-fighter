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
import latticex.rosetta as rtt
import channel_sdk


np.set_printoptions(suppress=True)
rtt.set_backend_loglevel(3)  # All(0), Trace(1), Debug(2), Info(3), Warn(4), Error(5), Fatal(6)
log = logging.getLogger(__name__)

class PrivacyXgbTrain(object):
    '''
    Privacy XGBoost train base on rosetta.
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
                "label_owner": "p1",
                "label_column": "Y",
                "algorithm_parameter": {
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.01,
                    "num_trees": 3,   # num of trees
                    "max_depth": 4,   # max depth of per tree
                    "num_bins": 5,    # num of bins of feature
                    "num_class": 2,   # num of class of label
                    "lambd": 1.0,     # L2 regular coefficient, [0, +∞)
                    "gamma": 0.0,     # Gamma, also known as "complexity control", is an important parameter we use to prevent over fitting
                    "use_validation_set": True,
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
        self.learning_rate = algorithm_parameter.get("learning_rate", 0.1)
        self.num_trees = algorithm_parameter.get("num_trees", 3)
        self.max_depth = algorithm_parameter.get("max_depth", 4)
        self.num_bins = algorithm_parameter.get("num_bins", 5)
        self.num_class = algorithm_parameter.get("num_class", 2)
        self.lambd = algorithm_parameter.get("lambd", 1.0)
        self.gamma = algorithm_parameter.get("gamma", 0.0)
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
        assert isinstance(self.num_trees, int) and self.num_trees > 0, "num_trees must be type(int) and greater 0"
        assert isinstance(self.max_depth, int) and self.max_depth > 0, "max_depth must be type(int) and greater 0"
        assert isinstance(self.num_bins, int) and self.num_bins > 0, "num_bins must be type(int) and greater 0"
        assert isinstance(self.num_class, int) and self.num_class > 1, "num_class must be type(int) and greater 1"
        assert isinstance(self.lambd, (float, int)) and self.lambd >= 0, "lambd must be type(float/int) and greater_equal 0"
        assert isinstance(self.gamma, (float, int)), "gamma must be type(float/int)"
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
        shard_x, shard_y, x_pmt_idx, x_inv_pmt_idx\
            = rtt.PrivateDatasetEx(data_owner=self.data_party, 
                                    label_owner=self.label_owner,
                                    dataset_type=rtt.DatasetType.SampleAligned,
                                    num_classes=self.num_class)\
                    .load_data(train_x, train_y, header=0)
        log.info("finish sharing train data.")
        column_total_num = shard_x.shape[1]
        log.info(f"column_total_num = {column_total_num}.")
        
        if self.use_validation_set:
            log.info("start sharing validation data.")
            shard_x_val, shard_y_val\
                = rtt.PrivateDataset(data_owner=self.data_party,
                                     label_owner=self.label_owner,
                                    dataset_type=rtt.DatasetType.SampleAligned,
                                    num_classes=self.num_class)\
                    .load_data(val_x, val_y, header=0)
            log.info("finish sharing validation data.")

        if self.party_id not in self.data_party:
            log.info("start build SecureXGBClassifier.")
            xgb = rtt.SecureXGBClassifier(epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        learning_rate=self.learning_rate,
                                        max_depth=self.max_depth,
                                        num_trees=self.num_trees,
                                        num_class=self.num_class,
                                        num_bins=self.num_bins,
                                        lambd=self.lambd,
                                        gamma=self.gamma)
            log.info("start train XGBoost.")
            xgb.FitEx(shard_x, shard_y, x_pmt_idx, x_inv_pmt_idx)
            log.info("start save model.")
            xgb.SaveModel(self.output_file)
            log.info("save model success.")    
            if self.use_validation_set:
                # predict Y
                rv_pred = xgb.Reveal(xgb.Predict(shard_x_val), self.result_party)
                y_shape = rv_pred.shape
                log.info(f"y_shape: {y_shape}, rv_pred: \n {rv_pred[:10]}")
                pred_y = [[float(ii_x) for ii_x in i_x] for i_x in rv_pred]
                log.info(f"pred_y: \n {pred_y[:10]}")
                pred_y = np.array(pred_y)
                pred_y.reshape(y_shape)
                Y_pred = np.squeeze(pred_y, 1)
                log.info(f"Y_pred:\n {Y_pred[:10]}")
                # actual Y
                Y_actual = xgb.Reveal(shard_y_val, self.result_party)
                log.info(f"Y_actual:\n {Y_actual[:10]}")

            running_stats = str(rtt.get_perf_stats(True)).replace('\n', '').replace(' ', '')
            log.info(f"running stats: {running_stats}")
        else:
            log.info("computing, please waiting for compute finish...")
        rtt.deactivate()
        log.info("rtt deactivate success.")
             
        if (self.party_id in self.result_party) and self.use_validation_set:
            log.info("result_party evaluate model.")
            Y_pred = np.squeeze(Y_pred.astype("float"))
            Y_true = np.squeeze(Y_actual.astype("float"))
            self.model_evaluation(Y_true, Y_pred)
        
        log.info("remove temp dir.")
        if self.party_id in (self.data_party + self.result_party):
            # self.remove_temp_dir()
            pass
        else:
            # delete the model in the compute party.
            self.remove_output_dir()
        log.info("train success.")
    
    def model_evaluation(self, Y_true, Y_pred):
        from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score
        if self.num_class == 2:
            average = 'binary'
            multi_class = 'raise'
            Y_pred_class = (Y_pred > self.predict_threshold).astype('int64')  # default threshold=0.5
        else:
            average = 'weighted'
            multi_class = 'ovr'
            Y_pred_class = np.argmax(Y_pred, axis=1)
        auc_score = roc_auc_score(Y_true, Y_pred, multi_class=multi_class)
        accuracy = accuracy_score(Y_true, Y_pred_class)
        f1_score = f1_score(Y_true, Y_pred_class, average=average)
        precision = precision_score(Y_true, Y_pred_class, average=average)
        recall = recall_score(Y_true, Y_pred_class, average=average)
        log.info("********************")
        log.info(f"AUC: {round(auc_score, 6)}")
        log.info(f"ACCURACY: {round(accuracy, 6)}")
        log.info(f"F1_SCORE: {round(f1_score, 6)}")
        log.info(f"PRECISION: {round(precision, 6)}")
        log.info(f"RECALL: {round(recall, 6)}")
        log.info("********************")
    
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
        temp_dir = self.results_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main(channel_config: str, cfg_dict: dict, data_party: list, result_party: list, results_dir: str):
    '''
    This is the entrance to this module
    '''
    privacy_xgb = PrivacyXgbTrain(channel_config, cfg_dict, data_party, result_party, results_dir)
    privacy_xgb.train()
