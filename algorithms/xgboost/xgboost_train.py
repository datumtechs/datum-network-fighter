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
import latticex.rosetta as rtt
import channel_sdk.pyio as io


np.set_printoptions(suppress=True)
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

class PrivacyXgbTrain(object):
    '''
    Privacy XGBoost train base on rosetta.
    '''

    def __init__(self,
                 channel_config: str,
                 cfg_dict: dict,
                 data_party: list,
                 compute_party: list,
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
                "label_owner": "data1",
                "label_column": "Y",
                "hyperparams": {
                    "epochs": 10,
                    "batch_size": 256,
                    "learning_rate": 0.01,
                    "num_trees": 1,   # num of trees
                    "max_depth": 3,   # max depth of per tree
                    "num_bins": 4,    # num of bins of feature
                    "num_class": 2,   # num of class of label
                    "lambd": 1.0,     # L2 regular coefficient, [0, +∞)
                    "gamma": 0.0,     # Gamma, also known as "complexity control", is an important parameter we use to prevent over fitting
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
        assert isinstance(compute_party, (list, tuple)), "type of compute_party must be list or tuple"
        assert isinstance(result_party, (list, tuple)), "type of result_party must be list or tuple"
        assert isinstance(results_dir, str), "type of results_dir must be str"
        
        log.info(f"start get input parameter.")
        self.channel_config = channel_config
        self.data_party = list(data_party)
        self.compute_party = list(compute_party)
        self.result_party = list(result_party)
        self.results_dir = results_dir
        self.output_dir = self.get_output_dir()
        self.output_file = os.path.join(self.output_dir, "model")
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
        self.learning_rate = hyperparams.get("learning_rate", 0.1)
        self.num_trees = hyperparams.get("num_trees", 1)
        self.max_depth = hyperparams.get("max_depth", 3)
        self.num_bins = hyperparams.get("num_bins", 4)
        self.num_class = hyperparams.get("num_class", 2)
        self.lambd = hyperparams.get("lambd", 1.0)
        self.gamma = hyperparams.get("gamma", 0.0)
        self.use_validation_set = hyperparams.get("use_validation_set", True)
        self.validation_set_rate = hyperparams.get("validation_set_rate", 0.2)
        self.predict_threshold = hyperparams.get("predict_threshold", 0.5)

    def _check_parameters(self):
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
        
        if self.party_id in self.data_party:
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
             
        result_path, result_type, evaluation_result = "", "", ""
        if self.party_id in self.result_party:
            log.info("result_party deal with the result.")
            result_path = self.output_dir
            result_type = "dir"
            if self.use_validation_set:
                log.info("result_party evaluate model.")
                Y_pred = np.squeeze(Y_pred.astype("float"))
                Y_true = np.squeeze(Y_actual.astype("float"))
                evaluation_result = self.model_evaluation(Y_true, Y_pred)
        
        log.info("start remove temp dir.")
        self.remove_temp_dir()
        log.info("train success all.")
        return result_path, result_type, evaluation_result
    
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
            
            input_data = pd.read_csv(self.input_file, usecols=usecols, dtype="str")
            input_data = input_data[usecols]
            assert input_data.shape[0] > 0, 'input file is no data.'
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
        self._mkdir(temp_dir)
        return temp_dir
    
    def get_output_dir(self):
        output_dir = os.path.join(self.results_dir, 'model')
        self._mkdir(output_dir)
        return output_dir

    def remove_temp_dir(self):
        if self.party_id in (self.data_party + self.result_party):
            # only delete the temp dir
            temp_dir = self.get_temp_dir()
        else:
            # delete the all results in the compute party.
            temp_dir = self.results_dir
        self._remove_dir(temp_dir)
    
    def _mkdir(self, _directory):
        if not os.path.exists(_directory):
            os.makedirs(_directory, exist_ok=True)

    def _remove_dir(self, _directory):
        if os.path.exists(_directory):
            shutil.rmtree(_directory)


def main(channel_config: str, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    algo_type = "privacy_xgb_train"
    try:
        log.info(f"start main function. {algo_type}.")
        privacy_xgb = PrivacyXgbTrain(channel_config, cfg_dict, data_party, compute_party, result_party, results_dir)
        result_path, result_type, extra = privacy_xgb.train()
        log.info(f"finish main function. {algo_type}.")
        return result_path, result_type, extra
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
        raise Exception(f"<ALGO>:{algo_type}. <RUN_STAGE>:{log.run_stage} "
                        f"<ERROR>: {error_filename},{error_lineno},{error_function},{error_msg}")
