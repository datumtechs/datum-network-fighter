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
import latticex.psi as psi


log = logging.getLogger(__name__)

class PrivateSetIntersection(object):
    '''
    private set intersection.
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
        self.use_validation_set = algorithm_parameter.get("use_validation_set", True)
        self.validation_set_rate = algorithm_parameter.get("validation_set_rate", 0.2)
        self.predict_threshold = algorithm_parameter.get("predict_threshold", 0.5)

        output_path = os.path.join(results_root_dir, f'{task_id}/{self.party_id}')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        self.output_file = os.path.join(output_path, "model")
                
    def run(self):
        '''
        run psi
        '''

        if self.party_id == 0:
            nodeId = 'P0'
        else:
            nodeId = 'P1'

        psihandler = psi.PSIHandler()
        log.info("start set log.")
        psihandler.log_to_stdout(True)
        psihandler.set_loglevel(psi.LogLevel.Info)

        log.info("start set recv party.")
        psihandler.set_recv_party(receiveParty, jobId) # 0:P0, 1:P1, 2: P0 and P1

        # IO
        iohandler = psi.IOHandler()
        log.info("start create io.")
        io = iohandler.create_io(jobId, nodeId, cfg_json)
        log.info("start set io.")
        psihandler.set_io(io, jobId)
        log.info("start activate.")
        psihandler.activate('T_V1_Basic_GLS254', jobId)
        log.info("finish activate.")

        log.info("start set batchsize.")
        psihandler.set_sender_batchsize(1000 * 1000 * 6, jobId)  # the number of row per batch
        psihandler.set_receiver_batchsize(1000 * 1000 * 6, jobId)
        psihandler.set_receiver_finder_batchsize(1000 * 1000 * 6, jobId)

        log.info("extract id.")
        input_file = extract_id()
        log.info("start prepare data.")
        psihandler.prepare(input_file, taskid=jobId)
        log.info("start run.")
        psihandler.run(input_file, output_file, taskid=jobId)
        log.info("finish run.")
        run_stats = psihandler.get_perf_stats(True, jobId)
        run_stats = run_stats.replace('\n', '').replace(' ', '')
        log.info(f"run stats: {run_stats}")
        log.info("start deactivate !")
        psihandler.deactivate(jobId)
        log.info("start release io !")
        iohandler.release_io(jobId)
        log.info("psi success!")
    
    def extract_id(self):
        '''
        Extract feature columns or label column from input file,
        and then divide them into train set and validation set.
        '''
        train_x = ""
        train_y = ""
        val_x = ""
        val_y = ""
        temp_dir = self.get_temp_dir()
        
        usecol = self.key_column
        input_data = pd.read_csv(self.input_file, usecols=usecol, dtype="str")
        input_file = os.path.join(temp_dir, f"input_{self.party_id}.csv")
        input_data.to_csv(input_file, header=True, index=False)
        return input_file
    
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
    psi = PrivateSetIntersection(task_id, cfg_dict, data_party, result_party, results_root_dir)
    psi.run()
