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
import codecs
import latticex.psi as psi
import channel_sdk.pyio as io


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

class PrivateSetIntersection(object):
    '''
    private set intersection.
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
                    }
                ]
            },
            "algorithm_dynamic_params": {
                "use_alignment": true,
                "label_owner": "data1",
                "label_column": "diagnosis",
                "psi_type": "T_V1_Basic_GLS254",
                "data_flow_restrict": {
                    "data1": ["compute1"],
                    "data2": ["compute2"],
                    "result1": ["compute1"],
                    "result2": ["compute2"]
                }
            }
        }
        '''
        log.info(f"channel_config:{channel_config}")
        log.info(f"cfg_dict:{cfg_dict}")
        log.info(f"data_party:{data_party}, compute_party:{compute_party}, result_party:{result_party}, results_dir:{results_dir}")
        assert isinstance(channel_config, str), "type of channel_config must be string"
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
        self._parse_algo_cfg(cfg_dict)
        self._check_parameters()
        self.result_recv_mode = self._get_result_recv_mode()
        self.temp_dir = self._get_temp_dir()   
        self.output_file = os.path.join(results_dir, "psi_result.csv")
        self.sdk_log_level = 3  # Trace=0, Debug=1, Audit=2, Info=3, Warn=4, Error=5, Fatal=6, Off=7

    def _parse_algo_cfg(self, cfg_dict):
        self.party_id = cfg_dict["self_cfg_params"]["party_id"]
        input_data = cfg_dict["self_cfg_params"]["input_data"]
        if self.party_id in self.data_party:
            for data in input_data:
                input_type = data["input_type"]
                data_type = data["data_type"]
                if input_type == 1:
                    self.input_file = data["data_path"]
                    self.key_column = data.get("key_column")
                    self.selected_columns = data.get("selected_columns")
                else:
                    raise Exception("paramter error. input_type only support 1")
        
        dynamic_parameter = cfg_dict["algorithm_dynamic_params"]
        self.use_alignment = dynamic_parameter.get("use_alignment")
        self.label_owner = dynamic_parameter.get("label_owner")
        if self.use_alignment and (self.party_id == self.label_owner):
            self.label_column = dynamic_parameter.get("label_column")
            self.data_with_label = True
        else:
            self.label_column = ""
            self.data_with_label = False
        if not self.use_alignment:
            self.selected_columns = []
        self.psi_type = dynamic_parameter.get("psi_type", "T_V1_Basic_GLS254")  # default 'T_V1_Basic_GLS254'
        self.data_flow_restrict = dynamic_parameter.get("data_flow_restrict")

    def _check_parameters(self):
        assert len(self.data_party) == 2, f"length of data_party must be 2, not {len(self.data_party)}."
        assert len(self.result_party) in [1, 2], f"length of result_party must be 1 or 2, not {len(self.result_party)}."
        if self.party_id in self.data_party:
            self._check_input_file()

    def _check_input_file(self):
        assert isinstance(self.input_file, str), "origin input_data must be type(string)"
        self.input_file = self.input_file.strip()
        if os.path.exists(self.input_file):
            file_suffix = os.path.splitext(self.input_file)[-1][1:]
            assert file_suffix == "csv", f"input_file must csv file, not {file_suffix}"
            assert self.key_column, f"key_column can not empty. key_column={self.key_column}"
            if self.use_alignment:
                assert self.selected_columns, f"selected_columns can not empty. selected_columns={self.selected_columns}"
            input_columns = pd.read_csv(self.input_file, nrows=0)
            input_columns = list(input_columns.columns)
            assert self.key_column in input_columns, f"key_column:{self.key_column} not in input_file"
            error_col = []
            for col in self.selected_columns:
                if col not in input_columns:
                    error_col.append(col)   
            assert not error_col, f"selected_columns:{error_col} not in input_file"
            assert self.key_column not in self.selected_columns, f"key_column:{self.key_column} can not in selected_columns"
            if self.data_with_label:
                assert self.label_column in input_columns, f"label_column:{self.label_column} not in input_file"
                assert self.label_column not in self.selected_columns, f"label_column:{self.label_column} can not in selected_columns"
        else:
            raise Exception(f"input_file is not exist. input_file={self.input_file}")
    
    def _get_result_recv_mode(self):
        '''
        assume data_party=[data1, data2], compute_party=[compute1, compute2], result_party=[result1, result2]:
            if result_party = [result1, result2], then result_recv_mode = 2
            if result_party = [result1], then result_recv_mode = 0
            if result_party = [result2], then result_recv_mode = 1
        '''
        if len(self.result_party) == 2:
            result_recv_mode = 2
        else:
            if self.party_id == self.data_flow_restrict[self.result_party[0]][0]:  # Determine the value according to the data flow chain
                result_recv_mode = 0
            else:
                result_recv_mode = 1
        return result_recv_mode

    def run(self):
        log.info("start create and set channel.")
        self._create_set_channel()
        log.info("start extract data.")
        usecols_file = self._extract_data_column()
        log.info("start send_data_to_compute_party.")
        self._send_data_to_compute_party(usecols_file)

        psi_output_file = os.path.join(self.temp_dir, "psi_sdk_output.csv")
        alignment_output_file = self.output_file
        if self.party_id in self.compute_party:
            log.info("start extract key column.")
            key_col_file, key_col_name, usecols_data = self._extract_key_column(usecols_file)
            log.info("start run psi sdk.")
            self._run_psi_sdk(key_col_file, psi_output_file)
            log.info("start alignment result.")
            self._alignment_result(psi_output_file, usecols_data, alignment_output_file, key_col_name)

        log.info("start send data to result party.")
        self._send_data_to_result_party(alignment_output_file)
        log.info("finish send data to result party.")
        result_path, result_type = '', ''
        if self.party_id in self.result_party:
            result_path = alignment_output_file
            result_type = 'csv'
        log.info("start remove temp dir.")
        self._remove_temp_dir()
        log.info("psi all success.")
        return result_path, result_type
    
    def _send_data_to_compute_party(self, data_path):
        if self.party_id in self.data_party:
            compute_party = self.data_flow_restrict[self.party_id][0]
            self.send_data_to_other_party(compute_party, data_path)
        elif self.party_id in self.compute_party:
            for party in self.data_party:
                if self.party_id == self.data_flow_restrict[party][0]:
                    self.recv_data_from_other_party(party, data_path)
        else:
            pass
    
    def _send_data_to_result_party(self, data_path):
        if self.party_id in self.compute_party:
            for party in self.result_party:
                if self.party_id == self.data_flow_restrict[party][0]:
                    self.send_data_to_other_party(party, data_path)
        elif self.party_id in self.result_party:
            compute_party = self.data_flow_restrict[self.party_id][0]
            self.recv_data_from_other_party(compute_party, data_path)
        else:
            pass

    def _extract_data_column(self):
        '''
        Extract data column from input file,
        and then write to a new file.
        '''
        usecols_file = os.path.join(self.temp_dir, f"usecols_{self.party_id}.csv")

        if self.party_id in self.data_party:
            use_cols = [self.key_column] + self.selected_columns
            if self.data_with_label:
                use_cols += [self.label_column]
            log.info("read input file and write to new file.")
            usecols_data = pd.read_csv(self.input_file, usecols=use_cols, dtype="str")
            usecols_data = usecols_data[use_cols]
            usecols_data.to_csv(usecols_file, header=True, index=False)
        return usecols_file
    
    def _extract_key_column(self, usecols_file):
        usecols_data = pd.read_csv(usecols_file, header=0, dtype="str")
        usecols = list(usecols_data.columns)
        key_col_name = usecols[0]
        if self.use_alignment:
            key_data = usecols_data[key_col_name]
            key_col_file = os.path.join(self.temp_dir, f"key_col_{self.party_id}.csv")
            key_data.to_csv(key_col_file, header=True, index=False)
        else:
            key_col_file = usecols_file
        return key_col_file, key_col_name, usecols_data

    def _run_psi_sdk(self, input_file, output_file):
        '''
        run psi sdk
        '''
        log.info("start create psihandler.")
        psihandler = psi.PSIHandler()
        log.info("start set log.")
        psihandler.log_to_stdout(True)
        psihandler.set_loglevel(self.sdk_log_level)
        log.info("start set recv party.")
        # psihandler.set_recv_party(self.result_recv_mode, "")
        psihandler.set_recv_party(2, "")

        log.info("start activate.")
        psihandler.activate(self.psi_type, "")
        log.info("finish activate.")
        log.info("start psihandler prepare data.")
        psihandler.prepare(input_file, taskid="")
        log.info("start psihandler run.")
        psihandler.run(input_file, output_file, taskid="")
        log.info("finish psihandler run.")
        run_stats = psihandler.get_perf_stats(True, "")
        run_stats = run_stats.replace('\n', '').replace(' ', '')
        log.info(f"run stats: {run_stats}")
        log.info("start deactivate.")
        psihandler.deactivate("")
        log.info("finish deactivate.")
    
    def _alignment_result(self, psi_output_file, usecols_data, alignment_output_file, key_col_name):
        '''
        for the compute_party, sort the key_col values and alignment the select_columns.
        '''
        if os.path.exists(psi_output_file):
            psi_result = pd.read_csv(psi_output_file, header=None, dtype="str")
            psi_result = pd.DataFrame(psi_result.values, columns=[key_col_name])
            psi_result.sort_values(by=[key_col_name], ascending=True, inplace=True)
            if self.use_alignment:
                alignment_result = pd.merge(psi_result, usecols_data, on=key_col_name)
            else:
                alignment_result = psi_result
            alignment_result.to_csv(alignment_output_file, index=False, header=True)
            log.info(f"alignment_result shape: {alignment_result.shape}")
        else:
            use_cols = list(usecols_data.columns)
            log.info(f"psi_result is Empty, only have Column name: {use_cols}")
            with open(alignment_output_file, 'w') as output_f:
                output_f.write(','.join(use_cols)+"\n")

    def _create_set_channel(self):
        '''
        create and set channel.
        '''
        log.info("start create iohandler.")
        iohandler = psi.IOHandler()
        self.io_channel = io.APIManager()
        log.info("start create channel.")
        channel = self.io_channel.create_channel(self.party_id, self.channel_config)
        log.info("start set channel.")
        iohandler.set_channel("", channel)
        log.info("set channel success.")
    
    def _get_temp_dir(self):
        '''
        Get the directory for temporarily saving files
        '''
        temp_dir = os.path.join(self.results_dir, 'temp')
        self._mkdir(temp_dir)
        return temp_dir
  
    def _remove_temp_dir(self):
        if self.party_id in self.result_party:
            # only delete the temp dir
            temp_dir = self.temp_dir
        else:
            # delete the all results in the non-result party.
            temp_dir = self.results_dir
        self._remove_dir(temp_dir)
    
    def _mkdir(self, _directory):
        if not os.path.exists(_directory):
            os.makedirs(_directory, exist_ok=True)

    def _remove_dir(self, _directory):
        if os.path.exists(_directory):
            shutil.rmtree(_directory)

    def send_data_to_other_party(self, remote_partyid, input_data_path):
        data = self.read_content(input_data_path, text=True)
        self.send_sth(remote_partyid, data)
    
    def recv_data_from_other_party(self, remote_partyid, output_data_path):
        _, data = self.recv_sth(remote_partyid)
        if data is None:
            raise ValueError(f'download data failed, recv nothing')
        self.write_content(output_data_path, data)
    
    def len_str(self, dat_len: int) -> str:
        """return hex string of len of data for transmission, always 8 chars"""
        lb = dat_len.to_bytes(4, byteorder='big')
        return codecs.encode(lb, 'hex').decode()

    def recv_sth(self, remote_nodeid):
        recv_data = self.io_channel.Recv(remote_nodeid, 8)
        if recv_data == '\x00'*8:
            return remote_nodeid, None
        data_len = int(recv_data, 16)
        recv_data = self.io_channel.Recv(remote_nodeid, data_len)
        h = ''.join('{:02x}'.format(ord(c)) for c in recv_data)
        recv_data = bytes.fromhex(h)
        log.info(f'recv {data_len} bytes data from {remote_nodeid}, in fact len: {len(recv_data)} bytes')
        return remote_nodeid, recv_data

    def send_sth(self, remote_nodeid, data: str) -> None:
        lens = self.len_str(len(data))
        self.io_channel.Send(remote_nodeid, lens)
        self.io_channel.Send(remote_nodeid, data)
        log.info(f'send {int(lens, 16)} bytes data to {remote_nodeid}, in fact len: {len(data)} bytes')
    
    def read_content(self, path, text=False):
        flag = 'r' if text else 'rb'
        with open(path, flag) as f:
            content = f.read()
        return content

    def write_content(self, path: str, data: bytes):
        with open(path, 'wb') as f:
            f.write(data)


def main(channel_config: str, cfg_dict: dict, data_party: list, compute_party: list, result_party: list, results_dir: str, **kwargs):
    '''
    This is the entrance to this module
    '''
    algo_type = "psi"
    try:
        log.info(f"start main function. {algo_type}.")
        psi = PrivateSetIntersection(channel_config, cfg_dict, data_party, compute_party, result_party, results_dir)
        result_path, result_type = psi.run()
        log.info(f"finish main function. {algo_type}.")
        return result_path, result_type
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
