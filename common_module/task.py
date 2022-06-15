import os
import sys
import time
import json
import copy
import logging
import threading
import hashlib
import importlib
import warnings
warnings.filterwarnings('ignore', message=r'Passing', category=FutureWarning)
import functools
import pandas as pd
from common_module.consts import DATA_EVENT, COMPUTE_EVENT, COMMON_EVENT
from common_module.event_engine import event_engine
from common_module.report_engine import  report_task_result, monitor_resource_usage, report_task_event
from common_module.io_channel_helper import get_channel_config, IOChannel
from pb.common.constant import carrier_enum_pb2


log = logging.getLogger(__name__)

class Task:
    def __init__(self, cfg, task_id, party_id, env_id, parties, algorithm_code, self_cfg_params, algorithm_dynamic_params,
                 data_party, computation_party, result_party, duration, limit_memory, limit_cpu,
                 limit_bandwidth, connect_policy):
        self.cfg = cfg
        self.id_ = task_id
        self.name = None
        self._party_id = party_id
        self.algorithm_code = algorithm_code
        self.env_id = env_id
        self.progress = None
        self.phase = None
        self.start_time = time.time()
        self.events = []
        self.parties = parties
        self.self_cfg_params = self_cfg_params
        self.algorithm_dynamic_params = algorithm_dynamic_params
        self.data_party = data_party
        self.computation_party = computation_party
        self.result_party = result_party
        self.limit_time = duration/1000  # s
        self.limit_memory = limit_memory
        self.limit_cpu = limit_cpu
        self.limit_bandwidth = limit_bandwidth
        self.connect_policy = connect_policy

        if self._party_id in (self.data_party + self.result_party):
            self.event_type = DATA_EVENT
            self.party_type = "data_svc"
        elif self._party_id in self.computation_party:
            self.event_type = COMPUTE_EVENT
            self.party_type = "compute_svc"
        else:
            raise Exception(f"{self._party_id} is not one of data_party/computation_party/result_party.")
        
        self.create_event = functools.partial(event_engine.create_event, task_id=self.id, party_id=self.party_id)
        self.fire_event = functools.partial(event_engine.fire_event, task_id=self.id, party_id=self.party_id)

    @property
    def id(self):
        return self.id_

    @property
    def party_id(self):
        return self._party_id

    def run(self):
        log.info(f'New task start run, run_cfg: {self.cfg}')
        report_event = functools.partial(report_task_event, self.cfg['schedule_svc'], self.create_event)
        report_event(self.event_type["TASK_START"], "task start.")
        current_task_pid = os.getpid()

        log.info(f'start monitor_resource thread.')
        monitor_resource = threading.Thread(target=monitor_resource_usage, args=(current_task_pid, self.limit_time,
                    self.limit_memory, self.limit_cpu, self.limit_bandwidth,self.cfg['schedule_svc'], self.create_event, self.event_type,
                    self.id, self.party_id, self.party_type, self.cfg['bind_ip'], self.cfg['port'], self.cfg['total_bandwidth']))
        monitor_resource.daemon = True
        monitor_resource.start()
        
        try:
            log.info(f'start download_algo.')
            self.download_algo()
            report_event(self.event_type["DOWNLOAD_ALGORITHM_SUCCESS"], "download algorithm success.")
        except Exception as e:
            log.exception(repr(e))
            report_event(self.event_type["DOWNLOAD_ALGORITHM_FAILED"], f"download algorithm fail. {str(e)[:900]}")
            report_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")
            self.clean()
            return

        log.info(f'start build_env.')
        self.build_env()

        try:
            log.info(f'start get_channel_config.')
            channel_config = get_channel_config(self.id, self.party_id, self.parties,
                            self.data_party, self.computation_party, self.result_party,
                            self.cfg, self.connect_policy,self.limit_time)
            log.info(f'channel_config: {channel_config}.')
            io_channel = IOChannel(self.party_id, channel_config)
            report_event(self.event_type["CREATE_CHANNEL_SUCCESS"], "create channel success.")
        except Exception as e:
            log.exception(repr(e))
            report_event(self.event_type["CREATE_CHANNEL_FAILED"], f"create channel failed. {str(e)[:900]}")
            report_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")
            self.clean()
            return          

        try:
            log.info(f'start get_input_data.')
            new_input_data = self.get_input_data()
            log.info(f'start assemble_cfg.')
            user_cfg = self.assemble_cfg(new_input_data)
            sys.path.insert(0, os.path.abspath(self._get_code_dir()))
            code_path = self._get_code_file_name()
            module_name = os.path.splitext(code_path)[0]
            log.info(f'start importlib.import_module. module_name: {module_name}')
            algo_module = importlib.import_module(module_name)
            log.info(f'finish importlib.import_module.')
            result_dir = self._get_result_dir()
            self._ensure_dir(result_dir)
            log.info(f'start execute algorithm.')
            report_event(self.event_type["ALGORITHM_EXECUTE_START"], "algorithm execute start.")
            algo_return = algo_module.main(io_channel, user_cfg, self.data_party, self.computation_party, self.result_party, result_dir)
            algo_return_len = len(algo_return)
            assert algo_return_len in [2,3], f"algo return must 2 or 3 params, not {algo_return_len}"
            if len(algo_return) == 3:
                result_path, result_type, extra = algo_return
            else:
                result_path, result_type = algo_return
                extra = ""
            assert isinstance(result_path, str), f"result_path must be type(string), not {type(result_path)}"
            assert isinstance(result_type, str), f"result_type must be type(string), not {type(result_type)}"
            assert isinstance(extra, str), f"extra must be type(string), not {type(extra)}"

            log.info(f'finish execute algorithm.')
            if self.party_id in self.result_party:
                assert result_path, f"result_path can not Empty. result_path={result_path}"
                result_path = os.path.abspath(result_path)
                assert os.path.exists(result_path), f"result_path is not exist. result_path={result_path}"
                data_type = map_data_type_to_int(result_type)
                origin_id, data_hash, metadata_option = get_metadata(result_path, data_type)
                file_summary = {"task_id": self.id, "origin_id": origin_id, "ip": self.cfg["bind_ip"], "port": self.cfg["port"],
                                "extra": extra, "data_hash": data_hash, "data_type": data_type, "metadata_option": metadata_option}
                log.info(f'start report task result file summary.')
                report_task_result(self.cfg['schedule_svc'], 'result_file', file_summary)
                log.info(f'finish report task result file summary. ')
            report_event(self.event_type["ALGORITHM_EXECUTE_SUCCESS"], "algorithm execute success.")
            report_event(COMMON_EVENT["END_FLAG_SUCCESS"], "task success.")
        except Exception as e:
            log.exception(repr(e))
            report_event(self.event_type["ALGORITHM_EXECUTE_FAILED"], f"algorithm execute failed. {str(e)[:900]}")
            report_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")
        finally:
            self.clean()

    def get_elapsed_time(self):
        now = time.time()
        return now - self.start_time

    def get_stdout(self, task_id):
        pass

    def download_algo(self):
        dir_ = self._get_code_dir()
        self._ensure_dir(dir_)
        code_path = os.path.join(dir_, self._get_code_file_name())
        log.info(f'code save to: {code_path}')
        with open(code_path, 'w') as f:
            f.write(self.algorithm_code)  # the algorithm_code is code itself at now
            f.write('\n')

    def build_env(self):
        log.info(f'build env {self.env_id}')

    def get_input_data(self):
        self_cfg_params = json.loads(self.self_cfg_params)
        input_data = self_cfg_params["input_data"]
        new_input_data = []
        for data in input_data:
            new_data = {}
            access_type = data["access_type"]
            input_type = data["input_type"]
            data_type = data["data_type"]
            data_path = data["data_path"]
            if access_type == 1:     # 0: unknown, 1: local, 2: url
                new_data["input_type"] = input_type
                new_data["data_type"] = data_type
                new_data["data_path"] = data_path
                if data_type in [1,4,5]:   # 0:unknown, 1:csv, 2:dir, 3:binary, 4:xls, 5:xlsx, 6:txt, 7:json
                    new_data["key_column"] = data.get("key_column")
                    new_data["selected_columns"] = data.get("selected_columns")
            else:
                raise NotImplementedError(f"todo access_type: {access_type}, only support 1.")
            new_input_data.append(new_data)
        return new_input_data
    
    def assemble_cfg(self, input_data):
        self_cfg_params = {"party_id": self.party_id, "input_data": input_data}
        algorithm_dynamic_params = json.loads(self.algorithm_dynamic_params)
        cfg_dict = {"self_cfg_params": self_cfg_params, "algorithm_dynamic_params": algorithm_dynamic_params}
        return cfg_dict

    def _get_code_dir(self):
        return os.path.join(self.cfg['code_root_dir'], self.id, self.party_id)

    def _get_result_dir(self):
        result_dir = os.path.join(self.cfg['results_root_dir'], f"{self.id}/{self.party_id}")
        return os.path.abspath(result_dir)

    def _ensure_dir(self, dir_):
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

    def _get_code_file_name(self):
        # name = hashlib.sha1(self.algorithm_code.encode()).hexdigest()[:6]
        # return f'C0DE_{name}.py'
        return 'main.py'

    def _get_code_cfg_file_name(self):
        return 'config.json'

    def clean(self):
        log.info('task final clean.')
        import shutil
        dir_ = self._get_code_dir()
        if os.path.exists(dir_):
            shutil.rmtree(dir_, ignore_errors=True)
        log.info(f'#################### task finish run, task_id: {self.id}, party_id: {self.party_id}')

def map_data_type_to_int(data_type_str):
    if data_type_str.lower() == 'csv':
        data_type_int = carrier_enum_pb2.OrigindataType_CSV
    elif data_type_str.lower() in ['dir', 'directory']:
        data_type_int = carrier_enum_pb2.OrigindataType_DIR
    elif data_type_str.lower() in ['bin', 'binary']:
        data_type_int = carrier_enum_pb2.OrigindataType_BINARY
    elif data_type_str.lower() == 'xls':
        data_type_int = carrier_enum_pb2.OrigindataType_XLS
    elif data_type_str.lower() == 'xlsx':
        data_type_int = carrier_enum_pb2.OrigindataType_XLSX
    elif data_type_str.lower() == 'txt':
        data_type_int = carrier_enum_pb2.OrigindataType_TXT
    elif data_type_str.lower() == 'json':
        data_type_int = carrier_enum_pb2.OrigindataType_JSON
    else:
        data_type_int = carrier_enum_pb2.OrigindataType_Unknown
    return data_type_int


def get_metadata(path, data_type):
    if data_type == carrier_enum_pb2.OrigindataType_DIR:
        assert os.path.isdir(path), f'{path} is not a directory.'
        origin_id, data_hash, metadata_option = get_directory_metadata(path)
    else:
        assert os.path.isfile(path), f'{path} is not a file.'
        origin_id, data_hash, metadata_option = get_file_metadata(path, data_type)
    metadata_option = json.dumps(metadata_option)
    return origin_id, data_hash, metadata_option

def get_file_metadata(path, data_type):
    m = hashlib.sha256()
    with open(path, 'rb') as f:
        chunk_size = 1024 * 1024 * 1024
        chunk = f.read(chunk_size)
        while chunk:
            m.update(chunk)
            chunk = f.read(chunk_size)
    data_hash = m.hexdigest()
    m.update(path.encode())
    origin_id = m.hexdigest()

    metadata_option = {"originId": origin_id}
    if data_type == carrier_enum_pb2.OrigindataType_CSV:
        metadata_option["dataPath"] = path
        metadata_option["size"] = os.path.getsize(path)
        with open(path, 'r') as f:
            rows = 0
            for line in f:
                rows += 1
        file_data = pd.read_csv(path, nrows=10)
        metadata_option["columns"] = file_data.shape[1]
        metadata_option["hasTitle"] = True
        if metadata_option["hasTitle"]:
            metadata_option["rows"] = rows - 1
            data_columns = list(file_data.columns)
            metadata_columns = []
            for index, col in enumerate(data_columns, 1):
                one_col_metadata = {}
                one_col_metadata["index"] = index
                one_col_metadata["name"] = col
                df_data_type = file_data[col].dtype
                if df_data_type == 'float':
                    col_data_type = 'float'
                elif df_data_type == 'int':
                    col_data_type = 'int'
                elif df_data_type == 'object':
                    col_data_type = 'string'
                else:
                    col_data_type = ''
                one_col_metadata["type"] = col_data_type
                one_col_metadata["size"] = 0
                one_col_metadata["comment"] = ""
                metadata_columns.append(one_col_metadata)
            metadata_option["metadataColumns"] = metadata_columns
        else:
            metadata_option["rows"] = 0
            metadata_option["metadataColumns"] = []
    else:
        metadata_option["dataPath"] = path
        metadata_option["size"] = os.path.getsize(path)

    return origin_id, data_hash, metadata_option

def get_directory_metadata(path):
    # List all subpaths's metadata
    metadata_temp = []
    for current_path, sub_dir, files in os.walk(path):
        current_abspath = os.path.abspath(current_path)
        sha256 = hashlib.sha256()
        sha256.update(current_abspath.encode())
        path_hash = sha256.hexdigest()
        one_dir_info = {}
        one_dir_info["originId"] = path_hash
        one_dir_info["dirPath"] = current_abspath
        one_dir_info["childs"] = []
        one_dir_info["last"] = False if sub_dir else True
        one_dir_info["filePaths"] = []
        for one_file in files:
            one_dir_info["filePaths"].append(os.path.join(current_abspath, one_file))
        metadata_temp.append(one_dir_info)

    # Assemble into the desired message structure
    metadata_option = copy.deepcopy(metadata_temp.pop(0))
    base_path = metadata_option['dirPath']
    base_path_list_len = len(base_path.split('/'))
    while metadata_temp:
        sub_path_list = metadata_temp[0]['dirPath'].split('/')
        sub_dirs = sub_path_list[base_path_list_len:]  # only get the sub directory
        # find the insert point to insert the one_dir_info
        full_dir = base_path
        insert_point = metadata_option["childs"]
        for dir_ in sub_dirs:
            full_dir = os.path.join(full_dir, dir_)
            for postion in insert_point:
                if full_dir == postion['dirPath']:
                    insert_point = postion['childs']
        insert_point.append(metadata_temp.pop(0))
    origin_id = metadata_option['originId']
    data_hash = origin_id
    return origin_id, data_hash, metadata_option
 
