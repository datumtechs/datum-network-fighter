import os
import sys
import time
import json
import logging
import threading
import hashlib
import importlib
import warnings
warnings.filterwarnings('ignore', message=r'Passing', category=FutureWarning)
import functools

from common.consts import DATA_EVENT, COMPUTE_EVENT, COMMON_EVENT
from common.event_engine import event_engine
from common.report_engine import  report_task_result, monitor_resource_usage, report_task_event
from common.io_channel_helper import get_channel_config


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
            self.event_type = None
        
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
            report_event(self.event_type["DOWNLOAD_CONTRACT_SUCCESS"], "download contract success.")
        except Exception as e:
            log.exception(repr(e))
            report_event(self.event_type["DOWNLOAD_CONTRACT_FAILED"], f"download contract fail.")
            report_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")

        log.info(f'start build_env.')
        self.build_env()

        try:
            log.info(f'start get_channel_config.')
            channel_config = get_channel_config(self.id, self.party_id, self.parties,
                            self.data_party, self.computation_party, self.result_party,
                            self.cfg, self.connect_policy,self.limit_time)
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
            log.info(f'start execute contract.')
            report_event(self.event_type["CONTRACT_EXECUTE_START"], "contract execute start.")
            extra = algo_module.main(channel_config, user_cfg, self.data_party, self.result_party, result_dir)
            if not extra:
                extra = ""
            log.info(f'finish execute contract.')
            if self.party_id in self.result_party:
                data_path = result_dir
                m = hashlib.sha256()
                m.update(data_path.encode())
                origin_id = m.hexdigest()
                file_summary = {"task_id": self.id, "origin_id": origin_id, "data_path": data_path,
                                "ip": self.cfg["bind_ip"], "port": self.cfg["port"], "extra": extra}
                log.info(f'start report task result file summary.')
                report_task_result(self.cfg['schedule_svc'], 'result_file', file_summary)
                log.info(f'finish report task result file summary. ')
            report_event(self.event_type["CONTRACT_EXECUTE_SUCCESS"], "contract execute success.")
            report_event(COMMON_EVENT["END_FLAG_SUCCESS"], "task success.")
        except Exception as e:
            log.exception(repr(e))
            report_event(self.event_type["CONTRACT_EXECUTE_FAILED"], f"contract execute failed.error:{str(e)[:900]}")
            report_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")
        finally:
            log.info('task final clean.')
            self.clean()
            log.info(f'#################### task finish run, task_id: {self.id}, party_id: {self.party_id}')

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
            if access_type == 1:     # 0: unknown, 1: local, 2: http, 3: https, 4: ftp
                new_data["input_type"] = input_type
                new_data["data_type"] = data_type
                new_data["data_path"] = data_path
                if data_type in [1,3,6]:   # 0:unknown, 1:csv, 2:folder, 3:xls, 4:txt, 5:json, 6:mysql, 7:bin
                    new_data["key_column"] = data.get("key_column")
                    new_data["selected_columns"] = data.get("selected_columns")
            else:
                raise NotImplementedError("todo access_type: {access_type}, only support 1.")
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
        import shutil
        dir_ = self._get_code_dir()
        shutil.rmtree(dir_, ignore_errors=True)
