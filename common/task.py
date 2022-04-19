import os
import sys
import time
import logging
import threading
import hashlib
import importlib
import warnings
warnings.filterwarnings('ignore', message=r'Passing', category=FutureWarning)
import functools

from common.consts import DATA_EVENT, COMPUTE_EVENT, COMMON_EVENT
from common.utils import decompose
from common.event_engine import event_engine
from common.report_engine import  report_task_result, monitor_resource_usage
from common.io_channel_helper import get_channel_config


log = logging.getLogger(__name__)

class Task:
    def __init__(self, cfg, task_id, party_id, contract_id, data_id, env_id, peers, contract_cfg,
                 data_party, computation_party, result_party, duration, limit_memory, limit_cpu,limit_bandwidth):
        log.info(f'thread id: {threading.get_ident()}')
        self.cfg = cfg
        self.id_ = task_id
        self.name = None
        self._party_id = party_id
        self.contract_id = contract_id
        self._main_file = None
        self.data_id = data_id
        self.env_id = env_id
        self.progress = None
        self.phase = None
        self.start_time = time.time()
        self.events = []
        self.peers = peers
        self.contract_cfg = contract_cfg
        self.data_party = data_party
        self.computation_party = computation_party
        self.result_party = result_party
        self.limit_time = duration/1000  # s
        self.limit_memory = limit_memory
        self.limit_cpu = limit_cpu
        self.limit_bandwidth = limit_bandwidth

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
        log.info(f'New task start run, task_id: {self.id}, party_id: {self.party_id}')
        log.info(f'thread id: {threading.get_ident()}')
        log.info(f'run_cfg: {self.cfg}')
        self.fire_event(self.event_type["TASK_START"], "task start.")
        current_task_pid = os.getpid()

        monitor_resource = threading.Thread(target=monitor_resource_usage, args=(current_task_pid, self.limit_time,
                    self.limit_memory, self.limit_cpu, self.limit_bandwidth,self.cfg['schedule_svc'], self.create_event, self.event_type,
                    self.id, self.party_id, self.party_type, self.cfg['bind_ip'], self.cfg['port'], self.cfg['total_bandwidth']))
        monitor_resource.daemon = True
        monitor_resource.start()
        
        try:
            self.download_algo()
            self.fire_event(self.event_type["DOWNLOAD_CONTRACT_SUCCESS"], "download contract success.")
        except Exception as e:
            log.exception(repr(e))
            self.fire_event(self.event_type["DOWNLOAD_CONTRACT_FAILED"], f"download contract fail.")
            self.fire_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")

        self.build_env()

        the_dir = os.path.dirname(__file__)
        pdir = os.path.dirname(the_dir)
        log.info(f'cwd: {os.getcwd()}, the dir: {the_dir}, parent dir: {pdir}')
        cwd_bak = os.getcwd()

        try:
            channel_config = get_channel_config(self.id, self.party_id, self.peers,
                            self.data_party, self.computation_party, self.result_party,
                            self.cfg, self.event_type)
         
            user_cfg = self.assemble_cfg()
            sys.path.insert(0, os.path.abspath(self._get_code_dir()))
            code_path = self.main_file
            module_name = os.path.splitext(code_path)[0]
            log.info(f'code path: {code_path}, module: {module_name}')

            m = importlib.import_module(module_name)
            result_dir = self._get_result_dir()
            data_dir = self._get_data_dir()
            self._ensure_dir(result_dir)
            self.fire_event(self.event_type["CONTRACT_EXECUTE_START"], "contract execute start.")
            os.chdir(self._get_code_dir())
            m.main(channel_config, user_cfg, self.data_party, self.result_party, result_dir, data_dir=data_dir)
            log.info(f'run task done')
            if self.party_id in self.result_party:
                file_path = result_dir
                m = hashlib.sha256()
                m.update(file_path.encode())
                data_id = m.hexdigest()
                file_summary = {"task_id": self.id, "origin_id": data_id, "file_path": file_path,
                                "ip": self.cfg["bind_ip"], "port": self.cfg["port"]}
                log.info(f'start report task result file summary.')
                report_task_result(self.cfg['schedule_svc'], 'result_file', file_summary)
                log.info(f'finish report task result file summary. ')
            self.fire_event(self.event_type["CONTRACT_EXECUTE_SUCCESS"], "contract execute success.")
            self.fire_event(COMMON_EVENT["END_FLAG_SUCCESS"], "task success.")
        except Exception as e:
            log.exception(repr(e))
            self.fire_event(self.event_type["CONTRACT_EXECUTE_FAILED"], f"contract execute failed.")
            self.fire_event(COMMON_EVENT["END_FLAG_FAILED"], "task fail.")
        finally:
            log.info('task final clean')
            self.clean()
            os.chdir(cwd_bak)
            log.info(f'#################### task finish run, task_id: {self.id}, party_id: {self.party_id}')

    def get_elapsed_time(self):
        now = time.time()
        return now - self.start_time

    def get_stdout(self, task_id):
        pass

    def download_algo(self):
        dir_ = self._get_code_dir()
        log.info(f'code save to: {dir_}')
        files = decompose(self.contract_id, dir_)  # the contract_id is code itself at now, may include more than one file
        self.main_file = files[0]

    def build_env(self):
        log.info(f'build env {self.env_id}')

    def assemble_cfg(self):
        import json
        cfg_dict = json.loads(self.contract_cfg)
        return cfg_dict

    def _get_code_dir(self):
        return os.path.join(self.cfg['code_root_dir'], self.id, self.party_id)

    def _get_data_dir(self):
        data_root = '.'
        if 'data_root' in self.cfg:
            data_root = self.cfg['data_root']
        return os.path.join(data_root, self.id, self.party_id)

    def _get_result_dir(self):
        result_dir = os.path.join(self.cfg['results_root_dir'], f"{self.id}/{self.party_id}")
        return os.path.abspath(result_dir)

    def _ensure_dir(self, dir_):
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

    @property
    def main_file(self):
        return self._main_file

    @main_file.setter
    def main_file(self, value):
        self._main_file = value

    def _get_code_cfg_file_name(self):
        return 'config.json'

    def clean(self):
        import shutil
        dir_ = self._get_code_dir()
        shutil.rmtree(dir_, ignore_errors=True)
