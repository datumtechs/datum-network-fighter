import os
import sys
import time
import logging
import threading
import multiprocessing as mp
import hashlib
import importlib
import warnings
warnings.filterwarnings('ignore', message=r'Passing', category=FutureWarning)
import functools

from common.consts import DATA_EVENT, COMPUTE_EVENT, COMMON_EVENT
from common.event_engine import event_engine
from common.report_engine import report_task_resource_usage, report_task_result_file_summary
from io_channel_helper import get_channel_config


log = logging.getLogger(__name__)

class Task:
    def __init__(self, cfg, task_id, party_id, contract_id, data_id, env_id, peers,
                 contract_cfg, data_party, computation_party, result_party):
        log.info(f'thread id: {threading.get_ident()}')
        self.cfg = cfg
        self.id_ = task_id
        self.name = None
        self._party_id = party_id
        self.contract_id = contract_id
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

        if self._party_id in (self.data_party + self.result_party):
            self.event_type = DATA_EVENT
            self.party_type = "data_svc"
        elif self._party_id in self.computation_party:
            self.event_type = COMPUTE_EVENT
            self.party_type = "compute_svc"
        else:
            self.event_type = None
        
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
        report_resource = threading.Thread(target=report_task_resource_usage, args=(current_task_pid, self.cfg['schedule_svc'], 
                    self.id, self.party_id, self.party_type, self.cfg['bind_ip'], self.cfg['port'], self.cfg['total_bandwidth'], 10))
        report_resource.daemon = True
        report_resource.start()
        
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

        try:
            channel_config = get_channel_config(self.id, self.party_id, self.peers,
                            self.data_party, self.computation_party, self.result_party,
                            self.cfg, self.event_type)
         
            user_cfg = self.assemble_cfg()
            sys.path.insert(0, os.path.abspath(self._get_code_dir()))
            code_path = self._get_code_file_name()
            module_name = os.path.splitext(code_path)[0]
            log.info(f'code path: {code_path}, module: {module_name}')

            m = importlib.import_module(module_name)
            result_dir = self._get_result_dir()
            self._ensure_dir(result_dir)
            self.fire_event(self.event_type["CONTRACT_EXECUTE_START"], "contract execute start.")
            m.main(channel_config, user_cfg, self.data_party, self.result_party, result_dir)
            log.info(f'run task done')
            if self.party_id in self.result_party:
                file_path = result_dir
                m = hashlib.sha256()
                m.update(file_path.encode())
                data_id = m.hexdigest()
                file_summary = {"task_id": self.id, "origin_id": data_id, "file_path": file_path,
                                "ip": self.cfg["bind_ip"], "port": self.cfg["port"]}
                log.info(f'start report task result file summary.')
                report_task_result_file_summary(self.cfg['schedule_svc'], file_summary)
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
            f.write(self.contract_id)  # the contract_id is code itself at now
            f.write('\n')

    def build_env(self):
        log.info(f'build env {self.env_id}')

    def assemble_cfg(self):
        import json
        cfg_dict = json.loads(self.contract_cfg)
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
        # name = hashlib.sha1(self.contract_id.encode()).hexdigest()[:6]
        # return f'C0DE_{name}.py'
        return 'main.py'

    def _get_code_cfg_file_name(self):
        return 'config.json'

    def clean(self):
        import shutil
        dir_ = self._get_code_dir()
        shutil.rmtree(dir_, ignore_errors=True)
