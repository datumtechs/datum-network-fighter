import os
import sys
import time
import logging
import threading

from common.consts import DATA_EVENT, COMPUTE_EVENT, COMMON_EVENT
from common.event_engine import event_engine

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
        elif self._party_id in self.computation_party:
            self.event_type = COMPUTE_EVENT
        else:
            self.event_type = None

    @property
    def id(self):
        return self.id_

    @property
    def party_id(self):
        return self._party_id

    def run(self):
        log.info(f'thread id: {threading.get_ident()}')
        log.info(self.cfg)
        event_engine.fire_event(self.event_type["TASK_START"], self.party_id, self.id_, "task start.")
        try:
            self.download_algo()
            event_engine.fire_event(self.event_type["DOWNLOAD_CONTRACT_SUCCESS"], self.party_id, self.id_,
                                    "download contract success.")
        except Exception as e:
            event_engine.fire_event(self.event_type["DOWNLOAD_CONTRACT_FAILED"], self.party_id, self.id_,
                                    f"download contract fail. {str(e)}")
            event_engine.fire_event(COMMON_EVENT["END_FLAG_FAILED"], self.party_id, self.id_, "service stop.")

        self.build_env()

        the_dir = os.path.dirname(__file__)
        pdir = os.path.dirname(the_dir)
        log.info(f'cwd: {os.getcwd()}\nthe dir: {the_dir}\nparent dir: {pdir}')

        import warnings
        warnings.filterwarnings('ignore', message=r'Passing', category=FutureWarning)
        from io_channel_helper import rtt_set_channel
        try:
            import importlib

            pass_via = self.cfg['pass_via']
            pproc_ip = self.cfg['bind_ip']
            certs = self.cfg['certs']

            rtt_set_channel(self.id, self.party_id, self.peers,
                            self.data_party, self.computation_party, self.result_party, 
                            pass_via, pproc_ip, certs, self.event_type)
         
            user_cfg = self.assemble_cfg()
            sys.path.insert(0, os.path.abspath(self._get_code_dir()))
            code_path = self._get_code_file_name()
            module_name = os.path.splitext(code_path)[0]
            log.info(f'code path: {code_path}, module: {module_name}')

            event_engine.fire_event(self.event_type["CONTRACT_EXECUTE_START"], self.party_id, self.id_, "contract execute start.")
            m = importlib.import_module(module_name)
            results_root_dir = self._get_result_dir()
            self._ensure_dir(results_root_dir)
            m.main(self.id, user_cfg, self.data_party, self.result_party, results_root_dir)
            log.info(f'run task done')
            event_engine.fire_event(self.event_type["CONTRACT_EXECUTE_SUCCESS"], self.party_id, self.id_,
                                    "contract execute success.")
            event_engine.fire_event(COMMON_EVENT["END_FLAG_SUCCESS"], self.party_id, self.id_, "task finish.")
        except Exception as e:
            log.exception(repr(e))
            event_engine.fire_event(self.event_type["CONTRACT_EXECUTE_FAILED"], self.party_id, self.id_,
                                    f"contract execute failed. {str(e)}")
            event_engine.fire_event(COMMON_EVENT["END_FLAG_FAILED"], self.party_id, self.id_, "service stop.")
        finally:
            log.info('task final clean')
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
            f.write(self.contract_id)  # the contract_id is code itself at now
            f.write('\n')

    def build_env(self):
        log.info(f'build env {self.env_id}')

    def assemble_cfg(self):
        import json
        cfg_dict = json.loads(self.contract_cfg)
        # dir_ = self._get_code_dir()
        # self._ensure_code_dir(dir_)
        # code_path_path = os.path.join(dir_, self._get_code_cfg_file_name())
        # log.info(code_path_path)
        # with open(code_path_path, 'w') as f:
        #     json.dump(cfg_dict, f)
        return cfg_dict

    def _get_code_dir(self):
        return os.path.join(self.cfg['code_root_dir'], self.id, self.party_id)

    def _get_result_dir(self):
        return os.path.abspath(self.cfg['results_root_dir'])

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
