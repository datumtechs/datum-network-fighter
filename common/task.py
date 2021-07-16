
import logging
import os
import threading
import time

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

    @property
    def id(self):
        return self.id_

    @property
    def party_id(self):
        return self._party_id

    def run(self):
        log.info(f'thread id: {threading.get_ident()}')
        log.info(self.cfg)
        self.download_algo()
        self.build_env()

        the_dir = os.path.dirname(__file__)
        pdir = os.path.dirname(the_dir)
        log.info(f'cwd: {os.getcwd()}\nthe dir: {the_dir}\nparent dir: {pdir}')

        import sys
        sys.path.append(os.path.join(pdir, 'via_svc'))
        log.info('sys.path:\n{}'.format('\n'.join(sys.path)))
        import warnings
        warnings.filterwarnings('ignore', message=r'Passing', category=FutureWarning)
        from io_channel_helper import rtt_set_channel
        try:
            import importlib

            pass_via = self.cfg['pass_via']
            pproc_ip = self.cfg['bind_ip']

            rtt_set_channel(self.id, self.party_id, self.peers,
                            self.data_party, self.computation_party, self.result_party, pass_via, pproc_ip)

            user_cfg = self.assemble_cfg()
            sys.path.insert(0, os.path.abspath(self._get_code_dir()))
            code_path = self._get_code_file_name()
            module_name = os.path.splitext(code_path)[0]
            log.info(f'code path: {code_path}, module: {module_name}')
            m = importlib.import_module(module_name)
            result_dir = self._get_result_dir()
            self._ensure_dir(result_dir)
            m.main(user_cfg, self.result_party, result_dir)
        except Exception as e:
            log.error(repr(e))
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
        return os.path.join(self.cfg['results_root_dir'], self.id, self.party_id)

    def _ensure_dir(self, dir_):
        if not os.path.exists(dir_):
            os.makedirs(dir_)

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
