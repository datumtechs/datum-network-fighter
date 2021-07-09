
import hashlib
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import psutil
from protos import compute_svc_pb2, via_svc_pb2

from config import cfg

log = logging.getLogger(__name__)


class Task:
    def __init__(self, task_id, party_id, contract_id, data_id, env_id, peers,
                 user_cfg, data_party, computation_party, result_party):
        log.info(f'thread id: {threading.get_ident()}')
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
        self.user_cfg = user_cfg
        self.data_party = data_party
        self.computation_party = computation_party
        self.result_party = result_party
        m = mp.Manager()
        self.bufs = {}
        for p in peers:
            self.bufs[p.party] = m.Queue()

    @property
    def id(self):
        return self.id_

    @property
    def party_id(self):
        return self._party_id

    def run(self):
        log.info(f'thread id: {threading.get_ident()}')
        log.info(cfg)
        self.download_algo()
        self.build_env()

        log.info(f'cwd:{os.getcwd()}')
        the_dir = os.path.dirname(__file__)
        pdir = os.path.dirname(the_dir)

        import sys
        sys.path.append(os.path.join(pdir, 'common'))
        sys.path.append(os.path.join(pdir, 'protos'))
        sys.path.append(os.path.join(pdir, 'third_party'))
        sys.path.append(os.path.join(pdir, 'via_svc'))

        from common import net_io
        try:
            import importlib
            import sys

            peers = {p.party: f'{p.ip}:{p.port}' for p in self.peers}
            
            pass_via =  cfg['pass_via']
            pproc_ip =  cfg['bind_ip']

            net_io.rtt_set_channel(self.id, self.party_id, peers,
                                   self.data_party, self.computation_party, self.result_party, pass_via, pproc_ip)

            user_cfg = self.assemble_cfg()
            sys.path.append(os.path.abspath(self._get_code_dir()))
            module_name = os.path.splitext(self._get_code_file_name())[0]
            log.info(module_name)
            m = importlib.import_module(module_name)
            m.main(user_cfg)
        except Exception as e:
            log.error(repr(e))
        finally:
            self.clean()

    def get_elapsed_time(self):
        now = time.time()
        return now - self.start_time

    def put_data(self, party_id, data):
        try:
            self.bufs[party_id].put(data, True, 10)
        except queue.Full as e:
            return False
        return True

    def get_data(self, party_id):
        self.bufs[party_id].get()

    def get_stdout(self, task_id):
        pass

    def download_algo(self):
        code = """
import numpy as np
def main(cfg):
    print(np.random.randint(10, size=(2, 3)))
"""
        code_path = os.path.join(self._get_code_dir(),
                                 self._get_code_file_name())
        log.info(code_path)
        with open(code_path, 'w') as f:
            f.write(code)
            f.write('\n')

    def build_env(self):
        log.info(f'build env {self.env_id}')

    def assemble_cfg(self):
        cfg = {p.name: p.party for i, p in enumerate(self.peers)}
        return cfg

    def _get_code_dir(self):
        return cfg['code_root_dir']

    def _get_code_file_name(self):
        name = hashlib.sha1(self.contract_id.encode()).hexdigest()[:6]
        return f'C0DE_{name}.py'

    def clean(self):
        code_path = os.path.join(self._get_code_dir(),
                                 self._get_code_file_name())
        if os.path.exists(code_path):
            os.remove(code_path)
