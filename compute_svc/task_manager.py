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

from config import cfg
from protos import compute_svc_pb2
from protos import via_svc_pb2

log = logging.getLogger(__name__)
TPeer = namedtuple('TPeer', ['ip', 'port', 'party', 'name'])


class Task:
    def __init__(self, task_id, party_id, contract_id, data_id, env_id, peers):
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

        from common.net_io import create_io
        try:
            import third_party.rosetta_helper as rtt
            import sys
            import importlib
            peers = {p.party: f'{p.ip}:{p.port}' for p in self.peers}
            io_ch = create_io(self, peers)
            rtt.set_io(io_ch)
            if cfg['pass_via']:
                from via_svc.svc import expose_me
                expose_me(cfg, self.id, via_svc_pb2.NET_COMM_SVC)
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
        code_path = os.path.join(self._get_code_dir(), self._get_code_file_name())
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
        code_path = os.path.join(self._get_code_dir(), self._get_code_file_name())
        if os.path.exists(code_path):
            os.remove(code_path)


class TaskManager:
    def __init__(self):
        self.executor = ProcessPoolExecutor(cfg['pool_size'])
        self.tasks = {}
        self.task_future = {}

    def start(self, task_id, party_id, contract_id, data_id, env_id, peers):
        if task_id in self.tasks:
            return False
        peers = tuple(TPeer(p.ip, p.port, p.party, p.name) for p in peers)
        task = Task(task_id, party_id, contract_id, data_id, env_id, peers)
        self.tasks[task_id] = task
        log.info(f'new task: {task.id}, thread id: {threading.get_ident()}')
        future = self.executor.submit(Task.run, task)
        self.task_future[task_id] = future
        log.info(f'task {task_id} done? {future.done()}')

    def cancel(self, task_id):
        future = self.task_future[task_id]
        future.cancel()

    def get_sys_stat(self):
        stat = compute_svc_pb2.GetStatusReply()
        _, _, load15 = psutil.getloadavg()
        stat.cpu = str(load15 / psutil.cpu_count() * 100)
        vm = psutil.virtual_memory()
        stat.mem = str(vm.percent)
        net = psutil.net_io_counters()
        b = net.bytes_sent + net.bytes_recv
        stat.bandwidth = str(b)
        return stat

    def get_task(self, task_id):
        return self.tasks.get(task_id, None)
