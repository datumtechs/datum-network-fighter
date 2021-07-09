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
from task import Task

log = logging.getLogger(__name__)
TPeer = namedtuple('TPeer', ['ip', 'port', 'party', 'name'])


class TaskManager:
    def __init__(self):
        self.executor = ProcessPoolExecutor(cfg['pool_size'])
        self.tasks = {}
        self.task_future = {}

    def start(self, req):
        task_id = req.task_id
        if task_id in self.tasks:
            return False, f'task: {task_id} repetitive submit'
        party_id = req.node_id
        contract_id = req.contract_id
        data_id = req.data_id
        env_id = req.env_id
        peers = req.peers
        user_cfg = req.user_cfg
        data_party = req.data_party
        computation_party = req.computation_party
        result_party = req.result_party

        peers = tuple(TPeer(p.ip, p.port, p.party, p.name) for p in peers)
        task = Task(task_id, party_id, contract_id, data_id, env_id, peers,
                    user_cfg, data_party, computation_party, result_party)
        self.tasks[task_id] = task
        log.info(f'new task: {task.id}, thread id: {threading.get_ident()}')
        future = self.executor.submit(Task.run, task)
        self.task_future[task_id] = future
        log.info(f'task {task_id} done? {future.done()}')
        return True, f'submit task {task_id}'

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
