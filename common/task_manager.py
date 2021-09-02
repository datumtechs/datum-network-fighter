import logging
import multiprocessing as mp
import threading
from collections import namedtuple

from .task import Task

log = logging.getLogger(__name__)
TPeer = namedtuple('TPeer', ['ip', 'port', 'party_id', 'name'])


class TaskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = {}
        self.procs = {}

    def start(self, req):
        task_id = req.task_id
        if task_id in self.tasks:
            return False, f'task: {task_id} repetitive submit'
        party_id = req.party_id
        contract_id = req.contract_id
        data_id = req.data_id
        env_id = req.env_id
        contract_cfg = req.contract_cfg
        data_party = tuple(req.data_party)
        computation_party = tuple(req.computation_party)
        result_party = tuple(req.result_party)
        peers = tuple(TPeer(p.ip, p.port, p.party_id, p.name) for p in req.peers)
        task = Task(self.cfg, task_id, party_id, contract_id, data_id, env_id, peers,
                    contract_cfg, data_party, computation_party, result_party)
        self.tasks[task_id] = task
        log.info(f'new task: {task.id}, thread id: {threading.get_ident()}')
        p = mp.Process(target=Task.run, args=(task,), daemon=True)
        self.procs[task_id] = p
        p.start()
        return True, f'submit task {task_id}'

    def get_task(self, task_id):
        return self.tasks.get(task_id, None)

    def cancel_task(self, task_id):
        if task_id not in self.procs:
            return False, f'process for task {task_id} not found'
        p = self.procs[task_id]
        p.kill()
        log.info(f'wait {task_id} terminate')
        p.join()
        msg = 'will soon' if p.is_alive() else 'succ'
        return True, f'cancel task {task_id} {msg}'

    def clean(self):
        exited = []
        for task_id, p in self.procs.items():
            if p.exitcode is not None:
                exited.append(task_id)
        log.info(f'detect {len(exited)} out of {len(self.procs)} tasks has terminated')
        for task_id in exited:
            self.tasks.pop(task_id, None)
            self.procs.pop(task_id, None)
