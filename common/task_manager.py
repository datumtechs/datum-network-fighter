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
        self.tasks = {}  # (task_id, party_id) => task
        self.procs = {}  # (task_id, party_id) => task

    def start(self, req):
        log.info(f"task Manager start: task: {req.task_id}, party_id: {req.party_id}")
        log.info(f"task request: {req}")
        task_id = req.task_id
        party_id = req.party_id
        uniq_task = (task_id, party_id)
        task_name = f'{task_id[:15]}-{party_id}'
        if uniq_task in self.tasks:
            return False, f'task: {task_name} repetitive submit'
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
        self.tasks[uniq_task] = task
        log.info(f'new task: {task_name}, thread id: {threading.get_ident()}')
        p = mp.Process(target=Task.run, args=(task,), name=task_name)
        self.procs[uniq_task] = p
        p.start()
        return True, f'submit task {task_name}'

    def get_task(self, task_id):
        raise NotImplementedError('deprecated, to remove')

    def cancel_task(self, task_id, party_id):
        uniq_task = (task_id, party_id)
        task_name = f'{task_id[:15]}-{party_id}'
        if uniq_task not in self.procs:
            return False, f'process for task {task_name} not found'
        p = self.procs[uniq_task]
        p.kill()
        log.info(f'wait {task_name} terminate')
        p.join()
        msg = 'will soon' if p.is_alive() else 'succ'
        return True, f'cancel task {task_name} {msg}'

    def clean(self):
        exited = []
        for uniq_task, p in self.procs.items():
            if p.exitcode is not None:
                exited.append(uniq_task)
        # log.info(f'detect {len(exited)} out of {len(self.procs)} tasks has terminated')
        for uniq_task in exited:
            self.tasks.pop(uniq_task, None)
            self.procs.pop(uniq_task, None)
