import logging
import threading
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from task import Task

log = logging.getLogger(__name__)
TPeer = namedtuple('TPeer', ['ip', 'port', 'party', 'name'])


class TaskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.executor = ProcessPoolExecutor(cfg['thread_pool_size'])
        self.tasks = {}
        self.task_future = {}        

    def start(self, req):
        task_id = req.task_id
        if task_id in self.tasks:
            return False, f'task: {task_id} repetitive submit'
        party_id = req.party_id
        contract_id = req.contract_id
        data_id = req.data_id
        env_id = req.env_id
        peers = req.peers
        contract_cfg = req.contract_cfg
        data_party = req.data_party
        computation_party = req.computation_party
        result_party = req.result_party

        peers = tuple(TPeer(p.ip, p.port, p.party, p.name) for p in peers)
        task = Task(task_id, self.cfg, party_id, contract_id, data_id, env_id, peers,
                    contract_cfg, data_party, computation_party, result_party)
        self.tasks[task_id] = task
        log.info(f'new task: {task.id}, thread id: {threading.get_ident()}')
        future = self.executor.submit(Task.run, task)
        self.task_future[task_id] = future
        log.info(f'task {task_id} done? {future.done()}')
        # TODO clean info after task finished
        return True, f'submit task {task_id}'

    def cancel(self, task_id):
        future = self.task_future[task_id]
        future.cancel()
        self.erase(task_id)
        
    def erase(self, task_id):
        del self.task_future[task_id]
        del self.tasks[task_id]

    def get_task(self, task_id):
        return self.tasks.get(task_id, None)
