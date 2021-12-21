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
        '''
        message TaskReadyGoReq {
            string task_id = 1;
            string contract_id = 2;
            string data_id = 3;
            string party_id = 4;
            string env_id = 5;
            message Peer {
                string ip = 1;
                int32 port = 2;
                string party_id = 3;
                string name = 4;
            }
            repeated Peer peers = 6;
            string contract_cfg = 7;
            repeated string data_party = 8;
            repeated string computation_party = 9;
            repeated string result_party = 10;
        }
        '''
        log.info("#################### Get a task request. the request information is as follows.")
        log.info(f"task_id: {req.task_id}, party_id: {req.party_id}")
        log.info(f"contract_cfg: {req.contract_cfg}")
        str_peers = str(req.peers).replace('\n', ' ')
        log.info(f"peers: {str_peers}")
        log.info(f"data_party: {req.data_party}, computation_party: {req.computation_party}, result_party: {req.result_party}")
        task_id = req.task_id
        party_id = req.party_id
        uniq_task = (task_id, party_id)
        task_name = f'{task_id[:15]}-{party_id}'
        if uniq_task in self.tasks:
            log.info(f'task: {task_name} repetitive submit')
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
        p = mp.Process(target=Task.run, args=(task,), name=task_name, daemon=True)
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
        p.close()
        return True, f'cancel task {task_name} {msg}'

    def clean(self):
        exited = []
        for uniq_task, p in self.procs.items():
            if p._closed or (p.exitcode is not None):
                # log.info(f"p.exitcode: {p.exitcode}, uniq_task: {uniq_task}")
                exited.append(uniq_task)
        # log.info(f'detect {len(exited)} out of {len(self.procs)} tasks has terminated')
        for uniq_task in exited:
            self.tasks.pop(uniq_task, None)
            self.procs.pop(uniq_task, None)
