from email import policy
import json
import logging
import multiprocessing as mp
import threading
from collections import namedtuple
from common.task import Task
from lib.api import sys_rpc_api_pb2 as pb2


log = logging.getLogger(__name__)
TParty = namedtuple('TParty', ['ip', 'port', 'party_id', 'name'])

class TaskManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = {}  # (task_id, party_id) => task
        self.procs = {}  # (task_id, party_id) => task

    def start(self, req):
        '''
        message Party {
            string ip = 1;
            int32 port = 2;
            string party_id = 3;
            string name = 4;
        }
        enum AlgorithmCfgType {
            AlgorithmCfgType_Unknown = 0;
            AlgorithmCfgType_2DTable = 1;
            AlgorithmCfgType_non2DTable = 2;
        }
        enum ConnectPolicyFormat {
            ConnectPolicyFormat_Unknown = 0;
            ConnectPolicyFormat_Str = 1;
            ConnectPolicyFormat_Json = 2;
        }
        message TaskReadyGoReq {
            string task_id = 1;
            string party_id = 2;
            string data_id = 3;
            string env_id = 4;
            repeated Party parties = 5;
            string algorithm_code = 6;
            AlgorithmCfgType algorithm_cfg_type = 7;
            string algorithm_cfg = 8;
            repeated string data_party_ids = 9;
            repeated string computation_party_ids = 10;
            repeated string result_party_ids = 11;
            uint64 duration = 12;
            uint64 memory = 13;
            uint32 processor = 14;
            uint64 bandwidth = 15;
            ConnectPolicyFormat connect_policy_format = 16;
            string connect_policy = 17;
        }
        '''
        log.info("#################### Get a task request. the request information is as follows.")
        log.info(f"task_id: {req.task_id}, party_id: {req.party_id}")
        log.info(f"algorithm_cfg: {req.algorithm_cfg}")
        str_parties = str(req.parties).replace('\n', ' ')
        log.info(f"parties: {str_parties}")
        log.info(f"data_party: {req.data_party}, computation_party: {req.computation_party}, result_party: {req.result_party}")
        task_id = req.task_id
        party_id = req.party_id
        uniq_task = (task_id, party_id)
        task_name = f'{task_id[:15]}-{party_id}'
        if uniq_task in self.tasks:
            log.info(f'task: {task_name} repetitive submit')
            return False, f'task: {task_name} repetitive submit'
        algorithm_code = req.algorithm_code
        data_id = req.data_id
        env_id = req.env_id
        algorithm_cfg = req.algorithm_cfg
        data_party = tuple(req.data_party_ids)
        computation_party = tuple(req.computation_party_ids)
        result_party = tuple(req.result_party_ids)
        duration = req.duration
        limit_memory = req.memory
        limit_cpu  = req.processor
        limit_bandwidth = req.bandwidth
        connect_policy_format = req.connect_policy_format
        if connect_policy_format == pb2.ConnectPolicyFormat_Json:
            connect_policy = json.loads(req.connect_policy)
        elif connect_policy_format == pb2.ConnectPolicyFormat_Str:
            connect_policy = req.connect_policy
        else:
            raise Exception('Unknown connect_policy_format. only support str/json.')
        
        parties = tuple(TParty(p.ip, p.port, p.party_id, p.name) for p in req.parties)
        task = Task(self.cfg, task_id, party_id, algorithm_code, data_id, env_id, parties, 
                    algorithm_cfg, data_party, computation_party, result_party, duration, 
                    limit_memory, limit_cpu, limit_bandwidth, connect_policy)
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
        if (not p._closed) and p.is_alive():
            p.kill()
            log.info(f'wait {task_name} terminate')
            p.join()
            msg = 'will soon' if p.is_alive() else 'succ'
            p.close()
            self.tasks.pop(uniq_task, None)
            self.procs.pop(uniq_task, None)
        else:
            msg = 'succ'
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
