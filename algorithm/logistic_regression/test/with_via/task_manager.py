import hashlib
import logging
import multiprocessing as mp
import os
import json
import queue
import threading
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import psutil

from config import cfg
from protos import compute_svc_pb2
from protos import via_svc_pb2
import latticex.rosetta as rtt
import io_channel

log = logging.getLogger(__name__)
TPeer = namedtuple('TPeer', ['ip', 'port', 'party', 'name'])


class Task:
    def __init__(self, task_id, party_id, contract_id, data_id, env_id, peers):
        log.info(f'thread id: {threading.get_ident()}')
        self.id_ = task_id
        self.name = None
        self._party_id = party_id
        self.contract_id = contract_id
        self.node_id = data_id
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

        try:
            import third_party.rosetta_helper as rtt
            import sys
            import importlib
            self.rtt_set_channel()
            # 注册服务
            if cfg['pass_via']:
                from via_svc.svc import expose_me
                expose_me(cfg, self.id_, via_svc_pb2.NET_COMM_SVC, self.node_id)

            user_cfg = self.assemble_cfg()
            sys.path.append(os.path.abspath(self._get_code_dir()))
            module_name = os.path.splitext(self._get_code_file_name())[0]
            log.info(module_name)
            m = importlib.import_module(module_name)
            m.main(user_cfg)
        except Exception as e:
            log.error(repr(e))
        # finally:
        #     self.clean()

    def rtt_set_channel(self):
        with open('rtt_config_with_via', 'r') as load_f:
            strJson = load_f.read()
        config_dict = json.loads(strJson)

        list_node_info = []
        for node_info in self.peers:
            one_node_info = dict(
                NODE_ID = node_info["party"],
                NAME = node_info["name"],
                ADDRESS = node_info["ip"]+str(node_info["port"]),
                VIA = "VIA1"
            )
            list_node_info.append(one_node_info)
        config_dict["NODE_INFO"] = list_node_info

        rtt_config = json.dumps(config_dict)

        def get_current_via_address(current_node_id):
            list_node_info = config_dict["NODE_INFO"]
            address = ""
            via = ""
            for node_info in list_node_info:
                nodeid = node_info["NODE_ID"]
                if nodeid == current_node_id:
                    address = node_info["ADDRESS"]
                    via = node_info["VIA"]
                    break

            if "" == via:
                return "", ""

            via_info = config_dict["VIA_INFO"]
            via_address = via_info[via]

            return address, via_address

        def create_channel(node_id, rtt_config: str):
            def error_callback(a, b, c, d, e):
                print("nodeid:{}, id:{}, errno:{}, error_msg:{}, ext_data:{}".format(a, b, c, d, e))

            # 启动服务
            res = io_channel.create_channel(node_id, rtt_config, error_callback)
            return res

        # 注册到via========================================
        # 获取via地址
        address, via_address = get_current_via_address(self.node_id)
        print("========current address:{}, current via address:{}".format(address, via_address))

        arr_ = address.split(':')
        ip = arr_[0]
        port = arr_[1]
        cfg['via_svc'] = via_address
        cfg['bind_ip'] = ip
        cfg['port'] = port
        print("current server ip:{}, port:{}".format(ip, port))

        channel = create_channel(self.node_id, rtt_config)
        rtt.set_channel(channel)
        print("set channel succeed==================")

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
        pass

    def build_env(self):
        log.info(f'build env {self.env_id}')

    def assemble_cfg(self):
        '''收集参与方的参数'''

        with open('lr_config.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        role_cfg = cfg_dict["user_cfg"]["role_cfg"]
        role_cfg["party_id"] = self.party_id
        role_cfg["input_file"] = ""
        if self.party_id == 0:
            role_cfg["input_file"] = f"../data/bank_train_data.csv"
        elif self.party_id == 1:
            role_cfg["input_file"] = f"../data/insurance_train_data.csv"
        role_cfg["output_file"] = f"../output/p{self.party_id}/my_result"
        if self.party_id != 0:
            role_cfg["with_label"] = False
            role_cfg["label_column_name"] = ""
        else:
            role_cfg["with_label"] = True
            role_cfg["label_column_name"] = "Y"

        return cfg_dict

    def _get_code_dir(self):
        return cfg['code_root_dir']

    def _get_code_file_name(self):
        # name = hashlib.sha1(self.contract_id.encode()).hexdigest()[:6]
        task_type = "lr_train"
        if task_type == "lr_train":
            return 'lr_train.py'
        elif task_type == "lr_predict":
            return 'lr_predict.py'

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
