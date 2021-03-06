
import os
import json
import copy
import argparse
import yaml
scripts_path = os.path.split(os.path.realpath(__file__))[0]
base_path = os.path.join(scripts_path, '../../..')
import sys
sys.path.insert(0, base_path)
from common.utils import load_cfg


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, default='config.yaml')
args = parser.parse_args()
cfg_file = args.cfg_file
cfg = load_cfg(cfg_file)
ip = cfg["ip"]
data_svc_port = cfg["data_svc_port"]
compute_svc_port = cfg["compute_svc_port"]
via_svc_port = cfg["via_svc_port"]
schedule_svc_port = cfg["schedule_svc_port"]
user = cfg["user"]
passwd = cfg["passwd"]
svc_list = cfg["svc_list"]


'''
{
    "host": "192.168.0.1",
    "port": 22,
    "user": "root",
    "passwd": "123456",
    "svc_type": "data_svc",
    "rpc_port": 55551,
    "data_dir": "../data_dir",
    "code_dir": "../contract_dir",
    "results_dir": "../result_dir",
    "pass_via": true,
    "via_svc": "192.168.0.1:55553",
    "schedule_svc": "192.168.0.1:55554"
}
'''
nodes_info = []
for host in ip:
    one_node = {}
    one_node["host"] = host
    one_node["port"] = 22
    one_node["user"] = user
    one_node["passwd"] = passwd
    one_node["pass_via"] = True
    one_node["via_svc"] = f"{host}:{via_svc_port}"
    one_node["schedule_svc"] = f"{host}:{schedule_svc_port}"
    for svc_type in svc_list:
        one_node["svc_type"] = svc_type
        port_list = []
        if svc_type == "data_svc":
            port_list = data_svc_port if isinstance(data_svc_port, list) else [data_svc_port]
        elif svc_type == "compute_svc":
            port_list = compute_svc_port if isinstance(compute_svc_port, list) else [compute_svc_port]
        elif svc_type == "tests/schedule_svc":
            port_list = [schedule_svc_port]
        else:
            raise Exception("svc list only support data_svc,compute_svc,tests/schedule_svc.")

        for port in port_list:
            one_node["rpc_port"] = port
            one_node["data_dir"] = f'../data{one_node["rpc_port"]}'
            one_node["code_dir"] = f'../contracts{one_node["rpc_port"]}'
            one_node["results_dir"] = f'../results{one_node["rpc_port"]}'
            nodes_info.append(copy.deepcopy(one_node))

with open("nodes_conf.json", 'w+') as f:
    json.dump(nodes_info, f, indent=4)
     
print("generate nodes conf success.")       
    
