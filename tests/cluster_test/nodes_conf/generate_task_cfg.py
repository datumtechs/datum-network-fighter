import os
import sys
import json
import copy
import argparse
import pandas as pd
scripts_path = os.path.split(os.path.realpath(__file__))[0]
base_path = os.path.join(scripts_path, '../../..')
import sys
sys.path.insert(0, base_path)
from common.utils import load_cfg


def join_base_path(file_path, base_path=base_path):
    abs_path = os.path.join(base_path, file_path)
    return abs_path

parser = argparse.ArgumentParser()
parser.add_argument('--algo_type', type=str, default='logistic_regression')
args = parser.parse_args()
algo_type = args.algo_type
# the relative path is based on data_svc or compute_svc directory
if algo_type == "logistic_regression":
    data_file_partyA = "../tests/test_data/binary_class/breast_cancel_partyA_min.csv"
    data_file_partyB = "../tests/test_data/binary_class/breast_cancel_partyB_min.csv"
    key_column = "id"
    label_column = "diagnosis"
    train_algorithm_file = "../algorithms/logistic_regression/logistic_reg_train.py"
    predict_algorithm_file = "../algorithms/logistic_regression/logistic_reg_predict.py"
    train_cfg_file_name = join_base_path("console/task_cfg_lr_train_cluster.json")
    predict_cfg_file_name = join_base_path("console/task_cfg_lr_predict_cluster.json")
    epochs = 10
    batch_size = 256
    learning_rate = 0.1
    model_restore_party = "p3"
    model_path = "../data_svc/results_root/abc/p3"  # the path to model
elif algo_type == "linear_regression":
    data_file_partyA = "../tests/test_data/regression/CarPricing_partyA_min.csv"
    data_file_partyB = "../tests/test_data/regression/CarPricing_partyB_min.csv"
    key_column = "Car_ID"
    label_column = "price"
    train_algorithm_file = "../algorithms/linear_regression/linear_reg_train.py"
    predict_algorithm_file = "../algorithms/linear_regression/linear_reg_predict.py"
    train_cfg_file_name = join_base_path("console/task_cfg_linr_train_cluster.json")
    predict_cfg_file_name = join_base_path("console/task_cfg_linr_predict_cluster.json")
    epochs = 50
    batch_size = 256
    learning_rate = 0.1
    model_restore_party = "p3"
    model_path = "../data_svc/results_root/abc/p3"
else:
    raise Exception("only support logistic_regression or linear_regression")

print(f'train_algorithm_file:{train_algorithm_file}')
print(f'predict_algorithm_file:{predict_algorithm_file}')
print(f'data_file_partyA:{data_file_partyA}')
print(f'data_file_partyB:{data_file_partyB}')
print(f'key_column:{key_column}')
print(f'label_column:{label_column}')

cfg_template = join_base_path("console/run_task_cfg_template/task_cfg_template_cluster.json")
with open(cfg_template, 'r') as f:
    cfg_dict = json.load(f)

'''
common config
'''
cfg_dict["data_id"] = f"{algo_type}_data"
cfg_dict["env_id"]  = f"test_environment"
cfg_dict["data_party"] = ["p1", "p2"],
cfg_dict["computation_party"] = ["p4", "p5", "p6"],
cfg_dict["result_party"] = ["p3"],

all_party_list = set(cfg_dict["data_party"] + cfg_dict["computation_party"] + cfg_dict["result_party"])
assert len(all_party_list) == (len(cfg_dict["data_party"]) + len(cfg_dict["computation_party"]) + len(cfg_dict["result_party"])),\
       "every party's NODE_ID must unique, can not be the same"
node_cfg = load_cfg(os.path.join(scripts_path, "config.yaml"))
peers = []
for i,party_id in enumerate(cfg_dict["data_party"]):
    one_node = {}
    one_node["NODE_ID"] = party_id
    one_node["ADDRESS"] = f'{node_cfg["ip"][i]}:{node_cfg["via_svc_port"]}'
    one_node["INTERNAL"] = f'{node_cfg["ip"][i]}:{node_cfg["data_svc_port"]}'
    peers.append(copy.deepcopy(one_node))

for i,party_id in enumerate(cfg_dict["computation_party"]):
    one_node = {}
    one_node["NODE_ID"] = party_id
    one_node["ADDRESS"] = f'{node_cfg["ip"][i]}:{node_cfg["via_svc_port"]}'
    one_node["INTERNAL"] = f'{node_cfg["ip"][i]}:{node_cfg["compute_svc_port"]}'
    peers.append(copy.deepcopy(one_node))

for i,party_id in enumerate(cfg_dict["result_party"]):
    one_node = {}
    one_node["NODE_ID"] = party_id
    one_node["ADDRESS"] = f'{node_cfg["ip"][i]}:{node_cfg["via_svc_port"]}'
    one_node["INTERNAL"] = f'{node_cfg["ip"][i]}:{node_cfg["data_svc_port"]}'
    peers.append(copy.deepcopy(one_node))

cfg_dict["peers"] = peers

'''
train task config
'''
cfg_dict["contract_id"] = train_algorithm_file
cfg_dict["dynamic_parameter"] = {
    "label_owner": cfg_dict["data_party"][0],  # default: the first data_party own label
    "label_column": label_column,
    "algorithm_parameter": {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }
}

cfg_dict["each_party"] = []
for party_id in all_party_list:
    party_info = {"party_id": party_id, 
                  "data_party": {
                        "input_file": "",
                        "key_column": "",
                        "selected_columns": []
                    }
                }
    label_owner = cfg_dict["dynamic_parameter"]["label_owner"]
    if party_id in cfg_dict["data_party"]:
        if party_id == label_owner:
            party_info["data_party"]["input_file"] = data_file_partyA
            input_file = join_base_path('data_svc/' + data_file_partyA)
            selected_columns = pd.read_csv(input_file, nrows=0).columns.tolist()
            selected_columns.remove(label_column)
        else:
            party_info["data_party"]["input_file"] = data_file_partyB
            input_file = join_base_path('data_svc/' + data_file_partyB)
            selected_columns = pd.read_csv(input_file, nrows=0).columns.tolist()
        selected_columns.remove(key_column)
        print(f'selected_columns:{selected_columns}')
        party_info["data_party"]["key_column"] = key_column
        party_info["data_party"]["selected_columns"] = selected_columns

    cfg_dict["each_party"].append(party_info)
        

with open(train_cfg_file_name, 'w+') as f:
    json.dump(cfg_dict, f, indent=4)
print(f"write to {train_cfg_file_name} success.")

'''
predict task config
'''
cfg_dict["contract_id"] = predict_algorithm_file
cfg_dict["dynamic_parameter"] = {
    "model_restore_party": model_restore_party, 
    "model_path": model_path
}
model_restore_party = cfg_dict["dynamic_parameter"]["model_restore_party"]
if model_restore_party not in cfg_dict["data_party"]:
    cfg_dict["data_party"].append(model_restore_party)

with open(predict_cfg_file_name, 'w+') as f:
    json.dump(cfg_dict, f, indent=4)
print(f"write to {predict_cfg_file_name} success.")