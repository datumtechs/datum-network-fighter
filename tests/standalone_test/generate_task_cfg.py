import os
import sys
import json
import argparse
import pandas as pd


scripts_path = os.path.split(os.path.realpath(__file__))[0]
base_path = os.path.join(scripts_path, '../..')
def join_base_path(file_path):
    abs_path = os.path.join(base_path, file_path)
    return abs_path

parser = argparse.ArgumentParser()
parser.add_argument('--algo_type', type=str, default='logistic_regression')
args = parser.parse_args()
algo_type = args.algo_type
if algo_type == "logistic_regression":
    data_file_partyA = join_base_path("tests/test_data/binary_class/breast_cancel_partyA_min.csv")
    data_file_partyB = join_base_path("tests/test_data/binary_class/breast_cancel_partyB_min.csv")
    key_column = "id"
    label_column = "diagnosis"
    train_algorithm_file = join_base_path("algorithms/logistic_regression/logistic_reg_train.py")
    predict_algorithm_file = join_base_path("algorithms/logistic_regression/logistic_reg_predict.py")
    train_cfg_file_name = join_base_path("console/task_cfg_lr_train.json")
    predict_cfg_file_name = join_base_path("console/task_cfg_lr_predict.json")
    epochs = 10
    batch_size = 256
    learning_rate = 0.1
    model_restore_party = "p3"
    model_path = join_base_path("data_svc/results_root/abc/p3")  # the path to model
elif algo_type == "linear_regression":
    data_file_partyA = join_base_path("tests/test_data/regression/CarPricing_partyA_min.csv")
    data_file_partyB = join_base_path("tests/test_data/regression/CarPricing_partyB_min.csv")
    key_column = "Car_ID"
    label_column = "price"
    train_algorithm_file = join_base_path("algorithms/linear_regression/linear_reg_train.py")
    predict_algorithm_file = join_base_path("algorithms/linear_regression/linear_reg_predict.py")
    train_cfg_file_name = join_base_path("console/task_cfg_linr_train.json")
    predict_cfg_file_name = join_base_path("console/task_cfg_linr_predict.json")
    epochs = 50
    batch_size = 256
    learning_rate = 0.1
    model_restore_party = "p3"
    model_path = join_base_path("data_svc/results_root/abc/p3")
else:
    raise Exception("only support logistic_regression or linear_regression")

print(f'train_algorithm_file:{train_algorithm_file}')
print(f'predict_algorithm_file:{predict_algorithm_file}')
print(f'data_file_partyA:{data_file_partyA}')
print(f'data_file_partyB:{data_file_partyB}')
print(f'key_column:{key_column}')
print(f'label_column:{label_column}')

cfg_template = join_base_path("console/run_task_cfg_template/task_cfg_template_local.json")
with open(cfg_template, 'r') as f:
    cfg_dict = json.load(f)

'''
common config
'''
cfg_dict["data_id"] = f"{algo_type}_data"
cfg_dict["env_id"]  = f"test_environment"
all_party_list = [party["NODE_ID"] for party in cfg_dict["peers"]]
assert len(all_party_list) == len(cfg_dict["peers"]), "every party's NODE_ID must unique."


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
            selected_columns = pd.read_csv(party_info["data_party"]["input_file"], nrows=0).columns.tolist()
            selected_columns.remove(label_column)
        else:
            party_info["data_party"]["input_file"] = data_file_partyB
            selected_columns = pd.read_csv(party_info["data_party"]["input_file"], nrows=0).columns.tolist()
        selected_columns.remove(key_column)
        print(f'selected_columns:{selected_columns}')
        party_info["data_party"]["key_column"] = key_column
        party_info["data_party"]["selected_columns"] = selected_columns

    cfg_dict["each_party"].append(party_info)
        

with open(train_cfg_file_name, 'w+') as f:
    json.dump(cfg_dict, f, indent=4)


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