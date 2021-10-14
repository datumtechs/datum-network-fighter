import json
import pandas as pd


config_file = 'run_task_cfg.json'
with open(config_file, 'r') as f:
    cfg_dict = json.load(f)
# print(f"cfg_dict:{cfg_dict}")


###### train task config
cfg_dict["contract_id"] = "../tests/standalone_test/lr_train.py"
cfg_dict["dynamic_parameter"] = {
    "label_owner": "p1",
    "label_column_name": "Y",
    "algorithm_parameter": {
        "epochs": 10,
        "batch_size": 256,
        "learning_rate": 0.1
    }
}

# party0
data_party = cfg_dict["each_party"][0]["data_party"]
data_party["input_file"] = "/home/juzix/work/fighter-py/data/big_data/train_data/bank_train_data_1w.csv"
input_file = data_party["input_file"]
columns = list(pd.read_csv(input_file, nrows=0).columns)
key_column = columns[0]
print(f'key_column:{key_column}')
selected_columns = columns[1:-1]
print(f'selected_columns:{selected_columns}')
data_party["key_column"] = key_column
data_party["selected_columns"] = selected_columns
with open("run_task_cfg_train.json", 'w+') as f:
    json.dump(cfg_dict, f, indent=4)

# party1
data_party = cfg_dict["each_party"][1]["data_party"]
data_party["input_file"] = "/home/juzix/work/fighter-py/data/big_data/train_data/insurance_train_data_1w.csv"
input_file = data_party["input_file"]
columns = list(pd.read_csv(input_file, nrows=0).columns)
key_column = columns[0]
print(f'key_column:{key_column}')
selected_columns = columns[1:]
print(f'selected_columns:{selected_columns}')
data_party["key_column"] = key_column
data_party["selected_columns"] = selected_columns

with open("run_task_cfg_train.json", 'w+') as f:
    json.dump(cfg_dict, f, indent=4)


###### predict task config
cfg_dict["contract_id"] = "../tests/standalone_test/lr_predict.py"
cfg_dict["dynamic_parameter"] = {
    "model_restore_party": "p3", 
    "model_path": "/home/juzix/fighter-v2/fighter-py/data_svc/results_root/abc/p7"
}

model_restore_party = cfg_dict["dynamic_parameter"]["model_restore_party"]
if model_restore_party not in cfg_dict["data_party"]:
    cfg_dict["data_party"].append(model_restore_party)

with open("run_task_cfg_predict.json", 'w+') as f:
    json.dump(cfg_dict, f, indent=4)