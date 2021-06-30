# coding:utf-8

import sys
sys.path.append("..")
from compute_svc.logistic_algorithm import PrivacyLogisticRegression


def main(cfg_dict):
    '''主函数，模块入口'''
    privacy_lr = PrivacyLogisticRegression(cfg_dict)
    privacy_lr.predict()


if __name__ == '__main__':
    import json
    import sys
    import argparse
    import io_channel
    import latticex.rosetta as rtt


    def error_callback(a, b, c, d, e):
        print("nodeid:{}, id:{}, errno:{}, error_msg:{}, ext_data:{}".format(a, b, c, d, e))
        return

    def create_channel(node_id_, strJson):
        print("_node_id======================:{}".format(node_id_))
        # 启动服务
        is_start_server = True
        res = io_channel.create_channel(node_id_, strJson, is_start_server, error_callback)
        return res

    def assemble_cfg_predict():
        '''收集参与方的参数'''

        party_id = int(sys.argv[1])
        with open('lr_config_predict.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        role_cfg = cfg_dict["user_cfg"]["role_cfg"]
        role_cfg["party_id"] = party_id
        role_cfg["input_file"] = ""
        if party_id == 0:
            role_cfg["input_file"] = f"../data/bank_predict_data.csv"
        elif party_id == 1:
            role_cfg["input_file"] = f"../data/insurance_predict_data.csv"
        role_cfg["model_file"] = f"../output/p{party_id}/my_result"
        role_cfg["output_file"] = f"../output/p{party_id}/predict"

        return cfg_dict

    _parser = argparse.ArgumentParser(description="LatticeX Rosetta")
    _parser.add_argument('--node_id', type=str, help="Node ID")  # node_id
    _args, _unparsed = _parser.parse_known_args()
    node_id_ = _args.node_id

    with open('rtt_config_no_via.json', 'r') as load_f:
        rtt_config = load_f.read()

    channel = create_channel(node_id_, rtt_config)
    rtt.set_channel(channel)
    print("set channel succeed==================")

    cfg_dict = assemble_cfg_predict()
    main(cfg_dict)