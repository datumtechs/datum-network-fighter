# coding:utf-8

from logistic_alogrithm import PrivacyLogisticRegression


def main(cfg_dict):
    '''主函数，模块入口'''
    privacy_lr = PrivacyLogisticRegression(cfg_dict)
    privacy_lr.predict()

if __name__ == '__main__':
    import json
    import sys

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


    cfg_dict = assemble_cfg_predict()
    main(cfg_dict)
