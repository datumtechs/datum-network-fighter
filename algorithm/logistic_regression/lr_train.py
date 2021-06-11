# coding:utf-8

from logistic_algorithm import PrivacyLogisticRegression


def main(cfg_dict):
    '''主函数，模块入口'''
    privacy_lr = PrivacyLogisticRegression(cfg_dict)
    privacy_lr.train()

if __name__ == '__main__':
    import json
    import sys
    
    def assemble_cfg():
        '''收集参与方的参数'''

        party_id = int(sys.argv[1])
        with open('lr_config.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        role_cfg = cfg_dict["user_cfg"]["role_cfg"]
        role_cfg["party_id"] = party_id
        role_cfg["input_file"] = ""
        # if party_id != 2:
        #     role_cfg["input_file"] = f"../output/p{party_id}/alignment_result.csv"
        # role_cfg["output_file"] = f"../output/p{party_id}/my_result"
        if party_id == 0:
            role_cfg["input_file"] = f"../data/bank_train_data.csv"
        elif party_id == 1:
            role_cfg["input_file"] = f"../data/insurance_train_data.csv"
        role_cfg["output_file"] = f"../output/p{party_id}/my_result"
        if party_id != 0:
            role_cfg["with_label"] = False
            role_cfg["label_column_name"] = ""
        else:
            role_cfg["with_label"] = True
            role_cfg["label_column_name"] = "Y"

        return cfg_dict

    cfg_dict = assemble_cfg()
    main(cfg_dict)
