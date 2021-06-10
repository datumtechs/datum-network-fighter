# coding: utf-8

import os
import time
import pandas as pd
import shutil
import sys
sys.path.append('..')
# os.environ['LD_LIBRARY_PATH'] = './install/lib'
# os.environ.setdefault("LD_LIBRARY_PATH", "./install/lib")
# os.system('export LD_LIBRARY_PATH=./install/lib')
# print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"))
import compute_svc.install.psi_lib.psi as sml


class PrivacyAlignment():
    '''隐私数据对齐'''

    def __init__(self, cfg_dict):
        system_cfg = cfg_dict["system_cfg"]
        self.task_id = system_cfg["task_id"]
        self.port = system_cfg["port"]
        self.role_p0_name = system_cfg["role_p0"]["name"]
        self.role_p0_host = system_cfg["role_p0"]["host"]
        self.role_p1_name = system_cfg["role_p1"]["name"]
        self.role_p1_host = system_cfg["role_p1"]["host"]

        user_cfg = cfg_dict["user_cfg"]
        self.result_save_mode = user_cfg["common_cfg"]["result_save_mode"]
        role_cfg = user_cfg["role_cfg"]
        self.party_id = role_cfg["party_id"]
        self.input_file = role_cfg["input_file"]
        self.output_file = role_cfg["output_file"]
        self.id_column_name = role_cfg["id_column_name"]

    def alignment_svc(self):
        '''
        多方数据的对齐，先做psi取交集，然后各方本地依据交集文件进行数据对齐
        '''
        psi_output_file = self.psi_svc()
        psi_output_data = pd.read_csv(psi_output_file, header=None)
        psi_output_data = pd.DataFrame(psi_output_data.values, columns=[self.id_column_name])
        input_data = pd.read_csv(self.input_file, dtype="str")  # numpy按float读取会有精度损失，所以这里按字符串读取
        alignment_result = pd.merge(psi_output_data, input_data, on=self.id_column_name, how='inner')
        alignment_result.to_csv(self.output_file, header=True, index=False)
        print("alignment finish.")

    def psi_svc(self):
        '''求出两方之间id列的交集，并生成交集文件'''
        temp_dir = self.get_temp_dir()
        psi_input_file = os.path.join(temp_dir, f'psi_input_file_{self.task_id}.csv')
        psi_output_file = os.path.join(temp_dir, f'psi_output_file_{self.task_id}.csv')
        # alignment_output_file = os.path.join(temp_dir, f'alignment_input_file_{task_id}.csv')
        self.generate_psi_input_file(psi_input_file)

        # 设置是否需要打开ssl，True-打开，False-关闭
        print("set if open ssl")
        self.set_open_gmssl(use_ssl=False)

        psicfg = self.get_psi_config()
        psiHandler = sml.PSIHandler()
        init_start_time = time.time()
        print("waiting other party...")
        try:
            retCode = psiHandler.init(psicfg, 30 * 1000)
        except Exception as e:
            wait_time = time.time() - init_start_time
            errorMsg = f"psi init error, {str(e)}! wait time:{round(wait_time, 3)}s"
            raise Exception(errorMsg)
        retCode = psiHandler.run(psi_input_file, psi_output_file)
        if retCode == False:
            print("psi run retCode={}".format(retCode))
            raise Exception(" PSI RUN FAIL! ")
        print("psi run success!")
        return psi_output_file

    def get_temp_dir(self):
        '''获取用于临时保存文件的目录路径'''
        temp_dir = os.path.join(os.path.dirname(self.output_file), f'temp/p{self.party_id}')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def remove_temp_dir(self):
        '''删除临时目录里的所有文件，这些文件都是一些临时保存的数据'''
        temp_dir = os.path.dirname(self.get_temp_dir())
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def generate_psi_input_file(self, psi_input_file):
        '''抽取id列，生成id输入文件，用于psi'''
        id_column = pd.read_csv(self.input_file, usecols=[self.id_column_name])
        id_column.to_csv(psi_input_file, header=True, index=False)

    def set_open_gmssl(self, use_ssl=False):
        '''设置是否打开ssl，如果打开则设置证书'''
        sml.set_open_gmssl(use_ssl)
        if use_ssl:
            if self.party_id == 0:
                sml.set_certs("certs/CA.crt", "certs/SS.crt",
                              "certs/SS.key", "certs/SE.crt", "certs/SE.key")
            elif self.party_id == 1:
                sml.set_certs("certs/CA.crt", "certs/CS.crt",
                              "certs/CS.key", "certs/CE.crt", "certs/CE.key")

    def get_psi_config(self):
        '''获取psi sdk所需的配置'''
        psicfg = sml.PSIConfig()
        psicfg.recv_party = self.result_save_mode
        psicfg.party = self.party_id
        if self.party_id == 0:
            psicfg.peer = self.role_p1_host
        elif self.party_id == 1:
            psicfg.peer = self.role_p0_host
        psicfg.port = self.port
        return psicfg


def main(cfg_dict):
    '''主函数，模块入口'''
    privacy_alignment = PrivacyAlignment(cfg_dict)
    privacy_alignment.alignment_svc()

if __name__=='__main__':
    def assemble_cfg():
        '''收集参与方的参数'''

        import json
        import sys

        party_id = int(sys.argv[1])
        with open('alignment_config.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        role_cfg = cfg_dict["user_cfg"]["role_cfg"]
        role_cfg["party_id"] = party_id
        role_cfg["input_file"] = ""
        if party_id == 0:
            role_cfg["input_file"] = "../data/1-test_bank.csv"
        elif party_id == 1:
            role_cfg["input_file"] = "../data/1-test_insurance.csv"
        role_cfg["output_file"] = f"../output/alignment_result_{party_id}.csv"

        return cfg_dict

    cfg_dict = assemble_cfg()
    main(cfg_dict)