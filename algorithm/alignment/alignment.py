# coding: utf-8

import os
import time
import pandas as pd
import shutil
import sys
# os.environ['LD_LIBRARY_PATH'] = './install/lib'
# os.environ.setdefault("LD_LIBRARY_PATH", "./install/lib")
# os.system('export LD_LIBRARY_PATH=./install/lib')
# print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"))
import install.psi_lib.psi as sml

def get_temp_dir(party_id, output_file):
    '''获取用于临时保存文件的目录路径'''
    temp_dir = os.path.join(os.path.dirname(output_file), f'temp/p{party_id}')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def remove_temp_dir(cfg_dict):
    '''删除临时目录里的所有文件，这些文件都是一些临时保存的数据'''
    party_id = cfg_dict["cal_cfg"]["party_id"]
    output_file = cfg_dict["cal_cfg"]["output_file"]
    temp_dir = os.path.dirname(get_temp_dir(party_id, output_file))
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def set_open_gmssl(party_id, use_ssl=False):
    '''设置是否打开ssl，如果打开则设置证书'''
    sml.set_open_gmssl(use_ssl)
    if use_ssl:
        if party_id == 0:
            sml.set_certs("certs/CA.crt", "certs/SS.crt",
                          "certs/SS.key", "certs/SE.crt", "certs/SE.key")
        elif party_id == 1:
            sml.set_certs("certs/CA.crt", "certs/CS.crt",
                          "certs/CS.key", "certs/CE.crt", "certs/CE.key")

def generate_psi_input_file(input_file, psi_input_file, id_column_name):
    '''抽取id列，生成id输入文件，用于psi'''
    id_column = pd.read_csv(input_file, usecols=[id_column_name])
    id_column.to_csv(psi_input_file, header=True, index=False)

def get_psi_config(cfg_dict):
    '''获取psi sdk所需的配置'''
    env_cfg = cfg_dict["env_cfg"]
    p0_host = env_cfg["role_p0"]["host"]
    p1_host = env_cfg["role_p1"]["host"]
    common_cfg = cfg_dict["common_cfg"]
    port = common_cfg["port"]
    result_save_mode = common_cfg["result_save_mode"]
    role_cfg = cfg_dict["role_cfg"]
    party_id = role_cfg["party_id"]

    psicfg = sml.PSIConfig()
    psicfg.recv_party = result_save_mode
    psicfg.party = party_id
    if party_id == 0:
        psicfg.peer = p1_host
    elif party_id == 1:
        psicfg.peer = p0_host
    psicfg.port = port
    return psicfg

def alignment_svc(cfg_dict):
    '''
    多方数据的对齐，先做psi取交集，然后各方本地依据交集文件进行数据对齐
    '''

    # 解析配置参数
    common_cfg = cfg_dict["common_cfg"]
    task_id = common_cfg["task_id"]
    role_cfg = cfg_dict["role_cfg"]
    party_id = role_cfg["party_id"]
    input_file = role_cfg["input_file"]
    output_file = role_cfg["output_file"]
    id_column_name = role_cfg["id_column_name"]

    temp_dir = get_temp_dir(party_id, output_file)
    psi_input_file = os.path.join(temp_dir, f'psi_input_file_{task_id}.csv')
    psi_output_file = os.path.join(temp_dir, f'psi_output_file_{task_id}.csv')
    # alignment_output_file = os.path.join(temp_dir, f'alignment_input_file_{task_id}.csv')
    generate_psi_input_file(input_file, psi_input_file, id_column_name)

    # 设置是否需要打开ssl，True-打开，False-关闭
    print("set if open ssl")
    set_open_gmssl(party_id, use_ssl=False)

    psicfg = get_psi_config(cfg_dict)
    psiHandler = sml.PSIHandler()
    init_start_time = time.time()
    print("waiting other party...")
    try:
        retCode = psiHandler.init(psicfg, 30*1000)
    except Exception as e:
        wait_time = time.time() - init_start_time
        errorMsg = f"psi init error, {str(e)}! wait time:{round(wait_time, 3)}s"
        raise Exception(errorMsg)
    retCode = psiHandler.run(psi_input_file, psi_output_file)
    if retCode == False:
        print("psi run retCode={}".format(retCode))
        raise Exception(" PSI RUN FAIL! ")
    print("psi run success!")

    psi_output_data = pd.read_csv(psi_output_file, header=None)
    psi_output_data = pd.DataFrame(psi_output_data.values, columns=[id_column_name])
    input_data = pd.read_csv(input_file, dtype="str")  # numpy按float读取会有精度损失，所以这里按字符串读取
    alignment_result = pd.merge(psi_output_data, input_data, on=id_column_name, how='inner')
    alignment_result.to_csv(output_file, header=True, index=False)
    print("alignment finish.")


def main(user_cfg):
    '''主函数，模块入口'''
    alignment_svc(user_cfg)

if __name__=='__main__':
    def assemble_cfg():
        '''收集参与方的参数'''

        import json
        import sys

        party_id = int(sys.argv[1])
        with open('alignment_config.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        role_cfg = cfg_dict["role_cfg"]
        role_cfg["party_id"] = party_id
        role_cfg["input_file"] = ""
        if party_id == 0:
            role_cfg["input_file"] = "../../data_svc/data/1-test_bank.csv"
        elif party_id == 1:
            role_cfg["input_file"] = "../../data_svc/data/1-test_insurance.csv"
        role_cfg["output_file"] = f"../../data_svc/data/alignment_result_{party_id}.csv"

        return cfg_dict

    cfg_dict = assemble_cfg()
    alignment_svc(cfg_dict)