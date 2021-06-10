# coding:utf-8

import os
import math
import json
import time
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import latticex.rosetta as rtt

np.set_printoptions(suppress=True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_rtt_config_str(cfg_dict, rtt_cfg_template_file):
    '''生成激活rosetta时所需要的配置'''

    with open(rtt_cfg_template_file, 'r') as load_f:
        rtt_cfg_template_dict = json.load(load_f)

    rtt_cfg_template_dict["PARTY_ID"] = cfg_dict["cal_cfg"]["party_id"]
    load_mpc = rtt_cfg_template_dict["MPC"]
    env_cfg = cfg_dict["env_cfg"]
    load_mpc["P0"]["HOST"] = env_cfg["role_p0"]["host"]
    load_mpc["P0"]["PORT"] = env_cfg["role_p0"]["port"]
    load_mpc["P1"]["HOST"] = env_cfg["role_p1"]["host"]
    load_mpc["P1"]["PORT"] = env_cfg["role_p1"]["port"]
    load_mpc["P2"]["HOST"] = env_cfg["role_p2"]["host"]
    load_mpc["P2"]["PORT"] = env_cfg["role_p2"]["port"]
    lr_cfg = cfg_dict["algorithm_cfg"]["lr"]
    load_mpc["FLOAT_PRECISION"] = lr_cfg["float_pricision"]
    load_mpc["SAVER_MODE"] = lr_cfg["result_save_mode"]

    rtt_cfg_str = json.dumps(rtt_cfg_template_dict)
    print("rtt_cfg_str = {}".format(rtt_cfg_str))
    return rtt_cfg_str

def get_temp_dir(output_file):
    '''获取用于临时保存文件的目录路径'''
    temp_dir = os.path.join(os.path.dirname(output_file), 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def remove_temp_dir(cfg_dict):
    '''删除临时目录里的所有文件，这些文件都是一些临时保存的数据'''
    output_file = cfg_dict["cal_cfg"]["output_file"]
    temp_dir = get_temp_dir(output_file)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def split_file_to_feature_label(input_file, temp_dir, task_id, id_column_name, label_col_name):
    '''
    为了满足rosetta的数据输入，
    1. 首先需要对数据文件去掉id列。
    2. 对于有label的一方，将输入的数据文件拆分成两个文件，分别是特征文件file_x及标签文件file_y。
    '''
    file_x = ""
    file_y = ""
    if input_file:
        input_data = pd.read_csv(input_file, dtype="str")
        if label_col_name:
            file_y = os.path.join(temp_dir, f"file_y_{task_id}.csv")
            y_data = input_data[label_col_name]
            y_data = pd.DataFrame(y_data.values, columns=[label_col_name])
            y_data.to_csv(file_y, header=True, index=False)
            del input_data[label_col_name]
        file_x = os.path.join(temp_dir, f"file_x_{task_id}.csv")
        x_data = input_data.drop(labels=id_column_name, axis=1)
        x_data.to_csv(file_x, header=True, index=False)
    return file_x, file_y

def set_open_gmssl(party_id, use_ssl=False):
    rtt.Netutil.set_open_gmssl(use_ssl)
    if use_ssl:
        if party_id == 0:
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/SS00.crt", "certs/SS00.key",
                                      "certs/SE00.crt", "certs/SE00.key", party_id, 0)
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/CS01.crt", "certs/CS01.key",
                                      "certs/CE01.crt", "certs/CE01.key", party_id, 1)
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/CS02.crt", "certs/CS02.key",
                                      "certs/CE02.crt", "certs/CE02.key", party_id, 2)
        elif party_id == 1:
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/CS10.crt", "certs/CS10.key",
                                      "certs/CE10.crt", "certs/CE10.key", party_id, 0)
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/SS11.crt", "certs/SS11.key",
                                      "certs/SE11.crt", "certs/SE11.key", party_id, 1)
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/CS12.crt", "certs/CS12.key",
                                      "certs/CE12.crt", "certs/CE12.key", party_id, 2)
        elif party_id == 2:
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/CS20.crt", "certs/CS20.key",
                                      "certs/CE20.crt", "certs/CE20.key", party_id, 0)
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/CS21.crt", "certs/CS21.key",
                                      "certs/CE21.crt", "certs/CE21.key", party_id, 1)
            rtt.Netutil.set_mpc_certs("certs/CA.crt", "certs/SS22.crt", "certs/SS22.key",
                                      "certs/SE22.crt", "certs/SE22.key", party_id, 2)

def lr_svc(cfg_dict):
    '''
    逻辑回归算法实现函数
    '''

    # 解析配置参数
    rtt_cfg_str = get_rtt_config_str(cfg_dict, './rtt_config_template.json')

    cal_cfg = cfg_dict["cal_cfg"]
    party_id = cal_cfg["party_id"]
    task_id = cal_cfg["task_id"]
    input_file = cal_cfg["input_file"]
    output_file = cal_cfg["output_file"]
    id_column_name = cal_cfg["id_column_name"]
    label_col_name = cal_cfg["label_col_name"]
    lr_cfg = cfg_dict["algorithm_cfg"]["lr"]
    epochs = lr_cfg["epochs"]
    batch_size = lr_cfg["batch_size"]
    learning_rate = lr_cfg["learning_rate"]

    temp_dir = get_temp_dir(output_file)
    file_x, file_y = split_file_to_feature_label(input_file, temp_dir, task_id, id_column_name, label_col_name)

    # 设置是否需要打开ssl，True-打开，False-关闭
    set_open_gmssl(party_id, use_ssl=False)

    print("rtt_cfg_str:", rtt_cfg_str)
    # active protocol and sharing data
    rtt.activate("Helix", rtt_cfg_str)
    shard_x, shard_y = rtt.PrivateDataset(data_owner=(0, 1), label_owner=0).load_data(file_x, file_y, header=0)
    column_total_num = shard_x.shape[1]

    X = tf.placeholder(tf.float64, [None, column_total_num])
    Y = tf.placeholder(tf.float64, [None, 1])
    W = tf.Variable(tf.zeros([column_total_num, 1], dtype=tf.float64))
    b = tf.Variable(tf.zeros([1], dtype=tf.float64))
    logits = tf.matmul(X, W) + b
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(loss)
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
    # predict
    pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
    reveal_Y = rtt.SecureReveal(pred_Y, 1)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # train
        batch_num = math.ceil(len(shard_x) / batch_size)
        for e in range(epochs):
            for i in range(batch_num):
                bX = shard_x[(i * batch_size): (i + 1) * batch_size]
                bY = shard_y[(i * batch_size): (i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: bX, Y: bY})

                if (i % 50 == 0) or (i == batch_num - 1):
                    print(f"epoch:{e + 1}/{epochs}, batch:{i + 1}/{batch_num}")

        saver.save(sess, output_file)
        print("save model success.")

        # test predict
        Y_pred_prob = sess.run(reveal_Y, feed_dict={X: shard_x, Y: shard_y})
        Y_pred_prob = Y_pred_prob.astype('str').astype("float")
        print("Y_pred_prob:\n", Y_pred_prob)
        if party_id == 0:
            id_column = pd.read_csv(input_file, usecols=[id_column_name])

            print("predict result write to file.")
            output_file_predict_prob = os.path.splitext(output_file)[0] + "_predict_prob.csv"
            Y_id_prob = pd.DataFrame(np.hstack((id_column.values, Y_pred_prob)), columns=[id_column_name, "Y_prob"])
            Y_id_prob.to_csv(output_file_predict_prob, header=True, index=False)

            output_file_predict_class = os.path.splitext(output_file)[0] + "_predict_class.csv"
            Y_class = (Y_pred_prob > 0.5) * 1  # 转化为分类
            Y_id_class = pd.DataFrame(np.hstack((id_column.values, Y_class)), columns=[id_column_name, "Y_class"])
            Y_id_class.to_csv(output_file_predict_class, header=True, index=False)

    rtt.deactivate()
    print("lr finish.")

def main(user_cfg):
    '''主函数，模块入口'''
    lr_svc(user_cfg)
    # remove_temp_dir(user_cfg)

if __name__ == '__main__':
    def assemble_cfg():
        '''收集参与方的参数'''

        import json
        import sys

        party_id = int(sys.argv[1])
        with open('lr_config.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        cal_cfg = cfg_dict["cal_cfg"]
        cal_cfg["party_id"] = party_id
        cal_cfg["input_file"] = ""
        if party_id != 2:
            cal_cfg["input_file"] = f"../../data_svc/data/alignment_result_{party_id}.csv"
        cal_cfg["output_file"] = f"../../data_svc/data/p{party_id}/my_result"
        if party_id != 0:
            cal_cfg["label_col_name"] = ""

        return cfg_dict

    cfg_dict = assemble_cfg()
    main(cfg_dict)
