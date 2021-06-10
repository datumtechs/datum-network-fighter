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


class PrivacyLogisticRegression(object):
    '''隐私的逻辑回归'''

    def __init__(self, cfg_dict: dict):
        print("cfg_dict:", cfg_dict)
        system_cfg = cfg_dict["system_cfg"]
        self.work_mode = system_cfg["work_mode"]
        self.task_id = system_cfg["task_id"]
        compute_node = system_cfg["compute_node"]
        self.role_p0_name = compute_node["role_p0"]["name"]
        self.role_p0_host = compute_node["role_p0"]["host"]
        self.role_p0_port = compute_node["role_p0"]["port"]
        self.role_p1_name = compute_node["role_p1"]["name"]
        self.role_p1_host = compute_node["role_p1"]["host"]
        self.role_p1_port = compute_node["role_p1"]["port"]
        self.role_p2_name = compute_node["role_p2"]["name"]
        self.role_p2_host = compute_node["role_p2"]["host"]
        self.role_p2_port = compute_node["role_p2"]["port"]

        common_cfg = cfg_dict["user_cfg"]["common_cfg"]
        role_cfg = cfg_dict["user_cfg"]["role_cfg"]
        self.float_pricision = common_cfg["float_pricision"]
        self.result_save_mode = common_cfg["result_save_mode"]
        algorithm_cfg = common_cfg["algorithm_cfg"]
        self.epochs = algorithm_cfg["epochs"]
        self.batch_size = algorithm_cfg["batch_size"]
        self.learning_rate = algorithm_cfg["learning_rate"]
        self.party_id = role_cfg["party_id"]
        self.input_file = role_cfg["input_file"]
        self.output_file = role_cfg["output_file"]
        self.id_column_name = role_cfg["id_column_name"]
        self.with_label = role_cfg["with_label"]
        self.label_column_name = role_cfg["label_column_name"]

    def lr_svc(self):
        '''
        逻辑回归算法实现函数
        '''

        np.set_printoptions(suppress=True)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # 获取rtt的配置参数
        rtt_cfg_str = self.get_rtt_config_str('./rtt_config_template.json')
        print("rtt_cfg_str = {}".format(rtt_cfg_str))
        file_x, file_y = self.split_file_to_feature_label()
        # 设置是否需要打开ssl，True-打开，False-关闭
        self.set_open_gmssl(use_ssl=False)

        print("waiting other party...")
        rtt.activate("Helix", rtt_cfg_str)
        # sharing data
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
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(var_list=None, max_to_keep=5, name='v2')
        # predict
        pred_Y = tf.sigmoid(tf.matmul(X, W) + b)
        reveal_Y = rtt.SecureReveal(pred_Y, 1)

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            # train
            train_start_time = time.time()
            batch_num = math.ceil(len(shard_x) / self.batch_size)
            for e in range(self.epochs):
                for i in range(batch_num):
                    bX = shard_x[(i * self.batch_size): (i + 1) * self.batch_size]
                    bY = shard_y[(i * self.batch_size): (i + 1) * self.batch_size]
                    sess.run(optimizer, feed_dict={X: bX, Y: bY})
                    if (i % 50 == 0) or (i == batch_num - 1):
                        print(f"epoch:{e + 1}/{self.epochs}, batch:{i + 1}/{batch_num}")
            saver.save(sess, self.output_file)
            train_use_time = round(time.time()-train_start_time, 3)
            print(f"save model success. train_use_time={train_use_time}s")

            # test predict
            predict_start_time = time.time()
            Y_pred_prob = sess.run(reveal_Y, feed_dict={X: shard_x, Y: shard_y})
            Y_pred_prob = Y_pred_prob.astype('str').astype("float")
            print("Y_pred_prob:\n", Y_pred_prob)
            if self.party_id == 0:
                id_column = pd.read_csv(self.input_file, usecols=[self.id_column_name])
                print("predict result write to file.")
                output_file_predict_prob = os.path.splitext(self.output_file)[0] + "_predict_prob.csv"
                Y_id_prob = pd.DataFrame(np.hstack((id_column.values, Y_pred_prob)), columns=[self.id_column_name, "Y_prob"])
                Y_id_prob.to_csv(output_file_predict_prob, header=True, index=False)
                output_file_predict_class = os.path.splitext(self.output_file)[0] + "_predict_class.csv"
                Y_class = (Y_pred_prob > 0.5) * 1  # 转化为分类
                Y_id_class = pd.DataFrame(np.hstack((id_column.values, Y_class)), columns=[self.id_column_name, "Y_class"])
                Y_id_class.to_csv(output_file_predict_class, header=True, index=False)
            predict_use_time = round(time.time() - predict_start_time, 3)
            print(f"trainset predict finish. predict_use_time={predict_use_time}s")
        rtt.deactivate()
        print("lr finish.")

    def set_open_gmssl(self, use_ssl: bool=False):
        '''设置是否打开ssl'''
        rtt.Netutil.set_open_gmssl(use_ssl)
        party_id = self.party_id
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

    def get_rtt_config_str(self, rtt_cfg_template_file: str):
        '''生成激活rosetta时所需要的配置'''
        with open(rtt_cfg_template_file, 'r') as load_f:
            rtt_cfg_template_dict = json.load(load_f)
        rtt_cfg_template_dict["PARTY_ID"] = self.party_id
        load_mpc = rtt_cfg_template_dict["MPC"]
        load_mpc["P0"]["HOST"] = self.role_p0_host
        load_mpc["P0"]["PORT"] = self.role_p0_port
        load_mpc["P1"]["HOST"] = self.role_p1_host
        load_mpc["P1"]["PORT"] = self.role_p1_port
        load_mpc["P2"]["HOST"] = self.role_p2_host
        load_mpc["P2"]["PORT"] = self.role_p2_port
        load_mpc["FLOAT_PRECISION"] = self.float_pricision
        load_mpc["SAVER_MODE"] = self.result_save_mode
        rtt_cfg_str = json.dumps(rtt_cfg_template_dict)
        return rtt_cfg_str

    def get_temp_dir(self):
        '''获取用于临时保存文件的目录路径'''
        temp_dir = os.path.join(os.path.dirname(self.output_file), 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def remove_temp_dir(self):
        '''删除临时目录里的所有文件，这些文件都是一些临时保存的数据'''
        temp_dir = self.get_temp_dir()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def split_file_to_feature_label(self):
        '''
        为了满足rosetta的数据输入，
        1. 首先需要对数据文件去掉id列。
        2. 对于有label的一方，将输入的数据文件拆分成两个文件，分别是特征文件file_x及标签文件file_y。
        '''
        file_x = ""
        file_y = ""
        temp_dir = self.get_temp_dir()
        if self.input_file:
            input_data = pd.read_csv(self.input_file, dtype="str")
            if self.with_label:
                file_y = os.path.join(temp_dir, f"file_y_{self.task_id}.csv")
                y_data = input_data[self.label_column_name]
                y_data = pd.DataFrame(y_data.values, columns=[self.label_column_name])
                y_data.to_csv(file_y, header=True, index=False)
                del input_data[self.label_column_name]
            file_x = os.path.join(temp_dir, f"file_x_{self.task_id}.csv")
            x_data = input_data.drop(labels=self.id_column_name, axis=1)
            x_data.to_csv(file_x, header=True, index=False)
        return file_x, file_y

def main(cfg_dict):
    '''主函数，模块入口'''
    privacy_lr = PrivacyLogisticRegression(cfg_dict)
    privacy_lr.lr_svc()


if __name__ == '__main__':
    def assemble_cfg():
        '''收集参与方的参数'''

        import json
        import sys

        party_id = int(sys.argv[1])
        with open('lr_config.json', 'r') as load_f:
            cfg_dict = json.load(load_f)

        role_cfg = cfg_dict["user_cfg"]["role_cfg"]
        role_cfg["party_id"] = party_id
        role_cfg["input_file"] = ""
        if party_id != 2:
            role_cfg["input_file"] = f"../output/p{party_id}/alignment_result.csv"
        role_cfg["output_file"] = f"../output/p{party_id}/my_result"
        if party_id != 0:
            role_cfg["with_label"] = False
            role_cfg["label_column_name"] = ""

        return cfg_dict

    cfg_dict = assemble_cfg()
    main(cfg_dict)
