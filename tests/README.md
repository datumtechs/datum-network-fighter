## 单机测试
作用：在单台机器起3个数据节点、3个计算节点、6个调度节点(仅测试用)、3个via服务。主要用于快速调试代码，排查bug，场景测试等。
步骤：
0. 进入目录：`cd standalone_test`
1. 生成任务配置文件： `bash generate_task_cfg.sh`
2. 运行多个服务：`bash run_standalone.sh`
3. 运行逻辑回归训练任务： `comp_run_task abc task_cfg_lr_train.json`
4. 运行逻辑回归预测任务： `comp_run_task abc_predict task_cfg_lr_predict.json`
5. 查看任务日志：当前目录下(tests/standalone), 执行指令`cd log`，查看对应服务的日志即可。
注意：运行训练任务时，task_id建议使用abc。如果训练使用自定义task_id，那么预测的时候，model_path需对应修改，否则运行预测任务会出现找不到模型的错误。
关闭所有服务：`bash kill.sh`


## 集群测试
作用：在三台机器中，对每台机器分别起1个数据节点、1个计算节点、1个调度节点(仅测试用)、1个via服务。主要用于测试多机器部署是否成功。
步骤：
0. 进入目录：`cd cluster_test`
1. 打开配置文件，修改三台机器的ip、用户名及登录密码，`vi nodes_conf/config.yaml`。
2. 生成部署配置文件、证书、任务配置文件，`bash generate_nodes_conf.json`
3. 部署并运行服务：`bash deploy_cluster.sh`
4. 连接某个数据节点，用于发起任务： `bash run_task.sh`
5. 发起任务：`comp_run_task abc task_cfg_lr_train_cluster.json`
关闭所有服务：`bash kill.sh`

