# 发起任务的配置文件说明

## 配置模板
1. 配置模板在console/run_task_cfg_template文件夹中
2. 可分为单机测试的模板与多机测试的模板，即local和cluster
3. 对于配置文件中的路径配置，可以配绝对路径或者相对路径。
4. 当使用相对路径时，对于字段contract_id是基于console目录；对于数据源的路径，即字段data_path，是基于与console同级的data_svc目录。

## 隐私算法任务的配置文件
1. 发起隐私算法任务的配置文件在console/privacy文件夹中

## 非隐私算法任务的配置文件
1. 发起非隐私算法任务的配置文件在console/non-privacy文件夹中