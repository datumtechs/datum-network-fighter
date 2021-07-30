### 步骤
0. ` git clone -b develop --recurse-submodules git@192.168.9.66:RosettaFlow/fighter-py.git && cd fighter-py`
1. 安装依赖：`pip install -r requirements.txt`
2. 编译 gRPC 协议：`python compile_proto_file.py`
3. 测试：`cd tests && ./fast_check.sh`
4. 进入到相应的服务目录

     * via服务：`cd via_svc`
     * 数据服务：`cd data_svc`
     * 计算服务：`cd compute_svc`
5. 编辑 `config.yaml`
6. 启动：`./start_svc.sh config.yaml`



### 一键部署

* 预先在可正常运行的环境把python的`bin,lib`等一并打成python环境包

* 参考`nodes_conf_example.json`填好配置
* `python onekey_deploy.py your_node_conf.json`，它有选项
  * `--remote_dir`：部署到目标机器的什么位置
  * `--py_env_zip`：python环境包，它相对固定，可以先由一个人做好
  * `--src_zip`: 代码压缩包，如果不指定，会自动打包最新代码
* 完成后就可以到相应目录启动服务了

