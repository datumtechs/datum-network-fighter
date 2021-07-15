### 步骤
0. `git clone git@192.168.9.66:RosettaFlow/fighter-py.git && cd fighter-py`

1. 编译 gRPC 协议：`python compile_proto_file.py`

2. 进入到相应的服务目录

     * via服务：`cd via_svc`
     * 数据服务：`cd data_svc`
     * 计算服务：`cd compute_svc`

3. 编辑 `config.yaml`

4. 启动：`./start_svc.sh config.yaml`
