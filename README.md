### 步骤
0. ` git clone -b develop --recurse-submodules git@192.168.9.66:RosettaFlow/fighter-py.git && cd fighter-py`
1. 安装依赖：`pip install -r requirements.txt`
2. 编译 gRPC 协议：`python tools/compile_proto_file.py`
3. 测试：`cd tests && ./fast_check.sh`
4. 进入到相应的服务目录

     * via服务：`cd via_svc`
     * 数据服务：`cd data_svc`
     * 计算服务：`cd compute_svc`
5. 编辑 `config.yaml`
6. 启动：`./start_svc.sh config.yaml`



### 打python环境包

* 如果需要，激活使用conda：`eval "$(~/miniconda3/bin/conda shell.bash hook)"`

* 新建一个python(版本3.7)虚拟环境：`conda create -n env_py37 python=3.7`

* 激活此环境：`conda activate env_py37`

* 安装tensorflow(版本1.14.0)：`pip install tensorflow==1.14.0`

* 编译并安装 [*latticex.rosetta*](https://github.com/LatticeX-Foundation/Rosetta)：

  ```bash
  $ cd Rosetta
  $ ./rosetta.sh clean
  $ ./rosetta.sh compile --enable-protocol-mpc-securenn --enable-protocol-mpc-helix;
  $ pip install dist/latticex_rosetta-1.0.0-cp37-cp37m-linux_x86_64.whl
  ```

* 编译并安装 [*channel_sdk*](https://github.com/Metisnetwork/Metis-Channel-sdk) 和 [*国密版 grpc*](https://github.com/part-c/grpc/tree/v2.0.0_gmssl/src/cpp)：

  ```bash
  # 编译 third_party/protobuf
  $ cd channel-sdk
  $ ./build.sh clean
  $ ./build.sh compile
  # 如果要国密版本，则需要先编译国密版grpc
  ./build.sh compile --ssl-type=2 --python-version=3.7
  
  $ pip install dist/channel_sdk-1.0.0-cp37-cp37m-linux_x86_64.whl
  ```

* 安装数据&计算服务依赖：

  ```bash
  $ cd fighter-py
  $ pip install -r requirements.txt
  ```

* 打包整个虚拟环境：

  ```bash
  $ cd ~/miniconda3/envs/
  $ tar -zcf env_py37.tar.gz env_py37/
  ```



### 一键部署

* 预先在可正常运行的环境把python的`bin,lib`等一并打成python环境包（参考上一节）

* 参考`nodes_conf_example.json`填好配置
* `python onekey_deploy.py your_node_conf.json`，它有选项
  * `--remote_dir`：部署到目标机器的什么位置
  * `--py_env_zip`：python环境包，它相对固定，可以先由一个人做好
  * `--src_zip`: 代码压缩包
  * `--repack_src`：打包最新代码
  * `--py_home`：指定python环境的相对于启动脚本的位置
* 完成后就可以到相应目录启动服务了



### Dockerize

* 做镜像

  * copy rosetta包 和 channel_sdk包到当前build context (在`fighter-py`)

  * 编译好数据和计算服务包 (`python -m build -w`，结果在`./dist`)

  * build

    ```bash
    docker build -t matelem/metis_dcv:v0.9 \
    --build-arg PKG_CHANNEL_SDK=./channel_sdk-1.0.0-cp37-cp37m-linux_x86_64.whl \
    --build-arg PKG_FIGHTER=./dist/metis_dcv-0.9-py3-none-any.whl \
    --build-arg PKG_ROSETTA=./latticex_rosetta-1.0.0-cp37-cp37m-linux_x86_64.whl \
    .
    ```

  * 分发

    ```bash
    docker image save -o metis_dcv.tar.gz matelem/metis_dcv:v0.9
    
    docker image load -i metis_dcv.tar.gz
    ```

    

* 运行

  * 准备好挂载目录，如`xxoo`

  * 准备好配置文件`*.yaml`（配置模板在安装目录下，如`site-packages/metis/data_svc/config.yaml`），其内容根据挂载目录而定，假如有如下配置

    ```bash
    $ ls cfg_dir/
    compute_config_30001.yaml  data_config_50001.yaml  via_config_20000.yaml
    ```

  * 运行相应的服务

    ```bash
    # via 服务
    docker run -d --rm -v $PWD/xxoo:/xxoo -v $PWD/cfg_dir:/cfg_dir -p 20000:20000 matelem/metis_dcv:v0.9 \
    ./start_via_svc.sh /cfg_dir/via_config_20000.yaml
    
    # 数据服务
    docker run -d --rm -v $PWD/xxoo:/xxoo -v $PWD/cfg_dir:/cfg_dir -p 50001:50001 --expose 1024-65535 -it matelem/metis_dcv:v0.9 \
    ./start_data_svc.sh /cfg_dir/data_config_50001.yaml
    
    # 计算服务
    docker run -d --rm -v $PWD/xxoo:/xxoo -v $PWD/cfg_dir:/cfg_dir -p 30001:30001 --expose 1024-65535 -it matelem/metis_dcv:v0.9 \
    ./start_compute_svc.sh /cfg_dir/compute_config_30001.yaml
    
    ```

    