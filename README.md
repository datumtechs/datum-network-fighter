### 注意
如果没有修改系统python的软链接的情况下，则需要你在执行以下操作中指定所需python3.7版本的绝对路径，例如：
/usb/bin/python3.7 -m pip list
### 步骤
0. ` git clone -b develop --recurse https://github.com/datumtechs/datum-network-fighter.git && cd datum-network-fighter`
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

* 安装 [*channel_sdk*](./third_party/README.md#1-channel-sdk)：
  ```bash
  $ cd channel-sdk 
  $ pip install dist/channel_sdk-*.whl
  ```

* 安装tensorflow(版本1.14.0)：`pip install tensorflow==1.14.0`

* 安装 [*latticex.rosetta*](./third_party/README.md#2-rosetta)：
  ```bash
  $ cd Rosetta
  $ pip install dist/latticex_rosetta-*.whl
  ```

* 安装 [*latticex.psi*](./third_party/README.md#3-psi)：
  ```bash
  $ cd PSI
  $ pip install dist/latticex_psi-*.whl
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
  
  * 执行下一步之前检查如果提示no module build，请执行python -m pip install build -i https://pypi.douban.com/simple/
  
  * 编译好数据和计算服务包 (`python -m build -w`，结果在`./dist`)

  * build

    ```bash
    docker build -t matelem/fighter_dcv:v0.9 \
    --build-arg PKG_CHANNEL_SDK=./channel_sdk-1.0.0-cp37-cp37m-linux_x86_64.whl \
    --build-arg PKG_FIGHTER=./dist/fighter_dcv-0.9-py3-none-any.whl \
    --build-arg PKG_ROSETTA=./latticex_rosetta-1.0.0-cp37-cp37m-linux_x86_64.whl \
    .
    ```

  * 分发

    ```bash
    docker image save -o fighter_dcv.tar.gz matelem/fighter_dcv:v0.9
    
    docker image load -i fighter_dcv.tar.gz
    ```

    

* 运行

  * 准备好挂载目录，如`xxoo`

  * 准备好配置文件`*.yaml`（配置模板在安装目录下，如`site-packages/fighter/data_svc/config.yaml`），其内容根据挂载目录而定，假如有如下配置

    ```bash
    $ ls cfg_dir/
    compute_config_30001.yaml  data_config_50001.yaml  via_config_20000.yaml
    ```

  * 运行相应的服务

    ```bash
    # via 服务
    docker run -d --rm -v $PWD/xxoo:/xxoo -v $PWD/cfg_dir:/cfg_dir -p 20000:20000 matelem/fighter_dcv:v0.9 \
    ./start_via_svc.sh /cfg_dir/via_config_20000.yaml
    
    # 数据服务
    docker run -d --rm -v $PWD/xxoo:/xxoo -v $PWD/cfg_dir:/cfg_dir -p 50001:50001 --expose 1024-65535 -it matelem/fighter_dcv:v0.9 \
    ./start_data_svc.sh /cfg_dir/data_config_50001.yaml
    
    # 计算服务
    docker run -d --rm -v $PWD/xxoo:/xxoo -v $PWD/cfg_dir:/cfg_dir -p 30001:30001 --expose 1024-65535 -it matelem/fighter_dcv:v0.9 \
    ./start_compute_svc.sh /cfg_dir/compute_config_30001.yaml
    
    ```


## v3

- 1.一键部署启动命令

  ~~~
  source start_v3_service.sh config.cfg service_type requirements.txt
  ~~~

  - config.cfg是数据或者计算服务的配置文件
  - service_type为data即为数据服务，为compute即为计算服务，需要安装的python 三方模块requirements.txt，例如channel-sdk、rosetta等