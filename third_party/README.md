# 第三方库说明

## 1. channel-sdk
+ 仓库：https://github.com/datumtechs/datum-network-channel-sdk
```
git clone https://github.com/datumtechs/datum-network-channel-sdk.git -b develop
cd channel-sdk
git checkout 0ec257328f7aa076c0f6f450774cff676db460b0
```

+ 安装依赖
```
sudo apt-get install build-essential libgflags-dev clang libc++-dev unzip
```

+ 编译
```
./build.sh compile --package-ice-via --python-version=3.7
```
--package-ice-via：是否打包IceGrid和Glacier2相关via文件，如果是ON表示是，OFF表示否（默认为否）；自测需要加上，生产环境不加。

--python-version: 指定python版本打包; 默认为当前系统python3对应的版本, ubuntu18.04默认为：python3.6;

+ 安装
```
pip3 install dist/channel_sdk-*
```

+ 卸载并清除
```
./build.sh clean
```


## 2. rosetta
+ 仓库：http://192.168.9.66/PlatON-Crypto/Rosetta
```
git clone --recurse http://192.168.9.66/PlatON-Crypto/Rosetta.git -b develop
cd Rosetta
git checkout 49e9cce5cc7d52f92bfb839723041d63eef543fe
```

+ 环境依赖确认
```
lsb_release -r       # e.g. Release:	18.04
python3 --version    # e.g. Python 3.7.5
pip3 --version       # e.g. pip 20.0.2
apt show libssl-dev  # e.g. Version: 1.1.1-1ubuntu2.1~18.04.5
cmake --version      # e.g. cmake version 3.10
pip3 show tensorflow # e.g. tensorflow version 1.14.0
```

+ 安装依赖
如果不满足上面的依赖，则安装，安装完成后，请再次检测版本是否符合要求
```
# install python3, pip3, openssl
sudo apt update
sudo apt install python3-dev python3-pip libssl-dev cmake
# upgrade pip3 to latest 
sudo pip3 install --upgrade pip
pip3 install tensorflow==1.14.0
```

+ 编译
```
./rosetta.sh compile --enable-protocol-mpc-securenn
```
执行`. rtt_completion`后，再`./rosetta.sh compile`加tab键可以查看该指令后面可加的参数

+ 安装
```
pip3 install dist/latticex_rosetta-*
```

+ 卸载并清除
```
./rosetta.sh clean
```


## 3. psi
仓库：http://192.168.9.66/PlatON-Crypto/PSI
```
git clone --recurse http://192.168.9.66/PlatON-Crypto/PSI.git -b develop
cd PSI
git checkout 8db892f9a39be02d2197350eec00c7c25296d004
```

+ 安装依赖
```
sudo apt install libgmp-dev libtbb-dev libtbb2 libgmpxx4ldbl
```

+ 编译
```
./compile.sh
# 可选，运行测试例  ./run_demo_local.sh
```
执行`. rtt_completion`后，再`./rosetta.sh compile`加tab键可以查看该指令后面可加的参数


+ 安装
```
pip3 install dist/latticex_psi-*
```

+ 卸载并清除
```
./clean.sh
```

