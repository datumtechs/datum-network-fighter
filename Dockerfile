FROM luodahui/channel-sdk:v2.0.3 as channel-sdk
# 基础镜像
FROM ubuntu:18.04

#更新为国内的镜像仓库，因为从默认官源拉取实在太慢了
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
COPY ./apt_src.ubuntu18.04 /etc/apt/sources.list

ARG PKG_ROSETTA
ARG PKG_PSI

# debconf: unable to initialize frontend: Dialog
ENV DEBIAN_FRONTEND noninteractive

# basic
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils gcc g++ clang make automake cmake build-essential autoconf \
    pkg-config bsdmainutils libssl-dev libtbb-dev libgmpxx4ldbl libtool libgflags-dev libgtest-dev libc++-dev && apt-get autoremove
RUN apt-get install -y --no-install-recommends vim wget unzip git tree dos2unix time sed gawk sudo && apt-get autoremove


# python/pip
RUN apt-get install -y --no-install-recommends python3.7 python3.7-dev python3-distutils python3-pip && apt-get autoremove
RUN cd /usr/bin && ln -sf python3.7 python && ln -sf python3.7 python3 && ln -sf pip3 pip && python -m pip install --upgrade pip
RUN pip3 install setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 --no-cache-dir install numpy==1.16.0 pandas sklearn tensorflow==1.14.0 -i https://mirrors.aliyun.com/pypi/simple/

RUN echo "$PKG_ROSETTA"
RUN echo "$PKG_PSI"

WORKDIR /Fighter

COPY $PKG_ROSETTA /Fighter/third_party/rosetta/
COPY $PKG_PSI /Fighter/third_party/psi/
COPY --from=channel-sdk /ChannelSDK/dist /Fighter/third_party/channel_sdk
# COPY ./third_party/rosetta /Fighter/third_party/rosetta
# COPY ./third_party/psi /Fighter/third_party/psi

RUN pip3 install /Fighter/third_party/rosetta/*.whl
RUN pip3 install /Fighter/third_party/channel_sdk/*.whl
RUN pip3 install /Fighter/third_party/psi/*.whl

COPY requirements.txt .
RUN pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN apt-get install -y --no-install-recommends \ 
    libssl1.0.0 \
    && apt-get autoremove
    
COPY common_module common_module
COPY compute_svc compute_svc
COPY data_svc data_svc
COPY pb pb

# unit test
COPY algorithms algorithms
COPY console console
COPY tests tests
COPY consul_client consul_client

