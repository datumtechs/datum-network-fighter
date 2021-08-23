FROM ubuntu:18.04

ARG gitlab_user
ARG gitlab_password

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
RUN pip3 install setuptools -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 --no-cache-dir install numpy==1.16.0 pandas sklearn tensorflow==1.14.0 -i https://mirrors.aliyun.com/pypi/simple/

COPY $pkg_rosetta /home
COPY $pkg_channel_sdk /home
COPY $pkg_fighter /home
