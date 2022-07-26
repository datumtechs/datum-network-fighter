# base image
FROM ubuntu:18.04

# speed up the install sources
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
COPY ./apt_src.ubuntu18.04 /etc/apt/sources.list
# debconf: unable to initialize frontend: Dialog
ENV DEBIAN_FRONTEND noninteractive

# basic
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils gcc g++ clang make automake cmake build-essential \
    bsdmainutils libssl-dev libtbb-dev libgmpxx4ldbl libgmp-dev libtbb2 && apt-get autoremove
RUN apt-get install -y --no-install-recommends vim wget unzip git tree dos2unix time sed gawk sudo && apt-get autoremove

# python/pip
RUN apt-get install -y --no-install-recommends python3.7 python3.7-dev python3-distutils python3-pip && apt-get autoremove
RUN cd /usr/bin && ln -sf python3.7 python && ln -sf python3.7 python3 && ln -sf pip3 pip && python -m pip install --upgrade pip
RUN pip3 install setuptools==57.5.0 wheel==0.37.1 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 --no-cache-dir install numpy==1.16.0 pandas sklearn tensorflow==1.14.0 protobuf==3.20.* -i https://mirrors.aliyun.com/pypi/simple/
RUN wget https://cmake.org/files/v3.15/cmake-3.15.2.tar.gz && tar -zxvf cmake-3.15.2.tar.gz && \
    cd cmake-3.15.2 && ./configure && make && make install && cd .. && rm -rf cmake-3.15.2

# check the version
RUN python3 --version && pip3 --version && apt show libssl-dev && cmake --version
RUN echo "e.g. python 3.7.5; pip 20.0.2; Version: 1.1.1-1ubuntu2.1~18.04.5; cmake version 3.15.2"

WORKDIR /opt/Rosetta
COPY ./Rosetta /opt/Rosetta
RUN ./rosetta.sh compile --enable-protocol-mpc-securenn

# test
RUN pip3 install dist/latticex_rosetta-*
RUN cd /opt/Rosetta/example/tutorials/code && ./tutorials.sh rtt logistic_regression_reveal && tail -n 60 log/logistic_regression_reveal-0.log
