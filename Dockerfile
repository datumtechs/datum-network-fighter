FROM ubuntu:18.04

ARG PKG_ROSETTA
ARG PKG_CHANNEL_SDK
ARG PKG_FIGHTER

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
RUN echo "$PKG_CHANNEL_SDK"
RUN echo "$PKG_FIGHTER"

COPY $PKG_ROSETTA /home
COPY $PKG_CHANNEL_SDK /home
COPY $PKG_FIGHTER /home

RUN cd /home && pip3 install "$(basename $PKG_ROSETTA)"
RUN cd /home && pip3 install "$(basename $PKG_CHANNEL_SDK)"
RUN cd /home && pip3 install "$(basename $PKG_FIGHTER)"

RUN cd /home && rm *.whl

COPY data_svc/start_data_svc.sh /home
COPY compute_svc/start_compute_svc.sh /home
COPY via_svc/start_via_svc.sh /home

WORKDIR /home
