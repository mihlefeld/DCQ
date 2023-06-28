# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y libopencv-dev libopencv-imgcodecs-dev cmake gdb
RUN apt-get install -y unzip wget
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip
RUN cp -rf libtorch/share/* /usr/share
RUN cp -rf libtorch/include/* /usr/include
RUN cp -rf libtorch/lib/* /usr/lib
RUN mkdir /home/DCQ
WORKDIR /home/DCQ
COPY * .

CMD while true; do sleep 1000; done