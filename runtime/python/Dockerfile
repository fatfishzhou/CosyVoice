# 使用 PyTorch 官方镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /opt/CosyVoice

# 修改 APT 源为 Google / Ubuntu 官方源
RUN sed -i 's@/mirrors.aliyun.com/@/archive.ubuntu.com/@g' /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install git unzip git-lfs curl
RUN apt-get install -y build-essential cmake
RUN git lfs install

# 克隆 CosyVoice 代码
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git

# 由于 Python 3.8 兼容性问题，使用 Python 3.10
RUN cd CosyVoice && pip3 install -r requirements.txt -i https://pypi.org/simple/ --trusted-host pypi.org

# 生成 gRPC 代码
RUN cd CosyVoice/runtime/python/grpc && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto