#!/bin/bash

# 确保在出现错误时退出
set -e

# 1. 下载模型文件到工作目录
echo "Downloading model..."
wget -O /opt/CosyVoice/CosyVoice2-0.5B \
    https://aiassistmldevw0107074910.blob.core.windows.net/model-cosyvoice/CosyVoice2-0.5B

# 2. 强制更新代码仓库
echo "Updating source code..."
if [ -d "/opt/CosyVoice/CosyVoice" ]; then
    # 进入已有仓库目录
    cd /opt/CosyVoice/CosyVoice
    # 丢弃所有本地修改
    git reset --hard HEAD
    # 清理未跟踪文件
    git clean -fd
    # 拉取最新代码
    git pull origin master
else
    # 首次克隆仓库
    git clone https://github.com/fatfishzhou/CosyVoice.git /opt/CosyVoice/CosyVoice
fi

# 3. 进入服务端代码目录
cd /opt/CosyVoice/CosyVoice/runtime/python/grpc

# 4. 启动gRPC服务
echo "Starting server..."
exec python3 server.py --port 50000 --max_conc 4 --model_dir /opt/CosyVoice/CosyVoice2-0.5B