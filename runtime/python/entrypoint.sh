#!/bin/bash

# 确保在出现错误时退出
# set -e

# 1. 设置模型下载 URL（使用环境变量，保证安全）
MODEL_SAS_URL=${MODEL_SAS_URL:-"https://aiassistmldevw0107074910.blob.core.windows.net/model-cosyvoice/CosyVoice2-0.5B?sp=rl&st=2025-02-11T14:35:13Z&se=2025-12-31T22:35:13Z&spr=https&sv=2022-11-02&sr=c&sig=04wDlFSHH8KK%2FKqMqtG5fdKdHZ%2BPFuUsPgWm8%2FIQ5nM%3D"}

echo "[Entrypoint] Starting initialization..."
echo "[Entrypoint] MODEL_SAS_URL is: '$MODEL_SAS_URL'"
# 2. 下载模型文件到工作目录
echo "[Entrypoint] Downloading model folder with AzCopy..."
azcopy copy "$MODEL_SAS_URL" "/opt/CosyVoice/" --recursive

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

# 重新生成.proto文件
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto

# 4. 启动 gRPC 服务
echo "Starting server..."
python3 server.py --port 50000 --max_conc 4 --model_dir /opt/CosyVoice/CosyVoice2-0.5B

# 5. 服务器崩溃后，保持容器运行（进入交互式 Shell）
echo "Server crashed or exited. Dropping into a shell for debugging..."
exec /bin/bash