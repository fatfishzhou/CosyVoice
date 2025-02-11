FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/CosyVoice

# 安装系统依赖（包含git和wget）
RUN apt-get update -y && \
    apt-get install -y g++ git wget && \
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制启动脚本并设置执行权限
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# 设置容器启动时执行的命令
ENTRYPOINT ["./entrypoint.sh"]