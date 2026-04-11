FROM python:3.12-slim

WORKDIR /app

COPY . /app

# 安装系统依赖项
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal

ENV PATH="/root/.cargo/bin:${PATH}"
ENV LYNSE_ROOT="/data"

# 构建并安装 LynseDB（含 Rust 扩展）
RUN pip install --no-cache-dir .

EXPOSE 7637
VOLUME ["/data"]

CMD ["python", "-m", "lynse.server", "run", "--host", "0.0.0.0"]
