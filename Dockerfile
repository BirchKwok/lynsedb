FROM python:3.12-slim

WORKDIR /app

COPY . /app

# 安装系统依赖项
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖项
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "/app/lynse/api/http_api/http_api.py", "run", "--host", "0.0.0.0", "--port", "7637"]
