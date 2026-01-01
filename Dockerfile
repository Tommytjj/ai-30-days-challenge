# Dockerfile —— Day 17 容器化你的 AI API

# Dockerfile（优化版）
FROM python:3.10-slim

WORKDIR /app

# 1. 先复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 只复制 api/ 和 models/（不复制 .git, __pycache__, etc.）
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]