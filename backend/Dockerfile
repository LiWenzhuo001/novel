# =============================================================================
# 后端 Docker 镜像：Python 3.12 + FastAPI + pgvector/psycopg2
#
# 分层策略：
#   1. 先装 CPU 版 torch（避免多 GB 的 CUDA 依赖）
#   2. 再装 requirements.txt 中其余依赖（国内镜像加速）
#   3. 最后拷贝源码（代码变更时缓存前两层）
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# 第一层：torch（CPU 版，仅 ~200MB；CUDA 版 > 2GB）
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# 第二层：其余依赖（langchain / fastapi / pgvector / psycopg2-binary 等）
# 走清华 PyPI 镜像加速（国内网络环境）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第三层：源码 + 数据目录
COPY . .
RUN mkdir -p data/raw

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
