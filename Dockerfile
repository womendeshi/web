FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

ENV PYTHONUNBUFFERED=1
# 微信云托管会注入 PORT 环境变量，这里使用它
ENV PORT=8000

# 启动 FastAPI 应用
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

