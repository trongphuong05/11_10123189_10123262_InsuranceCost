FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models /app/models
COPY service /app/service

ENV MODEL_PATH=/app/models/model.joblib \
    SCHEMA_PATH=/app/models/schema.json \
    METADATA_PATH=/app/models/metadata.json \
    AI_SERVICE_PORT=8001

EXPOSE 8001

WORKDIR /app/service

# Model được nạp ngay khi process Python khởi động (import app.py),
# không đợi request đầu tiên.
CMD ["python", "app.py"]
