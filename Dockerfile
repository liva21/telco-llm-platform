# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    TELCO_MODEL_MODE=mock \
    PORT=8000

WORKDIR /app

# Sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU torch + diger paketler
COPY requirements-cpu.txt /app/requirements-cpu.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-cpu.txt

# Uygulama kodu
COPY api/ /app/api/
COPY mcp_server/ /app/mcp_server/
COPY finetune/ /app/finetune/
COPY scripts/ /app/scripts/

# OpenShift uyumlulugu: root-olmayan rastgele UID icin grup izinleri
RUN mkdir -p /app/finetune/outputs /app/finetune/mlflow_tracking \
    && chgrp -R 0 /app && chmod -R g+rwX /app

EXPOSE 8000
USER 1001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD \
    curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
