FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CDC_CACHE_DB_PATH=/app/backend/data/cdc_cache.sqlite3

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/backend/data && \
    chown -R appuser:appuser /app

USER appuser
VOLUME ["/app/backend/data"]
EXPOSE 8765

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health', timeout=2)" || exit 1

CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]
