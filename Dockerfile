# Production Dockerfile for Voice Agent & Cyber Honeypot Engine
FROM python:3.11-slim as builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

# Copy installed python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application backend, frontend, and evals
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY evals/ ./evals/

ENV PYTHONPATH="/app/backend:/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Launch FastAPI ASGI server with uvicorn
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "backend"]
