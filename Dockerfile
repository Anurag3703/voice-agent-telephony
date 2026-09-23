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
ENV PORT=7860

EXPOSE 7860
EXPOSE 8080

# Launch FastAPI ASGI server with uvicorn (supports HuggingFace Spaces port 7860 and Render port 8080)
CMD ["sh", "-c", "python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860} --app-dir backend"]
