# streamlit/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 0. Install system dependencies (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# 1. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy Streamlit app
COPY app.py .

# 3. Ensure data dir exists (will be mounted from host)
RUN mkdir -p /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]