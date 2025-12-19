FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy API source + models
COPY api /app/api
COPY models /app/models

# Render requires binding to $PORT and 0.0.0.0
EXPOSE 8000
CMD ["sh","-c","uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
