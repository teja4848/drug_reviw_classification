FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY streamlit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy Streamlit application code
COPY streamlit /app/streamlit

# (Optional) copy data if Streamlit needs it
COPY data /app/data

# Expose Streamlit port
EXPOSE 8501

# Environment variable for API
ENV API_URL=http://api:8000

# Run Streamlit
CMD ["sh", "-c", "streamlit run /app/streamlit/app.py --server.address=0.0.0.0 --server.port ${PORT:-8501}"]
