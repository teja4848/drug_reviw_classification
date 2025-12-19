FROM python:3.12-slim

WORKDIR /app

# Install deps
COPY streamlit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code + any needed data
COPY streamlit /app/streamlit
COPY data /app/data

EXPOSE 8501

# Streamlit should call your file inside /app/streamlit
ENV API_URL=http://api:8000
CMD ["sh","-c","streamlit run /app/streamlit/app.py --server.address=0.0.0.0 --server.port ${PORT:-8501}"]
