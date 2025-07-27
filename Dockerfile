FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY utils/ ./utils/

# Start the API using Uvicorn
CMD ["uvicorn", "api.inferencing:app", "--host", "0.0.0.0", "--port", "80"]
