# Use a base image with Python
FROM python:3.10
 
# Set working directory
WORKDIR /app
 
# Copy everything into the container
COPY . .
 
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
 
# Run the FastAPI server
CMD ["uvicorn", "api.inferencing:app", "--host", "0.0.0.0", "--port", "80"]