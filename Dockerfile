# Use an official lightweight Python image
FROM python:3.9-slim

WORKDIR /app

# Copy files into the container
COPY requirements.txt .
COPY app.py .
COPY sentiment_model.onnx* .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
