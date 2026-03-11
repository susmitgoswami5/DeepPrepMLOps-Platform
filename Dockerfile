FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "src.api.router:app", "--host", "0.0.0.0", "--port", "8000"]
