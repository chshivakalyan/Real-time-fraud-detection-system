FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system deps (needed for scipy/xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caching optimization)
COPY apicontainer/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

