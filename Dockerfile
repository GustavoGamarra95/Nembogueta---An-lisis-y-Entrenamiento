# Dockerfile
FROM python:3.8-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Set environment variables (optional, can be overridden by .env)
ENV PYTHONUNBUFFERED=1

# Default command: open bash (can be overridden in docker-compose)
CMD ["/bin/bash"]

