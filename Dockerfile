# Use a secure and minimal Python base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy only requirements to install dependencies first (better caching)
COPY requirements.txt .

# Install build tools temporarily, install Python dependencies, and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libffi-dev libssl-dev && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove gcc g++ libffi-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the source code
COPY . .

# Pre-create folders with safe permissions (only if needed)
# 755 is more secure than 777
RUN mkdir -p /app/Data/processed /app/Data/segmented /app/Data/test /app/outputs /app/models && \
    chmod -R 755 /app/Data /app/outputs /app/models

# Use python as entrypoint to run your main script
CMD ["python", "src/main.py"]
