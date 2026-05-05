FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gradio==4.44.0  # For Web UI (Phase 3 demo)

# Copy the entire project
COPY . .

# Make the pipeline script executable
RUN chmod +x run_pipeline.sh

# The default command runs the full reproducible pipeline
CMD ["./run_pipeline.sh"]
