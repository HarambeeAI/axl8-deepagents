FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including tools for sandbox execution
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    grep \
    findutils \
    coreutils \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory for sandbox
RUN mkdir -p /workspace && chmod 777 /workspace

# Copy the entire project (needed for local package install)
COPY . /app/

# Install Python dependencies
WORKDIR /app/backend
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "langgraph-cli[inmem]"
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV SANDBOX_WORKSPACE=/workspace
ENV PYTHONUNBUFFERED=1

# Expose port for LangGraph API (Render uses PORT env var)
EXPOSE 10000

# Run FastAPI server with uvicorn
CMD python server.py
