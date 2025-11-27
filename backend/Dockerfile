FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project (needed for local package install)
COPY . /app/

# Install Python dependencies
WORKDIR /app/backend
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "langgraph-cli[inmem]"
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for LangGraph API (Render uses PORT env var)
EXPOSE 10000

# Run FastAPI server with uvicorn
CMD python server.py
