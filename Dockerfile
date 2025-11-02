# Use Python 3.12 slim base image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies needed for PyTorch and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in stages to avoid memory issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=1.12.0 numpy>=1.21.0 && \
    pip install --no-cache-dir fastapi>=0.104.0 pydantic>=2.0.0 uvicorn>=0.24.0 && \
    pip install --no-cache-dir \
    pandas>=1.4.0 \
    psutil>=5.9.0 \
    pytest>=7.0.0 \
    tokenizers>=0.12.1 \
    transformers>=4.20.0

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run the web API
CMD ["python", "web_api.py"]