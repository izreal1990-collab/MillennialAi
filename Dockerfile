# Use Python 3.12 slim base image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy and install minimal requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi numpy pydantic requests uvicorn

# Copy application files
COPY millennial_ai_live_chat.py .
COPY hybrid_brain.py .
COPY real_brain.py .
COPY continuous_learning.py .

# Create necessary directories
RUN mkdir -p logs data/learning_samples static

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application
CMD ["uvicorn", "millennial_ai_live_chat:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]