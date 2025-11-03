# Use Python 3.12 slim base image for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV OLLAMA_HOST=0.0.0.0:11434

# Install system dependencies including Ollama
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install minimal requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.9.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi \
        numpy \
        pydantic \
        requests \
        uvicorn \
        azure-ai-ml \
        azure-identity \
        azure-storage-blob \
        mlflow \
        azureml-mlflow

# Copy application files
COPY millennial_ai_live_chat.py .
COPY hybrid_brain.py .
COPY real_brain.py .
COPY continuous_learning.py .

# Create necessary directories
RUN mkdir -p logs data/learning_samples static

# Create startup script to run both Ollama and FastAPI
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
\n\
echo "⏳ Waiting for Ollama to be ready..."\n\
sleep 5\n\
\n\
echo "📥 Pulling llama3:8b model..."\n\
ollama pull llama3:8b || echo "⚠️ Model pull failed, will retry on startup"\n\
\n\
echo "🌟 Starting MillennialAi FastAPI server..."\n\
exec uvicorn millennial_ai_live_chat:app --host 0.0.0.0 --port 8000 --log-level info\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports (FastAPI and Ollama)
EXPOSE 8000 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the startup script
CMD ["/app/start.sh"]