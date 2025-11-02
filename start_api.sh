#!/bin/bash

# MillennialAi API Deployment Script
echo "🚀 Starting MillennialAi API Server..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "millennial_ai_api.py" ]; then
    echo "❌ Error: millennial_ai_api.py not found!"
    echo "Please run this script from the MillennialAi directory."
    exit 1
fi

# Check if static directory exists
if [ ! -d "static" ]; then
    echo "❌ Error: static directory not found!"
    echo "Please ensure the static/index.html file exists."
    exit 1
fi

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade -r api_requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies!"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8000"}

echo ""
echo "🌐 Server Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  URL: http://$HOST:$PORT"
echo ""

# Start the server
echo "🎯 Starting MillennialAi API..."
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn millennial_ai_api:app \
    --host $HOST \
    --port $PORT \
    --reload \
    --log-level info

echo ""
echo "👋 MillennialAi API server stopped."