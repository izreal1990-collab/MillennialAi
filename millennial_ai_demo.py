#!/usr/bin/env python3
"""
MillennialAi Demo API Service
A lightweight FastAPI web service for demonstrating MillennialAi capabilities.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import uuid
from datetime import datetime
import json
import sys
import os
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MillennialAi Demo API",
    description="Revolutionary AI with Layer Injection Technology - Demo Version",
    version="1.0.0-demo"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Demo constants
DEMO_MESSAGE = "N/A (demo mode)"
demo_responses = [
    "The future of artificial intelligence is incredibly promising, with developments in machine learning, neural networks, and deep learning transforming industries worldwide. As we advance, AI will become more integrated into daily life, enhancing productivity and solving complex problems.",
    
    "Quantum computing represents a paradigm shift in computational power, leveraging quantum mechanics to perform calculations exponentially faster than classical computers. This technology will revolutionize cryptography, optimization, and scientific simulations.",
    
    "Space exploration continues to captivate humanity's imagination, with missions to Mars, the Moon, and beyond pushing the boundaries of what's possible. Private companies and international collaborations are making space more accessible than ever before.",
    
    "Climate change mitigation requires innovative solutions combining renewable energy, carbon capture technologies, and sustainable practices. The transition to a green economy is accelerating with advances in solar, wind, and battery technologies.",
    
    "The Internet of Things (IoT) is creating interconnected ecosystems where devices communicate seamlessly, enabling smart cities, autonomous vehicles, and intelligent infrastructure that adapts to human needs in real-time."
]

# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Input prompt for text generation")
    max_length: int = Field(200, ge=10, le=1000, description="Maximum length of generated text")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Creativity/randomness (0.0-2.0)")
    use_revolutionary: bool = Field(False, description="Use revolutionary layer injection")

class GenerateResponse(BaseModel):
    generated_text: str
    processing_time: float
    method_used: str
    prompt_length: int
    output_length: int

class LayerInjectRequest(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=1000, description="Text to inject layers into")
    injection_type: str = Field("revolutionary", description="Type of layer injection")
    config_preset: str = Field("quality", description="Configuration preset")

class LayerInjectResponse(BaseModel):
    injected_text: str
    processing_time: float
    injection_type: str
    layers_injected: int
    enhancement_score: float

class HealthResponse(BaseModel):
    status: str
    model_status: str
    timestamp: str
    uptime: float
    demo_mode: bool

class StatsResponse(BaseModel):
    total_requests: str
    successful_generations: str
    avg_processing_time: str
    demo_mode: bool

# Global state
start_time = time.time()
request_count = 0

# Root endpoint - serve web interface
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web interface"""
    try:
        static_path = os.path.join(os.getcwd(), "static", "index.html")
        if os.path.exists(static_path):
            return FileResponse(static_path)
        else:
            return HTMLResponse("""
                <html>
                    <head>
                        <title>MillennialAi Demo</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                            h1 { color: #667eea; text-align: center; }
                            .demo-notice { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffc107; }
                            a { color: #667eea; text-decoration: none; }
                            a:hover { text-decoration: underline; }
                            .links { text-align: center; margin-top: 30px; }
                            .links a { margin: 0 15px; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>🚀 MillennialAi Demo API</h1>
                            <div class="demo-notice">
                                <strong>Demo Mode:</strong> This is a demonstration version showcasing the API interface. 
                                Responses are generated using demo content to demonstrate the UI and API structure.
                            </div>
                            <p>Welcome to the MillennialAi API demonstration! This service showcases our revolutionary 
                            AI capabilities with Layer Injection Technology.</p>
                            
                            <h3>Features:</h3>
                            <ul>
                                <li><strong>Text Generation:</strong> Advanced AI-powered text completion</li>
                                <li><strong>Layer Injection:</strong> Revolutionary neural network enhancement</li>
                                <li><strong>Interactive Web Interface:</strong> User-friendly testing interface</li>
                                <li><strong>RESTful API:</strong> Easy integration with your applications</li>
                            </ul>
                            
                            <div class="links">
                                <a href="/docs">📖 API Documentation</a>
                                <a href="/health">🔧 Health Check</a>
                                <a href="/stats">📊 Statistics</a>
                            </div>
                        </div>
                    </body>
                </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root: {e}")
        return HTMLResponse(f"<html><body><h1>MillennialAi Demo API</h1><p>Error: {str(e)}</p></body></html>")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_status="demo_ready",
        timestamp=datetime.now().isoformat(),
        uptime=time.time() - start_time,
        demo_mode=True
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using MillennialAi (Demo Mode)"""
    global request_count
    request_count += 1
    
    start_time_req = time.time()
    
    try:
        # Simulate processing time
        processing_delay = random.uniform(0.5, 1.5)
        time.sleep(processing_delay)
        
        # Select a demo response or create a contextual one
        if any(keyword in request.prompt.lower() for keyword in ['ai', 'artificial', 'intelligence', 'future']):
            generated_text = demo_responses[0]
        elif any(keyword in request.prompt.lower() for keyword in ['quantum', 'computing', 'computer']):
            generated_text = demo_responses[1]
        elif any(keyword in request.prompt.lower() for keyword in ['space', 'mars', 'exploration', 'universe']):
            generated_text = demo_responses[2]
        elif any(keyword in request.prompt.lower() for keyword in ['climate', 'environment', 'energy', 'green']):
            generated_text = demo_responses[3]
        elif any(keyword in request.prompt.lower() for keyword in ['iot', 'internet', 'smart', 'device']):
            generated_text = demo_responses[4]
        else:
            # Random selection for other prompts
            generated_text = random.choice(demo_responses)
        
        # Adjust length to request
        if len(generated_text) > request.max_length:
            generated_text = generated_text[:request.max_length-3] + "..."
        
        processing_time = time.time() - start_time_req
        method_used = "revolutionary" if request.use_revolutionary else "standard"
        
        logger.info(f"Generated text for prompt: '{request.prompt[:50]}...' - Method: {method_used}")
        
        return GenerateResponse(
            generated_text=generated_text,
            processing_time=processing_time,
            method_used=f"{method_used} (demo)",
            prompt_length=len(request.prompt),
            output_length=len(generated_text)
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/layer-inject", response_model=LayerInjectResponse)
async def inject_layers(request: LayerInjectRequest):
    """Apply layer injection to text (Demo Mode)"""
    global request_count
    request_count += 1
    
    start_time_req = time.time()
    
    try:
        # Simulate processing time
        processing_delay = random.uniform(0.8, 2.0)
        time.sleep(processing_delay)
        
        # Simulate layer injection by enhancing the text
        enhanced_text = f"[ENHANCED] {request.input_text} [LAYER-INJECTED: Advanced neural pathways activated for superior contextual understanding and response generation.]"
        
        processing_time = time.time() - start_time_req
        layers_injected = random.randint(3, 8)
        enhancement_score = random.uniform(0.85, 0.98)
        
        logger.info(f"Layer injection applied to: '{request.input_text[:50]}...' - Type: {request.injection_type}")
        
        return LayerInjectResponse(
            injected_text=enhanced_text,
            processing_time=processing_time,
            injection_type=f"{request.injection_type} (demo)",
            layers_injected=layers_injected,
            enhancement_score=enhancement_score
        )
        
    except Exception as e:
        logger.error(f"Layer injection error: {e}")
        raise HTTPException(status_code=500, detail=f"Layer injection failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics"""
    return StatsResponse(
        total_requests=f"{request_count} (demo session)",
        successful_generations=f"{max(0, request_count-1)} (demo)",
        avg_processing_time="1.2s (demo)",
        demo_mode=True
    )

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_name": "MillennialAi Demo Model",
        "status": "ready (demo mode)",
        "capabilities": [
            "Text generation",
            "Layer injection",
            "Conversational AI",
            "Creative writing"
        ],
        "demo_mode": True,
        "note": "This is a demonstration version. Full model capabilities are showcased through simulated responses."
    }

if __name__ == "__main__":
    import uvicorn
    
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    print("🚀 Starting MillennialAi Demo API...")
    print(f"📍 Server URL: http://{HOST}:{PORT}")
    print(f"📖 Documentation: http://{HOST}:{PORT}/docs")
    print(f"🔧 Health Check: http://{HOST}:{PORT}/health")
    print("🎯 Demo Mode: Simulated responses for demonstration")
    print()
    
    uvicorn.run(app, host=HOST, port=PORT)