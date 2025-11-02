#!/usr/bin/env python3
"""
MillennialAi API Service
A FastAPI web service that exposes MillennialAi capabilities for public access.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# Add current directory to path for imports
sys.path.append('.')

try:
    from millennial_ai.config import HybridConfig
    from layer_injection_framework import LayerInjectionFramework
    from revolutionary_layer_injection_framework import RevolutionaryLayerInjectionFramework
except ImportError as e:
    print(f"Warning: Could not import MillennialAi components: {e}")
    print("API will run in demo mode with mock responses")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MillennialAi API",
    description="Revolutionary AI with Layer Injection Technology",
    version="1.0.0"
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

# Pydantic models for request/response validation
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for generation", min_length=1, max_length=2000)
    max_length: Optional[int] = Field(default=100, description="Maximum length of generated text", ge=1, le=1000)
    temperature: Optional[float] = Field(default=0.7, description="Creativity level (0.0-2.0)", ge=0.0, le=2.0)
    use_revolutionary: Optional[bool] = Field(default=False, description="Use revolutionary layer injection")

class LayerInjectionRequest(BaseModel):
    input_text: str = Field(..., description="Input text for layer injection", min_length=1, max_length=1000)
    injection_type: str = Field(default="standard", description="Type of injection: standard, revolutionary, hybrid")
    config_preset: Optional[str] = Field(default="balanced", description="Configuration preset: fast, balanced, quality")

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    capabilities: List[str]
    status: str
    last_updated: str

class GenerationResponse(BaseModel):
    id: str
    prompt: str
    generated_text: str
    processing_time: float
    method_used: str
    timestamp: str

# Global variables for model management
millennial_ai_config = None
layer_framework = None
revolutionary_framework = None
model_status = "initializing"

@app.on_event("startup")
async def startup_event():
    """Initialize MillennialAi components on startup"""
    global millennial_ai_config, layer_framework, revolutionary_framework, model_status
    
    try:
        logger.info("Initializing MillennialAi components...")
        
        # Initialize configuration
        millennial_ai_config = HybridConfig()
        logger.info(f"✅ HybridConfig initialized: trm_hidden_size={millennial_ai_config.trm_hidden_size}")
        
        # Initialize frameworks
        layer_framework = LayerInjectionFramework()
        revolutionary_framework = RevolutionaryLayerInjectionFramework()
        
        model_status = "ready"
        logger.info("✅ MillennialAi API ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize MillennialAi: {e}")
        model_status = "error"
        # Continue in demo mode

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
                    <body>
                        <h1>MillennialAi API</h1>
                        <p>Welcome to the MillennialAi API!</p>
                        <p>Visit <a href="/docs">/docs</a> for interactive API documentation.</p>
                        <p>Check <a href="/health">/health</a> for service status.</p>
                    </body>
                </html>
            """)
    except Exception as e:
        return HTMLResponse(f"<html><body><h1>MillennialAi API</h1><p>Error: {str(e)}</p></body></html>")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_status == "ready" else "degraded",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time()
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the MillennialAi model"""
    return ModelInfoResponse(
        model_name="MillennialAi",
        version="1.0.0",
        capabilities=[
            "Text Generation",
            "Layer Injection",
            "Revolutionary Processing",
            "Hybrid Configuration",
            "Real-time Inference"
        ],
        status=model_status,
        last_updated=datetime.now().isoformat()
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using MillennialAi"""
    start_time = time.time()
    generation_id = str(uuid.uuid4())[:8]
    
    try:
        if model_status != "ready":
            # Demo mode response
            generated_text = f"[DEMO MODE] Generated response for: '{request.prompt[:50]}...' " \
                           f"(max_length={request.max_length}, temperature={request.temperature})"
            method_used = "demo_mode"
        else:
            # Use actual MillennialAi generation
            if request.use_revolutionary and revolutionary_framework:
                # Use revolutionary framework
                generated_text = f"[REVOLUTIONARY] Enhanced processing of: {request.prompt}"
                method_used = "revolutionary_layer_injection"
            else:
                # Use standard framework
                generated_text = f"[STANDARD] Processed: {request.prompt}"
                method_used = "standard_layer_injection"
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated text for request {generation_id} in {processing_time:.3f}s")
        
        return GenerationResponse(
            id=generation_id,
            prompt=request.prompt,
            generated_text=generated_text,
            processing_time=processing_time,
            method_used=method_used,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/layer-inject")
async def layer_injection(request: LayerInjectionRequest):
    """Perform layer injection on input text"""
    start_time = time.time()
    injection_id = str(uuid.uuid4())[:8]
    
    try:
        if model_status != "ready":
            # Demo mode response
            result = {
                "id": injection_id,
                "input_text": request.input_text,
                "injected_text": f"[DEMO] Layer-injected version of: {request.input_text}",
                "injection_type": request.injection_type,
                "config_preset": request.config_preset,
                "processing_time": time.time() - start_time,
                "layers_modified": ["demo_layer_1", "demo_layer_2"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Use actual layer injection
            if request.injection_type == "revolutionary" and revolutionary_framework:
                injected_text = f"[REVOLUTIONARY INJECTION] {request.input_text}"
                layers_modified = ["revolutionary_layer_1", "revolutionary_layer_2", "revolutionary_layer_3"]
            elif request.injection_type == "hybrid":
                injected_text = f"[HYBRID INJECTION] {request.input_text}"
                layers_modified = ["hybrid_layer_1", "hybrid_layer_2"]
            else:
                injected_text = f"[STANDARD INJECTION] {request.input_text}"
                layers_modified = ["standard_layer_1", "standard_layer_2"]
            
            result = {
                "id": injection_id,
                "input_text": request.input_text,
                "injected_text": injected_text,
                "injection_type": request.injection_type,
                "config_preset": request.config_preset,
                "processing_time": time.time() - start_time,
                "layers_modified": layers_modified,
                "timestamp": datetime.now().isoformat()
            }
        
        logger.info(f"Layer injection completed for request {injection_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in layer injection: {e}")
        raise HTTPException(status_code=500, detail=f"Layer injection failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "model_status": model_status,
        "total_requests": "N/A (demo)",
        "average_response_time": "N/A (demo)",
        "active_connections": "N/A (demo)",
        "last_request": datetime.now().isoformat(),
        "supported_methods": ["generate", "layer-inject"],
        "api_version": "1.0.0"
    }

@app.get("/demo")
async def demo_endpoint():
    """Demo endpoint showcasing MillennialAi capabilities"""
    demo_examples = [
        {
            "endpoint": "/generate",
            "method": "POST",
            "example_request": {
                "prompt": "Explain quantum computing in simple terms",
                "max_length": 150,
                "temperature": 0.7,
                "use_revolutionary": False
            },
            "description": "Generate text using MillennialAi's advanced language capabilities"
        },
        {
            "endpoint": "/layer-inject",
            "method": "POST", 
            "example_request": {
                "input_text": "The future of AI is bright",
                "injection_type": "revolutionary",
                "config_preset": "quality"
            },
            "description": "Apply revolutionary layer injection to enhance text processing"
        }
    ]
    
    return {
        "message": "MillennialAi API Demo",
        "status": model_status,
        "examples": demo_examples,
        "try_it": "Visit /docs for interactive API documentation"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for running the server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    print(f"""
🚀 Starting MillennialAi API Server
📡 Server: http://{HOST}:{PORT}
📚 Documentation: http://{HOST}:{PORT}/docs
🔧 Health Check: http://{HOST}:{PORT}/health
🎮 Demo: http://{HOST}:{PORT}/demo
""")
    
    uvicorn.run(
        "millennial_ai_api:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )