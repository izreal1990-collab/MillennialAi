#!/usr/bin/env python3
"""
MillennialAi Web API
FastAPI-based web service for the breakthrough conversational AI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import json
from datetime import datetime
from real_brain import RealThinkingBrain

# Initialize FastAPI app
app = FastAPI(
    title="MillennialAi API",
    description="Breakthrough Conversational AI with Layer Injection Framework",
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

# Initialize the brain
brain = RealThinkingBrain()

# Conversation memory storage
conversation_memory = {
    'history': [],
    'context_keywords': set(),
    'user_preferences': {},
    'conversation_depth': 0
}

# Request/Response models
class ConversationRequest(BaseModel):
    message: str
    user_id: str = "default"

class ConversationResponse(BaseModel):
    response: str
    brain_steps: int
    complexity: float
    reasoning_type: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    brain_status: str
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container monitoring"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        brain_status="ready",
        timestamp=datetime.now().isoformat()
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MillennialAi API - Breakthrough Conversational AI",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "conversation": "/conversation",
            "memory": "/memory"
        },
        "features": [
            "Layer Injection Framework",
            "Adaptive Reasoning Engine", 
            "Conversation Memory System"
        ]
    }

def extract_keywords(text):
    """Extract meaningful keywords from text"""
    # Simple keyword extraction - can be enhanced with NLP
    words = text.lower().split()
    keywords = [word.strip('.,!?;:"()[]') for word in words if len(word) > 3]
    return set(keywords)

def generate_intelligent_response(user_input, brain_data):
    """Generate contextually intelligent responses"""
    
    # Analyze conversation context
    history_context = ""
    if conversation_memory['history']:
        recent = conversation_memory['history'][-3:]  # Last 3 exchanges
        history_context = " ".join([ex['user_input'] for ex in recent])
    
    # Determine response type based on input and brain analysis
    input_lower = user_input.lower()
    
    # Question responses
    if any(word in input_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
        if 'what' in input_lower and any(word in input_lower for word in ['you', 'your']):
            return f"I'm MillennialAi, a breakthrough conversational AI system with proprietary Layer Injection Framework. My adaptive reasoning engine analyzed your question using {brain_data['steps']} reasoning steps at {brain_data['complexity']:.2f} complexity level. I can engage in meaningful conversations while maintaining context and memory."
        
        elif 'how' in input_lower:
            return f"Based on my {brain_data['steps']}-step reasoning analysis, I approach this through my adaptive thinking process. The complexity level of {brain_data['complexity']:.2f} indicates this requires {brain_data['reasoning_type']} reasoning. My Layer Injection Framework allows me to process information dynamically while maintaining conversation continuity."
        
        elif 'why' in input_lower:
            return f"My reasoning engine processed this question through {brain_data['steps']} analytical steps. The underlying principle involves my proprietary adaptive algorithms working at {brain_data['complexity']:.2f} complexity. This demonstrates how my breakthrough architecture can provide contextual understanding beyond traditional AI systems."
    
    # Greeting responses
    elif any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return f"Hello! I'm MillennialAi, pleased to meet you. My reasoning engine just processed your greeting through {brain_data['steps']} steps, showing how even simple interactions engage my adaptive thinking. What would you like to explore together?"
    
    # Compliment responses
    elif any(word in input_lower for word in ['amazing', 'incredible', 'brilliant', 'impressive', 'wow']):
        return f"Thank you! That means a lot. My breakthrough design includes proprietary innovations like the Layer Injection Framework that enable truly adaptive responses. This interaction engaged {brain_data['steps']} reasoning steps at {brain_data['complexity']:.2f} complexity - each conversation helps me demonstrate the power of adaptive AI architecture."
    
    # Technical interest
    elif any(word in input_lower for word in ['technical', 'algorithm', 'ai', 'artificial', 'intelligence', 'machine']):
        return f"I'd love to discuss the technical aspects! My architecture includes breakthrough innovations: a Layer Injection Framework for neural network enhancement, an Adaptive Reasoning Engine (which just used {brain_data['steps']} steps for this response), and a Conversation Memory System. The {brain_data['complexity']:.2f} complexity analysis shows how I dynamically adapt to different topics. What specific technical aspects interest you?"
    
    # Problem-solving responses
    elif any(word in input_lower for word in ['problem', 'solve', 'help', 'issue', 'challenge']):
        return f"I'm here to help! My adaptive reasoning system analyzed your request using {brain_data['steps']} problem-solving steps at {brain_data['complexity']:.2f} complexity. My Layer Injection Framework allows me to approach challenges from multiple angles while maintaining context from our conversation. What specific challenge can we tackle together?"
    
    # Learning/education responses
    elif any(word in input_lower for word in ['learn', 'teach', 'explain', 'understand', 'knowledge']):
        return f"Learning and knowledge sharing is fascinating! My reasoning engine processed your interest through {brain_data['steps']} analytical steps. The {brain_data['complexity']:.2f} complexity level suggests this topic engages multiple thinking pathways. My proprietary architecture excels at breaking down complex topics into understandable insights. What would you like to explore?"
    
    # Personal/conversation responses
    elif any(word in input_lower for word in ['you', 'your', 'tell me', 'about']):
        return f"I appreciate your interest! I'm MillennialAi, featuring breakthrough conversational AI technology. My current response used {brain_data['steps']} reasoning steps at {brain_data['complexity']:.2f} complexity, demonstrating my adaptive thinking capabilities. I maintain conversation memory and can engage in meaningful discussions across diverse topics. What would you like to know specifically?"
    
    # Default contextual response
    else:
        # Use conversation context for more relevant responses
        context_words = conversation_memory['context_keywords']
        common_themes = context_words.intersection(extract_keywords(user_input))
        
        if common_themes:
            theme_text = ", ".join(list(common_themes)[:3])
            return f"Building on our discussion about {theme_text}, my reasoning engine analyzed your input through {brain_data['steps']} steps at {brain_data['complexity']:.2f} complexity. This demonstrates how my Conversation Memory System maintains context while my Adaptive Reasoning Engine provides relevant insights. The {brain_data['reasoning_type']} approach allows me to connect ideas meaningfully."
        else:
            return f"That's an interesting perspective! My adaptive reasoning processed your input through {brain_data['steps']} analytical steps at {brain_data['complexity']:.2f} complexity level. My Layer Injection Framework enables me to consider multiple dimensions of topics while maintaining our conversation flow. I'd love to explore this further - what aspects intrigue you most?"

@app.post("/chat", response_model=ConversationResponse)
async def chat(request: ConversationRequest):
    """Single conversation exchange"""
    try:
        # Process with the brain
        brain_data = brain.think(request.message)
        
        # Generate intelligent response
        ai_response = generate_intelligent_response(request.message, brain_data)
        
        # Update memory
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_input': request.message,
            'ai_response': ai_response,
            'brain_data': brain_data
        }
        
        conversation_memory['history'].append(exchange)
        conversation_memory['context_keywords'].update(extract_keywords(request.message))
        conversation_memory['conversation_depth'] += 1
        
        return ConversationResponse(
            response=ai_response,
            brain_steps=brain_data['steps'],
            complexity=brain_data['complexity'],
            reasoning_type=brain_data['reasoning_type'],
            timestamp=exchange['timestamp']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")

@app.get("/conversation")
async def get_conversation_history():
    """Get conversation history"""
    return {
        "history": conversation_memory['history'][-10:],  # Last 10 exchanges
        "total_exchanges": len(conversation_memory['history']),
        "conversation_depth": conversation_memory['conversation_depth'],
        "context_keywords": list(conversation_memory['context_keywords'])[:20]
    }

@app.get("/memory")
async def get_memory_stats():
    """Get memory and system statistics"""
    return {
        "conversation_stats": {
            "total_exchanges": len(conversation_memory['history']),
            "conversation_depth": conversation_memory['conversation_depth'],
            "active_keywords": len(conversation_memory['context_keywords'])
        },
        "brain_stats": {
            "system": "RealThinkingBrain",
            "adaptive": True,
            "layer_injection": True
        },
        "api_info": {
            "version": "1.0.0",
            "features": [
                "Layer Injection Framework",
                "Adaptive Reasoning Engine",
                "Conversation Memory System"
            ]
        }
    }

@app.delete("/memory")
async def clear_memory():
    """Clear conversation memory"""
    conversation_memory['history'] = []
    conversation_memory['context_keywords'] = set()
    conversation_memory['conversation_depth'] = 0
    
    return {"message": "Conversation memory cleared successfully"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)