#!/usr/bin/env python3
"""
MillennialAi Live Chat API
Real-time conversational AI with continuous learning integration
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
import os
import threading
import queue
from pathlib import Path

# Import MillennialAi components
from hybrid_brain import HybridRevolutionaryBrain
from real_brain import RealThinkingBrain
from continuous_learning import continuous_learning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MillennialAi Live Chat API",
    description="Real-time conversational AI with continuous learning",
    version="2.0.0-live"
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

# Global state
start_time = time.time()
conversation_count = 0
active_conversations = {}
learning_data_queue = queue.Queue()

# Conversation memory system
class ConversationMemory:
    """Enhanced conversation memory for continuous learning"""

    def __init__(self):
        self.conversations = {}
        self.learning_data_path = Path("learning_data")
        self.learning_data_path.mkdir(exist_ok=True)

    def add_exchange(self, conversation_id: str, user_input: str, ai_response: str,
                    brain_data: Dict[str, Any], hybrid_data: Dict[str, Any]):
        """Add conversation exchange with full context"""

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'brain_data': brain_data,
            'hybrid_data': hybrid_data,
            'conversation_turn': len(self.conversations[conversation_id]) + 1
        }

        self.conversations[conversation_id].append(exchange)

        # Queue for continuous learning
        learning_sample = {
            'input': user_input,
            'response': ai_response,
            'brain_metrics': brain_data,
            'hybrid_metrics': hybrid_data,
            'timestamp': exchange['timestamp']
        }
        learning_data_queue.put(learning_sample)

    def get_conversation_context(self, conversation_id: str, max_turns: int = 5) -> str:
        """Get recent conversation context"""
        if conversation_id not in self.conversations:
            return ""

        recent_exchanges = self.conversations[conversation_id][-max_turns:]
        context = "Recent conversation:\n"

        for exchange in recent_exchanges:
            context += f"User: {exchange['user_input']}\n"
            context += f"AI: {exchange['ai_response'][:200]}...\n\n"

        return context

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics for continuous learning"""
        total_conversations = len(self.conversations)
        total_exchanges = sum(len(conv) for conv in self.conversations.values())

        return {
            'total_conversations': total_conversations,
            'total_exchanges': total_exchanges,
            'avg_conversation_length': total_exchanges / max(1, total_conversations),
            'learning_samples_queued': learning_data_queue.qsize()
        }

# Initialize AI components
print("🧠 Initializing MillennialAi Live Chat System...")

hybrid_brain = None
brain_available = False
ollama_available = False

try:
    # Try to initialize hybrid brain first
    hybrid_brain = HybridRevolutionaryBrain()
    brain_available = True
    ollama_available = hasattr(hybrid_brain, 'ollama') and hybrid_brain.ollama.available
    print("✅ Hybrid Revolutionary Brain loaded with Ollama integration")
except Exception as e:
    print(f"⚠️ Hybrid brain initialization failed: {e}")
    try:
        # Fallback to real brain
        hybrid_brain = RealThinkingBrain()
        brain_available = True
        ollama_available = False
        print("✅ Real Thinking Brain loaded (fallback mode)")
    except Exception as e2:
        print(f"❌ Brain initialization failed: {e2}")
        hybrid_brain = None

# Initialize conversation memory
memory = ConversationMemory()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID (auto-generated if not provided)")
    mode: str = Field("adaptive_selection", description="AI thinking mode")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    processing_time: float
    brain_metrics: Dict[str, Any]
    hybrid_metrics: Optional[Dict[str, Any]] = None
    conversation_turn: int

class HealthResponse(BaseModel):
    status: str
    brain_status: str
    ollama_status: str
    timestamp: str
    uptime: float
    active_conversations: int
    learning_samples_queued: int

class LearningStatsResponse(BaseModel):
    total_conversations: int
    total_exchanges: int
    avg_conversation_length: float
    learning_samples_queued: int
    brain_available: bool

# Background learning data processor
def process_learning_data():
    """Process queued learning data for continuous improvement"""
    while True:
        try:
            # Get learning sample
            sample = learning_data_queue.get(timeout=1)

            # Save to learning data file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_sample_{timestamp}.json"
            filepath = memory.learning_data_path / filename

            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)

            logger.info(f"💾 Learning sample saved: {filename}")

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error processing learning data: {e}")

# Start background learning processor
learning_thread = threading.Thread(target=process_learning_data, daemon=True)
learning_thread.start()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the live chat web interface"""
    return HTMLResponse("""
        <html>
            <head>
                <title>MillennialAi Live Chat</title>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                    }
                    .container {
                        max-width: 900px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 15px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                        margin-top: 20px;
                        margin-bottom: 20px;
                    }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        text-align: center;
                    }
                    .header h1 {
                        margin: 0;
                        font-size: 2.5em;
                        font-weight: 300;
                    }
                    .header p {
                        margin: 10px 0 0 0;
                        opacity: 0.9;
                        font-size: 1.1em;
                    }
                    .chat-container {
                        height: 600px;
                        display: flex;
                        flex-direction: column;
                    }
                    .chat-messages {
                        flex: 1;
                        padding: 20px;
                        overflow-y: auto;
                        background: #f8f9fa;
                        border-bottom: 1px solid #e9ecef;
                    }
                    .message {
                        margin-bottom: 20px;
                        padding: 15px;
                        border-radius: 10px;
                        max-width: 80%;
                        animation: fadeIn 0.3s ease-in;
                    }
                    .message.user {
                        background: #007bff;
                        color: white;
                        margin-left: auto;
                        text-align: right;
                    }
                    .message.ai {
                        background: white;
                        border: 1px solid #e9ecef;
                        color: #333;
                    }
                    .message-header {
                        font-weight: bold;
                        margin-bottom: 5px;
                        font-size: 0.9em;
                        opacity: 0.8;
                    }
                    .chat-input-container {
                        padding: 20px;
                        background: white;
                        border-top: 1px solid #e9ecef;
                    }
                    .chat-input {
                        display: flex;
                        gap: 10px;
                    }
                    #messageInput {
                        flex: 1;
                        padding: 15px;
                        border: 2px solid #e9ecef;
                        border-radius: 25px;
                        font-size: 16px;
                        outline: none;
                        transition: border-color 0.3s;
                    }
                    #messageInput:focus {
                        border-color: #667eea;
                    }
                    #sendButton {
                        padding: 15px 30px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        border-radius: 25px;
                        cursor: pointer;
                        font-size: 16px;
                        font-weight: 500;
                        transition: transform 0.2s;
                    }
                    #sendButton:hover {
                        transform: translateY(-2px);
                    }
                    #sendButton:disabled {
                        opacity: 0.6;
                        cursor: not-allowed;
                        transform: none;
                    }
                    .status {
                        text-align: center;
                        padding: 10px;
                        background: #f8f9fa;
                        border-top: 1px solid #e9ecef;
                        font-size: 0.9em;
                        color: #666;
                    }
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    .loading {
                        display: inline-block;
                        width: 20px;
                        height: 20px;
                        border: 3px solid #f3f3f3;
                        border-top: 3px solid #667eea;
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🧠 MillennialAi Live Chat</h1>
                        <p>Experience revolutionary AI with continuous learning</p>
                    </div>

                    <div class="chat-container">
                        <div id="chatMessages" class="chat-messages">
                            <div class="message ai">
                                <div class="message-header">MillennialAi</div>
                                Hello! I'm your revolutionary AI assistant with continuous learning capabilities. I combine advanced mathematical reasoning with vast knowledge integration. What would you like to discuss?
                            </div>
                        </div>

                        <div class="chat-input-container">
                            <div class="chat-input">
                                <input type="text" id="messageInput" placeholder="Type your message here..." maxlength="2000">
                                <button id="sendButton">Send</button>
                            </div>
                        </div>

                        <div id="status" class="status">
                            Ready for conversation
                        </div>
                    </div>
                </div>

                <script>
                    let conversationId = null;

                    const messageInput = document.getElementById('messageInput');
                    const sendButton = document.getElementById('sendButton');
                    const chatMessages = document.getElementById('chatMessages');
                    const status = document.getElementById('status');

                    // Send message function
                    async function sendMessage() {
                        const message = messageInput.value.trim();
                        if (!message) return;

                        // Disable input
                        messageInput.disabled = true;
                        sendButton.disabled = true;
                        sendButton.innerHTML = '<div class="loading"></div>';

                        // Add user message to chat
                        addMessage('user', 'You', message);
                        messageInput.value = '';

                        // Update status
                        status.textContent = '🧠 AI is thinking...';

                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    message: message,
                                    conversation_id: conversationId
                                })
                            });

                            const data = await response.json();

                            if (response.ok) {
                                conversationId = data.conversation_id;
                                addMessage('ai', 'MillennialAi', data.response);

                                // Show metrics
                                const metrics = `Thinking: ${data.brain_metrics.steps} steps, complexity: ${data.brain_metrics.complexity.toFixed(2)}`;
                                status.textContent = metrics;
                            } else {
                                addMessage('ai', 'System', `Error: ${data.detail || 'Unknown error'}`);
                                status.textContent = 'Error occurred';
                            }
                        } catch (error) {
                            addMessage('ai', 'System', `Network error: ${error.message}`);
                            status.textContent = 'Connection error';
                        }

                        // Re-enable input
                        messageInput.disabled = false;
                        sendButton.disabled = false;
                        sendButton.textContent = 'Send';
                        messageInput.focus();
                    }

                    // Add message to chat
                    function addMessage(type, sender, content) {
                        const messageDiv = document.createElement('div');
                        messageDiv.className = `message ${type}`;
                        messageDiv.innerHTML = `
                            <div class="message-header">${sender}</div>
                            <div>${content.replace(/\n/g, '<br>')}</div>
                        `;
                        chatMessages.appendChild(messageDiv);
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }

                    // Event listeners
                    sendButton.addEventListener('click', sendMessage);
                    messageInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });

                    // Focus input on load
                    messageInput.focus();
                </script>
            </body>
        </html>
    """)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Real-time chat with MillennialAi"""
    global conversation_count

    if not brain_available:
        raise HTTPException(status_code=503, detail="AI brain not available")

    start_time_req = time.time()

    # Generate conversation ID if not provided
    if not request.conversation_id:
        request.conversation_id = str(uuid.uuid4())
        conversation_count += 1

    try:
        # Get conversation context
        context = memory.get_conversation_context(request.conversation_id)

        # Prepare input for brain
        full_input = f"{context}\nCurrent: {request.message}"

        # Process with hybrid brain
        if hybrid_brain and hasattr(hybrid_brain, 'hybrid_think'):
            # Full hybrid processing
            result = hybrid_brain.hybrid_think(full_input, mode=request.mode)
            response = result['response']
            brain_data = result['revolutionary_analysis']
            hybrid_data = result['knowledge_enhancement']
        elif hybrid_brain and hasattr(hybrid_brain, 'think'):
            # Fallback to real brain
            result = hybrid_brain.think(full_input)
            response = result['response']
            brain_data = result
            hybrid_data = None
        else:
            raise HTTPException(status_code=503, detail="AI brain not properly initialized")

        # Add to conversation memory
        memory.add_exchange(
            request.conversation_id,
            request.message,
            response,
            brain_data,
            hybrid_data or {}
        )

        # Queue for continuous learning
        learning_sample = {
            'input': request.message,
            'response': response,
            'brain_metrics': {
                'steps': brain_data.get('steps', 0),
                'complexity': brain_data.get('complexity', 0.0),
                'reasoning_type': brain_data.get('reasoning_type', 'unknown'),
                'convergence': brain_data.get('convergence', 0.0)
            },
            'hybrid_metrics': hybrid_data or {},
            'conversation_id': request.conversation_id,
            'timestamp': datetime.now().isoformat()
        }
        continuous_learning.collect_learning_sample(learning_sample)

        processing_time = time.time() - start_time_req

        logger.info(f"Chat response generated for conversation {request.conversation_id} - {processing_time:.2f}s")

        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id,
            processing_time=processing_time,
            brain_metrics={
                'steps': brain_data.get('steps', 0),
                'complexity': brain_data.get('complexity', 0.0),
                'reasoning_type': brain_data.get('reasoning_type', 'unknown'),
                'convergence': brain_data.get('convergence', 0.0)
            },
            hybrid_metrics=hybrid_data,
            conversation_turn=len(memory.conversations[request.conversation_id])
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with detailed status"""
    ollama_status = "connected" if ollama_available else "offline"

    return HealthResponse(
        status="healthy" if brain_available else "degraded",
        brain_status="ready" if brain_available else "unavailable",
        ollama_status=ollama_status,
        timestamp=datetime.now().isoformat(),
        uptime=time.time() - start_time,
        active_conversations=len(memory.conversations),
        learning_samples_queued=learning_data_queue.qsize()
    )

@app.get("/learning/stats", response_model=LearningStatsResponse)
async def learning_stats():
    """Get continuous learning statistics"""
    stats = continuous_learning.get_learning_stats()

    return LearningStatsResponse(
        total_conversations=stats['total_conversations'],
        total_exchanges=stats['total_exchanges'],
        avg_conversation_length=stats['avg_conversation_length'],
        learning_samples_queued=stats['current_samples_count'],
        brain_available=brain_available
    )

@app.post("/learning/trigger-retraining")
async def trigger_retraining():
    """Trigger continuous learning retraining (Azure ML integration)"""
    try:
        stats = continuous_learning.get_learning_stats()

        if stats['current_samples_count'] < continuous_learning.min_samples_for_retraining:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {continuous_learning.min_samples_for_retraining} samples for retraining",
                "current_samples": stats['current_samples_count']
            }

        # Trigger retraining
        continuous_learning.trigger_retraining()

        return {
            "status": "triggered",
            "message": "Retraining job submitted to Azure ML",
            "samples_used": stats['current_samples_count'],
            "job_number": stats['retraining_jobs_submitted']
        }

    except Exception as e:
        logger.error(f"Retraining trigger error: {e}")
        return {
            "status": "error",
            "message": f"Failed to trigger retraining: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))

    print("🚀 Starting MillennialAi Live Chat API...")
    print(f"📍 Server URL: http://{HOST}:{PORT}")
    print(f"📖 Documentation: http://{HOST}:{PORT}/docs")
    print(f"🔧 Health Check: http://{HOST}:{PORT}/health")
    print(f"🧠 Brain Status: {'✅ Ready' if brain_available else '❌ Unavailable'}")
    print(f"🦙 Ollama Status: {'✅ Connected' if ollama_available else '❌ Offline'}")
    print("🎯 Live Chat Mode: Real-time conversations with continuous learning")
    print()

    uvicorn.run(app, host=HOST, port=PORT)