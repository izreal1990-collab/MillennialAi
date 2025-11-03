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
from datetime import datetime, timedelta
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

# Performance tracking
response_times = []
total_requests = 0
failed_requests = 0
successful_requests = 0

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
    uptime: float  # Keep for backward compatibility
    uptime_seconds: float  # Android app expects this field name
    active_conversations: int
    total_processed: int  # Android expects this
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
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>MillennialAi - Advanced AI Chat</title>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
                <style>
                    :root {
                        --primary: #6366f1;
                        --primary-dark: #4f46e5;
                        --primary-light: #818cf8;
                        --secondary: #8b5cf6;
                        --bg-main: #0f0f0f;
                        --bg-chat: #212121;
                        --bg-user: #2f2f2f;
                        --bg-ai: transparent;
                        --text-primary: #ececec;
                        --text-secondary: #b4b4b4;
                        --border: #404040;
                        --shadow: rgba(0, 0, 0, 0.3);
                    }
                    
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                        background: var(--bg-main);
                        color: var(--text-primary);
                        height: 100vh;
                        overflow: hidden;
                        -webkit-font-smoothing: antialiased;
                    }
                    
                    .app-container {
                        display: flex;
                        height: 100vh;
                    }
                    
                    .sidebar {
                        width: 260px;
                        background: var(--bg-main);
                        border-right: 1px solid var(--border);
                        display: flex;
                        flex-direction: column;
                        padding: 12px;
                    }
                    
                    .new-chat-btn {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        padding: 12px 16px;
                        background: transparent;
                        border: 1px solid var(--border);
                        border-radius: 8px;
                        color: var(--text-primary);
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 500;
                        transition: all 0.2s;
                        margin-bottom: 16px;
                    }
                    
                    .new-chat-btn:hover {
                        background: var(--bg-user);
                    }
                    
                    .brand {
                        padding: 20px 16px;
                        font-size: 18px;
                        font-weight: 700;
                        background: linear-gradient(135deg, var(--primary), var(--secondary));
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-top: auto;
                    }
                    
                    .main-content {
                        flex: 1;
                        display: flex;
                        flex-direction: column;
                        max-width: 100%;
                    }
                    
                    .chat-header {
                        padding: 16px 24px;
                        border-bottom: 1px solid var(--border);
                        background: var(--bg-main);
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                    }
                    
                    .model-selector {
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        padding: 8px 16px;
                        background: var(--bg-user);
                        border-radius: 8px;
                        font-size: 14px;
                        font-weight: 500;
                    }
                    
                    .model-badge {
                        padding: 2px 8px;
                        background: var(--primary);
                        border-radius: 4px;
                        font-size: 11px;
                        font-weight: 600;
                        text-transform: uppercase;
                    }
                    
                    .chat-area {
                        flex: 1;
                        overflow-y: auto;
                        background: var(--bg-chat);
                        scroll-behavior: smooth;
                    }
                    
                    .messages-container {
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 24px;
                    }
                    
                    .message-group {
                        margin-bottom: 24px;
                        display: flex;
                        gap: 16px;
                    }
                    
                    .avatar {
                        width: 32px;
                        height: 32px;
                        border-radius: 8px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 600;
                        font-size: 14px;
                        flex-shrink: 0;
                    }
                    
                    .avatar.user {
                        background: linear-gradient(135deg, var(--primary), var(--primary-light));
                    }
                    
                    .avatar.ai {
                        background: linear-gradient(135deg, var(--secondary), var(--primary));
                    }
                    
                    .message-content {
                        flex: 1;
                        padding-top: 4px;
                    }
                    
                    .message-text {
                        color: var(--text-primary);
                        line-height: 1.6;
                        font-size: 15px;
                    }
                    
                    .input-area {
                        padding: 24px;
                        background: var(--bg-chat);
                        border-top: 1px solid var(--border);
                    }
                    
                    .input-container {
                        max-width: 800px;
                        margin: 0 auto;
                        position: relative;
                    }
                    
                    .input-wrapper {
                        background: var(--bg-user);
                        border-radius: 24px;
                        border: 1px solid var(--border);
                        display: flex;
                        align-items: flex-end;
                        padding: 12px 16px;
                        transition: border-color 0.2s;
                    }
                    
                    .input-wrapper:focus-within {
                        border-color: var(--primary);
                    }
                    
                    #messageInput {
                        flex: 1;
                        background: transparent;
                        border: none;
                        color: var(--text-primary);
                        font-size: 15px;
                        font-family: inherit;
                        outline: none;
                        resize: none;
                        max-height: 200px;
                        min-height: 24px;
                        line-height: 24px;
                        padding: 0;
                    }
                    
                    #messageInput::placeholder {
                        color: var(--text-secondary);
                    }
                    
                    #sendButton {
                        width: 32px;
                        height: 32px;
                        background: var(--primary);
                        border: none;
                        border-radius: 8px;
                        color: white;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: all 0.2s;
                        flex-shrink: 0;
                        margin-left: 8px;
                    }
                    
                    #sendButton:hover:not(:disabled) {
                        background: var(--primary-dark);
                        transform: scale(1.05);
                    }
                    
                    #sendButton:disabled {
                        opacity: 0.4;
                        cursor: not-allowed;
                    }
                    
                    .thinking {
                        display: inline-flex;
                        gap: 4px;
                        padding: 8px 0;
                    }
                    
                    .thinking span {
                        width: 6px;
                        height: 6px;
                        background: var(--text-secondary);
                        border-radius: 50%;
                        animation: bounce 1.4s infinite ease-in-out both;
                    }
                    
                    .thinking span:nth-child(1) { animation-delay: -0.32s; }
                    .thinking span:nth-child(2) { animation-delay: -0.16s; }
                    
                    @keyframes bounce {
                        0%, 80%, 100% { transform: scale(0); }
                        40% { transform: scale(1); }
                    }
                    
                    @media (max-width: 768px) {
                        .sidebar {
                            position: absolute;
                            left: -260px;
                            z-index: 100;
                            transition: left 0.3s;
                        }
                        
                        .sidebar.open {
                            left: 0;
                        }
                        
                        .messages-container,
                        .input-container {
                            padding-left: 16px;
                            padding-right: 16px;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="app-container">
                    <div class="sidebar">
                        <button class="new-chat-btn" onclick="newChat()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 5v14M5 12h14"/>
                            </svg>
                            New chat
                        </button>
                        <div class="brand">MillennialAi</div>
                    </div>
                    
                    <div class="main-content">
                <div class="chat-header">
                    <div class="model-selector">
                        <span>MillennialAi</span>
                    </div>
                </div>                        <div class="chat-area" id="chatArea">
                            <div class="messages-container" id="messagesContainer">
                                <div class="message-group">
                                    <div class="avatar ai">AI</div>
                                    <div class="message-content">
                                        <div class="message-text">Hello! How can I assist you today?</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="input-area">
                            <div class="input-container">
                                <div class="input-wrapper">
                                    <textarea 
                                        id="messageInput" 
                                        placeholder="Message MillennialAi..." 
                                        rows="1"
                                        maxlength="2000"
                                    ></textarea>
                                    <button id="sendButton" type="button">
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                            <path d="M7 11L12 6L17 11M12 18V7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                        </svg>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <script>
                    let conversationId = null;
                    const messageInput = document.getElementById('messageInput');
                    const sendButton = document.getElementById('sendButton');
                    const messagesContainer = document.getElementById('messagesContainer');
                    const chatArea = document.getElementById('chatArea');
                    
                    // Auto-resize textarea
                    messageInput.addEventListener('input', function() {
                        this.style.height = 'auto';
                        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
                    });
                    
                    async function sendMessage() {
                        const message = messageInput.value.trim();
                        if (!message || sendButton.disabled) return;
                        
                        // Disable send
                        sendButton.disabled = true;
                        const userMessage = message;
                        messageInput.value = '';
                        messageInput.style.height = 'auto';
                        
                        // Add user message
                        addMessage('user', userMessage);
                        
                        // Add thinking indicator
                        const thinkingId = addThinking();
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ 
                                    message: userMessage, 
                                    conversation_id: conversationId 
                                })
                            });
                            
                            const data = await response.json();
                            
                            // Remove thinking indicator
                            removeThinking(thinkingId);
                            
                            if (response.ok) {
                                conversationId = data.conversation_id;
                                addMessage('ai', data.response);
                            } else {
                                addMessage('ai', 'Sorry, I encountered an error. Please try again.');
                            }
                        } catch (error) {
                            removeThinking(thinkingId);
                            addMessage('ai', 'Connection error. Please check your internet and try again.');
                        }
                        
                        sendButton.disabled = false;
                        messageInput.focus();
                    }
                    
                    function addMessage(type, content) {
                        const messageGroup = document.createElement('div');
                        messageGroup.className = 'message-group';
                        
                        const avatar = document.createElement('div');
                        avatar.className = `avatar ${type}`;
                        avatar.textContent = type === 'user' ? 'You' : 'AI';
                        
                        const messageContent = document.createElement('div');
                        messageContent.className = 'message-content';
                        
                        const messageText = document.createElement('div');
                        messageText.className = 'message-text';
                        messageText.innerHTML = content.replace(/\\n/g, '<br>');
                        
                        messageContent.appendChild(messageText);
                        messageGroup.appendChild(avatar);
                        messageGroup.appendChild(messageContent);
                        messagesContainer.appendChild(messageGroup);
                        
                        chatArea.scrollTop = chatArea.scrollHeight;
                    }
                    
                    function addThinking() {
                        const id = 'thinking-' + Date.now();
                        const messageGroup = document.createElement('div');
                        messageGroup.className = 'message-group';
                        messageGroup.id = id;
                        
                        const avatar = document.createElement('div');
                        avatar.className = 'avatar ai';
                        avatar.textContent = 'AI';
                        
                        const messageContent = document.createElement('div');
                        messageContent.className = 'message-content';
                        
                        const thinking = document.createElement('div');
                        thinking.className = 'thinking';
                        thinking.innerHTML = '<span></span><span></span><span></span>';
                        
                        messageContent.appendChild(thinking);
                        messageGroup.appendChild(avatar);
                        messageGroup.appendChild(messageContent);
                        messagesContainer.appendChild(messageGroup);
                        
                        chatArea.scrollTop = chatArea.scrollHeight;
                        return id;
                    }
                    
                    function removeThinking(id) {
                        const element = document.getElementById(id);
                        if (element) element.remove();
                    }
                    
                    function newChat() {
                        conversationId = null;
                        messagesContainer.innerHTML = `
                            <div class="message-group">
                                <div class="avatar ai">AI</div>
                                <div class="message-content">
                                    <div class="message-text">Hello! How can I assist you today?</div>
                                </div>
                            </div>
                        `;
                    }
                    
                    // Event listeners
                    sendButton.addEventListener('click', sendMessage);
                    
                    messageInput.addEventListener('keydown', (e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });
                    
                    messageInput.focus();
                </script>
            </body>
        </html>
    """)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Real-time chat with MillennialAi"""
    global conversation_count, total_requests, successful_requests, failed_requests

    # Track ALL requests
    total_requests += 1
    
    if not brain_available:
        failed_requests += 1
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
            failed_requests += 1
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
        
        # Track REAL response time and success
        response_times.append(processing_time)
        successful_requests += 1

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
        failed_requests += 1
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with detailed status"""
    ollama_status = "connected" if ollama_available else "offline"
    uptime_value = time.time() - start_time
    total_exchanges = sum(len(conv) for conv in memory.conversations.values())

    return HealthResponse(
        status="healthy" if brain_available else "degraded",
        brain_status="ready" if brain_available else "unavailable",
        ollama_status=ollama_status,
        timestamp=datetime.now().isoformat(),
        uptime=uptime_value,  # Keep for backward compatibility
        uptime_seconds=uptime_value,  # Android app expects this
        active_conversations=len(memory.conversations),
        total_processed=total_exchanges,  # Android expects this
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

@app.get("/metrics")
async def get_metrics():
    """EXTENSIVE ML Performance Metrics for Android monitoring app"""
    total_exchanges = sum(len(conv) for conv in memory.conversations.values())
    recent_conversations = [
        conv for conv in memory.conversations.values() 
        if conv and datetime.fromisoformat(conv[-1]['timestamp']) > datetime.now() - timedelta(hours=24)
    ]
    
    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        cpu_percent = process.cpu_percent(interval=0.1)
    except Exception:
        memory_usage_mb = 0.0
        memory_percent = 0.0
        cpu_percent = 0.0
    
    # ML Brain metrics
    brain_available = hybrid_brain is not None
    brain_status = "ready" if brain_available else "offline"
    brain_layers = 0
    layer_stats = []
    
    if brain_available:
        try:
            # Get brain layers count from HybridRevolutionaryBrain
            if hasattr(hybrid_brain, 'real_brain'):
                real = getattr(hybrid_brain, 'real_brain', None)
                if real and hasattr(real, 'brain'):
                    brain_obj = getattr(real, 'brain', None)
                    if brain_obj and hasattr(brain_obj, 'layers'):
                        brain_layers = len(brain_obj.layers)
                        # Get detailed layer stats
                        for i, layer in enumerate(brain_obj.layers):
                            if hasattr(layer, 'weight') and hasattr(layer.weight, 'shape'):
                                weight_shape = list(layer.weight.shape)
                                total_params = layer.weight.numel() if hasattr(layer.weight, 'numel') else 0
                                layer_stats.append({
                                    "layer_index": i,
                                    "layer_type": layer.__class__.__name__,
                                    "parameters": total_params,
                                    "shape": weight_shape
                                })
        except Exception as e:
            logger.warning(f"Error getting brain layer stats: {e}")
    
    brain_load = min(100, (len(memory.conversations) / 50) * 100)
    
    # Learning system metrics
    learning_queue_size = learning_data_queue.qsize()
    samples_processed = 0
    learning_active = False
    
    # Conversation quality metrics
    avg_conversation_length = sum(len(conv) for conv in memory.conversations.values()) / max(len(memory.conversations), 1)
    
    # REAL tracked response time - calculate from actual measurements
    avg_response_time = (sum(response_times) / len(response_times)) if response_times else 0.0
    
    # REAL success rate - calculate from actual request counts
    calculated_success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 100.0
    
    # REAL uptime - calculate from start_time global
    uptime_seconds = time.time() - start_time
    
    return {
        # Core metrics
        "timestamp": datetime.now().isoformat(),
        "total_requests": total_requests,  # Use global counter, not total_exchanges
        "active_conversations": len(memory.conversations),
        "conversations_24h": len(recent_conversations),
        "avg_conversation_length": round(avg_conversation_length, 2),
        
        # ML Brain metrics
        "brain_status": brain_status,
        "brain_load_percentage": round(brain_load, 2),
        "brain_layers_count": brain_layers,
        "brain_total_parameters": sum(ls["parameters"] for ls in layer_stats) if layer_stats else 0,
        "layer_details": layer_stats,
        
        # Learning system
        "learning_active": learning_active,
        "learning_queue_size": learning_queue_size,
        "learning_samples_processed": samples_processed,
        "learning_threshold": 10000,
        "learning_progress_percentage": min(100, (samples_processed / 10000) * 100) if samples_processed else 0,
        
        # Performance metrics - ALL REAL VALUES
        "avg_response_time_sec": round(avg_response_time, 3),
        "memory_usage_mb": round(memory_usage_mb, 2),
        "memory_percent": round(memory_percent, 2),
        "cpu_percent": round(cpu_percent, 2),
        "success_rate": round(calculated_success_rate, 2),
        
        # Context window
        "max_context_tokens": 4096,
        "avg_context_usage_percentage": round((avg_conversation_length / 4096) * 100, 2),
        
        # Uptime and health - REAL VALUE
        "uptime_seconds": round(uptime_seconds, 2),
        "health_status": "healthy"
    }

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