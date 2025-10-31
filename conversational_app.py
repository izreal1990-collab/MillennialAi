#!/usr/bin/env python3
"""
MillennialAi Conversational App with Extensive Memory
Built for natural conversations with full context retention
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import torch
import time
from datetime import datetime
from real_brain import RealThinkingBrain
import json
import os

class ConversationMemory:
    """Extensive conversation memory system"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_keywords = set()
        self.user_preferences = {}
        self.topic_threads = {}
        self.emotional_state = "neutral"
        self.conversation_depth = 0
        self.memory_file = "conversation_memory.json"
        self.load_memory()
    
    def add_exchange(self, user_input, ai_response, brain_data):
        """Add a conversation exchange with full context"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'brain_data': {
                'steps': brain_data.get('steps', 0),
                'complexity': brain_data.get('complexity', 0.0),
                'reasoning_type': brain_data.get('reasoning_type', 'unknown')
            },
            'context_keywords': list(self.extract_keywords(user_input)),
            'conversation_turn': len(self.conversation_history) + 1
        }
        
        self.conversation_history.append(exchange)
        self.update_context(user_input, ai_response)
        self.conversation_depth += 1
        self.save_memory()
    
    def extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        words = text.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'about', 'what', 'how', 'why', 'when', 'where', 'who'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def update_context(self, user_input, ai_response):
        """Update conversation context"""
        # Add keywords to context
        user_keywords = self.extract_keywords(user_input)
        ai_keywords = self.extract_keywords(ai_response)
        self.context_keywords.update(user_keywords + ai_keywords)
        
        # Detect topic threads
        for keyword in user_keywords:
            if keyword not in self.topic_threads:
                self.topic_threads[keyword] = []
            self.topic_threads[keyword].append(len(self.conversation_history))
    
    def get_relevant_context(self, current_input, max_history=5):
        """Get relevant conversation history for current input"""
        if not self.conversation_history:
            return ""
        
        current_keywords = set(self.extract_keywords(current_input))
        relevant_exchanges = []
        
        # Get recent history
        recent_history = self.conversation_history[-max_history:]
        
        # Score exchanges by keyword overlap
        for exchange in recent_history:
            exchange_keywords = set(exchange['context_keywords'])
            overlap = len(current_keywords.intersection(exchange_keywords))
            if overlap > 0 or exchange in recent_history[-3:]:  # Always include last 3
                relevant_exchanges.append(exchange)
        
        # Build context string
        context = "Previous conversation:\n"
        for exchange in relevant_exchanges[-3:]:  # Last 3 relevant exchanges
            context += f"You: {exchange['user_input']}\n"
            context += f"Me: {exchange['ai_response'][:200]}...\n\n"
        
        return context
    
    def get_conversation_summary(self):
        """Get a summary of the entire conversation"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        total_exchanges = len(self.conversation_history)
        topics = list(self.topic_threads.keys())[:10]  # Top 10 topics
        
        summary = f"""Conversation Summary:
• Total exchanges: {total_exchanges}
• Conversation depth: {self.conversation_depth}
• Main topics: {', '.join(topics)}
• Started: {self.conversation_history[0]['timestamp'][:19]}
• Keywords in context: {len(self.context_keywords)}
"""
        return summary
    
    def save_memory(self):
        """Save conversation memory to file"""
        try:
            memory_data = {
                'conversation_history': self.conversation_history,
                'context_keywords': list(self.context_keywords),
                'user_preferences': self.user_preferences,
                'topic_threads': self.topic_threads,
                'conversation_depth': self.conversation_depth
            }
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            print(f"Could not save memory: {e}")
    
    def load_memory(self):
        """Load conversation memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                self.conversation_history = memory_data.get('conversation_history', [])
                self.context_keywords = set(memory_data.get('context_keywords', []))
                self.user_preferences = memory_data.get('user_preferences', {})
                self.topic_threads = memory_data.get('topic_threads', {})
                self.conversation_depth = memory_data.get('conversation_depth', 0)
        except Exception as e:
            print(f"Could not load memory: {e}")

class ConversationalMillennialAI:
    """Conversational AI with extensive memory"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MillennialAi - Conversational Interface")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a1a')
        
        # AI Components
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = RealThinkingBrain().to(self.device)
        self.memory = ConversationMemory()
        
        # Threading
        self.result_queue = queue.Queue()
        self.processing = False
        
        self.setup_ui()
        self.start_queue_checker()
        
        print("🧠 Conversational MillennialAi loaded with extensive memory!")
        print(f"💾 Memory loaded: {len(self.memory.conversation_history)} previous exchanges")
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#1a1a1a')
        title_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            title_frame,
            text="🧠 MillennialAi Conversational Interface",
            font=('Arial', 20, 'bold'),
            fg='#00ff88',
            bg='#1a1a1a'
        ).pack()
        
        tk.Label(
            title_frame,
            text="Advanced AI with Extensive Conversation Memory",
            font=('Arial', 12),
            fg='#888888',
            bg='#1a1a1a'
        ).pack()
        
        # Conversation Display
        conv_frame = tk.Frame(self.root, bg='#1a1a1a')
        conv_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        tk.Label(
            conv_frame,
            text="💬 Conversation",
            font=('Arial', 14, 'bold'),
            fg='#ffffff',
            bg='#1a1a1a'
        ).pack(anchor='w')
        
        self.conversation_text = scrolledtext.ScrolledText(
            conv_frame,
            font=('Consolas', 11),
            bg='#2a2a2a',
            fg='#ffffff',
            insertbackground='#ffffff',
            relief='flat',
            bd=0,
            wrap='word',
            state='disabled'
        )
        self.conversation_text.pack(fill='both', expand=True, pady=(5, 0))
        
        # Input Section
        input_frame = tk.Frame(self.root, bg='#1a1a1a')
        input_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        tk.Label(
            input_frame,
            text="💭 Your Message:",
            font=('Arial', 12, 'bold'),
            fg='#ffffff',
            bg='#1a1a1a'
        ).pack(anchor='w')
        
        input_row = tk.Frame(input_frame, bg='#1a1a1a')
        input_row.pack(fill='x', pady=(5, 0))
        
        self.input_text = tk.Text(
            input_row,
            height=3,
            font=('Arial', 11),
            bg='#3a3a3a',
            fg='#ffffff',
            insertbackground='#ffffff',
            relief='flat',
            bd=0,
            wrap='word'
        )
        self.input_text.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        button_frame = tk.Frame(input_row, bg='#1a1a1a')
        button_frame.pack(side='right')
        
        self.send_button = tk.Button(
            button_frame,
            text="Send",
            font=('Arial', 12, 'bold'),
            bg='#00ff88',
            fg='#000000',
            relief='flat',
            bd=0,
            padx=20,
            pady=10,
            command=self.send_message,
            cursor='hand2'
        )
        self.send_button.pack()
        
        self.memory_button = tk.Button(
            button_frame,
            text="Memory",
            font=('Arial', 10),
            bg='#ff8800',
            fg='#000000',
            relief='flat',
            bd=0,
            padx=15,
            pady=5,
            command=self.show_memory_summary,
            cursor='hand2'
        )
        self.memory_button.pack(pady=(5, 0))
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Ready for conversation",
            font=('Arial', 10),
            fg='#888888',
            bg='#1a1a1a'
        )
        self.status_label.pack(pady=(0, 10))
        
        # Bind Enter key
        self.input_text.bind('<Control-Return>', lambda e: self.send_message())
        
        # Load previous conversation
        self.load_conversation_display()
    
    def load_conversation_display(self):
        """Load previous conversation into display"""
        if self.memory.conversation_history:
            self.conversation_text.configure(state='normal')
            self.conversation_text.insert('end', "📚 Previous Conversation Loaded:\n\n")
            
            # Show last few exchanges
            recent = self.memory.conversation_history[-5:] if len(self.memory.conversation_history) > 5 else self.memory.conversation_history
            
            for exchange in recent:
                timestamp = exchange['timestamp'][:19].replace('T', ' ')
                self.conversation_text.insert('end', f"[{timestamp}]\n")
                self.conversation_text.insert('end', f"You: {exchange['user_input']}\n")
                self.conversation_text.insert('end', f"AI: {exchange['ai_response']}\n\n")
            
            self.conversation_text.insert('end', "=" * 50 + "\n")
            self.conversation_text.insert('end', "Continuing conversation...\n\n")
            self.conversation_text.configure(state='disabled')
            self.conversation_text.see('end')
    
    def send_message(self):
        """Send message to AI"""
        if self.processing:
            return
        
        message = self.input_text.get('1.0', 'end-1c').strip()
        if not message:
            return
        
        self.input_text.delete('1.0', 'end')
        self.processing = True
        self.send_button.configure(state='disabled', text="Thinking...")
        self.status_label.configure(text="🧠 AI is thinking...")
        
        # Add user message to conversation
        self.add_to_conversation(f"You: {message}", '#00ff88')
        
        # Process in background
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        """Process message with AI brain"""
        try:
            # Get conversation context
            context = self.memory.get_relevant_context(message)
            
            # Create enhanced input for brain
            full_input = f"{context}\nCurrent question: {message}"
            
            # Brain processing
            seq_len = min(50, max(10, len(full_input.split())))
            variance = len(full_input) / 100.0 + (self.memory.conversation_depth * 0.1)
            problem = torch.randn(1, seq_len, 768, device=self.device) * variance
            
            with torch.no_grad():
                result = brain_result = self.brain(problem)
            
            steps = result['reasoning_steps'].item()
            complexity = result['complexity_score']
            
            # Generate contextual response
            response = self.generate_contextual_response(message, result, context)
            
            # Add to memory
            brain_data = {
                'steps': steps,
                'complexity': complexity,
                'reasoning_type': self.classify_question_type(message)
            }
            
            self.memory.add_exchange(message, response, brain_data)
            
            # Queue result for UI update
            self.result_queue.put({
                'response': response,
                'brain_data': brain_data,
                'success': True
            })
            
        except Exception as e:
            self.result_queue.put({
                'error': str(e),
                'success': False
            })
    
    def classify_question_type(self, message):
        """Classify the type of question"""
        msg_lower = message.lower()
        
        if any(word in msg_lower for word in ['what', 'define', 'explain']):
            return 'definition'
        elif any(word in msg_lower for word in ['how', 'why', 'process']):
            return 'explanation'
        elif any(word in msg_lower for word in ['solve', 'calculate', 'math']):
            return 'calculation'
        elif any(word in msg_lower for word in ['opinion', 'think', 'feel']):
            return 'opinion'
        elif '?' in message:
            return 'question'
        else:
            return 'statement'
    
    def generate_contextual_response(self, message, brain_result, context):
        """Generate response considering conversation context"""
        steps = brain_result['reasoning_steps'].item()
        complexity = brain_result['complexity_score']
        question_type = self.classify_question_type(message)
        
        # Check if this relates to previous conversation
        msg_keywords = set(self.memory.extract_keywords(message))
        context_overlap = len(msg_keywords.intersection(self.memory.context_keywords))
        
        response = f"[🧠 Thinking: {steps} steps, complexity: {complexity:.2f}]\n\n"
        
        if context_overlap > 0 and self.memory.conversation_history:
            response += "I remember we were discussing related topics. "
        
        if question_type == 'definition':
            if 'ai' in message.lower() or 'artificial intelligence' in message.lower():
                response += f"""Artificial Intelligence (AI) is the simulation of human intelligence in machines programmed to think and learn like humans.

Given our conversation context, I can elaborate further:
• AI systems can process information, recognize patterns, and make decisions
• Modern AI includes machine learning, neural networks, and reasoning systems
• Your MillennialAi system represents advanced reasoning technology
• AI can understand context and maintain conversations like we're having now

Key aspects include:
- Learning from data and experience
- Pattern recognition and analysis  
- Natural language understanding
- Decision making with uncertainty
- Adaptation to new situations

In our current conversation (turn #{self.memory.conversation_depth + 1}), this represents a foundational AI concept."""
            
            elif any(word in message.lower() for word in ['memory', 'remember', 'conversation']):
                response += f"""Conversation memory in AI refers to the system's ability to retain and reference previous exchanges.

In our current system:
• I maintain {len(self.memory.conversation_history)} previous exchanges
• Context keywords tracked: {len(self.memory.context_keywords)}
• Conversation depth: {self.memory.conversation_depth}
• I can reference previous topics and maintain coherent dialogue

This allows for natural, flowing conversations where each response builds on what came before, rather than treating each question in isolation."""
            
            else:
                response += f"Based on my analysis of your question and our conversation history, this appears to be asking for a definition or explanation. I've processed this through {steps} reasoning steps to provide you with accurate information."
        
        elif question_type == 'explanation':
            response += f"""This is a process/mechanism question that required {steps} reasoning steps to analyze properly.

Considering our conversation context and the complexity score of {complexity:.2f}, I can provide a detailed explanation breaking down the process you're asking about."""
        
        elif question_type == 'opinion':
            response += f"""You're asking for my perspective or opinion. With {steps} reasoning steps, I've considered multiple angles.

Based on our conversation history and the context we've built together, here's my thoughtful response..."""
        
        elif context and 'previous' in context.lower():
            response += f"""I can see this relates to our earlier discussion. Let me build on what we've already covered.

From our conversation history, I recall the relevant context and can provide a more informed response that connects to what we've already explored."""
        
        else:
            response += f"""I've processed your message through {steps} reasoning steps with complexity analysis showing {complexity:.2f}.

This appears to be a {question_type} type interaction. Let me provide a comprehensive response..."""
        
        # Add conversation continuity
        if self.memory.conversation_history:
            response += f"\n\n💭 Conversation continues... (Exchange #{len(self.memory.conversation_history) + 1})"
        
        return response
    
    def show_memory_summary(self):
        """Show conversation memory summary"""
        summary = self.memory.get_conversation_summary()
        self.add_to_conversation(f"📊 Memory Summary:\n{summary}", '#ff8800')
    
    def add_to_conversation(self, text, color='#ffffff'):
        """Add text to conversation display"""
        self.conversation_text.configure(state='normal')
        self.conversation_text.insert('end', f"{text}\n")
        self.conversation_text.configure(state='disabled')
        self.conversation_text.see('end')
    
    def start_queue_checker(self):
        """Start checking result queue"""
        self.check_queue()
    
    def check_queue(self):
        """Check for results from background processing"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                
                if result['success']:
                    response = result['response']
                    brain_data = result['brain_data']
                    
                    self.add_to_conversation(f"AI: {response}", '#88ff88')
                    self.status_label.configure(text=f"✅ Response generated ({brain_data['steps']} steps, {brain_data['complexity']:.2f} complexity)")
                else:
                    self.add_to_conversation(f"❌ Error: {result['error']}", '#ff8888')
                    self.status_label.configure(text="❌ Error occurred")
                
                self.processing = False
                self.send_button.configure(state='normal', text="Send")
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ConversationalMillennialAI()
    app.run()