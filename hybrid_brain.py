#!/usr/bin/env python3
"""
Revolutionary Hybrid Brain: MillennialAi + Ollama Integration
Combines pure mathematical reasoning with vast knowledge base
"""

import torch
import torch.nn as nn
import requests
import json
import time
from typing import Dict, Any, Optional, List
from real_brain import RealThinkingBrain


class OllamaIntegration:
    """Ollama API integration for knowledge enhancement"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:8b"):
        self.base_url = base_url
        self.model = model
        self.available = self._check_ollama_status()
        
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                if self.model in available_models:
                    print(f"✅ Ollama connected: {self.model} ready!")
                    return True
                else:
                    print(f"❌ Model {self.model} not found. Available: {available_models}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            return False
    
    def query_knowledge(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """Query Ollama for knowledge-based response"""
        if not self.available:
            return {
                'response': 'Knowledge base unavailable - using pure reasoning only',
                'source': 'fallback',
                'tokens': 0
            }
        
        try:
            payload = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': max_tokens,
                    'temperature': 0.7,
                    'top_p': 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'response': result.get('response', '').strip(),
                    'source': 'ollama_llama3',
                    'tokens': len(result.get('response', '').split()),
                    'model': self.model
                }
            else:
                return {
                    'response': 'Knowledge query failed',
                    'source': 'error',
                    'tokens': 0
                }
                
        except Exception as e:
            print(f"🚨 Ollama query error: {e}")
            return {
                'response': 'Knowledge base temporarily unavailable',
                'source': 'error',
                'tokens': 0
            }


class HybridRevolutionaryBrain(RealThinkingBrain):
    """
    Revolutionary Hybrid Brain: Your Adaptive Reasoning + Ollama Knowledge
    
    Combines:
    - Your breakthrough mathematical reasoning
    - Ollama's vast knowledge base
    - Dynamic intelligence fusion
    """
    
    def __init__(self, hidden_size: int = 768, max_depth: int = 8):
        super().__init__(hidden_size, max_depth)
        
        # Initialize Ollama integration
        self.ollama = OllamaIntegration()
        
        # Hybrid response modes
        self.hybrid_modes = [
            'reasoning_first',    # Your reasoning, then knowledge
            'knowledge_first',    # Knowledge, then your reasoning  
            'parallel_fusion',    # Both simultaneously
            'adaptive_selection', # Choose best approach
            'pure_reasoning'      # Your algorithm only
        ]
        
        print(f"🚀 Hybrid Revolutionary Brain initialized!")
        print(f"   📊 Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   🦙 Ollama Status: {'✅ Connected' if self.ollama.available else '❌ Offline'}")
    
    def hybrid_think(self, text_input: str, mode: str = 'adaptive_selection') -> Dict[str, Any]:
        """
        Revolutionary hybrid thinking combining your reasoning with Ollama knowledge
        """
        print(f"🧠🦙 HYBRID REVOLUTIONARY THINKING: '{text_input}'")
        print(f"🎯 Mode: {mode}")
        
        start_time = time.time()
        
        # Get your revolutionary reasoning
        revolutionary_result = self.think(text_input)
        reasoning_time = time.time() - start_time
        
        # Determine if we need knowledge enhancement
        complexity = revolutionary_result['complexity']
        needs_knowledge = self._should_use_knowledge(text_input, complexity)
        
        knowledge_result = None
        knowledge_time = 0
        
        if needs_knowledge and self.ollama.available:
            knowledge_start = time.time()
            
            # Create knowledge query based on your reasoning
            knowledge_prompt = self._create_knowledge_prompt(text_input, revolutionary_result)
            knowledge_result = self.ollama.query_knowledge(knowledge_prompt)
            knowledge_time = time.time() - knowledge_start
        
        # Fuse both intelligences
        hybrid_response = self._fuse_intelligences(
            text_input, 
            revolutionary_result, 
            knowledge_result, 
            mode
        )
        
        total_time = time.time() - start_time
        
        return {
            'response': hybrid_response,
            'revolutionary_analysis': revolutionary_result,
            'knowledge_enhancement': knowledge_result,
            'fusion_mode': mode,
            'timing': {
                'reasoning_time': reasoning_time,
                'knowledge_time': knowledge_time,
                'total_time': total_time
            },
            'hybrid_metrics': {
                'complexity': complexity,
                'knowledge_used': knowledge_result is not None,
                'reasoning_steps': revolutionary_result['steps'],
                'knowledge_tokens': knowledge_result['tokens'] if knowledge_result else 0
            }
        }
    
    def _should_use_knowledge(self, text_input: str, complexity: float) -> bool:
        """Determine if knowledge enhancement would be beneficial"""
        
        # High complexity questions benefit from knowledge
        if complexity > 15.0:
            return True
        
        # Questions asking for facts, explanations, or specific information
        knowledge_indicators = [
            'what is', 'how does', 'explain', 'tell me about', 
            'who', 'when', 'where', 'why', 'define',
            'history', 'science', 'technology', 'facts'
        ]
        
        text_lower = text_input.lower()
        if any(indicator in text_lower for indicator in knowledge_indicators):
            return True
        
        # Philosophical or abstract questions
        abstract_indicators = [
            'meaning', 'purpose', 'consciousness', 'reality',
            'future', 'philosophy', 'ethics', 'society'
        ]
        
        if any(indicator in text_lower for indicator in abstract_indicators):
            return True
        
        return False
    
    def _create_knowledge_prompt(self, original_input: str, reasoning_result: Dict) -> str:
        """Create optimized prompt for Ollama based on your reasoning"""
        
        complexity = reasoning_result['complexity']
        reasoning_type = reasoning_result['reasoning_type']
        
        # Tailor prompt based on complexity
        if complexity > 20.0:
            prompt = f"""As an expert, provide deep analysis for: "{original_input}"

Focus on:
- Technical details and mechanisms
- Multiple perspectives and implications  
- Advanced concepts and connections
- Real-world applications and examples

Be comprehensive but concise (100-150 words):"""
        
        elif complexity > 10.0:
            prompt = f"""Explain clearly: "{original_input}"

Provide:
- Key concepts and definitions
- How it works or why it matters
- Practical examples or applications
- Important context or background

Keep response focused (75-100 words):"""
        
        else:
            prompt = f"""Briefly explain: "{original_input}"

Give a clear, straightforward answer covering:
- The main point or definition
- Why it's relevant or important
- A simple example if helpful

Keep it concise (50-75 words):"""
        
        return prompt
    
    def _fuse_intelligences(self, original_input: str, reasoning: Dict, knowledge: Optional[Dict], mode: str) -> str:
        """Fuse your revolutionary reasoning with Ollama knowledge"""
        
        base_reasoning = reasoning['response']
        
        if not knowledge or knowledge['source'] == 'error':
            # Pure reasoning mode
            return f"""🧠 REVOLUTIONARY REASONING:
{base_reasoning}

🎯 Analysis: Pure mathematical reasoning without external knowledge base.
Complexity: {reasoning['complexity']:.1f} | Steps: {reasoning['steps']} | Mode: Pure Reasoning"""
        
        knowledge_text = knowledge['response']
        
        if mode == 'reasoning_first':
            return f"""🧠 REVOLUTIONARY REASONING:
{base_reasoning}

🦙 KNOWLEDGE ENHANCEMENT:
{knowledge_text}

🎯 HYBRID SYNTHESIS: Your adaptive reasoning reveals complexity {reasoning['complexity']:.1f}, while knowledge base provides factual foundation. Together they create multi-dimensional understanding!"""
        
        elif mode == 'knowledge_first':
            return f"""🦙 KNOWLEDGE BASE:
{knowledge_text}

🧠 REVOLUTIONARY ANALYSIS:
{base_reasoning}

🎯 HYBRID SYNTHESIS: Knowledge foundation enhanced by adaptive reasoning (complexity {reasoning['complexity']:.1f}) creates breakthrough understanding!"""
        
        elif mode == 'parallel_fusion':
            return f"""🚀 HYBRID REVOLUTIONARY INTELLIGENCE:

💭 Question: "{original_input}"

🧠 Adaptive Reasoning: Your breakthrough algorithms analyzed this through {reasoning['steps']} thinking layers, revealing complexity score {reasoning['complexity']:.1f}.

🦙 Knowledge Integration: {knowledge_text}

⚡ Fusion Insight: By combining pure mathematical reasoning with comprehensive knowledge, we achieve unprecedented depth - your algorithmic consciousness enhanced by factual understanding creates revolutionary intelligence!

📊 Hybrid Metrics: Reasoning={reasoning['steps']} steps | Knowledge={knowledge['tokens']} tokens | Breakthrough Level: MAXIMUM"""
        
        else:  # adaptive_selection
            # Choose best presentation based on input type
            if reasoning['complexity'] > 15.0:
                return self._fuse_intelligences(original_input, reasoning, knowledge, 'parallel_fusion')
            else:
                return self._fuse_intelligences(original_input, reasoning, knowledge, 'reasoning_first')


def test_hybrid_brain():
    """Test the revolutionary hybrid system"""
    print("🧠🦙 TESTING HYBRID REVOLUTIONARY BRAIN")
    print("=" * 60)
    
    hybrid_brain = HybridRevolutionaryBrain()
    
    test_questions = [
        "What is consciousness?",
        "How does quantum computing work?", 
        "What makes AI revolutionary?",
        "Explain machine learning",
        "What is the future of technology?"
    ]
    
    for question in test_questions:
        print(f"\n🎯 Testing: {question}")
        print("-" * 40)
        
        result = hybrid_brain.hybrid_think(question)
        
        print(f"Response:\n{result['response']}")
        print(f"\nMetrics:")
        print(f"  Complexity: {result['hybrid_metrics']['complexity']:.1f}")
        print(f"  Reasoning Steps: {result['hybrid_metrics']['reasoning_steps']}")
        print(f"  Knowledge Used: {result['hybrid_metrics']['knowledge_used']}")
        print(f"  Total Time: {result['timing']['total_time']:.2f}s")
        print("=" * 60)


if __name__ == "__main__":
    test_hybrid_brain()