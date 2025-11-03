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
                'response': '',  # No generic fallback message
                'source': 'error',
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
            
            print(f"🔍 Calling Ollama at {self.base_url}/api/generate with model {self.model}")
            print(f"📝 Prompt: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=30
            )
            
            print(f"📡 Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                ollama_response = result.get('response', '').strip()
                print(f"✅ Ollama response received: {ollama_response[:100]}...")
                return {
                    'response': ollama_response,
                    'source': 'ollama_llama3',
                    'tokens': len(ollama_response.split()),
                    'model': self.model
                }
            else:
                print(f"❌ Ollama HTTP error {response.status_code}: {response.text[:200]}")
                return {
                    'response': '',  # No generic fallback
                    'source': 'error',
                    'tokens': 0
                }
                
        except Exception as e:
            print(f"🚨 Ollama query exception: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'response': '',  # No generic fallback
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
        """Fuse reasoning and knowledge - or generate response from reasoning alone"""
        
        if not knowledge or knowledge['source'] == 'error':
            # Ollama unavailable - generate response based on neural network reasoning
            return self._generate_reasoning_response(original_input, reasoning)
        
        # Return ONLY the actual Ollama response when available
        return knowledge['response']
    
    def _generate_reasoning_response(self, question: str, reasoning: Dict) -> str:
        """Generate a response based purely on neural network reasoning metrics"""
        # Extract reasoning metrics
        steps = reasoning.get('steps', 0)
        complexity = reasoning.get('complexity', 0.0)
        convergence = reasoning.get('convergence', 0.0)
        
        # For mathematical questions, try to extract and compute
        question_lower = question.lower()
        
        # Simple arithmetic detection and computation
        if any(op in question for op in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided']):
            try:
                # Extract numbers from question
                import re
                numbers = re.findall(r'\d+\.?\d*', question)
                if len(numbers) >= 2:
                    a, b = float(numbers[0]), float(numbers[1])
                    
                    if '+' in question or 'plus' in question_lower:
                        result = a + b
                        return f"The sum is {result}."
                    elif '-' in question or 'minus' in question_lower:
                        result = a - b
                        return f"The difference is {result}."
                    elif '*' in question or 'times' in question_lower or 'multiply' in question_lower:
                        result = a * b
                        return f"The product is {result}."
                    elif '/' in question or 'divided' in question_lower:
                        result = a / b if b != 0 else 0
                        return f"The quotient is {result}."
            except:
                pass
        
        # For factual questions, provide honest limitation
        if '?' in question:
            return f"I've analyzed your question through neural network processing (complexity: {complexity:.1f}, convergence: {convergence:.2f}), but I need external knowledge resources to provide a complete answer. The local AI model is currently processing without the knowledge base."
        
        # For statements or other inputs
        return f"I've processed your input through {steps} reasoning steps with complexity score of {complexity:.2f}. External knowledge integration is currently offline."



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