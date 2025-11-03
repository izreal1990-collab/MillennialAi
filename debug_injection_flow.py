#!/usr/bin/env python3
"""
DEEP INJECTION POINT DIAGNOSTIC SCRIPT
======================================
Analyzes the complete forward pass through:
1. Real Brain reasoning (real_brain.py)
2. Ollama Integration (hybrid_brain.py)
3. Revolutionary Layer Injection Framework
4. Token Reasoning Module (TRM) flow
"""

import torch
import time
import json
import requests
from typing import Dict, Any
import traceback

# Import components
from real_brain import RealThinkingBrain
from hybrid_brain import HybridRevolutionaryBrain, OllamaIntegration

class InjectionFlowDebugger:
    """Deep diagnostic tool for analyzing injection point performance"""
    
    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'tests': [],
            'bottlenecks': [],
            'recommendations': []
        }
        
    def test_real_brain_forward(self):
        """Test 1: Real Brain forward pass performance"""
        print("\n" + "="*70)
        print("TEST 1: REAL BRAIN FORWARD PASS")
        print("="*70)
        
        brain = RealThinkingBrain(hidden_size=768, max_depth=8)
        
        test_cases = [
            ("Minimal", torch.randn(1, 5, 768)),
            ("Small", torch.randn(1, 10, 768)),
            ("Medium", torch.randn(1, 20, 768)),
            ("Large", torch.randn(1, 50, 768))
        ]
        
        for name, input_tensor in test_cases:
            print(f"\n🔍 Testing {name} input: {input_tensor.shape}")
            
            start = time.time()
            with torch.no_grad():
                result = brain.forward(input_tensor)
            forward_time = time.time() - start
            
            steps = result['reasoning_steps'].item()
            complexity = result['complexity_score']
            
            print(f"   ⏱️  Forward pass: {forward_time:.4f}s")
            print(f"   📊 Complexity: {complexity:.3f}")
            print(f"   🔢 Steps taken: {steps}")
            print(f"   ⚡ Speed: {forward_time/steps:.4f}s per step")
            
            self.results['tests'].append({
                'test': 'real_brain_forward',
                'input_size': name,
                'time': forward_time,
                'steps': steps,
                'complexity': complexity,
                'speed_per_step': forward_time/steps
            })
            
            if forward_time > 1.0:
                self.results['bottlenecks'].append({
                    'location': 'RealThinkingBrain.forward',
                    'issue': f'{name} input took {forward_time:.2f}s',
                    'severity': 'HIGH' if forward_time > 2.0 else 'MEDIUM'
                })
    
    def test_ollama_direct(self):
        """Test 2: Direct Ollama API performance"""
        print("\n" + "="*70)
        print("TEST 2: OLLAMA API DIRECT CALL")
        print("="*70)
        
        ollama = OllamaIntegration(base_url="http://localhost:11434", model="llama3:8b")
        
        if not ollama.available:
            print("❌ Ollama not available - skipping test")
            self.results['bottlenecks'].append({
                'location': 'OllamaIntegration',
                'issue': 'Ollama service not running',
                'severity': 'CRITICAL'
            })
            return
        
        test_prompts = [
            ("Tiny", "Hi", 10),
            ("Short", "What is 2+2?", 20),
            ("Medium", "Explain photosynthesis in one sentence.", 50),
            ("Long", "Describe the theory of relativity in detail.", 150)
        ]
        
        for name, prompt, max_tokens in test_prompts:
            print(f"\n🔍 Testing {name} prompt: '{prompt[:50]}...'")
            print(f"   Max tokens: {max_tokens}")
            
            start = time.time()
            result = ollama.query_knowledge(prompt, max_tokens=max_tokens)
            query_time = time.time() - start
            
            print(f"   ⏱️  Query time: {query_time:.4f}s")
            print(f"   📡 Source: {result['source']}")
            print(f"   🔢 Tokens: {result['tokens']}")
            
            if result['source'] == 'error':
                print(f"   ❌ ERROR - no response received")
            else:
                print(f"   ✅ Response: {result['response'][:100]}...")
                print(f"   ⚡ Speed: {result['tokens']/query_time:.2f} tokens/sec")
            
            self.results['tests'].append({
                'test': 'ollama_direct',
                'prompt_size': name,
                'time': query_time,
                'tokens': result['tokens'],
                'max_tokens': max_tokens,
                'source': result['source'],
                'tokens_per_sec': result['tokens']/query_time if result['tokens'] > 0 else 0
            })
            
            # Performance analysis
            if query_time > 30:
                self.results['bottlenecks'].append({
                    'location': 'Ollama API',
                    'issue': f'{name} prompt took {query_time:.2f}s (timeout risk)',
                    'severity': 'CRITICAL'
                })
            elif query_time > 10:
                self.results['bottlenecks'].append({
                    'location': 'Ollama API',
                    'issue': f'{name} prompt took {query_time:.2f}s (slow)',
                    'severity': 'HIGH'
                })
            
            # Token generation speed analysis
            if result['tokens'] > 0:
                tokens_per_sec = result['tokens'] / query_time
                if tokens_per_sec < 1:
                    self.results['bottlenecks'].append({
                        'location': 'Ollama inference',
                        'issue': f'Very slow: {tokens_per_sec:.2f} tokens/sec',
                        'severity': 'CRITICAL'
                    })
                elif tokens_per_sec < 5:
                    self.results['bottlenecks'].append({
                        'location': 'Ollama inference',
                        'issue': f'Slow: {tokens_per_sec:.2f} tokens/sec',
                        'severity': 'HIGH'
                    })
    
    def test_hybrid_integration(self):
        """Test 3: Full hybrid brain integration"""
        print("\n" + "="*70)
        print("TEST 3: HYBRID BRAIN INTEGRATION")
        print("="*70)
        
        hybrid = HybridRevolutionaryBrain(hidden_size=768, max_depth=8)
        
        test_inputs = [
            ("Simple Math", "What is 5+3?"),
            ("Factual", "What is the capital of France?"),
            ("Explanation", "How does photosynthesis work?")
        ]
        
        for name, text_input in test_inputs:
            print(f"\n🔍 Testing: {name}")
            print(f"   Input: '{text_input}'")
            
            start = time.time()
            result = hybrid.hybrid_think(text_input, mode='parallel_fusion')
            total_time = time.time() - start
            
            print(f"   ⏱️  Total time: {total_time:.4f}s")
            print(f"   🧠 Reasoning time: {result['timing']['reasoning_time']:.4f}s")
            print(f"   🦙 Knowledge time: {result['timing']['knowledge_time']:.4f}s")
            print(f"   📊 Complexity: {result['revolutionary_analysis']['complexity']:.3f}")
            print(f"   📝 Response: {result['response'][:100]}...")
            
            # Check if knowledge was used
            if result['knowledge_enhancement'] and result['knowledge_enhancement']['source'] == 'ollama_llama3':
                print(f"   ✅ Ollama responded successfully")
            else:
                print(f"   ❌ Ollama failed - used fallback")
            
            self.results['tests'].append({
                'test': 'hybrid_integration',
                'input_type': name,
                'total_time': total_time,
                'reasoning_time': result['timing']['reasoning_time'],
                'knowledge_time': result['timing']['knowledge_time'],
                'complexity': result['revolutionary_analysis']['complexity'],
                'knowledge_used': result['knowledge_enhancement'] is not None
            })
            
            # Analyze timing breakdown
            reasoning_pct = (result['timing']['reasoning_time'] / total_time) * 100
            knowledge_pct = (result['timing']['knowledge_time'] / total_time) * 100
            
            print(f"   📊 Time breakdown:")
            print(f"      - Reasoning: {reasoning_pct:.1f}%")
            print(f"      - Knowledge: {knowledge_pct:.1f}%")
            
            if knowledge_pct > 80:
                self.results['bottlenecks'].append({
                    'location': 'Hybrid integration',
                    'issue': f'Knowledge query dominates ({knowledge_pct:.1f}% of time)',
                    'severity': 'HIGH'
                })
    
    def test_injection_points(self):
        """Test 4: Revolutionary Layer Injection Framework"""
        print("\n" + "="*70)
        print("TEST 4: LAYER INJECTION FRAMEWORK")
        print("="*70)
        
        try:
            from revolutionary_layer_injection_framework import (
                RevolutionaryLayerInjectionFramework,
                InjectionMode,
                InjectionConfig
            )
            
            framework = RevolutionaryLayerInjectionFramework(InjectionMode.SIMULATION)
            
            test_query = "Explain quantum entanglement"
            
            print(f"\n🔍 Testing injection framework")
            print(f"   Query: '{test_query}'")
            
            start = time.time()
            result = framework.inject_and_enhance(
                test_query,
                llm_endpoint="http://localhost:11434",
                llm_architecture="llama"
            )
            injection_time = time.time() - start
            
            print(f"   ⏱️  Injection time: {injection_time:.4f}s")
            print(f"   📊 Complexity: {result.complexity_score:.3f}")
            print(f"   📍 Injection points: {result.injection_points}")
            print(f"   ✅ Success: {result.enhancement_success}")
            
            self.results['tests'].append({
                'test': 'injection_framework',
                'time': injection_time,
                'complexity': result.complexity_score,
                'injection_points': result.injection_points,
                'success': result.enhancement_success
            })
            
            if injection_time > 60:
                self.results['bottlenecks'].append({
                    'location': 'Layer Injection Framework',
                    'issue': f'Injection took {injection_time:.2f}s',
                    'severity': 'CRITICAL'
                })
                
        except Exception as e:
            print(f"   ❌ Framework test failed: {e}")
            traceback.print_exc()
            self.results['bottlenecks'].append({
                'location': 'Layer Injection Framework',
                'issue': f'Framework error: {str(e)}',
                'severity': 'CRITICAL'
            })
    
    def test_ollama_settings(self):
        """Test 5: Ollama configuration and model settings"""
        print("\n" + "="*70)
        print("TEST 5: OLLAMA CONFIGURATION ANALYSIS")
        print("="*70)
        
        try:
            # Check Ollama server status
            print("\n🔍 Checking Ollama server...")
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                print(f"   ✅ Ollama server running")
                print(f"   📦 Available models: {len(models)}")
                
                for model in models:
                    print(f"\n   📋 Model: {model['name']}")
                    print(f"      Size: {model.get('size', 'unknown')}")
                    print(f"      Modified: {model.get('modified_at', 'unknown')}")
                    
                    # Check if it's llama3:8b
                    if 'llama3' in model['name'].lower():
                        # Try to get model info
                        try:
                            info_response = requests.post(
                                "http://localhost:11434/api/show",
                                json={"name": model['name']},
                                timeout=10
                            )
                            if info_response.status_code == 200:
                                info = info_response.json()
                                print(f"      Parameters: {info.get('parameters', {})}")
                                print(f"      Template: {info.get('template', 'default')[:100]}...")
                        except Exception as e:
                            print(f"      ⚠️ Could not get model info: {e}")
            else:
                print(f"   ❌ Ollama server error: {response.status_code}")
                self.results['bottlenecks'].append({
                    'location': 'Ollama server',
                    'issue': f'HTTP {response.status_code}',
                    'severity': 'CRITICAL'
                })
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Cannot connect to Ollama: {e}")
            self.results['bottlenecks'].append({
                'location': 'Ollama server',
                'issue': f'Connection failed: {str(e)}',
                'severity': 'CRITICAL'
            })
    
    def analyze_and_recommend(self):
        """Analyze results and provide recommendations"""
        print("\n" + "="*70)
        print("ANALYSIS & RECOMMENDATIONS")
        print("="*70)
        
        # Analyze bottlenecks
        if self.results['bottlenecks']:
            print("\n⚠️  BOTTLENECKS DETECTED:")
            for i, bottleneck in enumerate(self.results['bottlenecks'], 1):
                severity_emoji = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                print(f"\n{i}. {severity_emoji.get(bottleneck['severity'], '⚪')} [{bottleneck['severity']}] {bottleneck['location']}")
                print(f"   Issue: {bottleneck['issue']}")
        else:
            print("\n✅ No significant bottlenecks detected!")
        
        # Generate recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        # Check Ollama performance
        ollama_tests = [t for t in self.results['tests'] if t['test'] == 'ollama_direct']
        if ollama_tests:
            avg_tokens_per_sec = sum(t['tokens_per_sec'] for t in ollama_tests) / len(ollama_tests)
            
            if avg_tokens_per_sec < 1:
                print("\n1. ⚡ CRITICAL: Ollama is extremely slow")
                print("   - Current speed: {:.2f} tokens/sec".format(avg_tokens_per_sec))
                print("   - Expected speed: 10-50 tokens/sec")
                print("   - Solutions:")
                print("     • Enable GPU acceleration (NVIDIA/AMD)")
                print("     • Use smaller model (llama3:3b instead of llama3:8b)")
                print("     • Increase CPU cores (currently 4, try dedicated server)")
                print("     • Use quantized model (Q4_0 or Q4_K_M)")
                print("     • Consider using API instead (OpenAI, Anthropic)")
                self.results['recommendations'].append({
                    'priority': 'CRITICAL',
                    'issue': 'Ollama CPU inference too slow',
                    'solution': 'Enable GPU or use smaller/quantized model'
                })
            elif avg_tokens_per_sec < 5:
                print("\n1. ⚠️  Ollama performance needs improvement")
                print("   - Current speed: {:.2f} tokens/sec".format(avg_tokens_per_sec))
                print("   - Solutions:")
                print("     • Consider GPU acceleration")
                print("     • Try quantized model for faster inference")
                self.results['recommendations'].append({
                    'priority': 'HIGH',
                    'issue': 'Ollama slow on CPU',
                    'solution': 'Use GPU or quantized model'
                })
        
        # Check hybrid integration
        hybrid_tests = [t for t in self.results['tests'] if t['test'] == 'hybrid_integration']
        if hybrid_tests:
            knowledge_failures = sum(1 for t in hybrid_tests if not t['knowledge_used'])
            if knowledge_failures > 0:
                print(f"\n2. ❌ Knowledge integration failing ({knowledge_failures}/{len(hybrid_tests)} tests)")
                print("   - Solutions:")
                print("     • Increase Ollama timeout (currently 120s)")
                print("     • Check Ollama memory allocation")
                print("     • Verify model is fully loaded")
                self.results['recommendations'].append({
                    'priority': 'HIGH',
                    'issue': 'Ollama integration failures',
                    'solution': 'Increase timeout and check memory'
                })
        
        # Save results
        with open('/home/jovan-blango/Desktop/MillennialAi/injection_diagnostic_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print("\n📊 Full results saved to: injection_diagnostic_results.json")
    
    def run_all_tests(self):
        """Run complete diagnostic suite"""
        print("\n" + "🔬"*35)
        print("DEEP INJECTION POINT DIAGNOSTIC")
        print("🔬"*35)
        
        try:
            self.test_real_brain_forward()
        except Exception as e:
            print(f"\n❌ Real Brain test failed: {e}")
            traceback.print_exc()
        
        try:
            self.test_ollama_settings()
        except Exception as e:
            print(f"\n❌ Ollama settings test failed: {e}")
            traceback.print_exc()
        
        try:
            self.test_ollama_direct()
        except Exception as e:
            print(f"\n❌ Ollama direct test failed: {e}")
            traceback.print_exc()
        
        try:
            self.test_hybrid_integration()
        except Exception as e:
            print(f"\n❌ Hybrid integration test failed: {e}")
            traceback.print_exc()
        
        try:
            self.test_injection_points()
        except Exception as e:
            print(f"\n❌ Injection framework test failed: {e}")
            traceback.print_exc()
        
        self.analyze_and_recommend()


if __name__ == "__main__":
    debugger = InjectionFlowDebugger()
    debugger.run_all_tests()
