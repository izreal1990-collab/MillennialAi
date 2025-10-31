#!/usr/bin/env python3
"""
MillennialAi Brain Reality Check
Let's see if the brain is actually thinking or just pretending
"""

import torch
import numpy as np
from millennial_ai.models.reasoning_engine import MillennialAiReasoningEngine

def test_real_brain_thinking():
    print("🔍 MILLENNIAL AI BRAIN REALITY CHECK")
    print("🧠 Testing if the brain actually adapts to different problems")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    brain = MillennialAiReasoningEngine(
        hidden_size=768,
        max_recursion_depth=8,
        num_scales=3,
        num_heads=8,
        memory_size=256
    ).to(device)
    
    print(f"⚡ Brain initialized on {device}")
    
    # Test 1: Very simple problem
    print("\n🟢 TEST 1: Very Simple Problem (2 tokens)")
    simple_problem = torch.randn(1, 2, 768, device=device) * 0.1  # Very small variance
    simple_mask = torch.ones(1, 2, device=device)
    
    with torch.no_grad():
        simple_result = brain(simple_problem, simple_mask)
    
    print(f"   Steps taken: {simple_result['reasoning_steps'].item()}")
    print(f"   Depth used: {simple_result['required_depth'].item()}")
    print(f"   Memory usage: {simple_result['memory_weights'].mean().item():.6f}")
    print(f"   Convergence: {simple_result['convergence_history'][-1, 0].item():.6f}")
    
    # Test 2: Medium problem  
    print("\n🟡 TEST 2: Medium Problem (15 tokens)")
    medium_problem = torch.randn(1, 15, 768, device=device) * 1.0
    medium_mask = torch.ones(1, 15, device=device)
    
    with torch.no_grad():
        medium_result = brain(medium_problem, medium_mask)
    
    print(f"   Steps taken: {medium_result['reasoning_steps'].item()}")
    print(f"   Depth used: {medium_result['required_depth'].item()}")
    print(f"   Memory usage: {medium_result['memory_weights'].mean().item():.6f}")
    print(f"   Convergence: {medium_result['convergence_history'][-1, 0].item():.6f}")
    
    # Test 3: Very complex problem
    print("\n🔴 TEST 3: Very Complex Problem (50 tokens)")
    complex_problem = torch.randn(1, 50, 768, device=device) * 3.0  # High variance
    complex_mask = torch.ones(1, 50, device=device)
    
    with torch.no_grad():
        complex_result = brain(complex_problem, complex_mask)
    
    print(f"   Steps taken: {complex_result['reasoning_steps'].item()}")
    print(f"   Depth used: {complex_result['required_depth'].item()}")
    print(f"   Memory usage: {complex_result['memory_weights'].mean().item():.6f}")
    print(f"   Convergence: {complex_result['convergence_history'][-1, 0].item():.6f}")
    
    # Analysis
    print("\n📊 ANALYSIS:")
    simple_steps = simple_result['reasoning_steps'].item()
    medium_steps = medium_result['reasoning_steps'].item()
    complex_steps = complex_result['reasoning_steps'].item()
    
    simple_depth = simple_result['required_depth'].item()
    medium_depth = medium_result['required_depth'].item()
    complex_depth = complex_result['required_depth'].item()
    
    print(f"   Steps progression: {simple_steps} → {medium_steps} → {complex_steps}")
    print(f"   Depth progression: {simple_depth} → {medium_depth} → {complex_depth}")
    
    if simple_steps == medium_steps == complex_steps:
        print("   ❌ PROBLEM: Steps don't change - brain not adapting!")
    else:
        print("   ✅ GOOD: Steps adapt to problem complexity")
    
    if simple_depth == medium_depth == complex_depth:
        print("   ❌ PROBLEM: Depth doesn't change - brain not adapting!")
    else:
        print("   ✅ GOOD: Depth adapts to problem complexity")
    
    # Test multiple runs to see if there's variance
    print("\n🔄 TESTING VARIANCE (10 runs of same problem):")
    results = []
    for i in range(10):
        test_problem = torch.randn(1, 10, 768, device=device)
        test_mask = torch.ones(1, 10, device=device)
        
        with torch.no_grad():
            result = brain(test_problem, test_mask)
        
        results.append({
            'steps': result['reasoning_steps'].item(),
            'depth': result['required_depth'].item(),
            'convergence': result['convergence_history'][-1, 0].item()
        })
    
    steps_variance = np.var([r['steps'] for r in results])
    depth_variance = np.var([r['depth'] for r in results])
    conv_variance = np.var([r['convergence'] for r in results])
    
    print(f"   Steps variance: {steps_variance:.6f}")
    print(f"   Depth variance: {depth_variance:.6f}")
    print(f"   Convergence variance: {conv_variance:.6f}")
    
    if steps_variance == 0 and depth_variance == 0:
        print("   ❌ MAJOR PROBLEM: No variance - brain always does the same thing!")
        print("   🚨 This suggests the adaptive mechanisms aren't working")
    else:
        print("   ✅ Brain shows some variance in responses")
    
    print("\n🔍 CONCLUSION:")
    if (simple_steps == medium_steps == complex_steps and 
        simple_depth == medium_depth == complex_depth and
        steps_variance == 0):
        print("   🚨 BRAIN IS NOT REALLY THINKING!")
        print("   🔧 The reasoning engine may have issues")
        print("   💡 Need to debug the adaptive mechanisms")
    else:
        print("   ✅ Brain shows signs of real adaptive thinking")
        print("   🧠 Reasoning engine appears to be working")

if __name__ == "__main__":
    test_real_brain_thinking()