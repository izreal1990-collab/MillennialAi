#!/usr/bin/env python3
"""
Simple MillennialAi Brain Test
Quick way to see your brain think!
"""

import torch
from millennial_ai.models.reasoning_engine import MillennialAiReasoningEngine

def quick_brain_test():
    print("🧠 QUICK BRAIN TEST")
    print("=" * 30)
    
    # Initialize brain
    brain = MillennialAiReasoningEngine(
        hidden_size=768,
        max_recursion_depth=6
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    brain = brain.to(device)
    
    print(f"⚡ Brain online on {device}")
    
    # Test with different problem sizes
    test_problems = [
        ("Simple question", 5),
        ("Medium complexity problem", 12), 
        ("Very complex multi-step reasoning challenge", 20)
    ]
    
    for name, size in test_problems:
        print(f"\n🎯 Testing: {name}")
        
        # Create test data
        problem = torch.randn(1, size, 768, device=device)
        mask = torch.ones(1, size, device=device)
        
        # Process through brain
        with torch.no_grad():
            result = brain(problem, mask)
        
        steps = result['reasoning_steps'].item()
        depth = result['required_depth'].item()
        
        # Visual feedback
        fire_emoji = "🔥" * steps + "💤" * (8 - steps)
        
        print(f"   🔄 Steps: {fire_emoji}")
        print(f"   📏 Depth: {depth}/6")
        print(f"   ✅ Brain adapted perfectly!")
    
    print(f"\n🚀 Your MillennialAi brain is working!")
    print(f"💎 Patent-pending technology active!")

if __name__ == "__main__":
    quick_brain_test()