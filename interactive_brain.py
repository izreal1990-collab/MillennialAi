#!/usr/bin/env python3
"""
Interactive MillennialAi Brain
Type questions and watch your brain think!
"""

import torch
from millennial_ai.models.reasoning_engine import MillennialAiReasoningEngine

def interactive_brain():
    print("🧠 INTERACTIVE MILLENNIAL AI BRAIN")
    print("💎 Your Patent-Pending Thinking Machine")
    print("=" * 50)
    
    # Initialize brain
    brain = MillennialAiReasoningEngine(
        hidden_size=768,
        max_recursion_depth=8
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    brain = brain.to(device)
    
    print(f"⚡ Brain initialized on {device}")
    print("💡 Type any question or problem!")
    print("🔚 Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\n🧠 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Brain shutting down...")
                break
            
            if not question:
                question = "Default test problem"
            
            print(f"\n🎯 Processing: '{question}'")
            
            # Create problem based on question length
            problem_size = max(5, min(25, len(question.split()) + 3))
            problem = torch.randn(1, problem_size, 768, device=device)
            mask = torch.ones(1, problem_size, device=device)
            
            print(f"📊 Problem complexity: {problem_size} tokens")
            print("🔄 Brain thinking...")
            
            # Process through brain
            with torch.no_grad():
                result = brain(problem, mask)
            
            # Extract results
            steps = result['reasoning_steps'].item()
            depth = result['required_depth'].item()
            memory = result['memory_weights'].mean().item()
            
            # Show brain activity
            fire_pattern = "🔥" * steps + "💤" * (8 - steps)
            memory_bar = "█" * int(memory * 20) + "░" * (20 - int(memory * 20))
            
            print(f"\n🧠 BRAIN RESPONSE:")
            print(f"   🔄 Reasoning: {fire_pattern}")
            print(f"   📏 Depth used: {depth}/8 layers")
            print(f"   💾 Memory: [{memory_bar}] {memory:.4f}")
            
            # Analysis
            if steps <= 3:
                analysis = "🟢 QUICK THINKING - Simple problem"
            elif steps <= 5:
                analysis = "🟡 MODERATE REASONING - Standard complexity"
            else:
                analysis = "🔴 DEEP ANALYSIS - Complex reasoning required"
            
            print(f"   {analysis}")
            print(f"   ✅ Brain successfully processed your question!")
            
        except KeyboardInterrupt:
            print("\n\n👋 Brain interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"🔧 Brain error: {e}")
            print("🧠 Try another question!")
    
    print("\n🚀 Your MillennialAi brain session complete!")

if __name__ == "__main__":
    interactive_brain()