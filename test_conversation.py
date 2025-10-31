#!/usr/bin/env python3
"""
Test the current app to verify the conversation issue
"""

import torch
from real_brain import RealThinkingBrain

def test_conversation_issue():
    print("🔍 TESTING CONVERSATION ISSUE")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    brain = RealThinkingBrain().to(device)
    
    print("🧠 Brain initialized")
    
    # Test multiple questions in sequence (like a conversation)
    questions = [
        "What is AI?",
        "How does it work?", 
        "Can you explain more?",
        "What about neural networks?",
        "Tell me about machine learning"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🎯 Question {i}: '{question}'")
        
        # Simulate what the app does
        seq_len = min(30, max(5, len(question.split())))
        variance = len(question) / 50.0
        problem = torch.randn(1, seq_len, 768, device=device) * variance
        
        print(f"   Problem tensor: {problem.shape}, variance: {torch.var(problem):.4f}")
        
        try:
            with torch.no_grad():
                result = brain(problem)
            
            steps = result['reasoning_steps'].item()
            complexity = result['complexity_score']
            
            print(f"   ✅ Steps: {steps}, Complexity: {complexity:.3f}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            break
    
    print(f"\n🔍 DIAGNOSIS:")
    print("If you see different steps/complexity for similar questions,")
    print("the brain is working. If they're all the same, there's an issue.")

if __name__ == "__main__":
    test_conversation_issue()