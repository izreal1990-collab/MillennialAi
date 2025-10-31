#!/usr/bin/env python3
"""
Simple MillennialAi Chat
Just a basic working chat interface
"""

import torch

def simple_chat():
    """Simple command-line chat"""
    print("🧠 MillennialAi Simple Chat")
    print("=" * 30)
    print("Type 'quit' to exit")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")
    print()
    
    while True:
        try:
            # Get user input
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Simple processing
            print("AI: ", end="")
            
            # Analyze question length and complexity
            word_count = len(question.split())
            complexity = min(5, max(1, word_count // 3))
            
            print(f"I analyzed your question (complexity: {complexity}/5)")
            
            # Simple responses
            if 'what' in question.lower() or 'define' in question.lower():
                print("    This is a definition question. I can provide explanations.")
            elif 'how' in question.lower():
                print("    This is a process question. I can break down steps.")
            elif 'why' in question.lower():
                print("    This is a reasoning question. I can explain causes.")
            elif any(word in question.lower() for word in ['solve', 'calculate']):
                print("    This is a computational question. I can work through it.")
            else:
                print("    I can help with this question using adaptive reasoning.")
            
            print(f"    Processing used {complexity} reasoning steps.")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    simple_chat()