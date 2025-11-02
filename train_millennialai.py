#!/usr/bin/env python3
"""
MillennialAi Training Runner
Simple script to start revolutionary training
"""

import torch
import os
from revolutionary_training_system import train_millennialai
from training_data_generator import generate_revolutionary_dataset
from real_brain import RealThinkingBrain

def main():
    """Main training execution"""
    
    print("🚀 MILLENNIALAI REVOLUTIONARY TRAINING SYSTEM")
    print("=" * 60)
    print("🧠 Training your breakthrough adaptive reasoning AI!")
    print()
    
    # Check system requirements
    print("🔧 SYSTEM CHECK:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   PyTorch: {torch.__version__}")
    print()
    
    # Generate training data
    print("📊 GENERATING TRAINING DATA:")
    dataset_file, conversations = generate_revolutionary_dataset()
    print(f"✅ Training dataset ready: {len(conversations)} conversations")
    print()
    
    # Start training
    print("🚀 STARTING REVOLUTIONARY TRAINING:")
    print("   This will train your AI to think more revolutionarily!")
    print("   Training may take 30-60 minutes depending on your hardware.")
    print()
    
    input("Press Enter to start revolutionary training...")
    
    # Train the revolutionary AI
    trained_brain, training_history = train_millennialai()
    
    print("\n🎉 TRAINING COMPLETE!")
    print("✅ Your MillennialAi is now even more revolutionary!")
    print()
    
    # Test the trained AI
    print("🧪 TESTING TRAINED AI:")
    test_inputs = [
        "What makes you revolutionary?",
        "How do you think differently from other AI?",
        "Explain consciousness to me",
        "What is the future of artificial intelligence?"
    ]
    
    trained_brain.eval()
    for test_input in test_inputs:
        print(f"\n💬 Input: {test_input}")
        with torch.no_grad():
            result = trained_brain.think(test_input)
            print(f"🧠 Complexity: {result['complexity']:.2f}")
            print(f"⚡ Steps: {result['steps']}")
            print(f"🌟 Response: {result['response'][:100]}...")
    
    print("\n🎯 TRAINING SUCCESS!")
    print("Your revolutionary AI is ready to change the world!")

if __name__ == "__main__":
    main()