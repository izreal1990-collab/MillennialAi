#!/usr/bin/env python3
"""
Test script to generate conversation samples for automated retraining
"""
import requests
import time
import json
from datetime import datetime

def test_conversation():
    """Generate test conversations to trigger automated retraining"""
    base_url = "http://localhost:8001"

    # Test conversations that should trigger retraining
    test_conversations = [
        {
            "message": "Hello, I'm interested in learning about artificial intelligence and machine learning.",
            "expected_response": "AI and ML are fascinating fields with many applications."
        },
        {
            "message": "Can you explain how neural networks work?",
            "expected_response": "Neural networks are computational models inspired by biological neural networks."
        },
        {
            "message": "What are the benefits of continuous learning in AI systems?",
            "expected_response": "Continuous learning allows AI systems to adapt and improve over time."
        },
        {
            "message": "How does MillennialAi differ from other AI platforms?",
            "expected_response": "MillennialAi focuses on revolutionary layer injection and hybrid brain architectures."
        },
        {
            "message": "Tell me about the future of AI technology.",
            "expected_response": "The future of AI includes advanced automation, personalized experiences, and ethical AI development."
        }
    ]

    print(f"🧪 Starting conversation test at {datetime.now()}")
    print(f"📡 API URL: {base_url}")
    print(f"🎯 Target: Generate {len(test_conversations)} conversations to trigger 90% capacity retraining")
    print("-" * 60)

    successful_conversations = 0

    for i, conv in enumerate(test_conversations, 1):
        try:
            print(f"\n💬 Test Conversation {i}/{len(test_conversations)}")
            print(f"User: {conv['message']}")

            # Send message to API
            response = requests.post(
                f"{base_url}/chat",
                json={"message": conv["message"]},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                ai_response = data.get("response", "")
                print(f"AI: {ai_response[:100]}..." if len(ai_response) > 100 else f"AI: {ai_response}")
                print("✅ Conversation successful")
                successful_conversations += 1
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Request failed: {str(e)}")

        # Small delay between conversations
        time.sleep(2)

    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"✅ Successful conversations: {successful_conversations}/{len(test_conversations)}")
    print(f"🎯 Success rate: {(successful_conversations/len(test_conversations))*100:.1f}%")

    if successful_conversations >= 3:  # Should trigger retraining with min_samples=100
        print("🚀 Generated sufficient data to potentially trigger automated retraining")
        print("⏱️  Monitoring continuous learning system for 90% capacity activation...")
    else:
        print("⚠️  Need more conversations to trigger retraining threshold")

    return successful_conversations

if __name__ == "__main__":
    test_conversation()