#!/usr/bin/env python3
"""
Simple test script for MillennialAi Live Chat API
"""
import requests
import json
import time

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_chat():
    """Test the chat endpoint"""
    try:
        payload = {
            "message": "Hello, can you tell me about MillennialAi?",
            "session_id": "test123"
        }
        response = requests.post("http://localhost:8001/chat", json=payload)
        if response.status_code == 200:
            print("✅ Chat endpoint working!")
            result = response.json()
            print(f"Response: {result.get('response', '')[:200]}...")
            return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False

def test_learning_stats():
    """Test the learning stats endpoint"""
    try:
        response = requests.get("http://localhost:8001/learning-stats")
        if response.status_code == 200:
            print("✅ Learning stats endpoint working!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Learning stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Learning stats error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing MillennialAi Live Chat API...")
    print("Note: Make sure the API server is running on http://localhost:8001")

    # Give server time to start if needed
    time.sleep(2)

    # Run tests
    health_ok = test_health()
    if health_ok:
        chat_ok = test_chat()
        learning_ok = test_learning_stats()

        if chat_ok and learning_ok:
            print("\n🎉 All tests passed! The API is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Check the server logs.")
    else:
        print("\n❌ Cannot proceed with tests - health check failed.")