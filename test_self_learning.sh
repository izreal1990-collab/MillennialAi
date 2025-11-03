#!/bin/bash
# Test Self-Learning Endpoints

API_URL="https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io"

echo "🧪 Testing MillennialAi Self-Learning System"
echo "=============================================="
echo ""

# Test 1: Learning Summary
echo "📊 Test 1: Getting Learning Summary"
curl -s "$API_URL/api/learning/summary" | python3 -m json.tool || echo "❌ Endpoint not yet deployed"
echo ""
echo ""

# Test 2: Submit Feedback
echo "⭐ Test 2: Submitting User Feedback"
curl -s -X POST "$API_URL/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test-123",
    "rating": 5,
    "helpful": true,
    "accurate": true,
    "corrections": "Excellent response!",
    "tags": ["clear", "helpful"]
  }' | python3 -m json.tool || echo "❌ Endpoint not yet deployed"
echo ""
echo ""

# Test 3: Self-Reflection
echo "🔍 Test 3: AI Self-Reflection"
curl -s -X POST "$API_URL/api/self-reflect" \
  -H "Content-Type: application/json" \
  -d '{
    "original_query": "What is AI?",
    "ai_response": "AI is artificial intelligence.",
    "reflection_type": "clarity"
  }' | python3 -m json.tool || echo "❌ Endpoint not yet deployed"
echo ""
echo ""

# Test 4: Training Loop Status
echo "🔄 Test 4: Training Loop Status"
curl -s "$API_URL/api/training-loop/status" | python3 -m json.tool || echo "❌ Endpoint not yet deployed"
echo ""
echo ""

# Test 5: Configure Training Loop
echo "⚙️  Test 5: Configure Training Loop"
curl -s -X POST "$API_URL/api/training-loop/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "min_samples": 50,
    "interval_hours": 24,
    "quality_threshold": 4.0
  }' | python3 -m json.tool || echo "❌ Endpoint not yet deployed"
echo ""
echo ""

# Test 6: Learn from Conversation
echo "💾 Test 6: Learn from Conversation"
curl -s -X POST "$API_URL/api/learn-from-conversation?conversation_id=test-456&quality_rating=4.5" | python3 -m json.tool || echo "❌ Endpoint not yet deployed"
echo ""
echo ""

echo "=============================================="
echo "✅ Testing Complete"
echo ""
echo "To deploy updated code to Azure:"
echo "1. Code is already in GitHub (commit 2319d71)"
echo "2. Azure will auto-sync within 5-10 minutes"
echo "3. Or manually restart: az containerapp revision restart --name millennialai-app --resource-group millennialai-rg"
echo ""
echo "📚 Documentation: SELF_LEARNING_GUIDE.md"
