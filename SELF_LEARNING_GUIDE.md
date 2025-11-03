# MillennialAi Self-Learning System Guide

## Overview

MillennialAi now includes **4 comprehensive self-learning mechanisms** that allow the system to improve from its own responses and user feedback.

## Features

### 1. Feedback System (User Ratings)
Users can rate AI responses to build supervised learning datasets.

**Endpoint:** `POST /api/feedback`

**Example:**
```bash
curl -X POST https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "abc123",
    "rating": 5,
    "helpful": true,
    "accurate": true,
    "corrections": "Could mention X for completeness",
    "tags": ["clear", "technical"]
  }'
```

**What it does:**
- ⭐ **4-5 star ratings** → Prioritized for training (high-quality examples)
- ⭐ **1-2 star ratings** with corrections → Used for improvement
- 📊 Tracks average rating and quality trends
- 🎯 Builds supervised learning dataset

---

### 2. Self-Reflection (AI Analyzes Itself)
The AI evaluates its own responses for quality and suggests improvements.

**Endpoint:** `POST /api/self-reflect`

**Example:**
```bash
curl -X POST https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/api/self-reflect \
  -H "Content-Type: application/json" \
  -d '{
    "original_query": "What is quantum computing?",
    "ai_response": "Quantum computing uses qubits...",
    "reflection_type": "accuracy"
  }'
```

**Reflection Types:**
- `general` - Overall quality assessment
- `accuracy` - Factual correctness check
- `clarity` - Readability and understandability
- `completeness` - Coverage of the topic

**What it does:**
- 🔍 Scores accuracy (1-10)
- 📝 Scores clarity (1-10)
- ✅ Scores completeness (1-10)
- 💡 Suggests specific improvements
- 🔄 Queues improvements for training

---

### 3. Automated Training Loop
Periodic automatic retraining based on accumulated feedback.

**Configure:** `POST /api/training-loop/configure`

**Example:**
```bash
curl -X POST https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/api/training-loop/configure \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "min_samples": 100,
    "interval_hours": 24,
    "quality_threshold": 4.0
  }'
```

**Check Status:** `GET /api/training-loop/status`

**Trigger Manually:** `POST /api/training-loop/trigger`

**What it does:**
- ⏰ Trains every N hours (configurable)
- 📊 Only uses high-quality samples (rating ≥ threshold)
- 🔄 Automatically improves based on feedback
- 📈 Tracks training history

---

### 4. Conversation Memory Feed
Feed complete conversations back for learning.

**Endpoint:** `POST /api/learn-from-conversation`

**Example:**
```bash
curl -X POST "https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/api/learn-from-conversation?conversation_id=abc123&quality_rating=4.5"
```

**What it does:**
- 💾 Stores entire conversation threads
- 🎯 Learns conversation flow patterns
- 📚 Builds contextual understanding
- 🔄 Improves multi-turn dialogue

---

## Comprehensive Summary

**Get Full Learning Status:** `GET /api/learning/summary`

**Example:**
```bash
curl https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/api/learning/summary
```

**Returns:**
```json
{
  "continuous_learning": {
    "total_samples": 150,
    "retraining_jobs": 2,
    "min_samples_for_retrain": 100
  },
  "feedback_system": {
    "total_feedback": 45,
    "average_rating": 4.2,
    "high_quality_count": 32
  },
  "self_reflection": {
    "total_sessions": 12,
    "improvements_identified": 8
  },
  "training_loop": {
    "enabled": true,
    "last_training": "2025-11-03T10:30:00"
  }
}
```

---

## Usage Workflows

### Workflow 1: User-Driven Improvement
1. User asks question → AI responds
2. User rates response (1-5 stars)
3. System queues high-rated responses for training
4. Low-rated responses with corrections improve future answers

### Workflow 2: Self-Improvement Loop
1. AI generates response
2. System triggers self-reflection
3. AI analyzes own response quality
4. Improvements queued for next training cycle

### Workflow 3: Automated Periodic Training
1. System accumulates feedback over 24 hours
2. Reaches 100+ samples threshold
3. Automatically triggers retraining
4. Neural brain updates with new patterns

### Workflow 4: Conversation Replay
1. Save successful conversation threads
2. Feed back to system with quality rating
3. System learns multi-turn dialogue patterns
4. Improves contextual understanding

---

## Best Practices

### For High-Quality Training Data:
- ✅ Provide specific corrections in feedback
- ✅ Use self-reflection on important responses
- ✅ Set quality_threshold to 4.0+ for training
- ✅ Tag feedback (technical, simple, creative, etc.)

### For Optimal Learning:
- 📊 Collect 100+ samples before first retrain
- ⏰ Retrain every 24-48 hours
- 🎯 Balance positive and corrective feedback
- 🔍 Run self-reflection on edge cases

### For Production:
- 🔄 Enable automated training loop
- 📈 Monitor learning summary regularly
- 🎯 Keep quality_threshold ≥ 4.0
- 💾 Feed back successful conversations

---

## Integration Examples

### Android App Integration:
```kotlin
// After user receives response
fun submitFeedback(conversationId: String, rating: Int) {
    val feedback = FeedbackRequest(
        conversation_id = conversationId,
        rating = rating,
        helpful = rating >= 4,
        tags = listOf("mobile-app")
    )
    
    apiService.submitFeedback(feedback)
}
```

### Python Script:
```python
import requests

def improve_ai_with_feedback(conversation_id, rating, corrections):
    response = requests.post(
        "https://millennialai-app.../api/feedback",
        json={
            "conversation_id": conversation_id,
            "rating": rating,
            "helpful": rating >= 4,
            "corrections": corrections
        }
    )
    return response.json()
```

---

## Monitoring

### Check Learning Progress:
```bash
# Overall summary
curl https://millennialai-app.../api/learning/summary

# Training loop status
curl https://millennialai-app.../api/training-loop/status

# Original continuous learning stats
curl https://millennialai-app.../api/learning/stats
```

### View Metrics:
- **Average Rating:** Target 4.0+
- **High Quality %:** Target 70%+
- **Training Frequency:** Every 24-48h
- **Improvement Rate:** Track over time

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Self-Learning System                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Interaction                                       │
│       │                                                 │
│       ├──► Feedback (1-5 ⭐) ──► High Quality Queue     │
│       │                                                 │
│       └──► Corrections ──────────► Improvement Queue    │
│                                                         │
│  AI Response                                            │
│       │                                                 │
│       └──► Self-Reflection ──────► Quality Analysis    │
│                    │                                    │
│                    └──► Suggested Improvements          │
│                                                         │
│  Automated Loop                                         │
│       │                                                 │
│       ├──► Collect Samples (100+)                       │
│       ├──► Filter by Quality (≥4.0⭐)                   │
│       └──► Trigger Retraining (every 24h)              │
│                                                         │
│  Conversation Memory                                    │
│       │                                                 │
│       └──► Store Thread ──────────► Pattern Learning   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## System Benefits

✅ **Continuous Improvement:** System gets better with every interaction  
✅ **User-Driven:** Real feedback shapes training data  
✅ **Self-Aware:** AI identifies own weaknesses  
✅ **Automated:** No manual intervention required  
✅ **Quality-Focused:** Only learns from good examples  
✅ **Transparent:** Full visibility into learning progress

---

## Next Steps

1. **Deploy to Azure** (run deploy script)
2. **Test endpoints** with sample data
3. **Enable training loop** with `/api/training-loop/configure`
4. **Integrate into Android app** for user feedback
5. **Monitor progress** via `/api/learning/summary`

---

**Status:** All 4 systems implemented and ready for deployment ✅
