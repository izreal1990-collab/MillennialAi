# Sample Data for MillennialAi Testing and Examples

## Sample Prompts for Testing
sample_prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Solve the equation: 5x + 3 = 18",
    "Write a short story about a robot learning to paint.",
    "What are the benefits of renewable energy?",
    "Describe the water cycle.",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Explain the concept of recursion.",
    "What are the main causes of climate change?"
]

## Sample Conversations for Testing
sample_conversations = [
    {
        "user": "Hello, can you help me understand Layer Injection?",
        "assistant": "Of course! Layer Injection is a technique where we enhance transformer models by injecting cognitive reasoning blocks between existing layers, without retraining the base model."
    },
    {
        "user": "What makes MillennialAi different?",
        "assistant": "MillennialAi uses proprietary TRM (Tiny Recursion Model) blocks that provide adaptive reasoning depth, multi-scale attention, and memory augmentation - all while being 100% IP-safe."
    }
]

## Sample Model Configurations
sample_configs = {
    "small_test": {
        "base_model_name": "gpt2",
        "base_model_layers": 12,
        "hidden_size": 768,
        "enhancement_dim": 512,
        "reasoning_stages": 2,
        "attention_heads": 8,
        "injection_points": [3, 6, 9]
    },
    "medium_test": {
        "base_model_name": "microsoft/DialoGPT-medium",
        "base_model_layers": 24,
        "hidden_size": 1024,
        "enhancement_dim": 768,
        "reasoning_stages": 3,
        "attention_heads": 12,
        "injection_points": [6, 12, 18]
    }
}

## Benchmark Datasets (Small samples)
benchmark_questions = [
    {
        "question": "What is 2 + 2?",
        "expected_answer": "4",
        "category": "math"
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "expected_answer": "William Shakespeare",
        "category": "literature"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "expected_answer": "Jupiter",
        "category": "science"
    }
]

## Performance Metrics Templates
performance_template = {
    "inference_time": 0.0,
    "memory_usage": 0.0,
    "injection_count": 0,
    "reasoning_depth": 0,
    "accuracy_score": 0.0,
    "response_quality": 0.0
}