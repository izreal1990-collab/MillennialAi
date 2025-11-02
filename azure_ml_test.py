#!/usr/bin/env python3
"""
Azure ML Test Script - Test MillennialAi with actual PyTorch models

This script tests the MillennialAi framework in Azure ML environment
using a small pre-trained model for validation.
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_pytorch():
    """Test basic PyTorch functionality"""
    logger.info("🧪 Testing basic PyTorch functionality...")

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        logger.info("⚠️ CUDA not available, using CPU")
        device = torch.device("cpu")

    # Test tensor operations
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)
    z = torch.matmul(x, y)

    logger.info(f"✅ Tensor operations successful: {z.shape}")
    return True

def test_small_model_loading():
    """Test loading a small pre-trained model"""
    logger.info("🤖 Testing small model loading...")

    try:
        # Use a small model for testing
        model_name = "distilgpt2"  # Much smaller than GPT-2

        logger.info(f"📥 Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        logger.info(f"✅ Model loaded successfully: {model_name}")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Device: {device}")

        return model, tokenizer, device

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None, None, None

def test_millennialai_config():
    """Test MillennialAi configuration loading"""
    logger.info("⚙️ Testing MillennialAi configuration...")

    try:
        # Import MillennialAi components
        from millennial_ai.config.config import HybridConfig, PresetConfigs
        from millennial_ai.config.constants import DEFAULT_TRM_HIDDEN_SIZE

        # Test basic config creation
        config = HybridConfig()
        logger.info("✅ Basic config created successfully")
        logger.info(f"   TRM hidden size: {config.trm_hidden_size}")
        logger.info(f"   Constant value: {DEFAULT_TRM_HIDDEN_SIZE}")

        # Test preset loading
        preset_config = PresetConfigs.minimal()
        logger.info("✅ Preset config loaded successfully")
        logger.info(f"   Preset TRM hidden size: {preset_config.trm_hidden_size}")

        return True

    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False

def test_model_inference(model, tokenizer, device, prompt="Hello, world!"):
    """Test model inference with a simple prompt"""
    logger.info("🔮 Testing model inference...")

    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("✅ Inference successful!")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Response: {response}")

        return True

    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 MILLENNIALAI AZURE ML TEST SUITE")
    print("=" * 50)

    results = {}

    # Test 1: Basic PyTorch
    results["pytorch"] = test_basic_pytorch()

    # Test 2: MillennialAi Config
    results["config"] = test_millennialai_config()

    # Test 3: Model Loading
    model, tokenizer, device = test_small_model_loading()
    results["model_loading"] = model is not None

    # Test 4: Model Inference (only if model loaded)
    if model is not None:
        results["inference"] = test_model_inference(model, tokenizer, device)
    else:
        results["inference"] = False
        logger.warning("⚠️ Skipping inference test due to model loading failure")

    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 30)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.upper()}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! MillennialAi is ready for Azure ML training.")
        return 0
    else:
        print("⚠️ Some tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())