#!/usr/bin/env python3
"""
Fixed Azure ML Test Script for MillennialAi

This script tests the MillennialAi framework in Azure ML environment
with proper imports and error handling.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_pytorch():
    """Test basic PyTorch functionality"""
    print("🧪 MillennialAi Azure ML Test Job -", datetime.now())
    print("=" * 50)
    
    print("Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Running on CPU (expected for CPU cluster)")
        device = torch.device("cpu")
    
    print("\nTesting tensor operations...")
    # Test tensor operations
    x = torch.randn(1000, 1000, device=device)
    y = torch.mm(x, x.t())
    print(f"Matrix multiplication successful: {y.shape}")
    
    return True

def test_millennial_ai_imports():
    """Test MillennialAi imports and basic functionality"""
    print("\nTesting MillennialAi imports...")
    
    try:
        # Add the current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Test imports
        from millennial_ai.config.config import HybridConfig
        from millennial_ai.config.constants import DEFAULT_TRM_HIDDEN_SIZE
        print("✅ MillennialAi config imports successful")
        
        # Test config creation
        config = HybridConfig()
        print(f"✅ HybridConfig created: hidden_size={config.trm_hidden_size}")
        
        return True
        
    except ImportError as e:
        print(f"❌ MillennialAi import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ MillennialAi config test failed: {e}")
        return False

def test_model_loading():
    """Test loading a small model for validation"""
    print("\nTesting model loading...")
    
    try:
        # Use a tiny model for testing
        model_name = "microsoft/DialoGPT-small"
        
        # Load tokenizer (lightweight)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenization successful: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Model loading test failed: {e}")
        print("This is expected if transformers is not installed")
        return False

def test_environment():
    """Test Azure ML environment"""
    print("\nTesting Azure ML environment...")
    
    # Check environment variables
    env_vars = ['AZUREML_RUN_ID', 'AZUREML_EXPERIMENT_NAME']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️ {var}: Not set")
    
    # Check working directory
    print(f"✅ Working directory: {os.getcwd()}")
    print(f"✅ Python version: {sys.version}")
    
    return True

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting MillennialAi Azure ML Tests")
    print("=" * 50)
    
    results = {
        'pytorch': False,
        'millennial_ai': False,
        'model_loading': False,
        'environment': False
    }
    
    try:
        # Test 1: PyTorch
        results['pytorch'] = test_basic_pytorch()
        
        # Test 2: MillennialAi imports
        results['millennial_ai'] = test_millennial_ai_imports()
        
        # Test 3: Model loading
        results['model_loading'] = test_model_loading()
        
        # Test 4: Environment
        results['environment'] = test_environment()
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return False
    
    # Summary
    print("\n📊 Test Results:")
    print("=" * 30)
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MillennialAi is ready for Azure ML")
        return True
    elif results['pytorch'] and results['millennial_ai']:
        print("✅ Core functionality working - ready for training")
        return True
    else:
        print("❌ Critical tests failed - requires investigation")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🚀 MillennialAi Azure ML validation completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed but core functionality may still work")
        sys.exit(1)