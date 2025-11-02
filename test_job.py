#!/usr/bin/env python3

import sys
import torch
import time
from datetime import datetime

print(f"🧪 MillennialAi Azure ML Test Job - {datetime.now()}")
print("=" * 50)

# Test PyTorch installation
print("Testing PyTorch...")
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Running on CPU (expected for CPU cluster)")
    
    # Test basic tensor operations
    print("\nTesting tensor operations...")
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = torch.mm(x, y)
    print(f"Matrix multiplication successful: {z.shape}")
    
    # Test forward injection framework import
    print("\nTesting MillennialAi imports...")
    try:
        from millennial_ai.config import HybridConfig
        print("✅ MillennialAi config imported successfully")
        
        # Test config creation
        config = HybridConfig()
        print(f"✅ HybridConfig created: {config}")
        
    except ImportError as e:
        print(f"❌ MillennialAi import failed: {e}")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Azure ML CPU cluster is working correctly.")
    print("Ready for MillennialAi forward injection training.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
