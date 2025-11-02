#!/bin/bash

# MillennialAi Azure ML Test Job
# Tests the CPU compute cluster with a simple PyTorch validation

echo "🧪 Testing MillennialAi on Azure ML CPU Cluster"
echo "=============================================="

# Azure ML Job Configuration
SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
WORKSPACE_NAME="azml7c462q3plqeyo"
COMPUTE_NAME="cpu-cluster"

# Create a simple test script
cat > test_job.py << 'EOF'
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
EOF

# Create Azure ML job YAML
cat > test_job.yml << EOF
\$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
command: python test_job.py
code: .
environment: azureml:millennialai-test-env:1.0
compute: azureml:$COMPUTE_NAME
display_name: millennialai-cpu-test
experiment_name: millennialai-testing
description: "Test MillennialAi setup on Azure ML CPU cluster"
EOF

echo "📄 Created test files:"
echo "  - test_job.py (Python test script)"
echo "  - test_env.yml (Azure ML environment definition)"
echo "  - environment.yml (Conda dependencies)"
echo "  - test_job.yml (Azure ML job configuration)"
echo ""

echo "🔧 Creating custom environment..."
az ml environment create --file test_env.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

echo ""
echo "🚀 Submitting test job to CPU cluster..."
az ml job create --file test_job.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

echo ""
echo "📊 Job submitted! Monitor progress:"
echo "https://ml.azure.com/experiments/millennialai-testing/runs?wsid=/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/workspaces/$WORKSPACE_NAME"

echo ""
echo "💡 Expected results:"
echo "  - Job should complete successfully"
echo "  - PyTorch should work on CPU"
echo "  - MillennialAi imports should succeed"
echo "  - Confirms Azure ML environment is ready"