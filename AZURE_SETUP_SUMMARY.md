# MillennialAi Azure Training Setup - Summary

## 🎯 What We've Created

Complete Azure ML infrastructure for training and testing MillennialAi on cloud GPU resources.

## 📁 New Files Created

### Infrastructure & Deployment

1. **`infra/main.bicep`** (Enhanced)
   - Azure ML Workspace with GPU compute clusters
   - Storage Account with blob containers (training-data, models, checkpoints)
   - Key Vault for secure credential management
   - Application Insights for monitoring
   - Auto-scaling GPU cluster (Standard_NC6s_v3, 0-4 nodes)

2. **`infra/main-no-gpu.bicep`** (Temporary)
   - Same as above but without GPU cluster (for zero-quota subscriptions)
   - Deploy this first while waiting for GPU quota approval

### Training Scripts

3. **`azure_ml_training.py`**
   - Main training script adapted for Azure ML
   - Features:
     - MLflow experiment tracking
     - Automatic checkpointing every 10 epochs
     - Azure Storage integration for model artifacts
     - Multi-GPU support (DataParallel)
     - Comprehensive logging and metrics

4. **`submit_azure_training.py`**
   - Job submission script for Azure ML
   - Creates training environment
   - Submits job to GPU cluster
   - Returns job URL for monitoring

5. **`azure_ml_testing.py`**
   - Comprehensive model testing suite
   - Test categories: Reasoning, Knowledge, Creativity, Conversation
   - Performance metrics and analysis
   - Results saved as JSON with MLflow logging

6. **`azure_ml_env.yaml`**
   - Conda environment definition for training
   - Includes PyTorch + CUDA 11.8
   - Azure ML SDK and dependencies
   - All required HuggingFace libraries

### Documentation

7. **`AZURE_ML_GUIDE.md`**
   - Complete guide for Azure ML training
   - Infrastructure overview
   - Training configuration and submission
   - Cost management strategies
   - Monitoring and troubleshooting

8. **`GPU_QUOTA_REQUEST_GUIDE.md`**
   - Step-by-step guide to request GPU quota
   - Three methods (Portal, CLI, Support Center)
   - Cost estimations for different GPU types
   - Alternative training options while waiting

### Updated Files

9. **`requirements.txt`** (Updated)
   - Added Azure ML SDK v2 (`azure-ai-ml>=1.11.0`)
   - Added Azure authentication (`azure-identity>=1.14.0`)
   - Added Azure Storage SDK (`azure-storage-blob>=12.19.0`)
   - Added MLflow integration (`mlflow>=2.8.0`, `azureml-mlflow>=1.53.0`)

## 🏗️ Azure Resources (Being Deployed)

```
Resource Group: rg-jblango-1749
Location: East US 2

├── Azure ML Workspace (azml7c462q3plqeyo)
│   ├── CPU Compute Cluster (cpu-cluster)
│   │   └── Standard_DS3_v2, 0-2 nodes
│   └── GPU Compute Cluster (gpu-cluster) [PENDING QUOTA]
│       └── Standard_NC6s_v3 (Tesla V100), 0-4 nodes
│
├── Storage Account (azsa7c462q3plqeyo)
│   ├── training-data/ (Training datasets)
│   ├── models/ (Trained models)
│   └── checkpoints/ (Training checkpoints)
│
├── Key Vault (azkv7c462q3plqeyo)
│   └── Secure storage for credentials
│
├── Application Insights (azai7c462q3plqeyo)
│   └── Training metrics and logging
│
├── Container Registry (azacr7c462q3plqeyo)
│   └── Docker images for training environments
│
└── Container Apps Environment (azcae7c462q3plqeyo)
    └── API deployment (optional)
```

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Install Azure ML dependencies
pip install -r requirements.txt

# Set up authentication
az login

# Set environment variables
export AZURE_SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
export AZURE_RESOURCE_GROUP="rg-jblango-1749"
export AZURE_ML_WORKSPACE_NAME="azml7c462q3plqeyo"
export AZURE_STORAGE_ACCOUNT_NAME="azsa7c462q3plqeyo"
```

### Workflow

#### Step 1: Request GPU Quota (Required First Time)

```bash
# Check current GPU quota
az ml quota list --location eastus2 | grep NC

# If quota is 0, follow GPU_QUOTA_REQUEST_GUIDE.md
# Typical approval time: 1-3 business days
```

#### Step 2: Deploy Infrastructure

```bash
# Option A: With GPU (if quota approved)
az deployment group create \
  --resource-group rg-jblango-1749 \
  --template-file infra/main.bicep \
  --parameters environmentName=millennialai-revolutionary location=eastus2

# Option B: Without GPU (while waiting for quota)
az deployment group create \
  --resource-group rg-jblango-1749 \
  --template-file infra/main-no-gpu.bicep \
  --parameters environmentName=millennialai-revolutionary location=eastus2
```

#### Step 3: Submit Training Job

```bash
# Submit to Azure ML GPU cluster
python submit_azure_training.py

# Monitor job
az ml job list --workspace-name azml7c462q3plqeyo

# Stream logs
az ml job stream --name <job-name> --workspace-name azml7c462q3plqeyo
```

#### Step 4: Test Trained Model

```bash
# Test model from Azure Storage
python azure_ml_testing.py \
  --use-azure \
  --azure-model-uri "https://azsa7c462q3plqeyo.blob.core.windows.net/models/..." \
  --output-dir ./test_results
```

## 💰 Cost Breakdown

### Current Deployment (CPU-Only)

- **Azure ML Workspace**: Free tier
- **Storage Account**: ~$0.02/GB/month
- **CPU Compute Cluster**: ~$0.18/hour (when running, auto-scales to 0)
- **Application Insights**: Pay-per-use (~$2/GB ingested)
- **Container Registry**: Basic tier (~$5/month)

**Monthly Estimate (Idle)**: ~$7-10/month

### After GPU Quota Approval

- **GPU Cluster (Low-Priority NC6s_v3)**:
  - Per node: ~$0.90/hour
  - 4 nodes for 2 hours training: ~$7.20
  - Auto-scales to 0 when idle (no cost)

**Per Training Run**: ~$5-15 (depending on duration and scale)

## 🎯 Training Capabilities

### Supported Models

1. **Small-Scale (7B models)**
   - Training time: 1-2 hours on 4x Tesla V100
   - Estimated cost: ~$7-15

2. **Medium-Scale (13B models)**
   - Training time: 3-5 hours on 4x Tesla V100
   - Estimated cost: ~$20-40

3. **Large-Scale (70B models)** [Distributed Training]
   - Training time: 8-12 hours on 4x Tesla V100
   - Estimated cost: ~$70-100

### Features

- ✅ Automatic checkpointing every 10 epochs
- ✅ MLflow experiment tracking
- ✅ Azure Storage integration
- ✅ Multi-GPU distributed training
- ✅ Cost-optimized with low-priority VMs
- ✅ Auto-scaling (0-4 nodes)
- ✅ Comprehensive testing suite

## 📊 Monitoring & Management

### Azure ML Studio

- Access: https://ml.azure.com
- View experiments, jobs, metrics
- Download artifacts and models

### MLflow UI

- Integrated in Azure ML Studio
- Track hyperparameters, metrics, artifacts
- Compare training runs

### Cost Management

```bash
# View current costs
az consumption usage list --start-date 2025-01-01 --end-date 2025-01-31

# Set budget alert
az consumption budget create \
  --budget-name "ml-training-budget" \
  --amount 100 \
  --time-grain Monthly
```

## 🐛 Known Issues & Solutions

### Issue: GPU Quota = 0

**Solution**: Follow `GPU_QUOTA_REQUEST_GUIDE.md` to request quota increase.

### Issue: Docker build OOM during `azd up`

**Solution**: Use Bicep deployment directly (already configured):
```bash
az deployment group create --template-file infra/main-no-gpu.bicep ...
```

### Issue: Training job fails to start

**Solutions**:
1. Check compute cluster status: `az ml compute show --name gpu-cluster ...`
2. Verify quota: `az ml compute list-usage --location eastus2`
3. Review job logs: `az ml job stream --name <job-name> ...`

## 🔄 Next Steps

### Immediate (While Waiting for GPU Quota)

1. ✅ Deploy CPU-only infrastructure (in progress)
2. ✅ Test Azure ML connectivity
3. ✅ Upload training data to blob storage
4. ✅ Train locally and upload models to Azure Storage

### After GPU Quota Approval

1. Deploy GPU compute cluster (update Bicep)
2. Submit first training job
3. Monitor performance and costs
4. Optimize hyperparameters
5. Set up automated ML pipelines

### Advanced

1. Implement distributed training for 70B models
2. Create Azure ML pipelines for continuous training
3. Deploy models to Azure ML endpoints
4. Set up A/B testing infrastructure
5. Implement model monitoring and retraining triggers

## 📚 Resources

- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [MLflow Integration](https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow)
- [GPU VM Pricing](https://azure.microsoft.com/pricing/details/virtual-machines/linux/)
- [Quota Management](https://learn.microsoft.com/azure/machine-learning/how-to-manage-quotas)

## ✅ Deployment Checklist

- [x] Created Azure ML training scripts
- [x] Created Bicep infrastructure files
- [x] Updated requirements.txt with Azure dependencies
- [x] Created comprehensive documentation
- [ ] Deployed Azure ML workspace (in progress)
- [ ] Requested GPU quota (user action required)
- [ ] Deployed GPU compute cluster (pending quota)
- [ ] Submitted test training job (pending GPU)
- [ ] Validated model testing pipeline

---

**Status**: Infrastructure deployment in progress. GPU quota request required before training can begin.

**Estimated Setup Time**: 
- Infrastructure deployment: ~10-15 minutes
- GPU quota approval: 1-3 business days
- First training job: 1-2 hours

**Total Cost (First Month)**: ~$50-100 (including infrastructure + 3-5 training runs)
