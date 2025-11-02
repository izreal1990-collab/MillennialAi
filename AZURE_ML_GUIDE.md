# Azure ML Training and Testing Guide

## 🌟 Overview

This guide covers training and testing MillennialAi on Azure Machine Learning with GPU compute clusters.

## 🏗️ Infrastructure Components

### Created by `infra/main.bicep`:

1. **Azure ML Workspace** - Central hub for ML operations
2. **GPU Compute Cluster** - `Standard_NC6s_v3` (Tesla V100, 6 cores, 112GB RAM)
   - Auto-scaling: 0-4 nodes
   - Low-priority for cost savings
3. **CPU Compute Cluster** - `Standard_DS3_v2` for preprocessing
4. **Storage Account** - Training data, models, checkpoints
5. **Key Vault** - Secure credential storage
6. **Application Insights** - Training monitoring and metrics
7. **Container Registry** - Docker images for training environments

## 🚀 Quick Start

### 1. Deploy Azure Infrastructure

```bash
# Provision all Azure resources
azd up

# This creates:
# - ML workspace with GPU clusters
# - Storage accounts with containers (training-data, models, checkpoints)
# - Application Insights for monitoring
# - Container registry and apps
```

### 2. Set Environment Variables

After deployment, get the values:

```bash
# Get Azure environment values
azd env get-values

# Export for local use
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
export AZURE_RESOURCE_GROUP="<your-resource-group>"
export AZURE_ML_WORKSPACE_NAME="<your-workspace-name>"
export AZURE_STORAGE_ACCOUNT_NAME="<your-storage-account>"
```

### 3. Submit Training Job

```bash
# Install Azure ML dependencies
pip install azure-ai-ml azure-identity azure-storage-blob mlflow

# Submit training to GPU cluster
python submit_azure_training.py
```

This will:
- Upload your code to Azure ML
- Create training environment with PyTorch + CUDA
- Allocate GPU compute (Tesla V100)
- Start training with MLflow tracking
- Save checkpoints to Azure Storage

### 4. Monitor Training

```bash
# View job status
az ml job list --workspace-name <workspace-name> --resource-group <resource-group>

# Stream logs
az ml job stream --name <job-name> --workspace-name <workspace-name> --resource-group <resource-group>

# Or use Azure ML Studio web interface
# URL provided in submit_azure_training.py output
```

## 📊 Training Configuration

### Default Settings (Configurable in `azure_ml_training.py`):

```python
--num-epochs 100              # Training epochs
--batch-size 4                # Batch size
--learning-rate 0.001         # Learning rate
--checkpoint-frequency 10     # Save every N epochs
--use-mlflow                  # Enable experiment tracking
--use-azure-storage           # Upload models to blob storage
```

### GPU Cluster Specifications:

- **VM Size**: `Standard_NC6s_v3`
  - GPU: NVIDIA Tesla V100 (16GB VRAM)
  - vCPUs: 6
  - Memory: 112 GB RAM
  - Ideal for MillennialAi training (handles 7B-13B models)

- **Scaling**: 
  - Min nodes: 0 (cost-effective)
  - Max nodes: 4 (distributed training capable)
  - Auto-scale down after 120s idle

## 🧪 Testing Trained Models

### Test Locally

```bash
# Test local model checkpoint
python azure_ml_testing.py \
    --model-path ./outputs/best_model_epoch_50.pt \
    --output-dir ./test_results
```

### Test Azure-Stored Model

```bash
# Test model from Azure Storage
python azure_ml_testing.py \
    --use-azure \
    --azure-model-uri "https://<storage>.blob.core.windows.net/models/millennialai/run_xxx/final_model.pt" \
    --output-dir ./test_results
```

### Comprehensive Test Suites:

1. **Reasoning** - Logic, math, problem-solving
2. **Knowledge** - Factual recall, comprehension
3. **Creativity** - Story generation, ideas
4. **Conversation** - Natural dialogue

Results saved as JSON with performance metrics.

## 💰 Cost Management

### Estimated Costs (East US 2):

**GPU Training (Standard_NC6s_v3 Low-Priority)**:
- Hourly: ~$0.90/hour
- 100 epochs (~1-2 hours): ~$1-2
- Monthly maximum (24/7): ~$650

**Storage**:
- Blob Storage: ~$0.02/GB/month
- Model checkpoints (5GB): ~$0.10/month

**Other Services**:
- ML Workspace: Free tier available
- Container Apps: ~$0.18/day
- Application Insights: Pay-per-use

### Cost Optimization:

1. **Use Low-Priority VMs** - 80% cheaper (already configured)
2. **Auto-scaling** - Scale to zero when idle
3. **Checkpoint Strategy** - Save every 10 epochs (configurable)
4. **Lifecycle Policies** - Auto-delete old checkpoints

```bash
# Stop all compute when not training
az ml compute stop --name gpu-cluster --resource-group <rg>

# Delete old checkpoints (30+ days)
az storage blob delete-batch \
    --source checkpoints \
    --account-name <storage> \
    --if-unmodified-since 30d
```

## 📁 Project Structure

```
MillennialAi/
├── azure_ml_training.py        # Main training script for Azure
├── submit_azure_training.py    # Job submission script
├── azure_ml_testing.py         # Model testing and validation
├── azure_ml_env.yaml           # Training environment definition
├── infra/
│   ├── main.bicep              # Infrastructure as Code
│   └── main.parameters.json    # Bicep parameters
├── outputs/                    # Local training outputs
│   ├── training_dataset.json
│   ├── checkpoint_*.pt
│   └── final_model.pt
└── test_outputs/               # Test results
    ├── test_results_detailed.json
    └── test_metrics.json
```

## 🔧 Advanced Usage

### Distributed Training (Multi-GPU)

Edit `submit_azure_training.py`:

```python
job = command(
    ...
    compute="gpu-cluster",
    instance_count=4,  # Use 4 GPU nodes
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 1
    }
)
```

### Custom Hyperparameters

```bash
# Edit azure_ml_training.py arguments
python submit_azure_training.py --custom-args \
    --num-epochs 200 \
    --batch-size 8 \
    --learning-rate 0.0005
```

### Resume from Checkpoint

Add to `azure_ml_training.py`:

```python
if args.resume_checkpoint:
    checkpoint = torch.load(args.resume_checkpoint)
    brain.load_state_dict(checkpoint['brain_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
```

## 📊 Monitoring and Logging

### MLflow Tracking

Access MLflow UI in Azure ML Studio:

1. Navigate to Experiments
2. Click experiment name: `millennialai-training`
3. View metrics, parameters, artifacts

### Application Insights

Query training metrics:

```kusto
traces
| where message contains "Epoch"
| project timestamp, message
| order by timestamp desc
```

### Storage Artifacts

```bash
# List models
az storage blob list \
    --container-name models \
    --account-name <storage>

# Download best model
az storage blob download \
    --container-name models \
    --name millennialai/run_xxx/best_model.pt \
    --file ./best_model.pt \
    --account-name <storage>
```

## 🐛 Troubleshooting

### Job Fails to Start

```bash
# Check compute cluster status
az ml compute show --name gpu-cluster --workspace-name <ws>

# Check quota
az ml compute list-usage --location eastus2
```

### Out of Memory (OOM)

Reduce batch size in `azure_ml_training.py`:

```python
--batch-size 2  # Instead of 4
```

Or use gradient accumulation:

```python
# Effective batch size = batch_size * accumulation_steps
accumulation_steps = 2
```

### Slow Training

Check GPU utilization:

```bash
# SSH to compute instance (if enabled)
nvidia-smi

# Or check in Application Insights
```

### Authentication Errors

```bash
# Re-authenticate
az login

# Set default subscription
az account set --subscription <subscription-id>

# Verify workspace access
az ml workspace show --name <workspace-name> --resource-group <rg>
```

## 🎯 Next Steps

1. **Experiment with Hyperparameters** - Learning rates, batch sizes
2. **Try Larger Models** - Scale to 70B with distributed training
3. **Deploy Models** - Use Azure ML endpoints for inference
4. **Automate Pipelines** - Create ML pipelines for continuous training
5. **Monitor Production** - Set up alerts and performance tracking

## 📚 Resources

- [Azure ML Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [GPU VM Sizes](https://learn.microsoft.com/azure/virtual-machines/sizes-gpu)
- [MLflow on Azure](https://learn.microsoft.com/azure/machine-learning/how-to-use-mlflow)
- [Cost Management](https://learn.microsoft.com/azure/cost-management-billing/)

---

**Need Help?** Check the main [README.md](../README.md) or open an issue on GitHub.
