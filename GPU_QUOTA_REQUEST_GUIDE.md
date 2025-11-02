# Azure GPU Quota Request Guide for MillennialAi Training

## ⚠️ Current Status

Your Azure subscription currently has **0 GPU vCPU quota**, which prevents GPU compute cluster deployment. This guide helps you request quota increase.

## 📊 Required GPU Quota

For MillennialAi training, you need one of the following:

### Option 1: Standard_NC6s_v3 (Recommended for Training)
- **GPU**: NVIDIA Tesla V100 (16GB VRAM)
- **vCPUs**: 6
- **RAM**: 112 GB
- **Quota needed**: 24 vCPUs (4 nodes × 6 vCPUs)
- **Best for**: Training 7B-13B models

### Option 2: Standard_NC4as_T4_v3 (Budget Option)
- **GPU**: NVIDIA T4 (16GB VRAM)
- **vCPUs**: 4
- **RAM**: 28 GB
- **Quota needed**: 16 vCPUs (4 nodes × 4 vCPUs)
- **Best for**: Smaller models, testing

### Option 3: Standard_NC6 (Basic Option)
- **GPU**: NVIDIA K80 (12GB VRAM)
- **vCPUs**: 6
- **RAM**: 56 GB
- **Quota needed**: 24 vCPUs (4 nodes × 6 vCPUs)
- **Best for**: Development, testing

## 🚀 How to Request GPU Quota

### Method 1: Azure Portal (Easiest)

1. **Navigate to Quotas**:
   ```
   Azure Portal → Search "Quotas" → Machine Learning Service → Select your subscription
   ```

2. **Find GPU VM Family**:
   - Search for "Standard NCSv3 Family vCPUs" (for NC6s_v3)
   - Or "Standard NCASv3_T4 Family vCPUs" (for T4)
   - Current quota will show: **0**

3. **Request Increase**:
   - Click "Request increase"
   - Select region: **East US 2** (where your resources are)
   - New quota limit: **24** (for 4 GPU nodes)
   - Business justification: "Machine Learning model training for research/development"
   - Submit request

4. **Wait for Approval**:
   - **Response time varies by severity**:
     - **Severity A**: 24x7 support (critical issues)
     - **Severity B**: Optionally 24x7 (important issues)
     - **Severity C**: Business hours only (general requests like quota)
   - **Quota requests (Severity C)**: Business hours only, typically 1-3 business days
   - Check email for updates
   - Azure Portal shows request status

### Method 2: Azure CLI (Faster)

```bash
# List current quotas
az ml quota list --location eastus2

# Create quota request
az support tickets create \
  --ticket-name "GPU Quota Request for ML Training" \
  --title "Request GPU quota increase for Azure ML" \
  --description "Requesting 24 vCPUs for Standard_NC6s_v3 in East US 2 for machine learning training workloads" \
  --severity 3 \
  --problem-classification "/providers/Microsoft.Support/services/quotas/problemClassifications/compute"
```

### Method 3: Azure Support Center

1. Visit: https://aka.ms/azuresupport
2. Create new support request
3. Issue type: **Service and subscription limits (quotas)**
4. Quota type: **Machine Learning Service**
5. Select your subscription
6. Region: **East US 2**
7. Quota: **Standard NCSv3 Family vCPUs**
8. New limit: **24**

## 💰 Cost Estimation

### Standard_NC6s_v3 (Tesla V100)

**Low-Priority Pricing** (Recommended):
- Cost per VM: ~$0.90/hour
- 4 VMs for 2 hours training: ~$7.20
- Monthly max (24/7): ~$2,592
- **Auto-scaling to 0 when idle** = Near-zero cost when not training

**Dedicated Pricing**:
- Cost per VM: ~$3.06/hour
- 4 VMs for 2 hours: ~$24.48
- Monthly max: ~$8,812

### Cost-Saving Tips:

1. **Use Low-Priority VMs** (Already configured in Bicep)
2. **Auto-scaling** - Scale to 0 when idle (configured: 0-4 nodes)
3. **Stop compute when not training**:
   ```bash
   az ml compute stop --name gpu-cluster --resource-group rg-jblango-1749
   ```
4. **Set budget alerts** in Azure Cost Management

## 🔧 Deployment After Quota Approval

Once your GPU quota is approved:

### 1. Update Bicep File

Replace `infra/main-no-gpu.bicep` content with original `infra/main.bicep` that includes GPU cluster:

```bicep
// GPU Compute Cluster for training
resource gpuComputeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  name: 'gpu-cluster'
  parent: mlWorkspace
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_NC6s_v3'  // Tesla V100
      vmPriority: 'LowPriority'    // Cost-effective
      scaleSettings: {
        minNodeCount: 0              // Scale to zero
        maxNodeCount: 4              // Up to 4 GPUs
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
}
```

### 2. Deploy Updated Infrastructure

```bash
# Deploy with GPU cluster
az deployment group create \
  --resource-group rg-jblango-1749 \
  --template-file infra/main.bicep \
  --parameters environmentName=millennialai-revolutionary location=eastus2
```

### 3. Submit Training Job

```bash
# Set environment variables
export AZURE_SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
export AZURE_RESOURCE_GROUP="rg-jblango-1749"
export AZURE_ML_WORKSPACE_NAME="<workspace-name-from-deployment>"

# Submit training
python submit_azure_training.py
```

## 🏃 Alternative: Deploy Now Without GPU

While waiting for GPU quota, you can deploy the current infrastructure (without GPU):

```bash
# Deploy CPU-only infrastructure
az deployment group create \
  --resource-group rg-jblango-1749 \
  --template-file infra/main-no-gpu.bicep \
  --parameters environmentName=millennialai-revolutionary location=eastus2
```

This creates:
- ✅ Azure ML Workspace
- ✅ Storage Account with blob containers
- ✅ Key Vault
- ✅ Application Insights
- ✅ CPU compute cluster (for data prep)
- ❌ GPU compute cluster (add later)

## 🧪 Local Training (No GPU Quota Needed)

Train locally while waiting for Azure GPU:

```bash
# Install dependencies
pip install -r requirements.txt

# Train on local GPU (if available)
python train_millennialai.py

# Or use CPU (slower)
CUDA_VISIBLE_DEVICES="" python train_millennialai.py
```

Upload trained models to Azure Storage:

```bash
# Install Azure CLI
pip install azure-storage-blob azure-identity

# Upload model
python -c "
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

storage_account = '<your-storage-account>'
credential = DefaultAzureCredential()
blob_service = BlobServiceClient(
    account_url=f'https://{storage_account}.blob.core.windows.net',
    credential=credential
)

container_client = blob_service.get_container_client('models')
with open('outputs/final_model.pt', 'rb') as data:
    container_client.upload_blob('local_trained_model.pt', data, overwrite=True)

print('✅ Model uploaded to Azure Storage')
"
```

## 📞 Support Resources

### Azure Support Response Times (by Severity)
Based on Azure support classification:
- **Severity A**: Critical issues - 24x7 support, immediate response
- **Severity B**: Important issues - Optionally 24x7 support
- **Severity C**: General requests (like quota increases) - **Business hours only**

**For GPU quota requests**:
- Classified as **Severity C** (business hours only)
- Processing: Monday-Friday, excludes weekends/holidays
- Expected timeline: 1-3 business days from submission
- Weekend submissions process on next business day

### Documentation Links
- **Azure Support**: https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade
- **ML Quota Docs**: https://learn.microsoft.com/azure/machine-learning/how-to-manage-quotas
- **GPU VM Sizes**: https://learn.microsoft.com/azure/virtual-machines/sizes-gpu
- **Azure Calculator**: https://azure.microsoft.com/pricing/calculator/

## ✅ Next Steps

1. **Submit GPU quota request** (using one of the methods above)
2. **Deploy CPU-only infrastructure** (optional, while waiting)
3. **Train locally or on CPU cluster** (to test pipeline)
4. **Once quota approved**: Deploy GPU cluster and submit training jobs
5. **Monitor costs** with Azure Cost Management

---

**Questions?** Check the main [AZURE_ML_GUIDE.md](AZURE_ML_GUIDE.md) for detailed training instructions.
