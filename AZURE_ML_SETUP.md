# Azure ML Setup Guide for MillennialAi

This guide will help you set up and run MillennialAi tests in Azure Machine Learning.

## Prerequisites

1. **Azure Subscription** with access to Azure Machine Learning
2. **Azure CLI** installed (`az` command)
3. **Azure ML CLI extension** (`az ml`)

## Quick Setup

### 1. Install Azure CLI and ML Extension

```bash
# Install Azure CLI (if not already installed)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Install Azure ML CLI extension
az extension add -n ml
```

### 2. Login to Azure

```bash
az login
```

### 3. Set Environment Variables

Create a `.env` file in the project root or export these variables:

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group-name"
export AZURE_ML_WORKSPACE_NAME="your-workspace-name"
```

To find these values:

```bash
# List subscriptions
az account list --output table

# List resource groups
az group list --output table

# List ML workspaces
az ml workspace list --output table
```

### 4. Create Azure ML Resources (if needed)

```bash
# Create resource group
az group create --name my-resource-group --location eastus

# Create ML workspace
az ml workspace create --name my-workspace --resource-group my-resource-group --location eastus

# Create compute cluster (CPU for testing)
az ml compute create --name cpu-cluster --type AmlCompute --size Standard_DS3_v2 --min-instances 0 --max-instances 4 --resource-group my-resource-group --workspace-name my-workspace
```

## Running Tests

### Local Testing (Recommended First)

Test locally before running in Azure ML:

```bash
python azure_ml_test.py
```

This will validate:
- ✅ PyTorch functionality
- ✅ MillennialAi configuration
- ✅ Model loading and inference

### Azure ML Testing

Once environment is configured:

```bash
python submit_azure_test.py
```

This will:
1. Connect to your Azure ML workspace
2. Create/update the test environment
3. Submit the test job to run in the cloud
4. Provide links to monitor progress

## Monitoring Jobs

### In Azure ML Studio

The submission script provides a direct link to view your job in Azure ML Studio.

### Using Azure CLI

```bash
# List recent jobs
az ml job list --output table

# Stream job logs
az ml job stream --name your-job-name

# Download job outputs
az ml job download --name your-job-name --download-path ./outputs
```

## Troubleshooting

### Common Issues

1. **"Authentication failed"**
   - Run `az login` again
   - Check that your account has access to the subscription

2. **"Workspace not found"**
   - Verify the workspace name and resource group
   - Check that the workspace exists: `az ml workspace list`

3. **"Compute cluster not available"**
   - The default is `cpu-cluster`
   - Create it if it doesn't exist (see setup above)
   - Or change the compute name in `submit_azure_test.py`

4. **"Environment creation failed"**
   - Check the `conda.yml` file for syntax errors
   - Ensure all packages are available

### Environment Variables

If you prefer not to use environment variables, you can:

1. **Create a `.env` file**:
   ```
   AZURE_SUBSCRIPTION_ID=your-subscription-id
   AZURE_RESOURCE_GROUP=your-resource-group
   AZURE_ML_WORKSPACE_NAME=your-workspace
   ```

2. **Modify the scripts** to hardcode values (not recommended for security)

## Test Results

The test suite validates:

- **PyTorch Operations**: Basic tensor operations and CUDA availability
- **MillennialAi Config**: Configuration loading and preset validation
- **Model Loading**: Loading pre-trained models from Hugging Face
- **Inference**: Running actual model inference with sample prompts

Expected output shows 4/4 tests passing with detailed logging.

## Next Steps

After successful testing:

1. **Scale up**: Modify `submit_azure_training.py` for full training jobs
2. **GPU Training**: Change compute to a GPU cluster for faster training
3. **Distributed Training**: Configure multi-node training for large models
4. **MLOps**: Set up pipelines for continuous training and deployment

## Support

For issues with Azure ML setup:
- Check Azure ML documentation: https://docs.microsoft.com/azure/machine-learning
- MillennialAi specific issues: Check the test logs and GitHub issues