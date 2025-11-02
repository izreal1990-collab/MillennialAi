#!/bin/bash

# MillennialAi Azure Monitoring Script
# Monitors deployment status, quota usage, and resource health

echo "🔍 MillennialAi Azure Monitor - $(date)"
echo "========================================"

SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
ML_WORKSPACE="azml7c462q3plqeyo"

echo ""
echo "📊 Subscription & Resource Group Status:"
echo "----------------------------------------"
az account show --query "{subscription:name, id:id}" --output table
az group show --name $RESOURCE_GROUP --query "{name:name, location:location, status:properties.provisioningState}" --output table

echo ""
echo "🤖 Azure ML Workspace Status:"
echo "-----------------------------"
az ml workspace show --name $ML_WORKSPACE --resource-group $RESOURCE_GROUP --query "{name:name, location:location, state:provisioningState}" --output table

echo ""
echo "🖥️  Compute Clusters Status:"
echo "---------------------------"
az ml compute list --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "[].{name:name, type:computeType, state:provisioningState, size:properties.vmSize, min:min_instances, max:max_instances}" --output table

echo ""
echo "📦 Container Apps Status:"
echo "-------------------------"
az containerapp list --resource-group $RESOURCE_GROUP --query "[].{name:name, location:location, state:provisioningState, url:properties.configuration.ingress.fqdn}" --output table

echo ""
echo "💾 Storage Accounts:"
echo "-------------------"
az storage account list --resource-group $RESOURCE_GROUP --query "[].{name:name, location:location, kind:kind, tier:sku.tier}" --output table

echo ""
echo "🔐 Key Vault Status:"
echo "-------------------"
az keyvault list --resource-group $RESOURCE_GROUP --query "[].{name:name, location:location, enabledForDeployment:properties.enabledForDeployment}" --output table

echo ""
echo "📈 Recent Deployments (Last 5):"
echo "-------------------------------"
az deployment group list --resource-group $RESOURCE_GROUP --query "[].{name:name, state:properties.provisioningState, timestamp:properties.timestamp} | sort_by(@, &timestamp) | reverse(@) | [0:5]" --output table

echo ""
echo "⚠️  Recent Deployment Errors:"
echo "----------------------------"
LATEST_FAILED=$(az deployment group list --resource-group $RESOURCE_GROUP --query "[?properties.provisioningState=='Failed'].name | [0]" --output tsv)
if [ ! -z "$LATEST_FAILED" ]; then
    echo "Latest failed deployment: $LATEST_FAILED"
    az deployment group show --name $LATEST_FAILED --resource-group $RESOURCE_GROUP --query "properties.error.details[].{code:code, message:message}" --output table 2>/dev/null || echo "Could not retrieve detailed error information"
else
    echo "No recent failed deployments found."
fi

echo ""
echo "📊 Quota Status Check:"
echo "----------------------"
echo "Note: CLI quota commands may not work reliably. Use Azure Portal for accurate quota information:"
echo "https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview"
echo ""
echo "Key quotas to monitor:"
echo "- Standard NCSv3 Family vCPUs (for GPU compute)"
echo "- Standard DSv3 Family vCPUs (for CPU compute)"
echo "- Total Regional vCPUs"
echo ""
echo "Current known limits:"
echo "- GPU vCPUs: 0 (blocking deployment)"
echo "- Request status: Check Azure Portal for approval status"

echo ""
echo "💡 Recommendations:"
echo "------------------"
CPU_CLUSTER_STATE=$(az ml compute show --name cpu-cluster --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "provisioningState" --output tsv)
GPU_CLUSTER_STATE=$(az ml compute show --name gpu-cluster --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "provisioningState" --output tsv)

if [ "$CPU_CLUSTER_STATE" = "Succeeded" ]; then
    echo "✅ CPU cluster is ready for data preprocessing and CPU-based tasks"
fi

if [ "$GPU_CLUSTER_STATE" = "Failed" ]; then
    echo "❌ GPU cluster failed - quota increase needed for GPU workloads"
    echo "   Check: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview"
fi

echo ""
echo "🔗 Useful Links:"
echo "---------------"
echo "Azure Portal: https://portal.azure.com/#@/resource/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/overview"
echo "ML Studio: https://ml.azure.com/?wsid=/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/workspaces/$ML_WORKSPACE"
echo "Quota Management: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview"

echo ""
echo "Monitor completed at $(date)"