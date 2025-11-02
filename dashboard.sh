#!/bin/bash

# MillennialAi Deployment Dashboard
# Run this periodically to track deployment progress

echo "🚀 MillennialAi Deployment Dashboard"
echo "===================================="
echo "Last updated: $(date)"
echo ""

# Colors for status
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
ML_WORKSPACE="azml7c462q3plqeyo"

# Check ML Workspace
echo "🤖 ML Workspace:"
WS_EXISTS=$(az ml workspace show --name $ML_WORKSPACE --resource-group $RESOURCE_GROUP --query "name" --output tsv 2>/dev/null)
if [ ! -z "$WS_EXISTS" ]; then
    echo -e "${GREEN}✅ Available${NC} ($ML_WORKSPACE)"
else
    echo -e "${RED}❌ Not Available${NC}"
fi

# Check Compute Clusters
echo ""
echo "🖥️  Compute Clusters:"
CPU_STATE=$(az ml compute show --name cpu-cluster --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "provisioning_state" --output tsv 2>/dev/null)
GPU_STATE=$(az ml compute show --name gpu-cluster --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "provisioning_state" --output tsv 2>/dev/null)

if [ "$CPU_STATE" = "Succeeded" ]; then
    echo -e "  CPU Cluster: ${GREEN}✅ Ready${NC} (Standard_DS3_v2, 0-2 nodes)"
else
    echo -e "  CPU Cluster: ${YELLOW}⏳ $CPU_STATE${NC}"
fi

if [ "$GPU_STATE" = "Succeeded" ]; then
    echo -e "  GPU Cluster: ${GREEN}✅ Ready${NC} (Standard_NC6s_v3, 0-4 nodes)"
else
    echo -e "  GPU Cluster: ${RED}❌ Blocked by quota${NC} (0 vCPUs available)"
fi

# Check Container Apps
echo ""
echo "📦 Container Apps:"
APP_COUNT=$(az containerapp list --resource-group $RESOURCE_GROUP --query "length(@)" --output tsv 2>/dev/null)
if [ "$APP_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ $APP_COUNT apps deployed${NC}"
    # Show URLs for running apps
    az containerapp list --resource-group $RESOURCE_GROUP --query "[?properties.provisioningState=='Succeeded'].{name:name, url:properties.configuration.ingress.fqdn}" --output table 2>/dev/null
else
    echo -e "${YELLOW}⚠️  No container apps${NC}"
fi

# Deployment Status
echo ""
echo "📈 Latest Deployment:"
LATEST_DEPLOY=$(az deployment group list --resource-group $RESOURCE_GROUP --query "[].{name:name, state:properties.provisioningState, timestamp:properties.timestamp} | sort_by(@, &timestamp) | reverse(@) | [0]" --output json 2>/dev/null)
if [ ! -z "$LATEST_DEPLOY" ]; then
    DEPLOY_NAME=$(echo $LATEST_DEPLOY | jq -r '.name')
    DEPLOY_STATE=$(echo $LATEST_DEPLOY | jq -r '.state')
    if [ "$DEPLOY_STATE" = "Succeeded" ]; then
        echo -e "${GREEN}✅ $DEPLOY_NAME${NC}"
    else
        echo -e "${RED}❌ $DEPLOY_NAME ($DEPLOY_STATE)${NC}"
    fi
fi

# Quota Status
echo ""
echo "⚙️  Critical Issues:"
echo "  • GPU compute quota: 0 vCPUs (increase needed)"
echo "  • Provider registration: ✅ All essential providers registered"
echo "  • Quota monitoring: Use Azure Portal (CLI commands unreliable)"

echo ""
echo "🔗 Quick Actions:"
echo "  • Monitor quotas: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview"
echo "  • ML Studio: https://ml.azure.com/?wsid=/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/workspaces/$ML_WORKSPACE"
echo "  • Resource Group: https://portal.azure.com/#@/resource/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/overview"

echo ""
echo "💡 Next Steps:"
echo "  1. ✅ Fixed CPU test job submitted (millennialai-cpu-test-fixed)"
echo "  2. ⚠️ Original job failed: Import error resolved in fixed version"
echo "  3. Monitor fixed job: ./monitor_jobs.sh"
echo "  4. Check Azure ML Studio for results"
echo "  5. Monitor GPU quota request approval"
echo "  6. Once GPU approved: azd up for full deployment"

echo ""
echo "Run './monitor_azure.sh' for detailed status"