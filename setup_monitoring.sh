#!/bin/bash

# MillennialAi Monitoring Schedule
# Sets up periodic monitoring of Azure deployment status

LOG_FILE="/home/jovan-blango/Desktop/MillennialAi/monitoring.log"
DASHBOARD_SCRIPT="/home/jovan-blango/Desktop/MillennialAi/dashboard.sh"

# Function to log with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Create log file if it doesn't exist
touch "$LOG_FILE"

echo "🔄 Setting up MillennialAi Azure Monitoring"
echo "=========================================="

# Run initial dashboard
echo "Running initial status check..."
$DASHBOARD_SCRIPT >> "$LOG_FILE" 2>&1
echo "Status logged to: $LOG_FILE"

echo ""
echo "📋 Monitoring Options:"
echo "----------------------"
echo "1. Run manually: ./dashboard.sh"
echo "2. View logs: tail -f monitoring.log"
echo "3. Detailed status: ./monitor_azure.sh"
echo ""

echo "⏰ To set up automatic monitoring every 30 minutes:"
echo "   Run: crontab -e"
echo "   Add: */30 * * * * cd /home/jovan-blango/Desktop/MillennialAi && ./dashboard.sh >> monitoring.log 2>&1"
echo ""

echo "📊 Current Status Summary:"
echo "-------------------------"
# Quick status check
SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
ML_WORKSPACE="azml7c462q3plqeyo"

WS_EXISTS=$(az ml workspace show --name $ML_WORKSPACE --resource-group $RESOURCE_GROUP --query "name" --output tsv 2>/dev/null)
if [ ! -z "$WS_EXISTS" ]; then
    echo "✅ ML Workspace: Created"
else
    echo "❌ ML Workspace: Missing"
fi

CPU_EXISTS=$(az ml compute show --name cpu-cluster --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "name" --output tsv 2>/dev/null)
if [ ! -z "$CPU_EXISTS" ]; then
    echo "✅ CPU Cluster: Created"
else
    echo "❌ CPU Cluster: Missing"
fi

GPU_EXISTS=$(az ml compute show --name gpu-cluster --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE --query "name" --output tsv 2>/dev/null)
if [ ! -z "$GPU_EXISTS" ]; then
    echo "⚠️  GPU Cluster: Created (but failed due to quota)"
else
    echo "❌ GPU Cluster: Not created"
fi

echo ""
echo "🎯 Next Action: Monitor quota approval in Azure Portal"
echo "   Link: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade/~/overview"