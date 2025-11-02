#!/bin/bash

# MillennialAi Job Monitor
# Tracks the status of Azure ML jobs

echo "📊 MillennialAi Job Monitor"
echo "=========================="
echo "Last check: $(date)"
echo ""

RESOURCE_GROUP="rg-jblango-1749"
WORKSPACE="azml7c462q3plqeyo"

# Check current jobs
echo "🏃 Active Jobs:"
echo "---------------"
az ml job list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --max-results 5 --query "[].{name:name, status:status, created:creation_context.created_at, compute:services.Studio.endpoint}" --output table

echo ""
echo "🔍 Latest Job Details:"
echo "---------------------"
LATEST_JOB=$(az ml job list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --max-results 1 --query "[0].name" --output tsv)

if [ ! -z "$LATEST_JOB" ]; then
    echo "Job: $LATEST_JOB"
    az ml job show --name $LATEST_JOB --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --query "{status:status, startTime:creation_context.created_at, compute:compute, duration:duration}" --output table
    
    echo ""
    echo "📝 Job Logs (if available):"
    echo "---------------------------"
    az ml job show --name $LATEST_JOB --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE --query "log_files" --output table 2>/dev/null || echo "Logs not yet available"
else
    echo "No jobs found"
fi

echo ""
echo "🔗 Monitor Links:"
echo "----------------"
echo "Azure ML Studio: https://ml.azure.com/?wsid=/subscriptions/639e13e9-b4be-4ba6-8e9e-f14db5b3a65c/resourcegroups/rg-jblango-1749/workspaces/azml7c462q3plqeyo"
echo "Jobs: https://ml.azure.com/jobs"

echo ""
echo "💡 Quick Commands:"
echo "-----------------"
echo "Check job status: az ml job show --name $LATEST_JOB --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE"
echo "Stream logs: az ml job stream --name $LATEST_JOB --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE"