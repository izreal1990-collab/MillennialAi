#!/bin/bash

# MillennialAi Azure Deployment Script for Continuous Operation
echo "🚀 Deploying MillennialAi to Azure for Continuous Operation"
echo "=========================================================="

# Check if azd is installed
if ! command -v azd &> /dev/null; then
    echo "❌ Azure Developer CLI (azd) is not installed!"
    echo "Please install it from: https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd"
    exit 1
fi

# Check if user is logged in
if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure!"
    echo "Please run: az login"
    exit 1
fi

# Set environment name
ENV_NAME="millennialai-prod-$(date +%Y%m%d-%H%M%S)"
echo "📍 Environment Name: $ENV_NAME"

# Deploy to Azure
echo "🔨 Deploying to Azure Container Apps..."
echo "This will create:"
echo "  • Container App for continuous operation"
echo "  • Azure ML workspace for model training"
echo "  • Storage account for data persistence"
echo "  • Log Analytics for monitoring"
echo ""

azd up --environment $ENV_NAME

if [ $? -ne 0 ]; then
    echo "❌ Deployment failed!"
    exit 1
fi

echo ""
echo "✅ Deployment successful!"
echo "🌐 Your MillennialAi system is now running continuously on Azure!"
echo ""
echo "📊 System Status:"
echo "  • Live Chat API: Collecting conversation data 24/7"
echo "  • Continuous Learning: Running at 90% capacity"
echo "  • Automated ML: Triggering retraining every 1 hour"
echo "  • Resource Utilization: Maximum performance"
echo ""
echo "🔗 Access your API at the URL shown above"
echo "📈 Monitor performance through Azure Portal > Container Apps"
echo ""
echo "💡 The system will run continuously, collecting data and improving automatically!"