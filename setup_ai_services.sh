#!/bin/bash
# Create AI Services resource for MillennialAi AI Studio integration

set -e

echo "🔧 Setting up AI Services for MillennialAi AI Studio Integration"
echo "==============================================================="

# Configuration
SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
LOCATION="eastus2"
AI_SERVICE_NAME="millennialai-ai-services"
HUB_NAME="millennialai-foundry-hub"
PROJECT_NAME="millennialai-project"

echo "📋 Creating AI Services resource for MillennialAi..."
echo "  AI Service Name: $AI_SERVICE_NAME"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"

# Create AI Services resource (this is what AI Studio recognizes)
echo ""
echo "🏗️ Creating AI Services resource..."
az cognitiveservices account create \
    --name "$AI_SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --kind "AIServices" \
    --sku "S0" \
    --custom-domain "$AI_SERVICE_NAME" \
    --assign-identity \
    --tags project="MillennialAi" framework="LayerInjection"

if [ $? -eq 0 ]; then
    echo "✅ AI Services resource created successfully!"
else
    echo "⚠️ AI Services resource might already exist or there was an issue"
fi

# Get the AI Services resource details
echo ""
echo "📊 Getting AI Services details..."
AI_SERVICE_ENDPOINT=$(az cognitiveservices account show \
    --name "$AI_SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.endpoint" \
    --output tsv)

AI_SERVICE_ID=$(az cognitiveservices account show \
    --name "$AI_SERVICE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "id" \
    --output tsv)

echo "  Endpoint: $AI_SERVICE_ENDPOINT"
echo "  Resource ID: $AI_SERVICE_ID"

# Connect AI Services to the AI Foundry Hub
echo ""
echo "🔗 Connecting AI Services to AI Foundry Hub..."

# Create connection between AI Services and the hub
az ml connection create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$HUB_NAME" \
    --file - << EOF
name: millennialai-ai-connection
type: azure_ai_services
target: $AI_SERVICE_ENDPOINT
ai_services_resource_id: $AI_SERVICE_ID
EOF

if [ $? -eq 0 ]; then
    echo "✅ AI Services connected to Foundry Hub!"
else
    echo "⚠️ Connection might already exist"
fi

# List all AI services to verify
echo ""
echo "🔍 Verification - All AI Services in resource group:"
az cognitiveservices account list --resource-group "$RESOURCE_GROUP" --output table

# Update configuration file
echo ""
echo "📝 Updating AI Foundry configuration..."
cat > ai_foundry_config.json << EOF
{
  "subscription_id": "$SUBSCRIPTION_ID",
  "resource_group": "$RESOURCE_GROUP",
  "hub_name": "$HUB_NAME",
  "project_name": "$PROJECT_NAME",
  "ai_services_name": "$AI_SERVICE_NAME",
  "ai_services_endpoint": "$AI_SERVICE_ENDPOINT",
  "location": "$LOCATION",
  "ai_studio_url": "https://ai.azure.com",
  "hub_endpoint": "https://ml.azure.com/workspaces/$HUB_NAME",
  "project_endpoint": "https://ml.azure.com/workspaces/$PROJECT_NAME",
  "capabilities": [
    "Model training and fine-tuning",
    "Model deployment and serving", 
    "Prompt engineering and testing",
    "AI application development",
    "Integration with existing ML workspace",
    "Advanced monitoring and observability",
    "Layer injection framework support",
    "AI Services integration"
  ],
  "setup_date": "$(date -Iseconds)"
}
EOF

echo "✅ Configuration updated with AI Services details"

echo ""
echo "🎉 AI Services setup completed!"
echo ""
echo "🎯 Now in AI Studio (https://ai.azure.com):"
echo "1. You should see '$AI_SERVICE_NAME' as an available resource"
echo "2. Navigate to Projects and select 'millennialai-project'"
echo "3. The project should now be properly connected to AI Services"
echo ""
echo "🔗 Direct Links:"
echo "  AI Studio: https://ai.azure.com"
echo "  AI Services: $AI_SERVICE_ENDPOINT"
echo "  Project: https://ml.azure.com/workspaces/$PROJECT_NAME"
echo ""
echo "✅ MillennialAi should now be visible in AI Studio!"