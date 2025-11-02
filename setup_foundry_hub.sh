#!/bin/bash
# AI Foundry Hub Setup for MillennialAi
# Creates Azure AI Foundry Hub with AI Studio integration

set -e

echo "🚀 Setting up Azure AI Foundry Hub for MillennialAi"
echo "=================================================="

# Configuration
SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
LOCATION="eastus2"
HUB_NAME="millennialai-foundry-hub"
PROJECT_NAME="millennialai-project"

echo "📋 Configuration:"
echo "  Subscription: $SUBSCRIPTION_ID"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Hub Name: $HUB_NAME"
echo "  Project Name: $PROJECT_NAME"

# Install/update Azure ML extension
echo ""
echo "🔧 Installing Azure ML extension..."
az extension add --name ml --yes --upgrade

# Create AI Foundry Hub
echo ""
echo "🏗️ Creating AI Foundry Hub..."
az ml workspace create \
    --kind hub \
    --name "$HUB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --description "MillennialAi AI Foundry Hub for advanced AI development with Layer Injection Framework"

if [ $? -eq 0 ]; then
    echo "✅ AI Foundry Hub created successfully!"
else
    echo "⚠️ Hub might already exist, checking..."
fi

# Create AI Project under the hub
echo ""
echo "📦 Creating AI Project..."
az ml workspace create \
    --kind project \
    --name "$PROJECT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --hub-id "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$HUB_NAME" \
    --description "MillennialAi training project with Layer Injection Framework and revolutionary training methods"

if [ $? -eq 0 ]; then
    echo "✅ AI Project created successfully!"
else
    echo "⚠️ Project might already exist, checking..."
fi

# Check the created resources
echo ""
echo "🔍 Validating created resources..."
echo "Listing workspaces in resource group:"
az ml workspace list --resource-group "$RESOURCE_GROUP" --output table

# Generate configuration for easy access
echo ""
echo "📝 Generating AI Foundry configuration..."
cat > ai_foundry_config.json << EOF
{
  "subscription_id": "$SUBSCRIPTION_ID",
  "resource_group": "$RESOURCE_GROUP",
  "hub_name": "$HUB_NAME",
  "project_name": "$PROJECT_NAME",
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
    "Layer injection framework support"
  ],
  "setup_date": "$(date -Iseconds)"
}
EOF

echo "✅ Configuration saved to ai_foundry_config.json"

# Create connection script for easy access
cat > connect_ai_studio.py << 'EOF'
#!/usr/bin/env python3
"""
Connect to AI Studio for MillennialAi project
"""

import json
import webbrowser
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def connect_to_ai_studio():
    """Open AI Studio and connect to MillennialAi project"""
    
    # Load configuration
    with open('ai_foundry_config.json', 'r') as f:
        config = json.load(f)
    
    print("🚀 Connecting to Azure AI Studio")
    print("=" * 35)
    print(f"Hub: {config['hub_name']}")
    print(f"Project: {config['project_name']}")
    print(f"Resource Group: {config['resource_group']}")
    
    # Create ML client for the project
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=config['subscription_id'],
            resource_group_name=config['resource_group'],
            workspace_name=config['project_name']
        )
        
        print("✅ Connected to AI Foundry project!")
        print(f"🌐 Opening AI Studio: {config['ai_studio_url']}")
        
        # Open AI Studio in browser
        webbrowser.open(config['ai_studio_url'])
        
        print("\n🎯 In AI Studio:")
        print("1. Navigate to your subscription")
        print(f"2. Select resource group: {config['resource_group']}")
        print(f"3. Open project: {config['project_name']}")
        print("4. Start building AI applications!")
        
        return ml_client
        
    except Exception as e:
        print(f"❌ Error connecting: {str(e)}")
        return None

if __name__ == "__main__":
    connect_to_ai_studio()
EOF

chmod +x connect_ai_studio.py

echo ""
echo "🎉 AI Foundry Hub setup completed!"
echo ""
echo "🎯 Next Steps:"
echo "1. Access AI Studio: https://ai.azure.com"
echo "2. Navigate to your MillennialAi project"
echo "3. Or run: python connect_ai_studio.py"
echo ""
echo "🔗 Quick Links:"
echo "  AI Studio: https://ai.azure.com"
echo "  Hub: https://ml.azure.com/workspaces/$HUB_NAME"
echo "  Project: https://ml.azure.com/workspaces/$PROJECT_NAME"
echo ""
echo "✅ MillennialAi is now ready for advanced AI development!"