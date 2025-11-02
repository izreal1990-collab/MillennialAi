#!/bin/bash
# GitHub Integration Setup Helper for MillennialAi

set -e

echo "🐙 Setting up GitHub Integration for MillennialAi"
echo "=============================================="

# Configuration
SUBSCRIPTION_ID="639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
RESOURCE_GROUP="rg-jblango-1749"
WORKSPACE_NAME="azml7c462q3plqeyo"
HUB_NAME="millennialai-foundry-hub"

echo "📋 Azure Configuration:"
echo "  Subscription: $SUBSCRIPTION_ID"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  ML Workspace: $WORKSPACE_NAME"
echo "  Foundry Hub: $HUB_NAME"

# Check if user is logged in to Azure
echo ""
echo "🔍 Checking Azure login status..."
if ! az account show > /dev/null 2>&1; then
    echo "❌ Not logged in to Azure. Please run 'az login' first."
    exit 1
fi

echo "✅ Azure login confirmed"

# Get current user info
CURRENT_USER=$(az account show --query user.name --output tsv)
TENANT_ID=$(az account show --query tenantId --output tsv)

echo "  Current user: $CURRENT_USER"
echo "  Tenant ID: $TENANT_ID"

# Create service principal for GitHub Actions
echo ""
echo "🔧 Creating service principal for GitHub Actions..."

SP_NAME="GitHub-MillennialAi-Actions-$(date +%s)"
echo "  Service Principal Name: $SP_NAME"

# Create the service principal
echo "Creating service principal..."
SP_OUTPUT=$(az ad sp create-for-rbac \
    --name "$SP_NAME" \
    --role "Contributor" \
    --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
    --sdk-auth)

if [ $? -eq 0 ]; then
    echo "✅ Service principal created successfully!"
else
    echo "❌ Failed to create service principal"
    exit 1
fi

# Extract service principal details
CLIENT_ID=$(echo "$SP_OUTPUT" | jq -r '.clientId')
echo "  Client ID: $CLIENT_ID"

# Add additional role assignments
echo ""
echo "🔐 Adding additional role assignments..."

# Cognitive Services Contributor
az role assignment create \
    --assignee "$CLIENT_ID" \
    --role "Cognitive Services Contributor" \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP"

# Machine Learning Workspace Contributor  
az role assignment create \
    --assignee "$CLIENT_ID" \
    --role "AzureML Data Scientist" \
    --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP"

echo "✅ Role assignments completed"

# Generate GitHub secrets configuration
echo ""
echo "📝 Generating GitHub secrets configuration..."

cat > github_secrets.txt << EOF
# GitHub Secrets Configuration for MillennialAi
# Copy these values to GitHub → Settings → Secrets and variables → Actions

## Azure Authentication (JSON format - copy exactly as shown)
AZURE_CREDENTIALS:
$SP_OUTPUT

## Azure Resource Configuration
AZURE_SUBSCRIPTION_ID:
$SUBSCRIPTION_ID

AZURE_RESOURCE_GROUP:
$RESOURCE_GROUP

AZURE_ML_WORKSPACE:
$WORKSPACE_NAME

AZURE_FOUNDRY_HUB:
$HUB_NAME

## Setup Instructions:
# 1. Go to: https://github.com/izreal1990-collab/MillennialAi/settings/secrets/actions
# 2. Click "New repository secret"
# 3. Copy each name and value above
# 4. Save each secret

## Test the setup:
# Push any commit to main branch to trigger GitHub Actions workflow
EOF

echo "✅ Configuration saved to github_secrets.txt"

# Test the service principal
echo ""
echo "🧪 Testing service principal access..."

# Test Azure ML access
echo "Testing Azure ML workspace access..."
az ml workspace show \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ Azure ML workspace access confirmed"
else
    echo "⚠️ Azure ML workspace access test failed"
fi

# Test Foundry Hub access
echo "Testing AI Foundry Hub access..."
az ml workspace show \
    --name "$HUB_NAME" \
    --resource-group "$RESOURCE_GROUP" > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ AI Foundry Hub access confirmed"
else
    echo "⚠️ AI Foundry Hub access test failed"
fi

echo ""
echo "🎉 GitHub integration setup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Review the configuration in github_secrets.txt"
echo "2. Add each secret to GitHub: https://github.com/izreal1990-collab/MillennialAi/settings/secrets/actions"
echo "3. Commit and push to trigger the first workflow run"
echo "4. Monitor workflow at: https://github.com/izreal1990-collab/MillennialAi/actions"
echo ""
echo "🔗 Files created:"
echo "  - github_secrets.txt (GitHub secrets configuration)"
echo "  - GITHUB_SETUP.md (detailed setup guide)"
echo "  - .github/workflows/azure-integration.yml (GitHub Actions workflow)"
echo ""
echo "✅ Ready for GitHub Actions integration!"

# Display important information
echo ""
echo "🚨 IMPORTANT: Save this service principal information securely!"
echo "Service Principal Details:"
echo "=========================="
echo "$SP_OUTPUT" | jq '.'