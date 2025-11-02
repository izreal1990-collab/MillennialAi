#!/usr/bin/env python3
"""
Azure AI Foundry Hub Setup for MillennialAi

This script sets up a proper AI Foundry Hub with AI Studio integration
for advanced AI development capabilities.
"""

import os
import json
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient


def setup_ai_foundry_hub():
    """Set up Azure AI Foundry Hub for MillennialAi"""
    
    print("🚀 Setting up Azure AI Foundry Hub for MillennialAi")
    print("=" * 55)
    
    # Configuration
    subscription_id = "639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
    resource_group = "rg-jblango-1749"
    location = "eastus2"
    
    # AI Foundry Hub configuration
    hub_name = "millennialai-foundry-hub"
    project_name = "millennialai-project"
    
    try:
        # Get credentials
        credential = DefaultAzureCredential()
        
        # Create ML client
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group
        )
        
        print(f"✅ Connected to Azure subscription: {subscription_id}")
        print(f"📁 Resource Group: {resource_group}")
        print(f"🌍 Location: {location}")
        
        # Check existing AI services
        print("\n🔍 Checking existing AI services...")
        
        # Create resource management client
        resource_client = ResourceManagementClient(credential, subscription_id)
        cognitive_client = CognitiveServicesManagementClient(credential, subscription_id)
        
        # List existing cognitive services
        existing_services = list(cognitive_client.accounts.list_by_resource_group(resource_group))
        
        print(f"📊 Found {len(existing_services)} existing AI services:")
        for service in existing_services:
            print(f"  - {service.name} ({service.kind}) - {service.location}")
        
        # Check if we need to create AI Foundry Hub
        print(f"\n🏗️ Setting up AI Foundry Hub: {hub_name}")
        
        # Create AI Foundry Hub using Azure CLI (more reliable)
        print("📦 Creating AI Foundry Hub via Azure CLI...")
        
        # Generate hub creation script
        setup_script = f"""
# Create AI Foundry Hub for MillennialAi
az extension add --name ml --yes

# Create the AI Hub (if not exists)
az ml workspace create \\
    --kind hub \\
    --name {hub_name} \\
    --resource-group {resource_group} \\
    --location {location} \\
    --description "MillennialAi AI Foundry Hub for advanced AI development" \\
    --friendly-name "MillennialAi Foundry Hub"

# Create AI Project under the hub
az ml workspace create \\
    --kind project \\
    --name {project_name} \\
    --resource-group {resource_group} \\
    --location {location} \\
    --hub-id /subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{hub_name} \\
    --description "MillennialAi training project with Layer Injection Framework" \\
    --friendly-name "MillennialAi Project"

echo "✅ AI Foundry Hub setup completed!"
"""
        
        # Save setup script
        with open('setup_foundry_hub.sh', 'w') as f:
            f.write(setup_script)
        
        print("✅ Generated setup script: setup_foundry_hub.sh")
        print("\n🎯 Next Steps:")
        print("1. Run: chmod +x setup_foundry_hub.sh")
        print("2. Run: ./setup_foundry_hub.sh")
        print("3. Access AI Studio: https://ai.azure.com")
        
        # Generate AI Studio configuration
        ai_studio_config = {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "hub_name": hub_name,
            "project_name": project_name,
            "location": location,
            "ai_studio_url": "https://ai.azure.com",
            "capabilities": [
                "Model training and fine-tuning",
                "Model deployment and serving",
                "Prompt engineering and testing",
                "AI application development",
                "Integration with existing ML workspace",
                "Advanced monitoring and observability"
            ]
        }
        
        with open('ai_foundry_config.json', 'w') as f:
            json.dump(ai_studio_config, f, indent=2)
        
        print("✅ Generated AI Foundry configuration: ai_foundry_config.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up AI Foundry Hub: {str(e)}")
        return False


def validate_ai_foundry_setup():
    """Validate AI Foundry Hub setup"""
    
    print("\n🔍 Validating AI Foundry Setup")
    print("=" * 35)
    
    # Check configuration files
    configs_to_check = [
        'setup_foundry_hub.sh',
        'ai_foundry_config.json'
    ]
    
    for config in configs_to_check:
        if os.path.exists(config):
            print(f"✅ {config} - Created")
        else:
            print(f"❌ {config} - Missing")
    
    print("\n📝 AI Foundry Setup Instructions:")
    print("1. Execute the setup script to create the hub")
    print("2. Access AI Studio at https://ai.azure.com")
    print("3. Navigate to your MillennialAi project")
    print("4. Configure model deployments and endpoints")
    print("5. Integrate with existing ML workspace")


if __name__ == "__main__":
    success = setup_ai_foundry_hub()
    if success:
        validate_ai_foundry_setup()
        print("\n🎉 AI Foundry setup preparation completed!")
    else:
        print("\n❌ AI Foundry setup failed. Check errors above.")