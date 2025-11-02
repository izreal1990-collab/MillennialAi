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
