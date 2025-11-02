#!/usr/bin/env python3
"""
AI Foundry Developer Tools for MillennialAi
Advanced AI development capabilities using Azure AI Foundry
"""

import json
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model, Environment, Component, Job
from azure.ai.ml.constants import AssetTypes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIFoundryDeveloper:
    """Advanced AI development tools for Azure AI Foundry"""
    
    def __init__(self):
        """Initialize AI Foundry connection"""
        self.config = self._load_config()
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=self.config['subscription_id'],
            resource_group_name=self.config['resource_group'],
            workspace_name=self.config['project_name']
        )
        logger.info(f"🚀 Connected to AI Foundry: {self.config['project_name']}")

    def _load_config(self):
        """Load AI Foundry configuration"""
        try:
            with open('ai_foundry_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("❌ AI Foundry config not found. Run setup_ai_foundry.py first")
            raise

    def list_models(self):
        """List all models in the AI Foundry workspace"""
        logger.info("📋 Listing AI Foundry models...")
        try:
            models = list(self.ml_client.models.list())
            if models:
                for model in models[:10]:  # Show first 10
                    logger.info(f"   📦 {model.name} v{model.version} - {model.description[:50]}...")
            else:
                logger.info("   No models found in workspace")
            return models
        except Exception as e:
            logger.error(f"❌ Error listing models: {e}")
            return []

    def list_environments(self):
        """List all environments in the AI Foundry workspace"""
        logger.info("🌍 Listing AI Foundry environments...")
        try:
            environments = list(self.ml_client.environments.list())
            if environments:
                for env in environments[:10]:  # Show first 10
                    logger.info(f"   🐳 {env.name} v{env.version}")
            else:
                logger.info("   No custom environments found")
            return environments
        except Exception as e:
            logger.error(f"❌ Error listing environments: {e}")
            return []

    def list_components(self):
        """List all components in the AI Foundry workspace"""
        logger.info("🔧 Listing AI Foundry components...")
        try:
            components = list(self.ml_client.components.list())
            if components:
                for comp in components[:10]:  # Show first 10
                    logger.info(f"   ⚙️  {comp.name} v{comp.version} - {comp.type}")
            else:
                logger.info("   No custom components found")
            return components
        except Exception as e:
            logger.error(f"❌ Error listing components: {e}")
            return []

    def create_millennialai_model_registration(self):
        """Register MillennialAi model in AI Foundry"""
        logger.info("📦 Registering MillennialAi model...")
        try:
            model = Model(
                name="millennialai-forward-injection",
                version="1",  # Integer version for Azure ML
                description="MillennialAi Forward Layer Injection Framework - Revolutionary AI Architecture",
                type=AssetTypes.CUSTOM_MODEL,
                path="./millennial_ai/",
                tags={
                    "framework": "pytorch",
                    "architecture": "forward-injection",
                    "innovation": "layer-injection",
                    "status": "revolutionary"
                },
                properties={
                    "injection_layers": "[32]",
                    "hidden_size": "4096",
                    "reasoning_engine": "adaptive",
                    "license": "proprietary"
                }
            )
            
            registered_model = self.ml_client.models.create_or_update(model)
            logger.info(f"✅ Model registered: {registered_model.name} v{registered_model.version}")
            return registered_model
            
        except Exception as e:
            logger.error(f"❌ Error registering model: {e}")
            return None

    def create_millennialai_environment(self):
        """Create MillennialAi development environment"""
        logger.info("🌍 Creating MillennialAi environment...")
        try:
            env = Environment(
                name="millennialai-dev-env",
                version="2",  # Integer version
                description="Development environment for MillennialAi Forward Injection Framework",
                conda_file="environment.yml",
                image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest",
                tags={
                    "purpose": "millennialai-development",
                    "gpu": "required",
                    "framework": "pytorch"
                }
            )
            
            created_env = self.ml_client.environments.create_or_update(env)
            logger.info(f"✅ Environment created: {created_env.name} v{created_env.version}")
            return created_env
            
        except Exception as e:
            logger.error(f"❌ Error creating environment: {e}")
            return None

    def create_training_component(self):
        """Create MillennialAi training component"""
        logger.info("⚙️ Creating MillennialAi training component...")
        
        # Create component YAML content
        component_yaml = """
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: millennialai_training
version: 1.0.0
display_name: MillennialAi Forward Injection Training
description: Train MillennialAi forward injection model with adaptive reasoning

type: command

inputs:
  training_data:
    type: uri_folder
    description: Training dataset
  config_preset:
    type: string
    default: "minimal"
    description: MillennialAi configuration preset
  max_epochs:
    type: integer
    default: 10
    description: Maximum training epochs

outputs:
  trained_model:
    type: uri_folder
    description: Trained MillennialAi model
  metrics:
    type: uri_file
    description: Training metrics

code: ./millennial_ai/

environment: azureml:millennialai-dev-env:1.0.0

command: >-
  python training_script.py
  --data ${{inputs.training_data}}
  --config ${{inputs.config_preset}}
  --epochs ${{inputs.max_epochs}}
  --output ${{outputs.trained_model}}
  --metrics ${{outputs.metrics}}
"""
        
        try:
            # Save component file
            with open('millennialai_training_component.yml', 'w') as f:
                f.write(component_yaml)
            
            component = Component.load('millennialai_training_component.yml')
            created_component = self.ml_client.components.create_or_update(component)
            logger.info(f"✅ Component created: {created_component.name} v{created_component.version}")
            return created_component
            
        except Exception as e:
            logger.error(f"❌ Error creating component: {e}")
            return None

    def list_compute_targets(self):
        """List available compute targets"""
        logger.info("💻 Listing compute targets...")
        try:
            computes = list(self.ml_client.compute.list())
            for compute in computes:
                logger.info(f"   🖥️  {compute.name} ({compute.type}) - {compute.provisioning_state}")
            return computes
        except Exception as e:
            logger.error(f"❌ Error listing compute: {e}")
            return []

    def check_gpu_quota(self):
        """Check GPU quota status"""
        logger.info("🎯 Checking GPU quota...")
        try:
            # This would need specific quota API calls
            logger.info("   Use Azure portal to check quota: https://portal.azure.com/#view/Microsoft_Azure_Capacity/QuotaMenuBlade")
            logger.info("   Or run: az vm list-usage --location eastus")
        except Exception as e:
            logger.error(f"❌ Error checking quota: {e}")

    def setup_foundry_workspace(self):
        """Complete AI Foundry workspace setup"""
        logger.info("🏗️ Setting up complete AI Foundry workspace...")
        
        results = {
            "models": self.list_models(),
            "environments": self.list_environments(), 
            "components": self.list_components(),
            "compute": self.list_compute_targets()
        }
        
        # Create MillennialAi assets
        logger.info("\n🚀 Creating MillennialAi assets...")
        results["model_registration"] = self.create_millennialai_model_registration()
        results["environment_creation"] = self.create_millennialai_environment()
        results["component_creation"] = self.create_training_component()
        
        return results

def main():
    """Main AI Foundry development function"""
    print("🚀 MILLENNIALAI AI FOUNDRY DEVELOPER TOOLS")
    print("=" * 50)
    
    try:
        developer = AIFoundryDeveloper()
        
        # Setup complete workspace
        results = developer.setup_foundry_workspace()
        
        print("\n📊 SETUP SUMMARY")
        print("=" * 30)
        print(f"✅ Models found: {len(results['models'])}")
        print(f"✅ Environments found: {len(results['environments'])}")
        print(f"✅ Components found: {len(results['components'])}")
        print(f"✅ Compute targets: {len(results['compute'])}")
        
        if results['model_registration']:
            print("✅ MillennialAi model registered")
        if results['environment_creation']:
            print("✅ MillennialAi environment created")
        if results['component_creation']:
            print("✅ MillennialAi training component created")
            
        print("\n🎯 Next Steps:")
        print("1. Open AI Studio: https://ai.azure.com")
        print("2. Navigate to your project: millennialai-project")
        print("3. Explore models, environments, and components")
        print("4. Start building AI applications!")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    main()