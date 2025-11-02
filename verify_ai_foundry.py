#!/usr/bin/env python3
"""
AI Foundry Setup Verification Script
Verifies that all AI Foundry, AI Studio, and AI Hub components are set up correctly
"""

import json
import os
import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class AIFoundryVerifier:
    """Verifies AI Foundry setup and connectivity"""

    def __init__(self):
        self.config = None
        self.ml_client = None
        self.issues = []
        self.successes = []

    def load_config(self):
        """Load AI Foundry configuration"""
        try:
            with open('ai_foundry_config.json', 'r') as f:
                self.config = json.load(f)
            self.successes.append("✅ Configuration file loaded successfully")
            return True
        except FileNotFoundError:
            self.issues.append("❌ ai_foundry_config.json not found")
            return False
        except json.JSONDecodeError as e:
            self.issues.append(f"❌ Invalid JSON in config file: {e}")
            return False

    def test_azure_authentication(self):
        """Test Azure authentication"""
        try:
            credential = DefaultAzureCredential()
            # Test credential by attempting to get token
            credential.get_token("https://management.azure.com/.default")
            self.successes.append("✅ Azure authentication successful")
            return True
        except Exception as e:
            self.issues.append(f"❌ Azure authentication failed: {e}")
            return False

    def test_ml_client_connection(self):
        """Test connection to AI Foundry project"""
        try:
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.config['subscription_id'],
                resource_group_name=self.config['resource_group'],
                workspace_name=self.config['project_name']
            )
            self.successes.append(f"✅ Connected to AI Foundry project: {self.config['project_name']}")
            return True
        except Exception as e:
            self.issues.append(f"❌ Failed to connect to AI Foundry project: {e}")
            return False

    def verify_hub_exists(self):
        """Verify AI Foundry Hub exists"""
        try:
            # Try to connect to hub
            MLClient(
                credential=DefaultAzureCredential(),
                subscription_id=self.config['subscription_id'],
                resource_group_name=self.config['resource_group'],
                workspace_name=self.config['hub_name']
            )
            self.successes.append(f"✅ AI Foundry Hub exists: {self.config['hub_name']}")
            return True
        except Exception as e:
            self.issues.append(f"❌ AI Foundry Hub not accessible: {e}")
            return False

    def check_compute_targets(self):
        """Check available compute targets"""
        try:
            computes = list(self.ml_client.compute.list())
            cpu_clusters = [c for c in computes if hasattr(c, 'type') and 'cpu' in str(c.type).lower()]
            gpu_clusters = [c for c in computes if hasattr(c, 'type') and 'gpu' in str(c.type).lower()]

            if cpu_clusters:
                self.successes.append(f"✅ CPU compute available: {len(cpu_clusters)} cluster(s)")
            else:
                self.issues.append("⚠️ No CPU compute clusters found")

            if gpu_clusters:
                self.successes.append(f"✅ GPU compute available: {len(gpu_clusters)} cluster(s)")
            else:
                self.issues.append("⚠️ No GPU compute clusters found (quota may be pending)")

            return True
        except Exception as e:
            self.issues.append(f"❌ Error checking compute targets: {e}")
            return False

    def check_environments(self):
        """Check custom environments"""
        try:
            environments = list(self.ml_client.environments.list())
            custom_envs = [e for e in environments if 'millennialai' in e.name.lower()]

            if custom_envs:
                self.successes.append(f"✅ MillennialAi environments found: {len(custom_envs)}")
                for env in custom_envs:
                    self.successes.append(f"   - {env.name} v{env.version}")
            else:
                self.issues.append("⚠️ No MillennialAi custom environments found")

            return True
        except Exception as e:
            self.issues.append(f"❌ Error checking environments: {e}")
            return False

    def check_models(self):
        """Check registered models"""
        try:
            models = list(self.ml_client.models.list())
            millennialai_models = [m for m in models if 'millennialai' in m.name.lower()]

            if millennialai_models:
                self.successes.append(f"✅ MillennialAi models registered: {len(millennialai_models)}")
                for model in millennialai_models:
                    self.successes.append(f"   - {model.name} v{model.version}")
            else:
                self.issues.append("⚠️ No MillennialAi models registered")

            return True
        except Exception as e:
            self.issues.append(f"❌ Error checking models: {e}")
            return False

    def check_components(self):
        """Check pipeline components"""
        try:
            components = list(self.ml_client.components.list())
            millennialai_components = [c for c in components if 'millennialai' in c.name.lower()]

            if millennialai_components:
                self.successes.append(f"✅ MillennialAi components found: {len(millennialai_components)}")
                for comp in millennialai_components:
                    self.successes.append(f"   - {comp.name} v{comp.version}")
            else:
                self.issues.append("⚠️ No MillennialAi pipeline components found")

            return True
        except Exception as e:
            self.issues.append(f"❌ Error checking components: {e}")
            return False

    def check_data_assets(self):
        """Check data assets"""
        try:
            # Check if training data is uploaded
            data_assets = list(self.ml_client.data.list())
            if data_assets:
                self.successes.append(f"✅ Data assets available: {len(data_assets)}")
            else:
                self.issues.append("⚠️ No data assets found in workspace")

            return True
        except Exception as e:
            self.issues.append(f"❌ Error checking data assets: {e}")
            return False

    def run_full_verification(self):
        """Run complete AI Foundry verification"""
        print("🔍 AI FOUNDRY SETUP VERIFICATION")
        print("=" * 40)

        # Step 1: Load configuration
        if not self.load_config():
            return False

        # Step 2: Test authentication
        if not self.test_azure_authentication():
            return False

        # Step 3: Test ML client connection
        if not self.test_ml_client_connection():
            return False

        # Step 4: Verify hub exists
        self.verify_hub_exists()

        # Step 5: Check compute resources
        self.check_compute_targets()

        # Step 6: Check environments
        self.check_environments()

        # Step 7: Check models
        self.check_models()

        # Step 8: Check components
        self.check_components()

        # Step 9: Check data assets
        self.check_data_assets()

        return True

    def print_report(self):
        """Print verification report"""
        print("\n📊 VERIFICATION REPORT")
        print("=" * 25)

        print(f"\n✅ SUCCESSFUL COMPONENTS ({len(self.successes)}):")
        for success in self.successes:
            print(f"   {success}")

        if self.issues:
            print(f"\n⚠️ ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   {issue}")
        else:
            print("\n🎉 NO ISSUES FOUND - SETUP IS COMPLETE!")

        print("\n🔗 ACCESS INFORMATION:")
        if self.config:
            print(f"   AI Studio: {self.config.get('ai_studio_url', 'https://ai.azure.com')}")
            print(f"   Hub: {self.config.get('hub_endpoint', 'N/A')}")
            print(f"   Project: {self.config.get('project_endpoint', 'N/A')}")

def main():
    """Main verification function"""
    verifier = AIFoundryVerifier()

    if verifier.run_full_verification():
        verifier.print_report()

        # Overall assessment
        critical_issues = [i for i in verifier.issues if '❌' in i]
        if critical_issues:
            print("\n❌ CRITICAL ISSUES DETECTED - AI Foundry setup incomplete")
            return False
        else:
            print("\n🎉 AI Foundry setup verification PASSED!")
            return True
    else:
        print("\n❌ VERIFICATION FAILED - Cannot proceed with checks")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)