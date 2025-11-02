#!/usr/bin/env python3
"""
Azure ML Environment Setup Helper

This script helps configure your Azure ML environment for MillennialAi testing.
Run this to interactively set up your Azure ML workspace connection.
"""

import os
import subprocess
import json
from pathlib import Path

# Constants for repeated strings
INVALID_CHOICE_MSG = "❌ Invalid choice. Please try again."


def run_command(cmd, capture_output=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def check_azure_cli():
    """Check if Azure CLI is installed"""
    success, _, _ = run_command("az --version")
    if success:
        print("✅ Azure CLI is installed")
        return True
    else:
        print("❌ Azure CLI not found. Please install it first:")
        print("   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash")
        return False


def check_ml_extension():
    """Check if Azure ML CLI extension is installed"""
    success, stdout, _ = run_command("az extension list --output json")
    if success:
        extensions = json.loads(stdout)
        ml_extensions = [ext for ext in extensions if ext.get('name') == 'ml']
        if ml_extensions:
            print("✅ Azure ML CLI extension is installed")
            return True

    print("❌ Azure ML CLI extension not found. Installing...")
    success, _, _ = run_command("az extension add -n ml")
    if success:
        print("✅ Azure ML CLI extension installed successfully")
        return True
    else:
        print("❌ Failed to install Azure ML CLI extension")
        return False


def azure_login():
    """Login to Azure"""
    print("\n🔐 Logging in to Azure...")
    print("Using device code flow for headless authentication.")
    print("A device code will be displayed below. Visit https://microsoft.com/devicelogin and enter the code.")

    success, _, stderr = run_command("az login --use-device-code")
    if success:
        print("✅ Azure login successful")
        return True
    else:
        print("❌ Azure login failed")
        print(f"Error: {stderr}")
        return False


def get_subscriptions():
    """Get available Azure subscriptions"""
    success, stdout, _ = run_command("az account list --output json")
    if success:
        try:
            subscriptions = json.loads(stdout)
            return [sub for sub in subscriptions if sub.get('isDefault', False) or sub.get('state') == 'Enabled']
        except json.JSONDecodeError:
            pass
    return []


def select_subscription():
    """Let user select a subscription"""
    subscriptions = get_subscriptions()
    if not subscriptions:
        print("❌ No active subscriptions found")
        return None

    print("\n📋 Available subscriptions:")
    for i, sub in enumerate(subscriptions):
        print(f"   {i+1}. {sub.get('name', 'Unknown')} ({sub['id']})")

    while True:
        try:
            choice = input(f"\nSelect subscription (1-{len(subscriptions)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(subscriptions):
                selected = subscriptions[idx]
                print(f"✅ Selected: {selected.get('name', 'Unknown')}")

                # Set the subscription
                run_command(f"az account set --subscription {selected['id']}")
                return selected
        except (ValueError, KeyboardInterrupt):
            pass
        print(INVALID_CHOICE_MSG)


def get_resource_groups():
    """Get available resource groups"""
    success, stdout, _ = run_command("az group list --output json")
    if success:
        try:
            groups = json.loads(stdout)
            return groups
        except json.JSONDecodeError:
            pass
    return []


def select_or_create_resource_group():
    """Let user select or create a resource group"""
    groups = get_resource_groups()

    print("\n📋 Available resource groups:")
    for i, group in enumerate(groups):
        print(f"   {i+1}. {group['name']} ({group['location']})")

    print(f"   {len(groups)+1}. Create new resource group")

    while True:
        try:
            choice = input(f"\nSelect resource group (1-{len(groups)+1}): ").strip()
            idx = int(choice) - 1

            if 0 <= idx < len(groups):
                selected = groups[idx]
                print(f"✅ Selected: {selected['name']}")
                return selected['name']
            elif idx == len(groups):
                # Create new resource group
                name = input("Enter new resource group name: ").strip()
                location = input("Enter location (e.g., eastus, westus2): ").strip() or "eastus"

                success, _, stderr = run_command(f"az group create --name {name} --location {location}")
                if success:
                    print(f"✅ Created resource group: {name}")
                    return name
                else:
                    print(f"❌ Failed to create resource group: {stderr}")

        except (ValueError, KeyboardInterrupt):
            pass
        print(INVALID_CHOICE_MSG)


def get_ml_workspaces(resource_group):
    """Get available ML workspaces in a resource group"""
    success, stdout, _ = run_command(f"az ml workspace list --resource-group {resource_group} --output json")
    if success:
        try:
            workspaces = json.loads(stdout)
            return workspaces
        except json.JSONDecodeError:
            pass
    return []


def select_or_create_workspace(resource_group):
    """Let user select or create an ML workspace"""
    workspaces = get_ml_workspaces(resource_group)

    print(f"\n📋 ML workspaces in '{resource_group}':")
    for i, ws in enumerate(workspaces):
        print(f"   {i+1}. {ws['name']} ({ws['location']})")

    print(f"   {len(workspaces)+1}. Create new ML workspace")

    while True:
        try:
            choice = input(f"\nSelect workspace (1-{len(workspaces)+1}): ").strip()
            idx = int(choice) - 1

            if 0 <= idx < len(workspaces):
                selected = workspaces[idx]
                print(f"✅ Selected: {selected['name']}")
                return selected['name']
            elif idx == len(workspaces):
                # Create new workspace
                name = input("Enter new workspace name: ").strip()
                location = input("Enter location (e.g., westus2): ").strip() or "westus2"

                success, _, stderr = run_command(
                    f"az ml workspace create --name {name} --resource-group {resource_group} --location {location}"
                )
                if success:
                    print(f"✅ Created ML workspace: {name}")
                    return name
                else:
                    print(f"❌ Failed to create workspace: {stderr}")

        except (ValueError, KeyboardInterrupt):
            pass
        print(INVALID_CHOICE_MSG)


def create_compute_cluster(resource_group, workspace):
    """Create a CPU compute cluster for testing"""
    print("\n🖥️ Setting up compute cluster...")

    # Check if cpu-cluster already exists
    success, _, _ = run_command(
        f"az ml compute show --name cpu-cluster --resource-group {resource_group} --workspace-name {workspace}"
    )

    if success:
        print("✅ CPU compute cluster already exists")
        return True

    # Create CPU compute cluster
    success, _, stderr = run_command(
        f"az ml compute create --name cpu-cluster --type AmlCompute "
        f"--size Standard_DS3_v2 --min-instances 0 --max-instances 4 "
        f"--resource-group {resource_group} --workspace-name {workspace}"
    )

    if success:
        print("✅ Created CPU compute cluster: cpu-cluster")
        return True
    else:
        print(f"❌ Failed to create compute cluster: {stderr}")
        return False


def save_environment_file(subscription_id, resource_group, workspace):
    """Save environment variables to a .env file"""
    env_content = f"""# Azure ML Environment Configuration
# Generated by setup_azure_ml.py

AZURE_SUBSCRIPTION_ID={subscription_id}
AZURE_RESOURCE_GROUP={resource_group}
AZURE_ML_WORKSPACE_NAME={workspace}
"""

    env_file = Path(".env")
    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"✅ Environment configuration saved to {env_file}")
    print("   Load these variables with: source .env")


def main():
    """Main setup function"""
    print("🚀 MILLENNIALAI AZURE ML ENVIRONMENT SETUP")
    print("=" * 50)

    # Check prerequisites
    if not check_azure_cli():
        return

    if not check_ml_extension():
        return

    # Login to Azure
    if not azure_login():
        return

    # Select subscription
    subscription = select_subscription()
    if not subscription:
        return

    # Select/create resource group
    resource_group = select_or_create_resource_group()
    if not resource_group:
        return

    # Select/create workspace
    workspace = select_or_create_workspace(resource_group)
    if not workspace:
        return

    # Create compute cluster
    if not create_compute_cluster(resource_group, workspace):
        print("⚠️ Compute cluster creation failed, but you can create it manually later")

    # Save configuration
    save_environment_file(subscription['id'], resource_group, workspace)

    print("\n🎉 Azure ML environment setup complete!")
    print("\n📋 Summary:")
    print(f"   Subscription: {subscription.get('name', 'Unknown')} ({subscription['id']})")
    print(f"   Resource Group: {resource_group}")
    print(f"   ML Workspace: {workspace}")
    print("   Compute: cpu-cluster (CPU compute for testing)")

    print("\n🧪 Ready to run tests:")
    print("   source .env")
    print("   python submit_azure_test.py")

    print("\n📖 For more information, see AZURE_ML_SETUP.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")