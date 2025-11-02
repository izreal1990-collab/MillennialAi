#!/usr/bin/env python3
"""
Submit MillennialAi Test Job to Azure ML

This script submits a test job to Azure ML to validate the MillennialAi
framework with actual PyTorch models in the cloud environment.
"""

import os
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from pathlib import Path


def submit_test_job():
    """Submit test job to Azure ML compute cluster"""

    # Get Azure credentials
    credential = DefaultAzureCredential()

    # Get workspace configuration from environment
    subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = os.getenv('AZURE_ML_WORKSPACE_NAME')

    if not all([subscription_id, resource_group, workspace_name]):
        print("❌ Missing Azure configuration. Required environment variables:")
        print("   - AZURE_SUBSCRIPTION_ID")
        print("   - AZURE_RESOURCE_GROUP")
        print("   - AZURE_ML_WORKSPACE_NAME")
        print("\n💡 Set these in your environment or create a .env file")
        return None

    # Connect to workspace
    print(f"🔗 Connecting to Azure ML workspace: {workspace_name}")
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    print("✅ Connected successfully")

    # Create or get environment
    env_name = "millennialai-test-env"
    env = None
    try:
        env = ml_client.environments.get(env_name, version="1")
        print(f"✅ Using existing test environment: {env.name}:{env.version}")
    except Exception:
        print("📦 Creating new test environment...")
        env = Environment(
            name=env_name,
            description="MillennialAI Test Environment with minimal dependencies",
            conda_file="./conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
        )
        env = ml_client.environments.create_or_update(env)
        print(f"✅ Test environment created: {env.name}:{env.version}")

    if env is None:
        raise RuntimeError("Failed to create or get environment")

    # Define test job
    print("\n🧪 Submitting test job...")

    job = command(
        code=".",  # Upload entire project directory
        command="python azure_ml_test.py",
        environment=f"{env.name}:{env.version}",
        compute="cpu-cluster",  # Use CPU for testing
        display_name="MillennialAi Azure ML Test",
        description="Test MillennialAi framework with PyTorch models in Azure ML",
        experiment_name="millennialai-testing",
        tags={
            "model": "MillennialAi",
            "framework": "pytorch",
            "task": "testing",
            "type": "validation"
        }
    )

    # Submit job
    returned_job = ml_client.jobs.create_or_update(job)

    print("\n✅ Test job submitted successfully!")
    print(f"   Job name: {returned_job.name}")
    print(f"   Status: {returned_job.status}")
    print(f"   Compute: {returned_job.compute}")
    print("\n🌐 View job in Azure ML Studio:")
    print(f"   {returned_job.studio_url}")
    print("\n📊 Monitor with:")
    print(f"   az ml job show --name {returned_job.name}")
    print("\n📋 Check logs with:")
    print(f"   az ml job stream --name {returned_job.name}")

    return returned_job


def check_azure_environment():
    """Check if Azure environment is properly configured"""
    print("🔍 Checking Azure ML environment configuration...")

    required_vars = [
        'AZURE_SUBSCRIPTION_ID',
        'AZURE_RESOURCE_GROUP',
        'AZURE_ML_WORKSPACE_NAME'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Set these variables in your environment or create a .env file")
        print("   Example:")
        print("   export AZURE_SUBSCRIPTION_ID='your-subscription-id'")
        print("   export AZURE_RESOURCE_GROUP='your-resource-group'")
        print("   export AZURE_ML_WORKSPACE_NAME='your-workspace-name'")
        return False

    print("✅ Azure environment configuration looks good")
    return True


if __name__ == "__main__":
    print("🧪 MILLENNIALAI AZURE ML TEST JOB SUBMISSION")
    print("=" * 60)

    # Check environment first
    if not check_azure_environment():
        print("\n❌ Cannot proceed without proper Azure configuration")
        exit(1)

    # Submit test job
    job = submit_test_job()

    if job:
        print("\n🎉 Test job submission complete!")
        print("   The test will validate MillennialAi with actual PyTorch models")
        print("   Check the Azure ML Studio link above to monitor progress")
    else:
        print("\n❌ Test job submission failed")
        exit(1)