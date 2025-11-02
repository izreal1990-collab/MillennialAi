#!/usr/bin/env python3
"""
Submit MillennialAi training job to Azure ML
"""

import os
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from pathlib import Path


def submit_training_job():
    """Submit training job to Azure ML compute cluster"""
    
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
        return
    
    # Connect to workspace
    print(f"🔗 Connecting to Azure ML workspace: {workspace_name}")
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    print("✅ Connected successfully")
    
    # Create or update environment
    print("\n📦 Creating training environment...")
    env_path = Path(__file__).parent / "azure_ml_env.yaml"
    
    if env_path.exists():
        env = Environment(
            name="millennialai-training-env",
            description="MillennialAi training environment",
            conda_file=str(env_path),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
        )
        env = ml_client.environments.create_or_update(env)
        print(f"✅ Environment created: {env.name}:{env.version}")
    
    # Define training job
    print("\n🚀 Submitting training job...")
    
    job = command(
        code=".",  # Upload entire project directory
        command="python azure_ml_training.py \
                --num-epochs 100 \
                --batch-size 4 \
                --learning-rate 0.001 \
                --use-mlflow \
                --use-azure-storage \
                --output-dir ./outputs",
        environment=f"{env.name}:{env.version}",
        compute="gpu-cluster",  # GPU compute cluster defined in Bicep
        display_name="MillennialAi Revolutionary Training",
        description="Training MillennialAi with revolutionary layer injection framework",
        experiment_name="millennialai-training",
        tags={
            "model": "MillennialAi",
            "framework": "pytorch",
            "task": "training"
        }
    )
    
    # Submit job
    returned_job = ml_client.jobs.create_or_update(job)
    
    print(f"\n✅ Training job submitted successfully!")
    print(f"   Job name: {returned_job.name}")
    print(f"   Status: {returned_job.status}")
    print(f"   Compute: {returned_job.compute}")
    print(f"\n🌐 View job in Azure ML Studio:")
    print(f"   {returned_job.studio_url}")
    print(f"\n📊 Monitor with:")
    print(f"   az ml job show --name {returned_job.name}")
    
    return returned_job


if __name__ == "__main__":
    print("🌟 MILLENNIALAI AZURE ML JOB SUBMISSION")
    print("=" * 70)
    
    job = submit_training_job()
    
    if job:
        print("\n🎉 Job submission complete!")
        print("   Training will begin when compute resources are allocated")
    else:
        print("\n❌ Job submission failed")
