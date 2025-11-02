#!/usr/bin/env python3
"""
Quick CPU Test Job for MillennialAi

Test the CPU compute cluster with a simple MillennialAi validation job.
"""

import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def submit_cpu_test_job():
    """Submit a quick test job to the CPU compute cluster"""
    
    print("🚀 Submitting MillennialAi CPU Test Job")
    print("=====================================")
    
    # Azure configuration
    subscription_id = "639e13e9-b4be-4ba6-8e9e-f14db5b3a65c"
    resource_group = "rg-jblango-1749"
    workspace_name = "azml7c462q3plqeyo"
    
    try:
        # Get credentials and create ML client
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        print(f"✅ Connected to Azure ML workspace: {workspace_name}")
        
        # Create environment for the job
        environment = Environment(
            name="millennialai-cpu-env",
            description="MillennialAi CPU testing environment",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            conda_file="environment.yml"
        )
        
        # Create the test job
        test_job = command(
            name="millennialai-cpu-test",
            display_name="MillennialAi CPU Cluster Test",
            description="Quick validation test on CPU compute cluster",
            code=".",
            command="python azure_ml_test.py",
            environment=environment,
            compute="cpu-cluster",
            instance_count=1,
            tags={
                "project": "MillennialAi",
                "test_type": "cpu_validation",
                "framework": "pytorch"
            }
        )
        
        print("📦 Submitting job to cpu-cluster...")
        
        # Submit the job
        submitted_job = ml_client.jobs.create_or_update(test_job)
        
        print("✅ Job submitted successfully!")
        print(f"   Job Name: {submitted_job.name}")
        print(f"   Job ID: {submitted_job.id}")
        print(f"   Status: {submitted_job.status}")
        print("")
        print("🔗 Monitor job progress:")
        print(f"   Azure ML Studio: https://ml.azure.com/runs/{submitted_job.name}")
        print(f"   Resource Group: https://portal.azure.com/#@/resource/subscriptions/{subscription_id}/resourceGroups/{resource_group}/overview")
        
        return submitted_job
        
    except Exception as e:
        print(f"❌ Error submitting job: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure you're logged in: az login")
        print("   2. Check compute cluster status: az ml compute list")
        print("   3. Verify workspace access: az ml workspace show")
        return None


if __name__ == "__main__":
    job = submit_cpu_test_job()
    
    if job:
        print("\n🎯 Next Steps:")
        print("   1. Monitor job in Azure ML Studio")
        print("   2. Check logs for MillennialAi validation results")
        print("   3. If successful, proceed with larger training jobs")
        print("   4. Continue monitoring GPU quota for full deployment")