#!/usr/bin/env python3
"""
Simple CPU Test Job for MillennialAi - Fixed Version

Test the CPU compute cluster with a corrected MillennialAi validation job.
"""

import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def submit_fixed_cpu_test():
    """Submit a fixed test job to the CPU compute cluster"""
    
    print("🚀 Submitting Fixed MillennialAi CPU Test Job")
    print("===========================================")
    
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
            name="millennialai-fixed-env",
            description="MillennialAi fixed testing environment",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            conda_file="environment.yml"
        )
        
        # Create the fixed test job
        test_job = command(
            name="millennialai-cpu-test-fixed-v2",
            display_name="MillennialAi CPU Test (Fixed v2)",
            description="Fixed validation test with proper imports and ACR auth",
            code=".",
            command="python azure_ml_test_fixed.py",
            environment=environment,
            compute="cpu-cluster",
            instance_count=1,
            tags={
                "project": "MillennialAi",
                "test_type": "cpu_validation_fixed",
                "framework": "pytorch",
                "version": "fixed"
            }
        )
        
        print("📦 Submitting fixed job to cpu-cluster...")
        
        # Submit the job
        submitted_job = ml_client.jobs.create_or_update(test_job)
        
        print("✅ Fixed job submitted successfully!")
        print(f"   Job Name: {submitted_job.name}")
        print(f"   Job ID: {submitted_job.id}")
        print(f"   Status: {submitted_job.status}")
        print("")
        print("🔗 Monitor job progress:")
        print(f"   Azure ML Studio: https://ml.azure.com/runs/{submitted_job.name}")
        print(f"   Resource Group: https://portal.azure.com/#@/resource/subscriptions/{subscription_id}/resourceGroups/{resource_group}/overview")
        
        return submitted_job
        
    except Exception as e:
        print(f"❌ Error submitting fixed job: {str(e)}")
        return None


if __name__ == "__main__":
    job = submit_fixed_cpu_test()
    
    if job:
        print("\n🎯 Next Steps:")
        print("   1. Monitor job in Azure ML Studio")
        print("   2. Check logs for improved MillennialAi validation")
        print("   3. Fixed import issues should resolve the failure")
        print("   4. Continue with GPU quota monitoring")