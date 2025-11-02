#!/usr/bin/env python3
"""
Azure ML Model Testing and Validation Script
Tests trained MillennialAi models for quality and performance
"""

import os
import json
import argparse
from pathlib import Path
import torch
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import mlflow

# Import MillennialAi components
from real_brain import RealThinkingBrain


class AzureMLModelTester:
    """Test and validate MillennialAi models on Azure"""
    
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Azure clients if needed
        self.ml_client = None
        self.blob_client = None
        if args.use_azure:
            self._setup_azure_clients()
    
    def _setup_azure_clients(self):
        """Initialize Azure ML and Storage clients"""
        try:
            credential = DefaultAzureCredential()
            
            # Azure ML client
            subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID', self.args.subscription_id)
            resource_group = os.getenv('AZURE_RESOURCE_GROUP', self.args.resource_group)
            workspace_name = os.getenv('AZURE_ML_WORKSPACE_NAME', self.args.workspace_name)
            
            if all([subscription_id, resource_group, workspace_name]):
                self.ml_client = MLClient(
                    credential=credential,
                    subscription_id=subscription_id,
                    resource_group_name=resource_group,
                    workspace_name=workspace_name
                )
                print(f"✅ Connected to Azure ML: {workspace_name}")
            
            # Storage client
            storage_account = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', self.args.storage_account)
            if storage_account:
                storage_url = f"https://{storage_account}.blob.core.windows.net"
                self.blob_client = BlobServiceClient(
                    account_url=storage_url,
                    credential=credential
                )
                print(f"✅ Connected to Azure Storage: {storage_account}")
        
        except Exception as e:
            print(f"⚠️ Azure setup failed: {e}")
    
    def load_model(self):
        """Load trained model from local or Azure storage"""
        print("\n📦 LOADING MODEL")
        
        if self.args.model_path:
            # Load from local path
            model_path = Path(self.args.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
        
        elif self.args.azure_model_uri and self.blob_client:
            # Download from Azure Storage
            print(f"   Downloading from Azure: {self.args.azure_model_uri}")
            model_path = self.output_dir / "downloaded_model.pt"
            
            # Parse blob URI
            parts = self.args.azure_model_uri.replace("https://", "").split("/")
            container_name = parts[1]
            blob_name = "/".join(parts[2:])
            
            blob_client = self.blob_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(model_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            print(f"   ✅ Downloaded to {model_path}")
        
        else:
            raise ValueError("Must provide either --model-path or --azure-model-uri")
        
        # Load checkpoint
        print(f"   Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize brain
        brain = RealThinkingBrain(device=self.device)
        brain.load_state_dict(checkpoint['brain_state_dict'])
        brain.eval()
        
        print(f"   ✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        return brain
    
    def run_comprehensive_tests(self, brain):
        """Run comprehensive test suite"""
        print("\n🧪 RUNNING COMPREHENSIVE TESTS")
        print("=" * 70)
        
        test_suites = {
            'reasoning': [
                "What is 2 + 2?",
                "If all A are B, and all B are C, are all A also C?",
                "Solve for x: 3x + 7 = 22",
                "What comes next: 2, 4, 8, 16, ?"
            ],
            'knowledge': [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What is photosynthesis?",
                "Explain quantum mechanics in simple terms"
            ],
            'creativity': [
                "Write a haiku about artificial intelligence",
                "Create a story about a robot learning to paint",
                "Design a new type of transportation",
                "Invent a solution to climate change"
            ],
            'conversation': [
                "How are you today?",
                "What makes you different from other AI?",
                "Tell me about yourself",
                "What do you think about the future?"
            ]
        }
        
        all_results = {}
        
        with torch.no_grad():
            for suite_name, test_cases in test_suites.items():
                print(f"\n{'='*70}")
                print(f"TEST SUITE: {suite_name.upper()}")
                print(f"{'='*70}")
                
                suite_results = []
                
                for i, test_input in enumerate(test_cases, 1):
                    print(f"\n[{i}/{len(test_cases)}] {test_input}")
                    
                    result = brain.think(test_input)
                    
                    complexity = result.get('complexity', 0.0)
                    steps = result.get('steps', 0)
                    response = result.get('response', '')
                    
                    print(f"   🧠 Complexity: {complexity:.2f}")
                    print(f"   ⚡ Thinking Steps: {steps}")
                    print(f"   💬 Response: {response[:200]}...")
                    
                    suite_results.append({
                        'input': test_input,
                        'complexity': complexity,
                        'steps': steps,
                        'response': response
                    })
                
                all_results[suite_name] = suite_results
        
        return all_results
    
    def analyze_performance(self, results):
        """Analyze model performance metrics"""
        print("\n📊 PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        metrics = {}
        
        for suite_name, suite_results in results.items():
            avg_complexity = sum(r['complexity'] for r in suite_results) / len(suite_results)
            avg_steps = sum(r['steps'] for r in suite_results) / len(suite_results)
            avg_response_length = sum(len(r['response']) for r in suite_results) / len(suite_results)
            
            metrics[suite_name] = {
                'avg_complexity': avg_complexity,
                'avg_steps': avg_steps,
                'avg_response_length': avg_response_length,
                'num_tests': len(suite_results)
            }
            
            print(f"\n{suite_name.upper()}:")
            print(f"   Avg Complexity: {avg_complexity:.2f}")
            print(f"   Avg Steps: {avg_steps:.1f}")
            print(f"   Avg Response Length: {avg_response_length:.0f} chars")
        
        return metrics
    
    def save_results(self, results, metrics):
        """Save test results and metrics"""
        print("\n💾 SAVING RESULTS")
        
        # Save detailed results
        results_path = self.output_dir / "test_results_detailed.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✅ Detailed results: {results_path}")
        
        # Save metrics summary
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   ✅ Metrics summary: {metrics_path}")
        
        # Log to MLflow if enabled
        if self.args.use_mlflow:
            mlflow.set_experiment("millennialai-testing")
            with mlflow.start_run():
                # Log metrics
                for suite_name, suite_metrics in metrics.items():
                    for metric_name, metric_value in suite_metrics.items():
                        mlflow.log_metric(f"{suite_name}_{metric_name}", metric_value)
                
                # Log artifacts
                mlflow.log_artifact(str(results_path))
                mlflow.log_artifact(str(metrics_path))
            
            print("   ✅ Logged to MLflow")
    
    def run(self):
        """Execute complete testing pipeline"""
        # Load model
        brain = self.load_model()
        
        # Run tests
        results = self.run_comprehensive_tests(brain)
        
        # Analyze performance
        metrics = self.analyze_performance(results)
        
        # Save results
        self.save_results(results, metrics)
        
        print("\n🎉 TESTING COMPLETE!")
        return results, metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Azure ML Model Testing")
    
    # Model parameters
    parser.add_argument('--model-path', type=str, default=None,
                       help='Local path to model checkpoint')
    parser.add_argument('--azure-model-uri', type=str, default=None,
                       help='Azure blob URI for model')
    
    # Azure configuration
    parser.add_argument('--use-azure', action='store_true',
                       help='Use Azure ML and Storage')
    parser.add_argument('--subscription-id', type=str, default=None,
                       help='Azure subscription ID')
    parser.add_argument('--resource-group', type=str, default=None,
                       help='Azure resource group')
    parser.add_argument('--workspace-name', type=str, default=None,
                       help='Azure ML workspace name')
    parser.add_argument('--storage-account', type=str, default=None,
                       help='Azure Storage account name')
    
    # MLflow parameters
    parser.add_argument('--use-mlflow', action='store_true', default=True,
                       help='Log results to MLflow')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./test_outputs',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main testing entry point"""
    args = parse_args()
    
    print("🌟 MILLENNIALAI MODEL TESTING")
    print("=" * 70)
    
    tester = AzureMLModelTester(args)
    tester.run()


if __name__ == "__main__":
    main()
