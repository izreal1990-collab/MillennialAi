#!/usr/bin/env python3
"""
Azure ML Training Script for MillennialAi
Optimized for distributed GPU training with checkpointing and monitoring
"""

import os
import argparse
import json
from pathlib import Path
import torch
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import mlflow
import mlflow.pytorch

# Import MillennialAi components
from revolutionary_training_system import RevolutionaryTrainer
from training_data_generator import generate_revolutionary_dataset
from real_brain import RealThinkingBrain


class AzureMLTrainingRunner:
    """Azure ML training orchestrator for MillennialAi"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._setup_device()
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Azure clients
        self.ml_client = None
        self.blob_client = None
        if args.use_azure_storage:
            self._setup_azure_clients()
        
        print("Azure ML Training Runner initialized")
        print(f"   Device: {self.device}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Azure Storage: {args.use_azure_storage}")
    
    def _setup_device(self):
        """Configure compute device with multi-GPU support"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print("GPU Configuration:")
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)")
            
            # Use first GPU by default (multi-GPU via DataParallel if needed)
            device = torch.device('cuda:0')
        else:
            print("⚠️ No GPU detected - using CPU (training will be slow)")
            device = torch.device('cpu')
        
        return device
    
    def _setup_azure_clients(self):
        """Initialize Azure ML and Storage clients"""
        try:
            # Azure ML client
            credential = DefaultAzureCredential()
            
            # Get workspace info from environment or args
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
                print(f"✅ Connected to Azure ML workspace: {workspace_name}")
            
            # Storage client for artifacts
            storage_account = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', self.args.storage_account)
            if storage_account:
                storage_url = f"https://{storage_account}.blob.core.windows.net"
                self.blob_client = BlobServiceClient(
                    account_url=storage_url,
                    credential=credential
                )
                print(f"✅ Connected to Azure Storage: {storage_account}")
        
        except Exception as e:
            print(f"⚠️ Azure client setup failed: {e}")
            print("   Continuing with local storage only")
    
    def prepare_training_data(self):
        """Generate or load training dataset"""
        print("\n📊 PREPARING TRAINING DATA")
        
        dataset_path = self.output_dir / "training_dataset.json"
        
        if dataset_path.exists() and not self.args.regenerate_data:
            print(f"   Loading existing dataset from {dataset_path}")
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            conversations = data['conversations']
        else:
            print("   Generating new revolutionary training dataset...")
            _, conversations = generate_revolutionary_dataset()
            
            # Save dataset
            with open(dataset_path, 'w') as f:
                json.dump({'conversations': conversations}, f, indent=2)
            print(f"   ✅ Saved dataset to {dataset_path}")
        
        print(f"   ✅ Dataset ready: {len(conversations)} conversations")
        return conversations
    
    def train(self):
        """Execute training with Azure ML integration"""
        print("\n🚀 STARTING AZURE ML TRAINING")
        print("=" * 70)
        
        # MLflow experiment tracking
        if self.args.use_mlflow:
            mlflow.set_experiment(self.args.experiment_name)
            mlflow.start_run(run_name=self.args.run_name)
            
            # Log parameters
            mlflow.log_params({
                'num_epochs': self.args.num_epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'device': str(self.device),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            })
        
        # Prepare data
        conversations = self.prepare_training_data()
        
        # Initialize model
        print("\n🧠 INITIALIZING REVOLUTIONARY BRAIN")
        brain = RealThinkingBrain(device=str(self.device))
        
        # Initialize trainer
        trainer = RevolutionaryTrainer(brain, device=str(self.device))
        
        # Training loop
        print(f"\n🏋️ TRAINING FOR {self.args.num_epochs} EPOCHS")
        best_score = 0.0
        
        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*70}")
            
            # Training step
            epoch_metrics = trainer.train_epoch(
                conversations=conversations,
                batch_size=self.args.batch_size,
                learning_rate=self.args.learning_rate
            )
            
            # Log metrics
            print(f"\n📊 Epoch {epoch + 1} Metrics:")
            for metric_name, metric_value in epoch_metrics.items():
                print(f"   {metric_name}: {metric_value:.4f}")
                
                if self.args.use_mlflow:
                    mlflow.log_metric(metric_name, metric_value, step=epoch)
            
            # Calculate revolutionary score
            revolutionary_score = epoch_metrics.get('revolutionary_score', 0.0)
            
            # Save checkpoint if improved
            if revolutionary_score > best_score:
                best_score = revolutionary_score
                checkpoint_path = self.output_dir / f"best_model_epoch_{epoch+1}.pt"
                self.save_checkpoint(brain, trainer, epoch, checkpoint_path)
                print(f"   ✅ New best model saved! Score: {best_score:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.args.checkpoint_frequency == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(brain, trainer, epoch, checkpoint_path)
                print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pt"
        self.save_checkpoint(brain, trainer, self.args.num_epochs, final_model_path)
        print(f"\n✅ Training complete! Final model: {final_model_path}")
        
        # Log model to MLflow
        if self.args.use_mlflow:
            mlflow.pytorch.log_model(brain, "model")
            mlflow.log_artifact(str(final_model_path))
            mlflow.end_run()
        
        # Upload to Azure Storage
        if self.args.use_azure_storage and self.blob_client:
            self.upload_to_azure(final_model_path)
        
        return brain, trainer
    
    def save_checkpoint(self, brain, trainer, epoch, path):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'brain_state_dict': brain.state_dict(),
            'complexity_optimizer': trainer.complexity_optimizer.state_dict(),
            'thinking_optimizer': trainer.thinking_optimizer.state_dict(),
            'convergence_optimizer': trainer.convergence_optimizer.state_dict(),
            'training_history': trainer.training_history,
        }
        torch.save(checkpoint, path)
    
    def upload_to_azure(self, model_path):
        """Upload trained model to Azure Blob Storage"""
        try:
            container_client = self.blob_client.get_container_client("models")
            
            # Create container if it doesn't exist
            try:
                container_client.create_container()
            except Exception:
                pass  # Container already exists
            
            blob_name = f"millennialai/{self.args.run_name}/{model_path.name}"
            
            with open(model_path, "rb") as data:
                container_client.upload_blob(
                    name=blob_name,
                    data=data,
                    overwrite=True
                )
            
            print(f"☁️ Model uploaded to Azure Storage: {blob_name}")
        
        except Exception as e:
            print(f"⚠️ Failed to upload to Azure Storage: {e}")
    
    def test_model(self, brain):
        """Test the trained model"""
        print("\n🧪 TESTING TRAINED MODEL")
        print("=" * 70)
        
        test_inputs = [
            "What makes you revolutionary?",
            "How do you think differently from other AI?",
            "Explain consciousness to me",
            "What is the future of artificial intelligence?",
            "Solve this problem: 2x + 5 = 15"
        ]
        
        brain.eval()
        results = []
        
        with torch.no_grad():
            for test_input in test_inputs:
                print(f"\n💬 Input: {test_input}")
                result = brain.think(test_input)
                
                print(f"   🧠 Complexity: {result['complexity']:.2f}")
                print(f"   ⚡ Steps: {result['steps']}")
                print(f"   🌟 Response: {result['response'][:150]}...")
                
                results.append({
                    'input': test_input,
                    'complexity': result['complexity'],
                    'steps': result['steps'],
                    'response': result['response']
                })
        
        # Save test results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Test results saved to {results_path}")
        
        if self.args.use_mlflow:
            mlflow.log_artifact(str(results_path))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Azure ML Training for MillennialAi")
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--checkpoint-frequency', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Data parameters
    parser.add_argument('--regenerate-data', action='store_true',
                       help='Regenerate training dataset')
    
    # Azure configuration
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
                       help='Enable MLflow experiment tracking')
    parser.add_argument('--experiment-name', type=str, default='millennialai-training',
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    
    # Storage parameters
    parser.add_argument('--use-azure-storage', action='store_true', default=True,
                       help='Upload models to Azure Storage')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Local output directory')
    
    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()
    
    # Set default run name if not provided
    if args.run_name is None:
        import datetime
        args.run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("MILLENNIALAI AZURE ML TRAINING")
    print("=" * 70)
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Run: {args.run_name}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print("=" * 70)
    
    # Initialize runner
    runner = AzureMLTrainingRunner(args)
    
    # Train model
    brain, _ = runner.train()
    
    # Test model
    runner.test_model(brain)
    
    print("\nAZURE ML TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
