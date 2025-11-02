#!/usr/bin/env python3
"""
MillennialAi Continuous Learning System
Integrates live chat data collection with Azure ML retraining
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Azure ML imports
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Job, PipelineJob
from azure.ai.ml.constants import AssetTypes

logger = logging.getLogger(__name__)

class ContinuousLearningManager:
    """Manages continuous learning pipeline for MillennialAi"""

    def __init__(self, learning_data_path: str = "learning_data"):
        self.learning_data_path = Path(learning_data_path)
        self.learning_data_path.mkdir(exist_ok=True)

        # Azure ML configuration
        self.subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
        self.resource_group = os.getenv('AZURE_RESOURCE_GROUP', 'millennialai-rg')
        self.workspace_name = os.getenv('AZURE_ML_WORKSPACE_NAME', 'millennialai-ws')

        # Continuous learning settings - OPTIMIZED FOR 85B PARAMETER MODEL
        self.min_samples_for_retraining = 10000  # Large models need massive data
        self.max_samples_per_batch = 50000  # Much larger batches for stability
        self.retraining_interval_hours = 168  # Weekly retraining (7 days) for 85B models
        self.last_retraining_time = None

        # Initialize Azure ML client
        self.ml_client = None
        self._setup_azure_ml()

        # Learning statistics
        self.stats = {
            'total_samples_collected': 0,
            'retraining_jobs_submitted': 0,
            'last_retraining_samples': 0,
            'avg_sample_quality': 0.0
        }

        logger.info("Continuous Learning Manager initialized")

    def _setup_azure_ml(self):
        """Setup Azure ML client"""
        try:
            if all([self.subscription_id, self.resource_group, self.workspace_name]):
                credential = DefaultAzureCredential()
                self.ml_client = MLClient(
                    credential=credential,
                    subscription_id=self.subscription_id,
                    resource_group_name=self.resource_group,
                    workspace_name=self.workspace_name
                )
                logger.info(f"✅ Connected to Azure ML workspace: {self.workspace_name}")
            else:
                logger.warning("⚠️ Azure ML credentials not configured - running in local mode only")
        except Exception as e:
            logger.error(f"❌ Azure ML setup failed: {e}")

    def collect_learning_sample(self, sample: Dict[str, Any]):
        """Collect a learning sample from live chat"""
        try:
            # Add metadata
            sample['collection_timestamp'] = datetime.now().isoformat()
            sample['sample_id'] = f"sample_{int(time.time() * 1000)}"

            # Calculate sample quality score
            quality_score = self._calculate_sample_quality(sample)
            sample['quality_score'] = quality_score

            # Save sample
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_sample_{timestamp}_{sample['sample_id']}.json"
            filepath = self.learning_data_path / filename

            with open(filepath, 'w') as f:
                json.dump(sample, f, indent=2)

            self.stats['total_samples_collected'] += 1
            self._update_avg_quality(quality_score)

            logger.info(f"💾 Learning sample collected: {filename} (quality: {quality_score:.2f})")

            # Check if retraining should be triggered
            self._check_retraining_trigger()

        except Exception as e:
            logger.error(f"Error collecting learning sample: {e}")

    def _calculate_sample_quality(self, sample: Dict[str, Any]) -> float:
        """Calculate quality score for learning sample"""
        quality = 0.5  # Base quality

        # Input length quality
        input_len = len(sample.get('input', ''))
        if 10 <= input_len <= 500:
            quality += 0.2
        elif input_len > 500:
            quality += 0.1

        # Response complexity
        brain_metrics = sample.get('brain_metrics', {})
        complexity = brain_metrics.get('complexity', 0.0)
        if complexity > 1.0:
            quality += 0.2
        elif complexity > 0.5:
            quality += 0.1

        # Reasoning steps
        steps = brain_metrics.get('steps', 0)
        if steps > 5:
            quality += 0.1

        # Hybrid enhancement
        if sample.get('hybrid_metrics') and sample['hybrid_metrics'].get('source') != 'error':
            quality += 0.1

        return min(1.0, quality)

    def _update_avg_quality(self, new_quality: float):
        """Update rolling average quality score"""
        total = self.stats['total_samples_collected']
        current_avg = self.stats['avg_sample_quality']
        self.stats['avg_sample_quality'] = (current_avg * (total - 1) + new_quality) / total

    def _check_retraining_trigger(self):
        """Check if retraining should be triggered"""
        current_samples = self.stats['total_samples_collected']

        # Check minimum samples
        if current_samples < self.min_samples_for_retraining:
            return

        # Check time interval
        if self.last_retraining_time:
            time_since_last = datetime.now() - self.last_retraining_time
            if time_since_last < timedelta(hours=self.retraining_interval_hours):
                return

        # Check sample quality
        if self.stats['avg_sample_quality'] < 0.6:
            logger.info("Sample quality too low for retraining, continuing data collection")
            return

        # Trigger retraining
        logger.info(f"🎯 Triggering retraining: {current_samples} samples collected")
        self.trigger_retraining()

    def trigger_retraining(self):
        """Trigger Azure ML retraining job"""
        try:
            if not self.ml_client:
                logger.warning("Azure ML not configured - skipping retraining")
                return

            # Prepare training data
            training_samples = self._prepare_training_batch()

            if len(training_samples) < self.min_samples_for_retraining:
                logger.info("Insufficient high-quality samples for retraining")
                return

            # Create training job
            job = self._create_training_job(training_samples)

            # Submit job
            submitted_job = self.ml_client.jobs.create_or_update(job)
            logger.info(f"🚀 Retraining job submitted: {submitted_job.name}")

            # Update stats
            self.stats['retraining_jobs_submitted'] += 1
            self.stats['last_retraining_samples'] = len(training_samples)
            self.last_retraining_time = datetime.now()

            # Archive used samples
            self._archive_used_samples(training_samples)

        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")

    def _prepare_training_batch(self) -> List[Dict[str, Any]]:
        """Prepare batch of high-quality samples for training"""
        # Get all sample files
        sample_files = list(self.learning_data_path.glob("learning_sample_*.json"))

        # Load and filter samples
        samples = []
        for filepath in sample_files:
            try:
                with open(filepath, 'r') as f:
                    sample = json.load(f)

                # Only include high-quality samples
                if sample.get('quality_score', 0) >= 0.7:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"Error loading sample {filepath}: {e}")

        # Sort by quality and recency
        samples.sort(key=lambda x: (x.get('quality_score', 0), x.get('collection_timestamp', '')), reverse=True)

        # Limit batch size
        return samples[:self.max_samples_per_batch]

    def _create_training_job(self, training_samples: List[Dict[str, Any]]) -> Job:
        """Create Azure ML training job"""
        from azure.ai.ml import command

        # Create training data file
        training_data_path = self.learning_data_path / "training_batch.json"
        with open(training_data_path, 'w') as f:
            json.dump(training_samples, f, indent=2)

        # Upload to Azure ML datastore (simplified - in practice use proper data asset)
        # For now, assume data is accessible via mounted storage

        # Create command job
        job = command(
            code="./millennial_ai/",
            command=f"python azure_ml_training.py --use-mlflow --experiment-name millennialai-continuous-learning --run-name continuous_{int(time.time())} --num-epochs 10 --batch-size 2048",
            environment="millennialai-dev-env:1.0.0",
            compute="gpu-cluster",  # Assumes GPU cluster exists
            display_name=f"MillennialAi Continuous Learning {datetime.now().strftime('%Y%m%d_%H%M')}",
            description=f"Continuous learning with {len(training_samples)} new samples",
            tags={
                "type": "continuous-learning",
                "samples": str(len(training_samples)),
                "avg_quality": f"{self.stats['avg_sample_quality']:.2f}"
            }
        )

        return job

    def _archive_used_samples(self, used_samples: List[Dict[str, Any]]):
        """Archive samples that were used for training"""
        archive_path = self.learning_data_path / "archive"
        archive_path.mkdir(exist_ok=True)

        for sample in used_samples:
            sample_id = sample.get('sample_id', 'unknown')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Move to archive
            for filepath in self.learning_data_path.glob(f"learning_sample_*_{sample_id}.json"):
                archive_file = archive_path / f"archived_{timestamp}_{filepath.name}"
                filepath.rename(archive_file)
                break

        logger.info(f"📦 Archived {len(used_samples)} samples for training")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        # Count current samples
        current_samples = list(self.learning_data_path.glob("learning_sample_*.json"))
        archived_samples = list((self.learning_data_path / "archive").glob("*.json"))

        return {
            **self.stats,
            'current_samples_count': len(current_samples),
            'archived_samples_count': len(archived_samples),
            'last_retraining_time': self.last_retraining_time.isoformat() if self.last_retraining_time else None,
            'next_retraining_eligible': self._check_next_retraining_eligibility(),
            'azure_ml_connected': self.ml_client is not None
        }

    def _check_next_retraining_eligibility(self) -> str:
        """Check when next retraining will be eligible"""
        current_samples = self.stats['total_samples_collected']

        if current_samples < self.min_samples_for_retraining:
            needed = self.min_samples_for_retraining - current_samples
            return f"Need {needed} more samples"

        if self.last_retraining_time:
            time_since_last = datetime.now() - self.last_retraining_time
            remaining_hours = self.retraining_interval_hours - (time_since_last.total_seconds() / 3600)
            if remaining_hours > 0:
                return f"Time-based: {remaining_hours:.1f} hours remaining"

        if self.stats['avg_sample_quality'] < 0.85:  # Higher quality threshold for 85B models
            return f"Quality threshold: {self.stats['avg_sample_quality']:.2f} < 0.85"

        return "Eligible now"

# Global continuous learning manager
continuous_learning = ContinuousLearningManager()

def process_continuous_learning():
    """Continuous process for automated machine learning - OPTIMIZED FOR 85B MODELS"""
    print("🚀 Starting Automated ML for 85B Parameter Model - Maximum Efficiency Mode")
    print("📊 Optimized Settings for Large-Scale AI:")
    print("   • Minimum samples: 10,000 (massive data requirements)")
    print("   • Batch size: 50,000 (stable training)")
    print("   • Retraining interval: 7 days (resource intensive)")
    print("   • Check frequency: Every 5 minutes (conservative monitoring)")
    print("🎯 Quality threshold: 0.85+ (high quality required for 85B models)")

    while True:
        try:
            # Check for retraining
            continuous_learning._check_retraining_trigger()

            # Get current stats
            stats = continuous_learning.get_learning_stats()
            logger.info(f"85B Model Status: {stats['total_samples_collected']} samples, Quality: {stats['avg_sample_quality']:.3f}, Next: {stats['next_retraining_eligible']}")

            # Sleep for 5 minutes (conservative checking for large models)
            time.sleep(300)

        except KeyboardInterrupt:
            print("\n🛑 85B Automated ML stopped by user")
            break
        except Exception as e:
            logger.error(f"85B Automated ML error: {e}")
            time.sleep(60)  # Longer wait on error for stability

# Start automated ML process
if __name__ == "__main__":
    process_continuous_learning()