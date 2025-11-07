"""
MillennialAI Enterprise Training System
llama3:8b + TRM Layer Injection Architecture

SINGLE MODEL SYSTEM:
- Base: llama3:8b via Ollama
- Enhancement: TRM layer injection for temporal reasoning
- Training: Fine-tune TRM layers on workspace knowledge
- Output: Complete MillennialAI system ready for deployment

Author: MillennialAI Team
Date: November 6, 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import subprocess
import time
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine project root intelligently
def find_project_root() -> Path:
    """Find MillennialAi workspace directory"""
    # Try current directory first
    if (Path.cwd() / "millennial_ai").exists():
        return Path.cwd()
    
    # Try common locations
    possible_paths = [
        Path(r"C:\Users\jblan\workspace\MillennialAi"),
        Path.home() / "workspace" / "MillennialAi",
        Path.home() / "MillennialAi",
        Path("/workspace/MillennialAi"),  # Linux/Docker
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "millennial_ai").exists():
            return path
    
    raise FileNotFoundError(
        "Could not find MillennialAi workspace. Please run from workspace directory or check paths."
    )

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
logger.info(f"📁 Project root: {PROJECT_ROOT}")

# Import MillennialAI components
try:
    from millennial_ai.core.hybrid_model import CombinedTRMLLM
    from millennial_ai.config.config import HybridConfig, PresetConfigs
    from millennial_ai.models.hybrid_trm import HybridTRMBlock
    logger.info("✅ MillennialAI core modules loaded")
    CORE_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import MillennialAI core: {e}")
    CORE_AVAILABLE = False

# Import brain components
try:
    from real_brain import RealThinkingBrain
    from hybrid_brain import HybridRevolutionaryBrain, OllamaIntegration
    logger.info("✅ Brain components loaded")
    BRAIN_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import brain components: {e}")
    BRAIN_AVAILABLE = False

# FAISS for vector storage
try:
    import faiss
    logger.info("✅ FAISS available for vector storage")
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  FAISS not available. Install: pip install faiss-cpu")
    FAISS_AVAILABLE = False


class EnterpriseDataset(Dataset):
    """
    Enterprise-grade dataset loader
    Loads real documentation from workspace, no hardcoded samples
    """
    
    def __init__(
        self, 
        tokenizer,
        data_dir: Path,
        max_length: int = 512,
        min_doc_length: int = 100
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_doc_length = min_doc_length
        
        # Load all markdown documentation
        self.documents = self._load_documents(data_dir)
        logger.info(f"📚 Loaded {len(self.documents)} documents from {data_dir}")
        
        # Tokenize and chunk
        self.samples = self._create_samples()
        logger.info(f"📊 Created {len(self.samples)} training samples")
    
    def _load_documents(self, data_dir: Path) -> List[str]:
        """Load all markdown and text files"""
        documents = []
        
        # File patterns to load
        patterns = ['*.md', '*.txt', '*.py']  # Include Python files for code understanding
        
        for pattern in patterns:
            for file_path in data_dir.rglob(pattern):
                # Skip certain directories
                if any(skip in str(file_path) for skip in [
                    '.git', 'node_modules', '__pycache__', 
                    '.gradle', 'sonar-scanner', 'millennial_api_env'
                ]):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        # Filter out very short or binary-like content
                        if len(content) > self.min_doc_length:
                            documents.append(content)
                            logger.debug(f"   Loaded: {file_path.name} ({len(content)} chars)")
                
                except Exception as e:
                    logger.warning(f"   Failed to load {file_path}: {e}")
        
        if not documents:
            raise ValueError(f"No documents found in {data_dir}")
        
        return documents
    
    def _create_samples(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize documents and create training samples"""
        samples = []
        
        for doc in self.documents:
            # Tokenize with sliding window
            tokens = self.tokenizer.encode(doc, add_special_tokens=True)
            
            # Create overlapping chunks
            stride = self.max_length // 2  # 50% overlap for context
            
            for i in range(0, len(tokens) - self.max_length + 1, stride):
                chunk = tokens[i:i + self.max_length]
                
                # Pad if needed
                if len(chunk) < self.max_length:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
                
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                # For causal LM, labels are the same as inputs (shifted internally by model)
                samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone(),
                    'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class OllamaManager:
    """Manages Ollama server lifecycle"""
    
    def __init__(self, model: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.process = None
    
    def is_running(self) -> bool:
        """Check if Ollama server is responding"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def has_model(self) -> bool:
        """Check if required model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available = [m['name'] for m in models]
                return self.model in available
        except:
            pass
        return False
    
    def start_server(self) -> bool:
        """Attempt to start Ollama server"""
        if self.is_running():
            logger.info("✅ Ollama already running")
            return True
        
        logger.info("🚀 Starting Ollama server...")
        
        try:
            if os.name == 'nt':  # Windows
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:  # Linux/Mac
                self.process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Wait for startup
            for i in range(10):
                time.sleep(1)
                if self.is_running():
                    logger.info("✅ Ollama server started")
                    return True
            
            logger.warning("⏱️  Ollama server timeout")
            return False
            
        except FileNotFoundError:
            logger.error("❌ Ollama not installed. Download from https://ollama.com")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to start Ollama: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Download model if not available"""
        if self.has_model():
            logger.info(f"✅ Model {self.model} already available")
            return True
        
        logger.info(f"📥 Pulling model {self.model}...")
        
        try:
            result = subprocess.run(
                ["ollama", "pull", self.model],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Model {self.model} downloaded")
                return True
            else:
                logger.error(f"❌ Failed to pull model: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to pull model: {e}")
            return False
    
    def setup(self) -> bool:
        """Complete Ollama setup"""
        if not self.start_server():
            return False
        
        if not self.pull_model():
            logger.warning("⚠️  Model not available, continuing without Ollama")
            return False
        
        return True


def create_rtx_optimized_config() -> HybridConfig:
    """
    Create configuration optimized for RTX 5060 Ti (16GB VRAM)
    Based on production_optimized preset but tuned for 16GB
    """
    return HybridConfig(
        # Strategic injection points (5 layers for memory efficiency)
        injection_layers=[4, 8, 12, 16, 20],
        
        # TRM architecture (scaled for 16GB)
        trm_hidden_size=2048,      # 2K hidden size
        trm_num_heads=16,          # 16 attention heads
        trm_ff_hidden_size=8192,   # 8K feedforward dimension
        trm_num_layers=4,          # 4-layer TRM stack
        num_recursion_steps=6,     # 6 recursion steps
        
        # Regularization
        dropout=0.1,
        recursion_dropout=0.12,
        
        # Injection configuration
        adaptive_injection=True,
        injection_strength=0.75,   # Balanced strength
        blending_strategy="attention_weighted",
        
        # Projection layers
        projection_bias=True,
        projection_activation="gelu",
        layer_norm_eps=1e-6,
        
        # Memory optimization
        gradient_checkpointing=True,  # Enable for memory savings
        mixed_precision=True,         # FP16 training
    )


def load_base_llm(model_name: str = "gpt2") -> Tuple[nn.Module, Any]:
    """
    Load base LLM model and tokenizer
    
    Args:
        model_name: HuggingFace model name (gpt2, gpt2-medium, gpt2-large)
    
    Returns:
        model, tokenizer
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library required")
    
    logger.info(f"📥 Loading base LLM: {model_name}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    logger.info(f"✅ Loaded {model_name}")
    logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"   Vocab size: {tokenizer.vocab_size}")
    
    return model, tokenizer


def create_hybrid_model(base_llm: nn.Module, config: HybridConfig) -> CombinedTRMLLM:
    """Create full hybrid model with layer injection"""
    if not CORE_AVAILABLE:
        raise ImportError("MillennialAI core not available")
    
    logger.info("🔨 Creating hybrid model with layer injection...")
    
    hybrid_model = CombinedTRMLLM(llm_model=base_llm, config=config)
    
    # Get parameter counts
    params = hybrid_model.get_parameter_count()
    logger.info(f"✅ Hybrid model created:")
    logger.info(f"   Total parameters: {params['total']:,}")
    logger.info(f"   LLM parameters: {params['llm_model']:,}")
    logger.info(f"   TRM parameters: {params['trm_block']:,}")
    logger.info(f"   Projection parameters: {params['projection']:,}")
    
    return hybrid_model


class EnterpriseTrainer:
    """Enterprise-grade training loop with all optimizations"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: HybridConfig,
        device: torch.device,
        save_dir: Path,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 500,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info("✅ Trainer initialized")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"   Mixed precision: {config.mixed_precision}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        
        # Activate injection
        if hasattr(self.model, 'activate_injection'):
            self.model.activate_injection()
        
        total_loss = 0
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        
        # Save epoch checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"⭐ Saved best model: {best_path}")
    
    def train(self):
        """Full training loop"""
        logger.info("🚀 Starting enterprise training...")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
        
        logger.info("🎉 Training complete!")


def extract_and_save_embeddings(model: nn.Module, save_dir: Path):
    """Extract embeddings and create FAISS index"""
    logger.info("📊 Extracting embeddings...")
    
    # Get embeddings from base LLM
    if hasattr(model, 'llm_model'):
        if hasattr(model.llm_model, 'transformer'):
            # GPT-2 style
            embeddings = model.llm_model.transformer.wte.weight.data.cpu()
        elif hasattr(model.llm_model, 'embeddings'):
            # BERT style
            embeddings = model.llm_model.embeddings.word_embeddings.weight.data.cpu()
        else:
            logger.warning("Could not find embeddings")
            return
    else:
        logger.warning("Model has no llm_model attribute")
        return
    
    # Save embeddings tensor
    embeddings_path = save_dir / "embeddings.pt"
    torch.save(embeddings, embeddings_path)
    logger.info(f"💾 Saved embeddings: {embeddings_path}")
    logger.info(f"   Shape: {embeddings.shape}")
    
    # Create FAISS index
    if FAISS_AVAILABLE:
        embeddings_np = embeddings.numpy().astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        # Create index
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity for normalized vectors
        index.add(embeddings_np)
        
        # Save index
        index_path = save_dir / "vectors.faiss"
        faiss.write_index(index, str(index_path))
        logger.info(f"💾 Saved FAISS index: {index_path}")
        logger.info(f"   Vectors: {index.ntotal}")
        logger.info(f"   Dimensions: {dim}")


def main():
    """Main training pipeline"""
    print("="*80)
    print("🚀 MILLENNIALAI ENTERPRISE TRAINING SYSTEM")
    print("🎯 Full Production Architecture - RTX 5060 Ti Optimized")
    print("="*80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logger.info(f"   CUDA: {torch.version.cuda}")
    
    # Note: Ollama is for HybridRevolutionaryBrain (inference/API use)
    # Training uses CombinedTRMLLM (GPT-2 + TRM injection only)
    logger.info("ℹ️  Training CombinedTRMLLM architecture (GPT-2 + TRM injection)")
    logger.info("ℹ️  Ollama is for HybridRevolutionaryBrain (inference layer, not needed for training)")
    ollama_ready = False  # Not used during training
    
    # Load base LLM
    base_llm, tokenizer = load_base_llm("gpt2")  # Can use gpt2-medium or gpt2-large
    
    # Create configuration
    config = create_rtx_optimized_config()
    logger.info(f"⚙️  Configuration: {config}")
    
    # Create hybrid model
    hybrid_model = create_hybrid_model(base_llm, config)
    hybrid_model = hybrid_model.to(device)
    
    # Create dataset
    dataset = EnterpriseDataset(
        tokenizer=tokenizer,
        data_dir=PROJECT_ROOT,
        max_length=512
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"📊 Dataset split: {train_size} train, {val_size} val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Small batch for 16GB VRAM
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup save directory
    save_dir = PROJECT_ROOT / "models"
    save_dir.mkdir(exist_ok=True)
    
    # Create trainer
    trainer = EnterpriseTrainer(
        model=hybrid_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=save_dir,
        learning_rate=2e-5,
        num_epochs=5,
        gradient_accumulation_steps=4,
    )
    
    # Train
    trainer.train()
    
    # Extract embeddings and create FAISS index
    extract_and_save_embeddings(hybrid_model, save_dir)
    
    # Save final model
    final_model_path = save_dir / "millennialai_enterprise.pt"
    torch.save({
        'model_state_dict': hybrid_model.state_dict(),
        'config': config.to_dict(),
        'tokenizer_name': "gpt2",
    }, final_model_path)
    logger.info(f"💾 Saved final model: {final_model_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'project_root': str(PROJECT_ROOT),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'base_llm': "gpt2",
        'config': config.to_dict(),
        'train_samples': train_size,
        'val_samples': val_size,
        'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else None,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None,
        'best_val_loss': trainer.best_val_loss,
        'ollama_enabled': ollama_ready,
        'components': {
            'core': CORE_AVAILABLE,
            'brain': BRAIN_AVAILABLE,
            'transformers': TRANSFORMERS_AVAILABLE,
            'faiss': FAISS_AVAILABLE,
        }
    }
    
    metadata_path = save_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"💾 Saved metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📦 Generated files in {save_dir}/:")
    print(f"   - best_model.pt (best validation checkpoint)")
    print(f"   - millennialai_enterprise.pt (final model)")
    print(f"   - checkpoint_epoch_*.pt (training checkpoints)")
    print(f"   - embeddings.pt (token embeddings)")
    print(f"   - vectors.faiss (FAISS vector database)")
    print(f"   - training_metadata.json (training info)")
    print("\n🚀 Next steps:")
    print("   1. Test model: python -c 'import torch; m=torch.load(\"models/best_model.pt\")'")
    print("   2. Deploy API: python millennial_ai_api.py")
    print("   3. Run inference: Use HybridRevolutionaryBrain for queries")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        raise
