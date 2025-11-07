"""
MillennialAI Hybrid Training Configuration
GPU + CPU Optimized for Maximum Performance

Hardware Profile:
- RTX 5060 Ti: 16GB VRAM (sm_120)
- Ryzen 7 7700: 8C/16T CPU
- 32GB DDR5 RAM
- B650 PCIe 5.0

Target: llama3.1-13B with 9 TRM injection points
"""

from pathlib import Path

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

GPU_VRAM_GB = 14  # Reserve 2GB for Windows
CPU_RAM_GB = 28   # Reserve 4GB for system
CPU_CORES = 6     # Use 6/8 cores (leave 2 for system)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Use 8B with 4-bit quantization (fits in 16GB VRAM easily)
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
MODEL_HIDDEN_SIZE = 4096  # 8B model hidden dimension
MODEL_LAYERS = 32         # llama3-8B has 32 layers

# Use 4-bit quantization for GPU-only training
USE_4BIT = True  # Fits easily in 16GB VRAM

# TRM Injection Points (strategic layer placement)
TRM_INJECTION_POINTS = [
    4,   # Early reasoning (GPU)
    8,   # Pattern recognition (GPU)
    12,  # Mid-layer synthesis (CPU)
    16,  # Deep reasoning (CPU)
    20,  # Advanced analysis (GPU)
    24,  # Integration layer (GPU)
    28,  # Refinement (GPU)
]

TRM_MAX_DEPTH = 8

# ============================================================================
# DEVICE MAPPING (GPU + CPU Hybrid)
# ============================================================================

# Simplified: GPU-only for training stability
# CPU offloading causes device mismatch errors during training
HYBRID_DEVICE_MAP = "auto"  # Let transformers handle it automatically

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    # Batch configuration for hybrid setup
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,  # Effective batch = 32
    
    # Training duration
    "num_train_epochs": 5,
    "max_steps": -1,  # Train full epochs
    
    # Learning rate
    "learning_rate": 1e-4,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",
    
    # Optimization
    "optim": "paged_adamw_32bit",  # Uses CPU RAM for optimizer states
    "max_grad_norm": 0.3,
    "weight_decay": 0.01,
    
    # Precision
    "fp16": True,
    "bf16": False,  # RTX 5060 Ti doesn't support BF16 efficiently
    
    # Memory optimization
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    
    # Multi-threading (Ryzen 7 7700)
    "dataloader_num_workers": CPU_CORES,
    "dataloader_prefetch_factor": 4,
    "dataloader_pin_memory": True,
    
    # Logging
    "logging_steps": 5,
    "logging_first_step": True,
    
    # Checkpointing
    "save_steps": 200,
    "save_total_limit": 3,
    "save_strategy": "steps",
    
    # Evaluation
    "evaluation_strategy": "no",
    
    # Misc
    "report_to": "none",
    "remove_unused_columns": False,
    "ddp_find_unused_parameters": False,
}

# ============================================================================
# LORA CONFIGURATION
# ============================================================================

LORA_CONFIG = {
    "r": 128,  # Larger rank for 13B model
    "lora_alpha": 32,
    "target_modules": [
        "q_proj",      # Query projection
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "o_proj",      # Output projection
        "gate_proj",   # MLP gate
        "up_proj",     # MLP up
        "down_proj",   # MLP down
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "inference_mode": False,
}

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

MAX_SEQ_LENGTH = 2048  # Longer context for better understanding
DATASET_PATTERNS = ["*.md", "*.txt", "*.py"]
EXCLUDE_PATTERNS = ["archive/*", "llama.cpp/*", "__pycache__/*"]

# Conversational datasets
USE_OPENASSISTANT = True  # High-quality multi-turn conversations
OPENASSISTANT_SAMPLES = 10000  # Use 10K examples (from 88K total)

# ============================================================================
# PATHS
# ============================================================================

def get_workspace():
    """Find workspace root"""
    paths = [
        Path.cwd(),
        Path(r"C:\Users\jblan\MillennialAi"),
        Path.home() / "MillennialAi",
    ]
    for p in paths:
        if p.exists() and (p / "real_brain.py").exists():
            return p
    raise FileNotFoundError("Workspace not found")

PROJECT_ROOT = get_workspace()
OUTPUT_DIR = PROJECT_ROOT / "models" / "hybrid_13b"
OFFLOAD_DIR = PROJECT_ROOT / "cpu_offload"
MERGED_MODEL_DIR = OUTPUT_DIR / "merged"
GGUF_DIR = OUTPUT_DIR / "gguf"

# ============================================================================
# SYSTEM OPTIMIZATION
# ============================================================================

import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', message='.*CUDA capability.*')
warnings.filterwarnings('ignore', message='.*TypedStorage is deprecated.*')

# Disable torch compile (stability)
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ============================================================================
# DISPLAY INFO
# ============================================================================

def print_config():
    """Print configuration summary"""
    print("=" * 80)
    print("🚀 MILLENNIALAI HYBRID TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"📦 Model: {BASE_MODEL}")
    print(f"🧠 TRM Injections: {len(TRM_INJECTION_POINTS)} points")
    print(f"🎮 GPU: RTX 5060 Ti ({GPU_VRAM_GB}GB VRAM)")
    print(f"💻 CPU: Ryzen 7 7700 ({CPU_CORES} cores)")
    print(f"💾 RAM: {CPU_RAM_GB}GB")
    print(f"📊 Batch: {TRAINING_CONFIG['per_device_train_batch_size']} × {TRAINING_CONFIG['gradient_accumulation_steps']} = {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"📈 Epochs: {TRAINING_CONFIG['num_train_epochs']}")
    print(f"📝 Context: {MAX_SEQ_LENGTH} tokens")
    print(f"💪 LoRA Rank: {LORA_CONFIG['r']}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()
