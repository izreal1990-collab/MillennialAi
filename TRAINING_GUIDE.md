# MillennialAi Training Guide

## Understanding the Hybrid Architecture

MillennialAi is **NOT** a replacement for existing LLMs. It's an **enhancement system** that injects additional intelligence into pre-trained models.

## Training Progression Levels

### 🥉 **Level 1: Beginner (Recommended Start)**
**Base Model**: Pre-trained LLaMA-2-7B or similar
**TRM Addition**: ~1-2B parameters
**Total Size**: ~8-9B parameters
**Memory**: ~16-18 GB (FP16)
**Hardware**: Single RTX 4090 or A100

```python
from millennial_ai.config.config import HybridConfig

# Beginner configuration
config = HybridConfig(
    injection_layers=[16, 32],     # Just 2 injection points
    trm_hidden_size=2048,          # Modest TRM size
    trm_num_heads=16,              # Reasonable attention heads
    trm_num_layers=2,              # Shallow TRM stack
    num_recursion_steps=4,         # Basic recursion
)
```

### 🥈 **Level 2: Intermediate**
**Base Model**: Pre-trained LLaMA-2-13B
**TRM Addition**: ~3-5B parameters  
**Total Size**: ~16-18B parameters
**Memory**: ~32-36 GB (FP16)
**Hardware**: 2x A100 40GB or 1x A100 80GB

```python
config = HybridConfig(
    injection_layers=[8, 16, 24, 32, 40],  # 5 injection points
    trm_hidden_size=4096,                   # Larger TRM
    trm_num_heads=32,
    trm_num_layers=3,
    num_recursion_steps=6,
)
```

### 🥇 **Level 3: Advanced Enterprise**
**Base Model**: Pre-trained LLaMA-2-70B
**TRM Addition**: ~15-20B parameters
**Total Size**: ~85-90B parameters  
**Memory**: ~170-180 GB (FP16)
**Hardware**: 8x A100 80GB minimum

```python
config = PresetConfigs.llama_2_70b_enterprise()  # Pre-configured
```

### 🏆 **Level 4: Research/Ultra Scale**
**Base Model**: LLaMA-2-70B+ or custom
**TRM Addition**: ~50B+ parameters
**Total Size**: ~120B+ parameters
**Memory**: ~240+ GB (FP16)
**Hardware**: 16+ H100 80GB cluster

```python
config = PresetConfigs.research_experimental()  # Maximum scale
```

## Training Steps for Each Level

### Step 1: Prepare Base Model
```python
# Load pre-trained model (NOT trained from scratch)
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",  # Start with 7B
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Step 2: Create Hybrid Architecture
```python
from millennial_ai.core.hybrid_model import CombinedTRMLLM

hybrid_model = CombinedTRMLLM(
    llm_model=base_model,
    config=config
)

# Only TRM components need training initially
for name, param in hybrid_model.named_parameters():
    if 'trm' not in name.lower():
        param.requires_grad = False  # Freeze base model
```

### Step 3: Training Strategy
```python
# Phase 1: Train only TRM injection (fast)
optimizer = torch.optim.AdamW(
    [p for p in hybrid_model.parameters() if p.requires_grad],
    lr=1e-4
)

# Phase 2: Fine-tune everything (optional)
for param in hybrid_model.parameters():
    param.requires_grad = True
    
optimizer = torch.optim.AdamW(
    hybrid_model.parameters(),
    lr=1e-5  # Lower learning rate
)
```

## Training Data Requirements

### Level 1 (Beginner): 1-10GB
- General text corpus
- Simple Q&A datasets
- Basic instruction following

### Level 2 (Intermediate): 10-100GB  
- Multi-domain datasets
- Code repositories
- Academic papers
- Instruction datasets

### Level 3 (Enterprise): 100GB-1TB
- Massive text corpora
- Specialized domain data
- Multi-modal datasets
- Enterprise-specific data

### Level 4 (Research): 1TB+
- Research datasets
- Multiple languages
- Complex reasoning tasks
- Cutting-edge capabilities

## Key Advantages

1. **You DON'T train a 70B model from scratch** (impossible for most)
2. **You start with pre-trained LLaMA-2-70B** (already trained by Meta)
3. **You only train the TRM injection layers** (much faster/cheaper)
4. **You get 85B total capability** with minimal training

## Cost Comparison

### Traditional Approach (Training 70B from scratch):
- **Cost**: $10-50 million
- **Time**: 6-12 months
- **Hardware**: Massive compute cluster
- **Risk**: Very high

### MillennialAi Approach (TRM Injection):
- **Cost**: $10,000-100,000  
- **Time**: 1-4 weeks
- **Hardware**: 8-16 GPUs
- **Risk**: Low (base model already works)

## Getting Started Recommendation

**Start with Level 1:**
1. Use LLaMA-2-7B as base
2. Add minimal TRM injection
3. Train on small dataset
4. Validate the concept
5. Scale up gradually

**Do NOT start with 70B models** unless you have enterprise infrastructure!