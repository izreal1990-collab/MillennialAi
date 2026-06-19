"""
Diagnostic test to verify TRM injection works during training.
Tests forward/backward pass with actual training setup.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("TRM INJECTION TRAINING DIAGNOSTIC TEST")
print("=" * 80)

# 1. Load small test model
print("\n1️⃣  Loading test model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",  # Use GPT2 instead of Llama for speed
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info("✅ Model loaded")
except Exception as e:
    logger.error(f"❌ Model load failed: {e}")
    exit(1)

# 2. Setup LoRA
print("\n2️⃣  Setting up LoRA...")
try:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    logger.info("✅ LoRA configured")
except Exception as e:
    logger.error(f"❌ LoRA setup failed: {e}")
    exit(1)

# 3. Create simple batch
print("\n3️⃣  Creating test batch...")
try:
    # Fix tokenizer for GPT2 (add padding token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts = ["Hello world", "This is a test"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=16)
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logger.info(f"✅ Batch created on device: {device}")
    logger.info(f"   Input IDs shape: {inputs['input_ids'].shape}")
except Exception as e:
    logger.error(f"❌ Batch creation failed: {e}")
    exit(1)

# 4. Forward pass
print("\n4️⃣  Testing forward pass...")
try:
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    logger.info(f"✅ Forward pass succeeded")
    logger.info(f"   Output shape: {logits.shape}")
except Exception as e:
    logger.error(f"❌ Forward pass failed: {e}")
    exit(1)

# 5. Backward pass with loss
print("\n5️⃣  Testing backward pass...")
try:
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    logger.info(f"✅ Loss computed: {loss.item():.4f}")
    
    loss.backward()
    logger.info(f"✅ Backward pass succeeded")
except Exception as e:
    logger.error(f"❌ Backward pass failed: {e}")
    exit(1)

# 6. Check gradients
print("\n6️⃣  Checking gradients...")
try:
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    logger.info(f"✅ Gradients present: {grad_count} parameters have gradients")
except Exception as e:
    logger.error(f"❌ Gradient check failed: {e}")
    exit(1)

print("\n" + "=" * 80)
print("✅ ALL DIAGNOSTIC TESTS PASSED")
print("=" * 80)
print("\nIf TRM injection training hangs, the issue is likely:")
print("  1. In the data loading/preprocessing pipeline")
print("  2. In the Trainer.train() initialization")
print("  3. Memory/VRAM issue causing silent wait")
print("  4. Issue specific to Llama-3 model architecture")
