"""
MillennialAI Hybrid Training System
GPU + CPU Optimized Training for llama3.1-13B + TRM

This is the CLEAN, PRODUCTION version
Run from Anaconda Prompt for stability
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import logging
from pathlib import Path
import gc
from typing import List

from hybrid_config import *
from real_brain import RealThinkingBrain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TRM INJECTION WRAPPER
# ============================================================================

class TRMInjectionLayer(nn.Module):
    """
    TRM injection that handles cross-device operation (GPU ↔ CPU)
    """
    def __init__(self, original_layer, trm_brain, layer_idx: int):
        super().__init__()
        self.original = original_layer
        self.trm = trm_brain
        self.layer_idx = layer_idx
        self.original_device = next(original_layer.parameters()).device
        self.trm_device = next(trm_brain.parameters()).device
        
        logger.info(f"  Layer {layer_idx}: Original={self.original_device}, TRM={self.trm_device}")
    
    def forward(self, hidden_states, *args, **kwargs):
        # Run original transformer layer
        outputs = self.original(hidden_states, *args, **kwargs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Move to TRM device if needed
        original_device = hidden.device
        if hidden.device != self.trm_device:
            hidden = hidden.to(self.trm_device)
        
        # Apply TRM reasoning
        trm_output = self.trm.thinking_modules[0](hidden)
        
        # Residual connection (10% TRM influence)
        enhanced = hidden + 0.1 * trm_output
        
        # Move back to original device
        if enhanced.device != original_device:
            enhanced = enhanced.to(original_device)
        
        # Return in original format
        return (enhanced,) + outputs[1:] if isinstance(outputs, tuple) else enhanced


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_base_model():
    """Load llama3-8B with 4-bit quantization (GPU-only)"""
    logger.info("="*80)
    logger.info("🚀 LOADING BASE MODEL")
    logger.info("="*80)
    logger.info(f"Model: {BASE_MODEL}")
    logger.info(f"Quantization: 4-bit NF4")
    logger.info(f"Device: GPU (cuda:0)")
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4-bit quantization config
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load with 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info("✅ Base model loaded (4-bit quantized)")
    
    return model, tokenizer


def inject_trm_layers(model):
    """Inject TRM at strategic points"""
    logger.info("="*80)
    logger.info("🧠 INJECTING TRM LAYERS")
    logger.info("="*80)
    logger.info(f"Injection Points: {TRM_INJECTION_POINTS}")
    
    for layer_idx in TRM_INJECTION_POINTS:
        if layer_idx >= len(model.model.layers):
            logger.warning(f"  ⚠️  Layer {layer_idx} out of range, skipping")
            continue
        
        # Get original layer
        original_layer = model.model.layers[layer_idx]
        layer_device = next(original_layer.parameters()).device
        
        # Create TRM on same device
        trm_brain = RealThinkingBrain(
            hidden_size=MODEL_HIDDEN_SIZE,
            max_depth=TRM_MAX_DEPTH
        ).to(layer_device)
        
        # Wrap with TRM injection
        model.model.layers[layer_idx] = TRMInjectionLayer(
            original_layer,
            trm_brain,
            layer_idx
        )
    
    logger.info(f"✅ {len(TRM_INJECTION_POINTS)} TRM layers injected")
    return model


def setup_lora(model):
    """Configure LoRA for parameter-efficient training"""
    logger.info("="*80)
    logger.info("🔧 CONFIGURING LORA")
    logger.info("="*80)
    
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    logger.info("✅ LoRA configured")
    
    return model


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def load_training_data(tokenizer):
    """Load and tokenize workspace documents + OpenAssistant conversations"""
    logger.info("="*80)
    logger.info("📚 LOADING TRAINING DATA")
    logger.info("="*80)
    
    all_texts = []
    
    # 1. Load workspace documents (technical knowledge)
    files = []
    for pattern in DATASET_PATTERNS:
        found = list(PROJECT_ROOT.glob(f"**/{pattern}"))
        # Filter out excluded patterns
        found = [f for f in found if not any(excl in str(f) for excl in EXCLUDE_PATTERNS)]
        files.extend(found)
    
    logger.info(f"📁 Found {len(files)} workspace files")
    
    for file in files:
        try:
            content = file.read_text(encoding='utf-8')
            if len(content.strip()) > 100:  # Skip tiny files
                all_texts.append(content)
        except Exception as e:
            logger.warning(f"  ⚠️  Could not read {file.name}: {e}")
    
    logger.info(f"✅ Loaded {len(all_texts)} workspace documents")
    
    # 2. Load OpenAssistant conversations
    if USE_OPENASSISTANT:
        logger.info("📥 Loading OpenAssistant dataset...")
        logger.info("   This may take 1-2 minutes on first run...")
        
        try:
            oasst_dataset = load_dataset(
                "OpenAssistant/oasst1",
                split=f"train[:{OPENASSISTANT_SAMPLES}]"
            )
            
            # Convert to conversation format
            logger.info("   Processing conversations...")
            conversations = {}
            
            for item in oasst_dataset:
                msg_id = item['message_id']
                parent_id = item['parent_id']
                text = item['text']
                role = item['role']
                
                # Build conversation threads
                if parent_id is None:
                    # Root message
                    conversations[msg_id] = f"User: {text}"
                else:
                    # Reply in thread
                    if parent_id in conversations:
                        prefix = "Assistant: " if role == "assistant" else "User: "
                        conversations[parent_id] += f"\n\n{prefix}{text}"
            
            # Add conversations to training data
            conv_list = list(conversations.values())
            all_texts.extend(conv_list)
            
            logger.info(f"✅ Added {len(conv_list)} OpenAssistant conversations")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load OpenAssistant: {e}")
            logger.warning("   Continuing with workspace documents only...")
    
    logger.info(f"📊 Total training samples: {len(all_texts)}")
    
    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({'text': all_texts})
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )
    
    logger.info("🔄 Tokenizing...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=CPU_CORES,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"✅ Dataset ready: {len(tokenized)} samples")
    return tokenized


# ============================================================================
# TRAINING
# ============================================================================

def train_hybrid_model():
    """Main training function"""
    print_config()
    
    # Load model
    model, tokenizer = load_base_model()
    
    # Inject TRM
    model = inject_trm_layers(model)
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load data
    dataset = load_training_data(tokenizer)
    
    # Training arguments
    logger.info("="*80)
    logger.info("⚙️  TRAINING CONFIGURATION")
    logger.info("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        **TRAINING_CONFIG
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Prevent Trainer from moving model (already on GPU+CPU)
    # Monkey-patch the model's .to() method to be a no-op
    original_to = model.to
    def no_op_to(*args, **kwargs):
        return model
    model.to = no_op_to
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Restore original .to() method after Trainer init
    model.to = original_to
    
    # Train
    logger.info("="*80)
    logger.info("🎯 STARTING TRAINING")
    logger.info("="*80)
    logger.info("⚠️  Monitor on iGPU recommended")
    logger.info("⚠️  Run from Anaconda Prompt for stability")
    logger.info("="*80)
    
    try:
        trainer.train()
        
        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        
        # Save final model
        final_dir = OUTPUT_DIR / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        
        logger.info(f"💾 Model saved to: {final_dir}")
        logger.info("\n🚀 Next steps:")
        logger.info("  1. python hybrid_merge.py    # Merge LoRA weights")
        logger.info("  2. python hybrid_deploy.py   # Convert to GGUF + Ollama")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        logger.info("💾 Checkpoints saved in: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_hybrid_model()
