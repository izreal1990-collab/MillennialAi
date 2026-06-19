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
import gc
from typing import Dict, Any, Tuple
import weakref

from hybrid_config import (
    BASE_MODEL,
    MODEL_HIDDEN_SIZE,
    TRM_MAX_DEPTH,
    TRM_INJECTION_POINTS,
    TRM_SHARED_CACHE,
    TRM_DEVICE_POLICY,
    TRM_BLEND_ALPHA,
    LORA_CONFIG,
    MAX_SEQ_LENGTH,
    DATASET_PATTERNS,
    PROJECT_ROOT,
    EXCLUDE_PATTERNS,
    USE_OPENASSISTANT,
    OPENASSISTANT_SAMPLES,
    OUTPUT_DIR,
    TRAINING_CONFIG,
    print_config,
)
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
    logger.info("LOADING BASE MODEL")
    logger.info("="*80)
    logger.info("Model: %s", BASE_MODEL)
    logger.info("Quantization: 4-bit NF4")
    logger.info("Device: GPU (cuda:0)")
    
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
    logger.info("INJECTING TRM LAYERS")
    logger.info("="*80)
    logger.info("Injection Points: %s", TRM_INJECTION_POINTS)

    # Discover transformer's layer container
    layers_container, parent_module, attr_name = discover_transformer_layers(model)
    layers = layers_container

    # TRM instances will be created lazily per device via get_trm_for_device

    # Manager to provide TRM instances per device (lazy)
    trm_cache: Dict[str, RealThinkingBrain] = {}

    def get_trm_for_device(device):
        key = str(device)
        if TRM_SHARED_CACHE and key in trm_cache:
            return trm_cache[key]
        # Clone lightweight TRM to device
        trm = RealThinkingBrain(hidden_size=MODEL_HIDDEN_SIZE, max_depth=TRM_MAX_DEPTH)
        trm.to(device)
        if TRM_SHARED_CACHE:
            trm_cache[key] = trm
        return trm

    injected = 0
    for layer_idx in TRM_INJECTION_POINTS:
        if layer_idx >= len(layers):
            logger.warning("Layer %d out of range, skipping", layer_idx)
            continue

        original_layer = layers[layer_idx]
        # Try to deduce layer device; fallback to CPU
        try:
            layer_device = next(original_layer.parameters()).device
        except Exception:
            layer_device = torch.device('cpu')

        # Choose TRM device per policy
        if TRM_DEVICE_POLICY == 'same_as_layer':
            trm_device = layer_device
        elif TRM_DEVICE_POLICY == 'auto':
            trm_device = layer_device if 'cuda' in str(layer_device) else torch.device('cpu')
        else:
            trm_device = torch.device('cpu')

        # Use cached TRM for the device
        trm_instance = get_trm_for_device(trm_device)

        # Wrap with safer TRM injection wrapper
        wrapper = TRMInjectionWrapper(
            original_layer=original_layer,
            trm_template=trm_instance,
            layer_idx=layer_idx,
            alpha=TRM_BLEND_ALPHA
        )

        # Replace in parent container
        if isinstance(layers, list):
            parent = parent_module
            getattr(parent, attr_name)[layer_idx] = wrapper
        else:
            setattr(parent_module, attr_name, layers)

        injected += 1

    logger.info("%d TRM layers injected", injected)
    return model


def discover_transformer_layers(model) -> Tuple[Any, Any, str]:
    """Attempt to locate the transformer layer list/module in a few common locations.

    Returns: (layers_container, parent_module, attr_name)
    """
    candidates = [
        (['model', 'layers'], 'layers'),
        (['transformer', 'h'], 'h'),
        (['base_model', 'model', 'layers'], 'layers'),
        (['model', 'decoder', 'layers'], 'layers'),
        (['model', 'encoder', 'layers'], 'layers'),
    ]

    for path, attr in candidates:
        cur = model
        parent = None
        found = True
        for p in path:
            parent = cur
            if not hasattr(cur, p):
                found = False
                break
            cur = getattr(cur, p)
        if found and (isinstance(cur, (list, torch.nn.ModuleList))):
            return cur, parent, attr

    # Fallback: try to find any ModuleList attribute containing many layers
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Module):
            for subname, sub in module.named_children():
                if isinstance(sub, torch.nn.ModuleList) and len(sub) > 4:
                    return sub, module, subname

    raise RuntimeError('Could not discover transformer layers container')


class TRMInjectionWrapper(nn.Module):
    """Wrapper that calls the original layer and blends TRM output into hidden states."""
    def __init__(self, original_layer: nn.Module, trm_template: RealThinkingBrain, layer_idx: int, alpha: float = 0.1):
        super().__init__()
        self.original = original_layer
        self.layer_idx = layer_idx
        self.alpha = alpha
        # Keep a weakref to template (not used for parameters)
        self._trm_template = weakref.ref(trm_template)

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.original(hidden_states, *args, **kwargs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs

        # Get trm instance from template and move hidden there
        trm_template = self._trm_template()
        trm_device = next(trm_template.parameters()).device if trm_template is not None else torch.device('cpu')

        original_device = hidden.device
        hidden_on_trm = hidden.to(trm_device)

        # Call TRM (assumes it accepts hidden tensor)
        if trm_template is not None and hasattr(trm_template, 'thinking_modules'):
            trm_out = trm_template.thinking_modules[0](hidden_on_trm)
        else:
            trm_out = trm_template(hidden_on_trm)

        enhanced = hidden_on_trm + self.alpha * trm_out

        if enhanced.device != original_device:
            enhanced = enhanced.to(original_device)

        # Reconstruct outputs preserving auxiliary outputs
        if isinstance(outputs, tuple):
            return (enhanced,) + outputs[1:]
        return enhanced


def revert_trm_injections(model):
    """Revert injected wrappers by replacing wrapper.original back into the model layers."""
    layers_container, parent_module, attr_name = discover_transformer_layers(model)
    layers = layers_container
    restored = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, TRMInjectionWrapper):
            layers[i] = layer.original
            restored += 1
    logger.info("%d TRM injections reverted", restored)
    return model


def setup_lora(model):
    """Configure LoRA for parameter-efficient training"""
    logger.info("="*80)
    logger.info("CONFIGURING LORA")
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
    
    # Stream workspace documents to avoid building a large in-memory list
    logger.info("📁 Scanning workspace files (streaming)")
    files = []
    for pattern in DATASET_PATTERNS:
        found = list(PROJECT_ROOT.glob(f"**/{pattern}"))
        # Filter out excluded patterns
        found = [f for f in found if not any(excl in str(f) for excl in EXCLUDE_PATTERNS)]
        files.extend(found)

    logger.info("Found %d workspace files (candidates)", len(files))

    def iter_texts():
        # Yield workspace file contents lazily
        for file in files:
            try:
                content = file.read_text(encoding='utf-8')
                if len(content.strip()) > 100:  # Skip tiny files
                    yield {"text": content}
            except Exception as e:
                logger.warning("Could not read %s: %s", file.name, e)

        # Optionally stream OpenAssistant samples (small slice)
        if USE_OPENASSISTANT:
            logger.info("📥 Streaming OpenAssistant dataset slice...")
            try:
                oasst_dataset = load_dataset(
                    "OpenAssistant/oasst1",
                    split=f"train[:{OPENASSISTANT_SAMPLES}]"
                )
                for item in oasst_dataset:
                    text = item.get('text') or item.get('instruction') or ''
                    if text and len(text.strip()) > 20:
                        yield {"text": text}
                logger.info("OpenAssistant slice streamed")
            except Exception as e:
                logger.warning("Could not stream OpenAssistant: %s", e)

    # Create a Dataset from the generator to avoid large in-memory lists
    from datasets import Dataset
    dataset = Dataset.from_generator(iter_texts)

    logger.info("Total training samples (lazy): %d", len(dataset))

    # Tokenize: return plain lists (not PyTorch tensors) so Dataset stays native
    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
        # Ensure returned dict contains lists (input_ids, attention_mask if present)
        result = {k: v for k, v in outputs.items() if k in ("input_ids", "attention_mask")}
        return result
    
    logger.info("Tokenizing (batched)...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info("Dataset ready: %d samples", len(tokenized))
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
        
        logger.info("Model saved to: %s", final_dir)
        logger.info("\n🚀 Next steps:")
        logger.info("  1. python hybrid_merge.py    # Merge LoRA weights")
        logger.info("  2. python hybrid_deploy.py   # Convert to GGUF + Ollama")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Checkpoints saved in: %s", OUTPUT_DIR)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    train_hybrid_model()
