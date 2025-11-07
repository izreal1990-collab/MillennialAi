"""
MillennialAI LoRA Merge Script
Merges trained LoRA adapters into base model using CPU offloading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from pathlib import Path
import gc
import shutil

from hybrid_config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_latest_checkpoint():
    """Find most recent training checkpoint"""
    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
    if not checkpoints:
        # Try final model
        if (OUTPUT_DIR / "final").exists():
            return OUTPUT_DIR / "final"
        raise FileNotFoundError(f"No checkpoints found in {OUTPUT_DIR}")
    return checkpoints[-1]


def merge_lora_weights():
    """Merge LoRA adapters using CPU offloading"""
    logger.info("="*80)
    logger.info("🔗 LORA MERGE (CPU OFFLOADING)")
    logger.info("="*80)
    
    # Find checkpoint
    checkpoint = find_latest_checkpoint()
    logger.info(f"📂 Checkpoint: {checkpoint}")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create temporary offload directory
    temp_offload = PROJECT_ROOT / "temp_merge_offload"
    temp_offload.mkdir(exist_ok=True)
    
    try:
        # Load base model with CPU offloading
        logger.info("🦙 Loading base model (FP16)...")
        logger.info("   Using CPU offloading for memory safety")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=str(temp_offload),
            offload_state_dict=True,
            max_memory={
                0: f"{GPU_VRAM_GB}GB",
                "cpu": f"{CPU_RAM_GB}GB"
            },
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        logger.info("✅ Base model loaded")
        
        # Load LoRA adapters
        logger.info("🔧 Loading LoRA adapters...")
        
        model = PeftModel.from_pretrained(
            base_model,
            str(checkpoint),
            device_map="auto",
            offload_folder=str(temp_offload),
        )
        
        logger.info("✅ LoRA adapters loaded")
        
        # Merge
        logger.info("🔗 Merging LoRA weights into base model...")
        logger.info("   ⏱️  This may take 5-10 minutes with CPU offloading")
        logger.info("   GPU layers: Fast (~2 min)")
        logger.info("   CPU layers: Slower (~5 min)")
        
        model = model.merge_and_unload()
        
        logger.info("✅ Merge complete!")
        
        # Save merged model
        logger.info("💾 Saving merged model...")
        MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(
            str(MERGED_MODEL_DIR),
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.save_pretrained(str(MERGED_MODEL_DIR))
        
        logger.info(f"✅ Merged model saved: {MERGED_MODEL_DIR}")
        
        # Cleanup
        logger.info("🧹 Cleaning up...")
        del model, base_model
        gc.collect()
        torch.cuda.empty_cache()
        
        if temp_offload.exists():
            shutil.rmtree(temp_offload)
        
        logger.info("="*80)
        logger.info("✅ MERGE COMPLETE!")
        logger.info("="*80)
        logger.info(f"📦 Output: {MERGED_MODEL_DIR}")
        logger.info("\n🚀 Next step:")
        logger.info("  python hybrid_deploy.py")
        
        return MERGED_MODEL_DIR
        
    except Exception as e:
        logger.error(f"❌ Merge failed: {e}", exc_info=True)
        
        # Cleanup on error
        if temp_offload.exists():
            shutil.rmtree(temp_offload)
        
        raise


if __name__ == "__main__":
    try:
        merge_lora_weights()
    except Exception as e:
        logger.error("\n💡 Troubleshooting:")
        logger.error("  - Make sure training completed successfully")
        logger.error("  - Close other applications to free memory")
        logger.error("  - Run from Anaconda Prompt")
        logger.error("  - Monitor on iGPU if possible")
