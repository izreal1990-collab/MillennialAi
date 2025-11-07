"""
MillennialAI Deployment Script
Converts merged model to GGUF and creates Ollama configuration
"""

import subprocess
import logging
from pathlib import Path
import shutil

from hybrid_config import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clone_llama_cpp():
    """Clone llama.cpp if not already present"""
    llama_cpp_dir = PROJECT_ROOT / "llama.cpp"
    
    if llama_cpp_dir.exists():
        logger.info("✅ llama.cpp already exists")
        return llama_cpp_dir
    
    logger.info("📥 Cloning llama.cpp...")
    subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
        cwd=str(PROJECT_ROOT),
        check=True
    )
    logger.info("✅ llama.cpp cloned")
    return llama_cpp_dir


def find_converter(llama_cpp_dir: Path):
    """Find the GGUF conversion script"""
    converters = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert.py",
    ]
    
    for conv in converters:
        if conv.exists():
            logger.info(f"✅ Found converter: {conv.name}")
            return conv
    
    raise FileNotFoundError("Could not find GGUF converter in llama.cpp")


def convert_to_gguf():
    """Convert merged model to GGUF format"""
    logger.info("="*80)
    logger.info("🔄 CONVERTING TO GGUF")
    logger.info("="*80)
    
    # Check merged model exists
    if not MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Merged model not found: {MERGED_MODEL_DIR}\n"
            "Run: python hybrid_merge.py first"
        )
    
    logger.info(f"📂 Input: {MERGED_MODEL_DIR}")
    
    # Clone llama.cpp
    llama_cpp_dir = clone_llama_cpp()
    converter = find_converter(llama_cpp_dir)
    
    # Create output directory
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    output_file = GGUF_DIR / "millennialai-13b-f16.gguf"
    
    # Run conversion
    logger.info("🔄 Converting to GGUF (FP16)...")
    logger.info("   ⏱️  This takes 5-10 minutes")
    
    cmd = [
        "python",
        str(converter),
        str(MERGED_MODEL_DIR),
        "--outtype", "f16",
        "--outfile", str(output_file)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Conversion failed:\n{result.stderr}")
        raise RuntimeError("GGUF conversion failed")
    
    logger.info("✅ GGUF conversion complete!")
    logger.info(f"📦 Output: {output_file}")
    logger.info(f"📊 Size: {output_file.stat().st_size / 1e9:.2f} GB")
    
    return output_file


def create_ollama_modelfile(gguf_path: Path):
    """Create Ollama Modelfile for deployment"""
    logger.info("📝 Creating Ollama Modelfile...")
    
    modelfile_content = f"""# MillennialAI - 13B with TRM Reasoning
FROM {gguf_path}

# System prompt
SYSTEM \"\"\"You are MillennialAI, an advanced AI assistant with enhanced reasoning capabilities through TRM (Thinking & Reasoning Modules).

You have deep knowledge of:
- Software development and engineering
- AI/ML concepts and implementation
- System architecture and optimization
- Problem-solving with structured reasoning

You think step-by-step, consider multiple perspectives, and provide well-reasoned answers.
\"\"\"

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# Stop tokens
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
"""
    
    modelfile_path = GGUF_DIR / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    
    logger.info(f"✅ Modelfile created: {modelfile_path}")
    return modelfile_path


def deploy_to_ollama(modelfile_path: Path):
    """Deploy model to Ollama"""
    logger.info("="*80)
    logger.info("🚀 DEPLOYING TO OLLAMA")
    logger.info("="*80)
    
    model_name = "millennialai"
    
    logger.info(f"📦 Creating Ollama model: {model_name}")
    
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("✅ Ollama model created!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ollama deployment failed:\n{e.stderr}")
        logger.info("\n💡 Make sure Ollama is installed:")
        logger.info("   Download: https://ollama.ai")
        return False
    except FileNotFoundError:
        logger.warning("⚠️  Ollama not found in PATH")
        logger.info("\n💡 Install Ollama to deploy:")
        logger.info("   Download: https://ollama.ai")
        logger.info("\nManual deployment:")
        logger.info(f"   1. Install Ollama")
        logger.info(f"   2. ollama create {model_name} -f {modelfile_path}")
        return False
    
    logger.info("="*80)
    logger.info("✅ DEPLOYMENT COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n🎉 Ready to use!")
    logger.info(f"\nStart chatting:")
    logger.info(f"  ollama run {model_name}")
    logger.info(f"\nExample:")
    logger.info(f'  ollama run {model_name} "Explain how MillennialAI works"')
    
    return True


def main():
    """Full deployment pipeline"""
    logger.info("="*80)
    logger.info("🚀 MILLENNIALAI DEPLOYMENT PIPELINE")
    logger.info("="*80)
    
    try:
        # Convert to GGUF
        gguf_file = convert_to_gguf()
        
        # Create Modelfile
        modelfile = create_ollama_modelfile(gguf_file)
        
        # Deploy to Ollama (optional)
        deploy_to_ollama(modelfile)
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL STEPS COMPLETE!")
        logger.info("="*80)
        logger.info(f"\n📦 GGUF Model: {gguf_file}")
        logger.info(f"📝 Modelfile: {modelfile}")
        logger.info("\n🎯 Your MillennialAI is ready!")
        
    except Exception as e:
        logger.error(f"\n❌ Deployment failed: {e}", exc_info=True)
        logger.info("\n💡 Troubleshooting:")
        logger.info("  1. Make sure merge completed: python hybrid_merge.py")
        logger.info("  2. Check merged model exists in: models/hybrid_13b/merged/")
        logger.info("  3. Ensure git is installed (for llama.cpp)")


if __name__ == "__main__":
    main()
