"""
MillenialAi: Enterprise-Grade Layer Injection Architecture

Revolutionary hybrid neural networks for 70B+ parameter models with seamless
TRM injection using PyTorch forward hooks. Designed for enterprise AI deployment.

This package enables the creation of hybrid models with 70-90+ billion parameters,
combining the power of large language models with revolutionary TRM processing.

ENTERPRISE FEATURES:
- 70B+ parameter model support (LLaMA-2/3, GPT-4 scale)
- Zero-modification layer injection via forward hooks
- Enterprise-grade distributed training support
- Production-optimized configurations
- Multi-GPU and memory-efficient processing
- Real-time injection activation/deactivation

SCALE TARGETS:
- LLaMA-2-70B + 15B TRM = 85B hybrid model
- LLaMA-3-70B + 20B TRM = 90B hybrid model  
- GPT-4-scale + 300B TRM = 2T+ hybrid model

Example (Enterprise):
    >>> from millennial_ai import CombinedTRMLLM, HybridConfig
    >>> from transformers import AutoModelForCausalLM
    >>> 
    >>> # Load 70B parameter model
    >>> llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
    >>> 
    >>> # Configure enterprise injection (85B total)
    >>> config = HybridConfig.from_preset('llama2-70b')
    >>> 
    >>> # Create hybrid model
    >>> hybrid = CombinedTRMLLM(llm_model=llm, config=config)
    >>> 
    >>> # Activate enterprise layer injection
    >>> hybrid.activate_injection()  # Now running 85B parameters!
"""

__version__ = "1.0.0"
__author__ = "Jovan Blango"
__email__ = "izreal1990@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/izreal1990-collab/MillenialAi"

# Core enterprise components
from millennial_ai.core.hybrid_model import CombinedTRMLLM, create_hybrid_model
from millennial_ai.config.config import HybridConfig, PresetConfigs
from millennial_ai.models.hybrid_trm import HybridTRMBlock
from millennial_ai.models.projection import DimensionalBridge, create_dimensional_bridge

# Enterprise presets for quick access
ENTERPRISE_PRESETS = {
    'llama2-70b': 'LLaMA-2-70B with 15B TRM injection (85B total)',
    'llama3-70b': 'LLaMA-3-70B with 20B TRM injection (90B total)', 
    'gpt4-scale': 'GPT-4 scale with 300B TRM injection (2T+ total)',
    'multimodal': 'Multimodal foundation with 25B TRM injection',
    'production': 'Production optimized (78B total)',
    'research': 'Experimental maximum capability configuration'
}

__all__ = [
    # Core classes
    "CombinedTRMLLM",
    "create_hybrid_model",
    "HybridConfig", 
    "PresetConfigs",
    "HybridTRMBlock",
    "DimensionalBridge",
    "create_dimensional_bridge",
    # Enterprise presets
    "ENTERPRISE_PRESETS",
]

def get_enterprise_info():
    """Get enterprise deployment information"""
    return {
        'version': __version__,
        'supported_models': ['LLaMA-2-70B', 'LLaMA-3-70B', 'GPT-4-scale', 'Custom 70B+'],
        'parameter_ranges': '70B - 2T+ parameters',
        'hardware_requirements': '8x A100 80GB minimum, 16x H100 recommended',
        'memory_optimization': 'Gradient checkpointing, mixed precision, model sharding',
        'enterprise_features': [
            'Multi-GPU distributed training',
            'Production monitoring',
            'Dynamic batch sizing', 
            'Zero-downtime injection toggling',
            'Enterprise SLA support'
        ]
    }

def quick_start_enterprise():
    """Show quick start for enterprise deployment"""
    print("🏢 MillenialAi Enterprise Quick Start")
    print("=" * 40)
    print("1. Load your 70B+ parameter model:")
    print("   model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-70b-hf')")
    print()
    print("2. Configure enterprise injection:")
    print("   config = HybridConfig.from_preset('llama2-70b')")
    print()
    print("3. Create hybrid model:")
    print("   hybrid = CombinedTRMLLM(llm_model=model, config=config)")
    print()
    print("4. Activate injection:")
    print("   hybrid.activate_injection()")
    print()
    print("5. Deploy at enterprise scale:")
    print("   outputs = hybrid(inputs)  # 85B parameter hybrid processing")
    print()
    print(f"📋 Available presets: {list(ENTERPRISE_PRESETS.keys())}")
    print(f"📞 Enterprise support: izreal1990@gmail.com")