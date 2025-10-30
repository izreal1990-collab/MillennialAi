"""
MillenialAi: Revolutionary Layer Injection Architecture

Revolutionary breakthrough in neural architecture: The first implementation of direct 
layer injection between Tiny Recursion Models (TRM) and Large Language Models (LLM) 
using PyTorch forward hooks.

This package enables seamless integration of recursive reasoning capabilities into 
existing transformer models without modification of the base architecture.

Key Features:
- Zero-interference integration via PyTorch forward hooks
- Dynamic activation/deactivation during inference
- Gradient-preserving hybrid training
- Multi-scale recursive processing
- HuggingFace transformer compatibility

Example:
    >>> from millennial_ai.core.hybrid_model import CombinedTRMLLM
    >>> from millennial_ai.config.config import HybridConfig
    >>> from transformers import GPT2LMHeadModel
    >>> 
    >>> # Load any transformer model
    >>> llm = GPT2LMHeadModel.from_pretrained("gpt2")
    >>> 
    >>> # Configure layer injection
    >>> config = HybridConfig(injection_layers=[4, 8])
    >>> 
    >>> # Create hybrid model
    >>> hybrid = CombinedTRMLLM(llm_model=llm, config=config)
    >>> 
    >>> # Activate layer injection
    >>> hybrid.activate_injection()
"""

__version__ = "1.0.0"
__author__ = "Jovan Blango"
__email__ = "izreal1990-collab@github.com"
__license__ = "MIT"
__url__ = "https://github.com/izreal1990-collab/MillenialAi"

# Core imports for easy access
from millennial_ai.core.hybrid_model import CombinedTRMLLM
from millennial_ai.config.config import HybridConfig
from millennial_ai.models.hybrid_trm import HybridTRMBlock
from millennial_ai.models.projection import DimensionalBridge

__all__ = [
    "CombinedTRMLLM",
    "HybridConfig", 
    "HybridTRMBlock",
    "DimensionalBridge",
    "__version__",
    "__author__",
    "__license__",
]