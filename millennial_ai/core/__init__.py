"""
Core module for MillenialAi Layer Injection Architecture

This module contains the main components for implementing hybrid TRM-LLM models:
- CombinedTRMLLM: Main hybrid model class
- Configuration management
- Utility functions
"""

__version__ = "1.0.0"

from .hybrid_model import CombinedTRMLLM, create_hybrid_model

__all__ = [
    'CombinedTRMLLM',
    'create_hybrid_model',
]