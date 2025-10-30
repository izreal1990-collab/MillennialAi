"""
Configuration system for MillenialAi hybrid architecture

This module provides configuration classes for the hybrid TRM-LLM architecture,
allowing fine-grained control over layer injection, model parameters, and
training settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import warnings


@dataclass
class HybridConfig:
    """
    Configuration class for hybrid TRM-LLM architecture
    
    This class defines all configurable parameters for the layer injection
    mechanism, TRM architecture, and training settings.
    
    Args:
        injection_layers: List of LLM layer indices where TRM injection should occur
        trm_hidden_size: Hidden dimension of the TRM blocks
        trm_num_heads: Number of attention heads in TRM blocks
        trm_ff_hidden_size: Feedforward dimension in TRM blocks
        trm_num_layers: Number of transformer layers in each TRM block
        num_recursion_steps: Number of recursive processing steps in TRM
        dropout: Dropout rate for TRM components
        layer_norm_eps: Epsilon for layer normalization
        projection_bias: Whether to use bias in projection layers
        projection_activation: Activation function for projection layers
        gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
        mixed_precision: Whether to enable mixed precision training
        injection_strength: Global injection strength (0.0 to 1.0)
        adaptive_injection: Whether to enable adaptive injection based on hidden states
        blending_strategy: Strategy for blending original and injected hidden states
    
    Example:
        >>> config = HybridConfig(
        ...     injection_layers=[4, 8, 12],
        ...     trm_hidden_size=768,
        ...     trm_num_heads=12,
        ...     adaptive_injection=True
        ... )
    """
    
    # Layer injection settings
    injection_layers: List[int] = field(default_factory=lambda: [4, 8])
    injection_strength: float = 1.0
    adaptive_injection: bool = False
    blending_strategy: str = "linear"  # "linear", "attention_weighted", "gated"
    
    # TRM architecture settings
    trm_hidden_size: int = 512
    trm_num_heads: int = 8
    trm_ff_hidden_size: int = 2048
    trm_num_layers: int = 2
    num_recursion_steps: int = 3
    
    # Regularization settings
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    recursion_dropout: float = 0.05
    
    # Projection layer settings
    projection_bias: bool = True
    projection_activation: str = "gelu"
    
    # Training and optimization settings
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    
    # Advanced settings
    custom_trm_configs: Optional[Dict[int, Dict[str, Any]]] = None
    injection_schedule: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration parameters after initialization"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate injection layers
        if not isinstance(self.injection_layers, list):
            raise TypeError("injection_layers must be a list of integers")
        
        if not all(isinstance(layer, int) and layer >= 0 for layer in self.injection_layers):
            raise ValueError("All injection_layers must be non-negative integers")
        
        # Validate TRM architecture
        if self.trm_hidden_size <= 0:
            raise ValueError("trm_hidden_size must be positive")
        
        if self.trm_num_heads <= 0:
            raise ValueError("trm_num_heads must be positive")
        
        if self.trm_hidden_size % self.trm_num_heads != 0:
            raise ValueError("trm_hidden_size must be divisible by trm_num_heads")
        
        if self.trm_ff_hidden_size <= 0:
            raise ValueError("trm_ff_hidden_size must be positive")
        
        if self.num_recursion_steps <= 0:
            raise ValueError("num_recursion_steps must be positive")
        
        # Validate dropout rates
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")
        
        if not 0.0 <= self.recursion_dropout <= 1.0:
            raise ValueError("recursion_dropout must be between 0.0 and 1.0")
        
        # Validate injection strength
        if not 0.0 <= self.injection_strength <= 1.0:
            raise ValueError("injection_strength must be between 0.0 and 1.0")
        
        # Validate blending strategy
        valid_strategies = ["linear", "attention_weighted", "gated"]
        if self.blending_strategy not in valid_strategies:
            raise ValueError(f"blending_strategy must be one of {valid_strategies}")
        
        # Validate activation function
        valid_activations = ["relu", "gelu", "tanh", "sigmoid", "swish"]
        if self.projection_activation not in valid_activations:
            warnings.warn(f"projection_activation '{self.projection_activation}' not in "
                         f"validated list {valid_activations}. Proceeding anyway.")
    
    def get_trm_config_for_layer(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get TRM configuration for a specific layer
        
        Args:
            layer_idx: Layer index to get configuration for
            
        Returns:
            Dictionary containing TRM configuration for the specified layer
        """
        if self.custom_trm_configs and layer_idx in self.custom_trm_configs:
            # Use custom configuration for this layer
            base_config = {
                'hidden_size': self.trm_hidden_size,
                'num_heads': self.trm_num_heads,
                'ff_hidden_size': self.trm_ff_hidden_size,
                'num_layers': self.trm_num_layers,
                'num_recursion_steps': self.num_recursion_steps,
                'dropout': self.dropout,
                'layer_norm_eps': self.layer_norm_eps,
            }
            
            # Update with custom settings
            base_config.update(self.custom_trm_configs[layer_idx])
            return base_config
        else:
            # Use default configuration
            return {
                'hidden_size': self.trm_hidden_size,
                'num_heads': self.trm_num_heads,
                'ff_hidden_size': self.trm_ff_hidden_size,
                'num_layers': self.trm_num_layers,
                'num_recursion_steps': self.num_recursion_steps,
                'dropout': self.dropout,
                'layer_norm_eps': self.layer_norm_eps,
            }
    
    def set_custom_trm_config(self, layer_idx: int, config: Dict[str, Any]):
        """
        Set custom TRM configuration for a specific layer
        
        Args:
            layer_idx: Layer index to configure
            config: Dictionary of configuration parameters to override
        """
        if self.custom_trm_configs is None:
            self.custom_trm_configs = {}
        
        self.custom_trm_configs[layer_idx] = config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'injection_layers': self.injection_layers,
            'injection_strength': self.injection_strength,
            'adaptive_injection': self.adaptive_injection,
            'blending_strategy': self.blending_strategy,
            'trm_hidden_size': self.trm_hidden_size,
            'trm_num_heads': self.trm_num_heads,
            'trm_ff_hidden_size': self.trm_ff_hidden_size,
            'trm_num_layers': self.trm_num_layers,
            'num_recursion_steps': self.num_recursion_steps,
            'dropout': self.dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'recursion_dropout': self.recursion_dropout,
            'projection_bias': self.projection_bias,
            'projection_activation': self.projection_activation,
            'gradient_checkpointing': self.gradient_checkpointing,
            'mixed_precision': self.mixed_precision,
            'custom_trm_configs': self.custom_trm_configs,
            'injection_schedule': self.injection_schedule,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def copy(self) -> 'HybridConfig':
        """Create a copy of the configuration"""
        return HybridConfig.from_dict(self.to_dict())
    
    def __repr__(self) -> str:
        """String representation of the configuration"""
        return (f"HybridConfig("
                f"injection_layers={self.injection_layers}, "
                f"trm_hidden_size={self.trm_hidden_size}, "
                f"trm_num_heads={self.trm_num_heads}, "
                f"adaptive_injection={self.adaptive_injection})")


# Predefined configurations for common use cases
class PresetConfigs:
    """Predefined configurations for common use cases"""
    
    @staticmethod
    def gpt2_small() -> HybridConfig:
        """Configuration optimized for GPT-2 Small (124M parameters)"""
        return HybridConfig(
            injection_layers=[3, 6, 9],
            trm_hidden_size=768,
            trm_num_heads=12,
            trm_ff_hidden_size=3072,
            trm_num_layers=2,
            num_recursion_steps=3,
            dropout=0.1,
        )
    
    @staticmethod
    def gpt2_medium() -> HybridConfig:
        """Configuration optimized for GPT-2 Medium (355M parameters)"""
        return HybridConfig(
            injection_layers=[4, 8, 12, 16],
            trm_hidden_size=1024,
            trm_num_heads=16,
            trm_ff_hidden_size=4096,
            trm_num_layers=2,
            num_recursion_steps=4,
            dropout=0.1,
        )
    
    @staticmethod
    def bert_base() -> HybridConfig:
        """Configuration optimized for BERT Base (110M parameters)"""
        return HybridConfig(
            injection_layers=[3, 6, 9],
            trm_hidden_size=768,
            trm_num_heads=12,
            trm_ff_hidden_size=3072,
            trm_num_layers=2,
            num_recursion_steps=3,
            dropout=0.1,
            adaptive_injection=True,
        )
    
    @staticmethod
    def minimal() -> HybridConfig:
        """Minimal configuration for testing and development"""
        return HybridConfig(
            injection_layers=[1],
            trm_hidden_size=256,
            trm_num_heads=4,
            trm_ff_hidden_size=1024,
            trm_num_layers=1,
            num_recursion_steps=2,
            dropout=0.0,
        )
    
    @staticmethod
    def adaptive_multi_scale() -> HybridConfig:
        """Advanced configuration with adaptive injection and multi-scale processing"""
        config = HybridConfig(
            injection_layers=[2, 4, 6, 8, 10],
            trm_hidden_size=768,
            trm_num_heads=12,
            trm_ff_hidden_size=3072,
            trm_num_layers=2,
            num_recursion_steps=4,
            dropout=0.1,
            adaptive_injection=True,
            blending_strategy="attention_weighted",
        )
        
        # Custom configurations for different layers
        config.set_custom_trm_config(2, {'num_recursion_steps': 2})  # Early layer: less recursion
        config.set_custom_trm_config(10, {'num_recursion_steps': 6})  # Late layer: more recursion
        
        return config