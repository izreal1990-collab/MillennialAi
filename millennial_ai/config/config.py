"""
Configuration system for MillennialAi hybrid architecture

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
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'HybridConfig':
        """Create configuration from enterprise preset"""
        if preset_name == 'minimal':
            return PresetConfigs.minimal()
        elif preset_name == 'llama2-70b':
            return PresetConfigs.llama_2_70b_enterprise()
        elif preset_name == 'llama3-70b':
            return PresetConfigs.llama_3_70b_revolutionary()
        elif preset_name == 'gpt4-scale':
            return PresetConfigs.gpt_4_scale_ultra()
        elif preset_name == 'multimodal':
            return PresetConfigs.multimodal_foundation()
        elif preset_name == 'research':
            return PresetConfigs.research_experimental()
        elif preset_name == 'production':
            return PresetConfigs.production_optimized()
        # Legacy support
        elif preset_name == 'gpt2':
            return PresetConfigs.minimal()  # Redirect to minimal
        elif preset_name == 'bert':
            return PresetConfigs.minimal()  # Redirect to minimal
        elif preset_name == 'adaptive':
            return PresetConfigs.production_optimized()  # Better alternative
        else:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: "
                           f"'llama2-70b', 'llama3-70b', 'gpt4-scale', 'multimodal', "
                           f"'research', 'production', 'minimal'")
    
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


# Predefined configurations for ENTERPRISE-SCALE use cases
class PresetConfigs:
    """ENTERPRISE-GRADE predefined configurations for massive language models"""
    
    @staticmethod
    def llama_2_70b_enterprise() -> HybridConfig:
        """
        ENTERPRISE-GRADE: LLaMA-2-70B with massive TRM injection
        
        Base model: 70B parameters
        TRM injection: ~15B parameters
        Total hybrid: ~85B parameters
        
        Designed for production deployment with distributed training
        """
        return HybridConfig(
            injection_layers=[8, 16, 24, 32, 40, 48, 56, 64, 72],  # 9 strategic injection points
            trm_hidden_size=8192,     # Massive 8K TRM hidden size
            trm_num_heads=64,         # 64 attention heads for rich representation
            trm_ff_hidden_size=32768, # 32K FF dimension for complex reasoning
            trm_num_layers=6,         # Deep TRM processing stack
            num_recursion_steps=8,    # Deep recursive processing
            dropout=0.05,             # Low dropout for enterprise stability
            recursion_dropout=0.1,
            adaptive_injection=True,
            injection_strength=0.8,   # Strong injection for maximum enhancement
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-6,
            gradient_checkpointing=True,  # Memory optimization for 70B+ models
            mixed_precision=True,         # FP16/BF16 for efficiency
        )
    
    @staticmethod
    def llama_3_70b_revolutionary() -> HybridConfig:
        """
        REVOLUTIONARY: LLaMA-3-70B with next-generation TRM injection
        
        Base model: 70B parameters
        TRM injection: ~20B parameters
        Total hybrid: ~90B parameters
        
        Next-generation architecture for cutting-edge AI capabilities
        """
        return HybridConfig(
            injection_layers=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80],  # 20 injection points
            trm_hidden_size=12288,    # Revolutionary 12K TRM hidden size
            trm_num_heads=96,         # 96 attention heads for ultra-rich representation
            trm_ff_hidden_size=49152, # 48K FF dimension for advanced reasoning
            trm_num_layers=8,         # Ultra-deep TRM processing
            num_recursion_steps=12,   # Maximum recursive depth
            dropout=0.03,             # Minimal dropout for maximum capability
            recursion_dropout=0.08,
            adaptive_injection=True,
            injection_strength=0.9,   # Near-maximum injection strength
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-7,
            gradient_checkpointing=True,
            mixed_precision=True,
        )
    
    @staticmethod
    def gpt_4_scale_ultra() -> HybridConfig:
        """
        ULTRA-SCALE: GPT-4 level hybrid architecture
        
        Base model: ~1.7T parameters (estimated)
        TRM injection: ~300B parameters
        Total hybrid: ~2T+ parameters
        
        Designed for the largest possible AI systems
        """
        return HybridConfig(
            injection_layers=list(range(10, 200, 6)),  # Every 6th layer from 10-200 (32 injection points)
            trm_hidden_size=16384,    # Massive 16K TRM hidden size
            trm_num_heads=128,        # 128 attention heads for ultimate representation
            trm_ff_hidden_size=65536, # 64K FF dimension for supreme reasoning
            trm_num_layers=12,        # Ultra-deep TRM processing stack
            num_recursion_steps=16,   # Maximum possible recursive processing
            dropout=0.02,             # Minimal dropout for stability at scale
            recursion_dropout=0.05,
            adaptive_injection=True,
            injection_strength=1.0,   # Maximum injection strength
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-8,
            gradient_checkpointing=True,
            mixed_precision=True,
        )
    
    @staticmethod
    def multimodal_foundation() -> HybridConfig:
        """
        MULTIMODAL FOUNDATION: Enterprise multimodal model configuration
        
        For vision-language-audio foundation models
        Base: 70B+ parameters
        TRM injection: ~25B parameters
        Total: 95B+ parameters
        """
        return HybridConfig(
            injection_layers=[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],  # 16 injection points
            trm_hidden_size=14336,    # 14K TRM hidden size for multimodal processing
            trm_num_heads=112,        # 112 attention heads for cross-modal attention
            trm_ff_hidden_size=57344, # 56K FF dimension for complex multimodal reasoning
            trm_num_layers=10,        # Deep TRM processing for multimodal fusion
            num_recursion_steps=14,   # Deep recursive processing for modality integration
            dropout=0.04,
            recursion_dropout=0.08,
            adaptive_injection=True,
            injection_strength=0.85,
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-7,
            gradient_checkpointing=True,
            mixed_precision=True,
        )
    
    @staticmethod
    def research_experimental() -> HybridConfig:
        """
        RESEARCH EXPERIMENTAL: Maximum capability research configuration
        
        For research into the limits of hybrid architectures
        WARNING: Requires massive computational resources
        """
        return HybridConfig(
            injection_layers=list(range(5, 250, 4)),  # Every 4th layer (61 injection points!)
            trm_hidden_size=20480,    # 20K TRM hidden size
            trm_num_heads=160,        # 160 attention heads
            trm_ff_hidden_size=81920, # 80K FF dimension
            trm_num_layers=16,        # Maximum TRM depth
            num_recursion_steps=20,   # Extreme recursive processing
            dropout=0.01,             # Minimal dropout
            recursion_dropout=0.03,
            adaptive_injection=True,
            injection_strength=1.0,   # Maximum injection
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-9,
            gradient_checkpointing=True,
            mixed_precision=True,
        )
    
    @staticmethod
    def production_optimized() -> HybridConfig:
        """
        PRODUCTION OPTIMIZED: Enterprise deployment configuration
        
        Balanced for performance and computational efficiency
        Base: 70B parameters
        TRM injection: ~8B parameters
        Total: ~78B parameters
        """
        return HybridConfig(
            injection_layers=[12, 24, 36, 48, 60],  # 5 strategic injection points
            trm_hidden_size=6144,     # 6K TRM hidden size
            trm_num_heads=48,         # 48 attention heads
            trm_ff_hidden_size=24576, # 24K FF dimension
            trm_num_layers=4,         # Moderate TRM depth for efficiency
            num_recursion_steps=6,    # Balanced recursive processing
            dropout=0.08,
            recursion_dropout=0.12,
            adaptive_injection=True,
            injection_strength=0.7,   # Balanced injection strength
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-6,
            gradient_checkpointing=True,
            mixed_precision=True,
        )
    
    # Legacy configurations for backward compatibility
    @staticmethod
    def minimal() -> HybridConfig:
        """Minimal configuration for testing (DEPRECATED - use production_optimized for real work)"""
        return HybridConfig(
            injection_layers=[32],     # Single middle layer
            trm_hidden_size=4096,     # Still substantial for enterprise
            trm_num_heads=32,
            trm_ff_hidden_size=16384,
            trm_num_layers=2,
            num_recursion_steps=4,
            dropout=0.1,
        )
    
    @staticmethod 
    def gpt2_small() -> HybridConfig:
        """DEPRECATED: Use llama_2_70b_enterprise() for enterprise work"""
        return PresetConfigs.minimal()
    
    @staticmethod
    def bert_base() -> HybridConfig:
        """DEPRECATED: Use llama_2_70b_enterprise() for enterprise work"""
        return PresetConfigs.minimal()