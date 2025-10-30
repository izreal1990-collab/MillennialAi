"""
Main Hybrid Model Implementation

This module contains the CombinedTRMLLM class, which implements the revolutionary
Layer Injection Architecture using PyTorch forward hooks to seamlessly integrate
Tiny Recursion Models into Large Language Models.

Key Features:
- Zero-modification LLM integration via forward hooks
- Dynamic injection activation/deactivation
- Gradient-preserving hybrid architecture
- Multi-layer injection support
- HuggingFace transformer compatibility
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union
import warnings
import logging

from ..models.hybrid_trm import HybridTRMBlock
from ..models.projection import DimensionalBridge, create_dimensional_bridge
from ..config.config import HybridConfig

logger = logging.getLogger(__name__)


class CombinedTRMLLM(nn.Module):
    """
    Combined TRM-LLM Model with Layer Injection Architecture
    
    This is the main class that implements the revolutionary layer injection
    mechanism. It wraps any existing LLM and injects TRM processing at
    specified layers using PyTorch forward hooks.
    
    The key innovation is that the LLM remains completely unchanged while
    TRM processing is seamlessly injected into the forward pass.
    
    Args:
        llm_model: The base LLM model (any PyTorch transformer)
        config: HybridConfig instance with injection parameters
        
    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> from millennial_ai.core.hybrid_model import CombinedTRMLLM
        >>> from millennial_ai.config.config import HybridConfig
        >>> 
        >>> # Load base model
        >>> llm = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> 
        >>> # Configure injection
        >>> config = HybridConfig(injection_layers=[4, 8])
        >>> 
        >>> # Create hybrid model
        >>> hybrid = CombinedTRMLLM(llm_model=llm, config=config)
        >>> 
        >>> # Activate injection
        >>> hybrid.activate_injection()
        >>> 
        >>> # Use normally
        >>> outputs = hybrid(input_ids)
    """
    
    def __init__(self, llm_model: nn.Module, config: HybridConfig):
        super().__init__()
        
        self.llm_model = llm_model
        self.config = config
        self.injection_active = False
        
        # Detect LLM hidden size
        self.llm_hidden_size = self._detect_llm_hidden_size()
        
        # Create TRM block
        self.trm_block = HybridTRMBlock(
            hidden_size=config.trm_hidden_size,
            num_heads=config.trm_num_heads,
            ff_hidden_size=config.trm_ff_hidden_size,
            num_layers=config.trm_num_layers,
            num_recursion_steps=config.num_recursion_steps,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps
        )
        
        # Create dimensional bridge
        self.projection = create_dimensional_bridge(
            llm_hidden_size=self.llm_hidden_size,
            trm_hidden_size=config.trm_hidden_size,
            projection_type='adaptive' if config.adaptive_injection else 'linear',
            bias=config.projection_bias,
            activation=config.projection_activation,
            dropout=config.dropout
        )
        
        # Hook management
        self.hooks: Dict[int, Any] = {}
        self.injection_statistics = {
            'total_injections': 0,
            'layer_counts': {layer: 0 for layer in config.injection_layers}
        }
        
        # Validate injection layers
        self._validate_injection_layers()
        
        logger.info(f"Created CombinedTRMLLM with injection at layers {config.injection_layers}")
    
    def _detect_llm_hidden_size(self) -> int:
        """Automatically detect the hidden size of the LLM"""
        # Common ways to detect hidden size
        if hasattr(self.llm_model, 'config') and hasattr(self.llm_model.config, 'hidden_size'):
            return self.llm_model.config.hidden_size
        elif hasattr(self.llm_model, 'config') and hasattr(self.llm_model.config, 'd_model'):
            return self.llm_model.config.d_model
        elif hasattr(self.llm_model, 'hidden_size'):
            return self.llm_model.hidden_size
        elif hasattr(self.llm_model, 'd_model'):
            return self.llm_model.d_model
        else:
            # Try to infer from a sample forward pass
            try:
                with torch.no_grad():
                    sample_input = torch.randint(0, 1000, (1, 10))
                    if hasattr(self.llm_model, 'transformer'):
                        # GPT-style models
                        embedding = self.llm_model.transformer.wte(sample_input)
                        return embedding.shape[-1]
                    elif hasattr(self.llm_model, 'embeddings'):
                        # BERT-style models
                        embedding = self.llm_model.embeddings.word_embeddings(sample_input)
                        return embedding.shape[-1]
                    else:
                        raise RuntimeError("Could not detect LLM hidden size")
            except Exception as e:
                raise RuntimeError(f"Could not detect LLM hidden size: {e}")
    
    def _validate_injection_layers(self):
        """Validate that injection layers exist in the LLM"""
        # This is a basic validation - in practice, you might want more sophisticated checking
        max_layer = max(self.config.injection_layers) if self.config.injection_layers else 0
        
        # Try to count layers in common architectures
        layer_count = 0
        if hasattr(self.llm_model, 'transformer') and hasattr(self.llm_model.transformer, 'h'):
            layer_count = len(self.llm_model.transformer.h)  # GPT-style
        elif hasattr(self.llm_model, 'encoder') and hasattr(self.llm_model.encoder, 'layer'):
            layer_count = len(self.llm_model.encoder.layer)  # BERT-style
        elif hasattr(self.llm_model, 'layers'):
            layer_count = len(self.llm_model.layers)  # Generic
        
        if layer_count > 0 and max_layer >= layer_count:
            warnings.warn(f"Injection layer {max_layer} may not exist (detected {layer_count} layers)")
    
    def _get_injection_target_modules(self) -> Dict[int, nn.Module]:
        """Get the actual modules to inject hooks into"""
        targets = {}
        
        for layer_idx in self.config.injection_layers:
            target_module = None
            
            # Try different common architectures
            if hasattr(self.llm_model, 'transformer') and hasattr(self.llm_model.transformer, 'h'):
                # GPT-style models
                if layer_idx < len(self.llm_model.transformer.h):
                    target_module = self.llm_model.transformer.h[layer_idx]
            
            elif hasattr(self.llm_model, 'encoder') and hasattr(self.llm_model.encoder, 'layer'):
                # BERT-style models
                if layer_idx < len(self.llm_model.encoder.layer):
                    target_module = self.llm_model.encoder.layer[layer_idx]
            
            elif hasattr(self.llm_model, 'layers'):
                # Generic transformer
                if layer_idx < len(self.llm_model.layers):
                    target_module = self.llm_model.layers[layer_idx]
            
            if target_module is not None:
                targets[layer_idx] = target_module
            else:
                warnings.warn(f"Could not find target module for layer {layer_idx}")
        
        return targets
    
    def _trm_injection_hook(self, layer_idx: int):
        """
        Create the injection hook function for a specific layer
        
        This is the core of the layer injection mechanism. The hook
        intercepts the output of an LLM layer and injects TRM processing.
        """
        def hook_fn(module, input_tuple, output):
            if not self.injection_active:
                return output
            
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_outputs = output[1:]
            else:
                hidden_states = output
                rest_outputs = ()
            
            # Ensure we have the right tensor
            if not isinstance(hidden_states, torch.Tensor):
                return output
            
            # Project to TRM space
            try:
                trm_input = self.projection.project_to_trm(hidden_states)
                
                # Apply TRM processing
                trm_output = self.trm_block(trm_input)
                
                # Project back to LLM space
                enhanced_hidden = self.projection.project_to_llm(trm_output)
                
                # Update statistics
                self.injection_statistics['total_injections'] += 1
                self.injection_statistics['layer_counts'][layer_idx] += 1
                
                # Return in the same format as input
                if rest_outputs:
                    return (enhanced_hidden,) + rest_outputs
                else:
                    return enhanced_hidden
                    
            except Exception as e:
                logger.warning(f"Injection failed at layer {layer_idx}: {e}")
                return output
        
        return hook_fn
    
    def activate_injection(self):
        """Activate layer injection by registering forward hooks"""
        if self.injection_active:
            logger.warning("Injection is already active")
            return
        
        target_modules = self._get_injection_target_modules()
        
        for layer_idx, module in target_modules.items():
            hook = module.register_forward_hook(self._trm_injection_hook(layer_idx))
            self.hooks[layer_idx] = hook
        
        self.injection_active = True
        logger.info(f"Activated injection on {len(self.hooks)} layers")
    
    def deactivate_injection(self):
        """Deactivate layer injection by removing forward hooks"""
        for hook in self.hooks.values():
            hook.remove()
        
        self.hooks.clear()
        self.injection_active = False
        logger.info("Deactivated layer injection")
    
    def toggle_injection(self):
        """Toggle injection on/off"""
        if self.injection_active:
            self.deactivate_injection()
        else:
            self.activate_injection()
    
    def get_injection_statistics(self) -> Dict[str, Any]:
        """Get statistics about injection usage"""
        return {
            'active': self.injection_active,
            'total_injections': self.injection_statistics['total_injections'],
            'layer_counts': self.injection_statistics['layer_counts'].copy(),
            'average_per_layer': (
                self.injection_statistics['total_injections'] / len(self.config.injection_layers)
                if self.config.injection_layers else 0
            )
        }
    
    def reset_injection_statistics(self):
        """Reset injection statistics"""
        self.injection_statistics = {
            'total_injections': 0,
            'layer_counts': {layer: 0 for layer in self.config.injection_layers}
        }
    
    def forward(self, *args, **kwargs):
        """Forward pass through the hybrid model"""
        return self.llm_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying LLM model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm_model, name)
    
    def __call__(self, *args, **kwargs):
        """Make the model callable"""
        return self.forward(*args, **kwargs)
    
    def parameters(self, recurse: bool = True):
        """Return parameters from both LLM and TRM components"""
        for param in self.llm_model.parameters(recurse):
            yield param
        for param in self.trm_block.parameters(recurse):
            yield param
        for param in self.projection.parameters(recurse):
            yield param
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters from both components"""
        for name, param in self.llm_model.named_parameters(prefix='llm_model.' + prefix, recurse=recurse):
            yield name, param
        for name, param in self.trm_block.named_parameters(prefix='trm_block.' + prefix, recurse=recurse):
            yield name, param
        for name, param in self.projection.named_parameters(prefix='projection.' + prefix, recurse=recurse):
            yield name, param
    
    def train(self, mode: bool = True):
        """Set training mode for all components"""
        super().train(mode)
        self.llm_model.train(mode)
        self.trm_block.train(mode)
        self.projection.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode for all components"""
        return self.train(False)
    
    def to(self, *args, **kwargs):
        """Move all components to specified device/dtype"""
        super().to(*args, **kwargs)
        self.llm_model.to(*args, **kwargs)
        self.trm_block.to(*args, **kwargs)
        self.projection.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        """Move to CUDA"""
        return self.to(device=device if device is not None else 'cuda')
    
    def cpu(self):
        """Move to CPU"""
        return self.to(device='cpu')
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count breakdown"""
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())
        
        llm_params = count_parameters(self.llm_model)
        trm_params = count_parameters(self.trm_block)
        projection_params = count_parameters(self.projection)
        
        return {
            'total': llm_params + trm_params + projection_params,
            'llm_model': llm_params,
            'trm_block': trm_params,
            'projection': projection_params,
            'overhead_params': trm_params + projection_params,
            'overhead_percentage': ((trm_params + projection_params) / llm_params * 100) if llm_params > 0 else 0
        }
    
    def save_pretrained(self, save_directory: str):
        """Save the hybrid model components"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, 'hybrid_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save TRM block
        trm_path = os.path.join(save_directory, 'trm_block.pt')
        torch.save(self.trm_block.state_dict(), trm_path)
        
        # Save projection
        projection_path = os.path.join(save_directory, 'projection.pt')
        torch.save(self.projection.state_dict(), projection_path)
        
        # Save base LLM if it has save_pretrained method
        if hasattr(self.llm_model, 'save_pretrained'):
            llm_path = os.path.join(save_directory, 'llm_model')
            self.llm_model.save_pretrained(llm_path)
        else:
            # Fallback: save as PyTorch state dict
            llm_path = os.path.join(save_directory, 'llm_model.pt')
            torch.save(self.llm_model.state_dict(), llm_path)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, llm_model: Optional[nn.Module] = None):
        """Load a hybrid model from saved components"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(load_directory, 'hybrid_config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = HybridConfig.from_dict(config_dict)
        
        # Load or use provided LLM model
        if llm_model is None:
            llm_path = os.path.join(load_directory, 'llm_model')
            if os.path.exists(llm_path):
                # Try to load using transformers
                try:
                    from transformers import AutoModel
                    llm_model = AutoModel.from_pretrained(llm_path)
                except ImportError:
                    raise RuntimeError("transformers library required to load LLM model")
            else:
                raise RuntimeError("No LLM model provided and none found in save directory")
        
        # Create hybrid model
        hybrid = cls(llm_model=llm_model, config=config)
        
        # Load TRM block
        trm_path = os.path.join(load_directory, 'trm_block.pt')
        if os.path.exists(trm_path):
            hybrid.trm_block.load_state_dict(torch.load(trm_path, map_location='cpu'))
        
        # Load projection
        projection_path = os.path.join(load_directory, 'projection.pt')
        if os.path.exists(projection_path):
            hybrid.projection.load_state_dict(torch.load(projection_path, map_location='cpu'))
        
        return hybrid
    
    def __repr__(self):
        """String representation of the hybrid model"""
        param_counts = self.get_parameter_count()
        return (f"CombinedTRMLLM(\n"
                f"  llm_model={type(self.llm_model).__name__}({param_counts['llm_model']:,} params),\n"
                f"  trm_block=HybridTRMBlock({param_counts['trm_block']:,} params),\n"
                f"  projection={type(self.projection).__name__}({param_counts['projection']:,} params),\n"
                f"  injection_layers={self.config.injection_layers},\n"
                f"  injection_active={self.injection_active},\n"
                f"  total_params={param_counts['total']:,},\n"
                f"  overhead={param_counts['overhead_percentage']:.1f}%\n"
                f")")


def create_hybrid_model(llm_model: nn.Module, **config_kwargs) -> CombinedTRMLLM:
    """
    Convenience function to create a hybrid model with default configuration
    
    Args:
        llm_model: The base LLM model
        **config_kwargs: Configuration arguments for HybridConfig
        
    Returns:
        CombinedTRMLLM instance
    """
    config = HybridConfig(**config_kwargs)
    return CombinedTRMLLM(llm_model=llm_model, config=config)