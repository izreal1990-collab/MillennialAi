"""
Dimensional Projection Layers for MillennialAi

This module implements projection layers that enable dimensional compatibility
between Large Language Models and Tiny Recursion Models with different hidden sizes.

Key Features:
- Bi-directional projection (LLM ↔ TRM)
- Learnable dimensional bridges
- Gradient-preserving transformations
- Adaptive projection strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


class ProjectionLayer(nn.Module):
    """
    Learnable projection layer with optional activation
    
    Projects from one dimensional space to another while preserving
    gradient flow and semantic information.
    """
    
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, 
                 activation: Optional[str] = None, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main projection
        self.projection = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Optional activation
        self.activation = None
        if activation is not None:
            if activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'swish' or activation == 'silu':
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward projection"""
        x = self.projection(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class AdaptiveProjection(nn.Module):
    """
    Adaptive projection that adjusts based on input characteristics
    
    Uses attention-like mechanisms to determine optimal projection
    weights based on the input hidden states.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        
        # Expert projections
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=False)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize expert and gate weights"""
        for expert in self.experts:
            nn.init.xavier_uniform_(expert.weight)
        
        # Initialize gate to uniform distribution
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adaptive forward projection"""
        batch_size, seq_len, _ = x.shape
        
        # Compute gating weights
        gate_weights = self.gate(x)  # [batch, seq_len, num_experts]
        
        # Apply each expert
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [batch, seq_len, output_dim]
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [batch, seq_len, num_experts, output_dim]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # [batch, seq_len, num_experts, 1]
        output = torch.sum(gate_weights * expert_outputs, dim=-2)  # [batch, seq_len, output_dim]
        
        return output


class DimensionalBridge(nn.Module):
    """
    Bidirectional dimensional bridge between LLM and TRM spaces
    
    This is the main projection module that handles dimensional compatibility
    between different model architectures. It supports both simple linear
    projections and more sophisticated adaptive projections.
    
    Args:
        llm_hidden_size: Hidden size of the LLM
        trm_hidden_size: Hidden size of the TRM
        projection_type: Type of projection ('linear', 'adaptive', 'residual')
        bias: Whether to use bias in projections
        activation: Activation function for projections
        dropout: Dropout rate for projections
        adaptive_experts: Number of experts for adaptive projection
    
    Example:
        >>> bridge = DimensionalBridge(
        ...     llm_hidden_size=768,
        ...     trm_hidden_size=512,
        ...     projection_type='adaptive'
        ... )
        >>> 
        >>> # Project LLM states to TRM space
        >>> trm_states = bridge.project_to_trm(llm_hidden_states)
        >>> 
        >>> # Project back to LLM space
        >>> llm_states = bridge.project_to_llm(trm_states)
    """
    
    def __init__(self, llm_hidden_size: int, trm_hidden_size: int,
                 projection_type: str = 'linear', bias: bool = True,
                 activation: Optional[str] = 'gelu', dropout: float = 0.0,
                 adaptive_experts: int = 4):
        super().__init__()
        
        self.llm_hidden_size = llm_hidden_size
        self.trm_hidden_size = trm_hidden_size
        self.projection_type = projection_type
        
        # Create forward projection (LLM -> TRM)
        if projection_type == 'linear':
            self.llm_to_trm = ProjectionLayer(
                llm_hidden_size, trm_hidden_size, bias, activation, dropout
            )
        elif projection_type == 'adaptive':
            self.llm_to_trm = AdaptiveProjection(
                llm_hidden_size, trm_hidden_size, adaptive_experts
            )
        elif projection_type == 'residual':
            self.llm_to_trm = self._create_residual_projection(
                llm_hidden_size, trm_hidden_size, bias, activation, dropout
            )
        else:
            raise ValueError(f"Unsupported projection_type: {projection_type}")
        
        # Create reverse projection (TRM -> LLM)
        if projection_type == 'linear':
            self.trm_to_llm = ProjectionLayer(
                trm_hidden_size, llm_hidden_size, bias, activation, dropout
            )
        elif projection_type == 'adaptive':
            self.trm_to_llm = AdaptiveProjection(
                trm_hidden_size, llm_hidden_size, adaptive_experts
            )
        elif projection_type == 'residual':
            self.trm_to_llm = self._create_residual_projection(
                trm_hidden_size, llm_hidden_size, bias, activation, dropout
            )
        
        # Layer norms for stability
        self.llm_norm = nn.LayerNorm(llm_hidden_size)
        self.trm_norm = nn.LayerNorm(trm_hidden_size)
        
        # Optional learnable scaling factors
        self.llm_to_trm_scale = nn.Parameter(torch.ones(1))
        self.trm_to_llm_scale = nn.Parameter(torch.ones(1))
    
    def _create_residual_projection(self, input_dim: int, output_dim: int,
                                  bias: bool, activation: Optional[str], dropout: float) -> nn.Module:
        """Create a residual projection with skip connection where possible"""
        if input_dim == output_dim:
            # Same dimensions - use identity with optional processing
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                ProjectionLayer(input_dim, output_dim, bias, activation, dropout)
            )
        else:
            # Different dimensions - use bottleneck architecture
            bottleneck_dim = min(input_dim, output_dim)
            return nn.Sequential(
                ProjectionLayer(input_dim, bottleneck_dim, bias, activation, dropout),
                ProjectionLayer(bottleneck_dim, output_dim, bias, activation, dropout)
            )
    
    def project_to_trm(self, llm_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project LLM hidden states to TRM dimensional space
        
        Args:
            llm_hidden_states: Hidden states from LLM [batch, seq_len, llm_hidden_size]
            
        Returns:
            Projected states [batch, seq_len, trm_hidden_size]
        """
        # Normalize input
        normalized = self.llm_norm(llm_hidden_states)
        
        # Apply projection
        projected = self.llm_to_trm(normalized)
        
        # Apply learnable scaling
        projected = projected * self.llm_to_trm_scale
        
        return projected
    
    def project_to_llm(self, trm_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project TRM hidden states back to LLM dimensional space
        
        Args:
            trm_hidden_states: Hidden states from TRM [batch, seq_len, trm_hidden_size]
            
        Returns:
            Projected states [batch, seq_len, llm_hidden_size]
        """
        # Normalize input
        normalized = self.trm_norm(trm_hidden_states)
        
        # Apply projection
        projected = self.trm_to_llm(normalized)
        
        # Apply learnable scaling
        projected = projected * self.trm_to_llm_scale
        
        return projected
    
    def get_parameter_count(self) -> dict:
        """Get parameter count breakdown"""
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'total': count_parameters(self),
            'llm_to_trm': count_parameters(self.llm_to_trm),
            'trm_to_llm': count_parameters(self.trm_to_llm),
            'layer_norms': count_parameters(self.llm_norm) + count_parameters(self.trm_norm),
            'scaling_factors': 2,  # Two scalar parameters
        }
    
    def forward(self, hidden_states: torch.Tensor, direction: str = 'to_trm') -> torch.Tensor:
        """
        Convenience forward method
        
        Args:
            hidden_states: Input hidden states
            direction: Projection direction ('to_trm' or 'to_llm')
            
        Returns:
            Projected hidden states
        """
        if direction == 'to_trm':
            return self.project_to_trm(hidden_states)
        elif direction == 'to_llm':
            return self.project_to_llm(hidden_states)
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'to_trm' or 'to_llm'")


class IdentityBridge(nn.Module):
    """
    Identity bridge for when LLM and TRM have the same hidden size
    
    This is an optimized version that avoids unnecessary projections
    when dimensional compatibility is not needed.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Optional layer norms for stability
        self.norm = nn.LayerNorm(hidden_size)
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1))
    
    def project_to_trm(self, llm_hidden_states: torch.Tensor) -> torch.Tensor:
        """Identity projection with optional normalization"""
        return self.norm(llm_hidden_states) * self.scale
    
    def project_to_llm(self, trm_hidden_states: torch.Tensor) -> torch.Tensor:
        """Identity projection with optional normalization"""
        return self.norm(trm_hidden_states) * self.scale
    
    def forward(self, hidden_states: torch.Tensor, direction: str = 'to_trm') -> torch.Tensor:
        """Convenience forward method"""
        return self.norm(hidden_states) * self.scale
    
    def get_parameter_count(self) -> dict:
        """Get parameter count breakdown"""
        return {
            'total': self.hidden_size * 2 + 1,  # LayerNorm + scale
            'layer_norm': self.hidden_size * 2,
            'scaling_factor': 1,
        }


def create_dimensional_bridge(llm_hidden_size: int, trm_hidden_size: int, **kwargs) -> Union[DimensionalBridge, IdentityBridge]:
    """
    Factory function to create appropriate dimensional bridge
    
    Automatically chooses between DimensionalBridge and IdentityBridge
    based on the hidden sizes.
    
    Args:
        llm_hidden_size: Hidden size of the LLM
        trm_hidden_size: Hidden size of the TRM
        **kwargs: Additional arguments for DimensionalBridge
        
    Returns:
        Appropriate bridge instance
    """
    if llm_hidden_size == trm_hidden_size:
        return IdentityBridge(llm_hidden_size)
    else:
        return DimensionalBridge(llm_hidden_size, trm_hidden_size, **kwargs)