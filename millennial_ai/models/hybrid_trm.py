"""
Hybrid TRM Block Implementation

This module implements the core Tiny Recursion Model (TRM) block designed specifically
for layer injection into Large Language Models. Unlike traditional TRM implementations,
this block has no input/output token embeddings - it processes hidden states directly.

Key Features:
- No token embeddings (works directly with hidden states)
- Recursive processing with configurable depth
- Multi-head attention with efficient implementation
- Designed for gradient-preserving injection into LLM layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for TRM blocks
    
    This provides position information without requiring absolute position embeddings,
    making it ideal for injection into pre-trained models.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values if sequence length changed"""
        if seq_len > self._cached_seq_len:
            seq = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(seq, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
            self._cached_seq_len = seq_len
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to query and key tensors"""
        self._update_cache(seq_len, q.device, q.dtype)
        
        if self._cached_cos is None or self._cached_sin is None:
            raise RuntimeError("Cache not initialized")
        
        cached_cos = self._cached_cos
        cached_sin = self._cached_sin
        
        def apply_rotary_pos_emb(x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, seq_len, heads, dim]
            x1, x2 = x[..., ::2], x[..., 1::2]
            cos, sin = cached_cos[:seq_len], cached_sin[:seq_len]
            
            # Expand for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim//2]
            
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        return apply_rotary_pos_emb(q), apply_rotary_pos_emb(k)


class TinyMultiHeadAttention(nn.Module):
    """
    Efficient multi-head attention for TRM blocks
    
    Optimized for recursive processing with minimal parameter overhead.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            
        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        
        # Apply rotary positional embedding
        q, k = self.rope(q, k, seq_len)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask to [batch, heads, seq_len, seq_len]
                mask = attention_mask.unsqueeze(1).unsqueeze(1)
                mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            else:
                mask = attention_mask.unsqueeze(1)  # Add heads dimension
            
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.output_proj(attn_output)
        
        return output


class TinyFeedForward(nn.Module):
    """
    Efficient feed-forward network for TRM blocks
    
    Uses SwiGLU activation for better performance with fewer parameters.
    """
    
    def __init__(self, hidden_size: int, ff_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.ff_hidden_size = ff_hidden_size
        
        # SwiGLU implementation
        self.w1 = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        self.w2 = nn.Linear(ff_hidden_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ff_hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: x -> SiLU(W1(x)) * W3(x) -> W2"""
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TinyTransformerLayer(nn.Module):
    """
    Single transformer layer for TRM blocks
    
    Combines multi-head attention and feed-forward with residual connections
    and layer normalization.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, ff_hidden_size: int, 
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        
        # Pre-norm architecture for better training stability
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = TinyMultiHeadAttention(hidden_size, num_heads, dropout)
        
        self.ff_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = TinyFeedForward(hidden_size, ff_hidden_size, dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with pre-norm and residual connections
        """
        # Self-attention with residual connection
        norm_hidden = self.attention_norm(hidden_states)
        attn_output = self.attention(norm_hidden, attention_mask)
        hidden_states = hidden_states + attn_output
        
        # Feed-forward with residual connection
        norm_hidden = self.ff_norm(hidden_states)
        ff_output = self.feed_forward(norm_hidden)
        hidden_states = hidden_states + ff_output
        
        return hidden_states


class RecursiveProcessor(nn.Module):
    """
    Recursive processing module for TRM blocks
    
    Implements recursive reasoning by repeatedly applying the same
    transformation with decreasing intensity.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Lightweight recursive layer
        self.recursive_attention = TinyMultiHeadAttention(hidden_size, num_heads, dropout)
        self.recursive_norm = nn.LayerNorm(hidden_size)
        
        # Gating mechanism for recursive depth control
        self.gate_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor, recursion_depth: int = 1) -> torch.Tensor:
        """
        Apply recursive processing
        
        Args:
            hidden_states: Input hidden states
            recursion_depth: Number of recursive steps to apply
            
        Returns:
            Recursively processed hidden states
        """
        current_states = hidden_states
        
        for step in range(recursion_depth):
            # Compute gating weights (decreasing with depth)
            gate_weights = torch.sigmoid(self.gate_proj(current_states))
            intensity = gate_weights * (0.8 ** step)  # Decreasing intensity
            
            # Apply recursive transformation
            norm_states = self.recursive_norm(current_states)
            recursive_output = self.recursive_attention(norm_states)
            
            # Gated residual connection
            current_states = current_states + intensity * recursive_output
        
        return current_states


class TinyTransformer(nn.Module):
    """
    Complete tiny transformer stack for TRM blocks
    
    This is the core transformer that processes hidden states
    without any token embeddings.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, ff_hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TinyTransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_hidden_size=ff_hidden_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all transformer layers
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return self.final_norm(hidden_states)


class HybridTRMBlock(nn.Module):
    """
    Complete Hybrid TRM Block for layer injection
    
    This is the main TRM block designed specifically for injection into LLM layers.
    It processes hidden states directly without token embeddings and includes
    recursive reasoning capabilities.
    
    Key Features:
    - No token embeddings (hidden state processing only)
    - Configurable recursive processing depth
    - Efficient multi-head attention
    - Designed for gradient-preserving injection
    - Minimal parameter overhead
    
    Args:
        hidden_size: Hidden dimension of the input states
        num_heads: Number of attention heads
        ff_hidden_size: Feedforward hidden dimension
        num_layers: Number of transformer layers
        num_recursion_steps: Number of recursive processing steps
        dropout: Dropout rate
        layer_norm_eps: Epsilon for layer normalization
    
    Example:
        >>> trm_block = HybridTRMBlock(
        ...     hidden_size=768,
        ...     num_heads=12,
        ...     ff_hidden_size=3072,
        ...     num_recursion_steps=3
        ... )
        >>> 
        >>> # Process hidden states from LLM layer
        >>> enhanced_states = trm_block(llm_hidden_states)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, ff_hidden_size: int,
                 num_layers: int = 2, num_recursion_steps: int = 3,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size
        self.num_layers = num_layers
        self.num_recursion_steps = num_recursion_steps
        
        # Input processing (no embeddings, just normalization)
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Core tiny transformer
        self.tiny_transformer = TinyTransformer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            ff_hidden_size=ff_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )
        
        # Recursive processor
        self.recursive_processor = RecursiveProcessor(
            hidden_size=hidden_size,
            num_heads=max(1, num_heads // 2),  # Fewer heads for recursion
            dropout=dropout
        )
        
        # Output processing
        self.output_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Parameter initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following best practices for transformers"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the hybrid TRM block
        
        Args:
            hidden_states: Input hidden states from LLM layer [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            
        Returns:
            Enhanced hidden states [batch, seq_len, hidden_size]
        """
        # Input normalization (no embeddings needed)
        normalized_states = self.input_layernorm(hidden_states)
        
        # Core transformer processing
        transformer_output = self.tiny_transformer(normalized_states, attention_mask)
        
        # Recursive reasoning (if enabled)
        if self.num_recursion_steps > 1:
            recursive_output = self.recursive_processor(
                transformer_output, 
                recursion_depth=self.num_recursion_steps - 1
            )
        else:
            recursive_output = transformer_output
        
        # Output normalization
        output = self.output_layernorm(recursive_output)
        
        return output
    
    def get_parameter_count(self) -> dict:
        """Get detailed parameter count breakdown"""
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            'total': count_parameters(self),
            'tiny_transformer': count_parameters(self.tiny_transformer),
            'recursive_processor': count_parameters(self.recursive_processor),
            'layer_norms': count_parameters(self.input_layernorm) + count_parameters(self.output_layernorm),
        }
    
    def set_recursion_steps(self, steps: int):
        """Dynamically change the number of recursion steps"""
        if steps < 1:
            raise ValueError("Recursion steps must be at least 1")
        self.num_recursion_steps = steps


class TRMBlock(nn.Module):
    """TRM Block for layer injection - simplified version of HybridTRMBlock"""
    
    def __init__(self, config):
        super().__init__()
        # Use enhancement_dim and attention_heads from MillennialAiConfig
        hidden_size = getattr(config, 'enhancement_dim', 512)
        num_heads = getattr(config, 'attention_heads', 8)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class HybridTRM(nn.Module):
    """Stack of TRM blocks for enhanced reasoning"""
    
    def __init__(self, config):
        super().__init__()
        # Use cognitive_layers from MillennialAiConfig, default to 4
        num_layers = getattr(config, 'cognitive_layers', 4)
        self.layers = nn.ModuleList([
            TRMBlock(config) for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x