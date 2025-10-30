"""
Basic Usage Example for MillenialAi

This example demonstrates how to use the Layer Injection Architecture
with a simple mock model to understand the core concepts.
"""

import torch
import torch.nn as nn

# Import MillenialAi components
from millennial_ai.core.hybrid_model import CombinedTRMLLM, create_hybrid_model
from millennial_ai.config.config import HybridConfig


class SimpleLLM(nn.Module):
    """Simple mock LLM for demonstration"""
    
    def __init__(self, vocab_size=1000, hidden_size=512, num_layers=6):
        super().__init__()
        
        # Simple transformer structure
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.ModuleDict({
            'h': nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 2,
                    batch_first=True,
                    norm_first=True
                ) for _ in range(num_layers)
            ])
        })
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Config for compatibility
        self.config = type('Config', (), {'hidden_size': hidden_size})()
    
    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        
        # Pass through transformer layers
        for layer in self.transformer.h:
            x = layer(x, x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return type('Output', (), {'logits': logits, 'last_hidden_state': x})()


def main():
    print("MillenialAi Basic Usage Example")
    print("=" * 40)
    
    # Step 1: Create a base LLM model
    print("\n1. Creating base LLM model...")
    llm = SimpleLLM(vocab_size=1000, hidden_size=512, num_layers=6)
    print(f"   Model created with {sum(p.numel() for p in llm.parameters()):,} parameters")
    
    # Step 2: Configure the hybrid architecture
    print("\n2. Configuring Layer Injection Architecture...")
    config = HybridConfig(
        injection_layers=[2, 4],  # Inject at layers 2 and 4
        trm_hidden_size=256,      # TRM hidden size
        trm_num_heads=8,          # TRM attention heads
        trm_num_layers=2,         # TRM layers
        num_recursion_steps=3     # Recursive processing steps
    )
    print(f"   Injection layers: {config.injection_layers}")
    print(f"   TRM architecture: {config.trm_hidden_size}d, {config.trm_num_heads} heads")
    
    # Step 3: Create the hybrid model
    print("\n3. Creating hybrid model...")
    hybrid = CombinedTRMLLM(llm_model=llm, config=config)
    
    # Show parameter breakdown
    params = hybrid.get_parameter_count()
    print(f"   Total parameters: {params['total']:,}")
    print(f"   LLM parameters: {params['llm_model']:,}")
    print(f"   TRM parameters: {params['trm_block']:,}")
    print(f"   Projection parameters: {params['projection']:,}")
    print(f"   Overhead: {params['overhead_percentage']:.1f}%")
    
    # Step 4: Prepare test data
    print("\n4. Preparing test data...")
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")
    
    # Step 5: Run without injection (baseline)
    print("\n5. Running baseline (no injection)...")
    hybrid.eval()
    with torch.no_grad():
        output_baseline = hybrid(input_ids)
    
    baseline_logits = output_baseline.logits
    print(f"   Output shape: {baseline_logits.shape}")
    print(f"   Output mean: {baseline_logits.mean().item():.4f}")
    print(f"   Output std: {baseline_logits.std().item():.4f}")
    
    # Step 6: Activate injection and run
    print("\n6. Running with Layer Injection...")
    hybrid.activate_injection()
    
    with torch.no_grad():
        output_injected = hybrid(input_ids)
    
    injected_logits = output_injected.logits
    print(f"   Output shape: {injected_logits.shape}")
    print(f"   Output mean: {injected_logits.mean().item():.4f}")
    print(f"   Output std: {injected_logits.std().item():.4f}")
    
    # Step 7: Compare outputs
    print("\n7. Analyzing injection effects...")
    difference = torch.abs(baseline_logits - injected_logits)
    print(f"   Mean absolute difference: {difference.mean().item():.6f}")
    print(f"   Max absolute difference: {difference.max().item():.6f}")
    
    # Check that outputs are indeed different
    if torch.allclose(baseline_logits, injected_logits, atol=1e-6):
        print("   ⚠️  WARNING: Outputs are nearly identical!")
    else:
        print("   ✓ Injection successfully modified the outputs")
    
    # Step 8: Show injection statistics
    print("\n8. Injection statistics...")
    stats = hybrid.get_injection_statistics()
    print(f"   Total injections: {stats['total_injections']}")
    print(f"   Per-layer counts: {stats['layer_counts']}")
    print(f"   Average per layer: {stats['average_per_layer']:.1f}")
    
    # Step 9: Demonstrate injection toggle
    print("\n9. Demonstrating injection toggle...")
    
    # Deactivate injection
    hybrid.deactivate_injection()
    print(f"   Injection active: {hybrid.injection_active}")
    
    # Reactivate injection
    hybrid.activate_injection()
    print(f"   Injection active: {hybrid.injection_active}")
    
    # Step 10: Demonstrate different configurations
    print("\n10. Testing different configurations...")
    
    # Minimal injection
    minimal_config = HybridConfig.from_preset('minimal')
    hybrid_minimal = create_hybrid_model(llm, **minimal_config.to_dict())
    hybrid_minimal.activate_injection()
    
    with torch.no_grad():
        output_minimal = hybrid_minimal(input_ids)
    
    print(f"    Minimal config parameters: {hybrid_minimal.get_parameter_count()['overhead_params']:,}")
    
    # Adaptive injection
    adaptive_config = HybridConfig(
        injection_layers=[1, 3, 5],
        adaptive_injection=True,
        projection_type='adaptive'
    )
    hybrid_adaptive = CombinedTRMLLM(llm_model=llm, config=adaptive_config)
    hybrid_adaptive.activate_injection()
    
    with torch.no_grad():
        output_adaptive = hybrid_adaptive(input_ids)
    
    print(f"    Adaptive config parameters: {hybrid_adaptive.get_parameter_count()['overhead_params']:,}")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("\nKey takeaways:")
    print("• Layer injection modifies LLM outputs without changing the base model")
    print("• Injection can be activated/deactivated dynamically")
    print("• Overhead is configurable based on TRM architecture")
    print("• Different projection strategies are available")


if __name__ == '__main__':
    main()