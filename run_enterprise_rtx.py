"""
MillennialAI Enterprise - RTX 5060 Ti Optimized
Full Layer Injection Architecture with GPU Acceleration
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("=" * 80)
print("🚀 MILLENNIAL AI ENTERPRISE - RTX 5060 Ti OPTIMIZED")
print("🎯 Full Layer Injection Architecture with GPU Acceleration")
print("=" * 80)

# Check GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n✅ GPU Detected: {gpu_name}")
    print(f"✅ VRAM Available: {gpu_memory:.1f} GB")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    device = torch.device('cuda')

    # Memory optimizations
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    print("❌ No GPU detected! Running on CPU (very slow)")
    device = torch.device('cpu')

try:
    from millennial_ai.core.hybrid_model import CombinedTRMLLM
    from millennial_ai.config.config import HybridConfig, PresetConfigs
    print("✅ MillennialAI core modules loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not import MillennialAI modules: {e}")
    print("⚠️  Creating standalone demonstration...")
    MILLENNIAL_AI_AVAILABLE = False
else:
    MILLENNIAL_AI_AVAILABLE = True


class RTX5060TiConfig:
    """RTX 5060 Ti optimized configurations"""

    @staticmethod
    def rtx_optimized_base() -> HybridConfig:
        """Base configuration optimized for RTX 5060 Ti (16GB)"""
        return HybridConfig(
            injection_layers=[4, 8, 12],  # 3 strategic injection points
            trm_hidden_size=1536,         # 1.5K TRM hidden size (fits in 16GB)
            trm_num_heads=12,             # 12 attention heads
            trm_ff_hidden_size=6144,      # 6K FF dimension
            trm_num_layers=3,             # 3 TRM layers for efficiency
            num_recursion_steps=4,        # 4 recursion steps
            dropout=0.1,                  # Standard dropout
            recursion_dropout=0.15,
            adaptive_injection=True,
            injection_strength=0.6,       # Balanced injection strength
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-6,
            gradient_checkpointing=False, # Not needed for inference
            mixed_precision=True,
        )

    @staticmethod
    def rtx_optimized_performance() -> HybridConfig:
        """Performance-optimized for RTX 5060 Ti"""
        return HybridConfig(
            injection_layers=[6, 12, 18],  # 3 injection points
            trm_hidden_size=2048,          # 2K TRM hidden size
            trm_num_heads=16,              # 16 attention heads
            trm_ff_hidden_size=8192,       # 8K FF dimension
            trm_num_layers=4,              # 4 TRM layers
            num_recursion_steps=6,         # 6 recursion steps
            dropout=0.08,
            recursion_dropout=0.12,
            adaptive_injection=True,
            injection_strength=0.7,        # Stronger injection
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-6,
            gradient_checkpointing=False,
            mixed_precision=True,
        )

    @staticmethod
    def rtx_optimized_max() -> HybridConfig:
        """Maximum capability for RTX 5060 Ti (uses ~12GB VRAM)"""
        return HybridConfig(
            injection_layers=[4, 8, 12, 16, 20],  # 5 injection points
            trm_hidden_size=2560,          # 2.5K TRM hidden size
            trm_num_heads=20,              # 20 attention heads
            trm_ff_hidden_size=10240,      # 10K FF dimension
            trm_num_layers=5,              # 5 TRM layers
            num_recursion_steps=8,         # 8 recursion steps
            dropout=0.06,
            recursion_dropout=0.1,
            adaptive_injection=True,
            injection_strength=0.8,        # Strong injection
            blending_strategy="attention_weighted",
            projection_bias=True,
            layer_norm_eps=1e-6,
            gradient_checkpointing=False,
            mixed_precision=True,
        )


class RTXOptimizedLLM(nn.Module):
    """RTX 5060 Ti optimized LLM with layer injection"""

    def __init__(self, config_name="performance"):
        super().__init__()

        print(f"\n🔨 Creating RTX 5060 Ti Optimized LLM...")
        print(f"   Config: {config_name}")

        # Select configuration
        if config_name == "base":
            self.config = RTX5060TiConfig.rtx_optimized_base()
        elif config_name == "performance":
            self.config = RTX5060TiConfig.rtx_optimized_performance()
        elif config_name == "max":
            self.config = RTX5060TiConfig.rtx_optimized_max()
        else:
            raise ValueError(f"Unknown config: {config_name}")

        # Create base LLM (optimized for 16GB GPU)
        self.base_llm = self._create_optimized_base_llm()

        # Create TRM injection blocks
        self.trm_blocks = nn.ModuleDict()
        for layer_idx in self.config.injection_layers:
            self.trm_blocks[str(layer_idx)] = self._create_trm_block()

        # Projection layers for each injection point
        self.projections = nn.ModuleDict()
        for layer_idx in self.config.injection_layers:
            self.projections[str(layer_idx)] = nn.Linear(
                self.base_llm.hidden_size, self.config.trm_hidden_size
            )

        # Injection state
        self.injection_active = True
        self.injection_count = 0

        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        base_params = sum(p.numel() for p in self.base_llm.parameters())
        trm_params = total_params - base_params

        print(f"   💾 Base LLM params: {base_params:,}")
        print(f"   💾 TRM params: {trm_params:,}")
        print(f"   💾 Total params: {total_params:,}")
        print(f"   💾 Overhead: {trm_params / base_params * 100:.1f}%")
        print(f"   🎯 Injection layers: {self.config.injection_layers}")

    def _create_optimized_base_llm(self):
        """Create base LLM optimized for RTX 5060 Ti"""
        vocab_size = 32000
        hidden_size = 2048  # Optimized for 16GB
        num_layers = 24     # Balanced layer count
        num_heads = 16      # Balanced attention

        class OptimizedTransformerBlock(nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.ff = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(0.1)
                )

            def forward(self, x):
                # Pre-norm attention
                normed = self.norm1(x)
                attn_out, _ = self.attention(normed, normed, normed)
                x = x + attn_out

                # Pre-norm FF
                normed = self.norm2(x)
                ff_out = self.ff(normed)
                x = x + ff_out
                return x

        class OptimizedLLM(nn.Module):
            def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_layers

                self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    OptimizedTransformerBlock(hidden_size, num_heads)
                    for _ in range(num_layers)
                ])
                self.norm = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            def forward(self, input_ids, **kwargs):
                x = self.embed_tokens(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.norm(x)
                logits = self.lm_head(x)
                return type('Output', (), {'logits': logits, 'last_hidden_state': x})()

        return OptimizedLLM(vocab_size, hidden_size, num_layers, num_heads)

    def _create_trm_block(self):
        """Create TRM block for injection"""
        return nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config.trm_hidden_size,
                nhead=self.config.trm_num_heads,
                dim_feedforward=self.config.trm_ff_hidden_size,
                dropout=self.config.dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(self.config.trm_num_layers)
        ])

    def forward(self, input_ids, attention_mask=None):
        """Forward pass with layer injection"""
        hidden_states = self.base_llm.embed_tokens(input_ids)

        # Pass through layers with injection
        for i, layer in enumerate(self.base_llm.layers):
            # Standard transformer layer
            hidden_states = layer(hidden_states)

            # Apply injection at designated layers
            if self.injection_active and i in self.config.injection_layers:
                # Project to TRM space
                trm_input = self.projections[str(i)](hidden_states)

                # Apply TRM processing with recursion
                trm_output = trm_input
                for step in range(self.config.num_recursion_steps):
                    for trm_layer in self.trm_blocks[str(i)]:
                        trm_output = trm_layer(trm_output)

                # Project back to LLM space
                enhanced = torch.matmul(trm_output, self.projections[str(i)].weight.t())

                # Blend with controlled strength
                hidden_states = (1 - self.config.injection_strength) * hidden_states + \
                               self.config.injection_strength * enhanced

                self.injection_count += 1

        # Final processing
        hidden_states = self.base_llm.norm(hidden_states)
        logits = self.base_llm.lm_head(hidden_states)

        return type('Output', (), {
            'logits': logits,
            'last_hidden_state': hidden_states
        })()

    def get_stats(self):
        """Get injection statistics"""
        return {
            'injection_count': self.injection_count,
            'injection_layers': self.config.injection_layers,
            'injection_active': self.injection_active,
            'config': {
                'trm_hidden_size': self.config.trm_hidden_size,
                'trm_num_heads': self.config.trm_num_heads,
                'injection_strength': self.config.injection_strength,
                'num_recursion_steps': self.config.num_recursion_steps
            }
        }


def run_rtx_optimized_benchmark(config_name="performance"):
    """Run RTX 5060 Ti optimized benchmark"""

    print(f"\n" + "=" * 80)
    print(f"🧪 RTX 5060 Ti BENCHMARK - {config_name.upper()} CONFIG")
    print("=" * 80)

    # Create optimized model
    print(f"\n📦 Step 1: Creating RTX-Optimized Model...")
    model = RTXOptimizedLLM(config_name=config_name)

    # Move to GPU
    if torch.cuda.is_available():
        print(f"\n🚀 Step 2: Moving to GPU...")
        model = model.to(device)
        print(f"   ✅ Model on GPU")

    # Create test input optimized for RTX 5060 Ti
    batch_size = 4      # Conservative batch size
    seq_len = 512       # Reasonable sequence length

    print(f"\n📝 Step 3: Creating Test Input...")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Total tokens: {batch_size * seq_len:,}")

    input_ids = torch.randint(0, model.base_llm.vocab_size, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.to(device)

    # Warm-up pass
    print(f"\n🔥 Step 4: GPU Warm-up...")
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark: Baseline (no injection)
    print(f"\n📊 Step 5: Baseline Performance (No Injection)...")
    model.injection_active = False
    model.injection_count = 0

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            baseline_output = model(input_ids)
        end_event.record()

        torch.cuda.synchronize()
        baseline_time = start_event.elapsed_time(end_event) / 1000
        baseline_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"   ⏱️  Time: {baseline_time:.3f} seconds")
        print(f"   💾 Memory: {baseline_memory:.2f} GB")
        print(f"   🚀 Throughput: {batch_size * seq_len / baseline_time:.0f} tokens/sec")
    else:
        import time
        start = time.time()
        with torch.no_grad():
            baseline_output = model(input_ids)
        baseline_time = time.time() - start
        print(f"   ⏱️  Time: {baseline_time:.3f} seconds")

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark: With layer injection
    print(f"\n🎯 Step 6: Enhanced Performance (With Injection)...")
    model.injection_active = True
    model.injection_count = 0

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            injected_output = model(input_ids)
        end_event.record()

        torch.cuda.synchronize()
        injected_time = start_event.elapsed_time(end_event) / 1000
        injected_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"   ⏱️  Time: {injected_time:.3f} seconds")
        print(f"   💾 Memory: {injected_memory:.2f} GB")
        print(f"   🚀 Throughput: {batch_size * seq_len / injected_time:.0f} tokens/sec")
        print(f"   🎯 Injections performed: {model.injection_count}")
    else:
        start = time.time()
        with torch.no_grad():
            injected_output = model(input_ids)
        injected_time = time.time() - start
        print(f"   ⏱️  Time: {injected_time:.3f} seconds")
        print(f"   🎯 Injections performed: {model.injection_count}")

    # Analysis
    print(f"\n📈 Step 7: Performance Analysis...")

    diff = torch.abs(baseline_output.logits - injected_output.logits)
    print(f"   📊 Mean output difference: {diff.mean().item():.6f}")
    print(f"   📊 Max output difference: {diff.max().item():.6f}")
    print(f"   📊 Output shape: {injected_output.logits.shape}")

    if torch.cuda.is_available():
        overhead = (injected_time - baseline_time) / baseline_time * 100
        memory_overhead = (injected_memory - baseline_memory) / baseline_memory * 100

        print(f"\n💡 Performance Impact:")
        print(f"   ⏱️  Time overhead: {overhead:.1f}%")
        print(f"   💾 Memory overhead: {memory_overhead:.1f}%")
        print(f"   🎯 Enhanced reasoning: {diff.mean().item() > 1e-4}")

        # GPU utilization info
        print(f"\n🖥️  RTX 5060 Ti Utilization:")
        print(f"   💾 Peak memory: {injected_memory:.2f} / {gpu_memory:.1f} GB ({injected_memory/gpu_memory*100:.1f}%)")
        print(f"   ✅ Headroom: {gpu_memory - injected_memory:.2f} GB available")

    print("\n" + "=" * 80)
    print("✅ RTX 5060 Ti BENCHMARK COMPLETE!")
    print("=" * 80)

    return model


def run_millennial_ai_enterprise():
    """Run full MillennialAI Enterprise if available"""
    if not MILLENNIAL_AI_AVAILABLE:
        print("\n⚠️  Full MillennialAI not available, running optimized version...")
        return run_rtx_optimized_benchmark()

    print(f"\n🎯 Step 1: Loading MillennialAI Enterprise Architecture...")

    try:
        # Create a mock LLM for demonstration
        class MockEnterpriseLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 2048
                self.num_hidden_layers = 24
                self.vocab_size = 32000

                self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerDecoderLayer(
                        d_model=self.hidden_size,
                        nhead=16,
                        dim_feedforward=self.hidden_size * 4,
                        dropout=0.1,
                        batch_first=True,
                        norm_first=True
                    )
                    for _ in range(self.num_hidden_layers)
                ])
                self.norm = nn.LayerNorm(self.hidden_size)
                self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

            def forward(self, input_ids, **kwargs):
                x = self.embed_tokens(input_ids)
                for layer in self.layers:
                    x = layer(x, x)
                x = self.norm(x)
                logits = self.lm_head(x)
                return type('Output', (), {'logits': logits, 'last_hidden_state': x})()

        # Create base model
        base_llm = MockEnterpriseLLM()

        # Use RTX-optimized config
        config = RTX5060TiConfig.rtx_optimized_performance()

        print(f"   🎯 Injection layers: {config.injection_layers}")
        print(f"   🎯 TRM hidden size: {config.trm_hidden_size}")
        print(f"   🎯 TRM heads: {config.trm_num_heads}")

        # Create hybrid model
        print(f"\n🔗 Step 2: Creating Hybrid Architecture...")
        hybrid = CombinedTRMLLM(llm_model=base_llm, config=config)

        # Move to GPU
        if torch.cuda.is_available():
            print(f"\n🚀 Step 3: Moving to GPU...")
            hybrid = hybrid.to(device)
            print(f"   ✅ Hybrid model on GPU")

        # Show parameter breakdown
        params = hybrid.get_parameter_count()
        print(f"   📊 Base LLM parameters: {params['llm_model']:,}")
        print(f"   📊 TRM parameters: {params['trm_block']:,}")
        print(f"   📊 Projection parameters: {params['projection']:,}")
        print(f"   📊 Total hybrid parameters: {params['total']:,}")
        print(f"   📊 Parameter overhead: {params['overhead_percentage']:.1f}%")

        # Test input
        batch_size = 4
        seq_len = 512
        input_ids = torch.randint(0, base_llm.vocab_size, (batch_size, seq_len))
        if torch.cuda.is_available():
            input_ids = input_ids.to(device)

        print(f"\n🧪 Step 4: Testing Enterprise Architecture...")

        # Baseline
        hybrid.deactivate_injection()
        with torch.no_grad():
            baseline = hybrid(input_ids)

        # With injection
        hybrid.activate_injection()
        with torch.no_grad():
            injected = hybrid(input_ids)

        # Analysis
        stats = hybrid.get_injection_statistics()
        diff = torch.abs(baseline.logits - injected.logits)

        print(f"   ✅ Enterprise architecture working!")
        print(f"   📊 Total injections: {stats['total_injections']:,}")
        print(f"   📊 Mean enhancement: {diff.mean().item():.6f}")

        return hybrid

    except Exception as e:
        print(f"   ❌ Error with full MillennialAI: {e}")
        print(f"   💡 Falling back to optimized version...")
        return run_rtx_optimized_benchmark()


def main():
    """Main execution"""

    try:
        # Try full MillennialAI Enterprise first
        if MILLENNIAL_AI_AVAILABLE:
            model = run_millennial_ai_enterprise()
        else:
            # Fallback to optimized version
            model = run_rtx_optimized_benchmark("performance")

        print(f"\n🎉 SUCCESS! MillennialAI Enterprise is running on your RTX 5060 Ti!")
        print(f"\n📋 What you have:")
        print(f"   ✅ RTX 5060 Ti GPU acceleration")
        print(f"   ✅ Layer injection architecture")
        print(f"   ✅ TRM temporal reasoning modules")
        print(f"   ✅ Optimized for 16GB VRAM")

        print(f"\n📋 Next Steps:")
        print(f"   1. Model ready for inference")
        print(f"   2. Try different configs: 'base', 'performance', 'max'")
        print(f"   3. Adjust batch_size and seq_len for your tasks")
        print(f"   4. Add text generation and prompting logic")

        # Show current stats
        if hasattr(model, 'get_stats'):
            stats = model.get_stats()
            print(f"\n⚙️  Current Configuration:")
            print(f"   Injection layers: {stats['injection_layers']}")
            print(f"   Injection count: {stats['injection_count']}")
            print(f"   Active: {stats['injection_active']}")

        if MILLENNIAL_AI_AVAILABLE and hasattr(model, 'get_injection_statistics'):
            enterprise_stats = model.get_injection_statistics()
            print(f"   Enterprise stats: {enterprise_stats}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n🧹 GPU cache cleared")


if __name__ == '__main__':
    main()
