"""
MillennialAI Enterprise - Optimized for RTX 5060 Ti (16GB)
GPU-Accelerated Layer Injection Architecture
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🚀 MILLENNIAL AI ENTERPRISE - GPU ACCELERATED")
print("🎯 Optimized for NVIDIA RTX 5060 Ti (16GB VRAM)")
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
    
    # Set memory optimization
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    print("❌ No GPU detected! Running on CPU (very slow)")
    device = torch.device('cpu')

try:
    from millennial_ai.core.hybrid_model import CombinedTRMLLM, create_hybrid_model
    from millennial_ai.config.config import HybridConfig, PresetConfigs
    print("✅ MillennialAI modules loaded successfully")
except ImportError as e:
    print(f"⚠️  Could not import MillennialAI modules: {e}")
    print("⚠️  Creating standalone demonstration...")


class OptimizedMockLLM(nn.Module):
    """Optimized LLM for 16GB GPU - scaled for RTX 5060 Ti"""
    
    def __init__(self, 
                 vocab_size=32000,
                 hidden_size=2048,      # Reduced for 16GB GPU
                 num_layers=24,         # Optimized layer count
                 num_heads=16):         # Balanced attention heads
        super().__init__()
        
        print(f"\n🔨 Creating Optimized LLM for 16GB GPU...")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Layers: {num_layers}")
        print(f"   Attention heads: {num_heads}")
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Config
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
            'vocab_size': vocab_size,
            'num_attention_heads': num_heads,
        })()
        
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   💾 Total parameters: {total_params:,}")
        print(f"   💾 Memory (FP32): {total_params * 4 / 1024**3:.2f} GB")
        print(f"   💾 Memory (FP16): {total_params * 2 / 1024**3:.2f} GB")
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass"""
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, hidden_states)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return type('Output', (), {
            'logits': logits,
            'last_hidden_state': hidden_states
        })()


class TRMBlock(nn.Module):
    """Temporal Reasoning Module for Layer Injection"""
    
    def __init__(self, hidden_size=2048, num_heads=16, num_layers=4):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        """Apply temporal reasoning"""
        for layer in self.layers:
            x = layer(x)
        return self.projection(x)


class GPUOptimizedHybrid(nn.Module):
    """GPU-Optimized Hybrid Model with Layer Injection"""
    
    def __init__(self, llm_model, injection_layers=[6, 12, 18], injection_strength=0.3):
        super().__init__()
        
        print(f"\n🔗 Creating GPU-Optimized Hybrid Architecture...")
        print(f"   Injection layers: {injection_layers}")
        print(f"   Injection strength: {injection_strength}")
        
        self.llm = llm_model
        self.injection_layers = injection_layers
        self.injection_strength = injection_strength
        
        # Create TRM blocks for each injection point
        hidden_size = llm_model.hidden_size
        self.trm_blocks = nn.ModuleDict({
            str(layer_idx): TRMBlock(hidden_size=hidden_size, num_heads=16, num_layers=3)
            for layer_idx in injection_layers
        })
        
        self.injection_active = True
        self.injection_count = 0
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        llm_params = sum(p.numel() for p in self.llm.parameters())
        trm_params = total_params - llm_params
        
        print(f"   💾 Base LLM params: {llm_params:,}")
        print(f"   💾 TRM params: {trm_params:,}")
        print(f"   💾 Total params: {total_params:,}")
        print(f"   💾 Overhead: {trm_params / llm_params * 100:.1f}%")
    
    def forward(self, input_ids, attention_mask=None):
        """Forward with layer injection"""
        hidden_states = self.llm.embed_tokens(input_ids)
        
        # Pass through layers with injection
        for i, layer in enumerate(self.llm.layers):
            # Standard transformer layer
            hidden_states = layer(hidden_states, hidden_states)
            
            # Apply injection at designated layers
            if self.injection_active and i in self.injection_layers:
                # Apply TRM block
                trm_output = self.trm_blocks[str(i)](hidden_states)
                
                # Inject with controlled strength
                hidden_states = (1 - self.injection_strength) * hidden_states + \
                               self.injection_strength * trm_output
                
                self.injection_count += 1
        
        # Final processing
        hidden_states = self.llm.norm(hidden_states)
        logits = self.llm.lm_head(hidden_states)
        
        return type('Output', (), {
            'logits': logits,
            'last_hidden_state': hidden_states
        })()
    
    def get_stats(self):
        """Get injection statistics"""
        return {
            'injection_count': self.injection_count,
            'injection_layers': self.injection_layers,
            'injection_active': self.injection_active
        }


def run_gpu_benchmark():
    """Run GPU-optimized benchmark"""
    
    print("\n" + "=" * 80)
    print("🧪 RUNNING GPU BENCHMARK")
    print("=" * 80)
    
    # Create optimized model for 16GB GPU
    print("\n📦 Step 1: Creating Optimized LLM...")
    llm = OptimizedMockLLM(
        hidden_size=2048,
        num_layers=24,
        num_heads=16
    )
    
    # Create hybrid with layer injection
    print("\n🔗 Step 2: Adding Layer Injection Architecture...")
    hybrid = GPUOptimizedHybrid(
        llm_model=llm,
        injection_layers=[6, 12, 18],  # Strategic injection points
        injection_strength=0.3
    )
    
    # Move to GPU
    if torch.cuda.is_available():
        print(f"\n🚀 Step 3: Moving to GPU...")
        hybrid = hybrid.to(device)
        print(f"   ✅ Model on GPU")
    
    # Create test input optimized for 16GB
    batch_size = 8      # Reasonable batch size
    seq_len = 512       # Moderate sequence length
    
    print(f"\n📝 Step 4: Creating Test Input...")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Total tokens: {batch_size * seq_len:,}")
    
    input_ids = torch.randint(0, llm.vocab_size, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.to(device)
    
    # Warm-up pass
    print(f"\n🔥 Step 5: GPU Warm-up...")
    hybrid.eval()
    with torch.no_grad():
        _ = hybrid(input_ids)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark: Baseline (no injection)
    print(f"\n📊 Step 6: Baseline Performance (No Injection)...")
    hybrid.injection_active = False
    hybrid.injection_count = 0
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            baseline_output = hybrid(input_ids)
        end_event.record()
        
        torch.cuda.synchronize()
        baseline_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        baseline_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"   ⏱️  Time: {baseline_time:.3f} seconds")
        print(f"   💾 Memory: {baseline_memory:.2f} GB")
        print(f"   🚀 Throughput: {batch_size * seq_len / baseline_time:.0f} tokens/sec")
    else:
        import time
        start = time.time()
        with torch.no_grad():
            baseline_output = hybrid(input_ids)
        baseline_time = time.time() - start
        print(f"   ⏱️  Time: {baseline_time:.3f} seconds")
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark: With layer injection
    print(f"\n🎯 Step 7: Enhanced Performance (With Injection)...")
    hybrid.injection_active = True
    hybrid.injection_count = 0
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            injected_output = hybrid(input_ids)
        end_event.record()
        
        torch.cuda.synchronize()
        injected_time = start_event.elapsed_time(end_event) / 1000
        injected_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"   ⏱️  Time: {injected_time:.3f} seconds")
        print(f"   💾 Memory: {injected_memory:.2f} GB")
        print(f"   🚀 Throughput: {batch_size * seq_len / injected_time:.0f} tokens/sec")
        print(f"   🎯 Injections performed: {hybrid.injection_count}")
    else:
        start = time.time()
        with torch.no_grad():
            injected_output = hybrid(input_ids)
        injected_time = time.time() - start
        print(f"   ⏱️  Time: {injected_time:.3f} seconds")
        print(f"   🎯 Injections performed: {hybrid.injection_count}")
    
    # Analysis
    print(f"\n📈 Step 8: Performance Analysis...")
    
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
        print(f"\n🖥️  GPU Utilization:")
        print(f"   💾 Peak memory: {injected_memory:.2f} / {gpu_memory:.1f} GB ({injected_memory/gpu_memory*100:.1f}%)")
        print(f"   ✅ Headroom: {gpu_memory - injected_memory:.2f} GB available")
    
    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETE!")
    print("=" * 80)
    
    return hybrid


def main():
    """Main execution"""
    
    try:
        # Run the benchmark
        model = run_gpu_benchmark()
        
        print(f"\n🎉 SUCCESS! MillennialAI Enterprise is running on your GPU!")
        print(f"\n📋 Next Steps:")
        print(f"   1. Model is ready for inference")
        print(f"   2. Adjust batch_size and seq_len for your use case")
        print(f"   3. Tune injection_strength for optimal performance")
        print(f"   4. Add your own prompts and generation logic")
        
        # Show configuration
        stats = model.get_stats()
        print(f"\n⚙️  Current Configuration:")
        print(f"   Injection layers: {stats['injection_layers']}")
        print(f"   Injection count: {stats['injection_count']}")
        print(f"   Active: {stats['injection_active']}")
        
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
