"""
MillennialAi Beginner Example - Level 1 Training

This example shows how to start with a small pre-trained model and add
TRM injection for enhancement. Perfect for learning and development.

Hardware Requirements:
- Single RTX 4090 (24GB) or A100 (40GB)
- 32GB+ system RAM
- Basic setup, not enterprise-scale
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from millennial_ai.core.hybrid_model import CombinedTRMLLM
from millennial_ai.config.config import HybridConfig


def create_beginner_config():
    """Create a beginner-friendly configuration"""
    return HybridConfig(
        # Only inject at 2 strategic points in a 32-layer model
        injection_layers=[16, 24],       
        
        # Modest TRM architecture
        trm_hidden_size=2048,            # Smaller than enterprise (8192)
        trm_num_heads=16,                # Reasonable attention heads
        trm_ff_hidden_size=8192,         # 4x hidden size (standard)
        trm_num_layers=2,                # Shallow TRM stack
        num_recursion_steps=4,           # Basic recursion
        
        # Training optimization
        dropout=0.1,
        gradient_checkpointing=True,     # Save memory
        mixed_precision=True,            # Use FP16
        
        # Moderate injection
        injection_strength=0.6,          # Not too aggressive
        adaptive_injection=True,
        blending_strategy="attention_weighted",
    )


def load_base_model():
    """Load a pre-trained 7B model (manageable size)"""
    print("🔄 Loading pre-trained LLaMA-2-7B model...")
    
    # Use a smaller model first - you can scale up later
    model_name = "meta-llama/Llama-2-7b-hf"  # 7B parameters
    
    try:
        # Load with memory optimization
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,    # Use FP16 to save memory
            device_map="auto",            # Automatic device placement
            low_cpu_mem_usage=True,       # Optimize CPU memory
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return base_model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 You may need to:")
        print("   1. Request access to LLaMA-2 on HuggingFace")
        print("   2. Login: huggingface-cli login")
        print("   3. Or use a different open model")
        
        # Fallback to a smaller open model
        print("🔄 Trying alternative model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",  # Smaller, fully open
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        return base_model, tokenizer


def create_hybrid_model(base_model, config):
    """Create the hybrid model with TRM injection"""
    print("🔗 Creating hybrid model with TRM injection...")
    
    try:
        hybrid = CombinedTRMLLM(
            llm_model=base_model,
            config=config
        )
        
        # Show parameter breakdown
        params = hybrid.get_parameter_count()
        print(f"📊 Parameter Analysis:")
        print(f"   Base LLM: {params['llm_model']:,} parameters")
        print(f"   TRM Injection: {params['trm_block']:,} parameters")
        print(f"   Projections: {params['projection']:,} parameters")
        print(f"   Total: {params['total']:,} parameters")
        print(f"   Overhead: {params['overhead_percentage']:.1f}%")
        
        # Memory estimate
        memory_gb = params['total'] * 2 / 1024**3  # FP16
        print(f"   Estimated memory: {memory_gb:.1f} GB (FP16)")
        
        return hybrid
        
    except Exception as e:
        print(f"❌ Error creating hybrid: {e}")
        return None


def setup_training(hybrid_model):
    """Set up training configuration"""
    print("⚙️ Setting up training...")
    
    # Phase 1: Train only TRM components (recommended start)
    print("🎯 Phase 1: Training only TRM injection layers")
    
    # Freeze base model parameters
    for name, param in hybrid_model.named_parameters():
        if any(keyword in name.lower() for keyword in ['trm', 'projection']):
            param.requires_grad = True
            print(f"   ✅ Training: {name}")
        else:
            param.requires_grad = False
            print(f"   ❄️ Frozen: {name}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    
    print(f"📊 Training Parameters:")
    print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Frozen: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.1f}%)")
    
    # Optimizer for TRM components only
    optimizer = torch.optim.AdamW(
        [p for p in hybrid_model.parameters() if p.requires_grad],
        lr=1e-4,           # Learning rate for TRM training
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    return optimizer


def simple_training_step(hybrid_model, tokenizer, optimizer):
    """Demonstrate a simple training step"""
    print("🏃 Demonstration training step...")
    
    # Simple training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is revolutionizing artificial intelligence.",
        "MillennialAi enhances pre-trained models with TRM injection.",
        "Layer injection allows adding capabilities without full retraining."
    ]
    
    # Tokenize
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        hybrid_model = hybrid_model.cuda()
    
    # Training step
    hybrid_model.train()
    
    # Activate TRM injection
    hybrid_model.activate_injection()
    
    # Forward pass
    outputs = hybrid_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    print(f"📈 Training loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✅ Training step completed successfully!")
    
    # Compare with non-injection
    hybrid_model.deactivate_injection()
    with torch.no_grad():
        baseline_outputs = hybrid_model(**inputs, labels=inputs["input_ids"])
        baseline_loss = baseline_outputs.loss
    
    print(f"📊 Comparison:")
    print(f"   With TRM injection: {loss.item():.4f}")
    print(f"   Without injection: {baseline_loss.item():.4f}")
    print(f"   Improvement: {((baseline_loss.item() - loss.item()) / baseline_loss.item() * 100):.1f}%")


def main():
    """Main beginner example"""
    print("🌟 MillennialAi Beginner Training Example")
    print("🌟 Level 1: Small-Scale TRM Injection")
    print("=" * 60)
    
    # Step 1: Create configuration
    config = create_beginner_config()
    print(f"✅ Created beginner configuration")
    
    # Step 2: Load base model
    base_model, tokenizer = load_base_model()
    print(f"✅ Loaded base model")
    
    # Step 3: Create hybrid
    hybrid_model = create_hybrid_model(base_model, config)
    if hybrid_model is None:
        return
    
    # Step 4: Setup training
    optimizer = setup_training(hybrid_model)
    print(f"✅ Training setup complete")
    
    # Step 5: Demonstration
    simple_training_step(hybrid_model, tokenizer, optimizer)
    
    print("\n" + "=" * 60)
    print("🎉 Beginner example completed!")
    print("\n📚 Next Steps:")
    print("   1. Train on your own dataset")
    print("   2. Experiment with different injection layers")
    print("   3. Try larger models (13B, 30B)")
    print("   4. Eventually scale to enterprise (70B+)")
    print("\n💡 Remember:")
    print("   - Start small and scale gradually")
    print("   - TRM injection is much cheaper than full training")
    print("   - You're enhancing, not replacing, the base model")


if __name__ == "__main__":
    main()