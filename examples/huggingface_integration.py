"""
HuggingFace Integration Example for MillenialAi

This example shows how to use the Layer Injection Architecture with
real HuggingFace transformer models for practical applications.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    BertModel, BertTokenizer,
    pipeline
)
import warnings
warnings.filterwarnings('ignore')

# Import MillenialAi components
from millennial_ai.core.hybrid_model import CombinedTRMLLM
from millennial_ai.config.config import HybridConfig


def test_gpt2_integration():
    """Test with GPT-2 for text generation"""
    print("\n" + "="*50)
    print("GPT-2 Integration Example")
    print("="*50)
    
    try:
        # Load GPT-2 model and tokenizer
        print("Loading GPT-2 model...")
        model_name = "gpt2"  # Use smallest GPT-2 for faster loading
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set padding token
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create hybrid configuration optimized for GPT-2
        config = HybridConfig.from_preset('gpt2')
        print(f"Injection layers: {config.injection_layers}")
        
        # Create hybrid model
        hybrid = CombinedTRMLLM(llm_model=model, config=config)
        param_counts = hybrid.get_parameter_count()
        print(f"Hybrid model overhead: {param_counts['overhead_percentage']:.1f}%")
        
        # Test text generation
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where machines can think,",
            "The key to understanding neural networks"
        ]
        
        print("\nGenerating text samples...")
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}: '{prompt}'")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors='pt', padding=True)
            
            # Generate without injection
            hybrid.deactivate_injection()
            with torch.no_grad():
                outputs_normal = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 20,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Generate with injection
            hybrid.activate_injection()
            with torch.no_grad():
                outputs_injected = hybrid.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 20,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode results
            text_normal = tokenizer.decode(outputs_normal[0], skip_special_tokens=True)
            text_injected = tokenizer.decode(outputs_injected[0], skip_special_tokens=True)
            
            print(f"Normal:   {text_normal}")
            print(f"Injected: {text_injected}")
            
            # Check if outputs differ
            if text_normal != text_injected:
                print("✓ Injection modified the generation")
            else:
                print("⚠️ Outputs are identical")
        
        # Show injection statistics
        stats = hybrid.get_injection_statistics()
        print(f"\nInjection Statistics:")
        print(f"Total injections: {stats['total_injections']}")
        print(f"Layer distribution: {stats['layer_counts']}")
        
        return True
        
    except ImportError:
        print("❌ transformers library not available. Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Error in GPT-2 integration: {e}")
        return False


def test_bert_integration():
    """Test with BERT for masked language modeling"""
    print("\n" + "="*50)
    print("BERT Integration Example")
    print("="*50)
    
    try:
        # Load BERT model and tokenizer
        print("Loading BERT model...")
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        
        print(f"Model loaded: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create hybrid configuration for BERT
        config = HybridConfig.from_preset('bert')
        print(f"Injection layers: {config.injection_layers}")
        
        # Create hybrid model
        hybrid = CombinedTRMLLM(llm_model=model, config=config)
        param_counts = hybrid.get_parameter_count()
        print(f"Hybrid model overhead: {param_counts['overhead_percentage']:.1f}%")
        
        # Test sentence encoding
        test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming our world.",
            "Neural networks learn complex patterns from data."
        ]
        
        print("\nProcessing sentences...")
        for i, sentence in enumerate(test_sentences):
            print(f"\nSentence {i+1}: '{sentence}'")
            
            # Tokenize input
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            
            # Process without injection
            hybrid.deactivate_injection()
            with torch.no_grad():
                outputs_normal = hybrid(**inputs)
            
            # Process with injection
            hybrid.activate_injection()
            with torch.no_grad():
                outputs_injected = hybrid(**inputs)
            
            # Compare embeddings
            embeddings_normal = outputs_normal.last_hidden_state.mean(dim=1)  # Average pooling
            embeddings_injected = outputs_injected.last_hidden_state.mean(dim=1)
            
            # Calculate similarity
            similarity = torch.cosine_similarity(embeddings_normal, embeddings_injected, dim=1)
            difference = torch.norm(embeddings_normal - embeddings_injected, dim=1)
            
            print(f"Cosine similarity: {similarity.item():.4f}")
            print(f"L2 difference: {difference.item():.4f}")
            
            if similarity.item() < 0.99:  # Not too similar
                print("✓ Injection modified the embeddings")
            else:
                print("⚠️ Embeddings are very similar")
        
        return True
        
    except ImportError:
        print("❌ transformers library not available. Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Error in BERT integration: {e}")
        return False


def test_custom_model_integration():
    """Test with a custom model architecture"""
    print("\n" + "="*50)
    print("Custom Model Integration Example")
    print("="*50)
    
    try:
        # Create a custom transformer model
        from transformers import AutoConfig, AutoModel
        
        # Define custom configuration
        config_dict = {
            "hidden_size": 384,
            "num_attention_heads": 6,
            "num_hidden_layers": 6,
            "intermediate_size": 1536,
            "vocab_size": 30522,
            "max_position_embeddings": 512,
            "model_type": "bert"
        }
        
        config = AutoConfig.from_dict(config_dict)
        model = AutoModel.from_config(config)
        
        print(f"Custom model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create hybrid configuration
        hybrid_config = HybridConfig(
            injection_layers=[1, 3, 5],
            trm_hidden_size=192,  # Half the model size
            trm_num_heads=6,
            trm_num_layers=2,
            adaptive_injection=True
        )
        
        # Create hybrid model
        hybrid = CombinedTRMLLM(llm_model=model, config=hybrid_config)
        param_counts = hybrid.get_parameter_count()
        
        print(f"Hybrid configuration:")
        print(f"  Injection layers: {hybrid_config.injection_layers}")
        print(f"  TRM hidden size: {hybrid_config.trm_hidden_size}")
        print(f"  Adaptive injection: {hybrid_config.adaptive_injection}")
        print(f"  Parameter overhead: {param_counts['overhead_percentage']:.1f}%")
        
        # Test with random input
        batch_size, seq_len = 3, 64
        input_ids = torch.randint(0, config_dict['vocab_size'], (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        
        print(f"\nTesting with input shape: {input_ids.shape}")
        
        # Process without injection
        hybrid.deactivate_injection()
        with torch.no_grad():
            outputs_normal = hybrid(input_ids=input_ids, attention_mask=attention_mask)
        
        # Process with injection
        hybrid.activate_injection()
        with torch.no_grad():
            outputs_injected = hybrid(input_ids=input_ids, attention_mask=attention_mask)
        
        # Analyze differences
        hidden_normal = outputs_normal.last_hidden_state
        hidden_injected = outputs_injected.last_hidden_state
        
        mean_diff = torch.abs(hidden_normal - hidden_injected).mean()
        max_diff = torch.abs(hidden_normal - hidden_injected).max()
        
        print(f"Hidden state differences:")
        print(f"  Mean absolute difference: {mean_diff.item():.6f}")
        print(f"  Max absolute difference: {max_diff.item():.6f}")
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        hybrid.train()
        
        # Forward pass with dummy loss
        outputs = hybrid(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        trm_has_grad = any(p.grad is not None for p in hybrid.trm_block.parameters())
        projection_has_grad = any(p.grad is not None for p in hybrid.projection.parameters())
        
        print(f"  TRM block gradients: {'✓' if trm_has_grad else '❌'}")
        print(f"  Projection gradients: {'✓' if projection_has_grad else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in custom model integration: {e}")
        return False


def benchmark_performance():
    """Benchmark performance with different configurations"""
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    try:
        from transformers import GPT2LMHeadModel
        import time
        
        # Load model
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        print("Running performance benchmarks...")
        
        # Test different configurations
        configs = [
            ("minimal", HybridConfig.from_preset('minimal')),
            ("standard", HybridConfig(injection_layers=[3, 6, 9])),
            ("heavy", HybridConfig(injection_layers=[2, 4, 6, 8, 10], trm_hidden_size=512))
        ]
        
        results = {}
        
        for name, config in configs:
            print(f"\nTesting {name} configuration...")
            
            hybrid = CombinedTRMLLM(llm_model=model, config=config)
            param_counts = hybrid.get_parameter_count()
            
            # Prepare test data
            batch_size, seq_len = 4, 128
            input_ids = torch.randint(0, 50257, (batch_size, seq_len))
            
            # Benchmark without injection
            hybrid.eval()
            times_normal = []
            for _ in range(5):
                start = time.time()
                with torch.no_grad():
                    _ = hybrid(input_ids)
                times_normal.append(time.time() - start)
            
            # Benchmark with injection
            hybrid.activate_injection()
            times_injected = []
            for _ in range(5):
                start = time.time()
                with torch.no_grad():
                    _ = hybrid(input_ids)
                times_injected.append(time.time() - start)
            
            results[name] = {
                'params_overhead_pct': param_counts['overhead_percentage'],
                'time_normal': np.mean(times_normal),
                'time_injected': np.mean(times_injected),
                'time_overhead_pct': (np.mean(times_injected) / np.mean(times_normal) - 1) * 100
            }
            
            print(f"  Parameter overhead: {results[name]['params_overhead_pct']:.1f}%")
            print(f"  Time overhead: {results[name]['time_overhead_pct']:.1f}%")
        
        # Summary
        print(f"\nPerformance Summary:")
        print(f"{'Config':<10} {'Param OH':<10} {'Time OH':<10}")
        print("-" * 30)
        for name, result in results.items():
            print(f"{name:<10} {result['params_overhead_pct']:<9.1f}% {result['time_overhead_pct']:<9.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in performance benchmark: {e}")
        return False


def main():
    """Run all integration examples"""
    print("MillenialAi HuggingFace Integration Examples")
    print("=" * 60)
    
    # Track success
    results = []
    
    # Run examples
    results.append(("GPT-2 Integration", test_gpt2_integration()))
    results.append(("BERT Integration", test_bert_integration()))
    results.append(("Custom Model Integration", test_custom_model_integration()))
    results.append(("Performance Benchmark", benchmark_performance()))
    
    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{name:<30} {status}")
    
    total_passed = sum(results[i][1] for i in range(len(results)))
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All integration tests passed!")
        print("\nNext steps:")
        print("• Try with your own models and datasets")
        print("• Experiment with different injection configurations")
        print("• Fine-tune the hybrid model on your specific tasks")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")


if __name__ == '__main__':
    main()