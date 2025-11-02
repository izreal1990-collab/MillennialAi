"""
Testing Framework for MillennialAi

This module provides comprehensive testing for the Layer Injection Architecture,
including unit tests, integration tests, and performance benchmarks.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import time
import json
import os
import tempfile

# Import our components
from millennial_ai.core.hybrid_model import CombinedTRMLLM, create_hybrid_model
from millennial_ai.models.hybrid_trm import HybridTRMBlock
from millennial_ai.models.projection import DimensionalBridge, create_dimensional_bridge
from millennial_ai.config.config import HybridConfig


class TestHybridConfig(unittest.TestCase):
    """Test the configuration system"""
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = HybridConfig()
        self.assertIsInstance(config.injection_layers, list)
        self.assertGreater(config.trm_hidden_size, 0)
        self.assertGreater(config.trm_num_heads, 0)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        config = HybridConfig(injection_layers=[0, 2, 4])
        self.assertEqual(len(config.injection_layers), 3)
        
        # Invalid config should raise error
        with self.assertRaises(ValueError):
            HybridConfig(injection_layers=[-1, 2])  # Negative layer
    
    def test_config_presets(self):
        """Test configuration presets"""
        # GPT-2 preset
        config = HybridConfig.from_preset('gpt2')
        self.assertIn(4, config.injection_layers)
        self.assertIn(8, config.injection_layers)
        
        # BERT preset
        config = HybridConfig.from_preset('bert')
        self.assertEqual(len(config.injection_layers), 3)
    
    def test_config_serialization(self):
        """Test config serialization and deserialization"""
        original = HybridConfig(injection_layers=[1, 3, 5])
        config_dict = original.to_dict()
        restored = HybridConfig.from_dict(config_dict)
        
        self.assertEqual(original.injection_layers, restored.injection_layers)
        self.assertEqual(original.trm_hidden_size, restored.trm_hidden_size)


class TestHybridTRMBlock(unittest.TestCase):
    """Test the TRM block implementation"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 256
        self.trm_block = HybridTRMBlock(
            hidden_size=self.hidden_size,
            num_heads=8,
            ff_hidden_size=512,
            num_layers=2,
            num_recursion_steps=3
        )
    
    def test_forward_pass(self):
        """Test basic forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        output = self.trm_block(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.allclose(output, x))  # Should change the input
    
    def test_attention_mask(self):
        """Test attention mask handling"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        mask = torch.ones(self.batch_size, self.seq_len).bool()
        mask[0, 5:] = False  # Mask second half of first sequence
        
        output = self.trm_block(x, attention_mask=mask)
        self.assertEqual(output.shape, x.shape)
    
    def test_recursion(self):
        """Test recursive processing"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Single recursion step
        trm_single = HybridTRMBlock(
            hidden_size=self.hidden_size,
            num_heads=8,
            ff_hidden_size=512,
            num_layers=1,
            num_recursion_steps=1
        )
        
        # Multiple recursion steps
        trm_multi = HybridTRMBlock(
            hidden_size=self.hidden_size,
            num_heads=8,
            ff_hidden_size=512,
            num_layers=1,
            num_recursion_steps=3
        )
        
        output_single = trm_single(x)
        output_multi = trm_multi(x)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output_single, output_multi))
    
    def test_gradient_flow(self):
        """Test gradient flow through TRM block"""
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        output = self.trm_block(x)
        loss = output.mean()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.allclose(x.grad, torch.zeros_like(x.grad)))


class TestDimensionalBridge(unittest.TestCase):
    """Test the projection system"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.llm_hidden = 768
        self.trm_hidden = 256
    
    def test_linear_projection(self):
        """Test linear projection bridge"""
        bridge = create_dimensional_bridge(
            llm_hidden_size=self.llm_hidden,
            trm_hidden_size=self.trm_hidden,
            projection_type='linear'
        )
        
        llm_hidden = torch.randn(self.batch_size, self.seq_len, self.llm_hidden)
        
        # Project to TRM space
        trm_hidden = bridge.project_to_trm(llm_hidden)
        self.assertEqual(trm_hidden.shape, (self.batch_size, self.seq_len, self.trm_hidden))
        
        # Project back to LLM space
        llm_restored = bridge.project_to_llm(trm_hidden)
        self.assertEqual(llm_restored.shape, llm_hidden.shape)
    
    def test_adaptive_projection(self):
        """Test adaptive projection bridge"""
        bridge = create_dimensional_bridge(
            llm_hidden_size=self.llm_hidden,
            trm_hidden_size=self.trm_hidden,
            projection_type='adaptive'
        )
        
        llm_hidden = torch.randn(self.batch_size, self.seq_len, self.llm_hidden)
        trm_hidden = bridge.project_to_trm(llm_hidden)
        llm_restored = bridge.project_to_llm(trm_hidden)
        
        self.assertEqual(trm_hidden.shape[-1], self.trm_hidden)
        self.assertEqual(llm_restored.shape, llm_hidden.shape)
    
    def test_identity_bridge(self):
        """Test identity bridge when dimensions match"""
        bridge = create_dimensional_bridge(
            llm_hidden_size=self.trm_hidden,
            trm_hidden_size=self.trm_hidden,
            projection_type='linear'
        )
        
        hidden = torch.randn(self.batch_size, self.seq_len, self.trm_hidden)
        trm_projected = bridge.project_to_trm(hidden)
        llm_projected = bridge.project_to_llm(trm_projected)
        
        # Should be approximately the same (within numerical precision)
        self.assertTrue(torch.allclose(hidden, llm_projected, atol=1e-6))


class MockLLM(nn.Module):
    """Mock LLM for testing"""
    
    def __init__(self, hidden_size=768, num_layers=12, vocab_size=50257):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Mock transformer structure
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, hidden_size),
            'wpe': nn.Embedding(1024, hidden_size),
            'h': nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                ) for _ in range(num_layers)
            ]),
            'ln_f': nn.LayerNorm(hidden_size)
        })
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Mock config
        self.config = type('Config', (), {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'vocab_size': vocab_size
        })()
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simple forward pass
        x = self.transformer.wte(input_ids)
        
        # Add position embeddings
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.transformer.wpe(position_ids)
        
        # Pass through layers
        for layer in self.transformer.h:
            x = layer(x, x, memory_mask=attention_mask)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        return type('Output', (), {'logits': logits, 'last_hidden_state': x})()


class TestCombinedTRMLLM(unittest.TestCase):
    """Test the main hybrid model"""
    
    def setUp(self):
        self.llm = MockLLM(hidden_size=768, num_layers=6)
        self.config = HybridConfig(
            injection_layers=[2, 4],
            trm_hidden_size=256,
            trm_num_heads=8
        )
        self.hybrid = CombinedTRMLLM(llm_model=self.llm, config=self.config)
        
        self.batch_size = 2
        self.seq_len = 10
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
    
    def test_initialization(self):
        """Test hybrid model initialization"""
        self.assertEqual(self.hybrid.llm_hidden_size, 768)
        self.assertFalse(self.hybrid.injection_active)
        self.assertEqual(len(self.hybrid.hooks), 0)
    
    def test_injection_activation(self):
        """Test injection activation and deactivation"""
        # Initially inactive
        self.assertFalse(self.hybrid.injection_active)
        
        # Activate injection
        self.hybrid.activate_injection()
        self.assertTrue(self.hybrid.injection_active)
        self.assertEqual(len(self.hybrid.hooks), 2)  # Two injection layers
        
        # Deactivate injection
        self.hybrid.deactivate_injection()
        self.assertFalse(self.hybrid.injection_active)
        self.assertEqual(len(self.hybrid.hooks), 0)
    
    def test_forward_without_injection(self):
        """Test forward pass without injection"""
        output = self.hybrid(self.input_ids)
        self.assertIsNotNone(output.logits)
        self.assertEqual(output.logits.shape, (self.batch_size, self.seq_len, self.llm.config.vocab_size))
    
    def test_forward_with_injection(self):
        """Test forward pass with injection"""
        # Activate injection
        self.hybrid.activate_injection()
        
        # Forward pass
        output = self.hybrid(self.input_ids)
        self.assertIsNotNone(output.logits)
        
        # Check that injection occurred
        stats = self.hybrid.get_injection_statistics()
        self.assertGreater(stats['total_injections'], 0)
    
    def test_injection_statistics(self):
        """Test injection statistics tracking"""
        self.hybrid.activate_injection()
        
        # Initial statistics
        stats = self.hybrid.get_injection_statistics()
        self.assertEqual(stats['total_injections'], 0)
        
        # Run forward pass
        self.hybrid(self.input_ids)
        
        # Check updated statistics
        stats = self.hybrid.get_injection_statistics()
        self.assertGreater(stats['total_injections'], 0)
        
        # Reset statistics
        self.hybrid.reset_injection_statistics()
        stats = self.hybrid.get_injection_statistics()
        self.assertEqual(stats['total_injections'], 0)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        param_counts = self.hybrid.get_parameter_count()
        
        self.assertIn('total', param_counts)
        self.assertIn('llm_model', param_counts)
        self.assertIn('trm_block', param_counts)
        self.assertIn('projection', param_counts)
        
        # Total should be sum of components
        expected_total = param_counts['llm_model'] + param_counts['trm_block'] + param_counts['projection']
        self.assertEqual(param_counts['total'], expected_total)
    
    def test_device_movement(self):
        """Test moving model to different devices"""
        # This test will only run if CUDA is available
        if torch.cuda.is_available():
            self.hybrid.cuda()
            
            # Check that all components are on CUDA
            self.assertTrue(next(self.hybrid.llm_model.parameters()).is_cuda)
            self.assertTrue(next(self.hybrid.trm_block.parameters()).is_cuda)
            self.assertTrue(next(self.hybrid.projection.parameters()).is_cuda)
            
            # Move back to CPU
            self.hybrid.cpu()
            self.assertFalse(next(self.hybrid.llm_model.parameters()).is_cuda)
    
    def test_training_mode(self):
        """Test training mode switching"""
        # Set to eval mode
        self.hybrid.eval()
        self.assertFalse(self.hybrid.training)
        self.assertFalse(self.hybrid.llm_model.training)
        self.assertFalse(self.hybrid.trm_block.training)
        
        # Set to train mode
        self.hybrid.train()
        self.assertTrue(self.hybrid.training)
        self.assertTrue(self.hybrid.llm_model.training)
        self.assertTrue(self.hybrid.trm_block.training)
    
    def test_save_and_load(self):
        """Test saving and loading hybrid model"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            self.hybrid.save_pretrained(tmp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'hybrid_config.json')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'trm_block.pt')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'projection.pt')))
            
            # Load model
            loaded_hybrid = CombinedTRMLLM.from_pretrained(tmp_dir, llm_model=self.llm)
            
            # Compare configurations
            self.assertEqual(self.hybrid.config.injection_layers, loaded_hybrid.config.injection_layers)
            self.assertEqual(self.hybrid.config.trm_hidden_size, loaded_hybrid.config.trm_hidden_size)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_huggingface_integration(self):
        """Test integration with HuggingFace models"""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            # Load small model for testing
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            
            # Create hybrid model
            config = HybridConfig.from_preset('gpt2')
            hybrid = CombinedTRMLLM(llm_model=model, config=config)
            
            # Test tokenization and generation
            text = "The future of AI is"
            inputs = tokenizer(text, return_tensors='pt', padding=True)
            
            # Generate without injection
            with torch.no_grad():
                outputs_normal = hybrid(**inputs)
            
            # Generate with injection
            hybrid.activate_injection()
            with torch.no_grad():
                outputs_injected = hybrid(**inputs)
            
            # Outputs should be different
            self.assertFalse(torch.allclose(outputs_normal.logits, outputs_injected.logits))
            
        except ImportError:
            self.skipTest("transformers library not available")
    
    def test_end_to_end_training(self):
        """Test end-to-end training capability"""
        llm = MockLLM(hidden_size=256, num_layers=4)
        config = HybridConfig(
            injection_layers=[1, 2],
            trm_hidden_size=128,
            trm_num_heads=4
        )
        hybrid = CombinedTRMLLM(llm_model=llm, config=config)
        
        # Activate injection
        hybrid.activate_injection()
        
        # Create dummy data
        input_ids = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))
        
        # Training step with proper hyperparameters
        optimizer = torch.optim.Adam(hybrid.parameters(), lr=1e-4, weight_decay=0.01)
        
        hybrid.train()
        outputs = hybrid(input_ids)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        self.assertIsNotNone(loss.item())


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    @staticmethod
    def benchmark_injection_overhead(llm_model, config: HybridConfig, num_runs: int = 10):
        """Benchmark the overhead of layer injection"""
        hybrid = CombinedTRMLLM(llm_model=llm_model, config=config)
        
        # Create test input
        batch_size, seq_len = 4, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Benchmark without injection
        hybrid.eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        times_normal = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = hybrid(input_ids)
            times_normal.append(time.time() - start_time)
        
        # Benchmark with injection
        hybrid.activate_injection()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        times_injected = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = hybrid(input_ids)
            times_injected.append(time.time() - start_time)
        
        return {
            'normal_mean': np.mean(times_normal),
            'normal_std': np.std(times_normal),
            'injected_mean': np.mean(times_injected),
            'injected_std': np.std(times_injected),
            'overhead_ratio': np.mean(times_injected) / np.mean(times_normal),
            'overhead_percentage': (np.mean(times_injected) / np.mean(times_normal) - 1) * 100
        }
    
    @staticmethod
    def benchmark_memory_usage(llm_model, config: HybridConfig):
        """Benchmark memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available for memory benchmarking"}
        
        hybrid = CombinedTRMLLM(llm_model=llm_model, config=config).cuda()
        
        # Measure baseline memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated()
        
        # Create test input
        input_ids = torch.randint(0, 1000, (4, 128)).cuda()
        
        # Memory without injection
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = hybrid(input_ids)
        memory_normal = torch.cuda.max_memory_allocated()
        
        # Memory with injection
        hybrid.activate_injection()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = hybrid(input_ids)
        memory_injected = torch.cuda.max_memory_allocated()
        
        return {
            'baseline_mb': baseline_memory / 1024 / 1024,
            'normal_peak_mb': memory_normal / 1024 / 1024,
            'injected_peak_mb': memory_injected / 1024 / 1024,
            'injection_overhead_mb': (memory_injected - memory_normal) / 1024 / 1024,
            'overhead_percentage': ((memory_injected - memory_normal) / memory_normal) * 100
        }


def run_all_tests():
    """Run all tests and return results"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestHybridConfig,
        TestHybridTRMBlock, 
        TestDimensionalBridge,
        TestCombinedTRMLLM,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the full pipeline"""
    
    def test_full_pipeline_inject_enhance_respond(self):
        """Test complete pipeline: inject → enhance → respond"""
        # Create mock LLM
        llm = MockLLM(hidden_size=512, num_layers=8)
        
        # Create config with injection points
        config = HybridConfig(injection_layers=[2, 5])
        
        # Create hybrid model
        model = create_hybrid_model(llm, config)
        
        # Test input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Enable injection
        model.activate_injection()
        
        # Run forward pass (inject → enhance → respond)
        with torch.no_grad():
            output = model(input_ids)
        
        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, llm.hidden_size))
        
        # Check that injection actually happened
        performance = model.get_reasoning_performance()
        self.assertGreater(performance['injection_statistics']['total_injections'], 0)
        
        # Verify enhancement layers were used
        self.assertTrue(any(layer.injection_active for layer in model.injection_layers.values()))
    
    def test_memory_efficiency(self):
        """Test memory usage with different configurations"""
        llm = MockLLM(hidden_size=768, num_layers=12)
        
        # Test with different injection strategies
        configs = [
            HybridConfig(injection_layers=[3, 6, 9]),  # 3 injections
            HybridConfig(injection_layers=[2, 5]),     # 2 injections
            HybridConfig(injection_layers=[4]),        # 1 injection
        ]
        
        for config in configs:
            model = create_hybrid_model(llm, config)
            model.activate_injection()
            
            # Test inference
            input_ids = torch.randint(0, 1000, (1, 32))
            with torch.no_grad():
                output = model(input_ids)
            
            # Should complete without memory errors
            self.assertIsNotNone(output)


def run_benchmarks():
    """Run performance benchmarks"""
    print("Running performance benchmarks...")
    
    # Create test model
    llm = MockLLM(hidden_size=512, num_layers=6)
    config = HybridConfig(
        injection_layers=[2, 4],
        trm_hidden_size=256,
        trm_num_heads=8
    )
    
    # Run benchmarks
    print("\n=== Performance Overhead ===")
    overhead_results = PerformanceBenchmark.benchmark_injection_overhead(llm, config)
    for key, value in overhead_results.items():
        print(f"{key}: {value:.4f}")
    
    print("\n=== Memory Usage ===")
    memory_results = PerformanceBenchmark.benchmark_memory_usage(llm, config)
    for key, value in memory_results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    print("MillennialAi Testing Framework")
    print("=" * 50)
    
    # Run tests
    print("\nRunning unit tests...")
    test_results = run_all_tests()
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    run_benchmarks()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests run: {test_results.testsRun}")
    print(f"Failures: {len(test_results.failures)}")
    print(f"Errors: {len(test_results.errors)}")
    
    if test_results.failures:
        print("\nFailures:")
        for test, traceback in test_results.failures:
            print(f"  - {test}: {traceback}")
    
    if test_results.errors:
        print("\nErrors:")
        for test, traceback in test_results.errors:
            print(f"  - {test}: {traceback}")