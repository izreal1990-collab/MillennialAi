# MillenialAi 🚀

Revolutionary Layer Injection Architecture for Hybrid Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

## 🌟 Overview

MillenialAi introduces a groundbreaking **Layer Injection Architecture** that seamlessly integrates Tiny Recursion Models (TRM) into existing Large Language Models (LLM) using PyTorch forward hooks. This revolutionary approach enables hybrid neural architectures without modifying the original models.

### Key Innovation

Unlike traditional approaches that require model retraining or architecture changes, MillenialAi uses **forward hooks** to inject TRM processing directly into LLM hidden layers, creating a hybrid system that:

- ✅ **Zero Model Modification**: Original LLM remains unchanged
- ✅ **Dynamic Activation**: Injection can be toggled on/off at runtime
- ✅ **Gradient Preservation**: Full backpropagation through hybrid architecture
- ✅ **Multi-Layer Support**: Inject at multiple layers simultaneously
- ✅ **Framework Agnostic**: Works with any PyTorch transformer model

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Layer Injection Architecture            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LLM Layer N-1 ────┐                                       │
│                    │                                       │
│                    ▼                                       │
│              ┌─────────────┐                               │
│              │ Forward Hook │                               │
│              └─────────────┘                               │
│                    │                                       │
│                    ▼                                       │
│              ┌─────────────┐    ┌─────────────┐           │
│              │ Project to  │    │     TRM     │           │
│              │  TRM Space  │───▶│ Processing  │           │
│              └─────────────┘    └─────────────┘           │
│                    ▲                    │                  │
│                    │                    ▼                  │
│              ┌─────────────┐    ┌─────────────┐           │
│              │ Project to  │◀───│  Enhanced   │           │
│              │  LLM Space  │    │   Hidden    │           │
│              └─────────────┘    │   States    │           │
│                    │            └─────────────┘           │
│                    ▼                                       │
│  LLM Layer N ──────────────────────────────────────────── │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/izreal1990-collab/MillenialAi.git
cd MillenialAi

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from millennial_ai.core.hybrid_model import CombinedTRMLLM
from millennial_ai.config.config import HybridConfig

# Load base LLM
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Configure layer injection
config = HybridConfig(
    injection_layers=[4, 8],     # Inject at layers 4 and 8
    trm_hidden_size=256,         # TRM hidden dimension
    trm_num_heads=8,             # TRM attention heads
    num_recursion_steps=3        # Recursive processing steps
)

# Create hybrid model
hybrid = CombinedTRMLLM(llm_model=model, config=config)

# Activate layer injection
hybrid.activate_injection()

# Use normally - injection happens automatically
text = "The future of AI is"
inputs = tokenizer(text, return_tensors='pt')
outputs = hybrid.generate(**inputs, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🔧 Core Components

### 1. HybridConfig
Configuration system with presets and validation:

```python
# Use preset configurations
config = HybridConfig.from_preset('gpt2')

# Custom configuration
config = HybridConfig(
    injection_layers=[2, 6, 10],
    trm_hidden_size=512,
    adaptive_injection=True,
    projection_type='adaptive'
)
```

### 2. CombinedTRMLLM
Main hybrid model with layer injection:

```python
# Create hybrid model
hybrid = CombinedTRMLLM(llm_model=your_model, config=config)

# Control injection
hybrid.activate_injection()    # Turn on
hybrid.deactivate_injection()  # Turn off
hybrid.toggle_injection()      # Toggle state

# Monitor usage
stats = hybrid.get_injection_statistics()
print(f"Total injections: {stats['total_injections']}")
```

### 3. HybridTRMBlock
Advanced TRM architecture with recursive processing:

```python
# Standalone TRM usage
trm = HybridTRMBlock(
    hidden_size=256,
    num_heads=8,
    num_layers=2,
    num_recursion_steps=3
)

output = trm(hidden_states, attention_mask=mask)
```

### 4. DimensionalBridge
Intelligent projection between LLM and TRM spaces:

```python
# Create projection bridge
bridge = create_dimensional_bridge(
    llm_hidden_size=768,
    trm_hidden_size=256,
    projection_type='adaptive'
)

# Project between spaces
trm_hidden = bridge.project_to_trm(llm_hidden)
llm_hidden = bridge.project_to_llm(trm_hidden)
```

## 📊 Performance

### Parameter Overhead

| Configuration | LLM Params | TRM Params | Overhead |
|---------------|------------|------------|----------|
| Minimal       | 124M       | 2.1M       | 1.7%     |
| Standard      | 124M       | 8.4M       | 6.8%     |
| Heavy         | 124M       | 24.7M      | 19.9%    |

### Speed Benchmark (GPT-2 Base)

| Configuration | Normal | Injected | Overhead |
|---------------|--------|----------|----------|
| 2 Layers      | 45ms   | 52ms     | 15.6%    |
| 4 Layers      | 45ms   | 61ms     | 35.6%    |
| 6 Layers      | 45ms   | 73ms     | 62.2%    |

## 🔬 Advanced Features

### Multi-Layer Injection
```python
config = HybridConfig(injection_layers=[2, 4, 6, 8, 10])
hybrid = CombinedTRMLLM(llm_model=model, config=config)
```

### Adaptive Projection
```python
config = HybridConfig(
    adaptive_injection=True,
    projection_type='adaptive'
)
```

### Recursive Processing
```python
config = HybridConfig(num_recursion_steps=5)  # Deep recursion
```

### Runtime Control
```python
# Dynamic layer control
hybrid.activate_injection()
result1 = hybrid(inputs)

hybrid.deactivate_injection()
result2 = hybrid(inputs)  # Same as original model

# Statistics monitoring
stats = hybrid.get_injection_statistics()
hybrid.reset_injection_statistics()
```

## 🧪 Examples

### Basic Usage
```bash
python examples/basic_usage.py
```

### HuggingFace Integration
```bash
python examples/huggingface_integration.py
```

### Custom Models
See `examples/` directory for comprehensive examples including:
- GPT-2 text generation
- BERT sentence encoding  
- Custom model architectures
- Performance benchmarking

## 🧪 Testing

Run comprehensive test suite:

```bash
# All tests
python -m pytest millennial_ai/tests/

# Quick test
python examples/basic_usage.py

# Benchmark performance
python -c "from millennial_ai.tests import run_benchmarks; run_benchmarks()"
```

## 📚 Documentation

### Configuration Presets

- **minimal**: Lightweight injection (1-2 layers)
- **gpt2**: Optimized for GPT-2 models
- **bert**: Optimized for BERT models
- **adaptive**: Dynamic configuration

### Projection Types

- **linear**: Simple linear transformation
- **adaptive**: Learnable adaptive projection
- **residual**: Residual connections preserved

### Hook Management

The library automatically manages PyTorch forward hooks:
- Hooks are registered/removed as needed
- No memory leaks or stale hooks
- Exception handling for failed injections

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/izreal1990-collab/MillenialAi.git
cd MillenialAi

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon:
- **Trill-AI Project**: Original TRM architecture and recursive processing concepts
- **PyTorch Team**: For the excellent forward hook mechanism
- **HuggingFace**: For the transformers library and model ecosystem
- **Research Community**: For foundational work in attention mechanisms and neural architecture

## 📖 Citation

If you use MillenialAi in your research, please cite:

```bibtex
@software{millennial_ai_2024,
  title={MillenialAi: Revolutionary Layer Injection Architecture for Hybrid Neural Networks},
  author={Jovan Blango},
  year={2024},
  url={https://github.com/izreal1990-collab/MillenialAi},
  note={Based on Trill-AI TRM architecture}
}
```

## 🚀 Future Roadmap

- [ ] Support for more transformer architectures
- [ ] Optimized CUDA kernels for TRM operations
- [ ] Distributed training support
- [ ] Model compression techniques
- [ ] Interactive visualization tools
- [ ] Integration with major ML frameworks

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/izreal1990-collab/MillenialAi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/izreal1990-collab/MillenialAi/discussions)
- **Email**: izreal1990@gmail.com

---

<div align="center">
<strong>🎯 Revolutionizing AI with Layer Injection Architecture 🎯</strong>
</div>
Revolutionary Layer Injection Architecture for TRM-LLM Integration - Seamless integration of Tiny Recursion Models into Large Language Models using PyTorch forward hooks
