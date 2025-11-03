<div align="center">
  
<img src="logo.png" alt="MillennialAi Logo" width="200"/>

# MillennialAi

**Layer Injection Architecture for Hybrid Neural Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![Azure](https://img.shields.io/badge/Azure-Container%20Apps-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/en-us/products/container-apps)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success?logo=azuredevops)](https://millennialai-app.lemongrass-179d661f.eastus2.azurecontainerapps.io/)

</div>

## Overview

MillennialAi integrates Tiny Recursion Models (TRM) into Large Language Models using PyTorch forward hooks for layer injection.

The system combines:
- Neural reasoning via adaptive PyTorch networks
- Knowledge integration through Ollama LLM (llama3:8b)
- Layer injection using forward hooks (zero LLM modification)
- Production API with FastAPI deployed on Azure

### Core Concept

This approach uses **forward hooks** to inject TRM processing directly into LLM hidden layers, creating a hybrid system that:

- ✅ **Zero Model Modification**: Original LLM remains unchanged
- ✅ **Dynamic Activation**: Injection can be toggled on/off at runtime
- ✅ **Gradient Preservation**: Full backpropagation through hybrid architecture
- ✅ **Multi-Layer Support**: Inject at multiple layers simultaneously
- ✅ **Framework Agnostic**: Works with any PyTorch transformer model

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   System Architecture                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Input → FastAPI → Hybrid Brain Controller            │
│                         │                              │
│            ┌────────────┴────────────┐                 │
│            ▼                         ▼                 │
│    ┌──────────────┐          ┌──────────────┐         │
│    │ Neural Brain │          │    Ollama    │         │
│    │  (PyTorch)   │          │  llama3:8b   │         │
│    │              │          │              │         │
│    │ • Complexity │          │ • Knowledge  │         │
│    │ • Steps 1-8  │          │ • Generation │         │
│    │ • Convergence│          │ • Context    │         │
│    └──────────────┘          └──────────────┘         │
│            │                         │                 │
│            └────────────┬────────────┘                 │
│                         ▼                              │
│              Response Synthesis                        │
│                                                        │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│              Layer Injection Mechanism                 │
├────────────────────────────────────────────────────────┤
│                                                        │
│  LLM Layer N-1                                         │
│       │                                                │
│       ▼                                                │
│  [Forward Hook] ────→ Inject TRM Processing           │
│       │                      │                         │
│       │                      ▼                         │
│       │              ┌──────────────┐                  │
│       │              │     TRM      │                  │
│       │              │ • Attention  │                  │
│       │              │ • Recursion  │                  │
│       │              │ • Transform  │                  │
│       │              └──────────────┘                  │
│       │                      │                         │
│       └──────────┬───────────┘                         │
│                  ▼                                     │
│  LLM Layer N (Enhanced Hidden States)                 │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/izreal1990-collab/MillennialAi.git
cd MillennialAi

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

## Core Components

### 1. HybridConfig
Configuration system with presets and validation:

```python
# Use preset configurations
config = HybridConfig.from_preset('production')

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

## Performance

### Production Metrics

Live deployment on Azure Container Apps (4 CPU, 8GB RAM):

**Response Times:**
- Average: 733ms
- P50: 733ms  
- P95: 734ms
- P99: 734ms

**Configuration:**
- CPU-only inference (no GPU)
- Ollama llama3:8b model
- 120s timeout
- Adaptive reasoning steps (1-8 based on complexity)

*Note: Performance varies based on query complexity and knowledge enhancement requirements.*

## Advanced Features

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

## Examples

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

## Testing

Run comprehensive test suite:

```bash
# All tests
python -m pytest millennial_ai/tests/

# Quick test
python examples/basic_usage.py

# Benchmark performance
python -c "from millennial_ai.tests import run_benchmarks; run_benchmarks()"
```

## Documentation

### Configuration Presets

- **minimal**: Lightweight injection (single layer)
- **production**: Balanced for performance and efficiency
- **llama2-70b**: Optimized for LLaMA-2 70B models
- **llama3-70b**: Optimized for LLaMA-3 70B models
- **gpt4-scale**: Ultra-scale configuration for largest models
- **multimodal**: Enterprise multimodal model configuration
- **research**: Experimental maximum capability configuration

### Projection Types

- **linear**: Simple linear transformation
- **adaptive**: Learnable adaptive projection
- **residual**: Residual connections preserved

### Hook Management

The library automatically manages PyTorch forward hooks:
- Hooks are registered/removed as needed
- No memory leaks or stale hooks
- Exception handling for failed injections

## Contributing

Contributions are welcome! 

### Development Setup

```bash
git clone https://github.com/izreal1990-collab/MillennialAi.git
cd MillennialAi

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Quality & SonarQube

MillennialAi uses SonarQube for continuous code quality analysis. Scans run automatically on commits, PRs, and CI/CD pipelines.

#### Local SonarQube Setup

1. **Install SonarQube Scanner**:
   ```bash
   # Download from: https://docs.sonarsource.com/sonarqube/latest/analyzing-source-code/scanners/sonarscanner/
   # Or use the included scanner
   export PATH="$PATH:$(pwd)/sonar-scanner-4.8.0.2856-linux/bin"
   ```

2. **Set up SonarCloud Token**:
   ```bash
   # Get token from: https://sonarcloud.io/account/security/
   export SONAR_TOKEN=your_token_here
   export SONAR_HOST_URL=https://sonarcloud.io
   ```

3. **Run Local Scan**:
   ```bash
   # Quick scan script
   ./scripts/run_sonar_scan.sh
   
   # Or manual scan
   sonar-scanner
   ```

#### CI/CD Integration

- **GitHub Actions**: Automatic scans on push/PR via `.github/workflows/ci-cd.yml`
- **Azure Pipelines**: Use `azure-pipelines.yml` for Azure DevOps integration
- **Pre-commit**: Local scans on commit via `.pre-commit-config.yaml`

#### Quality Gates

The project enforces these quality standards:
- **Test Coverage**: >80% target
- **Code Smells**: <10 per file
- **Duplications**: <3% of codebase
- **Security Issues**: 0 critical/blocking issues

View results at: [SonarCloud Dashboard](https://sonarcloud.io/dashboard?id=millennial-ai)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon:
- **Trill-AI Project**: Original TRM architecture and recursive processing concepts
- **PyTorch Team**: For the excellent forward hook mechanism
- **HuggingFace**: For the transformers library and model ecosystem
- **Research Community**: For foundational work in attention mechanisms and neural architecture

## Citation

If you use MillennialAi in your research, please cite:

```bibtex
@software{millennial_ai_2024,
  title={MillennialAi: Layer Injection Architecture for Hybrid Neural Networks},
  author={Jovan Blango},
  year={2024},
  url={https://github.com/izreal1990-collab/MillennialAi}
}
```

## Future Roadmap

- [ ] Support for more transformer architectures
- [ ] Optimized CUDA kernels for TRM operations
- [ ] Model compression techniques
- [ ] Integration with major ML frameworks

## Support

- **Issues**: [GitHub Issues](https://github.com/izreal1990-collab/MillennialAi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/izreal1990-collab/MillennialAi/discussions)
- **Email**: izreal1990@gmail.com

---

<div align="center">
<strong>Layer Injection Architecture for TRM-LLM Integration</strong>
</div>
