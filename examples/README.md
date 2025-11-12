# MillennialAi Examples

This directory contains practical examples demonstrating the Layer Injection Architecture for hybrid neural networks. Each example showcases different aspects and use cases of MillennialAi.

## Overview

The examples are organized from beginner-friendly to enterprise-scale, helping you learn the framework progressively.

## Examples

### 1. Basic Usage (`basic_usage.py`)

**Purpose**: Introduction to core MillennialAi concepts with a simple mock model.

**What you'll learn**:
- Creating a hybrid model with layer injection
- Configuring injection layers and TRM architecture
- Activating/deactivating injection dynamically
- Comparing outputs with and without injection
- Monitoring injection statistics

**Run**:
```bash
python examples/basic_usage.py
```

**Requirements**: PyTorch only

**Output**: Demonstrates how layer injection modifies model outputs and shows parameter overhead.

---

### 2. HuggingFace Integration (`huggingface_integration.py`)

**Purpose**: Real-world integration with HuggingFace transformer models.

**What you'll learn**:
- Using MillennialAi with GPT-2, BERT, and other HuggingFace models
- Text generation with layer injection
- Comparing injected vs. baseline outputs
- Working with real tokenizers and models

**Run**:
```bash
python examples/huggingface_integration.py
```

**Requirements**: 
- `transformers` library
- Internet connection (first run downloads models)

**Models tested**:
- GPT-2 for text generation
- BERT for encoding
- DistilGPT2 for efficient inference

---

### 3. Enterprise 70B Example (`enterprise_70b_example.py`)

**Purpose**: Demonstrates enterprise-scale deployment with massive 70B+ parameter models.

**What you'll learn**:
- Working with LLaMA-2-70B scale architectures
- Enterprise configuration presets
- Memory and performance considerations
- Multi-GPU distributed training concepts

**Run**:
```bash
python examples/enterprise_70b_example.py
```

**Requirements**: 
- High computational resources (simulated with mock models)
- Multiple GPUs for real deployment

**Key configurations**:
- LLaMA-2-70B + 15B TRM = 85B hybrid
- LLaMA-3-70B + 20B TRM = 90B hybrid
- GPT-4 scale configurations

**Note**: This example uses mock models to demonstrate the architecture without requiring actual 70B models.

---

### 4. Beginner Training Example (`beginner_training_example.py`)

**Purpose**: End-to-end training workflow for beginners.

**What you'll learn**:
- Setting up a training loop
- Preparing data for hybrid models
- Optimizing hyperparameters
- Saving and loading trained models
- Monitoring training progress

**Run**:
```bash
python examples/beginner_training_example.py
```

**Requirements**: PyTorch, sample dataset

---

### 5. Democratization Demo (`democratization_demo.py`)

**Purpose**: Illustrates the cost and accessibility benefits of MillennialAi.

**What you'll learn**:
- Before/after comparison of enterprise AI access
- Market impact of democratization
- Real-world scenarios across different organizations

**Run**:
```bash
python examples/democratization_demo.py
```

**Requirements**: None (information display only)

**Output**: Shows how MillennialAi makes enterprise AI accessible to organizations with limited budgets.

---

### 6. Cost Calculator (`cost_calculator.py`)

**Purpose**: Calculate and compare training costs for different approaches.

**What you'll learn**:
- Traditional training costs vs. MillennialAi approach
- Hardware requirements for different model sizes
- ROI analysis for enterprise deployment
- Cost optimization strategies

**Run**:
```bash
python examples/cost_calculator.py
```

**Requirements**: None (calculation utility)

**Scenarios**:
- Training models from scratch
- Using MillennialAi layer injection
- Comparative analysis across model sizes

---

### 7. Sample Data (`sample_data.py`)

**Purpose**: Utilities for generating and loading sample datasets.

**What you'll learn**:
- Creating synthetic datasets for testing
- Data preprocessing for hybrid models
- Dataset formats and requirements

**Usage**:
```python
from examples.sample_data import generate_sample_data

# Generate sample training data
data = generate_sample_data(num_samples=1000, seq_len=128)
```

---

## Quick Start Guide

### Running Your First Example

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start with basic usage**:
   ```bash
   python examples/basic_usage.py
   ```

3. **Try HuggingFace integration** (if you have `transformers` installed):
   ```bash
   python examples/huggingface_integration.py
   ```

### Recommended Learning Path

1. **Beginner**: Start with `basic_usage.py`
2. **Intermediate**: Move to `huggingface_integration.py`
3. **Advanced**: Explore `enterprise_70b_example.py`
4. **Planning**: Use `cost_calculator.py` for deployment planning

## Common Issues and Solutions

### Issue: Module not found

**Solution**: Install the package in development mode:
```bash
pip install -e .
```

### Issue: GPU out of memory

**Solution**: Reduce batch size or use smaller models:
```python
config = HybridConfig.from_preset('minimal')  # Use minimal preset
```

### Issue: HuggingFace models not downloading

**Solution**: Check internet connection or use cached models:
```bash
export TRANSFORMERS_CACHE=/path/to/cache
```

## Performance Considerations

### Memory Usage

- **Basic examples**: ~2GB RAM
- **HuggingFace examples**: ~4-8GB RAM (model dependent)
- **Enterprise examples**: 200GB+ GPU memory (for real 70B models)

### Execution Time

- **Basic examples**: < 1 minute
- **HuggingFace examples**: 2-5 minutes (first run includes model download)
- **Training examples**: 10-30 minutes

## Next Steps

After running the examples:

1. **Explore the API**: Check the main README for detailed API documentation
2. **Run tests**: `python -m pytest tests/`
3. **Build your application**: Adapt examples to your use case
4. **Join the community**: Visit [GitHub Discussions](https://github.com/izreal1990-collab/MillennialAi/discussions)

## Additional Resources

- **Main README**: [../README.md](../README.md) - Comprehensive documentation
- **API Reference**: Check docstrings in `millennial_ai/` modules
- **Configuration Guide**: See `millennial_ai/config/config.py`
- **Test Suite**: See `tests/` for more usage examples

## Contributing Examples

Have an interesting use case? We welcome example contributions!

1. Create a new Python file following the existing structure
2. Add comprehensive docstrings and comments
3. Update this README with your example
4. Submit a pull request

## Support

For questions or issues:
- **Issues**: [GitHub Issues](https://github.com/izreal1990-collab/MillennialAi/issues)
- **Email**: izreal1990@gmail.com
- **Discussions**: [GitHub Discussions](https://github.com/izreal1990-collab/MillennialAi/discussions)
