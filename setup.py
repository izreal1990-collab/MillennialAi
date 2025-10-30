"""
Setup configuration for MillenialAi

MillenialAi implements revolutionary Layer Injection Architecture for seamlessly
integrating Tiny Recursion Models into Large Language Models using PyTorch
forward hooks, enabling hybrid neural architectures without model modification.
"""

from setuptools import setup, find_packages
import os


def read_file(filename):
    """Read file contents"""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()


def read_requirements(filename):
    """Read requirements from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []


# Read long description from README
long_description = read_file('README.md')

# Read requirements
install_requires = read_requirements('requirements.txt')

setup(
    name="millennial-ai",
    version="1.0.0",
    
    # Author and contact information
    author="Jovan Blango",
    author_email="izreal1990@gmail.com",
    
    # Project description
    description="Revolutionary Layer Injection Architecture for Hybrid Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Project URLs
    url="https://github.com/izreal1990-collab/MillenialAi",
    project_urls={
        "Bug Reports": "https://github.com/izreal1990-collab/MillenialAi/issues",
        "Source": "https://github.com/izreal1990-collab/MillenialAi",
        "Documentation": "https://github.com/izreal1990-collab/MillenialAi/blob/main/README.md",
    },
    
    # Package configuration
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*']),
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=install_requires,
    
    # Optional dependencies for different use cases
    extras_require={
        'huggingface': [
            'transformers>=4.20.0',
            'tokenizers>=0.12.1',
            'datasets>=2.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
            'pre-commit>=2.20.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=0.18.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
            'ipywidgets>=7.7.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
        ],
        'benchmark': [
            'memory-profiler>=0.60.0',
            'psutil>=5.9.0',
            'matplotlib>=3.5.0',
            'pandas>=1.4.0',
        ],
        'all': [
            # Include all optional dependencies
            'transformers>=4.20.0',
            'tokenizers>=0.12.1',
            'datasets>=2.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
            'pre-commit>=2.20.0',
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'myst-parser>=0.18.0',
            'jupyter>=1.0.0',
            'notebook>=6.4.0',
            'ipywidgets>=7.7.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'memory-profiler>=0.60.0',
            'psutil>=5.9.0',
            'pandas>=1.4.0',
        ],
    },
    
    # Package data
    include_package_data=True,
    package_data={
        'millennial_ai': [
            'config/*.json',
            'docs/*.md',
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'millennial-ai-test=millennial_ai.tests:run_all_tests',
            'millennial-ai-benchmark=millennial_ai.tests:run_benchmarks',
        ],
    },
    
    # Classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English",
        
        # Framework
        "Framework :: PyTorch",
    ],
    
    # Keywords for package discovery
    keywords=[
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural networks",
        "transformers",
        "language models",
        "hybrid models",
        "layer injection",
        "pytorch",
        "recursion",
        "attention",
        "nlp",
        "ai",
        "ml",
        "research",
    ],
    
    # Zip safety
    zip_safe=False,
    
    # Test configuration
    test_suite='tests',
    tests_require=[
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
    ],
    
    # Additional metadata
    platforms=['any'],
    license="MIT",
)


# Post-installation message
print("""
🎉 MillenialAi has been installed successfully!

This package implements the revolutionary Layer Injection Architecture
for hybrid neural networks, enabling seamless integration of Tiny
Recursion Models into existing Large Language Models.

Quick Start:
-----------
from millennial_ai.core.hybrid_model import create_hybrid_model
from millennial_ai.config.config import HybridConfig

# Create hybrid model
config = HybridConfig(injection_layers=[2, 4, 6])
hybrid = create_hybrid_model(your_llm_model, **config.to_dict())

# Activate injection
hybrid.activate_injection()

# Use normally
outputs = hybrid(input_ids)

Documentation:
-------------
• GitHub: https://github.com/izreal1990-collab/MillenialAi
• Examples: See examples/ directory
• Tests: Run 'millennial-ai-test'
• Benchmarks: Run 'millennial-ai-benchmark'

For help or issues, visit:
https://github.com/izreal1990-collab/MillenialAi/issues

Happy researching! 🚀
""")