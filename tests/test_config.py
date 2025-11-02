"""
Tests for millennial_ai.config module

This module contains comprehensive unit tests for configuration classes
and constants to ensure proper validation and functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
import warnings

# Import only config components to avoid torch import issues
from millennial_ai.config.config import HybridConfig, PresetConfigs
from millennial_ai.config.constants import (
    DEFAULT_TRM_HIDDEN_SIZE,
    DEFAULT_TRM_NUM_HEADS,
    DEFAULT_TRM_FF_HIDDEN_SIZE,
    DEFAULT_TRM_NUM_LAYERS,
    DEFAULT_NUM_RECURSION_STEPS,
    DEFAULT_DROPOUT,
    DEFAULT_LAYER_NORM_EPS,
    DEFAULT_RECURSION_DROPOUT,
    DEFAULT_INJECTION_STRENGTH,
)


class TestHybridConfig(unittest.TestCase):
    """Test cases for HybridConfig class"""

    def setUp(self):
        """Set up test fixtures"""
        self.valid_config = HybridConfig(
            injection_layers=[4, 8, 12],
            trm_hidden_size=512,
            trm_num_heads=8,
            trm_ff_hidden_size=2048,
            trm_num_layers=2,
            num_recursion_steps=3,
            dropout=0.1,
            layer_norm_eps=1e-5,
            recursion_dropout=0.05,
            injection_strength=1.0,
        )

    def test_valid_config_creation(self):
        """Test creating a valid configuration"""
        config = HybridConfig()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.injection_layers, [4, 8])
        self.assertEqual(config.trm_hidden_size, DEFAULT_TRM_HIDDEN_SIZE)
        self.assertEqual(config.trm_num_heads, DEFAULT_TRM_NUM_HEADS)

    def test_config_validation_positive_values(self):
        """Test validation of positive values"""
        with self.assertRaises(ValueError):
            HybridConfig(trm_hidden_size=0)

        with self.assertRaises(ValueError):
            HybridConfig(trm_num_heads=0)

    def test_config_validation_hidden_size_divisible_by_heads(self):
        """Test validation that hidden_size is divisible by num_heads"""
        with self.assertRaises(ValueError):
            HybridConfig(trm_hidden_size=100, trm_num_heads=3)  # 100 % 3 != 0

    def test_config_validation_injection_layers(self):
        """Test validation of injection layers"""
        with self.assertRaises(TypeError):
            HybridConfig(injection_layers="invalid")

        with self.assertRaises(ValueError):
            HybridConfig(injection_layers=[-1, 0, 1])

    def test_from_preset_valid(self):
        """Test loading configuration from valid preset"""
        config = HybridConfig.from_preset('minimal')
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 256)

    def test_from_preset_invalid(self):
        """Test loading configuration from invalid preset"""
        with self.assertRaises(ValueError):
            HybridConfig.from_preset('invalid_preset')

    def test_constants_integration(self):
        """Test that constants are properly integrated"""
        config = HybridConfig()
        self.assertEqual(config.trm_hidden_size, DEFAULT_TRM_HIDDEN_SIZE)
        self.assertEqual(config.trm_num_heads, DEFAULT_TRM_NUM_HEADS)
        self.assertEqual(config.trm_ff_hidden_size, DEFAULT_TRM_FF_HIDDEN_SIZE)
        self.assertEqual(config.trm_num_layers, DEFAULT_TRM_NUM_LAYERS)
        self.assertEqual(config.num_recursion_steps, DEFAULT_NUM_RECURSION_STEPS)
        self.assertEqual(config.dropout, DEFAULT_DROPOUT)
        self.assertEqual(config.layer_norm_eps, DEFAULT_LAYER_NORM_EPS)
        self.assertEqual(config.recursion_dropout, DEFAULT_RECURSION_DROPOUT)
        self.assertEqual(config.injection_strength, DEFAULT_INJECTION_STRENGTH)


class TestPresetConfigs(unittest.TestCase):
    """Test cases for PresetConfigs class"""

    def test_minimal_preset(self):
        """Test minimal preset configuration"""
        config = PresetConfigs.minimal()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 4096)
        self.assertEqual(config.trm_num_heads, 32)
        self.assertEqual(config.injection_layers, [32])

    def test_llama_2_70b_enterprise_preset(self):
        """Test LLaMA-2-70B enterprise preset"""
        config = PresetConfigs.llama_2_70b_enterprise()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 8192)
        self.assertEqual(config.trm_num_heads, 64)
        self.assertEqual(config.injection_layers, [8, 16, 24, 32, 40, 48, 56, 64, 72])

    def test_llama_3_70b_revolutionary_preset(self):
        """Test LLaMA-3-70B revolutionary preset"""
        config = PresetConfigs.llama_3_70b_revolutionary()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 12288)
        self.assertEqual(config.trm_num_heads, 96)

    def test_gpt_4_scale_ultra_preset(self):
        """Test GPT-4 scale ultra preset"""
        config = PresetConfigs.gpt_4_scale_ultra()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 16384)
        self.assertEqual(config.trm_num_heads, 128)

    def test_multimodal_foundation_preset(self):
        """Test multimodal foundation preset"""
        config = PresetConfigs.multimodal_foundation()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 14336)
        self.assertEqual(config.trm_num_heads, 112)

    def test_research_experimental_preset(self):
        """Test research experimental preset"""
        config = PresetConfigs.research_experimental()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 20480)
        self.assertEqual(config.trm_num_heads, 160)

    def test_production_optimized_preset(self):
        """Test production optimized preset"""
        config = PresetConfigs.production_optimized()
        self.assertIsInstance(config, HybridConfig)
        self.assertEqual(config.trm_hidden_size, 6144)
        self.assertEqual(config.trm_num_heads, 48)

    def test_preset_consistency(self):
        """Test that all presets return valid configurations"""
        presets = [
            'minimal', 'llama2-70b', 'llama3-70b', 'gpt4-scale',
            'multimodal', 'research', 'production'
        ]

        for preset in presets:
            with self.subTest(preset=preset):
                config = HybridConfig.from_preset(preset)
                self.assertIsInstance(config, HybridConfig)
                # Validate basic constraints
                self.assertGreater(config.trm_hidden_size, 0)
                self.assertGreater(config.trm_num_heads, 0)
                self.assertEqual(config.trm_hidden_size % config.trm_num_heads, 0)


class TestConstants(unittest.TestCase):
    """Test cases for constants module"""

    def test_constants_defined(self):
        """Test that all required constants are defined"""
        # Import constants to ensure they exist
        from millennial_ai.config import constants

        # Check that key constants exist
        self.assertTrue(hasattr(constants, 'DEFAULT_TRM_HIDDEN_SIZE'))
        self.assertTrue(hasattr(constants, 'DEFAULT_TRM_NUM_HEADS'))
        self.assertTrue(hasattr(constants, 'DEFAULT_DROPOUT'))
        self.assertTrue(hasattr(constants, 'DEFAULT_WINDOW_WIDTH'))
        self.assertTrue(hasattr(constants, 'DEFAULT_WINDOW_HEIGHT'))

    def test_constant_values(self):
        """Test that constants have expected values"""
        from millennial_ai.config import constants

        self.assertEqual(constants.DEFAULT_TRM_HIDDEN_SIZE, 512)
        self.assertEqual(constants.DEFAULT_TRM_NUM_HEADS, 8)
        self.assertEqual(constants.DEFAULT_DROPOUT, 0.1)
        self.assertEqual(constants.DEFAULT_WINDOW_WIDTH, 1400)
        self.assertEqual(constants.DEFAULT_WINDOW_HEIGHT, 900)


if __name__ == '__main__':
    unittest.main()