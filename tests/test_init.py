"""
Tests for millennial_ai.__init__ module

This module contains unit tests for the package initialization,
enterprise presets, and utility functions.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

from millennial_ai import (
    __version__,
    __author__,
    __email__,
    __license__,
    ENTERPRISE_PRESETS,
    get_enterprise_info,
    quick_start_enterprise,
    CombinedTRMLLM,
    HybridConfig,
    PresetConfigs,
    HybridTRMBlock,
    DimensionalBridge,
    create_dimensional_bridge,
)


class TestPackageInit(unittest.TestCase):
    """Test cases for package initialization"""

    def test_version_info(self):
        """Test version and metadata constants"""
        self.assertEqual(__version__, "1.0.0")
        self.assertEqual(__author__, "Jovan Blango")
        self.assertEqual(__email__, "izreal1990@gmail.com")
        self.assertEqual(__license__, "MIT")

    def test_enterprise_presets(self):
        """Test enterprise presets dictionary"""
        self.assertIsInstance(ENTERPRISE_PRESETS, dict)
        self.assertIn('llama2-70b', ENTERPRISE_PRESETS)
        self.assertIn('llama3-70b', ENTERPRISE_PRESETS)
        self.assertIn('gpt4-scale', ENTERPRISE_PRESETS)
        self.assertIn('production', ENTERPRISE_PRESETS)

        # Test preset descriptions
        self.assertIn('85B total', ENTERPRISE_PRESETS['llama2-70b'])
        self.assertIn('90B total', ENTERPRISE_PRESETS['llama3-70b'])
        self.assertIn('2T+', ENTERPRISE_PRESETS['gpt4-scale'])

    def test_get_enterprise_info(self):
        """Test get_enterprise_info function"""
        info = get_enterprise_info()

        self.assertIsInstance(info, dict)
        self.assertEqual(info['version'], "1.0.0")
        self.assertIn('LLaMA-2-70B', info['supported_models'])
        self.assertIn('70B - 2T+', info['parameter_ranges'])
        self.assertIn('8x A100 80GB minimum', info['hardware_requirements'])
        self.assertIn('Multi-GPU distributed training', info['enterprise_features'])

    def test_quick_start_enterprise(self):
        """Test quick_start_enterprise function output"""
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            quick_start_enterprise()

        output = captured_output.getvalue()

        # Check that key information is present
        self.assertIn('MillennialAi Enterprise Quick Start', output)
        self.assertIn('AutoModelForCausalLM.from_pretrained', output)
        self.assertIn('HybridConfig.from_preset', output)
        self.assertIn('CombinedTRMLLM', output)
        self.assertIn('activate_injection', output)
        self.assertIn('85B parameter hybrid processing', output)
        self.assertIn('izreal1990@gmail.com', output)

    def test_imports_available(self):
        """Test that all expected classes are importable"""
        # Test that classes can be instantiated (basic smoke test)
        self.assertTrue(callable(CombinedTRMLLM))
        self.assertTrue(callable(HybridConfig))
        self.assertTrue(callable(PresetConfigs))
        self.assertTrue(callable(HybridTRMBlock))
        self.assertTrue(callable(DimensionalBridge))
        self.assertTrue(callable(create_dimensional_bridge))

    def test_enterprise_presets_completeness(self):
        """Test that enterprise presets cover all expected configurations"""
        expected_presets = [
            'llama2-70b', 'llama3-70b', 'gpt4-scale',
            'multimodal', 'production', 'research'
        ]

        for preset in expected_presets:
            with self.subTest(preset=preset):
                self.assertIn(preset, ENTERPRISE_PRESETS)
                # Ensure description is not empty
                self.assertGreater(len(ENTERPRISE_PRESETS[preset]), 0)


class TestPackageIntegration(unittest.TestCase):
    """Test cases for package-level integration"""

    def test_config_integration(self):
        """Test that config classes work together"""
        # Test that we can create a config from preset
        config = HybridConfig.from_preset('minimal')
        self.assertIsInstance(config, HybridConfig)

        # Test that preset configs are accessible
        llama_config = PresetConfigs.llama_2_70b_enterprise()
        self.assertIsInstance(llama_config, HybridConfig)

    def test_enterprise_workflow(self):
        """Test a basic enterprise workflow"""
        # This is a smoke test to ensure the basic workflow works
        # In a real scenario, this would require actual models

        # Test config creation
        config = HybridConfig.from_preset('minimal')
        self.assertIsInstance(config, HybridConfig)

        # Test that config has expected attributes
        self.assertTrue(hasattr(config, 'injection_layers'))
        self.assertTrue(hasattr(config, 'trm_hidden_size'))
        self.assertTrue(hasattr(config, 'trm_num_heads'))

        # Test enterprise info
        info = get_enterprise_info()
        self.assertIn('version', info)
        self.assertIn('supported_models', info)


if __name__ == '__main__':
    unittest.main()