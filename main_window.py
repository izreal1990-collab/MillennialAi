"""
MillennialAi Main Window - Unified GUI for Cognitive Enhancement App

A comprehensive PyQt5-based interface tailored to the Layer Injection Architecture.
Integrates chat, visualization, configuration, and enterprise tools.
"""
import sys
import logging
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QComboBox, QProgressBar, QSplitter,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg  # For real-time plotting

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('millennial_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MillennialAi')

# Import MillennialAi components
from millennial_ai.original_architecture import MillennialAiModel, MillennialAiConfig
from millennial_ai.core.ip_safe_integration import MillennialAiIPSafeModel
from brain_visualizer import BrainVisualizer
from examples.cost_calculator import calculate_millennial_ai_cost

class InferenceWorker(QThread):
    """Worker thread for model inference to avoid blocking UI."""
    result_ready = pyqtSignal(str, float, int)  # response, complexity, steps
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, model, prompt):
        super().__init__()
        self.model = model
        self.prompt = prompt
    
    def run(self):
        try:
            logger.info("Starting model inference")
            response = self.model.generate(self.prompt, max_length=100)
            # Simulate metrics since model doesn't provide them
            complexity = len(self.prompt) / 100.0  # Simple heuristic
            steps = max(1, int(complexity * 10))
            logger.info("Inference completed successfully")
            self.result_ready.emit(response, complexity, steps)
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MillennialAi Cognitive Enhancement Platform")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load default model
        self.config = MillennialAiConfig.for_enterprise_70b()
        self.model = None  # Lazy load
        
        # Setup UI
        self.init_ui()
        self.apply_millennialai_theme()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Header with branding
        header = QLabel("🧠 MillennialAi: Layer Injection Architecture for Enhanced Reasoning")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header)
        
        # Main tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Tab 1: Conversational Interface
        self.chat_tab = self.create_chat_tab()
        tabs.addTab(self.chat_tab, "💬 Cognitive Chat")
        
        # Tab 2: Architecture Visualization
        self.viz_tab = self.create_visualization_tab()
        tabs.addTab(self.viz_tab, "📊 Layer Injection Viz")
        
        # Tab 3: Configuration
        self.config_tab = self.create_config_tab()
        tabs.addTab(self.config_tab, "⚙️ Model Config")
        
        # Tab 4: Enterprise Tools
        self.tools_tab = self.create_tools_tab()
        tabs.addTab(self.tools_tab, "🏢 Enterprise Suite")
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - No model loaded")
    
    def create_chat_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input area
        input_group = QGroupBox("Input Prompt")
        input_layout = QVBoxLayout(input_group)
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter your query for cognitive enhancement...")
        input_layout.addWidget(self.input_text)
        layout.addWidget(input_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.send_btn = QPushButton("🚀 Enhance & Respond")
        self.send_btn.clicked.connect(self.handle_send)
        controls_layout.addWidget(self.send_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.input_text.clear)
        controls_layout.addWidget(self.clear_btn)
        
        layout.addLayout(controls_layout)
        
        # Output area
        output_group = QGroupBox("Enhanced Response")
        output_layout = QVBoxLayout(output_group)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        output_layout.addWidget(self.output_text)
        
        # Metrics display
        metrics_layout = QHBoxLayout()
        self.complexity_label = QLabel("Complexity: N/A")
        self.steps_label = QLabel("Steps: N/A")
        metrics_layout.addWidget(self.complexity_label)
        metrics_layout.addWidget(self.steps_label)
        output_layout.addLayout(metrics_layout)
        
        layout.addWidget(output_group)
        
        # Progress bar for inference
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return widget
    
    def create_visualization_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Layer diagram (text-based for now, can integrate pyqtgraph)
        diagram_text = QTextEdit()
        diagram_text.setPlainText("""
Enhanced LLM (70B) + MillennialAi Cognitive Layers
├── Standard Transformer Layers (1-30)
├── ✨ MillennialAi Injection Points (31-35) ← OUR INNOVATION
├── Standard Transformer Layers (36-65)
├── ✨ MillennialAi Enhancement Layer (66-70) ← OUR INNOVATION
└── Output Layer
        """)
        diagram_text.setReadOnly(True)
        layout.addWidget(diagram_text)
        
        # Real-time plot for brain activity
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Update timer for live viz
        self.viz_timer = QTimer()
        self.viz_timer.timeout.connect(self.update_visualization)
        self.viz_timer.start(1000)  # Update every second
        
        return widget
    
    def create_config_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        form_layout = QFormLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["GPT-2", "LLaMA-7B", "LLaMA-70B"])
        self.model_combo.setCurrentText("GPT-2")  # Set GPT-2 as default
        form_layout.addRow("Base Model:", self.model_combo)
        
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 100)
        self.layers_spin.setValue(80)
        form_layout.addRow("Total Layers:", self.layers_spin)
        
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 20)
        self.depth_spin.setValue(10)
        form_layout.addRow("Max Recursion Depth:", self.depth_spin)
        
        self.adaptive_check = QCheckBox("Adaptive Depth by Layer Position")
        self.adaptive_check.setChecked(True)
        form_layout.addRow(self.adaptive_check)
        
        layout.addLayout(form_layout)
        
        load_btn = QPushButton("🔄 Load/Reload Model")
        load_btn.clicked.connect(self.load_model)
        layout.addWidget(load_btn)
        
        return widget
    
    def create_tools_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Cost calculator
        cost_group = QGroupBox("Cost Estimation")
        cost_layout = QVBoxLayout(cost_group)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1000, 1000000)
        self.samples_spin.setValue(10000)
        cost_layout.addWidget(QLabel("Training Samples:"))
        cost_layout.addWidget(self.samples_spin)
        
        calc_cost_btn = QPushButton("💰 Calculate Cost")
        calc_cost_btn.clicked.connect(self.calculate_cost)
        cost_layout.addWidget(calc_cost_btn)
        
        self.cost_result = QLabel("Cost: N/A")
        cost_layout.addWidget(self.cost_result)
        layout.addWidget(cost_group)
        
        # Patent report generator
        patent_btn = QPushButton("📄 Generate Patent Report")
        patent_btn.clicked.connect(self.generate_report)
        layout.addWidget(patent_btn)
        
        return widget
    
    def apply_millennialai_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 248, 255))  # Light blue theme
        palette.setColor(QPalette.WindowText, QColor(0, 0, 139))  # Dark blue text
        self.setPalette(palette)
    
    def handle_send(self):
        prompt = self.input_text.toPlainText().strip()
        if not prompt:
            logger.warning("Empty prompt submitted")
            QMessageBox.warning(self, "Warning", "Please enter a prompt.")
            return
        
        if not self.model:
            logger.error("Attempted inference without loaded model")
            QMessageBox.critical(self, "Error", "No model loaded. Please load a model first.")
            return
        
        logger.info(f"Starting inference for prompt: {prompt[:50]}...")
        self.progress_bar.setVisible(True)
        self.send_btn.setEnabled(False)
        self.status_bar.showMessage("Running inference...")
        
        try:
            self.worker = InferenceWorker(self.model, prompt)
            self.worker.result_ready.connect(self.on_inference_done)
            self.worker.error_occurred.connect(self.on_inference_error)
            self.worker.start()
        except Exception as e:
            logger.error(f"Failed to start inference worker: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start inference: {str(e)}")
            self.progress_bar.setVisible(False)
            self.send_btn.setEnabled(True)
    
    def on_inference_done(self, response, complexity, steps):
        logger.info(f"Inference completed: complexity={complexity:.3f}, steps={steps}")
        self.output_text.setPlainText(response)
        self.complexity_label.setText(f"Complexity: {complexity:.3f}")
        self.steps_label.setText(f"Steps: {steps}")
        self.progress_bar.setVisible(False)
        self.send_btn.setEnabled(True)
        self.status_bar.showMessage("Inference completed")
    
    def on_inference_error(self, error_msg):
        logger.error(f"Inference failed: {error_msg}")
        QMessageBox.critical(self, "Inference Error", error_msg)
        self.output_text.setPlainText(f"Error: {error_msg}")
        self.complexity_label.setText("Complexity: N/A")
        self.steps_label.setText("Steps: N/A")
        self.progress_bar.setVisible(False)
        self.send_btn.setEnabled(True)
        self.status_bar.showMessage("Inference failed")
    
    def load_model(self):
        model_name = self.model_combo.currentText()
        logger.info(f"Loading model: {model_name}")
        
        # Map display names to HuggingFace model paths
        model_paths = {
            "GPT-2": "gpt2",
            "LLaMA-7B": "meta-llama/Llama-2-7b-hf",
            "LLaMA-70B": "meta-llama/Llama-2-70b-hf"
        }
        
        model_path = model_paths.get(model_name, model_name)
        self.config.base_model_name = model_path
        self.config.base_model_layers = self.layers_spin.value()
        self.config.reasoning_stages = self.depth_spin.value()
        
        try:
            self.model = MillennialAiModel(self.config)
            # Load the base model
            if self.model.load_base_model(model_path):
                # Update config with actual model layers
                if hasattr(self.model.base_model, 'config') and hasattr(self.model.base_model.config, 'n_layer'):
                    # GPT-2 style models
                    actual_layers = self.model.base_model.config.n_layer
                elif hasattr(self.model.base_model, 'config') and hasattr(self.model.base_model.config, 'num_hidden_layers'):
                    # LLaMA/BERT style models
                    actual_layers = self.model.base_model.config.num_hidden_layers
                else:
                    # Fallback
                    actual_layers = len([m for m in self.model.base_model.modules() if hasattr(m, 'weight') and len(m.weight.shape) > 1])
                
                self.config.base_model_layers = actual_layers
                # Recalculate injection points for the actual model
                self.config.injection_points = None  # Force recalculation
                self.config._suggest_injection_points()
                
                # Apply enhancements
                if self.model.enhance_model():
                    logger.info(f"Model {model_name} loaded and enhanced successfully")
                    self.status_bar.showMessage(f"Model {model_name} loaded and enhanced successfully")
                    QMessageBox.information(self, "Success", f"Model {model_name} loaded and enhanced successfully!")
                else:
                    raise RuntimeError("Failed to apply enhancements")
            else:
                raise RuntimeError("Failed to load base model")
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            self.status_bar.showMessage("Model loading failed")
            QMessageBox.critical(self, "Model Loading Error", error_msg)
            self.model = None
    
    def update_visualization(self):
        if self.model:
            # Simulate brain activity data (integrate with actual viz)
            data = torch.randn(100)  # Placeholder
            self.plot_widget.plot(data, clear=True)
    
    def calculate_cost(self):
        # Get samples value (currently not used but may be needed in future)
        _ = self.samples_spin.value()
        base_size = self.config.base_model_layers * self.config.hidden_size / 1e9  # in billions
        trm_size = base_size * 0.01  # 1% of base
        cost_dict = calculate_millennial_ai_cost(base_size, trm_size)
        cost = cost_dict.get('total_cost', 1000)
        self.cost_result.setText(f"Estimated Cost: ${cost:.2f}")
    
    def generate_report(self):
        # Integrate with create_patent_report.py
        self.status_bar.showMessage("Patent report generated (placeholder)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())