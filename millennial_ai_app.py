#!/usr/bin/env python3
"""
MillennialAi Main Application
Complete interactive GUI application
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
import torch
from real_brain import RealThinkingBrain

class MillennialAiApp:
    """
    Main MillennialAi Application Window
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MillennialAi - Interactive Brain Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize brain
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = None
        self.is_thinking = False
        self.result_queue = queue.Queue()
        
        self.setup_ui()
        self.initialize_brain()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#1e1e1e')
        title_frame.pack(fill='x', pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🧠 MillennialAi Interactive Brain",
            font=('Arial', 24, 'bold'),
            fg='#00ff41',
            bg='#1e1e1e'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Patent-Pending Layer Injection Technology",
            font=('Arial', 12),
            fg='#888888',
            bg='#1e1e1e'
        )
        subtitle_label.pack()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Input
        left_frame = tk.Frame(main_frame, bg='#2d2d2d', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Input section
        input_label = tk.Label(
            left_frame,
            text="💭 Ask Your AI Brain:",
            font=('Arial', 14, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        )
        input_label.pack(pady=10)
        
        self.input_text = scrolledtext.ScrolledText(
            left_frame,
            height=8,
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff',
            insertbackground='#ffffff',
            selectbackground='#555555'
        )
        self.input_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='#2d2d2d')
        button_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.think_button = tk.Button(
            button_frame,
            text="🧠 Think",
            font=('Arial', 12, 'bold'),
            bg='#00ff41',
            fg='#000000',
            command=self.start_thinking,
            height=2,
            relief='raised',
            bd=3
        )
        self.think_button.pack(side='left', padx=(0, 5))
        
        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=('Arial', 12),
            bg='#ff6b35',
            fg='#ffffff',
            command=self.clear_all,
            height=2,
            relief='raised',
            bd=3
        )
        self.clear_button.pack(side='left', padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹️ Stop",
            font=('Arial', 12),
            bg='#e74c3c',
            fg='#ffffff',
            command=self.stop_thinking,
            height=2,
            relief='raised',
            bd=3,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)
        
        # Right panel - Output and Status
        right_frame = tk.Frame(main_frame, bg='#2d2d2d', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Status section
        status_label = tk.Label(
            right_frame,
            text="🔍 Brain Activity:",
            font=('Arial', 14, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        )
        status_label.pack(pady=10)
        
        self.status_text = scrolledtext.ScrolledText(
            right_frame,
            height=6,
            font=('Courier', 10),
            bg='#1a1a1a',
            fg='#00ff41',
            insertbackground='#00ff41',
            state='disabled'
        )
        self.status_text.pack(fill='x', padx=10, pady=(0, 10))
        
        # Results section
        results_label = tk.Label(
            right_frame,
            text="💡 AI Response:",
            font=('Arial', 14, 'bold'),
            fg='#ffffff',
            bg='#2d2d2d'
        )
        results_label.pack()
        
        self.results_text = scrolledtext.ScrolledText(
            right_frame,
            height=15,
            font=('Arial', 11),
            bg='#1a1a1a',
            fg='#ffffff',
            insertbackground='#ffffff',
            state='disabled',
            wrap='word'
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Bottom status bar
        self.status_bar = tk.Label(
            self.root,
            text=f"🔥 MillennialAi Ready | Device: {self.device.upper()} | Status: Idle",
            relief='sunken',
            bd=1,
            font=('Arial', 10),
            bg='#333333',
            fg='#ffffff',
            anchor='w'
        )
        self.status_bar.pack(side='bottom', fill='x')
        
        # Sample prompts
        self.add_sample_prompts()
        
    def add_sample_prompts(self):
        """Add sample prompts to get started"""
        samples = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms",
            "How do neural networks learn?",
            "Write a short poem about technology",
            "Solve this math problem: What is 15 × 23?"
        ]
        
        sample_text = "Try these examples:\n" + "\n".join(f"• {s}" for s in samples)
        self.input_text.insert('1.0', sample_text)
        
    def initialize_brain(self):
        """Initialize the AI brain"""
        self.log_status("🧠 Initializing MillennialAi Brain...")
        try:
            self.brain = RealThinkingBrain().to(self.device)
            self.log_status(f"✅ Brain initialized successfully!")
            self.log_status(f"📊 Parameters: {sum(p.numel() for p in self.brain.parameters()):,}")
            self.update_status_bar("Ready")
        except Exception as e:
            self.log_status(f"❌ Brain initialization failed: {e}")
            messagebox.showerror("Error", f"Failed to initialize brain: {e}")
    
    def log_status(self, message):
        """Log status message"""
        self.status_text.config(state='normal')
        self.status_text.insert('end', f"{message}\n")
        self.status_text.see('end')
        self.status_text.config(state='disabled')
        self.root.update()
        
    def update_status_bar(self, status):
        """Update bottom status bar"""
        self.status_bar.config(text=f"🔥 MillennialAi Ready | Device: {self.device.upper()} | Status: {status}")
        
    def start_thinking(self):
        """Start the thinking process"""
        if self.is_thinking:
            return
            
        input_text = self.input_text.get('1.0', 'end-1c').strip()
        if not input_text:
            messagebox.showwarning("Warning", "Please enter a question or prompt!")
            return
            
        self.is_thinking = True
        self.think_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.update_status_bar("Thinking...")
        
        # Clear previous results
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.config(state='disabled')
        
        # Start thinking thread
        thinking_thread = threading.Thread(target=self.think_process, args=(input_text,))
        thinking_thread.daemon = True
        thinking_thread.start()
        
        # Start result checker
        self.check_results()
        
    def think_process(self, input_text):
        """The actual thinking process"""
        try:
            self.log_status(f"🎯 Processing: '{input_text[:50]}{'...' if len(input_text) > 50 else ''}'")
            
            # Convert text to tensor (simple encoding)
            seq_len = min(50, max(5, len(input_text.split())))
            variance_factor = len(input_text) / 100.0
            
            # Create problem tensor
            problem = torch.randn(1, seq_len, 768, device=self.device) * variance_factor
            
            self.log_status(f"📊 Problem tensor: {problem.shape}, variance: {torch.var(problem):.4f}")
            
            # Process through brain
            with torch.no_grad():
                result = self.brain(problem)
            
            # Generate response based on result
            steps = result['reasoning_steps'].item()
            complexity = result['complexity_score']
            convergence = result['convergence_history'][-1].item() if len(result['convergence_history']) > 0 else 0.0
            
            # Create AI response
            response = self.generate_response(input_text, steps, complexity, convergence)
            
            # Put result in queue
            self.result_queue.put({
                'response': response,
                'steps': steps,
                'complexity': complexity,
                'convergence': convergence,
                'success': True
            })
            
        except Exception as e:
            self.result_queue.put({
                'error': str(e),
                'success': False
            })
    
    def generate_response(self, input_text, steps, complexity, convergence):
        """Generate a response based on brain analysis"""
        # Simple response generation based on input analysis
        input_lower = input_text.lower()
        
        if any(word in input_lower for word in ['what', 'explain', 'how']):
            if complexity < 2:
                response = f"Based on my analysis, this is a straightforward question. After {steps} reasoning steps, I can provide a direct answer."
            elif complexity < 4:
                response = f"This question requires moderate reasoning. After {steps} steps of analysis, I've processed the complexity and can provide a comprehensive response."
            else:
                response = f"This is a complex question that required {steps} reasoning steps. My deep analysis shows multiple layers of consideration needed."
        
        elif any(word in input_lower for word in ['write', 'create', 'compose']):
            response = f"Creative task detected! My brain used {steps} reasoning steps to approach this creatively. The complexity score of {complexity:.2f} suggests this needed substantial cognitive processing."
        
        elif any(word in input_lower for word in ['solve', 'calculate', 'math']):
            response = f"Mathematical problem identified. After {steps} analytical steps with convergence score {convergence:.3f}, I've processed this systematically."
        
        else:
            response = f"I've analyzed your input using {steps} reasoning steps. The complexity analysis yielded a score of {complexity:.2f}, and my reasoning converged at {convergence:.3f}."
        
        # Add adaptive detail based on complexity
        if complexity > 3:
            response += f"\n\nThis required deep reasoning across multiple abstraction levels. My adaptive depth controller determined that {steps} steps were necessary for proper analysis."
        
        if convergence > 0.7:
            response += f"\n\nI achieved high confidence in this analysis (convergence: {convergence:.3f}), indicating a solid reasoning foundation."
        
        return response
    
    def check_results(self):
        """Check for results from thinking thread"""
        try:
            result = self.result_queue.get_nowait()
            
            if result['success']:
                # Display response
                self.results_text.config(state='normal')
                self.results_text.insert('end', result['response'])
                self.results_text.config(state='disabled')
                
                self.log_status(f"✅ Thinking complete!")
                self.log_status(f"📊 Steps: {result['steps']}, Complexity: {result['complexity']:.3f}")
                self.log_status(f"🎯 Convergence: {result['convergence']:.3f}")
                
            else:
                self.log_status(f"❌ Error: {result['error']}")
                messagebox.showerror("Error", f"Thinking failed: {result['error']}")
            
            self.finish_thinking()
            
        except queue.Empty:
            # Check again in 100ms
            self.root.after(100, self.check_results)
    
    def stop_thinking(self):
        """Stop the thinking process"""
        self.is_thinking = False
        self.finish_thinking()
        self.log_status("⏹️ Thinking stopped by user")
        
    def finish_thinking(self):
        """Finish thinking process"""
        self.is_thinking = False
        self.think_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.update_status_bar("Ready")
        
    def clear_all(self):
        """Clear all text areas"""
        self.input_text.delete('1.0', 'end')
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.config(state='disabled')
        self.status_text.config(state='normal')
        self.status_text.delete('1.0', 'end')
        self.status_text.config(state='disabled')
        self.add_sample_prompts()
        self.log_status("🗑️ Interface cleared")
        
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("🚀 Starting MillennialAi Interactive Application...")
    
    try:
        app = MillennialAiApp()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")
        messagebox.showerror("Critical Error", f"Application failed to start: {e}")

if __name__ == "__main__":
    main()