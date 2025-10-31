#!/usr/bin/env python3
"""
MillennialAi 3D Brain Visualizer
Windows Media Player style brain visualization!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import threading
import queue
from millennial_ai.models.reasoning_engine import MillennialAiReasoningEngine

class MillennialAi3DBrainVisualizer:
    """
    Clean Windows Media Player style brain visualization
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize brain
        self.brain = MillennialAiReasoningEngine(
            hidden_size=768,
            max_recursion_depth=8,
            num_scales=3,
            num_heads=8,
            memory_size=256
        ).to(self.device)
        
        # Visualization data
        self.brain_activity = queue.Queue()
        self.is_running = True
        
        # Clean color scheme
        self.colors = {
            'active': '#00ff41',     # Bright green
            'processing': '#ff6b35', # Orange
            'memory': '#4a90e2',     # Blue  
            'idle': '#333333',       # Dark
            'waves': '#e74c3c'       # Red
        }
        
        print("🧠 MillennialAi Brain Visualizer")
        print(f"⚡ Running on {self.device}")
        
    def generate_brain_activity(self):
        """
        Generate brain thinking data
        """
        while self.is_running:
            try:
                # Random problem complexity
                complexity = np.random.choice(['simple', 'medium', 'complex'])
                seq_len = np.random.randint(5, 25)
                
                # Generate and process problem  
                problem = torch.randn(1, seq_len, 768, device=self.device)
                attention_mask = torch.ones(1, seq_len, device=self.device)
                
                with torch.no_grad():
                    result = self.brain(problem, attention_mask)
                
                # Package visualization data
                brain_data = {
                    'steps': result['reasoning_steps'].item(),
                    'depth': result['required_depth'].item(),
                    'memory': result['memory_weights'].cpu().numpy(),
                    'convergence': result['convergence_history'].cpu().numpy(),
                    'complexity': complexity
                }
                
                if not self.brain_activity.full():
                    self.brain_activity.put(brain_data)
                
                time.sleep(0.3)  # Smooth updates
                
            except Exception as e:
                print(f"Brain error: {e}")
                time.sleep(1)
    
    def create_ultra_hd_visualization(self):
        """
        SUPER HIGH DEFINITION 4K brain visualization
        """
        # Ultra HD settings
        plt.rcParams['figure.dpi'] = 300  # 4K quality
        plt.rcParams['savefig.dpi'] = 300
        plt.style.use('dark_background')
        
        # Create massive HD figure
        fig = plt.figure(figsize=(24, 18), facecolor='black')
        fig.suptitle('🧠 MillennialAi Brain - ULTRA HD VISUALIZATION', 
                     fontsize=32, color='cyan', fontweight='bold')
        
        # Ultra precise grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.2, wspace=0.2)
        
        # MAIN 3D Brain (Ultra HD)
        ax_3d = fig.add_subplot(gs[0:3, 0:3], projection='3d')
        ax_3d.set_facecolor('black')
        
        # HD Side panels
        ax_steps = fig.add_subplot(gs[0, 3])
        ax_memory = fig.add_subplot(gs[1, 3]) 
        ax_depth = fig.add_subplot(gs[2, 3])
        ax_waves = fig.add_subplot(gs[3, :])
        
        # Ultra HD data storage
        brain_history = []
        wave_history = []
        
        def ultra_hd_update(frame):
            # Ultra smooth clearing
            for ax in [ax_3d, ax_steps, ax_memory, ax_depth, ax_waves]:
                ax.clear()
                ax.set_facecolor('black')
            
            # Get latest HD brain data
            if not self.brain_activity.empty():
                try:
                    data = self.brain_activity.get_nowait()
                    brain_history.append(data)
                    wave_history.append(data['convergence'][-1, 0])
                    
                    # Keep ultra smooth 100 frames
                    if len(brain_history) > 100:
                        brain_history.pop(0)
                        wave_history.pop(0)
                except:
                    pass
            
            if brain_history:
                latest = brain_history[-1]
                
                # ULTRA HD 3D BRAIN NETWORK
                n_nodes = 50  # More nodes for HD
                
                # Perfect sphere positioning
                phi = np.random.rand(n_nodes) * 2 * np.pi
                theta = np.random.rand(n_nodes) * np.pi
                radius = 8 + np.random.rand(n_nodes) * 4
                
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi) 
                z = radius * np.cos(theta)
                
                # Ultra HD colors and sizes
                colors = []
                sizes = []
                alphas = []
                
                for i in range(n_nodes):
                    if i < latest['steps'] * 6:  # More active nodes
                        colors.append('#00ff41')  # Bright green
                        sizes.append(300)
                        alphas.append(1.0)
                    elif i < latest['depth'] * 8:
                        colors.append('#ff6b35')  # Orange
                        sizes.append(200)
                        alphas.append(0.8)
                    else:
                        colors.append('#333333')  # Dark
                        sizes.append(50)
                        alphas.append(0.3)
                
                # Ultra HD 3D scatter
                ax_3d.scatter(x, y, z, c=colors, s=sizes, alpha=0.9, edgecolors='white', linewidth=0.5)
                
                # HD Neural connections
                active_nodes = min(latest['steps'] * 6, n_nodes)
                for i in range(active_nodes):
                    for j in range(i+1, min(i+4, active_nodes)):
                        ax_3d.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                                  color='#00ff41', alpha=0.6, linewidth=2)
                
                # Perfect 3D sphere
                ax_3d.set_xlim(-15, 15)
                ax_3d.set_ylim(-15, 15)
                ax_3d.set_zlim(-15, 15)
                ax_3d.set_title(f'🧠 ULTRA HD BRAIN - {latest["complexity"].upper()}', 
                               color='lime', fontsize=24, fontweight='bold')
                
                # Remove axes for clean look
                ax_3d.set_xticks([])
                ax_3d.set_yticks([])
                ax_3d.set_zticks([])
                
                # ULTRA HD REASONING BARS
                steps_data = [1.0 if i < latest['steps'] else 0.2 for i in range(8)]
                bars = ax_steps.bar(range(8), steps_data, 
                                   color=['#00ff41' if x > 0.5 else '#333333' for x in steps_data],
                                   edgecolor='white', linewidth=2, alpha=0.9)
                
                # HD bar effects
                for i, bar in enumerate(bars):
                    if i < latest['steps']:
                        bar.set_height(1.0 + 0.2 * np.sin(frame * 0.3 + i))  # Pulsing effect
                
                ax_steps.set_ylim(0, 1.5)
                ax_steps.set_title(f'🔄 STEPS: {latest["steps"]}/8', color='orange', fontsize=16, fontweight='bold')
                ax_steps.set_xticks([])
                
                # ULTRA HD MEMORY VISUALIZATION
                if latest['memory'].size > 0:
                    memory_slice = latest['memory'][0, :20, :20]  # Larger slice for HD
                    im = ax_memory.imshow(memory_slice, cmap='plasma', aspect='auto', interpolation='bilinear')
                    ax_memory.set_title('💾 HD MEMORY', color='blue', fontsize=16, fontweight='bold')
                    ax_memory.set_xticks([])
                    ax_memory.set_yticks([])
                
                # ULTRA HD DEPTH GAUGE
                depth_levels = [1.0 if i < latest['depth'] else 0.1 for i in range(8)]
                circle_colors = ['#4a90e2' if x > 0.5 else '#222222' for x in depth_levels]
                
                for i, (level, color) in enumerate(zip(depth_levels, circle_colors)):
                    circle = plt.Circle((i, 0.5), 0.4, color=color, alpha=0.9)
                    ax_depth.add_patch(circle)
                    if level > 0.5:
                        # Pulsing effect for active
                        pulse = 0.4 + 0.1 * np.sin(frame * 0.4 + i)
                        pulse_circle = plt.Circle((i, 0.5), pulse, color=color, alpha=0.3)
                        ax_depth.add_patch(pulse_circle)
                
                ax_depth.set_xlim(-0.5, 7.5)
                ax_depth.set_ylim(0, 1)
                ax_depth.set_title(f'� DEPTH: {latest["depth"]}/8', color='cyan', fontsize=16, fontweight='bold')
                ax_depth.set_xticks([])
                ax_depth.set_yticks([])
                
                # ULTRA HD CONVERGENCE WAVES
                if len(wave_history) > 2:
                    x_wave = np.arange(len(wave_history))
                    y_wave = wave_history
                    
                    # Ultra smooth interpolation
                    if len(x_wave) > 10:
                        x_smooth = np.linspace(0, len(x_wave)-1, len(x_wave)*4)
                        y_smooth = np.interp(x_smooth, x_wave, y_wave)
                        
                        # HD wave with gradient effect
                        ax_waves.plot(x_smooth, y_smooth, color='#e74c3c', linewidth=4, alpha=0.9)
                        ax_waves.fill_between(x_smooth, y_smooth, alpha=0.3, color='#e74c3c')
                        
                        # HD pulsing dots
                        for i in range(0, len(x_smooth), 8):
                            pulse_size = 200 * (0.8 + 0.4 * np.sin(frame * 0.2 + i * 0.1))
                            ax_waves.scatter(x_smooth[i], y_smooth[i], s=pulse_size, 
                                           c='#e74c3c', alpha=0.8, edgecolors='white', linewidth=2)
                    
                    ax_waves.set_ylim(0, 1.2)
                    ax_waves.set_title('🌊 ULTRA HD CONVERGENCE WAVES', color='magenta', fontsize=20, fontweight='bold')
                    ax_waves.grid(True, alpha=0.2, color='white')
                    ax_waves.set_facecolor('black')
            
            # Ultra HD styling for all axes
            for ax in [ax_steps, ax_memory, ax_depth, ax_waves]:
                ax.tick_params(colors='white', labelsize=12)
                for spine in ax.spines.values():
                    spine.set_color('white')
                    spine.set_linewidth(2)
        
        # Ultra smooth 60 FPS animation
        ani = FuncAnimation(fig, ultra_hd_update, interval=16, blit=False, cache_frame_data=False)
        
        return fig, ani
    
    def start_ultra_hd_visualization(self):
        """
        Start SUPER HIGH DEFINITION visualization
        """
        print("🚀 ULTRA HD Brain Visualization Loading...")
        print("� 4K Quality - 60 FPS - Maximum Detail")
        print("🔥 Super High Definition Mode ACTIVATED!")
        
        # Start brain activity thread
        brain_thread = threading.Thread(target=self.generate_brain_activity, daemon=True)
        brain_thread.start()
        
        # Create Ultra HD visualization
        fig, ani = self.create_ultra_hd_visualization()
        
        # Show in maximum quality window
        plt.show()
        
        self.is_running = False

def main():
    print("🧠 MILLENNIAL AI - ULTRA HD BRAIN VISUALIZER")
    print("� SUPER HIGH DEFINITION 4K BRAIN EFFECTS")
    print("⚡ 60 FPS REAL-TIME NEURAL VISUALIZATION")
    print("=" * 70)
    
    try:
        visualizer = MillennialAi3DBrainVisualizer()
        visualizer.start_ultra_hd_visualization()
    except KeyboardInterrupt:
        print("\n👋 Ultra HD Brain visualizer shutting down...")
    except Exception as e:
        print(f"🔧 HD Visualization error: {e}")
        print("💡 Ultra HD requires high-performance graphics")

if __name__ == "__main__":
    main()