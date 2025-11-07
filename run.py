"""
MillennialAI Training Launcher
Quick menu for all operations
"""

import sys
from pathlib import Path

def print_menu():
    print("=" * 80)
    print("🚀 MILLENNIALAI HYBRID TRAINING SYSTEM")
    print("=" * 80)
    print()
    print("Choose an option:")
    print()
    print("  1. 🎯 Train new model (45-60 min)")
    print("  2. 🔗 Merge LoRA weights (5-10 min)")
    print("  3. 📦 Deploy to Ollama (10-15 min)")
    print("  4. 🔄 Full pipeline (train → merge → deploy)")
    print("  5. ⚙️  Show configuration")
    print("  6. 📊 Check system status")
    print("  0. ❌ Exit")
    print()
    print("=" * 80)


def check_status():
    """Check what's been completed"""
    from hybrid_config import OUTPUT_DIR, MERGED_MODEL_DIR, GGUF_DIR
    
    print("\n📊 System Status:")
    print("-" * 80)
    
    # Check training
    checkpoints = list(OUTPUT_DIR.glob("checkpoint-*"))
    final_exists = (OUTPUT_DIR / "final").exists()
    
    if final_exists:
        print("✅ Training: COMPLETE (final model saved)")
    elif checkpoints:
        print(f"⏳ Training: IN PROGRESS ({len(checkpoints)} checkpoints)")
    else:
        print("❌ Training: NOT STARTED")
    
    # Check merge
    if MERGED_MODEL_DIR.exists() and list(MERGED_MODEL_DIR.glob("*.safetensors")):
        print("✅ Merge: COMPLETE")
    else:
        print("❌ Merge: NOT STARTED")
    
    # Check GGUF
    gguf_files = list(GGUF_DIR.glob("*.gguf")) if GGUF_DIR.exists() else []
    if gguf_files:
        gguf_file = gguf_files[0]
        size_gb = gguf_file.stat().st_size / 1e9
        print(f"✅ GGUF: COMPLETE ({size_gb:.2f} GB)")
    else:
        print("❌ GGUF: NOT STARTED")
    
    # Check Ollama
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "millennialai" in result.stdout:
            print("✅ Ollama: DEPLOYED")
        else:
            print("❌ Ollama: NOT DEPLOYED")
    except:
        print("⚠️  Ollama: Cannot check (not installed?)")
    
    print("-" * 80)


def main():
    print_menu()
    
    choice = input("Enter choice (0-6): ").strip()
    
    if choice == "0":
        print("\n👋 Goodbye!")
        return
    
    elif choice == "1":
        print("\n🎯 Starting training...")
        print("⚠️  Make sure you're running from Anaconda Prompt!")
        print("⚠️  Switch monitor to iGPU if possible")
        input("\nPress Enter to continue or Ctrl+C to cancel...")
        
        import hybrid_training
        hybrid_training.train_hybrid_model()
    
    elif choice == "2":
        print("\n🔗 Starting LoRA merge...")
        
        import hybrid_merge
        hybrid_merge.merge_lora_weights()
    
    elif choice == "3":
        print("\n📦 Starting deployment...")
        
        import hybrid_deploy
        hybrid_deploy.main()
    
    elif choice == "4":
        print("\n🔄 Starting full pipeline...")
        print("   This will take 60-90 minutes total")
        print("⚠️  Make sure you're running from Anaconda Prompt!")
        print("⚠️  Switch monitor to iGPU if possible")
        input("\nPress Enter to continue or Ctrl+C to cancel...")
        
        # Train
        print("\n" + "="*80)
        print("STEP 1/3: TRAINING")
        print("="*80)
        import hybrid_training
        hybrid_training.train_hybrid_model()
        
        # Merge
        print("\n" + "="*80)
        print("STEP 2/3: MERGING")
        print("="*80)
        import hybrid_merge
        hybrid_merge.merge_lora_weights()
        
        # Deploy
        print("\n" + "="*80)
        print("STEP 3/3: DEPLOYMENT")
        print("="*80)
        import hybrid_deploy
        hybrid_deploy.main()
        
        print("\n" + "="*80)
        print("🎉 FULL PIPELINE COMPLETE!")
        print("="*80)
        print("\nYour AI is ready:")
        print("  ollama run millennialai")
    
    elif choice == "5":
        print()
        import hybrid_config
        hybrid_config.print_config()
    
    elif choice == "6":
        check_status()
    
    else:
        print("\n❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
