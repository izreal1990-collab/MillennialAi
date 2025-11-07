# MillennialAI Hybrid Training System - Nov 7, 2025

## 🚀 What's New

**Clean, production-ready training system** that maximizes RTX 5060 Ti + Ryzen 7 7700 hardware.

## 📦 New Files

- **`hybrid_config.py`** - Centralized configuration (model, hardware, training params)
- **`hybrid_training.py`** - Main training script (GPU 4-bit + TRM + OpenAssistant)
- **`hybrid_merge.py`** - LoRA merge with CPU offloading
- **`hybrid_deploy.py`** - GGUF conversion + Ollama deployment
- **`run.py`** - Interactive launcher menu
- **`HYBRID_SYSTEM_README.md`** - Complete usage guide

## 🔧 Technical Improvements

- **GPU-only 4-bit training** (stable, no device conflicts)
- **7 TRM reasoning layers** (270M params each)
- **OpenAssistant dataset** (1,536 conversations + 164 workspace docs)
- **LoRA rank 128** (335M trainable params, 3.27%)
- **Automated pipeline** (train → merge → GGUF → Ollama)

## 📊 Training Specs

- **Model**: llama3-8B (4-bit quantized)
- **Dataset**: 1,700 samples (technical docs + conversations)
- **Hardware**: RTX 5060 Ti 16GB + Ryzen 7 7700 8C/16T
- **Effective batch**: 32 (2 × 16 grad accumulation)
- **Epochs**: 5
- **Context**: 2048 tokens

## 🎯 Usage

```bash
# Interactive menu
python run.py

# Or manual pipeline
python hybrid_training.py    # Train
python hybrid_merge.py        # Merge LoRA
python hybrid_deploy.py       # Deploy to Ollama
ollama run millennialai       # Use!
```

## 🗂️ Code Organization

- **New system**: Clean 5-file pipeline
- **Old code**: Moved to `archive/` folder
- **No breaking changes**: Old checkpoints preserved

## ✅ Benefits

- ✅ **Conversational AI** (trained on human dialogues)
- ✅ **Stable training** (GPU-only, no hybrid device conflicts)  
- ✅ **Production ready** (error handling, checkpoints, logging)
- ✅ **Fully automated** (one command deployment)
- ✅ **Well documented** (comprehensive README)

---

**Ready to train conversational AI with TRM reasoning on consumer hardware.**
