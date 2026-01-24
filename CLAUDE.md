# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> ⚠️ **DO NOT ADD TO THIS FILE** - Keep it under 500 lines. Document new features in module docstrings, config comments, or `docs/` directory instead.

## Project Overview

Voice Clone Pipeline is a system for creating personalized voice models by recording voice samples, auto-labeling prosody features, fine-tuning Sesame's CSM-1B model, and generating speech. Target hardware is M4 Pro Mac (64GB) and RTX 4090 (24GB).

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   RECORD     │───▶│    LABEL     │───▶│    TRAIN     │───▶│   GENERATE   │
│   (Frontend) │    │   (Backend)  │    │  (Training)  │    │  (Inference) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

- **Frontend (Next.js)**: Web-based recording interface with 3D prosody visualizer using Three.js
- **Backend (FastAPI)**: Handles audio upload, transcription (Whisper), and multi-layer prosody analysis (Parselmouth + Qwen2-Audio)
- **Training**: CSM-1B fine-tuning with DeepSeek techniques (MTP, MLA, custom LR schedule)
- **Inference**: Speech generation from fine-tuned models

### Prosody Analysis Pipeline

Audio is analyzed from 4 perspectives in `backend/prosody_analyzer.py`:
- **Semantic**: Emotion/tone detection via Qwen2-Audio
- **Acoustic**: Pitch, formants, HNR via Parselmouth
- **Rhythm**: Pauses, speaking rate, syllables via librosa
- **Contour**: Pitch trajectory time-series

### DeepSeek Training Enhancements

The training pipeline (`training/train_deepseek.py`) implements:
- **Multi-Token Prediction (MTP)**: Predicts 4 tokens ahead for denser training signal
- **Multi-head Latent Attention (MLA)**: Compresses KV cache for memory efficiency
- **DeepSeek LR Schedule**: Warmup → Stable → Cosine decay phases
- **M4 Pro optimizations**: torch.compile, FP32, larger batches (12)

## Common Commands

### Development Setup

```bash
# Mac setup (creates venv, installs deps)
./scripts/setup_mac.sh

# Linux/CUDA setup
./scripts/setup_linux.sh

# Download models (CSM-1B, Whisper, Qwen2-Audio)
python scripts/download_models.py
```

### Running the Application

```bash
# Backend API (Terminal 1)
cd backend
source venv/bin/activate
python main.py --port 8003  # Runs on http://localhost:8003

# Frontend (Terminal 2)
cd frontend
npm run dev -- -p 3003  # Runs on http://localhost:3003
```

**Default Ports**: Frontend 3003, Backend 8003

### Training

```bash
cd training

# Prepare dataset from labeled samples
python prepare_dataset.py --input ../data/labeled --output ../data/splits

# Train with DeepSeek enhancements + dashboard
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard
# Dashboard API: http://localhost:8001
# Frontend dashboard: http://localhost:3000/training

# Alternative: standard fine-tuning
python finetune_csm.py --config config/m4_pro.yaml --data_dir ../data/splits
```

### Inference

```bash
cd inference
python generate.py \
  --model ../models/checkpoints/voice_v1/best.pt \
  --text "Hello, this is my cloned voice!" \
  --output my_voice.wav
```

## Pocket TTS (Zero-Shot Voice Cloning)

**RECOMMENDED for quick voice cloning** - No training required!

Pocket TTS (by Kyutai) is a 100M parameter TTS model that does zero-shot voice cloning at inference time. Just provide a reference audio file and it clones the voice instantly.

### Setup

```bash
# Install (already in project venv)
pip install pocket-tts

# REQUIRED: Accept terms for voice cloning
# Go to: https://huggingface.co/kyutai/pocket-tts
# Click "Agree and access repository"
```

### Usage

```bash
cd inference

# List available voice samples
python test_pocket_tts.py --list-samples

# Clone voice from a reference audio
python test_pocket_tts.py -r ../data/voice_samples/session/calm_1.wav -t "Your text here"

# Generate all emotional variations
python test_pocket_tts.py --all-emotions

# Use built-in voice (no cloning)
python test_pocket_tts.py --builtin alba
```

### Python API

```python
from pocket_tts import TTSModel
import scipy.io.wavfile as wav
import numpy as np

model = TTSModel.load_model()

# Clone voice from reference audio
voice_state = model.get_state_for_audio_prompt("path/to/reference.wav")
audio = model.generate_audio(voice_state, "Text to synthesize")

# Save output
audio_np = (audio.cpu().numpy() * 32767).astype(np.int16)
wav.write("output.wav", 24000, audio_np)
```

### Comparison: Pocket TTS vs CSM Fine-tuning

| Aspect | Pocket TTS | CSM Fine-tuning |
|--------|------------|-----------------|
| Training required | No | Yes (hours) |
| Parameters | 100M | 1.5B |
| Hardware | CPU | GPU (24GB) |
| Voice samples needed | 1 | 42+ |
| Overfitting risk | None | High |
| Quality | Good | Variable |
| Prosody control | Limited | Custom (our goal) |

**Recommendation**: Use Pocket TTS for quick voice cloning. Use CSM fine-tuning only if you need custom prosody conditioning.

### Frontend Development

```bash
cd frontend
npm run dev      # Development server
npm run build    # Production build
npm run lint     # ESLint
```

## Remote Training on RTX 4090 (Windows WSL)

A Windows 10 machine with RTX 4090 (24GB VRAM) is available for faster training via Tailscale VPN.

**IMPORTANT**: Always use conda environment `voice` when running commands on the 4090 machine!

### Connection Details
- **Tailscale IP**: `100.83.78.111`
- **User**: `doc`
- **Project Path**: `~/dev/voice-clone-pipeline`
- **Conda Environment**: `voice` (REQUIRED for all Python commands)

### Connecting to Training Session

```bash
# SSH into WSL instance
ssh doc@100.83.78.111

# Or attach to existing tmux training session
ssh doc@100.83.78.111 -t "source ~/miniconda3/bin/activate && conda activate voice && tmux attach -t training"

# If no session exists, create one
ssh doc@100.83.78.111 -t "tmux new-session -s training"
```

### Running Training on RTX 4090

**IMPORTANT**: Use LoRA training for small datasets (< 500 samples) to prevent overfitting!

```bash
# On the Windows WSL machine
source ~/miniconda3/bin/activate
conda activate voice
cd ~/dev/voice-clone-pipeline/training

# RECOMMENDED: LoRA training (0.07% of params, prevents overfitting)
python train_lora_deepseek.py --config config/rtx_4090_lora.yaml

# Alternative: Full fine-tuning (only for large datasets 500+ samples)
python train_csm_final.py --config config/rtx_4090_deepseek.yaml
```

**Training approach selection:**
- < 100 samples: Use LoRA only
- 100-500 samples: LoRA recommended, full fine-tune with early stopping
- 500+ samples: Full fine-tuning is viable

### Syncing Project Files

```bash
# From Mac to Windows (run on Mac)
rsync -avz --progress --exclude 'node_modules' --exclude '.next' --exclude 'venv' \
  /Users/light/dev/web-apps/voice-clone-pipeline/ \
  doc@100.83.78.111:~/dev/voice-clone-pipeline/

# Don't forget to update data paths after syncing
# Mac paths (/Users/light/...) need to be changed to Linux paths (/home/doc/...)
```

### Monitoring GPU

```bash
# GPU usage (run in separate tmux pane) - USE WSL PATH!
watch -n 1 /usr/lib/wsl/lib/nvidia-smi

# Standard nvidia-smi won't work in WSL2 - always use the WSL path above
```

### Quick SSH Commands

```bash
# Simple SSH connection
ssh doc@100.83.78.111

# Run a command with conda environment
ssh doc@100.83.78.111 "source ~/miniconda3/bin/activate && conda activate voice && <your_command>"

# Check GPU status
ssh doc@100.83.78.111 "/usr/lib/wsl/lib/nvidia-smi"

# Check PyTorch CUDA
ssh doc@100.83.78.111 "source ~/miniconda3/bin/activate && conda activate voice && python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
```

### Known Issues & Required Fixes for LoRA Training

**CRITICAL**: These fixes are required for LoRA training to work on RTX 4090:

1. **cuDNN Frontend Error with SDPA Attention**
   - Error: `RuntimeError: cuDNN Frontend error: No execution plans support the graph`
   - Fix: Add `attn_implementation="eager"` when loading the model in `train_lora_deepseek.py`:
   ```python
   base_model = CsmForConditionalGeneration.from_pretrained(
       model_path,
       trust_remote_code=True,
       attn_implementation="eager",  # REQUIRED - fixes cuDNN error
       torch_dtype=dtype,
   )
   ```

2. **MTP Dtype Mismatch**
   - Error: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float`
   - Fix: Disable MTP in `config/rtx_4090_lora.yaml`:
   ```yaml
   use_mtp: false  # MTP has dtype issues with LoRA
   ```

3. **Gradient Checkpointing Incompatibility**
   - Issue: Gradient checkpointing conflicts with eager attention implementation
   - Fix: Disable in `config/rtx_4090_lora.yaml`:
   ```yaml
   gradient_checkpointing: false
   ```

**Result**: With these fixes, LoRA training uses only 3.4GB VRAM (vs 22GB for full fine-tuning) and trains 0.07% of parameters (1.2M out of 1.5B), preventing overfitting on small datasets.

## Key Configuration Files

- `training/config/m4_pro_deepseek.yaml`: M4 Pro training config with DeepSeek techniques
- `training/config/m4_pro.yaml`: Standard M4 Pro config
- `training/config/rtx_4090_deepseek.yaml`: RTX 4090 full fine-tuning config (500+ samples only)
- `training/config/rtx_4090_lora.yaml`: RTX 4090 LoRA config (RECOMMENDED for <500 samples)

## Data Flow

1. Audio uploaded via `/upload` endpoint → saved to `data/raw/`
2. Resampled to 24kHz mono → `data/processed/`
3. Transcribed via Whisper, prosody analyzed → stored in memory
4. User approves sample → saved to `data/labeled/` as JSON + WAV
5. `prepare_dataset.py` creates train/val/test splits in `data/splits/`
6. Training outputs checkpoints to `models/checkpoints/`

## API Endpoints (Backend)

- `POST /upload`: Upload audio file
- `POST /process/{sample_id}`: Full pipeline (transcribe + analyze)
- `PATCH /sample/{sample_id}`: Update transcript/prosody/approval
- `GET /samples`: List all samples
- `POST /export`: Export approved samples as training dataset
- `GET /stats`: Dataset statistics

## Tech Stack

- **Frontend**: Next.js 14, React 18, Three.js, Recharts, Tailwind CSS
- **Backend**: FastAPI, PyTorch, torchaudio, Whisper, Parselmouth, Transformers
- **Training**: PyTorch, PEFT (LoRA), Accelerate, WandB

## FREE Local Claude Code (claude-free)

Run Claude Code for FREE using local Ollama models instead of paid Anthropic API.

### Setup (One-time)

```bash
# 1. Install Ollama (if not installed)
brew install ollama

# 2. Download qwen3-coder model (18GB)
ollama pull qwen3-coder:30b

# 3. Create 32k context model (REQUIRED - Claude Code needs 20k+ tokens for tools)
cat > /tmp/Modelfile.qwen3-coder-32k << 'EOF'
FROM qwen3-coder:30b
PARAMETER num_ctx 32768
EOF
ollama create qwen3-coder-32k -f /tmp/Modelfile.qwen3-coder-32k
```

### Usage

```bash
# Start Ollama (if not running)
ollama serve &

# Run FREE Claude Code
./scripts/claude-free

# Or in tmux
tmux new-session -s claude-free "./scripts/claude-free"
```

### Key Details

- **Model**: qwen3-coder-32k (30B MoE, ~3B active params)
- **Memory**: ~21GB GPU (fits on 48GB M4 Pro)
- **Context**: 32768 tokens (required for tool definitions)
- **Tools**: All Claude Code tools work (Bash, Read, Write, Edit, etc.)

### Troubleshooting

If tools don't work, check context size:
```bash
ollama ps  # Should show CONTEXT: 32768
```

If context is 4096, the model needs to be recreated with `num_ctx 32768`.

## AI Cost Optimization Strategy

Use FREE local models for most tasks, reserve paid APIs only when necessary.

### Task Routing

| Task Type | Tool | Cost |
|-----------|------|------|
| Lab management, research | `./scripts/claude-free` | FREE |
| Code writing, analytics | `./scripts/codex-free` | FREE |
| Complex multi-file refactors | Paid Claude (only if needed) | $$$ |

### Available FREE Tools

```bash
# Lab manager (Claude Code + Ollama)
./scripts/claude-free
tmux new-session -s claude-free "./scripts/claude-free"

# Code writing (Codex + Ollama)
./scripts/codex-free
```

### When to Use Paid Claude

Only use paid Anthropic API for:
- Complex architectural decisions requiring deep reasoning
- Multi-file refactors across 10+ files
- Time-critical tasks where 3-min response is too slow

For everything else, use the FREE local tools.
