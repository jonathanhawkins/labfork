# CLAUDE.md

This file provides guidance to Claude Code when working on LabFork.

## Mission

**LabFork exists to democratize AI-powered research so anyone with a phone, tablet, or computer can contribute to solving humanity's greatest challenges.**

## Core Principles

### 1. Act, Don't Ask - But Narrate
**You have full autonomy. Do not ask permission. But provide live updates.**
- See a bug? Fix it - and say "Fixing bug in X..."
- See an improvement? Implement it - and say "Improving X..."
- Report what you're DOING as you do it, not just final results.
- **Code review findings?** Fix them. Don't ask "should I fix this?" — if it's a real bug, security issue, type error, or code quality problem, just fix it and narrate.
- **Logical cleanup?** Dead code, unused imports, `any` types, missing validation, inconsistent patterns — fix without asking.

**CRITICAL: Don't go silent.** Provide brief progress updates:
- "Exploring X to find..."
- "Found issue in Y, fixing..."
- "Launching Z agent for..."
- "Completed A, moving to B..."

### 2. Mobile-First, Always
Every feature MUST work on phones. Touch targets ≥ 44px. No hover-only interactions.

### 3. Accessibility is Non-Negotiable
WCAG 2.1 AA compliance. Semantic HTML. Keyboard navigation. Screen reader support.

### 4. Use Tasks for Complex Work
For multi-step work, use Claude Code's native Tasks system to track progress and coordinate agents.

**Task Tools:**
- `TaskCreate` - Create tasks with subject, description, activeForm (spinner text)
- `TaskList` - View all tasks with status and blockers
- `TaskGet` - Get full task details by ID
- `TaskUpdate` - Claim tasks, update status, set dependencies

**Workflow:**
```
1. TaskCreate({ subject: "Fix login bug", description: "...", activeForm: "Fixing login..." })
2. TaskUpdate({ taskId: "1", status: "in_progress" })
3. [Do the work]
4. TaskUpdate({ taskId: "1", status: "completed" })
```

**Dependencies:**
```
TaskUpdate({ taskId: "2", addBlockedBy: ["1"] })  // Task 2 waits for Task 1
```

---

## Project Overview

LabFork is an open platform where users fork research labs and AI agents implement papers. The Voice Clone domain is one example.

**Frontend**: Next.js 14, React 18, Three.js, Tailwind CSS
**Backend**: FastAPI, PyTorch, Whisper, Transformers
**Ports**: Frontend 3003, Backend 8003

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   RECORD     │───▶│    LABEL     │───▶│    TRAIN     │───▶│   GENERATE   │
│   (Frontend) │    │   (Backend)  │    │  (Training)  │    │  (Inference) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

- **Frontend (Next.js)**: Web-based recording interface with 3D prosody visualizer
- **Backend (FastAPI)**: Audio upload, transcription (Whisper), prosody analysis
- **Training**: CSM-1B fine-tuning with DeepSeek techniques (MTP, MLA)
- **Inference**: Speech generation from fine-tuned models

## Quick Start

```bash
# Backend with remote 4090 GPU (recommended)
./start-backend.sh

# Or manually:
cd backend && source venv/bin/activate && python main.py --port 8003

# Frontend
cd frontend && npm install && npm run dev -- -p 3003
```

### Development Setup

```bash
# Mac setup (creates venv, installs deps)
./scripts/setup_mac.sh

# Linux/CUDA setup
./scripts/setup_linux.sh

# Download models (CSM-1B, Whisper, Qwen2-Audio)
python scripts/download_models.py
```

## Voice Clone Domain

### Prosody Analysis (`backend/prosody_analyzer.py`)

Audio is analyzed from 4 perspectives:
- **Semantic**: Emotion/tone detection via Qwen2-Audio
- **Acoustic**: Pitch, formants, HNR via Parselmouth
- **Rhythm**: Pauses, speaking rate, syllables via librosa
- **Contour**: Pitch trajectory time-series

### DeepSeek Training Enhancements

The training pipeline (`training/train_deepseek.py`) implements:
- **Multi-Token Prediction (MTP)**: Predicts 4 tokens ahead for denser training signal
- **Multi-head Latent Attention (MLA)**: Compresses KV cache for memory efficiency
- **DeepSeek LR Schedule**: Warmup → Stable → Cosine decay phases

### Training Commands

```bash
cd training

# Prepare dataset from labeled samples
python prepare_dataset.py --input ../data/labeled --output ../data/splits

# Train with DeepSeek enhancements + dashboard
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard
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

Pocket TTS (by Kyutai) is a 100M parameter TTS model that does zero-shot voice cloning at inference time.

### Usage

```bash
cd inference

# Clone voice from a reference audio
python test_pocket_tts.py -r ../data/voice_samples/session/calm_1.wav -t "Your text"

# Generate all emotional variations
python test_pocket_tts.py --all-emotions

# List available samples
python test_pocket_tts.py --list-samples
```

### Python API

```python
from pocket_tts import TTSModel
import scipy.io.wavfile as wav
import numpy as np

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt("path/to/reference.wav")
audio = model.generate_audio(voice_state, "Text to synthesize")

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
| Prosody control | Limited | Custom (our goal) |

## Remote RTX 4090 Connection

The 4090 runs WSL2 on Windows. SSH to `doc@100.100.219.33` (WSL Tailscale IP).

**Environment Variables** (set in `.env` or by `./start-backend.sh`):
```bash
REMOTE_GPU_HOST=100.100.219.33
REMOTE_GPU_USER=doc
```

### SSH Commands

```bash
# Test SSH connection
ssh doc@100.100.219.33 "nvidia-smi"

# Check GPU stats via API (requires backend running)
curl http://localhost:8003/api/lab/gpu-stats

# Attach to existing tmux training session
ssh doc@100.100.219.33 -t "source ~/miniconda3/bin/activate && conda activate voice && tmux attach -t training"

# Create new training session
ssh doc@100.100.219.33 -t "tmux new-session -s training"
```

### Running Training on RTX 4090

**IMPORTANT**: Use LoRA training for small datasets (< 500 samples) to prevent overfitting!

```bash
ssh doc@100.100.219.33
source ~/miniconda3/bin/activate && conda activate voice
cd ~/dev/labfork/training

# RECOMMENDED: LoRA training (0.07% of params)
python train_lora_deepseek.py --config config/rtx_4090_lora.yaml

# Alternative: Full fine-tuning (500+ samples only)
python train_csm_final.py --config config/rtx_4090_deepseek.yaml
```

**Training approach:**
- < 100 samples: LoRA only
- 100-500 samples: LoRA recommended
- 500+ samples: Full fine-tuning viable

### Syncing Project Files

```bash
# From Mac to 4090 (run on Mac)
rsync -avz --progress --exclude 'node_modules' --exclude '.next' --exclude 'venv' \
  /Users/light/dev/web-apps/voice-clone-pipeline/ \
  doc@100.100.219.33:~/dev/labfork/
```

### GPU Monitoring

```bash
# In separate tmux pane (WSL-specific path)
watch -n 1 /usr/lib/wsl/lib/nvidia-smi
```

## LoRA Training Fixes (RTX 4090)

**CRITICAL**: These fixes are required for LoRA training:

1. **cuDNN Frontend Error with SDPA Attention**
   ```python
   # Add attn_implementation="eager" when loading model
   base_model = CsmForConditionalGeneration.from_pretrained(
       model_path,
       attn_implementation="eager",  # REQUIRED
       torch_dtype=dtype,
   )
   ```

2. **MTP Dtype Mismatch** - Set `use_mtp: false` in config

3. **Gradient Checkpointing** - Set `gradient_checkpointing: false`

**Result**: LoRA uses 3.4GB VRAM (vs 22GB for full fine-tuning), trains 0.07% of parameters.

## Key Config Files

- `training/config/rtx_4090_lora.yaml`: LoRA config (RECOMMENDED)
- `training/config/rtx_4090_deepseek.yaml`: Full fine-tuning
- `training/config/m4_pro_deepseek.yaml`: M4 Pro config

## Data Flow

1. Audio uploaded via `/upload` → saved to `data/raw/`
2. Resampled to 24kHz mono → `data/processed/`
3. Transcribed via Whisper, prosody analyzed → stored in memory
4. User approves sample → saved to `data/labeled/` as JSON + WAV
5. `prepare_dataset.py` creates train/val/test splits in `data/splits/`
6. Training outputs checkpoints to `models/checkpoints/`

## API Endpoints

- `POST /upload`: Upload audio file
- `POST /process/{sample_id}`: Full pipeline (transcribe + analyze)
- `PATCH /sample/{sample_id}`: Update transcript/prosody/approval
- `GET /samples`: List all samples
- `POST /export`: Export approved samples as training dataset
- `GET /stats`: Dataset statistics

## Frontend Development

```bash
cd frontend
npm run dev      # Development server
npm run build    # Production build
npm run lint     # ESLint
```

## Tech Stack

- **Frontend**: Next.js 14, React 18, Three.js, Recharts, Tailwind CSS
- **Backend**: FastAPI, PyTorch, torchaudio, Whisper, Parselmouth, Transformers
- **Training**: PyTorch, PEFT (LoRA), Accelerate, WandB

## Links

- **Repo**: https://github.com/jonathanhawkins/labfork
- **Docs**: [docs/OLLAMA_AND_COST_STRATEGY.md](docs/OLLAMA_AND_COST_STRATEGY.md)
