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

**CRITICAL: Don't go silent.** Provide brief progress updates:
- "Exploring X to find..."
- "Found issue in Y, fixing..."
- "Launching Z agent for..."
- "Completed A, moving to B..."

### 2. Mobile-First, Always
Every feature MUST work on phones. Touch targets ≥ 44px. No hover-only interactions.

### 3. Accessibility is Non-Negotiable
WCAG 2.1 AA compliance. Semantic HTML. Keyboard navigation. Screen reader support.

---

## Project Overview

LabFork is an open platform where users fork research labs and AI agents implement papers. The Voice Clone domain is one example.

**Frontend**: Next.js 14, React 18, Three.js, Tailwind CSS
**Backend**: FastAPI, PyTorch, Whisper, Transformers
**Ports**: Frontend 3003, Backend 8003

## Quick Start

```bash
# Frontend
cd frontend && npm install && npm run dev -- -p 3003

# Backend
cd backend && source venv/bin/activate && python main.py --port 8003
```

## Voice Clone Domain

The voice clone domain demonstrates prosody analysis and TTS training:

### Architecture
```
RECORD → LABEL → TRAIN → GENERATE
```

### Prosody Analysis (`backend/prosody_analyzer.py`)
- **Semantic**: Emotion/tone via Qwen2-Audio
- **Acoustic**: Pitch, formants, HNR via Parselmouth
- **Rhythm**: Pauses, speaking rate via librosa
- **Contour**: Pitch trajectory time-series

### Training Commands
```bash
cd training
python prepare_dataset.py --input ../data/labeled --output ../data/splits
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard
```

### Pocket TTS (Zero-Shot)
```bash
cd inference
python test_pocket_tts.py -r ../data/voice_samples/session/calm_1.wav -t "Your text"
```

## Remote GPU Training

Set in `.env`:
```bash
REMOTE_GPU_HOST=your-gpu-ip
REMOTE_GPU_USER=doc
```

```bash
ssh $REMOTE_GPU_USER@$REMOTE_GPU_HOST
cd ~/dev/labfork/training
python train_lora_deepseek.py --config config/rtx_4090_lora.yaml
```

**Training approach:**
- < 100 samples: LoRA only
- 100-500 samples: LoRA recommended
- 500+ samples: Full fine-tuning viable

## Key Config Files

- `training/config/rtx_4090_lora.yaml`: LoRA config (RECOMMENDED)
- `training/config/rtx_4090_deepseek.yaml`: Full fine-tuning

## LoRA Training Fixes (RTX 4090)

1. Add `attn_implementation="eager"` when loading model
2. Set `use_mtp: false` in config
3. Set `gradient_checkpointing: false`

## API Endpoints

- `POST /upload`: Upload audio
- `POST /process/{id}`: Transcribe + analyze
- `GET /samples`: List samples
- `POST /export`: Export training dataset

## Links

- **Repo**: https://github.com/jonathanhawkins/labfork
- **Docs**: [docs/OLLAMA_AND_COST_STRATEGY.md](docs/OLLAMA_AND_COST_STRATEGY.md)
