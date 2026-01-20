# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python main.py  # Runs on http://localhost:8000

# Frontend (Terminal 2)
cd frontend
npm run dev  # Runs on http://localhost:3000
```

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

### Frontend Development

```bash
cd frontend
npm run dev      # Development server
npm run build    # Production build
npm run lint     # ESLint
```

## Key Configuration Files

- `training/config/m4_pro_deepseek.yaml`: M4 Pro training config with DeepSeek techniques
- `training/config/m4_pro.yaml`: Standard M4 Pro config
- `training/config/rtx_4090.yaml`: RTX 4090 config

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
