# Voice Clone Pipeline

> **Research Question:** Can explicit multi-layer prosody labels improve voice cloning when training data is limited?

A complete exploration toolkit for prosody-controlled voice synthesis. This project tests whether analyzing audio from multiple perspectives (semantic, acoustic, rhythm, contour) and using that as conditioning during training can improve voice cloning results with limited data.

![Landing Page](docs/screenshots/landing.png)

## The Hypothesis

Traditional voice cloning works well with large datasets, but prosody (the rhythm, intonation, and emphasis that make speech sound natural) is hard to capture. Sesame's CSM-1B model learns prosody from conversation context, but with limited personal data (20-60 minutes vs their 1M hours), we hypothesized that **explicit prosody labels could accelerate learning**.

### The "Prosody Cube" Concept

Instead of treating audio as a single signal, we analyze it from 4 perspectives simultaneously:

| Layer | What it captures | Tool |
|-------|------------------|------|
| **Semantic** | Emotion, intent, tone | Qwen2-Audio |
| **Acoustic** | Pitch, formants, harmonics | Parselmouth |
| **Rhythm** | Pauses, speaking rate, syllables | librosa |
| **Contour** | Pitch trajectory over time | Time-series analysis |

This multi-layer representation creates a richer training signal than emotion labels alone.

## Features

### Data Collection
- **Studio** - Record voice samples with real-time waveform visualization and auto prosody labeling
- **Perform** - Script-based emotional recording with guided prompts from movies, speeches, Shakespeare

### Voice Generation
- **Generate** - Create speech with prosody control, voice cloning (Pocket TTS), or style transfer
- **Author** - Keyframe timeline for emotion transitions (like video editing but for voice)
- **Live** - Real-time voice transformation via WebSocket

### Analysis
- **Compare** - A/B test base models vs fine-tuned models with metrics
- **Training** - Real-time dashboard with loss curves, LR schedule visualization, memory usage

## Tech Stack

**Frontend:** Next.js 14, React 18, Three.js (3D visualizations), Tailwind CSS, shadcn/ui

**Backend:** FastAPI, PyTorch, Whisper (transcription), Qwen2-Audio (emotion), Parselmouth (acoustics)

**Training:** CSM-1B, PEFT/LoRA, DeepSeek techniques (MTP, custom LR schedule)

**Voice Cloning:** Pocket TTS (zero-shot), CSM-1B fine-tuning

## Quick Start

### Prerequisites
- M4 Pro Mac (64GB) or RTX 4090 (24GB)
- Python 3.10+
- Node.js 18+

### Setup

```bash
# Clone
git clone https://github.com/yourusername/voice-clone-pipeline
cd voice-clone-pipeline

# Mac setup
./scripts/setup_mac.sh

# Or Linux/CUDA
./scripts/setup_linux.sh

# Download models
python scripts/download_models.py
```

### Run

```bash
# Terminal 1: Backend
cd backend && source venv/bin/activate && python main.py --port 8003

# Terminal 2: Frontend
cd frontend && npm run dev -- -p 3003
```

Open http://localhost:3003

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VOICE CLONE PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COLLECT          LABEL           TRAIN          GENERATE       │
│  ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐      │
│  │Studio│───────▶│Prosody│───────▶│CSM-1B│───────▶│Output│      │
│  │Perform│       │ Cube │        │+LoRA │        │      │      │
│  └──────┘        └──────┘        └──────┘        └──────┘      │
│                      │                               ▲          │
│                      ▼                               │          │
│              ┌───────────────┐                       │          │
│              │ Qwen2-Audio   │ Semantic              │          │
│              │ Parselmouth   │ Acoustic        ┌─────┴─────┐   │
│              │ librosa       │ Rhythm          │ Generate  │   │
│              │ Time-series   │ Contour         │ Author    │   │
│              └───────────────┘                 │ Live      │   │
│                                                └───────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Training

### DeepSeek Techniques Applied

We incorporated techniques from DeepSeek-V3 for faster convergence:

- **Multi-Token Prediction (MTP):** Predicts 4 tokens ahead for denser training signal
- **Custom LR Schedule:** Warmup → Stable → Cosine decay phases
- **LoRA:** For efficient fine-tuning with limited data (0.07% of parameters)

```bash
cd training

# LoRA training (recommended for < 500 samples)
python train_lora_deepseek.py --config config/rtx_4090_lora.yaml

# Full fine-tune (for larger datasets)
python train_csm_final.py --config config/rtx_4090_deepseek.yaml
```

## Project Structure

```
voice-clone-pipeline/
├── frontend/                # Next.js web UI
│   ├── app/
│   │   ├── page.tsx        # Landing page (research overview)
│   │   ├── studio/         # Recording interface
│   │   ├── perform/        # Script-based recording
│   │   ├── generate/       # Speech synthesis
│   │   ├── author/         # Keyframe emotion editor
│   │   ├── live/           # Real-time transformation
│   │   ├── compare/        # A/B model testing
│   │   └── training/       # Training dashboard
│   └── components/
│       └── ProsodyMatrixVisualizer.tsx  # 3D cube
│
├── backend/                 # FastAPI server
│   ├── main.py             # API endpoints
│   ├── prosody_analyzer.py # Multi-layer analysis
│   ├── keyframe_prosody.py # Emotion interpolation
│   └── live_voice_transformer.py  # WebSocket
│
├── training/               # Training scripts
│   ├── train_lora_deepseek.py
│   ├── train_csm_final.py
│   └── config/             # Hardware configs
│
└── inference/              # Generation scripts
```

## Evaluation Methodology

To test the hypothesis, we run controlled A/B comparisons:

### Experiment Design

| Model | Description |
|-------|-------------|
| **Baseline** | Standard LoRA fine-tuning, no prosody conditioning |
| **Prosody** | LoRA + multi-layer prosody conditioning |

Both models trained on identical data with identical hyperparameters.

### Metrics

| Metric | What it measures | Tool |
|--------|------------------|------|
| Speaker Similarity | Does it sound like you? | SpeechBrain ECAPA-TDNN |
| Emotion Accuracy | Does happy sound happy? | Qwen2-Audio classification |
| Prosody Match | Pitch/rhythm correlation | Parselmouth + librosa |
| MCD | Acoustic similarity | Mel Cepstral Distortion |
| Human Preference | Blind A/B listening test | Web interface |

### Running Evaluations

```bash
cd evaluation

# Run full A/B comparison
python run_ab_comparison.py \
    --baseline ../models/checkpoints/baseline_no_prosody/best.pt \
    --prosody ../models/checkpoints/prosody_joint/best.pt \
    --reference ../data/voice_samples/reference.wav

# Test emotion accuracy specifically
python evaluate_emotion_accuracy.py \
    --baseline-dir results/baseline/ \
    --prosody-dir results/prosody/

# Use the web interface for blind listening tests
# Navigate to /evaluate in the frontend
```

### Success Criteria

- **Hypothesis Supported:** Prosody model scores 5+ points higher on average
- **Partially Supported:** Prosody model better for emotion accuracy but not similarity
- **Not Supported:** No significant difference or baseline performs better

## Research Status

**Pipeline Complete** - All tools functional, end-to-end workflow works

**Evaluation Complete** - Three model iterations tested with progressive improvements

### Results (January 2025)

| Version | Description | F0 Correlation | Emotion Accuracy |
|---------|-------------|----------------|------------------|
| v1 Baseline | No prosody conditioning | -0.006 | N/A |
| v2 Prosody | + Prosody encoder | 0.328 | 0/4 (0%) |
| **v3 Energy** | + Intensity fix, + Energy predictor | **0.328** | **2/4 (50%)** |

**Key Achievement:** Fixed inverted pitch patterns
- Happy: 144 Hz → **211 Hz** (now highest, correct)
- Sad: 274 Hz → **167 Hz** (now lowest, correct)

**Conclusion:** SUPPORTED (with caveats)

The energy predictor auxiliary loss and intensity mapping fix demonstrate that explicit prosody conditioning works. Pitch patterns now correctly differentiate emotions. Happy and sad are correctly detected; angry/neutral still need work.

See [evaluation/RESULTS.md](evaluation/RESULTS.md) for full analysis.

---

This is an exploration project. The hypothesis is interesting but unproven. The tools demonstrate:
- Full-stack ML pipeline development
- Real-time audio processing
- Multi-model orchestration (Whisper, Qwen2-Audio, CSM, Pocket TTS)
- Modern web UI with 3D visualization

## Hardware Tested

| Hardware | Use Case | Notes |
|----------|----------|-------|
| M4 Pro (64GB) | Development, full fine-tune | Larger batches, slower per-step |
| RTX 4090 (24GB) | Production training | 3-4x faster, LoRA recommended |

## License

MIT License

---

*Built by Jonathan Hawkins | Aligned Tools*
