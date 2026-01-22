# Voice Clone Pipeline - Product Requirements Document

## Executive Summary

A complete pipeline for creating a custom voice model by:
1. Recording and collecting voice samples
2. Auto-labeling prosody (emotion, tone, rhythm) using AI
3. Fine-tuning Sesame's CSM-1B model on your voice
4. Generating speech that sounds like you

**Target Hardware:** M4 Pro Mac (64GB) + RTX 4090 (24GB)
**Timeline:** 2-4 weeks from setup to working voice clone
**Data Required:** 20-60 minutes of your voice recordings

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Phase 1: Environment Setup](#phase-1-environment-setup)
4. [Phase 2: Data Collection](#phase-2-data-collection)
5. [Phase 3: Prosody Labeling](#phase-3-prosody-labeling)
6. [Phase 4: Model Training](#phase-4-model-training)
7. [Phase 5: Inference & Testing](#phase-5-inference--testing)
8. [Hardware Considerations](#hardware-considerations)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VOICE CLONE PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐│
│  │   COLLECT    │───▶│    LABEL     │───▶│    TRAIN     │───▶│  GENERATE  ││
│  │              │    │              │    │              │    │            ││
│  │ • Record     │    │ • Whisper    │    │ • CSM-1B     │    │ • Your     ││
│  │ • Upload     │    │ • Qwen2-Audio│    │ • LoRA/Full  │    │   Voice!   ││
│  │ • Review     │    │ • Parselmouth│    │ • 4090/M4    │    │            ││
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘│
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      3D PROSODY VISUALIZER                            │  │
│  │   Shows data flowing through semantic/acoustic/rhythm/contour layers  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

**Sesame's Insight:** Voice naturalness comes from prosody (rhythm, intonation, emphasis), not just pronunciation. Their CSM model learns prosody from conversation context.

**Your Advantage:** With limited data (20-60 min vs Sesame's 1M hours), we add explicit prosody labels to help the model learn faster. Think of it like adding yeast to speed up fermentation.

**The "Cube" Concept:** Audio is analyzed from 4 perspectives:
- **Semantic:** What emotion/intent is expressed (Qwen2-Audio)
- **Acoustic:** Physical properties - pitch, intensity, formants (Parselmouth)
- **Rhythm:** Timing - pauses, speaking rate, syllables (librosa)
- **Contour:** Pitch trajectory over time (time-series data)

---

## Project Structure

```
voice-clone-pipeline/
├── README.md                 # This file
├── PRD.md                    # Full product requirements
├── SETUP.md                  # Detailed setup instructions
├── TRAINING.md               # Training guide
├── TROUBLESHOOTING.md        # Common issues and fixes
│
├── backend/                  # FastAPI server
│   ├── main.py              # API endpoints
│   ├── prosody_analyzer.py  # Multi-layer prosody extraction
│   ├── requirements.txt     # Python dependencies
│   └── config.py            # Configuration settings
│
├── frontend/                 # Next.js web UI
│   ├── app/
│   │   ├── page.tsx         # Main recording interface
│   │   ├── training/
│   │   │   └── page.tsx     # Real-time training dashboard
│   │   ├── layout.tsx       # App layout
│   │   └── globals.css      # Styles
│   ├── components/
│   │   └── ProsodyMatrixVisualizer.tsx  # 3D cube visualization
│   ├── package.json
│   └── tailwind.config.js
│
├── training/                 # Model training code
│   ├── prepare_dataset.py   # Convert recordings to training format
│   ├── train_deepseek.py    # DeepSeek-enhanced training (recommended)
│   ├── training_api.py      # Real-time training dashboard API
│   ├── finetune_csm.py      # Standard training script
│   ├── config/
│   │   ├── m4_pro_deepseek.yaml  # M4 Pro + DeepSeek config
│   │   ├── m4_pro.yaml           # M4 Pro standard config
│   │   └── rtx_4090.yaml         # RTX 4090 config
│   └── requirements.txt     # Training dependencies
│
├── inference/                # Voice generation
│   ├── generate.py          # Generate speech from text
│   ├── interactive.py       # Interactive CLI
│   └── api_server.py        # REST API for generation
│
├── scripts/                  # Utility scripts
│   ├── setup_mac.sh         # One-click Mac setup
│   ├── setup_linux.sh       # One-click Linux/CUDA setup
│   ├── download_models.py   # Download required models
│   └── validate_audio.py    # Check audio quality
│
├── data/                     # Data directory (created on setup)
│   ├── raw/                 # Original recordings
│   ├── processed/           # Resampled & cleaned
│   ├── labeled/             # With prosody labels
│   └── training/            # Train/val/test splits
│
├── models/                   # Model checkpoints (created on setup)
│   ├── csm-1b/              # Base CSM model
│   ├── whisper/             # Whisper for transcription
│   ├── qwen2-audio/         # Qwen2-Audio for prosody
│   └── checkpoints/         # Your fine-tuned models
│
├── visualizer-demo.html      # Standalone 3D visualizer demo
│
└── docs/                     # Additional documentation
    ├── DEEPSEEK_TECHNIQUES.md  # DeepSeek innovations explained
    ├── ARCHITECTURE.md         # Technical deep-dive
    ├── PROSODY_FEATURES.md     # Prosody feature reference
    └── API_REFERENCE.md        # API documentation
```

---

## Phase 1: Environment Setup

### Prerequisites

- **Mac:** M4 Pro with 64GB RAM (or M1/M2/M3 with 32GB+)
- **Linux/Windows:** RTX 4090 (24GB VRAM) or RTX 3090
- **Storage:** 50GB free space
- **Python:** 3.10 or 3.11
- **Node.js:** 18+ (for frontend)

### Quick Start (Mac)

```bash
# Clone the project
git clone <your-repo-url> voice-clone-pipeline
cd voice-clone-pipeline

# Run setup script
chmod +x scripts/setup_mac.sh
./scripts/setup_mac.sh

# This will:
# 1. Create Python virtual environment
# 2. Install all dependencies
# 3. Download required models (CSM-1B, Whisper, Qwen2-Audio)
# 4. Set up the frontend
# 5. Validate the installation
```

### Quick Start (Linux/CUDA)

```bash
# Clone the project
git clone <your-repo-url> voice-clone-pipeline
cd voice-clone-pipeline

# Run setup script
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh

# This will also install CUDA dependencies
```

### Manual Setup

See [SETUP.md](./SETUP.md) for detailed step-by-step instructions.

---

## Phase 2: Data Collection

### Recording Guidelines

**Duration Target:** 20-60 minutes total (2-4 hours ideal for best quality)

**Quality Requirements:**
- Sample rate: 24kHz or higher (will be resampled to 24kHz)
- Bit depth: 16-bit or higher
- Format: WAV preferred (MP3/M4A accepted)
- Environment: Quiet room, minimal reverb
- Microphone: Consistent position throughout

**Content Variety (Important!):**
- Questions: "How are you today?" "What do you think about that?"
- Statements: "I believe this is the right approach."
- Exclamations: "That's amazing!" "I can't believe it!"
- Casual: "Yeah, I mean, it's kind of interesting..."
- Formal: "Thank you for your consideration."
- Emotional range: Happy, thoughtful, concerned, excited

**What to Record:**
```
Option A: Read prepared scripts
  - News articles (formal)
  - Children's stories (expressive)
  - Conversations from books (varied)
  
Option B: Spontaneous speech
  - Talk about your day
  - Explain a topic you know well
  - Tell stories from memory
  
Option C: Mixed (Recommended)
  - 50% scripted, 50% spontaneous
  - Best variety for natural prosody
```

### Using the Recording Interface

1. Start the backend:
   ```bash
   cd backend
   source venv/bin/activate
   python main.py
   ```

2. Start the frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Open http://localhost:3000

4. Recording workflow:
   - Click **Record** to start
   - Speak naturally for 10-30 seconds
   - Click **Stop**
   - Wait for processing (transcription + prosody analysis)
   - **Review** the labels - correct if wrong
   - **Approve** good samples, **Discard** bad ones
   - Repeat until you have 20+ minutes

### Importing Existing Audio

If you have existing recordings:

```bash
# Place files in data/raw/
cp ~/my_recordings/*.wav data/raw/

# Run the import script
python scripts/import_audio.py --input data/raw/ --output data/processed/

# This will:
# - Resample to 24kHz mono
# - Split long files into segments
# - Validate audio quality
# - Flag problematic files
```

---

## Phase 3: Prosody Labeling

### Automatic Labeling Pipeline

```bash
# Run the full labeling pipeline
python backend/prosody_analyzer.py --input data/processed/ --output data/labeled/

# Or use the web interface for interactive labeling
```

### What Gets Labeled

**Acoustic Features (Parselmouth/Praat):**
```json
{
  "pitch_mean": 180.5,      // Average F0 in Hz
  "pitch_min": 120.0,       // Lowest pitch
  "pitch_max": 280.0,       // Highest pitch
  "pitch_range": 160.0,     // Expressivity indicator
  "pitch_std": 35.2,        // Pitch variation
  "intensity_mean": 72.5,   // Average volume (dB)
  "intensity_std": 8.3,     // Volume variation
  "jitter": 0.012,          // Pitch stability
  "shimmer": 0.045,         // Amplitude stability
  "hnr": 18.5,              // Voice clarity (harmonics-to-noise)
  "f1_mean": 500.0,         // First formant (tongue height)
  "f2_mean": 1800.0,        // Second formant (tongue position)
  "f3_mean": 2800.0         // Third formant (lip rounding)
}
```

**Rhythm Features (librosa):**
```json
{
  "duration_seconds": 4.5,
  "speaking_rate": 4.2,         // Syllables per second
  "articulation_rate": 5.1,     // Excluding pauses
  "pause_count": 2,
  "pause_total_duration": 0.8,
  "pause_mean_duration": 0.4,
  "speech_to_pause_ratio": 4.6,
  "syllable_count": 19
}
```

**Semantic Features (Qwen2-Audio):**
```json
{
  "emotion": "friendly",
  "emotion_confidence": 0.85,
  "tone": "conversational",
  "energy_level": "medium",
  "pace_category": "normal",
  "emphasis_words": ["really", "important"],
  "mood": "warm",
  "notes": "Slight uptick at end suggests openness"
}
```

**Pitch Contour (time-series):**
```json
{
  "times": [0.0, 0.08, 0.16, ...],
  "values": [175.2, 180.1, 178.5, ...],
  "smoothed": [176.0, 178.5, 179.0, ...]
}
```

### Manual Review

The web interface shows:
- Audio playback
- Transcript
- Auto-detected prosody labels
- Edit buttons to correct mistakes

**Pro Tip:** Focus on correcting emotion and emphasis_words - these have the biggest impact on training.

---

## Phase 4: Model Training

### Training Options

| Method | VRAM | Quality | Speed | Use Case |
|--------|------|---------|-------|----------|
| Full Fine-tune | 20-24GB | Best | Slow | RTX 4090, final model |
| LoRA | 12-16GB | Good | Fast | Limited VRAM, iteration |
| QLoRA | 8-10GB | OK | Fast | Very limited VRAM |
| M4 Pro (MPS) | 64GB unified | Good | Medium | Mac development |

### Prepare Dataset

```bash
# Convert labeled data to training format
python training/prepare_dataset.py \
  --input data/labeled/ \
  --output data/training/ \
  --val_split 0.1 \
  --test_split 0.05

# Output structure:
# data/training/
#   train.json
#   val.json
#   test.json
```

### Training on RTX 4090 (Recommended for Final Model)

```bash
cd training

# DeepSeek-enhanced training (recommended)
python train_deepseek.py \
  --config config/rtx_4090.yaml \
  --dashboard

# Standard fine-tune (alternative)
python finetune_csm.py \
  --config config/rtx_4090.yaml \
  --data_dir ../data/training/ \
  --output_dir ../models/checkpoints/my_voice_v1
```

### Training on M4 Pro Mac

```bash
cd training

# DeepSeek-enhanced training with real-time dashboard
python train_deepseek.py \
  --config config/m4_pro_deepseek.yaml \
  --dashboard

# Dashboard runs at http://localhost:8001
# Frontend dashboard at http://localhost:3000/training

# Standard fine-tune (alternative)
python finetune_csm.py \
  --config config/m4_pro.yaml \
  --data_dir ../data/training/ \
  --output_dir ../models/checkpoints/my_voice_v1

# Note: M4 is ~3-4x slower per step than 4090, but larger batches help
```

### Training Hyperparameters

```yaml
# config/m4_pro_deepseek.yaml (example)

# Model
model_path: "../models/csm-1b"  # Local path to downloaded model

# Data
data_dir: "../data/training"
max_audio_length_ms: 30000

# ============== DeepSeek Enhancements ==============

# Multi-Token Prediction (denser training signal)
use_mtp: true
mtp_heads: 4          # Predict 4 tokens ahead

# Multi-head Latent Attention (memory efficiency)
use_mla: true
mla_compression_ratio: 4

# DeepSeek Learning Rate Schedule
use_deepseek_lr: true
lr_warmup_ratio: 0.1    # 10% warmup
lr_stable_ratio: 0.6    # 60% stable
lr_decay_ratio: 0.3     # 30% cosine decay

# ============== M4 Pro Optimizations ==============

batch_size: 12          # Larger batches on M4
learning_rate: 2e-5
gradient_accumulation: 2
use_torch_compile: false  # MPS doesn't support compile yet
device: "mps"

# Training
num_epochs: 50
max_grad_norm: 1.0
gradient_checkpointing: true

# CSM-specific
freeze_codec: true      # Keep Mimi codec frozen (recommended)
```

**Key DeepSeek Features:**
- **MTP (Multi-Token Prediction):** Predicts 4 tokens ahead for denser training signal
- **MLA (Multi-head Latent Attention):** Compresses KV cache for memory efficiency
- **DeepSeek LR Schedule:** Warmup → Stable → Cosine decay phases
- **Codec Freezing:** Keeps Mimi codec frozen, trains only backbone (recommended by Sesame)

### Monitoring Training

**Using the Real-Time Dashboard (Recommended):**

```bash
# Run training with --dashboard flag
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard

# Dashboard API: http://localhost:8001
# Frontend UI: http://localhost:3000/training
```

The dashboard shows:
- Live training/validation loss curves
- Learning rate schedule visualization
- Memory usage and training speed
- Current epoch and step progress
- Real-time status updates via WebSocket

**Using TensorBoard (Alternative):**

```bash
tensorboard --logdir models/checkpoints/my_voice_v1/logs

# Watch for:
# - Loss decreasing steadily
# - Validation loss not diverging from train loss (overfitting)
# - Audio samples improving (check every 5 epochs)
```

### When to Stop

- **Loss plateau:** If validation loss hasn't improved in 5 epochs, stop
- **Audio quality:** Generate samples every 5-10 epochs, stop when they sound good
- **Overfitting:** If val loss increases while train loss decreases, stop or reduce learning rate

---

## Phase 5: Inference & Testing

### Generate Speech

```bash
cd inference

# Single generation
python generate.py \
  --model ../models/checkpoints/my_voice_v1/best.pt \
  --text "Hello, this is my cloned voice speaking!" \
  --output ../outputs/test_1.wav

# With context (better prosody)
python generate.py \
  --model ../models/checkpoints/my_voice_v1/best.pt \
  --text "I think that's a great idea." \
  --context "What do you think about starting the project next week?" \
  --output ../outputs/test_2.wav
```

### Interactive CLI

```bash
python interactive.py --model ../models/checkpoints/my_voice_v1/best.pt

# Then type text and hear it spoken
> Hello, how are you today?
[Playing audio...]

> I'm doing great, thanks for asking!
[Playing audio...]

> quit
```

### API Server

```bash
# Start the generation API
python api_server.py --model ../models/checkpoints/my_voice_v1/best.pt --port 8001

# Use from anywhere
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "context": ""}' \
  --output speech.wav
```

### Quality Evaluation

```bash
# Run evaluation suite
python evaluate.py \
  --model ../models/checkpoints/my_voice_v1/best.pt \
  --test_data ../data/training/test.json \
  --output_dir ../evaluation/

# Generates:
# - MOS estimation (naturalness score)
# - Speaker similarity score
# - Prosody accuracy metrics
# - Sample audio files for listening
```

---

## Hardware Considerations

### RTX 4090 (24GB VRAM)

**Strengths:**
- Fastest training (3-4x faster than M4)
- Can do full fine-tune with batch_size=4
- Best for final production training

**Settings:**
```yaml
batch_size: 4
gradient_accumulation: 4
gradient_checkpointing: true
mixed_precision: fp16
```

**Memory Usage:**
- Model: ~4GB
- Gradients: ~4GB
- Optimizer: ~8GB
- Activations: ~6GB
- Total: ~22GB

### M4 Pro (64GB Unified Memory)

**Strengths:**
- Huge memory allows larger batches
- Can run Qwen2-Audio + CSM simultaneously
- Great for development and experimentation

**Settings:**
```yaml
batch_size: 8-16
gradient_accumulation: 2
gradient_checkpointing: false  # Not needed with 64GB
mixed_precision: fp32  # MPS works better with fp32
```

**Memory Usage:**
- Model: ~4GB
- Gradients: ~4GB
- Optimizer: ~8GB
- Activations: ~12GB
- Total: ~28GB (plenty of headroom)

### Hybrid Workflow (Recommended)

1. **Develop on M4:** Fast iteration, large batches, easy debugging
2. **Train on 4090:** Final training runs, much faster
3. **Inference on either:** Both work well for generation

---

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch_size
- Enable gradient_checkpointing
- Use LoRA instead of full fine-tune

**"MPS backend not available"**
- Update PyTorch: `pip install torch --upgrade`
- Check macOS version (needs 12.3+)

**"Model generates garbage audio"**
- Check audio preprocessing (should be 24kHz mono)
- Verify transcripts are accurate
- Reduce learning rate
- Add more training data

**"Prosody sounds flat"**
- Include more emotional variety in training data
- Use longer context during generation
- Try temperature > 0.8 for more variation

**"Voice doesn't sound like me"**
- Need more training data (aim for 60+ minutes)
- Train longer (more epochs)
- Check for audio quality issues in recordings

### Getting Help

1. Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions
2. Review training logs in TensorBoard
3. Generate samples at different checkpoints to find best epoch
4. Compare prosody labels between good and bad samples

---

## Appendix: Key Concepts

### Why CSM Works

Traditional TTS: Text → Phonemes → Audio (loses prosody intent)
CSM: [Text + Context] → Audio (prosody emerges from context)

The key insight is the **one-to-many problem**: "I didn't say he stole the money" has 7 different meanings depending on which word you emphasize. CSM uses conversation context to pick the right prosody.

### RVQ Codebooks

Mimi tokenizer converts audio to 32 codebooks:
- **c0:** Semantic content (WHAT you're saying)
- **c1-c31:** Acoustic details (HOW it sounds)

CSM predicts c0 with the backbone, then c1-c31 with the decoder. This split is why it's efficient.

### Compute Amortization

Sesame's key innovation: Train decoder on only 1/16 of frames, not all frames. Quality stays the same, memory drops 50%. This is what makes CSM trainable on consumer hardware.

---

## Version History

- **v1.0** - Initial release with full pipeline
- Built for M4 Pro (64GB) + RTX 4090 (24GB)
- Based on Sesame CSM-1B, Whisper large-v3, Qwen2-Audio-7B

---

*Created by Jonathan Hawkins | Aligned Tools*
