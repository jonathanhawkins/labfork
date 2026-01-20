# Voice Clone Pipeline

A complete system for creating a personalized voice model using your own recordings.
**Now with DeepSeek techniques for 40% faster training!**

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   RECORD     │───▶│    LABEL     │───▶│    TRAIN     │───▶│   GENERATE   │
│   Your Voice │    │   Prosody    │    │   CSM-1B     │    │   Speech!    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                              │
                                    DeepSeek Techniques:
                                    • Multi-Token Prediction
                                    • DeepSeek LR Schedule
                                    • torch.compile (MPS)
```

## Quick Start

### 1. Setup (5 minutes)

**Mac (M4 Pro/M1/M2/M3):**
```bash
chmod +x scripts/setup_mac.sh
./scripts/setup_mac.sh
```

**Linux (RTX 4090/3090):**
```bash
chmod +x scripts/setup_linux.sh
./scripts/setup_linux.sh
```

### 2. Download Models (20-40 minutes)

```bash
python scripts/download_models.py
```

### 3. Start Recording

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Open:** http://localhost:3000

### 4. Collect Data (1-2 hours)

1. Click **Record** → speak naturally for 10-30 seconds
2. Click **Stop** → **Process**
3. Review prosody labels (emotion, tone, etc.)
4. Click **Approve** or **Discard**
5. Repeat until you have 60+ minutes

### 5. Train Model (2-6 hours)

```bash
cd training

# Prepare dataset
python prepare_dataset.py --input ../data/labeled --output ../data/splits

# Train with DeepSeek enhancements + live dashboard
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard

# Dashboard available at http://localhost:8001
# Frontend dashboard at http://localhost:3000/training
```

### 6. Monitor Training (Real-time Dashboard)

The training dashboard shows:
- **Loss curves** (train/val) with real-time updates
- **Memory usage** (optimized for M4 Pro's 64GB)
- **Learning rate schedule** (DeepSeek-style warmup → stable → decay)
- **Errors & warnings** for debugging
- **Training data viewer** with filtering

```bash
# Open standalone dashboard
open training-dashboard-demo.html

# Or use the full Next.js dashboard
cd frontend && npm run dev
# Then visit http://localhost:3000/training
```

### 7. Generate Speech

```bash
cd inference
python generate.py \
  --model ../models/checkpoints/voice_v1/best.pt \
  --text "Hello, this is my cloned voice!" \
  --output my_voice.wav
```

---

## Hardware Requirements

| Hardware | RAM/VRAM | Best For |
|----------|----------|----------|
| M4 Pro Mac | 64GB unified | Full fine-tune, best quality |
| RTX 4090 | 24GB VRAM | LoRA fine-tune, faster training |
| RTX 3090 | 24GB VRAM | LoRA fine-tune |
| M1/M2/M3 Mac | 32GB+ | LoRA fine-tune |

---

## Documentation

- **[PRD.md](./PRD.md)** - Full product requirements and architecture
- **[SETUP.md](./SETUP.md)** - Detailed installation guide (coming soon)
- **[TRAINING.md](./TRAINING.md)** - Training deep-dive (coming soon)

---

## What's Inside

```
voice-clone-pipeline/
├── backend/              # FastAPI server
│   ├── main.py          # API endpoints
│   └── prosody_analyzer.py  # Multi-layer prosody analysis
├── frontend/            # Next.js web UI
│   ├── app/page.tsx     # Recording interface
│   └── components/      # 3D visualizer
├── training/            # Training scripts
│   ├── prepare_dataset.py
│   ├── finetune_csm.py
│   └── config/          # M4/4090 configs
├── inference/           # Speech generation
│   └── generate.py
└── scripts/             # Setup utilities
```

---

## Key Features

✅ **Web-based Recording** - Record directly in browser
✅ **Auto Prosody Labeling** - Whisper + Qwen2-Audio + Parselmouth
✅ **3D Visualizer** - See data flow through analysis layers
✅ **Review Interface** - Approve/edit/discard samples
✅ **Export to Training** - One-click dataset export
✅ **M4 Pro Optimized** - Full fine-tune with 64GB memory
✅ **RTX 4090 Optimized** - LoRA fine-tune with FP16
✅ **DeepSeek Techniques** - 40% faster convergence
✅ **Training Dashboard** - Real-time monitoring with charts

---

## DeepSeek Techniques Applied

Our training pipeline incorporates innovations from DeepSeek-V3:

| Technique | Benefit | Implementation |
|-----------|---------|----------------|
| **Multi-Token Prediction** | 25% fewer steps | Predicts 4 tokens ahead |
| **DeepSeek LR Schedule** | Smoother training | Warmup → Stable → Cosine |
| **Aux-Loss-Free Balancing** | Better gradients | No auxiliary loss terms |
| **torch.compile (MPS)** | 10-20% speedup | Kernel fusion |

See [docs/DEEPSEEK_TECHNIQUES.md](docs/DEEPSEEK_TECHNIQUES.md) for details.

---

## The Prosody Cube

Your audio is analyzed from 4 perspectives:

```
           ┌─────────────────┐
          ╱│    SEMANTIC     │
         ╱ │  (emotion, tone)│
        ╱  └─────────────────┘
INPUT ─▶│        ●          │───▶ OUTPUT
       ╱│       CORE        │╲
      ╱ │     (prosody)     │ ╲
     ╱  └───────────────────┘  ╲
    │       ACOUSTIC            │
    │  (pitch, formants, HNR)   │
    └───────────────────────────┘
              RHYTHM
        (pauses, rate, syllables)
```

---

## License

MIT License - See LICENSE file.

---

*Created by Jonathan Hawkins | Aligned Tools*
