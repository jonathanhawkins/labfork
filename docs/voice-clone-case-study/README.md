# Voice Clone Case Study

This folder contains documentation for the Voice Clone domain, which was the original research project that led to LabFork.

## Background

The Voice Clone lab explored a research question:

> **Can explicit multi-layer prosody labels improve voice cloning when training data is limited?**

This led to the development of:
- Multi-layer prosody analysis (semantic, acoustic, rhythm, contour)
- DeepSeek-inspired training techniques for faster convergence
- A full-stack pipeline from data collection to inference

## Documents

- [DEEPSEEK_TECHNIQUES.md](./DEEPSEEK_TECHNIQUES.md) - DeepSeek training techniques applied to voice cloning
- [RESEARCH_KEYFRAME_PROSODY.md](./RESEARCH_KEYFRAME_PROSODY.md) - Keyframe-based emotion interpolation research

## The "Prosody Cube" Concept

Instead of treating audio as a single signal, we analyze it from 4 perspectives simultaneously:

| Layer | What it captures | Tool |
|-------|------------------|------|
| **Semantic** | Emotion, intent, tone | Qwen2-Audio |
| **Acoustic** | Pitch, formants, harmonics | Parselmouth |
| **Rhythm** | Pauses, speaking rate, syllables | librosa |
| **Contour** | Pitch trajectory over time | Time-series analysis |

## Results (January 2025)

| Version | Description | F0 Correlation | Emotion Accuracy |
|---------|-------------|----------------|------------------|
| v1 Baseline | No prosody conditioning | -0.006 | N/A |
| v2 Prosody | + Prosody encoder | 0.328 | 0/4 (0%) |
| **v3 Energy** | + Intensity fix, + Energy predictor | **0.328** | **2/4 (50%)** |

**Key Achievement:** Fixed inverted pitch patterns
- Happy: 144 Hz -> **211 Hz** (now highest, correct)
- Sad: 274 Hz -> **167 Hz** (now lowest, correct)

**Conclusion:** SUPPORTED (with caveats)

The energy predictor auxiliary loss and intensity mapping fix demonstrate that explicit prosody conditioning works.

## Hardware Tested

| Hardware | Use Case | Notes |
|----------|----------|-------|
| M4 Pro (64GB) | Development, full fine-tune | Larger batches, slower per-step |
| RTX 4090 (24GB) | Production training | 3-4x faster, LoRA recommended |

## Related Code

- `backend/prosody_analyzer.py` - Multi-layer prosody analysis
- `training/train_lora_deepseek.py` - LoRA training with DeepSeek techniques
- `inference/test_pocket_tts.py` - Zero-shot voice cloning with Pocket TTS
- `.domains/voice-clone/` - Domain configuration

## From Case Study to Platform

This research project demonstrated the value of:
1. Structured domain configurations
2. AI agents finding and implementing papers
3. Real-time progress tracking

These insights led to LabFork - a platform where anyone can create research labs for any domain.
