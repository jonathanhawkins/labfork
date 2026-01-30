# Real Audio Testing Plan for Prosody Analysis

## Problem Statement

The current prosody comparison uses **synthetic audio** generated to match ground truth parameters. This is circular - it only validates that the analyzer can detect features that were programmatically inserted. The synthetic audio:

- Has near-zero pitch error (because pitch was explicitly programmed)
- Has 0% speaking rate match (doesn't simulate real syllables)
- Has 13% gender inference accuracy (pitch ranges overlap incorrectly)

We need **real speech audio** with **ground truth prosody annotations** to properly validate the Sesame/CSM-style prosody analysis pipeline.

## Available Datasets

### 1. LibriTTS-R (RECOMMENDED)

**Best match for our existing annotations**

| Attribute | Value |
|-----------|-------|
| Source | [HuggingFace: blabble-io/libritts_r](https://huggingface.co/datasets/blabble-io/libritts_r) |
| Audio | Yes - 24kHz WAV files |
| Size | 8.1GB (train-clean-100), streamable |
| Annotations | pitch (Hz + category), speaking_rate, gender, speech_monotony, text_description |
| License | CC BY 4.0 |

**Why it's perfect:**
- Annotations match our local `parler-tts/libritts_r_tags_tagged_10k_generated` dataset exactly
- Same 33,232 samples in train-clean-100
- Streaming support for efficient downloading of subsets
- Professional-quality restored speech

**Download command:**
```bash
python scripts/download_audio_datasets.py --dataset libritts_r --samples 100
```

### 2. RAVDESS

**Best for emotion recognition testing**

| Attribute | Value |
|-----------|-------|
| Source | [Zenodo: 1188976](https://zenodo.org/records/1188976), [HuggingFace: narad/ravdess](https://huggingface.co/datasets/narad/ravdess) |
| Audio | Yes - 48kHz WAV files |
| Size | ~2.4GB |
| Annotations | emotion, intensity, actor gender |
| License | CC BY-NC-SA 4.0 |

**Features:**
- 24 professional actors (12 female, 12 male)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 2 intensity levels: normal, strong
- Controlled recording environment

**Download command:**
```bash
python scripts/download_audio_datasets.py --dataset ravdess --samples 100
```

### 3. CREMA-D

**Crowdsourced emotional speech**

| Attribute | Value |
|-----------|-------|
| Source | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |
| Audio | Yes |
| Size | ~2.6GB |
| Annotations | emotion, emotion_level, demographics |
| License | Open Database License |

**Features:**
- 7,442 clips from 91 actors
- Diverse demographics (age, race, gender)
- Crowd-sourced emotion ratings
- ToBI framework prosody research available

### 4. EmoV-DB

**Emotional voice database for TTS**

| Attribute | Value |
|-----------|-------|
| Source | [OpenSLR: 115](https://www.openslr.org/115/) |
| Audio | Yes |
| Size | ~5GB |
| Annotations | emotion, phonetic alignment (TextGrid) |
| License | CC BY 4.0 |

**Features:**
- 4 speakers (2 male, 2 female)
- 5 emotions: neutral, sleepy, angry, disgust, amused
- Montreal Forced Aligner phoneme alignments
- Non-verbal vocalizations (laughs, yawns)

## Implementation Plan

### Phase 1: Download LibriTTS-R Audio (Immediate)

```bash
# Download 100 samples with full annotations
cd /Users/light/dev/web-apps/labfork
python scripts/download_audio_datasets.py --dataset libritts_r --samples 100
```

This will:
1. Stream audio from `blabble-io/libritts_r`
2. Match with annotations from `parler-tts/libritts_r_tags_tagged_10k_generated`
3. Save to `data/audio_datasets/libritts_r/`
4. Create `metadata.json` with full prosody annotations

**Expected output:**
- `data/audio_datasets/libritts_r/audio/` - 100 WAV files (~50-100MB)
- `data/audio_datasets/libritts_r/metadata.json` - annotations

### Phase 2: Run Real Audio Comparison

```bash
# Compare against real audio
python scripts/compare_prosody_real_audio.py --samples 50

# With semantic analysis (slower)
python scripts/compare_prosody_real_audio.py --use-qwen --samples 20
```

This will:
1. Load real audio from downloaded samples
2. Run acoustic analysis (Parselmouth)
3. Run rhythm analysis (librosa)
4. Optionally run semantic analysis (Qwen2-Audio)
5. Compare with ground truth annotations
6. Generate detailed comparison report

### Phase 3: Analyze Results

The comparison will measure:

**Pitch Analysis:**
- Mean absolute error (Hz) between detected and ground truth
- Percentage error
- Category match rate (very low / quite low / moderate / etc.)

**Speaking Rate:**
- Syllables per second vs. categorical annotations
- Category match rate

**Voice Quality:**
- Jitter (pitch stability)
- Shimmer (amplitude stability)
- HNR (harmonics-to-noise ratio)

**Gender Inference:**
- Accuracy based on F0

### Phase 4: Iterate and Improve

Based on results:
1. Tune category boundaries for pitch/rate mapping
2. Improve syllable detection algorithm
3. Calibrate gender inference thresholds
4. Add RAVDESS for emotion detection testing

## Expected Improvements

With real audio, we expect to see:

| Metric | Synthetic (Current) | Real Audio (Expected) |
|--------|--------------------|-----------------------|
| Pitch MAE | 0.05 Hz (artificial) | 10-30 Hz (realistic) |
| Pitch Category Match | 53% | 60-80% (target) |
| Speaking Rate Match | 0% | 40-60% (target) |
| Gender Inference | 13% | 70-90% (target) |

## Technical Notes

### LibriTTS-R Annotation Format

The `parler-tts/libritts_r_tags_tagged_10k_generated` dataset includes:

```json
{
  "id": "730_358_000003_000002",
  "text": "The moon I gazed with a kind of wonder.",
  "speaker_id": "730",
  "speaking_rate": "moderate speed",
  "gender": "female",
  "pitch": "quite low pitch",
  "utterance_pitch_mean": 144.72,
  "utterance_pitch_std": 35.88,
  "speech_monotony": "slightly monotone",
  "snr": 42.5,
  "c50": 15.2,
  "noise": "very noisy",
  "reverberation": "slightly reverberant",
  "text_description": "A female speaker with quite low pitch delivers..."
}
```

### Sesame/CSM Prosody Analysis

Our analyzer extracts similar features:

```python
# Acoustic features (Parselmouth)
pitch_mean, pitch_std, pitch_range
jitter, shimmer, hnr
f1_mean, f2_mean, f3_mean

# Rhythm features (librosa)
speaking_rate (syllables/sec)
articulation_rate
pause_count, pause_duration

# Semantic features (Qwen2-Audio)
emotion, tone, energy_level
pace_category, emphasis_words
```

## Quick Start Commands

```bash
# 1. Download real audio (100 samples, ~50MB)
python scripts/download_audio_datasets.py --dataset libritts_r --samples 100

# 2. Run comparison
python scripts/compare_prosody_real_audio.py --samples 50

# 3. Check results
cat data/prosody_comparison_results/real_audio_comparison_*.txt
```

## Sources

- [LibriTTS-R Audio (HuggingFace)](https://huggingface.co/datasets/blabble-io/libritts_r)
- [LibriTTS-R Annotations (Parler-TTS)](https://huggingface.co/datasets/parler-tts/libritts_r_tags_tagged_10k_generated)
- [LibriTTS-R OpenSLR](https://www.openslr.org/141/)
- [RAVDESS (Zenodo)](https://zenodo.org/records/1188976)
- [RAVDESS (HuggingFace)](https://huggingface.co/datasets/narad/ravdess)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [EmoV-DB](https://www.openslr.org/115/)
