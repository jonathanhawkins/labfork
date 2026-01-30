#!/usr/bin/env python3
"""
Extract Prosody for Emotion-Labeled Training Data (V6)

This script:
1. Uses RAVDESS emotion-labeled data (1440 samples)
2. Extracts full prosody features using CompleteProsodyAnalyzer
3. Creates train_large.json and val_large.json with emotion-balanced splits
4. Adds RAVDESS text transcripts (only 2 phrases in dataset)

RAVDESS Statements:
- Statement 01: "Kids are talking by the door"
- Statement 02: "Dogs are sitting by the door"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'backend'))

# RAVDESS transcript mapping (statement code -> text)
RAVDESS_TRANSCRIPTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


def get_ravdess_transcript(filename: str) -> str:
    """Extract transcript from RAVDESS filename."""
    # Format: 03-01-06-01-02-01-12.wav
    # Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor
    parts = Path(filename).stem.split("-")
    if len(parts) >= 5:
        statement_code = parts[4]
        return RAVDESS_TRANSCRIPTS.get(statement_code, "")
    return ""


def extract_prosody_batch(audio_paths: list, use_qwen: bool = False, device: str = "cpu"):
    """Extract prosody features for multiple audio files."""
    from prosody_analyzer import CompleteProsodyAnalyzer

    analyzer = CompleteProsodyAnalyzer(use_qwen=use_qwen, device=device)
    results = {}

    for audio_path in tqdm(audio_paths, desc="Extracting prosody"):
        try:
            result = analyzer.analyze(str(audio_path), transcript="")
            results[str(audio_path)] = result.to_dict()
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            results[str(audio_path)] = None

    return results


def normalize_prosody(prosody_dict: dict) -> dict:
    """
    Normalize prosody features to the format expected by training.

    Ensures:
    - semantic.emotion (str)
    - semantic.intensity (float, 0-1)
    - acoustic.pitch_mean (normalized Hz)
    - acoustic.pitch_std (normalized)
    - acoustic.energy (normalized RMS)
    - acoustic.speaking_rate (normalized)
    - rhythm.pause_ratio (0-1)
    - rhythm.syllable_rate (per second)
    - contour.pitch_contour (list of 64 normalized values)
    """
    if not prosody_dict:
        return None

    normalized = {"semantic": {}, "acoustic": {}, "rhythm": {}, "contour": {}}

    # Semantic - extract emotion label
    sem = prosody_dict.get("semantic", {}) or {}
    normalized["semantic"]["emotion"] = sem.get("emotion", "neutral")
    # Handle intensity (may be float or missing)
    intensity = sem.get("intensity", 0.5)
    if isinstance(intensity, (int, float)):
        normalized["semantic"]["intensity"] = float(intensity)
    else:
        normalized["semantic"]["intensity"] = 0.5
    normalized["semantic"]["emotion_confidence"] = sem.get("emotion_confidence", 0.8)

    # Acoustic - normalize pitch values
    aco = prosody_dict.get("acoustic", {}) or {}
    pitch_mean = aco.get("pitch_mean", 150.0)
    pitch_std = aco.get("pitch_std", 30.0)

    # Normalize pitch to 0-1 range (assume 50-400 Hz range)
    normalized["acoustic"]["pitch_mean"] = max(0, min(1, (pitch_mean - 50) / 350))
    normalized["acoustic"]["pitch_std"] = max(0, min(1, pitch_std / 100))

    # Energy (from intensity_mean or energy field)
    energy = aco.get("energy", aco.get("intensity_mean", 0.5))
    if energy > 1:  # If raw dB value, normalize
        energy = max(0, min(1, (energy + 50) / 100))
    normalized["acoustic"]["energy"] = float(energy)

    # Speaking rate
    rate = aco.get("speaking_rate", 0.5)
    if isinstance(rate, (int, float)) and rate > 2:  # If syllables/sec, normalize
        rate = min(1, rate / 10)
    normalized["acoustic"]["speaking_rate"] = float(rate)

    # Rhythm
    rhy = prosody_dict.get("rhythm", {}) or {}
    # Handle different field names
    pause_ratio = rhy.get("pause_ratio", rhy.get("speech_to_pause_ratio", 0.2))
    if pause_ratio > 1:  # speech_to_pause_ratio can be >1
        pause_ratio = 1 / (1 + pause_ratio)  # Convert to pause fraction
    normalized["rhythm"]["pause_ratio"] = float(pause_ratio)

    syllable_rate = rhy.get("syllable_rate", rhy.get("speaking_rate", 4.0))
    if syllable_rate > 1:  # If raw count, normalize
        syllable_rate = min(1, syllable_rate / 10)
    normalized["rhythm"]["syllable_rate"] = float(syllable_rate)

    # Contour - ensure 64-point normalized pitch trajectory
    con = prosody_dict.get("contour", {}) or {}
    pitch_contour = con.get("pitch_contour", con.get("smoothed", con.get("values", [])))

    if not pitch_contour or len(pitch_contour) == 0:
        pitch_contour = [0.5] * 64
    else:
        # Convert to list if needed
        if hasattr(pitch_contour, 'tolist'):
            pitch_contour = pitch_contour.tolist()

        # Remove zeros/NaN and normalize
        pitch_arr = np.array(pitch_contour, dtype=float)
        valid = pitch_arr[~np.isnan(pitch_arr) & (pitch_arr > 0)]

        if len(valid) > 0:
            # Normalize to 0-1
            p_min, p_max = valid.min(), valid.max()
            if p_max > p_min:
                pitch_arr = np.where(
                    (pitch_arr > 0) & (~np.isnan(pitch_arr)),
                    (pitch_arr - p_min) / (p_max - p_min),
                    0.5
                )
            else:
                pitch_arr = np.ones_like(pitch_arr) * 0.5
        else:
            pitch_arr = np.ones(len(pitch_arr)) * 0.5

        # Resample to 64 points
        if len(pitch_arr) != 64:
            indices = np.linspace(0, len(pitch_arr) - 1, 64)
            pitch_contour = np.interp(indices, np.arange(len(pitch_arr)), pitch_arr).tolist()
        else:
            pitch_contour = pitch_arr.tolist()

    normalized["contour"]["pitch_contour"] = pitch_contour

    return normalized


def create_balanced_split(samples: list, val_ratio: float = 0.1, seed: int = 42):
    """
    Create train/val split while maintaining emotion balance.

    Args:
        samples: List of sample dicts with prosody.semantic.emotion
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        (train_samples, val_samples)
    """
    random.seed(seed)

    # Group by emotion
    by_emotion = defaultdict(list)
    for s in samples:
        emotion = s.get("prosody", {}).get("semantic", {}).get("emotion", "unknown")
        by_emotion[emotion].append(s)

    train_samples = []
    val_samples = []

    print("\nEmotion distribution:")
    for emotion, emotion_samples in sorted(by_emotion.items()):
        n = len(emotion_samples)
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val

        random.shuffle(emotion_samples)
        train_samples.extend(emotion_samples[:n_train])
        val_samples.extend(emotion_samples[n_train:])

        print(f"  {emotion:12s}: {n:4d} total -> {n_train:4d} train, {n_val:3d} val")

    # Shuffle final lists
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="Extract prosody for emotion-labeled training data")
    parser.add_argument("--input", default="data/prosody_training/manifest.json",
                        help="Input manifest (RAVDESS prosody data)")
    parser.add_argument("--ravdess-dir", default="data/real_audio/ravdess",
                        help="RAVDESS audio directory (for re-extraction)")
    parser.add_argument("--output-dir", default="data/training",
                        help="Output directory for train/val JSONs")
    parser.add_argument("--re-extract", action="store_true",
                        help="Re-extract prosody using CompleteProsodyAnalyzer")
    parser.add_argument("--from-original", action="store_true",
                        help="Use original RAVDESS files (preserves transcripts)")
    parser.add_argument("--use-qwen", action="store_true",
                        help="Use Qwen2-Audio for semantic analysis (slow but better)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device for prosody extraction")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ravdess_dir = Path(args.ravdess_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing manifest
    print(f"\n{'='*60}")
    print("EMOTION-LABELED PROSODY EXTRACTION")
    print(f"{'='*60}")

    # Option 1: Use original RAVDESS files (preserves transcripts)
    if args.from_original:
        print("Using original RAVDESS files...")
        ravdess_meta = ravdess_dir / "metadata.json"
        if not ravdess_meta.exists():
            print(f"ERROR: RAVDESS metadata not found at {ravdess_meta}")
            return

        with open(ravdess_meta) as f:
            ravdess_data = json.load(f)

        samples = []
        for r in ravdess_data:
            # Get transcript from filename
            transcript = get_ravdess_transcript(r["audio_path"])
            # Map intensity string to float
            intensity = 0.5 if r.get("intensity") == "normal" else 0.8

            samples.append({
                "id": r["id"],
                "audio_path": r["audio_path"],
                "text": transcript,
                "duration": 0,
                "prosody": {
                    "semantic": {
                        "emotion": r["emotion"],
                        "intensity": intensity,
                        "emotion_confidence": 1.0,  # Ground truth label
                    },
                    "acoustic": {},
                    "rhythm": {},
                    "contour": {},
                },
                "source": "ravdess",
                "speaker_id": f"ravdess_actor_{r['actor']}",
            })
        print(f"Created {len(samples)} samples from RAVDESS metadata")

    # Option 2: Use existing prosody_training manifest
    elif input_path.exists():
        print(f"Loading existing manifest: {input_path}")
        with open(input_path) as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples")

    # Option 3: Create from scratch
    else:
        print(f"Manifest not found: {input_path}")
        print("Creating from RAVDESS metadata...")

        ravdess_meta = ravdess_dir / "metadata.json"
        if not ravdess_meta.exists():
            print(f"ERROR: RAVDESS metadata not found at {ravdess_meta}")
            return

        with open(ravdess_meta) as f:
            ravdess_data = json.load(f)

        samples = []
        for r in ravdess_data:
            samples.append({
                "id": r["id"],
                "audio_path": r["audio_path"],
                "text": "",
                "duration": 0,
                "prosody": {
                    "semantic": {
                        "emotion": r["emotion"],
                        "intensity": 0.5 if r.get("intensity") == "normal" else 0.8,
                    },
                    "acoustic": {},
                    "rhythm": {},
                    "contour": {},
                },
                "source": "ravdess",
                "speaker_id": f"ravdess_actor_{r['actor']}",
            })
        print(f"Created {len(samples)} sample entries")

    # Add transcripts to RAVDESS samples (if not already set)
    for sample in samples:
        if sample.get("source") == "ravdess" and not sample.get("text"):
            audio_path = sample.get("audio_path", "")
            sample["text"] = get_ravdess_transcript(audio_path)

    # Re-extract prosody if requested
    if args.re_extract:
        print(f"\nRe-extracting prosody features on {args.device}...")
        audio_paths = [s["audio_path"] for s in samples if s.get("audio_path")]

        # Make paths absolute
        abs_paths = []
        for p in audio_paths:
            if Path(p).is_absolute():
                abs_paths.append(p)
            else:
                # Try relative to project root
                full_path = project_root / "data/prosody_training" / p
                if full_path.exists():
                    abs_paths.append(str(full_path))
                else:
                    abs_paths.append(str(project_root / p))

        prosody_results = extract_prosody_batch(
            abs_paths,
            use_qwen=args.use_qwen,
            device=args.device
        )

        # Update samples with new prosody
        for sample in samples:
            audio_path = sample.get("audio_path", "")
            # Try to find matching result
            for path_key, prosody in prosody_results.items():
                if audio_path in path_key or path_key.endswith(audio_path):
                    if prosody:
                        # Preserve emotion label from original
                        orig_emotion = sample.get("prosody", {}).get("semantic", {}).get("emotion")
                        orig_intensity = sample.get("prosody", {}).get("semantic", {}).get("intensity", 0.5)

                        sample["prosody"] = prosody

                        # Restore emotion if Qwen wasn't used
                        if not args.use_qwen and orig_emotion:
                            sample["prosody"]["semantic"] = sample["prosody"].get("semantic") or {}
                            sample["prosody"]["semantic"]["emotion"] = orig_emotion
                            sample["prosody"]["semantic"]["intensity"] = orig_intensity
                    break

    # Normalize all prosody features
    print("\nNormalizing prosody features...")
    valid_samples = []
    for sample in tqdm(samples, desc="Normalizing"):
        prosody = normalize_prosody(sample.get("prosody"))
        if prosody:
            sample["prosody"] = prosody

            # Ensure audio path is absolute
            audio_path = sample.get("audio_path", "")
            if not Path(audio_path).is_absolute():
                # Check common locations
                for base in [
                    project_root / "data/prosody_training",
                    project_root / "data/real_audio/ravdess",
                    project_root,
                ]:
                    candidate = base / audio_path
                    if candidate.exists():
                        sample["audio_path"] = str(candidate)
                        break

            valid_samples.append(sample)

    print(f"Valid samples after normalization: {len(valid_samples)}")

    # Create balanced train/val split
    train_samples, val_samples = create_balanced_split(
        valid_samples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Save outputs
    train_path = output_dir / "train_large.json"
    val_path = output_dir / "val_large.json"

    print(f"\nSaving {len(train_samples)} training samples to {train_path}")
    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)

    print(f"Saving {len(val_samples)} validation samples to {val_path}")
    with open(val_path, "w") as f:
        json.dump(val_samples, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(valid_samples)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Emotion distribution
    emotions = defaultdict(int)
    for s in train_samples:
        e = s.get("prosody", {}).get("semantic", {}).get("emotion", "unknown")
        emotions[e] += 1

    print("\nTraining emotion distribution:")
    for e, c in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {e:12s}: {c:4d}")

    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    print("\nNext steps:")
    print("  1. Run V6 training:")
    print("     python training/train_prosody_conditioned.py --config config/prosody_v6.yaml")


if __name__ == "__main__":
    main()
