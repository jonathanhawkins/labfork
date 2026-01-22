#!/usr/bin/env python3
"""
Download LibriTTS-R with prosody annotations from HuggingFace.
This dataset has pre-computed pitch, speaking rate, and quality metrics.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
import numpy as np


def pitch_category_to_value(category: str) -> float:
    """Convert pitch category to normalized value (0-1)."""
    mapping = {
        "very low-pitch": 0.1,
        "low-pitch": 0.25,
        "slightly low-pitch": 0.4,
        "moderate pitch": 0.5,
        "slightly high-pitch": 0.6,
        "high-pitch": 0.75,
        "very high-pitch": 0.9,
    }
    return mapping.get(category, 0.5)


def speed_category_to_value(category: str) -> float:
    """Convert speaking rate category to normalized value (0-1)."""
    mapping = {
        "very slowly": 0.1,
        "quite slowly": 0.25,
        "slightly slowly": 0.4,
        "moderate speed": 0.5,
        "slightly fast": 0.6,
        "quite fast": 0.75,
        "very fast": 0.9,
    }
    return mapping.get(category, 0.5)


def monotony_to_expressiveness(monotony: str) -> tuple[str, float]:
    """Convert monotony to emotion and intensity."""
    # Higher monotony = calmer/neutral, lower = more expressive
    mapping = {
        "very monotone": ("neutral", 0.3),
        "quite monotone": ("neutral", 0.4),
        "slightly monotone": ("calm", 0.5),
        "moderate intonation": ("neutral", 0.5),
        "slightly expressive": ("happy", 0.5),
        "quite expressive": ("happy", 0.6),
        "very expressive": ("excited", 0.7),
    }
    return mapping.get(monotony, ("neutral", 0.5))


def process_sample(sample: dict, output_dir: Path, idx: int) -> dict | None:
    """Process a single sample and save to disk."""
    try:
        # Get audio
        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]

        # Get text
        text = sample.get("text_normalized") or sample.get("text_original", "")
        if not text:
            return None

        # Get prosody annotations
        pitch_mean = sample.get("utterance_pitch_mean", 0)
        pitch_std = sample.get("utterance_pitch_std", 0)
        speaking_rate = sample.get("speaking_rate", 0)
        snr = sample.get("snr", 0)

        # Get categorical annotations if available
        pitch_cat = sample.get("pitch", "moderate pitch")
        speed_cat = sample.get("speaking_rate_bin", sample.get("speed", "moderate speed"))
        monotony = sample.get("speech_monotony", "moderate intonation")

        # Convert to our prosody format
        emotion, intensity = monotony_to_expressiveness(monotony)

        # Build prosody dict matching our format
        prosody = {
            "semantic": {
                "emotion": emotion,
                "intensity": intensity,
                "tone": "narrative" if "monotone" in str(monotony) else "conversational",
            },
            "acoustic": {
                "pitch_mean": float(pitch_mean) if pitch_mean else pitch_category_to_value(pitch_cat),
                "pitch_std": float(pitch_std) if pitch_std else 0.1,
                "energy": 0.5 + (snr / 100 if snr else 0),  # Normalize SNR to energy proxy
                "speaking_rate": speed_category_to_value(speed_cat),
            },
            "rhythm": {
                "pause_ratio": 0.2 if "slow" in str(speed_cat) else 0.1,
                "syllable_rate": speaking_rate if speaking_rate else 4.0,
            },
            "contour": {
                # Generate a simple contour based on monotony
                "pitch_contour": generate_contour(monotony, 64),
            },
        }

        # Save audio
        audio_path = output_dir / "audio" / f"sample_{idx:06d}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), audio_array, sample_rate)

        # Build output record
        record = {
            "id": f"libritts_{idx:06d}",
            "text": text,
            "audio_path": str(audio_path.relative_to(output_dir)),
            "duration": len(audio_array) / sample_rate,
            "prosody": prosody,
            "source": "libritts-r",
            "speaker_id": sample.get("speaker_id", "unknown"),
        }

        return record

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return None


def generate_contour(monotony: str, length: int = 64) -> list[float]:
    """Generate a pitch contour based on monotony level."""
    base = 0.5

    if "very monotone" in monotony:
        variation = 0.02
    elif "monotone" in monotony:
        variation = 0.05
    elif "expressive" in monotony:
        variation = 0.15
    else:
        variation = 0.08

    # Generate smooth contour with appropriate variation
    t = np.linspace(0, 2 * np.pi, length)
    contour = base + variation * np.sin(t) + variation * 0.3 * np.sin(2 * t)
    contour = np.clip(contour, 0, 1)

    return contour.tolist()


def main():
    parser = argparse.ArgumentParser(description="Download LibriTTS-R with prosody annotations")
    parser.add_argument("--output", type=str, default="data/libritts_prosody", help="Output directory")
    parser.add_argument("--split", type=str, default="train.clean.100", help="Dataset split to download")
    parser.add_argument("--max-samples", type=int, default=10000, help="Maximum samples to download")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Minimum audio duration in seconds")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Maximum audio duration in seconds")
    parser.add_argument("--min-snr", type=float, default=20.0, help="Minimum SNR for quality filtering")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LibriTTS-R dataset (split: {args.split})...")
    print("This may take a while on first run as it downloads from HuggingFace...")

    # Load the pre-annotated dataset
    dataset = load_dataset(
        "parler-tts/libritts-r-filtered-speaker-descriptions",
        "clean",
        split=args.split,
        trust_remote_code=True,
    )

    print(f"Loaded {len(dataset)} samples")
    print(f"Processing up to {args.max_samples} samples...")

    records = []
    processed = 0

    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        if processed >= args.max_samples:
            break

        # Quality filtering
        duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
        if duration < args.min_duration or duration > args.max_duration:
            continue

        snr = sample.get("snr", 0)
        if snr and snr < args.min_snr:
            continue

        record = process_sample(sample, output_dir, processed)
        if record:
            records.append(record)
            processed += 1

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone! Processed {len(records)} samples")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Audio saved to: {output_dir / 'audio'}")

    # Print stats
    total_duration = sum(r["duration"] for r in records)
    emotions = {}
    for r in records:
        e = r["prosody"]["semantic"]["emotion"]
        emotions[e] = emotions.get(e, 0) + 1

    print(f"\nDataset stats:")
    print(f"  Total duration: {total_duration / 3600:.1f} hours")
    print(f"  Emotion distribution: {emotions}")


if __name__ == "__main__":
    main()
