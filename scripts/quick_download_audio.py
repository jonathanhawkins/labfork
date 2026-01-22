#!/usr/bin/env python3
"""
Quick download script to get real LibriTTS-R audio samples from HuggingFace.
Uses streaming to download only what we need.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "data" / "real_audio" / "libritts_r"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def download_libritts_r_samples(num_samples: int = 50, diverse: bool = False):
    """Download real LibriTTS-R audio samples with annotations.

    Args:
        num_samples: Total number of samples to download
        diverse: If True, try to get samples from diverse speakers
    """

    print(f"Loading LibriTTS-R dataset with audio (streaming)...")

    # Load the actual LibriTTS-R dataset with audio from blabble-io
    dataset = load_dataset(
        "blabble-io/libritts_r",
        "clean",
        split="train.clean.100",
        streaming=True,
    )

    print(f"Downloading {num_samples} samples with real audio...")
    if diverse:
        print("Diverse mode: sampling from multiple speakers")

    downloaded = 0
    metadata = []
    speakers_seen = set()
    samples_per_speaker = 5 if diverse else num_samples

    for sample in tqdm(dataset, total=num_samples * 10):  # Iterate more to find diverse samples
        if downloaded >= num_samples:
            break

        try:
            sample_id = sample["id"]
            speaker_id = str(sample.get("speaker_id", ""))

            # In diverse mode, limit samples per speaker
            if diverse:
                speaker_count = sum(1 for m in metadata if m.get("speaker_id") == speaker_id)
                if speaker_count >= samples_per_speaker:
                    continue

            audio_path = AUDIO_DIR / f"{sample_id}.wav"

            # Check if already downloaded
            if audio_path.exists():
                downloaded += 1
                speakers_seen.add(speaker_id)
                continue

            # Get audio data
            audio_data = sample["audio"]
            audio_array = np.array(audio_data["array"], dtype=np.float32)
            sample_rate = audio_data["sampling_rate"]

            # Save audio
            sf.write(str(audio_path), audio_array, sample_rate)

            # Save metadata (blabble-io/libritts_r format)
            metadata.append({
                "id": sample_id,
                "text": sample.get("text_normalized", sample.get("text_original", "")),
                "speaker_id": speaker_id,
                "audio_path": str(audio_path),
                "sample_rate": sample_rate,
            })

            speakers_seen.add(speaker_id)
            downloaded += 1

        except Exception as e:
            print(f"Error downloading {sample.get('id', 'unknown')}: {e}")
            continue

    # Save metadata
    import json
    metadata_path = AUDIO_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDownloaded {downloaded} samples from {len(speakers_seen)} speakers to {AUDIO_DIR}")
    print(f"Metadata saved to {metadata_path}")

    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", "-n", type=int, default=50)
    parser.add_argument("--diverse", "-d", action="store_true",
                        help="Sample from diverse speakers (5 samples per speaker max)")
    args = parser.parse_args()

    download_libritts_r_samples(args.samples, diverse=args.diverse)
