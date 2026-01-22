#!/usr/bin/env python3
"""
Download RAVDESS audio dataset - well-annotated emotional speech.
This is a smaller, easier to download dataset with clear annotations.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import json

BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "data" / "real_audio" / "ravdess"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# RAVDESS has standardized filenames:
# Modality-Vocal channel-Emotion-Emotional intensity-Statement-Repetition-Actor
# Emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}

def download_ravdess():
    """Download RAVDESS from Zenodo."""

    # RAVDESS is available on HuggingFace as narad/ravdess
    print("Loading RAVDESS dataset from HuggingFace...")

    from datasets import load_dataset

    try:
        # Try loading from HuggingFace
        dataset = load_dataset("narad/ravdess", split="train", streaming=True)

        downloaded = 0
        metadata = []
        max_samples = 100

        for sample in dataset:
            if downloaded >= max_samples:
                break

            try:
                # Get audio
                audio_data = sample["audio"]
                if not audio_data:
                    continue

                audio_array = audio_data["array"]
                sample_rate = audio_data["sampling_rate"]

                # Parse filename for metadata
                filename = sample.get("file", f"sample_{downloaded}")
                basename = Path(filename).stem

                # RAVDESS filename format: XX-XX-XX-XX-XX-XX-XX.wav
                parts = basename.split("-")
                if len(parts) >= 7:
                    emotion = EMOTION_MAP.get(parts[2], "unknown")
                    intensity = INTENSITY_MAP.get(parts[3], "unknown")
                    actor = parts[6]
                    gender = "female" if int(actor) % 2 == 0 else "male"
                else:
                    emotion = "unknown"
                    intensity = "unknown"
                    gender = "unknown"
                    actor = "unknown"

                # Save audio
                import soundfile as sf
                import numpy as np
                audio_path = AUDIO_DIR / f"{basename}.wav"
                sf.write(str(audio_path), np.array(audio_array, dtype=np.float32), sample_rate)

                metadata.append({
                    "id": basename,
                    "emotion": emotion,
                    "intensity": intensity,
                    "gender": gender,
                    "actor": actor,
                    "audio_path": str(audio_path),
                    "sample_rate": sample_rate,
                })

                downloaded += 1
                print(f"Downloaded {downloaded}/{max_samples}: {basename} - {emotion} ({gender})")

            except Exception as e:
                print(f"Error: {e}")
                continue

        # Save metadata
        metadata_path = AUDIO_DIR / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDownloaded {downloaded} samples to {AUDIO_DIR}")
        return downloaded

    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("\nTrying alternative: download from Zenodo...")
        return download_from_zenodo()


def download_from_zenodo():
    """Download directly from Zenodo as fallback."""

    # This is a single actor's speech files
    url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    zip_path = AUDIO_DIR / "ravdess.zip"

    print(f"Downloading from Zenodo (this may take a while)...")
    print(f"URL: {url}")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(AUDIO_DIR)

        # Clean up
        zip_path.unlink()
        print(f"Extracted to {AUDIO_DIR}")

        # Create metadata from extracted files
        create_metadata_from_files()
        return True

    except Exception as e:
        print(f"Zenodo download failed: {e}")
        return False


def create_metadata_from_files():
    """Create metadata.json from extracted files."""

    metadata = []
    for wav_file in AUDIO_DIR.rglob("*.wav"):
        basename = wav_file.stem
        parts = basename.split("-")

        if len(parts) >= 7:
            emotion = EMOTION_MAP.get(parts[2], "unknown")
            intensity = INTENSITY_MAP.get(parts[3], "unknown")
            actor = parts[6]
            gender = "female" if int(actor) % 2 == 0 else "male"
        else:
            continue

        metadata.append({
            "id": basename,
            "emotion": emotion,
            "intensity": intensity,
            "gender": gender,
            "actor": actor,
            "audio_path": str(wav_file),
        })

    metadata_path = AUDIO_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata for {len(metadata)} samples")


if __name__ == "__main__":
    download_ravdess()
