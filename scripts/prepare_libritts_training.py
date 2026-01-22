#!/usr/bin/env python3
"""
Prepare LibriTTS-R data for CSM training.
Converts downloaded audio + metadata into training format.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import random

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pyarrow.ipc as ipc

BASE_DIR = Path(__file__).parent.parent
AUDIO_DIR = BASE_DIR / "data" / "real_audio" / "libritts_r"
ANNOTATIONS_DIR = BASE_DIR / "data" / "libritts_annotated" / "train_clean_100"
OUTPUT_DIR = BASE_DIR / "data" / "training"


def load_annotations() -> Dict[str, Dict]:
    """Load ground truth annotations from arrow file."""
    arrow_path = ANNOTATIONS_DIR / "data-00000-of-00001.arrow"

    if not arrow_path.exists():
        print(f"Warning: No annotations file at {arrow_path}")
        return {}

    with open(arrow_path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    annotations = {}
    for i in range(table.num_rows):
        sample_id = table.column("id")[i].as_py()
        annotations[sample_id] = {
            "text": table.column("text")[i].as_py(),
            "speaker_id": str(table.column("speaker_id")[i].as_py()),
            "gender": table.column("gender")[i].as_py(),
            "pitch": table.column("pitch")[i].as_py(),
            "speaking_rate": table.column("speaking_rate")[i].as_py(),
            "speech_monotony": table.column("speech_monotony")[i].as_py(),
            "utterance_pitch_mean": float(table.column("utterance_pitch_mean")[i].as_py()),
            "utterance_pitch_std": float(table.column("utterance_pitch_std")[i].as_py()),
        }

    return annotations


def prepare_training_data():
    """Convert LibriTTS data to CSM training format."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = AUDIO_DIR / "metadata.json"
    with open(metadata_path) as f:
        samples = json.load(f)

    print(f"Found {len(samples)} audio samples")

    # Load annotations for prosody info
    annotations = load_annotations()
    print(f"Loaded {len(annotations)} annotations")

    # Build training samples
    training_samples = []

    for sample in samples:
        sample_id = sample["id"]
        audio_path = Path(sample["audio_path"])

        if not audio_path.exists():
            print(f"  Skipping {sample_id}: audio not found")
            continue

        # Get annotation data if available
        anno = annotations.get(sample_id, {})

        # Build training item in CSM format
        training_item = {
            "id": sample_id,
            "text": sample.get("text") or anno.get("text", ""),
            "path": str(audio_path),
            "speaker": int(sample.get("speaker_id", 0)),
        }

        # Add prosody metadata
        if anno:
            training_item["prosody"] = {
                "gender": anno.get("gender"),
                "pitch_category": anno.get("pitch"),
                "speaking_rate_category": anno.get("speaking_rate"),
                "pitch_mean_hz": anno.get("utterance_pitch_mean"),
                "pitch_std_hz": anno.get("utterance_pitch_std"),
            }

        training_samples.append(training_item)

    print(f"Prepared {len(training_samples)} training samples")

    # Split into train/val
    random.seed(42)
    random.shuffle(training_samples)

    n_val = max(5, int(len(training_samples) * 0.1))
    val_samples = training_samples[:n_val]
    train_samples = training_samples[n_val:]

    # Save
    train_path = OUTPUT_DIR / "train.json"
    val_path = OUTPUT_DIR / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=2)

    with open(val_path, "w") as f:
        json.dump(val_samples, f, indent=2)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_samples)} samples)")
    print(f"  Val: {val_path} ({len(val_samples)} samples)")

    # Print summary
    speakers = set(s["speaker"] for s in training_samples)
    total_text_chars = sum(len(s["text"]) for s in training_samples)

    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(training_samples)}")
    print(f"  Unique speakers: {len(speakers)}")
    print(f"  Total text: {total_text_chars:,} characters")
    print(f"  Avg text length: {total_text_chars // len(training_samples)} chars/sample")

    return train_samples, val_samples


if __name__ == "__main__":
    prepare_training_data()
