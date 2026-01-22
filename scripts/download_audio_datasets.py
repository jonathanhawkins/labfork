#!/usr/bin/env python3
"""
Audio Dataset Downloader for Prosody Analysis Testing

This script downloads real audio datasets with prosody/emotion annotations
to properly test the prosody analyzer against ground truth data.

Available datasets:
1. LibriTTS-R (HuggingFace) - Matches our existing annotations, has audio
2. RAVDESS - Emotion-labeled speech with professional actors
3. CREMA-D - Crowdsourced emotional speech dataset
4. EmoV-DB - Emotional voice database with phonetic annotations

Usage:
    python download_audio_datasets.py --dataset libritts_r --samples 100
    python download_audio_datasets.py --dataset ravdess --all
    python download_audio_datasets.py --list
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import urllib.request
import zipfile
import tarfile
import shutil


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_CACHE_DIR = DATA_DIR / "audio_datasets"


@dataclass
class DatasetInfo:
    """Information about an available dataset."""
    name: str
    description: str
    source: str
    size_estimate: str
    has_audio: bool
    annotations: List[str]
    download_method: str  # 'huggingface', 'zenodo', 'openslr', 'kaggle'
    license: str


AVAILABLE_DATASETS = {
    "libritts_r": DatasetInfo(
        name="LibriTTS-R",
        description="Sound quality improved LibriTTS corpus with prosody annotations",
        source="huggingface:blabble-io/libritts_r",
        size_estimate="8.1GB for train-clean-100 (full), ~50MB for 100 samples",
        has_audio=True,
        annotations=["pitch", "speaking_rate", "gender", "speech_monotony", "text_description"],
        download_method="huggingface",
        license="CC BY 4.0",
    ),
    "ravdess": DatasetInfo(
        name="RAVDESS",
        description="Ryerson Audio-Visual Database of Emotional Speech and Song",
        source="zenodo:1188976",
        size_estimate="~2.4GB (full), ~50MB for subset",
        has_audio=True,
        annotations=["emotion", "intensity", "actor_gender"],
        download_method="zenodo",
        license="CC BY-NC-SA 4.0",
    ),
    "crema_d": DatasetInfo(
        name="CREMA-D",
        description="Crowd-sourced Emotional Multimodal Actors Dataset",
        source="github:CheyneyComputerScience/CREMA-D",
        size_estimate="~2.6GB",
        has_audio=True,
        annotations=["emotion", "emotion_level", "actor_demographics"],
        download_method="github",
        license="Open Database License",
    ),
    "emov_db": DatasetInfo(
        name="EmoV-DB",
        description="Emotional Voices Database for speech synthesis",
        source="openslr:115",
        size_estimate="~5GB",
        has_audio=True,
        annotations=["emotion", "phonetic_alignment", "speaker_id"],
        download_method="openslr",
        license="CC BY 4.0",
    ),
}


def list_datasets():
    """Print information about available datasets."""
    print("=" * 70)
    print("AVAILABLE AUDIO DATASETS FOR PROSODY TESTING")
    print("=" * 70)
    print()

    for key, info in AVAILABLE_DATASETS.items():
        print(f"Dataset: {info.name}")
        print(f"  Key: {key}")
        print(f"  Description: {info.description}")
        print(f"  Source: {info.source}")
        print(f"  Size: {info.size_estimate}")
        print(f"  Annotations: {', '.join(info.annotations)}")
        print(f"  License: {info.license}")
        print()

    print("=" * 70)
    print("RECOMMENDED FOR PROSODY COMPARISON:")
    print("=" * 70)
    print()
    print("1. LibriTTS-R (libritts_r)")
    print("   - BEST MATCH: Has the same prosody annotations we already have locally")
    print("   - Includes: pitch (Hz), speaking_rate, gender, speech_monotony")
    print("   - Can stream small subsets via HuggingFace")
    print("   - Command: python download_audio_datasets.py --dataset libritts_r --samples 100")
    print()
    print("2. RAVDESS (ravdess)")
    print("   - Professional actor recordings with emotion labels")
    print("   - Good for testing emotion detection aspects")
    print("   - Smaller, self-contained dataset")
    print("   - Command: python download_audio_datasets.py --dataset ravdess --all")
    print()


def download_libritts_r_streaming(
    output_dir: Path,
    num_samples: int = 100,
    split: str = "train.clean.100"
) -> List[Dict[str, Any]]:
    """
    Download LibriTTS-R samples using HuggingFace streaming.

    This is the most efficient method as it:
    1. Streams only the samples we need
    2. Matches our existing annotation format
    3. Provides ground truth for direct comparison
    """
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install datasets soundfile")
        return []

    print(f"Downloading {num_samples} samples from LibriTTS-R ({split})...")
    print("Using HuggingFace streaming to minimize download size...")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Load dataset with streaming
    dataset = load_dataset(
        "blabble-io/libritts_r",
        "clean",
        split=split,
        streaming=True
    )

    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        sample_id = sample["id"]
        print(f"  [{i+1}/{num_samples}] {sample_id}")

        # Save audio file
        audio_data = sample["audio"]
        audio_path = audio_dir / f"{sample_id}.wav"

        # HuggingFace audio format: {"array": np.array, "sampling_rate": int}
        sf.write(
            str(audio_path),
            audio_data["array"],
            audio_data["sampling_rate"]
        )

        # Create metadata record
        metadata = {
            "id": sample_id,
            "text": sample["text_normalized"],
            "text_original": sample.get("text_original", ""),
            "speaker_id": sample["speaker_id"],
            "chapter_id": sample.get("chapter_id", ""),
            "audio_path": str(audio_path),
            "sampling_rate": audio_data["sampling_rate"],
            # Note: blabble-io/libritts_r doesn't have prosody annotations
            # We need to use parler-tts version for those
        }
        samples.append(metadata)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nDownloaded {len(samples)} samples to {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return samples


def download_libritts_r_with_annotations(
    output_dir: Path,
    num_samples: int = 100,
) -> List[Dict[str, Any]]:
    """
    Download LibriTTS-R with prosody annotations from parler-tts dataset.

    This combines:
    1. Audio from blabble-io/libritts_r
    2. Annotations from parler-tts/libritts_r_tags_tagged_10k_generated
    """
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install datasets soundfile")
        return []

    print(f"Downloading {num_samples} annotated samples from LibriTTS-R...")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Load annotations dataset (smaller, no audio)
    print("Loading annotation dataset...")
    annotations_ds = load_dataset(
        "parler-tts/libritts_r_tags_tagged_10k_generated",
        "clean",
        split="train.clean.100",
    )

    # Load audio dataset with streaming
    print("Loading audio dataset (streaming)...")
    audio_ds = load_dataset(
        "blabble-io/libritts_r",
        "clean",
        split="train.clean.100",
        streaming=True
    )

    # Create ID lookup from annotations
    print("Building annotation index...")
    annotation_lookup = {}
    for ann in annotations_ds:
        annotation_lookup[ann["id"]] = ann

    # Download audio and match with annotations
    samples = []
    downloaded = 0

    for sample in audio_ds:
        if downloaded >= num_samples:
            break

        sample_id = sample["id"]

        # Check if we have annotations for this sample
        if sample_id not in annotation_lookup:
            continue

        ann = annotation_lookup[sample_id]
        downloaded += 1

        print(f"  [{downloaded}/{num_samples}] {sample_id}")

        # Save audio file
        audio_data = sample["audio"]
        audio_path = audio_dir / f"{sample_id}.wav"

        sf.write(
            str(audio_path),
            audio_data["array"],
            audio_data["sampling_rate"]
        )

        # Create metadata record with full annotations
        metadata = {
            "id": sample_id,
            "text": ann["text"],
            "text_original": ann.get("text_original", ""),
            "speaker_id": ann["speaker_id"],
            "chapter_id": ann.get("chapter_id", ""),
            "audio_path": str(audio_path),
            "sampling_rate": audio_data["sampling_rate"],
            # Prosody annotations
            "speaking_rate": ann["speaking_rate"],
            "gender": ann["gender"],
            "pitch": ann["pitch"],  # categorical
            "utterance_pitch_mean": ann["utterance_pitch_mean"],  # Hz
            "utterance_pitch_std": ann["utterance_pitch_std"],  # Hz
            "speech_monotony": ann["speech_monotony"],
            "snr": ann["snr"],
            "c50": ann["c50"],
            "noise": ann["noise"],
            "reverberation": ann["reverberation"],
            "text_description": ann["text_description"],
        }
        samples.append(metadata)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nDownloaded {len(samples)} annotated samples to {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return samples


def download_ravdess(
    output_dir: Path,
    download_all: bool = False,
    num_samples: int = 100,
) -> List[Dict[str, Any]]:
    """
    Download RAVDESS emotional speech dataset.

    The dataset is available on HuggingFace or Zenodo.
    Filenames encode: modality-vocal_channel-emotion-intensity-statement-repetition-actor
    """
    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install datasets soundfile")
        return []

    print(f"Downloading RAVDESS emotional speech dataset...")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # RAVDESS emotion mapping
    EMOTIONS = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    INTENSITIES = {
        "01": "normal",
        "02": "strong"
    }

    # Load from HuggingFace
    print("Loading RAVDESS from HuggingFace...")
    dataset = load_dataset("narad/ravdess", split="train")

    samples = []
    limit = len(dataset) if download_all else min(num_samples, len(dataset))

    for i in range(limit):
        sample = dataset[i]

        # Parse filename to get annotations
        # Format: 03-01-05-01-02-01-12.wav
        # modality-vocal_channel-emotion-intensity-statement-repetition-actor
        filename = Path(sample["file"]).stem
        parts = filename.split("-")

        if len(parts) >= 7:
            emotion_code = parts[2]
            intensity_code = parts[3]
            actor_id = parts[6]

            emotion = EMOTIONS.get(emotion_code, "unknown")
            intensity = INTENSITIES.get(intensity_code, "unknown")
            gender = "female" if int(actor_id) % 2 == 0 else "male"
        else:
            emotion = sample.get("label", "unknown")
            intensity = "unknown"
            gender = "unknown"
            actor_id = "unknown"

        sample_id = f"ravdess_{filename}"
        print(f"  [{i+1}/{limit}] {sample_id} - {emotion} ({intensity})")

        # Save audio
        audio_data = sample["audio"]
        audio_path = audio_dir / f"{sample_id}.wav"

        sf.write(
            str(audio_path),
            audio_data["array"],
            audio_data["sampling_rate"]
        )

        metadata = {
            "id": sample_id,
            "audio_path": str(audio_path),
            "sampling_rate": audio_data["sampling_rate"],
            "emotion": emotion,
            "intensity": intensity,
            "gender": gender,
            "actor_id": actor_id,
            "original_file": sample["file"],
            # RAVDESS doesn't have pitch annotations - we'll compute them
            "text": "Kids are talking by the door" if "01" in filename else "Dogs are sitting by the door",
        }
        samples.append(metadata)

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nDownloaded {len(samples)} RAVDESS samples to {output_dir}")
    return samples


def check_local_annotations() -> Dict[str, Any]:
    """Check what annotations we already have locally."""
    local_annotations_path = DATA_DIR / "libritts_annotated" / "train_clean_100" / "data-00000-of-00001.arrow"

    if not local_annotations_path.exists():
        return {"exists": False, "count": 0}

    try:
        import pyarrow.ipc as ipc
        with open(local_annotations_path, "rb") as f:
            reader = ipc.open_stream(f)
            table = reader.read_all()

        return {
            "exists": True,
            "count": table.num_rows,
            "columns": table.column_names,
            "path": str(local_annotations_path),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Download audio datasets for prosody analysis testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python download_audio_datasets.py --list

  # Download 100 LibriTTS-R samples with annotations (RECOMMENDED)
  python download_audio_datasets.py --dataset libritts_r --samples 100

  # Download full RAVDESS dataset
  python download_audio_datasets.py --dataset ravdess --all

  # Check what annotations we already have
  python download_audio_datasets.py --check-local
"""
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Dataset to download"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Number of samples to download (default: 100)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download entire dataset (where applicable)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--check-local",
        action="store_true",
        help="Check locally available annotations"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.check_local:
        print("Checking local annotations...")
        info = check_local_annotations()
        print(json.dumps(info, indent=2))
        return

    if not args.dataset:
        parser.print_help()
        return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = AUDIO_CACHE_DIR / args.dataset

    # Download based on dataset type
    if args.dataset == "libritts_r":
        download_libritts_r_with_annotations(
            output_dir,
            num_samples=args.samples
        )
    elif args.dataset == "ravdess":
        download_ravdess(
            output_dir,
            download_all=args.all,
            num_samples=args.samples
        )
    else:
        print(f"Download for {args.dataset} not yet implemented.")
        print("Please use --list to see available options.")


if __name__ == "__main__":
    main()
