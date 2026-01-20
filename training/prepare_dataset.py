"""
Voice Clone Pipeline - Dataset Preparation
Converts labeled audio samples into training format for CSM fine-tuning.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
import torchaudio
from tqdm import tqdm


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load metadata.json from labeled directory."""
    with open(metadata_path) as f:
        return json.load(f)


def prepare_sample(sample: Dict[str, Any], output_dir: Path, tokenizer=None, mimi=None) -> Dict[str, Any]:
    """
    Prepare a single sample for training.
    
    If tokenizers are provided, tokenizes the audio and text.
    Otherwise, just organizes the data.
    """
    audio_path = Path(sample["audio_path"])
    
    # Load and validate audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        waveform = resampler(waveform)
    
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Build training item
    training_item = {
        "id": sample["id"],
        "text": sample["text"],
        "speaker": sample.get("speaker", 0),
        "duration": sample.get("duration", waveform.shape[1] / 24000),
        "audio_path": str(audio_path),
    }
    
    # Add prosody if available
    if "prosody" in sample and sample["prosody"]:
        training_item["prosody"] = sample["prosody"]
    
    # Tokenize if tokenizers provided
    if tokenizer is not None:
        speaker_id = sample.get("speaker", 0)
        text_with_speaker = f"[{speaker_id}]{sample['text']}"
        text_tokens = tokenizer.encode(text_with_speaker)
        training_item["text_tokens"] = text_tokens
    
    if mimi is not None:
        # Tokenize audio with Mimi
        with torch.no_grad():
            audio_tokens = mimi.encode(waveform.unsqueeze(0))
        training_item["audio_tokens"] = audio_tokens.squeeze(0).tolist()
    
    # Save waveform as tensor
    output_audio = output_dir / "audio" / f"{sample['id']}.pt"
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    torch.save(waveform, output_audio)
    training_item["audio_tensor_path"] = str(output_audio)
    
    return training_item


def split_dataset(samples: List[Dict], val_split: float = 0.1, test_split: float = 0.05):
    """Split samples into train/val/test sets."""
    random.shuffle(samples)
    
    n = len(samples)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    
    return train_samples, val_samples, test_samples


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    val_split: float = 0.1,
    test_split: float = 0.05,
    use_tokenizers: bool = False,
):
    """
    Prepare full dataset for training.
    
    Args:
        input_dir: Directory containing metadata.json and audio files
        output_dir: Output directory for prepared dataset
        val_split: Fraction for validation set
        test_split: Fraction for test set
        use_tokenizers: Whether to tokenize audio/text (requires CSM models)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {input_dir}")
    
    samples = load_metadata(metadata_path)
    print(f"Loaded {len(samples)} samples")
    
    # Filter valid samples
    valid_samples = []
    for s in samples:
        audio_path = Path(s["audio_path"])
        if audio_path.exists():
            valid_samples.append(s)
        else:
            print(f"  Warning: Audio not found: {audio_path}")
    
    print(f"Valid samples: {len(valid_samples)}")
    
    # Load tokenizers if requested
    tokenizer = None
    mimi = None
    
    if use_tokenizers:
        print("Loading tokenizers...")
        # This would load the actual CSM tokenizers
        # For now, we'll skip and just organize data
        pass
    
    # Process samples
    print("Processing samples...")
    processed = []
    for sample in tqdm(valid_samples):
        try:
            item = prepare_sample(sample, output_dir, tokenizer, mimi)
            processed.append(item)
        except Exception as e:
            print(f"  Error processing {sample['id']}: {e}")
    
    print(f"Processed {len(processed)} samples")
    
    # Split dataset
    train, val, test = split_dataset(processed, val_split, test_split)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Save splits
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_path = output_dir / f"{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {split_path}")
    
    # Calculate and save statistics
    total_duration = sum(s["duration"] for s in processed)
    stats = {
        "total_samples": len(processed),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "total_duration_minutes": round(total_duration / 60, 2),
        "avg_duration_seconds": round(total_duration / len(processed), 2) if processed else 0,
    }
    
    # Prosody distribution
    if any("prosody" in s for s in processed):
        emotions = {}
        for s in processed:
            if s.get("prosody", {}).get("semantic", {}).get("emotion"):
                emotion = s["prosody"]["semantic"]["emotion"]
                emotions[emotion] = emotions.get(emotion, 0) + 1
        stats["emotion_distribution"] = emotions
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {stats_path}")
    
    print("\nDataset preparation complete!")
    print(f"  Total duration: {stats['total_duration_minutes']} minutes")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for CSM training")
    parser.add_argument("--input", "-i", required=True, help="Input directory with metadata.json")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test_split", type=float, default=0.05, help="Test split fraction")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize audio/text (requires CSM)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        val_split=args.val_split,
        test_split=args.test_split,
        use_tokenizers=args.tokenize,
    )


if __name__ == "__main__":
    main()
