#!/usr/bin/env python3
"""
Prepare unified prosody training data from multiple sources:
1. LibriTTS-R (prosody-annotated read speech)
2. RAVDESS (emotional speech)
3. User's own labeled samples

Outputs training-ready data for prosody encoder.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm


# RAVDESS emotion mapping (code -> emotion name)
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "surprised",  # disgust in original, mapping to surprised
    "08": "surprised",
}

RAVDESS_INTENSITY = {
    "01": 0.5,  # normal
    "02": 0.8,  # strong
}


def load_ravdess(ravdess_dir: Path, output_dir: Path, start_idx: int = 0) -> list[dict]:
    """Load and process RAVDESS dataset."""
    records = []
    idx = start_idx

    if not ravdess_dir.exists():
        print(f"RAVDESS directory not found: {ravdess_dir}")
        return records

    # RAVDESS filename format: 03-01-06-01-02-01-12.wav
    # Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor
    for wav_file in tqdm(list(ravdess_dir.rglob("*.wav")), desc="Processing RAVDESS"):
        try:
            parts = wav_file.stem.split("-")
            if len(parts) != 7:
                continue

            emotion_code = parts[2]
            intensity_code = parts[3]
            actor_id = parts[6]

            emotion = RAVDESS_EMOTIONS.get(emotion_code, "neutral")
            intensity = RAVDESS_INTENSITY.get(intensity_code, 0.5)

            # Load audio
            audio, sr = librosa.load(str(wav_file), sr=24000)
            duration = len(audio) / sr

            if duration < 1.0 or duration > 10.0:
                continue

            # Analyze prosody
            pitch_contour = extract_pitch_contour(audio, sr)
            speaking_rate = estimate_speaking_rate(audio, sr)

            # Save audio copy
            audio_path = output_dir / "audio" / f"ravdess_{idx:06d}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(audio_path), audio, sr)

            prosody = {
                "semantic": {
                    "emotion": emotion,
                    "intensity": intensity,
                    "tone": "emotional",
                },
                "acoustic": {
                    "pitch_mean": float(np.mean(pitch_contour[pitch_contour > 0])) if np.any(pitch_contour > 0) else 0.5,
                    "pitch_std": float(np.std(pitch_contour[pitch_contour > 0])) if np.any(pitch_contour > 0) else 0.1,
                    "energy": float(np.mean(np.abs(audio))),
                    "speaking_rate": speaking_rate,
                },
                "rhythm": {
                    "pause_ratio": estimate_pause_ratio(audio, sr),
                    "syllable_rate": speaking_rate * 3,  # Approximate
                },
                "contour": {
                    "pitch_contour": resample_contour(pitch_contour, 64),
                },
            }

            record = {
                "id": f"ravdess_{idx:06d}",
                "text": "",  # RAVDESS doesn't have transcripts, will need ASR
                "audio_path": str(audio_path.relative_to(output_dir)),
                "duration": duration,
                "prosody": prosody,
                "source": "ravdess",
                "speaker_id": f"ravdess_actor_{actor_id}",
            }

            records.append(record)
            idx += 1

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

    return records


def extract_pitch_contour(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Extract pitch contour using librosa."""
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)
        return np.array(pitch_contour)
    except:
        return np.zeros(100)


def estimate_speaking_rate(audio: np.ndarray, sr: int) -> float:
    """Estimate speaking rate from audio."""
    try:
        # Use onset detection as proxy for syllable rate
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration = len(audio) / sr
        rate = len(onsets) / duration if duration > 0 else 3.0
        return min(max(rate / 10.0, 0.1), 1.0)  # Normalize to 0-1
    except:
        return 0.5


def estimate_pause_ratio(audio: np.ndarray, sr: int, threshold: float = 0.01) -> float:
    """Estimate ratio of silence/pauses in audio."""
    try:
        rms = librosa.feature.rms(y=audio)[0]
        silence_frames = np.sum(rms < threshold)
        return silence_frames / len(rms)
    except:
        return 0.2


def resample_contour(contour: np.ndarray, target_length: int) -> list[float]:
    """Resample contour to fixed length and normalize."""
    if len(contour) == 0:
        return [0.5] * target_length

    # Remove zeros and normalize
    contour = contour[contour > 0] if np.any(contour > 0) else contour

    if len(contour) == 0:
        return [0.5] * target_length

    # Normalize to 0-1
    contour_min = contour.min()
    contour_max = contour.max()
    if contour_max > contour_min:
        contour = (contour - contour_min) / (contour_max - contour_min)
    else:
        contour = np.ones_like(contour) * 0.5

    # Resample
    indices = np.linspace(0, len(contour) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(contour)), contour)

    return resampled.tolist()


def load_user_samples(labeled_dir: Path, output_dir: Path, start_idx: int = 0) -> list[dict]:
    """Load user's own labeled samples."""
    records = []
    idx = start_idx

    if not labeled_dir.exists():
        print(f"Labeled directory not found: {labeled_dir}")
        return records

    for json_file in tqdm(list(labeled_dir.glob("*.json")), desc="Processing user samples"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            wav_file = json_file.with_suffix(".wav")
            if not wav_file.exists():
                continue

            # Copy audio
            audio, sr = librosa.load(str(wav_file), sr=24000)
            audio_path = output_dir / "audio" / f"user_{idx:06d}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(audio_path), audio, sr)

            record = {
                "id": f"user_{idx:06d}",
                "text": data.get("transcript", ""),
                "audio_path": str(audio_path.relative_to(output_dir)),
                "duration": len(audio) / sr,
                "prosody": data.get("prosody", {}),
                "source": "user",
                "speaker_id": "user",
            }

            records.append(record)
            idx += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare unified prosody training data")
    parser.add_argument("--output", type=str, default="data/prosody_training", help="Output directory")
    parser.add_argument("--ravdess-dir", type=str, default="data/ravdess", help="RAVDESS dataset directory")
    parser.add_argument("--labeled-dir", type=str, default="data/labeled", help="User labeled samples directory")
    parser.add_argument("--include-libritts", action="store_true", help="Include LibriTTS-R (large download)")
    parser.add_argument("--libritts-samples", type=int, default=5000, help="Number of LibriTTS samples")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    idx = 0

    # 1. Load RAVDESS (emotional speech)
    print("\n=== Loading RAVDESS ===")
    ravdess_records = load_ravdess(Path(args.ravdess_dir), output_dir, idx)
    all_records.extend(ravdess_records)
    idx += len(ravdess_records)
    print(f"Loaded {len(ravdess_records)} RAVDESS samples")

    # 2. Load user samples
    print("\n=== Loading User Samples ===")
    user_records = load_user_samples(Path(args.labeled_dir), output_dir, idx)
    all_records.extend(user_records)
    idx += len(user_records)
    print(f"Loaded {len(user_records)} user samples")

    # 3. Optionally load LibriTTS-R
    if args.include_libritts:
        print("\n=== Loading LibriTTS-R ===")
        print("This will download data from HuggingFace (may take a while)...")

        try:
            dataset = load_dataset(
                "parler-tts/libritts-r-filtered-speaker-descriptions",
                "clean",
                split="train.clean.100",
                trust_remote_code=True,
            )

            for i, sample in enumerate(tqdm(dataset, desc="Processing LibriTTS")):
                if i >= args.libritts_samples:
                    break

                try:
                    audio = sample["audio"]["array"]
                    sr = sample["audio"]["sampling_rate"]
                    duration = len(audio) / sr

                    if duration < 1.0 or duration > 15.0:
                        continue

                    # Save audio
                    audio_path = output_dir / "audio" / f"libritts_{idx:06d}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(audio_path), audio, sr)

                    # Extract prosody from annotations
                    monotony = sample.get("speech_monotony", "moderate intonation")
                    emotion = "neutral"
                    intensity = 0.5
                    if "expressive" in str(monotony):
                        emotion = "happy"
                        intensity = 0.6

                    prosody = {
                        "semantic": {
                            "emotion": emotion,
                            "intensity": intensity,
                            "tone": "narrative",
                        },
                        "acoustic": {
                            "pitch_mean": sample.get("utterance_pitch_mean", 0.5),
                            "pitch_std": sample.get("utterance_pitch_std", 0.1),
                            "energy": 0.5,
                            "speaking_rate": sample.get("speaking_rate", 0.5),
                        },
                        "rhythm": {
                            "pause_ratio": 0.2,
                            "syllable_rate": 4.0,
                        },
                        "contour": {
                            "pitch_contour": [0.5] * 64,
                        },
                    }

                    record = {
                        "id": f"libritts_{idx:06d}",
                        "text": sample.get("text_normalized", ""),
                        "audio_path": str(audio_path.relative_to(output_dir)),
                        "duration": duration,
                        "prosody": prosody,
                        "source": "libritts",
                        "speaker_id": str(sample.get("speaker_id", "unknown")),
                    }

                    all_records.append(record)
                    idx += 1

                except Exception as e:
                    continue

            print(f"Loaded {idx - len(ravdess_records) - len(user_records)} LibriTTS samples")

        except Exception as e:
            print(f"Error loading LibriTTS: {e}")

    # Save unified manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"PROSODY TRAINING DATA PREPARED")
    print(f"{'='*50}")
    print(f"Total samples: {len(all_records)}")

    total_duration = sum(r["duration"] for r in all_records)
    print(f"Total duration: {total_duration / 3600:.2f} hours")

    sources = {}
    for r in all_records:
        src = r["source"]
        sources[src] = sources.get(src, 0) + 1
    print(f"By source: {sources}")

    emotions = {}
    for r in all_records:
        e = r["prosody"].get("semantic", {}).get("emotion", "unknown")
        emotions[e] = emotions.get(e, 0) + 1
    print(f"Emotion distribution: {emotions}")

    print(f"\nManifest: {manifest_path}")
    print(f"Audio dir: {output_dir / 'audio'}")


if __name__ == "__main__":
    main()
