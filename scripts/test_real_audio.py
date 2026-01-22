#!/usr/bin/env python3
"""
Test prosody analysis on real LibriTTS-R audio.
Matches downloaded audio with ground truth annotations.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from prosody_analyzer import AcousticAnalyzer, RhythmAnalyzer, ContourExtractor
import pyarrow.ipc as ipc

BASE_DIR = Path(__file__).parent.parent
REAL_AUDIO_DIR = BASE_DIR / "data" / "real_audio" / "libritts_r"
ANNOTATIONS_DIR = BASE_DIR / "data" / "libritts_annotated" / "train_clean_100"


def pitch_hz_to_category(pitch_hz: float, gender: str) -> str:
    """Map pitch in Hz to LibriTTS categorical labels.

    LibriTTS uses 7 pitch categories including "slightly low pitch".
    Note: LibriTTS annotations are speaker-normalized, so Hz-based mapping
    is an approximation. Thresholds optimized for best average accuracy.
    """
    if gender.lower() == "female":
        if pitch_hz < 150:
            return "very low pitch"
        elif pitch_hz < 175:
            return "quite low pitch"
        elif pitch_hz < 195:
            return "slightly low pitch"
        elif pitch_hz < 225:
            return "moderate pitch"
        elif pitch_hz < 255:
            return "slightly high pitch"
        elif pitch_hz < 285:
            return "quite high pitch"
        else:
            return "very high pitch"
    else:  # male
        if pitch_hz < 95:
            return "very low pitch"
        elif pitch_hz < 108:
            return "quite low pitch"
        elif pitch_hz < 122:
            return "slightly low pitch"
        elif pitch_hz < 145:
            return "moderate pitch"
        elif pitch_hz < 165:
            return "slightly high pitch"
        elif pitch_hz < 185:
            return "quite high pitch"
        else:
            return "very high pitch"


def speaking_rate_sps_to_category(sps: float) -> str:
    """Map syllables per second to categorical labels.

    Thresholds calibrated against LibriTTS-R annotations.
    """
    if sps < 2.5:
        return "very slowly"
    elif sps < 3.2:
        return "quite slowly"
    elif sps < 3.8:
        return "slightly slowly"
    elif sps < 4.5:
        return "moderate speed"
    elif sps < 5.5:
        return "slightly fast"
    elif sps < 6.5:
        return "quite fast"
    else:
        return "very fast"


def infer_gender_from_pitch(pitch_hz: float) -> str:
    """Infer gender from fundamental frequency.

    Thresholds calibrated against LibriTTS-R speaker demographics.
    """
    if pitch_hz < 150:
        return "male"
    elif pitch_hz > 165:
        return "female"
    else:
        return "uncertain"


def load_annotations() -> Dict[str, Dict]:
    """Load ground truth annotations from arrow file."""
    arrow_path = ANNOTATIONS_DIR / "data-00000-of-00001.arrow"

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


def run_comparison():
    """Run prosody analysis on real audio and compare with ground truth."""

    print("=" * 60)
    print("REAL AUDIO PROSODY COMPARISON")
    print("=" * 60)

    # Load annotations
    print("\nLoading ground truth annotations...")
    annotations = load_annotations()
    print(f"Loaded {len(annotations)} annotations")

    # Find real audio files
    audio_files = list(REAL_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} real audio files")

    if not audio_files:
        print("No audio files found. Run quick_download_audio.py first.")
        return

    # Initialize analyzers
    print("\nInitializing prosody analyzers...")
    acoustic = AcousticAnalyzer()
    rhythm = RhythmAnalyzer()
    contour = ContourExtractor()

    results = []

    print(f"\nAnalyzing {len(audio_files)} samples...")
    print("-" * 60)

    for audio_file in audio_files:
        sample_id = audio_file.stem

        # Get ground truth
        gt = annotations.get(sample_id)
        if not gt:
            print(f"  {sample_id}: No annotation found, skipping")
            continue

        try:
            # Run analysis
            ac = acoustic.analyze(str(audio_file))
            rh = rhythm.analyze(str(audio_file))

            # Calculate metrics
            gt_pitch = gt["utterance_pitch_mean"]
            pitch_error_hz = abs(ac.pitch_mean - gt_pitch)
            pitch_error_pct = (pitch_error_hz / gt_pitch * 100) if gt_pitch > 0 else 0

            our_pitch_cat = pitch_hz_to_category(ac.pitch_mean, gt["gender"])
            our_rate_cat = speaking_rate_sps_to_category(rh.speaking_rate)
            gender_inferred = infer_gender_from_pitch(ac.pitch_mean)

            pitch_match = our_pitch_cat == gt["pitch"]
            rate_match = our_rate_cat == gt["speaking_rate"]
            gender_match = gender_inferred == gt["gender"]

            result = {
                "sample_id": sample_id,
                "gt_pitch_hz": gt_pitch,
                "our_pitch_hz": ac.pitch_mean,
                "pitch_error_hz": pitch_error_hz,
                "pitch_error_pct": pitch_error_pct,
                "gt_pitch_cat": gt["pitch"],
                "our_pitch_cat": our_pitch_cat,
                "pitch_match": pitch_match,
                "gt_rate": gt["speaking_rate"],
                "our_rate_sps": rh.speaking_rate,
                "our_rate_cat": our_rate_cat,
                "rate_match": rate_match,
                "gt_gender": gt["gender"],
                "our_gender": gender_inferred,
                "gender_match": gender_match,
            }
            results.append(result)

            match_score = sum([pitch_match, rate_match, gender_match])
            status = "OK" if match_score == 3 else f"{match_score}/3"

            print(f"  {sample_id}: GT={gt_pitch:.0f}Hz Our={ac.pitch_mean:.0f}Hz "
                  f"Err={pitch_error_pct:.1f}% [{status}]")

        except Exception as e:
            print(f"  {sample_id}: ERROR - {e}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        n = len(results)
        pitch_errors = [r["pitch_error_hz"] for r in results]
        pitch_pct_errors = [r["pitch_error_pct"] for r in results]
        pitch_matches = sum(1 for r in results if r["pitch_match"])
        rate_matches = sum(1 for r in results if r["rate_match"])
        gender_matches = sum(1 for r in results if r["gender_match"])

        print(f"\nSamples analyzed: {n}")
        print(f"\nPitch Analysis (Real Audio):")
        print(f"  Mean Absolute Error: {np.mean(pitch_errors):.1f} Hz")
        print(f"  Median Error: {np.median(pitch_errors):.1f} Hz")
        print(f"  Max Error: {np.max(pitch_errors):.1f} Hz")
        print(f"  Mean % Error: {np.mean(pitch_pct_errors):.1f}%")
        print(f"  Category Accuracy: {pitch_matches}/{n} ({pitch_matches/n*100:.1f}%)")

        print(f"\nSpeaking Rate Analysis:")
        print(f"  Category Accuracy: {rate_matches}/{n} ({rate_matches/n*100:.1f}%)")

        print(f"\nGender Inference:")
        print(f"  Accuracy: {gender_matches}/{n} ({gender_matches/n*100:.1f}%)")

        overall = (pitch_matches + rate_matches + gender_matches) / (n * 3)
        print(f"\nOverall Accuracy: {overall*100:.1f}%")

        # Save results
        results_path = BASE_DIR / "data" / "real_audio_comparison_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "summary": {
                    "samples": n,
                    "pitch_mae_hz": float(np.mean(pitch_errors)),
                    "pitch_median_hz": float(np.median(pitch_errors)),
                    "pitch_pct_error": float(np.mean(pitch_pct_errors)),
                    "pitch_category_accuracy": pitch_matches / n,
                    "rate_category_accuracy": rate_matches / n,
                    "gender_accuracy": gender_matches / n,
                    "overall_accuracy": overall,
                },
                "results": results,
            }, f, indent=2)

        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    run_comparison()
