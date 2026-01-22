#!/usr/bin/env python3
"""
Prosody Comparison with Real Audio

Compares the Voice Clone Pipeline's prosody analysis against real audio
from LibriTTS-R with ground truth annotations.

This script:
1. Loads real audio from downloaded LibriTTS-R samples
2. Runs our prosody analyzer on each sample
3. Compares results with ground truth annotations
4. Generates detailed comparison reports

Usage:
    # First, download real audio:
    python download_audio_datasets.py --dataset libritts_r --samples 100

    # Then run comparison:
    python compare_prosody_real_audio.py --samples 50

    # Include Qwen2-Audio semantic analysis:
    python compare_prosody_real_audio.py --use-qwen --samples 20
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from prosody_analyzer import (
    AcousticAnalyzer,
    RhythmAnalyzer,
    ContourExtractor,
    CompleteProsodyAnalyzer,
)


# ============== Configuration ==============

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio_datasets" / "libritts_r"
RESULTS_DIR = DATA_DIR / "prosody_comparison_results"


# ============== Data Classes ==============

@dataclass
class RealAudioSample:
    """A sample with real audio and ground truth annotations."""
    id: str
    text: str
    audio_path: str
    speaker_id: str

    # Ground truth prosody annotations
    speaking_rate_category: str  # categorical
    gender: str
    pitch_category: str  # categorical
    utterance_pitch_mean: float  # Hz
    utterance_pitch_std: float  # Hz
    speech_monotony: str

    # Quality metrics
    snr: float
    c50: float
    noise: str
    reverberation: str

    # Natural language description
    text_description: str


@dataclass
class RealAudioComparisonResult:
    """Comparison between ground truth and our analysis on real audio."""
    sample_id: str
    text: str
    audio_path: str

    # Ground truth (LibriTTS annotations)
    gt_pitch_category: str
    gt_pitch_mean_hz: float
    gt_pitch_std_hz: float
    gt_speaking_rate: str
    gt_monotony: str
    gt_gender: str
    gt_description: str

    # Our acoustic analysis
    our_pitch_mean_hz: float
    our_pitch_std_hz: float
    our_pitch_range_hz: float
    our_jitter: float
    our_shimmer: float
    our_hnr: float
    our_formants: Tuple[float, float, float]

    # Our rhythm analysis
    our_duration_sec: float
    our_speaking_rate_sps: float
    our_articulation_rate: float
    our_pause_count: int
    our_syllable_count: int

    # Our semantic analysis (if available)
    our_emotion: Optional[str]
    our_tone: Optional[str]
    our_energy_level: Optional[str]
    our_pace_category: Optional[str]

    # Computed metrics
    pitch_mean_error_hz: float
    pitch_mean_error_pct: float
    pitch_std_error_hz: float
    pitch_category_match: bool
    speaking_rate_category_match: bool
    gender_inferred: str
    gender_match: bool


# ============== Mapping Functions ==============

def pitch_hz_to_category(pitch_hz: float, gender: str) -> str:
    """Map pitch in Hz to LibriTTS categorical labels."""
    if gender.lower() == "female":
        if pitch_hz < 165:
            return "very low pitch"
        elif pitch_hz < 190:
            return "quite low pitch"
        elif pitch_hz < 220:
            return "moderate pitch"
        elif pitch_hz < 250:
            return "slightly high pitch"
        elif pitch_hz < 280:
            return "quite high pitch"
        else:
            return "very high pitch"
    else:  # male
        if pitch_hz < 95:
            return "very low pitch"
        elif pitch_hz < 110:
            return "quite low pitch"
        elif pitch_hz < 130:
            return "moderate pitch"
        elif pitch_hz < 150:
            return "slightly high pitch"
        elif pitch_hz < 170:
            return "quite high pitch"
        else:
            return "very high pitch"


def speaking_rate_sps_to_category(sps: float) -> str:
    """Map syllables per second to LibriTTS categorical labels."""
    if sps < 2.5:
        return "very slowly"
    elif sps < 3.5:
        return "quite slowly"
    elif sps < 4.0:
        return "slightly slowly"
    elif sps < 5.0:
        return "moderate speed"
    elif sps < 6.0:
        return "slightly fast"
    elif sps < 7.0:
        return "quite fast"
    else:
        return "very fast"


def infer_gender_from_pitch(pitch_hz: float) -> str:
    """Infer gender from fundamental frequency."""
    if pitch_hz < 155:
        return "male"
    elif pitch_hz > 175:
        return "female"
    else:
        return "uncertain"


def pitch_std_to_monotony(pitch_std: float, pitch_mean: float) -> str:
    """Map pitch standard deviation to monotony category."""
    if pitch_mean == 0:
        return "quite monotone"

    cv = pitch_std / pitch_mean

    if cv < 0.10:
        return "very monotone"
    elif cv < 0.15:
        return "quite monotone"
    elif cv < 0.20:
        return "slightly monotone"
    elif cv < 0.25:
        return "slightly expressive"
    elif cv < 0.35:
        return "quite expressive"
    else:
        return "very expressive"


# ============== Data Loading ==============

def load_real_audio_samples(
    audio_dir: Path,
    limit: int = 50
) -> List[RealAudioSample]:
    """Load samples from downloaded LibriTTS-R audio."""
    metadata_path = audio_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"Error: No metadata found at {metadata_path}")
        print("Please run download_audio_datasets.py first:")
        print("  python scripts/download_audio_datasets.py --dataset libritts_r --samples 100")
        return []

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    samples = []
    for item in metadata[:limit]:
        # Check if audio file exists
        audio_path = Path(item["audio_path"])
        if not audio_path.exists():
            print(f"  Warning: Audio not found: {audio_path}")
            continue

        sample = RealAudioSample(
            id=item["id"],
            text=item["text"],
            audio_path=str(audio_path),
            speaker_id=item["speaker_id"],
            speaking_rate_category=item.get("speaking_rate", "unknown"),
            gender=item.get("gender", "unknown"),
            pitch_category=item.get("pitch", "unknown"),
            utterance_pitch_mean=float(item.get("utterance_pitch_mean", 0)),
            utterance_pitch_std=float(item.get("utterance_pitch_std", 0)),
            speech_monotony=item.get("speech_monotony", "unknown"),
            snr=float(item.get("snr", 0)),
            c50=float(item.get("c50", 0)),
            noise=item.get("noise", "unknown"),
            reverberation=item.get("reverberation", "unknown"),
            text_description=item.get("text_description", ""),
        )
        samples.append(sample)

    return samples


# ============== Analysis ==============

def analyze_real_sample(
    sample: RealAudioSample,
    acoustic_analyzer: AcousticAnalyzer,
    rhythm_analyzer: RhythmAnalyzer,
    semantic_analyzer=None,
) -> RealAudioComparisonResult:
    """Run prosody analysis on a real audio sample and compare with ground truth."""

    # Run acoustic analysis
    acoustic = acoustic_analyzer.analyze(sample.audio_path)

    # Run rhythm analysis
    rhythm = rhythm_analyzer.analyze(sample.audio_path)

    # Optional: semantic analysis
    semantic = None
    if semantic_analyzer:
        try:
            semantic = semantic_analyzer.analyze(sample.audio_path, sample.text)
        except Exception as e:
            print(f"  Semantic analysis failed: {e}")

    # Compute errors
    gt_pitch = sample.utterance_pitch_mean if sample.utterance_pitch_mean > 0 else 150.0
    pitch_mean_error = abs(acoustic.pitch_mean - gt_pitch)
    pitch_mean_error_pct = (pitch_mean_error / gt_pitch * 100) if gt_pitch > 0 else 0
    pitch_std_error = abs(acoustic.pitch_std - sample.utterance_pitch_std)

    # Check category matches
    our_pitch_category = pitch_hz_to_category(acoustic.pitch_mean, sample.gender)
    pitch_category_match = our_pitch_category.lower() == sample.pitch_category.lower()

    our_speaking_rate_category = speaking_rate_sps_to_category(rhythm.speaking_rate)
    # Normalize for comparison
    gt_rate_normalized = sample.speaking_rate_category.lower().replace("_", " ")
    our_rate_normalized = our_speaking_rate_category.lower()
    speaking_rate_match = gt_rate_normalized == our_rate_normalized

    # Infer gender from pitch
    gender_inferred = infer_gender_from_pitch(acoustic.pitch_mean)
    gender_match = gender_inferred.lower() == sample.gender.lower()

    return RealAudioComparisonResult(
        sample_id=sample.id,
        text=sample.text[:60] + "..." if len(sample.text) > 60 else sample.text,
        audio_path=sample.audio_path,

        # Ground truth
        gt_pitch_category=sample.pitch_category,
        gt_pitch_mean_hz=gt_pitch,
        gt_pitch_std_hz=sample.utterance_pitch_std,
        gt_speaking_rate=sample.speaking_rate_category,
        gt_monotony=sample.speech_monotony,
        gt_gender=sample.gender,
        gt_description=sample.text_description[:80] + "..." if len(sample.text_description) > 80 else sample.text_description,

        # Our acoustic analysis
        our_pitch_mean_hz=acoustic.pitch_mean,
        our_pitch_std_hz=acoustic.pitch_std,
        our_pitch_range_hz=acoustic.pitch_range,
        our_jitter=acoustic.jitter,
        our_shimmer=acoustic.shimmer,
        our_hnr=acoustic.hnr,
        our_formants=(acoustic.f1_mean, acoustic.f2_mean, acoustic.f3_mean),

        # Our rhythm analysis
        our_duration_sec=rhythm.duration_seconds,
        our_speaking_rate_sps=rhythm.speaking_rate,
        our_articulation_rate=rhythm.articulation_rate,
        our_pause_count=rhythm.pause_count,
        our_syllable_count=rhythm.syllable_count,

        # Our semantic analysis
        our_emotion=semantic.emotion if semantic else None,
        our_tone=semantic.tone if semantic else None,
        our_energy_level=semantic.energy_level if semantic else None,
        our_pace_category=semantic.pace_category if semantic else None,

        # Metrics
        pitch_mean_error_hz=pitch_mean_error,
        pitch_mean_error_pct=pitch_mean_error_pct,
        pitch_std_error_hz=pitch_std_error,
        pitch_category_match=pitch_category_match,
        speaking_rate_category_match=speaking_rate_match,
        gender_inferred=gender_inferred,
        gender_match=gender_match,
    )


# ============== Reporting ==============

def generate_report(results: List[RealAudioComparisonResult]) -> str:
    """Generate a formatted comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PROSODY COMPARISON REPORT - REAL AUDIO")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("Using real LibriTTS-R audio with ground truth annotations")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    n = len(results)
    pitch_errors = [r.pitch_mean_error_hz for r in results]
    pitch_pct_errors = [r.pitch_mean_error_pct for r in results]
    pitch_category_matches = sum(1 for r in results if r.pitch_category_match)
    speaking_rate_matches = sum(1 for r in results if r.speaking_rate_category_match)
    gender_matches = sum(1 for r in results if r.gender_match)

    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total samples analyzed: {n}")
    lines.append("")

    lines.append("Pitch Analysis (vs Ground Truth):")
    lines.append(f"  Mean absolute error: {np.mean(pitch_errors):.2f} Hz")
    lines.append(f"  Median absolute error: {np.median(pitch_errors):.2f} Hz")
    lines.append(f"  Max error: {np.max(pitch_errors):.2f} Hz")
    lines.append(f"  Mean percentage error: {np.mean(pitch_pct_errors):.1f}%")
    lines.append(f"  Category match rate: {pitch_category_matches}/{n} ({pitch_category_matches/n*100:.1f}%)")
    lines.append("")

    lines.append("Speaking Rate Analysis:")
    lines.append(f"  Category match rate: {speaking_rate_matches}/{n} ({speaking_rate_matches/n*100:.1f}%)")
    lines.append("")

    lines.append("Gender Inference (from F0):")
    lines.append(f"  Match rate: {gender_matches}/{n} ({gender_matches/n*100:.1f}%)")
    lines.append("")

    # Voice quality summary
    avg_jitter = np.mean([r.our_jitter for r in results])
    avg_shimmer = np.mean([r.our_shimmer for r in results])
    avg_hnr = np.mean([r.our_hnr for r in results])

    lines.append("Voice Quality Metrics (averaged):")
    lines.append(f"  Jitter: {avg_jitter*100:.3f}%")
    lines.append(f"  Shimmer: {avg_shimmer*100:.3f}%")
    lines.append(f"  HNR: {avg_hnr:.1f} dB")
    lines.append("")

    # Detailed results table
    lines.append("=" * 80)
    lines.append("DETAILED PITCH COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    pitch_table = []
    for r in results:
        pitch_table.append([
            r.sample_id[:20],
            f"{r.gt_pitch_mean_hz:.1f}",
            f"{r.our_pitch_mean_hz:.1f}",
            f"{r.pitch_mean_error_hz:.1f}",
            f"{r.pitch_mean_error_pct:.1f}%",
            "Y" if r.pitch_category_match else "N",
            r.gt_pitch_category[:15],
        ])

    lines.append(tabulate(
        pitch_table,
        headers=["Sample ID", "GT Hz", "Our Hz", "Error", "Err%", "Cat?", "GT Category"],
        tablefmt="simple"
    ))
    lines.append("")

    # Speaking rate comparison
    lines.append("=" * 80)
    lines.append("SPEAKING RATE COMPARISON")
    lines.append("=" * 80)
    lines.append("")

    rate_table = []
    for r in results:
        our_category = speaking_rate_sps_to_category(r.our_speaking_rate_sps)
        rate_table.append([
            r.sample_id[:20],
            r.gt_speaking_rate[:15],
            f"{r.our_speaking_rate_sps:.2f}",
            our_category[:15],
            "Y" if r.speaking_rate_category_match else "N",
        ])

    lines.append(tabulate(
        rate_table,
        headers=["Sample ID", "GT Rate", "Our SPS", "Our Category", "Match?"],
        tablefmt="simple"
    ))
    lines.append("")

    return "\n".join(lines)


def generate_json_report(results: List[RealAudioComparisonResult]) -> Dict[str, Any]:
    """Generate JSON-formatted report."""
    n = len(results)

    return {
        "timestamp": datetime.now().isoformat(),
        "sample_count": n,
        "audio_type": "real",
        "dataset": "LibriTTS-R",
        "summary": {
            "pitch": {
                "mean_absolute_error_hz": float(np.mean([r.pitch_mean_error_hz for r in results])),
                "median_absolute_error_hz": float(np.median([r.pitch_mean_error_hz for r in results])),
                "max_error_hz": float(np.max([r.pitch_mean_error_hz for r in results])),
                "mean_percentage_error": float(np.mean([r.pitch_mean_error_pct for r in results])),
                "category_match_rate": sum(1 for r in results if r.pitch_category_match) / n,
            },
            "speaking_rate": {
                "category_match_rate": sum(1 for r in results if r.speaking_rate_category_match) / n,
            },
            "gender_inference": {
                "match_rate": sum(1 for r in results if r.gender_match) / n,
            },
            "voice_quality": {
                "avg_jitter": float(np.mean([r.our_jitter for r in results])),
                "avg_shimmer": float(np.mean([r.our_shimmer for r in results])),
                "avg_hnr": float(np.mean([r.our_hnr for r in results])),
            },
        },
        "samples": [asdict(r) for r in results],
    }


# ============== Main ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare prosody analysis against real LibriTTS-R audio"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Number of samples to analyze (default: 50)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory containing downloaded audio (default: data/audio_datasets/libritts_r)"
    )
    parser.add_argument(
        "--use-qwen",
        action="store_true",
        help="Include Qwen2-Audio semantic analysis (slow)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Setup directories
    audio_dir = Path(args.audio_dir) if args.audio_dir else AUDIO_DIR
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Prosody Comparison - Real Audio")
    print("=" * 60)
    print(f"Audio directory: {audio_dir}")
    print(f"Samples to analyze: {args.samples}")
    print(f"Qwen2-Audio: {'Yes' if args.use_qwen else 'No'}")
    print()

    # Check for downloaded audio
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        print()
        print("Please download real audio first:")
        print("  python scripts/download_audio_datasets.py --dataset libritts_r --samples 100")
        sys.exit(1)

    # Load samples
    print("Loading real audio samples...")
    samples = load_real_audio_samples(audio_dir, limit=args.samples)

    if len(samples) == 0:
        print("Error: No samples found!")
        print("Please download audio using download_audio_datasets.py")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")
    print()

    # Initialize analyzers
    print("Initializing analyzers...")
    acoustic_analyzer = AcousticAnalyzer()
    rhythm_analyzer = RhythmAnalyzer()

    semantic_analyzer = None
    if args.use_qwen:
        from prosody_analyzer import SemanticAnalyzer
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Qwen2-Audio on {device}...")
        semantic_analyzer = SemanticAnalyzer(device=device)
        semantic_analyzer.load_model()

    print()

    # Process samples
    results = []
    print("Analyzing samples...")

    for i, sample in enumerate(tqdm(samples, disable=args.quiet)):
        if not args.quiet:
            tqdm.write(f"\n[{i+1}/{len(samples)}] {sample.id}")
            tqdm.write(f"  Text: {sample.text[:50]}...")
            tqdm.write(f"  GT: pitch={sample.pitch_category} ({sample.utterance_pitch_mean:.1f}Hz)")

        try:
            result = analyze_real_sample(
                sample,
                acoustic_analyzer,
                rhythm_analyzer,
                semantic_analyzer,
            )
            results.append(result)

            if not args.quiet:
                tqdm.write(f"  Our: pitch={result.our_pitch_mean_hz:.1f}Hz (err: {result.pitch_mean_error_pct:.1f}%)")
                tqdm.write(f"       rate={result.our_speaking_rate_sps:.2f} sps")

        except Exception as e:
            if not args.quiet:
                tqdm.write(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"Successfully analyzed: {len(results)}/{len(samples)} samples")
    print()

    if len(results) == 0:
        print("No samples were successfully analyzed!")
        sys.exit(1)

    # Generate reports
    report_text = generate_report(results)
    report_json = generate_json_report(results)

    # Print report
    print(report_text)

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    text_path = output_dir / f"real_audio_comparison_{timestamp}.txt"
    with open(text_path, "w") as f:
        f.write(report_text)
    print(f"\nText report saved: {text_path}")

    json_path = output_dir / f"real_audio_comparison_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"JSON report saved: {json_path}")

    # Print final summary
    print()
    print("=" * 60)
    print("FINAL SUMMARY - REAL AUDIO ANALYSIS")
    print("=" * 60)
    summary = report_json["summary"]
    print(f"Pitch Mean Absolute Error: {summary['pitch']['mean_absolute_error_hz']:.2f} Hz")
    print(f"Pitch Category Match Rate: {summary['pitch']['category_match_rate']*100:.1f}%")
    print(f"Speaking Rate Match: {summary['speaking_rate']['category_match_rate']*100:.1f}%")
    print(f"Gender Inference Accuracy: {summary['gender_inference']['match_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
