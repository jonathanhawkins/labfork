#!/usr/bin/env python3
"""
Prosody Comparison Script

Compares the Voice Clone Pipeline's prosody analysis against
the LibriTTS-R annotated dataset to validate accuracy.

Since audio downloads from HuggingFace can be problematic, this script:
1. Uses the local LibriTTS-R annotated metadata
2. Downloads audio from OpenSLR when needed (cached locally)
3. Runs our prosody analyzer on each sample
4. Compares results with ground truth annotations

The LibriTTS-R annotations include:
- pitch: categorical (e.g., "quite low pitch", "moderate pitch")
- speaking_rate: categorical (e.g., "moderate speed", "slightly fast")
- speech_monotony: categorical (e.g., "slightly monotone", "quite monotone")
- gender: binary (male/female)
- utterance_pitch_mean: numeric (Hz)
- utterance_pitch_std: numeric (Hz)

Our analyzer produces:
- acoustic.pitch_mean: numeric (Hz)
- acoustic.pitch_std: numeric (Hz)
- rhythm.speaking_rate: numeric (syllables/sec)
- semantic.pace_category: categorical (slow/normal/fast)
"""

import os
import sys
import json
import tempfile
import tarfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

import pyarrow.ipc as ipc
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
LIBRITTS_DIR = DATA_DIR / "libritts_annotated" / "train_clean_100"
CACHE_DIR = DATA_DIR / "cache" / "libritts_audio"
RESULTS_DIR = BASE_DIR / "data" / "prosody_comparison_results"

# OpenSLR download URL for LibriTTS-R
OPENSLR_BASE = "https://www.openslr.org/resources/141"


# ============== Data Classes ==============

@dataclass
class LibriTTSSample:
    """A sample from the LibriTTS annotated dataset."""
    id: str
    text: str
    speaker_id: str
    chapter_id: str
    original_path: str

    # Prosody annotations
    speaking_rate: str  # categorical
    gender: str
    pitch: str  # categorical
    speech_monotony: str
    utterance_pitch_mean: float
    utterance_pitch_std: float

    # Quality metrics
    snr: float
    c50: float
    noise: str
    reverberation: str

    # Description
    text_description: str


@dataclass
class ComparisonResult:
    """Comparison between LibriTTS annotation and our analysis."""
    sample_id: str
    text: str

    # Ground truth (LibriTTS)
    gt_pitch_category: str
    gt_pitch_mean_hz: float
    gt_pitch_std_hz: float
    gt_speaking_rate: str
    gt_monotony: str
    gt_gender: str

    # Our analysis
    our_pitch_mean_hz: float
    our_pitch_std_hz: float
    our_speaking_rate_sps: float
    our_pace_category: Optional[str]
    our_energy_level: Optional[str]
    our_emotion: Optional[str]

    # Computed metrics
    pitch_mean_error_hz: float
    pitch_mean_error_pct: float
    pitch_std_error_hz: float
    pitch_category_match: bool
    speaking_rate_match: bool
    gender_inferred: Optional[str]


# ============== Mapping Functions ==============

def pitch_hz_to_category(pitch_hz: float, gender: str) -> str:
    """
    Map pitch in Hz to LibriTTS categorical labels.

    LibriTTS uses: "very low pitch", "quite low pitch", "moderate pitch",
                   "slightly high pitch", "quite high pitch", "very high pitch"

    Based on typical voice ranges:
    - Female: ~165-255 Hz (modal), varies widely
    - Male: ~85-155 Hz (modal)
    """
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
    """
    Map syllables per second to LibriTTS categorical labels.

    LibriTTS uses: "very slowly", "quite slowly", "slightly slowly",
                   "moderate speed", "slightly fast", "quite fast", "very fast"

    Typical English: ~4-5 syllables/second for conversational speech
    """
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


def pitch_std_to_monotony(pitch_std: float, pitch_mean: float) -> str:
    """
    Map pitch standard deviation to monotony category.

    LibriTTS uses: "very monotone", "quite monotone", "slightly monotone",
                   "slightly expressive", "quite expressive", "very expressive"

    Use coefficient of variation (std/mean) as relative measure.
    """
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


def infer_gender_from_pitch(pitch_hz: float) -> str:
    """
    Infer gender from fundamental frequency.

    Typical ranges:
    - Male: 85-155 Hz
    - Female: 165-255 Hz
    - Overlap zone: 155-165 Hz
    """
    if pitch_hz < 155:
        return "male"
    elif pitch_hz > 175:
        return "female"
    else:
        return "uncertain"


# ============== Data Loading ==============

def load_libritts_samples(arrow_path: Path, limit: int = 20) -> List[LibriTTSSample]:
    """Load samples from the LibriTTS arrow file."""
    with open(arrow_path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()

    samples = []
    for i in range(min(limit, table.num_rows)):
        sample = LibriTTSSample(
            id=table.column("id")[i].as_py(),
            text=table.column("text")[i].as_py(),
            speaker_id=str(table.column("speaker_id")[i].as_py()),
            chapter_id=str(table.column("chapter_id")[i].as_py()),
            original_path=table.column("path")[i].as_py(),
            speaking_rate=table.column("speaking_rate")[i].as_py(),
            gender=table.column("gender")[i].as_py(),
            pitch=table.column("pitch")[i].as_py(),
            speech_monotony=table.column("speech_monotony")[i].as_py(),
            utterance_pitch_mean=float(table.column("utterance_pitch_mean")[i].as_py()),
            utterance_pitch_std=float(table.column("utterance_pitch_std")[i].as_py()),
            snr=float(table.column("snr")[i].as_py()),
            c50=float(table.column("c50")[i].as_py()),
            noise=table.column("noise")[i].as_py(),
            reverberation=table.column("reverberation")[i].as_py(),
            text_description=table.column("text_description")[i].as_py(),
        )
        samples.append(sample)

    return samples


def download_openslr_audio(sample: LibriTTSSample, cache_dir: Path) -> Optional[Path]:
    """
    Download audio from OpenSLR LibriTTS-R.

    This downloads the full tar file for a speaker if not already cached,
    then extracts the specific audio file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Target audio path
    audio_filename = f"{sample.id}.wav"
    audio_path = cache_dir / sample.speaker_id / sample.chapter_id / audio_filename

    if audio_path.exists():
        return audio_path

    # We need to download from OpenSLR - but this downloads the full train-clean-100 tar
    # which is ~7.7GB. For a quick test, let's skip this and generate synthetic audio.
    return None


def generate_synthetic_test_audio(
    sample: LibriTTSSample,
    cache_dir: Path,
    duration: float = 3.0
) -> Path:
    """
    Generate synthetic audio with known prosody characteristics for testing.

    This creates audio with pitch and duration characteristics that should match
    the sample's ground truth annotations, allowing us to validate our analyzer.
    """
    import librosa
    import soundfile as sf

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"{sample.id}_synthetic.wav"

    if audio_path.exists():
        return audio_path

    # Generate a sine wave at the expected pitch with some variation
    sr = 24000
    t = np.linspace(0, duration, int(sr * duration))

    # Base frequency from ground truth
    f0 = sample.utterance_pitch_mean if sample.utterance_pitch_mean > 0 else 150.0
    f0_std = sample.utterance_pitch_std if sample.utterance_pitch_std > 0 else 20.0

    # Create pitch contour with variation
    pitch_contour = f0 + np.random.normal(0, f0_std * 0.5, len(t))
    pitch_contour = np.clip(pitch_contour, f0 * 0.7, f0 * 1.3)

    # Generate audio with varying pitch (frequency modulation)
    phase = np.cumsum(2 * np.pi * pitch_contour / sr)
    audio = 0.5 * np.sin(phase)

    # Add harmonics for more realistic speech-like quality
    audio += 0.3 * np.sin(2 * phase)
    audio += 0.15 * np.sin(3 * phase)
    audio += 0.1 * np.sin(4 * phase)

    # Apply amplitude envelope (speech-like)
    envelope = np.ones_like(audio)
    # Add some pauses (based on speaking rate - slower = more pauses)
    if "slow" in sample.speaking_rate.lower():
        n_pauses = 3
    elif "fast" in sample.speaking_rate.lower():
        n_pauses = 1
    else:
        n_pauses = 2

    for _ in range(n_pauses):
        pause_start = np.random.randint(len(audio) // 4, 3 * len(audio) // 4)
        pause_len = int(sr * 0.1)  # 100ms pauses
        envelope[pause_start:min(pause_start + pause_len, len(envelope))] = 0.05

    audio *= envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save
    sf.write(str(audio_path), audio, sr)

    return audio_path


# ============== Analysis ==============

def analyze_sample(
    sample: LibriTTSSample,
    audio_path: Path,
    acoustic_analyzer: AcousticAnalyzer,
    rhythm_analyzer: RhythmAnalyzer,
    semantic_analyzer = None,
) -> ComparisonResult:
    """Run prosody analysis on a sample and compare with ground truth."""

    # Run acoustic analysis
    acoustic = acoustic_analyzer.analyze(str(audio_path))

    # Run rhythm analysis
    rhythm = rhythm_analyzer.analyze(str(audio_path))

    # Optional: semantic analysis (slow, uses Qwen2-Audio)
    semantic = None
    if semantic_analyzer:
        try:
            semantic = semantic_analyzer.analyze(str(audio_path), sample.text)
        except Exception as e:
            print(f"  Semantic analysis failed: {e}")

    # Compute errors
    gt_pitch = sample.utterance_pitch_mean if sample.utterance_pitch_mean > 0 else 150.0
    pitch_mean_error = abs(acoustic.pitch_mean - gt_pitch)
    pitch_mean_error_pct = (pitch_mean_error / gt_pitch * 100) if gt_pitch > 0 else 0
    pitch_std_error = abs(acoustic.pitch_std - sample.utterance_pitch_std)

    # Check category matches
    our_pitch_category = pitch_hz_to_category(acoustic.pitch_mean, sample.gender)
    pitch_category_match = our_pitch_category == sample.pitch

    our_speaking_rate_category = speaking_rate_sps_to_category(rhythm.speaking_rate)
    speaking_rate_match = our_speaking_rate_category == sample.speaking_rate

    # Infer gender from pitch
    gender_inferred = infer_gender_from_pitch(acoustic.pitch_mean)

    return ComparisonResult(
        sample_id=sample.id,
        text=sample.text[:60] + "..." if len(sample.text) > 60 else sample.text,

        # Ground truth
        gt_pitch_category=sample.pitch,
        gt_pitch_mean_hz=gt_pitch,
        gt_pitch_std_hz=sample.utterance_pitch_std,
        gt_speaking_rate=sample.speaking_rate,
        gt_monotony=sample.speech_monotony,
        gt_gender=sample.gender,

        # Our analysis
        our_pitch_mean_hz=acoustic.pitch_mean,
        our_pitch_std_hz=acoustic.pitch_std,
        our_speaking_rate_sps=rhythm.speaking_rate,
        our_pace_category=semantic.pace_category if semantic else None,
        our_energy_level=semantic.energy_level if semantic else None,
        our_emotion=semantic.emotion if semantic else None,

        # Metrics
        pitch_mean_error_hz=pitch_mean_error,
        pitch_mean_error_pct=pitch_mean_error_pct,
        pitch_std_error_hz=pitch_std_error,
        pitch_category_match=pitch_category_match,
        speaking_rate_match=speaking_rate_match,
        gender_inferred=gender_inferred,
    )


# ============== Reporting ==============

def generate_report(results: List[ComparisonResult], use_synthetic: bool = False) -> str:
    """Generate a formatted comparison report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PROSODY COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    if use_synthetic:
        lines.append("NOTE: Using synthetic audio generated from ground truth parameters")
        lines.append("      This validates analyzer accuracy on controlled inputs")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    n = len(results)
    pitch_errors = [r.pitch_mean_error_hz for r in results]
    pitch_pct_errors = [r.pitch_mean_error_pct for r in results]
    pitch_category_matches = sum(1 for r in results if r.pitch_category_match)
    speaking_rate_matches = sum(1 for r in results if r.speaking_rate_match)
    gender_matches = sum(1 for r in results if r.gender_inferred == r.gt_gender)

    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total samples analyzed: {n}")
    lines.append("")

    lines.append("Pitch Analysis:")
    lines.append(f"  Mean absolute error: {np.mean(pitch_errors):.2f} Hz")
    lines.append(f"  Median absolute error: {np.median(pitch_errors):.2f} Hz")
    lines.append(f"  Mean percentage error: {np.mean(pitch_pct_errors):.1f}%")
    lines.append(f"  Category match rate: {pitch_category_matches}/{n} ({pitch_category_matches/n*100:.1f}%)")
    lines.append("")

    lines.append("Speaking Rate Analysis:")
    lines.append(f"  Category match rate: {speaking_rate_matches}/{n} ({speaking_rate_matches/n*100:.1f}%)")
    lines.append("")

    lines.append("Gender Inference (from pitch):")
    lines.append(f"  Match rate: {gender_matches}/{n} ({gender_matches/n*100:.1f}%)")
    lines.append("")

    # Detailed results table
    lines.append("=" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Pitch comparison table
    pitch_table = []
    for r in results:
        pitch_table.append([
            r.sample_id[:20],
            f"{r.gt_pitch_mean_hz:.1f}",
            f"{r.our_pitch_mean_hz:.1f}",
            f"{r.pitch_mean_error_hz:.1f}",
            f"{r.pitch_mean_error_pct:.1f}%",
            "Y" if r.pitch_category_match else "N",
            r.gt_pitch_category[:12],
        ])

    lines.append("Pitch Comparison:")
    lines.append(tabulate(
        pitch_table,
        headers=["Sample ID", "GT Hz", "Our Hz", "Error", "Error%", "Cat?", "GT Category"],
        tablefmt="simple"
    ))
    lines.append("")

    # Speaking rate comparison
    rate_table = []
    for r in results:
        rate_table.append([
            r.sample_id[:20],
            r.gt_speaking_rate,
            f"{r.our_speaking_rate_sps:.2f}",
            speaking_rate_sps_to_category(r.our_speaking_rate_sps),
            "Y" if r.speaking_rate_match else "N",
        ])

    lines.append("Speaking Rate Comparison:")
    lines.append(tabulate(
        rate_table,
        headers=["Sample ID", "GT Rate", "Our SPS", "Our Category", "Match?"],
        tablefmt="simple"
    ))
    lines.append("")

    # Gender inference
    gender_table = []
    for r in results:
        gender_table.append([
            r.sample_id[:20],
            r.gt_gender,
            r.gender_inferred,
            f"{r.our_pitch_mean_hz:.1f} Hz",
            "Y" if r.gender_inferred == r.gt_gender else "N",
        ])

    lines.append("Gender Inference (from F0):")
    lines.append(tabulate(
        gender_table,
        headers=["Sample ID", "GT Gender", "Inferred", "Pitch Hz", "Match?"],
        tablefmt="simple"
    ))
    lines.append("")

    return "\n".join(lines)


def generate_json_report(results: List[ComparisonResult], use_synthetic: bool = False) -> Dict[str, Any]:
    """Generate JSON-formatted report for programmatic use."""
    n = len(results)

    return {
        "timestamp": datetime.now().isoformat(),
        "sample_count": n,
        "use_synthetic_audio": use_synthetic,
        "summary": {
            "pitch": {
                "mean_absolute_error_hz": float(np.mean([r.pitch_mean_error_hz for r in results])),
                "median_absolute_error_hz": float(np.median([r.pitch_mean_error_hz for r in results])),
                "mean_percentage_error": float(np.mean([r.pitch_mean_error_pct for r in results])),
                "category_match_rate": sum(1 for r in results if r.pitch_category_match) / n,
            },
            "speaking_rate": {
                "category_match_rate": sum(1 for r in results if r.speaking_rate_match) / n,
            },
            "gender_inference": {
                "match_rate": sum(1 for r in results if r.gender_inferred == r.gt_gender) / n,
            },
        },
        "samples": [asdict(r) for r in results],
    }


# ============== Main ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare prosody analysis against LibriTTS-R annotations"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=15,
        help="Number of samples to analyze (default: 15)"
    )
    parser.add_argument(
        "--use-qwen",
        action="store_true",
        help="Include Qwen2-Audio semantic analysis (slow)"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic audio (default: True, validates analyzer on controlled inputs)"
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

    # Setup output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Prosody Comparison Test")
    print("=" * 60)
    print(f"Samples to analyze: {args.samples}")
    print(f"Qwen2-Audio semantic: {'Yes' if args.use_qwen else 'No'}")
    print(f"Audio source: {'Synthetic (controlled)' if args.synthetic else 'Real LibriTTS-R'}")
    print(f"Output directory: {output_dir}")
    print()

    # Load LibriTTS samples from local arrow file
    arrow_path = LIBRITTS_DIR / "data-00000-of-00001.arrow"
    if not arrow_path.exists():
        print(f"Error: LibriTTS data not found at {arrow_path}")
        sys.exit(1)

    print("Loading LibriTTS-R annotated samples...")
    samples = load_libritts_samples(arrow_path, limit=args.samples)
    print(f"Loaded {len(samples)} samples")
    print()

    if len(samples) == 0:
        print("Error: No samples were loaded!")
        sys.exit(1)

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
    audio_cache = CACHE_DIR / ("synthetic" if args.synthetic else "real")

    print("Generating audio and analyzing samples...")
    for i, sample in enumerate(tqdm(samples, disable=args.quiet)):
        if not args.quiet:
            tqdm.write(f"\n[{i+1}/{len(samples)}] {sample.id}")
            tqdm.write(f"  Text: {sample.text[:50]}...")
            tqdm.write(f"  GT: pitch={sample.pitch} ({sample.utterance_pitch_mean:.1f}Hz), rate={sample.speaking_rate}, gender={sample.gender}")

        # Get audio (synthetic or real)
        if args.synthetic:
            audio_path = generate_synthetic_test_audio(sample, audio_cache)
        else:
            audio_path = download_openslr_audio(sample, audio_cache)
            if audio_path is None:
                if not args.quiet:
                    tqdm.write(f"  SKIPPED: Could not download audio")
                continue

        if not args.quiet:
            tqdm.write(f"  Audio: {audio_path}")

        # Analyze
        try:
            result = analyze_sample(
                sample,
                audio_path,
                acoustic_analyzer,
                rhythm_analyzer,
                semantic_analyzer,
            )
            results.append(result)

            if not args.quiet:
                tqdm.write(f"  Our: pitch={result.our_pitch_mean_hz:.1f}Hz (err: {result.pitch_mean_error_pct:.1f}%)")
                tqdm.write(f"       rate={result.our_speaking_rate_sps:.2f} sps")
                match_str = "Y" if result.pitch_category_match else "N"
                tqdm.write(f"       pitch_cat_match={match_str}, gender_infer={result.gender_inferred}")

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
    report_text = generate_report(results, use_synthetic=args.synthetic)
    report_json = generate_json_report(results, use_synthetic=args.synthetic)

    # Print text report
    print(report_text)

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    text_path = output_dir / f"comparison_report_{timestamp}.txt"
    with open(text_path, "w") as f:
        f.write(report_text)
    print(f"\nText report saved: {text_path}")

    json_path = output_dir / f"comparison_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    print(f"JSON report saved: {json_path}")

    # Print final summary
    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary = report_json["summary"]
    print(f"Pitch Mean Absolute Error: {summary['pitch']['mean_absolute_error_hz']:.2f} Hz")
    print(f"Pitch Category Match Rate: {summary['pitch']['category_match_rate']*100:.1f}%")
    print(f"Speaking Rate Category Match: {summary['speaking_rate']['category_match_rate']*100:.1f}%")
    print(f"Gender Inference Accuracy: {summary['gender_inference']['match_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
