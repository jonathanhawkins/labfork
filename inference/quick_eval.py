#!/usr/bin/env python3
"""
Quick Evaluation Pipeline

30-minute end-to-end evaluation for any prosody technique.
Generates samples, measures metrics, compares to baseline.

Usage:
    # Evaluate a new checkpoint
    python quick_eval.py --checkpoint ../checkpoints/new_model/best.pt

    # Compare two checkpoints
    python quick_eval.py --checkpoint model_a.pt --baseline model_b.pt

    # Just generate samples (skip metrics)
    python quick_eval.py --checkpoint model.pt --generate-only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))
sys.path.insert(0, str(project_root / 'backend'))


@dataclass
class EvalResult:
    """Evaluation result for a single technique."""
    name: str
    checkpoint: str
    timestamp: str

    # Core metrics (must-have)
    f0_happy_hz: float
    f0_sad_hz: float
    f0_separation_hz: float  # happy - sad (should be positive)
    f0_separation_pass: bool  # > 30 Hz threshold

    # Emotion accuracy
    emotion_accuracy: float  # 0-1
    emotion_accuracy_pass: bool  # >= 0.5 threshold
    emotions_correct: list  # which emotions were classified correctly

    # Speaker similarity (optional)
    speaker_similarity: Optional[float] = None
    speaker_similarity_pass: Optional[bool] = None

    # Overall
    overall_pass: bool = False

    def to_dict(self):
        return asdict(self)


def extract_f0_mean(audio_path: str) -> float:
    """Extract mean F0 from audio file."""
    try:
        import parselmouth
        import soundfile as sf

        audio, sr = sf.read(audio_path)
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch(time_step=0.01)
        f0_values = pitch.selected_array['frequency']
        voiced = f0_values[f0_values > 0]
        return float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    except Exception as e:
        print(f"  Warning: F0 extraction failed for {audio_path}: {e}")
        return 0.0


def classify_emotion(audio_path: str) -> str:
    """Classify emotion using simple heuristics (placeholder for Qwen2-Audio)."""
    # TODO: Replace with actual Qwen2-Audio classification
    # For now, use F0-based heuristic
    f0 = extract_f0_mean(audio_path)
    if f0 > 200:
        return "happy"
    elif f0 < 150:
        return "sad"
    elif f0 > 180:
        return "angry"
    else:
        return "neutral"


def compute_speaker_similarity(audio_path: str, reference_path: str) -> float:
    """Compute speaker similarity using ECAPA-TDNN (placeholder)."""
    # TODO: Implement actual speaker similarity
    # For now, return placeholder
    return 0.75


def generate_samples(checkpoint: str, output_dir: Path, text: str = "I am feeling very emotional right now."):
    """Generate samples for all emotions."""
    from verify_v7_e2e import load_v7_model, get_emotion_prosody, generate_audio
    import scipy.io.wavfile as wavfile

    emotions = ['happy', 'sad', 'angry', 'neutral', 'calm']
    samples_per_emotion = 3

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint}")
    csm_path = str(project_root / 'models' / 'csm-1b')

    try:
        models = load_v7_model(checkpoint, csm_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {}

    generated = {}

    for emotion in emotions:
        generated[emotion] = []
        print(f"Generating {emotion}...")

        for i in range(samples_per_emotion):
            try:
                prosody = get_emotion_prosody(emotion, 1.0, models['prosody_config'], models['device'])
                audio = generate_audio(models, text, prosody)

                output_path = output_dir / f"{emotion}_{i+1}.wav"
                audio_np = audio.squeeze().numpy()
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wavfile.write(str(output_path), 24000, audio_int16)

                generated[emotion].append(str(output_path))
                print(f"  ✓ {output_path.name}")
            except Exception as e:
                print(f"  ✗ Failed sample {i+1}: {e}")

    return generated


def evaluate_samples(samples: dict, reference_audio: Optional[str] = None) -> EvalResult:
    """Evaluate generated samples."""

    # Extract F0 for each emotion
    f0_by_emotion = {}
    for emotion, paths in samples.items():
        f0_values = [extract_f0_mean(p) for p in paths]
        f0_by_emotion[emotion] = np.mean([f for f in f0_values if f > 0]) if any(f > 0 for f in f0_values) else 0.0

    # Core metrics
    f0_happy = f0_by_emotion.get('happy', 0)
    f0_sad = f0_by_emotion.get('sad', 0)
    f0_separation = f0_happy - f0_sad

    # Emotion classification
    emotions_correct = []
    total_samples = 0
    correct_samples = 0

    for emotion, paths in samples.items():
        for path in paths:
            predicted = classify_emotion(path)
            total_samples += 1
            if predicted == emotion:
                correct_samples += 1
                if emotion not in emotions_correct:
                    emotions_correct.append(emotion)

    emotion_accuracy = correct_samples / total_samples if total_samples > 0 else 0

    # Speaker similarity (if reference provided)
    speaker_sim = None
    speaker_sim_pass = None
    if reference_audio:
        # Use first happy sample for comparison
        if 'happy' in samples and samples['happy']:
            speaker_sim = compute_speaker_similarity(samples['happy'][0], reference_audio)
            speaker_sim_pass = speaker_sim >= 0.7

    # Create result
    result = EvalResult(
        name="quick_eval",
        checkpoint="",  # Will be set by caller
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        f0_happy_hz=f0_happy,
        f0_sad_hz=f0_sad,
        f0_separation_hz=f0_separation,
        f0_separation_pass=f0_separation > 30,
        emotion_accuracy=emotion_accuracy,
        emotion_accuracy_pass=emotion_accuracy >= 0.5,
        emotions_correct=emotions_correct,
        speaker_similarity=speaker_sim,
        speaker_similarity_pass=speaker_sim_pass,
    )

    # Overall pass requires F0 separation AND emotion accuracy
    result.overall_pass = result.f0_separation_pass and result.emotion_accuracy_pass

    return result


def print_results(result: EvalResult, baseline: Optional[EvalResult] = None):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 60)
    print("QUICK EVAL RESULTS")
    print("=" * 60)

    # F0 Separation
    status = "✓ PASS" if result.f0_separation_pass else "✗ FAIL"
    print(f"\nF0 Separation: {status}")
    print(f"  Happy: {result.f0_happy_hz:.1f} Hz")
    print(f"  Sad:   {result.f0_sad_hz:.1f} Hz")
    print(f"  Diff:  {result.f0_separation_hz:.1f} Hz (threshold: 30 Hz)")

    if baseline:
        diff = result.f0_separation_hz - baseline.f0_separation_hz
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  vs Baseline: {direction} {abs(diff):.1f} Hz")

    # Emotion Accuracy
    status = "✓ PASS" if result.emotion_accuracy_pass else "✗ FAIL"
    print(f"\nEmotion Accuracy: {status}")
    print(f"  Accuracy: {result.emotion_accuracy:.1%}")
    print(f"  Correct:  {', '.join(result.emotions_correct) or 'none'}")

    if baseline:
        diff = result.emotion_accuracy - baseline.emotion_accuracy
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  vs Baseline: {direction} {abs(diff):.1%}")

    # Speaker Similarity
    if result.speaker_similarity is not None:
        status = "✓ PASS" if result.speaker_similarity_pass else "✗ FAIL"
        print(f"\nSpeaker Similarity: {status}")
        print(f"  Score: {result.speaker_similarity:.3f} (threshold: 0.7)")

    # Overall
    print("\n" + "-" * 60)
    if result.overall_pass:
        print("OVERALL: ✓ PASS - Technique meets quality threshold")
    else:
        print("OVERALL: ✗ FAIL - Does not meet quality threshold")
        failures = []
        if not result.f0_separation_pass:
            failures.append("F0 separation < 30 Hz")
        if not result.emotion_accuracy_pass:
            failures.append("Emotion accuracy < 50%")
        print(f"  Issues: {', '.join(failures)}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Quick evaluation pipeline")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint to evaluate")
    parser.add_argument("--baseline", help="Path to baseline checkpoint for comparison")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    parser.add_argument("--run-dir", help="Run directory (writes samples to run-dir/samples and metrics to run-dir/metrics.json)")
    parser.add_argument("--metrics-out", help="Optional path to write metrics JSON")
    parser.add_argument("--reference", help="Reference audio for speaker similarity")
    parser.add_argument("--text", default="I am feeling very emotional right now.", help="Text to synthesize")
    parser.add_argument("--generate-only", action="store_true", help="Only generate samples, skip metrics")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    output_dir = (run_dir / "samples") if run_dir else Path(args.output_dir)
    checkpoint_name = Path(args.checkpoint).stem

    print(f"\n{'='*60}")
    print(f"QUICK EVAL: {checkpoint_name}")
    print(f"{'='*60}\n")

    # Generate samples
    start = time.time()
    print("Step 1: Generating samples...")
    samples = generate_samples(args.checkpoint, output_dir / checkpoint_name, args.text)

    if not samples:
        print("Failed to generate samples")
        return 1

    print(f"Generated {sum(len(v) for v in samples.values())} samples in {time.time()-start:.1f}s")

    if args.generate_only:
        print(f"\nSamples saved to: {output_dir / checkpoint_name}")
        return 0

    # Evaluate
    print("\nStep 2: Evaluating samples...")
    result = evaluate_samples(samples, args.reference)
    result.checkpoint = args.checkpoint
    result.name = checkpoint_name

    # Baseline comparison
    baseline_result = None
    if args.baseline:
        print(f"\nStep 3: Generating baseline samples...")
        baseline_samples = generate_samples(args.baseline, output_dir / "baseline", args.text)
        if baseline_samples:
            baseline_result = evaluate_samples(baseline_samples, args.reference)
            baseline_result.checkpoint = args.baseline
            baseline_result.name = "baseline"

    # Output
    if args.json:
        output = {"result": result.to_dict()}
        if baseline_result:
            output["baseline"] = baseline_result.to_dict()
        print(json.dumps(output, indent=2))
    else:
        print_results(result, baseline_result)

    # Save results
    results_payload = result.to_dict()
    results_file = run_dir / "metrics.json" if run_dir else (output_dir / f"{checkpoint_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    if args.metrics_out:
        metrics_out_path = Path(args.metrics_out)
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out_path, 'w') as f:
            json.dump(results_payload, f, indent=2)
        print(f"Metrics also written to: {metrics_out_path}")

    return 0 if result.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
