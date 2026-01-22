#!/usr/bin/env python3
"""
Test Seed-VC Voice Conversion

Tests the Seed-VC model with emotion reference samples.
"""

import sys
import time
from pathlib import Path

# Add seed-vc to path
SEED_VC_PATH = Path(__file__).parent.parent / "seed-vc"
sys.path.insert(0, str(SEED_VC_PATH))

import torch
import numpy as np
import scipy.io.wavfile as wavfile


def load_seed_vc():
    """Load Seed-VC model."""
    print("Loading Seed-VC model...")
    print("This may take a minute on first run (downloading models)...")

    from seed_vc_wrapper import SeedVCWrapper

    wrapper = SeedVCWrapper()
    print(f"Model loaded on device: {wrapper.device}")
    print(f"Sample rate: {wrapper.sr}")

    return wrapper


def convert_emotion(
    wrapper,
    source_path: str,
    reference_path: str,
    output_path: str,
    diffusion_steps: int = 10,
):
    """
    Convert voice emotion using Seed-VC.

    Args:
        wrapper: Loaded SeedVCWrapper
        source_path: Path to source audio (what to convert)
        reference_path: Path to reference audio (target voice/emotion)
        output_path: Path to save output
        diffusion_steps: Number of diffusion steps (4-10 for real-time, 25+ for quality)
    """
    print(f"\n{'='*60}")
    print(f"Converting voice:")
    print(f"  Source: {Path(source_path).name}")
    print(f"  Reference: {Path(reference_path).name}")
    print(f"  Diffusion steps: {diffusion_steps}")

    start_time = time.time()

    # Perform conversion (stream_output=False returns numpy array)
    result = None
    for chunk in wrapper.convert_voice(
        source=source_path,
        target=reference_path,
        diffusion_steps=diffusion_steps,
        stream_output=True,
        inference_cfg_rate=0.7,
    ):
        mp3_bytes, full_audio = chunk
        result = full_audio

    inference_time = time.time() - start_time

    if result is not None:
        # Result is tuple: (sample_rate, audio_data)
        if isinstance(result, tuple):
            sr, audio_data = result
        else:
            sr = wrapper.sr
            audio_data = result

        # Convert to numpy if tensor
        if hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()

        # Save as WAV (audio is float32 in range [-1, 1])
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(output_path, sr, audio_int16)

        audio_duration = len(audio_data) / sr
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Real-time factor: {inference_time / audio_duration:.2f}x")
        print(f"  Output saved: {output_path}")
    else:
        print("  ERROR: No output generated!")

    return result


def find_voice_samples():
    """Find available voice samples."""
    samples_dir = Path(__file__).parent.parent / "data" / "voice_samples"

    if not samples_dir.exists():
        print(f"Voice samples directory not found: {samples_dir}")
        return None, {}

    # Find sample session
    session_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not session_dirs:
        print("No voice sample sessions found!")
        return None, {}

    session_dir = session_dirs[0]
    print(f"Using samples from: {session_dir.name}")

    # Find samples for different emotions
    samples = {}
    for wav_file in session_dir.glob("*.wav"):
        if "_raw" in wav_file.name:
            continue
        emotion = wav_file.stem.split("_")[0]
        if emotion not in samples:
            samples[emotion] = wav_file

    print(f"Found emotions: {list(samples.keys())}")
    return session_dir, samples


def test_emotion_conversion():
    """Test converting between emotions using recorded samples."""
    print("\n" + "=" * 60)
    print("SEED-VC EMOTION CONVERSION TEST")
    print("=" * 60)

    session_dir, samples = find_voice_samples()
    if not samples:
        return

    # Create output directory
    output_dir = Path(__file__).parent / "outputs" / "seed_vc_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    wrapper = load_seed_vc()

    # Define test conversions
    conversions = [
        # (source_emotion, target_emotion)
        ("calm", "happy"),
        ("calm", "angry"),
        ("calm", "sad"),
        ("happy", "sad"),
        ("angry", "calm"),
    ]

    # Filter to available emotions
    available = list(samples.keys())
    conversions = [
        (src, tgt)
        for src, tgt in conversions
        if src in available and tgt in available
    ]

    if not conversions:
        # Fallback: use first two emotions
        if len(available) >= 2:
            conversions = [(available[0], available[1])]
        else:
            print("Not enough emotion samples for testing!")
            return

    print(f"\nWill test {len(conversions)} conversions:")
    for src, tgt in conversions:
        print(f"  {src} -> {tgt}")

    # Run conversions
    for source_emotion, target_emotion in conversions:
        output_path = output_dir / f"{source_emotion}_to_{target_emotion}.wav"
        convert_emotion(
            wrapper,
            str(samples[source_emotion]),
            str(samples[target_emotion]),
            str(output_path),
            diffusion_steps=10,
        )

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


def test_speed_comparison():
    """Compare different diffusion step counts."""
    print("\n" + "=" * 60)
    print("SPEED COMPARISON TEST")
    print("=" * 60)

    session_dir, samples = find_voice_samples()
    if not samples or len(samples) < 2:
        print("Need at least 2 emotion samples!")
        return

    output_dir = Path(__file__).parent / "outputs" / "seed_vc_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_seed_vc()

    emotions = list(samples.keys())
    source = samples[emotions[0]]
    target = samples[emotions[1]]

    print(f"\nComparing diffusion steps: {source.stem} -> {target.stem}")

    # Test different step counts
    for steps in [4, 6, 10, 25]:
        output_path = output_dir / f"speed_test_{steps}_steps.wav"
        convert_emotion(
            wrapper,
            str(source),
            str(target),
            str(output_path),
            diffusion_steps=steps,
        )


def test_latency_benchmark():
    """Benchmark inference latency."""
    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)

    session_dir, samples = find_voice_samples()
    if not samples or len(samples) < 2:
        print("Need at least 2 emotion samples!")
        return

    wrapper = load_seed_vc()

    emotions = list(samples.keys())
    source = samples[emotions[0]]
    target = samples[emotions[1]]

    # Warmup
    print("\nWarming up...")
    output_dir = Path(__file__).parent / "outputs" / "seed_vc_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(2):
        for chunk in wrapper.convert_voice(
            source=str(source),
            target=str(target),
            diffusion_steps=4,
            stream_output=True,
        ):
            pass

    # Benchmark
    print("\nBenchmarking with 4 diffusion steps...")
    latencies = []
    for i in range(5):
        start = time.time()
        for chunk in wrapper.convert_voice(
            source=str(source),
            target=str(target),
            diffusion_steps=4,
            stream_output=True,
        ):
            pass
        latency = time.time() - start
        latencies.append(latency)
        print(f"  Run {i+1}: {latency:.2f}s")

    avg = np.mean(latencies)
    std = np.std(latencies)
    print(f"\nResults:")
    print(f"  Average: {avg:.2f}s (±{std:.2f}s)")
    print(f"  For real-time, need < audio duration")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Seed-VC voice conversion")
    parser.add_argument("--speed", action="store_true", help="Run speed comparison")
    parser.add_argument("--latency", action="store_true", help="Run latency benchmark")
    parser.add_argument("--convert", action="store_true", help="Run emotion conversion")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.all or (not args.speed and not args.latency and not args.convert):
        test_emotion_conversion()
        test_speed_comparison()
    elif args.speed:
        test_speed_comparison()
    elif args.latency:
        test_latency_benchmark()
    elif args.convert:
        test_emotion_conversion()
