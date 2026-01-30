#!/usr/bin/env python3
"""
PitchFlow Inference Script

Generate speech with explicit pitch control using PitchFlow's quantized pitch bins.

Features:
- Automatic pitch prediction from mel-spectrogram
- Explicit pitch control via F0 contour specification
- Pitch shifting in semitones
- Pitch range modification (compress/expand)

Usage:
    # Automatic pitch prediction
    python generate_with_pitchflow.py \\
        --text "Hello, how are you?" \\
        --reference reference.wav \\
        --checkpoint ../checkpoints/pitchflow/best.pt \\
        --output output.wav

    # Pitch shifting
    python generate_with_pitchflow.py \\
        --text "Hello, how are you?" \\
        --reference reference.wav \\
        --shift-semitones 5 \\
        --output output_higher.wav

    # Custom F0 contour
    python generate_with_pitchflow.py \\
        --text "Hello, how are you?" \\
        --custom-f0 100,150,200,180,120 \\
        --output custom_pitch.wav

    # Pitch range modification
    python generate_with_pitchflow.py \\
        --text "Hello, how are you?" \\
        --reference reference.wav \\
        --pitch-scale 1.5 \\
        --output expanded_range.wav
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from pitchflow import (
    PitchFlowConfig,
    PitchFlowAdapter,
    LogF0Quantizer,
)


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> PitchFlowAdapter:
    """Load PitchFlow model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = PitchFlowConfig(
            num_pitch_bins=config_dict.get('num_pitch_bins', 50),
            f0_min=config_dict.get('f0_min', 50.0),
            f0_max=config_dict.get('f0_max', 800.0),
            prosody_dim=config_dict.get('prosody_dim', 2048),
            text_dim=config_dict.get('text_dim', 768),
        )
    else:
        config = PitchFlowConfig()

    model = PitchFlowAdapter(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def extract_mel(
    audio_path: str,
    config: PitchFlowConfig,
) -> torch.Tensor:
    """Extract mel-spectrogram from audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )

    mel = mel_transform(waveform)
    mel = torch.log(mel.clamp(min=1e-5))

    return mel


def parse_custom_f0(f0_string: str) -> torch.Tensor:
    """Parse comma-separated F0 values."""
    values = [float(v.strip()) for v in f0_string.split(',')]
    return torch.tensor(values, dtype=torch.float32)


def scale_pitch_range(
    f0: torch.Tensor,
    scale: float,
    center: Optional[float] = None,
) -> torch.Tensor:
    """
    Scale pitch range around center frequency.

    Args:
        f0: F0 values
        scale: Scale factor (>1 expands range, <1 compresses)
        center: Center frequency (default: mean of voiced values)

    Returns:
        Scaled F0 values
    """
    voiced_mask = f0 > 0

    if center is None:
        if voiced_mask.any():
            center = f0[voiced_mask].mean()
        else:
            return f0

    # Scale around center
    f0_scaled = f0.clone()
    f0_scaled[voiced_mask] = center + (f0[voiced_mask] - center) * scale

    return f0_scaled


def main():
    parser = argparse.ArgumentParser(description="Generate with PitchFlow pitch control")

    # Input/Output
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--reference', type=str, default=None,
                       help='Reference audio for pitch extraction')
    parser.add_argument('--output', type=str, default='pitchflow_output.wav',
                       help='Output audio path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PitchFlow checkpoint')

    # Pitch control
    parser.add_argument('--shift-semitones', type=float, default=0,
                       help='Shift pitch by N semitones (positive=up, negative=down)')
    parser.add_argument('--custom-f0', type=str, default=None,
                       help='Custom F0 contour (comma-separated Hz values)')
    parser.add_argument('--pitch-scale', type=float, default=1.0,
                       help='Scale pitch range (>1 expands, <1 compresses)')
    parser.add_argument('--pitch-center', type=float, default=None,
                       help='Center frequency for pitch scaling (default: auto)')

    # Generation settings
    parser.add_argument('--num-steps', type=int, default=50,
                       help='ODE integration steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')

    # Sweep modes
    parser.add_argument('--sweep-semitones', action='store_true',
                       help='Generate outputs at multiple pitch shifts')
    parser.add_argument('--sweep-range', type=str, default='-6,6',
                       help='Semitone range for sweep (min,max)')

    args = parser.parse_args()

    print("=" * 70)
    print("PitchFlow Inference")
    print("Quantized Pitch Control for TTS")
    print("=" * 70)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device)
    config = model.config

    # Get mel-spectrogram
    if args.reference:
        print(f"Extracting mel from {args.reference}...")
        mel = extract_mel(args.reference, config)
        mel = mel.to(device)
    else:
        # Create synthetic mel for testing
        print("No reference provided, using synthetic mel...")
        mel = torch.randn(1, config.n_mels, 100, device=device)

    # Text conditioning (placeholder - in practice, use text encoder)
    batch_size = 1
    text_cond = torch.randn(batch_size, 20, config.text_dim, device=device)
    text_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)

    # Handle different pitch control modes
    if args.sweep_semitones:
        # Generate multiple outputs at different pitch shifts
        min_st, max_st = map(int, args.sweep_range.split(','))
        semitones = list(range(min_st, max_st + 1, 2))

        output_dir = os.path.dirname(args.output) or '.'
        base_name = os.path.splitext(os.path.basename(args.output))[0]

        print(f"\nGenerating sweep from {min_st} to {max_st} semitones...")

        for st in semitones:
            print(f"  Generating at {st:+d} semitones...")

            with torch.no_grad():
                result = model.shift_pitch(
                    mel, st, text_cond, text_mask
                )

            prosody_tokens = result['prosody_tokens']
            print(f"    Prosody tokens: {prosody_tokens.shape}")
            print(f"    Original F0 mean: {result['original_f0'][result['original_f0'] > 0].mean():.1f} Hz")
            print(f"    Shifted F0 mean: {result['shifted_f0'][result['shifted_f0'] > 0].mean():.1f} Hz")

            # Save output path for each semitone value
            output_path = os.path.join(output_dir, f"{base_name}_{st:+d}st.wav")
            # In practice, pass prosody_tokens to CSM for audio generation
            print(f"    Output would be saved to: {output_path}")

    elif args.custom_f0:
        # Use custom F0 contour
        print(f"\nUsing custom F0 contour...")
        f0_contour = parse_custom_f0(args.custom_f0).unsqueeze(0).to(device)
        print(f"  F0 values: {args.custom_f0}")

        with torch.no_grad():
            result = model.from_pitch_contour(
                f0_contour, text_cond, text_mask,
                num_steps=args.num_steps
            )

        prosody_tokens = result['prosody_tokens']
        print(f"  Prosody tokens: {prosody_tokens.shape}")

    elif args.shift_semitones != 0:
        # Shift pitch by semitones
        print(f"\nShifting pitch by {args.shift_semitones:+.1f} semitones...")

        with torch.no_grad():
            result = model.shift_pitch(
                mel, args.shift_semitones, text_cond, text_mask
            )

        prosody_tokens = result['prosody_tokens']
        print(f"  Original F0 range: [{result['original_f0'][result['original_f0'] > 0].min():.1f}, "
              f"{result['original_f0'][result['original_f0'] > 0].max():.1f}] Hz")
        print(f"  Shifted F0 range: [{result['shifted_f0'][result['shifted_f0'] > 0].min():.1f}, "
              f"{result['shifted_f0'][result['shifted_f0'] > 0].max():.1f}] Hz")

    elif args.pitch_scale != 1.0:
        # Scale pitch range
        print(f"\nScaling pitch range by {args.pitch_scale}x...")

        with torch.no_grad():
            # Get predicted pitch
            result = model(mel, text_cond=text_cond, text_mask=text_mask)
            f0 = result['pitch_f0']

            # Scale the range
            f0_scaled = scale_pitch_range(f0, args.pitch_scale, args.pitch_center)

            # Generate with scaled pitch
            result = model.from_pitch_contour(f0_scaled, text_cond, text_mask)

        prosody_tokens = result['prosody_tokens']

        voiced = f0 > 0
        if voiced.any():
            original_range = f0[voiced].max() - f0[voiced].min()
            scaled_range = f0_scaled[voiced].max() - f0_scaled[voiced].min()
            print(f"  Original pitch range: {original_range:.1f} Hz")
            print(f"  Scaled pitch range: {scaled_range:.1f} Hz")

    else:
        # Automatic pitch prediction
        print("\nUsing automatic pitch prediction...")

        with torch.no_grad():
            result = model(mel, text_cond=text_cond, text_mask=text_mask)

        prosody_tokens = result['prosody_tokens']
        f0 = result['pitch_f0']

        voiced = f0 > 0
        if voiced.any():
            print(f"  Predicted F0 range: [{f0[voiced].min():.1f}, {f0[voiced].max():.1f}] Hz")
            print(f"  Predicted F0 mean: {f0[voiced].mean():.1f} Hz")

    print(f"\nProsody tokens shape: {prosody_tokens.shape}")
    print(f"Output would be saved to: {args.output}")
    print("\nNote: In full pipeline, pass prosody_tokens to CSM for audio generation:")
    print("  combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)")
    print("  audio = csm_model.generate(input_ids, prosody_prefix=combined_prefix)")

    # Show bin distribution
    if 'pitch_bins' in result:
        bins = result['pitch_bins']
        unvoiced_pct = (bins == 0).float().mean() * 100
        print(f"\nPitch bin statistics:")
        print(f"  Unvoiced frames: {unvoiced_pct:.1f}%")
        print(f"  Bin range: [{bins.min()}, {bins.max()}]")

    print("\n" + "=" * 70)
    print("Inference complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
