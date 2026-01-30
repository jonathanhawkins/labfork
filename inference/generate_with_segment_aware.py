#!/usr/bin/env python3
"""
Inference script for Segment-Aware Conditioning.

Training-free intra-utterance emotion control based on arXiv:2601.03170.

Usage:
    # Basic intra-utterance emotion control
    python generate_with_segment_aware.py \
        --text "I was sad at first, but then something amazing happened!" \
        --segments "I was sad at first,|sad:0.7" "but then something amazing happened!|happy:0.9" \
        --reference speaker.wav \
        --output emotional.wav

    # Using inline markup
    python generate_with_segment_aware.py \
        --markup "I was [sad:0.7]feeling down[/sad] but then [happy:0.9]everything changed[/happy]!" \
        --reference speaker.wav \
        --output emotional.wav

    # Emotion trajectory (time-based)
    python generate_with_segment_aware.py \
        --text "A long sentence with emotional progression" \
        --trajectory "0.0-0.3:sad:0.7,0.3-0.7:neutral:0.5,0.7-1.0:happy:0.9" \
        --reference speaker.wav \
        --output trajectory.wav

    # Sweep emotions for comparison
    python generate_with_segment_aware.py \
        --text "The same text with different emotions" \
        --sweep-emotions happy,sad,angry \
        --reference speaker.wav \
        --output outputs/

    # Analyze segments without generation
    python generate_with_segment_aware.py \
        --markup "I was [sad]feeling down[/sad]..." \
        --analyze-only
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.segment_aware_conditioning import (
    SegmentAwareConfig,
    SegmentAwareAdapter,
    SegmentAwareInferenceWrapper,
    EmotionSegment,
    create_emotion_segments,
    parse_emotion_markup,
    estimate_segment_durations,
    visualize_emotion_trajectory,
    create_segment_aware_adapter,
)


def parse_segment_arg(segment_arg: str) -> Tuple[str, str, float]:
    """
    Parse a segment argument.

    Format: "text|emotion:intensity" or "text|emotion"

    Examples:
        "Hello world|happy:0.8"
        "Goodbye|sad"
    """
    parts = segment_arg.split("|")
    if len(parts) != 2:
        raise ValueError(f"Invalid segment format: {segment_arg}. Use 'text|emotion:intensity'")

    text = parts[0]
    emotion_part = parts[1]

    if ":" in emotion_part:
        emotion, intensity_str = emotion_part.split(":")
        intensity = float(intensity_str)
    else:
        emotion = emotion_part
        intensity = 1.0

    return text, emotion, intensity


def parse_trajectory_arg(trajectory_arg: str) -> List[Tuple[float, float, str, float]]:
    """
    Parse a trajectory argument.

    Format: "start-end:emotion:intensity,start-end:emotion:intensity,..."

    Example:
        "0.0-0.3:sad:0.7,0.3-0.7:neutral:0.5,0.7-1.0:happy:0.9"
    """
    segments = []

    for part in trajectory_arg.split(","):
        parts = part.strip().split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid trajectory format: {part}")

        time_range = parts[0]
        emotion = parts[1]
        intensity = float(parts[2]) if len(parts) > 2 else 1.0

        start, end = time_range.split("-")
        segments.append((float(start), float(end), emotion, intensity))

    return segments


def load_model(checkpoint: Optional[str] = None):
    """
    Load TTS model for generation.

    Falls back to mock model for testing if checkpoint not provided.
    """
    if checkpoint and Path(checkpoint).exists():
        # Try to load CSM or similar model
        try:
            from csm.api import CSM

            model = CSM.from_pretrained(checkpoint)
            return model
        except ImportError:
            pass

        # Try loading state dict directly
        try:
            state_dict = torch.load(checkpoint, map_location='cpu')
            # Would need model class to instantiate
            raise NotImplementedError("Direct checkpoint loading not implemented")
        except Exception as e:
            warnings.warn(f"Could not load checkpoint: {e}")

    # Return mock model for testing
    class MockModel:
        def __init__(self):
            self.sample_rate = 24000
            self.device = torch.device('cpu')

        def parameters(self):
            return iter([torch.tensor([0.0])])

        def generate(self, text: str, **kwargs) -> torch.Tensor:
            """Generate mock audio."""
            # Create simple sine wave based on text length
            duration = max(1.0, len(text) * 0.1)
            samples = int(duration * self.sample_rate)
            t = torch.linspace(0, duration, samples)
            audio = 0.5 * torch.sin(2 * 3.14159 * 440 * t)
            return audio

    warnings.warn("Using mock model for generation (no checkpoint provided)")
    return MockModel()


def generate_with_segments(
    model,
    text: str,
    segments: List[EmotionSegment],
    reference_audio: Optional[torch.Tensor] = None,
    config: SegmentAwareConfig = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Generate audio with segment-aware emotion conditioning.

    Args:
        model: TTS model
        text: Full text to synthesize
        segments: Emotion segments
        reference_audio: Optional reference for voice cloning
        config: Segment-aware config
        device: Target device

    Returns:
        Generated audio waveform
    """
    if config is None:
        config = SegmentAwareConfig()

    # Create wrapper
    wrapper = SegmentAwareInferenceWrapper(model, config)

    # Generate
    audio = wrapper.generate(
        text=text,
        segments=segments,
        reference_audio=reference_audio,
    )

    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with intra-utterance emotion control"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text", type=str,
        help="Full text to synthesize (use with --segments)"
    )
    input_group.add_argument(
        "--markup", type=str,
        help="Text with inline emotion markup"
    )

    # Segment specification
    parser.add_argument(
        "--segments", type=str, nargs="+",
        help="Segments in format 'text|emotion:intensity'"
    )
    parser.add_argument(
        "--trajectory", type=str,
        help="Emotion trajectory: 'start-end:emotion:intensity,...'"
    )

    # Reference and output
    parser.add_argument(
        "--reference", type=str,
        help="Reference audio for voice cloning"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output.wav",
        help="Output audio path"
    )

    # Model
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to TTS model checkpoint"
    )

    # Generation options
    parser.add_argument(
        "--transition-type", type=str, default="sigmoid",
        choices=["sigmoid", "linear", "cosine", "step"],
        help="Transition type between emotions"
    )
    parser.add_argument(
        "--transition-duration", type=int, default=10,
        help="Transition duration in frames"
    )
    parser.add_argument(
        "--mask-softness", type=float, default=0.1,
        help="Mask transition softness (0-1)"
    )

    # Analysis options
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Analyze segments without generating audio"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize emotion trajectory"
    )
    parser.add_argument(
        "--plot-output", type=str,
        help="Save trajectory plot to file"
    )

    # Sweep options
    parser.add_argument(
        "--sweep-emotions", type=str,
        help="Comma-separated emotions to sweep"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.text and not args.markup:
        parser.error("Either --text or --markup is required")

    if args.text and not args.segments and not args.trajectory:
        parser.error("--text requires either --segments or --trajectory")

    # Parse input
    if args.markup:
        text, segments = parse_emotion_markup(args.markup)
        print(f"Parsed markup:")
        print(f"  Clean text: {text}")
    elif args.trajectory:
        text = args.text
        traj = parse_trajectory_arg(args.trajectory)
        segments = []
        for start, end, emotion, intensity in traj:
            # Create segment with estimated text portion
            total_len = len(text)
            seg_text = text[int(start * total_len):int(end * total_len)]
            segments.append(EmotionSegment(
                text=seg_text,
                emotion=emotion,
                intensity=intensity,
            ))
    else:
        text = args.text
        segment_texts = []
        emotions = []
        intensities = []

        for seg_arg in args.segments:
            seg_text, emotion, intensity = parse_segment_arg(seg_arg)
            segment_texts.append(seg_text)
            emotions.append(emotion)
            intensities.append(intensity)

        segments = create_emotion_segments(text, segment_texts, emotions, intensities)

    # Print segment info
    print(f"\nSegments ({len(segments)}):")
    for i, seg in enumerate(segments):
        print(f"  {i+1}. [{seg.emotion}:{seg.intensity:.2f}] \"{seg.text}\"")

    if args.analyze_only:
        print("\nAnalysis complete (--analyze-only mode)")
        return

    # Create config
    config = SegmentAwareConfig(
        transition_type=args.transition_type,
        transition_duration_frames=args.transition_duration,
        mask_softness=args.mask_softness,
    )

    # Load model
    model = load_model(args.checkpoint)

    # Load reference audio if provided
    reference_audio = None
    if args.reference:
        reference_path = Path(args.reference)
        if reference_path.exists():
            waveform, sr = torchaudio.load(reference_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            reference_audio = waveform
            print(f"\nLoaded reference: {reference_path}")

    # Handle sweep mode
    if args.sweep_emotions:
        sweep_emotions = args.sweep_emotions.split(",")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSweeping emotions: {sweep_emotions}")

        for emotion in sweep_emotions:
            # Create uniform emotion segments
            uniform_segments = [
                EmotionSegment(text=seg.text, emotion=emotion.strip(), intensity=seg.intensity)
                for seg in segments
            ]

            audio = generate_with_segments(
                model, text, uniform_segments, reference_audio, config, args.device
            )

            output_path = output_dir / f"{emotion.strip()}.wav"
            torchaudio.save(str(output_path), audio.unsqueeze(0), model.sample_rate)
            print(f"  Saved: {output_path}")

        return

    # Generate single output
    print(f"\nGenerating with segment-aware conditioning...")

    audio = generate_with_segments(
        model, text, segments, reference_audio, config, args.device
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio.unsqueeze(0), model.sample_rate)
    print(f"\nSaved: {output_path}")

    # Visualize if requested
    if args.visualize or args.plot_output:
        print("\nGenerating visualization...")

        # Create adapter to get emotion weights
        adapter = SegmentAwareAdapter(config, input_dim=768)
        dummy_features = torch.randn(1, 100, 768)
        result = adapter(dummy_features, segments, return_all=True)

        if 'emotion_weights' in result:
            visualize_emotion_trajectory(
                result['emotion_weights'],
                [0, 30, 70, 100],  # Example boundaries
                config.emotion_labels,
                args.plot_output,
            )

            if args.plot_output:
                print(f"Saved plot: {args.plot_output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
