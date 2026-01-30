#!/usr/bin/env python3
"""
Inference script for IndexTTS2: 8-Dimensional Emotion Vector Control

Based on arXiv:2506.21619 - "IndexTTS 2: Controllable Emotional Text-to-Speech"

Usage:
    # Single emotion
    python generate_with_indextts2.py \
        --text "Hello, how are you?" \
        --emotion happy --intensity 0.8 \
        --output outputs/happy.wav

    # 8-dimensional vector control
    python generate_with_indextts2.py \
        --text "This is amazing!" \
        --vector "0.8,0.0,0.0,0.0,0.0,0.0,0.2,0.3" \
        --output outputs/custom.wav

    # Multiple emotions (blended)
    python generate_with_indextts2.py \
        --text "I have mixed feelings" \
        --emotions "happy:0.5,sad:0.3,calm:0.2" \
        --output outputs/mixed.wav

    # Duration control (for video dubbing)
    python generate_with_indextts2.py \
        --text "Timed speech" \
        --emotion calm \
        --duration 150 \
        --output outputs/timed.wav

    # Sweep all emotions
    python generate_with_indextts2.py \
        --text "Testing emotion variations" \
        --sweep-emotions \
        --output outputs/

    # Intensity sweep
    python generate_with_indextts2.py \
        --text "Varying intensity" \
        --emotion angry \
        --sweep-intensity \
        --output outputs/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from indextts2 import (
    IndexTTS2Config,
    IndexTTS2,
    IndexTTS2Adapter,
    create_emotion_vector,
    emotion_vector_to_description,
    EMOTION_LABELS,
    EMOTION_TO_IDX,
    EMOTION_PROFILES,
)


def parse_vector_string(vector_str: str) -> List[float]:
    """Parse comma-separated vector string."""
    values = [float(v.strip()) for v in vector_str.split(",")]
    if len(values) != 8:
        raise ValueError(f"Vector must have 8 values, got {len(values)}")
    return values


def parse_emotions_string(emotions_str: str) -> Dict[str, float]:
    """Parse emotion:intensity pairs."""
    emotions = {}
    for pair in emotions_str.split(","):
        if ":" in pair:
            emotion, intensity = pair.split(":")
            emotions[emotion.strip().lower()] = float(intensity.strip())
        else:
            emotions[pair.strip().lower()] = 0.8  # Default intensity
    return emotions


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with IndexTTS2 8-dimensional emotion control"
    )

    # Input options
    parser.add_argument("--text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--reference", type=str, default=None,
                       help="Reference audio for timbre (optional)")

    # Emotion options (mutually exclusive)
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument("--emotion", type=str, default=None,
                              help=f"Single emotion: {', '.join(EMOTION_LABELS)}")
    emotion_group.add_argument("--vector", type=str, default=None,
                              help="8-dim vector: 'v1,v2,v3,v4,v5,v6,v7,v8'")
    emotion_group.add_argument("--emotions", type=str, default=None,
                              help="Multiple: 'happy:0.5,sad:0.3'")
    emotion_group.add_argument("--profile", type=str, default=None,
                              choices=list(EMOTION_PROFILES.keys()),
                              help="Predefined emotion profile")

    # Intensity
    parser.add_argument("--intensity", type=float, default=0.8,
                       help="Overall emotion intensity (0-1)")

    # Duration control
    parser.add_argument("--duration", type=int, default=None,
                       help="Target duration in tokens (for video dubbing)")
    parser.add_argument("--duration-scale", type=float, default=1.0,
                       help="Scale predicted duration")

    # Generation options
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling threshold")

    # Sweep options
    parser.add_argument("--sweep-emotions", action="store_true",
                       help="Generate all 8 emotions")
    parser.add_argument("--sweep-intensity", action="store_true",
                       help="Sweep intensity from 0.2 to 1.0")
    parser.add_argument("--sweep-blend", type=str, default=None,
                       help="Sweep blend: 'happy,sad' for happy-sad interpolation")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")

    # Output
    parser.add_argument("--output", type=str, required=True,
                       help="Output audio path or directory (for sweeps)")

    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Create config and adapter
    config = IndexTTS2Config()
    adapter = IndexTTS2Adapter(config).to(device)

    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Note: Adapter only loads emotion encoder, not full model
        if "model_state_dict" in checkpoint:
            # Filter for adapter-relevant keys
            adapter_keys = {k: v for k, v in checkpoint["model_state_dict"].items()
                          if k.startswith("emotion_encoder")}
            adapter.emotion_encoder.load_state_dict(adapter_keys, strict=False)
            print("  Loaded emotion encoder weights")

    adapter.eval()

    # Handle sweeps
    if args.sweep_emotions:
        print("\n=== Emotion Sweep ===")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for emotion in EMOTION_LABELS:
            print(f"\nGenerating: {emotion}")
            result = adapter.from_profile(emotion, intensity=args.intensity)

            prosody_tokens = result["prosody_tokens"]
            vector = result["emotion_vector"]
            desc = emotion_vector_to_description(vector)

            print(f"  Description: {desc}")
            print(f"  Tokens shape: {prosody_tokens.shape}")

            # Save metadata
            output_path = output_dir / f"{emotion}.json"
            with open(output_path, "w") as f:
                json.dump({
                    "emotion": emotion,
                    "intensity": args.intensity,
                    "vector": vector.cpu().tolist(),
                    "description": desc,
                    "text": args.text,
                }, f, indent=2)

        print(f"\nSaved to: {output_dir}")
        return

    if args.sweep_intensity:
        print("\n=== Intensity Sweep ===")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        emotion = args.emotion or "happy"
        intensities = [0.2, 0.4, 0.6, 0.8, 1.0]

        for intensity in intensities:
            print(f"\nGenerating: {emotion} @ {intensity}")
            result = adapter.from_profile(emotion, intensity=intensity)

            prosody_tokens = result["prosody_tokens"]
            vector = result["emotion_vector"]
            desc = emotion_vector_to_description(vector)

            print(f"  Description: {desc}")

            # Save metadata
            output_path = output_dir / f"{emotion}_intensity_{intensity:.1f}.json"
            with open(output_path, "w") as f:
                json.dump({
                    "emotion": emotion,
                    "intensity": intensity,
                    "vector": vector.cpu().tolist(),
                    "description": desc,
                    "text": args.text,
                }, f, indent=2)

        print(f"\nSaved to: {output_dir}")
        return

    if args.sweep_blend:
        print("\n=== Blend Sweep ===")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        emotions = args.sweep_blend.split(",")
        if len(emotions) != 2:
            print("Error: --sweep-blend requires exactly 2 emotions")
            sys.exit(1)

        emotion1, emotion2 = emotions[0].strip(), emotions[1].strip()

        # Get base vectors
        vector1 = create_emotion_vector(profile=emotion1, device=torch.device(device))
        vector2 = create_emotion_vector(profile=emotion2, device=torch.device(device))

        blend_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for t in blend_values:
            print(f"\nGenerating: {emotion1}->{emotion2} @ t={t}")
            result = adapter.interpolate(
                vector1.tolist(),
                vector2.tolist(),
                t=t,
            )

            prosody_tokens = result["prosody_tokens"]
            vector = result["emotion_vector"]
            desc = emotion_vector_to_description(vector)

            print(f"  Description: {desc}")

            # Save metadata
            output_path = output_dir / f"blend_{emotion1}_{emotion2}_t{t:.2f}.json"
            with open(output_path, "w") as f:
                json.dump({
                    "emotion1": emotion1,
                    "emotion2": emotion2,
                    "blend_t": t,
                    "vector": vector.cpu().tolist(),
                    "description": desc,
                    "text": args.text,
                }, f, indent=2)

        print(f"\nSaved to: {output_dir}")
        return

    # Single generation
    print("\n=== Single Generation ===")
    print(f"Text: {args.text}")

    # Determine emotion vector
    with torch.no_grad():
        if args.vector:
            # Direct vector input
            vector = parse_vector_string(args.vector)
            print(f"Using 8-dim vector: {vector}")
            result = adapter.from_emotion_vector(vector)

        elif args.emotions:
            # Multiple emotions
            emotions = parse_emotions_string(args.emotions)
            print(f"Using emotions: {emotions}")
            result = adapter.from_emotions(**emotions)

        elif args.profile:
            # Predefined profile
            print(f"Using profile: {args.profile} @ intensity={args.intensity}")
            result = adapter.from_profile(args.profile, intensity=args.intensity)

        elif args.emotion:
            # Single emotion
            print(f"Using emotion: {args.emotion} @ intensity={args.intensity}")
            result = adapter.from_profile(args.emotion, intensity=args.intensity)

        else:
            # Default to calm/neutral
            print("Using default: calm")
            result = adapter.from_profile("calm", intensity=0.5)

    prosody_tokens = result["prosody_tokens"]
    emotion_vector = result["emotion_vector"]
    dominant = EMOTION_LABELS[result["dominant_emotion"].item()]
    description = emotion_vector_to_description(emotion_vector)

    print(f"\nResults:")
    print(f"  Prosody tokens shape: {prosody_tokens.shape}")
    print(f"  Emotion vector: {emotion_vector.cpu().tolist()}")
    print(f"  Dominant emotion: {dominant}")
    print(f"  Description: {description}")

    if args.duration:
        print(f"  Target duration: {args.duration} tokens")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata (actual audio generation would require full model)
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "text": args.text,
            "emotion_vector": emotion_vector.cpu().tolist(),
            "dominant_emotion": dominant,
            "description": description,
            "intensity": args.intensity,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "target_duration": args.duration,
            "prosody_tokens_shape": list(prosody_tokens.shape),
        }, f, indent=2)

    print(f"\nSaved metadata to: {meta_path}")

    # Note about full generation
    print("\nNote: Full audio generation requires:")
    print("  1. Trained IndexTTS2 model checkpoint")
    print("  2. Reference audio for timbre extraction")
    print("  3. Integration with audio decoder (e.g., HiFi-GAN)")
    print("\nThe prosody tokens can be used with ProsodyControlledCSM:")
    print("  combined_prefix = torch.cat([prosody_prefix, prosody_tokens], dim=1)")
    print("  output = csm_model(input_ids, prosody_prefix=combined_prefix)")


if __name__ == "__main__":
    main()
