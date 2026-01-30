#!/usr/bin/env python3
"""
Generate speech with mismatch-aware CFG emotion control.

This script extends activation steering with automatic intensity adjustment
based on semantic alignment between emotion and content.

Features:
- Dynamic guidance: Strong guidance when emotion matches content, weak when mismatched
- No manual tuning: Automatically adapts intensity to content
- VAD support: Use VAD coordinates with mismatch awareness
- Analysis mode: Inspect alignment scores before generation

Usage:
    # Generate with automatic intensity adjustment
    python generate_with_mismatch_aware.py \\
        --text "I'm so excited about this amazing news!" \\
        --emotion happy --base-intensity 0.8 \\
        --output outputs/happy_auto.wav

    # Compare with mismatched content
    python generate_with_mismatch_aware.py \\
        --text "This is terrible news, I'm devastated." \\
        --emotion happy --base-intensity 0.8 \\
        --output outputs/happy_mismatch.wav

    # Analyze alignment without generating
    python generate_with_mismatch_aware.py \\
        --text "I'm furious about this outrage!" \\
        --emotion angry --analyze-only

    # VAD-based steering with mismatch awareness
    python generate_with_mismatch_aware.py \\
        --text "What wonderful news!" \\
        --vad 0.8,0.6,0.4 --base-intensity 0.8 \\
        --output outputs/vad_happy.wav

    # Compare all emotions on same text
    python generate_with_mismatch_aware.py \\
        --text "Hello, how are you today?" \\
        --sweep-emotions \\
        --output outputs/

References:
    - EmoSteer-TTS (arXiv:2508.03543): Activation steering for emotion control
    - Mismatch-Aware Guidance (arXiv:2510.13293): Dynamic CFG based on semantic alignment
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from mismatch_aware_cfg import (
    MismatchAwareCFGConfig,
    SemanticMismatchDiscriminator,
    MismatchAwareActivationSteering,
    SphericalMismatchAwareSteering,
    create_mismatch_aware_steerer,
    analyze_alignment,
)
from activation_steering import (
    SteeringConfig,
    SteeringVectorExtractor,
    ActivationSteering,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load TTS model from checkpoint.

    This is a placeholder - replace with your actual model loading code.
    """
    print(f"Loading model from {checkpoint_path}...")

    # TODO: Replace with actual model loading
    # from csm_model import CsmForConditionalGeneration
    # model = CsmForConditionalGeneration.from_pretrained(checkpoint_path)

    # For now, create a dummy model for demonstration
    class DummyTTSModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
                for _ in range(12)
            ])
            # Add self_attn attribute for steering to find layers
            for layer in self.layers:
                layer.self_attn = layer.self_attn

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def generate(self, text: str, **kwargs):
            """Generate audio from text."""
            # Placeholder: return dummy audio
            return torch.randn(1, 24000)  # 1 second at 24kHz

    model = DummyTTSModel()
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def load_steering_vectors(path: str) -> Tuple[Dict, SteeringConfig]:
    """Load pre-extracted steering vectors."""
    print(f"Loading steering vectors from {path}...")
    data = torch.load(path, weights_only=False)
    return data["steering_vectors"], data.get("config", SteeringConfig())


def create_dummy_steering_vectors(
    hidden_dim: int = 512,
    seq_len: int = 100,
    target_layers: List[int] = None,
    emotions: List[str] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Create dummy steering vectors for testing."""
    if target_layers is None:
        target_layers = [1, 4, 7, 10]
    if emotions is None:
        emotions = ["happy", "sad", "angry", "fearful", "surprised", "calm", "neutral"]

    vectors = {}
    for emotion in emotions:
        vectors[emotion] = {
            layer: torch.randn(seq_len, hidden_dim)
            for layer in target_layers
        }
    return vectors


def save_audio(audio: torch.Tensor, path: str, sample_rate: int = 24000):
    """Save audio tensor to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    try:
        torchaudio.save(str(path), audio, sample_rate)
    except ImportError:
        # Fallback to scipy if torchcodec not available
        import scipy.io.wavfile as wav
        import numpy as np
        audio_np = audio.squeeze().cpu().numpy()
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        wav.write(str(path), sample_rate, audio_np)
    print(f"Saved audio to {path}")


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_with_mismatch_aware(
    model,
    steerer: MismatchAwareActivationSteering,
    text: str,
    emotion: str,
    base_intensity: float = 0.8,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate audio with mismatch-aware emotion steering.

    Args:
        model: TTS model
        steerer: Mismatch-aware activation steerer
        text: Input text
        emotion: Target emotion
        base_intensity: Base steering intensity before alignment scaling
        device: Computation device

    Returns:
        (audio, info) tuple with audio tensor and steering info
    """
    # Get steering info
    info = steerer.steer_with_info(emotion, base_intensity=base_intensity, text=text)
    print(f"  Emotion: {emotion}")
    print(f"  Text: {text[:50]}...")
    print(f"  Alignment: {info['alignment']:.3f}")
    print(f"  Base intensity: {info['base_intensity']:.3f}")
    print(f"  Dynamic intensity: {info['dynamic_intensity']:.3f}")
    print(f"  Intensity ratio: {info['intensity_ratio']:.3f}")

    # Generate with adaptive steering
    with steerer.steer_adaptive(emotion, base_intensity=base_intensity, text=text):
        audio = model.generate(text)

    return audio, info


def generate_with_vad_mismatch_aware(
    model,
    spherical_steerer: SphericalMismatchAwareSteering,
    text: str,
    vad: Tuple[float, float, float],
    base_intensity: float = 0.8,
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate audio with VAD-based mismatch-aware steering.

    Args:
        model: TTS model
        spherical_steerer: Spherical mismatch-aware steerer
        text: Input text
        vad: (valence, arousal, dominance) tuple
        base_intensity: Base steering intensity

    Returns:
        (audio, info) tuple
    """
    valence, arousal, dominance = vad

    print(f"  VAD: V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}")
    print(f"  Text: {text[:50]}...")

    with spherical_steerer.steer_vad_adaptive(
        valence, arousal, dominance, text=text, base_intensity=base_intensity
    ):
        audio = model.generate(text)
        alignment = spherical_steerer.mismatch_steerer.current_alignment
        intensity = spherical_steerer.mismatch_steerer.current_dynamic_intensity

    info = {
        "vad": vad,
        "alignment": alignment,
        "dynamic_intensity": intensity,
    }
    print(f"  Alignment: {alignment:.3f}")
    print(f"  Dynamic intensity: {intensity:.3f}")

    return audio, info


def analyze_text_emotions(
    discriminator: SemanticMismatchDiscriminator,
    text: str,
    emotions: List[str] = None,
) -> Dict[str, float]:
    """
    Analyze alignment scores for text against multiple emotions.

    Args:
        discriminator: Mismatch discriminator
        text: Input text
        emotions: List of emotions to analyze

    Returns:
        Dict mapping emotion to alignment score
    """
    if emotions is None:
        emotions = ["happy", "sad", "angry", "fearful", "surprised", "calm", "neutral"]

    alignments = {}
    for emotion in emotions:
        alignment = discriminator.compute_alignment_from_text(emotion, text)
        alignments[emotion] = alignment

    return alignments


def sweep_emotions(
    model,
    steerer: MismatchAwareActivationSteering,
    text: str,
    base_intensity: float = 0.8,
    output_dir: str = "outputs",
) -> List[Dict]:
    """
    Generate audio for all emotions and compare alignments.

    Args:
        model: TTS model
        steerer: Mismatch-aware steerer
        text: Input text
        base_intensity: Base intensity
        output_dir: Output directory

    Returns:
        List of results per emotion
    """
    emotions = steerer.list_emotions()
    results = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSweeping {len(emotions)} emotions on text: '{text[:50]}...'")
    print("-" * 60)

    for emotion in emotions:
        print(f"\n[{emotion}]")
        audio, info = generate_with_mismatch_aware(
            model, steerer, text, emotion, base_intensity
        )

        # Save audio
        output_path = output_dir / f"{emotion}_{info['alignment']:.2f}.wav"
        save_audio(audio, output_path)

        results.append({
            "emotion": emotion,
            "alignment": info["alignment"],
            "dynamic_intensity": info["dynamic_intensity"],
            "output_path": str(output_path),
        })

    # Sort by alignment
    results_sorted = sorted(results, key=lambda x: x["alignment"], reverse=True)

    print("\n" + "=" * 60)
    print("Results ranked by alignment:")
    print("-" * 60)
    for r in results_sorted:
        print(f"  {r['emotion']:12} align={r['alignment']:.3f} α={r['dynamic_intensity']:.3f}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with mismatch-aware CFG emotion control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("--text", "-t", type=str, required=True,
                        help="Input text to synthesize")

    # Emotion control
    parser.add_argument("--emotion", "-e", type=str,
                        help="Target emotion (happy, sad, angry, etc.)")
    parser.add_argument("--vad", type=str,
                        help="VAD coordinates as 'valence,arousal,dominance'")
    parser.add_argument("--blend", type=str,
                        help="Emotion blend as 'emo1:w1,emo2:w2'")

    # Intensity control
    parser.add_argument("--base-intensity", type=float, default=0.8,
                        help="Base steering intensity (default: 0.8)")
    parser.add_argument("--fixed-intensity", action="store_true",
                        help="Disable mismatch-aware adjustment, use fixed intensity")

    # Output
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                        help="Output path for audio")

    # Model
    parser.add_argument("--checkpoint", type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--steering-vectors", type=str,
                        help="Path to pre-extracted steering vectors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")

    # Analysis modes
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze alignment, don't generate")
    parser.add_argument("--sweep-emotions", action="store_true",
                        help="Generate all emotions and compare alignments")

    # Configuration
    parser.add_argument("--min-scale", type=float, default=0.1,
                        help="Minimum guidance scale for mismatched cases")
    parser.add_argument("--max-scale", type=float, default=2.0,
                        help="Maximum guidance scale for matched cases")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Run with dummy model for testing")

    args = parser.parse_args()

    # Validate arguments
    if not args.analyze_only and not args.sweep_emotions:
        if args.emotion is None and args.vad is None and args.blend is None:
            parser.error("Must specify --emotion, --vad, or --blend")

    print("=" * 60)
    print("Mismatch-Aware CFG Emotion Generation")
    print("=" * 60)

    # Create configuration
    config = MismatchAwareCFGConfig(
        min_guidance_scale=args.min_scale,
        max_guidance_scale=args.max_scale,
    )

    # Load or create model
    if args.test:
        print("\n[Test mode] Creating dummy model and steering vectors...")
        model = load_model("dummy", device=args.device)
        steering_vectors = create_dummy_steering_vectors()
        steering_config = SteeringConfig()
    else:
        if args.checkpoint:
            model = load_model(args.checkpoint, device=args.device)
        else:
            print("ERROR: Must specify --checkpoint or use --test mode")
            return 1

        if args.steering_vectors:
            steering_vectors, steering_config = load_steering_vectors(args.steering_vectors)
        else:
            print("ERROR: Must specify --steering-vectors or use --test mode")
            return 1

    # Create discriminator and steerer
    print("\nCreating mismatch-aware steerer...")
    discriminator = SemanticMismatchDiscriminator(config)
    steerer = MismatchAwareActivationSteering(
        model, steering_vectors, discriminator, config, steering_config
    )

    # Analysis mode
    if args.analyze_only:
        print("\n[Analysis Mode]")
        print(f"Text: {args.text}")
        print("-" * 40)

        alignments = analyze_text_emotions(discriminator, args.text)
        sorted_alignments = sorted(alignments.items(), key=lambda x: x[1], reverse=True)

        print("\nAlignment scores:")
        for emotion, score in sorted_alignments:
            bar = "#" * int(score * 20)
            print(f"  {emotion:12} {score:.3f} |{bar}")

        if args.emotion:
            info = steerer.steer_with_info(args.emotion, args.base_intensity, text=args.text)
            print(f"\nFor emotion '{args.emotion}':")
            print(f"  Alignment: {info['alignment']:.3f}")
            print(f"  Base intensity: {info['base_intensity']:.3f}")
            print(f"  Dynamic intensity: {info['dynamic_intensity']:.3f}")

        return 0

    # Sweep emotions mode
    if args.sweep_emotions:
        results = sweep_emotions(
            model, steerer, args.text, args.base_intensity, args.output
        )

        # Save results JSON
        results_path = Path(args.output) / "results.json"
        with open(results_path, "w") as f:
            json.dump({"text": args.text, "results": results}, f, indent=2)
        print(f"\nResults saved to {results_path}")
        return 0

    # Standard generation
    print("\n[Generation Mode]")

    if args.vad:
        # VAD-based generation
        vad = tuple(map(float, args.vad.split(",")))
        if len(vad) != 3:
            print("ERROR: VAD must be 3 values: valence,arousal,dominance")
            return 1

        spherical = SphericalMismatchAwareSteering(steerer)
        audio, info = generate_with_vad_mismatch_aware(
            model, spherical, args.text, vad, args.base_intensity
        )

    elif args.blend:
        # Blended emotions
        blend = {}
        for part in args.blend.split(","):
            emo, weight = part.split(":")
            blend[emo] = float(weight)

        print(f"  Blend: {blend}")
        with steerer.steer_adaptive(
            list(blend.keys())[0],  # Primary emotion for alignment
            blend=blend,
            text=args.text,
            base_intensity=args.base_intensity,
        ):
            audio = model.generate(args.text)
            info = {
                "blend": blend,
                "alignment": steerer.current_alignment,
                "dynamic_intensity": steerer.current_dynamic_intensity,
            }
        print(f"  Alignment: {info['alignment']:.3f}")
        print(f"  Dynamic intensity: {info['dynamic_intensity']:.3f}")

    else:
        # Standard emotion
        if args.fixed_intensity:
            # Use base steering without mismatch adjustment
            with steerer.base_steerer.steer(args.emotion, intensity=args.base_intensity):
                audio = model.generate(args.text)
            info = {"emotion": args.emotion, "intensity": args.base_intensity, "fixed": True}
            print(f"  [Fixed intensity mode] α={args.base_intensity}")
        else:
            audio, info = generate_with_mismatch_aware(
                model, steerer, args.text, args.emotion, args.base_intensity
            )

    # Save audio
    save_audio(audio, args.output)

    # Save metadata
    meta_path = Path(args.output).with_suffix(".json")
    metadata = {
        "text": args.text,
        **info,
        "base_intensity": args.base_intensity,
        "config": {
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
