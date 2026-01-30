"""
Inference with ECE-TTS EASV (Emotion-Adaptive Spherical Vectors)

This script demonstrates how to use EASV for continuous intensity-controlled
emotion synthesis using the ECE-TTS arithmetic formulation:

    emb_intensity = emb_neutral + α * (emb_emotion - emb_neutral)

Key Features:
1. Continuous intensity control (α from 0.0 to 2.0)
2. Emotion exaggeration (α > 1.0)
3. Emotion suppression (α < 1.0)
4. Emotion interpolation with SLERP
5. Direct VAD control

SUCCESS CRITERIA Verification:
- Intensity=0.5 should produce weaker emotion than intensity=1.0
- Intensity=1.5 should produce stronger/exaggerated emotion
- Happy pitch contour at intensity=1.5 > intensity=1.0 > intensity=0.5

Example usage:
    # Basic intensity control
    python generate_with_easv.py \\
        --text "I can't believe this actually worked!" \\
        --emotion happy --intensity 1.0 \\
        --output happy_normal.wav

    # Exaggerated emotion
    python generate_with_easv.py \\
        --text "I'm absolutely thrilled about this!" \\
        --emotion happy --intensity 1.5 \\
        --output happy_exaggerated.wav

    # Intensity sweep for verification
    python generate_with_easv.py \\
        --text "This is wonderful news" \\
        --emotion happy --sweep \\
        --output happy_sweep

    # Emotion interpolation
    python generate_with_easv.py \\
        --text "I'm not sure how to feel" \\
        --interpolate "sad:happy:0.5" --intensity 1.0 \\
        --output mixed_emotion.wav
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))

from easv_intensity import (
    EASVConfig,
    EASVProsodyAdapter,
    EASVIntensityControl,
    create_easv_adapter,
    verify_intensity_monotonicity,
)
from spherical_emotion import (
    VAD_PROTOTYPES,
    CORE_EMOTIONS,
    cartesian_to_spherical,
    vad_to_emotion_name,
    EmotionInterpolator,
)


def intensity_description(intensity: float) -> str:
    """Get human-readable description of intensity."""
    if intensity <= 0.0:
        return "neutral (no emotion)"
    elif intensity < 0.5:
        return "very weak emotion"
    elif intensity < 1.0:
        return f"suppressed emotion ({intensity*100:.0f}%)"
    elif intensity == 1.0:
        return "normal emotion (training level)"
    elif intensity < 1.5:
        return f"enhanced emotion ({intensity*100:.0f}%)"
    elif intensity <= 2.0:
        return f"exaggerated emotion ({intensity*100:.0f}%)"
    else:
        return f"extreme exaggeration ({intensity*100:.0f}%)"


class EASVInference:
    """
    EASV-based speech synthesis with intensity control.

    Provides:
    1. Emotion encoding with continuous intensity
    2. Emotion exaggeration/suppression
    3. Smooth emotion interpolation
    4. Intensity sweep for verification
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
    ):
        self.device = torch.device(device)

        # Load configuration and adapter
        if checkpoint_path and Path(checkpoint_path).exists():
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self._load_from_checkpoint()
        else:
            self.checkpoint = None
            self._initialize_default()

        print(f"EASV Inference initialized on {self.device}")

    def _initialize_default(self):
        """Initialize with default configuration."""
        self.config = EASVConfig()
        self.adapter = create_easv_adapter(self.config).to(self.device)
        self.adapter.eval()
        print(f"  Intensity range: [{self.config.min_intensity}, {self.config.max_intensity}]")
        print(f"  Shift centers: {'enabled' if self.config.use_shift_centers else 'disabled'}")
        print(f"  IQR normalization: {'enabled' if self.config.use_iqr_normalization else 'disabled'}")

    def _load_from_checkpoint(self):
        """Load from checkpoint."""
        config_dict = self.checkpoint.get('easv_config', {})
        self.config = EASVConfig(
            embedding_dim=config_dict.get('embedding_dim', 256),
            hidden_dim=config_dict.get('hidden_dim', 512),
            output_dim=config_dict.get('output_dim', 2048),
            num_prosody_tokens=config_dict.get('num_prosody_tokens', 4),
        )

        self.adapter = create_easv_adapter(self.config).to(self.device)

        if 'easv_adapter' in self.checkpoint:
            self.adapter.load_state_dict(self.checkpoint['easv_adapter'])

        self.adapter.eval()

    @torch.no_grad()
    def encode_emotion(
        self,
        emotion: str,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion with EASV intensity control.

        Args:
            emotion: Emotion name
            intensity: Intensity α (0.0=neutral, 1.0=full, >1.0=exaggerated)

        Returns:
            Dict with prosody tokens and analysis
        """
        return self.adapter.encode_emotion(emotion, intensity)

    @torch.no_grad()
    def intensity_sweep(
        self,
        emotion: str,
        intensities: List[float] = None,
    ) -> Dict[str, List]:
        """
        Generate at multiple intensities for verification.

        SUCCESS CRITERIA:
        - Intensity=0.5 < Intensity=1.0 < Intensity=1.5 in effect

        Args:
            emotion: Target emotion
            intensities: List of α values

        Returns:
            Dict with results per intensity
        """
        if intensities is None:
            intensities = [0.5, 1.0, 1.5]

        return self.adapter.intensity_sweep(emotion, intensities)

    @torch.no_grad()
    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotions.

        Args:
            emotion1: Source emotion
            emotion2: Target emotion
            t: Interpolation factor [0, 1]
            intensity: Overall intensity

        Returns:
            Dict with prosody tokens
        """
        return self.adapter.interpolate_emotions(emotion1, emotion2, t, intensity)

    @torch.no_grad()
    def encode_vad(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode from direct VAD coordinates.

        Args:
            valence: -1 (negative) to +1 (positive)
            arousal: -1 (calm) to +1 (excited)
            dominance: -1 (submissive) to +1 (dominant)
            intensity: Intensity scaling

        Returns:
            Dict with prosody tokens
        """
        vad = torch.tensor([[valence, arousal, dominance]], dtype=torch.float32)
        vad = vad.to(self.device)
        return self.adapter.encode_vad(vad, intensity)

    def get_prosody_tokens(
        self,
        emotion: Optional[str] = None,
        intensity: float = 1.0,
        vad: Optional[Tuple[float, float, float]] = None,
        interpolate: Optional[Tuple[str, str, float]] = None,
    ) -> torch.Tensor:
        """
        Get prosody tokens for conditioning.

        Args:
            emotion: Emotion name
            intensity: Intensity α
            vad: Direct VAD coordinates
            interpolate: (emotion1, emotion2, t) for interpolation

        Returns:
            Prosody tokens [1, num_tokens, hidden]
        """
        if interpolate is not None:
            result = self.interpolate_emotions(
                interpolate[0], interpolate[1], interpolate[2], intensity
            )
        elif vad is not None:
            result = self.encode_vad(vad[0], vad[1], vad[2], intensity)
        elif emotion is not None:
            result = self.encode_emotion(emotion, intensity)
        else:
            result = self.encode_emotion("neutral", 0.0)

        return result["prosody_tokens"]

    def visualize_emotion(
        self,
        emotion: Optional[str] = None,
        vad: Optional[Tuple[float, float, float]] = None,
        interpolate: Optional[Tuple[str, str, float]] = None,
        intensity: float = 1.0,
    ):
        """Print visualization of EASV encoding."""
        print("\n" + "=" * 60)
        print("ECE-TTS EASV (Emotion-Adaptive Spherical Vectors)")
        print("=" * 60)

        # EASV formula explanation
        print("\nEASV Formula:")
        print("  emb_out = emb_neutral + α * (emb_emotion - emb_neutral)")
        print(f"  α (intensity) = {intensity}")
        print(f"  Intensity description: {intensity_description(intensity)}")

        # Determine VAD
        if interpolate:
            vad1 = torch.tensor(VAD_PROTOTYPES[interpolate[0].lower()])
            vad2 = torch.tensor(VAD_PROTOTYPES[interpolate[1].lower()])
            t = interpolate[2]
            vad_interp = EmotionInterpolator.slerp(
                vad1.unsqueeze(0), vad2.unsqueeze(0), t
            ).squeeze(0)
            vad_final = vad_interp
            print(f"\nMode: Interpolation")
            print(f"  From: {interpolate[0]} → To: {interpolate[1]}")
            print(f"  t = {t:.2f}")
        elif vad:
            vad_final = torch.tensor(vad)
            print(f"\nMode: Direct VAD")
        elif emotion:
            vad_final = torch.tensor(VAD_PROTOTYPES[emotion.lower()])
            print(f"\nMode: Single Emotion")
            print(f"  Emotion: {emotion}")
        else:
            vad_final = torch.zeros(3)
            print(f"\nMode: Neutral")

        # Display VAD
        v, a, d = vad_final.tolist()
        print(f"\nVAD Coordinates (unscaled):")
        print(f"  Valence (V):   {v:+.3f}  {'😊' if v > 0.3 else '😢' if v < -0.3 else '😐'}")
        print(f"  Arousal (A):   {a:+.3f}  {'⚡' if a > 0.3 else '😴' if a < -0.3 else '➖'}")
        print(f"  Dominance (D): {d:+.3f}  {'👑' if d > 0.3 else '🙈' if d < -0.3 else '➖'}")

        # Spherical coordinates
        vad_tensor = vad_final.unsqueeze(0) * intensity
        r, theta, phi = cartesian_to_spherical(vad_tensor)
        print(f"\nSpherical Coordinates (intensity-scaled):")
        print(f"  Radius (r):    {r.item():.3f}  (effective intensity)")
        print(f"  Theta (θ):     {theta.item():.3f}  (dominance angle)")
        print(f"  Phi (φ):       {phi.item():.3f}  (valence-arousal angle)")

        # Intensity effect
        print(f"\nIntensity Effect (α = {intensity}):")
        if intensity < 1.0:
            print(f"  → Emotion SUPPRESSED to {intensity*100:.0f}%")
        elif intensity > 1.0:
            print(f"  → Emotion EXAGGERATED to {intensity*100:.0f}%")
        else:
            print(f"  → Full emotion (training level)")

        # Nearest emotion
        scaled_vad = vad_final * intensity
        nearest = vad_to_emotion_name(scaled_vad)
        print(f"\nNearest prototype: {nearest}")

        print("=" * 60 + "\n")


def parse_interpolate(s: str) -> Optional[Tuple[str, str, float]]:
    """Parse interpolate string: 'emotion1:emotion2:t'"""
    if not s:
        return None
    parts = s.split(':')
    if len(parts) >= 3:
        return (parts[0], parts[1], float(parts[2]))
    return None


def parse_vad(s: str) -> Optional[Tuple[float, float, float]]:
    """Parse VAD string: 'v,a,d'"""
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(',')]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return None


def run_verification(inference: EASVInference, emotion: str = "happy"):
    """
    Run SUCCESS CRITERIA verification.

    Verifies:
    - Intensity=0.5 produces weaker effect than intensity=1.0
    - Intensity=1.5 produces stronger/exaggerated effect
    """
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA VERIFICATION")
    print("=" * 60)
    print(f"\nEmotion: {emotion}")
    print("Testing: α=0.5 < α=1.0 < α=1.5\n")

    # Get neutral reference
    neutral_result = inference.encode_emotion("neutral", intensity=0.0)
    neutral_emb = neutral_result["embedding"]

    # Test intensities
    intensities = [0.5, 1.0, 1.5]
    results = inference.intensity_sweep(emotion, intensities)

    print("Results:")
    print("-" * 40)

    distances = []
    for i, α in enumerate(intensities):
        emb = results["embeddings"][i]
        dist = (emb - neutral_emb).norm().item()
        distances.append(dist)
        print(f"  α={α:.1f}: distance from neutral = {dist:.4f}")

    print("-" * 40)

    # Verify monotonicity
    criteria_met = distances[0] < distances[1] < distances[2]
    print(f"\nMonotonic increase: {'✓ YES' if criteria_met else '✗ NO'}")

    if criteria_met:
        print("\n✓ SUCCESS CRITERIA MET")
        print("  • α=0.5 produces WEAKER emotion than α=1.0")
        print("  • α=1.5 produces STRONGER emotion than α=1.0")
    else:
        print("\n✗ SUCCESS CRITERIA NOT MET")
        print("  Check model training or configuration")

    # Ratio analysis
    ratio_05_10 = distances[1] / (distances[0] + 1e-8)
    ratio_15_10 = distances[2] / (distances[1] + 1e-8)
    print(f"\nRatios:")
    print(f"  dist(α=1.0) / dist(α=0.5) = {ratio_05_10:.2f}x")
    print(f"  dist(α=1.5) / dist(α=1.0) = {ratio_15_10:.2f}x")

    print("=" * 60 + "\n")

    return criteria_met


def main():
    parser = argparse.ArgumentParser(
        description='Generate speech with ECE-TTS EASV intensity control'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=False,
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output_easv.wav',
                       help='Output path')

    # Emotion specification
    parser.add_argument('--emotion', type=str, default=None,
                       help='Emotion name (happy, sad, angry, etc.)')
    parser.add_argument('--intensity', type=float, default=1.0,
                       help='Intensity α (0.0-2.0)')
    parser.add_argument('--interpolate', type=str, default=None,
                       help='Interpolate: "emotion1:emotion2:t"')
    parser.add_argument('--vad', type=str, default=None,
                       help='Direct VAD: "valence,arousal,dominance"')

    # Verification modes
    parser.add_argument('--sweep', action='store_true',
                       help='Run intensity sweep')
    parser.add_argument('--verify', action='store_true',
                       help='Run SUCCESS CRITERIA verification')

    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda/mps)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization')

    args = parser.parse_args()

    # Initialize inference
    inference = EASVInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Run verification if requested
    if args.verify:
        emotion = args.emotion or "happy"
        run_verification(inference, emotion)
        return

    # Run intensity sweep if requested
    if args.sweep:
        emotion = args.emotion or "happy"
        print(f"\n[Intensity Sweep for {emotion}]")
        print("-" * 40)

        intensities = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        results = inference.intensity_sweep(emotion, intensities)

        for i, α in enumerate(intensities):
            pred_int = results["predicted_intensities"][i].item()
            token_norm = results["tokens"][i].norm().item()
            print(f"  α={α:.2f}: predicted={pred_int:.3f}, token_norm={token_norm:.4f}")

        print("-" * 40)
        return

    # Parse emotion specifications
    interpolate = parse_interpolate(args.interpolate)
    vad = parse_vad(args.vad)

    # Visualize
    if not args.no_visualize:
        inference.visualize_emotion(
            emotion=args.emotion,
            vad=vad,
            interpolate=interpolate,
            intensity=args.intensity,
        )

    # Get prosody tokens
    prosody_tokens = inference.get_prosody_tokens(
        emotion=args.emotion,
        intensity=args.intensity,
        vad=vad,
        interpolate=interpolate,
    )

    if args.text:
        print(f"Text: {args.text}")
    print(f"Prosody tokens shape: {prosody_tokens.shape}")

    # Save prosody data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prosody_path = output_path.with_suffix('.easv.pt')
    save_data = {
        'tokens': prosody_tokens,
        'text': args.text,
        'emotion': args.emotion,
        'intensity': args.intensity,
        'interpolate': args.interpolate,
        'vad': args.vad,
        'method': 'ECE-TTS EASV',
    }
    torch.save(save_data, prosody_path)
    print(f"Saved EASV prosody data to: {prosody_path}")

    print("\n[NOTE] This demonstrates EASV token generation.")
    print("       Integrate with CSM model for audio synthesis.")

    # Show available emotions
    print("\nAvailable emotions:")
    print("-" * 40)
    for emotion in CORE_EMOTIONS:
        v, a, d = VAD_PROTOTYPES[emotion]
        print(f"  {emotion:12s}: V={v:+.2f}, A={a:+.2f}, D={d:+.2f}")


if __name__ == "__main__":
    main()
