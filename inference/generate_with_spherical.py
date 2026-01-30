"""
Inference with Spherical Emotion Vectors (EmoSphere-TTS approach)
with ECE-TTS Emotion-Adaptive Spherical Vectors (EASV) for intensity control

This script demonstrates how to use spherical emotion embeddings for
continuous, intensity-controllable emotion synthesis.

Key features:
1. EASV intensity control: neutral + α * (emotion - neutral)
2. Continuous intensity range: 0.0 (neutral) to 2.0 (exaggerated)
3. Smooth emotion interpolation (LERP/SLERP)
4. Multi-emotion blending with weights
5. VAD-space emotion specification
6. Emotion trajectories for temporal control

EASV Benefits (ECE-TTS 2025):
- intensity=0.0 → neutral emotion (always!)
- intensity=0.5 → weak emotion
- intensity=1.0 → standard emotion
- intensity=1.5 → strong/exaggerated emotion
- Linear scaling enables precise emotion control

Example usage:
    # Single emotion with intensity (EASV enabled by default)
    python generate_with_spherical.py \
        --checkpoint ../models/checkpoints/prosody_v6/best.pt \
        --text "I can't believe this actually worked!" \
        --emotion happy --intensity 0.9 \
        --output happy_speech.wav

    # Exaggerated emotion (intensity > 1.0)
    python generate_with_spherical.py \
        --text "This is AMAZING!" \
        --emotion happy --intensity 1.5 \
        --output exaggerated_happy.wav

    # Emotion interpolation
    python generate_with_spherical.py \
        --text "Things are looking up, but I'm still worried" \
        --interpolate "sad:happy:0.6" --intensity 0.7 \
        --output mixed_speech.wav

    # Multi-emotion blend
    python generate_with_spherical.py \
        --text "This is interesting but also a bit concerning" \
        --blend "surprised:0.4,fearful:0.3,calm:0.3" --intensity 0.8 \
        --output complex_speech.wav

    # Direct VAD specification
    python generate_with_spherical.py \
        --text "Testing direct VAD control" \
        --vad "0.5,0.7,0.3" --intensity 0.8 \
        --output vad_speech.wav

    # Disable EASV (use basic scaling instead)
    python generate_with_spherical.py \
        --text "Test basic scaling" \
        --emotion happy --intensity 0.8 --no-easv \
        --output basic_scaling.wav
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'training'))

from spherical_emotion import (
    SphericalEmotionConfig,
    SphericalEmotionEncoder,
    SphericalEmotionAdapter,
    EmotionInterpolator,
    VAD_PROTOTYPES,
    CORE_EMOTIONS,
    cartesian_to_spherical,
    vad_to_emotion_name,
    easv_scale,  # ECE-TTS EASV functions
)
from prosody_conditioning import ProsodyConfig, ProsodyEncoder


class SphericalEmotionInference:
    """
    Spherical emotion-based speech synthesis.

    Provides continuous, intensity-controllable emotion conditioning
    using VAD (Valence-Arousal-Dominance) space representation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        use_easv: bool = True,  # ECE-TTS EASV formula
    ):
        self.device = torch.device(device)
        self.use_easv = use_easv

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self._load_from_checkpoint()
        else:
            self.checkpoint = None
            self._initialize_default()

        scaling_mode = "EASV" if use_easv else "basic"
        print(f"Spherical Emotion Inference initialized on {self.device} (scaling={scaling_mode})")

    def _initialize_default(self):
        """Initialize with default configurations."""
        self.emotion_config = SphericalEmotionConfig(use_easv=self.use_easv)
        self.prosody_config = ProsodyConfig()

        self.adapter = SphericalEmotionAdapter(
            self.emotion_config,
            prosody_hidden=self.prosody_config.hidden_size,
        ).to(self.device)
        self.adapter.eval()

    def _load_from_checkpoint(self):
        """Load configurations and models from checkpoint."""
        # Load configs
        emotion_cfg = self.checkpoint.get('emotion_config', {})
        self.emotion_config = SphericalEmotionConfig(
            embedding_dim=emotion_cfg.get('embedding_dim', 256),
            hidden_dim=emotion_cfg.get('hidden_dim', 512),
            output_dim=emotion_cfg.get('output_dim', 2048),
            num_prosody_tokens=emotion_cfg.get('num_prosody_tokens', 4),
            use_easv=self.use_easv,  # Use EASV from instance setting
        )

        prosody_cfg = self.checkpoint.get('prosody_config', {})
        self.prosody_config = ProsodyConfig(
            hidden_size=prosody_cfg.get('hidden_size', 2048),
            num_prosody_tokens=prosody_cfg.get('num_prosody_tokens', 4),
        )

        # Initialize adapter
        self.adapter = SphericalEmotionAdapter(
            self.emotion_config,
            prosody_hidden=self.prosody_config.hidden_size,
        ).to(self.device)

        # Load weights if available
        if 'spherical_encoder' in self.checkpoint:
            self.adapter.encoder.load_state_dict(self.checkpoint['spherical_encoder'])
        if 'spherical_classifier' in self.checkpoint:
            self.adapter.classifier.load_state_dict(self.checkpoint['spherical_classifier'])

        self.adapter.eval()

    @torch.no_grad()
    def encode_emotion(
        self,
        emotion: str,
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a single emotion with intensity.

        Args:
            emotion: Emotion name (happy, sad, angry, etc.)
            intensity: Intensity factor α ∈ [0, 1.5]

        Returns:
            Dict with prosody tokens and analysis
        """
        return self.adapter.encode_emotion(emotion, intensity)

    @torch.no_grad()
    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 0.7,
        use_slerp: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotions.

        Args:
            emotion1: Source emotion
            emotion2: Target emotion
            t: Interpolation factor [0, 1]
            intensity: Overall intensity
            use_slerp: Use spherical interpolation (recommended)

        Returns:
            Dict with prosody tokens and analysis
        """
        return self.adapter.interpolate_emotions(
            emotion1, emotion2, t, intensity, use_slerp
        )

    @torch.no_grad()
    def blend_emotions(
        self,
        emotions: List[Tuple[str, float]],
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Blend multiple emotions with weights.

        Args:
            emotions: List of (emotion_name, weight) tuples
            intensity: Overall intensity

        Returns:
            Dict with prosody tokens and analysis
        """
        return self.adapter.blend_emotions(emotions, intensity)

    @torch.no_grad()
    def encode_vad(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode from direct VAD coordinates.

        Args:
            valence: -1 (negative) to +1 (positive)
            arousal: -1 (calm) to +1 (excited)
            dominance: -1 (submissive) to +1 (dominant)
            intensity: Intensity scaling

        Returns:
            Dict with prosody tokens and analysis
        """
        vad = torch.tensor([[valence, arousal, dominance]], dtype=torch.float32)
        vad = vad.to(self.device)
        return self.adapter.forward(vad, intensity)

    @torch.no_grad()
    def create_trajectory(
        self,
        emotions: List[Tuple[str, float]],
        num_segments: int = 4,
        use_slerp: bool = True,
    ) -> torch.Tensor:
        """
        Create emotion trajectory for temporal control.

        Args:
            emotions: List of (emotion_name, duration_ratio) waypoints
            num_segments: Number of output segments
            use_slerp: Use spherical interpolation

        Returns:
            Trajectory tensor [num_segments, 3] of VAD coordinates
        """
        return EmotionInterpolator.create_trajectory(
            emotions,
            self.adapter.encoder,
            num_steps=num_segments,
            use_slerp=use_slerp,
        )

    def get_prosody_tokens(
        self,
        emotion: Optional[str] = None,
        intensity: float = 0.7,
        vad: Optional[Tuple[float, float, float]] = None,
        interpolate: Optional[Tuple[str, str, float]] = None,
        blend: Optional[List[Tuple[str, float]]] = None,
    ) -> torch.Tensor:
        """
        Get prosody tokens for conditioning.

        Supports multiple specification methods (use one):
        - emotion: Single emotion name
        - vad: Direct VAD coordinates (V, A, D)
        - interpolate: (emotion1, emotion2, t) interpolation
        - blend: List of (emotion, weight) for blending

        Args:
            emotion: Emotion name
            intensity: Intensity factor
            vad: Direct VAD tuple
            interpolate: Interpolation tuple
            blend: Blend list

        Returns:
            Prosody tokens [1, num_tokens, hidden]
        """
        if interpolate is not None:
            result = self.interpolate_emotions(
                interpolate[0], interpolate[1], interpolate[2], intensity
            )
        elif blend is not None:
            result = self.blend_emotions(blend, intensity)
        elif vad is not None:
            result = self.encode_vad(vad[0], vad[1], vad[2], intensity)
        elif emotion is not None:
            result = self.encode_emotion(emotion, intensity)
        else:
            result = self.encode_emotion('neutral', intensity)

        return result['prosody_tokens']

    def visualize_emotion(
        self,
        emotion: Optional[str] = None,
        vad: Optional[Tuple[float, float, float]] = None,
        interpolate: Optional[Tuple[str, str, float]] = None,
        blend: Optional[List[Tuple[str, float]]] = None,
        intensity: float = 0.7,
    ):
        """Print visualization of emotion encoding."""
        print("\n" + "=" * 60)
        print("Spherical Emotion Analysis")
        print("=" * 60)

        # Determine the VAD coordinates
        if interpolate:
            vad1 = self.adapter.encoder.get_vad_for_emotion(interpolate[0])
            vad2 = self.adapter.encoder.get_vad_for_emotion(interpolate[1])
            t = interpolate[2]
            vad_interp = EmotionInterpolator.slerp(
                vad1.unsqueeze(0), vad2.unsqueeze(0), t
            ).squeeze(0)
            vad_final = vad_interp * intensity
            print(f"Mode: Interpolation")
            print(f"  From: {interpolate[0]} -> To: {interpolate[1]}")
            print(f"  t = {t:.2f}")
        elif blend:
            vad_blend = EmotionInterpolator.blend_emotions(blend, self.adapter.encoder)
            vad_final = vad_blend * intensity
            print(f"Mode: Blend")
            print(f"  Components: {blend}")
        elif vad:
            vad_final = torch.tensor(vad) * intensity
            print(f"Mode: Direct VAD")
        elif emotion:
            vad_e = self.adapter.encoder.get_vad_for_emotion(emotion)
            vad_final = vad_e * intensity
            print(f"Mode: Single Emotion")
            print(f"  Emotion: {emotion}")
        else:
            vad_final = torch.zeros(3)
            print(f"Mode: Neutral")

        # Display VAD
        v, a, d = vad_final.tolist()
        print(f"\nVAD Coordinates (scaled by intensity {intensity:.2f}):")
        print(f"  Valence (V):   {v:+.3f}  {'😊' if v > 0.3 else '😢' if v < -0.3 else '😐'}")
        print(f"  Arousal (A):   {a:+.3f}  {'⚡' if a > 0.3 else '😴' if a < -0.3 else '➖'}")
        print(f"  Dominance (D): {d:+.3f}  {'👑' if d > 0.3 else '🙈' if d < -0.3 else '➖'}")

        # Spherical coordinates
        vad_tensor = vad_final.unsqueeze(0) if vad_final.dim() == 1 else vad_final
        r, theta, phi = cartesian_to_spherical(vad_tensor)
        print(f"\nSpherical Coordinates:")
        print(f"  Radius (r):    {r.item():.3f}  (emotion intensity)")
        print(f"  Theta (θ):     {theta.item():.3f}  (dominance angle)")
        print(f"  Phi (φ):       {phi.item():.3f}  (valence-arousal angle)")

        # Nearest emotion
        nearest = vad_to_emotion_name(vad_final)
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


def parse_blend(s: str) -> Optional[List[Tuple[str, float]]]:
    """Parse blend string: 'emotion1:weight1,emotion2:weight2'"""
    if not s:
        return None
    result = []
    for part in s.split(','):
        sub = part.strip().split(':')
        if len(sub) >= 2:
            result.append((sub[0], float(sub[1])))
    return result if result else None


def parse_vad(s: str) -> Optional[Tuple[float, float, float]]:
    """Parse VAD string: 'v,a,d'"""
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(',')]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate speech with spherical emotion control'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output_spherical.wav',
                       help='Output audio path')

    # Emotion specification (use one)
    parser.add_argument('--emotion', type=str, default=None,
                       help='Single emotion name')
    parser.add_argument('--intensity', type=float, default=0.7,
                       help='Emotion intensity (0.0-2.0 with EASV)')
    parser.add_argument('--interpolate', type=str, default=None,
                       help='Interpolate: "emotion1:emotion2:t"')
    parser.add_argument('--blend', type=str, default=None,
                       help='Blend: "emotion1:weight1,emotion2:weight2"')
    parser.add_argument('--vad', type=str, default=None,
                       help='Direct VAD: "valence,arousal,dominance"')

    # ECE-TTS EASV options
    parser.add_argument('--use-easv', action='store_true', default=True,
                       help='Use EASV formula (default: enabled)')
    parser.add_argument('--no-easv', action='store_true',
                       help='Disable EASV, use basic intensity scaling')

    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda/mps)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization')
    args = parser.parse_args()

    # Determine EASV mode
    use_easv = not args.no_easv  # EASV is default

    # Initialize inference
    inference = SphericalEmotionInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_easv=use_easv,
    )

    # Parse emotion specifications
    interpolate = parse_interpolate(args.interpolate)
    blend = parse_blend(args.blend)
    vad = parse_vad(args.vad)

    # Visualize
    if not args.no_visualize:
        inference.visualize_emotion(
            emotion=args.emotion,
            vad=vad,
            interpolate=interpolate,
            blend=blend,
            intensity=args.intensity,
        )

    # Get prosody tokens
    prosody_tokens = inference.get_prosody_tokens(
        emotion=args.emotion,
        intensity=args.intensity,
        vad=vad,
        interpolate=interpolate,
        blend=blend,
    )

    print(f"Text: {args.text}")
    print(f"Prosody tokens shape: {prosody_tokens.shape}")

    # Save prosody data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prosody_path = output_path.with_suffix('.prosody.pt')
    save_data = {
        'tokens': prosody_tokens,
        'text': args.text,
        'emotion': args.emotion,
        'intensity': args.intensity,
        'interpolate': args.interpolate,
        'blend': args.blend,
        'vad': args.vad,
    }
    torch.save(save_data, prosody_path)
    print(f"Saved prosody data to: {prosody_path}")

    print("\n[NOTE] This script demonstrates spherical emotion token generation.")
    print("       To generate actual audio, integrate with CSM model:")
    print("""
    from transformers import CsmForConditionalGeneration

    csm = CsmForConditionalGeneration.from_pretrained('sesame/csm-1b')

    # Get text embeddings
    text_embeds = csm.embed_text_tokens(tokenized_text)

    # Prepend emotion tokens
    combined = torch.cat([prosody_tokens, text_embeds], dim=1)

    # Generate audio
    audio = csm.generate(inputs_embeds=combined, ...)
    """)

    # Demo: show all available emotions
    print("\nAvailable emotions and their VAD coordinates:")
    print("-" * 50)
    for emotion in CORE_EMOTIONS:
        v, a, d = VAD_PROTOTYPES[emotion]
        print(f"  {emotion:12s}: V={v:+.2f}, A={a:+.2f}, D={d:+.2f}")


if __name__ == "__main__":
    main()
