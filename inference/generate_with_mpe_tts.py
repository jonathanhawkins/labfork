"""
Inference script for MPE-TTS multi-modal emotion control.

Based on "MPE-TTS: Multi-Modal Prompt Emotion Encoder for Expressive TTS"
Interspeech 2025 - arXiv:2505.18453

Enables emotion control from:
1. Text descriptions ("expressing genuine happiness")
2. Images (facial expressions, emotion images)
3. Reference speech audio

Usage:
    # From text emotion description
    python generate_with_mpe_tts.py \
        --text "Hello, how are you?" \
        --emotion-desc "expressing warm friendliness and genuine interest" \
        --output outputs/friendly_greeting.wav

    # From emotion label
    python generate_with_mpe_tts.py \
        --text "I can't believe this happened!" \
        --emotion happy --intensity 0.9 \
        --output outputs/surprised.wav

    # From facial expression image
    python generate_with_mpe_tts.py \
        --text "Hello, how are you?" \
        --emotion-image face_happy.jpg \
        --output outputs/from_face.wav

    # From reference speech audio
    python generate_with_mpe_tts.py \
        --text "Hello, how are you?" \
        --emotion-audio reference.wav \
        --output outputs/from_audio.wav

    # Emotion interpolation
    python generate_with_mpe_tts.py \
        --text "Things are complicated right now" \
        --interpolate "happy:sad:0.4" \
        --output outputs/mixed.wav

    # Sweep all emotions
    python generate_with_mpe_tts.py \
        --text "Testing variations" \
        --sweep-emotions \
        --output outputs/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torchaudio

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.mpe_tts import (
    MPETTSConfig,
    MPETTSAdapter,
    create_mpetts_adapter,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# EMOTION DESCRIPTIONS
# =============================================================================

EMOTION_DESCRIPTIONS = {
    "neutral": [
        "speaking in a calm, neutral tone",
        "expressing matter-of-fact clarity",
        "conveying composed neutrality",
    ],
    "happy": [
        "expressing genuine happiness and warmth",
        "speaking with joyful enthusiasm",
        "conveying delighted satisfaction",
    ],
    "sad": [
        "expressing deep sadness and melancholy",
        "speaking with sorrowful resignation",
        "conveying heartfelt disappointment",
    ],
    "angry": [
        "expressing intense frustration and anger",
        "speaking with fierce indignation",
        "conveying barely contained rage",
    ],
    "surprised": [
        "expressing sudden astonishment",
        "speaking with shocked disbelief",
        "conveying startled amazement",
    ],
    "fearful": [
        "expressing anxious apprehension",
        "speaking with nervous anticipation",
        "conveying worried uncertainty",
    ],
    "disgusted": [
        "expressing strong disgust and revulsion",
        "speaking with disdainful contempt",
        "conveying visceral repulsion",
    ],
    "contempt": [
        "expressing cold contempt and disdain",
        "speaking with dismissive superiority",
        "conveying patronizing mockery",
    ],
}


def get_emotion_description(emotion: str, variant: int = 0) -> str:
    """Get emotion description for a given emotion label."""
    descriptions = EMOTION_DESCRIPTIONS.get(emotion, EMOTION_DESCRIPTIONS["neutral"])
    return descriptions[variant % len(descriptions)]


# =============================================================================
# INFERENCE
# =============================================================================

class MPETTSGenerator:
    """
    Generator for MPE-TTS multi-modal emotion control.

    Supports:
    - Text emotion descriptions
    - Image emotion (facial expressions)
    - Reference speech audio
    - Emotion interpolation
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[MPETTSConfig] = None,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Load adapter
        self.adapter = create_mpetts_adapter(config, checkpoint_path)
        self.adapter.to(self.device)
        self.adapter.eval()

        self.config = self.adapter.config

        logger.info(f"MPE-TTS generator initialized on {self.device}")

    @torch.no_grad()
    def get_prosody_tokens(
        self,
        emotion_desc: Optional[str] = None,
        emotion_label: Optional[str] = None,
        emotion_image: Optional[Any] = None,
        emotion_audio: Optional[torch.Tensor] = None,
        intensity: float = 0.8,
    ) -> torch.Tensor:
        """
        Get prosody tokens from any modality.

        Args:
            emotion_desc: Natural language emotion description
            emotion_label: Categorical emotion label
            emotion_image: PIL Image for facial expression
            emotion_audio: Audio tensor [samples] at 16kHz
            intensity: Emotion intensity scaling

        Returns:
            prosody_tokens: [1, num_tokens, output_dim]
        """
        if emotion_audio is not None:
            # From reference speech
            audio = emotion_audio.unsqueeze(0).to(self.device)
            result = self.adapter.from_speech(audio, intensity=intensity)
        elif emotion_image is not None:
            # From facial expression image
            result = self.adapter.from_image(emotion_image, intensity=intensity)
        elif emotion_desc is not None:
            # From text description
            result = self.adapter.from_text(emotion_desc, intensity=intensity)
        elif emotion_label is not None:
            # From emotion label
            result = self.adapter.from_emotion_label(emotion_label, intensity=intensity)
        else:
            # Default neutral
            result = self.adapter.from_emotion_label("neutral", intensity=0.5)

        return result['prosody_tokens']

    @torch.no_grad()
    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float = 0.5,
        intensity: float = 0.8,
        method: str = "spherical",
    ) -> torch.Tensor:
        """
        Interpolate between two emotions.

        Args:
            emotion1: First emotion (label or description)
            emotion2: Second emotion (label or description)
            t: Interpolation factor (0.0 = emotion1, 1.0 = emotion2)
            intensity: Emotion intensity
            method: "linear" or "spherical"

        Returns:
            prosody_tokens: [1, num_tokens, output_dim]
        """
        # Convert labels to descriptions if needed
        if emotion1 in EMOTION_DESCRIPTIONS:
            desc1 = get_emotion_description(emotion1)
        else:
            desc1 = emotion1

        if emotion2 in EMOTION_DESCRIPTIONS:
            desc2 = get_emotion_description(emotion2)
        else:
            desc2 = emotion2

        result = self.adapter.interpolate_emotions(desc1, desc2, t=t, method=method)

        return result['prosody_tokens'] * intensity

    def generate_audio(
        self,
        text: str,
        prosody_tokens: torch.Tensor,
        reference_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate audio from text and prosody tokens.

        Note: This is a placeholder. In practice, integrate with full TTS model.

        Args:
            text: Text to synthesize
            prosody_tokens: [1, num_tokens, output_dim] prosody conditioning
            reference_audio: Optional speaker reference audio

        Returns:
            audio: Generated audio tensor
        """
        # Placeholder - in practice, use full MPE-TTS acoustic model
        logger.warning(
            "Full audio generation not implemented. "
            "Prosody tokens ready for CSM integration."
        )

        # Return prosody tokens info
        return prosody_tokens


def load_image(image_path: str):
    """Load image from file."""
    try:
        from PIL import Image
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio from file."""
    try:
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Mono
        waveform = waveform.mean(dim=0)

        return waveform
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return None


def parse_interpolation(interp_str: str) -> Tuple[str, str, float]:
    """
    Parse interpolation string like "happy:sad:0.4".

    Returns:
        (emotion1, emotion2, t)
    """
    parts = interp_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid interpolation format: {interp_str}")

    return parts[0], parts[1], float(parts[2])


def main():
    parser = argparse.ArgumentParser(description="Generate speech with MPE-TTS multi-modal emotion control")

    # Required
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")

    # Emotion sources (mutually exclusive in priority)
    parser.add_argument("--emotion-desc", type=str, help="Natural language emotion description")
    parser.add_argument("--emotion", type=str, help="Categorical emotion label")
    parser.add_argument("--emotion-image", type=str, help="Path to emotion image (facial expression)")
    parser.add_argument("--emotion-audio", type=str, help="Path to reference emotion audio")

    # Emotion parameters
    parser.add_argument("--intensity", type=float, default=0.8, help="Emotion intensity (0.0-1.5)")

    # Interpolation
    parser.add_argument("--interpolate", type=str, help="Interpolate emotions: 'happy:sad:0.4'")
    parser.add_argument("--interpolation-method", type=str, default="spherical",
                        choices=["linear", "spherical"], help="Interpolation method")

    # Sweep
    parser.add_argument("--sweep-emotions", action="store_true", help="Generate all emotions")
    parser.add_argument("--sweep-intensities", action="store_true", help="Sweep intensity levels")

    # Model
    parser.add_argument("--checkpoint", type=str, help="Path to MPE-TTS checkpoint")

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output path (file or directory)")

    args = parser.parse_args()

    # Initialize generator
    generator = MPETTSGenerator(checkpoint_path=args.checkpoint)

    output_path = Path(args.output)

    # Sweep all emotions
    if args.sweep_emotions:
        output_path.mkdir(parents=True, exist_ok=True)

        for emotion in EMOTION_DESCRIPTIONS.keys():
            logger.info(f"Generating with emotion: {emotion}")

            tokens = generator.get_prosody_tokens(
                emotion_label=emotion,
                intensity=args.intensity,
            )

            # Save prosody tokens (placeholder for full generation)
            token_path = output_path / f"{emotion}.pt"
            torch.save({
                'text': args.text,
                'emotion': emotion,
                'intensity': args.intensity,
                'prosody_tokens': tokens,
            }, token_path)
            logger.info(f"Saved: {token_path}")

        logger.info(f"Emotion sweep complete. Saved to {output_path}")
        return

    # Sweep intensities
    if args.sweep_intensities:
        output_path.mkdir(parents=True, exist_ok=True)
        emotion = args.emotion or "happy"

        for intensity in [0.0, 0.3, 0.5, 0.7, 0.9, 1.2]:
            logger.info(f"Generating with intensity: {intensity}")

            tokens = generator.get_prosody_tokens(
                emotion_label=emotion,
                intensity=intensity,
            )

            token_path = output_path / f"{emotion}_intensity{intensity:.1f}.pt"
            torch.save({
                'text': args.text,
                'emotion': emotion,
                'intensity': intensity,
                'prosody_tokens': tokens,
            }, token_path)
            logger.info(f"Saved: {token_path}")

        logger.info(f"Intensity sweep complete. Saved to {output_path}")
        return

    # Single generation
    if args.interpolate:
        # Emotion interpolation
        emotion1, emotion2, t = parse_interpolation(args.interpolate)
        logger.info(f"Interpolating {emotion1} -> {emotion2} (t={t})")

        tokens = generator.interpolate_emotions(
            emotion1, emotion2, t,
            intensity=args.intensity,
            method=args.interpolation_method,
        )
    elif args.emotion_audio:
        # From reference audio
        audio = load_audio(args.emotion_audio)
        if audio is None:
            logger.error("Failed to load emotion audio")
            return

        logger.info(f"Extracting emotion from audio: {args.emotion_audio}")
        tokens = generator.get_prosody_tokens(emotion_audio=audio, intensity=args.intensity)

    elif args.emotion_image:
        # From facial expression image
        image = load_image(args.emotion_image)
        if image is None:
            logger.error("Failed to load emotion image")
            return

        logger.info(f"Extracting emotion from image: {args.emotion_image}")
        tokens = generator.get_prosody_tokens(emotion_image=image, intensity=args.intensity)

    elif args.emotion_desc:
        # From text description
        logger.info(f"Using emotion description: {args.emotion_desc}")
        tokens = generator.get_prosody_tokens(emotion_desc=args.emotion_desc, intensity=args.intensity)

    elif args.emotion:
        # From emotion label
        logger.info(f"Using emotion label: {args.emotion}")
        tokens = generator.get_prosody_tokens(emotion_label=args.emotion, intensity=args.intensity)

    else:
        # Default neutral
        logger.info("Using default neutral emotion")
        tokens = generator.get_prosody_tokens(emotion_label="neutral", intensity=0.5)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'text': args.text,
        'emotion_desc': args.emotion_desc,
        'emotion_label': args.emotion,
        'intensity': args.intensity,
        'prosody_tokens': tokens,
    }

    # Save as .pt file (prosody tokens for CSM integration)
    if output_path.suffix == '.pt':
        torch.save(result, output_path)
    else:
        # Save prosody tokens alongside
        token_path = output_path.with_suffix('.pt')
        torch.save(result, token_path)

    logger.info(f"Prosody tokens saved to: {output_path}")
    logger.info(f"Token shape: {tokens.shape}")
    logger.info(f"\nTo use with CSM:")
    logger.info(f"  prosody_tokens = torch.load('{output_path}')['prosody_tokens']")
    logger.info(f"  output = csm_model(input_ids, prosody_prefix=prosody_tokens)")


if __name__ == "__main__":
    main()
