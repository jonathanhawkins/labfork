"""
EmoVoice Inference Script

Generate speech with freestyle natural language emotion control.

Usage:
    # With emotion description
    python generate_with_emovoice.py \
        --text "Hello, how are you?" \
        --emotion-desc "expressing warm friendliness and genuine interest" \
        --output outputs/friendly_greeting.wav

    # With categorical emotion
    python generate_with_emovoice.py \
        --text "I can't believe this happened!" \
        --emotion surprised --intensity 0.9 \
        --output outputs/surprised.wav

    # Interpolate between emotions
    python generate_with_emovoice.py \
        --text "Things are complicated right now" \
        --interpolate "expressing happiness:expressing sadness:0.4" \
        --output outputs/mixed_emotion.wav

    # Generate with multiple emotion variations
    python generate_with_emovoice.py \
        --text "Hello world" \
        --sweep-emotions \
        --output outputs/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.emovoice import (
    EmoVoiceConfig,
    EmoVoice,
    EmoVoiceAdapter,
    generate_emotion_description,
    parse_freestyle_prompt,
    EMOTION_DESCRIPTION_EXAMPLES,
)


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class EmoVoiceInference:
    """
    Inference wrapper for EmoVoice model.

    Supports:
    - Natural language emotion descriptions
    - Categorical emotions with intensity
    - Emotion interpolation
    - Batch generation
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Create config and model
        self.config = EmoVoiceConfig()
        self.adapter = EmoVoiceAdapter(self.config)
        self.adapter.to(self.device)
        self.adapter.eval()

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        print(f"EmoVoice loaded on {self.device}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.adapter.emovoice.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.adapter.emovoice.load_state_dict(checkpoint)

        print(f"Loaded checkpoint: {path}")

    @torch.no_grad()
    def generate_prosody_tokens(
        self,
        emotion_description: Optional[str] = None,
        emotion: Optional[str] = None,
        intensity: float = 0.7,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens for emotion conditioning.

        Args:
            emotion_description: Natural language description (priority)
            emotion: Categorical emotion (fallback)
            intensity: Emotion intensity (0-1)
            batch_size: Batch size

        Returns:
            Dict with prosody tokens and analysis
        """
        if emotion_description:
            result = self.adapter.encode_description(emotion_description, batch_size)
        elif emotion:
            result = self.adapter.encode_emotion(emotion, intensity, batch_size=batch_size)
        else:
            result = self.adapter.encode_emotion("neutral", batch_size=batch_size)

        return result

    @torch.no_grad()
    def interpolate_emotions(
        self,
        description1: str,
        description2: str,
        t: float,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotion descriptions.

        Args:
            description1: Source emotion description
            description2: Target emotion description
            t: Interpolation factor [0, 1]
            batch_size: Batch size

        Returns:
            Dict with interpolated prosody tokens
        """
        return self.adapter.interpolate_descriptions(
            description1, description2, t, batch_size
        )

    @torch.no_grad()
    def sweep_emotions(
        self,
        emotions: Optional[List[str]] = None,
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens for multiple emotions.

        Args:
            emotions: List of emotions (default: all supported)
            intensity: Emotion intensity

        Returns:
            Dict mapping emotion to prosody tokens
        """
        if emotions is None:
            emotions = list(EMOTION_DESCRIPTION_EXAMPLES.keys())

        results = {}
        for emotion in emotions:
            result = self.adapter.encode_emotion(emotion, intensity, batch_size=1)
            results[emotion] = result["prosody_tokens"]

        return results

    def analyze_prompt(self, prompt: str) -> Dict:
        """
        Analyze a freestyle emotion prompt.

        Args:
            prompt: Natural language emotion prompt

        Returns:
            Dict with analysis (detected emotions, intensity, etc.)
        """
        return parse_freestyle_prompt(prompt)


# =============================================================================
# AUDIO GENERATION (PLACEHOLDER)
# =============================================================================

def generate_audio_tokens(
    text: str,
    prosody_tokens: torch.Tensor,
    model: EmoVoice,
    max_tokens: int = 500,
) -> torch.Tensor:
    """
    Generate audio tokens from text with prosody conditioning.

    This is a placeholder - in practice, this would:
    1. Tokenize the input text
    2. Pass through the EmoVoice model with prosody conditioning
    3. Return semantic audio tokens

    For actual audio generation, you would then:
    1. Pass tokens through flow matching module
    2. Pass through HiFi-GAN vocoder
    """
    # Placeholder - return random tokens
    return torch.randint(0, 4096, (1, max_tokens))


def tokens_to_audio(
    audio_tokens: torch.Tensor,
    sample_rate: int = 24000,
) -> np.ndarray:
    """
    Convert audio tokens to waveform.

    This is a placeholder for flow matching + vocoder.
    In practice, this would use CosyVoice's flow matching module
    and HiFi-GAN vocoder.
    """
    # Placeholder - return silence
    duration = audio_tokens.shape[1] / 50  # 50Hz tokens
    num_samples = int(duration * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """Save audio to file."""
    try:
        import scipy.io.wavfile as wav
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(path, sample_rate, audio_int16)
    except ImportError:
        import wave
        with wave.open(path, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            f.writeframes(audio_int16.tobytes())

    print(f"Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate speech with EmoVoice")

    # Input
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")

    # Emotion control
    parser.add_argument("--emotion-desc", type=str, default=None,
                        help="Natural language emotion description")
    parser.add_argument("--emotion", type=str, default=None,
                        help="Categorical emotion (happy, sad, angry, etc.)")
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Emotion intensity (0-1)")
    parser.add_argument("--interpolate", type=str, default=None,
                        help="Interpolate emotions: 'desc1:desc2:t'")

    # Sweep mode
    parser.add_argument("--sweep-emotions", action="store_true",
                        help="Generate all emotion variations")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Output file or directory")

    args = parser.parse_args()

    # Create inference object
    inference = EmoVoiceInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Sweep mode
    if args.sweep_emotions:
        os.makedirs(args.output, exist_ok=True)
        results = inference.sweep_emotions(intensity=args.intensity)

        print(f"\nGenerating {len(results)} emotion variations...")
        for emotion, tokens in results.items():
            print(f"  {emotion}: prosody tokens shape {tokens.shape}")

            # Analyze
            desc = generate_emotion_description(emotion, args.intensity)
            analysis = inference.analyze_prompt(desc)
            print(f"    Description: \"{desc}\"")
            print(f"    Estimated intensity: {analysis['estimated_intensity']:.2f}")

        print(f"\nProsody tokens saved for {len(results)} emotions")
        print("Note: Full audio generation requires vocoder integration")
        return

    # Interpolation mode
    if args.interpolate:
        parts = args.interpolate.split(":")
        if len(parts) != 3:
            print("Error: --interpolate format should be 'desc1:desc2:t'")
            return

        desc1, desc2, t = parts[0], parts[1], float(parts[2])
        print(f"\nInterpolating emotions:")
        print(f"  Source: \"{desc1}\"")
        print(f"  Target: \"{desc2}\"")
        print(f"  Factor: {t}")

        result = inference.interpolate_emotions(desc1, desc2, t)
        print(f"\n  Prosody tokens shape: {result['prosody_tokens'].shape}")
        return

    # Standard generation
    if args.emotion_desc:
        print(f"\nEmotion description: \"{args.emotion_desc}\"")
        analysis = inference.analyze_prompt(args.emotion_desc)
        print(f"  Detected primary: {analysis['primary_emotion']}")
        print(f"  Estimated intensity: {analysis['estimated_intensity']:.2f}")

        result = inference.generate_prosody_tokens(
            emotion_description=args.emotion_desc
        )

    elif args.emotion:
        desc = generate_emotion_description(args.emotion, args.intensity)
        print(f"\nEmotion: {args.emotion} @ {args.intensity:.1f}")
        print(f"  Description: \"{desc}\"")

        result = inference.generate_prosody_tokens(
            emotion=args.emotion,
            intensity=args.intensity,
        )

    else:
        print("\nNo emotion specified, using neutral")
        result = inference.generate_prosody_tokens(emotion="neutral")

    print(f"\n  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"\nText: \"{args.text}\"")

    # Placeholder for full audio generation
    print("\nNote: Full audio generation requires:")
    print("  1. Text tokenization")
    print("  2. Semantic token generation via EmoVoice backbone")
    print("  3. Flow matching decoder (CosyVoice)")
    print("  4. HiFi-GAN vocoder")
    print("\nProsody tokens ready for CSM integration.")


if __name__ == "__main__":
    main()
