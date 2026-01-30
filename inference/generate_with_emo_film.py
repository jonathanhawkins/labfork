"""
Inference script for Emo-FiLM word-level emotion generation.

Based on "Beyond Global Emotion" (arXiv:2509.20378).

Usage:
    # Global emotion
    python generate_with_emo_film.py \
        --text "I'm so excited about this!" \
        --emotion happy --intensity 0.9 \
        --output excited.wav

    # Word-level emotion specification
    python generate_with_emo_film.py \
        --text "I was calm but then something surprised me" \
        --word-emotions "calm:0.5,calm:0.5,calm:0.5,calm:0.5,surprised:0.9,surprised:0.9" \
        --output transition.wav

    # Emotion transition (smooth interpolation)
    python generate_with_emo_film.py \
        --text "Things started badly but ended well" \
        --transition sad:happy \
        --output transition.wav

    # Extract emotions from reference audio
    python generate_with_emo_film.py \
        --text "Your text here" \
        --reference reference.wav \
        --output cloned.wav

    # Sweep all emotions
    python generate_with_emo_film.py \
        --text "Hello world" \
        --sweep-emotions \
        --output outputs/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.emo_film import (
    EmoFiLMConfig,
    EmoFiLMAdapter,
    Emotion2VecExtractor,
    FEDDEvaluator,
)


def parse_word_emotions(spec: str) -> Tuple[List[str], List[float]]:
    """
    Parse word emotions specification.

    Format: "emotion1:intensity1,emotion2:intensity2,..."
    Example: "happy:0.8,happy:0.9,sad:0.7"
    """
    emotions = []
    intensities = []

    for item in spec.split(","):
        parts = item.strip().split(":")
        emotion = parts[0].strip()
        intensity = float(parts[1]) if len(parts) > 1 else 1.0

        emotions.append(emotion)
        intensities.append(intensity)

    return emotions, intensities


def parse_transition(spec: str) -> Tuple[str, str]:
    """
    Parse emotion transition specification.

    Format: "start_emotion:end_emotion"
    Example: "sad:happy"
    """
    parts = spec.split(":")
    return parts[0].strip(), parts[1].strip()


class EmoFiLMGenerator:
    """Generator using Emo-FiLM for word-level emotion control."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[EmoFiLMConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = torch.device(device)
        self.config = config or EmoFiLMConfig()

        # Load adapter
        self.adapter = EmoFiLMAdapter(self.config).to(self.device)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")

        self.adapter.eval()

        # Emotion extractor (for reference audio)
        self.emotion_extractor = Emotion2VecExtractor(self.config)

        # Mock text encoder (replace with real one in production)
        self.text_encoder = torch.nn.Sequential(
            torch.nn.Linear(100, self.config.text_hidden_dim),
            torch.nn.GELU(),
        ).to(self.device)

    def tokenize_text(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """
        Tokenize text into words and create embeddings.

        In production, use real text encoder (e.g., from CSM).
        """
        words = text.split()
        num_words = len(words)

        # Mock embeddings
        text_input = torch.randn(1, num_words, 100, device=self.device)
        text_embeddings = self.text_encoder(text_input)

        return text_embeddings, words

    @torch.no_grad()
    def generate_prosody_tokens(
        self,
        text: str,
        emotion: Optional[str] = None,
        intensity: float = 1.0,
        word_emotions: Optional[List[str]] = None,
        word_intensities: Optional[List[float]] = None,
        start_emotion: Optional[str] = None,
        end_emotion: Optional[str] = None,
        reference_audio: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens for text with emotion control.

        Modes:
        1. Global emotion: Same emotion for all words
        2. Word-level: Different emotion per word
        3. Transition: Smooth interpolation between start and end
        4. Reference: Extract from reference audio
        """
        # Tokenize text
        text_embeddings, words = self.tokenize_text(text)
        num_words = len(words)

        # Generate prosody tokens based on mode
        if reference_audio is not None:
            # Extract from reference
            if reference_audio.dim() == 1:
                reference_audio = reference_audio.unsqueeze(0)

            result = self.adapter(
                audio=reference_audio.to(self.device),
                text_embeddings=text_embeddings,
                sample_rate=sample_rate,
            )

        elif word_emotions is not None:
            # Word-level specification
            if len(word_emotions) != num_words:
                # Repeat or truncate to match
                if len(word_emotions) < num_words:
                    word_emotions = word_emotions + [word_emotions[-1]] * (num_words - len(word_emotions))
                else:
                    word_emotions = word_emotions[:num_words]

            if word_intensities is None:
                word_intensities = [intensity] * num_words
            elif len(word_intensities) != num_words:
                if len(word_intensities) < num_words:
                    word_intensities = word_intensities + [word_intensities[-1]] * (num_words - len(word_intensities))
                else:
                    word_intensities = word_intensities[:num_words]

            result = self.adapter.from_emotion_trajectory(
                text_embeddings=text_embeddings,
                word_emotions=word_emotions,
                intensities=word_intensities,
            )

        elif start_emotion is not None and end_emotion is not None:
            # Smooth transition
            result = self.adapter.interpolate_emotions(
                text_embeddings=text_embeddings,
                start_emotion=start_emotion,
                end_emotion=end_emotion,
                start_intensity=intensity,
                end_intensity=intensity,
            )

        else:
            # Global emotion
            emotion = emotion or "neutral"
            result = self.adapter.from_global_emotion(
                text_embeddings=text_embeddings,
                emotion=emotion,
                intensity=intensity,
            )

        result['words'] = words
        result['text'] = text

        return result

    def generate_speech(
        self,
        text: str,
        prosody_tokens: torch.Tensor,
        output_path: str,
    ):
        """
        Generate speech using prosody tokens.

        This is a placeholder - in production, integrate with CSM.
        """
        print(f"Text: {text}")
        print(f"Prosody tokens shape: {prosody_tokens.shape}")
        print(f"Would save to: {output_path}")

        # Mock: save silence
        try:
            import torchaudio

            silence = torch.zeros(1, 24000 * 2)  # 2 seconds
            torchaudio.save(output_path, silence, 24000)
            print(f"Saved mock output to {output_path}")
        except Exception as e:
            print(f"Could not save audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate with Emo-FiLM")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--emotion", type=str, help="Global emotion")
    parser.add_argument("--intensity", type=float, default=0.8, help="Emotion intensity (0-1)")
    parser.add_argument("--word-emotions", type=str, help="Word-level emotions (emotion:intensity,...)")
    parser.add_argument("--transition", type=str, help="Emotion transition (start:end)")
    parser.add_argument("--reference", type=str, help="Reference audio for emotion extraction")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path")
    parser.add_argument("--sweep-emotions", action="store_true", help="Generate all emotions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize generator
    generator = EmoFiLMGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    print("=" * 60)
    print("Emo-FiLM Word-Level Emotion Generation")
    print("=" * 60)
    print(f"Text: {args.text}")

    if args.sweep_emotions:
        # Generate all emotions
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        emotions = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"]

        for emotion in emotions:
            print(f"\nGenerating: {emotion}")

            result = generator.generate_prosody_tokens(
                text=args.text,
                emotion=emotion,
                intensity=args.intensity,
            )

            output_path = output_dir / f"{emotion}_{args.intensity:.1f}.wav"
            generator.generate_speech(args.text, result['prosody_tokens'], str(output_path))

        print(f"\nGenerated {len(emotions)} variations in {output_dir}")

    elif args.word_emotions:
        # Word-level emotions
        word_emotions, word_intensities = parse_word_emotions(args.word_emotions)

        print(f"Word emotions: {word_emotions}")
        print(f"Word intensities: {word_intensities}")

        result = generator.generate_prosody_tokens(
            text=args.text,
            word_emotions=word_emotions,
            word_intensities=word_intensities,
        )

        generator.generate_speech(args.text, result['prosody_tokens'], args.output)

    elif args.transition:
        # Emotion transition
        start_emotion, end_emotion = parse_transition(args.transition)

        print(f"Transition: {start_emotion} -> {end_emotion}")

        result = generator.generate_prosody_tokens(
            text=args.text,
            start_emotion=start_emotion,
            end_emotion=end_emotion,
            intensity=args.intensity,
        )

        generator.generate_speech(args.text, result['prosody_tokens'], args.output)

    elif args.reference:
        # Reference-based
        import torchaudio

        print(f"Reference: {args.reference}")

        waveform, sr = torchaudio.load(args.reference)
        waveform = waveform.mean(dim=0)  # Mono

        result = generator.generate_prosody_tokens(
            text=args.text,
            reference_audio=waveform,
            sample_rate=sr,
        )

        generator.generate_speech(args.text, result['prosody_tokens'], args.output)

    else:
        # Global emotion
        emotion = args.emotion or "neutral"
        print(f"Emotion: {emotion} (intensity: {args.intensity})")

        result = generator.generate_prosody_tokens(
            text=args.text,
            emotion=emotion,
            intensity=args.intensity,
        )

        generator.generate_speech(args.text, result['prosody_tokens'], args.output)

    # Show prosody info
    print(f"\nProsody tokens: {result['prosody_tokens'].shape}")
    print(f"Words: {result['words']}")

    if result.get('word_emotions') is not None:
        print(f"Word emotions shape: {result['word_emotions'].shape}")


if __name__ == "__main__":
    main()
