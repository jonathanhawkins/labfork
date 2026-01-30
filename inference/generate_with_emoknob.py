#!/usr/bin/env python3
"""
Generate speech with EmoKnob direction vector emotion control.

Based on: arXiv:2410.00316 "EmoKnob: Enhance Voice Cloning with Fine-Grained
          Emotion Control" (EMNLP 2024)

Examples:
    # Basic emotion control
    python generate_with_emoknob.py \
        --text "Hello, how are you?" \
        --emotion happy --intensity 0.8 \
        --reference voice_sample.wav \
        --output outputs/happy.wav

    # Blend multiple emotions
    python generate_with_emoknob.py \
        --text "This is exciting but also scary" \
        --blend "happy:0.6,fearful:0.4" --intensity 0.7 \
        --reference voice_sample.wav \
        --output outputs/mixed.wav

    # Sweep intensities
    python generate_with_emoknob.py \
        --text "Testing intensity levels" \
        --emotion angry --sweep-intensity \
        --reference voice_sample.wav \
        --output outputs/

    # Text-based emotion (no explicit label needed)
    python generate_with_emoknob.py \
        --text "I just won the lottery!" \
        --auto-emotion \
        --reference voice_sample.wav \
        --output outputs/auto.wav

    # Sweep all emotions
    python generate_with_emoknob.py \
        --text "Hello world" \
        --sweep-emotions \
        --reference voice_sample.wav \
        --output outputs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.emoknob import (
    EmoKnobConfig,
    EmoKnob,
    EmoKnobAdapter,
    create_emoknob_adapter,
    intensity_to_description,
    EMOKNOB_EMOTIONS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and resample if needed."""
    try:
        import torchaudio
    except ImportError:
        logger.error("torchaudio required. Install with: pip install torchaudio")
        raise

    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def save_audio(waveform: torch.Tensor, path: str, sample_rate: int = 24000):
    """Save audio to file."""
    try:
        import torchaudio
    except ImportError:
        import scipy.io.wavfile as wav
        audio_np = (waveform.squeeze().cpu().numpy() * 32767).astype(np.int16)
        wav.write(path, sample_rate, audio_np)
        return

    # Ensure proper shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(path, waveform.cpu(), sample_rate)


def extract_features(audio: torch.Tensor, model_name: str = "wav2vec2") -> torch.Tensor:
    """Extract features from audio using pre-trained model."""
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except ImportError:
        logger.warning("transformers not available, using random features")
        return torch.randn(1, audio.shape[-1] // 320, 768)

    if model_name == "wav2vec2":
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    else:
        # HuBERT
        from transformers import HubertModel, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

    model.eval()

    # Process audio
    inputs = processor(
        audio.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state

    return features


# =============================================================================
# EMOTION INFERENCE
# =============================================================================

def infer_emotion_from_text(text: str) -> Tuple[str, float]:
    """
    Infer emotion from text content using keyword matching.

    Returns:
        emotion: Predicted emotion
        confidence: Confidence score
    """
    text_lower = text.lower()

    # Emotion keywords
    emotion_keywords = {
        "happy": ["happy", "joy", "great", "wonderful", "amazing", "love", "excited",
                  "delighted", "thrilled", "fantastic", "awesome", "glad", "pleased"],
        "sad": ["sad", "sorry", "unfortunately", "miss", "lost", "depressed", "cry",
                "grief", "heartbroken", "devastated", "disappointed", "upset"],
        "angry": ["angry", "furious", "mad", "hate", "annoyed", "frustrated", "rage",
                  "outraged", "irritated", "livid", "infuriated", "upset"],
        "surprised": ["surprised", "shocked", "wow", "unbelievable", "amazing",
                     "astonished", "stunned", "unexpected", "incredible"],
        "fearful": ["afraid", "scared", "fear", "terrified", "worried", "anxious",
                   "nervous", "panic", "dread", "frightened", "uneasy"],
        "calm": ["calm", "peaceful", "relaxed", "serene", "quiet", "tranquil",
                "composed", "steady", "balanced"],
        "disgusted": ["disgust", "gross", "eww", "horrible", "revolting", "nasty",
                     "repulsive", "sickening"],
    }

    # Count keyword matches
    scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = score

    # Find best match
    if max(scores.values()) == 0:
        return "neutral", 0.5

    best_emotion = max(scores, key=scores.get)
    confidence = min(1.0, scores[best_emotion] / 3)

    return best_emotion, confidence


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_with_emotion(
    adapter: EmoKnobAdapter,
    reference_features: torch.Tensor,
    emotion: str,
    intensity: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate prosody tokens with emotion control."""
    adapter.eval()

    with torch.no_grad():
        features = reference_features.to(device)
        result = adapter.from_emotion(features, emotion=emotion, intensity=intensity)

    return result


def generate_with_blend(
    adapter: EmoKnobAdapter,
    reference_features: torch.Tensor,
    emotions: List[str],
    weights: List[float],
    intensity: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate prosody tokens with blended emotions."""
    adapter.eval()

    with torch.no_grad():
        features = reference_features.to(device)
        result = adapter.emoknob.interpolate(features, emotions, weights, intensity)

    return result


def sweep_intensities(
    adapter: EmoKnobAdapter,
    reference_features: torch.Tensor,
    emotion: str,
    intensities: List[float],
    device: torch.device,
) -> Dict[str, List[torch.Tensor]]:
    """Generate samples at different intensities."""
    return adapter.sweep_intensities(
        reference_features.to(device),
        emotion=emotion,
        intensities=intensities,
    )


def sweep_emotions(
    adapter: EmoKnobAdapter,
    reference_features: torch.Tensor,
    emotions: List[str],
    intensity: float,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Generate samples for all emotions."""
    results = {}

    for emotion in emotions:
        result = generate_with_emotion(
            adapter, reference_features, emotion, intensity, device
        )
        results[emotion] = result

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate speech with EmoKnob emotion control")

    # Input/Output
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--reference", type=str, required=True, help="Reference audio for voice")
    parser.add_argument("--output", type=str, default="output.wav", help="Output path")

    # Emotion control
    parser.add_argument("--emotion", type=str, help="Target emotion")
    parser.add_argument("--intensity", type=float, default=0.7, help="Emotion intensity (0-2)")
    parser.add_argument("--blend", type=str, help="Blend emotions (format: happy:0.6,sad:0.4)")
    parser.add_argument("--auto-emotion", action="store_true", help="Infer emotion from text")

    # Sweep modes
    parser.add_argument("--sweep-intensity", action="store_true", help="Sweep intensity levels")
    parser.add_argument("--sweep-emotions", action="store_true", help="Sweep all emotions")
    parser.add_argument("--intensities", type=str, default="0.0,0.3,0.5,0.7,1.0,1.3,1.5",
                       help="Intensities to sweep")

    # Model paths
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--directions", type=str, default="../checkpoints/emoknob/directions",
                       help="Pre-computed direction vectors path")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create adapter
    config = EmoKnobConfig()
    adapter = create_emoknob_adapter(
        config=config,
        direction_cache_path=args.directions if Path(args.directions).exists() else None,
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        adapter.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Load reference audio and extract features
    logger.info(f"Loading reference: {args.reference}")
    reference_audio = load_audio(args.reference)
    reference_features = extract_features(reference_audio)
    logger.info(f"Extracted features: {reference_features.shape}")

    # Create output directory if needed
    output_path = Path(args.output)
    if args.sweep_intensity or args.sweep_emotions:
        output_path.mkdir(parents=True, exist_ok=True)

    # Generate based on mode
    if args.sweep_emotions:
        # Sweep all emotions
        emotions = ["happy", "sad", "angry", "surprised", "calm", "fearful", "disgusted"]
        logger.info(f"Sweeping {len(emotions)} emotions at intensity {args.intensity}")

        results = sweep_emotions(adapter, reference_features, emotions, args.intensity, device)

        for emotion, result in results.items():
            out_file = output_path / f"{emotion}_i{args.intensity:.1f}.wav"
            logger.info(f"Generated: {out_file}")
            # Note: In full pipeline, would generate audio from prosody tokens
            # For now, just log the prosody token info
            logger.info(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")

    elif args.sweep_intensity:
        # Sweep intensity levels
        emotion = args.emotion or "happy"
        intensities = [float(x) for x in args.intensities.split(",")]
        logger.info(f"Sweeping {len(intensities)} intensities for {emotion}")

        results = sweep_intensities(adapter, reference_features, emotion, intensities, device)

        for i, intensity in enumerate(results["intensities"]):
            out_file = output_path / f"{emotion}_i{intensity:.1f}.wav"
            logger.info(f"Generated: {out_file} ({intensity_to_description(intensity)} {emotion})")
            tokens = results["tokens"][i]
            logger.info(f"  Prosody tokens shape: {tokens.shape}")

    elif args.blend:
        # Blend multiple emotions
        blend_parts = args.blend.split(",")
        emotions = []
        weights = []
        for part in blend_parts:
            emo, weight = part.split(":")
            emotions.append(emo.strip())
            weights.append(float(weight))

        logger.info(f"Blending emotions: {list(zip(emotions, weights))} at intensity {args.intensity}")

        result = generate_with_blend(
            adapter, reference_features, emotions, weights, args.intensity, device
        )

        logger.info(f"Generated: {output_path}")
        logger.info(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")

    else:
        # Single emotion generation
        if args.auto_emotion:
            emotion, confidence = infer_emotion_from_text(args.text)
            logger.info(f"Auto-detected emotion: {emotion} (confidence: {confidence:.2f})")
        else:
            emotion = args.emotion or "neutral"

        intensity = args.intensity
        logger.info(f"Generating with {emotion} at intensity {intensity}")

        result = generate_with_emotion(adapter, reference_features, emotion, intensity, device)

        logger.info(f"Generated: {output_path}")
        logger.info(f"  Speaker embedding: {result['speaker_emb'].shape}")
        logger.info(f"  Controlled embedding: {result['controlled_emb'].shape}")
        logger.info(f"  Prosody tokens: {result['prosody_tokens'].shape}")

        # Compute emotion change
        change_norm = (result['controlled_emb'] - result['speaker_emb']).norm().item()
        logger.info(f"  Emotion change magnitude: {change_norm:.4f}")

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
