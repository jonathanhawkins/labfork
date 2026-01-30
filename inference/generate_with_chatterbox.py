#!/usr/bin/env python3
"""
Inference script for Chatterbox-style emotion exaggeration.

Based on Chatterbox by Resemble AI (December 2025).

Features:
1. Single-parameter emotion exaggeration (0.0=monotone to 2.0=dramatic)
2. Native paralinguistic tags ([laugh], [sigh], [cough], etc.)
3. CFG weight control for pacing
4. Zero-shot voice cloning from reference audio

Usage:
    # Basic generation with exaggeration
    python generate_with_chatterbox.py \
      --text "Hello, how are you?" \
      --reference speaker.wav \
      --exaggeration 0.7 \
      --output hello.wav

    # With paralinguistic tags
    python generate_with_chatterbox.py \
      --text "Hi there [laugh], have you got a minute?" \
      --reference speaker.wav \
      --exaggeration 0.8 \
      --output greeting.wav

    # Exaggeration sweep (generate at multiple levels)
    python generate_with_chatterbox.py \
      --text "I can't believe it!" \
      --reference speaker.wav \
      --sweep-exaggeration \
      --output outputs/

    # List supported paralinguistic tags
    python generate_with_chatterbox.py --list-tags

    # Custom CFG weight
    python generate_with_chatterbox.py \
      --text "Hello world" \
      --reference speaker.wav \
      --exaggeration 1.5 --cfg-weight 0.35 \
      --output dramatic.wav
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.chatterbox_emotion import (
    ChatterboxConfig,
    Chatterbox,
    ChatterboxAdapter,
    parse_paralinguistic_tags,
    exaggeration_to_description,
    suggest_cfg_weight,
    get_supported_tags,
    describe_tag,
    format_text_with_tags,
    PARALINGUISTIC_TAGS,
    TAG_CATEGORIES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def extract_features(
    audio: torch.Tensor,
    feature_type: str = "mock",
    device: torch.device = None,
) -> torch.Tensor:
    """
    Extract features from audio.

    In production, this would use wav2vec2/HuBERT. For testing,
    we use mock features.
    """
    if device is None:
        device = torch.device('cpu')

    if feature_type == "mock":
        # Mock features for testing
        # Assume 50 features per second at 16kHz
        num_samples = audio.shape[-1]
        num_frames = num_samples // 320  # ~50 Hz
        features = torch.randn(1, num_frames, 768).to(device)
        return features

    elif feature_type == "wav2vec2":
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

            inputs = processor(audio.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            return outputs.last_hidden_state

        except ImportError:
            logger.warning("transformers not installed, using mock features")
            return extract_features(audio, "mock", device)

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


# =============================================================================
# GENERATION
# =============================================================================

def generate_prosody_tokens(
    model: Chatterbox,
    features: torch.Tensor,
    text: str,
    exaggeration: float = 0.5,
    cfg_weight: Optional[float] = None,
    use_adaptive_cfg: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate prosody tokens using Chatterbox model.

    Args:
        model: Chatterbox model
        features: Reference audio features
        text: Text with optional paralinguistic tags
        exaggeration: Emotion exaggeration level (0.0-2.0)
        cfg_weight: CFG weight (auto if None)
        use_adaptive_cfg: Use adaptive CFG based on exaggeration

    Returns:
        Dictionary with prosody tokens and auxiliary info
    """
    model.eval()

    with torch.no_grad():
        result = model(
            features,
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            use_adaptive_cfg=use_adaptive_cfg,
        )

    return result


def generate_exaggeration_sweep(
    model: Chatterbox,
    features: torch.Tensor,
    text: str,
    levels: List[float] = None,
) -> Dict[str, any]:
    """
    Generate prosody tokens at multiple exaggeration levels.

    Useful for comparing different emotional intensities.
    """
    if levels is None:
        levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    model.eval()

    results = {
        'levels': levels,
        'tokens': [],
        'descriptions': [],
        'cfg_weights': [],
    }

    with torch.no_grad():
        for level in levels:
            result = model(features, text=text, exaggeration=level)
            results['tokens'].append(result['prosody_tokens'])
            results['descriptions'].append(exaggeration_to_description(level))
            results['cfg_weights'].append(suggest_cfg_weight(level))

    return results


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def list_supported_tags():
    """Print all supported paralinguistic tags."""
    print("\n" + "=" * 60)
    print("SUPPORTED PARALINGUISTIC TAGS")
    print("=" * 60)

    # Group by category
    by_category = {}
    for tag_name, info in PARALINGUISTIC_TAGS.items():
        cat = info['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((tag_name, info))

    for category in sorted(by_category.keys()):
        print(f"\n{category.upper()}:")
        for tag_name, info in sorted(by_category[category]):
            duration_ms = info['duration_frames'] * 10
            print(f"  [{tag_name}] - intensity: {info['intensity']:.1f}, ~{duration_ms}ms")

    print("\n" + "=" * 60)
    print("USAGE EXAMPLE:")
    print('  "Hi there [laugh], have you got a minute to chat?"')
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech with Chatterbox emotion exaggeration"
    )

    # Input/Output
    parser.add_argument(
        '--text', type=str,
        help='Text to synthesize (may contain paralinguistic tags like [laugh])'
    )
    parser.add_argument(
        '--reference', type=str,
        help='Reference audio for voice cloning'
    )
    parser.add_argument(
        '--output', type=str, default='output.wav',
        help='Output path (file or directory for sweep)'
    )
    parser.add_argument(
        '--checkpoint', type=str,
        help='Path to model checkpoint'
    )

    # Exaggeration control
    parser.add_argument(
        '--exaggeration', type=float, default=0.5,
        help='Emotion exaggeration level (0.0=monotone to 2.0=dramatic, default: 0.5)'
    )
    parser.add_argument(
        '--cfg-weight', type=float, default=None,
        help='CFG weight for pacing (auto if not specified)'
    )
    parser.add_argument(
        '--no-adaptive-cfg', action='store_true',
        help='Disable adaptive CFG (use manual cfg-weight)'
    )

    # Sweep mode
    parser.add_argument(
        '--sweep-exaggeration', action='store_true',
        help='Generate at multiple exaggeration levels'
    )
    parser.add_argument(
        '--sweep-levels', type=str, default=None,
        help='Custom exaggeration levels for sweep (comma-separated)'
    )

    # Tag utilities
    parser.add_argument(
        '--list-tags', action='store_true',
        help='List all supported paralinguistic tags'
    )
    parser.add_argument(
        '--describe-tag', type=str, default=None,
        help='Get description of a specific tag'
    )
    parser.add_argument(
        '--parse-text', type=str, default=None,
        help='Parse text and show extracted tags (no generation)'
    )

    # Feature extraction
    parser.add_argument(
        '--feature-type', type=str, default='mock',
        choices=['mock', 'wav2vec2'],
        help='Feature extraction method'
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_tags:
        list_supported_tags()
        return

    if args.describe_tag:
        print(describe_tag(args.describe_tag))
        return

    if args.parse_text:
        clean_text, tags = parse_paralinguistic_tags(args.parse_text)
        print(f"Original: {args.parse_text}")
        print(f"Clean: {clean_text}")
        print(f"Tags found: {len(tags)}")
        for tag in tags:
            print(f"  - [{tag.tag_name}] at word {tag.word_index}")
            print(f"    Category: {tag.category}, Intensity: {tag.intensity}")
        return

    # Validate required arguments for generation
    if not args.text:
        parser.error("--text is required for generation")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    config = ChatterboxConfig()
    model = Chatterbox(config).to(device)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Extract features from reference audio
    if args.reference and os.path.exists(args.reference):
        logger.info(f"Loading reference audio: {args.reference}")
        audio = load_audio(args.reference)
        features = extract_features(audio, args.feature_type, device)
    else:
        logger.info("Using mock features (no reference audio)")
        features = torch.randn(1, 100, config.input_dim).to(device)

    # Parse text for info
    clean_text, parsed_tags = parse_paralinguistic_tags(args.text)
    logger.info(f"Text: {args.text}")
    logger.info(f"Clean text: {clean_text}")
    if parsed_tags:
        logger.info(f"Paralinguistic tags: {[t.tag_name for t in parsed_tags]}")

    # Generate
    if args.sweep_exaggeration:
        # Sweep mode
        levels = None
        if args.sweep_levels:
            levels = [float(l) for l in args.sweep_levels.split(',')]

        logger.info("Generating exaggeration sweep...")
        results = generate_exaggeration_sweep(model, features, args.text, levels)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        for i, (level, desc, cfg) in enumerate(zip(
            results['levels'], results['descriptions'], results['cfg_weights']
        )):
            tokens = results['tokens'][i]
            token_path = output_dir / f"exag_{level:.2f}_{desc}.pt"
            torch.save({
                'prosody_tokens': tokens.cpu(),
                'exaggeration': level,
                'description': desc,
                'suggested_cfg': cfg,
                'text': args.text,
            }, token_path)
            logger.info(f"Saved: {token_path} (variance: {tokens.var().item():.4f})")

        # Print summary
        print("\nExaggeration Sweep Summary:")
        print("-" * 50)
        for level, desc, cfg in zip(results['levels'], results['descriptions'], results['cfg_weights']):
            print(f"  {level:.2f}: {desc:12s} (suggested CFG: {cfg:.2f})")

    else:
        # Single generation
        use_adaptive_cfg = not args.no_adaptive_cfg

        if args.cfg_weight is None:
            suggested_cfg = suggest_cfg_weight(args.exaggeration)
            logger.info(f"Using suggested CFG weight: {suggested_cfg:.2f}")
        else:
            suggested_cfg = args.cfg_weight

        logger.info(f"Generating with exaggeration={args.exaggeration:.2f}")
        result = generate_prosody_tokens(
            model,
            features,
            args.text,
            exaggeration=args.exaggeration,
            cfg_weight=None if use_adaptive_cfg else args.cfg_weight,
            use_adaptive_cfg=use_adaptive_cfg,
        )

        # Save prosody tokens
        output_path = Path(args.output)
        if output_path.suffix == '.pt':
            torch.save({
                'prosody_tokens': result['prosody_tokens'].cpu(),
                'speaker_emb': result['speaker_emb'].cpu(),
                'exaggeration': args.exaggeration,
                'text': args.text,
                'clean_text': result['clean_text'],
                'parsed_tags': [t.tag_name for t in result['parsed_tags']],
            }, output_path)
        else:
            # For .wav output, just save the tokens as .pt
            pt_path = output_path.with_suffix('.pt')
            torch.save({
                'prosody_tokens': result['prosody_tokens'].cpu(),
                'speaker_emb': result['speaker_emb'].cpu(),
                'exaggeration': args.exaggeration,
                'text': args.text,
                'clean_text': result['clean_text'],
                'parsed_tags': [t.tag_name for t in result['parsed_tags']],
            }, pt_path)
            logger.info(f"Saved prosody tokens to: {pt_path}")
            logger.info("Note: Full audio generation requires CSM model integration")

        logger.info(f"Prosody tokens shape: {result['prosody_tokens'].shape}")
        logger.info(f"Exaggeration description: {exaggeration_to_description(args.exaggeration)}")

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
