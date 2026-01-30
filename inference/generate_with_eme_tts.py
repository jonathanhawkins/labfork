#!/usr/bin/env python3
"""
Generate speech with EME-TTS emphasis-emotion coupling.

EME-TTS enables expressive TTS by modeling the link between emphasis and emotion.
This script provides multiple modes for emphasis control:

1. Automatic (LLM): Use LLM to predict emphasis positions based on text and emotion
2. Automatic (Neural): Use trained neural predictor for emphasis
3. Manual: Specify emphasis levels per word explicitly
4. Annotated: Parse user-provided text with emphasis markers (*word*, **word**)

Usage:
    # LLM-based emphasis prediction
    python generate_with_eme_tts.py \\
        --text "I am so excited about this amazing opportunity!" \\
        --emotion happy \\
        --mode llm \\
        --output excited.wav

    # Manual emphasis specification
    python generate_with_eme_tts.py \\
        --text "This is absolutely incredible" \\
        --emotion surprised \\
        --emphasis "0,0,3,2" \\
        --output incredible.wav

    # Annotated text with emphasis markers
    python generate_with_eme_tts.py \\
        --annotated "This is **absolutely** *incredible*!" \\
        --emotion surprised \\
        --output incredible.wav

    # Sweep all emotions with auto emphasis
    python generate_with_eme_tts.py \\
        --text "Hello, how are you?" \\
        --sweep-emotions \\
        --output outputs/

Reference: https://arxiv.org/abs/2507.12015
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.eme_tts import (
    EMETTSConfig,
    EMETTSAdapter,
    create_eme_tts_adapter,
    emphasis_level_to_description,
    parse_annotated_text,
    EMPHASIS_LEVELS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_adapter(
    checkpoint_path: Optional[str] = None,
    config: Optional[EMETTSConfig] = None,
    device: str = 'cpu',
) -> EMETTSAdapter:
    """Load EME-TTS adapter from checkpoint or create new one."""
    if config is None:
        config = EMETTSConfig()

    adapter = EMETTSAdapter(config).to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        adapter.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    adapter.eval()
    return adapter


def create_mock_text_embeddings(
    text: str,
    config: EMETTSConfig,
    device: str = 'cpu',
) -> torch.Tensor:
    """Create mock text embeddings for testing (replace with actual encoder in production)."""
    words = text.split()
    num_words = len(words)

    # Generate random embeddings (in production, use actual text encoder)
    embeddings = torch.randn(1, num_words, config.text_hidden_dim, device=device)
    return embeddings


def generate_with_llm_emphasis(
    adapter: EMETTSAdapter,
    text: str,
    emotion: str,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Generate prosody tokens using LLM for emphasis prediction."""
    text_emb = create_mock_text_embeddings(text, adapter.config, device)

    with torch.no_grad():
        result = adapter.from_text(
            text=text,
            text_embeddings=text_emb,
            emotion=emotion,
            use_llm=True,
        )

    return {
        'prosody_tokens': result['prosody_tokens'],
        'emphasis_levels': result['emphasis_levels'],
        'llm_emphasis': result.get('llm_emphasis'),
        'text': text,
        'emotion': emotion,
    }


def generate_with_neural_emphasis(
    adapter: EMETTSAdapter,
    text: str,
    emotion: str,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Generate prosody tokens using neural emphasis predictor."""
    text_emb = create_mock_text_embeddings(text, adapter.config, device)

    with torch.no_grad():
        result = adapter.from_text(
            text=text,
            text_embeddings=text_emb,
            emotion=emotion,
            use_llm=False,  # Use neural predictor instead
        )

    return {
        'prosody_tokens': result['prosody_tokens'],
        'emphasis_levels': result['emphasis_levels'],
        'text': text,
        'emotion': emotion,
    }


def generate_with_manual_emphasis(
    adapter: EMETTSAdapter,
    text: str,
    emotion: str,
    emphasis_levels: List[int],
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Generate prosody tokens with manually specified emphasis levels."""
    text_emb = create_mock_text_embeddings(text, adapter.config, device)

    # Ensure emphasis_levels matches text length
    num_words = len(text.split())
    if len(emphasis_levels) < num_words:
        emphasis_levels = emphasis_levels + [0] * (num_words - len(emphasis_levels))
    elif len(emphasis_levels) > num_words:
        emphasis_levels = emphasis_levels[:num_words]

    with torch.no_grad():
        result = adapter.from_emphasis_trajectory(
            text_embeddings=text_emb,
            emotion=emotion,
            emphasis_levels=emphasis_levels,
        )

    return {
        'prosody_tokens': result['prosody_tokens'],
        'emphasis_levels': emphasis_levels,
        'text': text,
        'emotion': emotion,
    }


def generate_from_annotated(
    adapter: EMETTSAdapter,
    annotated_text: str,
    emotion: str,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Generate from text with emphasis markers (*word* or **word**)."""
    clean_text, emphasized_indices, level_map = parse_annotated_text(annotated_text)

    # Build emphasis levels array
    words = clean_text.split()
    emphasis_levels = [level_map.get(i, 0) for i in range(len(words))]

    result = generate_with_manual_emphasis(
        adapter, clean_text, emotion, emphasis_levels, device
    )

    result['original_annotated'] = annotated_text
    return result


def sweep_emotions(
    adapter: EMETTSAdapter,
    text: str,
    output_dir: Path,
    mode: str = 'neural',
    device: str = 'cpu',
) -> List[Dict[str, Any]]:
    """Generate for all emotions."""
    results = []

    for emotion in adapter.config.emotion_labels:
        logger.info(f"Generating for emotion: {emotion}")

        if mode == 'llm':
            result = generate_with_llm_emphasis(adapter, text, emotion, device)
        else:
            result = generate_with_neural_emphasis(adapter, text, emotion, device)

        results.append(result)

        # Log emphasis info
        emphasis_str = ' '.join([
            f"{w}({emphasis_level_to_description(l)})"
            for w, l in zip(text.split(), result['emphasis_levels'][0].tolist())
        ])
        logger.info(f"  {emotion}: {emphasis_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate with EME-TTS emphasis-emotion coupling")

    # Input
    parser.add_argument('--text', type=str, help="Input text")
    parser.add_argument('--annotated', type=str, help="Annotated text with emphasis markers")
    parser.add_argument('--emotion', type=str, default='neutral',
                        help="Target emotion")
    parser.add_argument('--emphasis', type=str,
                        help="Comma-separated emphasis levels per word (0-3)")

    # Mode
    parser.add_argument('--mode', type=str, choices=['llm', 'neural', 'manual'],
                        default='neural', help="Emphasis prediction mode")

    # Model
    parser.add_argument('--checkpoint', type=str, help="Path to checkpoint")
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output
    parser.add_argument('--output', type=str, default='output.wav',
                        help="Output path")

    # Sweep
    parser.add_argument('--sweep-emotions', action='store_true',
                        help="Generate for all emotions")

    args = parser.parse_args()

    logger.info("EME-TTS Inference")
    logger.info("=" * 60)

    # Validate input
    if not args.text and not args.annotated:
        logger.error("Please provide --text or --annotated")
        sys.exit(1)

    # Load adapter
    config = EMETTSConfig()
    adapter = load_adapter(args.checkpoint, config, args.device)
    logger.info(f"Loaded adapter on {args.device}")

    # Handle annotated text
    if args.annotated:
        result = generate_from_annotated(
            adapter, args.annotated, args.emotion, args.device
        )
        logger.info(f"Original annotated: {args.annotated}")
        logger.info(f"Clean text: {result['text']}")

    # Handle sweep
    elif args.sweep_emotions:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = sweep_emotions(
            adapter, args.text, output_dir, args.mode, args.device
        )

        # Save results
        for result in results:
            emotion = result['emotion']
            tokens_path = output_dir / f"{emotion}_tokens.pt"
            torch.save(result['prosody_tokens'], tokens_path)
            logger.info(f"Saved {emotion} tokens to {tokens_path}")

        logger.info(f"\nGenerated {len(results)} emotion variants")
        return

    # Handle manual emphasis
    elif args.emphasis:
        emphasis_levels = [int(x) for x in args.emphasis.split(',')]
        result = generate_with_manual_emphasis(
            adapter, args.text, args.emotion, emphasis_levels, args.device
        )

    # Handle automatic emphasis
    else:
        if args.mode == 'llm':
            result = generate_with_llm_emphasis(
                adapter, args.text, args.emotion, args.device
            )
            if result.get('llm_emphasis'):
                logger.info(f"LLM annotated: {result['llm_emphasis'].get('annotated_text', '')}")
        else:
            result = generate_with_neural_emphasis(
                adapter, args.text, args.emotion, args.device
            )

    # Log results
    logger.info(f"\nText: {result['text']}")
    logger.info(f"Emotion: {result['emotion']}")

    emphasis_tensor = result['emphasis_levels']
    if isinstance(emphasis_tensor, torch.Tensor):
        emphasis_list = emphasis_tensor[0].tolist()
    else:
        emphasis_list = emphasis_tensor

    words = result['text'].split()
    for word, level in zip(words, emphasis_list):
        logger.info(f"  {word}: {emphasis_level_to_description(level)}")

    logger.info(f"\nProsody tokens shape: {result['prosody_tokens'].shape}")

    # Save tokens
    output_path = Path(args.output)
    if output_path.suffix == '.wav':
        tokens_path = output_path.with_suffix('.pt')
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        tokens_path = output_path / 'prosody_tokens.pt'

    torch.save(result['prosody_tokens'], tokens_path)
    logger.info(f"Saved prosody tokens to {tokens_path}")

    # Save metadata
    meta = {
        'text': result['text'],
        'emotion': result['emotion'],
        'emphasis_levels': emphasis_list,
        'emphasis_descriptions': [emphasis_level_to_description(l) for l in emphasis_list],
    }
    meta_path = tokens_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
