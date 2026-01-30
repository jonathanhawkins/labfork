#!/usr/bin/env python3
"""
Train EMORL-TTS: VAD-space Intensity with Local Emphasis Regulation

Based on EMORL-TTS (arXiv:2510.05758) - combines global emotion intensity
in VAD space with local word-level emphasis regulation using RL.

Key Features:
1. Global VAD-based emotion intensity control at utterance level
2. Local word-level emphasis positions and strengths
3. Dual reward system: VAD matching + emphasis clarity
4. Emphasis curriculum learning (gradual introduction)
5. SFT regularization to preserve naturalness

Usage:
    # Train from scratch
    python train_emorl_tts.py --config config/emorl_tts.yaml \
        --manifest ../data/emotion_manifest.json

    # Train from V7 baseline checkpoint
    python train_emorl_tts.py --config config/emorl_tts.yaml \
        --checkpoint ../checkpoints/prosody_v7/best.pt \
        --manifest ../data/emotion_manifest.json

    # Test mode
    python train_emorl_tts.py --test

Dependencies: Requires V7 baseline training (#6) completed first.
Builds on Multi-Reward RL (#28).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from emorl_tts import (
    EMORLConfig,
    EMORLDataset,
    EMORLTrainer,
    emorl_collate_fn,
    test_emorl_tts,
)


def load_config(config_path: str) -> EMORLConfig:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        # Filter to only valid EMORLConfig fields
        valid_fields = {
            f.name for f in EMORLConfig.__dataclass_fields__.values()
        }
        filtered_dict = {
            k: v for k, v in config_dict.items()
            if k in valid_fields
        }
        return EMORLConfig(**filtered_dict)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        return EMORLConfig()


def load_prosody_encoder(checkpoint_path: str, config: EMORLConfig):
    """Load pre-trained prosody encoder from checkpoint."""
    from prosody_conditioning import ProsodyConfig, ProsodyEncoder

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Try to get prosody config from checkpoint
        if 'prosody_config' in checkpoint:
            prosody_config = ProsodyConfig(**checkpoint['prosody_config'])
        else:
            prosody_config = ProsodyConfig(hidden_size=config.hidden_size)

        prosody_encoder = ProsodyEncoder(prosody_config)

        # Load weights
        if 'prosody_encoder' in checkpoint:
            prosody_encoder.load_state_dict(checkpoint['prosody_encoder'])
            print(f"Loaded prosody encoder from {checkpoint_path}")
        else:
            print(f"Warning: No prosody_encoder weights in {checkpoint_path}")
            print("Using randomly initialized encoder")
    else:
        prosody_config = ProsodyConfig(hidden_size=config.hidden_size)
        prosody_encoder = ProsodyEncoder(prosody_config)
        print("Created fresh prosody encoder (no checkpoint provided)")

    return prosody_encoder


def create_data_loaders(
    config: EMORLConfig,
    train_manifest: str,
    val_manifest: str = None,
    prosody_cache_dir: str = 'data/prosody_cache',
):
    """Create training and validation data loaders."""
    train_dataset = EMORLDataset(
        manifest_path=train_manifest,
        prosody_cache_dir=prosody_cache_dir,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=emorl_collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = None
    if val_manifest:
        val_dataset = EMORLDataset(
            manifest_path=val_manifest,
            prosody_cache_dir=prosody_cache_dir,
            config=config,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=emorl_collate_fn,
            num_workers=0,
        )

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train EMORL-TTS: VAD + Local Emphasis RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train_emorl_tts.py --manifest ../data/emotion_manifest.json

  # Train from V7 baseline
  python train_emorl_tts.py --checkpoint ../checkpoints/prosody_v7/best.pt \\
      --manifest ../data/emotion_manifest.json

  # Custom config
  python train_emorl_tts.py --config config/emorl_tts.yaml \\
      --checkpoint ../checkpoints/prosody_v7/best.pt \\
      --manifest ../data/emotion_manifest.json

Key EMORL-TTS Features:
  - Global VAD-based emotion intensity at utterance level
  - Local word-level emphasis positions (1-4 words)
  - Emphasis strength levels (0=none to 4=very strong)
  - Dual reward: VAD matching + emphasis clarity
  - Emphasis curriculum learning
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/emorl_tts.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Pre-trained prosody encoder checkpoint (e.g., V7 baseline)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        help='Training manifest JSON file'
    )
    parser.add_argument(
        '--val_manifest',
        type=str,
        help='Validation manifest JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../checkpoints/emorl_tts',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--prosody_cache',
        type=str,
        default='../data/prosody_cache',
        help='Prosody cache directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--no_emphasis_curriculum',
        action='store_true',
        help='Disable emphasis curriculum learning'
    )
    parser.add_argument(
        '--emphasis_weight',
        type=float,
        help='Local emphasis reward weight (overrides config)'
    )
    parser.add_argument(
        '--vad_weight',
        type=float,
        help='Global VAD reward weight (overrides config)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test mode with synthetic data'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use for training'
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("EMORL-TTS: VAD-space Intensity + Local Emphasis Regulation")
    print("=" * 70)

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.no_emphasis_curriculum:
        config.use_emphasis_curriculum = False
    if args.emphasis_weight:
        config.local_emphasis_reward_weight = args.emphasis_weight
    if args.vad_weight:
        config.global_vad_reward_weight = args.vad_weight

    # Test mode
    if args.test:
        print("\nRunning test mode with synthetic data...")
        test_emorl_tts(config)
        return

    # Check for manifest
    if not args.manifest:
        print("\nNo manifest provided!")
        print("\nEMORL-TTS requires a training manifest with:")
        print("  - prosody features (semantic, acoustic, rhythm, contour)")
        print("  - emotion labels")
        print("  - optional: word-level emphasis annotations")
        print("\nRun with --test for synthetic data test")
        print("Or provide --manifest path/to/manifest.json")
        return

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load prosody encoder
    prosody_encoder = load_prosody_encoder(args.checkpoint, config)

    # Create trainer
    trainer = EMORLTrainer(
        config=config,
        prosody_encoder=prosody_encoder,
        device=device,
    )

    # Print config summary
    print("\nConfiguration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  GRPO group size: {config.group_size}")
    print(f"  Global VAD weight: {config.global_vad_reward_weight}")
    print(f"  Local emphasis weight: {config.local_emphasis_reward_weight}")
    print(f"  Emphasis curriculum: {config.use_emphasis_curriculum}")
    print(f"  Max emphasis words: {config.max_emphasis_words}")
    print(f"  Emphasis levels: {config.num_emphasis_levels}")
    print(f"  Output dir: {config.output_dir}")

    # Create data loaders
    print(f"\nLoading data from: {args.manifest}")
    train_loader, val_loader = create_data_loaders(
        config,
        train_manifest=args.manifest,
        val_manifest=args.val_manifest,
        prosody_cache_dir=args.prosody_cache,
    )

    print(f"  Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Validation samples: {len(val_loader.dataset)}")

    # Train
    trainer.train(train_loader, val_loader)

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {config.output_dir}")
    print("\nNext steps:")
    print("  1. Run verification with generate_with_emorl.py")
    print("  2. Test emphasis on different words in same sentence")
    print("  3. Verify energy/F0 peaks align with emphasis targets")
    print("  4. Run emotion2vec to confirm global emotion preserved")


if __name__ == "__main__":
    main()
