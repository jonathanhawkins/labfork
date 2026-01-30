#!/usr/bin/env python3
"""
Training script for EASPO: Emotion-Aware Stepwise Preference Optimization

Based on arXiv:2509.25416 (Sept 2025). Post-training framework for emotion
alignment with dense supervision at each denoising step.

Usage:
    # Train EASPO with pre-trained flow model
    python train_easpo.py --config config/easpo.yaml \
        --manifest ../data/emotion_manifest.json \
        --flow-checkpoint ../checkpoints/prosody_flow/best.pt

    # Resume training
    python train_easpo.py --config config/easpo.yaml \
        --manifest ../data/emotion_manifest.json \
        --resume ../checkpoints/easpo/latest.pt

    # Test mode (no data needed)
    python train_easpo.py --test
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from easpo import (
    EASPOConfig,
    EASPM,
    EASPOTrainer,
    EASPOAdapter,
    EASPODataset,
    StepwiseCandidateSampler,
    StepwisePreferenceLoss,
    TimeConditionedEmotionEncoder,
    NoisyStateEncoder,
    collate_easpo,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION,
)
from prosody_flow import ProsodyFlowConfig, ProsodyFlow


def load_config(config_path: str) -> EASPOConfig:
    """Load config from YAML file."""
    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        return EASPOConfig()

    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    # Handle nested base_config
    if 'base_config' in config_dict:
        base_dict = config_dict.pop('base_config')
        base_config = ProsodyFlowConfig(**{k: v for k, v in base_dict.items()
                                           if hasattr(ProsodyFlowConfig, k)})
    else:
        base_config = ProsodyFlowConfig()

    # Create main config
    config = EASPOConfig(
        base_config=base_config,
        **{k: v for k, v in config_dict.items()
           if hasattr(EASPOConfig, k) and k != 'base_config'}
    )

    return config


def create_dataloaders(
    manifest_path: str,
    val_manifest_path: Optional[str],
    config: EASPOConfig,
) -> tuple:
    """Create train and validation dataloaders."""

    # Check if manifest exists
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Create datasets
    train_dataset = EASPODataset(
        manifest_path=manifest_path,
        prosody_dim=config.base_config.prosody_dim,
        text_dim=config.base_config.text_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_easpo,
    )

    val_loader = None
    if val_manifest_path and Path(val_manifest_path).exists():
        val_dataset = EASPODataset(
            manifest_path=val_manifest_path,
            prosody_dim=config.base_config.prosody_dim,
            text_dim=config.base_config.text_dim,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_easpo,
        )

    return train_loader, val_loader


def create_synthetic_loader(config: EASPOConfig, num_samples: int = 100) -> DataLoader:
    """Create synthetic data loader for testing."""

    class SyntheticEASPODataset(Dataset):
        def __init__(self, num_samples: int, config: EASPOConfig):
            self.num_samples = num_samples
            self.config = config

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            # Create synthetic prosody with emotion-specific patterns
            emotion_id = idx % config.num_emotions

            # Base prosody
            prosody = torch.randn(config.base_config.prosody_dim)

            # Add emotion-specific bias
            bias = torch.zeros(config.base_config.prosody_dim)
            bias[:8] = torch.tensor([
                0.8 if emotion_id == 1 else -0.5 if emotion_id == 2 else 0.0,  # happy/sad
                0.6 if emotion_id == 3 else 0.0,  # angry
                -0.7 if emotion_id == 4 else 0.0,  # fearful
                0.8 if emotion_id == 5 else 0.0,  # surprised
                0.4 if emotion_id == 7 else 0.0,  # calm
                0.0, 0.0, 0.0,
            ])
            prosody = prosody + bias

            # Text conditioning (synthetic)
            text_cond = torch.randn(64, config.base_config.text_dim)
            text_mask = torch.ones(64, dtype=torch.bool)

            return {
                'prosody': prosody,
                'emotion_ids': torch.tensor(emotion_id, dtype=torch.long),
                'text_cond': text_cond,
                'text_mask': text_mask,
            }

    dataset = SyntheticEASPODataset(num_samples, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_easpo,
    )


def train_easpo(
    config: EASPOConfig,
    flow_model: ProsodyFlow,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    output_dir: str,
    resume_path: Optional[str] = None,
):
    """Main training function."""

    # Create trainer
    trainer = EASPOTrainer(config, flow_model)

    # Resume from checkpoint
    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=trainer.device)

        if 'flow_model' in checkpoint:
            trainer.flow_model.load_state_dict(checkpoint['flow_model'])
        if 'easpm' in checkpoint:
            trainer.easpm.load_state_dict(checkpoint['easpm'])
        if 'flow_optimizer' in checkpoint:
            trainer.flow_optimizer.load_state_dict(checkpoint['flow_optimizer'])
        if 'easpm_optimizer' in checkpoint:
            trainer.easpm_optimizer.load_state_dict(checkpoint['easpm_optimizer'])
        if 'global_step' in checkpoint:
            trainer.global_step = checkpoint['global_step']
        if 'best_accuracy' in checkpoint:
            trainer.best_accuracy = checkpoint['best_accuracy']

        print(f"  Resumed at step {trainer.global_step}")
        print(f"  Best accuracy so far: {trainer.best_accuracy:.2%}")

    # Train
    trainer.train(train_loader, val_loader)

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="EASPO: Emotion-Aware Stepwise Preference Optimization Training"
    )
    parser.add_argument('--config', type=str, default='config/easpo.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--manifest', type=str,
                        help='Path to training manifest JSON')
    parser.add_argument('--val-manifest', type=str,
                        help='Path to validation manifest JSON')
    parser.add_argument('--flow-checkpoint', type=str,
                        help='Path to pre-trained flow model checkpoint')
    parser.add_argument('--resume', type=str,
                        help='Path to EASPO checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='../checkpoints/easpo',
                        help='Output directory for checkpoints')
    parser.add_argument('--test', action='store_true',
                        help='Run tests with synthetic data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Print header
    print("=" * 70)
    print("EASPO: Emotion-Aware Stepwise Preference Optimization")
    print("Based on arXiv:2509.25416 (Sept 2025)")
    print("=" * 70)

    if args.test:
        # Run test mode
        from easpo import run_tests
        run_tests()
        return

    # Load config
    config = load_config(args.config)

    print(f"\nConfiguration:")
    print(f"  Denoising steps: {config.num_denoising_steps}")
    print(f"  Candidates per step: {config.num_candidates_per_step}")
    print(f"  Step weight schedule: {config.step_weight_schedule}")
    print(f"  Beta (KL penalty): {config.beta}")
    print(f"  Use multiscale: {config.use_multiscale}")

    # Create flow model
    flow_model = ProsodyFlow(config.base_config)

    # Load pre-trained flow model
    if args.flow_checkpoint:
        print(f"\nLoading pre-trained flow model from: {args.flow_checkpoint}")
        checkpoint = torch.load(args.flow_checkpoint, map_location='cpu')

        if 'flow_model' in checkpoint:
            flow_model.load_state_dict(checkpoint['flow_model'])
        elif 'model' in checkpoint:
            flow_model.load_state_dict(checkpoint['model'])
        else:
            # Try loading directly
            try:
                flow_model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"  Warning: Could not load checkpoint directly: {e}")
                print("  Starting with fresh flow model")

    # Create dataloaders
    if args.manifest:
        print(f"\nLoading training data from: {args.manifest}")
        train_loader, val_loader = create_dataloaders(
            args.manifest,
            args.val_manifest,
            config,
        )
        print(f"  Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Validation samples: {len(val_loader.dataset)}")
    else:
        print("\nNo manifest provided, using synthetic data for testing")
        train_loader = create_synthetic_loader(config, num_samples=200)
        val_loader = create_synthetic_loader(config, num_samples=50)
        print(f"  Synthetic training samples: {len(train_loader.dataset)}")
        print(f"  Synthetic validation samples: {len(val_loader.dataset)}")

    # Train
    trainer = train_easpo(
        config=config,
        flow_model=flow_model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        resume_path=args.resume,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print(f"Best accuracy: {trainer.best_accuracy:.2%}")
    print(f"Total steps: {trainer.global_step}")

    # Show usage example
    print("\nTo use the trained model:")
    print("-" * 40)
    print(f"""
from easpo import EASPOConfig, EASPOAdapter
from prosody_flow import ProsodyFlow
import torch

# Load config and model
config = EASPOConfig()
flow_model = ProsodyFlow(config.base_config).cuda()

# Load checkpoint
checkpoint = torch.load('{args.output_dir}/best.pt')
flow_model.load_state_dict(checkpoint['flow_model'])

# Create adapter with EASPM
from easpo import EASPM
easpm = EASPM(config).cuda()
easpm.load_state_dict(checkpoint['easpm'])

adapter = EASPOAdapter(config, flow_model, easpm)

# Generate emotion-aligned prosody with EASPM guidance
out = adapter(
    text_cond,
    emotion_ids=torch.tensor([1]),  # happy
    use_easpm_guidance=True,
    guidance_scale=2.0,
)

prosody_tokens = out['prosody_tokens']
# Use with ProsodyControlledCSM
""")


if __name__ == "__main__":
    main()
