#!/usr/bin/env python3
"""
Training script for UDDETTS ADV-Space Unified Emotion Control.

Based on UDDETTS (arXiv:2505.10599): Unified Discrete-Dimensional Emotion Control.

Features:
1. ADV encoder with nonlinear quantization
2. Semi-supervised training (discrete + ADV annotations)
3. OT-CFM integration for expressive prosody
4. Integration with existing spherical emotion (VAD) vectors

Usage:
    # Train UDDETTS model
    python train_uddetts_adv.py --config config/uddetts_adv.yaml

    # Resume from checkpoint
    python train_uddetts_adv.py --config config/uddetts_adv.yaml \\
        --resume ../checkpoints/uddetts_adv/best.pt

    # Test mode (synthetic data)
    python train_uddetts_adv.py --test
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uddetts_adv import (
    UDDETTSConfig,
    UDDETTSAdapter,
    UDDETTSSemiSupervisedLoss,
    ADV_PROTOTYPES,
    DISCRETE_EMOTIONS,
    EMOTION_TO_IDX,
    adv_to_emotion_name,
    describe_adv,
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC DATASET (for testing)
# =============================================================================

class SyntheticEmotionDataset(Dataset):
    """
    Synthetic dataset for testing UDDETTS training.

    Generates samples with:
    - Random audio features (simulating wav2vec2/HuBERT)
    - Random discrete emotion labels
    - Random ADV coordinates
    - Mixed supervision masks
    """

    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 768,
        num_emotions: int = 8,
        semi_supervised_ratio: float = 0.3,  # Ratio with only partial labels
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_emotions = num_emotions
        self.semi_supervised_ratio = semi_supervised_ratio

        # Generate data
        self.features = torch.randn(num_samples, input_dim)
        self.emotion_ids = torch.randint(0, num_emotions, (num_samples,))

        # Generate ADV coordinates based on emotion prototypes + noise
        self.adv_coords = torch.zeros(num_samples, 3)
        for i, emo_id in enumerate(self.emotion_ids):
            emotion = DISCRETE_EMOTIONS[emo_id.item()]
            proto = torch.tensor(ADV_PROTOTYPES[emotion])
            noise = torch.randn(3) * 0.1  # Small noise
            self.adv_coords[i] = proto + noise

        # Generate supervision masks
        self.has_discrete = torch.ones(num_samples, dtype=torch.bool)
        self.has_adv = torch.ones(num_samples, dtype=torch.bool)

        # Randomly mask some labels for semi-supervised testing
        num_partial = int(num_samples * semi_supervised_ratio)
        partial_indices = torch.randperm(num_samples)[:num_partial]

        for idx in partial_indices[:num_partial // 2]:
            self.has_discrete[idx] = False  # Only ADV

        for idx in partial_indices[num_partial // 2:]:
            self.has_adv[idx] = False  # Only discrete

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'emotion_id': self.emotion_ids[idx],
            'adv': self.adv_coords[idx],
            'has_discrete': self.has_discrete[idx],
            'has_adv': self.has_adv[idx],
        }


# =============================================================================
# TRAINER
# =============================================================================

class UDDETTSTrainer:
    """Trainer for UDDETTS ADV-space unified emotion control."""

    def __init__(
        self,
        config: UDDETTSConfig,
        train_config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.config = config
        self.train_config = train_config
        self.device = device

        # Initialize model
        self.model = UDDETTSAdapter(config).to(device)

        # Loss function
        self.loss_fn = UDDETTSSemiSupervisedLoss(config).to(device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.get('max_epochs', 100),
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(train_config.get('checkpoint_dir', '../checkpoints/uddetts_adv'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_discrete_loss = 0.0
        total_adv_loss = 0.0
        total_consistency_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move to device
            emotion_ids = batch['emotion_id'].to(self.device)
            adv = batch['adv'].to(self.device)
            has_discrete = batch['has_discrete'].to(self.device)
            has_adv = batch['has_adv'].to(self.device)

            # Forward pass
            # Use ADV coordinates if available, otherwise use emotion IDs
            encoder_output = self.model(
                adv=adv * has_adv.unsqueeze(-1).float(),  # Zero out if no ADV
                emotion_ids=emotion_ids,
                intensity=0.7,
            )

            # Compute loss
            losses = self.loss_fn(
                encoder_output,
                target_emotion_ids=emotion_ids,
                target_adv=adv,
                has_discrete=has_discrete,
                has_adv=has_adv,
            )

            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.train_config.get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config['gradient_clip'],
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_discrete_loss += losses['discrete'].item()
            total_adv_loss += losses['adv_regression'].item()
            total_consistency_loss += losses['consistency'].item()
            total_accuracy += losses['discrete_accuracy'].item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.train_config.get('log_every_n_steps', 100) == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"loss={loss.item():.4f}, "
                    f"discrete={losses['discrete'].item():.4f}, "
                    f"adv={losses['adv_regression'].item():.4f}, "
                    f"acc={losses['discrete_accuracy'].item():.4f}"
                )

        # Average metrics
        return {
            'loss': total_loss / num_batches,
            'discrete_loss': total_discrete_loss / num_batches,
            'adv_loss': total_adv_loss / num_batches,
            'consistency_loss': total_consistency_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()

        total_loss = 0.0
        total_discrete_loss = 0.0
        total_adv_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch in dataloader:
            emotion_ids = batch['emotion_id'].to(self.device)
            adv = batch['adv'].to(self.device)
            has_discrete = batch['has_discrete'].to(self.device)
            has_adv = batch['has_adv'].to(self.device)

            encoder_output = self.model(
                adv=adv * has_adv.unsqueeze(-1).float(),
                emotion_ids=emotion_ids,
                intensity=0.7,
            )

            losses = self.loss_fn(
                encoder_output,
                target_emotion_ids=emotion_ids,
                target_adv=adv,
                has_discrete=has_discrete,
                has_adv=has_adv,
            )

            total_loss += losses['total'].item()
            total_discrete_loss += losses['discrete'].item()
            total_adv_loss += losses['adv_regression'].item()
            total_accuracy += losses['discrete_accuracy'].item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_discrete_loss': total_discrete_loss / num_batches,
            'val_adv_loss': total_adv_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches,
        }

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 100,
    ):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("Starting UDDETTS ADV-Space Training")
        logger.info("=" * 60)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {num_epochs}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {train_metrics}")

            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val: {val_metrics}")

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best.pt')
                    logger.info(f"New best model saved (val_loss={self.best_val_loss:.4f})")

            # Update scheduler
            self.scheduler.step()

            # Periodic checkpoint
            save_every = self.train_config.get('save_every_n_epochs', 5)
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')

        # Final checkpoint
        self.save_checkpoint('final.pt')
        logger.info("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path}")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Tuple[UDDETTSConfig, Dict[str, Any]]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Extract model config
    model_config = UDDETTSConfig(
        input_dim=config_dict.get('input_dim', 768),
        embedding_dim=config_dict.get('embedding_dim', 256),
        hidden_dim=config_dict.get('hidden_dim', 512),
        output_dim=config_dict.get('output_dim', 2048),
        num_discrete_emotions=config_dict.get('num_discrete_emotions', 8),
        num_adv_anchors=config_dict.get('num_adv_anchors', 64),
        use_learnable_anchors=config_dict.get('use_learnable_anchors', True),
        quantization_temperature=config_dict.get('quantization_temperature', 1.0),
        discrete_label_weight=config_dict.get('discrete_label_weight', 1.0),
        adv_regression_weight=config_dict.get('adv_regression_weight', 1.0),
        consistency_weight=config_dict.get('consistency_weight', 0.5),
        dropout=config_dict.get('dropout', 0.1),
        use_layer_norm=config_dict.get('use_layer_norm', True),
        num_prosody_tokens=config_dict.get('num_prosody_tokens', 4),
    )

    return model_config, config_dict


def run_test_mode():
    """Run test mode with synthetic data."""
    logger.info("Running in TEST MODE with synthetic data...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create config
    config = UDDETTSConfig()
    train_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_epochs': 5,
        'gradient_clip': 1.0,
        'log_every_n_steps': 10,
        'save_every_n_epochs': 2,
        'checkpoint_dir': '../checkpoints/uddetts_adv_test',
    }

    # Create synthetic dataset
    full_dataset = SyntheticEmotionDataset(
        num_samples=500,
        input_dim=config.input_dim,
        num_emotions=config.num_discrete_emotions,
        semi_supervised_ratio=0.3,
    )

    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create trainer
    trainer = UDDETTSTrainer(config, train_config, device)

    # Train
    trainer.train(train_loader, val_loader, num_epochs=5)

    # Test inference
    logger.info("\nTesting inference...")
    trainer.model.eval()

    with torch.no_grad():
        # From emotion label
        result = trainer.model.from_emotion("happy", intensity=0.8)
        logger.info(f"From 'happy': tokens shape = {result['prosody_tokens'].shape}")

        # From ADV coordinates
        adv = torch.tensor([[0.7, 0.5, 0.8]], device=device)
        result = trainer.model.from_adv(adv, intensity=0.9)
        logger.info(f"From ADV: tokens shape = {result['prosody_tokens'].shape}")

        # Interpolation
        result = trainer.model.interpolate_emotions("sad", "happy", t=0.5, intensity=0.7)
        logger.info(f"Interpolation: tokens shape = {result['prosody_tokens'].shape}")

        # Blending
        result = trainer.model.blend_emotions([
            ("happy", 0.5),
            ("surprised", 0.3),
            ("calm", 0.2),
        ], intensity=0.8)
        logger.info(f"Blending: tokens shape = {result['prosody_tokens'].shape}")

    logger.info("\nTest mode completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Train UDDETTS ADV-Space Emotion Control')
    parser.add_argument('--config', type=str, default='config/uddetts_adv.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with synthetic data')

    args = parser.parse_args()

    if args.test:
        run_test_mode()
        return

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)

    model_config, train_config = load_config(config_path)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create synthetic dataset for now (replace with real data loading)
    # In production, load from train_config['data_manifest']
    logger.info("Creating synthetic dataset (replace with real data loading)...")

    full_dataset = SyntheticEmotionDataset(
        num_samples=2000,
        input_dim=model_config.input_dim,
        num_emotions=model_config.num_discrete_emotions,
        semi_supervised_ratio=0.3,
    )

    # Split
    train_size = int(train_config.get('train_split', 0.9) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 32),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 32),
        shuffle=False,
        num_workers=0,
    )

    # Create trainer
    trainer = UDDETTSTrainer(model_config, train_config, device)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=train_config.get('max_epochs', 100),
    )


if __name__ == '__main__':
    main()
