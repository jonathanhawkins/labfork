#!/usr/bin/env python3
"""
Train MSR-Codec: Multi-Stream Residual Speech Codec

4-stream disentanglement with cascaded residual architecture:
1. Semantic (HuBERT) - linguistic content
2. Timbre (speaker embedding) - speaker identity
3. Prosody (VQ1) - prosodic patterns
4. Residual (VQ2) - fine-grained details

Usage:
    # Full training
    python train_msr_codec.py --config config/msr_codec.yaml

    # Resume from checkpoint
    python train_msr_codec.py --config config/msr_codec.yaml \
        --resume ../checkpoints/msr_codec/latest.pt

    # Test mode (synthetic data)
    python train_msr_codec.py --test
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from msr_codec import (
    MSRCodecConfig,
    MSRCodec,
    MSRCodecLoss,
    MSRCodecAdapter,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# SYNTHETIC DATASET (FOR TESTING)
# =============================================================================

class SyntheticMSRDataset(Dataset):
    """Synthetic dataset for testing MSR-Codec training."""

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 100,
        mel_dim: int = 80,
        semantic_dim: int = 768,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.mel_dim = mel_dim
        self.semantic_dim = semantic_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic mel spectrogram
        mel = torch.randn(self.seq_len, self.mel_dim)

        # Generate synthetic HuBERT features
        semantic = torch.randn(self.seq_len, self.semantic_dim)

        return {
            'mel': mel,
            'semantic': semantic,
            'id': f'synthetic_{idx}',
        }


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

def get_default_training_config() -> Dict:
    """Default training configuration."""
    return {
        # Model
        'model': asdict(MSRCodecConfig()),

        # Training
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 5,
            'grad_clip': 1.0,
            'log_interval': 50,
            'save_interval': 1,
            'val_interval': 1,
        },

        # Data
        'data': {
            'train_manifest': '../data/train_manifest.json',
            'val_manifest': '../data/val_manifest.json',
            'num_workers': 4,
            'pin_memory': True,
        },

        # Checkpoints
        'checkpoint': {
            'dir': '../checkpoints/msr_codec',
            'save_best': True,
            'save_last': True,
        },
    }


def load_config(config_path: Optional[str]) -> Dict:
    """Load configuration from YAML file or use defaults."""
    config = get_default_training_config()

    if config_path and Path(config_path).exists():
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed, using default config")
            return config

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        # Deep merge
        for key in loaded:
            if key in config and isinstance(config[key], dict):
                config[key].update(loaded[key])
            else:
                config[key] = loaded[key]

    return config


# =============================================================================
# TRAINER
# =============================================================================

class MSRCodecTrainer:
    """Trainer for MSR-Codec model."""

    def __init__(
        self,
        config: Dict,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        # Create model config
        model_config = MSRCodecConfig(**config['model'])
        self.model_config = model_config

        # Create model
        self.model = MSRCodec(model_config).to(device)
        self.loss_fn = MSRCodecLoss(model_config)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_epochs = self.config['training']['warmup_epochs']
        total_epochs = self.config['training']['epochs']

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_commitment_loss = 0.0
        total_prosody_perplexity = 0.0
        total_residual_perplexity = 0.0
        num_batches = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if TQDM_AVAILABLE else train_loader

        for batch_idx, batch in enumerate(iterator):
            mel = batch['mel'].to(self.device)
            semantic = batch['semantic'].to(self.device)

            # Forward pass
            output = self.model(mel, semantic)

            # Compute loss
            losses = self.loss_fn(output, mel)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip'],
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += losses['total'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_commitment_loss += losses['total_commitment'].item()
            total_prosody_perplexity += losses['prosody_perplexity'].item()
            total_residual_perplexity += losses['residual_perplexity'].item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.config['training']['log_interval'] == 0:
                if TQDM_AVAILABLE:
                    iterator.set_postfix({
                        'loss': f"{losses['total'].item():.4f}",
                        'recon': f"{losses['reconstruction_loss'].item():.4f}",
                        'ppl_p': f"{losses['prosody_perplexity'].item():.1f}",
                        'ppl_r': f"{losses['residual_perplexity'].item():.1f}",
                    })
                else:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"loss={losses['total'].item():.4f}, "
                        f"recon={losses['reconstruction_loss'].item():.4f}"
                    )

        # Average metrics
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_recon_loss': total_recon_loss / num_batches,
            'train_commitment_loss': total_commitment_loss / num_batches,
            'train_prosody_perplexity': total_prosody_perplexity / num_batches,
            'train_residual_perplexity': total_residual_perplexity / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_prosody_perplexity = 0.0
        total_residual_perplexity = 0.0
        num_batches = 0

        for batch in val_loader:
            mel = batch['mel'].to(self.device)
            semantic = batch['semantic'].to(self.device)

            # Forward pass
            output = self.model(mel, semantic)

            # Compute loss
            losses = self.loss_fn(output, mel)

            # Accumulate metrics
            total_loss += losses['total'].item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_prosody_perplexity += losses['prosody_perplexity'].item()
            total_residual_perplexity += losses['residual_perplexity'].item()
            num_batches += 1

        metrics = {
            'val_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_prosody_perplexity': total_prosody_perplexity / num_batches,
            'val_residual_perplexity': total_residual_perplexity / num_batches,
        }

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        # Save latest
        if self.config['checkpoint']['save_last']:
            torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save best
        if is_best and self.config['checkpoint']['save_best']:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            logger.info(f"Saved best model (val_loss={metrics.get('val_loss', 0):.4f})")

        # Save periodic checkpoint
        if epoch % self.config['training']['save_interval'] == 0:
            torch.save(
                checkpoint,
                self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            )

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        if 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
            self.best_val_loss = checkpoint['metrics']['val_loss']

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Full training loop."""
        epochs = self.config['training']['epochs']
        start_epoch = self.epoch + 1

        logger.info(f"Starting training from epoch {start_epoch}")
        logger.info(f"Total epochs: {epochs}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")

        for epoch in range(start_epoch, epochs + 1):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # Step scheduler
            self.scheduler.step()

            # Log training metrics
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"recon={train_metrics['train_recon_loss']:.4f}, "
                f"prosody_ppl={train_metrics['train_prosody_perplexity']:.1f}, "
                f"residual_ppl={train_metrics['train_residual_perplexity']:.1f}"
            )

            # Validation
            if val_loader is not None and epoch % self.config['training']['val_interval'] == 0:
                val_metrics = self.validate(val_loader)

                logger.info(
                    f"Epoch {epoch} [VAL]: "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"recon={val_metrics['val_recon_loss']:.4f}"
                )

                # Check if best
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']

                # Save checkpoint
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, is_best)
            else:
                self.save_checkpoint(epoch, train_metrics, is_best=False)

        logger.info("Training completed!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MSR-Codec")
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run test mode with synthetic data'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    if args.test:
        logger.info("=" * 60)
        logger.info("MSR-Codec Training - TEST MODE")
        logger.info("=" * 60)

        # Override config for test
        config['training']['epochs'] = 3
        config['training']['batch_size'] = 4
        config['training']['log_interval'] = 1

        # Create synthetic datasets
        train_dataset = SyntheticMSRDataset(
            num_samples=20,
            seq_len=100,
            mel_dim=config['model']['mel_dim'],
            semantic_dim=config['model']['semantic_dim'],
        )
        val_dataset = SyntheticMSRDataset(
            num_samples=10,
            seq_len=100,
            mel_dim=config['model']['mel_dim'],
            semantic_dim=config['model']['semantic_dim'],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
        )

        # Create trainer
        trainer = MSRCodecTrainer(config, args.device)

        # Train
        trainer.train(train_loader, val_loader)

        logger.info("Test mode completed successfully!")

    else:
        logger.info("=" * 60)
        logger.info("MSR-Codec Training")
        logger.info("=" * 60)

        # Create trainer
        trainer = MSRCodecTrainer(config, args.device)

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # TODO: Create real data loaders from manifests
        # For now, use synthetic data
        logger.warning("Using synthetic data - implement real data loading")

        train_dataset = SyntheticMSRDataset(
            num_samples=1000,
            seq_len=100,
            mel_dim=config['model']['mel_dim'],
            semantic_dim=config['model']['semantic_dim'],
        )
        val_dataset = SyntheticMSRDataset(
            num_samples=100,
            seq_len=100,
            mel_dim=config['model']['mel_dim'],
            semantic_dim=config['model']['semantic_dim'],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
        )

        # Train
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
