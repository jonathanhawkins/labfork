#!/usr/bin/env python3
"""
Training script for SpeechTripleNet VAE.

This script trains the SpeechTripleNet model for triple disentanglement
of content, timbre, and prosody from speech.

Usage:
    # Train from scratch
    python train_speech_triple_net.py --config config/speech_triple_net.yaml

    # Resume from checkpoint
    python train_speech_triple_net.py --config config/speech_triple_net.yaml \\
        --resume ../checkpoints/speech_triple_net/best.pt

    # Test mode (quick validation)
    python train_speech_triple_net.py --test
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.speech_triple_net import (
    SpeechTripleNetConfig,
    SpeechTripleNet,
    SpeechTripleNetLoss,
    SpeechTripleNetAdapter,
)


# =============================================================================
# DATASET
# =============================================================================

class MelSpectrogramDataset(Dataset):
    """
    Dataset for mel spectrograms.

    Loads pre-computed mel spectrograms from a manifest file.
    """

    def __init__(
        self,
        manifest_path: str,
        max_len: int = 400,
        mel_dim: int = 80,
    ):
        self.max_len = max_len
        self.mel_dim = mel_dim

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Filter valid entries
        self.samples = [
            s for s in self.manifest
            if 'mel_path' in s and os.path.exists(s['mel_path'])
        ]

        print(f"Loaded {len(self.samples)} samples from manifest")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load mel spectrogram
        mel = torch.load(sample['mel_path'])

        # Ensure correct shape [T, D]
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        if mel.shape[0] == self.mel_dim:
            mel = mel.t()

        # Truncate or pad
        if mel.shape[0] > self.max_len:
            start = torch.randint(0, mel.shape[0] - self.max_len, (1,)).item()
            mel = mel[start:start + self.max_len]
        elif mel.shape[0] < self.max_len:
            pad_len = self.max_len - mel.shape[0]
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad_len))

        return {
            'mel': mel,
            'speaker_id': sample.get('speaker_id', 0),
        }


class SyntheticMelDataset(Dataset):
    """Synthetic dataset for testing/debugging."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 100,
        mel_dim: int = 80,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.mel_dim = mel_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mel = torch.randn(self.seq_len, self.mel_dim)
        return {
            'mel': mel,
            'speaker_id': idx % 10,
        }


# =============================================================================
# TRAINER
# =============================================================================

class SpeechTripleNetTrainer:
    """Trainer for SpeechTripleNet model."""

    def __init__(
        self,
        config: SpeechTripleNetConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: str = '../checkpoints/speech_triple_net',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        log_interval: int = 100,
        save_interval: int = 1000,
        device: str = 'cuda',
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Create model
        self.model = SpeechTripleNet(config).to(device)
        self.loss_fn = SpeechTripleNetLoss(config)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.save_interval = save_interval

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        mel = batch['mel'].to(self.device)

        # Forward pass
        output = self.model(mel)
        losses = self.loss_fn(output, mel)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        total_losses = {}
        num_batches = 0

        for batch in self.val_dataloader:
            mel = batch['mel'].to(self.device)
            output = self.model(mel)
            losses = self.loss_fn(output, mel)

            for k, v in losses.items():
                val = v.item() if torch.is_tensor(v) else v
                total_losses[k] = total_losses.get(k, 0) + val

            num_batches += 1

        # Average
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = path.parent / 'best.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Loaded checkpoint from step {self.global_step}")

    def train(self, num_epochs: int, num_steps: Optional[int] = None):
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            num_steps: Optional max steps (overrides epochs if set)
        """
        print(f"\nStarting training...")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.train_dataloader.batch_size}")
        print(f"  Steps per epoch: {len(self.train_dataloader)}")

        for epoch in range(num_epochs):
            epoch_losses = {}
            num_batches = 0

            for batch in self.train_dataloader:
                losses = self.train_step(batch)

                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
                num_batches += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    log_str = (
                        f"Step {self.global_step} | "
                        f"Loss: {losses['total']:.4f} | "
                        f"Recon: {losses['reconstruction']:.4f} | "
                        f"VQ: {losses['content_commitment']:.4f} | "
                        f"T-KL: {losses['timbre_kl']:.4f} | "
                        f"P-KL: {losses['prosody_kl']:.4f} | "
                        f"Ortho: {losses['orthogonality']:.4f} | "
                        f"PPL: {losses['content_perplexity']:.1f} | "
                        f"LR: {lr:.2e}"
                    )
                    print(log_str)

                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    ckpt_path = self.checkpoint_dir / f'step_{self.global_step}.pt'
                    self.save_checkpoint(ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")

                # Check step limit
                if num_steps and self.global_step >= num_steps:
                    print(f"Reached {num_steps} steps, stopping.")
                    return

            # End of epoch
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train loss: {avg_losses['total']:.4f}")
            print(f"  Reconstruction: {avg_losses['reconstruction']:.4f}")
            print(f"  Content VQ: {avg_losses['content_commitment']:.4f}")
            print(f"  Timbre KL: {avg_losses['timbre_kl']:.4f}")
            print(f"  Prosody KL: {avg_losses['prosody_kl']:.4f}")
            print(f"  Orthogonality: {avg_losses['orthogonality']:.4f}")
            print(f"  Perplexity: {avg_losses['content_perplexity']:.1f}")

            # Validation
            if self.val_dataloader is not None:
                val_losses = self.validate()
                print(f"  Val loss: {val_losses['total']:.4f}")

                # Save best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    ckpt_path = self.checkpoint_dir / f'epoch_{epoch + 1}.pt'
                    self.save_checkpoint(ckpt_path, is_best=True)
                    print(f"  New best model saved! (loss: {self.best_val_loss:.4f})")

            print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train SpeechTripleNet VAE')
    parser.add_argument('--config', type=str, help='Path to config YAML')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create config
    if args.config and os.path.exists(args.config):
        yaml_config = load_config(args.config)
        config = SpeechTripleNetConfig(**yaml_config.get('model', {}))
    else:
        config = SpeechTripleNetConfig()

    print(f"Configuration:")
    print(f"  Content codebook: {config.content_codebook_size} codes")
    print(f"  Timbre latent: {config.timbre_latent_dim}-dim")
    print(f"  Prosody latent: {config.prosody_latent_dim}-dim")
    print(f"  KL weight: {config.kl_weight}")

    # Create datasets
    if args.test:
        print("\nRunning in TEST mode with synthetic data...")
        train_dataset = SyntheticMelDataset(num_samples=100, seq_len=100)
        val_dataset = SyntheticMelDataset(num_samples=20, seq_len=100)
        args.epochs = 2
    else:
        # Load from manifest
        if args.config and os.path.exists(args.config):
            yaml_config = load_config(args.config)
            train_manifest = yaml_config.get('data', {}).get('train_manifest', '../data/train_manifest.json')
            val_manifest = yaml_config.get('data', {}).get('val_manifest', '../data/val_manifest.json')
        else:
            train_manifest = '../data/train_manifest.json'
            val_manifest = '../data/val_manifest.json'

        if os.path.exists(train_manifest):
            train_dataset = MelSpectrogramDataset(train_manifest)
            val_dataset = MelSpectrogramDataset(val_manifest) if os.path.exists(val_manifest) else None
        else:
            print(f"Warning: Manifest not found at {train_manifest}, using synthetic data")
            train_dataset = SyntheticMelDataset(num_samples=1000, seq_len=100)
            val_dataset = SyntheticMelDataset(num_samples=200, seq_len=100)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    # Create trainer
    trainer = SpeechTripleNetTrainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.lr,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    trainer.train(num_epochs=args.epochs)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()
