#!/usr/bin/env python3
"""
Training script for Soft Frequency-Band Disentanglement Codec.

Based on "Soft Frequency-Band Disentanglement for Neural Audio Codecs"
(EUSIPCO 2025) - arXiv:2510.03735

Usage:
    # Train soft frequency-band codec
    python train_soft_freq_band.py --config config/soft_freq_band.yaml

    # Resume from checkpoint
    python train_soft_freq_band.py --config config/soft_freq_band.yaml \\
      --resume ../checkpoints/soft_freq_band/latest.pt

    # Test mode (synthetic data)
    python train_soft_freq_band.py --test
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from soft_freq_band_codec import (
    SoftFreqBandConfig,
    SoftFreqBandCodec,
    SoftFreqBandLoss,
    SoftFreqBandAdapter,
    compute_bitrate,
    analyze_frequency_separation,
)


# =============================================================================
# DATASET
# =============================================================================

class SoftFreqBandDataset(Dataset):
    """
    Dataset for Soft Frequency-Band Disentanglement Codec training.

    Loads audio files and returns waveform segments.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 32000,
        segment_seconds: float = 2.0,
        augment: bool = True,
    ):
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.augment = augment

        # Load manifest
        self.samples = []

        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                if isinstance(manifest, list):
                    self.samples = manifest
                elif 'samples' in manifest:
                    self.samples = manifest['samples']

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio_path = sample.get('audio_path', sample.get('path', ''))

        try:
            import torchaudio
            audio, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)

            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            audio = audio.squeeze(0)

        except Exception as e:
            # Return random audio if file loading fails
            print(f"Warning: Could not load {audio_path}: {e}")
            audio = torch.randn(self.segment_samples) * 0.1

        # Segment or pad
        if audio.shape[0] >= self.segment_samples:
            # Random crop
            start = torch.randint(0, audio.shape[0] - self.segment_samples + 1, (1,)).item()
            audio = audio[start:start + self.segment_samples]
        else:
            # Pad
            padding = self.segment_samples - audio.shape[0]
            audio = F.pad(audio, (0, padding))

        # Augmentation
        if self.augment:
            # Random gain
            gain = torch.empty(1).uniform_(0.8, 1.2).item()
            audio = audio * gain

            # Random noise
            noise_level = torch.empty(1).uniform_(0.0, 0.01).item()
            audio = audio + torch.randn_like(audio) * noise_level

        return {
            'audio': audio,
            'idx': idx,
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        sample_rate: int = 32000,
        segment_seconds: float = 2.0,
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic audio with multiple frequency components
        t = torch.linspace(0, self.segment_samples / self.sample_rate, self.segment_samples)

        # Low-frequency component (0-4kHz)
        lf_freq = torch.randint(100, 2000, (1,)).item()
        lf = 0.5 * torch.sin(2 * math.pi * lf_freq * t)

        # Mid-frequency component (4-8kHz)
        mf_freq = torch.randint(4000, 8000, (1,)).item()
        mf = 0.3 * torch.sin(2 * math.pi * mf_freq * t)

        # High-frequency component (8-16kHz)
        hf_freq = torch.randint(8000, 14000, (1,)).item()
        hf = 0.2 * torch.sin(2 * math.pi * hf_freq * t)

        # Combine
        audio = lf + mf + hf

        # Add noise
        audio = audio + torch.randn_like(audio) * 0.01

        return {
            'audio': audio,
            'idx': idx,
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    config: SoftFreqBandConfig,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': config.__dict__,
    }, checkpoint_path)

    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    return checkpoint


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: SoftFreqBandCodec,
    loss_fn: SoftFreqBandLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    config: SoftFreqBandConfig,
    log_interval: int = 50,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_band_recon = 0.0
    total_full_recon = 0.0
    total_commitment = 0.0
    total_spectral = 0.0
    total_cross_band = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        audio = batch['audio'].to(device)

        # Forward pass
        output = model(audio)
        losses = loss_fn(output, audio)

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Accumulate losses
        total_loss += losses['total'].item()
        total_band_recon += losses['band_reconstruction'].item()
        total_full_recon += losses['full_reconstruction'].item()
        total_commitment += losses['commitment'].item()
        total_spectral += losses['spectral'].item()
        total_cross_band += losses['cross_band'].item()
        num_batches += 1

        # Logging
        if batch_idx % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {losses['total'].item():.4f} "
                  f"BandRecon: {losses['band_reconstruction'].item():.4f} "
                  f"FullRecon: {losses['full_reconstruction'].item():.4f} "
                  f"Commit: {losses['commitment'].item():.4f} "
                  f"Spectral: {losses['spectral'].item():.4f} "
                  f"CrossBand: {losses['cross_band'].item():.4f} "
                  f"Perplexity: {losses['mean_perplexity'].item():.2f} "
                  f"LR: {lr:.2e}")

    return {
        'loss': total_loss / num_batches,
        'band_reconstruction': total_band_recon / num_batches,
        'full_reconstruction': total_full_recon / num_batches,
        'commitment': total_commitment / num_batches,
        'spectral': total_spectral / num_batches,
        'cross_band': total_cross_band / num_batches,
    }


def validate(
    model: SoftFreqBandCodec,
    loss_fn: SoftFreqBandLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_band_recon = 0.0
    total_full_recon = 0.0
    total_disentanglement = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            audio = batch['audio'].to(device)

            output = model(audio)
            losses = loss_fn(output, audio)

            total_loss += losses['total'].item()
            total_band_recon += losses['band_reconstruction'].item()
            total_full_recon += losses['full_reconstruction'].item()
            num_batches += 1

            # Analyze frequency separation
            sep_analysis = analyze_frequency_separation(model, audio)
            total_disentanglement += sep_analysis['disentanglement_score'].item()

    return {
        'loss': total_loss / num_batches,
        'band_reconstruction': total_band_recon / num_batches,
        'full_reconstruction': total_full_recon / num_batches,
        'disentanglement_score': total_disentanglement / num_batches,
    }


def train(
    config: SoftFreqBandConfig,
    manifest_path: str,
    checkpoint_dir: str,
    resume_path: Optional[str] = None,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    log_interval: int = 50,
    save_interval: int = 5,
    val_split: float = 0.1,
    use_synthetic: bool = False,
):
    """Main training function."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    if use_synthetic:
        dataset = SyntheticDataset(
            num_samples=1000,
            sample_rate=config.sample_rate,
        )
    else:
        dataset = SoftFreqBandDataset(
            manifest_path=manifest_path,
            sample_rate=config.sample_rate,
        )

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = SoftFreqBandCodec(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Print bitrate
    bitrate = compute_bitrate(config)
    print(f"Theoretical bitrate: {bitrate['total_kbps']:.2f} kbps")

    # Loss function
    loss_fn = SoftFreqBandLoss(config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Scheduler
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')

    if resume_path and os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('loss', float('inf'))

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, epoch, config, log_interval,
        )

        # Validate
        val_metrics = validate(model, loss_fn, val_loader, device)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Disentanglement: {val_metrics['disentanglement_score']:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 0,
                val_metrics['loss'], config, checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt",
            )

        # Save best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, 0,
                val_metrics['loss'], config, checkpoint_dir,
                "best.pt",
            )

        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch, 0,
            val_metrics['loss'], config, checkpoint_dir,
            "latest.pt",
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Soft Frequency-Band Disentanglement Codec"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/soft_freq_band.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="../data/manifest.json",
        help="Path to training manifest",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="../checkpoints/soft_freq_band",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with synthetic data",
    )

    args = parser.parse_args()

    # Load or create config
    config = SoftFreqBandConfig()

    if os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            print(f"Loaded config from {args.config}")
        except ImportError:
            print("Warning: PyYAML not installed, using default config")

    # Test mode
    if args.test:
        print("Running in test mode with synthetic data...")
        train(
            config=config,
            manifest_path="",
            checkpoint_dir=args.checkpoint_dir,
            resume_path=args.resume,
            num_epochs=3,
            batch_size=4,
            learning_rate=args.lr,
            warmup_steps=10,
            log_interval=5,
            save_interval=1,
            use_synthetic=True,
        )
    else:
        train(
            config=config,
            manifest_path=args.manifest,
            checkpoint_dir=args.checkpoint_dir,
            resume_path=args.resume,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_steps=1000,
            log_interval=50,
            save_interval=5,
            use_synthetic=False,
        )


if __name__ == "__main__":
    main()
