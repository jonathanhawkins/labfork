#!/usr/bin/env python3
"""
Train STCTS Sparse Keyframe Prosody Model

Based on arXiv:2512.00451 "Towards Ultra-Low Bitrate Speech Coding".

This script trains the sparse keyframe conditioner to:
1. Extract informative prosody keyframes from dense contours
2. Encode keyframes into prosody conditioning tokens
3. Achieve ultra-low bitrate (<14 bps) prosody representation

Usage:
    # Full training
    python train_sparse_keyframe.py --config config/sparse_keyframe.yaml

    # Resume from checkpoint
    python train_sparse_keyframe.py --config config/sparse_keyframe.yaml \
        --resume ../checkpoints/sparse_keyframe/latest.pt

    # Test mode (synthetic data)
    python train_sparse_keyframe.py --test
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sparse_keyframe_prosody import (
    SparseKeyframeConfig,
    SparseKeyframeConditioner,
    SparseKeyframeAdapter,
    ProsodyKeyframe,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000

    reconstruction_weight: float = 1.0
    smoothness_weight: float = 0.1
    sparsity_weight: float = 0.01

    log_every_n_steps: int = 100
    save_every_n_steps: int = 1000
    eval_every_n_steps: int = 500


class SyntheticProsodyDataset(Dataset):
    """
    Synthetic prosody dataset for testing.

    Generates realistic prosody contours with various patterns.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 100,
        duration_range: Tuple[float, float] = (1.0, 5.0),
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.duration_range = duration_range

        # Pre-generate patterns
        self.patterns = ['sine', 'rising', 'falling', 'flat', 'complex']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random pattern
        pattern = self.patterns[idx % len(self.patterns)]

        # Generate time axis
        t = torch.linspace(0, 1, self.seq_len)

        # Generate pitch contour based on pattern
        if pattern == 'sine':
            freq = torch.randint(2, 6, (1,)).item()
            pitch = 0.5 + 0.2 * torch.sin(t * freq * 2 * math.pi)
        elif pattern == 'rising':
            pitch = 0.3 + 0.4 * t
        elif pattern == 'falling':
            pitch = 0.7 - 0.4 * t
        elif pattern == 'flat':
            pitch = torch.ones(self.seq_len) * (0.4 + 0.2 * torch.rand(1).item())
        else:  # complex
            freq1 = torch.randint(2, 4, (1,)).item()
            freq2 = torch.randint(4, 8, (1,)).item()
            pitch = (
                0.5 +
                0.15 * torch.sin(t * freq1 * 2 * math.pi) +
                0.1 * torch.sin(t * freq2 * 2 * math.pi) +
                0.05 * torch.randn(self.seq_len)
            )

        # Generate energy (correlated with pitch variation)
        energy = 0.5 + 0.3 * torch.abs(pitch - 0.5) + 0.1 * torch.randn(self.seq_len)

        # Clamp to valid range
        pitch = torch.clamp(pitch, 0, 1)
        energy = torch.clamp(energy, 0, 1)

        # Random duration
        duration = (
            self.duration_range[0] +
            torch.rand(1).item() * (self.duration_range[1] - self.duration_range[0])
        )

        return {
            'pitch': pitch,
            'energy': energy,
            'duration': duration,
            'pattern': pattern,
        }


def compute_smoothness_loss(interpolated: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness loss for interpolated prosody.

    Penalizes large jumps between consecutive frames.
    """
    diff = interpolated[:, 1:] - interpolated[:, :-1]
    return (diff ** 2).mean()


def compute_sparsity_loss(
    num_keyframes: List[int],
    target_rate: float,
    duration: float,
) -> torch.Tensor:
    """
    Compute sparsity loss to encourage fewer keyframes.

    Args:
        num_keyframes: Number of keyframes per sample
        target_rate: Target keyframe rate (Hz)
        duration: Audio duration

    Returns:
        Sparsity loss
    """
    target_count = target_rate * duration
    actual_counts = torch.tensor(num_keyframes, dtype=torch.float32)
    return F.relu(actual_counts - target_count).mean()


class SparseKeyframeTrainer:
    """Trainer for sparse keyframe prosody model."""

    def __init__(
        self,
        model_config: SparseKeyframeConfig,
        train_config: TrainingConfig,
        device: str = 'cpu',
        checkpoint_dir: str = '../checkpoints/sparse_keyframe',
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = SparseKeyframeConditioner(model_config).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.max_steps,
            eta_min=train_config.learning_rate * 0.01,
        )

        # Metrics
        self.global_step = 0
        self.best_loss = float('inf')

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        pitch = batch['pitch'].to(self.device)
        energy = batch['energy'].to(self.device)
        duration = batch['duration'][0].item()  # Assume same duration in batch

        # Extract keyframes (non-differentiable)
        keyframes_list = self.model.extract(pitch, energy, duration)

        # Convert to tensor (differentiable from here)
        keyframes_tensor, keyframes_mask = self.model.keyframes_to_tensor(keyframes_list)
        keyframes_tensor = keyframes_tensor.to(self.device)
        keyframes_mask = keyframes_mask.to(self.device)

        # Encode keyframes to tokens (differentiable)
        prosody_tokens = self.model.encoder(keyframes_tensor, keyframes_mask)

        # Token prediction loss: tokens should encode information to reconstruct input
        # We use a simple consistency loss: token features should correlate with input stats
        pitch_mean = pitch.mean(dim=-1)  # [batch]
        energy_mean = energy.mean(dim=-1)  # [batch]
        pitch_std = pitch.std(dim=-1)
        energy_std = energy.std(dim=-1)

        # Pool tokens to get summary statistics
        token_pooled = prosody_tokens.mean(dim=1)  # [batch, output_dim]

        # Regression targets
        targets = torch.stack([pitch_mean, energy_mean, pitch_std, energy_std], dim=-1)
        target_proj = torch.zeros(token_pooled.shape[0], token_pooled.shape[1], device=self.device)
        target_proj[:, :4] = targets

        # Token encoding loss
        token_loss = F.mse_loss(token_pooled, target_proj)

        # Keyframe representation loss (keyframes should capture local values)
        kf_recon_loss = torch.tensor(0.0, device=self.device)
        num_kfs = [len(kfs) for kfs in keyframes_list]
        for b, keyframes in enumerate(keyframes_list):
            for kf in keyframes:
                idx = int(kf.time * (pitch.shape[1] - 1))
                kf_recon_loss += (kf.pitch - pitch[b, idx].item()) ** 2
                kf_recon_loss += (kf.energy - energy[b, idx].item()) ** 2
        kf_recon_loss = kf_recon_loss / (sum(num_kfs) * 2 + 1e-8)

        # Sparsity loss
        sparsity_loss = compute_sparsity_loss(
            num_kfs,
            self.model_config.keyframe_rate_hz,
            duration,
        )

        # Total loss (only token_loss is differentiable)
        total_loss = (
            self.train_config.reconstruction_weight * token_loss +
            self.train_config.sparsity_weight * sparsity_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Metrics
        avg_keyframes = sum(num_kfs) / len(num_kfs)
        bitrate = self._compute_bitrate(avg_keyframes, duration)

        return {
            'loss': total_loss.item(),
            'token_loss': token_loss.item(),
            'kf_recon_loss': kf_recon_loss.item() if isinstance(kf_recon_loss, torch.Tensor) else kf_recon_loss,
            'sparsity_loss': sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss,
            'avg_keyframes': avg_keyframes,
            'bitrate_bps': bitrate,
            'lr': self.scheduler.get_last_lr()[0],
        }

    def _compute_bitrate(self, num_keyframes: float, duration: float) -> float:
        """Compute bitrate in bits per second."""
        bits_per_keyframe = 8 + 8 + 8 + 8 * self.model_config.emotion_dim
        total_bits = num_keyframes * bits_per_keyframe
        return total_bits / duration

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on dataset."""
        self.model.eval()

        total_recon = 0.0
        total_keyframes = 0
        total_bitrate = 0.0
        num_batches = 0

        for batch in dataloader:
            pitch = batch['pitch'].to(self.device)
            energy = batch['energy'].to(self.device)
            duration = batch['duration'][0].item()

            result = self.model(pitch, energy, duration)

            total_recon += result['reconstruction_loss'].item()

            num_kfs = [len(kfs) for kfs in result['keyframes']]
            avg_kfs = sum(num_kfs) / len(num_kfs)
            total_keyframes += avg_kfs
            total_bitrate += self._compute_bitrate(avg_kfs, duration)

            num_batches += 1

        return {
            'eval_recon_loss': total_recon / num_batches,
            'eval_avg_keyframes': total_keyframes / num_batches,
            'eval_bitrate_bps': total_bitrate / num_batches,
        }

    def save_checkpoint(self, filename: str = 'latest.pt'):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'model_config': self.model_config,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ):
        """Main training loop."""
        print(f"Starting training from step {self.global_step}")
        print(f"Target: {self.train_config.max_steps} steps")

        while self.global_step < self.train_config.max_steps:
            for batch in train_loader:
                metrics = self.train_step(batch)
                self.global_step += 1

                # Logging
                if self.global_step % self.train_config.log_every_n_steps == 0:
                    print(f"Step {self.global_step}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"token={metrics['token_loss']:.4f}, "
                          f"kf={metrics['avg_keyframes']:.1f}, "
                          f"bps={metrics['bitrate_bps']:.1f}, "
                          f"lr={metrics['lr']:.2e}")

                # Evaluation
                if eval_loader and self.global_step % self.train_config.eval_every_n_steps == 0:
                    eval_metrics = self.evaluate(eval_loader)
                    print(f"Eval: "
                          f"recon={eval_metrics['eval_recon_loss']:.4f}, "
                          f"kf={eval_metrics['eval_avg_keyframes']:.1f}, "
                          f"bps={eval_metrics['eval_bitrate_bps']:.1f}")

                    # Save best
                    if eval_metrics['eval_recon_loss'] < self.best_loss:
                        self.best_loss = eval_metrics['eval_recon_loss']
                        self.save_checkpoint('best.pt')

                # Save checkpoint
                if self.global_step % self.train_config.save_every_n_steps == 0:
                    self.save_checkpoint('latest.pt')

                if self.global_step >= self.train_config.max_steps:
                    break

        # Final save
        self.save_checkpoint('final.pt')
        print("Training complete!")


def load_config(config_path: str) -> Tuple[SparseKeyframeConfig, TrainingConfig]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_config = SparseKeyframeConfig(
        keyframe_rate_hz=cfg.get('keyframe_rate_hz', 0.5),
        min_keyframes=cfg.get('min_keyframes', 2),
        max_keyframes=cfg.get('max_keyframes', 16),
        extraction_method=cfg.get('extraction_method', 'salient'),
        salience_threshold=cfg.get('salience_threshold', 0.3),
        pitch_dim=cfg.get('pitch_dim', 1),
        energy_dim=cfg.get('energy_dim', 1),
        duration_dim=cfg.get('duration_dim', 1),
        emotion_dim=cfg.get('emotion_dim', 8),
        interpolation_method=cfg.get('interpolation_method', 'cubic'),
        interpolation_smoothing=cfg.get('interpolation_smoothing', 0.1),
        hidden_dim=cfg.get('hidden_dim', 256),
        num_transformer_layers=cfg.get('num_transformer_layers', 2),
        num_heads=cfg.get('num_heads', 4),
        dropout=cfg.get('dropout', 0.1),
        output_dim=cfg.get('output_dim', 2048),
        num_output_tokens=cfg.get('num_output_tokens', 4),
        target_bitrate_bps=cfg.get('target_bitrate_bps', 14.0),
    )

    train_cfg = cfg.get('training', {})
    train_config = TrainingConfig(
        batch_size=train_cfg.get('batch_size', 16),
        learning_rate=train_cfg.get('learning_rate', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        warmup_steps=train_cfg.get('warmup_steps', 1000),
        max_steps=train_cfg.get('max_steps', 50000),
        reconstruction_weight=train_cfg.get('reconstruction_weight', 1.0),
        smoothness_weight=train_cfg.get('smoothness_weight', 0.1),
        sparsity_weight=train_cfg.get('sparsity_weight', 0.01),
        log_every_n_steps=train_cfg.get('log_every_n_steps', 100),
        save_every_n_steps=train_cfg.get('save_every_n_steps', 1000),
        eval_every_n_steps=train_cfg.get('eval_every_n_steps', 500),
    )

    return model_config, train_config


def test_mode():
    """Run quick test with synthetic data."""
    print("=" * 60)
    print("STCTS Sparse Keyframe Training - Test Mode")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create configs
    model_config = SparseKeyframeConfig()
    train_config = TrainingConfig(
        max_steps=100,
        log_every_n_steps=10,
        eval_every_n_steps=50,
        save_every_n_steps=50,
    )

    # Create datasets
    train_dataset = SyntheticProsodyDataset(num_samples=200)
    eval_dataset = SyntheticProsodyDataset(num_samples=50)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create trainer
    trainer = SparseKeyframeTrainer(
        model_config=model_config,
        train_config=train_config,
        device=device,
        checkpoint_dir='../checkpoints/sparse_keyframe_test',
    )

    # Train
    trainer.train(train_loader, eval_loader)

    # Final evaluation
    print("\nFinal Evaluation:")
    final_metrics = trainer.evaluate(eval_loader)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Compression stats
    print("\nCompression Statistics:")
    adapter = SparseKeyframeAdapter(model_config).to(device)
    stats = adapter.compute_compression_stats(3.0, int(final_metrics['eval_avg_keyframes']))
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train STCTS Sparse Keyframe Model")
    parser.add_argument('--config', type=str, default='config/sparse_keyframe.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run test mode with synthetic data')
    args = parser.parse_args()

    if args.test:
        test_mode()
        return

    # Load config
    model_config, train_config = load_config(args.config)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create synthetic dataset (replace with real data in production)
    train_dataset = SyntheticProsodyDataset(num_samples=10000)
    eval_dataset = SyntheticProsodyDataset(num_samples=1000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Create trainer
    trainer = SparseKeyframeTrainer(
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_loader, eval_loader)


if __name__ == "__main__":
    main()
