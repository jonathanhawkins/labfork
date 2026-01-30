#!/usr/bin/env python3
"""
FreeCodec Training Script

Trains the parallel-encoder disentangled speech codec based on arXiv:2412.01053.

Three training phases:
1. Warmup: Train content encoder with semantic targets
2. Joint: Train all encoders + decoder together
3. Fine-tune: Optional discriminator training

Usage:
    # Train from scratch
    python train_freecodec.py --config config/freecodec.yaml

    # Resume training
    python train_freecodec.py --config config/freecodec.yaml \
        --resume ../checkpoints/freecodec/latest.pt

    # Train specific variant
    python train_freecodec.py --config config/freecodec.yaml --variant v3

    # Test mode (synthetic data)
    python train_freecodec.py --test
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, Dataset

import yaml

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))

from freecodec import (
    FreeCodec,
    FreeCodecAdapter,
    FreeCodecConfig,
    FreeCodecLoss,
    FreeCodecVsFACodecComparison,
)


# =============================================================================
# DATASET
# =============================================================================

class FreeCodecDataset(Dataset):
    """
    Dataset for FreeCodec training.

    Loads mel spectrograms and optional pre-extracted HuBERT features
    for semantic learning targets.
    """

    def __init__(
        self,
        manifest_path: str,
        mel_dir: str,
        semantic_dir: Optional[str] = None,
        max_seq_len: int = 500,
        min_seq_len: int = 50,
    ):
        self.mel_dir = Path(mel_dir)
        self.semantic_dir = Path(semantic_dir) if semantic_dir else None
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Filter by valid length
        self.samples = []
        for item in self.manifest:
            if 'mel_path' in item:
                self.samples.append(item)

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        # Load mel spectrogram
        mel_path = self.mel_dir / item['mel_path']
        mel = torch.load(mel_path, weights_only=True)  # [seq, mel_dim]

        # Truncate if too long
        if mel.shape[0] > self.max_seq_len:
            start = torch.randint(0, mel.shape[0] - self.max_seq_len, (1,)).item()
            mel = mel[start:start + self.max_seq_len]

        # Load semantic features if available
        semantic = None
        if self.semantic_dir is not None and 'semantic_path' in item:
            semantic_path = self.semantic_dir / item['semantic_path']
            if semantic_path.exists():
                semantic = torch.load(semantic_path, weights_only=True)
                # Align length with mel
                if semantic.shape[0] > mel.shape[0]:
                    semantic = semantic[:mel.shape[0]]
                elif semantic.shape[0] < mel.shape[0]:
                    # Pad semantic
                    pad_len = mel.shape[0] - semantic.shape[0]
                    semantic = F.pad(semantic, (0, 0, 0, pad_len))

        result = {
            'mel': mel,
            'length': mel.shape[0],
        }

        if semantic is not None:
            result['semantic'] = semantic

        return result


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 100,
        mel_dim: int = 80,
        semantic_dim: int = 768,
        max_seq_len: int = 200,
    ):
        self.num_samples = num_samples
        self.mel_dim = mel_dim
        self.semantic_dim = semantic_dim
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random length
        length = torch.randint(50, self.max_seq_len, (1,)).item()

        return {
            'mel': torch.randn(length, self.mel_dim),
            'semantic': torch.randn(length, self.semantic_dim),
            'length': length,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    # Find max length
    max_len = max(item['length'] for item in batch)

    # Pad sequences
    mels = []
    semantics = []
    lengths = []
    masks = []

    for item in batch:
        mel = item['mel']
        pad_len = max_len - mel.shape[0]

        # Pad mel
        mel_padded = F.pad(mel, (0, 0, 0, pad_len))
        mels.append(mel_padded)

        # Pad semantic if present
        if 'semantic' in item:
            semantic = item['semantic']
            semantic_padded = F.pad(semantic, (0, 0, 0, pad_len))
            semantics.append(semantic_padded)

        # Length and mask
        lengths.append(item['length'])
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[item['length']:] = True
        masks.append(mask)

    result = {
        'mel': torch.stack(mels),
        'lengths': torch.tensor(lengths),
        'mask': torch.stack(masks),
    }

    if semantics:
        result['semantic'] = torch.stack(semantics)

    return result


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """Cosine learning rate schedule with warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# TRAINER
# =============================================================================

class FreeCodecTrainer:
    """
    Trainer for FreeCodec.

    Handles:
    - Training loop with gradient accumulation
    - Mixed precision training
    - Checkpoint saving/loading
    - Validation and metrics logging
    """

    def __init__(
        self,
        config: FreeCodecConfig,
        model: FreeCodec,
        loss_fn: FreeCodecLoss,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints/freecodec",
        log_every: int = 50,
        save_every: int = 1000,
        eval_every: int = 500,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and torch.cuda.is_available()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.device = device
        self.num_epochs = num_epochs

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        total_steps = num_epochs * len(train_loader) // gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.metrics = {}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        meters = {
            'total': AverageMeter(),
            'reconstruction': AverageMeter(),
            'commitment': AverageMeter(),
            'semantic': AverageMeter(),
            'orthogonality': AverageMeter(),
            'perplexity': AverageMeter(),
        }

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            mel = batch['mel'].to(self.device)
            mask = batch['mask'].to(self.device)
            semantic = batch.get('semantic')
            if semantic is not None:
                semantic = semantic.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                output = self.model(mel, semantic, mask)
                losses = self.loss_fn(output, mel, mask=mask)

                # Scale loss for gradient accumulation
                loss = losses['total'] / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update meters
            batch_size = mel.shape[0]
            meters['total'].update(losses['total'].item(), batch_size)
            meters['reconstruction'].update(losses['reconstruction_loss'].item(), batch_size)
            meters['commitment'].update(losses['commitment_loss'].item(), batch_size)
            meters['semantic'].update(losses['semantic_loss'].item(), batch_size)
            meters['orthogonality'].update(losses['orthogonality_loss'].item(), batch_size)
            meters['perplexity'].update(losses['perplexity'].item(), batch_size)

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

                self.global_step += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"[Epoch {epoch}] Step {self.global_step} | "
                        f"Loss: {meters['total'].avg:.4f} | "
                        f"Recon: {meters['reconstruction'].avg:.4f} | "
                        f"Perplexity: {meters['perplexity'].avg:.2f} | "
                        f"LR: {lr:.2e}"
                    )

                # Checkpointing
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

                # Validation
                if self.val_loader is not None and self.global_step % self.eval_every == 0:
                    val_metrics = self.validate()
                    self.model.train()

                    if val_metrics['total'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['total']
                        self.save_checkpoint("best.pt")
                        print(f"  New best model saved! Val loss: {val_metrics['total']:.4f}")

        return {k: v.avg for k, v in meters.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        meters = {
            'total': AverageMeter(),
            'reconstruction': AverageMeter(),
            'perplexity': AverageMeter(),
        }

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            mask = batch['mask'].to(self.device)
            semantic = batch.get('semantic')
            if semantic is not None:
                semantic = semantic.to(self.device)

            output = self.model(mel, semantic, mask)
            losses = self.loss_fn(output, mel, mask=mask)

            batch_size = mel.shape[0]
            meters['total'].update(losses['total'].item(), batch_size)
            meters['reconstruction'].update(losses['reconstruction_loss'].item(), batch_size)
            meters['perplexity'].update(losses['perplexity'].item(), batch_size)

        print(
            f"  [Validation] Loss: {meters['total'].avg:.4f} | "
            f"Recon: {meters['reconstruction'].avg:.4f} | "
            f"Perplexity: {meters['perplexity'].avg:.2f}"
        )

        return {k: v.avg for k, v in meters.items()}

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

        # Also save as latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        print(f"  Saved checkpoint: {filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")

    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Total steps: {self.num_epochs * len(self.train_loader) // self.gradient_accumulation_steps}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print()

        for epoch in range(self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("=" * 60)

            train_metrics = self.train_epoch(epoch + 1)

            print(
                f"\nEpoch {epoch + 1} Summary: "
                f"Loss: {train_metrics['total']:.4f} | "
                f"Perplexity: {train_metrics['perplexity']:.2f}"
            )

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        print("\nTraining complete!")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train FreeCodec")
    parser.add_argument(
        "--config", type=str, default="config/freecodec.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--variant", type=str, default=None, choices=["v1", "v2", "v3"],
        help="FreeCodec variant to train",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode with synthetic data",
    )
    args = parser.parse_args()

    # Test mode
    if args.test:
        print("=" * 60)
        print("FreeCodec Training - Test Mode")
        print("=" * 60)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")

        # Create synthetic dataset
        train_dataset = SyntheticDataset(num_samples=50)
        val_dataset = SyntheticDataset(num_samples=10)

        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
        )

        # Create model
        config = FreeCodecConfig(variant="v2")
        model = FreeCodec(config)
        loss_fn = FreeCodecLoss(config)

        # Quick training test
        trainer = FreeCodecTrainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=1e-4,
            warmup_steps=10,
            log_every=5,
            save_every=50,
            eval_every=25,
            checkpoint_dir="/tmp/freecodec_test",
            device=device,
        )

        trainer.train()

        print("\n" + "=" * 60)
        print("Test mode completed successfully!")
        print("=" * 60)
        return

    # Load configuration
    config_dict = load_config(args.config)

    # Override variant if specified
    if args.variant:
        config_dict['model']['variant'] = args.variant

    # Create config
    model_config = config_dict['model']
    config = FreeCodecConfig(
        variant=model_config.get('variant', 'v2'),
        mel_dim=model_config.get('mel_dim', 80),
        sample_rate=model_config.get('sample_rate', 16000),
        hop_length=model_config.get('hop_length', 256),
        timbre_dim=model_config.get('timbre_dim', 256),
        timbre_hidden_dim=model_config.get('timbre_hidden_dim', 512),
        timbre_num_layers=model_config.get('timbre_num_layers', 3),
        prosody_dim=model_config.get('prosody_dim', 128),
        prosody_hidden_dim=model_config.get('prosody_hidden_dim', 256),
        prosody_num_layers=model_config.get('prosody_num_layers', 4),
        prosody_stride=model_config.get('prosody_stride', 4),
        prosody_kernel_size=model_config.get('prosody_kernel_size', 8),
        content_dim=model_config.get('content_dim', 256),
        content_hidden_dim=model_config.get('content_hidden_dim', 512),
        content_num_layers=model_config.get('content_num_layers', 6),
        content_codebook_size=model_config.get('content_codebook_size', 1024),
        content_commitment_cost=model_config.get('content_commitment_cost', 0.25),
        content_ema_decay=model_config.get('content_ema_decay', 0.99),
        semantic_dim=model_config.get('semantic_dim', 768),
        use_semantic_target=model_config.get('use_semantic_target', True),
        semantic_weight=model_config.get('semantic_weight', 1.0),
        decoder_hidden_dim=model_config.get('decoder_hidden_dim', 512),
        decoder_num_layers=model_config.get('decoder_num_layers', 6),
        decoder_num_heads=model_config.get('decoder_num_heads', 8),
        decoder_ffn_dim=model_config.get('decoder_ffn_dim', 2048),
        dropout=model_config.get('dropout', 0.1),
        use_orthogonality=model_config.get('use_orthogonality', True),
        ortho_weight=model_config.get('ortho_weight', 0.01),
        use_discriminator=model_config.get('use_discriminator', False),
        output_dim=model_config.get('output_dim', 2048),
        num_prefix_tokens=model_config.get('num_prefix_tokens', 4),
    )

    print("=" * 60)
    print(f"FreeCodec Training - Variant: {config.variant}")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Timbre dim: {config.timbre_dim}")
    print(f"  Prosody dim: {config.prosody_dim} (stride={config.prosody_stride})")
    print(f"  Content codebook: {config.content_codebook_size}")
    print(f"  Semantic target: {config.use_semantic_target}")
    print()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create datasets
    data_config = config_dict['data']
    train_dataset = FreeCodecDataset(
        manifest_path=data_config['manifest_path'],
        mel_dir=data_config['mel_dir'],
        semantic_dir=data_config.get('semantic_dir'),
        max_seq_len=data_config.get('max_seq_len', 500),
        min_seq_len=data_config.get('min_seq_len', 50),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict['training']['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collate_fn,
    )

    # Create model
    model = FreeCodec(config)
    loss_fn = FreeCodecLoss(config)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,} total, {num_trainable:,} trainable")

    # Training config
    train_config = config_dict['training']

    # Create trainer
    trainer = FreeCodecTrainer(
        config=config,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=None,  # Optional validation loader
        learning_rate=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 1000),
        num_epochs=train_config.get('num_epochs', 100),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 1),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        use_amp=train_config.get('use_amp', True),
        checkpoint_dir=config_dict['paths'].get('checkpoint_dir', 'checkpoints/freecodec'),
        log_every=config_dict['logging'].get('log_every', 50),
        save_every=train_config.get('save_every', 1000),
        eval_every=train_config.get('eval_every', 500),
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
