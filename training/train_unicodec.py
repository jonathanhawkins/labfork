#!/usr/bin/env python3
"""
Train UniCodec: Universal Speech Token Learning

Based on UniCodec (arXiv:2503.12115, IEEE J-STSP 2025).
Trains a neural codec that unifies semantic and acoustic tokens.

Two-Stage Training:
1. Stage 1: Token Learning - Train codec with SSL distillation
2. Stage 2: Token Generation - Train LM to generate tokens from text (optional)

Usage:
    # Train UniCodec (Stage 1)
    python train_unicodec.py --config config/unicodec.yaml

    # Resume training
    python train_unicodec.py --config config/unicodec.yaml \
        --resume ../checkpoints/unicodec/latest.pt

    # Test mode (synthetic data)
    python train_unicodec.py --test
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unicodec import (
    UniCodecConfig,
    UniCodec,
    UniCodecLoss,
    UniCodecAdapter,
    SSLFeatureExtractor,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class UniCodecDataset(Dataset):
    """
    Dataset for UniCodec training.

    Expects manifest with audio paths and optional mel/SSL features.
    """

    def __init__(
        self,
        manifest_path: str,
        mel_dim: int = 80,
        sample_rate: int = 16000,
        max_len: int = 500,
        precomputed_dir: Optional[str] = None,
    ):
        self.mel_dim = mel_dim
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.precomputed_dir = Path(precomputed_dir) if precomputed_dir else None

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        logger.info(f"Loaded {len(self.manifest)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]

        # Try loading precomputed features
        if self.precomputed_dir is not None:
            item_id = item.get('id', str(idx))
            feature_path = self.precomputed_dir / f"{item_id}.pt"

            if feature_path.exists():
                features = torch.load(feature_path)
                return {
                    'mel': features['mel'][:self.max_len],
                    'ssl_target': features.get('ssl_target', torch.zeros(1))[:self.max_len],
                    'audio': features.get('audio', torch.zeros(1)),
                }

        # Load audio and compute features on the fly
        try:
            import torchaudio

            audio_path = item.get('audio_path', item.get('path'))
            audio, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            # Mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            audio = audio.squeeze(0)

            # Compute mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=320,
                n_mels=self.mel_dim,
            )
            mel = mel_transform(audio.unsqueeze(0)).squeeze(0)
            mel = torch.log(mel.clamp(min=1e-5)).transpose(0, 1)  # [time, mel_dim]

            # Truncate
            mel = mel[:self.max_len]

            return {
                'mel': mel,
                'ssl_target': torch.zeros(mel.shape[0], 768),  # Placeholder
                'audio': audio,
            }

        except Exception as e:
            logger.warning(f"Failed to load {item}: {e}")
            # Return dummy data
            return {
                'mel': torch.randn(self.max_len, self.mel_dim),
                'ssl_target': torch.randn(self.max_len, 768),
                'audio': torch.randn(self.sample_rate * 5),  # 5 seconds
            }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, size: int = 100, seq_len: int = 100, mel_dim: int = 80):
        self.size = size
        self.seq_len = seq_len
        self.mel_dim = mel_dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'mel': torch.randn(self.seq_len, self.mel_dim),
            'ssl_target': torch.randn(self.seq_len, 768),
            'audio': torch.randn(16000 * 2),  # 2 seconds
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    mel_list = [b['mel'] for b in batch]
    ssl_list = [b['ssl_target'] for b in batch]

    # Pad to max length in batch
    max_mel_len = max(m.shape[0] for m in mel_list)
    max_ssl_len = max(s.shape[0] for s in ssl_list)
    max_len = max(max_mel_len, max_ssl_len)

    mel_padded = []
    ssl_padded = []
    mask = []

    for mel, ssl in zip(mel_list, ssl_list):
        mel_len = mel.shape[0]
        ssl_len = ssl.shape[0]

        # Pad mel
        if mel_len < max_len:
            mel = F.pad(mel, (0, 0, 0, max_len - mel_len))
        mel_padded.append(mel[:max_len])

        # Pad SSL
        if ssl_len < max_len:
            ssl = F.pad(ssl, (0, 0, 0, max_len - ssl_len))
        ssl_padded.append(ssl[:max_len])

        # Create mask (True = valid, False = padded)
        m = torch.ones(max_len, dtype=torch.bool)
        m[mel_len:] = False
        mask.append(m)

    return {
        'mel': torch.stack(mel_padded),
        'ssl_target': torch.stack(ssl_padded),
        'mask': torch.stack(mask),
    }


# =============================================================================
# TRAINER
# =============================================================================

class UniCodecTrainer:
    """Trainer for UniCodec."""

    def __init__(
        self,
        config: UniCodecConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        checkpoint_dir: str = "../checkpoints/unicodec",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        use_ssl_extraction: bool = False,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = UniCodec(config).to(device)
        self.loss_fn = UniCodecLoss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # Learning rate scheduler with warmup
        pct_start = min(warmup_steps / max_steps, 0.3)  # Cap at 30% warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=max_steps,
            pct_start=pct_start,
        )

        # SSL feature extractor (optional)
        if use_ssl_extraction:
            self.ssl_extractor = SSLFeatureExtractor(
                model_name=config.ssl_model,
                target_layer=config.ssl_layer,
            ).to(device)
        else:
            self.ssl_extractor = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Metrics tracking
        self.train_losses = []

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self):
        """Main training loop."""
        logger.info("Starting UniCodec training...")

        self.model.train()
        train_iter = iter(self.train_loader)

        while self.global_step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move to device
            mel = batch['mel'].to(self.device)
            ssl_target = batch['ssl_target'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = ~mask.to(self.device)  # Invert for attention mask

            # Forward pass
            output = self.model(mel, ssl_target, mask)
            losses = self.loss_fn(output, mel, ssl_target, mask)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            self.train_losses.append(losses['total'].item())
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                avg_loss = sum(self.train_losses[-self.log_interval:]) / self.log_interval
                lr = self.scheduler.get_last_lr()[0]

                logger.info(
                    f"Step {self.global_step}/{self.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Recon: {losses['reconstruction'].item():.4f} | "
                    f"Distill: {losses['distillation'].item():.4f} | "
                    f"Sem Perp: {losses['semantic_perplexity'].item():.2f} | "
                    f"LR: {lr:.2e}"
                )

            # Evaluation
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                logger.info(f"Step {self.global_step} | Validation Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")

                self.model.train()

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
                self.save_checkpoint("latest.pt")

        logger.info("Training complete!")
        self.save_checkpoint("final.pt")

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            ssl_target = batch['ssl_target'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = ~mask.to(self.device)

            output = self.model(mel, ssl_target, mask)
            losses = self.loss_fn(output, mel, ssl_target, mask)

            total_loss += losses['total'].item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train UniCodec")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--manifest", type=str, default=None, help="Path to training manifest")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Run in test mode with synthetic data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Load config
    config_dict = {}
    if args.config:
        config_dict = load_config(args.config)

    # Create UniCodec config
    config = UniCodecConfig(**config_dict.get('model', {}))

    # Create datasets
    if args.test:
        logger.info("Running in TEST mode with synthetic data")
        train_dataset = SyntheticDataset(size=100, seq_len=100, mel_dim=config.mel_dim)
        val_dataset = SyntheticDataset(size=20, seq_len=100, mel_dim=config.mel_dim)
        max_steps = 500
    else:
        manifest_path = args.manifest or config_dict.get('data', {}).get('train_manifest')
        if manifest_path is None:
            raise ValueError("Must provide --manifest or set data.train_manifest in config")

        train_dataset = UniCodecDataset(
            manifest_path=manifest_path,
            mel_dim=config.mel_dim,
            precomputed_dir=config_dict.get('data', {}).get('precomputed_dir'),
        )

        val_manifest = config_dict.get('data', {}).get('val_manifest')
        if val_manifest:
            val_dataset = UniCodecDataset(
                manifest_path=val_manifest,
                mel_dim=config.mel_dim,
                precomputed_dir=config_dict.get('data', {}).get('precomputed_dir'),
            )
        else:
            val_dataset = None

        max_steps = args.max_steps

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if not args.test else 0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if 'val_dataset' in dir() and val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2 if not args.test else 0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Create trainer
    trainer = UniCodecTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr,
        max_steps=max_steps,
        checkpoint_dir=config_dict.get('training', {}).get('checkpoint_dir', '../checkpoints/unicodec'),
        log_interval=50 if args.test else 100,
        eval_interval=100 if args.test else 1000,
        save_interval=200 if args.test else 5000,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    # Final evaluation
    if val_loader is not None:
        final_loss = trainer.evaluate()
        logger.info(f"Final validation loss: {final_loss:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
