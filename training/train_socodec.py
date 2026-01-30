"""
SoCodec Training Script

Trains the Semantic-Ordered Speech Codec with Ordered Product Quantization (OPQ).

Features:
- HuBERT/WavLM semantic feature extraction
- Ordered Product Quantization with multi-stream conditioning
- ECAPA-TDNN acoustic encoder
- Multi-stream delayed LM training
- Checkpoint saving and resuming
- Validation and metrics logging

Usage:
    # Full training
    python train_socodec.py --config config/socodec.yaml

    # Resume from checkpoint
    python train_socodec.py --config config/socodec.yaml \
        --resume ../checkpoints/socodec/latest.pt

    # Test mode (synthetic data)
    python train_socodec.py --test
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from socodec import (
    SoCodecConfig,
    SoCodec,
    SoCodecLoss,
    SoCodecAdapter,
    compute_bitrate,
)


# =============================================================================
# DATASET
# =============================================================================

class SoCodecDataset(Dataset):
    """
    Dataset for SoCodec training.

    Loads mel spectrograms and pre-extracted semantic features (HuBERT/WavLM).
    """

    def __init__(
        self,
        manifest_path: str,
        mel_dir: Optional[str] = None,
        semantic_dir: Optional[str] = None,
        max_length: int = 500,  # Max frames
        min_length: int = 50,   # Min frames
    ):
        self.manifest_path = manifest_path
        self.mel_dir = mel_dir
        self.semantic_dir = semantic_dir
        self.max_length = max_length
        self.min_length = min_length

        # Load manifest
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.samples = json.load(f)
        else:
            self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load mel spectrogram
        mel_path = sample.get('mel_path', '')
        if self.mel_dir:
            mel_path = os.path.join(self.mel_dir, mel_path)

        if os.path.exists(mel_path):
            mel = torch.load(mel_path)
        else:
            # Fallback: random mel
            mel = torch.randn(self.max_length, 80)

        # Load semantic features
        semantic_path = sample.get('semantic_path', '')
        if self.semantic_dir:
            semantic_path = os.path.join(self.semantic_dir, semantic_path)

        if os.path.exists(semantic_path):
            semantic = torch.load(semantic_path)
        else:
            # Fallback: random semantic features
            semantic = torch.randn(mel.shape[0], 768)

        # Ensure same length
        min_len = min(mel.shape[0], semantic.shape[0])
        mel = mel[:min_len]
        semantic = semantic[:min_len]

        # Truncate to max length
        if mel.shape[0] > self.max_length:
            start = torch.randint(0, mel.shape[0] - self.max_length, (1,)).item()
            mel = mel[start:start + self.max_length]
            semantic = semantic[start:start + self.max_length]

        # Pad if too short
        if mel.shape[0] < self.min_length:
            pad_len = self.min_length - mel.shape[0]
            mel = F.pad(mel, (0, 0, 0, pad_len))
            semantic = F.pad(semantic, (0, 0, 0, pad_len))

        return {
            'mel': mel,
            'semantic': semantic,
            'length': min_len,
        }


class SyntheticSoCodecDataset(Dataset):
    """Synthetic dataset for testing."""

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
        mel = torch.randn(self.seq_len, self.mel_dim)
        semantic = torch.randn(self.seq_len, self.semantic_dim)

        return {
            'mel': mel,
            'semantic': semantic,
            'length': self.seq_len,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    max_len = max(item['mel'].shape[0] for item in batch)

    mels = []
    semantics = []
    lengths = []

    for item in batch:
        mel = item['mel']
        semantic = item['semantic']

        # Pad to max length
        if mel.shape[0] < max_len:
            pad_len = max_len - mel.shape[0]
            mel = F.pad(mel, (0, 0, 0, pad_len))
            semantic = F.pad(semantic, (0, 0, 0, pad_len))

        mels.append(mel)
        semantics.append(semantic)
        lengths.append(item['length'])

    return {
        'mel': torch.stack(mels),
        'semantic': torch.stack(semantics),
        'lengths': torch.tensor(lengths),
    }


# =============================================================================
# TRAINER
# =============================================================================

class SoCodecTrainer:
    """Trainer for SoCodec model."""

    def __init__(
        self,
        config: SoCodecConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: str = "../checkpoints/socodec",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        log_interval: int = 50,
        save_interval: int = 5,
        device: str = "cuda",
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Model
        self.model = SoCodec(config).to(self.device)
        self.loss_fn = SoCodecLoss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            self.val_loader = None

        # Scheduler
        num_training_steps = len(self.train_loader) * num_epochs
        num_warmup_steps = len(self.train_loader) * warmup_epochs

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=num_training_steps,
            pct_start=num_warmup_steps / num_training_steps,
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print model configuration and statistics."""
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("SoCodec Training Configuration")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nModel Config:")
        print(f"  Num streams: {self.config.num_streams}")
        print(f"  Codebook size: {self.config.codebook_size}")
        print(f"  Code dim: {self.config.code_dim}")
        print(f"  Downsample factor: {self.config.temporal_downsample_factor}")
        print(f"  Use delayed LM: {self.config.use_delayed_lm}")

        bitrate = compute_bitrate(self.config)
        print(f"\nTheoretical Bitrate:")
        print(f"  Semantic: {bitrate['semantic_bps']:.2f} bps")
        print(f"  Total: {bitrate['total_bps']:.2f} bps")
        print(f"  Frame rate: {bitrate['frame_rate_hz']:.2f} Hz")

        print(f"\nTraining Config:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {self.num_epochs}")
        print("=" * 60 + "\n")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save epoch checkpoint
        if (epoch + 1) % self.save_interval == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch+1}.pt"
            torch.save(checkpoint, epoch_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'commitment': 0.0,
            'lm': 0.0,
        }
        epoch_perplexities = []

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            mel = batch['mel'].to(self.device)
            semantic = batch['semantic'].to(self.device)

            # Forward pass
            output = self.model(mel, semantic)
            losses = self.loss_fn(output, mel)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()
            self.scheduler.step()

            # Accumulate losses
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['reconstruction'] += losses['reconstruction'].item()
            epoch_losses['commitment'] += losses['commitment'].item()
            epoch_losses['lm'] += losses['lm_loss'].item()
            epoch_perplexities.append(losses['mean_perplexity'].item())

            self.global_step += 1

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                avg_perplexity = sum(epoch_perplexities[-self.log_interval:]) / self.log_interval
                print(
                    f"  Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss: {losses['total'].item():.4f}, "
                    f"recon: {losses['reconstruction'].item():.4f}, "
                    f"commit: {losses['commitment'].item():.4f}, "
                    f"perplexity: {avg_perplexity:.2f}, "
                    f"lr: {lr:.2e}"
                )

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        epoch_losses['perplexity'] = sum(epoch_perplexities) / len(epoch_perplexities)
        epoch_losses['time'] = time.time() - start_time

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'commitment': 0.0,
        }
        val_perplexities = []

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            semantic = batch['semantic'].to(self.device)

            output = self.model(mel, semantic)
            losses = self.loss_fn(output, mel)

            val_losses['total'] += losses['total'].item()
            val_losses['reconstruction'] += losses['reconstruction'].item()
            val_losses['commitment'] += losses['commitment'].item()
            val_perplexities.append(losses['mean_perplexity'].item())

        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        val_losses['perplexity'] = sum(val_perplexities) / len(val_perplexities)

        return val_losses

    def train(self):
        """Full training loop."""
        print(f"\nStarting training from epoch {self.start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.start_epoch, self.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")

            # Train
            train_losses = self.train_epoch(epoch)
            print(
                f"\nTrain - loss: {train_losses['total']:.4f}, "
                f"recon: {train_losses['reconstruction']:.4f}, "
                f"perplexity: {train_losses['perplexity']:.2f}, "
                f"time: {train_losses['time']:.1f}s"
            )

            # Validate
            if self.val_loader:
                val_losses = self.validate()
                print(
                    f"Val   - loss: {val_losses['total']:.4f}, "
                    f"recon: {val_losses['reconstruction']:.4f}, "
                    f"perplexity: {val_losses['perplexity']:.2f}"
                )

                # Check for best model
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
            else:
                is_best = False

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("Warning: PyYAML not installed, using default config")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train SoCodec model")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run in test mode with synthetic data"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="../checkpoints/socodec",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config)"
    )

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        cfg = {}

    # Create model config
    model_cfg = cfg.get('model', {})
    config = SoCodecConfig(
        num_streams=model_cfg.get('num_streams', 4),
        codebook_size=model_cfg.get('codebook_size', 1024),
        code_dim=model_cfg.get('code_dim', 64),
        temporal_downsample_factor=model_cfg.get('temporal_downsample_factor', 6),
        semantic_input_dim=model_cfg.get('semantic_input_dim', 768),
        acoustic_dim=model_cfg.get('acoustic_dim', 256),
        use_delayed_lm=model_cfg.get('use_delayed_lm', True),
        conditioning_type=model_cfg.get('conditioning_type', 'concat'),
        use_cosine_similarity=model_cfg.get('use_cosine_similarity', True),
        dropout=model_cfg.get('dropout', 0.1),
        output_dim=model_cfg.get('output_dim', 2048),
        num_prefix_tokens=model_cfg.get('num_prefix_tokens', 4),
    )

    # Training config
    train_cfg = cfg.get('training', {})
    epochs = args.epochs or train_cfg.get('epochs', 100)
    batch_size = args.batch_size or train_cfg.get('batch_size', 16)
    lr = args.lr or train_cfg.get('learning_rate', 1e-4)
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    log_interval = train_cfg.get('log_interval', 50)
    save_interval = train_cfg.get('save_interval', 5)

    # Data config
    data_cfg = cfg.get('data', {})

    # Create datasets
    if args.test:
        print("Running in TEST mode with synthetic data")
        train_dataset = SyntheticSoCodecDataset(num_samples=200, seq_len=100)
        val_dataset = SyntheticSoCodecDataset(num_samples=50, seq_len=100)
        epochs = 3
        batch_size = 4
        log_interval = 10
    else:
        train_manifest = data_cfg.get('train_manifest', '../data/train_manifest.json')
        val_manifest = data_cfg.get('val_manifest', '../data/val_manifest.json')

        if os.path.exists(train_manifest):
            train_dataset = SoCodecDataset(
                train_manifest,
                mel_dir=data_cfg.get('mel_dir'),
                semantic_dir=data_cfg.get('semantic_dir'),
            )
        else:
            print(f"Warning: {train_manifest} not found, using synthetic data")
            train_dataset = SyntheticSoCodecDataset(num_samples=200, seq_len=100)

        if os.path.exists(val_manifest):
            val_dataset = SoCodecDataset(
                val_manifest,
                mel_dir=data_cfg.get('mel_dir'),
                semantic_dir=data_cfg.get('semantic_dir'),
            )
        else:
            val_dataset = SyntheticSoCodecDataset(num_samples=50, seq_len=100)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create trainer
    trainer = SoCodecTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=epochs,
        warmup_epochs=warmup_epochs,
        grad_clip=grad_clip,
        log_interval=log_interval,
        save_interval=save_interval,
        device=device,
    )

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
