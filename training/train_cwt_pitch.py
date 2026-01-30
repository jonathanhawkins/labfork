#!/usr/bin/env python3
"""
Training script for CWT F0 Spectrogram Prediction.

This script trains the CWT-based pitch predictor and compares it against
direct F0 prediction to demonstrate the benefits of the FastSpeech 2 approach.

Usage:
    # Train CWT pitch predictor
    python train_cwt_pitch.py --config config/cwt_pitch.yaml

    # Resume training
    python train_cwt_pitch.py --config config/cwt_pitch.yaml \
        --resume ../checkpoints/cwt_pitch/best.pt

    # Test mode (quick validation)
    python train_cwt_pitch.py --test

    # Compare CWT vs Direct prediction
    python train_cwt_pitch.py --compare --checkpoint ../checkpoints/cwt_pitch/best.pt
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
import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.cwt_pitch import (
    CWTPitchConfig,
    CWTPitchModule,
    CWTPitchAdapter,
    CWTEncoder,
    InverseCWT,
)


# =============================================================================
# SYNTHETIC DATASET (for testing/demonstration)
# =============================================================================

class SyntheticF0Dataset(Dataset):
    """
    Synthetic dataset with various F0 patterns for training/testing.

    Generates realistic F0 contours with:
    - Different pitch patterns (rising, falling, questioning, statement)
    - Voiced/unvoiced regions
    - Text embeddings (random but consistent)

    This is useful for:
    1. Testing the CWT approach without real data
    2. Controlled experiments with known patterns
    3. Demonstration and debugging
    """

    def __init__(
        self,
        num_samples: int = 1000,
        time_len: int = 100,
        input_dim: int = 256,
        f0_mean: float = 150.0,
        f0_std: float = 30.0,
        voiced_ratio: float = 0.85,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.time_len = time_len
        self.input_dim = input_dim
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        self.voiced_ratio = voiced_ratio

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Pre-generate all data
        self.data = self._generate_data()

    def _generate_data(self) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic F0 patterns."""
        data = []
        patterns = ["rising", "falling", "question", "statement", "emphasis", "flat"]

        for i in range(self.num_samples):
            pattern = patterns[i % len(patterns)]
            f0, text_emb, voiced_mask = self._generate_sample(pattern)
            data.append({
                'f0': f0,
                'text_embeddings': text_emb,
                'voiced_mask': voiced_mask,
                'pattern': pattern,
            })

        return data

    def _generate_sample(
        self, pattern: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a single sample with specified pattern."""
        t = torch.linspace(0, 1, self.time_len)

        # Base F0 with pattern
        if pattern == "rising":
            # Rising intonation (question-like)
            f0_contour = self.f0_mean + self.f0_std * (t - 0.5) * 2
        elif pattern == "falling":
            # Falling intonation (statement)
            f0_contour = self.f0_mean + self.f0_std * (0.5 - t) * 2
        elif pattern == "question":
            # Rise-fall-rise (yes/no question)
            f0_contour = self.f0_mean + self.f0_std * torch.sin(t * 3 * math.pi)
        elif pattern == "statement":
            # Gradual decline with emphasis
            f0_contour = self.f0_mean + self.f0_std * (0.3 - t * 0.6 + 0.2 * torch.sin(t * 4 * math.pi))
        elif pattern == "emphasis":
            # Peak in the middle (emphatic)
            f0_contour = self.f0_mean + self.f0_std * torch.exp(-((t - 0.4) ** 2) / 0.02)
        else:  # flat
            f0_contour = torch.full((self.time_len,), self.f0_mean)

        # Add micro-prosody (small random variations)
        f0_contour = f0_contour + torch.randn(self.time_len) * 5

        # Clamp to valid range
        f0_contour = f0_contour.clamp(min=50, max=800)

        # Generate voiced mask
        voiced_mask = torch.rand(self.time_len) < self.voiced_ratio

        # Apply voiced mask (unvoiced = 0)
        f0_contour = f0_contour * voiced_mask.float()

        # Generate text embeddings (random but could correlate with pattern)
        text_emb = torch.randn(self.time_len, self.input_dim) * 0.1

        return f0_contour, text_emb, voiced_mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        'f0': torch.stack([b['f0'] for b in batch]),
        'text_embeddings': torch.stack([b['text_embeddings'] for b in batch]),
        'voiced_mask': torch.stack([b['voiced_mask'] for b in batch]),
    }


# =============================================================================
# REAL DATA LOADING (for actual training)
# =============================================================================

class F0Dataset(Dataset):
    """
    Dataset for loading real F0 data from preprocessed files.

    Expected manifest format (JSON):
    [
        {
            "audio_path": "path/to/audio.wav",
            "f0_path": "path/to/f0.pt",  # or "f0.npy"
            "text_embedding_path": "path/to/text_emb.pt",
            "duration": 3.5
        },
        ...
    ]

    Or can load from a directory with .pt files containing:
    {
        "f0": torch.Tensor,
        "text_embeddings": torch.Tensor,
        "voiced_mask": torch.Tensor (optional)
    }
    """

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        max_len: int = 500,
        input_dim: int = 256,
    ):
        self.max_len = max_len
        self.input_dim = input_dim
        self.samples = []

        if manifest_path and os.path.exists(manifest_path):
            self._load_from_manifest(manifest_path)
        elif data_dir and os.path.exists(data_dir):
            self._load_from_directory(data_dir)
        else:
            print("Warning: No data source provided, using empty dataset")

    def _load_from_manifest(self, manifest_path: str) -> None:
        """Load samples from manifest JSON."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        for entry in manifest:
            self.samples.append({
                'f0_path': entry.get('f0_path'),
                'text_embedding_path': entry.get('text_embedding_path'),
            })

    def _load_from_directory(self, data_dir: str) -> None:
        """Load samples from directory of .pt files."""
        for filename in os.listdir(data_dir):
            if filename.endswith('.pt'):
                self.samples.append({
                    'data_path': os.path.join(data_dir, filename),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if 'data_path' in sample:
            # Load from combined .pt file
            data = torch.load(sample['data_path'])
            f0 = data['f0']
            text_emb = data['text_embeddings']
            voiced_mask = data.get('voiced_mask', f0 > 0)
        else:
            # Load from separate files
            f0 = torch.load(sample['f0_path'])
            text_emb = torch.load(sample['text_embedding_path'])
            voiced_mask = f0 > 0

        # Truncate/pad to max_len
        if f0.shape[0] > self.max_len:
            f0 = f0[:self.max_len]
            text_emb = text_emb[:self.max_len]
            voiced_mask = voiced_mask[:self.max_len]
        elif f0.shape[0] < self.max_len:
            pad_len = self.max_len - f0.shape[0]
            f0 = F.pad(f0, (0, pad_len))
            text_emb = F.pad(text_emb, (0, 0, 0, pad_len))
            voiced_mask = F.pad(voiced_mask, (0, pad_len))

        return {
            'f0': f0,
            'text_embeddings': text_emb,
            'voiced_mask': voiced_mask,
        }


# =============================================================================
# TRAINER
# =============================================================================

class CWTTrainer:
    """
    Trainer for CWT pitch prediction module.

    Features:
    - Training with spectrogram and reconstruction losses
    - Comparison against direct F0 prediction
    - Logging and checkpointing
    - Learning rate scheduling
    """

    def __init__(
        self,
        config: CWTPitchConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        checkpoint_dir: str = "../checkpoints/cwt_pitch",
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated based on epochs
            eta_min=1e-6,
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_spec_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            f0 = batch['f0'].to(self.device)
            text_emb = batch['text_embeddings'].to(self.device)
            voiced_mask = batch['voiced_mask'].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            result = self.model(text_emb, f0, voiced_mask)

            # Backward
            loss = result['total_loss']
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_spec_loss += result['spectrogram_loss'].item()
            total_recon_loss += result['reconstruction_loss'].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_spec = total_spec_loss / num_batches
        avg_recon = total_recon_loss / num_batches

        self.scheduler.step()

        return {
            'loss': avg_loss,
            'spectrogram_loss': avg_spec,
            'reconstruction_loss': avg_recon,
            'lr': self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_spec_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            f0 = batch['f0'].to(self.device)
            text_emb = batch['text_embeddings'].to(self.device)
            voiced_mask = batch['voiced_mask'].to(self.device)

            result = self.model(text_emb, f0, voiced_mask)

            total_loss += result['total_loss'].item()
            total_spec_loss += result['spectrogram_loss'].item()
            total_recon_loss += result['reconstruction_loss'].item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_spectrogram_loss': total_spec_loss / num_batches,
            'val_reconstruction_loss': total_recon_loss / num_batches,
        }

    @torch.no_grad()
    def compare_methods(self) -> Dict[str, float]:
        """Compare CWT vs direct prediction."""
        if not hasattr(self.model, 'cwt_module'):
            # Model is CWTPitchModule, not CWTPitchAdapter
            module = self.model
        else:
            module = self.model.cwt_module

        module.eval()
        cwt_losses = []
        direct_losses = []

        loader = self.val_loader or self.train_loader

        for batch in loader:
            f0 = batch['f0'].to(self.device)
            text_emb = batch['text_embeddings'].to(self.device)
            voiced_mask = batch['voiced_mask'].to(self.device)

            comparison = module.compare_with_direct(text_emb, f0, voiced_mask)
            cwt_losses.append(comparison['cwt_reconstruction_loss'].item())
            direct_losses.append(comparison['direct_loss'].item())

        return {
            'cwt_avg_loss': sum(cwt_losses) / len(cwt_losses),
            'direct_avg_loss': sum(direct_losses) / len(direct_losses),
            'improvement': (sum(direct_losses) - sum(cwt_losses)) / sum(direct_losses) * 100,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint. Returns starting epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0) + 1

    def train(self, num_epochs: int, start_epoch: int = 0) -> None:
        """Full training loop."""
        print(f"\nStarting training from epoch {start_epoch + 1}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)

        for epoch in range(start_epoch, num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics['val_loss'])

            # Check for best model
            current_loss = val_metrics.get('val_loss', train_metrics['loss'])
            is_best = current_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_loss

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Log progress
            log_str = (
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Spec: {train_metrics['spectrogram_loss']:.4f}, "
                f"Recon: {train_metrics['reconstruction_loss']:.4f})"
            )
            if val_metrics:
                log_str += f" | Val Loss: {val_metrics['val_loss']:.4f}"
            if is_best:
                log_str += " [BEST]"
            print(log_str)

            # Periodic comparison
            if (epoch + 1) % 10 == 0:
                comparison = self.compare_methods()
                print(f"  CWT vs Direct: CWT={comparison['cwt_avg_loss']:.4f}, "
                      f"Direct={comparison['direct_avg_loss']:.4f}, "
                      f"Improvement={comparison['improvement']:.1f}%")

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train CWT F0 Spectrogram Predictor")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--test", action="store_true", help="Run in test mode with synthetic data")
    parser.add_argument("--compare", action="store_true", help="Compare CWT vs direct prediction")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint for comparison")
    parser.add_argument("--manifest", type=str, help="Path to training manifest JSON")
    parser.add_argument("--data-dir", type=str, help="Path to preprocessed data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configuration
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)
        config = CWTPitchConfig(**cfg.get('model', {}))
        train_cfg = cfg.get('training', {})
    else:
        config = CWTPitchConfig()
        train_cfg = {}

    # Override with command line args
    epochs = args.epochs if args.epochs else train_cfg.get('epochs', 100)
    batch_size = args.batch_size if args.batch_size else train_cfg.get('batch_size', 32)
    lr = args.lr if args.lr else train_cfg.get('learning_rate', 1e-4)

    # Test mode
    if args.test:
        print("\n" + "=" * 60)
        print("Running in TEST mode with synthetic data")
        print("=" * 60)

        # Create synthetic datasets
        train_dataset = SyntheticF0Dataset(num_samples=200, seed=42)
        val_dataset = SyntheticF0Dataset(num_samples=50, seed=123)

        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
        )

        # Create model (use adapter for full integration test)
        model = CWTPitchAdapter(config, input_dim=256, prosody_hidden=2048)

        # Create trainer
        trainer = CWTTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=lr,
            checkpoint_dir="../checkpoints/cwt_pitch_test",
            device=device,
        )

        # Train for a few epochs
        trainer.train(num_epochs=5)

        # Final comparison
        print("\nFinal CWT vs Direct Comparison:")
        comparison = trainer.compare_methods()
        print(f"  CWT reconstruction loss: {comparison['cwt_avg_loss']:.4f}")
        print(f"  Direct prediction loss: {comparison['direct_avg_loss']:.4f}")
        print(f"  CWT improvement: {comparison['improvement']:.1f}%")

        print("\n[TEST MODE COMPLETE]")
        return

    # Comparison mode
    if args.compare:
        if not args.checkpoint:
            print("Error: --checkpoint required for comparison mode")
            return

        print("\n" + "=" * 60)
        print("CWT vs Direct F0 Prediction Comparison")
        print("=" * 60)

        # Load model
        model = CWTPitchAdapter(config, input_dim=256, prosody_hidden=2048)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Create test dataset
        test_dataset = SyntheticF0Dataset(num_samples=100, seed=999)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        # Run comparison
        model.eval()
        cwt_losses = []
        direct_losses = []

        with torch.no_grad():
            for batch in test_loader:
                f0 = batch['f0'].to(device)
                text_emb = batch['text_embeddings'].to(device)
                voiced_mask = batch['voiced_mask'].to(device)

                comparison = model.cwt_module.compare_with_direct(text_emb, f0, voiced_mask)
                cwt_losses.append(comparison['cwt_reconstruction_loss'].item())
                direct_losses.append(comparison['direct_loss'].item())

        avg_cwt = sum(cwt_losses) / len(cwt_losses)
        avg_direct = sum(direct_losses) / len(direct_losses)
        improvement = (avg_direct - avg_cwt) / avg_direct * 100

        print(f"\nResults:")
        print(f"  CWT reconstruction loss: {avg_cwt:.4f}")
        print(f"  Direct prediction loss: {avg_direct:.4f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"\nConclusion: CWT approach {'outperforms' if improvement > 0 else 'underperforms'} direct F0 prediction")

        return

    # Full training mode
    print("\n" + "=" * 60)
    print("CWT F0 Spectrogram Prediction Training")
    print("=" * 60)

    # Load or create datasets
    if args.manifest or args.data_dir:
        train_dataset = F0Dataset(
            manifest_path=args.manifest,
            data_dir=args.data_dir,
        )
        # Split for validation
        val_size = min(int(len(train_dataset) * 0.1), 500)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    else:
        print("No data source provided, using synthetic data for demonstration")
        train_dataset = SyntheticF0Dataset(num_samples=1000, seed=42)
        val_dataset = SyntheticF0Dataset(num_samples=200, seed=123)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    model = CWTPitchAdapter(config, input_dim=256, prosody_hidden=2048)

    # Create trainer
    trainer = CWTTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        checkpoint_dir=train_cfg.get('checkpoint_dir', '../checkpoints/cwt_pitch'),
        device=device,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # Train
    trainer.train(num_epochs=epochs, start_epoch=start_epoch)

    # Final comparison
    print("\n" + "=" * 60)
    print("Final Evaluation: CWT vs Direct Prediction")
    print("=" * 60)
    comparison = trainer.compare_methods()
    print(f"  CWT reconstruction loss: {comparison['cwt_avg_loss']:.4f}")
    print(f"  Direct prediction loss: {comparison['direct_avg_loss']:.4f}")
    print(f"  Improvement: {comparison['improvement']:.1f}%")


if __name__ == "__main__":
    main()
