#!/usr/bin/env python3
"""
Training Script for Dual-Codebook VQ-VAE

Trains the dual-codebook model for separate F0 and content learning.
Based on "Improved Prosody from Learned F0 Codebook Representations for VQ-VAE"
(Interspeech 2020).

Usage:
    # Basic training
    python train_dual_codebook.py --config config/dual_codebook.yaml

    # Resume from checkpoint
    python train_dual_codebook.py --config config/dual_codebook.yaml \
        --resume ../checkpoints/dual_codebook/best.pt

    # Test mode (synthetic data)
    python train_dual_codebook.py --test

    # Preprocess dataset (extract F0)
    python train_dual_codebook.py --preprocess \
        --manifest ../data/manifest.json \
        --output ../data/dual_codebook_features
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import yaml
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dual_codebook_vqvae import (
    DualCodebookConfig,
    DualCodebookVQVAE,
    DualCodebookLoss,
    DualCodebookProsodyAdapter,
    analyze_codebook_usage,
)


# =============================================================================
# DATASET
# =============================================================================

class DualCodebookDataset(Dataset):
    """
    Dataset for dual-codebook VQ-VAE training.

    Expects preprocessed data with:
    - mel: Mel spectrogram features
    - f0: F0 trajectory (fundamental frequency)
    - speaker_id: Speaker identifier (optional)
    """

    def __init__(
        self,
        manifest_path: str,
        features_dir: str,
        max_len: int = 500,
        min_len: int = 50,
    ):
        """
        Args:
            manifest_path: Path to JSON manifest with sample info
            features_dir: Directory containing preprocessed features
            max_len: Maximum sequence length (frames)
            min_len: Minimum sequence length (frames)
        """
        self.features_dir = Path(features_dir)
        self.max_len = max_len
        self.min_len = min_len

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Filter by length if available
        self.samples = [
            s for s in self.manifest
            if s.get('num_frames', max_len) >= min_len
            and s.get('num_frames', min_len) <= max_len
        ]

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sample_id = sample['id']

        # Load features
        mel_path = self.features_dir / f"{sample_id}_mel.pt"
        f0_path = self.features_dir / f"{sample_id}_f0.pt"

        mel = torch.load(mel_path)
        f0 = torch.load(f0_path)

        # Get speaker ID
        speaker_id = sample.get('speaker_id', 0)

        # Truncate or pad to max_len
        if mel.shape[0] > self.max_len:
            # Random crop
            start = torch.randint(0, mel.shape[0] - self.max_len, (1,)).item()
            mel = mel[start:start + self.max_len]
            f0 = f0[start:start + self.max_len]
        elif mel.shape[0] < self.max_len:
            # Pad with zeros
            pad_len = self.max_len - mel.shape[0]
            mel = F.pad(mel, (0, 0, 0, pad_len))
            f0 = F.pad(f0, (0, pad_len))

        return {
            'mel': mel,
            'f0': f0,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'length': torch.tensor(min(sample.get('num_frames', self.max_len), self.max_len)),
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 100,
        mel_dim: int = 80,
        num_speakers: int = 10,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.mel_dim = mel_dim
        self.num_speakers = num_speakers

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic mel spectrogram
        mel = torch.randn(self.seq_len, self.mel_dim)

        # Generate synthetic F0 (with some structure)
        # Base frequency + random variations
        base_f0 = 150 + 50 * torch.sin(torch.linspace(0, 4 * np.pi, self.seq_len))
        f0 = base_f0 + 20 * torch.randn(self.seq_len)
        f0 = torch.clamp(f0, min=50, max=400)

        # Random unvoiced regions (set F0 to 0)
        unvoiced_mask = torch.rand(self.seq_len) < 0.1
        f0[unvoiced_mask] = 0

        speaker_id = idx % self.num_speakers

        return {
            'mel': mel,
            'f0': f0,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'length': torch.tensor(self.seq_len),
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

class DualCodebookTrainer:
    """Trainer for Dual-Codebook VQ-VAE."""

    def __init__(
        self,
        config: DualCodebookConfig,
        model: DualCodebookVQVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        checkpoint_dir: str = "../checkpoints/dual_codebook",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_steps = max_steps

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.loss_fn = DualCodebookLoss(config)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max_steps // 4,
            T_mult=2,
        )

        # Warmup
        self.warmup_steps = warmup_steps

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Logging
        self.train_losses: List[Dict[str, float]] = []
        self.val_losses: List[Dict[str, float]] = []

    def _warmup_lr(self) -> float:
        """Linear warmup for learning rate."""
        if self.global_step < self.warmup_steps:
            return self.global_step / self.warmup_steps
        return 1.0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        mel = batch['mel'].to(self.device)
        f0 = batch['f0'].to(self.device)
        speaker_ids = batch['speaker_id'].to(self.device)

        # Forward pass
        output = self.model(mel, f0, speaker_ids)

        # Compute loss
        losses = self.loss_fn(output, mel)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Apply warmup
        warmup_factor = self._warmup_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * warmup_factor

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Return losses as float dict
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_losses = {}
        num_batches = 0

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            f0 = batch['f0'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)

            output = self.model(mel, f0, speaker_ids)
            losses = self.loss_fn(output, mel)

            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                total_losses[k] = total_losses.get(k, 0) + val

            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from step {self.global_step}")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Dual-Codebook VQ-VAE Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Max steps: {self.max_steps}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print("=" * 60 + "\n")

        train_iter = iter(self.train_loader)

        while self.global_step < self.max_steps:
            # Get next batch (with cycling)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            losses = self.train_step(batch)
            self.global_step += 1

            # Log training
            if self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.global_step:6d} | "
                      f"loss: {losses['total']:.4f} | "
                      f"recon: {losses['reconstruction']:.4f} | "
                      f"f0_ppl: {losses['f0_perplexity']:.2f} | "
                      f"cont_ppl: {losses['content_perplexity']:.2f} | "
                      f"lr: {lr:.2e}")

            # Validate
            if self.global_step % self.eval_interval == 0:
                val_losses = self.validate()
                if val_losses:
                    print(f"\n[Validation] Step {self.global_step}")
                    print(f"  loss: {val_losses['total']:.4f} | "
                          f"recon: {val_losses['reconstruction']:.4f}")

                    # Check for best model
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint(
                            self.checkpoint_dir / f"step_{self.global_step}.pt",
                            is_best=True,
                        )
                    print()

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"step_{self.global_step}.pt"
                )
                print(f"  Saved checkpoint at step {self.global_step}")

        # Final save
        self.save_checkpoint(self.checkpoint_dir / "final.pt")
        print("\nTraining complete!")


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_dataset(
    manifest_path: str,
    audio_dir: str,
    output_dir: str,
    sample_rate: int = 16000,
    hop_length: int = 256,
    n_mels: int = 80,
    f0_min: float = 50.0,
    f0_max: float = 600.0,
):
    """
    Preprocess dataset: extract mel spectrograms and F0.

    Args:
        manifest_path: Path to input manifest
        audio_dir: Directory containing audio files
        output_dir: Output directory for features
        sample_rate: Target sample rate
        hop_length: Hop length for feature extraction
        n_mels: Number of mel bands
        f0_min: Minimum F0 for extraction
        f0_max: Maximum F0 for extraction
    """
    try:
        import torchaudio
        import librosa
    except ImportError:
        print("Please install torchaudio and librosa for preprocessing")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    processed = []

    for i, sample in enumerate(manifest):
        try:
            audio_path = Path(audio_dir) / sample['audio_file']

            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze().numpy()

            # Extract mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=sample_rate,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            mel = torch.from_numpy(mel.T).float()  # [T, n_mels]

            # Extract F0 using librosa's pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=f0_min,
                fmax=f0_max,
                sr=sample_rate,
                hop_length=hop_length,
            )

            # Handle NaN values (unvoiced regions)
            f0 = np.nan_to_num(f0, nan=0.0)
            f0 = torch.from_numpy(f0).float()

            # Align lengths
            min_len = min(mel.shape[0], len(f0))
            mel = mel[:min_len]
            f0 = f0[:min_len]

            # Save features
            sample_id = sample.get('id', f"sample_{i:06d}")
            torch.save(mel, output_dir / f"{sample_id}_mel.pt")
            torch.save(f0, output_dir / f"{sample_id}_f0.pt")

            # Update sample info
            processed.append({
                **sample,
                'id': sample_id,
                'num_frames': min_len,
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(manifest)} samples")

        except Exception as e:
            print(f"Error processing {sample}: {e}")
            continue

    # Save processed manifest
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(processed, f, indent=2)

    print(f"\nPreprocessed {len(processed)} samples to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Dual-Codebook VQ-VAE")
    parser.add_argument('--config', type=str, help="Path to config YAML")
    parser.add_argument('--resume', type=str, help="Path to checkpoint to resume")
    parser.add_argument('--test', action='store_true', help="Test mode with synthetic data")
    parser.add_argument('--preprocess', action='store_true', help="Preprocess dataset")
    parser.add_argument('--manifest', type=str, help="Manifest path for preprocessing")
    parser.add_argument('--audio-dir', type=str, help="Audio directory for preprocessing")
    parser.add_argument('--output', type=str, help="Output directory for preprocessing")

    args = parser.parse_args()

    # Preprocessing mode
    if args.preprocess:
        if not args.manifest or not args.output:
            print("Please provide --manifest and --output for preprocessing")
            return

        preprocess_dataset(
            manifest_path=args.manifest,
            audio_dir=args.audio_dir or str(Path(args.manifest).parent),
            output_dir=args.output,
        )
        return

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = DualCodebookConfig(**config_dict.get('model', {}))
        train_config = config_dict.get('training', {})
    else:
        config = DualCodebookConfig()
        train_config = {}

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = DualCodebookVQVAE(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test mode
    if args.test:
        print("\n[Test Mode] Using synthetic data")

        train_dataset = SyntheticDataset(
            num_samples=1000,
            seq_len=100,
            mel_dim=config.mel_dim,
            num_speakers=config.num_speakers,
        )
        val_dataset = SyntheticDataset(
            num_samples=100,
            seq_len=100,
            mel_dim=config.mel_dim,
            num_speakers=config.num_speakers,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=False,
            num_workers=0,
        )

        # Short training for test
        train_config['max_steps'] = 500
        train_config['log_interval'] = 50
        train_config['eval_interval'] = 100
        train_config['save_interval'] = 200

    else:
        # Load real dataset
        manifest_path = train_config.get('manifest', '../data/dual_codebook_features/manifest.json')
        features_dir = train_config.get('features_dir', '../data/dual_codebook_features')

        if not Path(manifest_path).exists():
            print(f"Manifest not found: {manifest_path}")
            print("Please preprocess the dataset first with --preprocess")
            print("Or run in --test mode for synthetic data")
            return

        train_dataset = DualCodebookDataset(
            manifest_path=manifest_path,
            features_dir=features_dir,
            max_len=train_config.get('max_len', 500),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=True,
            num_workers=train_config.get('num_workers', 4),
            pin_memory=True,
        )
        val_loader = None  # TODO: Add validation split

    # Create trainer
    trainer = DualCodebookTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 1000),
        max_steps=train_config.get('max_steps', 100000),
        checkpoint_dir=train_config.get('checkpoint_dir', '../checkpoints/dual_codebook'),
        log_interval=train_config.get('log_interval', 100),
        eval_interval=train_config.get('eval_interval', 1000),
        save_interval=train_config.get('save_interval', 5000),
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
