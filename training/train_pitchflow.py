#!/usr/bin/env python3
"""
PitchFlow Training Script

Trains the PitchFlow model for quantized pitch control in flow-matching TTS.

Training Strategy:
1. Stage 1 (Classifier Only): Train pitch classifier on noisy mel-spectrograms
2. Stage 2 (Full Model): Train complete flow model with pitch conditioning

Usage:
    # Full training
    python train_pitchflow.py --config config/pitchflow.yaml

    # Resume from checkpoint
    python train_pitchflow.py --config config/pitchflow.yaml \\
        --resume ../checkpoints/pitchflow/latest.pt

    # Classifier-only stage
    python train_pitchflow.py --config config/pitchflow.yaml --classifier-only

    # Test mode (synthetic data)
    python train_pitchflow.py --test
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
from torch.utils.data import Dataset, DataLoader
import torchaudio
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pitchflow import (
    PitchFlowConfig,
    PitchFlow,
    PitchFlowAdapter,
    PitchFlowLoss,
    LogF0Quantizer,
)

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: parselmouth not available, using synthetic F0")


# =============================================================================
# DATASET
# =============================================================================

class PitchFlowDataset(Dataset):
    """Dataset for PitchFlow training with F0 extraction."""

    def __init__(
        self,
        manifest_path: str,
        config: PitchFlowConfig,
        split: str = "train",
        max_audio_len: float = 10.0,
    ):
        self.config = config
        self.max_audio_len = max_audio_len
        self.max_frames = int(max_audio_len * config.sample_rate / config.hop_length)

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Filter by split
        self.samples = [s for s in manifest if s.get('split', 'train') == split]

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.win_length,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax,
        )

        print(f"Loaded {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['audio_path']

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate if too long
        max_samples = int(self.max_audio_len * self.config.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # Compute mel spectrogram
        mel = self.mel_transform(waveform)  # [1, n_mels, time]
        mel = mel.squeeze(0)  # [n_mels, time]

        # Convert to log scale
        mel = torch.log(mel.clamp(min=1e-5))

        # Extract F0
        f0 = self._extract_f0(audio_path, mel.shape[1])

        # Get prosody target if available
        if 'prosody_embedding' in sample:
            prosody = torch.tensor(sample['prosody_embedding'], dtype=torch.float32)
        else:
            prosody = torch.randn(self.config.prosody_dim)

        # Text embedding (placeholder)
        text_emb = torch.randn(20, self.config.text_dim)
        text_mask = torch.ones(20, dtype=torch.bool)

        return {
            'mel': mel,
            'f0': f0,
            'prosody': prosody,
            'text_emb': text_emb,
            'text_mask': text_mask,
            'audio_path': audio_path,
        }

    def _extract_f0(self, audio_path: str, target_len: int) -> torch.Tensor:
        """Extract F0 using Praat or fallback to synthetic."""
        if PARSELMOUTH_AVAILABLE:
            try:
                sound = parselmouth.Sound(audio_path)
                pitch = call(
                    sound,
                    "To Pitch (ac)",
                    self.config.time_step,
                    self.config.pitch_floor,
                    15,
                    "no",
                    self.config.voicing_threshold,
                    0.01,
                    0.35,
                    0.25,
                    0.01,
                    self.config.pitch_ceiling,
                )

                num_frames = call(pitch, "Get number of frames")
                f0_values = []
                for i in range(1, num_frames + 1):
                    f0 = call(pitch, "Get value in frame", i, "Hertz")
                    f0_values.append(f0 if f0 == f0 else 0.0)

                f0 = torch.tensor(f0_values, dtype=torch.float32)

                # Interpolate to match mel length
                if len(f0) != target_len:
                    f0 = F.interpolate(
                        f0.unsqueeze(0).unsqueeze(0),
                        size=target_len,
                        mode='linear',
                        align_corners=False
                    ).squeeze()

                return f0

            except Exception as e:
                print(f"F0 extraction failed for {audio_path}: {e}")

        # Fallback: synthetic F0
        t = torch.linspace(0, 4 * math.pi, target_len)
        f0 = 150 + 50 * torch.sin(t) + 20 * torch.sin(3 * t)
        # Add some unvoiced regions
        mask = torch.rand(target_len) > 0.1
        f0 = f0 * mask.float()
        return f0


def collate_fn(batch):
    """Collate function with padding."""
    # Find max lengths
    max_mel_len = max(b['mel'].shape[1] for b in batch)
    max_text_len = max(b['text_emb'].shape[0] for b in batch)

    # Pad sequences
    mels = []
    f0s = []
    prosodies = []
    text_embs = []
    text_masks = []

    for b in batch:
        # Pad mel
        mel = b['mel']
        mel_pad = F.pad(mel, (0, max_mel_len - mel.shape[1]))
        mels.append(mel_pad)

        # Pad F0
        f0 = b['f0']
        f0_pad = F.pad(f0, (0, max_mel_len - len(f0)))
        f0s.append(f0_pad)

        # Prosody (no padding needed)
        prosodies.append(b['prosody'])

        # Pad text
        text_emb = b['text_emb']
        text_pad = F.pad(text_emb, (0, 0, 0, max_text_len - text_emb.shape[0]))
        text_embs.append(text_pad)

        # Text mask
        text_mask = b['text_mask']
        mask_pad = F.pad(text_mask, (0, max_text_len - len(text_mask)), value=False)
        text_masks.append(mask_pad)

    return {
        'mel': torch.stack(mels),
        'f0': torch.stack(f0s),
        'prosody': torch.stack(prosodies),
        'text_emb': torch.stack(text_embs),
        'text_mask': torch.stack(text_masks),
    }


# =============================================================================
# SYNTHETIC DATASET (for testing)
# =============================================================================

class SyntheticPitchFlowDataset(Dataset):
    """Synthetic dataset for testing without real data."""

    def __init__(
        self,
        config: PitchFlowConfig,
        num_samples: int = 1000,
        time_len: int = 100,
    ):
        self.config = config
        self.num_samples = num_samples
        self.time_len = time_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Synthetic mel
        mel = torch.randn(self.config.n_mels, self.time_len)

        # Synthetic F0 with varying patterns
        t = torch.linspace(0, 4 * math.pi, self.time_len)
        pattern = idx % 4
        if pattern == 0:
            f0 = 150 + 50 * torch.sin(t)
        elif pattern == 1:
            f0 = 100 + torch.linspace(0, 100, self.time_len)
        elif pattern == 2:
            f0 = 200 - torch.linspace(0, 100, self.time_len)
        else:
            f0 = 180 + 30 * torch.sin(2 * t) + 20 * torch.cos(3 * t)

        # Add some unvoiced regions
        mask = torch.rand(self.time_len) > 0.1
        f0 = f0 * mask.float()

        # Prosody and text
        prosody = torch.randn(self.config.prosody_dim)
        text_emb = torch.randn(20, self.config.text_dim)
        text_mask = torch.ones(20, dtype=torch.bool)

        return {
            'mel': mel,
            'f0': f0,
            'prosody': prosody,
            'text_emb': text_emb,
            'text_mask': text_mask,
        }


# =============================================================================
# TRAINER
# =============================================================================

class PitchFlowTrainer:
    """Trainer for PitchFlow model."""

    def __init__(
        self,
        model: PitchFlowAdapter,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=tuple(config.get('betas', [0.9, 0.999])),
        )

        # Learning rate scheduler
        total_steps = config.get('epochs', 100) * config.get('steps_per_epoch', 1000)
        warmup_steps = config.get('warmup_steps', 1000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(config.get('min_lr', 1e-6) / config.get('lr', 1e-4),
                      0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        classifier_only: bool = False,
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            mel = batch['mel'].to(self.device)
            f0 = batch['f0'].to(self.device)
            prosody = batch['prosody'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)

            # Forward pass
            if classifier_only:
                # Only train classifier
                losses = self.model.pitchflow.compute_classifier_loss(
                    mel, f0, step=self.global_step
                )
                loss = losses['loss']
            else:
                # Full training
                result = self.model(
                    mel, f0, prosody, text_emb, text_mask, step=self.global_step
                )
                loss = result['total_loss']
                losses = {k: v for k, v in result.items() if 'loss' in k or 'accuracy' in k}

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()
            self.scheduler.step()

            # Accumulate losses
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                total_losses[k] = total_losses.get(k, 0) + v

            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.config.get('log_every', 100) == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Step {self.global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_losses = {}
        num_batches = 0

        for batch in dataloader:
            mel = batch['mel'].to(self.device)
            f0 = batch['f0'].to(self.device)
            prosody = batch['prosody'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)

            # Forward pass
            result = self.model(mel, f0, prosody, text_emb, text_mask)

            for k, v in result.items():
                if 'loss' in k or 'accuracy' in k:
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    total_losses[k] = total_losses.get(k, 0) + v

            num_batches += 1

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        is_best: bool = False,
    ):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config,
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)

        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train PitchFlow model")
    parser.add_argument('--config', type=str, default='config/pitchflow.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--classifier-only', action='store_true',
                       help='Train classifier only')
    parser.add_argument('--test', action='store_true',
                       help='Use synthetic data for testing')
    args = parser.parse_args()

    print("=" * 70)
    print("PitchFlow Training")
    print("Quantized Pitch Control for Flow-Matching TTS")
    print("=" * 70)

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"\nLoaded config from {args.config}")
    else:
        print(f"\nConfig not found, using defaults")
        config = {}

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model config
    model_config = PitchFlowConfig(
        num_pitch_bins=config.get('num_pitch_bins', 50),
        f0_min=config.get('f0_min', 50.0),
        f0_max=config.get('f0_max', 800.0),
        use_log_scale=config.get('use_log_scale', True),
        n_mels=config.get('n_mels', 80),
        sample_rate=config.get('sample_rate', 24000),
        hop_length=config.get('hop_length', 256),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 256),
        classifier_num_layers=config.get('classifier_num_layers', 4),
        train_noise_std=config.get('train_noise_std', 0.1),
        train_noise_schedule=config.get('train_noise_schedule', 'cosine_decay'),
        flow_hidden_dim=config.get('flow_hidden_dim', 512),
        flow_num_layers=config.get('flow_num_layers', 4),
        prosody_dim=config.get('prosody_dim', 2048),
        num_prosody_tokens=config.get('num_prosody_tokens', 4),
        text_dim=config.get('text_dim', 768),
        classifier_ce_weight=config.get('classifier_ce_weight', 1.0),
        flow_loss_weight=config.get('flow_loss_weight', 1.0),
    )

    # Create model
    model = PitchFlowAdapter(model_config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create datasets
    if args.test:
        print("\nUsing synthetic data for testing...")
        train_dataset = SyntheticPitchFlowDataset(model_config, num_samples=500)
        val_dataset = SyntheticPitchFlowDataset(model_config, num_samples=100)
        collate = None
    else:
        manifest_path = config.get('manifest', '../data/manifest.json')
        if not os.path.exists(manifest_path):
            print(f"\nManifest not found at {manifest_path}, using synthetic data...")
            train_dataset = SyntheticPitchFlowDataset(model_config, num_samples=500)
            val_dataset = SyntheticPitchFlowDataset(model_config, num_samples=100)
            collate = None
        else:
            train_dataset = PitchFlowDataset(manifest_path, model_config, split='train')
            val_dataset = PitchFlowDataset(manifest_path, model_config, split='val')
            collate = collate_fn

    # Create dataloaders
    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate,
        pin_memory=config.get('pin_memory', True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate,
        pin_memory=config.get('pin_memory', True),
    )

    # Update config with steps per epoch
    config['steps_per_epoch'] = len(train_loader)

    # Create trainer
    trainer = PitchFlowTrainer(model, config, device)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Training loop
    epochs = config.get('epochs', 100)
    classifier_only_epochs = config.get('classifier_only_epochs', 20)
    checkpoint_dir = config.get('checkpoint_dir', '../checkpoints/pitchflow')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Classifier-only phase: epochs 1-{classifier_only_epochs}")
    print(f"Full training phase: epochs {classifier_only_epochs + 1}-{epochs}")
    print()

    for epoch in range(start_epoch, epochs):
        # Determine training mode
        classifier_only = (epoch < classifier_only_epochs) or args.classifier_only

        mode = "Classifier Only" if classifier_only else "Full Training"
        print(f"Epoch {epoch + 1}/{epochs} [{mode}]")
        print("-" * 40)

        # Train
        train_losses = trainer.train_epoch(train_loader, epoch, classifier_only)
        print(f"  Train: " + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items()))

        # Evaluate
        val_losses = trainer.evaluate(val_loader)
        print(f"  Val:   " + " | ".join(f"{k}: {v:.4f}" for k, v in val_losses.items()))

        # Check for best model
        current_accuracy = val_losses.get('accuracy', val_losses.get('classifier_accuracy', 0))
        is_best = current_accuracy > trainer.best_accuracy
        if is_best:
            trainer.best_accuracy = current_accuracy
            print(f"  New best accuracy: {current_accuracy:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            trainer.save_checkpoint(checkpoint_path, epoch + 1, is_best)

        # Save latest
        latest_path = os.path.join(checkpoint_dir, 'latest.pt')
        trainer.save_checkpoint(latest_path, epoch + 1, is_best=False)

        print()

    print("=" * 70)
    print("Training complete!")
    print(f"Best accuracy: {trainer.best_accuracy:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
