#!/usr/bin/env python3
"""
ProMode Training Script

Trains the ProMode stand-alone prosody model for TTS.

Based on "ProMode: A Speech Prosody Model Conditioned on Acoustic and Textual Inputs"
(Interspeech 2025) - https://arxiv.org/abs/2508.09389

Usage:
    # Train from scratch
    python train_promode.py --config config/promode.yaml

    # Resume from checkpoint
    python train_promode.py --config config/promode.yaml \
        --resume ../checkpoints/promode/best.pt

    # Test mode (synthetic data)
    python train_promode.py --test

    # With specific GPU
    python train_promode.py --config config/promode.yaml --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Local imports
from promode import (
    ProModeConfig,
    ProMode,
    ProModeAdapter,
    ProModeLoss,
    create_promode,
)


# =============================================================================
# DATASET
# =============================================================================

class ProModeDataset(Dataset):
    """
    Dataset for ProMode training.

    Loads audio and extracts:
    - Mel spectrogram (acoustic features)
    - Text features (from text encoder)
    - F0 contour
    - Energy
    - (Optional) Duration
    """

    def __init__(
        self,
        manifest_path: str,
        config: ProModeConfig,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
        max_audio_duration: float = 15.0,
        min_audio_duration: float = 1.0,
        is_training: bool = True,
    ):
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_audio_duration = max_audio_duration
        self.min_audio_duration = min_audio_duration
        self.is_training = is_training

        # Load manifest
        self.samples = self._load_manifest(manifest_path)

        # Audio processing
        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.n_mels = config.acoustic_dim

        if HAS_TORCHAUDIO:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load manifest file."""
        samples = []

        if not os.path.exists(manifest_path):
            print(f"Warning: Manifest not found at {manifest_path}")
            return samples

        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        duration = sample.get('duration', 10.0)
                        if self.min_audio_duration <= duration <= self.max_audio_duration:
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(samples)} samples from {manifest_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from audio."""
        if HAS_TORCHAUDIO:
            mel = self.mel_transform(audio)  # [n_mels, time]
            mel = torch.log(mel + 1e-6)
            return mel.transpose(0, 1)  # [time, n_mels]
        else:
            num_frames = audio.shape[0] // self.hop_length
            return torch.randn(num_frames, self.n_mels)

    def _extract_f0(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract F0 and voiced/unvoiced from audio.

        Returns:
            f0: [time] F0 in Hz (0 for unvoiced)
            vuv: [time] 1 for voiced, 0 for unvoiced
        """
        try:
            import pyworld as pw
            import numpy as np

            audio_np = audio.numpy().astype(np.float64)

            # DIO for F0 extraction
            f0, t = pw.dio(audio_np, self.sample_rate,
                          frame_period=self.hop_length / self.sample_rate * 1000)
            f0 = pw.stonemask(audio_np, f0, t, self.sample_rate)

            f0 = torch.from_numpy(f0).float()
            vuv = (f0 > 0).float()

            return f0, vuv
        except ImportError:
            # Fallback: synthetic F0
            num_frames = audio.shape[0] // self.hop_length
            f0 = torch.rand(num_frames) * 200 + 100  # 100-300 Hz
            vuv = (torch.rand(num_frames) > 0.2).float()  # 80% voiced
            return f0, vuv

    def _extract_energy(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract energy from mel spectrogram."""
        return mel.mean(dim=-1)  # [time]

    def _get_text_features(self, text: str) -> torch.Tensor:
        """Get text features from text encoder."""
        if self.text_encoder is not None and self.tokenizer is not None:
            # Use actual text encoder
            with torch.no_grad():
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                outputs = self.text_encoder(**inputs)
                # Use last hidden state
                features = outputs.last_hidden_state.squeeze(0)  # [seq, dim]
            return features
        else:
            # Synthetic text features
            text_len = len(text.split()) + 5
            return torch.randn(text_len, self.config.text_dim)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio_path = sample.get('audio_path', sample.get('audio', ''))
        text = sample.get('text', sample.get('transcript', ''))

        try:
            if HAS_TORCHAUDIO and os.path.exists(audio_path):
                audio, sr = torchaudio.load(audio_path)
                audio = audio[0]  # Mono

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    audio = resampler(audio)
            else:
                # Synthetic audio
                duration = sample.get('duration', 5.0)
                audio = torch.randn(int(duration * self.sample_rate))
        except Exception as e:
            # Fallback to synthetic
            audio = torch.randn(int(5.0 * self.sample_rate))

        # Extract features
        mel = self._extract_mel(audio)  # [time, n_mels]
        f0, vuv = self._extract_f0(audio)  # [time], [time]
        energy = self._extract_energy(mel)  # [time]

        # Get text features
        text_features = self._get_text_features(text)  # [text_len, text_dim]

        # Align lengths
        acoustic_len = mel.shape[0]
        f0 = f0[:acoustic_len] if f0.shape[0] > acoustic_len else F.pad(f0, (0, acoustic_len - f0.shape[0]))
        vuv = vuv[:acoustic_len] if vuv.shape[0] > acoustic_len else F.pad(vuv, (0, acoustic_len - vuv.shape[0]))
        energy = energy[:acoustic_len] if energy.shape[0] > acoustic_len else F.pad(energy, (0, acoustic_len - energy.shape[0]))

        return {
            'mel': mel,
            'text_features': text_features,
            'f0': f0,
            'vuv': vuv,
            'energy': energy,
            'text': text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Get max lengths
    max_acoustic_len = max(item['mel'].shape[0] for item in batch)
    max_text_len = max(item['text_features'].shape[0] for item in batch)

    # Pad and stack
    mel_list = []
    text_list = []
    f0_list = []
    vuv_list = []
    energy_list = []

    for item in batch:
        # Pad acoustic
        acoustic_pad = max_acoustic_len - item['mel'].shape[0]
        mel_list.append(F.pad(item['mel'], (0, 0, 0, acoustic_pad)))
        f0_list.append(F.pad(item['f0'], (0, acoustic_pad)))
        vuv_list.append(F.pad(item['vuv'], (0, acoustic_pad)))
        energy_list.append(F.pad(item['energy'], (0, acoustic_pad)))

        # Pad text
        text_pad = max_text_len - item['text_features'].shape[0]
        text_list.append(F.pad(item['text_features'], (0, 0, 0, text_pad)))

    return {
        'mel': torch.stack(mel_list),
        'text_features': torch.stack(text_list),
        'f0': torch.stack(f0_list),
        'vuv': torch.stack(vuv_list),
        'energy': torch.stack(energy_list),
    }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, self.base_lrs[i] * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    config: ProModeConfig,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': getattr(scheduler, 'step_count', 0),
        'config': asdict(config),
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model from {path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: ProMode,
    loss_fn: ProModeLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    epoch: int,
    config: Dict,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    loss_components = {}
    num_batches = 0
    log_every = config.get('log_every', 50)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        mel = batch['mel'].to(device)
        text_features = batch['text_features'].to(device)
        f0 = batch['f0'].to(device)
        vuv = batch['vuv'].to(device)
        energy = batch['energy'].to(device)

        # Forward pass
        losses = model.compute_loss(
            acoustic_features=mel,
            text_features=text_features,
            f0_target=f0,
            energy_target=energy,
            vuv_target=vuv,
        )

        # Scale loss for gradient accumulation
        loss = losses['total'] / grad_accum_steps
        loss.backward()

        # Accumulate loss components
        for k, v in losses.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()

        # Gradient step
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += losses['total'].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            lr = scheduler.get_lr()
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

    # Average losses
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return {
        'loss': avg_loss,
        **avg_components,
    }


@torch.no_grad()
def validate(
    model: ProMode,
    loss_fn: ProModeLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0
    loss_components = {}
    num_batches = 0

    for batch in dataloader:
        mel = batch['mel'].to(device)
        text_features = batch['text_features'].to(device)
        f0 = batch['f0'].to(device)
        vuv = batch['vuv'].to(device)
        energy = batch['energy'].to(device)

        losses = model.compute_loss(
            acoustic_features=mel,
            text_features=text_features,
            f0_target=f0,
            energy_target=energy,
            vuv_target=vuv,
        )

        for k, v in losses.items():
            if k not in loss_components:
                loss_components[k] = 0
            loss_components[k] += v.item()

        total_loss += losses['total'].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return {
        'val_loss': avg_loss,
        **{f'val_{k}': v for k, v in avg_components.items()},
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ProMode prosody model")
    parser.add_argument('--config', type=str, default='config/promode.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with synthetic data')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    args = parser.parse_args()

    print("=" * 60)
    print("ProMode Training")
    print("Stand-alone Prosody Model for TTS")
    print("=" * 60)

    # Load config
    config_dict = {}
    if HAS_YAML and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")
    else:
        print("Using default configuration")

    # Create model config
    model_config = config_dict.get('model', {})
    config = ProModeConfig(
        acoustic_dim=model_config.get('acoustic_dim', 80),
        text_dim=model_config.get('text_dim', 512),
        latent_dim=model_config.get('latent_dim', 256),
        num_latent_queries=model_config.get('num_latent_queries', 32),
        encoder_num_layers=model_config.get('encoder_num_layers', 6),
        encoder_num_heads=model_config.get('encoder_num_heads', 8),
        encoder_ffn_dim=model_config.get('encoder_ffn_dim', 1024),
        encoder_dropout=model_config.get('encoder_dropout', 0.1),
        num_cross_attn_layers=model_config.get('num_cross_attn_layers', 4),
        cross_attn_heads=model_config.get('cross_attn_heads', 8),
        decoder_num_layers=model_config.get('decoder_num_layers', 4),
        decoder_num_heads=model_config.get('decoder_num_heads', 8),
        decoder_ffn_dim=model_config.get('decoder_ffn_dim', 1024),
        decoder_dropout=model_config.get('decoder_dropout', 0.1),
        acoustic_mask_ratio=model_config.get('acoustic_mask_ratio', 0.5),
        text_mask_ratio=model_config.get('text_mask_ratio', 0.3),
        mask_patch_size=model_config.get('mask_patch_size', 4),
        reconstruction_weight=model_config.get('reconstruction_weight', 1.0),
        f0_weight=model_config.get('f0_weight', 1.0),
        energy_weight=model_config.get('energy_weight', 0.5),
        duration_weight=model_config.get('duration_weight', 0.5),
        consistency_weight=model_config.get('consistency_weight', 0.1),
        sample_rate=model_config.get('sample_rate', 16000),
        hop_length=model_config.get('hop_length', 256),
        f0_min=model_config.get('f0_min', 50.0),
        f0_max=model_config.get('f0_max', 800.0),
        output_dim=model_config.get('output_dim', 2048),
        num_prefix_tokens=model_config.get('num_prefix_tokens', 4),
    )

    # Training config
    train_config = config_dict.get('training', {})
    epochs = args.epochs or train_config.get('epochs', 100)
    batch_size = args.batch_size or train_config.get('batch_size', 16)
    lr = train_config.get('learning_rate', 1e-4)
    warmup_epochs = train_config.get('warmup_epochs', 5)
    grad_accum_steps = train_config.get('gradient_accumulation_steps', 2)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    checkpoint_dir = train_config.get('checkpoint_dir', '../checkpoints/promode')
    log_every = train_config.get('log_every', 50)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")

    # Create model
    print("\nCreating model...")
    model = ProMode(config).to(device)
    loss_fn = ProModeLoss(config)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_config.get('weight_decay', 0.01),
        betas=(
            train_config.get('adam_beta1', 0.9),
            train_config.get('adam_beta2', 0.98),
        ),
        eps=train_config.get('adam_eps', 1e-9),
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    # Test mode
    if args.test:
        print("\n[TEST MODE] Using synthetic data")

        # Create synthetic dataset
        class SyntheticDataset(Dataset):
            def __init__(self, size: int = 100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                acoustic_len = torch.randint(50, 200, (1,)).item()
                text_len = torch.randint(10, 50, (1,)).item()

                return {
                    'mel': torch.randn(acoustic_len, config.acoustic_dim),
                    'text_features': torch.randn(text_len, config.text_dim),
                    'f0': torch.rand(acoustic_len) * 200 + 100,
                    'vuv': (torch.rand(acoustic_len) > 0.2).float(),
                    'energy': torch.randn(acoustic_len),
                }

        train_dataset = SyntheticDataset(100)
        val_dataset = SyntheticDataset(20)

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn,
        )

        epochs = 3
    else:
        # Create real dataset
        manifest_path = train_config.get('manifest', '../data/manifest.json')

        if not os.path.exists(manifest_path):
            print(f"\nWarning: Manifest not found at {manifest_path}")
            print("Creating synthetic dataset for testing...")

            class SyntheticDataset(Dataset):
                def __init__(self, size: int = 100):
                    self.size = size

                def __len__(self):
                    return self.size

                def __getitem__(self, idx):
                    acoustic_len = torch.randint(50, 200, (1,)).item()
                    text_len = torch.randint(10, 50, (1,)).item()

                    return {
                        'mel': torch.randn(acoustic_len, config.acoustic_dim),
                        'text_features': torch.randn(text_len, config.text_dim),
                        'f0': torch.rand(acoustic_len) * 200 + 100,
                        'vuv': (torch.rand(acoustic_len) > 0.2).float(),
                        'energy': torch.randn(acoustic_len),
                    }

            train_dataset = SyntheticDataset(500)
            val_dataset = SyntheticDataset(50)
        else:
            train_dataset = ProModeDataset(
                manifest_path,
                config,
                max_audio_duration=train_config.get('max_duration', 15.0),
                min_audio_duration=train_config.get('min_duration', 1.0),
                is_training=True,
            )

            val_dataset = ProModeDataset(
                manifest_path.replace('train', 'val'),
                config,
                is_training=False,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Create scheduler
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = len(train_loader) * warmup_epochs // grad_accum_steps
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=train_config.get('min_lr', 1e-6),
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = train_config.get('early_stopping_patience', 20)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch + 1,
            config={'log_every': log_every},
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
        )

        # Validate
        val_metrics = validate(
            model=model,
            loss_fn=loss_fn,
            dataloader=val_loader,
            device=device,
        )

        # Print metrics
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

        for k, v in train_metrics.items():
            if k != 'loss':
                print(f"    {k}: {v:.4f}")

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=epoch * len(train_loader),
                loss=best_val_loss,
                path=os.path.join(checkpoint_dir, 'best.pt'),
                config=config,
            )
            print(f"  New best model! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # Regular checkpoint
        if (epoch + 1) % train_config.get('save_every', 5) == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=epoch * len(train_loader),
                loss=val_metrics['val_loss'],
                path=os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt'),
                config=config,
            )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epochs - 1,
        step=epochs * len(train_loader),
        loss=val_metrics['val_loss'],
        path=os.path.join(checkpoint_dir, 'final.pt'),
        config=config,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
