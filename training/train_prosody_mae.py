#!/usr/bin/env python3
"""
Training Script for Prosody-MAE Self-Supervised Pre-training

Based on Prosody-MAE (ACL Findings 2023). Pre-trains a masked autoencoder
on unlabeled speech data to learn prosodic representations.

Usage:
    # Pre-train on unlabeled audio
    python train_prosody_mae.py --config config/prosody_mae.yaml

    # Resume training
    python train_prosody_mae.py --config config/prosody_mae.yaml \
        --resume ../checkpoints/prosody_mae/latest.pt

    # Test mode (synthetic data)
    python train_prosody_mae.py --test

    # Fine-tune on labeled data
    python train_prosody_mae.py --config config/prosody_mae.yaml \
        --pretrained ../checkpoints/prosody_mae/best.pt \
        --finetune
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

from prosody_mae import (
    ProsodyMAEConfig,
    ProsodyMAE,
    ProsodyMAEAdapter,
    ProsodyMAELoss,
    get_mask_ratio_schedule,
    create_prosody_mae_adapter,
)


# =============================================================================
# DATASET
# =============================================================================

class UnlabeledAudioDataset(Dataset):
    """
    Dataset for unlabeled audio files for self-supervised pre-training.

    Supports:
    - Directory of audio files
    - Manifest JSON file with paths
    - Random cropping for training
    """

    def __init__(
        self,
        audio_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: float = 10.0,  # Max duration in seconds
        min_duration: float = 1.0,   # Min duration in seconds
        random_crop: bool = True,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.min_samples = int(min_duration * sample_rate)
        self.random_crop = random_crop

        # Collect audio files
        self.audio_files = []

        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if isinstance(manifest, list):
                self.audio_files = [item['audio_path'] if isinstance(item, dict) else item for item in manifest]
            elif isinstance(manifest, dict) and 'files' in manifest:
                self.audio_files = manifest['files']
        elif audio_dir and os.path.exists(audio_dir):
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                self.audio_files.extend(Path(audio_dir).rglob(ext))
            self.audio_files = [str(f) for f in self.audio_files]

        print(f"Found {len(self.audio_files)} audio files")

    def __len__(self) -> int:
        return len(self.audio_files)

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and resample audio file."""
        if HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(path)
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze(0)
        else:
            # Fallback: return random noise
            return torch.randn(self.max_samples)

    def _crop_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Crop audio to max duration."""
        if len(audio) > self.max_samples:
            if self.random_crop:
                start = random.randint(0, len(audio) - self.max_samples)
            else:
                start = 0
            audio = audio[start:start + self.max_samples]
        return audio

    def _pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad audio to min duration if needed."""
        if len(audio) < self.min_samples:
            padding = self.min_samples - len(audio)
            audio = F.pad(audio, (0, padding), mode='constant', value=0)
        return audio

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path = self.audio_files[idx]

        try:
            audio = self._load_audio(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return random noise as fallback
            audio = torch.randn(self.max_samples)

        # Crop and pad
        audio = self._crop_audio(audio)
        audio = self._pad_audio(audio)

        # Normalize
        audio = audio / (audio.abs().max() + 1e-8)

        return {
            'audio': audio,
            'path': audio_path,
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        sample_rate: int = 16000,
        duration: float = 3.0,
    ):
        self.num_samples = num_samples
        self.audio_length = int(sample_rate * duration)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic audio with prosodic patterns
        t = torch.linspace(0, 3, self.audio_length)

        # Base frequency with pitch variation
        f0 = 200 + 50 * torch.sin(2 * 3.14159 * 0.5 * t)  # Slow pitch modulation

        # Audio signal
        audio = torch.sin(2 * 3.14159 * f0 * t)

        # Add some noise
        audio = audio + 0.1 * torch.randn_like(audio)

        # Normalize
        audio = audio / audio.abs().max()

        return {
            'audio': audio,
            'path': f'synthetic_{idx}',
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of audio samples."""
    # Find max length
    max_len = max(b['audio'].shape[0] for b in batch)

    # Pad to max length
    audios = []
    for b in batch:
        audio = b['audio']
        if len(audio) < max_len:
            audio = F.pad(audio, (0, max_len - len(audio)))
        audios.append(audio)

    return {
        'audio': torch.stack(audios),
    }


# =============================================================================
# TRAINING
# =============================================================================

class ProsodyMAETrainer:
    """Trainer for Prosody-MAE pre-training."""

    def __init__(
        self,
        config: ProsodyMAEConfig,
        model: ProsodyMAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        checkpoint_dir: str = "../checkpoints/prosody_mae",
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler (cosine with warmup)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Get mask ratio for this epoch
        mask_ratio = get_mask_ratio_schedule(epoch, self.total_epochs, self.config)

        total_loss = 0.0
        num_batches = 0
        loss_components = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.total_epochs}")

        for batch in pbar:
            audio = batch['audio'].to(self.device)

            # Forward pass
            losses = self.model.compute_pretraining_loss(
                audio=audio,
                mask_ratio=mask_ratio,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += losses['total'].item()
            num_batches += 1

            # Track loss components
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'mask': f"{mask_ratio:.2f}",
            })

        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return {
            'loss': avg_loss,
            'mask_ratio': mask_ratio,
            **loss_components,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch['audio'].to(self.device)

            losses = self.model.compute_pretraining_loss(
                audio=audio,
                mask_ratio=self.config.mask_ratio,
            )

            total_loss += losses['total'].item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])

        print(f"Resumed from epoch {self.current_epoch}")

    def train(self):
        """Full training loop."""
        print(f"\nStarting pre-training for {self.total_epochs} epochs")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Mask ratio: {self.config.mask_ratio}")
        print(f"Warmup epochs: {self.warmup_epochs}")

        for epoch in range(self.current_epoch, self.total_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Check if best
            is_best = False
            if 'val_loss' in val_metrics:
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    is_best = True

            # Log
            metrics = {**train_metrics, **val_metrics, 'lr': self.scheduler.get_last_lr()[0]}
            self.training_history.append(metrics)

            print(f"\nEpoch {epoch+1}/{self.total_epochs}")
            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Mask ratio: {train_metrics['mask_ratio']:.2f}")
            if 'val_loss' in val_metrics:
                print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            print(f"  LR: {metrics['lr']:.6f}")

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.training_history


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str) -> Tuple[ProsodyMAEConfig, dict]:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        print("Warning: PyYAML not installed, using default config")
        return ProsodyMAEConfig(), {}

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Extract model config
    model_cfg = cfg.get('model', {})
    prosody_mae_config = ProsodyMAEConfig(
        # Audio processing
        sample_rate=model_cfg.get('sample_rate', 16000),
        hop_length=model_cfg.get('hop_length', 160),
        n_mels=model_cfg.get('n_mels', 80),

        # Patch and masking
        patch_size=model_cfg.get('patch_size', 8),
        embed_dim=model_cfg.get('embed_dim', 384),
        mask_ratio=model_cfg.get('mask_ratio', 0.70),

        # Encoder
        encoder_num_layers=model_cfg.get('encoder_num_layers', 8),
        encoder_num_heads=model_cfg.get('encoder_num_heads', 6),
        encoder_ffn_dim=model_cfg.get('encoder_ffn_dim', 1536),
        encoder_dropout=model_cfg.get('encoder_dropout', 0.1),

        # Decoder
        decoder_embed_dim=model_cfg.get('decoder_embed_dim', 192),
        decoder_num_layers=model_cfg.get('decoder_num_layers', 4),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 4),
        decoder_ffn_dim=model_cfg.get('decoder_ffn_dim', 768),
        decoder_dropout=model_cfg.get('decoder_dropout', 0.1),

        # Reconstruction
        reconstruct_mel=model_cfg.get('reconstruct_mel', True),
        reconstruct_pitch=model_cfg.get('reconstruct_pitch', True),
        reconstruct_energy=model_cfg.get('reconstruct_energy', True),

        # Loss weights
        mel_loss_weight=model_cfg.get('mel_loss_weight', 1.0),
        pitch_loss_weight=model_cfg.get('pitch_loss_weight', 0.5),
        energy_loss_weight=model_cfg.get('energy_loss_weight', 0.5),

        # Training
        warmup_epochs=model_cfg.get('warmup_epochs', 10),
        min_mask_ratio=model_cfg.get('min_mask_ratio', 0.5),

        # Output
        output_dim=model_cfg.get('output_dim', 2048),
        num_prefix_tokens=model_cfg.get('num_prefix_tokens', 4),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
    )

    return prosody_mae_config, cfg


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Prosody-MAE")
    parser.add_argument('--config', type=str, help='Path to config YAML')
    parser.add_argument('--audio-dir', type=str, help='Directory of audio files')
    parser.add_argument('--manifest', type=str, help='Path to manifest JSON')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, help='Pre-trained checkpoint for fine-tuning')
    parser.add_argument('--finetune', action='store_true', help='Fine-tune mode')
    parser.add_argument('--test', action='store_true', help='Test mode with synthetic data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints/prosody_mae',
                        help='Checkpoint directory')

    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config, full_cfg = load_config(args.config)
        training_cfg = full_cfg.get('training', {})
    else:
        config = ProsodyMAEConfig()
        training_cfg = {}

    # Override from args
    epochs = training_cfg.get('epochs', args.epochs)
    batch_size = training_cfg.get('batch_size', args.batch_size)
    lr = training_cfg.get('learning_rate', args.lr)
    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("Prosody-MAE Pre-training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Encoder layers: {config.encoder_num_layers}")
    print(f"  Decoder layers: {config.decoder_num_layers}")

    # Create dataset
    if args.test:
        print("\n[Test Mode] Using synthetic data")
        train_dataset = SyntheticDataset(num_samples=500, sample_rate=config.sample_rate)
        val_dataset = SyntheticDataset(num_samples=100, sample_rate=config.sample_rate)
        epochs = 5  # Reduce for testing
    else:
        audio_dir = args.audio_dir or training_cfg.get('audio_dir')
        manifest = args.manifest or training_cfg.get('manifest')

        train_dataset = UnlabeledAudioDataset(
            audio_dir=audio_dir,
            manifest_path=manifest,
            sample_rate=config.sample_rate,
            max_duration=training_cfg.get('max_duration', 10.0),
            min_duration=training_cfg.get('min_duration', 1.0),
        )

        # Simple split for validation
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    print(f"\nDataset:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset) if val_dataset else 0}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    ) if val_dataset else None

    # Create model
    model = ProsodyMAE(config)

    # Load pretrained if provided
    if args.pretrained:
        print(f"\nLoading pre-trained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = ProsodyMAETrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=training_cfg.get('weight_decay', 0.05),
        warmup_epochs=training_cfg.get('warmup_epochs', 10),
        total_epochs=epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    # Resume if provided
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    start_time = time.time()

    try:
        history = trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch)
        history = trainer.training_history

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed / 3600:.2f} hours")

    # Save training history
    history_path = Path(args.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Test adapter
    print("\n" + "=" * 70)
    print("Testing Adapter...")
    print("=" * 70)

    adapter = ProsodyMAEAdapter(config, model).to(device)
    test_audio = torch.randn(2, config.sample_rate * 2).to(device)

    with torch.no_grad():
        prefix_tokens = adapter(audio=test_audio)

    print(f"Test audio: {test_audio.shape}")
    print(f"Prefix tokens: {prefix_tokens.shape}")
    print("\nAdapter test passed!")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"Best checkpoint: {args.checkpoint_dir}/best.pt")

    print("\nNext steps:")
    print("-" * 40)
    print("""
1. Use pre-trained embeddings for downstream tasks:

   from prosody_mae import create_prosody_mae_adapter
   adapter = create_prosody_mae_adapter(checkpoint='checkpoints/prosody_mae/best.pt')
   prosody_tokens = adapter(audio)  # [batch, 4, 2048]

2. Integrate with CSM:

   combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
   output = csm_model(input_ids, prosody_prefix=combined_prefix)

3. Fine-tune on labeled emotion data:

   python train_prosody_mae.py --config config/prosody_mae.yaml \\
       --pretrained checkpoints/prosody_mae/best.pt --finetune
""")


if __name__ == "__main__":
    import math
    main()
