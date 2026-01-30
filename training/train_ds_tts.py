#!/usr/bin/env python3
"""
Training script for DS-TTS (Dual-Style TTS) Model

Based on DS-TTS (arXiv:2506.01020) - Dual-style feature modulation for
superior speaker similarity in voice cloning.

Usage:
    # Train DS-TTS model
    python train_ds_tts.py --config config/ds_tts.yaml

    # Resume from checkpoint
    python train_ds_tts.py --config config/ds_tts.yaml --resume ../checkpoints/ds_tts/latest.pt

    # Test mode (synthetic data)
    python train_ds_tts.py --test
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ds_tts import (
    DSTTSConfig,
    DSTTSAdapter,
    DSTTSLoss,
    create_dstts_adapter,
    analyze_style_contribution,
)


# =============================================================================
# DATASET
# =============================================================================

class DSTTSDataset(Dataset):
    """
    Dataset for DS-TTS training.

    Expects pre-extracted mel-spectrograms and optionally MFCC/prosody features:
        {
            "mel": tensor [time, mel_dim],
            "mfcc": tensor [time, mfcc_dim] (optional),
            "pitch": tensor [time] (optional),
            "energy": tensor [time] (optional),
            "duration": tensor [phoneme_len] (optional),
            "text_emb": tensor [seq, text_dim] (optional),
            "speaker_id": int,
        }
    """

    def __init__(
        self,
        manifest_path: str,
        feature_dir: str,
        max_mel_len: int = 500,
        mel_dim: int = 80,
    ):
        self.feature_dir = Path(feature_dir)
        self.max_mel_len = max_mel_len
        self.mel_dim = mel_dim

        # Load manifest
        if Path(manifest_path).exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
            self.samples = self.manifest.get('samples', self.manifest)
            if isinstance(self.samples, dict):
                self.samples = list(self.samples.values())
        else:
            self.samples = []

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            return self._generate_synthetic_sample(idx)

        sample = self.samples[idx]

        # Load pre-extracted features
        feature_path = self.feature_dir / f"{sample.get('id', idx)}.pt"
        if feature_path.exists():
            data = torch.load(feature_path)
            mel = data.get('mel', torch.randn(100, self.mel_dim))
            mfcc = data.get('mfcc', None)
            pitch = data.get('pitch', None)
            energy = data.get('energy', None)
            duration = data.get('duration', None)
            text_emb = data.get('text_emb', None)
        else:
            return self._generate_synthetic_sample(idx)

        # Truncate/pad mel to max length
        if mel.shape[0] > self.max_mel_len:
            mel = mel[:self.max_mel_len]

        mel_len = mel.shape[0]
        mask = torch.ones(mel_len)

        result = {
            'mel': mel,
            'mask': mask,
            'speaker_id': sample.get('speaker_id', 0),
            'sample_id': sample.get('id', str(idx)),
        }

        if mfcc is not None:
            result['mfcc'] = mfcc[:self.max_mel_len] if mfcc.shape[0] > self.max_mel_len else mfcc

        if pitch is not None:
            result['pitch'] = pitch[:self.max_mel_len] if pitch.shape[0] > self.max_mel_len else pitch

        if energy is not None:
            result['energy'] = energy[:self.max_mel_len] if energy.shape[0] > self.max_mel_len else energy

        if duration is not None:
            result['duration'] = duration

        if text_emb is not None:
            result['text_emb'] = text_emb

        return result

    def _generate_synthetic_sample(self, idx: int) -> Dict:
        """Generate synthetic sample for testing."""
        mel_len = torch.randint(50, self.max_mel_len, (1,)).item()
        mel = torch.randn(mel_len, self.mel_dim)
        mask = torch.ones(mel_len)

        return {
            'mel': mel,
            'mask': mask,
            'speaker_id': idx % 10,
            'sample_id': str(idx),
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 100,
        max_mel_len: int = 200,
        mel_dim: int = 80,
    ):
        self.num_samples = num_samples
        self.max_mel_len = max_mel_len
        self.mel_dim = mel_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mel_len = torch.randint(50, self.max_mel_len, (1,)).item()
        mel = torch.randn(mel_len, self.mel_dim)
        mask = torch.ones(mel_len)
        speaker_id = idx % 10

        return {
            'mel': mel,
            'mask': mask,
            'speaker_id': speaker_id,
            'sample_id': str(idx),
        }


def collate_fn(batch):
    """Collate function for variable length mel-spectrograms."""
    # Find max length
    max_len = max(item['mel'].shape[0] for item in batch)

    mel_list = []
    mask_list = []
    speaker_ids = []

    for item in batch:
        mel_len = item['mel'].shape[0]
        pad_len = max_len - mel_len

        # Pad mel
        if pad_len > 0:
            mel = F.pad(item['mel'], (0, 0, 0, pad_len))
            mask = F.pad(item['mask'], (0, pad_len))
        else:
            mel = item['mel']
            mask = item['mask']

        mel_list.append(mel)
        mask_list.append(mask)
        speaker_ids.append(item['speaker_id'])

    return {
        'mel': torch.stack(mel_list),
        'mask': torch.stack(mask_list),
        'speaker_ids': torch.tensor(speaker_ids),
    }


# =============================================================================
# TRAINER
# =============================================================================

class DSTTSTrainer:
    """Trainer for DS-TTS model."""

    def __init__(
        self,
        config: DSTTSConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: str = "../checkpoints/ds_tts",
        device: str = "cuda",
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        num_epochs: int = 100,
        log_every: int = 50,
        save_every: int = 5,
        gradient_accumulation: int = 1,
        use_amp: bool = True,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.save_every = save_every
        self.gradient_accumulation = gradient_accumulation
        self.use_amp = use_amp and self.device.type == "cuda"

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4 if not isinstance(train_dataset, SyntheticDataset) else 0,
            pin_memory=True,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4 if not isinstance(val_dataset, SyntheticDataset) else 0,
                pin_memory=True,
            )

        # Create model
        self.model = DSTTSAdapter(config).to(self.device)

        # Create loss function
        self.loss_fn = DSTTSLoss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'style_consistency': 0.0,
            'gate_balance': 0.0,
            'orthogonality': 0.0,
        }
        gate_stats = {'mel': 0.0, 'mfcc': 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            mel = batch['mel'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                output = self.model(mel, mel_mask=mask)
                losses = self.loss_fn(output)

            # Backward pass
            loss = losses['total'] / self.gradient_accumulation

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()

            # Accumulate gate stats
            if 'gates' in output:
                gates = output['gates']
                gate_stats['mel'] += gates[:, 0].mean().item()
                gate_stats['mfcc'] += gates[:, 1].mean().item()

            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.log_every == 0:
                avg_loss = epoch_losses['total'] / num_batches
                print(f"  Step {self.global_step} | Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Mel gate: {gate_stats['mel'] / num_batches:.3f} | "
                      f"MFCC gate: {gate_stats['mfcc'] / num_batches:.3f}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Average gate stats
        for key in gate_stats:
            gate_stats[key] /= num_batches

        epoch_losses['gate_mel'] = gate_stats['mel']
        epoch_losses['gate_mfcc'] = gate_stats['mfcc']

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {
            'total': 0.0,
            'style_consistency': 0.0,
            'gate_balance': 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            mask = batch['mask'].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                output = self.model(mel, mel_mask=mask)
                losses = self.loss_fn(output)

            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()

            num_batches += 1

        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)

        return val_losses

    def train(self):
        """Full training loop."""
        print(f"Starting DS-TTS training on {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_losses = self.train_epoch()
            print(f"  Train Loss: {train_losses['total']:.4f} | "
                  f"Style: {train_losses['style_consistency']:.4f} | "
                  f"Gate: {train_losses['gate_balance']:.4f}")

            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                print(f"  Val Loss: {val_losses['total']:.4f}")
                current_loss = val_losses['total']
            else:
                current_loss = train_losses['total']

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint('latest.pt')

            # Save best model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint('best.pt')
                print(f"  New best model saved (loss: {self.best_loss:.4f})")

            print()

        print("Training complete!")
        print(f"Best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DS-TTS model")
    parser.add_argument("--config", type=str, default="config/ds_tts.yaml",
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with synthetic data")
    args = parser.parse_args()

    # Test mode
    if args.test:
        print("Running DS-TTS in test mode with synthetic data")
        print()

        config = DSTTSConfig()
        train_dataset = SyntheticDataset(num_samples=50, max_mel_len=100)
        val_dataset = SyntheticDataset(num_samples=10, max_mel_len=100)

        trainer = DSTTSTrainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir="../checkpoints/ds_tts_test",
            device="cpu",
            num_epochs=3,
            batch_size=4,
            log_every=5,
        )

        trainer.train()

        # Test inference
        print("\nTesting inference...")
        trainer.model.eval()

        with torch.no_grad():
            test_mel = torch.randn(2, 100, config.mel_dim)
            output = trainer.model(test_mel)

            print(f"  Prosody tokens shape: {output['prosody_tokens'].shape}")
            print(f"  Style shape: {output['style'].shape}")
            print(f"  Style mel shape: {output['style_mel'].shape}")
            print(f"  Style mfcc shape: {output['style_mfcc'].shape}")
            print(f"  Gates shape: {output['gates'].shape}")

            # Analyze style contribution
            gate_analysis = analyze_style_contribution(output['gates'])
            print(f"\n  Style contribution analysis:")
            print(f"    Mel contribution: {gate_analysis['mel_contribution']:.3f}")
            print(f"    MFCC contribution: {gate_analysis['mfcc_contribution']:.3f}")
            print(f"    Balance ratio: {gate_analysis['balance_ratio']:.3f}")
            print(f"    Dominant encoder: {gate_analysis['dominant_encoder']}")

            # Test style interpolation
            style1 = output['style'][0:1]
            style2 = output['style'][1:2]

            interpolated = trainer.model.interpolate_styles(style1, style2, t=0.5)
            print(f"\n  Interpolated style shape: {interpolated.shape}")

            tokens_from_style = trainer.model.style_to_tokens(interpolated)
            print(f"  Tokens from interpolated style: {tokens_from_style.shape}")

        print("\nDS-TTS test completed successfully!")
        return

    # Load configuration
    if Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        cfg = {}

    # Create config object
    model_cfg = cfg.get('model', {})
    config = DSTTSConfig(
        mel_dim=model_cfg.get('mel_dim', 80),
        mfcc_dim=model_cfg.get('mfcc_dim', 13),
        style_dim=model_cfg.get('style_dim', 128),
        combined_style_dim=model_cfg.get('combined_style_dim', 256),
        text_hidden_dim=model_cfg.get('text_hidden_dim', 256),
        output_dim=model_cfg.get('output_dim', 2048),
        num_prosody_tokens=model_cfg.get('num_prosody_tokens', 4),
        dropout=model_cfg.get('dropout', 0.1),
    )

    # Create datasets
    train_manifest = cfg.get('train_manifest', '../data/ds_tts_train.json')
    val_manifest = cfg.get('val_manifest', '../data/ds_tts_val.json')
    feature_dir = cfg.get('feature_dir', '../data/ds_tts_features')

    if Path(train_manifest).exists():
        train_dataset = DSTTSDataset(
            manifest_path=train_manifest,
            feature_dir=feature_dir,
            max_mel_len=cfg.get('max_mel_len', 500),
            mel_dim=config.mel_dim,
        )
    else:
        print(f"Train manifest not found: {train_manifest}")
        print("Using synthetic data for training")
        train_dataset = SyntheticDataset(
            num_samples=cfg.get('num_synthetic_samples', 1000),
            max_mel_len=cfg.get('max_mel_len', 200),
            mel_dim=config.mel_dim,
        )

    val_dataset = None
    if Path(val_manifest).exists():
        val_dataset = DSTTSDataset(
            manifest_path=val_manifest,
            feature_dir=feature_dir,
            max_mel_len=cfg.get('max_mel_len', 500),
            mel_dim=config.mel_dim,
        )
    elif not isinstance(train_dataset, SyntheticDataset):
        print("Validation manifest not found, skipping validation")

    # Create trainer
    trainer = DSTTSTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=cfg.get('checkpoint_dir', '../checkpoints/ds_tts'),
        device=cfg.get('device', 'cuda'),
        learning_rate=cfg.get('learning_rate', 1e-4),
        batch_size=cfg.get('batch_size', 16),
        num_epochs=cfg.get('num_epochs', 100),
        log_every=cfg.get('log_every', 50),
        save_every=cfg.get('save_every', 5),
        gradient_accumulation=cfg.get('gradient_accumulation', 1),
        use_amp=cfg.get('use_amp', True),
    )

    # Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
