#!/usr/bin/env python3
"""
PEPA (Phoneme-Emotion Projection Adapter) Training Script

Based on "Emotional TTS via MI-Guided Disentanglement" (arXiv:2510.01722)

PEPA bridges phoneme embeddings to acoustic emotion space via two successive
1D convolutions. This enables phoneme-level emotion prediction without
reference audio at inference time.

Usage:
    # Train PEPA model
    python train_pepa.py --config config/pepa.yaml

    # Resume from checkpoint
    python train_pepa.py --config config/pepa.yaml \\
        --resume ../checkpoints/pepa/best.pt

    # Test mode with synthetic data
    python train_pepa.py --test
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pepa import PEPAConfig, PEPAAdapter, PEPALoss, PEPAEmotionModule


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class PEPADataset(Dataset):
    """
    Dataset for PEPA training.

    Expected manifest format (JSON):
    [
        {
            "audio_path": "path/to/audio.wav",
            "mel_path": "path/to/mel.pt",  # Pre-computed mel spectrogram
            "phonemes": [1, 2, 3, ...],    # Phoneme IDs
            "emotion": "happy",             # Emotion label
            "speaker_id": 0,               # Speaker ID (optional)
            "duration": 2.5                # Duration in seconds
        },
        ...
    ]
    """

    def __init__(
        self,
        manifest_path: str,
        config: PEPAConfig,
        max_mel_len: int = 1000,
        max_phoneme_len: int = 200,
        augment: bool = False,
    ):
        self.config = config
        self.max_mel_len = max_mel_len
        self.max_phoneme_len = max_phoneme_len
        self.augment = augment

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)

        # Emotion to index mapping
        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(config.emotion_labels)
        }

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load mel spectrogram
        if 'mel_path' in sample and os.path.exists(sample['mel_path']):
            mel = torch.load(sample['mel_path'])
        else:
            # Mock mel for testing
            mel = torch.randn(100, self.config.mel_dim)

        # Get phonemes
        if 'phonemes' in sample:
            phonemes = torch.tensor(sample['phonemes'], dtype=torch.long)
        else:
            # Mock phonemes for testing
            phonemes = torch.randint(1, self.config.phoneme_vocab_size, (50,))

        # Get emotion label
        emotion_str = sample.get('emotion', 'neutral')
        emotion_idx = self.emotion_to_idx.get(emotion_str.lower(), 0)

        # Get speaker ID (optional)
        speaker_id = sample.get('speaker_id', 0)

        # Truncate/pad mel
        if mel.shape[0] > self.max_mel_len:
            mel = mel[:self.max_mel_len]
        mel_len = mel.shape[0]

        # Truncate/pad phonemes
        if phonemes.shape[0] > self.max_phoneme_len:
            phonemes = phonemes[:self.max_phoneme_len]
        phoneme_len = phonemes.shape[0]

        # Pad to max length
        mel_padded = torch.zeros(self.max_mel_len, self.config.mel_dim)
        mel_padded[:mel_len] = mel

        phoneme_padded = torch.zeros(self.max_phoneme_len, dtype=torch.long)
        phoneme_padded[:phoneme_len] = phonemes

        # Create masks
        phoneme_mask = torch.zeros(self.max_phoneme_len, dtype=torch.bool)
        phoneme_mask[:phoneme_len] = True

        return {
            'mel': mel_padded,
            'mel_len': torch.tensor(mel_len),
            'phoneme_ids': phoneme_padded,
            'phoneme_mask': phoneme_mask,
            'phoneme_len': torch.tensor(phoneme_len),
            'emotion_label': torch.tensor(emotion_idx),
            'speaker_id': torch.tensor(speaker_id),
        }


class SyntheticPEPADataset(Dataset):
    """Synthetic dataset for testing without real data."""

    def __init__(
        self,
        config: PEPAConfig,
        num_samples: int = 1000,
        max_mel_len: int = 200,
        max_phoneme_len: int = 50,
    ):
        self.config = config
        self.num_samples = num_samples
        self.max_mel_len = max_mel_len
        self.max_phoneme_len = max_phoneme_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random lengths
        mel_len = random.randint(50, self.max_mel_len)
        phoneme_len = random.randint(10, self.max_phoneme_len)

        # Generate synthetic data
        mel = torch.randn(mel_len, self.config.mel_dim)
        phonemes = torch.randint(1, self.config.phoneme_vocab_size, (phoneme_len,))
        emotion_idx = random.randint(0, self.config.num_emotions - 1)
        speaker_id = random.randint(0, 99)

        # Pad
        mel_padded = torch.zeros(self.max_mel_len, self.config.mel_dim)
        mel_padded[:mel_len] = mel

        phoneme_padded = torch.zeros(self.max_phoneme_len, dtype=torch.long)
        phoneme_padded[:phoneme_len] = phonemes

        phoneme_mask = torch.zeros(self.max_phoneme_len, dtype=torch.bool)
        phoneme_mask[:phoneme_len] = True

        return {
            'mel': mel_padded,
            'mel_len': torch.tensor(mel_len),
            'phoneme_ids': phoneme_padded,
            'phoneme_mask': phoneme_mask,
            'phoneme_len': torch.tensor(phoneme_len),
            'emotion_label': torch.tensor(emotion_idx),
            'speaker_id': torch.tensor(speaker_id),
        }


# =============================================================================
# TRAINING
# =============================================================================

class PEPATrainer:
    """Trainer for PEPA model."""

    def __init__(
        self,
        config: dict,
        device: str = 'cpu',
    ):
        self.config = config
        self.device = device

        # Build model config
        model_config = config.get('model', {})
        self.model_config = PEPAConfig(
            phoneme_embed_dim=model_config.get('phoneme_embed_dim', 512),
            phoneme_vocab_size=model_config.get('phoneme_vocab_size', 256),
            reference_encoder_dim=model_config.get('reference_encoder_dim', 256),
            reference_hidden_dim=model_config.get('reference_hidden_dim', 512),
            num_gst_tokens=model_config.get('num_gst_tokens', 10),
            gst_head_dim=model_config.get('gst_head_dim', 256),
            mel_dim=model_config.get('mel_dim', 80),
            pepa_hidden_dim=model_config.get('pepa_hidden_dim', 512),
            pepa_kernel_size=model_config.get('pepa_kernel_size', 3),
            pepa_dropout=model_config.get('pepa_dropout', 0.1),
            emotion_dim=model_config.get('emotion_dim', 256),
            num_emotions=model_config.get('num_emotions', 8),
            emotion_labels=model_config.get('emotion_labels', [
                "neutral", "happy", "sad", "angry",
                "surprised", "calm", "fearful", "disgusted"
            ]),
            output_dim=model_config.get('output_dim', 2048),
            num_prosody_tokens=model_config.get('num_prosody_tokens', 4),
            use_mine=model_config.get('use_mine', True),
            num_speakers=model_config.get('num_speakers', 1000),
            speaker_embed_dim=model_config.get('speaker_embed_dim', 256),
        )

        # Build model
        self.model = PEPAEmotionModule(self.model_config).to(device)

        # Build loss function
        loss_config = config.get('loss', {})
        self.loss_fn = PEPALoss(
            config=self.model_config,
            reconstruction_weight=loss_config.get('reconstruction_weight', 1.0),
            classification_weight=loss_config.get('emotion_cls_weight', 0.5),
            mine_weight=loss_config.get('mine_weight', 0.3),
            smoothness_weight=loss_config.get('smoothness_weight', 0.1),
        )

        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.accumulation_steps = train_config.get('accumulation_steps', 1)
        self.log_every = train_config.get('log_every_n_steps', 50)
        self.eval_every = train_config.get('eval_every_n_epochs', 1)
        self.save_every = train_config.get('save_every_n_epochs', 5)
        self.checkpoint_dir = Path(train_config.get('checkpoint_dir', '../checkpoints/pepa'))

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=tuple(train_config.get('betas', [0.9, 0.999])),
        )

        # Scheduler
        warmup_epochs = train_config.get('warmup_epochs', 3)
        min_lr = train_config.get('min_lr', 1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - warmup_epochs,
            eta_min=min_lr,
        )

        # Mixed precision
        use_amp = train_config.get('use_amp', True)
        amp_dtype = train_config.get('amp_dtype', 'bfloat16')
        if use_amp and device == 'cuda':
            self.amp_dtype = torch.float16 if amp_dtype == 'float16' else torch.bfloat16
            self.scaler = torch.amp.GradScaler('cuda')
        elif use_amp and device == 'mps':
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        else:
            self.amp_dtype = None
            self.scaler = None

        # Tracking
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            mel = batch['mel'].to(self.device)
            phoneme_ids = batch['phoneme_ids'].to(self.device)
            phoneme_mask = batch['phoneme_mask'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)

            # Forward pass
            with torch.amp.autocast('cuda' if self.device == 'cuda' else 'cpu',
                                   enabled=self.amp_dtype is not None,
                                   dtype=self.amp_dtype if self.amp_dtype else torch.float32):
                output = self.model(
                    phoneme_ids=phoneme_ids,
                    phoneme_mask=phoneme_mask,
                    mel=mel,
                    emotion_labels=emotion_labels,
                    epoch=epoch,
                )

                losses = self.loss_fn(output, emotion_labels)
                loss = losses['total'] / self.accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate losses
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()

            # Optimizer step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            num_batches += 1

            # Logging
            if batch_idx % self.log_every == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {losses['total'].item():.4f} "
                    f"Recon: {losses['reconstruction'].item():.4f} "
                    f"Cls: {losses['pepa_cls'].item():.4f}"
                )

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
        correct = 0
        total = 0

        for batch in dataloader:
            mel = batch['mel'].to(self.device)
            phoneme_ids = batch['phoneme_ids'].to(self.device)
            phoneme_mask = batch['phoneme_mask'].to(self.device)
            emotion_labels = batch['emotion_label'].to(self.device)

            output = self.model(
                phoneme_ids=phoneme_ids,
                phoneme_mask=phoneme_mask,
                mel=mel,
                emotion_labels=emotion_labels,
            )

            losses = self.loss_fn(output, emotion_labels)

            # Accumulate losses
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()

            # Accuracy
            preds = output['pepa_emotion_logits'].argmax(dim=-1)
            correct += (preds == emotion_labels).sum().item()
            total += emotion_labels.size(0)

            num_batches += 1

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total

        return avg_losses

    def save_checkpoint(
        self,
        path: Path,
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = path.parent / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            logger.info(
                f"Epoch {epoch} Train - "
                f"Loss: {train_losses['total']:.4f} "
                f"Recon: {train_losses['reconstruction']:.4f} "
                f"Cls: {train_losses['pepa_cls']:.4f} "
                f"MINE: {train_losses.get('mine', 0):.4f}"
            )

            # Update scheduler
            self.scheduler.step()

            # Evaluate
            if val_loader is not None and (epoch + 1) % self.eval_every == 0:
                val_losses = self.evaluate(val_loader)
                logger.info(
                    f"Epoch {epoch} Val - "
                    f"Loss: {val_losses['total']:.4f} "
                    f"Accuracy: {val_losses['accuracy']:.2%}"
                )

                # Save best model
                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.save_checkpoint(
                        self.checkpoint_dir / f'epoch_{epoch}.pt',
                        is_best=True,
                    )

            # Regular checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f'epoch_{epoch}.pt')

        logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PEPA model')
    parser.add_argument('--config', type=str, default='config/pepa.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with synthetic data')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    hardware_config = config.get('hardware', {})
    device = hardware_config.get('device', device)
    logger.info(f"Using device: {device}")

    # Test mode
    if args.test:
        logger.info("Running in TEST mode with synthetic data")

        # Create model config
        model_config = PEPAConfig()

        # Create synthetic data
        train_dataset = SyntheticPEPADataset(model_config, num_samples=200)
        val_dataset = SyntheticPEPADataset(model_config, num_samples=50)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Create trainer
        trainer = PEPATrainer(config, device=device)

        # Train for a few epochs
        config['training'] = config.get('training', {})
        config['training']['epochs'] = 3
        trainer.epochs = 3
        trainer.train(train_loader, val_loader)

        logger.info("TEST mode complete!")
        return

    # Load real data
    training_config = config.get('training', {})
    manifest_path = training_config.get('manifest_path', '../data/emotion_manifest.json')

    if not Path(manifest_path).exists():
        logger.error(f"Manifest not found at {manifest_path}")
        logger.info("Run with --test flag to use synthetic data")
        return

    # Create datasets
    model_config = PEPAConfig(**config.get('model', {}))
    train_dataset = PEPADataset(manifest_path, model_config, augment=True)

    # Split for validation (simple 90/10 split)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=True,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=False,
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
    )

    # Create trainer
    trainer = PEPATrainer(config, device=device)

    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
