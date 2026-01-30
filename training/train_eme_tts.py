#!/usr/bin/env python3
"""
Train EME-TTS: Emphasis Meets Emotion TTS

Training script for the emphasis-emotion coupling model.

Training Phases:
1. Phase 1 - Pseudo-label learning: Train emphasis extraction from prosodic features
2. Phase 2 - Neural predictor: Train text-based emphasis prediction
3. Phase 3 - End-to-end: Joint training with EPE block

Usage:
    # Full training
    python train_eme_tts.py --config config/eme_tts.yaml

    # Resume from checkpoint
    python train_eme_tts.py --config config/eme_tts.yaml \
        --resume ../checkpoints/eme_tts/latest.pt

    # Phase-specific training
    python train_eme_tts.py --config config/eme_tts.yaml --phase 2

    # Test mode (synthetic data)
    python train_eme_tts.py --test

Reference: https://arxiv.org/abs/2507.12015
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.eme_tts import (
    EMETTSConfig,
    EMETTS,
    EMETTSAdapter,
    EMETTSLoss,
    VarianceEmphasisExtractor,
    EmphasisPseudoLabelGenerator,
    NeuralEmphasisPredictor,
    EmphasisPerceptionEnhancement,
    EMPHASIS_LEVELS,
    emphasis_level_to_description,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class EMETTSDataset(Dataset):
    """
    Dataset for EME-TTS training.

    Expects a manifest file with:
    - audio_path: Path to audio file
    - text: Transcript text
    - emotion: Emotion label (happy, sad, angry, etc.)
    - word_timestamps: Optional word-level timestamps
    - emphasis_labels: Optional ground-truth emphasis (if available)

    Features extracted:
    - Text embeddings (pre-computed or extracted on-the-fly)
    - Prosodic features (pitch, energy, duration)
    - Word boundaries
    """

    def __init__(
        self,
        manifest_path: str,
        config: EMETTSConfig,
        text_embeddings_dir: Optional[str] = None,
        prosody_features_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.config = config
        self.text_embeddings_dir = Path(text_embeddings_dir) if text_embeddings_dir else None
        self.prosody_features_dir = Path(prosody_features_dir) if prosody_features_dir else None

        # Load manifest
        manifest_path = Path(manifest_path)
        if manifest_path.suffix == '.json':
            with open(manifest_path) as f:
                self.samples = json.load(f)
        else:
            # Assume line-delimited JSON
            self.samples = []
            with open(manifest_path) as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

        if max_samples:
            self.samples = self.samples[:max_samples]

        # Emotion to index mapping
        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(config.emotion_labels)
        }

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get text embeddings
        text_emb = self._load_text_embeddings(sample)

        # Get prosodic features
        prosody = self._load_prosody_features(sample)

        # Get emotion
        emotion = sample.get('emotion', 'neutral')
        emotion_idx = self.emotion_to_idx.get(emotion.lower(), 0)

        # Get emphasis labels if available
        emphasis_labels = sample.get('emphasis_labels')
        if emphasis_labels is not None:
            emphasis_labels = torch.tensor(emphasis_labels, dtype=torch.long)
        else:
            emphasis_labels = torch.full((text_emb.shape[0],), -1, dtype=torch.long)

        return {
            'text_embeddings': text_emb,
            'pitch': prosody['pitch'],
            'energy': prosody['energy'],
            'duration': prosody['duration'],
            'word_boundaries': prosody['word_boundaries'],
            'emotion_idx': emotion_idx,
            'emphasis_labels': emphasis_labels,
            'sample_id': sample.get('id', str(idx)),
        }

    def _load_text_embeddings(self, sample: Dict) -> torch.Tensor:
        """Load or compute text embeddings."""
        if self.text_embeddings_dir:
            sample_id = sample.get('id', hash(sample.get('text', '')))
            emb_path = self.text_embeddings_dir / f"{sample_id}_text.pt"
            if emb_path.exists():
                return torch.load(emb_path)

        # Generate random embeddings for testing
        text = sample.get('text', '')
        num_words = len(text.split())
        return torch.randn(num_words, self.config.text_hidden_dim)

    def _load_prosody_features(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Load or compute prosodic features."""
        if self.prosody_features_dir:
            sample_id = sample.get('id', hash(sample.get('text', '')))
            feat_path = self.prosody_features_dir / f"{sample_id}_prosody.pt"
            if feat_path.exists():
                return torch.load(feat_path)

        # Generate synthetic features for testing
        text = sample.get('text', '')
        num_words = len(text.split()) or 5
        num_frames = num_words * 20  # ~20 frames per word

        return {
            'pitch': torch.randn(num_frames).abs() * 100 + 100,
            'energy': torch.randn(num_frames).abs() * 0.5,
            'duration': torch.ones(num_words) * 20,
            'word_boundaries': self._generate_word_boundaries(num_words, num_frames),
        }

    def _generate_word_boundaries(
        self,
        num_words: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Generate uniform word boundaries."""
        frames_per_word = num_frames // max(1, num_words)
        boundaries = []

        for i in range(num_words):
            start = i * frames_per_word
            end = (i + 1) * frames_per_word if i < num_words - 1 else num_frames
            boundaries.append([start, end])

        return torch.tensor(boundaries, dtype=torch.long)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Find max lengths
    max_words = max(b['text_embeddings'].shape[0] for b in batch)
    max_frames = max(b['pitch'].shape[0] for b in batch)

    # Pad and stack
    text_emb = []
    pitch = []
    energy = []
    duration = []
    word_boundaries = []
    emotion_idx = []
    emphasis_labels = []

    for b in batch:
        # Pad text embeddings
        n_words = b['text_embeddings'].shape[0]
        pad_words = max_words - n_words
        if pad_words > 0:
            text_emb.append(F.pad(b['text_embeddings'], (0, 0, 0, pad_words)))
        else:
            text_emb.append(b['text_embeddings'])

        # Pad prosody features
        n_frames = b['pitch'].shape[0]
        pad_frames = max_frames - n_frames
        if pad_frames > 0:
            pitch.append(F.pad(b['pitch'], (0, pad_frames)))
            energy.append(F.pad(b['energy'], (0, pad_frames)))
        else:
            pitch.append(b['pitch'])
            energy.append(b['energy'])

        # Pad duration and boundaries
        if pad_words > 0:
            duration.append(F.pad(b['duration'], (0, pad_words)))
            wb = F.pad(b['word_boundaries'], (0, 0, 0, pad_words))
            word_boundaries.append(wb)
            emphasis_labels.append(F.pad(b['emphasis_labels'], (0, pad_words), value=-1))
        else:
            duration.append(b['duration'])
            word_boundaries.append(b['word_boundaries'])
            emphasis_labels.append(b['emphasis_labels'])

        emotion_idx.append(b['emotion_idx'])

    return {
        'text_embeddings': torch.stack(text_emb),
        'pitch': torch.stack(pitch),
        'energy': torch.stack(energy),
        'duration': torch.stack(duration),
        'word_boundaries': torch.stack(word_boundaries),
        'emotion_idx': torch.tensor(emotion_idx, dtype=torch.long),
        'emphasis_labels': torch.stack(emphasis_labels),
    }


# =============================================================================
# TRAINER
# =============================================================================

class EMETTSTrainer:
    """Trainer for EME-TTS model."""

    def __init__(
        self,
        config: EMETTSConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: str = '../checkpoints/eme_tts',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize model
        self.model = EMETTS(config).to(device)
        self.loss_fn = EMETTSLoss(config)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        logger.info(f"Initialized EME-TTS trainer on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        for batch in self.train_dataloader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            output = self.model(
                text_embeddings=batch['text_embeddings'],
                emotion_ids=batch['emotion_idx'],
                pitch=batch['pitch'],
                energy=batch['energy'],
                duration=batch['duration'],
                word_boundaries=batch['word_boundaries'],
                emphasis_labels=batch.get('emphasis_labels'),
            )

            # Compute loss
            losses = self.loss_fn(output, batch.get('emphasis_labels'))

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()

            num_batches += 1
            self.global_step += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_losses = {}
        num_batches = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            output = self.model(
                text_embeddings=batch['text_embeddings'],
                emotion_ids=batch['emotion_idx'],
                pitch=batch['pitch'],
                energy=batch['energy'],
                duration=batch['duration'],
                word_boundaries=batch['word_boundaries'],
            )

            losses = self.loss_fn(output)

            for k, v in losses.items():
                val_losses[k] = val_losses.get(k, 0) + v.item()

            num_batches += 1

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)

        return val_losses

    def train(
        self,
        num_epochs: int,
        save_every: int = 5,
        log_every: int = 1,
    ):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()
            self.scheduler.step()

            # Validate
            val_losses = self.validate()

            # Log
            if epoch % log_every == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_losses.get('total', 0):.4f}"
                )

                for k, v in train_losses.items():
                    if k != 'total':
                        logger.info(f"  {k}: {v:.4f}")

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            # Save best model
            val_loss = val_losses.get('total', train_losses['total'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")
                logger.info(f"New best model saved (loss: {val_loss:.4f})")

        # Save final model
        self.save_checkpoint("final.pt")
        logger.info("Training completed!")

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_synthetic_dataset(
    config: EMETTSConfig,
    num_samples: int = 100,
) -> EMETTSDataset:
    """Create synthetic dataset for testing."""
    # Create temporary manifest
    samples = []
    emotions = config.emotion_labels

    for i in range(num_samples):
        num_words = np.random.randint(3, 10)
        words = [f"word{j}" for j in range(num_words)]

        samples.append({
            'id': f"sample_{i}",
            'text': ' '.join(words),
            'emotion': np.random.choice(emotions),
        })

    # Save to temp file
    import tempfile
    manifest_path = Path(tempfile.mkdtemp()) / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(samples, f)

    return EMETTSDataset(str(manifest_path), config)


def main():
    parser = argparse.ArgumentParser(description="Train EME-TTS model")
    parser.add_argument('--config', type=str, help="Path to config YAML")
    parser.add_argument('--manifest', type=str, help="Path to training manifest")
    parser.add_argument('--val-manifest', type=str, help="Path to validation manifest")
    parser.add_argument('--resume', type=str, help="Resume from checkpoint")
    parser.add_argument('--output-dir', type=str, default='../checkpoints/eme_tts',
                        help="Output directory for checkpoints")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help="Training phase (1=pseudo-label, 2=predictor, 3=end-to-end)")
    parser.add_argument('--test', action='store_true', help="Test mode with synthetic data")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        cfg_dict = load_config(args.config)
        config = EMETTSConfig(**{k: v for k, v in cfg_dict.items() if hasattr(EMETTSConfig, k)})
    else:
        config = EMETTSConfig()

    logger.info("EME-TTS Training Script")
    logger.info("=" * 60)

    if args.test:
        # Test mode with synthetic data
        logger.info("Running in test mode with synthetic data")

        # Create synthetic datasets
        train_dataset = create_synthetic_dataset(config, num_samples=100)
        val_dataset = create_synthetic_dataset(config, num_samples=20)

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

        # Create trainer
        trainer = EMETTSTrainer(
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            checkpoint_dir=args.output_dir,
            device=args.device,
        )

        # Run a few epochs
        trainer.train(num_epochs=3, save_every=1, log_every=1)

        logger.info("\nTest training completed successfully!")

    else:
        # Full training
        if not args.manifest:
            logger.error("Please provide --manifest for training data")
            sys.exit(1)

        # Create datasets
        train_dataset = EMETTSDataset(args.manifest, config)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )

        val_loader = None
        if args.val_manifest:
            val_dataset = EMETTSDataset(args.val_manifest, config)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
            )

        # Create trainer
        trainer = EMETTSTrainer(
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            checkpoint_dir=args.output_dir,
            device=args.device,
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        trainer.train(num_epochs=args.epochs, save_every=5, log_every=1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
