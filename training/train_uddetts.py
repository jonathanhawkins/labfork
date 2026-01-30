"""
Training script for UDDETTS: Unified Dimensional Emotion TTS with ADV Control

Based on UDDETTS (arXiv:2505.10599).

Features:
- Semi-supervised training on mixed annotation types (discrete labels + ADV)
- OT-CFM (Optimal Transport Conditional Flow Matching) decoder
- Nonlinear ADV quantization for fine-grained control
- Integration with existing prosody pipeline

Usage:
    # Train UDDETTS model
    python train_uddetts.py --config config/uddetts.yaml

    # Resume from checkpoint
    python train_uddetts.py --config config/uddetts.yaml \
        --resume ../checkpoints/uddetts/latest.pt

    # Test mode (synthetic data)
    python train_uddetts.py --test
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.uddetts import (
    UDDETTSConfig,
    UDDETTS,
    UDDETTSAdapter,
    ADV_PROTOTYPES,
    CORE_EMOTIONS,
    EMOTION_TO_IDX,
    adv_to_vad,
    vad_to_adv,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class UDDETTSDataset(Dataset):
    """
    Dataset for UDDETTS training with mixed annotation types.

    Supports:
    1. Samples with discrete emotion labels only
    2. Samples with continuous ADV annotations only
    3. Samples with both annotations
    4. Samples with neither (for unsupervised learning)
    """

    def __init__(
        self,
        manifest_path: str,
        feature_dir: str,
        max_samples: int = -1,
    ):
        """
        Args:
            manifest_path: Path to JSON manifest file
            feature_dir: Directory containing pre-extracted features
            max_samples: Maximum samples to load (-1 for all)
        """
        self.feature_dir = Path(feature_dir)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        if max_samples > 0:
            self.manifest = self.manifest[:max_samples]

        logger.info(f"Loaded {len(self.manifest)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]

        # Load prosody features (target tokens)
        feature_path = self.feature_dir / f"{item['id']}_prosody.pt"
        if feature_path.exists():
            prosody_tokens = torch.load(feature_path)
        else:
            # Generate placeholder if not exists
            prosody_tokens = torch.randn(4, 2048)

        result = {
            'id': item['id'],
            'prosody_tokens': prosody_tokens,
        }

        # Add discrete emotion if available
        if 'emotion' in item and item['emotion'] is not None:
            emotion = item['emotion'].lower()
            if emotion in EMOTION_TO_IDX:
                result['emotion_label'] = torch.tensor(EMOTION_TO_IDX[emotion])
                result['has_emotion'] = torch.tensor(True)
            else:
                result['emotion_label'] = torch.tensor(0)  # neutral
                result['has_emotion'] = torch.tensor(False)
        else:
            result['emotion_label'] = torch.tensor(0)
            result['has_emotion'] = torch.tensor(False)

        # Add ADV if available
        if 'adv' in item and item['adv'] is not None:
            result['adv'] = torch.tensor(item['adv'], dtype=torch.float32)
            result['has_adv'] = torch.tensor(True)
        elif 'vad' in item and item['vad'] is not None:
            # Convert VAD to ADV
            vad = torch.tensor(item['vad'], dtype=torch.float32)
            result['adv'] = vad_to_adv(vad)
            result['has_adv'] = torch.tensor(True)
        else:
            result['adv'] = torch.zeros(3)
            result['has_adv'] = torch.tensor(False)

        return result


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int = 1000, config: UDDETTSConfig = None):
        self.num_samples = num_samples
        self.config = config or UDDETTSConfig()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random prosody tokens
        prosody_tokens = torch.randn(
            self.config.num_prosody_tokens,
            self.config.output_dim
        )

        # Random emotion
        has_emotion = torch.rand(1) > 0.3  # 70% have emotion labels
        emotion_label = torch.randint(0, len(CORE_EMOTIONS), (1,)).squeeze()

        # Random ADV
        has_adv = torch.rand(1) > 0.4  # 60% have ADV
        adv = torch.randn(3).tanh()  # Bounded ADV

        return {
            'id': f'synthetic_{idx}',
            'prosody_tokens': prosody_tokens,
            'emotion_label': emotion_label,
            'has_emotion': has_emotion.squeeze(),
            'adv': adv,
            'has_adv': has_adv.squeeze(),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length features."""
    result = {
        'prosody_tokens': torch.stack([b['prosody_tokens'] for b in batch]),
        'emotion_label': torch.stack([b['emotion_label'] for b in batch]),
        'has_emotion': torch.stack([b['has_emotion'] for b in batch]),
        'adv': torch.stack([b['adv'] for b in batch]),
        'has_adv': torch.stack([b['has_adv'] for b in batch]),
    }
    return result


# =============================================================================
# TRAINER
# =============================================================================

class UDDETTSTrainer:
    """Trainer for UDDETTS model."""

    def __init__(
        self,
        config: UDDETTSConfig,
        model: UDDETTS,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        checkpoint_dir: str = '../checkpoints/uddetts',
        log_interval: int = 50,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        self.model.to(self.device)
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
            T_max=100,
            eta_min=1e-6,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {
            'flow': 0.0,
            'discrete': 0.0,
            'adv': 0.0,
            'consistency': 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            prosody_tokens = batch['prosody_tokens'].to(self.device)
            emotion_label = batch['emotion_label'].to(self.device)
            has_emotion = batch['has_emotion'].to(self.device)
            adv = batch['adv'].to(self.device)
            has_adv = batch['has_adv'].to(self.device)

            # Forward pass
            result = self.model(
                adv=adv if has_adv.any() else None,
                emotion_labels=emotion_label if has_emotion.any() else None,
                target_tokens=prosody_tokens,
                emotion_mask=has_emotion,
                adv_mask=has_adv,
            )

            # Compute total loss
            loss = torch.tensor(0.0, device=self.device)

            if 'flow_loss' in result:
                loss = loss + result['flow_loss']
                loss_components['flow'] += result['flow_loss'].item()

            if 'semi_supervised_loss' in result:
                ssl = result['semi_supervised_loss']
                loss = loss + ssl['total']
                loss_components['discrete'] += ssl['discrete'].item()
                loss_components['adv'] += ssl['adv'].item()
                loss_components['consistency'] += ssl['consistency'].item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                logger.info(
                    f"Epoch {self.epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return {
            'total': avg_loss,
            **loss_components,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Move to device
            prosody_tokens = batch['prosody_tokens'].to(self.device)
            emotion_label = batch['emotion_label'].to(self.device)
            has_emotion = batch['has_emotion'].to(self.device)
            adv = batch['adv'].to(self.device)
            has_adv = batch['has_adv'].to(self.device)

            # Forward pass
            result = self.model(
                adv=adv if has_adv.any() else None,
                emotion_labels=emotion_label if has_emotion.any() else None,
                target_tokens=prosody_tokens,
                emotion_mask=has_emotion,
                adv_mask=has_adv,
            )

            # Compute loss
            loss = torch.tensor(0.0, device=self.device)
            if 'flow_loss' in result:
                loss = loss + result['flow_loss']
            if 'semi_supervised_loss' in result:
                loss = loss + result['semi_supervised_loss']['total']

            total_loss += loss.item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        }

    def save_checkpoint(self, filename: str = 'latest.pt', is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, self.checkpoint_dir / filename)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
        logger.info(f"Saved checkpoint to {self.checkpoint_dir / filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, num_epochs: int):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_losses['total']:.4f} | "
                f"Flow: {train_losses['flow']:.4f} | "
                f"Discrete: {train_losses['discrete']:.4f} | "
                f"ADV: {train_losses['adv']:.4f}"
            )

            # Validate
            val_losses = self.validate()
            if val_losses:
                logger.info(f"Epoch {epoch} | Val Loss: {val_losses['val_loss']:.4f}")

                # Save best model
                if val_losses['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['val_loss']
                    self.save_checkpoint('best.pt', is_best=True)

            # Update scheduler
            self.scheduler.step()

            # Save latest checkpoint
            self.save_checkpoint('latest.pt')

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

        logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train UDDETTS model')
    parser.add_argument('--config', type=str, default='config/uddetts.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to training manifest')
    parser.add_argument('--feature-dir', type=str, default=None,
                        help='Directory with pre-extracted features')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with synthetic data')
    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config_dict = load_config(args.config)
    else:
        config_dict = {}
        logger.warning(f"Config file not found: {args.config}, using defaults")

    # Create model config
    model_config = UDDETTSConfig(**{
        k: v for k, v in config_dict.get('model', {}).items()
        if k in UDDETTSConfig.__dataclass_fields__
    })

    # Create model
    model = UDDETTS(model_config)
    logger.info(f"Created UDDETTS model with config: {model_config}")

    # Create dataset
    if args.test:
        logger.info("Running in test mode with synthetic data")
        train_dataset = SyntheticDataset(num_samples=1000, config=model_config)
        val_dataset = SyntheticDataset(num_samples=100, config=model_config)
    else:
        manifest_path = args.manifest or config_dict.get('data', {}).get('manifest')
        feature_dir = args.feature_dir or config_dict.get('data', {}).get('feature_dir')

        if manifest_path is None or feature_dir is None:
            logger.error("Must provide --manifest and --feature-dir or config file")
            sys.exit(1)

        train_dataset = UDDETTSDataset(manifest_path, feature_dir)
        val_dataset = None  # Add validation split if available

    # Create data loaders
    batch_size = config_dict.get('training', {}).get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    ) if val_dataset else None

    # Create trainer
    trainer = UDDETTSTrainer(
        config=model_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config_dict.get('training', {}).get('learning_rate', 1e-4),
        weight_decay=config_dict.get('training', {}).get('weight_decay', 0.01),
        checkpoint_dir=config_dict.get('training', {}).get('checkpoint_dir', '../checkpoints/uddetts'),
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    num_epochs = config_dict.get('training', {}).get('num_epochs', 100)
    if args.test:
        num_epochs = 2  # Quick test

    trainer.train(num_epochs)


if __name__ == '__main__':
    main()
