#!/usr/bin/env python3
"""
Training script for ParaMETA: Disentangled Paralinguistic Representation Learning

Based on arXiv:2601.12289 (January 2025)

Features:
- Multi-task learning for emotion, gender, age, accent, etc.
- Task-specific subspaces with orthogonality constraints
- Prototype-based text-speech alignment
- CSM integration via prosody tokens

Usage:
    # Train ParaMETA
    python train_parameta.py --config config/parameta.yaml

    # Resume from checkpoint
    python train_parameta.py --config config/parameta.yaml \
        --resume ../checkpoints/parameta/best.pt

    # Test mode (synthetic data)
    python train_parameta.py --test
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parameta import (
    ParaMETAConfig,
    ParaMETA,
    ParaMETAAdapter,
    PARALINGUISTIC_TASKS,
    get_task_labels,
    get_num_classes,
    label_to_idx,
    describe_predictions,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class ParalinguisticDataset(Dataset):
    """
    Dataset for paralinguistic attribute learning.

    Expects a manifest file with entries containing:
    - audio_path: Path to audio file
    - features_path: Optional pre-extracted features
    - emotion: Emotion label
    - gender: Gender label
    - age: Age group label
    - accent: Accent label (optional)
    - speaking_rate: Speaking rate label (optional)
    - energy: Energy level label (optional)
    - text: Transcript (optional, for text alignment)
    """

    def __init__(
        self,
        manifest_path: str,
        config: ParaMETAConfig,
        split: str = "train",
        max_length: int = 500,
        use_precomputed_features: bool = True,
    ):
        self.config = config
        self.max_length = max_length
        self.use_precomputed_features = use_precomputed_features

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)

        # Filter by split if present
        if 'split' in self.samples[0]:
            self.samples = [s for s in self.samples if s.get('split') == split]

        logger.info(f"Loaded {len(self.samples)} samples for {split}")

        # Task label mappings
        self.task_labels = {}
        for task in PARALINGUISTIC_TASKS.keys():
            self.task_labels[task] = get_task_labels(task)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load features
        if self.use_precomputed_features and 'features_path' in sample:
            features = torch.load(sample['features_path'])
            if isinstance(features, dict):
                features = features['features']
        else:
            # Load and process audio (would use wav2vec2/HuBERT)
            # For now, placeholder
            features = torch.randn(100, self.config.input_dim)

        # Truncate/pad to max_length
        if features.shape[0] > self.max_length:
            features = features[:self.max_length]
        elif features.shape[0] < self.max_length:
            pad = torch.zeros(
                self.max_length - features.shape[0],
                features.shape[1]
            )
            features = torch.cat([features, pad], dim=0)

        # Create mask
        original_len = min(len(self.samples[idx].get('features', features)), self.max_length)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:original_len] = True

        result = {
            'features': features,
            'mask': mask,
        }

        # Add task labels
        for task in PARALINGUISTIC_TASKS.keys():
            if task in sample:
                label = sample[task]
                label_idx = label_to_idx(task, label)
                result[f'{task}_labels'] = torch.tensor(label_idx, dtype=torch.long)
            else:
                # Use -1 for missing labels (will be ignored in loss)
                result[f'{task}_labels'] = torch.tensor(-1, dtype=torch.long)

        # Text for alignment (optional)
        if 'text' in sample:
            result['text'] = sample['text']

        return result


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        config: ParaMETAConfig,
        num_samples: int = 1000,
        max_length: int = 100,
    ):
        self.config = config
        self.num_samples = num_samples
        self.max_length = max_length

        # Pre-generate samples
        self.samples = []
        for _ in range(num_samples):
            sample = {
                'features': torch.randn(max_length, config.input_dim),
                'mask': torch.ones(max_length, dtype=torch.bool),
            }

            # Random labels for each task
            for task, labels in PARALINGUISTIC_TASKS.items():
                sample[f'{task}_labels'] = torch.randint(0, len(labels), (1,)).item()

            self.samples.append(sample)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'features': sample['features'],
            'mask': sample['mask'],
            'emotion_labels': torch.tensor(sample['emotion_labels'], dtype=torch.long),
            'gender_labels': torch.tensor(sample['gender_labels'], dtype=torch.long),
            'age_labels': torch.tensor(sample['age_labels'], dtype=torch.long),
            'accent_labels': torch.tensor(sample['accent_labels'], dtype=torch.long),
            'speaking_rate_labels': torch.tensor(sample['speaking_rate_labels'], dtype=torch.long),
            'energy_labels': torch.tensor(sample['energy_labels'], dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for dataloader."""
    result = {}

    # Stack tensors
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        elif key == 'text':
            result[key] = [b[key] for b in batch]

    return result


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: ParaMETAConfig,
    gradient_accumulation: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0
    task_metrics = {task: {'loss': 0.0, 'correct': 0, 'total': 0}
                    for task in PARALINGUISTIC_TASKS.keys()}

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)

        # Prepare task labels (filter out missing labels)
        task_labels = {}
        for task in PARALINGUISTIC_TASKS.keys():
            key = f'{task}_labels'
            if key in batch:
                labels = batch[key].to(device)
                # Only include if not -1 (missing)
                valid_mask = labels >= 0
                if valid_mask.any():
                    task_labels[key] = labels

        # Forward pass
        output = model(features, mask=mask, **task_labels)

        # Backward pass
        loss = output['total_loss'] / gradient_accumulation
        loss.backward()

        # Update metrics
        total_loss += output['total_loss'].item()
        total_samples += features.shape[0]

        for task in PARALINGUISTIC_TASKS.keys():
            if f'{task}_cls_loss' in output:
                task_metrics[task]['loss'] += output[f'{task}_cls_loss'].item()
            if f'{task}_accuracy' in output:
                acc = output[f'{task}_accuracy'].item()
                batch_size = features.shape[0]
                task_metrics[task]['correct'] += int(acc * batch_size)
                task_metrics[task]['total'] += batch_size

        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {output['total_loss'].item():.4f}"
            )

    # Final gradient step if needed
    if len(dataloader) % gradient_accumulation != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    metrics = {'loss': avg_loss}

    for task, tm in task_metrics.items():
        if tm['total'] > 0:
            metrics[f'{task}_loss'] = tm['loss'] / len(dataloader)
            metrics[f'{task}_accuracy'] = tm['correct'] / tm['total']

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    task_metrics = {task: {'loss': 0.0, 'correct': 0, 'total': 0}
                    for task in PARALINGUISTIC_TASKS.keys()}

    for batch in dataloader:
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)

        task_labels = {}
        for task in PARALINGUISTIC_TASKS.keys():
            key = f'{task}_labels'
            if key in batch:
                labels = batch[key].to(device)
                valid_mask = labels >= 0
                if valid_mask.any():
                    task_labels[key] = labels

        output = model(features, mask=mask, **task_labels)

        total_loss += output['total_loss'].item()

        for task in PARALINGUISTIC_TASKS.keys():
            if f'{task}_cls_loss' in output:
                task_metrics[task]['loss'] += output[f'{task}_cls_loss'].item()
            if f'{task}_accuracy' in output:
                acc = output[f'{task}_accuracy'].item()
                batch_size = features.shape[0]
                task_metrics[task]['correct'] += int(acc * batch_size)
                task_metrics[task]['total'] += batch_size

    avg_loss = total_loss / len(dataloader)
    metrics = {'loss': avg_loss}

    for task, tm in task_metrics.items():
        if tm['total'] > 0:
            metrics[f'{task}_loss'] = tm['loss'] / len(dataloader)
            metrics[f'{task}_accuracy'] = tm['correct'] / tm['total']

    return metrics


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(
    config: ParaMETAConfig,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_dir: str,
    device: torch.device,
    resume_path: Optional[str] = None,
):
    """Main training loop."""

    # Create model
    model = ParaMETA(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler: warmup + cosine decay
    warmup_epochs = min(5, num_epochs // 10)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, device, epoch, config,
            gradient_accumulation=config.gradient_accumulation,
        )

        # Validate
        if val_dataloader is not None:
            val_metrics = validate(model, val_dataloader, device)
        else:
            val_metrics = train_metrics

        # Step scheduler
        scheduler.step()

        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} completed in {epoch_time:.1f}s | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

        # Log per-task accuracies
        for task in PARALINGUISTIC_TASKS.keys():
            train_acc = train_metrics.get(f'{task}_accuracy', 0)
            val_acc = val_metrics.get(f'{task}_accuracy', 0)
            logger.info(f"  {task}: Train {train_acc:.4f} | Val {val_acc:.4f}")

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss,
            'config': asdict(config),
        }

        # Save latest
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))
            logger.info(f"  New best model saved! Val Loss: {val_metrics['loss']:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            )

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return model


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ParaMETA for paralinguistic learning"
    )
    parser.add_argument(
        '--config', type=str, default='config/parameta.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--manifest', type=str, default=None,
        help='Path to training manifest'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run in test mode with synthetic data'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Override learning rate'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Create config
    config = ParaMETAConfig(
        input_dim=config_dict.get('input_dim', 768),
        meta_dim=config_dict.get('meta_dim', 512),
        hidden_dim=config_dict.get('hidden_dim', 512),
        task_subspace_dim=config_dict.get('task_subspace_dim', 128),
        num_prototypes=config_dict.get('num_prototypes', 128),
        prototype_dim=config_dict.get('prototype_dim', 256),
        num_layers=config_dict.get('num_layers', 6),
        num_heads=config_dict.get('num_heads', 8),
        dropout=config_dict.get('dropout', 0.1),
        classification_weight=config_dict.get('classification_weight', 1.0),
        contrastive_weight=config_dict.get('contrastive_weight', 0.5),
        prototype_weight=config_dict.get('prototype_weight', 0.3),
        orthogonality_weight=config_dict.get('orthogonality_weight', 0.1),
        intra_class_weight=config_dict.get('intra_class_weight', 0.2),
    )

    # Training settings
    num_epochs = args.epochs or int(config_dict.get('num_epochs', 100))
    batch_size = args.batch_size or int(config_dict.get('batch_size', 32))
    learning_rate = args.lr or float(config_dict.get('learning_rate', 1e-4))
    weight_decay = float(config_dict.get('weight_decay', 0.01))
    checkpoint_dir = config_dict.get('checkpoint_dir', '../checkpoints/parameta')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if args.test:
        logger.info("Running in TEST mode with synthetic data")

        # Create synthetic datasets
        train_dataset = SyntheticDataset(config, num_samples=500, max_length=100)
        val_dataset = SyntheticDataset(config, num_samples=100, max_length=100)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Short training
        num_epochs = 3

        # Train
        model = train(
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            checkpoint_dir=checkpoint_dir,
            device=device,
            resume_path=args.resume,
        )

        # Test inference
        logger.info("\nTesting inference...")
        model.eval()
        with torch.no_grad():
            test_features = torch.randn(2, 100, config.input_dim, device=device)
            predictions = model.predict(test_features)
            for i in range(2):
                pred_i = {k: v[i] if isinstance(v, list) else v[i].item()
                          for k, v in predictions.items()}
                desc = describe_predictions(pred_i)
                logger.info(f"Sample {i}: {desc}")

        # Test adapter
        logger.info("\nTesting adapter...")
        adapter = ParaMETAAdapter(config).to(device)
        adapter.parameta = model.parameta if hasattr(model, 'parameta') else model

        tokens = adapter(test_features)['prosody_tokens']
        logger.info(f"Prosody tokens shape: {tokens.shape}")

        # Test from attributes
        attr_tokens = adapter.from_attributes(
            emotion="happy",
            gender="female",
            age="young_adult",
            device=device,
        )['prosody_tokens']
        logger.info(f"Attribute tokens shape: {attr_tokens.shape}")

        logger.info("\nTest completed successfully!")

    else:
        # Real training
        manifest_path = args.manifest or config_dict.get('manifest_path')
        if manifest_path is None:
            raise ValueError("Must provide --manifest or manifest_path in config")

        # Create datasets
        train_dataset = ParalinguisticDataset(
            manifest_path, config, split='train'
        )
        val_dataset = ParalinguisticDataset(
            manifest_path, config, split='val'
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Train
        train(
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            checkpoint_dir=checkpoint_dir,
            device=device,
            resume_path=args.resume,
        )


if __name__ == '__main__':
    main()
