#!/usr/bin/env python3
"""
Training script for SAVC (Self-Adversarial Voice Conversion) Prosody Model

Based on SAVC (arXiv:2405.00603) - Adversarial style augmentation for
learning speaker-invariant prosody representations.

Usage:
    # Train SAVC model
    python train_savc.py --config config/savc.yaml

    # Resume from checkpoint
    python train_savc.py --config config/savc.yaml --resume ../checkpoints/savc/latest.pt

    # Test mode (synthetic data)
    python train_savc.py --test
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

from savc import (
    SAVCConfig,
    SAVCAdapter,
    SAVCModule,
    AdversarialStyleAugmentor,
    StyleConsistencyLoss,
)


# =============================================================================
# DATASET
# =============================================================================

class SAVCDataset(Dataset):
    """
    Dataset for SAVC training.

    Expects pre-extracted features (HuBERT/wav2vec2) stored as:
        {
            "features": tensor [seq, dim],
            "speaker_id": int,
            "duration": float (optional),
            "emotion": str (optional),
        }
    """

    def __init__(
        self,
        manifest_path: str,
        feature_dir: str,
        max_seq_len: int = 500,
    ):
        self.feature_dir = Path(feature_dir)
        self.max_seq_len = max_seq_len

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.samples = self.manifest.get('samples', self.manifest)
        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load pre-extracted features
        feature_path = self.feature_dir / f"{sample['id']}.pt"
        if feature_path.exists():
            data = torch.load(feature_path)
            features = data['features']
        else:
            # Fallback: generate random features for testing
            seq_len = min(sample.get('duration', 5) * 50, self.max_seq_len)
            features = torch.randn(int(seq_len), 768)

        # Truncate/pad to max length
        if features.shape[0] > self.max_seq_len:
            features = features[:self.max_seq_len]

        seq_len = features.shape[0]
        mask = torch.ones(seq_len)

        return {
            'features': features,
            'mask': mask,
            'speaker_id': sample.get('speaker_id', 0),
            'sample_id': sample.get('id', str(idx)),
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int = 100, seq_len: int = 100, feature_dim: int = 768):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features = torch.randn(self.seq_len, self.feature_dim)
        mask = torch.ones(self.seq_len)
        speaker_id = idx % 10

        return {
            'features': features,
            'mask': mask,
            'speaker_id': speaker_id,
            'sample_id': str(idx),
        }


def collate_fn(batch):
    """Collate function for variable length sequences."""
    # Find max length
    max_len = max(item['features'].shape[0] for item in batch)

    features_list = []
    masks_list = []
    speaker_ids = []

    for item in batch:
        seq_len = item['features'].shape[0]
        pad_len = max_len - seq_len

        # Pad features
        if pad_len > 0:
            features = F.pad(item['features'], (0, 0, 0, pad_len))
            mask = F.pad(item['mask'], (0, pad_len))
        else:
            features = item['features']
            mask = item['mask']

        features_list.append(features)
        masks_list.append(mask)
        speaker_ids.append(item['speaker_id'])

    return {
        'features': torch.stack(features_list),
        'mask': torch.stack(masks_list),
        'speaker_id': torch.tensor(speaker_ids),
    }


# =============================================================================
# TRAINER
# =============================================================================

class SAVCTrainer:
    """
    Trainer for SAVC model.

    Training procedure:
    1. Extract features from audio (pre-computed or on-the-fly)
    2. Generate augmented views via statistic perturbation
    3. Encode all views with prosody encoder
    4. Compute consistency loss (same sample → similar prosody)
    5. Optional: Add GRL/MINE for additional disentanglement
    """

    def __init__(
        self,
        config: SAVCConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        checkpoint_dir: str = "../checkpoints/savc",
        log_interval: int = 50,
        use_disentanglement: bool = False,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.use_disentanglement = use_disentanglement

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = SAVCModule(config).to(device)

        # Optional: Add disentanglement loss
        self.disentanglement_loss = None
        if use_disentanglement:
            try:
                from disentanglement import DisentanglementConfig, DisentanglementLoss
                disentangle_config = DisentanglementConfig(
                    use_grl=True,
                    use_scheduled_grl=True,
                    num_speakers=100,
                )
                self.disentanglement_loss = DisentanglementLoss(
                    disentangle_config,
                    prosody_dim=config.prosody_dim,
                ).to(device)
            except ImportError:
                print("Warning: Disentanglement module not available")

        # Simple speaker encoder for disentanglement
        if use_disentanglement:
            from disentanglement import SimpleSpeakerEncoder
            self.speaker_encoder = SimpleSpeakerEncoder(
                input_dim=config.input_dim,
                output_dim=config.prosody_dim,
            ).to(device)
        else:
            self.speaker_encoder = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
        )

        # Tracking
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        if self.speaker_encoder:
            self.speaker_encoder.train()

        total_loss = 0.0
        total_consistency = 0.0
        total_contrastive = 0.0
        total_disentangle = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)

            # Forward pass
            result = self.model(features, mask)

            # SAVC losses
            savc_loss = result['losses']['total']

            # Optional disentanglement losses
            disentangle_loss = torch.tensor(0.0, device=self.device)
            if self.disentanglement_loss is not None and self.speaker_encoder is not None:
                # Get speaker embedding
                timbre_emb = self.speaker_encoder(features.mean(dim=1))

                disentangle_losses = self.disentanglement_loss(
                    prosody_emb=result['prosody_emb'],
                    timbre_emb=timbre_emb,
                    speaker_labels=speaker_ids,
                    epoch=self.epoch,
                )
                disentangle_loss = disentangle_losses['total']

            # Total loss
            loss = savc_loss + 0.5 * disentangle_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Tracking
            total_loss += loss.item()
            total_consistency += result['losses']['consistency_positive'].item()
            total_contrastive += result['losses']['consistency_contrastive'].item()
            total_disentangle += disentangle_loss.item()
            num_batches += 1
            self.step += 1

            # Log
            if batch_idx % self.log_interval == 0:
                print(
                    f"  Batch {batch_idx}/{len(self.train_loader)}: "
                    f"loss={loss.item():.4f}, "
                    f"consistency={result['losses']['consistency_positive'].item():.4f}, "
                    f"strength={result['augmentation_strength']:.3f}"
                )

        # Update disentanglement schedule
        if self.disentanglement_loss is not None:
            self.disentanglement_loss.update_grl_lambda(self.epoch, self.max_epochs)

        return {
            'loss': total_loss / num_batches,
            'consistency': total_consistency / num_batches,
            'contrastive': total_contrastive / num_batches,
            'disentangle': total_disentangle / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_consistency = 0.0
        num_batches = 0

        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward (no augmentation in eval)
            result = self.model(features, mask, extract_content=False)

            # For validation, just compute embedding quality metrics
            prosody_emb = result['prosody_emb']

            # Simple self-consistency metric
            # (perturb and check similarity)
            self.model.adapter.augmentor.train()  # Enable augmentation
            aug_result = self.model.adapter.augmentor.augment_single(features, mask)
            self.model.adapter.augmentor.eval()

            aug_emb = self.model.adapter.prosody_encoder(
                aug_result['perturbed'], mask
            )['prosody_emb']

            # Cosine similarity
            emb_norm = F.normalize(prosody_emb, p=2, dim=-1)
            aug_norm = F.normalize(aug_emb, p=2, dim=-1)
            consistency = (emb_norm * aug_norm).sum(dim=-1).mean()

            total_consistency += consistency.item()
            total_loss += (1.0 - consistency).item()
            num_batches += 1

        return {
            'val_loss': total_loss / max(1, num_batches),
            'val_consistency': total_consistency / max(1, num_batches),
        }

    def save_checkpoint(self, filename: str = "latest.pt"):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        if self.speaker_encoder is not None:
            checkpoint['speaker_encoder_state_dict'] = self.speaker_encoder.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint to {self.checkpoint_dir / filename}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.speaker_encoder is not None and 'speaker_encoder_state_dict' in checkpoint:
            self.speaker_encoder.load_state_dict(checkpoint['speaker_encoder_state_dict'])

        print(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    def train(self):
        """Full training loop."""
        print(f"\nStarting SAVC training for {self.max_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()
            print(f"\nTrain: loss={train_metrics['loss']:.4f}, "
                  f"consistency={train_metrics['consistency']:.4f}")

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                print(f"Val: loss={val_metrics['val_loss']:.4f}, "
                      f"consistency={val_metrics['val_consistency']:.4f}")

            # Step scheduler
            self.scheduler.step()

            # Save checkpoint
            self.save_checkpoint("latest.pt")

            # Save best
            val_loss = val_metrics.get('val_loss', train_metrics['loss'])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best.pt")
                print("  New best model!")

        print("\nTraining complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SAVC prosody model")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Run test mode with synthetic data")
    parser.add_argument("--manifest", type=str, help="Path to data manifest")
    parser.add_argument("--feature-dir", type=str, help="Directory with pre-extracted features")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--use-disentanglement", action="store_true",
                        help="Enable GRL/MINE disentanglement")

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
        config = SAVCConfig(**config_dict.get('savc', {}))
    else:
        config = SAVCConfig()

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    if args.test:
        print("\n*** TEST MODE: Using synthetic data ***\n")
        train_dataset = SyntheticDataset(num_samples=100, seq_len=100)
        val_dataset = SyntheticDataset(num_samples=20, seq_len=100)
    else:
        if not args.manifest or not args.feature_dir:
            print("Error: --manifest and --feature-dir required for training")
            print("Use --test for test mode with synthetic data")
            return

        train_dataset = SAVCDataset(
            manifest_path=args.manifest,
            feature_dir=args.feature_dir,
        )
        val_dataset = None  # Add validation manifest if available

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if not args.test else 0,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4 if not args.test else 0,
            collate_fn=collate_fn,
        )

    # Trainer
    trainer = SAVCTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        max_epochs=args.epochs if not args.test else 2,
        use_disentanglement=args.use_disentanglement,
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print("SAVC Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
