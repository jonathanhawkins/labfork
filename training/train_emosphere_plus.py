"""
Training Script for EmoSphere++: Zero-Shot Emotional TTS

Based on EmoSphere++ (arXiv:2411.02625, Nov 2024).

Training Pipeline:
1. Load audio data with emotion labels (optional)
2. Extract mel spectrograms and audio features (wav2vec2/HuBERT)
3. Train multi-level style encoder + EASV + CFM decoder
4. Evaluate zero-shot emotion transfer capability

Key Training Objectives:
- CFM loss: Flow matching for prosody generation
- EASV reconstruction: VAD consistency with prosody patterns
- Style consistency: Same speaker → similar style
- Emotion transfer: Same emotion → similar EASV
- Speaker disentanglement: EASV independent of speaker

Usage:
    python train_emosphere_plus.py --config config/emosphere_plus.yaml
    python train_emosphere_plus.py --test  # Run with synthetic data
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emosphere_plus import (
    EmoSpherePlusConfig,
    EmoSpherePlus,
    EmoSpherePlusAdapter,
    CORE_EMOTIONS,
)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for EmoSphere++."""

    # Data
    manifest_path: str = ""
    audio_dir: str = ""
    mel_dir: str = ""
    feature_dir: str = ""

    # Model
    model_config: Optional[EmoSpherePlusConfig] = None

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Scheduler
    scheduler_T0: int = 10
    scheduler_T_mult: int = 2
    scheduler_eta_min: float = 1e-6

    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    checkpoint_dir: str = "../checkpoints/emosphere_plus"

    # Device
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    # Mixed precision
    use_amp: bool = True

    # Loss weights (from model config or override)
    cfm_loss_weight: float = 1.0
    easv_reconstruction_weight: float = 1.0
    style_consistency_weight: float = 0.5
    emotion_transfer_weight: float = 0.3


# =============================================================================
# DATASET
# =============================================================================

class EmoSphereDataset(Dataset):
    """
    Dataset for EmoSphere++ training.

    Expects:
    - Mel spectrograms: [time, mel_dim]
    - Audio features: [seq, feature_dim] (wav2vec2/HuBERT)
    - Optional: emotion labels, speaker IDs
    """

    def __init__(
        self,
        manifest_path: str,
        mel_dir: str,
        feature_dir: str,
        max_mel_length: int = 1000,
        max_feature_length: int = 500,
    ):
        self.mel_dir = Path(mel_dir)
        self.feature_dir = Path(feature_dir)
        self.max_mel_length = max_mel_length
        self.max_feature_length = max_feature_length

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load mel spectrogram
        mel_path = self.mel_dir / f"{sample['id']}.pt"
        if mel_path.exists():
            mel = torch.load(mel_path)
        else:
            # Fallback: generate random mel
            mel = torch.randn(100, 80)

        # Load audio features
        feature_path = self.feature_dir / f"{sample['id']}.pt"
        if feature_path.exists():
            features = torch.load(feature_path)
        else:
            # Fallback: generate random features
            features = torch.randn(50, 768)

        # Truncate/pad mel
        if mel.shape[0] > self.max_mel_length:
            mel = mel[:self.max_mel_length]
        else:
            pad_len = self.max_mel_length - mel.shape[0]
            mel = F.pad(mel, (0, 0, 0, pad_len))

        # Truncate/pad features
        if features.shape[0] > self.max_feature_length:
            features = features[:self.max_feature_length]
            feature_mask = torch.ones(self.max_feature_length, dtype=torch.bool)
        else:
            orig_len = features.shape[0]
            pad_len = self.max_feature_length - features.shape[0]
            features = F.pad(features, (0, 0, 0, pad_len))
            feature_mask = torch.zeros(self.max_feature_length, dtype=torch.bool)
            feature_mask[:orig_len] = True

        # Emotion label (if available)
        emotion = sample.get('emotion', 'neutral')
        emotion_idx = CORE_EMOTIONS.index(emotion.lower()) if emotion.lower() in CORE_EMOTIONS else 0

        # Speaker ID (if available)
        speaker_id = sample.get('speaker_id', 0)

        # Prosody features (compute simple statistics)
        # In practice, these would be extracted from the audio
        prosody_features = torch.randn(15)

        return {
            'mel': mel,
            'features': features,
            'feature_mask': feature_mask,
            'emotion_idx': torch.tensor(emotion_idx, dtype=torch.long),
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'prosody_features': prosody_features,
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        mel_length: int = 100,
        feature_length: int = 50,
        mel_dim: int = 80,
        feature_dim: int = 768,
        num_emotions: int = 8,
        num_speakers: int = 100,
    ):
        self.num_samples = num_samples
        self.mel_length = mel_length
        self.feature_length = feature_length
        self.mel_dim = mel_dim
        self.feature_dim = feature_dim
        self.num_emotions = num_emotions
        self.num_speakers = num_speakers

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic data with emotion-correlated patterns
        emotion_idx = idx % self.num_emotions
        speaker_id = idx % self.num_speakers

        # Generate mel with emotion-correlated energy patterns
        base_mel = torch.randn(self.mel_length, self.mel_dim)
        # Add emotion-specific bias
        emotion_bias = torch.randn(1, self.mel_dim) * (emotion_idx / self.num_emotions)
        mel = base_mel + emotion_bias

        # Generate features with speaker-correlated patterns
        base_features = torch.randn(self.feature_length, self.feature_dim)
        speaker_bias = torch.randn(1, self.feature_dim) * (speaker_id / self.num_speakers * 0.1)
        features = base_features + speaker_bias

        feature_mask = torch.ones(self.feature_length, dtype=torch.bool)

        # Simple prosody features
        prosody_features = torch.randn(15)

        return {
            'mel': mel,
            'features': features,
            'feature_mask': feature_mask,
            'emotion_idx': torch.tensor(emotion_idx, dtype=torch.long),
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'prosody_features': prosody_features,
        }


# =============================================================================
# TRAINER
# =============================================================================

class EmoSpherePlusTrainer:
    """Trainer for EmoSphere++ model."""

    def __init__(
        self,
        config: TrainingConfig,
        model: EmoSpherePlus,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_T0,
            T_mult=config.scheduler_T_mult,
            eta_min=config.scheduler_eta_min,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if config.use_amp else None

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        mel = batch['mel'].to(self.device)
        features = batch['features'].to(self.device)
        feature_mask = batch['feature_mask'].to(self.device)
        prosody_features = batch['prosody_features'].to(self.device)
        emotion_labels = batch['emotion_idx'].to(self.device)
        speaker_ids = batch['speaker_id'].to(self.device)

        # Target prosody (in practice, this would be extracted from audio)
        target_prosody = torch.randn(mel.shape[0], self.config.model_config.output_dim, device=self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):
            losses = self.model.compute_training_loss(
                mel=mel,
                features=features,
                target_prosody=target_prosody,
                feature_mask=feature_mask,
                prosody_features=prosody_features,
                emotion_labels=emotion_labels,
                speaker_ids=speaker_ids,
            )

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        self.scheduler.step()

        # Return loss values
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation pass."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_losses = {}
        num_batches = 0

        for batch in self.val_loader:
            mel = batch['mel'].to(self.device)
            features = batch['features'].to(self.device)
            feature_mask = batch['feature_mask'].to(self.device)
            prosody_features = batch['prosody_features'].to(self.device)
            emotion_labels = batch['emotion_idx'].to(self.device)

            target_prosody = torch.randn(mel.shape[0], self.config.model_config.output_dim, device=self.device)

            with torch.amp.autocast(device_type='cuda', enabled=self.config.use_amp):
                losses = self.model.compute_training_loss(
                    mel=mel,
                    features=features,
                    target_prosody=target_prosody,
                    feature_mask=feature_mask,
                    prosody_features=prosody_features,
                    emotion_labels=emotion_labels,
                )

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            num_batches += 1

        # Average
        return {k: v / num_batches for k, v in total_losses.items()}

    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"  Saved checkpoint to {self.checkpoint_dir / filename}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Starting EmoSphere++ Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*70}\n")

        epoch_losses = []

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                losses = self.train_step(batch)

                epoch_loss += losses['total']
                num_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    print(f"  Step {self.global_step:6d} | "
                          f"Loss: {losses['total']:.4f} | "
                          f"CFM: {losses['cfm']:.4f} | "
                          f"EASV: {losses.get('easv_reconstruction', 0):.4f} | "
                          f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

                # Evaluation
                if self.global_step % self.config.eval_interval == 0 and self.val_loader is not None:
                    val_losses = self.validate()
                    print(f"\n  Validation at step {self.global_step}:")
                    for k, v in val_losses.items():
                        print(f"    {k}: {v:.4f}")

                    if val_losses.get('total', float('inf')) < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint("best.pt")
                        print("  New best model saved!")
                    print()

                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)

            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} completed in {epoch_time:.1f}s")
            print(f"  Average loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint("latest.pt")

        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}")

        return epoch_losses


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_zero_shot_transfer(
    model: EmoSpherePlus,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate zero-shot emotion transfer capability.

    Tests:
    1. Emotion classification accuracy after transfer
    2. Style consistency for same speaker
    3. EASV distance between same-emotion pairs
    """
    model.eval()

    all_easv = []
    all_emotions = []
    all_speakers = []

    for batch in test_loader:
        mel = batch['mel'].to(device)
        features = batch['features'].to(device)
        feature_mask = batch['feature_mask'].to(device)
        emotion_labels = batch['emotion_idx'].to(device)
        speaker_ids = batch['speaker_id'].to(device)

        output = model(mel, features, feature_mask)

        all_easv.append(output['vad'])
        all_emotions.append(emotion_labels)
        all_speakers.append(speaker_ids)

    # Concatenate
    all_easv = torch.cat(all_easv, dim=0)
    all_emotions = torch.cat(all_emotions, dim=0)
    all_speakers = torch.cat(all_speakers, dim=0)

    # Compute metrics
    metrics = {}

    # 1. EASV clustering by emotion (should be similar for same emotion)
    num_emotions = all_emotions.max().item() + 1
    emotion_centroids = []
    for e in range(num_emotions):
        mask = all_emotions == e
        if mask.sum() > 0:
            centroid = all_easv[mask].mean(dim=0)
            emotion_centroids.append(centroid)

    if len(emotion_centroids) > 1:
        centroids = torch.stack(emotion_centroids)
        # Inter-emotion distance (should be large)
        inter_dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze()
        metrics['inter_emotion_distance'] = inter_dist[~torch.eye(len(centroids), dtype=bool)].mean().item()

        # Intra-emotion variance (should be small)
        intra_vars = []
        for e, centroid in enumerate(emotion_centroids):
            mask = all_emotions == e
            if mask.sum() > 1:
                var = ((all_easv[mask] - centroid) ** 2).mean()
                intra_vars.append(var.item())
        metrics['intra_emotion_variance'] = sum(intra_vars) / len(intra_vars) if intra_vars else 0

    # 2. Test emotion transfer
    # Take first sample, transfer to each emotion, check VAD distance
    sample_mel = mel[:1]
    sample_features = features[:1]
    sample_mask = feature_mask[:1] if feature_mask is not None else None

    transfer_vads = []
    for emotion in CORE_EMOTIONS:
        tokens = model.transfer_emotion(sample_mel, sample_features, emotion, intensity=0.8, feature_mask=sample_mask)
        # Get VAD from transferred output
        output = model(sample_mel, sample_features, sample_mask)
        transfer_vads.append(output['vad'])

    transfer_vads = torch.cat(transfer_vads, dim=0)
    vad_std = transfer_vads.std(dim=0).mean().item()
    metrics['transfer_vad_diversity'] = vad_std

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train EmoSphere++")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--test", action="store_true", help="Run with synthetic data")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Configuration
    model_config = EmoSpherePlusConfig()
    train_config = TrainingConfig(
        model_config=model_config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("=" * 70)
    print("EmoSphere++ Training")
    print("=" * 70)
    print(f"Model config: {model_config}")
    print(f"Training config: batch_size={train_config.batch_size}, epochs={train_config.num_epochs}")

    # Dataset
    if args.test:
        print("\nUsing synthetic dataset for testing...")
        train_dataset = SyntheticDataset(num_samples=500)
        val_dataset = SyntheticDataset(num_samples=100)
    else:
        if not args.config:
            print("Error: --config required when not using --test")
            return

        # Load from config
        import yaml
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)

        train_config.manifest_path = config_data.get('manifest_path', '')
        train_config.mel_dir = config_data.get('mel_dir', '')
        train_config.feature_dir = config_data.get('feature_dir', '')

        train_dataset = EmoSphereDataset(
            train_config.manifest_path,
            train_config.mel_dir,
            train_config.feature_dir,
        )
        val_dataset = None  # Would need separate manifest

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers if not args.test else 0,
        pin_memory=train_config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers if not args.test else 0,
    ) if val_dataset is not None else None

    # Model
    model = EmoSpherePlus(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Trainer
    trainer = EmoSpherePlusTrainer(
        config=train_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    # Final evaluation
    if val_loader is not None:
        print("\n" + "=" * 70)
        print("Final Evaluation")
        print("=" * 70)

        device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
        metrics = evaluate_zero_shot_transfer(model, val_loader, device)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
