#!/usr/bin/env python3
"""
Training script for Marco-Voice rotational emotion embeddings.

Based on arXiv:2508.02038 - A Unified Framework for Expressive Speech Synthesis

Key features:
- Rotational emotion embeddings (direction from neutral)
- Cross-orthogonal constraint for speaker-emotion disentanglement
- In-batch contrastive learning for emotion distinction
- Cross-attention integration with LM outputs

Usage:
    # Basic training
    python train_marco_voice.py --config config/marco_voice.yaml

    # Resume from checkpoint
    python train_marco_voice.py --config config/marco_voice.yaml \
        --resume ../checkpoints/marco_voice/best.pt

    # Test mode (verify setup)
    python train_marco_voice.py --test
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import yaml

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marco_voice import (
    MarcoVoiceConfig,
    MarcoVoiceEmotionModule,
    MarcoVoiceAdapter,
    MarcoVoiceLoss,
    MARCO_EMOTIONS,
    EMOTION_TO_IDX,
)


# =============================================================================
# DATASET
# =============================================================================

class EmotionPairDataset(Dataset):
    """
    Dataset that provides (emotional, neutral) pairs for rotational encoding.

    Each sample contains:
    - Emotional audio features
    - Neutral audio features (same speaker)
    - Emotion label
    - Speaker ID
    """

    def __init__(
        self,
        manifest_path: str,
        feature_extractor: Optional[nn.Module] = None,
        max_duration: float = 15.0,
        min_duration: float = 0.5,
        sample_rate: int = 16000,
        require_neutral_pairs: bool = True,
    ):
        self.manifest_path = manifest_path
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.require_neutral_pairs = require_neutral_pairs

        # Load manifest
        self.samples = self._load_manifest()

        # Build speaker -> neutral samples mapping
        self.speaker_neutrals = self._build_neutral_mapping()

        # Filter samples that have neutral pairs if required
        if require_neutral_pairs:
            self.samples = [
                s for s in self.samples
                if s['speaker_id'] in self.speaker_neutrals
            ]

        print(f"Loaded {len(self.samples)} samples with neutral pairs")
        print(f"Speakers with neutrals: {len(self.speaker_neutrals)}")

    def _load_manifest(self) -> List[Dict]:
        """Load samples from manifest file."""
        samples = []

        if not os.path.exists(self.manifest_path):
            print(f"Warning: Manifest not found at {self.manifest_path}")
            return samples

        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)

        for entry in manifest:
            # Check required fields
            if 'audio_path' not in entry:
                continue

            # Get emotion label
            emotion = entry.get('emotion', 'neutral').lower()
            if emotion not in EMOTION_TO_IDX:
                continue

            # Get speaker ID
            speaker_id = entry.get('speaker_id', entry.get('speaker', 'unknown'))

            # Check duration
            duration = entry.get('duration', 0)
            if duration > 0:
                if duration < self.min_duration or duration > self.max_duration:
                    continue

            samples.append({
                'audio_path': entry['audio_path'],
                'emotion': emotion,
                'emotion_idx': EMOTION_TO_IDX[emotion],
                'speaker_id': speaker_id,
                'duration': duration,
                'text': entry.get('text', ''),
            })

        return samples

    def _build_neutral_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from speaker_id to neutral audio paths."""
        speaker_neutrals = defaultdict(list)

        for sample in self.samples:
            if sample['emotion'] == 'neutral':
                speaker_neutrals[sample['speaker_id']].append(sample['audio_path'])

        return dict(speaker_neutrals)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Trim to max duration
        max_samples = int(self.max_duration * self.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        return waveform.squeeze(0)  # [time]

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform."""
        if self.feature_extractor is not None:
            with torch.no_grad():
                features = self.feature_extractor(waveform.unsqueeze(0))
                if isinstance(features, dict):
                    features = features.get('last_hidden_state', features.get('features'))
                return features.squeeze(0)  # [seq, dim]
        else:
            # Return raw waveform for later processing
            return waveform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load emotional audio
        try:
            emotional_waveform = self._load_audio(sample['audio_path'])
            emotional_features = self._extract_features(emotional_waveform)
        except Exception as e:
            print(f"Error loading {sample['audio_path']}: {e}")
            # Return dummy data
            return self.__getitem__((idx + 1) % len(self))

        # Load neutral audio from same speaker
        neutral_features = None
        if sample['speaker_id'] in self.speaker_neutrals:
            neutral_path = random.choice(self.speaker_neutrals[sample['speaker_id']])
            try:
                neutral_waveform = self._load_audio(neutral_path)
                neutral_features = self._extract_features(neutral_waveform)
            except Exception as e:
                print(f"Error loading neutral {neutral_path}: {e}")

        # If no neutral, use zero vector (will use learnable neutral)
        if neutral_features is None:
            neutral_features = torch.zeros_like(emotional_features)

        return {
            'emotional_features': emotional_features,
            'neutral_features': neutral_features,
            'emotion_idx': torch.tensor(sample['emotion_idx'], dtype=torch.long),
            'speaker_id': sample['speaker_id'],
            'emotion': sample['emotion'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Find max sequence length
    max_len = max(b['emotional_features'].shape[0] for b in batch)

    emotional_features = []
    neutral_features = []
    emotion_idx = []
    mask = []

    for b in batch:
        emo_feat = b['emotional_features']
        neu_feat = b['neutral_features']
        seq_len = emo_feat.shape[0]

        # Pad to max length
        if emo_feat.dim() == 1:
            # Raw waveform
            pad_len = max_len - seq_len
            emo_padded = F.pad(emo_feat, (0, pad_len))
            neu_padded = F.pad(neu_feat, (0, pad_len))
            m = torch.ones(max_len)
            m[seq_len:] = 0
        else:
            # Features [seq, dim]
            pad_len = max_len - seq_len
            emo_padded = F.pad(emo_feat, (0, 0, 0, pad_len))
            neu_padded = F.pad(neu_feat, (0, 0, 0, pad_len))
            m = torch.ones(max_len)
            m[seq_len:] = 0

        emotional_features.append(emo_padded)
        neutral_features.append(neu_padded)
        emotion_idx.append(b['emotion_idx'])
        mask.append(m)

    return {
        'emotional_features': torch.stack(emotional_features),
        'neutral_features': torch.stack(neutral_features),
        'emotion_idx': torch.stack(emotion_idx),
        'mask': torch.stack(mask),
    }


# =============================================================================
# TRAINER
# =============================================================================

class MarcoVoiceTrainer:
    """Trainer for Marco-Voice rotational emotion embeddings."""

    def __init__(
        self,
        config: Dict,
        model: MarcoVoiceEmotionModule,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.loss_fn = MarcoVoiceLoss(
            MarcoVoiceConfig(**config.get('model', {}))
        ).to(device)

        # Optimizer
        training_config = config.get('training', {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-4),
            weight_decay=training_config.get('weight_decay', 0.01),
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.get('max_steps', 100000),
            eta_min=training_config.get('min_lr', 1e-6),
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(training_config.get('checkpoint_dir', '../checkpoints/marco_voice'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_every = training_config.get('log_every_steps', 100)
        self.eval_every = training_config.get('eval_every_steps', 500)
        self.save_every = training_config.get('save_every_steps', 2000)

        # AMP
        self.use_amp = training_config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and 'cuda' in device else None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        emotional_features = batch['emotional_features'].to(self.device)
        neutral_features = batch['neutral_features'].to(self.device)
        emotion_idx = batch['emotion_idx'].to(self.device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(self.device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.use_amp and 'cuda' in self.device):
            output = self.model(
                emotional_features,
                neutral_features,
                lm_output=None,  # No LM output during initial training
                emotion_labels=emotion_idx,
                compute_losses=True,
            )

            losses = output['losses']
            total_loss = losses['total']

        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('training', {}).get('max_grad_norm', 1.0)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('training', {}).get('max_grad_norm', 1.0)
            )
            self.optimizer.step()

        self.scheduler.step()

        # Return metrics
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_losses = defaultdict(float)
        num_batches = 0
        correct = 0
        total = 0

        for batch in self.val_loader:
            emotional_features = batch['emotional_features'].to(self.device)
            neutral_features = batch['neutral_features'].to(self.device)
            emotion_idx = batch['emotion_idx'].to(self.device)

            output = self.model(
                emotional_features,
                neutral_features,
                emotion_labels=emotion_idx,
                compute_losses=True,
            )

            losses = output['losses']
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    total_losses[k] += v.item()

            # Compute accuracy
            preds = output['emotion_logits'].argmax(dim=-1)
            correct += (preds == emotion_idx).sum().item()
            total += emotion_idx.shape[0]

            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses['accuracy'] = correct / total if total > 0 else 0

        return avg_losses

    def train(self, num_epochs: int = 100):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

        max_steps = self.config.get('training', {}).get('max_steps', 100000)
        patience = self.config.get('training', {}).get('patience', 10)

        for epoch in range(num_epochs):
            epoch_losses = defaultdict(float)
            num_batches = 0

            for batch in self.train_loader:
                losses = self.train_step(batch)

                for k, v in losses.items():
                    epoch_losses[k] += v
                num_batches += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {self.global_step} | "
                          f"Loss: {avg_losses['total']:.4f} | "
                          f"Ortho: {avg_losses['orthogonal_loss']:.4f} | "
                          f"Contrast: {avg_losses['contrastive_loss']:.4f} | "
                          f"Class: {avg_losses['classification_loss']:.4f} | "
                          f"LR: {lr:.2e}")

                # Validation
                if self.global_step % self.eval_every == 0:
                    val_losses = self.validate()
                    if val_losses:
                        print(f"Validation | "
                              f"Loss: {val_losses['total']:.4f} | "
                              f"Acc: {val_losses['accuracy']:.4f}")

                        # Check for improvement
                        if val_losses['total'] < self.best_val_loss:
                            self.best_val_loss = val_losses['total']
                            self.patience_counter = 0
                            self.save_checkpoint('best.pt')
                        else:
                            self.patience_counter += 1

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

                # Check stopping conditions
                if self.global_step >= max_steps:
                    print(f"Reached max steps ({max_steps})")
                    return

                if self.patience_counter >= patience:
                    print(f"Early stopping after {patience} epochs without improvement")
                    return

            # Epoch summary
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_losses['total']:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {self.checkpoint_dir / filename}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_mode():
    """Run quick test with synthetic data."""
    print("=" * 60)
    print("Marco-Voice Training - Test Mode")
    print("=" * 60)

    # Create config
    config = {
        'model': {
            'input_dim': 768,
            'emotion_dim': 256,
            'speaker_dim': 256,
            'hidden_dim': 512,
            'output_dim': 2048,
            'num_emotions': 7,
            'lambda_orth': 0.1,
            'lambda_contrast': 0.5,
        },
        'training': {
            'learning_rate': 1e-4,
            'max_grad_norm': 1.0,
            'checkpoint_dir': '/tmp/marco_voice_test',
            'log_every_steps': 1,
            'eval_every_steps': 5,
            'save_every_steps': 10,
            'max_steps': 20,
        },
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model_config = MarcoVoiceConfig(**config['model'])
    model = MarcoVoiceEmotionModule(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create synthetic dataset
    class SyntheticDataset(Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'emotional_features': torch.randn(50, 768),
                'neutral_features': torch.randn(50, 768),
                'emotion_idx': torch.randint(0, 7, ()),
            }

    train_loader = DataLoader(
        SyntheticDataset(100),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        SyntheticDataset(20),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Create trainer
    trainer = MarcoVoiceTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train for a few steps
    print("\nRunning test training...")
    trainer.train(num_epochs=5)

    print("\n" + "=" * 60)
    print("Test mode completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train Marco-Voice rotational emotion embeddings")
    parser.add_argument('--config', type=str, default='config/marco_voice.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run test mode with synthetic data')
    args = parser.parse_args()

    if args.test:
        test_mode()
        return

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Determine device
    hardware_config = config.get('hardware', {})
    device = hardware_config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Create model
    model_config = MarcoVoiceConfig(**config.get('model', {}))
    model = MarcoVoiceEmotionModule(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset
    training_config = config.get('training', {})
    manifest_path = training_config.get('manifest_path', '../data/emotion_manifest.json')

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please create an emotion manifest with the following format:")
        print("""
[
    {
        "audio_path": "path/to/audio.wav",
        "emotion": "happy",
        "speaker_id": "speaker_001",
        "duration": 3.5,
        "text": "Optional transcript"
    },
    ...
]
        """)
        print("\nRun with --test to verify setup works correctly.")
        return

    dataset = EmotionPairDataset(
        manifest_path=manifest_path,
        max_duration=config.get('features', {}).get('max_duration', 15.0),
        min_duration=config.get('features', {}).get('min_duration', 0.5),
        sample_rate=config.get('features', {}).get('sample_rate', 16000),
        require_neutral_pairs=training_config.get('require_neutral_pairs', True),
    )

    if len(dataset) == 0:
        print("Error: No valid samples found in manifest")
        return

    # Split into train/val
    val_size = min(int(len(dataset) * 0.1), 500)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=True,
        num_workers=hardware_config.get('num_workers', 4),
        pin_memory=hardware_config.get('pin_memory', True),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get('batch_size', 16),
        shuffle=False,
        num_workers=hardware_config.get('num_workers', 4),
        pin_memory=hardware_config.get('pin_memory', True),
        collate_fn=collate_fn,
    )

    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

    # Create trainer
    trainer = MarcoVoiceTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    num_epochs = training_config.get('max_steps', 100000) // len(train_loader) + 1
    trainer.train(num_epochs=num_epochs)


if __name__ == '__main__':
    main()
