#!/usr/bin/env python3
"""
Training script for E3-VITS batch-permuted style perturbation.

Based on E3-VITS (ICML 2023): "Emotional End-to-End TTS with Cross-speaker Style Transfer"
Paper: https://openreview.net/forum?id=qL47xtuEuv
GitHub: https://github.com/Wonbin-Jung/e3-vits

Usage:
    # Train E3-VITS model
    python train_e3_vits.py --config config/e3_vits.yaml

    # Resume from checkpoint
    python train_e3_vits.py --config config/e3_vits.yaml \
      --resume ../checkpoints/e3_vits/best.pt

    # Test mode with synthetic data
    python train_e3_vits.py --test
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    yaml = None

from training.e3_vits import (
    E3VITSConfig,
    E3VITSAdapter,
    E3VITSLoss,
    EMOTION_LABELS,
    EMOTION_TO_IDX,
)


# =============================================================================
# SYNTHETIC DATASET FOR TESTING
# =============================================================================

class SyntheticEmotionDataset(Dataset):
    """Synthetic dataset for testing E3-VITS training."""

    def __init__(
        self,
        num_samples: int = 1000,
        num_speakers: int = 100,
        seq_len: int = 100,
        feature_dim: int = 768,
    ):
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_emotions = len(EMOTION_LABELS)

        # Generate random assignments
        self.speaker_ids = torch.randint(0, num_speakers, (num_samples,))
        self.emotion_ids = torch.randint(0, self.num_emotions, (num_samples,))

        # Simulate disjoint dataset (not all speakers have all emotions)
        # Each speaker only has 3-5 emotions
        self.speaker_emotions = {}
        for speaker in range(num_speakers):
            num_emo = torch.randint(3, 6, (1,)).item()
            available = torch.randperm(self.num_emotions)[:num_emo]
            self.speaker_emotions[speaker] = available

        # Reassign emotions to match disjoint constraint
        for i in range(num_samples):
            speaker = self.speaker_ids[i].item()
            available = self.speaker_emotions[speaker]
            self.emotion_ids[i] = available[torch.randint(0, len(available), (1,))].item()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        speaker_id = self.speaker_ids[idx]
        emotion_id = self.emotion_ids[idx]

        # Generate synthetic features with some emotion-dependent patterns
        features = torch.randn(self.seq_len, self.feature_dim)

        # Add emotion-specific bias (to make emotions distinguishable)
        emotion_bias = emotion_id.float() / self.num_emotions * 0.5
        features[:, :10] += emotion_bias

        # Add speaker-specific bias
        speaker_bias = speaker_id.float() / self.num_speakers * 0.3
        features[:, 10:20] += speaker_bias

        return {
            'features': features,
            'speaker_id': speaker_id,
            'emotion_id': emotion_id,
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: E3VITSAdapter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: E3VITSConfig,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        speaker_ids = batch['speaker_id'].to(device)
        emotion_ids = batch['emotion_id'].to(device)

        # Forward pass with batch permutation
        result = model(
            features=features,
            speaker_ids=speaker_ids,
            emotion_ids=emotion_ids,
            use_permutation=True,
        )

        # Get losses
        losses = result['losses']
        total_loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()

        num_batches += 1

        # Log progress
        if batch_idx % 10 == 0:
            perm_info = result['permutation_info']
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"loss={total_loss.item():.4f}, "
                  f"permuted={perm_info['permuted']}")

    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches

    return total_losses


def validate(
    model: E3VITSAdapter,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_losses = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            emotion_ids = batch['emotion_id'].to(device)

            # Forward pass without permutation for validation
            result = model(
                features=features,
                speaker_ids=speaker_ids,
                emotion_ids=emotion_ids,
                use_permutation=False,
            )

            # We still want to check that the model works with permutation
            result_perm = model(
                features=features,
                speaker_ids=speaker_ids,
                emotion_ids=emotion_ids,
                use_permutation=True,
            )

            # Compute style consistency
            style_diff = (result['style_original'] - result_perm['style_original']).norm(dim=-1).mean()

            for k, v in result['losses'].items():
                if isinstance(v, torch.Tensor):
                    if k not in total_losses:
                        total_losses[k] = 0.0
                    total_losses[k] += v.item()

            total_losses['style_consistency'] = total_losses.get('style_consistency', 0.0) + style_diff.item()
            num_batches += 1

    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches

    return total_losses


def evaluate_cross_speaker_transfer(
    model: E3VITSAdapter,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate cross-speaker emotion transfer quality."""
    model.eval()

    style_distances = []
    emotion_preservations = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            speaker_ids = batch['speaker_id'].to(device)
            emotion_ids = batch['emotion_id'].to(device)

            # Get original style
            result_orig = model(
                features=features,
                speaker_ids=speaker_ids,
                emotion_ids=emotion_ids,
                use_permutation=False,
            )

            # Get perturbed style (cross-speaker)
            result_perm = model(
                features=features,
                speaker_ids=speaker_ids,
                emotion_ids=emotion_ids,
                use_permutation=True,
            )

            # Compute style distance (should be different due to speaker change)
            style_dist = (result_orig['style_original'] - result_perm['style_perturbed']).norm(dim=-1)
            style_distances.extend(style_dist.cpu().tolist())

            # Check if emotion is preserved after permutation
            # (perturbed emotion should come from different sample but be consistent)
            perm_info = result_perm['permutation_info']
            emotion_perm_idx = perm_info['emotion_perm_idx']
            original_emotions = emotion_ids
            perturbed_emotions = emotion_ids[emotion_perm_idx]

            # Compute emotion embedding similarity
            orig_emo_emb = result_orig['emotion_emb']
            perm_emo_emb = result_perm['emotion_emb'][emotion_perm_idx]
            emo_sim = F.cosine_similarity(orig_emo_emb, perm_emo_emb, dim=-1)
            emotion_preservations.extend(emo_sim.cpu().tolist())

    metrics = {
        'style_distance_mean': sum(style_distances) / len(style_distances),
        'style_distance_std': (sum((x - sum(style_distances)/len(style_distances))**2 for x in style_distances) / len(style_distances)) ** 0.5,
        'emotion_preservation_mean': sum(emotion_preservations) / len(emotion_preservations),
    }

    return metrics


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    config_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    test_mode: bool = False,
):
    """Main training function."""
    print("=" * 60)
    print("E3-VITS Training")
    print("=" * 60)

    # Load config
    if config_path and yaml:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        config_dict = {}

    # Create config
    config = E3VITSConfig(
        input_dim=config_dict.get('input_dim', 768),
        hidden_dim=config_dict.get('hidden_dim', 512),
        style_dim=config_dict.get('style_dim', 256),
        output_dim=config_dict.get('output_dim', 2048),
        flow_num_layers=config_dict.get('flow_num_layers', 4),
        flow_num_flows=config_dict.get('flow_num_flows', 4),
        permutation_probability=config_dict.get('permutation_probability', 0.5),
        use_discriminator=config_dict.get('use_discriminator', True),
        warmup_steps=config_dict.get('warmup_steps', 1000),
    )

    # Training settings
    num_epochs = config_dict.get('num_epochs', 100 if not test_mode else 2)
    batch_size = config_dict.get('batch_size', 32 if not test_mode else 8)
    learning_rate = config_dict.get('learning_rate', 1e-4)
    num_speakers = config_dict.get('num_speakers', 100)
    num_samples = config_dict.get('num_samples', 1000 if not test_mode else 100)
    checkpoint_dir = config_dict.get('checkpoint_dir', '../checkpoints/e3_vits')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SyntheticEmotionDataset(
        num_samples=num_samples,
        num_speakers=num_speakers,
        feature_dim=config.input_dim,
    )
    val_dataset = SyntheticEmotionDataset(
        num_samples=num_samples // 5,
        num_speakers=num_speakers,
        feature_dim=config.input_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Print dataset info
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Number of speakers: {num_speakers}")
    print(f"Number of emotions: {len(EMOTION_LABELS)}")

    # Check disjoint property
    total_pairs = 0
    for speaker in range(min(10, num_speakers)):
        available = train_dataset.speaker_emotions[speaker]
        print(f"  Speaker {speaker}: emotions {[EMOTION_LABELS[e] for e in available.tolist()]}")
        total_pairs += len(available)
    print(f"Average emotions per speaker: {total_pairs / min(10, num_speakers):.1f}")

    # Create model
    print("\nCreating E3-VITS model...")
    model = E3VITSAdapter(config, num_speakers=num_speakers).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Train
        train_losses = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch,
        )

        print(f"\nTrain losses:")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.4f}")

        # Validate
        val_losses = validate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(f"\nValidation losses:")
        for k, v in val_losses.items():
            print(f"  {k}: {v:.4f}")

        # Evaluate cross-speaker transfer
        transfer_metrics = evaluate_cross_speaker_transfer(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print(f"\nCross-speaker transfer metrics:")
        for k, v in transfer_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Update scheduler
        scheduler.step()

        # Save checkpoint
        val_loss = val_losses.get('total', 0.0)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'transfer_metrics': transfer_metrics,
            'config': config.__dict__,
        }

        # Save latest
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best.pt'))
            print(f"  Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)

    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train E3-VITS model")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with synthetic data'
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_path=args.resume,
        test_mode=args.test,
    )
