"""
Training script for AnCoGen attribute editor.

Trains the attribute extraction and injection networks on codec outputs
with optional supervised labels for attributes.

Usage:
    # Train with synthetic data (test mode)
    python train_ancogen.py --test

    # Train on real data with attribute labels
    python train_ancogen.py --config config/ancogen.yaml

    # Resume from checkpoint
    python train_ancogen.py --config config/ancogen.yaml \
        --resume ../checkpoints/ancogen/latest.pt

    # Analyze only (no training)
    python train_ancogen.py --analyze \
        --checkpoint ../checkpoints/ancogen/best.pt \
        --manifest ../data/manifest.json

Example config (config/ancogen.yaml):
    ```yaml
    # Model
    num_rvq_layers: 8
    codebook_size: 1024
    embedding_dim: 256

    # Training
    batch_size: 32
    learning_rate: 1e-4
    num_epochs: 50
    warmup_steps: 1000

    # Data
    manifest_path: ../data/manifest.json
    codec_type: encodec  # encodec, facodec, or custom

    # Loss weights
    reconstruction_weight: 1.0
    prediction_weight: 1.0
    orthogonality_weight: 0.1

    # Checkpointing
    checkpoint_dir: ../checkpoints/ancogen
    save_every_n_epochs: 5
    ```
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ancogen_editor import (
    AnCoGenConfig,
    AnCoGenEditor,
    AnCoGenLoss,
    AnCoGenAdapter,
    create_ancogen_editor,
    analyze_layer_contributions,
)


# =============================================================================
# SYNTHETIC DATASET FOR TESTING
# =============================================================================

class SyntheticCodecDataset(Dataset):
    """
    Synthetic dataset for testing AnCoGen training.

    Generates random codes with correlated attribute labels.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_rvq_layers: int = 8,
        seq_length: int = 100,
        codebook_size: int = 1024,
        num_speakers: int = 10,
        num_emotions: int = 8,
    ):
        self.num_samples = num_samples
        self.num_rvq_layers = num_rvq_layers
        self.seq_length = seq_length
        self.codebook_size = codebook_size
        self.num_speakers = num_speakers
        self.num_emotions = num_emotions

        # Generate synthetic data
        self.codes = torch.randint(
            0, codebook_size,
            (num_samples, num_rvq_layers, seq_length)
        )

        # Generate correlated labels
        self.speaker_ids = torch.randint(0, num_speakers, (num_samples,))
        self.emotion_ids = torch.randint(0, num_emotions, (num_samples,))
        self.f0_values = torch.randn(num_samples) * 50 + 200  # Mean 200 Hz
        self.energy_values = torch.randn(num_samples) * 0.2 + 0.5  # Mean 0.5

        # Make later layers correlate with speaker
        for i in range(num_samples):
            speaker = self.speaker_ids[i].item()
            # Bias later layer codes toward speaker-specific patterns
            self.codes[i, 4:, :] = (
                self.codes[i, 4:, :] + speaker * 100
            ) % codebook_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'codes': self.codes[idx],
            'speaker_id': self.speaker_ids[idx],
            'emotion_id': self.emotion_ids[idx],
            'f0_mean': self.f0_values[idx],
            'energy_mean': self.energy_values[idx],
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: AnCoGenEditor,
    loss_fn: AnCoGenLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'prediction': 0.0,
        'orthogonality': 0.0,
    }
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        codes = batch['codes'].to(device)

        # Prepare labels
        labels = {}
        if 'speaker_id' in batch:
            labels['speaker'] = batch['speaker_id'].to(device)
        if 'emotion_id' in batch:
            labels['emotion'] = batch['emotion_id'].to(device)
        if 'f0_mean' in batch:
            labels['pitch'] = batch['f0_mean'].to(device)
        if 'energy_mean' in batch:
            labels['energy'] = batch['energy_mean'].to(device)

        # Forward pass
        losses = loss_fn(model, codes, labels if labels else None)

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate losses
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_losses['total'] / num_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

    # Average losses
    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def validate(
    model: AnCoGenEditor,
    loss_fn: AnCoGenLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validation pass.
    """
    model.eval()
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'prediction': 0.0,
        'orthogonality': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            codes = batch['codes'].to(device)

            labels = {}
            if 'speaker_id' in batch:
                labels['speaker'] = batch['speaker_id'].to(device)
            if 'emotion_id' in batch:
                labels['emotion'] = batch['emotion_id'].to(device)
            if 'f0_mean' in batch:
                labels['pitch'] = batch['f0_mean'].to(device)
            if 'energy_mean' in batch:
                labels['energy'] = batch['energy_mean'].to(device)

            losses = loss_fn(model, codes, labels if labels else None)

            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


def train(
    config: AnCoGenConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "../checkpoints/ancogen",
    device: torch.device = None,
    resume_path: Optional[str] = None,
):
    """
    Full training loop.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model
    model = AnCoGenEditor(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize loss
    loss_fn = AnCoGenLoss(
        config,
        reconstruction_weight=1.0,
        prediction_weight=1.0,
        orthogonality_weight=0.1,
    )

    # Add attribute predictors for supervised training
    # Get attribute dimensions from dataset
    sample_batch = next(iter(train_loader))
    if 'speaker_id' in sample_batch:
        num_speakers = sample_batch['speaker_id'].max().item() + 1
        loss_fn.add_attribute_predictor('speaker', config.speaker_dim, num_speakers)
    if 'emotion_id' in sample_batch:
        num_emotions = sample_batch['emotion_id'].max().item() + 1
        loss_fn.add_attribute_predictor('emotion', config.emotion_dim, num_emotions)
    if 'f0_mean' in sample_batch:
        loss_fn.add_attribute_predictor('pitch', config.pitch_dim, None)  # Regression
    if 'energy_mean' in sample_batch:
        loss_fn.add_attribute_predictor('energy', config.energy_dim, None)

    loss_fn = loss_fn.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.attribute_predictors.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    print(f"Training on {device}")
    print(f"Starting from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_losses = train_epoch(model, loss_fn, train_loader, optimizer, device, epoch)
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"Recon: {train_losses['reconstruction']:.4f}, "
              f"Pred: {train_losses['prediction']:.4f}, "
              f"Ortho: {train_losses['orthogonality']:.4f}")

        # Validate
        if val_loader:
            val_losses = validate(model, loss_fn, val_loader, device)
            print(f"Val   - Total: {val_losses['total']:.4f}, "
                  f"Recon: {val_losses['reconstruction']:.4f}")

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'best_val_loss': best_val_loss,
                }, os.path.join(checkpoint_dir, 'best.pt'))
                print("  Saved best model!")

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'best_val_loss': best_val_loss,
            }, os.path.join(checkpoint_dir, 'latest.pt'))

    print("\nTraining complete!")
    return model, loss_fn


# =============================================================================
# ANALYSIS MODE
# =============================================================================

def analyze_model(
    model: AnCoGenEditor,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str = "analysis_output",
):
    """
    Analyze trained model's attribute extraction quality.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    print("\nAnalyzing model...")

    # Collect predictions and labels
    all_attributes = {
        'speaker': [], 'pitch': [], 'content': [], 'emotion': [], 'energy': []
    }
    all_labels = {
        'speaker': [], 'pitch': [], 'emotion': [], 'energy': []
    }

    with torch.no_grad():
        for batch in dataloader:
            codes = batch['codes'].to(device)

            # Extract attributes
            output = model(codes)
            attributes = output['attributes']

            for key in all_attributes:
                if key in attributes:
                    all_attributes[key].append(attributes[key].cpu())

            # Collect labels
            if 'speaker_id' in batch:
                all_labels['speaker'].append(batch['speaker_id'])
            if 'emotion_id' in batch:
                all_labels['emotion'].append(batch['emotion_id'])
            if 'f0_mean' in batch:
                all_labels['pitch'].append(batch['f0_mean'])
            if 'energy_mean' in batch:
                all_labels['energy'].append(batch['energy_mean'])

    # Concatenate
    for key in all_attributes:
        if all_attributes[key]:
            all_attributes[key] = torch.cat(all_attributes[key], dim=0)
    for key in all_labels:
        if all_labels[key]:
            all_labels[key] = torch.cat(all_labels[key], dim=0)

    # Analyze layer contributions
    print("\nLayer contributions per attribute:")
    sample_codes = next(iter(dataloader))['codes'][:1].to(device)

    for attr in ['speaker', 'pitch', 'content', 'emotion', 'energy']:
        contributions = analyze_layer_contributions(model, sample_codes, attr)
        print(f"\n  {attr}:")
        for layer_idx, weight in sorted(contributions.items()):
            print(f"    Layer {layer_idx}: {weight:.4f}")

    # Save analysis
    analysis_results = {
        'attributes': {k: v.numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in all_attributes.items()},
        'labels': {k: v.numpy() if isinstance(v, torch.Tensor) else v
                  for k, v in all_labels.items()},
    }

    torch.save(analysis_results, os.path.join(output_dir, 'analysis_results.pt'))
    print(f"\nAnalysis saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train AnCoGen attribute editor")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--test", action="store_true", help="Run with synthetic data")
    parser.add_argument("--analyze", action="store_true", help="Analysis mode only")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint for analysis")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--manifest", type=str, help="Data manifest path")
    parser.add_argument("--output", type=str, default="../checkpoints/ancogen",
                       help="Output directory")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.test:
        # Test mode with synthetic data
        print("Running in test mode with synthetic data...")

        config = AnCoGenConfig(
            num_rvq_layers=8,
            codebook_size=1024,
            embedding_dim=256,
        )

        # Create synthetic datasets
        train_dataset = SyntheticCodecDataset(
            num_samples=500,
            num_rvq_layers=config.num_rvq_layers,
            codebook_size=config.codebook_size,
        )
        val_dataset = SyntheticCodecDataset(
            num_samples=100,
            num_rvq_layers=config.num_rvq_layers,
            codebook_size=config.codebook_size,
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Train
        model, loss_fn = train(
            config,
            train_loader,
            val_loader,
            num_epochs=10,
            learning_rate=1e-4,
            checkpoint_dir=args.output,
            device=device,
        )

        # Quick analysis
        analyze_model(model, val_loader, device, os.path.join(args.output, 'analysis'))

        print("\nTest mode complete!")

    elif args.analyze and args.checkpoint:
        # Analysis mode
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        config = checkpoint.get('config', AnCoGenConfig())
        model = AnCoGenEditor(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Need to create a dataloader for analysis
        if args.manifest:
            print(f"Loading data from {args.manifest}")
            # TODO: Implement real data loading
            pass
        else:
            # Use synthetic data for demo
            dataset = SyntheticCodecDataset(
                num_samples=200,
                num_rvq_layers=config.num_rvq_layers,
                codebook_size=config.codebook_size,
            )
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            analyze_model(model, dataloader, device, os.path.join(args.output, 'analysis'))

    else:
        print("Please specify --test, --analyze, or --config")
        parser.print_help()


if __name__ == "__main__":
    main()
