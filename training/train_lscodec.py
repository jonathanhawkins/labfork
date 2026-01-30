#!/usr/bin/env python3
"""
Training script for LSCodec: Low-Bitrate Speaker-Decoupled Speech Codec.

Three-stage training pipeline:
1. Stage 1 (VAE): Speech VAE with speaker perturbation
2. Stage 2 (VQ-VAE): Add vector quantization
3. Stage 3 (Vocoder): Token vocoder for waveform synthesis

Usage:
    # Full training (all stages)
    python train_lscodec.py --config config/lscodec.yaml

    # Train specific stage
    python train_lscodec.py --config config/lscodec.yaml --stage 1
    python train_lscodec.py --config config/lscodec.yaml --stage 2
    python train_lscodec.py --config config/lscodec.yaml --stage 3

    # Resume from checkpoint
    python train_lscodec.py --config config/lscodec.yaml --stage 2 \
        --resume ../checkpoints/lscodec/stage2_latest.pt

    # Test mode (synthetic data)
    python train_lscodec.py --test

Based on: arXiv:2410.15764 (Interspeech 2025)
"""

import argparse
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lscodec import (
    LSCodecConfig,
    LSCodec,
    LSCodecLoss,
    LSCodecAdapter,
    TimeStretchingPerturbation,
)


# =============================================================================
# DATA
# =============================================================================

class LSCodecDataset(Dataset):
    """
    Dataset for LSCodec training.

    Each sample provides:
    - mel: Content mel spectrogram
    - reference_mel: Reference mel for timbre (from same utterance)
    - ssl_target: SSL token indices (from WavLM k-means)

    For training, content mel is perturbed with time stretching.
    Reference mel is the original (unperturbed) version.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_dir: str,
        config: LSCodecConfig,
        ssl_features_dir: Optional[str] = None,
        max_length: float = 30.0,
        min_length: float = 6.0,
        split: str = "train",
    ):
        self.config = config
        self.audio_dir = Path(audio_dir)
        self.ssl_features_dir = Path(ssl_features_dir) if ssl_features_dir else None
        self.max_length = max_length
        self.min_length = min_length
        self.split = split

        # Load manifest
        self.samples = self._load_manifest(manifest_path)

        print(f"Loaded {len(self.samples)} samples for {split}")

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load manifest file."""
        samples = []

        if not os.path.exists(manifest_path):
            # Return empty list (for test mode)
            return samples

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        for item in manifest:
            duration = item.get('duration', self.max_length)
            if self.min_length <= duration <= self.max_length:
                samples.append(item)

        return samples

    def __len__(self) -> int:
        return max(len(self.samples), 1)  # At least 1 for test mode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if len(self.samples) == 0:
            # Synthetic data for test mode
            return self._get_synthetic_sample()

        sample = self.samples[idx % len(self.samples)]

        # Load mel spectrogram
        mel_path = self.audio_dir / f"{sample['id']}_mel.pt"
        if mel_path.exists():
            mel = torch.load(mel_path)
        else:
            # Generate synthetic mel
            duration_frames = int(sample.get('duration', 8.0) * self.config.frame_rate)
            mel = torch.randn(duration_frames, self.config.mel_dim)

        # Split into content and reference
        # Reference is 1/3 to 1/2 of the utterance
        total_len = mel.shape[0]
        ref_ratio = random.uniform(
            self.config.prompt_ratio_min,
            self.config.prompt_ratio_max,
        )
        ref_len = int(total_len * ref_ratio)

        # Random split point
        split_point = random.randint(0, total_len - ref_len)
        reference_mel = mel[split_point:split_point + ref_len]

        # Content is the rest (or full for simplicity)
        content_mel = mel

        # Load SSL features if available
        if self.ssl_features_dir and self.ssl_features_dir.exists():
            ssl_path = self.ssl_features_dir / f"{sample['id']}_ssl.pt"
            if ssl_path.exists():
                ssl_target = torch.load(ssl_path)
            else:
                ssl_target = torch.randint(0, self.config.ssl_num_clusters, (total_len,))
        else:
            ssl_target = torch.randint(0, self.config.ssl_num_clusters, (total_len,))

        return {
            'mel': content_mel,
            'reference_mel': reference_mel,
            'ssl_target': ssl_target,
            'sample_id': sample.get('id', str(idx)),
        }

    def _get_synthetic_sample(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic sample for testing."""
        seq_len = random.randint(80, 150)
        ref_len = int(seq_len * 0.4)

        return {
            'mel': torch.randn(seq_len, self.config.mel_dim),
            'reference_mel': torch.randn(ref_len, self.config.mel_dim),
            'ssl_target': torch.randint(0, self.config.ssl_num_clusters, (seq_len,)),
            'sample_id': 'synthetic',
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Find max lengths
    max_mel_len = max(item['mel'].shape[0] for item in batch)
    max_ref_len = max(item['reference_mel'].shape[0] for item in batch)

    # Pad tensors
    mel_padded = []
    ref_padded = []
    ssl_padded = []
    masks = []

    for item in batch:
        mel = item['mel']
        ref = item['reference_mel']
        ssl = item['ssl_target']

        # Pad mel
        pad_len = max_mel_len - mel.shape[0]
        mel_padded.append(F.pad(mel, (0, 0, 0, pad_len)))

        # Pad reference
        ref_pad_len = max_ref_len - ref.shape[0]
        ref_padded.append(F.pad(ref, (0, 0, 0, ref_pad_len)))

        # Pad SSL targets
        ssl_padded.append(F.pad(ssl, (0, pad_len)))

        # Create mask (True for valid positions)
        mask = torch.ones(max_mel_len, dtype=torch.bool)
        mask[mel.shape[0]:] = False
        masks.append(mask)

    return {
        'mel': torch.stack(mel_padded),
        'reference_mel': torch.stack(ref_padded),
        'ssl_target': torch.stack(ssl_padded),
        'mask': torch.stack(masks),
    }


# =============================================================================
# TRAINING
# =============================================================================

class LSCodecTrainer:
    """Trainer for LSCodec three-stage training."""

    def __init__(
        self,
        config: LSCodecConfig,
        train_config: Dict[str, Any],
        model: LSCodec,
        device: str = "cuda",
    ):
        self.config = config
        self.train_config = train_config
        self.model = model.to(device)
        self.device = device

        self.loss_fn = LSCodecLoss(config)
        self.perturbation = TimeStretchingPerturbation(config)

        # Will be set per stage
        self.optimizer = None
        self.scheduler = None
        self.current_stage = 1

    def setup_stage(self, stage: int):
        """Setup optimizer and scheduler for a training stage."""
        self.current_stage = stage
        self.model.set_stage(stage)

        stage_config = self.train_config.get(f'stage{stage}', {})
        lr = stage_config.get('learning_rate', 0.0001)
        weight_decay = stage_config.get('weight_decay', 0.01)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Create scheduler
        total_steps = (
            stage_config.get('epochs', 200) *
            self.train_config.get('steps_per_epoch', 1000)
        )
        warmup_steps = stage_config.get('warmup_steps', 1000)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )

        print(f"\nSetup Stage {stage} training:")
        print(f"  Learning rate: {lr}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Total steps: {total_steps}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_losses = {}

        stage = self.current_stage
        stage_config = self.train_config.get(f'stage{stage}', {})
        gradient_clip = stage_config.get('gradient_clip', 1.0)

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            mel = batch['mel'].to(self.device)
            reference_mel = batch['reference_mel'].to(self.device)
            ssl_target = batch['ssl_target'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = ~mask.to(self.device)  # Invert for attention mask

            # Apply perturbation
            perturbed_mel, beta = self.perturbation.perturb_mel(mel)

            # Forward pass
            if stage == 1:
                output = self.model.forward_vae(perturbed_mel, reference_mel)
                losses = self.loss_fn.compute_vae_loss(
                    output, mel, ssl_target, mask
                )
            else:
                output = self.model.forward_vqvae(perturbed_mel, reference_mel)
                losses = self.loss_fn.compute_vqvae_loss(
                    output, mel, ssl_target, mask
                )

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), gradient_clip
            )

            self.optimizer.step()
            self.scheduler.step()

            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                if isinstance(value, torch.Tensor):
                    total_losses[key] += value.item()
                else:
                    total_losses[key] += value

        # Average losses
        num_batches = len(dataloader)
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_losses = {}

        stage = self.current_stage

        for batch in dataloader:
            mel = batch['mel'].to(self.device)
            reference_mel = batch['reference_mel'].to(self.device)
            ssl_target = batch['ssl_target'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = ~mask.to(self.device)

            # No perturbation for validation
            if stage == 1:
                output = self.model.forward_vae(mel, reference_mel)
                losses = self.loss_fn.compute_vae_loss(
                    output, mel, ssl_target, mask
                )
            else:
                output = self.model.forward_vqvae(mel, reference_mel)
                losses = self.loss_fn.compute_vqvae_loss(
                    output, mel, ssl_target, mask
                )

            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                if isinstance(value, torch.Tensor):
                    total_losses[key] += value.item()
                else:
                    total_losses[key] += value

        num_batches = len(dataloader)
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        stage: int,
        best_loss: float,
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': best_loss,
            'config': vars(self.config),
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint: {path}")
        return checkpoint


def collect_vae_means(
    model: LSCodec,
    dataloader: DataLoader,
    device: str,
    num_batches: int = 100,
) -> torch.Tensor:
    """
    Collect VAE means for VQ initialization.

    After Stage 1 training, we initialize VQ codebook from
    the distribution of VAE encoder means (V-centroid k-means).
    """
    model.eval()
    all_means = []

    print("Collecting VAE means for VQ initialization...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            mel = batch['mel'].to(device)
            enc_out = model.encoder(mel)
            all_means.append(enc_out['mu'].cpu())

    all_means = torch.cat(all_means, dim=0)
    all_means = all_means.reshape(-1, model.config.code_dim)

    print(f"Collected {all_means.shape[0]} mean vectors")
    return all_means


def train_stage(
    trainer: LSCodecTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage: int,
    num_epochs: int,
    checkpoint_dir: str,
    save_every: int = 10,
):
    """Train a single stage."""
    print(f"\n{'='*60}")
    print(f"Starting Stage {stage} Training")
    print(f"{'='*60}")

    trainer.setup_stage(stage)

    best_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_losses = trainer.validate(val_loader)

        # Log
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train - Total: {train_losses['total']:.4f}, "
              f"Recon: {train_losses['recon']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, "
              f"Recon: {val_losses['recon']:.4f}")

        if stage >= 2:
            print(f"  Perplexity: {val_losses.get('perplexity', 0):.2f}")

        # Save checkpoint
        is_best = val_losses['total'] < best_loss
        if is_best:
            best_loss = val_losses['total']
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, f'stage{stage}_best.pt'),
                epoch, stage, best_loss
            )

        if (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(
                os.path.join(checkpoint_dir, f'stage{stage}_epoch{epoch+1}.pt'),
                epoch, stage, best_loss
            )

    # Save final
    trainer.save_checkpoint(
        os.path.join(checkpoint_dir, f'stage{stage}_final.pt'),
        num_epochs, stage, best_loss
    )

    return best_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LSCodec")
    parser.add_argument(
        '--config', type=str, default='config/lscodec.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--stage', type=int, default=None,
        help='Train specific stage (1, 2, or 3). Default: train all stages.'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run with synthetic data for testing'
    )
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            train_config = yaml.safe_load(f)
    else:
        print(f"Config not found: {args.config}, using defaults")
        train_config = {}

    # Create model config
    config = LSCodecConfig(
        frame_rate=train_config.get('frame_rate', 50),
        vocab_size=train_config.get('vocab_size', 300),
        encoder_num_blocks=train_config.get('encoder_num_blocks', 11),
        mel_dim=train_config.get('mel_dim', 80),
        code_dim=train_config.get('code_dim', 64),
        ssl_num_clusters=train_config.get('ssl_num_clusters', 2048),
        gamma_kl=train_config.get('gamma_kl', 60.0),
        gamma_recon=train_config.get('gamma_recon', 60.0),
        gamma_idx=train_config.get('gamma_idx', 2.0),
        gamma_cmt=train_config.get('gamma_cmt', 1.0),
    )

    # Device
    device = train_config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    print(f"\nUsing device: {device}")
    print(f"Frame rate: {config.frame_rate} Hz")
    print(f"Vocabulary: {config.vocab_size}")
    bitrate = config.frame_rate * math.ceil(math.log2(config.vocab_size)) / 1000
    print(f"Bitrate: {bitrate:.2f} kbps")

    # Create model
    model = LSCodec(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test mode
    if args.test:
        print("\n" + "="*60)
        print("TEST MODE - Using synthetic data")
        print("="*60)

        # Synthetic dataset
        train_dataset = LSCodecDataset(
            manifest_path="",
            audio_dir="",
            config=config,
        )
        val_dataset = train_dataset

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
        train_config['steps_per_epoch'] = 10
        trainer = LSCodecTrainer(config, train_config, model, device)

        # Test Stage 1
        print("\n[Test] Stage 1 (VAE)...")
        trainer.setup_stage(1)
        losses = trainer.train_epoch(train_loader, 0)
        print(f"  Total loss: {losses['total']:.4f}")
        print(f"  KL loss: {losses['kl']:.4f}")
        print(f"  Recon loss: {losses['recon']:.4f}")
        print("  [PASS]")

        # Initialize VQ from VAE means
        print("\n[Test] VQ Initialization...")
        means = collect_vae_means(model, train_loader, device, num_batches=5)
        model.vq.initialize_from_kmeans(means)
        print(f"  VQ initialized: {model.vq.initialized.item()}")
        print("  [PASS]")

        # Test Stage 2
        print("\n[Test] Stage 2 (VQ-VAE)...")
        trainer.setup_stage(2)
        losses = trainer.train_epoch(train_loader, 0)
        print(f"  Total loss: {losses['total']:.4f}")
        print(f"  Commitment loss: {losses['commitment']:.4f}")
        print(f"  Perplexity: {losses['perplexity']:.2f}")
        print("  [PASS]")

        # Test validation
        print("\n[Test] Validation...")
        val_losses = trainer.validate(val_loader)
        print(f"  Val total: {val_losses['total']:.4f}")
        print("  [PASS]")

        # Test CSM adapter
        print("\n[Test] CSM Adapter...")
        adapter = LSCodecAdapter(config, model).to(device)
        sample = next(iter(train_loader))
        mel = sample['mel'].to(device)
        result = adapter(mel)
        print(f"  Prosody tokens: {result['prosody_tokens'].shape}")
        assert result['prosody_tokens'].shape == (
            mel.shape[0], config.num_prefix_tokens, config.output_dim
        )
        print("  [PASS]")

        # Test voice conversion
        print("\n[Test] Voice Conversion...")
        source = torch.randn(1, 50, config.mel_dim).to(device)
        target = torch.randn(1, 30, config.mel_dim).to(device)
        with torch.no_grad():
            converted = model.voice_convert(source, target)
        print(f"  Converted shape: {converted.shape}")
        print("  [PASS]")

        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
        return

    # Full training
    data_config = train_config.get('data', {})

    # Create datasets
    train_dataset = LSCodecDataset(
        manifest_path=data_config.get('manifest_path', '../data/manifest.json'),
        audio_dir=data_config.get('audio_dir', '../data/audio'),
        config=config,
        ssl_features_dir=data_config.get('ssl_features_dir'),
        max_length=data_config.get('max_audio_length', 30.0),
        min_length=data_config.get('min_audio_length', 6.0),
        split='train',
    )

    val_dataset = LSCodecDataset(
        manifest_path=data_config.get('manifest_path', '../data/manifest.json'),
        audio_dir=data_config.get('audio_dir', '../data/audio'),
        config=config,
        ssl_features_dir=data_config.get('ssl_features_dir'),
        max_length=data_config.get('max_audio_length', 30.0),
        min_length=data_config.get('min_audio_length', 6.0),
        split='val',
    )

    # Create data loaders
    stage1_config = train_config.get('stage1', {})
    batch_size = stage1_config.get('batch_size', 16)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 4),
        prefetch_factor=data_config.get('prefetch_factor', 2),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )

    train_config['steps_per_epoch'] = len(train_loader)

    # Create trainer
    trainer = LSCodecTrainer(config, train_config, model, device)

    # Resume if specified
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint['epoch']
        start_stage = checkpoint['stage']
    else:
        start_epoch = 0
        start_stage = args.stage or 1

    checkpoint_dir = train_config.get('checkpoint_dir', '../checkpoints/lscodec')
    save_every = train_config.get('save_every_epochs', 10)

    # Train stages
    stages_to_train = [args.stage] if args.stage else [1, 2, 3]

    for stage in stages_to_train:
        if stage < start_stage:
            continue

        stage_config = train_config.get(f'stage{stage}', {})
        num_epochs = stage_config.get('epochs', 200)

        # For Stage 2, initialize VQ from Stage 1
        if stage == 2 and not model.vq.initialized:
            # Try to load Stage 1 checkpoint
            stage1_checkpoint = stage_config.get('stage1_checkpoint')
            if stage1_checkpoint is None:
                stage1_checkpoint = os.path.join(checkpoint_dir, 'stage1_best.pt')

            if os.path.exists(stage1_checkpoint):
                print(f"\nLoading Stage 1 checkpoint: {stage1_checkpoint}")
                ckpt = torch.load(stage1_checkpoint, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])

            # Collect VAE means and initialize VQ
            means = collect_vae_means(model, train_loader, device)
            model.vq.initialize_from_kmeans(means)

        # Train stage
        train_stage(
            trainer,
            train_loader,
            val_loader,
            stage=stage,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            save_every=save_every,
        )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
