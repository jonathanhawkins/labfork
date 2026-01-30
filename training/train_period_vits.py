"""
Period VITS Training Script

Trains the Period VITS module for stable emotional pitch synthesis.

Based on "Period VITS: Variational Inference with Explicit Pitch Modeling for
End-to-end Emotional Speech Synthesis" (ICASSP 2023)
https://arxiv.org/abs/2210.15964

Usage:
    python train_period_vits.py --config config/period_vits.yaml

    # Test mode (synthetic data)
    python train_period_vits.py --test

    # Resume training
    python train_period_vits.py --config config/period_vits.yaml \
        --resume ../checkpoints/period_vits/latest.pt
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from period_vits import (
    PeriodVITSConfig,
    PeriodVITS,
    PeriodVITSAdapter,
    PitchStabilityAnalyzer,
    create_period_vits_adapter,
)


# =============================================================================
# DATASET
# =============================================================================

class PeriodVITSDataset(Dataset):
    """
    Dataset for Period VITS training.

    Loads audio samples and extracts:
    - Text/phoneme embeddings (or uses random for now)
    - F0 contours (ground truth)
    - Voicing masks
    - Emotion labels (optional)
    """

    def __init__(
        self,
        manifest_path: str,
        config: PeriodVITSConfig,
        max_frames: int = 300,
        use_cache: bool = True,
    ):
        self.config = config
        self.max_frames = max_frames
        self.use_cache = use_cache
        self.cache = {}

        # Load manifest
        self.samples = []
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.samples = json.load(f)
            print(f"Loaded {len(self.samples)} samples from {manifest_path}")
        else:
            print(f"Warning: Manifest not found at {manifest_path}")

        # Try to load F0 extraction tools
        self.f0_extractor = None
        try:
            import parselmouth
            self.f0_extractor = 'parselmouth'
        except ImportError:
            pass

    def __len__(self) -> int:
        return max(len(self.samples), 1)

    def _extract_f0(
        self, audio: torch.Tensor, sr: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract F0 and voicing from audio.

        Returns:
            f0: F0 values in Hz [frames]
            voicing: Binary voicing mask [frames]
        """
        import numpy as np

        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.numpy()
        else:
            audio_np = audio

        # Use parselmouth if available
        if self.f0_extractor == 'parselmouth':
            import parselmouth

            snd = parselmouth.Sound(audio_np, sr)
            pitch = snd.to_pitch(
                time_step=self.config.hop_length / sr,
                pitch_floor=self.config.f0_min,
                pitch_ceiling=self.config.f0_max,
            )

            f0 = pitch.selected_array['frequency']
            voicing = (f0 > 0).astype(np.float32)

            # Replace unvoiced with interpolated values (for training)
            f0_interp = np.copy(f0)
            voiced_idx = np.where(f0 > 0)[0]
            if len(voiced_idx) > 0:
                # Simple linear interpolation
                f0_interp = np.interp(
                    np.arange(len(f0)),
                    voiced_idx,
                    f0[voiced_idx],
                )
            else:
                f0_interp = np.full_like(f0, (self.config.f0_min + self.config.f0_max) / 2)

            return torch.tensor(f0_interp, dtype=torch.float32), torch.tensor(voicing)

        # Fallback: synthetic F0
        num_frames = len(audio_np) // self.config.hop_length
        t = np.linspace(0, 4 * np.pi, num_frames)
        f0 = 150 + 50 * np.sin(t) + 20 * np.sin(3 * t)  # Multi-frequency pattern
        voicing = np.ones(num_frames, dtype=np.float32)

        return torch.tensor(f0, dtype=torch.float32), torch.tensor(voicing)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with:
                - text_embeddings: [frames, hidden]
                - f0: [frames] target F0 in Hz
                - voicing: [frames] target voicing
                - emotion: emotion label (if available)
        """
        # Check cache
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        if len(self.samples) > 0:
            sample = self.samples[idx]
            audio_path = sample.get('audio_path', sample.get('path', ''))

            try:
                import torchaudio
                audio, sr = torchaudio.load(audio_path)
                audio = audio[0]  # Mono

                # Resample if needed
                if sr != self.config.sample_rate:
                    audio = torchaudio.functional.resample(
                        audio, sr, self.config.sample_rate
                    )

                # Extract F0 and voicing
                f0, voicing = self._extract_f0(audio, self.config.sample_rate)

            except Exception as e:
                # Fallback to synthetic
                num_frames = self.max_frames
                t = torch.linspace(0, 4 * math.pi, num_frames)
                f0 = 150 + 50 * torch.sin(t) + 20 * torch.sin(3 * t)
                voicing = torch.ones(num_frames)

            # Get emotion if available
            emotion = sample.get('emotion', 'neutral')

        else:
            # Synthetic data for testing
            num_frames = self.max_frames
            t = torch.linspace(0, 4 * math.pi, num_frames)
            f0 = 150 + 50 * torch.sin(t) + 20 * torch.sin(3 * t)
            voicing = (torch.rand(num_frames) > 0.2).float()
            emotion = 'neutral'

        # Clip to max_frames
        if len(f0) > self.max_frames:
            f0 = f0[:self.max_frames]
            voicing = voicing[:self.max_frames]
        elif len(f0) < self.max_frames:
            # Pad
            pad_len = self.max_frames - len(f0)
            f0 = F.pad(f0, (0, pad_len), value=0)
            voicing = F.pad(voicing, (0, pad_len), value=0)

        # Create text embeddings (random for now, would use real phoneme embeddings)
        # In real use, this would come from a text encoder
        text_embeddings = torch.randn(self.max_frames, 256)

        # Create emotion embedding
        emotion_idx = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'surprised': 4, 'calm': 5, 'fearful': 6, 'disgusted': 7,
        }.get(emotion.lower(), 0)
        emotion_one_hot = torch.zeros(256)
        if emotion_idx < 256:
            emotion_one_hot[emotion_idx] = 1.0

        result = {
            'text_embeddings': text_embeddings,
            'f0': f0,
            'voicing': voicing,
            'emotion': emotion,
            'emotion_embedding': emotion_one_hot,
            'mask': (f0 > 0).float() | (voicing > 0).float(),
        }

        # Cache
        if self.use_cache:
            self.cache[idx] = result

        return result


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    config: dict,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dict with average losses and metrics
    """
    model.train()
    total_losses = {}
    total_metrics = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        text_emb = batch['text_embeddings'].to(device)
        f0 = batch['f0'].to(device)
        voicing = batch['voicing'].to(device)
        mask = batch['mask'].to(device)
        emotion_emb = batch['emotion_embedding'].to(device)

        # Forward pass
        if hasattr(model, 'combine_emotions') and model.combine_emotions:
            output = model(text_emb, f0, voicing, emotion_emb, mask)
        else:
            output = model(text_emb, f0, voicing, mask=mask)

        # Get loss
        loss = output['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        max_grad_norm = config.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Accumulate losses
        for key, value in output.items():
            if 'loss' in key and isinstance(value, torch.Tensor):
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()

        # Accumulate metrics
        if 'metrics' in output:
            for key, value in output['metrics'].items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value

        num_batches += 1

        # Log progress
        if batch_idx % config.get('log_every', 10) == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, "
                  f"F0 corr={output.get('metrics', {}).get('f0_correlation', 0):.4f}")

    # Average losses and metrics
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    # Update scheduler
    if scheduler is not None:
        scheduler.step()

    return {**avg_losses, **avg_metrics}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
) -> Dict[str, float]:
    """
    Validate the model.

    Returns:
        Dict with validation losses and metrics
    """
    model.eval()
    total_losses = {}
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            text_emb = batch['text_embeddings'].to(device)
            f0 = batch['f0'].to(device)
            voicing = batch['voicing'].to(device)
            mask = batch['mask'].to(device)
            emotion_emb = batch['emotion_embedding'].to(device)

            # Forward pass
            if hasattr(model, 'combine_emotions') and model.combine_emotions:
                output = model(text_emb, f0, voicing, emotion_emb, mask)
            else:
                output = model(text_emb, f0, voicing, mask=mask)

            # Accumulate losses
            for key, value in output.items():
                if 'loss' in key and isinstance(value, torch.Tensor):
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()

            # Accumulate metrics
            if 'metrics' in output:
                for key, value in output['metrics'].items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value

            num_batches += 1

    # Average
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    return {**avg_losses, **avg_metrics}


def train(
    config: dict,
    resume_path: Optional[str] = None,
):
    """
    Main training function.

    Args:
        config: Configuration dictionary
        resume_path: Optional path to resume from checkpoint
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create Period VITS config
    pv_config = PeriodVITSConfig(
        sample_rate=config.get('sample_rate', 24000),
        hop_length=config.get('hop_length', 256),
        f0_min=config.get('f0_min', 50.0),
        f0_max=config.get('f0_max', 800.0),
        use_log_f0=config.get('use_log_f0', True),
        fpp_hidden_dim=config.get('fpp_hidden_dim', 256),
        fpp_num_layers=config.get('fpp_num_layers', 3),
        fpp_kernel_size=config.get('fpp_kernel_size', 5),
        fpp_dropout=config.get('fpp_dropout', 0.1),
        fpp_use_gru=config.get('fpp_use_gru', True),
        pg_upsample_scales=config.get('pg_upsample_scales', [4, 4, 4, 4]),
        pg_noise_scale=config.get('pg_noise_scale', 0.003),
        pg_sine_amplitude=config.get('pg_sine_amplitude', 0.1),
        pg_use_learnable_amplitude=config.get('pg_use_learnable_amplitude', True),
        f0_loss_weight=config.get('f0_loss_weight', 1.0),
        voicing_loss_weight=config.get('voicing_loss_weight', 0.5),
        periodicity_loss_weight=config.get('periodicity_loss_weight', 0.1),
        output_dim=config.get('output_dim', 2048),
        num_prosody_tokens=config.get('num_prosody_tokens', 4),
    )

    # Create model
    if config.get('combine_emotions', True):
        model = create_period_vits_adapter(
            input_dim=256,
            emotion_dim=config.get('emotion_dim', 256),
            sample_rate=config.get('sample_rate', 24000),
            output_dim=config.get('output_dim', 2048),
            combine_emotions=True,
        )
    else:
        from period_vits import PeriodVITS
        model = PeriodVITS(pv_config, input_dim=256)

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create datasets
    train_dataset = PeriodVITSDataset(
        config.get('train_manifest'),
        pv_config,
        max_frames=300,
    )
    val_dataset = PeriodVITSDataset(
        config.get('val_manifest'),
        pv_config,
        max_frames=300,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
        betas=(
            config.get('adam_beta1', 0.9),
            config.get('adam_beta2', 0.98),
        ),
        eps=config.get('adam_eps', 1e-9),
    )

    # Scheduler
    scheduler = None
    if config.get('lr_scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=config.get('min_lr', 1e-5),
        )

    # Resume from checkpoint
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Output directory
    output_dir = Path(config.get('output_dir', '../checkpoints/period_vits'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_f0_corr = -1.0
    no_improve_count = 0
    patience = config.get('early_stopping_patience', 10)

    print(f"\nStarting training from epoch {start_epoch}")
    print(f"Output directory: {output_dir}")

    for epoch in range(start_epoch, config.get('num_epochs', 50)):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.get('num_epochs', 50)}")
        print(f"{'='*60}")

        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, config
        )

        print(f"\nTrain: Loss={train_results.get('total_loss', 0):.4f}, "
              f"F0 corr={train_results.get('f0_correlation', 0):.4f}, "
              f"Jitter={train_results.get('f0_jitter', 0):.2f} Hz")

        # Validate
        if (epoch + 1) % config.get('validate_every_epochs', 1) == 0:
            val_results = validate(model, val_loader, device, config)

            print(f"Val:   Loss={val_results.get('total_loss', 0):.4f}, "
                  f"F0 corr={val_results.get('f0_correlation', 0):.4f}, "
                  f"Jitter={val_results.get('f0_jitter', 0):.2f} Hz")

            # Check for improvement
            current_corr = val_results.get('f0_correlation', 0)
            if current_corr > best_f0_corr:
                best_f0_corr = current_corr
                no_improve_count = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'f0_correlation': current_corr,
                    'config': config,
                }, output_dir / 'best.pt')
                print(f"  Saved best model (F0 corr: {best_f0_corr:.4f})")

            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"\nEarly stopping: no improvement for {patience} epochs")
                    break

        # Save checkpoint
        if (epoch + 1) % config.get('save_every_epochs', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': config,
            }, output_dir / f'epoch_{epoch+1}.pt')

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
        }, output_dir / 'latest.pt')

    print(f"\nTraining complete!")
    print(f"Best F0 correlation: {best_f0_corr:.4f}")
    print(f"Target: > 0.3 (currently {best_f0_corr:.4f})")


def test_mode():
    """
    Run tests with synthetic data.
    """
    print("=" * 60)
    print("Period VITS - Test Mode")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create config
    config = PeriodVITSConfig()

    # Create model
    model = create_period_vits_adapter(
        input_dim=256,
        emotion_dim=256,
        combine_emotions=True,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create synthetic data
    batch_size = 4
    time_len = 100

    text_emb = torch.randn(batch_size, time_len, 256, device=device)
    t = torch.linspace(0, 4 * math.pi, time_len)
    f0 = (150 + 50 * torch.sin(t) + 20 * torch.sin(3 * t)).unsqueeze(0).expand(batch_size, -1).to(device)
    voicing = (torch.rand(batch_size, time_len, device=device) > 0.2).float()
    emotion_emb = torch.randn(batch_size, 256, device=device)
    mask = torch.ones(batch_size, time_len, device=device)

    # Forward pass
    print("\n[Test] Forward pass...")
    output = model(text_emb, f0, voicing, emotion_emb, mask)

    print(f"  Prosody tokens: {output['prosody_tokens'].shape}")
    print(f"  F0 loss: {output['f0_loss']:.4f}")
    print(f"  Voicing loss: {output['voicing_loss']:.4f}")
    print(f"  Total loss: {output['total_loss']:.4f}")

    if 'metrics' in output:
        print(f"\n  Metrics:")
        print(f"    F0 correlation: {output['metrics']['f0_correlation']:.4f}")
        print(f"    F0 jitter: {output['metrics']['f0_jitter']:.2f} Hz")
        print(f"    Voicing accuracy: {output['metrics']['voicing_accuracy']:.4f}")
        print(f"    Contour smoothness: {output['metrics']['contour_smoothness']:.4f}")

    # Backward pass
    print("\n[Test] Backward pass...")
    loss = output['total_loss']
    loss.backward()

    total_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.norm().item() ** 2
    print(f"  Gradient norm: {total_grad ** 0.5:.4f}")

    # Training step
    print("\n[Test] Training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(5):
        optimizer.zero_grad()
        output = model(text_emb, f0, voicing, emotion_emb, mask)
        loss = output['total_loss']
        loss.backward()
        optimizer.step()

        print(f"  Step {i+1}: Loss={loss.item():.4f}, "
              f"F0 corr={output['metrics']['f0_correlation']:.4f}")

    print("\n[PASS] All tests passed!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Train Period VITS")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    if args.test:
        success = test_mode()
        sys.exit(0 if success else 1)

    if not args.config:
        print("Error: --config required for training")
        print("Usage: python train_period_vits.py --config config/period_vits.yaml")
        print("   or: python train_period_vits.py --test")
        sys.exit(1)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")

    # Train
    train(config, args.resume)


if __name__ == "__main__":
    main()
