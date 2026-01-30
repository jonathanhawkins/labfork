#!/usr/bin/env python3
"""
Training script for PE-wav2vec (Prosody-Enhanced wav2vec 2.0).

This script trains the PE-wav2vec model with LPC residual supervision
on initial transformer blocks to learn prosodic features.

Training Stages:
1. Pretrain: LPC residual supervision on wav2vec layers 1-4
2. Fine-tune: Prosody prediction for TTS integration

Usage:
    # Train PE-wav2vec with LPC supervision
    python train_pe_wav2vec.py --config config/pe_wav2vec.yaml

    # Resume from checkpoint
    python train_pe_wav2vec.py --config config/pe_wav2vec.yaml --resume ../checkpoints/pe_wav2vec/best.pt

    # Test mode (no GPU, synthetic data)
    python train_pe_wav2vec.py --test

Reference:
    PE-Wav2vec: A Prosody-Enhanced Speech Model for Self-Supervised Prosody Learning in TTS
    IEEE/ACM TASLP 2024 - https://ieeexplore.ieee.org/document/10645206/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pe_wav2vec import (
    PEWav2VecConfig,
    LPCResidualExtractor,
    PEWav2VecEncoder,
    PEWav2VecAdapter,
    PEWav2VecLoss,
    S4LPRProsodyPredictor,
)


# =============================================================================
# DATASET
# =============================================================================

class AudioProsodyDataset(Dataset):
    """
    Dataset for PE-wav2vec training.

    Loads audio files and extracts LPC residual targets on-the-fly.
    """

    def __init__(
        self,
        manifest_path: str,
        config: PEWav2VecConfig,
        max_audio_len: int = 16000 * 10,  # 10 seconds max
        min_audio_len: int = 16000 * 1,   # 1 second min
    ):
        self.config = config
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio_path = sample['audio_path']

        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            # Truncate or pad
            if waveform.shape[0] > self.max_audio_len:
                start = torch.randint(0, waveform.shape[0] - self.max_audio_len, (1,)).item()
                waveform = waveform[start:start + self.max_audio_len]
            elif waveform.shape[0] < self.min_audio_len:
                # Pad with zeros
                pad_len = self.min_audio_len - waveform.shape[0]
                waveform = F.pad(waveform, (0, pad_len))

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return random audio as fallback
            waveform = torch.randn(self.max_audio_len)

        return {
            'audio': waveform,
            'path': audio_path,
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int = 100, audio_len: int = 32000):
        self.num_samples = num_samples
        self.audio_len = audio_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic audio with prosodic variation
        t = torch.linspace(0, 2 * math.pi * 10, self.audio_len)

        # Base frequency with variation
        freq_mod = 1 + 0.5 * torch.sin(t / 1000)
        audio = 0.5 * torch.sin(t * freq_mod * 440 / self.audio_len * 16000)

        # Add harmonics
        audio += 0.3 * torch.sin(t * freq_mod * 880 / self.audio_len * 16000)
        audio += 0.1 * torch.sin(t * freq_mod * 1320 / self.audio_len * 16000)

        # Add amplitude envelope
        envelope = 0.5 + 0.5 * torch.sin(t / 5000)
        audio = audio * envelope

        # Add noise
        audio = audio + 0.01 * torch.randn_like(audio)

        return {
            'audio': audio,
            'path': f'synthetic_{idx}',
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    max_len = max(item['audio'].shape[0] for item in batch)

    audios = []
    masks = []

    for item in batch:
        audio = item['audio']
        pad_len = max_len - audio.shape[0]

        if pad_len > 0:
            audio = F.pad(audio, (0, pad_len))
            mask = torch.ones(max_len)
            mask[-pad_len:] = 0
        else:
            mask = torch.ones(max_len)

        audios.append(audio)
        masks.append(mask)

    return {
        'audio': torch.stack(audios),
        'attention_mask': torch.stack(masks).bool(),
    }


# =============================================================================
# TRAINER
# =============================================================================

class PEWav2VecTrainer:
    """Trainer for PE-wav2vec model."""

    def __init__(
        self,
        config: PEWav2VecConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        log_every: int = 100,
        eval_every: int = 1000,
        save_every: int = 5000,
        checkpoint_dir: str = "../checkpoints/pe_wav2vec",
        device: str = "cuda",
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.encoder = PEWav2VecEncoder(config).to(device)
        self.loss_fn = PEWav2VecLoss(config)

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # Scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                # Cosine decay after warmup
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.encoder.train()

        audio = batch['audio'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass with LPC supervision
        output = self.encoder(
            audio,
            attention_mask=attention_mask,
            compute_lpc_loss=True,
        )

        # Compute loss
        losses = self.loss_fn(output)
        total_loss = losses['total']

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}

        self.encoder.eval()
        total_losses = {}
        num_batches = 0

        for batch in self.val_dataloader:
            audio = batch['audio'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            output = self.encoder(
                audio,
                attention_mask=attention_mask,
                compute_lpc_loss=True,
            )

            losses = self.loss_fn(output)

            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item() if torch.is_tensor(v) else v

            num_batches += 1

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Resumed from step {self.global_step}")

    def train(self):
        """Main training loop."""
        print(f"Starting training from step {self.global_step}")
        print(f"Total steps: {self.max_steps}")

        data_iter = iter(self.train_dataloader)

        pbar = tqdm(total=self.max_steps, initial=self.global_step, desc="Training")

        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Train step
            losses = self.train_step(batch)
            self.train_losses.append(losses['total'])
            self.global_step += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'lpc': f"{losses.get('lpc_loss', 0):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Log
            if self.global_step % self.log_every == 0:
                avg_loss = sum(self.train_losses[-self.log_every:]) / min(len(self.train_losses), self.log_every)
                print(f"\nStep {self.global_step}: avg_loss={avg_loss:.4f}, lr={self.scheduler.get_last_lr()[0]:.2e}")

            # Validate
            if self.global_step % self.eval_every == 0:
                val_losses = self.validate()
                if val_losses:
                    print(f"Validation: {val_losses}")

                    # Save best model
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint("best")

            # Save checkpoint
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f"step_{self.global_step}")
                self.save_checkpoint("latest")

        pbar.close()
        self.save_checkpoint("final")
        print("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML not installed, using default config")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Train PE-wav2vec model")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--manifest', type=str, default=None, help='Path to training manifest')
    parser.add_argument('--test', action='store_true', help='Run in test mode with synthetic data')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Load config
    if args.config and os.path.exists(args.config):
        config_dict = load_config(args.config)
    else:
        config_dict = {}

    # Create PE-wav2vec config
    config = PEWav2VecConfig(**{
        k: v for k, v in config_dict.get('model', {}).items()
        if hasattr(PEWav2VecConfig, k)
    })

    print("=" * 60)
    print("PE-wav2vec Training")
    print("=" * 60)
    print(f"Config: {config}")

    # Determine device
    if args.test:
        device = 'cpu'
        print("\n[TEST MODE] Using CPU and synthetic data")
    else:
        device = args.device if torch.cuda.is_available() else 'cpu'
        print(f"\nDevice: {device}")

    # Create datasets
    if args.test:
        train_dataset = SyntheticDataset(num_samples=50, audio_len=32000)
        val_dataset = SyntheticDataset(num_samples=10, audio_len=32000)
        batch_size = 2
        max_steps = 100
    else:
        manifest_path = args.manifest or config_dict.get('data', {}).get('manifest')
        if manifest_path and os.path.exists(manifest_path):
            train_dataset = AudioProsodyDataset(manifest_path, config)
            val_dataset = None  # Can add validation manifest
        else:
            print("No manifest found, using synthetic data")
            train_dataset = SyntheticDataset(num_samples=1000, audio_len=32000)
            val_dataset = SyntheticDataset(num_samples=100, audio_len=32000)

        batch_size = config_dict.get('training', {}).get('batch_size', 8)
        max_steps = config_dict.get('training', {}).get('max_steps', 100000)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 if args.test else 4,
        pin_memory=device == 'cuda',
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0 if args.test else 2,
        )

    # Create trainer
    trainer = PEWav2VecTrainer(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=config_dict.get('training', {}).get('learning_rate', 1e-4),
        weight_decay=config_dict.get('training', {}).get('weight_decay', 0.01),
        warmup_steps=config_dict.get('training', {}).get('warmup_steps', 1000),
        max_steps=max_steps,
        log_every=config_dict.get('training', {}).get('log_every', 100),
        eval_every=config_dict.get('training', {}).get('eval_every', 1000),
        save_every=config_dict.get('training', {}).get('save_every', 5000),
        checkpoint_dir=config_dict.get('training', {}).get('checkpoint_dir', '../checkpoints/pe_wav2vec'),
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("-" * 40)
    print("""
1. Use trained encoder for prosody extraction:

   from pe_wav2vec import PEWav2VecEncoder, PEWav2VecConfig

   config = PEWav2VecConfig()
   encoder = PEWav2VecEncoder(config)
   encoder.load_state_dict(torch.load('checkpoints/pe_wav2vec/best.pt')['encoder_state_dict'])

   # Extract prosody
   prosody_emb = encoder.get_prosody_embedding(audio)

2. Integrate with CSM:

   from pe_wav2vec import PEWav2VecAdapter

   adapter = PEWav2VecAdapter(config, encoder)
   prefix_tokens = adapter(audio)  # [batch, 4, 2048]

3. Compare with vanilla wav2vec:

   from pe_wav2vec import ProsodyComparison

   comparison = ProsodyComparison(config)
   f0_corr = comparison.compute_f0_correlation(prosody_emb, f0_values)
   print(f"F0 correlation: {f0_corr}")
""")


if __name__ == "__main__":
    main()
