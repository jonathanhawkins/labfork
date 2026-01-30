"""
Training Script for DDGAN-Accelerated Prosody Diffusion

Trains the DiffProsody-style model for fast prosody generation (16x speedup).

Two-stage training process:
1. Stage 1: Pre-train Prosody VQ-VAE (reconstruction)
2. Stage 2: Train DDGAN Generator + Discriminator (adversarial)

Usage:
    # Full training (both stages)
    python train_ddgan_prosody.py --config config/ddgan_prosody.yaml

    # Stage 1 only (VQ-VAE)
    python train_ddgan_prosody.py --config config/ddgan_prosody.yaml --stage 1

    # Stage 2 only (DDGAN, requires pretrained VQ-VAE)
    python train_ddgan_prosody.py --config config/ddgan_prosody.yaml --stage 2 \\
        --vqvae-checkpoint ../checkpoints/ddgan_prosody/vqvae_best.pt

    # Test mode
    python train_ddgan_prosody.py --test

References:
    - DiffProsody: https://arxiv.org/abs/2307.16549
    - DiffGAN-TTS: https://arxiv.org/abs/2201.11972
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddgan_prosody import (
    DDGANProsodyConfig,
    DDGANProsody,
    DDGANProsodyAdapter,
    DDGANProsodyLoss,
    ProsodyVQVAE,
)


# =============================================================================
# DATASET
# =============================================================================

class ProsodyDataset(Dataset):
    """
    Dataset for DDGAN prosody training.

    Expected manifest format:
    [
        {
            "audio_path": "path/to/audio.wav",
            "mel_path": "path/to/mel.pt",
            "text_embedding_path": "path/to/text_emb.pt",
            "speaker_id": 0,
            "duration": 3.5
        },
        ...
    ]
    """

    def __init__(
        self,
        manifest_path: str,
        max_mel_len: int = 500,
        max_text_len: int = 100,
        speaker_embedding_path: Optional[str] = None,
    ):
        self.manifest_path = manifest_path
        self.max_mel_len = max_mel_len
        self.max_text_len = max_text_len

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        # Load speaker embeddings if provided
        self.speaker_embeddings = None
        if speaker_embedding_path and os.path.exists(speaker_embedding_path):
            self.speaker_embeddings = torch.load(speaker_embedding_path)

        print(f"Loaded {len(self.manifest)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]

        # Load mel spectrogram
        mel = torch.load(item['mel_path'])
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # Add batch dim if needed

        # Ensure [mel_dim, time]
        if mel.shape[0] > mel.shape[-1]:
            mel = mel.squeeze(0)
        if mel.dim() == 3:
            mel = mel.squeeze(0)

        # Truncate/pad mel
        mel_len = mel.shape[-1]
        if mel_len > self.max_mel_len:
            mel = mel[:, :self.max_mel_len]
            mel_len = self.max_mel_len

        # Load text embeddings
        text_emb = torch.load(item['text_embedding_path'])
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(0)

        # Truncate/pad text
        text_len = text_emb.shape[0]
        if text_len > self.max_text_len:
            text_emb = text_emb[:self.max_text_len]
            text_len = self.max_text_len

        # Get speaker embedding
        speaker_id = item.get('speaker_id', 0)
        if self.speaker_embeddings is not None:
            speaker_emb = self.speaker_embeddings[speaker_id]
        else:
            # Random speaker embedding (will be learned)
            speaker_emb = torch.randn(256)

        return {
            'mel': mel,
            'text_emb': text_emb,
            'speaker_emb': speaker_emb,
            'mel_len': torch.tensor(mel_len),
            'text_len': torch.tensor(text_len),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    # Find max lengths
    max_mel_len = max(item['mel'].shape[-1] for item in batch)
    max_text_len = max(item['text_emb'].shape[0] for item in batch)

    # Get dimensions
    mel_dim = batch[0]['mel'].shape[0]
    text_dim = batch[0]['text_emb'].shape[-1]
    speaker_dim = batch[0]['speaker_emb'].shape[-1]

    # Initialize padded tensors
    batch_size = len(batch)
    mel_batch = torch.zeros(batch_size, mel_dim, max_mel_len)
    text_batch = torch.zeros(batch_size, max_text_len, text_dim)
    speaker_batch = torch.zeros(batch_size, speaker_dim)
    mel_mask = torch.zeros(batch_size, max_mel_len, dtype=torch.bool)
    text_mask = torch.zeros(batch_size, max_text_len, dtype=torch.bool)

    # Fill tensors
    for i, item in enumerate(batch):
        mel_len = item['mel'].shape[-1]
        text_len = item['text_emb'].shape[0]

        mel_batch[i, :, :mel_len] = item['mel']
        text_batch[i, :text_len] = item['text_emb']
        speaker_batch[i] = item['speaker_emb']
        mel_mask[i, :mel_len] = True
        text_mask[i, :text_len] = True

    return {
        'mel': mel_batch,
        'text_emb': text_batch,
        'speaker_emb': speaker_batch,
        'mel_mask': mel_mask,
        'text_mask': text_mask,
    }


class SyntheticProsodyDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 100,
        mel_dim: int = 80,
        text_dim: int = 256,
        speaker_dim: int = 256,
        mel_len_range: Tuple[int, int] = (50, 200),
        text_len_range: Tuple[int, int] = (10, 50),
    ):
        self.num_samples = num_samples
        self.mel_dim = mel_dim
        self.text_dim = text_dim
        self.speaker_dim = speaker_dim
        self.mel_len_range = mel_len_range
        self.text_len_range = text_len_range

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random lengths
        mel_len = torch.randint(*self.mel_len_range, (1,)).item()
        text_len = torch.randint(*self.text_len_range, (1,)).item()

        return {
            'mel': torch.randn(self.mel_dim, mel_len),
            'text_emb': torch.randn(text_len, self.text_dim),
            'speaker_emb': torch.randn(self.speaker_dim),
            'mel_len': torch.tensor(mel_len),
            'text_len': torch.tensor(text_len),
        }


# =============================================================================
# TRAINER
# =============================================================================

class DDGANProsodyTrainer:
    """
    Trainer for DDGAN Prosody model.

    Two-stage training:
    1. Pre-train VQ-VAE for prosody encoding
    2. Train DDGAN with adversarial loss
    """

    def __init__(
        self,
        config: DDGANProsodyConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "../checkpoints/ddgan_prosody",
        device: str = "cuda",
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Initialize model
        self.model = DDGANProsody(config).to(device)

        # Loss function
        self.loss_fn = DDGANProsodyLoss(config)

        # Optimizers
        self.optimizer_vqvae = torch.optim.AdamW(
            self.model.vqvae.parameters(),
            lr=config.generator_lr,
            betas=(0.9, 0.999),
        )

        self.optimizer_g = torch.optim.AdamW(
            list(self.model.vqvae.parameters()) +
            list(self.model.generator.parameters()) +
            list(self.model.token_projection.parameters()),
            lr=config.generator_lr,
            betas=(0.9, 0.999),
        )

        self.optimizer_d = torch.optim.AdamW(
            self.model.discriminator.parameters(),
            lr=config.discriminator_lr,
            betas=(0.9, 0.999),
        )

        # Learning rate schedulers
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=100, eta_min=1e-6
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=100, eta_min=1e-6
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_vqvae_epoch(self, epoch: int) -> Dict[str, float]:
        """Train VQ-VAE for one epoch (Stage 1)."""
        self.model.vqvae.train()

        epoch_losses = {
            'reconstruction': 0.0,
            'commitment': 0.0,
            'perplexity': 0.0,
            'total': 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            mel = batch['mel'].to(self.device)
            mel_mask = batch['mel_mask'].to(self.device)

            # Forward through VQ-VAE
            output = self.model.vqvae(mel, mel_mask)

            # Compute loss
            loss = output['reconstruction_loss'] + self.config.vq_weight * output['commitment_loss']

            # Backward
            self.optimizer_vqvae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.vqvae.parameters(), 1.0)
            self.optimizer_vqvae.step()

            # Track losses
            epoch_losses['reconstruction'] += output['reconstruction_loss'].item()
            epoch_losses['commitment'] += output['commitment_loss'].item()
            epoch_losses['perplexity'] += output['perplexity'].item()
            epoch_losses['total'] += loss.item()
            num_batches += 1

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    def train_ddgan_epoch(self, epoch: int) -> Dict[str, float]:
        """Train DDGAN for one epoch (Stage 2)."""
        self.model.train()

        epoch_losses = {
            'g_total': 0.0,
            'd_total': 0.0,
            'reconstruction': 0.0,
            'adversarial': 0.0,
            'feature_matching': 0.0,
            'd_real': 0.0,
            'd_fake': 0.0,
            'r1_penalty': 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            mel = batch['mel'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            speaker_emb = batch['speaker_emb'].to(self.device)
            mel_mask = batch['mel_mask'].to(self.device)
            text_mask = batch['text_mask'].to(self.device)

            # =========== Train Discriminator ===========
            self.optimizer_d.zero_grad()

            # Forward
            with torch.no_grad():
                model_output = self.model(
                    mel, text_emb, speaker_emb, mel_mask, text_mask
                )

            # Get real latent with gradient
            real_latent = model_output['real_latent'].detach().clone()
            real_latent.requires_grad_(True)

            # Discriminator on real
            real_score = self.model.discriminator(
                real_latent, text_emb, speaker_emb, text_mask
            )

            # Discriminator on fake
            fake_score = self.model.discriminator(
                model_output['pred_latent'].detach(),
                text_emb, speaker_emb, text_mask
            )

            # Hinge loss
            d_loss_real = F.relu(1.0 - real_score['score']).mean()
            d_loss_fake = F.relu(1.0 + fake_score['score']).mean()

            # R1 penalty
            r1_grads = torch.autograd.grad(
                outputs=real_score['score'].sum(),
                inputs=real_latent,
                create_graph=True,
            )[0]
            r1_penalty = r1_grads.pow(2).sum(dim=-1).mean()

            d_loss = d_loss_real + d_loss_fake + self.config.gradient_penalty_weight * r1_penalty

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 1.0)
            self.optimizer_d.step()

            # =========== Train Generator ===========
            self.optimizer_g.zero_grad()

            # Forward
            model_output = self.model(
                mel, text_emb, speaker_emb, mel_mask, text_mask
            )

            # Add real score for feature matching
            with torch.no_grad():
                model_output['real_score'] = self.model.discriminator(
                    model_output['real_latent'].detach(),
                    text_emb, speaker_emb, text_mask
                )

            # Generator losses
            g_losses = self.loss_fn.generator_loss(model_output)

            g_losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.vqvae.parameters()) +
                list(self.model.generator.parameters()),
                1.0
            )
            self.optimizer_g.step()

            # Track losses
            epoch_losses['g_total'] += g_losses['total'].item()
            epoch_losses['d_total'] += d_loss.item()
            epoch_losses['reconstruction'] += g_losses['reconstruction'].item()
            epoch_losses['adversarial'] += g_losses['adversarial'].item()
            epoch_losses['feature_matching'] += g_losses['feature_matching'].item()
            epoch_losses['d_real'] += d_loss_real.item()
            epoch_losses['d_fake'] += d_loss_fake.item()
            epoch_losses['r1_penalty'] += r1_penalty.item()
            num_batches += 1

            self.global_step += 1

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # Step schedulers
        self.scheduler_g.step()
        self.scheduler_d.step()

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = {
            'reconstruction': 0.0,
            'perplexity': 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                mel = batch['mel'].to(self.device)
                mel_mask = batch['mel_mask'].to(self.device)

                output = self.model.vqvae(mel, mel_mask)

                val_losses['reconstruction'] += output['reconstruction_loss'].item()
                val_losses['perplexity'] += output['perplexity'].item()
                num_batches += 1

        for k in val_losses:
            val_losses[k] /= num_batches

        return val_losses

    def save_checkpoint(self, path: str, epoch: int, stage: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_vqvae_state_dict': self.optimizer_vqvae.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_vqvae.load_state_dict(checkpoint['optimizer_vqvae_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0), checkpoint.get('stage', 1)

    def load_vqvae_checkpoint(self, path: str):
        """Load VQ-VAE checkpoint only."""
        checkpoint = torch.load(path, map_location=self.device)
        vqvae_state = {
            k.replace('vqvae.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('vqvae.')
        }
        self.model.vqvae.load_state_dict(vqvae_state)
        print(f"Loaded VQ-VAE from {path}")

    def train_stage1(self, num_epochs: int) -> Dict[str, List[float]]:
        """Stage 1: Train VQ-VAE."""
        print("\n" + "=" * 60)
        print("STAGE 1: Training Prosody VQ-VAE")
        print("=" * 60)

        history = {k: [] for k in ['reconstruction', 'commitment', 'perplexity', 'total']}

        for epoch in range(num_epochs):
            losses = self.train_vqvae_epoch(epoch)

            for k in history:
                history[k].append(losses[k])

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Recon: {losses['reconstruction']:.4f} | "
                  f"Commit: {losses['commitment']:.4f} | "
                  f"Perplexity: {losses['perplexity']:.1f}")

            # Validate
            if self.val_loader is not None and (epoch + 1) % 5 == 0:
                val_losses = self.validate()
                print(f"  Val Recon: {val_losses['reconstruction']:.4f}")

                if val_losses['reconstruction'] < self.best_val_loss:
                    self.best_val_loss = val_losses['reconstruction']
                    self.save_checkpoint(
                        str(self.checkpoint_dir / "vqvae_best.pt"), epoch, 1
                    )

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    str(self.checkpoint_dir / f"vqvae_epoch_{epoch+1}.pt"), epoch, 1
                )

        # Final checkpoint
        self.save_checkpoint(str(self.checkpoint_dir / "vqvae_final.pt"), num_epochs, 1)

        return history

    def train_stage2(self, num_epochs: int) -> Dict[str, List[float]]:
        """Stage 2: Train DDGAN."""
        print("\n" + "=" * 60)
        print("STAGE 2: Training DDGAN Generator + Discriminator")
        print("=" * 60)

        history = {k: [] for k in [
            'g_total', 'd_total', 'reconstruction', 'adversarial',
            'feature_matching', 'd_real', 'd_fake', 'r1_penalty'
        ]}

        for epoch in range(num_epochs):
            losses = self.train_ddgan_epoch(epoch)

            for k in history:
                history[k].append(losses[k])

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"G: {losses['g_total']:.4f} | "
                  f"D: {losses['d_total']:.4f} | "
                  f"Adv: {losses['adversarial']:.4f} | "
                  f"FM: {losses['feature_matching']:.4f}")

            # Validate
            if self.val_loader is not None and (epoch + 1) % 5 == 0:
                val_losses = self.validate()
                print(f"  Val Recon: {val_losses['reconstruction']:.4f}")

                if val_losses['reconstruction'] < self.best_val_loss:
                    self.best_val_loss = val_losses['reconstruction']
                    self.save_checkpoint(
                        str(self.checkpoint_dir / "best.pt"), epoch, 2
                    )

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    str(self.checkpoint_dir / f"ddgan_epoch_{epoch+1}.pt"), epoch, 2
                )

        # Final checkpoint
        self.save_checkpoint(str(self.checkpoint_dir / "final.pt"), num_epochs, 2)

        return history


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DDGAN Prosody Model")
    parser.add_argument("--config", type=str, default="config/ddgan_prosody.yaml",
                       help="Path to config file")
    parser.add_argument("--manifest", type=str, default=None,
                       help="Path to training manifest")
    parser.add_argument("--val-manifest", type=str, default=None,
                       help="Path to validation manifest")
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints/ddgan_prosody",
                       help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--vqvae-checkpoint", type=str, default=None,
                       help="Pre-trained VQ-VAE checkpoint for stage 2")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                       help="Train specific stage only (1=VQ-VAE, 2=DDGAN)")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with synthetic data")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config_dict = load_config(args.config)
    else:
        config_dict = {}

    # Create config
    config = DDGANProsodyConfig(**{
        k: v for k, v in config_dict.items()
        if k in DDGANProsodyConfig.__dataclass_fields__
    })

    # Get training params
    batch_size = config_dict.get('batch_size', 8)
    num_workers = config_dict.get('num_workers', 4)
    stage1_epochs = config_dict.get('stage1_epochs', 50)
    stage2_epochs = config_dict.get('stage2_epochs', 100)

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print(f"\nUsing device: {device}")
    print(f"Config: {config}")

    # Create datasets
    if args.test:
        print("\n[TEST MODE] Using synthetic data")
        train_dataset = SyntheticProsodyDataset(
            num_samples=100,
            mel_dim=config.mel_dim,
            text_dim=config.text_dim,
            speaker_dim=config.speaker_dim,
        )
        val_dataset = SyntheticProsodyDataset(
            num_samples=20,
            mel_dim=config.mel_dim,
            text_dim=config.text_dim,
            speaker_dim=config.speaker_dim,
        )
        stage1_epochs = 3
        stage2_epochs = 3
    else:
        if args.manifest is None:
            print("Error: --manifest required for training")
            return

        train_dataset = ProsodyDataset(args.manifest)
        val_dataset = None
        if args.val_manifest:
            val_dataset = ProsodyDataset(args.val_manifest)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if not args.test else 0,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if not args.test else 0,
            collate_fn=collate_fn,
        )

    # Create trainer
    trainer = DDGANProsodyTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    # Resume from checkpoint
    start_epoch = 0
    start_stage = 1
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, start_stage = trainer.load_checkpoint(args.resume)

    # Load VQ-VAE for stage 2
    if args.vqvae_checkpoint:
        trainer.load_vqvae_checkpoint(args.vqvae_checkpoint)

    # Determine which stages to train
    if args.stage is not None:
        stages_to_train = [args.stage]
    else:
        stages_to_train = [1, 2]

    # Train
    print("\n" + "=" * 60)
    print("DDGAN Prosody Training")
    print("=" * 60)
    print(f"Stages to train: {stages_to_train}")
    print(f"Stage 1 epochs: {stage1_epochs}")
    print(f"Stage 2 epochs: {stage2_epochs}")

    history = {}

    if 1 in stages_to_train:
        history['stage1'] = trainer.train_stage1(stage1_epochs)

    if 2 in stages_to_train:
        history['stage2'] = trainer.train_stage2(stage2_epochs)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # Test generation
    print("\nTesting few-step generation...")
    trainer.model.eval()
    with torch.no_grad():
        # Get a batch
        batch = next(iter(train_loader))
        text_emb = batch['text_emb'][:1].to(device)
        speaker_emb = batch['speaker_emb'][:1].to(device)

        for n_steps in [1, 2, 4]:
            latent = trainer.model.generate_prosody(
                text_emb, speaker_emb, num_steps=n_steps
            )
            tokens = trainer.model.to_tokens(latent)
            print(f"  {n_steps} step(s): tokens shape={tokens.shape}")

    print("\nCheckpoints saved to:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
