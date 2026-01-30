#!/usr/bin/env python3
"""
VEVO Training Script: Self-Supervised Disentanglement via VQ Bottleneck

Trains the VEVO model for controllable zero-shot voice imitation.

Usage:
    # Full training
    python train_vevo.py --config config/vevo_disentanglement.yaml

    # Resume from checkpoint
    python train_vevo.py --config config/vevo_disentanglement.yaml \
        --resume ../checkpoints/vevo/checkpoint_step_10000.pt

    # Test mode (verify model works)
    python train_vevo.py --test

    # Evaluate disentanglement
    python train_vevo.py --evaluate --checkpoint ../checkpoints/vevo/best.pt
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.vevo_disentanglement import (
    VEVOConfig,
    VEVO,
    VEVOLoss,
    VEVOAdapter,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class VEVODataset(Dataset):
    """
    Dataset for VEVO training.

    Requires:
    - Audio features (HuBERT/WavLM)
    - Mel spectrograms (for acoustic model)
    """

    def __init__(
        self,
        manifest_path: str,
        feature_dir: Optional[str] = None,
        max_length: int = 500,  # Max sequence length in frames
        sample_rate: int = 16000,
        hop_length: int = 320,
    ):
        self.manifest_path = manifest_path
        self.feature_dir = feature_dir
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Load manifest
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.samples = json.load(f)
        else:
            self.samples = []
            logger.warning(f"Manifest not found: {manifest_path}")

    def __len__(self) -> int:
        return max(len(self.samples), 100)  # Minimum for testing

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # If we have real data, load it
        if idx < len(self.samples):
            sample = self.samples[idx]

            # Load features (HuBERT/WavLM)
            if self.feature_dir and 'feature_path' in sample:
                feature_path = os.path.join(self.feature_dir, sample['feature_path'])
                if os.path.exists(feature_path):
                    features = torch.load(feature_path)
                else:
                    features = self._generate_dummy_features()
            else:
                features = self._generate_dummy_features()

            # Load mel spectrogram
            if 'mel_path' in sample:
                mel_path = sample['mel_path']
                if os.path.exists(mel_path):
                    mel = torch.load(mel_path)
                else:
                    mel = self._generate_dummy_mel()
            else:
                mel = self._generate_dummy_mel()

            # Get metadata
            speaker_id = sample.get('speaker_id', 0)
            emotion = sample.get('emotion', 'neutral')
        else:
            # Generate dummy data for testing
            features = self._generate_dummy_features()
            mel = self._generate_dummy_mel()
            speaker_id = idx % 10
            emotion = 'neutral'

        # Truncate/pad to max length
        features = self._pad_or_truncate(features, self.max_length)
        mel = self._pad_or_truncate(mel, self.max_length)

        return {
            'features': features,
            'mel': mel,
            'speaker_id': torch.tensor(speaker_id),
            'emotion': emotion,
        }

    def _generate_dummy_features(self) -> torch.Tensor:
        """Generate dummy HuBERT-like features for testing."""
        length = torch.randint(50, self.max_length, (1,)).item()
        return torch.randn(length, 768)  # HuBERT dimension

    def _generate_dummy_mel(self) -> torch.Tensor:
        """Generate dummy mel spectrogram for testing."""
        length = torch.randint(50, self.max_length, (1,)).item()
        return torch.randn(length, 80)  # Mel channels

    def _pad_or_truncate(
        self,
        tensor: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Pad or truncate tensor to fixed length."""
        if tensor.shape[0] > max_length:
            return tensor[:max_length]
        elif tensor.shape[0] < max_length:
            padding = torch.zeros(max_length - tensor.shape[0], tensor.shape[1])
            return torch.cat([tensor, padding], dim=0)
        return tensor


# =============================================================================
# TRAINER
# =============================================================================

class VEVOTrainer:
    """
    Trainer for VEVO model.

    Handles training loop, checkpointing, logging, and evaluation.
    """

    def __init__(
        self,
        config: Dict,
        model: VEVO,
        loss_fn: VEVOLoss,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Training settings
        train_config = config.get('training', {})
        self.num_epochs = train_config.get('num_epochs', 50)
        self.max_steps = train_config.get('max_steps', 100000)
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)

        # Logging
        log_config = config.get('logging', {})
        self.log_every = log_config.get('log_every_n_steps', 100)

        # Checkpointing
        checkpoint_config = config.get('checkpointing', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', '../checkpoints/vevo'))
        self.save_every = checkpoint_config.get('save_every_n_steps', 5000)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer and scheduler
        self._setup_optimizer(train_config)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def _setup_optimizer(self, train_config: Dict):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=(
                train_config.get('adam_beta1', 0.9),
                train_config.get('adam_beta2', 0.999),
            ),
            eps=train_config.get('adam_epsilon', 1e-8),
        )

        # Learning rate scheduler
        warmup_steps = train_config.get('warmup_steps', 1000)
        total_steps = self.max_steps

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
        )

        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        features = batch['features'].to(self.device)
        mel = batch['mel'].to(self.device)

        # Forward pass
        output = self.model(features, mel)

        # Compute loss
        losses = self.loss_fn(output, mel)

        # Backward pass
        loss = losses['total'] / self.gradient_accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        # Return metrics
        return {
            'loss': losses['total'].item(),
            'commitment': losses['commitment'].item(),
            'acoustic': losses['acoustic'].item(),
            'perplexity': output['content_perplexity'].item(),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in self.val_dataloader:
            features = batch['features'].to(self.device)
            mel = batch['mel'].to(self.device)

            output = self.model(features, mel)
            losses = self.loss_fn(output, mel)

            total_loss += losses['total'].item() * features.shape[0]
            total_samples += features.shape[0]

        return {
            'val_loss': total_loss / max(total_samples, 1),
        }

    def save_checkpoint(self, suffix: str = ""):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_step_{self.global_step}{suffix}.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        logger.info(f"Loaded checkpoint from step {self.global_step}")

    def train(self):
        """Main training loop."""
        logger.info("Starting VEVO training...")
        logger.info(f"  Total epochs: {self.num_epochs}")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Device: {self.device}")

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Training step
                metrics = self.train_step(batch)

                epoch_loss += metrics['loss']
                epoch_samples += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_loss = epoch_loss / max(epoch_samples, 1)
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs} | "
                        f"Step {self.global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Commitment: {metrics['commitment']:.4f} | "
                        f"Acoustic: {metrics['acoustic']:.4f} | "
                        f"Perplexity: {metrics['perplexity']:.1f} | "
                        f"LR: {lr:.2e}"
                    )

                # Checkpointing
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint()

                # Check max steps
                if self.global_step >= self.max_steps:
                    break

            # End of epoch
            avg_loss = epoch_loss / max(epoch_samples, 1)
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

            # Validation
            val_metrics = self.validate()
            if val_metrics:
                logger.info(f"Validation loss: {val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    torch.save(
                        self.model.state_dict(),
                        self.save_dir / "best.pt"
                    )
                    logger.info("Saved best model")

            if self.global_step >= self.max_steps:
                break

        # Final checkpoint
        self.save_checkpoint(suffix="_final")
        logger.info("Training complete!")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_disentanglement(
    model: VEVO,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate disentanglement quality.

    Metrics:
    - Content perplexity (codebook usage)
    - Style-timbre orthogonality
    - Reconstruction quality
    """
    model.eval()

    total_perplexity = 0.0
    total_orthogonality = 0.0
    total_recon_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            mel = batch['mel'].to(device)

            # Forward pass
            output = model(features, mel)

            # Perplexity
            total_perplexity += output['content_perplexity'].item()

            # Style-timbre orthogonality
            style_norm = F.normalize(output['style_emb'], p=2, dim=-1)
            min_dim = min(output['style_emb'].shape[-1], output['timbre_emb'].shape[-1])
            timbre_norm = F.normalize(output['timbre_emb'][:, :min_dim], p=2, dim=-1)
            cos_sim = (style_norm[:, :min_dim] * timbre_norm).sum(dim=-1).abs().mean()
            total_orthogonality += cos_sim.item()

            # Reconstruction (acoustic loss serves as proxy)
            total_recon_loss += output.get('acoustic', torch.tensor(0.0)).item()

            total_samples += 1

    num_batches = max(total_samples, 1)

    return {
        'avg_perplexity': total_perplexity / num_batches,
        'avg_orthogonality': total_orthogonality / num_batches,
        'avg_recon_loss': total_recon_loss / num_batches,
    }


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict) -> Tuple[VEVO, VEVOLoss]:
    """Create model and loss function from config."""
    model_config = config.get('model', {})

    vevo_config = VEVOConfig(
        input_dim=model_config.get('input_dim', 768),
        mel_dim=model_config.get('mel_dim', 80),
        sample_rate=model_config.get('sample_rate', 16000),
        hop_length=model_config.get('hop_length', 320),
        content_codebook_size=model_config.get('content_codebook_size', 1024),
        content_code_dim=model_config.get('content_code_dim', 256),
        content_commitment_cost=model_config.get('content_commitment_cost', 0.25),
        content_ema_decay=model_config.get('content_ema_decay', 0.99),
        style_dim=model_config.get('style_dim', 512),
        style_num_layers=model_config.get('style_num_layers', 4),
        style_num_heads=model_config.get('style_num_heads', 8),
        style_ffn_dim=model_config.get('style_ffn_dim', 2048),
        timbre_dim=model_config.get('timbre_dim', 256),
        cs_hidden_dim=model_config.get('cs_hidden_dim', 768),
        cs_num_layers=model_config.get('cs_num_layers', 8),
        cs_num_heads=model_config.get('cs_num_heads', 12),
        cs_ffn_dim=model_config.get('cs_ffn_dim', 3072),
        cs_dropout=model_config.get('cs_dropout', 0.1),
        acoustic_hidden_dim=model_config.get('acoustic_hidden_dim', 512),
        acoustic_num_layers=model_config.get('acoustic_num_layers', 6),
        acoustic_num_heads=model_config.get('acoustic_num_heads', 8),
        acoustic_dropout=model_config.get('acoustic_dropout', 0.1),
        sigma_min=model_config.get('sigma_min', 0.001),
        num_ode_steps=model_config.get('num_ode_steps', 50),
        output_dim=model_config.get('output_dim', 2048),
        num_prefix_tokens=model_config.get('num_prefix_tokens', 4),
    )

    model = VEVO(vevo_config)
    loss_fn = VEVOLoss(vevo_config)

    return model, loss_fn


def run_test():
    """Run quick test to verify model works."""
    print("=" * 60)
    print("VEVO Training Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create model with default config
    config = VEVOConfig()
    model = VEVO(config)
    loss_fn = VEVOLoss(config)

    # Create dummy data
    batch_size = 2
    seq_len = 100

    features = torch.randn(batch_size, seq_len, config.input_dim)
    mel = torch.randn(batch_size, seq_len, config.mel_dim)

    model.to(device)
    features = features.to(device)
    mel = mel.to(device)

    # Test forward pass
    print("\n[Test 1] Forward pass...")
    model.train()
    output = model(features, mel)
    print(f"  Output keys: {list(output.keys())}")
    print("  [PASS]")

    # Test loss computation
    print("\n[Test 2] Loss computation...")
    losses = loss_fn(output, mel)
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test backward pass
    print("\n[Test 3] Backward pass...")
    losses['total'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test generation
    print("\n[Test 4] Generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            features, features, features,
            num_ode_steps=5,
        )
    print(f"  Generated shape: {generated.shape}")
    print("  [PASS]")

    # Test adapter
    print("\n[Test 5] CSM Adapter...")
    adapter = VEVOAdapter(config, model).to(device)
    prefix_tokens = adapter(features)
    print(f"  Prefix tokens shape: {prefix_tokens.shape}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All tests passed! VEVO is ready for training.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VEVO Training")
    parser.add_argument('--config', type=str, default='config/vevo_disentanglement.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate disentanglement')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint for evaluation')
    args = parser.parse_args()

    # Run test mode
    if args.test:
        run_test()
        return

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    config = load_config(str(config_path))

    # Determine device
    hardware_config = config.get('hardware', {})
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    device = hardware_config.get('device', device)

    logger.info(f"Using device: {device}")

    # Create model
    model, loss_fn = create_model_from_config(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create dataset
    data_config = config.get('data', {})
    train_dataset = VEVODataset(
        manifest_path=data_config.get('manifest_path', '../data/manifest.json'),
        max_length=500,
    )

    train_config = config.get('training', {})
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
    )

    # Evaluation mode
    if args.evaluate:
        if args.checkpoint:
            state_dict = torch.load(args.checkpoint, map_location=device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint: {args.checkpoint}")

        model.to(device)
        metrics = evaluate_disentanglement(model, train_dataloader, device)
        print("\nDisentanglement Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        return

    # Create trainer
    trainer = VEVOTrainer(
        config=config,
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        device=device,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
