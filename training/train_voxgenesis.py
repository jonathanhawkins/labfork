#!/usr/bin/env python3
"""
Training script for VoxGenesis latent speaker manifold.

Based on "VoxGenesis: Interpretable Voice Synthesis and Manipulation"
arXiv:2403.00529 (March 2024)

Usage:
    # Train from scratch
    python train_voxgenesis.py --config config/voxgenesis.yaml

    # Resume training
    python train_voxgenesis.py --config config/voxgenesis.yaml \
        --resume ../checkpoints/voxgenesis/latest.pt

    # Discover directions from trained model
    python train_voxgenesis.py --discover-directions \
        --checkpoint ../checkpoints/voxgenesis/best.pt

    # Test mode (synthetic data)
    python train_voxgenesis.py --test
"""

import argparse
import json
import logging
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

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from voxgenesis import (
    VoxGenesisConfig,
    VoxGenesis,
    VoxGenesisLoss,
    VoxGenesisAdapter,
    discover_directions,
    label_directions_from_analysis,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class VoxGenesisDataset(Dataset):
    """Dataset for VoxGenesis training."""

    def __init__(
        self,
        manifest_path: str,
        audio_dir: str,
        feature_dim: int = 768,
        mel_dim: int = 80,
        max_length: float = 15.0,
        min_length: float = 1.0,
        sample_rate: int = 16000,
        hop_length: int = 320,
    ):
        self.audio_dir = Path(audio_dir)
        self.feature_dim = feature_dim
        self.mel_dim = mel_dim
        self.max_frames = int(max_length * sample_rate / hop_length)
        self.min_frames = int(min_length * sample_rate / hop_length)
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Load manifest
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.samples = json.load(f)
        else:
            # Create dummy samples for testing
            self.samples = [
                {'audio_path': 'dummy.wav', 'speaker_id': i % 10}
                for i in range(100)
            ]

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # In real implementation, load and extract features
        # For now, generate synthetic data
        seq_len = torch.randint(self.min_frames, self.max_frames, (1,)).item()
        mel_len = seq_len * 4  # Upsampled length

        features = torch.randn(seq_len, self.feature_dim)
        semantic_tokens = torch.randn(seq_len, self.feature_dim)
        mel = torch.randn(mel_len, self.mel_dim)

        speaker_id = sample.get('speaker_id', 0)

        return {
            'features': features,
            'semantic_tokens': semantic_tokens,
            'mel': mel,
            'speaker_id': torch.tensor(speaker_id),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    # Find max lengths
    max_feat_len = max(b['features'].shape[0] for b in batch)
    max_mel_len = max(b['mel'].shape[0] for b in batch)

    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[-1]
    mel_dim = batch[0]['mel'].shape[-1]

    # Initialize padded tensors
    features = torch.zeros(batch_size, max_feat_len, feature_dim)
    semantic_tokens = torch.zeros(batch_size, max_feat_len, feature_dim)
    mel = torch.zeros(batch_size, max_mel_len, mel_dim)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)

    # Fill in
    for i, b in enumerate(batch):
        feat_len = b['features'].shape[0]
        mel_len = b['mel'].shape[0]

        features[i, :feat_len] = b['features']
        semantic_tokens[i, :feat_len] = b['semantic_tokens']
        mel[i, :mel_len] = b['mel']
        speaker_ids[i] = b['speaker_id']

    return {
        'features': features,
        'semantic_tokens': semantic_tokens,
        'mel': mel,
        'speaker_id': speaker_ids,
    }


# =============================================================================
# TRAINER
# =============================================================================

class VoxGenesisTrainer:
    """Trainer for VoxGenesis model."""

    def __init__(
        self,
        config: Dict,
        model: VoxGenesis,
        loss_fn: VoxGenesisLoss,
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

        # Training config
        train_config = config.get('training', {})
        self.max_steps = train_config.get('max_steps', 100000)
        self.log_every = config.get('logging', {}).get('log_every_n_steps', 100)
        self.save_every = config.get('checkpointing', {}).get('save_every_n_steps', 5000)
        self.eval_every = config.get('evaluation', {}).get('eval_every_n_steps', 5000)

        # Separate optimizers for generator and discriminator
        self.optimizer_g = torch.optim.AdamW(
            list(model.speaker_encoder.parameters()) +
            list(model.mapping_network.parameters()) +
            list(model.semantic_embedding.parameters()) +
            list(model.transformation.parameters()) +
            list(model.deconvolution.parameters()) +
            list(model.output_proj.parameters()),
            lr=train_config.get('lr_generator', 1e-4),
            betas=(train_config.get('adam_beta1', 0.5), train_config.get('adam_beta2', 0.9)),
            weight_decay=train_config.get('weight_decay', 0.01),
        )

        self.optimizer_d = torch.optim.AdamW(
            model.discriminator.parameters(),
            lr=train_config.get('lr_discriminator', 1e-4),
            betas=(train_config.get('adam_beta1', 0.5), train_config.get('adam_beta2', 0.9)),
            weight_decay=train_config.get('weight_decay', 0.01),
        )

        # Schedulers
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=self.max_steps
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=self.max_steps
        )

        # Gradient accumulation
        self.grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)

        # D updates per G update
        self.d_updates_per_g = train_config.get('d_updates_per_g', 1)

        # Checkpointing
        self.save_dir = Path(config.get('checkpointing', {}).get('save_dir', '../checkpoints/voxgenesis'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Step counter
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        features = batch['features'].to(self.device)
        semantic_tokens = batch['semantic_tokens'].to(self.device)
        mel = batch['mel'].to(self.device)

        losses_dict = {}

        # ========== Train Discriminator ==========
        self.optimizer_d.zero_grad()

        # Forward pass
        with torch.no_grad():
            gen_output = self.model(features, semantic_tokens, mel)
            fake_mel = gen_output['mel'].detach()

        # Discriminator on real
        disc_real = self.model.discriminator(mel, semantic_tokens)
        # Discriminator on fake
        disc_fake = self.model.discriminator(fake_mel, semantic_tokens)

        # D loss
        d_real = F.mse_loss(disc_real['logits'], torch.ones_like(disc_real['logits']))
        d_fake = F.mse_loss(disc_fake['logits'], torch.zeros_like(disc_fake['logits']))
        d_loss = (d_real + d_fake) / 2

        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.max_grad_norm)
        self.optimizer_d.step()

        losses_dict['d_loss'] = d_loss.item()
        losses_dict['d_real'] = d_real.item()
        losses_dict['d_fake'] = d_fake.item()

        # ========== Train Generator ==========
        self.optimizer_g.zero_grad()

        # Full forward pass with gradients
        gen_output = self.model(features, semantic_tokens, mel)

        # Compute losses
        losses = self.loss_fn(gen_output, mel, step=self.global_step)

        g_total = losses['g_total']
        g_total.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.model.speaker_encoder.parameters()) +
            list(self.model.mapping_network.parameters()) +
            list(self.model.semantic_embedding.parameters()) +
            list(self.model.transformation.parameters()) +
            list(self.model.deconvolution.parameters()),
            self.max_grad_norm
        )
        self.optimizer_g.step()

        # Collect losses
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                losses_dict[k] = v.item()
            else:
                losses_dict[k] = v

        return losses_dict

    def train(self):
        """Main training loop."""
        logger.info("Starting VoxGenesis training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Max steps: {self.max_steps}")

        self.model.train()
        running_losses = {}

        while self.global_step < self.max_steps:
            for batch in self.train_dataloader:
                if self.global_step >= self.max_steps:
                    break

                # Training step
                losses = self.train_step(batch)

                # Update running losses
                for k, v in losses.items():
                    if k not in running_losses:
                        running_losses[k] = 0.0
                    running_losses[k] += v

                self.global_step += 1

                # Update schedulers
                self.scheduler_g.step()
                self.scheduler_d.step()

                # Logging
                if self.global_step % self.log_every == 0:
                    avg_losses = {k: v / self.log_every for k, v in running_losses.items()}
                    log_str = f"Step {self.global_step}: "
                    log_str += ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
                    logger.info(log_str)
                    running_losses = {}

                # Checkpointing
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

                # Evaluation
                if self.val_dataloader and self.global_step % self.eval_every == 0:
                    self.evaluate()

        # Final save
        self.save_checkpoint("final.pt")

        # Discover directions after training
        logger.info("Discovering interpretable directions...")
        self.discover_and_save_directions()

        logger.info("Training complete!")

    @torch.no_grad()
    def evaluate(self):
        """Evaluation on validation set."""
        self.model.eval()

        total_losses = {}
        num_batches = 0

        for batch in self.val_dataloader:
            features = batch['features'].to(self.device)
            semantic_tokens = batch['semantic_tokens'].to(self.device)
            mel = batch['mel'].to(self.device)

            # Forward pass
            gen_output = self.model(features, semantic_tokens, mel)
            losses = self.loss_fn(gen_output, mel, step=self.global_step)

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    if k not in total_losses:
                        total_losses[k] = 0.0
                    total_losses[k] += v.item()

            num_batches += 1

            if num_batches >= 50:  # Limit eval batches
                break

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        log_str = f"[EVAL] Step {self.global_step}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        logger.info(log_str)

        self.model.train()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
        }

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

        # Also save as 'latest.pt'
        latest_path = self.save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.global_step = checkpoint['global_step']

        logger.info(f"Loaded checkpoint from {checkpoint_path}, step {self.global_step}")

    def discover_and_save_directions(self):
        """Discover interpretable directions and save them."""
        self.model.eval()

        # Discover directions via PCA
        directions_info = self.model.direction_discovery.discover_from_samples(
            self.model.mapping_network,
            num_samples=self.config.get('direction_discovery', {}).get('num_samples', 10000),
            device=torch.device(self.device),
        )

        # Label directions with defaults
        default_labels = self.config.get('direction_discovery', {}).get('default_labels', {})
        for idx_str, label in default_labels.items():
            idx = int(idx_str)
            if idx < self.model.config.num_directions:
                self.model.direction_discovery.label_direction(idx, label)

        # Save directions
        directions_path = self.save_dir / "directions.pt"
        torch.save({
            'directions': self.model.direction_discovery.directions,
            'eigenvalues': self.model.direction_discovery.eigenvalues,
            'mean': self.model.direction_discovery.mean,
            'labels': self.model.direction_discovery.direction_labels,
            'explained_variance_ratio': directions_info['explained_variance_ratio'],
        }, directions_path)

        logger.info(f"Saved directions to {directions_path}")
        logger.info(f"Top 5 explained variance: {directions_info['explained_variance_ratio'][:5].cpu().numpy()}")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict) -> Tuple[VoxGenesis, VoxGenesisLoss]:
    """Create model and loss from config."""
    model_config = config.get('model', {})
    training_config = config.get('training', {})

    vox_config = VoxGenesisConfig(
        input_dim=model_config.get('input_dim', 768),
        semantic_dim=model_config.get('semantic_dim', 768),
        mel_dim=model_config.get('mel_dim', 80),
        sample_rate=model_config.get('sample_rate', 16000),
        hop_length=model_config.get('hop_length', 320),
        latent_dim=model_config.get('latent_dim', 512),
        mapping_dim=model_config.get('mapping_dim', 512),
        num_mapping_layers=model_config.get('num_mapping_layers', 7),
        semantic_hidden_dim=model_config.get('semantic_hidden_dim', 512),
        semantic_num_layers=model_config.get('semantic_num_layers', 4),
        semantic_num_heads=model_config.get('semantic_num_heads', 8),
        transform_hidden_dim=model_config.get('transform_hidden_dim', 512),
        transform_num_layers=model_config.get('transform_num_layers', 4),
        deconv_hidden_dim=model_config.get('deconv_hidden_dim', 512),
        deconv_num_layers=model_config.get('deconv_num_layers', 8),
        deconv_upsample_rates=model_config.get('deconv_upsample_rates', [10, 4, 2, 2]),
        disc_hidden_dim=model_config.get('disc_hidden_dim', 512),
        disc_num_layers=model_config.get('disc_num_layers', 4),
        dropout=model_config.get('dropout', 0.1),
        use_spectral_norm=model_config.get('use_spectral_norm', True),
        gan_feature_matching_weight=training_config.get('gan_feature_matching_weight', 10.0),
        gan_disc_weight=training_config.get('gan_disc_weight', 1.0),
        kl_weight=training_config.get('kl_weight', 0.01),
        kl_anneal_steps=training_config.get('kl_anneal_steps', 10000),
        num_directions=model_config.get('num_directions', 32),
        output_dim=model_config.get('output_dim', 2048),
        num_prefix_tokens=model_config.get('num_prefix_tokens', 4),
    )

    model = VoxGenesis(vox_config)
    loss_fn = VoxGenesisLoss(vox_config)

    return model, loss_fn


def main():
    parser = argparse.ArgumentParser(description="Train VoxGenesis model")
    parser.add_argument("--config", type=str, default="config/voxgenesis.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--discover-directions", action="store_true",
                        help="Discover directions from trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint for direction discovery")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with synthetic data")

    args = parser.parse_args()

    # Test mode
    if args.test:
        logger.info("Running in test mode...")
        run_test_mode()
        return

    # Direction discovery mode
    if args.discover_directions:
        if not args.checkpoint:
            logger.error("--checkpoint required for direction discovery")
            return
        discover_directions_from_checkpoint(args.config, args.checkpoint)
        return

    # Load config
    config = load_config(args.config)

    # Create model
    model, loss_fn = create_model_from_config(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create datasets
    data_config = config.get('data', {})
    train_dataset = VoxGenesisDataset(
        manifest_path=data_config.get('manifest_path', '../data/manifest.json'),
        audio_dir=data_config.get('audio_dir', '../data/audio'),
        feature_dim=config.get('model', {}).get('input_dim', 768),
        mel_dim=config.get('model', {}).get('mel_dim', 80),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=data_config.get('pin_memory', True),
    )

    # Validation dataloader (optional)
    val_dataloader = None

    # Device
    device = config.get('hardware', {}).get('device', 'cuda')
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create trainer
    trainer = VoxGenesisTrainer(
        config=config,
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


def run_test_mode():
    """Run in test mode with synthetic data."""
    logger.info("=" * 60)
    logger.info("VoxGenesis Test Mode")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create model
    config = VoxGenesisConfig()
    model = VoxGenesis(config).to(device)
    loss_fn = VoxGenesisLoss(config)

    # Create synthetic data
    batch_size = 2
    seq_len = 100
    mel_len = 400

    features = torch.randn(batch_size, seq_len, config.input_dim).to(device)
    semantic_tokens = torch.randn(batch_size, seq_len, config.semantic_dim).to(device)
    mel = torch.randn(batch_size, mel_len, config.mel_dim).to(device)

    # Forward pass
    logger.info("\n[Test] Forward pass...")
    model.train()
    output = model(features, semantic_tokens, mel)
    logger.info(f"  Generated mel shape: {output['mel'].shape}")
    logger.info(f"  z shape: {output['z'].shape}")
    logger.info(f"  w shape: {output['w'].shape}")

    # Compute loss
    logger.info("\n[Test] Loss computation...")
    losses = loss_fn(output, mel, step=1000)
    logger.info(f"  G total loss: {losses['g_total'].item():.4f}")
    logger.info(f"  D loss: {losses['d_loss'].item():.4f}")

    # Backward pass
    logger.info("\n[Test] Backward pass...")
    losses['g_total'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    logger.info(f"  Gradient norm: {grad_norm:.4f}")

    # Direction discovery
    logger.info("\n[Test] Direction discovery...")
    model.eval()
    directions = model.direction_discovery.discover_from_samples(
        model.mapping_network,
        num_samples=500,
        device=torch.device(device),
    )
    logger.info(f"  Discovered {config.num_directions} directions")
    logger.info(f"  Top 5 explained variance: {directions['explained_variance_ratio'][:5].cpu().numpy()}")

    # Voice editing
    logger.info("\n[Test] Voice editing...")
    edit_output = model.voice_editing(
        features, semantic_tokens,
        direction_idx=2,
        scale=0.5,
    )
    logger.info(f"  Original mel shape: {edit_output['original_mel'].shape}")
    logger.info(f"  Edited mel shape: {edit_output['edited_mel'].shape}")

    # Novel speaker generation
    logger.info("\n[Test] Novel speaker generation...")
    novel_output = model.generate_novel_speaker(semantic_tokens, num_speakers=3)
    logger.info(f"  Generated {len(novel_output['generations'])} novel speakers")

    # CSM adapter
    logger.info("\n[Test] CSM adapter...")
    adapter = VoxGenesisAdapter(config, model).to(device)
    adapter_output = adapter(features, semantic_tokens)
    logger.info(f"  Prosody tokens shape: {adapter_output['prosody_tokens'].shape}")

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


def discover_directions_from_checkpoint(config_path: str, checkpoint_path: str):
    """Discover directions from a trained checkpoint."""
    logger.info("Discovering directions from checkpoint...")

    config = load_config(config_path)
    model, _ = create_model_from_config(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Discover directions
    directions_info = model.direction_discovery.discover_from_samples(
        model.mapping_network,
        num_samples=config.get('direction_discovery', {}).get('num_samples', 10000),
        device=torch.device(device),
    )

    # Label directions
    default_labels = config.get('direction_discovery', {}).get('default_labels', {})
    for idx_str, label in default_labels.items():
        idx = int(idx_str)
        if idx < model.config.num_directions:
            model.direction_discovery.label_direction(idx, label)

    # Print results
    logger.info("\nDiscovered Directions:")
    logger.info("-" * 40)
    for i in range(min(10, model.config.num_directions)):
        var = directions_info['explained_variance_ratio'][i].item()
        label = ""
        for name, idx in model.direction_discovery.direction_labels.items():
            if idx == i:
                label = f" ({name})"
                break
        logger.info(f"  PC{i}: {var*100:.2f}% variance{label}")

    cumulative = directions_info['explained_variance_ratio'].cumsum(dim=0)
    logger.info(f"\nCumulative variance (top 5): {cumulative[4].item()*100:.2f}%")
    logger.info(f"Cumulative variance (top 10): {cumulative[9].item()*100:.2f}%")

    # Save directions
    save_dir = Path(checkpoint_path).parent
    directions_path = save_dir / "directions.pt"
    torch.save({
        'directions': model.direction_discovery.directions,
        'eigenvalues': model.direction_discovery.eigenvalues,
        'mean': model.direction_discovery.mean,
        'labels': model.direction_discovery.direction_labels,
        'explained_variance_ratio': directions_info['explained_variance_ratio'],
    }, directions_path)
    logger.info(f"\nSaved directions to {directions_path}")


if __name__ == "__main__":
    main()
