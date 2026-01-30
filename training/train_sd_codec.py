"""
Training script for SD-Codec: Source Disentanglement via Joint Coding and Separation.

Based on "Learning Source Disentanglement in Neural Audio Codec" (2025)

Usage:
    # Train SD-Codec model
    python train_sd_codec.py --config config/sd_codec.yaml

    # Resume from checkpoint
    python train_sd_codec.py --config config/sd_codec.yaml \\
        --resume ../checkpoints/sd_codec/latest.pt

    # Test mode with synthetic data
    python train_sd_codec.py --test
"""

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import yaml

from sd_codec import (
    SDCodecConfig,
    SDCodec,
    SDCodecLoss,
    SDCodecAdapter,
    compute_source_statistics,
    analyze_source_separation,
)


# =============================================================================
# DATASET
# =============================================================================

class MelSpectrogramDataset(Dataset):
    """Dataset for mel spectrograms."""

    def __init__(
        self,
        manifest_path: str,
        mel_dir: str,
        segment_frames: int = 200,
        augment: bool = False,
    ):
        self.mel_dir = Path(mel_dir)
        self.segment_frames = segment_frames
        self.augment = augment

        # Load manifest
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.samples = self.manifest.get("samples", self.manifest)
        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load mel spectrogram
        mel_path = self.mel_dir / sample.get("mel_path", f"{sample['id']}.pt")
        mel = torch.load(mel_path)

        # Ensure shape is [seq, mel_dim]
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        if mel.dim() == 1:
            mel = mel.unsqueeze(-1)

        # Random crop or pad
        if mel.shape[0] >= self.segment_frames:
            start = random.randint(0, mel.shape[0] - self.segment_frames)
            mel = mel[start:start + self.segment_frames]
        else:
            # Pad with zeros
            pad = torch.zeros(self.segment_frames - mel.shape[0], mel.shape[1])
            mel = torch.cat([mel, pad], dim=0)

        # Augmentation
        if self.augment:
            # Random gain
            gain = 0.8 + 0.4 * random.random()
            mel = mel * gain

            # Random noise
            noise = torch.randn_like(mel) * 0.01
            mel = mel + noise

        return {"mel": mel, "id": sample.get("id", str(idx))}


class SyntheticMelDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 200,
        mel_dim: int = 80,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.mel_dim = mel_dim

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate synthetic mel with some structure
        t = torch.linspace(0, 1, self.seq_len).unsqueeze(1)

        # Base frequency pattern (prosody-like)
        freq = 2 + 3 * random.random()
        prosody = torch.sin(2 * math.pi * freq * t)

        # Harmonic structure (content-like)
        harmonics = sum(
            torch.sin(2 * math.pi * (i + 1) * freq * 0.3 * t) / (i + 1)
            for i in range(5)
        )

        # Random spectral envelope (timbre-like)
        envelope = torch.randn(1, self.mel_dim) * 0.5
        envelope = envelope.expand(self.seq_len, -1)

        # Combine
        mel = prosody * 0.3 + harmonics * 0.3 + envelope * 0.4
        mel = mel + torch.randn_like(mel) * 0.1

        return {"mel": mel, "id": f"synthetic_{idx}"}


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: SDCodec,
    loss_fn: SDCodecLoss,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    config: dict,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        mel = batch["mel"].to(device)

        # Forward pass
        output = model(mel, return_routing=True)
        losses = loss_fn(output, mel)

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        if config.get("gradient_clip", 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"]
            )

        optimizer.step()

        # Accumulate losses
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()

        num_batches += 1

        # Log progress
        if batch_idx % config.get("log_interval", 50) == 0:
            print(
                f"  Batch {batch_idx}/{len(dataloader)}: "
                f"loss={losses['total'].item():.4f}, "
                f"recon={losses['reconstruction'].item():.4f}, "
                f"perp={losses['mean_perplexity'].item():.2f}"
            )

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def validate(
    model: SDCodec,
    loss_fn: SDCodecLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_losses = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)

            output = model(mel)
            losses = loss_fn(output, mel)

            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()

            num_batches += 1

    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def analyze_model(
    model: SDCodec,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
) -> Dict[str, float]:
    """Analyze model disentanglement."""
    model.eval()

    all_stats = []
    all_analysis = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            mel = batch["mel"].to(device)

            stats = compute_source_statistics(model, mel)
            all_stats.append(stats)

            if model.separation_heads is not None:
                analysis = analyze_source_separation(model, mel)
                all_analysis.append(analysis)

    # Aggregate statistics
    results = {}

    # Average routing probabilities
    routing_probs = torch.stack([s["routing_probs"] for s in all_stats]).mean(0)
    for i, prob in enumerate(routing_probs):
        results[f"routing_prob_source_{i}"] = prob.item()

    # Average perplexities
    perplexities = torch.stack([s["perplexities"] for s in all_stats]).mean(0)
    for i, perp in enumerate(perplexities):
        results[f"perplexity_source_{i}"] = perp.item()

    # Disentanglement score
    if all_analysis:
        disentangle_scores = torch.stack([a["disentanglement_score"] for a in all_analysis])
        results["disentanglement_score"] = disentangle_scores.mean().item()

        recon_errors = torch.stack([a["sum_reconstruction_error"] for a in all_analysis])
        results["separation_recon_error"] = recon_errors.mean().item()

    return results


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    config_path: str,
    resume_path: Optional[str] = None,
    test_mode: bool = False,
):
    """Main training function."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model config
    model_config = SDCodecConfig(
        num_sources=config.get("num_sources", 3),
        source_names=tuple(config.get("source_names", ["prosody", "content", "timbre"])),
        mel_dim=config.get("mel_dim", 80),
        encoder_hidden_dim=config.get("encoder_hidden_dim", 512),
        encoder_num_layers=config.get("encoder_num_layers", 6),
        encoder_num_heads=config.get("encoder_num_heads", 8),
        encoder_ffn_dim=config.get("encoder_ffn_dim", 2048),
        decoder_hidden_dim=config.get("decoder_hidden_dim", 512),
        decoder_num_layers=config.get("decoder_num_layers", 6),
        decoder_num_heads=config.get("decoder_num_heads", 8),
        decoder_ffn_dim=config.get("decoder_ffn_dim", 2048),
        router_hidden_dim=config.get("router_hidden_dim", 256),
        router_temperature=config.get("router_temperature", 0.5),
        source_encoder_hidden_dim=config.get("source_encoder_hidden_dim", 256),
        source_encoder_num_layers=config.get("source_encoder_num_layers", 3),
        default_codebook_size=config.get("default_codebook_size", 512),
        default_code_dim=config.get("default_code_dim", 64),
        commitment_cost=config.get("commitment_cost", 0.25),
        use_separation_head=config.get("use_separation_head", True),
        separation_hidden_dim=config.get("separation_hidden_dim", 256),
        reconstruction_weight=config.get("reconstruction_weight", 1.0),
        separation_weight=config.get("separation_weight", 0.5),
        commitment_weight=config.get("commitment_weight", 0.25),
        routing_entropy_weight=config.get("routing_entropy_weight", 0.1),
        orthogonality_weight=config.get("orthogonality_weight", 0.05),
        output_dim=config.get("output_dim", 2048),
        num_prefix_tokens=config.get("num_prefix_tokens", 4),
        dropout=config.get("dropout", 0.1),
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SDCodec(model_config).to(device)
    loss_fn = SDCodecLoss(model_config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create datasets
    training_config = config.get("training", {})

    if test_mode:
        train_dataset = SyntheticMelDataset(
            num_samples=500,
            seq_len=training_config.get("segment_frames", 200),
            mel_dim=model_config.mel_dim,
        )
        val_dataset = SyntheticMelDataset(
            num_samples=100,
            seq_len=training_config.get("segment_frames", 200),
            mel_dim=model_config.mel_dim,
        )
    else:
        train_dataset = MelSpectrogramDataset(
            manifest_path=config["train_manifest"],
            mel_dir=config["mel_dir"],
            segment_frames=training_config.get("segment_frames", 200),
            augment=True,
        )
        val_dataset = MelSpectrogramDataset(
            manifest_path=config["val_manifest"],
            mel_dir=config["mel_dir"],
            segment_frames=training_config.get("segment_frames", 200),
            augment=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=True,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.get("batch_size", 16),
        shuffle=False,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    # Learning rate scheduler
    num_epochs = training_config.get("num_epochs", 100)
    warmup_steps = training_config.get("warmup_steps", 1000)
    total_steps = len(train_loader) * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        global_step = checkpoint.get("global_step", 0)

    # Checkpoint directory
    checkpoint_dir = Path(config.get("checkpoint_dir", "../checkpoints/sd_codec"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_losses = train_epoch(
            model, loss_fn, optimizer, train_loader, device, epoch, training_config
        )

        # Update scheduler
        scheduler.step()

        # Validate
        val_losses = validate(model, loss_fn, val_loader, device)

        # Analyze disentanglement
        if epoch % training_config.get("analyze_interval", 10) == 0:
            analysis = analyze_model(model, val_loader, device)
            print(f"  Disentanglement: {analysis.get('disentanglement_score', 0):.4f}")
            for i in range(model_config.num_sources):
                print(
                    f"    Source {i}: "
                    f"routing={analysis.get(f'routing_prob_source_{i}', 0):.3f}, "
                    f"perp={analysis.get(f'perplexity_source_{i}', 0):.2f}"
                )

        # Log
        print(
            f"  Train: loss={train_losses['total']:.4f}, "
            f"recon={train_losses['reconstruction']:.4f}, "
            f"sep={train_losses['separation']:.4f}"
        )
        print(
            f"  Val: loss={val_losses['total']:.4f}, "
            f"recon={val_losses['reconstruction']:.4f}, "
            f"sep={val_losses['separation']:.4f}"
        )

        # Save checkpoints
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["total"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "global_step": global_step,
            "config": config,
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"  New best model saved! (loss={best_val_loss:.4f})")

        # Save periodic
        if (epoch + 1) % training_config.get("save_interval", 10) == 0:
            torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1}.pt")

        global_step += len(train_loader)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SD-Codec model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sd_codec.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with synthetic data",
    )

    args = parser.parse_args()

    if args.test:
        # Create minimal config for testing
        test_config = {
            "num_sources": 3,
            "source_names": ["prosody", "content", "timbre"],
            "mel_dim": 80,
            "encoder_hidden_dim": 256,
            "encoder_num_layers": 2,
            "encoder_num_heads": 4,
            "encoder_ffn_dim": 512,
            "decoder_hidden_dim": 256,
            "decoder_num_layers": 2,
            "decoder_num_heads": 4,
            "decoder_ffn_dim": 512,
            "router_hidden_dim": 128,
            "source_encoder_hidden_dim": 128,
            "source_encoder_num_layers": 2,
            "default_codebook_size": 256,
            "default_code_dim": 32,
            "use_separation_head": True,
            "output_dim": 512,
            "checkpoint_dir": "../checkpoints/sd_codec_test",
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4,
                "num_epochs": 3,
                "warmup_steps": 10,
                "segment_frames": 100,
                "log_interval": 10,
                "analyze_interval": 1,
                "save_interval": 1,
                "gradient_clip": 1.0,
                "num_workers": 0,
            },
        }

        # Save temp config
        os.makedirs("config", exist_ok=True)
        with open("config/sd_codec_test.yaml", "w") as f:
            yaml.dump(test_config, f)

        train("config/sd_codec_test.yaml", test_mode=True)
    else:
        train(args.config, args.resume)


if __name__ == "__main__":
    main()
