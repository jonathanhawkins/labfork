#!/usr/bin/env python3
"""
EmoKnob Training Script

Two-phase training:
1. Direction Vector Extraction: Extract emotion directions from paired samples
2. Fine-tuning: Train model to apply emotion control with intensity

Based on: arXiv:2410.00316 "EmoKnob: Enhance Voice Cloning with Fine-Grained
          Emotion Control" (EMNLP 2024)

Usage:
    # Extract directions from paired samples
    python train_emoknob.py --extract --config config/emoknob.yaml

    # Train full model
    python train_emoknob.py --config config/emoknob.yaml

    # Resume from checkpoint
    python train_emoknob.py --config config/emoknob.yaml --resume

    # Test mode with synthetic data
    python train_emoknob.py --test
"""

import argparse
import json
import logging
import math
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.emoknob import (
    EmoKnobConfig,
    EmoKnob,
    EmoKnobAdapter,
    EmoKnobLoss,
    SpeakerEncoder,
    DirectionVectorExtractor,
    compute_emotion_statistics,
    EMOKNOB_EMOTIONS,
    EMOTION_TO_IDX,
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

class EmoKnobDataset(Dataset):
    """
    Dataset for EmoKnob training.

    Expects manifest with paired samples:
    {
        "speaker_id": "speaker_001",
        "emotion": "happy",
        "emotional_audio": "path/to/happy.wav",
        "neutral_audio": "path/to/neutral.wav",
        "text": "transcript"
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: EmoKnobConfig,
        feature_extractor: Optional[nn.Module] = None,
        max_length: int = 500,
    ):
        self.config = config
        self.max_length = max_length

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        # Group by emotion for direction extraction
        self.emotion_groups: Dict[str, List[dict]] = {}
        for sample in self.samples:
            emotion = sample.get("emotion", "neutral")
            if emotion not in self.emotion_groups:
                self.emotion_groups[emotion] = []
            self.emotion_groups[emotion].append(sample)

        # Feature extractor (optional - can be passed or created)
        self.feature_extractor = feature_extractor

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")
        logger.info(f"Emotion distribution: {[(k, len(v)) for k, v in self.emotion_groups.items()]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # In real implementation, load and process audio
        # For now, return placeholder features
        batch = {
            "speaker_id": sample.get("speaker_id", "unknown"),
            "emotion": sample.get("emotion", "neutral"),
            "emotion_idx": EMOTION_TO_IDX.get(sample.get("emotion", "neutral"), 0),
            "text": sample.get("text", ""),
        }

        # Features would be extracted from audio files
        # emotional_features: from sample["emotional_audio"]
        # neutral_features: from sample["neutral_audio"]

        return batch

    def get_paired_samples(
        self,
        emotion: str,
        num_pairs: int = 5,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get paired (emotional, neutral) samples for direction extraction.
        """
        if emotion not in self.emotion_groups:
            return [], []

        samples = self.emotion_groups[emotion][:num_pairs]
        emotional_features = []
        neutral_features = []

        for sample in samples:
            # In real implementation, load and extract features
            # emo_feat = self.feature_extractor(load_audio(sample["emotional_audio"]))
            # neu_feat = self.feature_extractor(load_audio(sample["neutral_audio"]))
            pass

        return emotional_features, neutral_features


class SyntheticEmoKnobDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples: int = 1000, config: EmoKnobConfig = None):
        self.num_samples = num_samples
        self.config = config or EmoKnobConfig()

        # Pre-generate synthetic data
        self.data = []
        emotions = ["happy", "sad", "angry", "surprised", "calm", "fearful"]

        for i in range(num_samples):
            emotion = emotions[i % len(emotions)]
            self.data.append({
                "speaker_id": f"speaker_{i % 10}",
                "emotion": emotion,
                "emotion_idx": EMOTION_TO_IDX.get(emotion, 0),
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # Generate synthetic features
        features = torch.randn(100, self.config.input_dim)  # [seq, dim]

        return {
            **sample,
            "features": features,
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def extract_direction_vectors(
    model: EmoKnob,
    dataset: EmoKnobDataset,
    config: EmoKnobConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Phase 1: Extract direction vectors from paired samples.
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Extracting Direction Vectors")
    logger.info("=" * 60)

    model.eval()
    extractor = DirectionVectorExtractor(model.speaker_encoder, config)

    directions = {}
    stats = {}

    for emotion in EMOKNOB_EMOTIONS:
        if emotion == "neutral":
            continue

        emo_features, neu_features = dataset.get_paired_samples(
            emotion, num_pairs=config.num_pairs_for_extraction
        )

        if not emo_features:
            logger.warning(f"No samples found for emotion: {emotion}")
            continue

        # Move to device
        emo_features = [f.to(device) for f in emo_features]
        neu_features = [f.to(device) for f in neu_features]

        # Extract direction
        direction = extractor.extract_direction(emo_features, neu_features)
        directions[emotion] = direction

        # Register with model
        model.manipulator.register_direction(emotion, direction)

        logger.info(f"Extracted direction for {emotion}: norm={direction.norm().item():.4f}")

    # Save directions
    cache_path = Path(config.direction_cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    for emotion, direction in directions.items():
        torch.save(direction.cpu(), cache_path / f"{emotion}.pt")

    logger.info(f"Saved {len(directions)} direction vectors to {cache_path}")

    return directions


def train_epoch(
    model: EmoKnob,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: EmoKnobLoss,
    device: torch.device,
    epoch: int,
    config: dict,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_direction = 0.0
    total_contrastive = 0.0
    total_consistency = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        features = batch["features"].to(device)
        emotion_idx = batch["emotion_idx"].to(device)
        emotions = batch["emotion"]

        # Forward pass
        # Sample intensity for training
        intensity = torch.rand(1).item() * config["model"]["max_intensity"]

        results = []
        for i, emotion in enumerate(emotions):
            result = model(features[i:i+1], emotion=emotion, intensity=intensity)
            results.append(result)

        # Stack results
        speaker_embs = torch.cat([r["speaker_emb"] for r in results], dim=0)
        controlled_embs = torch.cat([r["controlled_emb"] for r in results], dim=0)

        # Compute loss
        losses = loss_fn(speaker_embs, controlled_embs, emotion_idx)

        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))

        optimizer.step()

        # Accumulate metrics
        total_loss += losses["total"].item()
        total_direction += losses["direction"].item()
        total_contrastive += losses["contrastive"].item()
        total_consistency += losses["consistency"].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % config.get("log_every", 100) == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {losses['total'].item():.4f} "
                f"Dir: {losses['direction'].item():.4f} "
                f"Cont: {losses['contrastive'].item():.4f}"
            )

    return {
        "loss": total_loss / num_batches,
        "direction": total_direction / num_batches,
        "contrastive": total_contrastive / num_batches,
        "consistency": total_consistency / num_batches,
    }


def evaluate(
    model: EmoKnob,
    dataloader: DataLoader,
    loss_fn: EmoKnobLoss,
    device: torch.device,
    config: dict,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_direction = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            emotion_idx = batch["emotion_idx"].to(device)
            emotions = batch["emotion"]

            results = []
            for i, emotion in enumerate(emotions):
                result = model(features[i:i+1], emotion=emotion)
                results.append(result)

            speaker_embs = torch.cat([r["speaker_emb"] for r in results], dim=0)
            controlled_embs = torch.cat([r["controlled_emb"] for r in results], dim=0)

            losses = loss_fn(speaker_embs, controlled_embs, emotion_idx)

            total_loss += losses["total"].item()
            total_direction += losses["direction"].item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "direction": total_direction / num_batches,
    }


def train(
    model: EmoKnob,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Full training loop."""
    logger.info("=" * 60)
    logger.info("Phase 2: Training EmoKnob Model")
    logger.info("=" * 60)

    # Loss function
    loss_fn = EmoKnobLoss(
        EmoKnobConfig(**config["model"]),
        direction_weight=config.get("direction_weight", 1.0),
        contrastive_weight=config.get("contrastive_weight", 0.5),
        consistency_weight=config.get("consistency_weight", 0.3),
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
        betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.999)),
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * config.get("epochs", 100)
    warmup_steps = config.get("warmup_steps", 1000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(config.get("checkpoint_dir", "../checkpoints/emoknob"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.get("epochs", 100) + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.get('epochs', 100)}")
        logger.info(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, config
        )

        # Update scheduler
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device, config)

        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save checkpoint
        if epoch % config.get("save_every", 5) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = checkpoint_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, best_path)
            logger.info(f"New best model saved: {best_path}")

    # Save final directions
    model.manipulator.save_directions(config["model"]["direction_cache_path"])
    logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train EmoKnob model")
    parser.add_argument("--config", type=str, default="config/emoknob.yaml",
                       help="Path to config file")
    parser.add_argument("--extract", action="store_true",
                       help="Only extract direction vectors")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str,
                       help="Checkpoint path to resume from")
    parser.add_argument("--test", action="store_true",
                       help="Run with synthetic data for testing")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu/mps)")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {
            "model": asdict(EmoKnobConfig()),
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "warmup_steps": 1000,
            "weight_decay": 0.01,
            "checkpoint_dir": "../checkpoints/emoknob",
            "save_every": 5,
            "log_every": 100,
        }

    # Device (command line overrides config)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Create model
    model_config = EmoKnobConfig(**config.get("model", {}))
    model = EmoKnob(model_config).to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test mode with synthetic data
    if args.test:
        logger.info("Running in test mode with synthetic data")

        # Create synthetic datasets
        train_dataset = SyntheticEmoKnobDataset(num_samples=100, config=model_config)
        val_dataset = SyntheticEmoKnobDataset(num_samples=20, config=model_config)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        # Create synthetic direction vectors
        for emotion in ["happy", "sad", "angry", "surprised", "calm", "fearful"]:
            direction = torch.randn(model_config.speaker_dim)
            direction = F.normalize(direction, p=2, dim=0)
            model.manipulator.register_direction(emotion, direction)

        # Run short training
        config["epochs"] = 2
        config["log_every"] = 10
        train(model, train_loader, val_loader, config, device)

        logger.info("Test training complete!")
        return

    # Load datasets
    train_manifest = config.get("train_manifest", "../data/emotion_train.json")
    val_manifest = config.get("val_manifest", "../data/emotion_val.json")

    if not Path(train_manifest).exists():
        logger.error(f"Training manifest not found: {train_manifest}")
        logger.info("Use --test flag for synthetic data testing")
        return

    train_dataset = EmoKnobDataset(train_manifest, model_config)
    val_dataset = EmoKnobDataset(val_manifest, model_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 16),
        num_workers=config.get("num_workers", 4),
    )

    # Resume from checkpoint
    if args.resume or args.checkpoint:
        checkpoint_path = args.checkpoint or Path(config["checkpoint_dir"]) / "best.pt"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    # Extract-only mode
    if args.extract:
        extract_direction_vectors(model, train_dataset, model_config, device)
        return

    # Full training
    train(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
