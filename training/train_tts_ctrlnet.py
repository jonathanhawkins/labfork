#!/usr/bin/env python3
"""
Training script for TTS-CtrlNet: ControlNet-style Emotion Control

Based on TTS-CtrlNet (arXiv:2507.04349). Trains emotion control on top
of a pretrained ProsodyFlow model without degrading base model quality.

Training stages:
1. Load pretrained ProsodyFlow base model (frozen)
2. Initialize ControlNet control branch
3. Train only control branch on emotion-labeled data
4. Fine-tune with emotion-specific flow steps

Usage:
    # Train TTS-CtrlNet with pretrained base
    python train_tts_ctrlnet.py --config config/tts_ctrlnet.yaml \
        --base-checkpoint ../checkpoints/prosody_flow/best.pt

    # Resume training
    python train_tts_ctrlnet.py --config config/tts_ctrlnet.yaml \
        --resume ../checkpoints/tts_ctrlnet/latest.pt

    # Test mode
    python train_tts_ctrlnet.py --test
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.tts_ctrlnet import (
    TTSCtrlNetConfig,
    TTSCtrlNet,
    TTSCtrlNetAdapter,
    TTSCtrlNetLoss,
    EMOTION_TO_IDX,
    IDX_TO_EMOTION,
    VAD_PROTOTYPES,
)
from training.prosody_flow import (
    ProsodyFlowConfig,
    ProsodyFlow,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for TTS-CtrlNet."""

    # Model paths
    base_checkpoint: str = ""  # Path to pretrained ProsodyFlow
    output_dir: str = "../checkpoints/tts_ctrlnet"

    # Data
    manifest_path: str = "../data/emotion_manifest.json"
    val_split: float = 0.1

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0

    # Scheduler
    scheduler_type: str = "cosine"  # cosine, linear, constant

    # Control scale curriculum
    use_control_curriculum: bool = True
    control_scale_start: float = 0.1
    control_scale_end: float = 1.0
    control_curriculum_epochs: int = 20

    # Logging
    log_interval: int = 100
    val_interval: int = 500
    save_interval: int = 1000
    use_wandb: bool = False
    wandb_project: str = "tts-ctrlnet"

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = True


# =============================================================================
# DATASET
# =============================================================================

class EmotionProsodyDataset(Dataset):
    """
    Dataset for emotion-labeled prosody training.

    Expected manifest format:
    [
        {
            "audio_path": "path/to/audio.wav",
            "text": "transcript text",
            "emotion": "happy",
            "vad": [0.8, 0.6, 0.6],  # optional
            "intensity": 0.8,  # optional
            "prosody_features": {...}  # optional preprocessed features
        },
        ...
    ]
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_dim: int = 2048,
        text_dim: int = 768,
        text_seq_len: int = 20,
        split: str = "train",
        val_ratio: float = 0.1,
    ):
        self.prosody_dim = prosody_dim
        self.text_dim = text_dim
        self.text_seq_len = text_seq_len

        # Load manifest
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.data = json.load(f)
        else:
            logger.warning(f"Manifest not found: {manifest_path}, using synthetic data")
            self.data = self._create_synthetic_data()

        # Split data
        split_idx = int(len(self.data) * (1 - val_ratio))
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        logger.info(f"Loaded {len(self.data)} samples for {split}")

    def _create_synthetic_data(self, num_samples: int = 1000) -> List[Dict]:
        """Create synthetic data for testing."""
        emotions = list(EMOTION_TO_IDX.keys())
        data = []

        for i in range(num_samples):
            emotion = emotions[i % len(emotions)]
            vad = list(VAD_PROTOTYPES[emotion])

            data.append({
                "audio_path": f"synthetic_{i}.wav",
                "text": f"This is sample {i} with emotion {emotion}",
                "emotion": emotion,
                "vad": vad,
                "intensity": 0.5 + 0.5 * (i % 10) / 10,
            })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Get emotion label
        emotion = item.get("emotion", "neutral")
        emotion_idx = EMOTION_TO_IDX.get(emotion, 0)

        # Get VAD coordinates
        if "vad" in item:
            vad = torch.tensor(item["vad"], dtype=torch.float32)
        else:
            vad = torch.tensor(VAD_PROTOTYPES[emotion], dtype=torch.float32)

        # Get intensity
        intensity = item.get("intensity", 0.8)

        # Get prosody features (or create synthetic)
        if "prosody_features" in item:
            prosody = torch.tensor(item["prosody_features"], dtype=torch.float32)
        else:
            # Create synthetic prosody conditioned on emotion
            prosody = self._generate_synthetic_prosody(emotion, vad, intensity)

        # Get text embeddings (or create synthetic)
        if "text_embeddings" in item:
            text_emb = torch.tensor(item["text_embeddings"], dtype=torch.float32)
        else:
            text_emb = torch.randn(self.text_seq_len, self.text_dim)

        return {
            "prosody": prosody,
            "text_embeddings": text_emb,
            "emotion_label": torch.tensor(emotion_idx, dtype=torch.long),
            "vad": vad,
            "intensity": torch.tensor(intensity, dtype=torch.float32),
        }

    def _generate_synthetic_prosody(
        self,
        emotion: str,
        vad: torch.Tensor,
        intensity: float,
    ) -> torch.Tensor:
        """Generate synthetic prosody features conditioned on emotion."""
        # Base prosody (random)
        prosody = torch.randn(self.prosody_dim) * 0.5

        # Add emotion-specific bias based on VAD
        # This creates prosody that has structure based on emotion
        emotion_bias = torch.zeros(self.prosody_dim)

        # Map VAD to prosody dimensions
        # Valence -> first quarter
        emotion_bias[:self.prosody_dim // 4] = vad[0] * intensity

        # Arousal -> second quarter
        emotion_bias[self.prosody_dim // 4:self.prosody_dim // 2] = vad[1] * intensity

        # Dominance -> third quarter
        emotion_bias[self.prosody_dim // 2:3 * self.prosody_dim // 4] = vad[2] * intensity

        prosody = prosody + emotion_bias

        return prosody


# =============================================================================
# TRAINING LOOP
# =============================================================================

class TTSCtrlNetTrainer:
    """Trainer for TTS-CtrlNet model."""

    def __init__(
        self,
        model: TTSCtrlNet,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainingConfig,
        ctrlnet_config: TTSCtrlNetConfig,
    ):
        self.model = model
        self.config = config
        self.ctrlnet_config = ctrlnet_config

        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        if config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
            )
        elif config.scheduler_type == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
        else:
            self.scheduler = None

        # Loss function
        self.loss_fn = TTSCtrlNetLoss()

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_control_scale(self, epoch: int) -> float:
        """Get control scale based on curriculum."""
        if not self.config.use_control_curriculum:
            return 1.0

        if epoch >= self.config.control_curriculum_epochs:
            return self.config.control_scale_end

        progress = epoch / self.config.control_curriculum_epochs
        return self.config.control_scale_start + (
            self.config.control_scale_end - self.config.control_scale_start
        ) * progress

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        control_scale = self.get_control_scale(epoch)
        logger.info(f"Epoch {epoch + 1}: control_scale = {control_scale:.3f}")

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            prosody = batch["prosody"].to(self.device)
            text_emb = batch["text_embeddings"].to(self.device)
            emotion_labels = batch["emotion_label"].to(self.device)
            vad = batch["vad"].to(self.device)
            intensity = batch["intensity"].to(self.device)

            # Forward pass
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss_output = self.model.compute_loss(
                        prosody, text_emb,
                        emotion_labels=emotion_labels,
                        vad_coords=vad,
                        intensity=intensity,
                        control_scale=control_scale,
                    )
                    loss = loss_output['loss']

                # Backward pass with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_output = self.model.compute_loss(
                    prosody, text_emb,
                    emotion_labels=emotion_labels,
                    vad_coords=vad,
                    intensity=intensity,
                    control_scale=control_scale,
                )
                loss = loss_output['loss']

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.gradient_clip
                )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Step {self.global_step}: loss={avg_loss:.4f}, "
                    f"lr={lr:.2e}, control_scale={control_scale:.3f}"
                )

            # Validation
            if self.global_step % self.config.val_interval == 0:
                val_loss = self.validate()
                logger.info(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint("latest.pt")
                self.save_checkpoint(f"step_{self.global_step}.pt")

        return {
            "loss": total_loss / num_batches,
            "control_scale": control_scale,
        }

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            prosody = batch["prosody"].to(self.device)
            text_emb = batch["text_embeddings"].to(self.device)
            emotion_labels = batch["emotion_label"].to(self.device)
            vad = batch["vad"].to(self.device)
            intensity = batch["intensity"].to(self.device)

            loss_output = self.model.compute_loss(
                prosody, text_emb,
                emotion_labels=emotion_labels,
                vad_coords=vad,
                intensity=intensity,
                control_scale=1.0,
            )

            total_loss += loss_output['loss'].item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": {
                "training": vars(self.config),
                "ctrlnet": vars(self.ctrlnet_config),
            }
        }
        torch.save(checkpoint, self.output_dir / filename)
        logger.info(f"Saved checkpoint to {self.output_dir / filename}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        logger.info(f"Loaded checkpoint from {path}, step {self.global_step}")

    def train(self):
        """Main training loop."""
        logger.info("Starting TTS-CtrlNet training...")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        logger.info(f"Frozen parameters: {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")

            metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} metrics: {metrics}")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Tuple[TrainingConfig, TTSCtrlNetConfig]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    training_cfg = TrainingConfig(**cfg.get("training", {}))

    # Base config
    base_config = ProsodyFlowConfig(**cfg.get("base_model", {}))

    # CtrlNet config
    ctrlnet_cfg_dict = cfg.get("ctrlnet", {})
    ctrlnet_cfg_dict["base_config"] = base_config
    ctrlnet_cfg = TTSCtrlNetConfig(**ctrlnet_cfg_dict)

    return training_cfg, ctrlnet_cfg


def main():
    parser = argparse.ArgumentParser(description="Train TTS-CtrlNet")
    parser.add_argument("--config", type=str, default="config/tts_ctrlnet.yaml",
                        help="Path to config file")
    parser.add_argument("--base-checkpoint", type=str, default="",
                        help="Path to pretrained ProsodyFlow checkpoint")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to resume training from")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with synthetic data")
    args = parser.parse_args()

    if args.test:
        # Quick test mode
        logger.info("Running in test mode...")

        # Create small configs
        base_config = ProsodyFlowConfig(
            prosody_dim=256,
            hidden_dim=128,
            text_dim=128,
            num_layers=2,
            num_heads=4,
        )

        ctrlnet_config = TTSCtrlNetConfig(
            base_config=base_config,
            emotion_dim=64,
        )

        training_config = TrainingConfig(
            batch_size=4,
            num_epochs=2,
            log_interval=10,
            val_interval=50,
            save_interval=100,
            output_dir="../checkpoints/tts_ctrlnet_test",
        )

        # Create model
        base_flow = ProsodyFlow(base_config)
        model = TTSCtrlNet(ctrlnet_config, base_flow)

        # Create datasets
        train_dataset = EmotionProsodyDataset(
            manifest_path="",
            prosody_dim=base_config.prosody_dim,
            text_dim=base_config.text_dim,
            split="train",
        )

        val_dataset = EmotionProsodyDataset(
            manifest_path="",
            prosody_dim=base_config.prosody_dim,
            text_dim=base_config.text_dim,
            split="val",
        )

        # Create trainer
        trainer = TTSCtrlNetTrainer(
            model, train_dataset, val_dataset,
            training_config, ctrlnet_config
        )

        # Run one epoch
        trainer.train_epoch(0)
        logger.info("Test mode complete!")
        return

    # Load config
    if os.path.exists(args.config):
        training_config, ctrlnet_config = load_config(args.config)
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
        base_config = ProsodyFlowConfig()
        ctrlnet_config = TTSCtrlNetConfig(base_config=base_config)
        training_config = TrainingConfig()

    # Override base checkpoint if provided
    if args.base_checkpoint:
        training_config.base_checkpoint = args.base_checkpoint

    # Create base model
    base_flow = ProsodyFlow(ctrlnet_config.base_config)

    # Load pretrained weights if available
    if training_config.base_checkpoint and os.path.exists(training_config.base_checkpoint):
        logger.info(f"Loading base model from {training_config.base_checkpoint}")
        checkpoint = torch.load(training_config.base_checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            base_flow.load_state_dict(checkpoint["model_state_dict"])
        else:
            base_flow.load_state_dict(checkpoint)

    # Create TTS-CtrlNet model
    model = TTSCtrlNet(ctrlnet_config, base_flow)

    # Create datasets
    train_dataset = EmotionProsodyDataset(
        manifest_path=training_config.manifest_path,
        prosody_dim=ctrlnet_config.base_config.prosody_dim,
        text_dim=ctrlnet_config.base_config.text_dim,
        split="train",
        val_ratio=training_config.val_split,
    )

    val_dataset = EmotionProsodyDataset(
        manifest_path=training_config.manifest_path,
        prosody_dim=ctrlnet_config.base_config.prosody_dim,
        text_dim=ctrlnet_config.base_config.text_dim,
        split="val",
        val_ratio=training_config.val_split,
    )

    # Create trainer
    trainer = TTSCtrlNetTrainer(
        model, train_dataset, val_dataset,
        training_config, ctrlnet_config
    )

    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
