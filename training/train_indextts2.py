#!/usr/bin/env python3
"""
Training script for IndexTTS2: 8-Dimensional Emotion Vector Control

Based on arXiv:2506.21619 - "IndexTTS 2: Controllable Emotional Text-to-Speech"

Three-Stage Training Curriculum:
- Stage 1: Pre-training on neutral TTS (establish synthesis capability)
- Stage 2: Emotion classification training (learn emotion representations)
- Stage 3: Full emotional TTS with disentanglement

Usage:
    # Full three-stage training
    python train_indextts2.py --config config/indextts2.yaml

    # Resume from checkpoint
    python train_indextts2.py --config config/indextts2.yaml \
        --resume ../checkpoints/indextts2/stage2_best.pt --stage 3

    # Start from specific stage
    python train_indextts2.py --config config/indextts2.yaml --stage 2

    # Test mode (synthetic data)
    python train_indextts2.py --test
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import yaml

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from indextts2 import (
    IndexTTS2Config,
    IndexTTS2,
    IndexTTS2Loss,
    IndexTTS2Adapter,
    create_emotion_vector,
    EMOTION_LABELS,
    EMOTION_TO_IDX,
)


# =============================================================================
# DATASET
# =============================================================================

class EmotionTTSDataset(Dataset):
    """
    Dataset for IndexTTS2 training.

    Expected manifest format:
    {
        "samples": [
            {
                "audio_path": "path/to/audio.wav",
                "text": "transcription",
                "speaker_id": 0,
                "emotion": "happy",
                "emotion_vector": [0.8, 0.0, ...],  # Optional 8-dim vector
                "duration_tokens": 150,  # Optional
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: IndexTTS2Config,
        feature_extractor=None,
        max_samples: Optional[int] = None,
        is_neutral: bool = False,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.is_neutral = is_neutral

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.samples = manifest.get("samples", [])

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Build speaker mapping
        speaker_ids = set(s.get("speaker_id", 0) for s in self.samples)
        self.speaker_to_idx = {s: i for i, s in enumerate(sorted(speaker_ids))}
        self.num_speakers = len(self.speaker_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get audio features (placeholder - in practice load and extract)
        # Simulate audio features
        seq_len = random.randint(50, 200)
        audio_features = torch.randn(seq_len, self.config.input_dim)

        # Get text (placeholder tokenization)
        text = sample.get("text", "placeholder text")
        text_ids = torch.randint(0, 50000, (min(len(text.split()) * 2, 100),))

        # Get speaker ID
        speaker_id = self.speaker_to_idx.get(sample.get("speaker_id", 0), 0)

        # Get emotion
        if self.is_neutral:
            # Stage 1: neutral only
            emotion_vector = torch.zeros(8)
            emotion_vector[7] = 1.0  # calm
            emotion_idx = 7
        else:
            emotion = sample.get("emotion", "neutral")
            emotion_idx = EMOTION_TO_IDX.get(emotion.lower(), 7)

            if "emotion_vector" in sample:
                emotion_vector = torch.tensor(sample["emotion_vector"])
            else:
                emotion_vector = torch.zeros(8)
                emotion_vector[emotion_idx] = 1.0

        # Get target tokens (placeholder)
        target_duration = sample.get("duration_tokens", random.randint(50, 200))
        target_tokens = torch.randint(0, 4096, (target_duration,))

        return {
            "audio_features": audio_features,
            "text_ids": text_ids,
            "speaker_id": torch.tensor(speaker_id),
            "emotion_vector": emotion_vector,
            "emotion_idx": torch.tensor(emotion_idx),
            "target_tokens": target_tokens,
            "target_duration": torch.tensor(target_duration),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    # Find max lengths
    max_audio_len = max(b["audio_features"].shape[0] for b in batch)
    max_text_len = max(b["text_ids"].shape[0] for b in batch)
    max_token_len = max(b["target_tokens"].shape[0] for b in batch)

    batch_size = len(batch)
    input_dim = batch[0]["audio_features"].shape[1]

    # Pad sequences
    audio_features = torch.zeros(batch_size, max_audio_len, input_dim)
    audio_mask = torch.zeros(batch_size, max_audio_len)
    text_ids = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    text_mask = torch.zeros(batch_size, max_text_len)
    target_tokens = torch.zeros(batch_size, max_token_len, dtype=torch.long)

    speaker_ids = []
    emotion_vectors = []
    emotion_indices = []
    target_durations = []

    for i, b in enumerate(batch):
        # Audio
        audio_len = b["audio_features"].shape[0]
        audio_features[i, :audio_len] = b["audio_features"]
        audio_mask[i, :audio_len] = 1

        # Text
        text_len = b["text_ids"].shape[0]
        text_ids[i, :text_len] = b["text_ids"]
        text_mask[i, :text_len] = 1

        # Tokens
        token_len = b["target_tokens"].shape[0]
        target_tokens[i, :token_len] = b["target_tokens"]

        # Others
        speaker_ids.append(b["speaker_id"])
        emotion_vectors.append(b["emotion_vector"])
        emotion_indices.append(b["emotion_idx"])
        target_durations.append(b["target_duration"])

    return {
        "audio_features": audio_features,
        "audio_mask": audio_mask,
        "text_ids": text_ids,
        "text_mask": text_mask,
        "target_tokens": target_tokens,
        "speaker_ids": torch.stack(speaker_ids),
        "emotion_vectors": torch.stack(emotion_vectors),
        "emotion_indices": torch.stack(emotion_indices),
        "target_durations": torch.stack(target_durations),
    }


# =============================================================================
# TRAINER
# =============================================================================

class IndexTTS2Trainer:
    """Trainer for IndexTTS2 with three-stage curriculum."""

    def __init__(
        self,
        config: IndexTTS2Config,
        train_config: Dict,
        model: Optional[IndexTTS2] = None,
        device: str = "cpu",
    ):
        self.config = config
        self.train_config = train_config
        self.device = device

        # Create model
        if model is None:
            num_speakers = train_config.get("num_speakers", 1000)
            self.model = IndexTTS2(config, num_speakers=num_speakers).to(device)
        else:
            self.model = model.to(device)

        # Loss function
        self.loss_fn = IndexTTS2Loss(
            config,
            tts_weight=train_config.get("loss", {}).get("tts_weight", 1.0),
            emotion_class_weight=train_config.get("loss", {}).get("emotion_class_weight", 0.5),
            disentangle_weight=train_config.get("loss", {}).get("disentangle_weight", 0.1),
            duration_weight=train_config.get("loss", {}).get("duration_weight", 0.3),
        )

        # Optimizer and scheduler (created per stage)
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if train_config.get("training", {}).get("use_amp", True) else None

        # Training state
        self.current_stage = 1
        self.global_step = 0
        self.best_loss = float("inf")

        # Logging
        self.log_dir = Path(train_config.get("logging", {}).get("log_dir", "../logs/indextts2"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Checkpointing
        self.checkpoint_dir = Path(
            train_config.get("training", {}).get("checkpoint_dir", "../checkpoints/indextts2")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def setup_optimizer_for_stage(self, stage: int):
        """Setup optimizer for specific training stage."""
        self.current_stage = stage
        self.model.set_training_stage(stage)

        # Get stage-specific learning rate
        stage_config = self.train_config.get("training", {}).get(f"stage{stage}", {})
        lr = stage_config.get("learning_rate", 1e-4)

        # Create optimizer (only for trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.train_config.get("training", {}).get("weight_decay", 0.01),
        )

        # Create scheduler
        num_epochs = stage_config.get("epochs", 50)
        warmup_steps = self.train_config.get("training", {}).get("warmup_steps", 2000)

        # Simple linear warmup + cosine decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * 1000,  # Approximate steps per epoch
            eta_min=self.train_config.get("training", {}).get("min_lr", 1e-6),
        )

        print(f"Setup optimizer for Stage {stage}:")
        print(f"  Learning rate: {lr}")
        print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_losses = {
            "total": 0.0,
            "tts": 0.0,
            "emotion_class": 0.0,
            "disentangle": 0.0,
            "duration": 0.0,
        }
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            with autocast(enabled=self.scaler is not None):
                model_output = self.model(
                    text_ids=batch["text_ids"],
                    audio_features=batch["audio_features"],
                    emotion_vector=batch["emotion_vectors"],
                    speaker_labels=batch["speaker_ids"],
                    emotion_labels=batch["emotion_indices"],
                    target_tokens=batch["target_tokens"],
                )

                # Compute loss
                losses = self.loss_fn(
                    model_output,
                    batch["target_tokens"],
                    target_emotion_idx=batch["emotion_indices"],
                    target_duration=batch["target_durations"],
                    stage=self.current_stage,
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.get("training", {}).get("max_grad_norm", 1.0),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.get("training", {}).get("max_grad_norm", 1.0),
                )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update GRL scale for stage 3
            if self.current_stage == 3 and self.model.disentanglement is not None:
                grl_warmup = self.train_config.get("training", {}).get("stage3", {}).get(
                    "grl_warmup_epochs", 10
                )
                max_scale = self.train_config.get("loss", {}).get("grl_max_scale", 0.5)
                progress = min(1.0, epoch / grl_warmup)
                self.model.disentanglement.set_grl_scale(progress * max_scale)

            # Accumulate losses
            for k, v in losses.items():
                if k in total_losses:
                    total_losses[k] += v.item()
            num_batches += 1

            self.global_step += 1

            # Log
            log_every = self.train_config.get("training", {}).get("log_every_steps", 100)
            if self.global_step % log_every == 0:
                avg_loss = total_losses["total"] / num_batches
                print(f"  Step {self.global_step}: loss={avg_loss:.4f}")

        # Average losses
        for k in total_losses:
            total_losses[k] /= max(1, num_batches)

        return total_losses

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        total_losses = {
            "total": 0.0,
            "tts": 0.0,
            "emotion_class": 0.0,
            "disentangle": 0.0,
            "duration": 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            model_output = self.model(
                text_ids=batch["text_ids"],
                audio_features=batch["audio_features"],
                emotion_vector=batch["emotion_vectors"],
                speaker_labels=batch["speaker_ids"],
                emotion_labels=batch["emotion_indices"],
                target_tokens=batch["target_tokens"],
            )

            losses = self.loss_fn(
                model_output,
                batch["target_tokens"],
                target_emotion_idx=batch["emotion_indices"],
                target_duration=batch["target_durations"],
                stage=self.current_stage,
            )

            for k, v in losses.items():
                if k in total_losses:
                    total_losses[k] += v.item()
            num_batches += 1

        for k in total_losses:
            total_losses[k] /= max(1, num_batches)

        return total_losses

    def save_checkpoint(self, epoch: int, losses: Dict[str, float], is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "stage": self.current_stage,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "losses": losses,
            "config": self.config.__dict__,
        }

        # Save latest
        path = self.checkpoint_dir / f"stage{self.current_stage}_latest.pt"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / f"stage{self.current_stage}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint: {best_path}")

        # Save periodic
        save_every = self.train_config.get("training", {}).get("save_every_steps", 5000)
        if self.global_step % save_every == 0:
            step_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, step_path)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.current_stage = checkpoint.get("stage", 1)

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from {path}")
        print(f"  Stage: {self.current_stage}, Step: {self.global_step}")

    def train_stage(
        self,
        stage: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Train a specific stage."""
        print(f"\n{'='*60}")
        print(f"Starting Stage {stage} Training")
        print(f"{'='*60}")

        self.setup_optimizer_for_stage(stage)

        stage_config = self.train_config.get("training", {}).get(f"stage{stage}", {})
        num_epochs = stage_config.get("epochs", 50)

        self.best_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            start_time = time.time()

            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"  Train loss: {train_losses['total']:.4f}")

            # Evaluate
            if val_loader is not None:
                val_losses = self.evaluate(val_loader)
                print(f"  Val loss: {val_losses['total']:.4f}")
                current_loss = val_losses["total"]
            else:
                current_loss = train_losses["total"]

            # Checkpoint
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss

            self.save_checkpoint(epoch, train_losses, is_best)

            elapsed = time.time() - start_time
            print(f"  Epoch time: {elapsed:.1f}s")

        print(f"\nStage {stage} complete. Best loss: {self.best_loss:.4f}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        start_stage: int = 1,
    ):
        """Run full three-stage training."""
        for stage in range(start_stage, 4):
            self.train_stage(stage, train_loader, val_loader)


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train IndexTTS2 model")
    parser.add_argument("--config", type=str, default="config/indextts2.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--stage", type=int, default=1,
                       help="Start from specific stage (1, 2, or 3)")
    parser.add_argument("--manifest", type=str, default=None,
                       help="Override manifest path")
    parser.add_argument("--test", action="store_true",
                       help="Run test mode with synthetic data")
    args = parser.parse_args()

    # Test mode
    if args.test:
        print("=" * 60)
        print("IndexTTS2 Training Test Mode")
        print("=" * 60)

        config = IndexTTS2Config()

        # Create synthetic data
        print("\n[Test] Creating synthetic dataset...")
        samples = []
        for i in range(100):
            emotion = random.choice(EMOTION_LABELS)
            samples.append({
                "audio_path": f"audio_{i}.wav",
                "text": f"Test sentence number {i}",
                "speaker_id": i % 10,
                "emotion": emotion,
            })

        # Save temporary manifest
        import tempfile
        manifest_path = tempfile.mktemp(suffix=".json")
        with open(manifest_path, "w") as f:
            json.dump({"samples": samples}, f)

        # Create dataset
        dataset = EmotionTTSDataset(manifest_path, config, max_samples=50)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

        print(f"  Dataset size: {len(dataset)}")
        print(f"  Num speakers: {dataset.num_speakers}")

        # Create trainer
        train_config = {
            "training": {
                "stage1": {"epochs": 2, "learning_rate": 1e-4},
                "stage2": {"epochs": 2, "learning_rate": 5e-5},
                "stage3": {"epochs": 2, "learning_rate": 1e-5, "grl_warmup_epochs": 1},
                "warmup_steps": 10,
                "max_grad_norm": 1.0,
                "use_amp": False,
                "log_every_steps": 5,
            },
            "loss": {
                "tts_weight": 1.0,
                "emotion_class_weight": 0.5,
                "disentangle_weight": 0.1,
                "duration_weight": 0.3,
            },
            "num_speakers": dataset.num_speakers,
            "logging": {"log_dir": "/tmp/indextts2_test"},
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            device = "mps"

        print(f"\n[Test] Creating trainer (device: {device})...")
        trainer = IndexTTS2Trainer(config, train_config, device=device)

        print("\n[Test] Running mini training...")
        try:
            trainer.train(loader, val_loader=None, start_stage=1)
            print("\n[Test] Training completed successfully!")
        except Exception as e:
            print(f"\n[Test] Training failed: {e}")
            raise

        # Test adapter
        print("\n[Test] Testing IndexTTS2Adapter...")
        adapter = IndexTTS2Adapter(config).to(device)

        result = adapter.from_profile("happy", intensity=0.8)
        print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")

        result = adapter.from_emotions(happy=0.6, surprised=0.3)
        print(f"  From emotions - dominant: {EMOTION_LABELS[result['dominant_emotion'].item()]}")

        # Cleanup
        os.remove(manifest_path)

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return

    # Load config
    if os.path.exists(args.config):
        train_config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
        train_config = {}

    # Create model config
    model_cfg = train_config.get("model", {})
    config = IndexTTS2Config(
        input_dim=model_cfg.get("input_dim", 768),
        emotion_dim=model_cfg.get("emotion_dim", 256),
        timbre_dim=model_cfg.get("timbre_dim", 256),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        output_dim=model_cfg.get("output_dim", 2048),
        gpt_hidden_dim=model_cfg.get("gpt_hidden_dim", 1024),
        gpt_num_layers=model_cfg.get("gpt_num_layers", 12),
        dropout=model_cfg.get("dropout", 0.1),
        enable_duration_control=model_cfg.get("enable_duration_control", True),
        use_soft_instruction=model_cfg.get("use_soft_instruction", True),
        use_adversarial_disentanglement=model_cfg.get("use_adversarial_disentanglement", True),
    )

    # Get manifest path
    manifest_path = args.manifest or train_config.get("training", {}).get(
        "manifest_path", "../data/emotion_manifest.json"
    )

    # Check manifest exists
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        print("Please provide a manifest file with --manifest or in config.")
        sys.exit(1)

    # Create datasets
    print("Loading datasets...")
    train_dataset = EmotionTTSDataset(manifest_path, config)
    val_dataset = EmotionTTSDataset(manifest_path, config, max_samples=100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("training", {}).get("batch_size", 16),
        shuffle=True,
        num_workers=train_config.get("hardware", {}).get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=train_config.get("hardware", {}).get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("training", {}).get("batch_size", 16),
        shuffle=False,
        num_workers=train_config.get("hardware", {}).get("num_workers", 4),
        collate_fn=collate_fn,
    )

    # Determine device
    device = train_config.get("hardware", {}).get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"Using device: {device}")

    # Update config with speaker count
    train_config["num_speakers"] = train_dataset.num_speakers

    # Create trainer
    trainer = IndexTTS2Trainer(config, train_config, device=device)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Run training
    trainer.train(train_loader, val_loader, start_stage=args.stage)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
