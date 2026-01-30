"""
Training script for DrawSpeech: Sketch-Conditioned Prosody Generation

Based on DrawSpeech (ICASSP 2025) - trains a model to generate prosody
from user-drawn pitch/energy curves.

Usage:
    python train_draw_speech.py --config config/draw_speech.yaml

Training data requirements:
    - Audio files with emotion labels
    - Extracted prosody features (from prosody_analyzer)
    - Ground truth sketches are automatically derived from prosody contours
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import yaml
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from draw_speech import (
    DrawSpeechConfig,
    DrawSpeech,
    SketchConfig,
    prosody_to_sketch,
    sketch_from_emotion_profile,
)
from prosody_flow import ProsodyFlowConfig
from prosody_conditioning import extract_prosody_for_conditioning, ProsodyConfig


# =============================================================================
# DATASET
# =============================================================================

class DrawSpeechDataset(Dataset):
    """
    Dataset for DrawSpeech training.

    Loads audio samples with prosody annotations and creates:
    - Target prosody vectors (from prosody encoder or pre-extracted)
    - Ground truth sketches (derived from pitch contour and energy)
    - Text embeddings (from pre-trained text encoder)

    Supports augmentation:
    - Sketch noise injection
    - Sketch smoothing variation
    - Sketch shift (pitch/energy offset)
    """

    def __init__(
        self,
        manifest_path: str,
        config: dict,
        split: str = "train",
        text_encoder: Optional[nn.Module] = None,
    ):
        self.config = config
        self.split = split
        self.text_encoder = text_encoder

        # Data settings
        data_config = config.get("data", {})
        self.max_audio_length = data_config.get("max_audio_length", 15.0)
        self.min_audio_length = data_config.get("min_audio_length", 0.5)
        self.sample_rate = data_config.get("sample_rate", 24000)

        # Sketch settings
        sketch_config = config.get("sketch", {})
        self.sketch_length = sketch_config.get("sketch_length", 100)
        self.smooth_sigma = sketch_config.get("smooth_sigma", 2.0)

        # Augmentation
        self.augment = split == "train" and data_config.get("augment_sketches", True)
        self.noise_std = data_config.get("sketch_noise_std", 0.05)
        self.smooth_range = data_config.get("sketch_smooth_range", [1.0, 4.0])
        self.shift_range = data_config.get("sketch_shift_range", [-0.1, 0.1])

        # Load manifest
        self.samples = self._load_manifest(manifest_path, split)
        print(f"Loaded {len(self.samples)} samples for {split}")

    def _load_manifest(self, manifest_path: str, split: str) -> List[dict]:
        """Load and filter manifest for split."""
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        samples = []
        for entry in manifest:
            # Filter by split if specified
            entry_split = entry.get("split", "train")
            if entry_split != split:
                continue

            # Filter by duration
            duration = entry.get("duration", 0)
            if duration < self.min_audio_length or duration > self.max_audio_length:
                continue

            # Must have prosody or audio path
            if not entry.get("prosody") and not entry.get("audio_path"):
                continue

            samples.append(entry)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get prosody features
        if "prosody" in sample:
            prosody_dict = sample["prosody"]
        else:
            # Extract prosody from audio (fallback, slower)
            prosody_dict = self._extract_prosody(sample["audio_path"])

        # Convert to conditioning format
        prosody_config = ProsodyConfig()
        conditioning = extract_prosody_for_conditioning(prosody_dict, prosody_config)

        # Create target prosody vector (concatenated features)
        prosody_target = torch.cat([
            conditioning["semantic"].squeeze(0),
            conditioning["acoustic"].squeeze(0),
            conditioning["rhythm"].squeeze(0),
            conditioning["contour"].squeeze(0),
        ], dim=-1)

        # Pad/expand to prosody_dim
        flow_config = self.config.get("flow", {})
        prosody_dim = flow_config.get("prosody_dim", 2048)
        if prosody_target.shape[-1] < prosody_dim:
            padding = torch.zeros(prosody_dim - prosody_target.shape[-1])
            prosody_target = torch.cat([prosody_target, padding], dim=-1)

        # Create ground truth sketches from prosody
        pitch_sketch, energy_sketch = prosody_to_sketch(
            conditioning,
            sketch_length=self.sketch_length,
        )

        # Apply augmentation
        if self.augment:
            pitch_sketch, energy_sketch = self._augment_sketches(
                pitch_sketch, energy_sketch
            )

        # Get text
        text = sample.get("text", sample.get("transcript", ""))

        # Get text embeddings
        if self.text_encoder is not None:
            with torch.no_grad():
                text_emb = self.text_encoder(text)
        else:
            # Placeholder - will be computed during training
            text_dim = flow_config.get("text_dim", 768)
            text_emb = torch.randn(10, text_dim)  # [seq_len, text_dim]

        return {
            "prosody_target": prosody_target,
            "pitch_sketch": pitch_sketch.squeeze(0),
            "energy_sketch": energy_sketch.squeeze(0),
            "text_emb": text_emb,
            "text": text,
            "emotion": sample.get("emotion", "neutral"),
        }

    def _extract_prosody(self, audio_path: str) -> dict:
        """Extract prosody from audio file."""
        # Import analyzer
        sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
        from prosody_analyzer import CompleteProsodyAnalyzer

        analyzer = CompleteProsodyAnalyzer()
        return analyzer.analyze(audio_path).to_dict()

    def _augment_sketches(
        self,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to sketches."""
        # Add noise
        if self.noise_std > 0:
            pitch_sketch = pitch_sketch + torch.randn_like(pitch_sketch) * self.noise_std
            energy_sketch = energy_sketch + torch.randn_like(energy_sketch) * self.noise_std

        # Random smoothing
        if self.smooth_range[0] != self.smooth_range[1]:
            sigma = random.uniform(*self.smooth_range)
            pitch_sketch = self._smooth_curve(pitch_sketch, sigma)
            energy_sketch = self._smooth_curve(energy_sketch, sigma)

        # Random shift
        if self.shift_range[0] != self.shift_range[1]:
            pitch_shift = random.uniform(*self.shift_range)
            energy_shift = random.uniform(*self.shift_range)
            pitch_sketch = pitch_sketch + pitch_shift
            energy_sketch = energy_sketch + energy_shift

        # Clamp to valid range
        pitch_sketch = torch.clamp(pitch_sketch, 0, 1)
        energy_sketch = torch.clamp(energy_sketch, 0, 1)

        return pitch_sketch, energy_sketch

    def _smooth_curve(self, curve: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        if sigma <= 0:
            return curve

        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=curve.dtype, device=curve.device)
        x = x - kernel_size // 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        # Apply convolution
        curve_expanded = curve.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K]
        padding = kernel_size // 2

        smoothed = F.conv1d(curve_expanded, kernel_expanded, padding=padding)
        return smoothed.squeeze(0).squeeze(0)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length text."""
    prosody_target = torch.stack([b["prosody_target"] for b in batch])
    pitch_sketch = torch.stack([b["pitch_sketch"] for b in batch])
    energy_sketch = torch.stack([b["energy_sketch"] for b in batch])

    # Pad text embeddings
    max_len = max(b["text_emb"].shape[0] for b in batch)
    text_dim = batch[0]["text_emb"].shape[-1]

    text_emb = torch.zeros(len(batch), max_len, text_dim)
    text_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        seq_len = b["text_emb"].shape[0]
        text_emb[i, :seq_len] = b["text_emb"]
        text_mask[i, :seq_len] = True

    return {
        "prosody_target": prosody_target,
        "pitch_sketch": pitch_sketch,
        "energy_sketch": energy_sketch,
        "text_emb": text_emb,
        "text_mask": text_mask,
        "texts": [b["text"] for b in batch],
        "emotions": [b["emotion"] for b in batch],
    }


# =============================================================================
# TRAINER
# =============================================================================

class DrawSpeechTrainer:
    """Trainer for DrawSpeech model."""

    def __init__(
        self,
        config: dict,
        model: DrawSpeech,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training settings
        train_config = config.get("training", {})
        self.batch_size = train_config.get("batch_size", 16)
        self.gradient_accumulation = train_config.get("gradient_accumulation", 1)
        self.learning_rate = train_config.get("learning_rate", 1e-4)
        self.weight_decay = train_config.get("weight_decay", 0.01)
        self.warmup_steps = train_config.get("warmup_steps", 500)
        self.max_steps = train_config.get("max_steps", 50000)
        self.eval_interval = train_config.get("eval_interval", 500)
        self.save_interval = train_config.get("save_interval", 2000)
        self.log_interval = train_config.get("log_interval", 100)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)

        # Device
        device_config = config.get("device", "auto")
        if device_config == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_config)

        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Paths
        paths_config = config.get("paths", {})
        self.checkpoint_dir = Path(paths_config.get("checkpoint_dir", "../checkpoints/draw_speech"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

        # WandB
        wandb_config = config.get("wandb", {})
        self.use_wandb = wandb_config.get("project") is not None
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_config.get("project", "draw-speech"),
                    name=wandb_config.get("name", "draw_speech_v1"),
                    tags=wandb_config.get("tags", []),
                    config=config,
                )
            except ImportError:
                print("WandB not available, logging disabled")
                self.use_wandb = False

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        train_config = self.config.get("training", {})
        schedule = train_config.get("lr_schedule", "cosine")
        lr_min = train_config.get("lr_min", 1e-6)

        if schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps - self.warmup_steps,
                eta_min=lr_min,
            )
        elif schedule == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=lr_min / self.learning_rate,
                total_iters=self.max_steps - self.warmup_steps,
            )
        else:
            return None

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_steps} steps")

        self.model.train()
        accumulation_counter = 0
        accumulated_loss = 0.0

        pbar = tqdm(total=self.max_steps, desc="Training")

        while self.global_step < self.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.max_steps:
                    break

                # Move to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                loss_output = self.model.compute_loss(
                    batch["prosody_target"],
                    batch["pitch_sketch"],
                    batch["energy_sketch"],
                    batch["text_emb"],
                    batch["text_mask"],
                )

                loss = loss_output["loss"] / self.gradient_accumulation
                loss.backward()

                accumulated_loss += loss.item()
                accumulation_counter += 1

                # Gradient step
                if accumulation_counter >= self.gradient_accumulation:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Learning rate warmup
                    if self.global_step < self.warmup_steps:
                        warmup_factor = self.global_step / self.warmup_steps
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate * warmup_factor
                    elif self.scheduler is not None:
                        self.scheduler.step()

                    self.global_step += 1
                    pbar.update(1)

                    # Logging
                    if self.global_step % self.log_interval == 0:
                        avg_loss = accumulated_loss / accumulation_counter
                        lr = self.optimizer.param_groups[0]["lr"]

                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })

                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/step": self.global_step,
                            })

                    accumulated_loss = 0.0
                    accumulation_counter = 0

                    # Validation
                    if self.global_step % self.eval_interval == 0:
                        if self.val_loader is not None:
                            val_loss = self.validate()
                            print(f"\nStep {self.global_step}: val_loss = {val_loss:.4f}")

                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_checkpoint("best.pt")

                    # Save checkpoint
                    if self.global_step % self.save_interval == 0:
                        self.save_checkpoint(f"step_{self.global_step}.pt")

        pbar.close()
        self.save_checkpoint("final.pt")
        print("Training complete!")

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            loss_output = self.model.compute_loss(
                batch["prosody_target"],
                batch["pitch_sketch"],
                batch["energy_sketch"],
                batch["text_emb"],
                batch["text_mask"],
            )

            total_loss += loss_output["loss"].item()
            num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)

        if self.use_wandb:
            import wandb
            wandb.log({
                "val/loss": avg_loss,
                "val/step": self.global_step,
            })

        return avg_loss

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from step {self.global_step}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train DrawSpeech model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/draw_speech.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override manifest path",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override manifest if provided
    if args.manifest:
        config["data"]["manifest"] = args.manifest

    # Create model config
    sketch_config = SketchConfig(**config.get("sketch", {}))
    flow_config = ProsodyFlowConfig(**config.get("flow", {}))

    model_config = DrawSpeechConfig(
        sketch=sketch_config,
        flow=flow_config,
        sketch_conditioning=config.get("sketch_conditioning", "cross_attention"),
        sketch_dropout=config.get("sketch_dropout", 0.1),
        use_cfg=config.get("use_cfg", True),
        cfg_scale=config.get("cfg_scale", 2.0),
        cfg_dropout=config.get("cfg_dropout", 0.1),
        combine_text_sketch=config.get("combine_text_sketch", True),
    )

    # Create model
    model = DrawSpeech(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    manifest_path = config["data"]["manifest"]

    train_dataset = DrawSpeechDataset(manifest_path, config, split="train")
    val_dataset = DrawSpeechDataset(manifest_path, config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    ) if len(val_dataset) > 0 else None

    # Create trainer
    trainer = DrawSpeechTrainer(config, model, train_loader, val_loader)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
