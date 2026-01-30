"""
Training Script for Soft Speech Units

Trains the soft speech units model for prosody-preserving voice conversion.

Usage:
    python train_soft_speech_units.py --config config/soft_speech_units.yaml
    python train_soft_speech_units.py --config config/soft_speech_units.yaml --resume ../checkpoints/soft_speech_units/best.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "training"))

from soft_speech_units import (
    SoftSpeechUnitsConfig,
    SoftSpeechUnitsModel,
    SoftSpeechUnitsAdapter,
)


# =============================================================================
# DATASET
# =============================================================================

class SoftUnitsDataset(Dataset):
    """
    Dataset for soft speech units training.

    Loads audio files and mel spectrograms for reconstruction training.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        max_duration: float = 15.0,
        mel_dim: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.mel_dim = mel_dim
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Load manifest
        with open(manifest_path, "r") as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=mel_dim,
        )

        # Build speaker to ID mapping
        speakers = set()
        for sample in self.samples:
            speaker = sample.get("speaker_id") or sample.get("speaker") or "default"
            speakers.add(speaker)
        self.speaker_to_id = {s: i for i, s in enumerate(sorted(speakers))}
        print(f"Found {len(self.speaker_to_id)} unique speakers")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get audio path
        audio_path = sample.get("audio_path") or sample.get("path")
        if not os.path.isabs(audio_path):
            # Relative to manifest
            manifest_dir = Path(self.samples[idx].get("_manifest_dir", "."))
            audio_path = manifest_dir / audio_path

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)  # [samples]

        # Truncate if too long
        max_samples = int(self.max_duration * self.sample_rate)
        if audio.shape[0] > max_samples:
            start = torch.randint(0, audio.shape[0] - max_samples, (1,)).item()
            audio = audio[start : start + max_samples]

        # Compute mel spectrogram
        mel = self.mel_transform(audio)  # [mel_dim, frames]
        mel = mel.transpose(0, 1)  # [frames, mel_dim]

        # Log mel
        mel = torch.log(mel.clamp(min=1e-5))

        # Get speaker ID
        speaker = sample.get("speaker_id") or sample.get("speaker") or "default"
        speaker_id = self.speaker_to_id[speaker]

        return {
            "audio": audio,
            "mel": mel,
            "speaker_id": torch.tensor(speaker_id),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Find max lengths
    max_audio_len = max(b["audio"].shape[0] for b in batch)
    max_mel_len = max(b["mel"].shape[0] for b in batch)

    # Pad audio
    audios = []
    audio_masks = []
    for b in batch:
        audio = b["audio"]
        pad_len = max_audio_len - audio.shape[0]
        audios.append(F.pad(audio, (0, pad_len)))
        mask = torch.ones(audio.shape[0])
        mask = F.pad(mask, (0, pad_len))
        audio_masks.append(mask)

    # Pad mel
    mels = []
    mel_masks = []
    mel_dim = batch[0]["mel"].shape[1]
    for b in batch:
        mel = b["mel"]
        pad_len = max_mel_len - mel.shape[0]
        mels.append(F.pad(mel, (0, 0, 0, pad_len)))
        mask = torch.ones(mel.shape[0])
        mask = F.pad(mask, (0, pad_len))
        mel_masks.append(mask)

    return {
        "audio": torch.stack(audios),
        "audio_mask": torch.stack(audio_masks),
        "mel": torch.stack(mels),
        "mel_mask": torch.stack(mel_masks),
        "speaker_id": torch.stack([b["speaker_id"] for b in batch]),
    }


# =============================================================================
# TRAINER
# =============================================================================

class SoftUnitsTrainer:
    """Trainer for soft speech units model."""

    def __init__(
        self,
        config: dict,
        model: SoftSpeechUnitsModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Training config
        train_config = config.get("training", {})

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.get("learning_rate", 1e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
            betas=train_config.get("betas", (0.9, 0.98)),
        )

        # Scheduler
        total_steps = len(train_loader) * train_config.get("epochs", 50)
        warmup_steps = len(train_loader) * train_config.get("warmup_epochs", 3)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=train_config.get("learning_rate", 1e-4),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
            final_div_factor=train_config.get("learning_rate", 1e-4) / train_config.get("min_lr", 1e-6),
        )

        # Loss weights
        self.l1_weight = train_config.get("l1_weight", 1.0)
        self.l2_weight = train_config.get("l2_weight", 1.0)
        self.entropy_weight = train_config.get("entropy_weight", 0.01)

        # Gradient clipping
        self.gradient_clip = train_config.get("gradient_clip", 1.0)

        # Checkpointing
        ckpt_config = config.get("checkpoints", {})
        self.save_dir = Path(ckpt_config.get("save_dir", "../checkpoints/soft_speech_units"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = ckpt_config.get("save_every_n_epochs", 5)
        self.keep_top_k = ckpt_config.get("keep_top_k", 3)

        # Best tracking
        self.best_val_loss = float("inf")
        self.best_checkpoints = []

        # FP16
        self.use_fp16 = config.get("hardware", {}).get("fp16", True)
        self.scaler = torch.amp.GradScaler("cuda") if self.use_fp16 and device == "cuda" else None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        total_entropy = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            audio = batch["audio"].to(self.device)
            mel = batch["mel"].to(self.device)
            speaker_id = batch["speaker_id"].to(self.device)
            mel_mask = batch["mel_mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            with torch.amp.autocast("cuda", enabled=self.use_fp16 and self.device == "cuda"):
                output = self.model(audio, speaker_id)

                # Compute loss
                losses = self.model.compute_loss(
                    mel,
                    output["mel_reconstructed"],
                    output["soft_probs"],
                    mel_mask,
                )

                loss = (
                    self.l1_weight * losses["l1_reconstruction"]
                    + self.l2_weight * losses["l2_reconstruction"]
                    + self.entropy_weight * losses["entropy_loss"]
                )

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Track
            total_loss += loss.item()
            total_l1 += losses["l1_reconstruction"].item()
            total_l2 += losses["l2_reconstruction"].item()
            total_entropy += losses["entropy_loss"].item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "l1_loss": total_l1 / num_batches,
            "l2_loss": total_l2 / num_batches,
            "entropy_loss": total_entropy / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device)
            mel = batch["mel"].to(self.device)
            speaker_id = batch["speaker_id"].to(self.device)
            mel_mask = batch["mel_mask"].to(self.device)

            output = self.model(audio, speaker_id)
            losses = self.model.compute_loss(
                mel,
                output["mel_reconstructed"],
                output["soft_probs"],
                mel_mask,
            )

            loss = losses["l1_reconstruction"] + losses["l2_reconstruction"]
            total_loss += loss.item()
            total_l1 += losses["l1_reconstruction"].item()
            total_l2 += losses["l2_reconstruction"].item()
            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_l1_loss": total_l1 / num_batches,
            "val_l2_loss": total_l2 / num_batches,
        }

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint."""
        ckpt_path = self.save_dir / f"epoch_{epoch}.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }, ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}")

        # Track best
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.save_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "config": self.config,
            }, best_path)
            print(f"New best model saved: {best_path}")

    def train(self, epochs: int, resume_from: Optional[str] = None):
        """Full training loop."""
        start_epoch = 0

        # Resume if specified
        if resume_from is not None:
            print(f"Resuming from {resume_from}")
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1

        print(f"Starting training from epoch {start_epoch}")

        for epoch in range(start_epoch, epochs):
            print(f"\n{'=' * 40}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 40}")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"L1: {train_metrics['l1_loss']:.4f}, "
                  f"L2: {train_metrics['l2_loss']:.4f}, "
                  f"LR: {train_metrics['lr']:.6f}")

            # Validate
            val_metrics = self.validate()
            print(f"Val - Loss: {val_metrics['val_loss']:.4f}, "
                  f"L1: {val_metrics['val_l1_loss']:.4f}, "
                  f"L2: {val_metrics['val_l2_loss']:.4f}")

            # Save checkpoint
            all_metrics = {**train_metrics, **val_metrics}
            if (epoch + 1) % self.save_every == 0 or epoch == epochs - 1:
                self.save_checkpoint(epoch + 1, all_metrics)

        print("\nTraining complete!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Soft Speech Units model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")

    # Device
    device = config.get("hardware", {}).get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Create model config
    model_config = config.get("model", {})
    soft_config = SoftSpeechUnitsConfig(
        hubert_model=model_config.get("hubert_model", "facebook/hubert-base-ls960"),
        hubert_layer=model_config.get("hubert_layer", 6),
        num_units=model_config.get("num_units", 100),
        soft_unit_dim=model_config.get("soft_unit_dim", 256),
        hidden_dim=model_config.get("hidden_dim", 512),
        prosody_dim=model_config.get("prosody_dim", 4),
        prosody_hidden_dim=model_config.get("prosody_hidden_dim", 256),
        decoder_layers=model_config.get("decoder_layers", 4),
        decoder_heads=model_config.get("decoder_heads", 8),
        decoder_ffn_dim=model_config.get("decoder_ffn_dim", 2048),
        num_speakers=model_config.get("num_speakers", 100),
        speaker_embed_dim=model_config.get("speaker_embed_dim", 256),
        mel_dim=model_config.get("mel_dim", 80),
        output_dim=model_config.get("output_dim", 2048),
        dropout=model_config.get("dropout", 0.1),
        temperature=model_config.get("temperature", 1.0),
    )

    # Create model
    model = SoftSpeechUnitsModel(soft_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Create datasets
    data_config = config.get("data", {})
    audio_config = config.get("audio", {})

    train_dataset = SoftUnitsDataset(
        manifest_path=data_config.get("train_manifest", "../data/splits/train.json"),
        sample_rate=audio_config.get("sample_rate", 16000),
        max_duration=data_config.get("max_duration", 15.0),
        mel_dim=audio_config.get("n_mels", 80),
        n_fft=audio_config.get("n_fft", 1024),
        hop_length=audio_config.get("hop_length", 256),
    )

    val_dataset = SoftUnitsDataset(
        manifest_path=data_config.get("val_manifest", "../data/splits/val.json"),
        sample_rate=audio_config.get("sample_rate", 16000),
        max_duration=data_config.get("max_duration", 15.0),
        mel_dim=audio_config.get("n_mels", 80),
        n_fft=audio_config.get("n_fft", 1024),
        hop_length=audio_config.get("hop_length", 256),
    )

    # Create dataloaders
    train_config = config.get("training", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 16),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    # Create trainer
    trainer = SoftUnitsTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train
    trainer.train(
        epochs=train_config.get("epochs", 50),
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
