"""
EmoVoice Training Script

Two-phase training:
1. Phase 1 (Pre-training): Standard TTS on neutral data - establishes baseline synthesis
2. Phase 2 (Fine-tuning): Emotion-labeled data with natural language descriptions

Based on EmoVoice: arXiv:2504.12867
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.emovoice import (
    EmoVoiceConfig,
    EmoVoice,
    EmoVoiceAdapter,
    EmoVoiceLoss,
    EMOTION_DESCRIPTION_EXAMPLES,
    generate_emotion_description,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for EmoVoice."""

    # Model
    model: EmoVoiceConfig = field(default_factory=EmoVoiceConfig)

    # Data
    train_manifest: str = "../data/emovoice_train.json"
    val_manifest: str = "../data/emovoice_val.json"
    max_audio_length: int = 500  # Max audio tokens
    max_text_length: int = 256  # Max text tokens
    max_desc_length: int = 64  # Max description tokens

    # Training phase 1 (pre-training)
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-4
    pretrain_batch_size: int = 16

    # Training phase 2 (fine-tuning)
    finetune_epochs: int = 30
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 8

    # Optimizer
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Loss weights
    audio_weight: float = 1.0
    phoneme_weight: float = 0.3
    emotion_class_weight: float = 0.2
    contrastive_weight: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "../checkpoints/emovoice"
    save_every: int = 5
    log_every: int = 100

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    gradient_accumulation: int = 1
    use_amp: bool = True

    # Phase control
    skip_pretrain: bool = False
    pretrain_checkpoint: Optional[str] = None


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Create model config
    model_config = EmoVoiceConfig(**config_dict.get("model", {}))

    # Remove model from dict and create training config
    config_dict.pop("model", None)
    config = TrainingConfig(model=model_config, **config_dict)

    return config


# =============================================================================
# DATASET
# =============================================================================

class EmoVoiceDataset(Dataset):
    """
    Dataset for EmoVoice training.

    Manifest format:
    {
        "samples": [
            {
                "audio_path": "path/to/audio.wav",
                "text": "The transcription",
                "emotion": "happy",  # Optional categorical
                "emotion_description": "expressing genuine happiness",  # Optional freestyle
                "audio_tokens": [1, 2, 3, ...],  # Pre-extracted semantic tokens
                "phonemes": [10, 20, 30, ...],  # Pre-extracted phoneme IDs
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: TrainingConfig,
        phase: str = "pretrain",  # "pretrain" or "finetune"
    ):
        self.config = config
        self.phase = phase

        # Load manifest
        with open(manifest_path) as f:
            data = json.load(f)
        self.samples = data.get("samples", data)

        # Simple tokenizer (placeholder)
        self.text_vocab = self._build_vocab()

    def _build_vocab(self) -> Dict[str, int]:
        """Build text vocabulary from samples."""
        vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3}
        idx = 4

        words = set()
        for sample in self.samples[:1000]:  # Use first 1000 for vocab
            text = sample.get("text", "")
            words.update(text.lower().split())

            desc = sample.get("emotion_description", "")
            words.update(desc.lower().split())

        # Add emotion description words
        for descriptions in EMOTION_DESCRIPTION_EXAMPLES.values():
            for desc in descriptions:
                words.update(desc.lower().split())

        for word in sorted(words):
            if word not in vocab:
                vocab[word] = idx
                idx += 1

        return vocab

    def tokenize(self, text: str, max_length: int) -> torch.Tensor:
        """Tokenize text to IDs."""
        words = text.lower().split()[:max_length]
        ids = [self.text_vocab.get(w, 1) for w in words]  # 1 = <unk>

        # Pad or truncate
        if len(ids) < max_length:
            ids = ids + [0] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Text
        text = sample.get("text", "")
        text_ids = self.tokenize(text, self.config.max_text_length)

        # Audio tokens (pre-extracted)
        audio_tokens = sample.get("audio_tokens", [])
        if len(audio_tokens) < self.config.max_audio_length:
            audio_tokens = audio_tokens + [0] * (self.config.max_audio_length - len(audio_tokens))
        else:
            audio_tokens = audio_tokens[:self.config.max_audio_length]
        audio_tokens = torch.tensor(audio_tokens, dtype=torch.long)

        # Phonemes (if available)
        phonemes = sample.get("phonemes", [])
        if phonemes:
            phoneme_len = self.config.max_audio_length // 4  # Lower rate
            if len(phonemes) < phoneme_len:
                phonemes = phonemes + [0] * (phoneme_len - len(phonemes))
            else:
                phonemes = phonemes[:phoneme_len]
        else:
            phonemes = [0] * (self.config.max_audio_length // 4)
        phonemes = torch.tensor(phonemes, dtype=torch.long)

        # Emotion description
        if self.phase == "finetune":
            # Use emotion description for fine-tuning
            description = sample.get("emotion_description", "")
            if not description:
                # Generate from categorical emotion
                emotion = sample.get("emotion", "neutral")
                description = generate_emotion_description(emotion, intensity=0.7)
        else:
            # Pre-training uses neutral description
            description = "speaking naturally and clearly"

        desc_ids = self.tokenize(description, self.config.max_desc_length)

        # Emotion label (for classification loss)
        emotion = sample.get("emotion", "neutral")
        emotion_map = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "fearful": 4, "surprised": 5, "disgusted": 6,
            "excited": 7, "tender": 8, "contempt": 9,
        }
        emotion_idx = emotion_map.get(emotion.lower(), 0)

        return {
            "text_ids": text_ids,
            "audio_tokens": audio_tokens,
            "phonemes": phonemes,
            "description_ids": desc_ids,
            "emotion_idx": torch.tensor(emotion_idx, dtype=torch.long),
        }


# =============================================================================
# TRAINER
# =============================================================================

class EmoVoiceTrainer:
    """Trainer for EmoVoice model."""

    def __init__(
        self,
        model: EmoVoice,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Loss function
        self.loss_fn = EmoVoiceLoss(
            config.model,
            audio_weight=config.audio_weight,
            phoneme_weight=config.phoneme_weight,
            emotion_class_weight=config.emotion_class_weight,
            contrastive_weight=config.contrastive_weight,
        )

        # Optimizer (initialized per phase)
        self.optimizer = None
        self.scheduler = None

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if config.use_amp else None

        # Tracking
        self.global_step = 0
        self.current_epoch = 0

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _create_optimizer(self, lr: float):
        """Create optimizer with given learning rate."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = self.config.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.1, 1.0 - (step - warmup_steps) / (num_training_steps - warmup_steps))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        if self.config.use_amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(
                    text_ids=batch["text_ids"],
                    description_ids=batch["description_ids"],
                    target_audio_tokens=batch["audio_tokens"],
                    target_phonemes=batch["phonemes"],
                )

                losses = self.loss_fn(
                    outputs,
                    target_audio_tokens=batch["audio_tokens"],
                    target_phonemes=batch["phonemes"],
                    target_emotion_idx=batch["emotion_idx"],
                )

            # Backward pass with scaling
            self.scaler.scale(losses["total"]).backward()

            if (self.global_step + 1) % self.config.gradient_accumulation == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
        else:
            outputs = self.model(
                text_ids=batch["text_ids"],
                description_ids=batch["description_ids"],
                target_audio_tokens=batch["audio_tokens"],
                target_phonemes=batch["phonemes"],
            )

            losses = self.loss_fn(
                outputs,
                target_audio_tokens=batch["audio_tokens"],
                target_phonemes=batch["phonemes"],
                target_emotion_idx=batch["emotion_idx"],
            )

            losses["total"].backward()

            if (self.global_step + 1) % self.config.gradient_accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()

        total_losses = {}
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                text_ids=batch["text_ids"],
                description_ids=batch["description_ids"],
                target_audio_tokens=batch["audio_tokens"],
                target_phonemes=batch["phonemes"],
            )

            losses = self.loss_fn(
                outputs,
                target_audio_tokens=batch["audio_tokens"],
                target_phonemes=batch["phonemes"],
                target_emotion_idx=batch["emotion_idx"],
            )

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def train_phase(
        self,
        phase: str,
        epochs: int,
        lr: float,
        batch_size: int,
    ):
        """
        Train a single phase.

        Args:
            phase: "pretrain" or "finetune"
            epochs: Number of epochs
            lr: Learning rate
            batch_size: Batch size
        """
        print(f"\n{'=' * 60}")
        print(f"Starting {phase} phase")
        print(f"Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
        print(f"{'=' * 60}\n")

        # Create datasets for this phase
        train_dataset = EmoVoiceDataset(
            self.config.train_manifest,
            self.config,
            phase=phase,
        )
        val_dataset = None
        if os.path.exists(self.config.val_manifest):
            val_dataset = EmoVoiceDataset(
                self.config.val_manifest,
                self.config,
                phase=phase,
            )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

        # Create optimizer and scheduler
        self._create_optimizer(lr)
        num_training_steps = len(train_loader) * epochs
        self._create_scheduler(num_training_steps)

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Training
            epoch_losses = {}
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                losses = self.train_step(batch)

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v

                # Update progress bar
                if self.global_step % self.config.log_every == 0:
                    pbar.set_postfix({
                        "loss": f"{losses['total']:.4f}",
                        "audio": f"{losses['audio']:.4f}",
                    })

            # Average epoch losses
            num_steps = len(train_loader)
            epoch_losses = {k: v / num_steps for k, v in epoch_losses.items()}

            print(f"\nEpoch {epoch + 1} Training - "
                  f"Total: {epoch_losses['total']:.4f}, "
                  f"Audio: {epoch_losses['audio']:.4f}")

            # Validation
            if val_loader:
                val_losses = self.validate(val_loader)
                print(f"Epoch {epoch + 1} Validation - "
                      f"Total: {val_losses['total']:.4f}, "
                      f"Audio: {val_losses['audio']:.4f}")

                # Save best model
                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    self.save_checkpoint(f"{phase}_best.pt")

            # Regular checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"{phase}_epoch_{epoch + 1}.pt")

        # Save final checkpoint
        self.save_checkpoint(f"{phase}_final.pt")

    def train(self):
        """Run full two-phase training."""
        # Phase 1: Pre-training
        if not self.config.skip_pretrain:
            if self.config.pretrain_checkpoint:
                print(f"Loading pre-train checkpoint: {self.config.pretrain_checkpoint}")
                self.load_checkpoint(self.config.pretrain_checkpoint)
            else:
                self.train_phase(
                    phase="pretrain",
                    epochs=self.config.pretrain_epochs,
                    lr=self.config.pretrain_lr,
                    batch_size=self.config.pretrain_batch_size,
                )

        # Phase 2: Fine-tuning
        self.train_phase(
            phase="finetune",
            epochs=self.config.finetune_epochs,
            lr=self.config.finetune_lr,
            batch_size=self.config.finetune_batch_size,
        )

        print("\nTraining complete!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)

        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint: {path}")
        print(f"  Step: {self.global_step}, Epoch: {self.current_epoch}")


# =============================================================================
# MAIN
# =============================================================================

def create_mock_manifest(path: str, num_samples: int = 100):
    """Create a mock manifest for testing."""
    import random

    samples = []
    emotions = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

    for i in range(num_samples):
        emotion = random.choice(emotions)
        description = generate_emotion_description(emotion, intensity=random.uniform(0.3, 0.9))

        samples.append({
            "text": f"This is test sentence number {i + 1}.",
            "emotion": emotion,
            "emotion_description": description,
            "audio_tokens": [random.randint(0, 4095) for _ in range(random.randint(50, 200))],
            "phonemes": [random.randint(0, 255) for _ in range(random.randint(10, 50))],
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)

    print(f"Created mock manifest with {num_samples} samples: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train EmoVoice model")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock data")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pre-training phase")
    args = parser.parse_args()

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = TrainingConfig()

    # Override with command line args
    if args.skip_pretrain:
        config.skip_pretrain = True
    if args.resume:
        config.pretrain_checkpoint = args.resume

    # Test mode
    if args.test:
        print("Running in test mode with mock data...")
        config.pretrain_epochs = 2
        config.finetune_epochs = 2
        config.pretrain_batch_size = 4
        config.finetune_batch_size = 4
        config.log_every = 10
        config.save_every = 1

        # Create mock manifests
        create_mock_manifest(config.train_manifest, num_samples=50)
        create_mock_manifest(config.val_manifest, num_samples=10)

    # Create model
    print("Creating EmoVoice model...")
    model = EmoVoice(config.model)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {param_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    # Create trainer
    trainer = EmoVoiceTrainer(model, config)

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
