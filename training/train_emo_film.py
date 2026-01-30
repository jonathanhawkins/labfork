"""
Training script for Emo-FiLM word-level emotion modulation.

Based on "Beyond Global Emotion" (arXiv:2509.20378).

Training Pipeline:
1. Load emotion-labeled audio dataset
2. Extract frame-level emotions via emotion2vec
3. Align emotions to word boundaries
4. Train FiLM modulation to match word-level emotion targets
5. Evaluate with FEDD metrics

Usage:
    # Train Emo-FiLM model
    python train_emo_film.py --config config/emo_film.yaml

    # Resume from checkpoint
    python train_emo_film.py --config config/emo_film.yaml \
        --resume ../checkpoints/emo_film/best.pt

    # Test mode (synthetic data)
    python train_emo_film.py --test
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.emo_film import (
    EmoFiLMConfig,
    EmoFiLMAdapter,
    EmoFiLMModulator,
    EmoFiLMLoss,
    FEDDEvaluator,
    Emotion2VecExtractor,
    WordLevelAligner,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class EmoFiLMDataset(Dataset):
    """
    Dataset for training Emo-FiLM.

    Expects manifest with:
    - audio_path: Path to audio file
    - text: Transcript
    - words: List of word info (word, start, end, emotion, intensity)
    - utterance_emotion: Global emotion label

    Example manifest entry:
    {
        "audio_path": "data/train/sample_001.wav",
        "text": "I am so happy today!",
        "words": [
            {"word": "I", "start": 0.0, "end": 0.15, "emotion": "neutral", "intensity": 0.5},
            {"word": "am", "start": 0.15, "end": 0.3, "emotion": "neutral", "intensity": 0.5},
            {"word": "so", "start": 0.3, "end": 0.5, "emotion": "happy", "intensity": 0.7},
            {"word": "happy", "start": 0.5, "end": 0.8, "emotion": "happy", "intensity": 0.9},
            {"word": "today", "start": 0.8, "end": 1.2, "emotion": "happy", "intensity": 0.8}
        ],
        "utterance_emotion": "happy",
        "utterance_intensity": 0.8
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: EmoFiLMConfig,
        max_audio_length: float = 15.0,  # seconds
        sample_rate: int = 16000,
    ):
        self.config = config
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

        # Emotion label mapping
        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(config.emotion_labels)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        try:
            import torchaudio
            waveform, sr = torchaudio.load(sample['audio_path'])

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Mono and truncate
            waveform = waveform.mean(dim=0)
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
        except Exception as e:
            logger.warning(f"Failed to load {sample['audio_path']}: {e}")
            waveform = torch.zeros(self.sample_rate)  # 1 second silence

        # Extract word info
        words = sample.get('words', [])
        num_words = len(words)

        word_emotions = []
        word_intensities = []
        word_boundaries = []

        for word_info in words:
            emotion = word_info.get('emotion', 'neutral')
            intensity = word_info.get('intensity', 0.5)
            start = word_info.get('start', 0)
            end = word_info.get('end', start + 0.1)

            emotion_idx = self.emotion_to_idx.get(emotion.lower(), 4)  # Default neutral
            word_emotions.append(emotion_idx)
            word_intensities.append(intensity)
            word_boundaries.append((start, end))

        # Create tensors
        if not word_emotions:
            word_emotions = [4]  # neutral
            word_intensities = [0.5]
            word_boundaries = [(0, len(waveform) / self.sample_rate)]

        word_emotions = torch.tensor(word_emotions, dtype=torch.long)
        word_intensities = torch.tensor(word_intensities, dtype=torch.float32)

        # Compute word durations in frames
        word_durations = torch.tensor([
            int((end - start) * self.config.frame_rate)
            for start, end in word_boundaries
        ], dtype=torch.long)

        # Utterance-level info
        utterance_emotion = self.emotion_to_idx.get(
            sample.get('utterance_emotion', 'neutral').lower(), 4
        )
        utterance_intensity = sample.get('utterance_intensity', 0.5)

        return {
            'audio': waveform,
            'word_emotions': word_emotions,
            'word_intensities': word_intensities,
            'word_durations': word_durations,
            'word_boundaries': word_boundaries,
            'utterance_emotion': torch.tensor(utterance_emotion, dtype=torch.long),
            'utterance_intensity': torch.tensor(utterance_intensity, dtype=torch.float32),
            'num_words': num_words,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    # Find max lengths
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    max_words = max(item['num_words'] for item in batch)

    # Pad and stack
    audios = []
    word_emotions = []
    word_intensities = []
    word_durations = []
    word_boundaries_list = []
    utterance_emotions = []
    utterance_intensities = []
    word_masks = []

    for item in batch:
        # Pad audio
        audio = item['audio']
        if audio.shape[0] < max_audio_len:
            audio = torch.nn.functional.pad(audio, (0, max_audio_len - audio.shape[0]))
        audios.append(audio)

        # Pad word-level tensors
        num_words = item['num_words']
        we = item['word_emotions']
        wi = item['word_intensities']
        wd = item['word_durations']

        if num_words < max_words:
            we = torch.nn.functional.pad(we, (0, max_words - num_words), value=-1)
            wi = torch.nn.functional.pad(wi, (0, max_words - num_words), value=0)
            wd = torch.nn.functional.pad(wd, (0, max_words - num_words), value=0)

        word_emotions.append(we)
        word_intensities.append(wi)
        word_durations.append(wd)

        # Word mask
        mask = torch.zeros(max_words, dtype=torch.bool)
        mask[:num_words] = True
        word_masks.append(mask)

        # Word boundaries (list format)
        word_boundaries_list.append(item['word_boundaries'])

        # Utterance-level
        utterance_emotions.append(item['utterance_emotion'])
        utterance_intensities.append(item['utterance_intensity'])

    return {
        'audio': torch.stack(audios),
        'word_emotions': torch.stack(word_emotions),
        'word_intensities': torch.stack(word_intensities),
        'word_durations': torch.stack(word_durations),
        'word_boundaries': word_boundaries_list,
        'word_masks': torch.stack(word_masks),
        'utterance_emotions': torch.stack(utterance_emotions),
        'utterance_intensities': torch.stack(utterance_intensities),
    }


# =============================================================================
# SYNTHETIC DATASET FOR TESTING
# =============================================================================

class SyntheticEmoFiLMDataset(Dataset):
    """Synthetic dataset for testing without real audio."""

    def __init__(
        self,
        num_samples: int = 100,
        config: EmoFiLMConfig = None,
        sample_rate: int = 16000,
    ):
        self.num_samples = num_samples
        self.config = config or EmoFiLMConfig()
        self.sample_rate = sample_rate

        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(self.config.emotion_labels)
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random audio length (1-5 seconds)
        duration = random.uniform(1.0, 5.0)
        num_samples = int(duration * self.sample_rate)
        audio = torch.randn(num_samples) * 0.1

        # Random number of words (3-8)
        num_words = random.randint(3, 8)

        # Random word durations
        word_durations = torch.randint(5, 20, (num_words,))
        word_durations = word_durations.float()
        word_durations = (word_durations / word_durations.sum() * duration * self.config.frame_rate).long()

        # Random emotions and intensities
        word_emotions = torch.randint(0, self.config.num_emotions, (num_words,))
        word_intensities = torch.rand(num_words) * 0.5 + 0.5  # 0.5-1.0

        # Word boundaries
        time_per_frame = 1.0 / self.config.frame_rate
        boundaries = []
        current_time = 0.0
        for d in word_durations:
            end_time = current_time + d.item() * time_per_frame
            boundaries.append((current_time, end_time))
            current_time = end_time

        # Utterance-level (mode of word emotions)
        utterance_emotion = word_emotions[0]
        utterance_intensity = word_intensities.mean()

        return {
            'audio': audio,
            'word_emotions': word_emotions,
            'word_intensities': word_intensities,
            'word_durations': word_durations,
            'word_boundaries': boundaries,
            'utterance_emotion': utterance_emotion,
            'utterance_intensity': utterance_intensity,
            'num_words': num_words,
        }


# =============================================================================
# TRAINER
# =============================================================================

class EmoFiLMTrainer:
    """Trainer for Emo-FiLM model."""

    def __init__(
        self,
        config: EmoFiLMConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        checkpoint_dir: str = "../checkpoints/emo_film",
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_epochs: int = 50,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
        else:
            self.val_loader = None

        # Model
        self.adapter = EmoFiLMAdapter(config).to(self.device)

        # Mock text encoder for training
        self.text_encoder = nn.Sequential(
            nn.Linear(100, config.text_hidden_dim),
            nn.GELU(),
        ).to(self.device)

        # Loss
        self.loss_fn = EmoFiLMLoss(config)

        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.adapter.parameters()) + list(self.text_encoder.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader),
            eta_min=1e-6,
        )

        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Evaluator
        self.evaluator = FEDDEvaluator(config)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.adapter.train()
        self.text_encoder.train()

        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            audio = batch['audio'].to(self.device)
            word_emotions = batch['word_emotions'].to(self.device)
            word_durations = batch['word_durations'].to(self.device)
            word_masks = batch['word_masks'].to(self.device)

            # Generate mock text embeddings
            batch_size, max_words = word_durations.shape
            text_input = torch.randn(batch_size, max_words, 100, device=self.device)
            text_embeddings = self.text_encoder(text_input)

            # Forward pass
            output = self.adapter(
                audio=audio,
                text_embeddings=text_embeddings,
                word_durations=word_durations.float(),
            )

            # Compute loss
            losses = self.loss_fn(
                modulator_output=output,
                target_emotions=word_emotions,
                original_text_embeddings=text_embeddings,
            )

            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return {
            'train_loss': total_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.adapter.eval()
        self.text_encoder.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch['audio'].to(self.device)
            word_emotions = batch['word_emotions'].to(self.device)
            word_durations = batch['word_durations'].to(self.device)

            batch_size, max_words = word_durations.shape
            text_input = torch.randn(batch_size, max_words, 100, device=self.device)
            text_embeddings = self.text_encoder(text_input)

            output = self.adapter(
                audio=audio,
                text_embeddings=text_embeddings,
                word_durations=word_durations.float(),
            )

            losses = self.loss_fn(
                modulator_output=output,
                target_emotions=word_emotions,
                original_text_embeddings=text_embeddings,
            )

            total_loss += losses['total'].item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'adapter_state_dict': self.adapter.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
        }

        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            logger.info(f"Saved best checkpoint at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']

    def train(self, resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """Full training loop."""
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        for epoch in range(start_epoch, self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 40)

            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Learning Rate: {train_metrics['learning_rate']:.6f}")

            history['train_loss'].append(train_metrics['train_loss'])
            history['learning_rate'].append(train_metrics['learning_rate'])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
                history['val_loss'].append(val_metrics['val_loss'])

                # Save best checkpoint
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=False)

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        logger.info("\nTraining complete!")
        return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Emo-FiLM model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--manifest", type=str, help="Path to training manifest")
    parser.add_argument("--val-manifest", type=str, help="Path to validation manifest")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Run test mode with synthetic data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = EmoFiLMConfig(**config_dict.get('model', {}))
    else:
        config = EmoFiLMConfig()

    logger.info("=" * 60)
    logger.info("Emo-FiLM Training")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Config: {config}")

    if args.test:
        # Test mode with synthetic data
        logger.info("\nRunning in TEST mode with synthetic data")

        train_dataset = SyntheticEmoFiLMDataset(num_samples=100, config=config)
        val_dataset = SyntheticEmoFiLMDataset(num_samples=20, config=config)

        trainer = EmoFiLMTrainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir="../checkpoints/emo_film_test",
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=3,  # Quick test
            device=args.device,
        )

        history = trainer.train()
        logger.info("\nTest training complete!")
        logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")

    else:
        # Real training
        if not args.manifest:
            logger.error("--manifest required for training")
            return

        train_dataset = EmoFiLMDataset(args.manifest, config)

        val_dataset = None
        if args.val_manifest:
            val_dataset = EmoFiLMDataset(args.val_manifest, config)

        trainer = EmoFiLMTrainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir="../checkpoints/emo_film",
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=args.device,
        )

        history = trainer.train(resume_from=args.resume)

        # Save history
        with open(trainer.checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
