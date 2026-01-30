"""
Training script for MPE-TTS multi-modal emotion encoder.

Based on "MPE-TTS: Multi-Modal Prompt Emotion Encoder for Expressive TTS"
Interspeech 2025 - arXiv:2505.18453

Three-Stage Training Pipeline:
1. Emotion Stage (100 epochs): Train MPEE to align text/image with speech emotions
2. Acoustic Stage (500k + 50 epochs): Train diffusion decoder
3. Prosody Stage (50 epochs): Train prosody predictor with ECL

Usage:
    # Full three-stage training
    python train_mpe_tts.py --config config/mpe_tts.yaml

    # Stage 1 only (MPEE training)
    python train_mpe_tts.py --config config/mpe_tts.yaml --stage 1

    # Resume from checkpoint
    python train_mpe_tts.py --config config/mpe_tts.yaml \
        --resume ../checkpoints/mpe_tts/stage1_best.pt

    # Test mode (synthetic data)
    python train_mpe_tts.py --test
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

from training.mpe_tts import (
    MPETTSConfig,
    MultiModalPromptEmotionEncoder,
    ProsodyPredictor,
    EmotionConsistencyLoss,
    MPETTSLoss,
    MPETTSAdapter,
    EMOTION_TO_IDX,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class MPETTSDataset(Dataset):
    """
    Dataset for MPE-TTS training.

    Expects manifest with:
    - audio_path: Path to audio file (for speech emotion)
    - text: Transcript
    - emotion: Emotion label
    - emotion_description: Natural language emotion description (optional)
    - image_path: Path to emotion image like facial expression (optional)

    Example manifest entry:
    {
        "audio_path": "data/train/sample_001.wav",
        "text": "I am so happy today!",
        "emotion": "happy",
        "emotion_description": "expressing genuine happiness and warmth",
        "image_path": "data/images/happy_face_001.jpg"
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: MPETTSConfig,
        max_audio_length: float = 15.0,  # seconds
        sample_rate: int = 16000,
        include_images: bool = False,
    ):
        self.config = config
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.include_images = include_images

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

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

            # Pad if needed
            if len(waveform) < self.sample_rate:  # Min 1 second
                waveform = F.pad(waveform, (0, self.sample_rate - len(waveform)))

        except Exception as e:
            logger.warning(f"Failed to load {sample.get('audio_path', 'unknown')}: {e}")
            waveform = torch.zeros(self.sample_rate)  # 1 second silence

        # Get emotion label
        emotion = sample.get('emotion', 'neutral')
        emotion_idx = EMOTION_TO_IDX.get(emotion, EMOTION_TO_IDX['neutral'])

        # Get emotion description
        emotion_desc = sample.get('emotion_description', f"speaking with {emotion} emotion")

        result = {
            'audio': waveform,
            'emotion_idx': torch.tensor(emotion_idx, dtype=torch.long),
            'emotion_description': emotion_desc,
            'text': sample.get('text', ''),
        }

        # Load image if available and requested
        if self.include_images and 'image_path' in sample:
            try:
                from PIL import Image
                result['image'] = Image.open(sample['image_path']).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {sample.get('image_path')}: {e}")
                result['image'] = None

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for MPE-TTS dataset."""
    # Pad audio to max length in batch
    audios = [item['audio'] for item in batch]
    max_len = max(a.shape[0] for a in audios)
    padded_audios = torch.stack([
        F.pad(a, (0, max_len - a.shape[0])) for a in audios
    ])

    return {
        'audio': padded_audios,
        'emotion_idx': torch.stack([item['emotion_idx'] for item in batch]),
        'emotion_description': [item['emotion_description'] for item in batch],
        'text': [item['text'] for item in batch],
        'images': [item.get('image') for item in batch] if 'image' in batch[0] else None,
    }


# =============================================================================
# TRAINER
# =============================================================================

class MPETTSTrainer:
    """
    Trainer for MPE-TTS with three-stage training support.

    Stage 1: MPEE training - align text/image with speech emotions
    Stage 2: Acoustic model training (placeholder - uses external diffusion)
    Stage 3: Prosody predictor training with ECL
    """

    def __init__(
        self,
        config: MPETTSConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize models
        self.mpee = MultiModalPromptEmotionEncoder(config).to(device)
        self.prosody_predictor = ProsodyPredictor(config).to(device)
        self.adapter = MPETTSAdapter(config).to(device)

        # Loss function
        self.loss_fn = MPETTSLoss(config).to(device)

        # Optimizers (will be set per stage)
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.current_stage = 1
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def setup_stage1_optimizer(self):
        """Setup optimizer for Stage 1 (MPEE training)."""
        # Only train adapters, freeze speech encoder (it's pre-trained)
        trainable_params = []

        # Text adapter
        trainable_params.extend(self.mpee.text_encoder.adapter.parameters())

        # Image adapter
        trainable_params.extend(self.mpee.image_encoder.adapter.parameters())

        # Emotion classifier
        trainable_params.extend(self.mpee.emotion_classifier.parameters())

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=1e-4,
            betas=(0.8, 0.99),
            weight_decay=0.01,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.stage1_epochs,
        )

        logger.info(f"Stage 1 optimizer setup with {sum(p.numel() for p in trainable_params)} parameters")

    def setup_stage3_optimizer(self):
        """Setup optimizer for Stage 3 (Prosody predictor training)."""
        trainable_params = list(self.prosody_predictor.parameters())

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=5e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=5e-5,
            epochs=self.config.stage3_epochs,
            steps_per_epoch=len(self.train_loader),
        )

        logger.info(f"Stage 3 optimizer setup with {sum(p.numel() for p in trainable_params)} parameters")

    def train_stage1_epoch(self) -> Dict[str, float]:
        """
        Train one epoch for Stage 1 (MPEE alignment).

        Goal: Train text/image adapters to match speech emotion embeddings.
        Loss: MSE(E_text, E_speech) + MSE(E_image, E_speech) + CE(emotion)
        """
        self.mpee.train()
        epoch_losses = {
            'text_mse': 0.0,
            'emotion_ce': 0.0,
            'total': 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            audio = batch['audio'].to(self.device)
            emotion_idx = batch['emotion_idx'].to(self.device)
            emotion_descriptions = batch['emotion_description']

            # Forward pass
            self.optimizer.zero_grad()

            # Get alignment losses
            alignment_losses = self.mpee.compute_alignment_loss(
                speech_audio=audio,
                text=emotion_descriptions,
            )

            # Forward for classification loss
            mpee_out = self.mpee(speech_audio=audio)

            # Emotion classification loss
            emotion_ce = F.cross_entropy(mpee_out['emotion_logits'], emotion_idx)

            # Total loss
            total_loss = alignment_losses['mpee_total'] + 0.5 * emotion_ce

            # Backward
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.mpee.parameters(), 1.0)
            self.optimizer.step()

            # Track losses
            if 'text_mse' in alignment_losses:
                epoch_losses['text_mse'] += alignment_losses['text_mse'].item()
            epoch_losses['emotion_ce'] += emotion_ce.item()
            epoch_losses['total'] += total_loss.item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % 100 == 0:
                logger.info(
                    f"Stage 1 Step {self.global_step}: "
                    f"text_mse={alignment_losses.get('text_mse', 0):.4f}, "
                    f"emotion_ce={emotion_ce.item():.4f}"
                )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    def train_stage3_epoch(self) -> Dict[str, float]:
        """
        Train one epoch for Stage 3 (Prosody predictor with ECL).

        Uses frozen MPEE for emotion embeddings.
        Trains prosody predictor with emotion consistency loss.
        """
        self.prosody_predictor.train()
        self.mpee.eval()  # Freeze MPEE

        epoch_losses = {
            'prosody_ce': 0.0,
            'ecl': 0.0,
            'total': 0.0,
        }
        num_batches = 0

        for batch in self.train_loader:
            audio = batch['audio'].to(self.device)
            emotion_idx = batch['emotion_idx'].to(self.device)

            self.optimizer.zero_grad()

            # Get emotion embedding from MPEE (frozen)
            with torch.no_grad():
                mpee_out = self.mpee(speech_audio=audio)
                emotion_emb = mpee_out['emotion_embedding']

            # Mock content and timbre (in practice, from separate encoders)
            batch_size, seq_len = audio.shape[0], audio.shape[1] // 320  # ~50Hz
            content = torch.randn(
                batch_size, seq_len, self.config.content_encoder_dim,
                device=self.device
            )
            timbre = torch.randn(batch_size, self.config.timbre_dim, device=self.device)

            # Mock prosody targets (in practice, from VQ encoder)
            prosody_target = torch.randint(
                0, self.config.prosody_codebook_size,
                (batch_size, seq_len),
                device=self.device
            )

            # Forward pass
            pred_out = self.prosody_predictor(
                content=content,
                timbre=timbre,
                emotion=emotion_emb,
                prosody_target=prosody_target,
                teacher_forcing=True,
            )

            # Losses
            losses = self.loss_fn(
                mpee_outputs=mpee_out,
                prosody_outputs=pred_out,
                prosody_targets=prosody_target,
                emotion_labels=emotion_idx,
            )

            # Backward
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.prosody_predictor.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Track losses
            if 'prosody_ce' in losses:
                epoch_losses['prosody_ce'] += losses['prosody_ce'].item()
            if 'ecl' in losses:
                epoch_losses['ecl'] += losses['ecl'].item()
            epoch_losses['total'] += losses['total'].item()
            num_batches += 1
            self.global_step += 1

            if self.global_step % 100 == 0:
                logger.info(
                    f"Stage 3 Step {self.global_step}: "
                    f"prosody_ce={losses.get('prosody_ce', 0):.4f}, "
                    f"ecl={losses.get('ecl', 0):.4f}"
                )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        self.mpee.eval()
        self.prosody_predictor.eval()

        val_losses = {
            'text_mse': 0.0,
            'emotion_ce': 0.0,
            'emotion_accuracy': 0.0,
            'total': 0.0,
        }
        num_batches = 0
        correct = 0
        total = 0

        for batch in self.val_loader:
            audio = batch['audio'].to(self.device)
            emotion_idx = batch['emotion_idx'].to(self.device)
            emotion_descriptions = batch['emotion_description']

            # MPEE forward
            mpee_out = self.mpee(speech_audio=audio, text=emotion_descriptions)

            # Alignment loss
            if 'speech_emotion' in mpee_out and 'text_emotion' in mpee_out:
                text_mse = F.mse_loss(mpee_out['text_emotion'], mpee_out['speech_emotion'])
                val_losses['text_mse'] += text_mse.item()

            # Classification
            logits = mpee_out['emotion_logits']
            emotion_ce = F.cross_entropy(logits, emotion_idx)
            val_losses['emotion_ce'] += emotion_ce.item()

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct += (preds == emotion_idx).sum().item()
            total += emotion_idx.shape[0]

            val_losses['total'] += (val_losses['text_mse'] + emotion_ce.item())
            num_batches += 1

        # Average
        for key in val_losses:
            if key != 'emotion_accuracy':
                val_losses[key] /= max(num_batches, 1)

        val_losses['emotion_accuracy'] = correct / max(total, 1)

        return val_losses

    def train_stage(
        self,
        stage: int,
        num_epochs: int,
        checkpoint_dir: str,
    ):
        """Train a specific stage."""
        self.current_stage = stage
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer for stage
        if stage == 1:
            self.setup_stage1_optimizer()
        elif stage == 3:
            self.setup_stage3_optimizer()
        else:
            logger.warning(f"Stage {stage} training not implemented (acoustic model)")
            return

        logger.info(f"Starting Stage {stage} training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            if stage == 1:
                train_losses = self.train_stage1_epoch()
            else:
                train_losses = self.train_stage3_epoch()

            # Validate
            val_losses = self.validate()

            # Step scheduler
            if stage == 1:
                self.scheduler.step()

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train: {train_losses} - Val: {val_losses}"
            )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"stage{stage}_epoch{epoch + 1}.pt"
                )

            # Save best
            current_loss = val_losses.get('total', train_losses['total'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint(checkpoint_dir / f"stage{stage}_best.pt")
                logger.info(f"New best model saved (loss={current_loss:.4f})")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'mpee_state_dict': self.mpee.state_dict(),
            'prosody_predictor_state_dict': self.prosody_predictor.state_dict(),
            'adapter_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': asdict(self.config),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.mpee.load_state_dict(checkpoint['mpee_state_dict'])
        self.prosody_predictor.load_state_dict(checkpoint['prosody_predictor_state_dict'])
        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])

        self.current_stage = checkpoint['current_stage']
        self.current_epoch = checkpoint['current_epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        logger.info(
            f"Loaded checkpoint from {path} "
            f"(stage={self.current_stage}, epoch={self.current_epoch})"
        )


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_synthetic_dataset(
    num_samples: int,
    config: MPETTSConfig,
    output_path: str,
):
    """Create synthetic dataset for testing."""
    import torch.nn.functional as F

    emotions = config.emotion_labels
    descriptions = {
        "neutral": "speaking in a calm, neutral tone",
        "happy": "expressing genuine happiness and warmth",
        "sad": "speaking with deep sadness and melancholy",
        "angry": "expressing intense anger and frustration",
        "surprised": "speaking with sudden astonishment",
        "fearful": "expressing anxious fear and worry",
        "disgusted": "speaking with disgust and revulsion",
        "contempt": "expressing cold contempt and disdain",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for i in range(num_samples):
            emotion = emotions[i % len(emotions)]
            sample = {
                "audio_path": f"synthetic/sample_{i:04d}.wav",
                "text": f"This is sample {i} with {emotion} emotion.",
                "emotion": emotion,
                "emotion_description": descriptions.get(emotion, descriptions["neutral"]),
            }
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Created synthetic dataset with {num_samples} samples at {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train MPE-TTS multi-modal emotion encoder")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Training stage (1, 2, or 3)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--test", action="store_true", help="Run in test mode with synthetic data")
    parser.add_argument("--manifest", type=str, help="Path to training manifest")
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints/mpe_tts")

    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load or create config
    if args.config:
        config_dict = load_config(args.config)
        config = MPETTSConfig(**config_dict.get('model', {}))
    else:
        config = MPETTSConfig()

    # Test mode
    if args.test:
        logger.info("Running in test mode with synthetic data")

        # Create synthetic manifest
        manifest_path = create_synthetic_dataset(100, config, "/tmp/mpe_tts_test/manifest.jsonl")

        # Create mock dataset that works without real audio
        class MockDataset(Dataset):
            def __init__(self, num_samples, config):
                self.num_samples = num_samples
                self.config = config
                self.emotions = config.emotion_labels
                self.descriptions = {
                    "neutral": "speaking in a calm, neutral tone",
                    "happy": "expressing genuine happiness and warmth",
                    "sad": "speaking with deep sadness and melancholy",
                    "angry": "expressing intense anger and frustration",
                    "surprised": "speaking with sudden astonishment",
                    "fearful": "expressing anxious fear and worry",
                    "disgusted": "speaking with disgust and revulsion",
                    "contempt": "expressing cold contempt and disdain",
                }

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                emotion = self.emotions[idx % len(self.emotions)]
                return {
                    'audio': torch.randn(16000),  # 1 second mock audio
                    'emotion_idx': torch.tensor(EMOTION_TO_IDX.get(emotion, 0)),
                    'emotion_description': self.descriptions.get(emotion, "neutral"),
                    'text': f"Sample {idx}",
                }

        train_dataset = MockDataset(80, config)
        val_dataset = MockDataset(20, config)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        # Create trainer
        trainer = MPETTSTrainer(config, train_loader, val_loader, device)

        # Test Stage 1 training (reduced epochs)
        logger.info("Testing Stage 1 training...")
        trainer.train_stage(1, num_epochs=2, checkpoint_dir=args.checkpoint_dir)

        # Test Stage 3 training (reduced epochs)
        logger.info("Testing Stage 3 training...")
        trainer.train_stage(3, num_epochs=2, checkpoint_dir=args.checkpoint_dir)

        logger.info("Test mode completed successfully!")
        return

    # Full training
    if args.manifest:
        manifest_path = args.manifest
    else:
        logger.error("Please provide --manifest path or use --test mode")
        return

    # Load datasets
    train_dataset = MPETTSDataset(manifest_path, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_manifest = config_dict.get('training', {}).get('val_manifest_path')
    val_loader = None
    if val_manifest and Path(val_manifest).exists():
        val_dataset = MPETTSDataset(val_manifest, config)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config_dict.get('training', {}).get('batch_size', 8),
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

    # Create trainer
    trainer = MPETTSTrainer(config, train_loader, val_loader, device)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train specific stage or all stages
    if args.stage == 1:
        trainer.train_stage(1, config.stage1_epochs, args.checkpoint_dir)
    elif args.stage == 3:
        trainer.train_stage(3, config.stage3_epochs, args.checkpoint_dir)
    else:
        # Full three-stage training
        logger.info("Starting full three-stage training...")

        # Stage 1: MPEE
        trainer.train_stage(1, config.stage1_epochs, args.checkpoint_dir)

        # Stage 2: Acoustic model (placeholder)
        logger.info("Stage 2 (Acoustic) - using external diffusion trainer")

        # Stage 3: Prosody predictor
        trainer.train_stage(3, config.stage3_epochs, args.checkpoint_dir)

    logger.info("Training completed!")


if __name__ == "__main__":
    # Handle missing F import
    import torch.nn.functional as F
    main()
