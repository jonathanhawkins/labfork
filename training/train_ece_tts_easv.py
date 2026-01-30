"""
Training Script for ECE-TTS EASV (Emotion-Adaptive Spherical Vectors)

Trains the EASV module for intensity-controllable emotion synthesis using:
    emb_out = emb_neutral + α * (emb_emotion - emb_neutral)

Key Training Objectives:
1. Intensity monotonicity: Higher α → larger distance from neutral
2. Emotion discrimination: Different emotions have distinct directions
3. VAD reconstruction: Embedding should encode VAD information
4. Prosody alignment: Tokens should condition TTS appropriately

SUCCESS CRITERIA:
- Intensity=0.5 should produce weaker emotion than intensity=1.0
- Intensity=1.5 should produce stronger/exaggerated emotion
- Happy pitch contour at intensity=1.5 > intensity=1.0 > intensity=0.5

VERIFICATION:
- Run inference with same prompt at intensities [0.5, 1.0, 1.5]
- Extract F0 contours and verify monotonic relationship with intensity
- Use emotion2vec to classify outputs and check confidence scales with intensity

Usage:
    python train_ece_tts_easv.py --config config/ece_tts_easv.yaml

    # With intensity verification
    python train_ece_tts_easv.py --config config/ece_tts_easv.yaml --verify
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import yaml

# Project imports
from ece_tts_easv import (
    EASVConfig,
    EASVAdapter,
    EASVLoss,
    EASVIntensityController,
    create_easv_adapter,
    CORE_EMOTIONS,
)
from spherical_emotion import VAD_PROTOTYPES, EMOTION_TO_IDX

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA
# =============================================================================

class EmotionDataset(Dataset):
    """
    Dataset for EASV training.

    Each sample contains:
    - emotion_idx: Emotion label index
    - vad: Ground truth VAD coordinates
    - prosody_features: Optional prosody features for alignment
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: Optional[str] = None,
        intensity_range: Tuple[float, float] = (0.0, 2.0),
        augment_intensity: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.prosody_cache_dir = Path(prosody_cache_dir) if prosody_cache_dir else None
        self.intensity_range = intensity_range
        self.augment_intensity = augment_intensity

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get emotion
        emotion = sample.get('emotion', 'neutral').lower()
        if emotion not in EMOTION_TO_IDX:
            emotion = 'neutral'
        emotion_idx = EMOTION_TO_IDX[emotion]

        # Get VAD
        if 'vad' in sample:
            vad = torch.tensor(sample['vad'], dtype=torch.float32)
        else:
            vad = torch.tensor(VAD_PROTOTYPES[emotion], dtype=torch.float32)

        # Sample random intensity for training
        if self.augment_intensity:
            intensity = torch.empty(1).uniform_(*self.intensity_range).item()
        else:
            intensity = 1.0

        # Load prosody features if available
        if self.prosody_cache_dir:
            prosody_path = self.prosody_cache_dir / f"{sample['id']}.pt"
            if prosody_path.exists():
                prosody_data = torch.load(prosody_path, map_location='cpu')
                prosody_features = prosody_data.get('features', None)
            else:
                prosody_features = None
        else:
            prosody_features = None

        return {
            'emotion_idx': emotion_idx,
            'vad': vad,
            'intensity': intensity,
            'prosody_features': prosody_features,
            'sample_id': sample.get('id', str(idx)),
        }


def collate_fn(batch):
    """Custom collate for variable prosody features."""
    emotion_idx = torch.tensor([b['emotion_idx'] for b in batch], dtype=torch.long)
    vad = torch.stack([b['vad'] for b in batch])
    intensity = torch.tensor([b['intensity'] for b in batch], dtype=torch.float32)

    # Handle optional prosody features
    has_prosody = any(b['prosody_features'] is not None for b in batch)

    return {
        'emotion_idx': emotion_idx,
        'vad': vad,
        'intensity': intensity,
        'has_prosody': has_prosody,
        'sample_ids': [b['sample_id'] for b in batch],
    }


# =============================================================================
# TRAINING
# =============================================================================

class EASVTrainer:
    """
    Trainer for EASV module.

    Implements:
    1. Multi-intensity training with monotonicity constraint
    2. Curriculum learning for intensity range
    3. Verification of SUCCESS CRITERIA
    """

    def __init__(
        self,
        config_path: str,
        device: str = 'cuda',
    ):
        self.device = torch.device(device)

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create EASV config
        easv_cfg = self.config.get('easv', {})
        self.easv_config = EASVConfig(
            embedding_dim=easv_cfg.get('embedding_dim', 256),
            hidden_dim=easv_cfg.get('hidden_dim', 512),
            output_dim=easv_cfg.get('output_dim', 2048),
            num_emotions=easv_cfg.get('num_emotions', 8),
            use_learnable_neutral=easv_cfg.get('use_learnable_neutral', True),
            use_learnable_directions=easv_cfg.get('use_learnable_directions', True),
            use_adaptive_scaling=easv_cfg.get('use_adaptive_scaling', True),
            default_intensity=easv_cfg.get('default_intensity', 1.0),
            min_intensity=easv_cfg.get('min_intensity', 0.0),
            max_intensity=easv_cfg.get('max_intensity', 2.0),
            num_prosody_tokens=easv_cfg.get('num_prosody_tokens', 4),
            dropout=easv_cfg.get('dropout', 0.1),
            use_layer_norm=easv_cfg.get('use_layer_norm', True),
        )

        # Create adapter
        self.adapter = create_easv_adapter(self.easv_config).to(self.device)

        # Loss function
        loss_cfg = self.config.get('loss', {})
        self.loss_fn = EASVLoss(
            self.easv_config,
            reconstruction_weight=loss_cfg.get('reconstruction_weight', 1.0),
            monotonicity_weight=loss_cfg.get('monotonicity_weight', 0.5),
            orthogonality_weight=loss_cfg.get('orthogonality_weight', 0.2),
            neutral_reg_weight=loss_cfg.get('neutral_reg_weight', 0.1),
            classification_weight=loss_cfg.get('classification_weight', 0.5),
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.loss_fn.parameters()),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(
                self.config.get('adam_beta1', 0.9),
                self.config.get('adam_beta2', 0.98),
            ),
            eps=self.config.get('adam_epsilon', 1e-9),
        )

        # Scheduler
        self.scheduler = None  # Set up in train()

        # Mixed precision
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Output directory
        self.output_dir = Path(self.config.get('output_dir', './checkpoints/easv'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EASV Trainer initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output: {self.output_dir}")

    def get_intensity_range(self, epoch: int) -> Tuple[float, float]:
        """Get curriculum-adjusted intensity range."""
        intensity_cfg = self.config.get('intensity_training', {})

        if not intensity_cfg.get('use_curriculum', False):
            return (
                self.easv_config.min_intensity,
                self.easv_config.max_intensity,
            )

        curriculum_epochs = intensity_cfg.get('curriculum_epochs', 10)
        start_range = intensity_cfg.get('curriculum_start_range', [0.7, 1.3])
        end_range = intensity_cfg.get('curriculum_end_range', [0.0, 2.0])

        if epoch >= curriculum_epochs:
            return tuple(end_range)

        # Linear interpolation
        progress = epoch / curriculum_epochs
        min_int = start_range[0] + progress * (end_range[0] - start_range[0])
        max_int = start_range[1] + progress * (end_range[1] - start_range[1])

        return (min_int, max_int)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        self.adapter.train()

        emotion_idx = batch['emotion_idx'].to(self.device)
        vad = batch['vad'].to(self.device)
        intensity = batch['intensity'].to(self.device)

        batch_size = emotion_idx.shape[0]

        with autocast(enabled=self.use_amp):
            # Forward pass at current intensity
            result = self.adapter.controller(emotion_idx, intensity, batch_size)

            # Forward pass at lower intensity (for monotonicity)
            intensity_low = (intensity * 0.5).clamp(
                self.easv_config.min_intensity,
                self.easv_config.max_intensity
            )
            result_low = self.adapter.controller(emotion_idx, intensity_low, batch_size)

            # Compute losses
            losses = self.loss_fn(
                result,
                target_emotion=emotion_idx,
                target_vad=vad,
                controller_output_low=result_low,
            )

            # Additional monotonicity loss
            mono_loss = self.loss_fn.intensity_monotonicity_loss(
                result_low["embedding"],
                result["embedding"],
                result["neutral_emb"],
            )
            losses["monotonicity"] = mono_loss

            # Direction orthogonality
            directions = self.adapter.controller.direction_bank.directions
            ortho_loss = self.loss_fn.direction_orthogonality_loss(directions)
            losses["orthogonality"] = ortho_loss

            # Neutral regularization
            neutral_vad = self.adapter.controller.neutral_anchor.neutral_vad
            neutral_loss = self.loss_fn.neutral_regularization_loss(neutral_vad)
            losses["neutral_reg"] = neutral_loss

            # Total loss
            loss_cfg = self.config.get('loss', {})
            total_loss = (
                losses.get("classification", 0.0) * loss_cfg.get('classification_weight', 0.5) +
                losses.get("reconstruction", 0.0) * loss_cfg.get('reconstruction_weight', 1.0) +
                losses["monotonicity"] * loss_cfg.get('monotonicity_weight', 0.5) +
                losses["orthogonality"] * loss_cfg.get('orthogonality_weight', 0.2) +
                losses["neutral_reg"] * loss_cfg.get('neutral_reg_weight', 0.1)
            )

        # Backward
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(),
                self.config.get('max_grad_norm', 1.0),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(),
                self.config.get('max_grad_norm', 1.0),
            )
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'classification': losses.get("classification", torch.tensor(0.0)).item(),
            'reconstruction': losses.get("reconstruction", torch.tensor(0.0)).item(),
            'monotonicity': losses["monotonicity"].item(),
            'orthogonality': losses["orthogonality"].item(),
            'neutral_reg': losses["neutral_reg"].item(),
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation with SUCCESS CRITERIA checking."""
        self.adapter.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        monotonic_pairs = 0
        total_pairs = 0

        for batch in val_loader:
            emotion_idx = batch['emotion_idx'].to(self.device)
            vad = batch['vad'].to(self.device)
            batch_size = emotion_idx.shape[0]

            # Test at multiple intensities
            intensities = [0.5, 1.0, 1.5]
            results = []

            neutral_result = self.adapter.controller(
                torch.zeros_like(emotion_idx), 0.0, batch_size
            )
            neutral_emb = neutral_result["neutral_emb"]

            for α in intensities:
                result = self.adapter.controller(emotion_idx, α, batch_size)
                results.append(result)

            # Check monotonicity
            for i in range(len(intensities) - 1):
                dist_low = (results[i]["embedding"] - neutral_emb).norm(dim=-1)
                dist_high = (results[i + 1]["embedding"] - neutral_emb).norm(dim=-1)
                monotonic = (dist_low < dist_high).float()
                monotonic_pairs += monotonic.sum().item()
                total_pairs += batch_size

            # Classification accuracy at α=1.0
            result_10 = results[1]  # α=1.0
            logits = self.loss_fn.emotion_classifier(result_10["embedding"])
            preds = logits.argmax(dim=-1)
            correct += (preds == emotion_idx).sum().item()
            total += batch_size

        metrics = {
            'monotonicity_ratio': monotonic_pairs / max(total_pairs, 1),
            'classification_accuracy': correct / max(total, 1),
        }

        return metrics

    def verify_success_criteria(self, emotion: str = "happy") -> bool:
        """
        Verify SUCCESS CRITERIA.

        Tests that:
        - Intensity=0.5 produces weaker emotion than intensity=1.0
        - Intensity=1.5 produces stronger/exaggerated emotion
        """
        self.adapter.eval()

        emotion_idx = EMOTION_TO_IDX[emotion]
        intensities = [0.5, 1.0, 1.5]

        with torch.no_grad():
            # Get neutral reference
            neutral_result = self.adapter.controller(
                torch.tensor([0], device=self.device), 0.0, 1
            )
            neutral_emb = neutral_result["neutral_emb"]

            # Get embeddings at different intensities
            distances = []
            for α in intensities:
                result = self.adapter.controller(
                    torch.tensor([emotion_idx], device=self.device), α, 1
                )
                dist = (result["embedding"] - neutral_emb).norm().item()
                distances.append(dist)

        # Check monotonicity
        is_monotonic = distances[0] < distances[1] < distances[2]

        logger.info(f"\nSUCCESS CRITERIA Verification ({emotion}):")
        logger.info(f"  α=0.5: distance = {distances[0]:.4f}")
        logger.info(f"  α=1.0: distance = {distances[1]:.4f}")
        logger.info(f"  α=1.5: distance = {distances[2]:.4f}")
        logger.info(f"  Monotonic: {'YES ✓' if is_monotonic else 'NO ✗'}")

        return is_monotonic

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'easv_config': vars(self.easv_config),
            'easv_adapter': self.adapter.state_dict(),
            'loss_fn': self.loss_fn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }

        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
    ):
        """Main training loop."""
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 30)

        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = self.config.get('warmup_steps', 500)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.get('min_lr', 1e-6),
        )

        log_every = self.config.get('log_every', 10)
        save_every = self.config.get('save_every_epochs', 5)
        validate_every = self.config.get('validate_every_epochs', 1)

        logger.info(f"\nStarting EASV training for {num_epochs} epochs")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Get curriculum intensity range
            int_range = self.get_intensity_range(epoch)
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Intensity range: [{int_range[0]:.2f}, {int_range[1]:.2f}]")

            # Update dataset intensity range
            if hasattr(train_loader.dataset, 'intensity_range'):
                train_loader.dataset.intensity_range = int_range

            # Training
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                self.global_step += 1

                metrics = self.train_step(batch)
                epoch_losses.append(metrics['total_loss'])

                if batch_idx % log_every == 0:
                    logger.info(
                        f"  Step {batch_idx}/{len(train_loader)}: "
                        f"loss={metrics['total_loss']:.4f}, "
                        f"mono={metrics['monotonicity']:.4f}, "
                        f"class={metrics['classification']:.4f}"
                    )

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            # Validation
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"  Validation:")
                logger.info(f"    Monotonicity ratio: {val_metrics['monotonicity_ratio']:.4f}")
                logger.info(f"    Classification acc: {val_metrics['classification_accuracy']:.4f}")

                # Check if best
                if val_metrics['monotonicity_ratio'] > self.best_metric:
                    self.best_metric = val_metrics['monotonicity_ratio']
                    is_best = True
                else:
                    is_best = False

            # Verify SUCCESS CRITERIA
            if self.config.get('evaluation', {}).get('verify_every_epoch', False):
                self.verify_success_criteria("happy")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                ckpt_path = self.output_dir / f'epoch_{epoch + 1}.pt'
                self.save_checkpoint(str(ckpt_path), is_best=is_best if 'is_best' in dir() else False)

        # Final verification
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUCCESS CRITERIA VERIFICATION")
        logger.info("=" * 60)

        for emotion in ["happy", "sad", "angry"]:
            self.verify_success_criteria(emotion)

        # Save final checkpoint
        final_path = self.output_dir / 'final.pt'
        self.save_checkpoint(str(final_path))

        logger.info(f"\nTraining complete!")
        logger.info(f"  Best monotonicity ratio: {self.best_metric:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ECE-TTS EASV')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu/mps)')
    parser.add_argument('--verify', action='store_true',
                       help='Run verification only (no training)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    args = parser.parse_args()

    # Create trainer
    trainer = EASVTrainer(args.config, device=args.device)

    # Load checkpoint if provided
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.adapter.load_state_dict(ckpt['easv_adapter'])
        trainer.loss_fn.load_state_dict(ckpt['loss_fn'])
        trainer.epoch = ckpt.get('epoch', 0)
        trainer.global_step = ckpt.get('global_step', 0)
        trainer.best_metric = ckpt.get('best_metric', 0.0)
        logger.info(f"Resumed from checkpoint: epoch {trainer.epoch}")

    # Verification only mode
    if args.verify:
        logger.info("Running SUCCESS CRITERIA verification only...")
        for emotion in CORE_EMOTIONS[:4]:  # Test first 4 emotions
            trainer.verify_success_criteria(emotion)
        return

    # Load config for data paths
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create datasets
    train_dataset = EmotionDataset(
        manifest_path=config.get('train_manifest', '../data/splits/train.json'),
        prosody_cache_dir=config.get('prosody_cache_dir'),
        intensity_range=(0.0, 2.0),
        augment_intensity=True,
    )

    val_dataset = EmotionDataset(
        manifest_path=config.get('val_manifest', '../data/splits/val.json'),
        prosody_cache_dir=config.get('prosody_cache_dir'),
        intensity_range=(0.5, 1.5),  # Narrower range for validation
        augment_intensity=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', True),
    )

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
