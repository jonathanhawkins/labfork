"""
MaskGCT Training Script for Parallel Prosody Generation

Trains the MaskGCT-style masked prosody prediction model for non-autoregressive TTS.
This enables 2x+ faster inference compared to autoregressive baselines.

Key Features:
- Mask-and-predict training objective (like BERT/T5)
- Mask scheduling: high ratio (0.8) → low ratio (0.15) during training
- Two-stage prediction: semantic → acoustic
- Parallel iterative decoding for fast inference

Usage:
    python train_maskgct.py --config config/maskgct_prosody.yaml

References:
- MaskGCT: Zero-Shot TTS with Masked Generative Codec Transformer (ICLR 2025)
- Integrates with existing prosody conditioning pipeline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))
sys.path.insert(0, str(project_root / 'training'))

from maskgct_prosody import (
    MaskGCTConfig,
    MaskGCTProsodyModel,
    MaskGCTWithProsodyEncoder,
    MaskScheduler,
)
from prosody_conditioning import ProsodyConfig, extract_prosody_for_conditioning


class MaskGCTDataset(Dataset):
    """
    Dataset for MaskGCT training.

    Loads prosody features from manifest and provides them in dict format
    suitable for masked prediction training.
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        config: MaskGCTConfig,
    ):
        self.manifest_path = Path(manifest_path)
        self.prosody_cache_dir = Path(prosody_cache_dir)
        self.config = config

        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.prosody_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loaded {len(self.samples)} samples for MaskGCT training")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get prosody from cache or extract
        prosody = self._get_prosody(sample, idx)

        # Get emotion label if available
        emotion_label = self._get_emotion_label(sample)

        return {
            'prosody_semantic': prosody['semantic'].squeeze(0),
            'prosody_acoustic': prosody['acoustic'].squeeze(0),
            'prosody_rhythm': prosody['rhythm'].squeeze(0),
            'prosody_contour': prosody['contour'].squeeze(0),
            'emotion_label': emotion_label,
        }

    def _get_prosody(self, sample: dict, idx: int) -> Dict[str, torch.Tensor]:
        """Get prosody from cache or extract."""
        audio_path = sample.get('audio_path') or sample.get('path') or sample.get('audio')
        if not audio_path:
            return self._default_prosody()

        import hashlib
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        cache_path = self.prosody_cache_dir / f"prosody_{path_hash}.pt"

        if cache_path.exists():
            return torch.load(cache_path)

        if 'prosody' in sample:
            prosody = extract_prosody_for_conditioning(sample['prosody'])
            torch.save(prosody, cache_path)
            return prosody

        return self._default_prosody()

    def _default_prosody(self) -> Dict[str, torch.Tensor]:
        """Return default prosody if extraction fails."""
        return {
            'semantic': torch.zeros(1, self.config.semantic_dim),
            'acoustic': torch.zeros(1, self.config.acoustic_dim),
            'rhythm': torch.zeros(1, self.config.rhythm_dim),
            'contour': torch.zeros(1, self.config.contour_dim),
        }

    EMOTION_TO_IDX = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
        'excited': 8, 'contempt': 9,
    }

    def _get_emotion_label(self, sample: dict) -> int:
        """Extract emotion label index from sample metadata."""
        emotion = sample.get('emotion', '').lower()
        if not emotion:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion', '').lower()
        if not emotion:
            emotions = sample.get('prosody', {}).get('semantic', {}).get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()
        return self.EMOTION_TO_IDX.get(emotion, -1)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch for MaskGCT training."""
    return {
        'prosody_semantic': torch.stack([item['prosody_semantic'] for item in batch]),
        'prosody_acoustic': torch.stack([item['prosody_acoustic'] for item in batch]),
        'prosody_rhythm': torch.stack([item['prosody_rhythm'] for item in batch]),
        'prosody_contour': torch.stack([item['prosody_contour'] for item in batch]),
        'emotion_label': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long),
    }


class MaskGCTTrainer:
    """
    Trainer for MaskGCT prosody model.

    Handles:
    - Mask scheduling during training
    - Loss computation on masked positions only
    - Validation with parallel generation
    - Speed benchmarking vs autoregressive
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()

        # Create MaskGCT config
        self.maskgct_config = MaskGCTConfig(
            hidden_size=config.get('hidden_size', 768),
            num_semantic_tokens=config.get('num_semantic_tokens', 32),
            num_acoustic_tokens=config.get('num_acoustic_tokens', 128),
            semantic_dim=config.get('semantic_dim', 8),
            acoustic_dim=config.get('acoustic_dim', 12),
            rhythm_dim=config.get('rhythm_dim', 8),
            contour_dim=config.get('contour_dim', 64),
            max_prosody_length=config.get('max_prosody_length', 64),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            feedforward_dim=config.get('feedforward_dim', 3072),
            dropout=config.get('dropout', 0.1),
            initial_mask_ratio=config.get('initial_mask_ratio', 0.8),
            final_mask_ratio=config.get('final_mask_ratio', 0.15),
            mask_schedule=config.get('mask_schedule', 'cosine'),
            num_parallel_iterations=config.get('num_parallel_iterations', 8),
            temperature=config.get('temperature', 0.8),
        )

        # Create model
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()

        # Mask scheduler (for logging mask ratio progress)
        total_steps = config.get('num_epochs', 30) * config.get('steps_per_epoch', 100)
        self.mask_scheduler = MaskScheduler(
            initial_ratio=self.maskgct_config.initial_mask_ratio,
            final_ratio=self.maskgct_config.final_mask_ratio,
            total_steps=total_steps,
            schedule=self.maskgct_config.mask_schedule,
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _setup_model(self) -> MaskGCTWithProsodyEncoder:
        """Setup MaskGCT model with prosody encoder bridge."""
        model = MaskGCTWithProsodyEncoder(
            self.maskgct_config,
            hidden_size=self.config.get('prosody_hidden_size', 2048),
            num_prosody_tokens=self.config.get('num_prosody_tokens', 4),
        )
        model = model.to(self.device)

        # Print parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"MaskGCT model parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable:,}")

        return model

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with masked prediction loss."""
        self.model.train()

        # Move to device
        prosody_dict = {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }

        self.optimizer.zero_grad()

        # Forward with masked prediction loss
        loss_dict = self.model(prosody_dict)

        # Backward
        loss = loss_dict['loss']
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get('max_grad_norm', 1.0)
        )

        self.optimizer.step()
        self.global_step += 1

        return {
            'loss': loss_dict['loss'].item(),
            'semantic_loss': loss_dict['semantic_loss'].item(),
            'acoustic_loss': loss_dict['acoustic_loss'].item(),
            'commitment_loss': loss_dict['commitment_loss'].item(),
            'mask_ratio': loss_dict['mask_ratio'].item(),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation with parallel generation test."""
        self.model.eval()

        total_loss = 0.0
        total_semantic_loss = 0.0
        total_acoustic_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                prosody_dict = {
                    'semantic': batch['prosody_semantic'].to(self.device),
                    'acoustic': batch['prosody_acoustic'].to(self.device),
                    'rhythm': batch['prosody_rhythm'].to(self.device),
                    'contour': batch['prosody_contour'].to(self.device),
                }

                loss_dict = self.model(prosody_dict)

                total_loss += loss_dict['loss'].item()
                total_semantic_loss += loss_dict['semantic_loss'].item()
                total_acoustic_loss += loss_dict['acoustic_loss'].item()
                num_batches += 1

        return {
            'val_loss': total_loss / max(1, num_batches),
            'val_semantic_loss': total_semantic_loss / max(1, num_batches),
            'val_acoustic_loss': total_acoustic_loss / max(1, num_batches),
        }

    def benchmark_inference_speed(self, batch_size: int = 1, num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark parallel vs autoregressive inference speed.

        MaskGCT parallel decoding should be 2x+ faster than autoregressive baseline.

        The speedup comes from:
        - Parallel: 4-8 iterations, each predicting ALL tokens
        - AR: seq_len iterations, each predicting ONE token

        For seq_len=64 and num_iterations=4:
        - Parallel: ~4 full forward passes
        - AR: ~64 forward passes (but with smaller context)

        Expected speedup: ~2-5x depending on hardware and batch size.
        """
        self.model.eval()
        device = self.device

        # Create dummy prompt
        prompt_embeds = torch.randn(
            batch_size, 4, self.maskgct_config.hidden_size,
            device=device
        )

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = self.model.generate_prefix_parallel(prompt_embeds, batch_size)

        # Benchmark parallel generation (MaskGCT approach)
        parallel_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model.generate_prefix_parallel(prompt_embeds, batch_size)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                parallel_times.append(time.time() - start)

        parallel_mean = sum(parallel_times) / len(parallel_times) * 1000  # ms

        # Simulate autoregressive baseline with ACTUAL transformer forward passes
        # This gives a realistic estimate of AR inference time
        seq_len = self.maskgct_config.max_prosody_length
        ar_times = []

        # Create a full transformer encoder for AR simulation (matching MaskGCT depth)
        ar_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.maskgct_config.hidden_size,
                nhead=self.maskgct_config.num_heads,
                dim_feedforward=self.maskgct_config.feedforward_dim,
                dropout=0.0,  # No dropout for inference
                batch_first=True,
            ),
            num_layers=self.maskgct_config.num_layers,
        ).to(device)
        ar_encoder.eval()

        # Token embedding for AR simulation
        ar_token_embed = nn.Embedding(256, self.maskgct_config.hidden_size).to(device)

        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                # Simulate AR: generate tokens one at a time with growing context
                # Start with prompt tokens
                context = prompt_embeds.clone()

                for step in range(seq_len):
                    # Each AR step does a full forward pass on the growing context
                    _ = ar_encoder(context)

                    # Simulate appending a new token to context
                    new_token = torch.randint(0, 256, (batch_size, 1), device=device)
                    new_embed = ar_token_embed(new_token)
                    context = torch.cat([context, new_embed], dim=1)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    torch.mps.synchronize()
                ar_times.append(time.time() - start)

        ar_mean = sum(ar_times) / len(ar_times) * 1000  # ms

        # Speedup calculation
        speedup = ar_mean / parallel_mean if parallel_mean > 0 else 0

        # Also compute tokens-per-second for comparison
        num_tokens = seq_len * 2  # semantic + acoustic tokens
        parallel_tokens_per_sec = num_tokens / (parallel_mean / 1000) if parallel_mean > 0 else 0
        ar_tokens_per_sec = num_tokens / (ar_mean / 1000) if ar_mean > 0 else 0

        return {
            'parallel_time_ms': parallel_mean,
            'ar_baseline_time_ms': ar_mean,
            'speedup': speedup,
            'parallel_tokens_per_sec': parallel_tokens_per_sec,
            'ar_tokens_per_sec': ar_tokens_per_sec,
            'num_iterations': self.maskgct_config.num_parallel_iterations,
            'seq_len': seq_len,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
    ):
        """Main training loop."""
        print(f"\nStarting MaskGCT training for {num_epochs} epochs")
        print(f"Mask schedule: {self.maskgct_config.mask_schedule}")
        print(f"Initial mask ratio: {self.maskgct_config.initial_mask_ratio}")
        print(f"Final mask ratio: {self.maskgct_config.final_mask_ratio}")

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])

                if self.global_step % self.config.get('log_every', 10) == 0:
                    avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                    print(f"Epoch {epoch+1}, Step {self.global_step}: "
                          f"loss={avg_loss:.4f} "
                          f"(sem={metrics['semantic_loss']:.4f}, "
                          f"aco={metrics['acoustic_loss']:.4f}, "
                          f"mask_ratio={metrics['mask_ratio']:.2f})")

            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                print(f"Validation: loss={val_metrics['val_loss']:.4f}")

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best')

            # Benchmark inference speed every 5 epochs
            if (epoch + 1) % 5 == 0:
                speed_metrics = self.benchmark_inference_speed()
                print(f"Inference speed: parallel={speed_metrics['parallel_time_ms']:.2f}ms, "
                      f"AR baseline={speed_metrics['ar_baseline_time_ms']:.2f}ms, "
                      f"speedup={speed_metrics['speedup']:.2f}x")

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

        self.save_checkpoint('final')

        # Final speed benchmark
        print("\n" + "=" * 50)
        print("Final Inference Speed Benchmark")
        print("=" * 50)
        speed_metrics = self.benchmark_inference_speed(num_runs=20)
        print(f"Parallel generation: {speed_metrics['parallel_time_ms']:.2f}ms")
        print(f"AR baseline (simulated): {speed_metrics['ar_baseline_time_ms']:.2f}ms")
        print(f"Speedup: {speed_metrics['speedup']:.2f}x")

        print("\nMaskGCT training complete!")

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        output_dir = Path(self.config.get('output_dir', 'checkpoints/maskgct'))
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'maskgct_config': {
                'hidden_size': self.maskgct_config.hidden_size,
                'num_semantic_tokens': self.maskgct_config.num_semantic_tokens,
                'num_acoustic_tokens': self.maskgct_config.num_acoustic_tokens,
                'num_layers': self.maskgct_config.num_layers,
                'num_heads': self.maskgct_config.num_heads,
                'initial_mask_ratio': self.maskgct_config.initial_mask_ratio,
                'final_mask_ratio': self.maskgct_config.final_mask_ratio,
                'num_parallel_iterations': self.maskgct_config.num_parallel_iterations,
            },
        }

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/maskgct_prosody.yaml')
    parser.add_argument('--manifest', type=str, help='Path to training manifest')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'train_manifest': '../data/splits/train.json',
            'val_manifest': '../data/splits/val.json',
            'prosody_cache_dir': '../data/prosody_cache',
            'output_dir': '../models/checkpoints/maskgct',
            'num_epochs': 30,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'hidden_size': 768,
            'num_layers': 6,
            'num_heads': 8,
            'initial_mask_ratio': 0.8,
            'final_mask_ratio': 0.15,
            'mask_schedule': 'cosine',
            'num_parallel_iterations': 8,
        }

    print("MaskGCT Prosody Training")
    print("=" * 50)
    print(f"Config: {config}")

    # Create trainer
    trainer = MaskGCTTrainer(config)

    # Create datasets
    manifest_path = args.manifest or config.get('train_manifest')
    if manifest_path and Path(manifest_path).exists():
        train_dataset = MaskGCTDataset(
            manifest_path,
            config.get('prosody_cache_dir', '../data/prosody_cache'),
            trainer.maskgct_config,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0),
        )

        # Validation loader
        val_loader = None
        val_manifest = config.get('val_manifest')
        if val_manifest and Path(val_manifest).exists():
            val_dataset = MaskGCTDataset(
                val_manifest,
                config.get('prosody_cache_dir', '../data/prosody_cache'),
                trainer.maskgct_config,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('batch_size', 8),
                shuffle=False,
                collate_fn=collate_fn,
            )
            print(f"Validation set: {len(val_dataset)} samples")

        trainer.train(
            train_loader,
            val_loader=val_loader,
            num_epochs=config.get('num_epochs', 30),
        )
    else:
        print("\nNo manifest provided or file not found.")
        print(f"Expected: {manifest_path}")
        print("\nModule loaded successfully - ready for training when data is available.")

        # Run a quick test
        print("\nRunning inference speed benchmark...")
        speed_metrics = trainer.benchmark_inference_speed(num_runs=5)
        print(f"Parallel generation: {speed_metrics['parallel_time_ms']:.2f}ms")
        print(f"Speedup vs AR baseline: {speed_metrics['speedup']:.2f}x")


if __name__ == "__main__":
    main()
