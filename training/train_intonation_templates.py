#!/usr/bin/env python3
"""
Training script for Intonation Template Clustering (Into-TTS approach).

Two-phase training:
1. Template Discovery: Extract F0 contours and cluster into templates
2. Encoder Training: Train template encoder and predictor

Usage:
    # Phase 1: Discover templates from dataset
    python train_intonation_templates.py --discover \
        --manifest ../data/manifest.json \
        --output ../checkpoints/intonation_templates

    # Phase 2: Train encoder and predictor
    python train_intonation_templates.py --config config/intonation_templates.yaml

    # Resume training
    python train_intonation_templates.py --config config/intonation_templates.yaml \
        --resume ../checkpoints/intonation_templates/checkpoint_latest.pt

    # Test mode
    python train_intonation_templates.py --test

References:
    - Into-TTS: https://arxiv.org/abs/2204.01271
"""

import argparse
import json
import glob
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import yaml

from intonation_templates import (
    IntonationTemplateConfig,
    IntonationTemplateAdapter,
    IntonationTemplateClustering,
    IntonationTemplateLoss,
    F0Extractor,
    extract_templates_from_dataset,
    visualize_templates,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""

    # Paths
    manifest_path: str = "../data/manifest.json"
    output_dir: str = "../checkpoints/intonation_templates"
    clustering_path: str = ""  # Path to pre-computed clustering

    # Data
    sample_rate: int = 24000
    max_audio_duration: float = 10.0  # Max audio duration in seconds

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    gradient_clip: float = 1.0

    # Validation
    val_split: float = 0.1
    val_every: int = 500  # Validate every N steps

    # Checkpointing
    save_every: int = 1000
    keep_last_n: int = 3

    # Model
    intonation_config: IntonationTemplateConfig = field(
        default_factory=IntonationTemplateConfig
    )

    # Text encoder (for predictor training)
    text_encoder: str = "phoneme"  # "phoneme" or "bert"
    text_embed_dim: int = 256

    # Logging
    log_every: int = 50

    # Hardware
    device: str = "auto"
    num_workers: int = 4


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    # Extract intonation config
    intonation_dict = yaml_config.pop('intonation', {})
    intonation_config = IntonationTemplateConfig(**intonation_dict)

    # Create training config
    config = TrainingConfig(**yaml_config, intonation_config=intonation_config)

    return config


# =============================================================================
# DATASET
# =============================================================================

class IntonationTemplateDataset(Dataset):
    """
    Dataset for intonation template training.

    Loads audio files, extracts F0 contours, and provides:
    - Normalized F0 contour
    - Template index (from clustering)
    - Text features (for predictor training)
    """

    def __init__(
        self,
        manifest: List[Dict],
        config: TrainingConfig,
        clustering: IntonationTemplateClustering,
        f0_extractor: F0Extractor,
        text_encoder: Optional[nn.Module] = None,
        is_train: bool = True,
    ):
        self.manifest = manifest
        self.config = config
        self.clustering = clustering
        self.f0_extractor = f0_extractor
        self.text_encoder = text_encoder
        self.is_train = is_train

        # Cache for extracted contours
        self._contour_cache = {}
        self._template_cache = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]

        # Load audio
        audio_path = item['audio_path']

        # Extract or cache F0 contour
        if audio_path in self._contour_cache:
            contour = self._contour_cache[audio_path]
            template_idx = self._template_cache[audio_path]
        else:
            try:
                import torchaudio
                audio, sr = torchaudio.load(audio_path)

                # Resample if needed
                if sr != self.config.sample_rate:
                    audio = torchaudio.functional.resample(
                        audio, sr, self.config.sample_rate
                    )

                # Truncate if too long
                max_samples = int(self.config.max_audio_duration * self.config.sample_rate)
                if audio.shape[1] > max_samples:
                    audio = audio[:, :max_samples]

                # Extract and normalize contour
                contour = self.f0_extractor.process_audio(audio, self.config.sample_rate)

                # Get template assignment
                template_idx = self.clustering.predict(contour.reshape(1, -1))[0]

                # Cache
                self._contour_cache[audio_path] = contour
                self._template_cache[audio_path] = template_idx

            except Exception as e:
                print(f"Warning: Failed to process {audio_path}: {e}")
                # Return zeros as fallback
                contour = np.zeros(self.config.intonation_config.contour_length)
                template_idx = 0

        # Convert to tensors
        contour_tensor = torch.from_numpy(contour).float()
        template_tensor = torch.tensor(template_idx, dtype=torch.long)

        result = {
            'contour': contour_tensor,
            'template_index': template_tensor,
        }

        # Add text features if encoder available
        if self.text_encoder is not None and 'text' in item:
            text = item['text']
            # Simple character encoding (replace with proper phoneme/BERT in production)
            text_ids = [ord(c) % 256 for c in text[:100]]  # Truncate to 100 chars
            text_ids = text_ids + [0] * (100 - len(text_ids))  # Pad
            result['text_ids'] = torch.tensor(text_ids, dtype=torch.long)
            result['text_mask'] = torch.tensor(
                [1] * min(len(text), 100) + [0] * max(0, 100 - len(text)),
                dtype=torch.bool
            )

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    result = {
        'contour': torch.stack([b['contour'] for b in batch]),
        'template_index': torch.stack([b['template_index'] for b in batch]),
    }

    if 'text_ids' in batch[0]:
        result['text_ids'] = torch.stack([b['text_ids'] for b in batch])
        result['text_mask'] = torch.stack([b['text_mask'] for b in batch])

    return result


# =============================================================================
# SIMPLE TEXT ENCODER
# =============================================================================

class SimpleTextEncoder(nn.Module):
    """
    Simple character-level text encoder.

    For production, replace with:
    - Phoneme encoder (G2P + embedding)
    - Pre-trained BERT/RoBERTa
    """

    def __init__(self, vocab_size: int = 256, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            text_ids: [batch, seq_len]
            text_mask: [batch, seq_len]

        Returns:
            Text embeddings [batch, seq_len, embed_dim]
        """
        x = self.embedding(text_ids)

        if text_mask is not None:
            # TransformerEncoder expects True = masked
            src_key_padding_mask = ~text_mask
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x


# =============================================================================
# TRAINING LOOP
# =============================================================================

class IntonationTemplateTrainer:
    """Trainer for intonation template system."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.f0_extractor = F0Extractor(config.intonation_config)
        self.clustering = None
        self.adapter = None
        self.text_encoder = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def discover_templates(
        self,
        manifest_path: str,
        output_path: Optional[str] = None,
    ) -> IntonationTemplateClustering:
        """
        Discover intonation templates from dataset.

        Args:
            manifest_path: Path to manifest JSON file
            output_path: Optional path to save clustering model

        Returns:
            Fitted clustering model
        """
        print("\n" + "=" * 60)
        print("Phase 1: Template Discovery")
        print("=" * 60)

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        if isinstance(manifest, dict):
            manifest = manifest.get('samples', manifest.get('data', []))

        audio_paths = [item['audio_path'] for item in manifest if 'audio_path' in item]
        print(f"Found {len(audio_paths)} audio files")

        # Extract templates
        self.clustering = extract_templates_from_dataset(
            audio_paths,
            self.config.intonation_config,
            verbose=True,
        )

        # Save clustering
        if output_path is None:
            output_path = self.output_dir / "clustering.pkl"
        self.clustering.save(output_path)
        print(f"Saved clustering model to {output_path}")

        # Visualize templates
        viz_path = self.output_dir / "templates_visualization.png"
        visualize_templates(self.clustering, save_path=str(viz_path))

        # Print descriptions
        print("\nDiscovered Templates:")
        for desc in self.clustering.describe_templates():
            print(f"  {desc}")

        return self.clustering

    def setup_training(
        self,
        manifest_path: str,
        clustering_path: Optional[str] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training components.

        Args:
            manifest_path: Path to manifest file
            clustering_path: Path to pre-computed clustering model

        Returns:
            (train_loader, val_loader)
        """
        print("\n" + "=" * 60)
        print("Phase 2: Training Setup")
        print("=" * 60)

        # Load clustering
        if clustering_path and Path(clustering_path).exists():
            self.clustering = IntonationTemplateClustering(self.config.intonation_config)
            self.clustering.load(clustering_path)
            print(f"Loaded clustering from {clustering_path}")
        elif self.clustering is None:
            # Try default path
            default_path = self.output_dir / "clustering.pkl"
            if default_path.exists():
                self.clustering = IntonationTemplateClustering(self.config.intonation_config)
                self.clustering.load(default_path)
                print(f"Loaded clustering from {default_path}")
            else:
                raise RuntimeError(
                    "No clustering model found. Run with --discover first."
                )

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        if isinstance(manifest, dict):
            manifest = manifest.get('samples', manifest.get('data', []))

        # Split into train/val
        random.shuffle(manifest)
        val_size = int(len(manifest) * self.config.val_split)
        train_manifest = manifest[val_size:]
        val_manifest = manifest[:val_size]

        print(f"Train samples: {len(train_manifest)}")
        print(f"Val samples: {len(val_manifest)}")

        # Create text encoder
        self.text_encoder = SimpleTextEncoder(
            embed_dim=self.config.text_embed_dim
        ).to(self.device)

        # Create datasets
        train_dataset = IntonationTemplateDataset(
            train_manifest,
            self.config,
            self.clustering,
            self.f0_extractor,
            self.text_encoder,
            is_train=True,
        )

        val_dataset = IntonationTemplateDataset(
            val_manifest,
            self.config,
            self.clustering,
            self.f0_extractor,
            self.text_encoder,
            is_train=False,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Create adapter
        self.config.intonation_config.text_embed_dim = self.config.text_embed_dim
        self.adapter = IntonationTemplateAdapter(
            self.config.intonation_config
        ).to(self.device)

        # Load clustering into adapter
        self.adapter.clustering = self.clustering
        self.adapter.module.load_templates(self.clustering)

        # Create optimizer
        params = list(self.adapter.parameters()) + list(self.text_encoder.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Create scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps,
        )

        # Create loss function
        self.loss_fn = IntonationTemplateLoss(self.config.intonation_config)

        # Print model info
        total_params = sum(p.numel() for p in self.adapter.parameters())
        trainable_params = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"\nAdapter parameters: {total_params:,} ({trainable_params:,} trainable)")

        return train_loader, val_loader

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        self.adapter.train()
        self.text_encoder.train()

        # Move to device
        contour = batch['contour'].to(self.device)
        template_index = batch['template_index'].to(self.device)

        # Encode text
        text_ids = batch.get('text_ids')
        text_mask = batch.get('text_mask')

        if text_ids is not None:
            text_ids = text_ids.to(self.device)
            text_mask = text_mask.to(self.device)
            text_emb = self.text_encoder(text_ids, text_mask)
        else:
            # No text - use random embeddings (for testing)
            batch_size = contour.shape[0]
            text_emb = torch.randn(
                batch_size, 20, self.config.text_embed_dim,
                device=self.device
            )
            text_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=self.device)

        # Forward pass
        result = self.adapter.module(
            text_embeddings=text_emb,
            text_mask=text_mask,
            return_contour=True,
        )

        # Compute loss
        losses = self.loss_fn(result, template_index, contour)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.adapter.parameters()) + list(self.text_encoder.parameters()),
                self.config.gradient_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        # Return metrics
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                   for k, v in losses.items()}
        metrics['lr'] = self.scheduler.get_last_lr()[0]

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.adapter.eval()
        self.text_encoder.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_recon = 0.0
        num_batches = 0

        for batch in val_loader:
            contour = batch['contour'].to(self.device)
            template_index = batch['template_index'].to(self.device)

            text_ids = batch.get('text_ids')
            text_mask = batch.get('text_mask')

            if text_ids is not None:
                text_ids = text_ids.to(self.device)
                text_mask = text_mask.to(self.device)
                text_emb = self.text_encoder(text_ids, text_mask)
            else:
                batch_size = contour.shape[0]
                text_emb = torch.randn(
                    batch_size, 20, self.config.text_embed_dim,
                    device=self.device
                )
                text_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=self.device)

            result = self.adapter.module(
                text_embeddings=text_emb,
                text_mask=text_mask,
                return_contour=True,
            )

            losses = self.loss_fn(result, template_index, contour)

            total_loss += losses['total'].item()
            total_acc += losses.get('accuracy', torch.tensor(0.0)).item()
            total_recon += losses.get('reconstruction_loss', torch.tensor(0.0)).item()
            num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_acc / num_batches,
            'val_reconstruction': total_recon / num_batches,
        }

    def save_checkpoint(self, name: str, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'adapter_state_dict': self.adapter.state_dict(),
            'text_encoder_state_dict': self.text_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        path = self.output_dir / name
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Clean up old checkpoints
        checkpoints = sorted(
            self.output_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for old_ckpt in checkpoints[self.config.keep_last_n:]:
            old_ckpt.unlink()

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Optional[str] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_path: Optional checkpoint path to resume from
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)

        num_epochs = self.config.num_epochs
        log_every = self.config.log_every
        val_every = self.config.val_every
        save_every = self.config.save_every

        start_time = time.time()
        running_loss = 0.0
        running_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                running_loss += metrics['total']
                running_acc += metrics.get('accuracy', 0.0)

                # Logging
                if self.global_step % log_every == 0:
                    avg_loss = running_loss / log_every
                    avg_acc = running_acc / log_every
                    elapsed = time.time() - start_time

                    print(
                        f"  Step {self.global_step:6d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Acc: {avg_acc:.4f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )

                    running_loss = 0.0
                    running_acc = 0.0

                # Validation
                if self.global_step % val_every == 0:
                    val_metrics = self.validate(val_loader)
                    print(
                        f"  Validation | "
                        f"Loss: {val_metrics['val_loss']:.4f} | "
                        f"Acc: {val_metrics['val_accuracy']:.4f} | "
                        f"Recon: {val_metrics['val_reconstruction']:.4f}"
                    )

                    # Save best
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(
                            f"checkpoint_step_{self.global_step}.pt",
                            is_best=True
                        )

                # Save checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        # Final save
        self.save_checkpoint("checkpoint_final.pt")
        print("\nTraining complete!")


# =============================================================================
# TEST MODE
# =============================================================================

def run_test():
    """Run test with synthetic data."""
    print("=" * 70)
    print("Intonation Template Training - Test Mode")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Check sklearn availability
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("\nWARNING: scikit-learn not installed. Skipping clustering tests.")
        print("Install with: pip install scikit-learn")
        print("\nRunning encoder-only tests...")
        run_encoder_only_test(device)
        return


def run_encoder_only_test(device: str):
    """Run minimal test without sklearn."""
    from intonation_templates import (
        IntonationTemplateConfig,
        IntonationEncoder,
        IntonationPredictor,
        IntonationTemplateModule,
    )

    config = IntonationTemplateConfig(
        num_templates=4,
        contour_length=30,
    )

    # Test encoder
    print("\n[Test] Intonation Encoder...")
    encoder = IntonationEncoder(config).to(device)
    template_indices = torch.randint(0, config.num_templates, (4,), device=device)
    result = encoder(template_indices=template_indices)
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    assert result['prosody_tokens'].shape == (4, config.num_prosody_tokens, config.output_dim)
    print("  [PASS]")

    # Test predictor
    print("\n[Test] Intonation Predictor...")
    predictor = IntonationPredictor(config).to(device)
    text_emb = torch.randn(4, 20, config.text_embed_dim, device=device)
    text_mask = torch.ones(4, 20, dtype=torch.bool, device=device)
    pred = predictor(text_emb, text_mask)
    print(f"  Predicted templates: {pred['predicted'].tolist()}")
    print("  [PASS]")

    # Test module
    print("\n[Test] Complete Module...")
    module = IntonationTemplateModule(config).to(device)
    result = module(text_embeddings=text_emb, text_mask=text_mask, return_contour=True)
    print(f"  Tokens shape: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("Encoder-only tests passed!")
    print("Install scikit-learn for full clustering tests.")
    print("=" * 70)

    # Create config
    config = TrainingConfig(
        batch_size=4,
        num_epochs=2,
        val_every=10,
        save_every=20,
        log_every=5,
        output_dir="/tmp/intonation_test",
    )

    config.intonation_config.num_templates = 4
    config.intonation_config.contour_length = 30

    # Create trainer
    trainer = IntonationTemplateTrainer(config)

    # Generate synthetic data
    print("\nGenerating synthetic data...")

    np.random.seed(42)
    n_samples = 100
    contour_length = config.intonation_config.contour_length

    # Generate contours with known patterns
    contours = []
    labels = []

    patterns = ['rising', 'falling', 'flat', 'rise-fall']
    for i in range(n_samples):
        pattern = patterns[i % len(patterns)]
        t = np.linspace(0, 1, contour_length)

        if pattern == 'rising':
            contour = t + np.random.randn(contour_length) * 0.1
        elif pattern == 'falling':
            contour = -t + np.random.randn(contour_length) * 0.1
        elif pattern == 'flat':
            contour = np.random.randn(contour_length) * 0.1
        else:  # rise-fall
            contour = np.sin(np.pi * t) + np.random.randn(contour_length) * 0.1

        contours.append(contour)
        labels.append(i % len(patterns))

    contours = np.array(contours)
    labels = np.array(labels)

    # Fit clustering
    print("\nFitting clustering...")
    trainer.clustering = IntonationTemplateClustering(config.intonation_config)
    trainer.clustering.fit(contours)

    print("\nTemplate descriptions:")
    for desc in trainer.clustering.describe_templates():
        print(f"  {desc}")

    # Create adapter
    trainer.text_encoder = SimpleTextEncoder(
        embed_dim=config.text_embed_dim
    ).to(device)

    trainer.adapter = IntonationTemplateAdapter(
        config.intonation_config
    ).to(device)

    trainer.adapter.clustering = trainer.clustering
    trainer.adapter.module.load_templates(trainer.clustering)

    # Create loss function
    trainer.loss_fn = IntonationTemplateLoss(config.intonation_config)

    # Simple training test
    print("\nRunning training test...")

    # Create optimizer
    params = list(trainer.adapter.parameters()) + list(trainer.text_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    # Mini training loop
    for step in range(10):
        # Random batch
        batch_idx = np.random.choice(len(contours), size=config.batch_size)

        contour_batch = torch.from_numpy(contours[batch_idx]).float().to(device)
        label_batch = torch.from_numpy(labels[batch_idx]).long().to(device)

        # Random text
        text_emb = torch.randn(
            config.batch_size, 20, config.text_embed_dim,
            device=device
        )
        text_mask = torch.ones(config.batch_size, 20, dtype=torch.bool, device=device)

        # Forward
        result = trainer.adapter.module(
            text_embeddings=text_emb,
            text_mask=text_mask,
            return_contour=True,
        )

        # Loss
        losses = trainer.loss_fn(result, label_batch, contour_batch)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        if step % 2 == 0:
            print(
                f"  Step {step}: "
                f"Loss={losses['total'].item():.4f}, "
                f"Acc={losses.get('accuracy', torch.tensor(0)).item():.4f}"
            )

    # Test inference
    print("\nTesting inference...")

    # Manual template
    result = trainer.adapter.from_template(template_index=0, batch_size=2)
    print(f"  From template 0: tokens shape = {result['prosody_tokens'].shape}")

    # From text
    result = trainer.adapter.from_text(text_emb[:2], text_mask[:2])
    print(f"  From text: predicted = {result['predicted_template'].tolist()}")

    # Template to sketch
    sketch = trainer.adapter.template_to_sketch(0)
    print(f"  Template 0 sketch: pitch={sketch['pitch_sketch'].shape}, energy={sketch['energy_sketch'].shape}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Intonation Template Clustering (Into-TTS approach)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/intonation_templates.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run template discovery phase",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest file (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default=None,
        help="Path to pre-computed clustering model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode with synthetic data",
    )

    args = parser.parse_args()

    if args.test:
        run_test()
        return

    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = TrainingConfig()

    # Override from command line
    if args.manifest:
        config.manifest_path = args.manifest
    if args.output:
        config.output_dir = args.output
    if args.clustering:
        config.clustering_path = args.clustering

    # Create trainer
    trainer = IntonationTemplateTrainer(config)

    # Run template discovery if requested
    if args.discover:
        trainer.discover_templates(config.manifest_path)
        print("\nTemplate discovery complete. Run again without --discover to train.")
        return

    # Setup and train
    train_loader, val_loader = trainer.setup_training(
        config.manifest_path,
        config.clustering_path or None,
    )

    trainer.train(train_loader, val_loader, args.resume)


if __name__ == "__main__":
    main()
