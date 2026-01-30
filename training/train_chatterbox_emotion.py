#!/usr/bin/env python3
"""
Training script for Chatterbox-style emotion exaggeration.

Based on Chatterbox by Resemble AI (December 2025).

Features:
1. Single-parameter emotion exaggeration (0.0=monotone to 2.0=dramatic)
2. Native paralinguistic tags ([laugh], [sigh], [cough], etc.)
3. CFG weight control for pacing
4. Zero-shot voice cloning support

Usage:
    # Train full model
    python train_chatterbox_emotion.py --config config/chatterbox_emotion.yaml

    # Resume from checkpoint
    python train_chatterbox_emotion.py --config config/chatterbox_emotion.yaml \
      --resume ../checkpoints/chatterbox_emotion/latest.pt

    # Test mode (synthetic data)
    python train_chatterbox_emotion.py --test
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from chatterbox_emotion import (
    ChatterboxConfig,
    Chatterbox,
    ChatterboxAdapter,
    ChatterboxLoss,
    parse_paralinguistic_tags,
    exaggeration_to_description,
    suggest_cfg_weight,
    get_supported_tags,
    PARALINGUISTIC_TAGS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class ChatterboxDataset(Dataset):
    """Dataset for training Chatterbox emotion exaggeration."""

    def __init__(
        self,
        manifest_path: str,
        feature_extractor: str = "wav2vec2",
        max_audio_length: int = 500,
        max_seq_length: int = 1000,
        include_tags: bool = True,
    ):
        self.manifest_path = manifest_path
        self.feature_extractor = feature_extractor
        self.max_audio_length = max_audio_length
        self.max_seq_length = max_seq_length
        self.include_tags = include_tags

        # Load manifest
        self.samples = []
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        else:
            logger.warning(f"Manifest not found: {manifest_path}")

        # Extract unique speakers for speaker preservation loss
        self.speakers = list(set(s.get('speaker_id', 'unknown') for s in self.samples))
        self.speaker_to_idx = {s: i for i, s in enumerate(self.speakers)}

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        if not self.samples:
            # Return synthetic data for testing
            return self._create_synthetic_sample()

        sample = self.samples[idx % len(self.samples)]

        # Extract features (placeholder - would use actual feature extraction)
        features = torch.randn(self.max_audio_length, 768)

        # Get text with potential tags
        text = sample.get('text', 'Hello, how are you?')

        # Parse any paralinguistic tags
        clean_text, parsed_tags = parse_paralinguistic_tags(text)

        # Get emotion info
        emotion = sample.get('emotion', 'neutral')
        exaggeration = sample.get('exaggeration', 0.5)

        # Speaker ID
        speaker_id = sample.get('speaker_id', 'unknown')
        speaker_idx = self.speaker_to_idx.get(speaker_id, 0)

        # Target prosody (placeholder)
        target_prosody = torch.randn(4, 2048)

        return {
            'features': features,
            'text': text,
            'clean_text': clean_text,
            'num_tags': len(parsed_tags),
            'emotion': emotion,
            'exaggeration': exaggeration,
            'speaker_idx': speaker_idx,
            'target_prosody': target_prosody,
        }

    def _create_synthetic_sample(self):
        """Create synthetic sample for testing."""
        return {
            'features': torch.randn(self.max_audio_length, 768),
            'text': 'Hello [laugh], how are you today?',
            'clean_text': 'Hello, how are you today?',
            'num_tags': 1,
            'emotion': random.choice(['happy', 'sad', 'neutral', 'angry']),
            'exaggeration': random.uniform(0.0, 2.0),
            'speaker_idx': 0,
            'target_prosody': torch.randn(4, 2048),
        }


def collate_fn(batch):
    """Custom collate function for Chatterbox dataset."""
    features = torch.stack([b['features'] for b in batch])
    texts = [b['text'] for b in batch]
    clean_texts = [b['clean_text'] for b in batch]
    exaggerations = torch.tensor([b['exaggeration'] for b in batch])
    speaker_idxs = torch.tensor([b['speaker_idx'] for b in batch])
    target_prosody = torch.stack([b['target_prosody'] for b in batch])

    return {
        'features': features,
        'texts': texts,
        'clean_texts': clean_texts,
        'exaggerations': exaggerations,
        'speaker_idxs': speaker_idxs,
        'target_prosody': target_prosody,
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: Chatterbox,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ChatterboxLoss,
    device: torch.device,
    epoch: int,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0
    loss_components = {}

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        features = batch['features'].to(device)
        texts = batch['texts']
        exaggerations = batch['exaggerations'].to(device)
        target_prosody = batch['target_prosody'].to(device)
        speaker_idxs = batch['speaker_idxs'].to(device)

        # Forward pass with random exaggeration from batch
        exag = exaggerations[0].item()
        result = model(features, text=texts[0], exaggeration=exag)

        # Compute loss
        losses = loss_fn(
            result,
            target_prosody=target_prosody,
            speaker_labels=speaker_idxs,
        )

        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track losses
        total_loss += losses['total'].item()
        num_batches += 1

        for key, value in losses.items():
            if key not in loss_components:
                loss_components[key] = 0
            loss_components[key] += value.item()

        # Log progress
        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | Exag: {exag:.2f}"
            )

    # Average losses
    for key in loss_components:
        loss_components[key] /= num_batches

    return loss_components


def validate(
    model: Chatterbox,
    dataloader: DataLoader,
    loss_fn: ChatterboxLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0
    num_batches = 0
    loss_components = {}

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            texts = batch['texts']
            exaggerations = batch['exaggerations'].to(device)
            target_prosody = batch['target_prosody'].to(device)
            speaker_idxs = batch['speaker_idxs'].to(device)

            exag = exaggerations[0].item()
            result = model(features, text=texts[0], exaggeration=exag)

            losses = loss_fn(
                result,
                target_prosody=target_prosody,
                speaker_labels=speaker_idxs,
            )

            total_loss += losses['total'].item()
            num_batches += 1

            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value.item()

    for key in loss_components:
        loss_components[key] /= num_batches

    return loss_components


def evaluate_exaggeration_sweep(
    model: Chatterbox,
    features: torch.Tensor,
    device: torch.device,
    text: str = "Hello, how are you today?",
    levels: List[float] = None,
) -> Dict[str, any]:
    """Evaluate model at different exaggeration levels."""
    model.eval()

    if levels is None:
        levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    features = features.to(device)
    results = model.sweep_exaggeration(features, levels=levels, text=text)

    # Compute variance in prosody tokens across levels
    token_variances = []
    for i, tokens in enumerate(results['tokens']):
        var = tokens.var().item()
        token_variances.append((levels[i], var))

    return {
        'levels': levels,
        'token_variances': token_variances,
        'descriptions': [(l, exaggeration_to_description(l)) for l in levels],
        'suggested_cfg': [(l, suggest_cfg_weight(l)) for l in levels],
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(config_path: str, resume: str = None):
    """Main training function."""

    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create model config
    model_config = ChatterboxConfig(
        input_dim=config_dict['model'].get('input_dim', 768),
        hidden_dim=config_dict['model'].get('hidden_dim', 512),
        emotion_dim=config_dict['model'].get('emotion_dim', 256),
        output_dim=config_dict['model'].get('output_dim', 2048),
        speaker_dim=config_dict['model'].get('speaker_dim', 256),
        tag_embedding_dim=config_dict['model'].get('tag_embedding_dim', 128),
        default_exaggeration=config_dict['model'].get('default_exaggeration', 0.5),
        min_exaggeration=config_dict['model'].get('min_exaggeration', 0.0),
        max_exaggeration=config_dict['model'].get('max_exaggeration', 2.0),
        default_cfg_weight=config_dict['model'].get('default_cfg_weight', 0.5),
        use_single_step_decoder=config_dict['model'].get('use_single_step_decoder', True),
        decoder_hidden_dim=config_dict['model'].get('decoder_hidden_dim', 512),
        decoder_num_layers=config_dict['model'].get('decoder_num_layers', 4),
        decoder_num_heads=config_dict['model'].get('decoder_num_heads', 8),
        dropout=config_dict['model'].get('dropout', 0.1),
        use_layer_norm=config_dict['model'].get('use_layer_norm', True),
        num_prosody_tokens=config_dict['model'].get('num_prosody_tokens', 4),
        max_tags_per_utterance=config_dict['model'].get('max_tags_per_utterance', 10),
    )

    # Device setup
    device_str = config_dict.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = Chatterbox(model_config).to(device)
    logger.info(f"Created Chatterbox model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create loss function
    loss_fn = ChatterboxLoss(
        model_config,
        exag_consistency_weight=config_dict.get('exag_consistency_weight', 1.0),
        tag_weight=config_dict.get('tag_weight', 0.5),
        speaker_weight=config_dict.get('speaker_weight', 0.5),
        prosody_weight=config_dict.get('prosody_weight', 1.0),
    ).to(device)

    # Create datasets
    train_dataset = ChatterboxDataset(
        manifest_path=config_dict.get('train_manifest', '../data/emotion_train.json'),
        max_audio_length=config_dict.get('max_audio_length', 500),
        max_seq_length=config_dict.get('max_seq_length', 1000),
    )

    val_dataset = ChatterboxDataset(
        manifest_path=config_dict.get('val_manifest', '../data/emotion_val.json'),
        max_audio_length=config_dict.get('max_audio_length', 500),
        max_seq_length=config_dict.get('max_seq_length', 1000),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict.get('batch_size', 16),
        shuffle=True,
        num_workers=config_dict.get('num_workers', 4),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config_dict.get('batch_size', 16),
        shuffle=False,
        num_workers=config_dict.get('num_workers', 4),
        collate_fn=collate_fn,
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config_dict.get('learning_rate', 1e-4),
        weight_decay=config_dict.get('weight_decay', 0.01),
        betas=(config_dict.get('adam_beta1', 0.9), config_dict.get('adam_beta2', 0.999)),
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config_dict.get('epochs', 100),
        eta_min=config_dict.get('learning_rate', 1e-4) * 0.1,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if resume and os.path.exists(resume):
        logger.info(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Checkpoint directory
    checkpoint_dir = Path(config_dict.get('checkpoint_dir', '../checkpoints/chatterbox_emotion'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    epochs = config_dict.get('epochs', 100)
    save_every = config_dict.get('save_every', 5)
    log_every = config_dict.get('log_every', 100)
    eval_every = config_dict.get('eval_every', 500)

    for epoch in range(start_epoch, epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch + 1, log_every
        )

        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)

        # Update learning rate
        scheduler.step()

        # Log epoch results
        logger.info(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Evaluate exaggeration sweep
        if (epoch + 1) % 10 == 0:
            sample_features = torch.randn(1, 100, model_config.input_dim).to(device)
            sweep_results = evaluate_exaggeration_sweep(model, sample_features, device)
            logger.info(f"Exaggeration sweep variances: {sweep_results['token_variances'][:4]}...")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': model_config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_path = checkpoint_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': model_config,
            }, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Save latest
        latest_path = checkpoint_dir / "latest.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'config': model_config,
        }, latest_path)

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return model


# =============================================================================
# TEST MODE
# =============================================================================

def run_test():
    """Run test with synthetic data."""
    logger.info("Running Chatterbox emotion test mode...")

    # Create config
    config = ChatterboxConfig()

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Chatterbox(config).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test paralinguistic tag parsing
    test_texts = [
        "Hello [laugh], how are you today?",
        "I'm not sure about this [sigh].",
        "That's amazing [gasp]! Wow [chuckle]!",
        "[hmm] Let me think about that [um] for a moment.",
        "I'm so sorry [sob] about what happened.",
    ]

    logger.info("\nTesting paralinguistic tag parsing:")
    for text in test_texts:
        clean, tags = parse_paralinguistic_tags(text)
        tag_names = [t.tag_name for t in tags]
        logger.info(f"  Original: {text}")
        logger.info(f"  Clean: {clean}")
        logger.info(f"  Tags: {tag_names}")

    # Test exaggeration sweep
    logger.info("\nTesting exaggeration sweep:")
    features = torch.randn(1, 100, config.input_dim).to(device)

    with torch.no_grad():
        sweep = model.sweep_exaggeration(features, text="Hello [laugh], nice to meet you!")

    for level, tokens in zip(sweep['levels'], sweep['tokens']):
        desc = exaggeration_to_description(level)
        cfg = suggest_cfg_weight(level)
        variance = tokens.var().item()
        logger.info(f"  Exag {level:.2f} ({desc}): variance={variance:.4f}, suggested_cfg={cfg:.2f}")

    # Test full forward pass with tags
    logger.info("\nTesting forward pass with paralinguistic tags:")
    with torch.no_grad():
        result = model(
            features,
            text="Hi there [chuckle], have you got a minute to chat?",
            exaggeration=0.7,
            cfg_weight=0.45,
        )

    logger.info(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    logger.info(f"  Speaker embedding shape: {result['speaker_emb'].shape}")
    logger.info(f"  Emotion embedding shape: {result['emotion_emb'].shape}")
    logger.info(f"  Exaggerated emotion shape: {result['exaggerated_emotion'].shape}")
    logger.info(f"  Clean text: {result['clean_text']}")
    logger.info(f"  Parsed tags: {[t.tag_name for t in result['parsed_tags']]}")

    # Test adapter
    logger.info("\nTesting ChatterboxAdapter:")
    adapter = ChatterboxAdapter(config).to(device)

    with torch.no_grad():
        adapter_result = adapter(
            features,
            text="Testing [sigh] the adapter [laugh].",
            exaggeration=1.0,
        )

    logger.info(f"  Adapter output shape: {adapter_result['prosody_tokens'].shape}")

    # Test loss function
    logger.info("\nTesting loss function:")
    loss_fn = ChatterboxLoss(config).to(device)

    target_prosody = torch.randn(1, config.num_prosody_tokens, config.output_dim).to(device)
    losses = loss_fn(result, target_prosody=target_prosody)

    for key, value in losses.items():
        logger.info(f"  {key}: {value.item():.4f}")

    # List supported tags
    logger.info(f"\nSupported paralinguistic tags ({len(PARALINGUISTIC_TAGS)}):")
    for tag_name in sorted(PARALINGUISTIC_TAGS.keys()):
        info = PARALINGUISTIC_TAGS[tag_name]
        logger.info(f"  [{tag_name}]: {info['category']}, intensity={info['intensity']:.1f}")

    logger.info("\nTest completed successfully!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Chatterbox emotion exaggeration model"
    )
    parser.add_argument(
        '--config', type=str, default='config/chatterbox_emotion.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run test mode with synthetic data'
    )

    args = parser.parse_args()

    if args.test:
        run_test()
    else:
        train(args.config, args.resume)


if __name__ == "__main__":
    main()
