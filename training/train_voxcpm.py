#!/usr/bin/env python3
"""
VoxCPM Training Script

Trains the VoxCPM tokenizer-free TTS model with flow-matching objective.

Usage:
    # Full training
    python train_voxcpm.py --config config/voxcpm.yaml

    # Resume training
    python train_voxcpm.py --config config/voxcpm.yaml \
        --resume ../checkpoints/voxcpm/latest.pt

    # VAE pretraining only
    python train_voxcpm.py --config config/voxcpm.yaml --vae-only

    # Compact model for testing
    python train_voxcpm.py --config config/voxcpm.yaml --compact

    # Test mode (synthetic data)
    python train_voxcpm.py --test

Based on arXiv:2509.24650:
"VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation
and True-to-Life Voice Cloning"
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

from voxcpm import (
    VoxCPMConfig,
    VoxCPM,
    VoxCPMAdapter,
    estimate_rtf,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class VoxCPMDataset(Dataset):
    """Dataset for VoxCPM training."""

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        max_audio_len: int = 480000,
        min_audio_len: int = 16000,
        max_text_len: int = 512,
    ):
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.min_audio_len = min_audio_len
        self.max_text_len = max_text_len

        # Load manifest
        self.samples = []
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            sample = json.loads(line)
                            self.samples.append(sample)
                        except json.JSONDecodeError:
                            continue

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

        # Simple character-to-index mapping
        self.char_to_idx = {}
        self.build_vocab()

    def build_vocab(self):
        """Build character vocabulary from samples."""
        chars = set()
        for sample in self.samples:
            text = sample.get('text', sample.get('transcript', ''))
            chars.update(text)

        self.char_to_idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
        self.char_to_idx['<pad>'] = 0
        self.char_to_idx['<unk>'] = len(self.char_to_idx)

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token indices."""
        tokens = [
            self.char_to_idx.get(c, self.char_to_idx['<unk>'])
            for c in text[:self.max_text_len]
        ]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio
        audio_path = sample.get('audio_path', sample.get('path', ''))
        if os.path.exists(audio_path):
            import torchaudio
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            audio = audio.mean(dim=0, keepdim=True)  # Mono
        else:
            # Synthetic audio for testing
            length = torch.randint(self.min_audio_len, self.max_audio_len, (1,)).item()
            audio = torch.randn(1, length) * 0.1

        # Truncate/pad audio
        if audio.shape[-1] > self.max_audio_len:
            start = torch.randint(0, audio.shape[-1] - self.max_audio_len, (1,)).item()
            audio = audio[:, start:start + self.max_audio_len]
        elif audio.shape[-1] < self.min_audio_len:
            audio = F.pad(audio, (0, self.min_audio_len - audio.shape[-1]))

        # Get text
        text = sample.get('text', sample.get('transcript', 'hello world'))
        text_tokens = self.text_to_tokens(text)

        return {
            'audio': audio,
            'text_tokens': text_tokens,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Pad audio
    max_audio_len = max(b['audio'].shape[-1] for b in batch)
    audios = []
    audio_masks = []
    for b in batch:
        audio = b['audio']
        pad_len = max_audio_len - audio.shape[-1]
        audios.append(F.pad(audio, (0, pad_len)))
        mask = torch.ones(audio.shape[-1] // 640)  # Frame-level mask
        mask = F.pad(mask, (0, max_audio_len // 640 - mask.shape[0]))
        audio_masks.append(mask)

    # Pad text
    max_text_len = max(b['text_tokens'].shape[0] for b in batch)
    texts = []
    text_masks = []
    for b in batch:
        text = b['text_tokens']
        pad_len = max_text_len - text.shape[0]
        texts.append(F.pad(text, (0, pad_len)))
        mask = torch.ones(text.shape[0], dtype=torch.bool)
        mask = F.pad(mask, (0, pad_len), value=False)
        text_masks.append(mask)

    return {
        'audio': torch.stack(audios),
        'audio_mask': torch.stack(audio_masks),
        'text_tokens': torch.stack(texts),
        'text_mask': torch.stack(text_masks),
    }


# =============================================================================
# TRAINING
# =============================================================================

class VoxCPMTrainer:
    """Trainer for VoxCPM model."""

    def __init__(
        self,
        config: VoxCPMConfig,
        train_config: Dict,
        model: Optional[VoxCPM] = None,
    ):
        self.config = config
        self.train_config = train_config
        self.device = torch.device(train_config.get('device', 'cuda'))

        # Create model
        if model is not None:
            self.model = model
        else:
            self.model = VoxCPM(config)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.get('learning_rate', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.get('max_steps', 500000),
        )

        # Mixed precision
        self.use_amp = train_config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Gradient accumulation
        self.grad_accum = train_config.get('gradient_accumulation', 1)
        self.max_grad_norm = train_config.get('max_grad_norm', 1.0)

        # Logging
        self.step = 0
        self.best_loss = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(train_config.get('checkpoint_dir', '../checkpoints/voxcpm'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        audio = batch['audio'].to(self.device)
        text_tokens = batch['text_tokens'].to(self.device)
        text_mask = batch['text_mask'].to(self.device)
        audio_mask = batch.get('audio_mask')
        if audio_mask is not None:
            audio_mask = audio_mask.to(self.device)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            losses = self.model.compute_loss(
                text_tokens=text_tokens,
                audio=audio,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )

        loss = losses['total'] / self.grad_accum

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.step + 1) % self.grad_accum == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

        self.step += 1

        return {k: v.item() for k, v in losses.items()}

    def train_vae_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """VAE pretraining step (optional separate stage)."""
        audio = batch['audio'].to(self.device)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            vae_out = self.model.vae(audio)

            # Reconstruction loss
            recon_loss = F.mse_loss(vae_out['reconstructed'], audio)

            # KL divergence
            kl_loss = -0.5 * torch.mean(
                1 + vae_out['logvar'] - vae_out['mean'].pow(2) - vae_out['logvar'].exp()
            )

            loss = recon_loss + 5e-5 * kl_loss

        loss = loss / self.grad_accum

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (self.step + 1) % self.grad_accum == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.vae.parameters(),
                self.max_grad_norm
            )

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        self.step += 1

        return {
            'total': loss.item() * self.grad_accum,
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_losses = {}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                audio = batch['audio'].to(self.device)
                text_tokens = batch['text_tokens'].to(self.device)
                text_mask = batch['text_mask'].to(self.device)

                losses = self.model.compute_loss(
                    text_tokens=text_tokens,
                    audio=audio,
                    text_mask=text_mask,
                )

                for k, v in losses.items():
                    if k not in total_losses:
                        total_losses[k] = 0
                    total_losses[k] += v.item()

                num_batches += 1

                if num_batches >= 50:  # Limit eval batches
                    break

        self.model.train()
        return {k: v / num_batches for k, v in total_losses.items()}

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = self.checkpoint_dir / f'checkpoint_step_{self.step}.pt'

        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Loaded checkpoint from {path} at step {self.step}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        max_steps: Optional[int] = None,
        vae_only: bool = False,
    ):
        """Main training loop."""
        if max_steps is None:
            max_steps = self.train_config.get('max_steps', 500000)

        log_interval = self.train_config.get('log_interval', 100)
        eval_interval = self.train_config.get('eval_interval', 5000)
        save_interval = self.train_config.get('save_interval', 10000)

        self.model.train()
        running_losses = {}
        num_logs = 0

        logger.info(f"Starting training from step {self.step}")
        logger.info(f"Max steps: {max_steps}")

        while self.step < max_steps:
            for batch in train_dataloader:
                if self.step >= max_steps:
                    break

                # Training step
                if vae_only:
                    losses = self.train_vae_step(batch)
                else:
                    losses = self.train_step(batch)

                # Accumulate losses
                for k, v in losses.items():
                    if k not in running_losses:
                        running_losses[k] = 0
                    running_losses[k] += v
                num_logs += 1

                # Logging
                if self.step % log_interval == 0 and num_logs > 0:
                    avg_losses = {k: v / num_logs for k, v in running_losses.items()}
                    lr = self.scheduler.get_last_lr()[0]

                    log_msg = f"Step {self.step} | LR: {lr:.2e}"
                    for k, v in avg_losses.items():
                        log_msg += f" | {k}: {v:.4f}"
                    logger.info(log_msg)

                    running_losses = {}
                    num_logs = 0

                # Evaluation
                if self.step % eval_interval == 0 and val_dataloader is not None:
                    eval_losses = self.evaluate(val_dataloader)

                    log_msg = f"Eval Step {self.step}"
                    for k, v in eval_losses.items():
                        log_msg += f" | {k}: {v:.4f}"
                    logger.info(log_msg)

                    # Check best
                    if eval_losses['total'] < self.best_loss:
                        self.best_loss = eval_losses['total']
                        self.save_checkpoint(is_best=True)

                # Save checkpoint
                if self.step % save_interval == 0:
                    self.save_checkpoint()

        # Final save
        self.save_checkpoint()
        logger.info(f"Training complete at step {self.step}")


# =============================================================================
# MAIN
# =============================================================================

def create_config(yaml_config: Dict, compact: bool = False) -> VoxCPMConfig:
    """Create VoxCPMConfig from YAML config."""
    model_config = yaml_config.get('model', {})

    if compact:
        # Use compact variant
        compact_config = yaml_config.get('model_compact', {})
        model_config.update(compact_config)

    return VoxCPMConfig(
        sample_rate=model_config.get('sample_rate', 16000),
        vae_frame_rate=model_config.get('vae_frame_rate', 25),
        vae_latent_dim=model_config.get('vae_latent_dim', 64),
        tslm_hidden_dim=model_config.get('tslm_hidden_dim', 1024),
        tslm_ffn_dim=model_config.get('tslm_ffn_dim', 4096),
        tslm_num_layers=model_config.get('tslm_num_layers', 24),
        tslm_num_heads=model_config.get('tslm_num_heads', 16),
        tslm_vocab_size=model_config.get('tslm_vocab_size', 50000),
        fsq_dim=model_config.get('fsq_dim', 256),
        fsq_levels=model_config.get('fsq_levels', 9),
        fsq_enabled=model_config.get('fsq_enabled', True),
        ralm_hidden_dim=model_config.get('ralm_hidden_dim', 1024),
        ralm_ffn_dim=model_config.get('ralm_ffn_dim', 4096),
        ralm_num_layers=model_config.get('ralm_num_layers', 6),
        ralm_num_heads=model_config.get('ralm_num_heads', 16),
        locdit_hidden_dim=model_config.get('locdit_hidden_dim', 1024),
        locdit_ffn_dim=model_config.get('locdit_ffn_dim', 4096),
        locdit_num_layers=model_config.get('locdit_num_layers', 4),
        locdit_num_heads=model_config.get('locdit_num_heads', 16),
        locdit_patch_size=model_config.get('locdit_patch_size', 4),
        num_diffusion_steps=model_config.get('num_diffusion_steps', 50),
        cfg_scale=model_config.get('cfg_scale', 2.0),
        cfg_mask_prob=model_config.get('cfg_mask_prob', 0.1),
        vae_hidden_dim=model_config.get('vae_hidden_dim', 512),
        vae_num_blocks=model_config.get('vae_num_blocks', 4),
        vae_strides=model_config.get('vae_strides', [2, 5, 8, 8]),
        dropout=model_config.get('dropout', 0.1),
        stop_loss_weight=model_config.get('stop_loss_weight', 1.0),
        flow_loss_weight=model_config.get('flow_loss_weight', 1.0),
        output_dim=model_config.get('output_dim', 2048),
        num_prefix_tokens=model_config.get('num_prefix_tokens', 4),
    )


def run_test_mode():
    """Run in test mode with synthetic data."""
    logger.info("Running in test mode with synthetic data...")

    # Create config
    config = VoxCPMConfig(
        tslm_num_layers=4,
        tslm_hidden_dim=256,
        tslm_ffn_dim=512,
        ralm_num_layers=2,
        ralm_hidden_dim=256,
        locdit_num_layers=2,
        locdit_hidden_dim=256,
        num_diffusion_steps=10,
    )

    # Create model
    model = VoxCPM(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    text_len = 50
    audio_len = 32000  # 2 seconds

    text_tokens = torch.randint(0, config.tslm_vocab_size, (batch_size, text_len))
    audio = torch.randn(batch_size, 1, audio_len) * 0.1
    text_mask = torch.ones(batch_size, text_len, dtype=torch.bool)

    logger.info("Testing forward pass...")
    losses = model.compute_loss(text_tokens, audio, text_mask)

    for k, v in losses.items():
        logger.info(f"  {k}: {v.item():.4f}")

    # Test generation
    logger.info("Testing generation...")
    with torch.no_grad():
        generated = model.generate(
            text_tokens,
            max_len=50,
            num_steps=5,
        )
    logger.info(f"  Generated shape: {generated.shape}")

    # Test adapter
    logger.info("Testing CSM adapter...")
    adapter = VoxCPMAdapter(config, model)
    result = adapter(audio)
    logger.info(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")

    # Test voice cloning
    logger.info("Testing voice cloning...")
    ref_result = adapter.from_reference(audio)
    logger.info(f"  Speaker embedding shape: {ref_result['speaker_embedding'].shape}")

    logger.info("All tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Train VoxCPM model")
    parser.add_argument('--config', type=str, default='config/voxcpm.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--vae-only', action='store_true',
                        help='Only pretrain VAE')
    parser.add_argument('--compact', action='store_true',
                        help='Use compact model variant')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with synthetic data')
    args = parser.parse_args()

    if args.test:
        run_test_mode()
        return

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    # Create model config
    config = create_config(yaml_config, compact=args.compact)
    train_config = yaml_config.get('training', {})
    data_config = yaml_config.get('data', {})

    logger.info(f"VoxCPM Configuration:")
    logger.info(f"  TSLM: {config.tslm_num_layers} layers, {config.tslm_hidden_dim} dim")
    logger.info(f"  RALM: {config.ralm_num_layers} layers, {config.ralm_hidden_dim} dim")
    logger.info(f"  LocDiT: {config.locdit_num_layers} layers, {config.locdit_hidden_dim} dim")
    logger.info(f"  FSQ: {config.fsq_dim} dims, {config.fsq_levels} levels")
    logger.info(f"  Diffusion: {config.num_diffusion_steps} steps, CFG {config.cfg_scale}")

    # Create datasets
    train_manifest = data_config.get('train_manifest', '../data/manifest.json')
    val_manifest = data_config.get('val_manifest', '../data/val_manifest.json')

    train_dataset = VoxCPMDataset(
        train_manifest,
        sample_rate=config.sample_rate,
        max_audio_len=data_config.get('max_audio_len', 480000),
        min_audio_len=data_config.get('min_audio_len', 16000),
        max_text_len=data_config.get('max_text_len', 512),
    )

    val_dataset = None
    if os.path.exists(val_manifest):
        val_dataset = VoxCPMDataset(
            val_manifest,
            sample_rate=config.sample_rate,
            max_audio_len=data_config.get('max_audio_len', 480000),
            min_audio_len=data_config.get('min_audio_len', 16000),
            max_text_len=data_config.get('max_text_len', 512),
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 8),
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.get('batch_size', 8),
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Create trainer
    trainer = VoxCPMTrainer(config, train_config)

    logger.info(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    max_steps = train_config.get('vae_pretrain_steps', 50000) if args.vae_only else None
    trainer.train(train_loader, val_loader, max_steps=max_steps, vae_only=args.vae_only)


if __name__ == '__main__':
    main()
