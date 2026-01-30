#!/usr/bin/env python3
"""
Training Script for FACodec-Conditioned Prosody Model

Uses NaturalSpeech3's FACodec to extract disentangled prosody codes
and trains the model to condition on these clean prosody representations.

Benefits over multi-vector approach:
1. Pre-trained disentanglement - no speaker/content leakage
2. Discrete codes - can use as conditioning tokens directly
3. Standardized representation - works across different speakers

Usage:
    # Train with FACodec prosody conditioning
    python train_facodec_prosody.py --config config/facodec_prosody.yaml

    # Resume training
    python train_facodec_prosody.py --config config/facodec_prosody.yaml \
        --resume ../checkpoints/facodec_prosody/best.pt

    # Preprocess dataset (extract FACodec features first)
    python train_facodec_prosody.py --preprocess \
        --manifest ../data/manifest.json \
        --output ../data/facodec_features
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facodec_integration import (
    FACodecConfig,
    FACodecProsodyExtractor,
    FACodecProsodyAdapter,
    FACodecControlledCSM,
    extract_facodec_prosody,
    preprocess_dataset_with_facodec,
)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    manifest_path: str = "../data/manifest.json"
    facodec_features_dir: str = "../data/facodec_features"
    val_split: float = 0.1

    # Model
    model_path: str = "../models/sesame-csm-1b"
    hidden_size: int = 2048
    num_prosody_tokens: int = 4
    freeze_csm: bool = True

    # FACodec
    facodec_config: FACodecConfig = field(default_factory=FACodecConfig)

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 10000
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # Checkpointing
    output_dir: str = "../checkpoints/facodec_prosody"
    save_steps: int = 500
    eval_steps: int = 250
    log_steps: int = 50

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4


# =============================================================================
# DATASET
# =============================================================================

class FACodecProsodyDataset(Dataset):
    """
    Dataset for training with pre-extracted FACodec prosody features.

    Expects manifest with:
    - audio_path: Path to audio file
    - text: Transcript
    - facodec_path: Path to pre-extracted FACodec features (.pt file)

    Each .pt file should contain:
    - prosody_codes: [time] discrete codes
    - prosody_emb: [time, 256] embeddings
    - speaker_emb: [256] speaker embedding
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        max_length: int = 512,
        sample_rate: int = 16000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_rate = sample_rate

        # Load manifest
        with open(manifest_path) as f:
            self.samples = json.load(f)

        # Filter to samples with FACodec features
        self.samples = [
            s for s in self.samples
            if 'facodec_path' in s and os.path.exists(s['facodec_path'])
        ]

        print(f"Loaded {len(self.samples)} samples with FACodec features")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load FACodec features
        facodec = torch.load(sample['facodec_path'])

        # Tokenize text
        text = sample.get('text', '')
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        # Load audio for labels (if training decoder)
        audio = None
        if 'audio_path' in sample:
            try:
                waveform, sr = torchaudio.load(sample['audio_path'])
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audio = waveform.squeeze(0)
            except Exception as e:
                print(f"Failed to load audio {sample['audio_path']}: {e}")

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'prosody_codes': facodec['prosody_codes'].squeeze(0),  # [time]
            'prosody_emb': facodec['prosody_emb'].squeeze(0),      # [time, 256]
            'speaker_emb': facodec['speaker_emb'].squeeze(0),      # [256]
            'audio': audio,
            'text': text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length prosody codes."""

    # Standard fields
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])

    # Pad prosody codes to same length
    max_prosody_len = max(b['prosody_codes'].shape[0] for b in batch)

    prosody_codes = []
    prosody_embs = []
    prosody_masks = []

    for b in batch:
        codes = b['prosody_codes']
        emb = b['prosody_emb']
        length = codes.shape[0]

        # Pad codes
        padded_codes = F.pad(codes, (0, max_prosody_len - length), value=0)
        prosody_codes.append(padded_codes)

        # Pad embeddings
        padded_emb = F.pad(emb, (0, 0, 0, max_prosody_len - length), value=0)
        prosody_embs.append(padded_emb)

        # Create mask
        mask = torch.zeros(max_prosody_len)
        mask[:length] = 1
        prosody_masks.append(mask)

    # Stack speaker embeddings
    speaker_embs = torch.stack([b['speaker_emb'] for b in batch])

    # Stack audio if available
    audio = None
    if batch[0]['audio'] is not None:
        max_audio_len = max(b['audio'].shape[0] for b in batch if b['audio'] is not None)
        audio_list = []
        for b in batch:
            if b['audio'] is not None:
                padded = F.pad(b['audio'], (0, max_audio_len - b['audio'].shape[0]))
                audio_list.append(padded)
        if audio_list:
            audio = torch.stack(audio_list)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'prosody_codes': torch.stack(prosody_codes),
        'prosody_emb': torch.stack(prosody_embs),
        'prosody_mask': torch.stack(prosody_masks),
        'speaker_emb': speaker_embs,
        'audio': audio,
    }


# =============================================================================
# TRAINING
# =============================================================================

class FACodecProsodyTrainer:
    """
    Trainer for FACodec-conditioned prosody model.

    Trains the model to generate speech conditioned on FACodec prosody codes.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1,
        )

        # Mixed precision
        self.scaler = None
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')

        # State
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def warmup_lr(self):
        """Apply linear warmup to learning rate."""
        if self.global_step < self.config.warmup_steps:
            factor = self.global_step / self.config.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * factor

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        prosody_codes = batch['prosody_codes'].to(self.device)
        prosody_emb = batch['prosody_emb'].to(self.device)
        prosody_mask = batch['prosody_mask'].to(self.device)

        # Forward pass
        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            # Use prosody codes or embeddings based on configuration
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prosody_codes=prosody_codes,
                labels=input_ids,  # Causal LM objective
            )

            loss = outputs.loss / self.config.gradient_accumulation

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {
            'loss': loss.item() * self.config.gradient_accumulation,
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0
        num_steps = 0

        progress = tqdm(self.train_loader, desc=f"Training")

        for batch_idx, batch in enumerate(progress):
            # Training step
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            num_steps += 1

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update LR
                self.warmup_lr()
                self.scheduler.step()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    avg_loss = total_loss / num_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': self.global_step,
                    })

                # Evaluation
                if self.global_step % self.config.eval_steps == 0 and self.val_loader:
                    val_loss = self.evaluate()
                    print(f"\nStep {self.global_step} - Val loss: {val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best.pt')

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')

                # Max steps
                if self.global_step >= self.config.max_steps:
                    break

        return {'loss': total_loss / max(num_steps, 1)}

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            prosody_codes = batch['prosody_codes'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prosody_codes=prosody_codes,
                labels=input_ids,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.config.output_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self):
        """Main training loop."""
        print(f"Starting training from step {self.global_step}")
        print(f"  Device: {self.device}")
        print(f"  Max steps: {self.config.max_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")

        while self.global_step < self.config.max_steps:
            metrics = self.train_epoch()
            print(f"Epoch complete - Loss: {metrics['loss']:.4f}")

        # Final checkpoint
        self.save_checkpoint('final.pt')
        print("Training complete!")


# =============================================================================
# MOCK CSM MODEL (for testing without full model)
# =============================================================================

class MockCSM(nn.Module):
    """Mock CSM model for testing the training pipeline."""

    def __init__(self, hidden_size: int = 2048, vocab_size: int = 32000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def embed_text_tokens(self, input_ids):
        return self.embed(input_ids)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        num_prefix_tokens: int = 0,
        **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)

        hidden = self.layers(inputs_embeds)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # Skip prefix tokens when computing loss
            if num_prefix_tokens > 0:
                logits_for_loss = logits[:, num_prefix_tokens:, :]
            else:
                logits_for_loss = logits

            # Shift for causal LM
            shift_logits = logits_for_loss[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Ensure shapes match
            min_len = min(shift_logits.shape[1], shift_labels.shape[1])
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        return Output(loss, logits)

    def generate(self, **kwargs):
        return torch.randn(1, 16000)


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for config loading. Install with: pip install pyyaml")

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Convert nested FACodec config
    if 'facodec_config' in cfg_dict:
        cfg_dict['facodec_config'] = FACodecConfig(**cfg_dict['facodec_config'])
    else:
        cfg_dict['facodec_config'] = FACodecConfig()

    return TrainingConfig(**cfg_dict)


def main():
    parser = argparse.ArgumentParser(description="Train FACodec-conditioned prosody model")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset")
    parser.add_argument("--manifest", type=str, help="Manifest path for preprocessing")
    parser.add_argument("--output", type=str, help="Output dir for preprocessing")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()

    # Preprocessing mode
    if args.preprocess:
        if not args.manifest or not args.output:
            print("Error: --manifest and --output required for preprocessing")
            return

        preprocess_dataset_with_facodec(
            manifest_path=args.manifest,
            output_dir=args.output,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return

    # Test mode
    if args.test:
        print("Running in test mode with mock CSM...")

        config = TrainingConfig(
            batch_size=2,
            max_steps=10,
            log_steps=1,
            eval_steps=5,
            save_steps=10,
        )

        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {
                    'input_ids': torch.randint(0, 1000, (1, 128)),
                    'attention_mask': torch.ones(1, 128),
                }

        # Mock dataset
        class MockDataset(Dataset):
            def __len__(self):
                return 20

            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, 1000, (128,)),
                    'attention_mask': torch.ones(128),
                    'prosody_codes': torch.randint(0, 1024, (40,)),
                    'prosody_emb': torch.randn(40, 256),
                    'speaker_emb': torch.randn(256),
                    'audio': None,
                    'text': 'Test text',
                }

        dataset = MockDataset()
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
        )

        # Mock model
        csm = MockCSM(hidden_size=config.hidden_size)
        model = FACodecControlledCSM(csm, config.facodec_config, freeze_csm=False)

        # Train
        trainer = FACodecProsodyTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
        )

        trainer.train()
        print("\nTest mode completed successfully!")
        return

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig()

    print("FACodec Prosody Training")
    print("=" * 50)
    print(f"Config: {args.config or 'default'}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {config.device}")

    # TODO: Load real tokenizer and CSM model
    # tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # csm_model = AutoModelForCausalLM.from_pretrained(config.model_path)

    # For now, print instructions
    print("\nTo run training with real models, implement:")
    print("1. Load tokenizer from config.model_path")
    print("2. Load CSM model from config.model_path")
    print("3. Create FACodecProsodyDataset with real manifest")
    print("4. Initialize FACodecControlledCSM wrapper")
    print("5. Run trainer.train()")
    print("\nRun with --test flag for mock testing")


if __name__ == "__main__":
    main()
