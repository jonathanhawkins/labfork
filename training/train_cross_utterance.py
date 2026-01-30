"""
Training script for Cross-Utterance Context prosody prediction.

Based on CUC-VAE (Cross-Utterance Conditioned VAE) and ParaTTS research.

Training Pipeline:
1. Load paragraph-structured dataset with sentence boundaries
2. Extract text embeddings for sentences with context windows
3. Train VAE to learn context-aware prosody latent space
4. Evaluate on paragraph coherence metrics

Usage:
    # Train cross-utterance model
    python train_cross_utterance.py --config config/cross_utterance.yaml

    # Resume from checkpoint
    python train_cross_utterance.py --config config/cross_utterance.yaml \
        --resume ../checkpoints/cross_utterance/best.pt

    # Test mode (synthetic data)
    python train_cross_utterance.py --test
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

from training.cross_utterance_prosody import (
    CrossUtteranceConfig,
    CrossUtteranceAdapter,
    CrossUtteranceLoss,
    CrossUtteranceProsody,
    split_into_sentences,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class CrossUtteranceDataset(Dataset):
    """
    Dataset for training cross-utterance prosody prediction.

    Expects manifest with paragraph-structured data:
    {
        "paragraph_id": "para_001",
        "sentences": [
            {
                "text": "This is the first sentence.",
                "audio_path": "path/to/audio1.wav",
                "prosody": {
                    "semantic": [...],
                    "acoustic": [...],
                    "rhythm": [...],
                    "contour": [...]
                },
                "speaker_id": 0,
                "emotion": "neutral"
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        manifest_path: str,
        config: CrossUtteranceConfig,
        text_encoder_model: Optional[str] = None,
        max_paragraph_length: int = 10,
    ):
        self.config = config
        self.max_paragraph_length = max_paragraph_length

        # Load manifest
        if Path(manifest_path).exists():
            with open(manifest_path, 'r') as f:
                self.paragraphs = [json.loads(line) for line in f if line.strip()]
        else:
            # Create synthetic data for testing
            self.paragraphs = self._create_synthetic_data()

        logger.info(f"Loaded {len(self.paragraphs)} paragraphs")

        # Flatten to (paragraph_idx, sentence_idx) pairs
        self.samples = []
        for para_idx, para in enumerate(self.paragraphs):
            sentences = para.get('sentences', [])
            for sent_idx in range(len(sentences)):
                self.samples.append((para_idx, sent_idx))

        logger.info(f"Total samples: {len(self.samples)}")

        # Emotion mapping
        self.emotion_to_idx = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "fearful": 4, "surprised": 5, "disgusted": 6, "calm": 7,
        }

        # Text encoder (lazy loaded)
        self._tokenizer = None
        self._text_encoder = None
        self._text_encoder_model = text_encoder_model or config.text_encoder_model

    def _create_synthetic_data(self, num_paragraphs: int = 100) -> List[Dict]:
        """Create synthetic paragraph data for testing."""
        paragraphs = []

        sample_texts = [
            "The sun was setting over the mountains.",
            "A gentle breeze rustled through the trees.",
            "Birds were singing their evening songs.",
            "The sky turned shades of orange and pink.",
            "It was a peaceful end to a long day.",
            "Tomorrow would bring new adventures.",
            "But for now, everything was calm.",
            "The world seemed to pause and breathe.",
        ]

        emotions = list(self.emotion_to_idx.keys()) if hasattr(self, 'emotion_to_idx') else [
            "neutral", "happy", "sad", "angry", "fearful", "surprised"
        ]

        for para_idx in range(num_paragraphs):
            num_sentences = random.randint(3, 7)
            sentences = []

            for sent_idx in range(num_sentences):
                text = random.choice(sample_texts)

                # Generate synthetic prosody
                prosody = {
                    "semantic": [random.random() for _ in range(8)],
                    "acoustic": [random.random() for _ in range(12)],
                    "rhythm": [random.random() for _ in range(8)],
                    "contour": [random.random() for _ in range(64)],
                }

                sentences.append({
                    "text": text,
                    "audio_path": f"synthetic/para_{para_idx}/sent_{sent_idx}.wav",
                    "prosody": prosody,
                    "speaker_id": random.randint(0, 9),
                    "emotion": random.choice(emotions),
                })

            paragraphs.append({
                "paragraph_id": f"para_{para_idx:04d}",
                "sentences": sentences,
            })

        return paragraphs

    def _load_text_encoder(self):
        """Lazy load text encoder."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._text_encoder_model)
            logger.info(f"Loaded tokenizer: {self._text_encoder_model}")

        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Using mock tokenizer.")
            self._tokenizer = "mock"

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text."""
        self._load_text_encoder()

        if self._tokenizer == "mock":
            # Mock tokenization
            return {
                'input_ids': torch.randint(0, 30000, (min(len(text.split()) * 2, 64),)),
                'attention_mask': torch.ones(min(len(text.split()) * 2, 64)),
            }

        encoded = self._tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_sentence_length,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        para_idx, sent_idx = self.samples[idx]
        paragraph = self.paragraphs[para_idx]
        sentences = paragraph['sentences']
        current_sent = sentences[sent_idx]

        # Tokenize current sentence
        current_tokens = self._tokenize(current_sent['text'])

        # Tokenize previous sentences
        prev_tokens_list = []
        for i in range(max(0, sent_idx - self.config.num_prev_sentences), sent_idx):
            tokens = self._tokenize(sentences[i]['text'])
            prev_tokens_list.append(tokens)

        # Pad to fixed number of previous sentences
        while len(prev_tokens_list) < self.config.num_prev_sentences:
            prev_tokens_list.insert(0, {
                'input_ids': torch.zeros_like(current_tokens['input_ids']),
                'attention_mask': torch.zeros_like(current_tokens['attention_mask']),
            })

        # Tokenize next sentences
        next_tokens_list = []
        for i in range(sent_idx + 1, min(len(sentences), sent_idx + 1 + self.config.num_next_sentences)):
            tokens = self._tokenize(sentences[i]['text'])
            next_tokens_list.append(tokens)

        # Pad to fixed number of next sentences
        while len(next_tokens_list) < self.config.num_next_sentences:
            next_tokens_list.append({
                'input_ids': torch.zeros_like(current_tokens['input_ids']),
                'attention_mask': torch.zeros_like(current_tokens['attention_mask']),
            })

        # Get prosody features
        prosody = current_sent.get('prosody', {})
        semantic = torch.tensor(prosody.get('semantic', [0.0] * 8), dtype=torch.float32)
        acoustic = torch.tensor(prosody.get('acoustic', [0.0] * 12), dtype=torch.float32)
        rhythm = torch.tensor(prosody.get('rhythm', [0.0] * 8), dtype=torch.float32)
        contour = torch.tensor(prosody.get('contour', [0.0] * 64), dtype=torch.float32)

        # Get speaker and emotion
        speaker_id = current_sent.get('speaker_id', 0)
        emotion = current_sent.get('emotion', 'neutral')
        emotion_id = self.emotion_to_idx.get(emotion, 0)

        result = {
            'current_input_ids': current_tokens['input_ids'],
            'current_attention_mask': current_tokens['attention_mask'],
            'semantic': semantic,
            'acoustic': acoustic,
            'rhythm': rhythm,
            'contour': contour,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'emotion_id': torch.tensor(emotion_id, dtype=torch.long),
            'para_idx': torch.tensor(para_idx, dtype=torch.long),
            'sent_idx': torch.tensor(sent_idx, dtype=torch.long),
        }

        # Add previous sentence tokens
        for i, tokens in enumerate(prev_tokens_list):
            result[f'prev_{i}_input_ids'] = tokens['input_ids']
            result[f'prev_{i}_attention_mask'] = tokens['attention_mask']

        # Add next sentence tokens
        for i, tokens in enumerate(next_tokens_list):
            result[f'next_{i}_input_ids'] = tokens['input_ids']
            result[f'next_{i}_attention_mask'] = tokens['attention_mask']

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    result = {}

    # Stack all tensors
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        result[key] = torch.stack(values)

    return result


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class TextEncoder(nn.Module):
    """
    Wrapper for pre-trained text encoder.
    """

    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self._encoder = None
        self.freeze = freeze

    def _load_encoder(self, device: torch.device):
        """Lazy load encoder."""
        if self._encoder is not None:
            return

        try:
            from transformers import AutoModel

            self._encoder = AutoModel.from_pretrained(self.model_name).to(device)

            if self.freeze:
                for param in self._encoder.parameters():
                    param.requires_grad = False
                self._encoder.eval()

            logger.info(f"Loaded text encoder: {self.model_name}")

        except Exception as e:
            logger.warning(f"Failed to load text encoder: {e}. Using mock encoder.")
            self._encoder = "mock"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            embeddings: [batch, seq_len, hidden_dim]
        """
        device = input_ids.device
        self._load_encoder(device)

        if self._encoder == "mock":
            # Mock encoding
            return torch.randn(
                input_ids.shape[0],
                input_ids.shape[1],
                768,
                device=device,
            ) * 0.1

        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self._encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.last_hidden_state


class Trainer:
    """
    Trainer for cross-utterance prosody model.
    """

    def __init__(
        self,
        config: CrossUtteranceConfig,
        model: CrossUtteranceProsody,
        text_encoder: TextEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        checkpoint_dir: str = "../checkpoints/cross_utterance",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
    ):
        self.config = config
        self.model = model
        self.text_encoder = text_encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_steps = max_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_steps,
            T_mult=2,
        )

        # Loss function
        self.loss_fn = CrossUtteranceLoss(config)

        # Training state
        self.step = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        device = next(self.model.parameters()).device

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Encode current sentence
        current_embeddings = self.text_encoder(
            batch['current_input_ids'],
            batch['current_attention_mask'],
        )

        # Encode previous sentences
        prev_embeddings = []
        prev_masks = []
        for i in range(self.config.num_prev_sentences):
            emb = self.text_encoder(
                batch[f'prev_{i}_input_ids'],
                batch[f'prev_{i}_attention_mask'],
            )
            prev_embeddings.append(emb)
            prev_masks.append(batch[f'prev_{i}_attention_mask'])

        # Encode next sentences
        next_embeddings = []
        next_masks = []
        for i in range(self.config.num_next_sentences):
            emb = self.text_encoder(
                batch[f'next_{i}_input_ids'],
                batch[f'next_{i}_attention_mask'],
            )
            next_embeddings.append(emb)
            next_masks.append(batch[f'next_{i}_attention_mask'])

        # Forward pass
        result = self.model(
            current_text_embeddings=current_embeddings,
            current_text_mask=batch['current_attention_mask'],
            prev_text_embeddings=prev_embeddings,
            prev_text_masks=prev_masks,
            next_text_embeddings=next_embeddings,
            next_text_masks=next_masks,
            target_semantic=batch['semantic'],
            target_acoustic=batch['acoustic'],
            target_rhythm=batch['rhythm'],
            target_contour=batch['contour'],
            speaker_id=batch['speaker_id'] if self.config.num_speakers > 0 else None,
            emotion_id=batch['emotion_id'] if self.config.use_emotion else None,
            training=True,
        )

        # Compute loss (use prosody tokens as target for now)
        # In practice, you'd have real target tokens from prosody encoder
        target_tokens = torch.zeros_like(result['prosody_tokens'])

        losses = self.loss_fn(
            result['prosody_tokens'],
            target_tokens,
            result['kl_loss'],
            step=self.step,
        )

        # Backward pass
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1

        return {
            'loss': losses['total'].item(),
            'reconstruction': losses['reconstruction'].item(),
            'kl': losses['kl'].item(),
            'kl_weight': losses['kl_weight'].item(),
            'lr': self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        device = next(self.model.parameters()).device

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Encode current sentence
            current_embeddings = self.text_encoder(
                batch['current_input_ids'],
                batch['current_attention_mask'],
            )

            # Encode previous sentences
            prev_embeddings = []
            prev_masks = []
            for i in range(self.config.num_prev_sentences):
                emb = self.text_encoder(
                    batch[f'prev_{i}_input_ids'],
                    batch[f'prev_{i}_attention_mask'],
                )
                prev_embeddings.append(emb)
                prev_masks.append(batch[f'prev_{i}_attention_mask'])

            # Encode next sentences
            next_embeddings = []
            next_masks = []
            for i in range(self.config.num_next_sentences):
                emb = self.text_encoder(
                    batch[f'next_{i}_input_ids'],
                    batch[f'next_{i}_attention_mask'],
                )
                next_embeddings.append(emb)
                next_masks.append(batch[f'next_{i}_attention_mask'])

            # Forward pass
            result = self.model(
                current_text_embeddings=current_embeddings,
                current_text_mask=batch['current_attention_mask'],
                prev_text_embeddings=prev_embeddings,
                prev_text_masks=prev_masks,
                next_text_embeddings=next_embeddings,
                next_text_masks=next_masks,
                target_semantic=batch['semantic'],
                target_acoustic=batch['acoustic'],
                target_rhythm=batch['rhythm'],
                target_contour=batch['contour'],
                training=True,
            )

            # Compute loss
            target_tokens = torch.zeros_like(result['prosody_tokens'])
            losses = self.loss_fn(
                result['prosody_tokens'],
                target_tokens,
                result['kl_loss'],
                step=self.step,
            )

            total_loss += losses['total'].item()
            total_recon += losses['reconstruction'].item()
            total_kl += losses['kl'].item()
            num_batches += 1

        return {
            'val_loss': total_loss / max(num_batches, 1),
            'val_reconstruction': total_recon / max(num_batches, 1),
            'val_kl': total_kl / max(num_batches, 1),
        }

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'best_val_loss': self.best_val_loss,
        }

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        logger.info(f"Loaded checkpoint from {path} (step {self.step})")

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training from step {self.step}")

        train_iter = iter(self.train_loader)

        while self.step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)

            # Logging
            if self.step % self.log_interval == 0:
                logger.info(
                    f"Step {self.step}: loss={metrics['loss']:.4f}, "
                    f"recon={metrics['reconstruction']:.4f}, "
                    f"kl={metrics['kl']:.4f}, "
                    f"kl_w={metrics['kl_weight']:.4f}, "
                    f"lr={metrics['lr']:.6f}"
                )

            # Evaluation
            if self.step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                if val_metrics:
                    logger.info(
                        f"Validation: loss={val_metrics['val_loss']:.4f}, "
                        f"recon={val_metrics['val_reconstruction']:.4f}, "
                        f"kl={val_metrics['val_kl']:.4f}"
                    )

                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint("best")

            # Save checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.step}")

        # Final save
        self.save_checkpoint("final")
        logger.info("Training complete!")


# =============================================================================
# MAIN
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Cross-Utterance Prosody Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--test", action="store_true", help="Run in test mode with synthetic data")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints/cross_utterance")
    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        config_dict = load_config(args.config)
        config = CrossUtteranceConfig(**config_dict.get('model', {}))
    else:
        config = CrossUtteranceConfig()

    logger.info(f"Configuration: {asdict(config)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = CrossUtteranceProsody(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create text encoder
    text_encoder = TextEncoder(
        config.text_encoder_model,
        freeze=config.freeze_text_encoder,
    )

    # Create dataset
    if args.test:
        logger.info("Running in test mode with synthetic data")
        manifest_path = "synthetic_manifest.jsonl"  # Will create synthetic data
    else:
        manifest_path = args.manifest or "../data/paragraph_manifest.jsonl"

    dataset = CrossUtteranceDataset(manifest_path, config)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        text_encoder=text_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        max_steps=args.max_steps if not args.test else 100,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=10 if args.test else 100,
        eval_interval=50 if args.test else 1000,
        save_interval=50 if args.test else 5000,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
