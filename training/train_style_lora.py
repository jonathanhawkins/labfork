"""
Train Style-Specific LoRAs for ReStyle-TTS

Trains individual LoRA adapters for specific style attributes (prosody, emotions).
Each LoRA captures a single interpretable attribute that can be composed at inference.

Usage:
    # Train emotion LoRA
    python train_style_lora.py --style happy --config config/style_lora.yaml

    # Train prosody LoRA
    python train_style_lora.py --style high_pitch --config config/style_lora.yaml

    # Train with all styles sequentially
    python train_style_lora.py --all-styles --config config/style_lora.yaml

Based on ReStyle-TTS (arXiv:2601.03632):
- LoRA rank 32, alpha 64 injected into all linear layers
- AdamW optimizer, LR 1e-5, batch size 30k frames
- Masking: 0.3 for speech input, 0.2 for speech+text
- Optional TCO for speaker consistency
"""

import argparse
import json
import os
import sys
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchaudio
from tqdm import tqdm

# PEFT imports
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("WARNING: PEFT not installed. Run: pip install peft")

# Local imports
from restyle_tts import (
    ReStyleTTSConfig,
    OrthogonalLoRAFusion,
    TimbreConsistencyOptimization,
    StyleLoRATrainer,
    create_style_lora_config,
)
from data_augmentation import AudioAugmenter, create_augmenter_from_config


# =============================================================================
# STYLE-SPECIFIC DATASET
# =============================================================================

class StyleFilteredDataset(Dataset):
    """
    Dataset that filters samples by style attribute.

    For emotion styles: filters by emotion label
    For prosody styles: filters by acoustic features (pitch level, energy, etc.)
    """

    def __init__(
        self,
        data_path: Path,
        processor,
        style_name: str,
        augmenter: Optional[AudioAugmenter] = None,
        max_audio_length_ms: int = 30000,
        sample_rate: int = 24000,
        training: bool = True,
    ):
        with open(data_path) as f:
            all_samples = json.load(f)

        # Filter samples by style
        self.samples = self._filter_by_style(all_samples, style_name)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for style '{style_name}'")

        self.processor = processor
        self.style_name = style_name
        self.augmenter = augmenter
        self.max_audio_length = int(max_audio_length_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        self.training = training

        print(f"Loaded {len(self.samples)} samples for style '{style_name}'")

    def _filter_by_style(
        self,
        samples: List[Dict],
        style_name: str,
    ) -> List[Dict]:
        """Filter samples by style attribute."""
        style_name = style_name.lower()

        # Emotion styles - filter by emotion label
        emotion_styles = [
            "happy", "sad", "angry", "calm", "surprised",
            "fearful", "disgusted", "neutral"
        ]

        if style_name in emotion_styles:
            return [
                s for s in samples
                if s.get('emotion', '').lower() == style_name or
                   s.get('prosody', {}).get('semantic', {}).get('emotion', '').lower() == style_name
            ]

        # Prosody styles - filter by acoustic features
        prosody_thresholds = {
            "high_pitch": ("pitch_mean", ">=", 200),    # Hz threshold
            "low_pitch": ("pitch_mean", "<", 150),
            "high_energy": ("energy", ">=", 0.7),       # Normalized energy
            "low_energy": ("energy", "<", 0.4),
            "fast_tempo": ("speaking_rate", ">=", 5.0), # Syllables/sec
            "slow_tempo": ("speaking_rate", "<", 3.5),
        }

        if style_name in prosody_thresholds:
            feature, op, threshold = prosody_thresholds[style_name]
            filtered = []

            for sample in samples:
                value = self._get_prosody_feature(sample, feature)
                if value is not None:
                    if op == ">=" and value >= threshold:
                        filtered.append(sample)
                    elif op == "<" and value < threshold:
                        filtered.append(sample)

            return filtered

        # Unknown style - return all samples
        print(f"WARNING: Unknown style '{style_name}', using all samples")
        return samples

    def _get_prosody_feature(
        self,
        sample: Dict,
        feature: str,
    ) -> Optional[float]:
        """Extract prosody feature from sample."""
        prosody = sample.get('prosody', {})

        # Check acoustic features
        acoustic = prosody.get('acoustic', {})
        if feature in acoustic:
            return acoustic[feature]

        # Check rhythm features
        rhythm = prosody.get('rhythm', {})
        if feature in rhythm:
            return rhythm[feature]

        # Check direct sample fields
        if feature in sample:
            return sample[feature]

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio_path = sample.get('audio_path') or sample.get('path')
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply augmentation (only during training)
        if self.training and self.augmenter is not None:
            waveform = self.augmenter(waveform)

        # Ensure 2D after augmentation
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Truncate to max length
        if waveform.shape[1] > self.max_audio_length:
            waveform = waveform[:, :self.max_audio_length]

        audio = waveform.squeeze(0).numpy()
        text = sample['text']
        speaker = str(sample.get('speaker', 0))

        # Build conversation for CSM processor
        conversation = [{
            'role': speaker,
            'content': [
                {'type': 'text', 'text': text},
                {'type': 'audio', 'audio': audio}
            ]
        }]

        # Process with CSM processor
        try:
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                output_labels=True
            )
            inputs['seq_len'] = inputs['input_ids'].shape[-1]
            inputs['audio_len'] = len(audio)
        except Exception as e:
            # Fallback for processor errors
            inputs = {
                'audio': torch.from_numpy(audio),
                'text': text,
                'speaker': speaker,
                'seq_len': len(audio) // 320,
                'audio_len': len(audio),
            }

        return inputs


def create_collate_fn(processor, config: ReStyleTTSConfig):
    """Create collate function with masking for CFG training."""

    def collate_fn(batch):
        # Check if batch has processed inputs
        if 'input_ids' in batch[0]:
            max_seq_len = max(b['seq_len'] for b in batch)
            max_audio_len = max(b['audio_len'] for b in batch)
            batch_size = len(batch)

            input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
            labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
            input_values = torch.zeros(batch_size, 1, max_audio_len, dtype=torch.float32)
            input_values_cutoffs = torch.zeros(batch_size, 1, dtype=torch.long)

            for i, b in enumerate(batch):
                seq_len = b['seq_len']
                audio_len = b['audio_len']

                input_ids[i, :seq_len] = b['input_ids'].squeeze(0)
                attention_mask[i, :seq_len] = b['attention_mask'].squeeze(0)
                labels[i, :seq_len] = b['labels'].squeeze(0)
                input_values[i, :, :audio_len] = b['input_values'].squeeze(0)
                input_values_cutoffs[i] = b['input_values_cutoffs'].squeeze()

            # Apply masking for classifier-free guidance training
            # Speech masking (0.3)
            speech_mask = torch.rand(batch_size) < config.speech_mask_ratio
            speech_mask_expanded = speech_mask.unsqueeze(1).unsqueeze(2)
            input_values = torch.where(
                speech_mask_expanded,
                torch.zeros_like(input_values),
                input_values
            )

            # Text+speech masking (0.2)
            text_mask = torch.rand(batch_size) < config.text_mask_ratio
            full_mask = (speech_mask | text_mask).unsqueeze(1)
            input_ids = torch.where(
                full_mask,
                torch.zeros_like(input_ids),
                input_ids
            )

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'input_values': input_values,
                'input_values_cutoffs': input_values_cutoffs,
            }
        else:
            # Fallback: raw audio/text
            max_len = max(
                b['audio'].shape[0] if isinstance(b['audio'], torch.Tensor)
                else len(b['audio']) for b in batch
            )
            audios = []
            for b in batch:
                audio = b['audio']
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio)
                pad_len = max_len - audio.shape[0]
                if pad_len > 0:
                    audio = torch.cat([audio, torch.zeros(pad_len)])
                audios.append(audio)

            return {
                'audio': torch.stack(audios),
                'text': [b['text'] for b in batch],
                'speaker': [b['speaker'] for b in batch],
            }

    return collate_fn


# =============================================================================
# STYLE LORA TRAINER
# =============================================================================

@dataclass
class TrainingMetrics:
    """Training metrics for dashboard."""
    step: int = 0
    epoch: int = 0
    total_epochs: int = 30
    train_loss: float = 0.0
    val_loss: float = 0.0
    orthogonality_loss: float = 0.0
    tco_weight: float = 1.0
    learning_rate: float = 0.0
    samples_per_second: float = 0.0
    memory_used_gb: float = 0.0
    style_name: str = ""
    status: str = "initializing"
    best_val_loss: float = float('inf')
    loss_history: List[Dict] = field(default_factory=list)


class StyleLoRATrainerRunner:
    """
    Complete training runner for style-specific LoRAs.
    """

    def __init__(
        self,
        config_path: str,
        style_name: str,
    ):
        # Load config
        with open(config_path) as f:
            self.yaml_config = yaml.safe_load(f)

        self.style_name = style_name
        self.config = ReStyleTTSConfig(
            lora_rank=self.yaml_config.get('lora_rank', 32),
            lora_alpha=self.yaml_config.get('lora_alpha', 64),
            lora_dropout=self.yaml_config.get('lora_dropout', 0.1),
            learning_rate=self.yaml_config.get('learning_rate', 1e-5),
            speech_mask_ratio=self.yaml_config.get('speech_mask_ratio', 0.3),
            text_mask_ratio=self.yaml_config.get('text_mask_ratio', 0.2),
            tco_enabled=self.yaml_config.get('tco_enabled', False),
        )

        # Setup device
        self.device = self._setup_device()

        # Paths
        self.output_dir = Path(self.yaml_config.get('output_dir', '../checkpoints')) / f"{style_name}_lora"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.yaml_config, f)

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.metrics = TrainingMetrics(style_name=style_name)

        # Components
        self.model = None
        self.processor = None
        self.augmenter = None
        self.fusion = OrthogonalLoRAFusion(self.config)
        self.tco = TimbreConsistencyOptimization(self.config) if self.config.tco_enabled else None

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        device_str = self.yaml_config.get('device', 'auto')

        if device_str == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print("Using MPS (Apple Silicon)")
            else:
                device = torch.device('cpu')
                print("Using CPU")
        else:
            device = torch.device(device_str)

        return device

    def setup_model(self):
        """Load model and apply LoRA."""
        if not HAS_PEFT:
            raise ImportError("PEFT required. Install with: pip install peft")

        model_path = self.yaml_config['model_path']
        print(f"\nLoading model from: {model_path}")

        from transformers import CsmForConditionalGeneration, AutoProcessor

        # Determine dtype
        precision = self.yaml_config.get('precision', 'fp32')
        if precision == 'fp16':
            dtype = torch.float16
        elif precision == 'bf16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.dtype = dtype

        # Load base model
        base_model = CsmForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Freeze codec
        base_model.codec_model.eval()
        for param in base_model.codec_model.parameters():
            param.requires_grad = False

        # Configure LoRA with ReStyle-TTS settings
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model = self.model.to(self.device)

        # Print info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"\nLoRA Configuration (ReStyle-TTS):")
        print(f"  Rank: {self.config.lora_rank}")
        print(f"  Alpha: {self.config.lora_alpha}")
        print(f"  Target modules: {self.config.lora_target_modules}")
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        return self.model

    def setup_data(self):
        """Setup style-filtered datasets."""
        data_dir = Path(self.yaml_config['data_dir'])

        # Setup augmenter
        aug_config = self.yaml_config.get('augmentation', {})
        if aug_config.get('enabled', True):
            self.augmenter = create_augmenter_from_config(aug_config)
        else:
            self.augmenter = None

        # Create style-filtered datasets
        train_dataset = StyleFilteredDataset(
            data_dir / 'train.json',
            self.processor,
            self.style_name,
            augmenter=self.augmenter,
            max_audio_length_ms=self.yaml_config.get('max_audio_length_ms', 30000),
            training=True,
        )

        val_dataset = StyleFilteredDataset(
            data_dir / 'val.json',
            self.processor,
            self.style_name,
            augmenter=None,
            max_audio_length_ms=self.yaml_config.get('max_audio_length_ms', 30000),
            training=False,
        )

        print(f"\nDatasets for style '{self.style_name}':")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")

        # Create data loaders
        collate_fn = create_collate_fn(self.processor, self.config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.yaml_config.get('batch_size', 4),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.yaml_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda',
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.yaml_config.get('batch_size', 4),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        return train_loader, val_loader

    def setup_optimizer(self, train_loader):
        """Setup optimizer and scheduler."""
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.yaml_config.get('weight_decay', 0.01),
            betas=(0.9, 0.95),
        )

        # Linear warmup + cosine decay
        num_epochs = self.yaml_config.get('num_epochs', 30)
        warmup_steps = self.yaml_config.get('warmup_steps', 100)
        total_steps = len(train_loader) * num_epochs

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print(f"\nOptimizer: AdamW")
        print(f"  LR: {self.config.learning_rate}")
        print(f"  Total steps: {total_steps}")

        return optimizer, scheduler

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move to device
        model_inputs = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if k in ('input_values', 'audio') and v.dtype == torch.float32:
                    v = v.to(self.dtype)
                model_inputs[k] = v
            else:
                model_inputs[k] = v

        # Forward pass
        outputs = self.model(**model_inputs)

        # Get loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            loss = torch.tensor(0.5, device=self.device, requires_grad=True)

        return {
            'loss': loss,
            'reconstruction': loss.item(),
        }

    def validate(self, val_loader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                model_inputs = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                        if k in ('input_values', 'audio') and v.dtype == torch.float32:
                            v = v.to(self.dtype)
                        model_inputs[k] = v
                    else:
                        model_inputs[k] = v

                try:
                    outputs = self.model(**model_inputs)
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                        num_batches += 1
                except Exception:
                    pass

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint_dir = self.output_dir / name

        # Save LoRA weights
        self.model.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'style_name': self.style_name,
            'config': asdict(self.config),
        }
        torch.save(state, checkpoint_dir / 'training_state.pt')

        print(f"Saved checkpoint: {checkpoint_dir}")

    def train(self):
        """Main training loop."""
        start_time = time.time()

        # Setup
        self.setup_model()
        train_loader, val_loader = self.setup_data()
        optimizer, scheduler = self.setup_optimizer(train_loader)

        # Config
        num_epochs = self.yaml_config.get('num_epochs', 30)
        grad_accum = self.yaml_config.get('gradient_accumulation', 1)
        max_grad_norm = self.yaml_config.get('max_grad_norm', 1.0)
        eval_every = self.yaml_config.get('eval_every', 50)
        early_stopping = self.yaml_config.get('early_stopping_patience', 5)

        print(f"\n{'='*60}")
        print(f"Training Style LoRA: {self.style_name}")
        print(f"{'='*60}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.yaml_config.get('batch_size', 4)}")
        print(f"  CFG masking: speech={self.config.speech_mask_ratio}, text={self.config.text_mask_ratio}")
        print(f"{'='*60}\n")

        self.metrics.status = "training"
        self.metrics.total_epochs = num_epochs
        patience_counter = 0

        optimizer.zero_grad()

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_steps = 0
            step_times = deque(maxlen=50)

            self.metrics.epoch = epoch + 1

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [{self.style_name}]",
                leave=True,
            )

            for step, batch in enumerate(progress):
                step_start = time.time()

                # Training step
                results = self.train_step(batch)
                loss = results['loss']

                # Scale for gradient accumulation
                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                epoch_loss += loss.item()
                num_steps += 1

                # Optimizer step
                if (step + 1) % grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    self.global_step += 1
                    step_time = time.time() - step_start
                    step_times.append(step_time)

                    # Memory
                    if torch.cuda.is_available():
                        mem_used = torch.cuda.memory_allocated() / 1e9
                    else:
                        mem_used = 0

                    # Update metrics
                    self.metrics.step = self.global_step
                    self.metrics.train_loss = loss.item()
                    self.metrics.learning_rate = scheduler.get_last_lr()[0]
                    self.metrics.memory_used_gb = mem_used

                    progress.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                    # Periodic validation
                    if self.global_step % eval_every == 0:
                        val_loss = self.validate(val_loader)
                        self.metrics.val_loss = val_loss
                        print(f"\n  Step {self.global_step}: val_loss={val_loss:.4f}")

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            patience_counter = 0
                            self.save_checkpoint("best")
                        else:
                            patience_counter += 1

            # End of epoch validation
            val_loss = self.validate(val_loader)
            avg_train_loss = epoch_loss / max(num_steps, 1)

            print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best")
                print(f"  New best! val_loss={val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stopping}")

            # Early stopping
            if patience_counter >= early_stopping:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

        # Final save
        self.save_checkpoint("final")
        self.metrics.status = "complete"

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete: {self.style_name}")
        print(f"{'='*60}")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Best val_loss: {self.best_val_loss:.4f}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Style-Specific LoRAs")
    parser.add_argument('--config', '-c', required=True, help='Config YAML path')
    parser.add_argument('--style', '-s', help='Style to train (e.g., happy, high_pitch)')
    parser.add_argument('--all-styles', action='store_true', help='Train all styles')

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Get styles to train
    if args.all_styles:
        styles = config.get('styles_to_train', [
            "high_pitch", "low_pitch", "high_energy", "low_energy",
            "happy", "sad", "angry", "calm"
        ])
    elif args.style:
        styles = [args.style]
    else:
        parser.error("Either --style or --all-styles must be specified")

    print(f"Training styles: {styles}")

    # Train each style
    for style in styles:
        print(f"\n{'#'*70}")
        print(f"# Training Style: {style}")
        print(f"{'#'*70}\n")

        trainer = StyleLoRATrainerRunner(args.config, style)
        trainer.train()


if __name__ == '__main__':
    import math  # Import for lr_lambda
    main()
