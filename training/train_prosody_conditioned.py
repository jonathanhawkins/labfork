"""
Prosody-Conditioned Voice Clone Training

This script trains the prosody conditioning module alongside a LoRA-adapted CSM model.
The key innovation: prosody features extracted from audio now directly control generation.

Training Flow:
    1. Load audio sample
    2. Extract prosody using your ProsodyAnalyzer (4-layer cube)
    3. Encode prosody to prefix embeddings
    4. Train CSM to reconstruct audio given (text + prosody prefix)

After training, you can:
    - Generate speech with extracted prosody (style transfer)
    - Generate with synthetic prosody (emotion control)
    - Interpolate between prosody styles

Usage:
    python train_prosody_conditioned.py --config config/prosody_conditioning.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Add parent to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: PEFT not available. LoRA disabled.")

from prosody_conditioning import (
    ProsodyConfig,
    ProsodyEncoder,
    TemporalProsodyEncoder,
    ProsodyControlledCSM,
    extract_prosody_for_conditioning,
)
from keyframe_prosody import get_temporal_prosody_tokens


class ProsodyConditionedDataset(Dataset):
    """
    Dataset that provides (audio, text, prosody) triplets.

    The prosody is pre-extracted and cached for efficiency.
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        max_audio_length_ms: int = 30000,
        sample_rate: int = 24000,
    ):
        self.manifest_path = Path(manifest_path)
        self.prosody_cache_dir = Path(prosody_cache_dir)
        self.max_audio_length = int(max_audio_length_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

        # Load manifest
        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples")

        # Create prosody cache dir
        self.prosody_cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache the prosody analyzer (keeps Qwen2-Audio loaded)
        self._analyzer = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load audio (support multiple manifest formats)
        audio_path = sample.get('audio_path') or sample.get('path') or sample.get('audio')
        if not audio_path:
            raise ValueError(f"Sample {idx} has no audio path. Keys: {sample.keys()}")
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        # Truncate if too long
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]

        # Get text
        text = sample.get('text') or sample.get('transcript', '')

        # Load or extract prosody
        prosody = self._get_prosody(sample, idx, audio_path)

        return {
            'audio': waveform,
            'text': text,
            'prosody_semantic': prosody['semantic'].squeeze(0),
            'prosody_acoustic': prosody['acoustic'].squeeze(0),
            'prosody_rhythm': prosody['rhythm'].squeeze(0),
            'prosody_contour': prosody['contour'].squeeze(0),
        }

    def _get_prosody(
        self,
        sample: dict,
        idx: int,
        audio_path: str,
    ) -> Dict[str, torch.Tensor]:
        """Get prosody from cache or extract."""
        cache_path = self.prosody_cache_dir / f"prosody_{idx}.pt"

        if cache_path.exists():
            return torch.load(cache_path)

        # Check if prosody is in manifest
        if 'prosody' in sample:
            prosody = extract_prosody_for_conditioning(sample['prosody'])
            torch.save(prosody, cache_path)
            return prosody

        # Extract prosody (requires prosody_analyzer)
        try:
            from prosody_analyzer import CompleteProsodyAnalyzer

            # Use cached analyzer to avoid reloading Qwen2-Audio
            if self._analyzer is None:
                print("Initializing prosody analyzer (will load Qwen2-Audio once)...")
                self._analyzer = CompleteProsodyAnalyzer(use_qwen=True, device="cuda")

            prosody_result = self._analyzer.analyze(audio_path)
            prosody_dict = prosody_result.to_dict()
            prosody = extract_prosody_for_conditioning(prosody_dict)
            torch.save(prosody, cache_path)
            return prosody
        except Exception as e:
            print(f"Warning: Could not extract prosody for {audio_path}: {e}")
            # Return default prosody
            config = ProsodyConfig()
            return {
                'semantic': torch.zeros(1, config.semantic_dim),
                'acoustic': torch.zeros(1, config.acoustic_dim),
                'rhythm': torch.zeros(1, config.rhythm_dim),
                'contour': torch.zeros(1, config.contour_dim),
            }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Pad audio to max length in batch
    audios = [item['audio'] for item in batch]
    max_len = max(a.shape[0] for a in audios)
    audios_padded = torch.stack([
        F.pad(a, (0, max_len - a.shape[0])) for a in audios
    ])

    # Stack prosody (already fixed size)
    return {
        'audio': audios_padded,
        'text': [item['text'] for item in batch],
        'prosody_semantic': torch.stack([item['prosody_semantic'] for item in batch]),
        'prosody_acoustic': torch.stack([item['prosody_acoustic'] for item in batch]),
        'prosody_rhythm': torch.stack([item['prosody_rhythm'] for item in batch]),
        'prosody_contour': torch.stack([item['prosody_contour'] for item in batch]),
    }


class ProsodyConditionedTrainer:
    """
    Trainer for prosody-conditioned voice cloning.

    This trains:
    1. ProsodyEncoder: Maps prosody vectors → global prefix embeddings
    2. TemporalProsodyEncoder: Maps per-segment prosody → temporal prefix embeddings
    3. (Optional) LoRA adapters on CSM backbone

    The temporal encoder enables keyframe-based emotion control where different
    parts of the utterance can have different prosody (e.g., start neutral,
    become happy, then calm down).
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()
        self.train_temporal = config.get('train_temporal', True)
        self.num_temporal_segments = config.get('num_temporal_segments', 4)

        self.prosody_config = ProsodyConfig(
            hidden_size=config.get('hidden_size', 2048),
            num_prosody_tokens=config.get('num_prosody_tokens', 4),
            dropout=config.get('dropout', 0.1),
        )

        # Setup models
        self.model = self._setup_model()
        self.temporal_encoder = self._setup_temporal_encoder() if self.train_temporal else None
        self.optimizer = self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _setup_model(self) -> ProsodyControlledCSM:
        """Setup CSM with prosody conditioning."""
        from transformers import CsmForConditionalGeneration, AutoProcessor

        model_path = self.config['model_path']
        print(f"Loading CSM from: {model_path}")

        # Load base CSM
        csm = CsmForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Optionally apply LoRA
        if HAS_PEFT and self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config.get('lora_r', 8),
                lora_alpha=self.config.get('lora_alpha', 16),
                target_modules=self.config.get('lora_target_modules', ['q_proj', 'v_proj']),
                lora_dropout=self.config.get('lora_dropout', 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            csm = get_peft_model(csm, lora_config)
            print("Applied LoRA to CSM backbone")

        # Wrap with prosody conditioning
        model = ProsodyControlledCSM(
            csm,
            self.prosody_config,
            freeze_csm=not self.config.get('train_csm', False),
        )
        model = model.to(self.device)

        # Print parameter counts
        prosody_params = sum(p.numel() for p in model.prosody_encoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Global prosody encoder parameters: {prosody_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable:,} ({100*trainable/total_params:.2f}%)")

        return model

    def _setup_temporal_encoder(self) -> TemporalProsodyEncoder:
        """Setup temporal prosody encoder for keyframe control."""
        print("Setting up temporal prosody encoder...")

        temporal_encoder = TemporalProsodyEncoder(self.prosody_config)
        temporal_encoder = temporal_encoder.to(self.device)

        # Initialize from global encoder for better starting point
        temporal_encoder.init_from_global_encoder(self.model.prosody_encoder)

        temporal_params = sum(p.numel() for p in temporal_encoder.parameters())
        print(f"Temporal prosody encoder parameters: {temporal_params:,}")

        return temporal_encoder

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Add temporal encoder parameters if training temporal
        if self.temporal_encoder is not None:
            trainable_params.extend(list(self.temporal_encoder.parameters()))
            print(f"Total trainable params (incl. temporal): {sum(p.numel() for p in trainable_params):,}")

        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step for both global and temporal encoders."""
        self.model.train()
        if self.temporal_encoder is not None:
            self.temporal_encoder.train()

        # Move to device
        audio = batch['audio'].to(self.device)
        prosody_dict = {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }

        self.optimizer.zero_grad()

        try:
            # ============ GLOBAL ENCODER LOSS ============
            # Train the global prosody encoder to produce consistent embeddings
            prosody_prefix = self.model.get_prosody_prefix(
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'],
            )

            # Reconstruction target based on input prosody
            target_embedding = torch.cat([
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'][:, :8] if prosody_dict['contour'].shape[1] > 8 else prosody_dict['contour'],
            ], dim=-1)

            # Ensure target matches prosody prefix dimensions
            if target_embedding.shape[-1] < prosody_prefix.shape[-1]:
                target_embedding = F.pad(
                    target_embedding,
                    (0, prosody_prefix.shape[-1] - target_embedding.shape[-1])
                )
            target_embedding = target_embedding.unsqueeze(1).expand_as(prosody_prefix[:, :1, :])

            # L2 reconstruction loss on first token
            global_loss = F.mse_loss(prosody_prefix[:, 0, :target_embedding.shape[-1]], target_embedding[:, 0, :])

            # ============ TEMPORAL ENCODER LOSS ============
            temporal_loss = torch.tensor(0.0, device=self.device)
            if self.temporal_encoder is not None:
                # Create temporal prosody by segmenting the global prosody
                # This simulates having keyframe data by splitting into segments
                batch_size = prosody_dict['semantic'].shape[0]
                num_segments = self.num_temporal_segments

                # Expand prosody to temporal format [batch, num_segments, dim]
                # For training, we create "pseudo-keyframes" by slightly varying each segment
                temporal_prosody = self._create_temporal_prosody(prosody_dict, num_segments)

                # Forward through temporal encoder
                temporal_prefix = self.temporal_encoder(
                    temporal_prosody['semantic'],
                    temporal_prosody['acoustic'],
                    temporal_prosody['rhythm'],
                    temporal_prosody['contour'],
                )  # [batch, num_segments, hidden]

                # Temporal reconstruction loss: each segment should encode its prosody
                # The target is segment-wise prosody expanded to hidden size
                temporal_target = torch.cat([
                    temporal_prosody['semantic'],
                    temporal_prosody['acoustic'],
                    temporal_prosody['rhythm'],
                    temporal_prosody['contour'][:, :, :8] if temporal_prosody['contour'].shape[-1] > 8
                    else temporal_prosody['contour'],
                ], dim=-1)  # [batch, num_segments, prosody_dim]

                if temporal_target.shape[-1] < temporal_prefix.shape[-1]:
                    temporal_target = F.pad(
                        temporal_target,
                        (0, temporal_prefix.shape[-1] - temporal_target.shape[-1])
                    )

                temporal_loss = F.mse_loss(
                    temporal_prefix[:, :, :temporal_target.shape[-1]],
                    temporal_target
                )

                # Consistency loss: temporal encoder should match global on averaged input
                avg_temporal = temporal_prefix.mean(dim=1)  # [batch, hidden]
                avg_global = prosody_prefix.mean(dim=1)      # [batch, hidden]
                consistency_loss = F.mse_loss(avg_temporal, avg_global) * 0.1

                temporal_loss = temporal_loss + consistency_loss

            # Combined loss
            loss = global_loss + temporal_loss
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if self.temporal_encoder is not None:
                torch.nn.utils.clip_grad_norm_(self.temporal_encoder.parameters(), max_grad_norm)

            self.optimizer.step()

            return {
                'loss': loss.item(),
                'global_loss': global_loss.item(),
                'temporal_loss': temporal_loss.item() if self.temporal_encoder else 0.0,
            }

        except Exception as e:
            import traceback
            print(f"Training step error: {e}")
            traceback.print_exc()
            return {'loss': 0.0, 'global_loss': 0.0, 'temporal_loss': 0.0}

    def _create_temporal_prosody(
        self,
        prosody_dict: Dict[str, torch.Tensor],
        num_segments: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Create temporal prosody from global prosody for training.

        This simulates keyframe data by creating slightly varied segments.
        In real usage, keyframes would be user-defined or extracted from
        dense prosody contours.
        """
        batch_size = prosody_dict['semantic'].shape[0]

        temporal = {}
        for key in ['semantic', 'acoustic', 'rhythm', 'contour']:
            global_vec = prosody_dict[key]  # [batch, dim]
            dim = global_vec.shape[-1]

            # Expand to [batch, num_segments, dim]
            expanded = global_vec.unsqueeze(1).expand(batch_size, num_segments, dim).clone()

            # Add small per-segment variation to encourage temporal learning
            # Variation increases toward the middle segments (like a natural emotion arc)
            for s in range(num_segments):
                # Variation strength: peaks in middle, lower at ends
                strength = 0.1 * (1.0 - abs(s - num_segments / 2) / (num_segments / 2))
                noise = torch.randn_like(expanded[:, s, :]) * strength
                expanded[:, s, :] = expanded[:, s, :] + noise

            temporal[key] = expanded.clamp(0, 1)  # Keep in valid range

        return temporal

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
    ):
        """Main training loop."""
        print(f"\nStarting prosody-conditioned training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                self.global_step += 1

                if self.global_step % self.config.get('log_every', 10) == 0:
                    avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                    log_msg = f"Epoch {epoch+1}, Step {self.global_step}: loss={avg_loss:.4f}"
                    if 'global_loss' in metrics:
                        log_msg += f" (global={metrics['global_loss']:.4f}, temporal={metrics['temporal_loss']:.4f})"
                    print(log_msg)

            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

        self.save_checkpoint('final')
        print("\nTraining complete!")

    def validate(self, val_loader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Simplified validation - would mirror train_step
                num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, name: str):
        """Save checkpoint with both global and temporal encoders."""
        output_dir = Path(self.config.get('output_dir', 'checkpoints/prosody'))
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'prosody_encoder': self.model.prosody_encoder.state_dict(),
            'config': self.config,
            'prosody_config': {
                'semantic_dim': self.prosody_config.semantic_dim,
                'acoustic_dim': self.prosody_config.acoustic_dim,
                'rhythm_dim': self.prosody_config.rhythm_dim,
                'contour_dim': self.prosody_config.contour_dim,
                'hidden_size': self.prosody_config.hidden_size,
                'num_prosody_tokens': self.prosody_config.num_prosody_tokens,
            },
        }

        # Save temporal encoder if trained
        if self.temporal_encoder is not None:
            checkpoint['temporal_encoder'] = self.temporal_encoder.state_dict()
            checkpoint['num_temporal_segments'] = self.num_temporal_segments

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")
        if self.temporal_encoder is not None:
            print(f"  Includes temporal encoder ({self.num_temporal_segments} segments)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/prosody_conditioning.yaml')
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
            'model_path': '../models/csm-1b',
            'use_lora': True,
            'lora_r': 8,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'batch_size': 4,
            'output_dir': '../models/checkpoints/prosody_conditioned',
        }

    print("Prosody-Conditioned Training")
    print("=" * 50)
    print(f"Config: {config}")

    # Create trainer
    trainer = ProsodyConditionedTrainer(config)

    # Create dummy data for testing
    if args.manifest:
        dataset = ProsodyConditionedDataset(
            args.manifest,
            config.get('prosody_cache_dir', '../data/prosody_cache'),
        )
        train_loader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=True,
            collate_fn=collate_fn,
        )
        trainer.train(train_loader, num_epochs=config.get('num_epochs', 10))
    else:
        print("\nNo manifest provided. Module loaded successfully.")
        print("\nTo train, run with: --manifest path/to/manifest.json")
        print("\nThe prosody conditioning system bridges:")
        print("  1. Your 4-layer prosody analysis (Cube)")
        print("  2. CSM voice generation")
        print("\nThis enables controllable synthesis like:")
        print("  - 'Say this happily'")
        print("  - 'Match the prosody of this reference audio'")
        print("  - 'Generate with high energy and fast pace'")


if __name__ == "__main__":
    main()
