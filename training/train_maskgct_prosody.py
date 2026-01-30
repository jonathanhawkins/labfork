"""
MaskGCT Prosody Training Script

Trains the MaskGCT masked prosody prediction model for non-autoregressive
prosody-conditioned TTS. Key features:

1. **Mask Scheduling**: Cosine schedule starts with high mask ratio and decreases
2. **Two-Stage Training**: Semantic tokens first, then acoustic
3. **Prompt Conditioning**: Learn prosody from reference audio style
4. **Parallel Decoding**: Inference with iterative refinement for 2x+ speedup

Usage:
    python train_maskgct_prosody.py --config config/maskgct_prosody.yaml

    # Quick test run
    python train_maskgct_prosody.py --config config/maskgct_prosody.yaml --debug

Reference: MaskGCT (ICLR 2025)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

from maskgct_prosody import (
    MaskGCTConfig,
    MaskGCTProsodyModel,
    MaskGCTWithProsodyEncoder,
    MaskScheduler,
)
from prosody_conditioning import (
    ProsodyConfig,
    extract_prosody_for_conditioning,
)


class MaskGCTDataset(Dataset):
    """
    Dataset for MaskGCT masked prosody training.

    Extends base prosody dataset with:
    - Temporal prosody sequences (not just global average)
    - Prompt/reference prosody for style conditioning
    - Support for two-stage training (semantic → acoustic)
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        max_seq_len: int = 64,
        sample_rate: int = 24000,
        use_prompt: bool = True,
        prompt_ratio: float = 0.15,  # Fraction of sequence used as prompt
    ):
        self.manifest_path = Path(manifest_path)
        self.prosody_cache_dir = Path(prosody_cache_dir)
        self.max_seq_len = max_seq_len
        self.sample_rate = sample_rate
        self.use_prompt = use_prompt
        self.prompt_ratio = prompt_ratio

        # Load manifest
        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples")

        # Create prosody cache dir
        self.prosody_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load prosody (from cache or extract)
        prosody = self._get_prosody(sample, idx)

        # Expand to temporal sequence
        temporal_prosody = self._expand_to_temporal(prosody)

        # Split into prompt and target if using prompt conditioning
        if self.use_prompt:
            prompt_len = max(1, int(self.max_seq_len * self.prompt_ratio))
            prompt_prosody = {k: v[:, :prompt_len] for k, v in temporal_prosody.items()}
            target_prosody = {k: v[:, prompt_len:] for k, v in temporal_prosody.items()}
        else:
            prompt_prosody = None
            target_prosody = temporal_prosody

        # Get emotion label
        emotion_label = self._get_emotion_label(sample)

        result = {
            'prosody_semantic': target_prosody['semantic'].squeeze(0),
            'prosody_acoustic': target_prosody['acoustic'].squeeze(0),
            'prosody_rhythm': target_prosody['rhythm'].squeeze(0),
            'prosody_contour': target_prosody['contour'].squeeze(0),
            'emotion_label': emotion_label,
        }

        if prompt_prosody is not None:
            result['prompt_semantic'] = prompt_prosody['semantic'].squeeze(0)
            result['prompt_acoustic'] = prompt_prosody['acoustic'].squeeze(0)
            result['prompt_rhythm'] = prompt_prosody['rhythm'].squeeze(0)
            result['prompt_contour'] = prompt_prosody['contour'].squeeze(0)

        return result

    def _expand_to_temporal(
        self,
        prosody: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Expand global prosody to temporal sequence.

        For now, we create a sequence by adding slight variations to simulate
        temporal dynamics. In real usage, this would come from dense prosody
        extraction at multiple time points.
        """
        temporal = {}

        for key in ['semantic', 'acoustic', 'rhythm']:
            vec = prosody[key]  # [1, dim]
            # Expand to sequence with small variations
            expanded = vec.expand(1, self.max_seq_len, -1).clone()

            # Add temporal variation (smooth random walk)
            noise_scale = 0.1
            noise = torch.randn_like(expanded) * noise_scale
            # Smooth the noise
            for t in range(1, self.max_seq_len):
                noise[:, t] = 0.9 * noise[:, t-1] + 0.1 * noise[:, t]

            expanded = torch.clamp(expanded + noise, 0, 1)
            temporal[key] = expanded

        # Contour: split into segments
        contour = prosody['contour']  # [1, contour_dim]
        contour_dim = contour.shape[-1]

        # Create per-timestep contour (subsample or interpolate)
        if contour_dim >= self.max_seq_len:
            # Subsample
            indices = torch.linspace(0, contour_dim - 1, self.max_seq_len).long()
            temporal_contour = contour[:, indices]
        else:
            # Interpolate
            temporal_contour = F.interpolate(
                contour.unsqueeze(1),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        # Expand to [1, seq_len, contour_dim] where each timestep has full contour
        temporal['contour'] = contour.expand(1, self.max_seq_len, -1).clone()

        return temporal

    def _get_prosody(self, sample: dict, idx: int) -> Dict[str, torch.Tensor]:
        """Get prosody from cache or extract."""
        import hashlib

        # Use audio path hash as cache key
        audio_path = sample.get('audio_path') or sample.get('path') or sample.get('audio', '')
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        cache_path = self.prosody_cache_dir / f"prosody_{path_hash}.pt"

        if cache_path.exists():
            return torch.load(cache_path)

        # Check if prosody is in manifest
        if 'prosody' in sample:
            prosody = extract_prosody_for_conditioning(sample['prosody'])
            torch.save(prosody, cache_path)
            return prosody

        # Return default prosody
        config = ProsodyConfig()
        return {
            'semantic': torch.zeros(1, config.semantic_dim),
            'acoustic': torch.zeros(1, config.acoustic_dim),
            'rhythm': torch.zeros(1, config.rhythm_dim),
            'contour': torch.zeros(1, config.contour_dim),
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
    """Collate batch."""
    result = {
        'prosody_semantic': torch.stack([item['prosody_semantic'] for item in batch]),
        'prosody_acoustic': torch.stack([item['prosody_acoustic'] for item in batch]),
        'prosody_rhythm': torch.stack([item['prosody_rhythm'] for item in batch]),
        'prosody_contour': torch.stack([item['prosody_contour'] for item in batch]),
        'emotion_label': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long),
    }

    if 'prompt_semantic' in batch[0]:
        result['prompt_semantic'] = torch.stack([item['prompt_semantic'] for item in batch])
        result['prompt_acoustic'] = torch.stack([item['prompt_acoustic'] for item in batch])
        result['prompt_rhythm'] = torch.stack([item['prompt_rhythm'] for item in batch])
        result['prompt_contour'] = torch.stack([item['prompt_contour'] for item in batch])

    return result


class MaskGCTTrainer:
    """
    Trainer for MaskGCT masked prosody prediction.

    Training procedure:
    1. Tokenize prosody features into discrete tokens
    2. Apply random masking with scheduled ratio
    3. Predict masked tokens with transformer
    4. Compute cross-entropy loss on masked positions
    5. Train tokenizer with commitment loss (VQ-VAE style)
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()

        # Build MaskGCT config
        self.maskgct_config = MaskGCTConfig(
            hidden_size=config.get('hidden_size', 2048),
            num_semantic_tokens=config.get('num_semantic_tokens', 32),
            num_acoustic_tokens=config.get('num_acoustic_tokens', 128),
            max_prosody_length=config.get('max_seq_len', 64),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            feedforward_dim=config.get('feedforward_dim', 4096),
            dropout=config.get('dropout', 0.1),
            initial_mask_ratio=config.get('initial_mask_ratio', 0.9),
            final_mask_ratio=config.get('final_mask_ratio', 0.1),
            mask_schedule=config.get('mask_schedule', 'cosine'),
            num_parallel_iterations=config.get('num_parallel_iterations', 4),
            temperature=config.get('temperature', 0.8),
        )

        # Setup model
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Update mask scheduler with total steps
        total_steps = config.get('num_epochs', 30) * config.get('steps_per_epoch', 100)
        self.model.maskgct.mask_scheduler.total_steps = total_steps

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _setup_model(self) -> MaskGCTWithProsodyEncoder:
        """Setup MaskGCT model with prosody encoder bridge."""
        print("Setting up MaskGCT model...")

        model = MaskGCTWithProsodyEncoder(
            self.maskgct_config,
            hidden_size=self.config.get('hidden_size', 2048),
            num_prosody_tokens=self.config.get('num_prosody_tokens', 4),
        )
        model = model.to(self.device)

        # Print parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable:,}")

        return model

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.99),
        )

    def _build_prosody_dict(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Build prosody dict from batch."""
        return {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }

    def _build_prompt_embeds(self, batch: Dict) -> Optional[torch.Tensor]:
        """Build prompt embeddings from batch."""
        if 'prompt_semantic' not in batch:
            return None

        # Encode prompt prosody to embeddings
        # First, average over time dimension if present (temporal prosody)
        prompt_semantic = batch['prompt_semantic'].to(self.device)
        prompt_acoustic = batch['prompt_acoustic'].to(self.device)
        prompt_rhythm = batch['prompt_rhythm'].to(self.device)
        prompt_contour = batch['prompt_contour'].to(self.device)

        # If 3D (batch, seq, dim), average over time dimension
        if prompt_semantic.dim() == 3:
            prompt_semantic = prompt_semantic.mean(dim=1)
            prompt_acoustic = prompt_acoustic.mean(dim=1)
            prompt_rhythm = prompt_rhythm.mean(dim=1)
            prompt_contour = prompt_contour.mean(dim=1)

        prompt_dict = {
            'semantic': prompt_semantic,
            'acoustic': prompt_acoustic,
            'rhythm': prompt_rhythm,
            'contour': prompt_contour,
        }

        # Use the model's prosody encoder to get embeddings
        with torch.no_grad():
            prefix = self.model.encode_prosody_from_dict(prompt_dict)

        return prefix

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Build inputs
        prosody_dict = self._build_prosody_dict(batch)
        prompt_embeds = self._build_prompt_embeds(batch)

        # Forward pass with masked prediction
        output = self.model(prosody_dict, prompt_embeds)

        loss = output['loss']
        loss.backward()

        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()
        self.global_step += 1

        return {
            'loss': loss.item(),
            'semantic_loss': output['semantic_loss'].item(),
            'acoustic_loss': output['acoustic_loss'].item(),
            'commitment_loss': output['commitment_loss'].item(),
            'mask_ratio': output['mask_ratio'].item(),
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()

        total_loss = 0.0
        total_semantic = 0.0
        total_acoustic = 0.0
        num_batches = 0

        for batch in val_loader:
            prosody_dict = self._build_prosody_dict(batch)
            prompt_embeds = self._build_prompt_embeds(batch)

            # Use fixed mask ratio for validation
            output = self.model.maskgct.compute_masked_loss(
                prosody_dict, prompt_embeds, mask_ratio=0.5
            )

            total_loss += output['loss'].item()
            total_semantic += output['semantic_loss'].item()
            total_acoustic += output['acoustic_loss'].item()
            num_batches += 1

        return {
            'val_loss': total_loss / max(1, num_batches),
            'val_semantic_loss': total_semantic / max(1, num_batches),
            'val_acoustic_loss': total_acoustic / max(1, num_batches),
        }

    @torch.no_grad()
    def evaluate_parallel_generation(
        self,
        val_loader: DataLoader,
        num_samples: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate parallel generation quality and speed.

        Compares generated prosody tokens with ground truth.
        """
        self.model.eval()

        total_semantic_acc = 0.0
        total_acoustic_acc = 0.0
        total_time = 0.0
        num_evaluated = 0

        for batch in val_loader:
            if num_evaluated >= num_samples:
                break

            prosody_dict = self._build_prosody_dict(batch)
            prompt_embeds = self._build_prompt_embeds(batch)

            batch_size = prosody_dict['semantic'].shape[0]

            # Get ground truth tokens
            _, gt_semantic, _ = self.model.maskgct.tokenizer.encode_semantic(
                prosody_dict['semantic'][:, 0, :]  # First timestep
            )
            _, gt_acoustic, _ = self.model.maskgct.tokenizer.encode_acoustic(
                prosody_dict['acoustic'][:, 0, :],
                prosody_dict['rhythm'][:, 0, :],
                prosody_dict['contour'][:, 0, :],
            )

            # Generate with timing
            start_time = time.time()
            gen_semantic, gen_acoustic = self.model.maskgct.generate_parallel(
                prompt_embeds, batch_size
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            # Compute accuracy (first timestep)
            semantic_acc = (gen_semantic[:, 0] == gt_semantic).float().mean().item()
            acoustic_acc = (gen_acoustic[:, 0] == gt_acoustic).float().mean().item()

            total_semantic_acc += semantic_acc
            total_acoustic_acc += acoustic_acc
            num_evaluated += batch_size

        return {
            'parallel_semantic_acc': total_semantic_acc / max(1, num_evaluated) * batch_size,
            'parallel_acoustic_acc': total_acoustic_acc / max(1, num_evaluated) * batch_size,
            'parallel_gen_time_ms': total_time / max(1, num_evaluated) * batch_size * 1000,
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
                    print(
                        f"Epoch {epoch+1}, Step {self.global_step}: "
                        f"loss={avg_loss:.4f} "
                        f"(sem={metrics['semantic_loss']:.4f}, "
                        f"aco={metrics['acoustic_loss']:.4f}, "
                        f"commit={metrics['commitment_loss']:.4f}) "
                        f"mask={metrics['mask_ratio']:.2f}"
                    )

            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                print(f"Validation: loss={val_metrics['val_loss']:.4f}")

                # Evaluate parallel generation periodically
                if (epoch + 1) % 5 == 0:
                    gen_metrics = self.evaluate_parallel_generation(val_loader)
                    print(
                        f"Parallel gen: sem_acc={gen_metrics['parallel_semantic_acc']:.2%}, "
                        f"aco_acc={gen_metrics['parallel_acoustic_acc']:.2%}, "
                        f"time={gen_metrics['parallel_gen_time_ms']:.1f}ms"
                    )

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best')

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

        self.save_checkpoint('final')
        print("\nTraining complete!")

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
                'max_prosody_length': self.maskgct_config.max_prosody_length,
                'num_layers': self.maskgct_config.num_layers,
                'num_heads': self.maskgct_config.num_heads,
                'num_parallel_iterations': self.maskgct_config.num_parallel_iterations,
            },
        }

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']

        trainer = cls(config)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']

        return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/maskgct_prosody.yaml')
    parser.add_argument('--manifest', type=str, help='Path to training manifest')
    parser.add_argument('--debug', action='store_true', help='Debug mode with small run')
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
            'hidden_size': 2048,
            'num_prosody_tokens': 4,
            'max_seq_len': 64,
            'num_layers': 6,
            'num_heads': 8,
            'learning_rate': 1e-4,
            'num_epochs': 30,
            'batch_size': 8,
            'output_dir': '../models/checkpoints/maskgct',
            'initial_mask_ratio': 0.9,
            'final_mask_ratio': 0.1,
            'mask_schedule': 'cosine',
        }

    if args.debug:
        config['num_epochs'] = 2
        config['batch_size'] = 2
        print("Debug mode: reduced epochs and batch size")

    print("MaskGCT Prosody Training")
    print("=" * 50)
    print(f"Config: {config}")

    # Create trainer
    trainer = MaskGCTTrainer(config)

    # Create dataset
    manifest_path = args.manifest or config.get('train_manifest')
    if manifest_path:
        dataset = MaskGCTDataset(
            manifest_path,
            config.get('prosody_cache_dir', '../data/prosody_cache'),
            max_seq_len=config.get('max_seq_len', 64),
            use_prompt=config.get('use_prompt', True),
        )

        train_loader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            collate_fn=collate_fn,
        )

        # Validation loader
        val_loader = None
        val_manifest = config.get('val_manifest')
        if val_manifest:
            val_dataset = MaskGCTDataset(
                val_manifest,
                config.get('prosody_cache_dir', '../data/prosody_cache'),
                max_seq_len=config.get('max_seq_len', 64),
                use_prompt=config.get('use_prompt', True),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('batch_size', 8),
                shuffle=False,
                collate_fn=collate_fn,
            )

        trainer.train(
            train_loader,
            val_loader=val_loader,
            num_epochs=config.get('num_epochs', 30),
        )
    else:
        print("\nNo manifest provided. Running module test...")

        # Test with dummy data
        print("\nTesting MaskGCT components...")

        # Create dummy batch
        batch_size = 2
        seq_len = 16

        dummy_batch = {
            'prosody_semantic': torch.rand(batch_size, seq_len, trainer.maskgct_config.semantic_dim),
            'prosody_acoustic': torch.rand(batch_size, seq_len, trainer.maskgct_config.acoustic_dim),
            'prosody_rhythm': torch.rand(batch_size, seq_len, trainer.maskgct_config.rhythm_dim),
            'prosody_contour': torch.rand(batch_size, seq_len, trainer.maskgct_config.contour_dim),
            'prompt_semantic': torch.rand(batch_size, 4, trainer.maskgct_config.semantic_dim),
            'prompt_acoustic': torch.rand(batch_size, 4, trainer.maskgct_config.acoustic_dim),
            'prompt_rhythm': torch.rand(batch_size, 4, trainer.maskgct_config.rhythm_dim),
            'prompt_contour': torch.rand(batch_size, 4, trainer.maskgct_config.contour_dim),
            'emotion_label': torch.tensor([1, 2]),
        }

        # Test training step
        print("\nTesting training step...")
        metrics = trainer.train_step(dummy_batch)
        print(f"Train step metrics: {metrics}")

        # Test parallel generation
        print("\nTesting parallel generation...")
        prompt_embeds = torch.rand(batch_size, 4, trainer.maskgct_config.hidden_size).to(trainer.device)
        with torch.no_grad():
            sem_tokens, aco_tokens = trainer.model.maskgct.generate_parallel(
                prompt_embeds, batch_size, seq_len=16
            )
        print(f"Generated semantic tokens: {sem_tokens.shape}")
        print(f"Generated acoustic tokens: {aco_tokens.shape}")

        print("\nMaskGCT training module ready!")
        print("\nTo train, run with: --manifest path/to/manifest.json")
        print("\nKey features:")
        print("  - Mask-and-predict training for parallel prosody generation")
        print("  - Cosine mask scheduling (0.9 → 0.1)")
        print("  - VQ-VAE tokenization with commitment loss")
        print("  - Iterative parallel decoding for 2x+ inference speedup")


if __name__ == "__main__":
    main()
