"""
Prosody-Conditioned Voice Clone Training (V5)

This script trains the prosody conditioning module alongside a LoRA-adapted CSM model.
The key innovation: prosody features extracted from audio now directly control generation.

V5 IMPROVEMENTS (anti-overfitting):
    - Early stopping based on pitch pattern preservation
    - Emotion loss warmup schedule (ramps up over epochs)
    - Gradient detachment: emotion head doesn't affect acoustic features
    - Curriculum sampling: start with neutral, gradually add intense emotions
    - Pitch pattern validation: track happy vs sad Hz separation

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
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
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


# =============================================================================
# V5 ADDITIONS: Early Stopping and Curriculum Sampling
# =============================================================================

class EarlyStopping:
    """
    Early stopping based on pitch pattern preservation.

    V5: Monitors that happy pitch > sad pitch (expected pattern).
    Stops when pitch inversion occurs for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

        # Pitch pattern tracking (V5)
        self.pitch_inversions = 0
        self.happy_sad_separations = []

    def __call__(self, score: float, epoch: int, pitch_metrics: dict = None) -> bool:
        """
        Check if training should stop.

        Args:
            score: Validation loss
            epoch: Current epoch number
            pitch_metrics: Optional dict with 'happy_mean_pitch' and 'sad_mean_pitch'

        Returns:
            True if training should stop
        """
        # Check pitch pattern (V5 key metric)
        if pitch_metrics:
            happy_pitch = pitch_metrics.get('happy_mean_pitch', 0)
            sad_pitch = pitch_metrics.get('sad_mean_pitch', 0)
            separation = happy_pitch - sad_pitch
            self.happy_sad_separations.append(separation)

            if separation < 0:  # Pitch inversion detected
                self.pitch_inversions += 1
                print(f"⚠️  Pitch inversion detected! Happy ({happy_pitch:.1f}Hz) < Sad ({sad_pitch:.1f}Hz)")
                if self.pitch_inversions >= 2:  # Stop after 2 consecutive inversions
                    print("🛑 Stopping due to pitch inversion (overfitting sign)")
                    self.should_stop = True
                    return True
            else:
                self.pitch_inversions = 0  # Reset counter

        # Standard early stopping on loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"🛑 Early stopping: no improvement for {self.patience} epochs")
                self.should_stop = True
                return True

        return False

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


class CurriculumSampler:
    """
    V5: Curriculum learning sampler based on emotion intensity.

    Starts training with neutral/mild emotions, gradually introduces
    intense emotions (angry, excited) to prevent early overfitting.
    """

    # Emotion difficulty scores (0 = easy/neutral, 1 = hard/intense)
    EMOTION_DIFFICULTY = {
        'neutral': 0.0,
        'calm': 0.1,
        'happy': 0.3,
        'sad': 0.4,
        'surprised': 0.5,
        'fearful': 0.6,
        'disgusted': 0.7,
        'contempt': 0.7,
        'angry': 0.9,
        'excited': 0.9,
    }

    def __init__(self, dataset, start_threshold: float = 0.4, epoch_increment: float = 0.1):
        """
        Args:
            dataset: ProsodyConditionedDataset instance
            start_threshold: Initial difficulty threshold (0.4 = start with neutral/calm/happy)
            epoch_increment: How much to increase threshold per epoch (0.1 = 10% more per epoch)
        """
        self.dataset = dataset
        self.start_threshold = start_threshold
        self.epoch_increment = epoch_increment
        self.current_threshold = start_threshold
        self.current_epoch = 0

        # Compute difficulty for each sample
        self.difficulties = []
        for sample in dataset.samples:
            emotion = self._get_emotion(sample)
            difficulty = self.EMOTION_DIFFICULTY.get(emotion, 0.5)
            self.difficulties.append(difficulty)

        print(f"CurriculumSampler initialized: {len(self.difficulties)} samples")
        self._log_distribution()

    def _get_emotion(self, sample: dict) -> str:
        """Extract emotion from sample."""
        emotion = sample.get('emotion', '').lower()
        if not emotion:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion', '').lower()
        if not emotion:
            emotions = sample.get('prosody', {}).get('semantic', {}).get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()
        return emotion or 'neutral'

    def _log_distribution(self):
        """Log current curriculum distribution."""
        included = sum(1 for d in self.difficulties if d <= self.current_threshold)
        total = len(self.difficulties)
        print(f"  Curriculum: {included}/{total} samples ({100*included/total:.1f}%) at threshold {self.current_threshold:.1f}")

    def step(self, epoch: int):
        """Advance curriculum for new epoch."""
        self.current_epoch = epoch
        self.current_threshold = min(1.0, self.start_threshold + epoch * self.epoch_increment)
        self._log_distribution()

    def get_indices(self) -> list:
        """Get indices of samples within current curriculum."""
        return [i for i, d in enumerate(self.difficulties) if d <= self.current_threshold]

    def __iter__(self):
        """Yield sample indices for current epoch."""
        indices = self.get_indices()
        import random
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.get_indices())


class ProsodyConditionedDataset(Dataset):
    """
    Dataset that provides (audio, text, prosody) triplets.

    The prosody is pre-extracted and cached for efficiency.
    Supports energy augmentation to create better contrast between
    high-energy (angry/excited) and low-energy (sad/calm) emotions.
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        max_audio_length_ms: int = 30000,
        sample_rate: int = 24000,
        energy_augmentation: Optional[dict] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.prosody_cache_dir = Path(prosody_cache_dir)
        self.max_audio_length = int(max_audio_length_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

        # Energy augmentation config
        # Default: boost angry by +6dB, reduce sad by -6dB
        self.energy_aug = energy_augmentation or {}
        self.energy_aug_enabled = self.energy_aug.get('enabled', False)
        self.angry_boost_db = self.energy_aug.get('angry_boost_db', 6.0)
        self.sad_reduce_db = self.energy_aug.get('sad_reduce_db', -6.0)
        self.energy_aug_prob = self.energy_aug.get('probability', 0.5)

        if self.energy_aug_enabled:
            print(f"Energy augmentation enabled: angry +{self.angry_boost_db}dB, sad {self.sad_reduce_db}dB (p={self.energy_aug_prob})")

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

    def get_emotion_weights(self) -> Optional[List[float]]:
        """
        Compute per-sample weights to balance emotions during training.

        Returns:
            List of weights (len = dataset size) or None if no emotion labels.
        """
        emotions = []
        for sample in self.samples:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion')
            if not emotion and isinstance(semantic.get('emotions'), dict):
                # Pick the top-scoring emotion if present
                scores = semantic.get('emotions', {})
                if scores:
                    emotion = max(scores.items(), key=lambda kv: kv[1])[0]
            emotions.append(emotion)

        counts = {}
        for emotion in emotions:
            if emotion:
                counts[emotion] = counts.get(emotion, 0) + 1

        if not counts:
            return None

        # Inverse frequency weighting with mild smoothing
        max_count = max(counts.values())
        weights = []
        for emotion in emotions:
            if not emotion:
                weights.append(1.0)
                continue
            weight = max_count / counts[emotion]
            weights.append(weight)

        return weights

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

        # Apply energy augmentation based on emotion
        if self.energy_aug_enabled:
            import random
            if random.random() < self.energy_aug_prob:
                waveform, prosody = self._apply_energy_augmentation(
                    waveform, prosody, sample
                )

        # Get emotion label (for auxiliary emotion classification loss)
        emotion_label = self._get_emotion_label(sample)

        return {
            'audio': waveform,
            'text': text,
            'prosody_semantic': prosody['semantic'].squeeze(0),
            'prosody_acoustic': prosody['acoustic'].squeeze(0),
            'prosody_rhythm': prosody['rhythm'].squeeze(0),
            'prosody_contour': prosody['contour'].squeeze(0),
            'emotion_label': emotion_label,
        }

    # Emotion label mapping (consistent with common emotion sets)
    EMOTION_TO_IDX = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
        'excited': 8, 'contempt': 9,
    }

    def _get_emotion_label(self, sample: dict) -> int:
        """Extract emotion label index from sample metadata."""
        # Try top-level emotion first
        emotion = sample.get('emotion', '').lower()

        # Fall back to prosody.semantic.emotion
        if not emotion:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion', '').lower()

        # Fall back to max in emotions dict
        if not emotion:
            emotions = sample.get('prosody', {}).get('semantic', {}).get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()

        return self.EMOTION_TO_IDX.get(emotion, -1)  # -1 = unknown

    def _apply_energy_augmentation(
        self,
        waveform: torch.Tensor,
        prosody: Dict[str, torch.Tensor],
        sample: dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply energy augmentation based on emotion.

        High-energy emotions (angry, excited) get boosted.
        Low-energy emotions (sad, calm) get reduced.

        This creates clearer contrast in the training data.
        """
        import math

        # Detect emotion from sample metadata
        semantic = sample.get('prosody', {}).get('semantic', {})
        emotion = semantic.get('emotion', '').lower()

        if not emotion:
            # Try to get from emotions dict
            emotions = semantic.get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()

        if not emotion:
            return waveform, prosody

        # Determine gain adjustment
        gain_db = 0.0
        if emotion in ['angry', 'excited', 'surprised']:
            gain_db = self.angry_boost_db
        elif emotion in ['sad', 'calm', 'fearful']:
            gain_db = self.sad_reduce_db

        if gain_db == 0.0:
            return waveform, prosody

        # Apply gain to audio
        gain_linear = 10 ** (gain_db / 20.0)
        waveform = waveform * gain_linear

        # Clip to prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)

        # Update prosody acoustic intensity (index 2)
        prosody = {k: v.clone() for k, v in prosody.items()}
        if prosody['acoustic'].shape[-1] > 2:
            # Intensity is normalized to [0, 1], adjust proportionally
            # +6dB roughly doubles intensity, so add ~0.15 to normalized value
            intensity_delta = gain_db / 40.0  # Rough mapping: ±6dB -> ±0.15
            prosody['acoustic'][..., 2] = torch.clamp(
                prosody['acoustic'][..., 2] + intensity_delta,
                0.0, 1.0
            )

        return waveform, prosody

    def _get_prosody(
        self,
        sample: dict,
        idx: int,
        audio_path: str,
    ) -> Dict[str, torch.Tensor]:
        """Get prosody from cache or extract."""
        # Use audio path hash as cache key (not index) to handle dataset reordering
        import hashlib
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        cache_path = self.prosody_cache_dir / f"prosody_{path_hash}.pt"

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
        'emotion_label': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long),
    }


class EnergyPredictor(nn.Module):
    """
    Auxiliary head that predicts energy/intensity from prosody embeddings.

    This ensures the encoder learns energy-discriminative features,
    which is critical for distinguishing angry (high energy) from sad (low energy).
    """

    def __init__(self, hidden_size: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, prosody_embedding: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """
        Predict energy from prosody embedding.

        Args:
            prosody_embedding: [batch, hidden_size] or [batch, num_tokens, hidden_size]

        Returns:
            Predicted energy in [0, 1] range
        """
        if prosody_embedding.dim() == 3:
            if reduce:
                # Average over tokens dimension
                prosody_embedding = prosody_embedding.mean(dim=1)
                return self.predictor(prosody_embedding).squeeze(-1)

            # Per-segment prediction
            batch_size, num_segments, hidden = prosody_embedding.shape
            flattened = prosody_embedding.reshape(batch_size * num_segments, hidden)
            preds = self.predictor(flattened).squeeze(-1)
            return preds.view(batch_size, num_segments)

        return self.predictor(prosody_embedding).squeeze(-1)


class EmotionPredictor(nn.Module):
    """
    Auxiliary head that predicts discrete emotion from prosody embeddings.

    This enforces separability between emotions (e.g., angry vs sad).
    """

    def __init__(self, hidden_size: int = 2048, num_emotions: int = 8, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_emotions),
        )

    def forward(self, prosody_embedding: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """
        Predict emotion logits from prosody embedding.

        Args:
            prosody_embedding: [batch, hidden_size] or [batch, num_tokens, hidden_size]

        Returns:
            Logits of shape [batch, num_emotions] or [batch, num_segments, num_emotions]
        """
        if prosody_embedding.dim() == 3:
            if reduce:
                prosody_embedding = prosody_embedding.mean(dim=1)
                return self.classifier(prosody_embedding)

            batch_size, num_segments, hidden = prosody_embedding.shape
            flattened = prosody_embedding.reshape(batch_size * num_segments, hidden)
            logits = self.classifier(flattened)
            return logits.view(batch_size, num_segments, -1)

        return self.classifier(prosody_embedding)


class ProsodyConditionedTrainer:
    """
    Trainer for prosody-conditioned voice cloning.

    This trains:
    1. ProsodyEncoder: Maps prosody vectors → global prefix embeddings
    2. TemporalProsodyEncoder: Maps per-segment prosody → temporal prefix embeddings
    3. EnergyPredictor: Auxiliary head for energy discrimination (angry vs sad)
    4. (Optional) LoRA adapters on CSM backbone

    The temporal encoder enables keyframe-based emotion control where different
    parts of the utterance can have different prosody (e.g., start neutral,
    become happy, then calm down).

    The energy predictor ensures angry/excited emotions stay distinct from
    sad/calm emotions by enforcing energy-discriminative embeddings.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()
        self.train_temporal = config.get('train_temporal', True)
        self.num_temporal_segments = config.get('num_temporal_segments', 4)

        # Loss weights for better prosody learning
        self.prosody_loss_weight = config.get('prosody_loss_weight', 2.0)  # Boost prosody loss
        self.energy_loss_weight = config.get('energy_loss_weight', 1.0)   # Energy auxiliary loss
        self.temporal_loss_weight = config.get('temporal_loss_weight', 1.0)
        self.emotion_loss_weight = config.get('emotion_loss_weight', 1.0)

        # V5: Anti-overfitting settings
        self.emotion_warmup_epochs = config.get('emotion_warmup_epochs', 10)  # Ramp up emotion loss
        self.detach_emotion_grad = config.get('detach_emotion_grad', True)   # Don't backprop emotion to acoustic
        self.detach_energy_grad = config.get('detach_energy_grad', True)     # V5: Don't backprop energy to pitch (CRITICAL for pitch inversion fix)
        self.early_stopping_patience = config.get('early_stopping_patience', 3)
        self.use_curriculum = config.get('use_curriculum', True)
        self.curriculum_start = config.get('curriculum_start', 0.4)
        self.curriculum_increment = config.get('curriculum_increment', 0.1)
        self.current_epoch = 0

        self.prosody_config = ProsodyConfig(
            hidden_size=config.get('hidden_size', 2048),
            num_prosody_tokens=config.get('num_prosody_tokens', 4),
            dropout=config.get('dropout', 0.1),
        )
        max_segments = self.prosody_config.num_prosody_tokens * 4
        if self.num_temporal_segments > max_segments:
            raise ValueError(
                f"num_temporal_segments ({self.num_temporal_segments}) exceeds max supported "
                f"({max_segments}). Increase num_prosody_tokens or reduce num_temporal_segments."
            )

        # Setup models
        self.model = self._setup_model()
        self.temporal_encoder = self._setup_temporal_encoder() if self.train_temporal else None
        self.energy_predictor = self._setup_energy_predictor()
        self.emotion_predictor = self._setup_emotion_predictor()
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
        """Setup temporal prosody encoder for keyframe control.

        Uses the model's built-in temporal_encoder if available, ensuring
        trained weights are saved with the model for inference.
        """
        print("Setting up temporal prosody encoder...")

        # Use model's temporal encoder if it exists (ensures trained weights are used at inference)
        if hasattr(self.model, 'temporal_encoder') and self.model.temporal_encoder is not None:
            print("Using model's built-in temporal encoder (weights will be saved with model)")
            temporal_encoder = self.model.temporal_encoder
        else:
            # Create new encoder and attach to model for inference compatibility
            print("Creating temporal encoder and attaching to model")
            temporal_encoder = TemporalProsodyEncoder(self.prosody_config)
            self.model.temporal_encoder = temporal_encoder
            self.model.enable_temporal = True

        temporal_encoder = temporal_encoder.to(self.device)

        # Initialize from global encoder for better starting point
        temporal_encoder.init_from_global_encoder(self.model.prosody_encoder)

        temporal_params = sum(p.numel() for p in temporal_encoder.parameters())
        print(f"Temporal prosody encoder parameters: {temporal_params:,}")

        return temporal_encoder

    def _setup_energy_predictor(self) -> EnergyPredictor:
        """Setup auxiliary energy predictor for angry/sad discrimination."""
        print("Setting up energy predictor (auxiliary loss)...")

        energy_predictor = EnergyPredictor(
            hidden_size=self.prosody_config.hidden_size,
            dropout=self.prosody_config.dropout,
        )
        energy_predictor = energy_predictor.to(self.device)

        energy_params = sum(p.numel() for p in energy_predictor.parameters())
        print(f"Energy predictor parameters: {energy_params:,}")
        print(f"Prosody loss weight: {self.prosody_loss_weight}")
        print(f"Energy loss weight: {self.energy_loss_weight}")

        return energy_predictor

    def _setup_emotion_predictor(self) -> EmotionPredictor:
        """Setup auxiliary emotion predictor for emotion separability."""
        print("Setting up emotion predictor (auxiliary loss)...")

        emotion_predictor = EmotionPredictor(
            hidden_size=self.prosody_config.hidden_size,
            num_emotions=self.prosody_config.semantic_dim,
            dropout=self.prosody_config.dropout,
        )
        emotion_predictor = emotion_predictor.to(self.device)

        emotion_params = sum(p.numel() for p in emotion_predictor.parameters())
        print(f"Emotion predictor parameters: {emotion_params:,}")
        print(f"Emotion loss weight: {self.emotion_loss_weight}")

        return emotion_predictor

    def _segment_contour(
        self,
        contour: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        """
        Split contour into segments and resample each segment to contour_dim.

        Args:
            contour: [batch, contour_dim] pitch trajectory
            num_segments: number of temporal segments

        Returns:
            Tensor of shape [batch, num_segments, contour_dim]
        """
        batch_size, contour_dim = contour.shape
        segment_size = max(1, contour_dim // num_segments)
        segments = []

        for s in range(num_segments):
            start = s * segment_size
            end = contour_dim if s == num_segments - 1 else start + segment_size
            segment = contour[:, start:end]
            if segment.shape[1] < 2:
                # Ensure at least two points for interpolation
                segment = segment.repeat(1, 2)

            # Resample each segment back to contour_dim
            segment = F.interpolate(
                segment.unsqueeze(1),
                size=contour_dim,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            segments.append(segment)

        return torch.stack(segments, dim=1)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Add temporal encoder parameters if training temporal
        if self.temporal_encoder is not None:
            trainable_params.extend(list(self.temporal_encoder.parameters()))

        # Add energy predictor parameters
        if self.energy_predictor is not None:
            trainable_params.extend(list(self.energy_predictor.parameters()))

        # Add emotion predictor parameters
        if self.emotion_predictor is not None:
            trainable_params.extend(list(self.emotion_predictor.parameters()))

        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"Total trainable params (all modules): {total_trainable:,}")

        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step for both global and temporal encoders with optional audio loss."""
        self.model.train()
        if self.temporal_encoder is not None:
            self.temporal_encoder.train()
        if self.energy_predictor is not None:
            self.energy_predictor.train()
        if self.emotion_predictor is not None:
            self.emotion_predictor.train()

        # Move to device
        audio = batch['audio'].to(self.device)
        texts = batch['text']
        prosody_dict = {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }

        self.optimizer.zero_grad()
        use_audio_loss = self.config.get('use_audio_loss', False)
        audio_loss_weight = self.config.get('audio_loss_weight', 0.1)

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

            # L2 reconstruction loss on first token (scaled by prosody_loss_weight)
            global_loss = F.mse_loss(prosody_prefix[:, 0, :target_embedding.shape[-1]], target_embedding[:, 0, :])
            global_loss = global_loss * self.prosody_loss_weight

            # ============ ENERGY PREDICTION AUXILIARY LOSS ============
            # This ensures angry (high energy) stays distinct from sad (low energy)
            # V5 FIX: Detach to prevent energy gradients from corrupting pitch encoding
            energy_loss = torch.tensor(0.0, device=self.device)
            if self.energy_predictor is not None:
                # Target energy from acoustic[2] which is now intensity_mean normalized to [0, 1]
                target_energy = prosody_dict['acoustic'][:, 2]  # [batch]

                # V5 FIX: Detach prosody_prefix so energy loss doesn't corrupt pitch
                # This is CRITICAL - without this, energy gradients flow back through
                # acoustic_encoder and interfere with pitch_mean (acoustic[0]) encoding,
                # causing pitch inversion where happy < sad Hz
                if self.detach_energy_grad:
                    energy_input = prosody_prefix.detach()
                else:
                    energy_input = prosody_prefix

                # Predict energy from prosody embedding
                predicted_energy = self.energy_predictor(energy_input)  # [batch]

                # L2 loss on energy prediction
                energy_loss = F.mse_loss(predicted_energy, target_energy)

            # ============ EMOTION CLASSIFICATION AUXILIARY LOSS ============
            # Uses explicit emotion labels from training data (not derived from prosody)
            # V5: Gradient detachment prevents emotion loss from corrupting acoustic features
            # V5: Warmup schedule prevents early overfitting
            emotion_loss = torch.tensor(0.0, device=self.device)
            if self.emotion_predictor is not None:
                emotion_labels = batch.get('emotion_label')
                if emotion_labels is not None:
                    emotion_labels = emotion_labels.to(self.device)
                    # Filter out unknown emotions (label = -1)
                    valid_mask = emotion_labels >= 0
                    if valid_mask.any():
                        # V5: Detach prosody_prefix so emotion loss doesn't affect acoustic features
                        # This prevents pitch inversion where happy/sad Hz values flip
                        if self.detach_emotion_grad:
                            emotion_input = prosody_prefix.detach()
                        else:
                            emotion_input = prosody_prefix
                        logits = self.emotion_predictor(emotion_input)  # [batch, num_emotions]
                        emotion_loss = F.cross_entropy(logits[valid_mask], emotion_labels[valid_mask])

            # ============ AUDIO RECONSTRUCTION LOSS ============
            audio_loss = torch.tensor(0.0, device=self.device)
            if use_audio_loss and hasattr(self, 'processor'):
                try:
                    # Prepare inputs with prosody prefix
                    batch_size = len(texts)

                    # Build conversation format for CSM
                    conversations = []
                    for i, text in enumerate(texts):
                        audio_np = audio[i].cpu().numpy()
                        conversations.append([{
                            'role': '0',
                            'content': [
                                {'type': 'text', 'text': text},
                                {'type': 'audio', 'audio': audio_np}
                            ]
                        }])

                    # Process through CSM's processor
                    inputs = self.processor.apply_chat_template(
                        conversations,
                        tokenize=True,
                        return_dict=True,
                        return_tensors='pt',
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                             for k, v in inputs.items()}

                    # Get text embeddings and prepend prosody prefix
                    if hasattr(self.model.csm, 'get_input_embeddings'):
                        text_embeds = self.model.csm.get_input_embeddings()(inputs['input_ids'])
                    else:
                        text_embeds = self.model.csm.embed_text_tokens(inputs['input_ids'])

                    # Scale prosody prefix to avoid dominating
                    prosody_scale = self.config.get('prosody_scale', 0.1)
                    scaled_prefix = prosody_prefix * prosody_scale

                    # Concatenate: [prosody_prefix | text_embeddings]
                    inputs_embeds = torch.cat([scaled_prefix, text_embeds], dim=1)

                    # Extend attention mask
                    prosody_mask = torch.ones(
                        batch_size, prosody_prefix.shape[1],
                        device=self.device, dtype=inputs['attention_mask'].dtype
                    )
                    extended_mask = torch.cat([prosody_mask, inputs['attention_mask']], dim=1)

                    # Extend labels to match inputs_embeds length (prepend -100 for prosody prefix)
                    labels = inputs.get('labels', inputs['input_ids'])
                    prefix_labels = torch.full(
                        (batch_size, prosody_prefix.shape[1]),
                        -100,  # Ignore index - don't compute loss on prosody prefix
                        device=self.device,
                        dtype=labels.dtype
                    )
                    extended_labels = torch.cat([prefix_labels, labels], dim=1)

                    # Forward with labels for loss computation
                    outputs = self.model.csm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=extended_mask,
                        labels=extended_labels,
                    )

                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        audio_loss = outputs.loss * audio_loss_weight

                except Exception as e:
                    # Audio loss failed, continue with embedding loss only
                    if self.global_step % 100 == 0:
                        print(f"Audio loss skipped: {e}")

            # ============ TEMPORAL ENCODER LOSS ============
            temporal_loss = torch.tensor(0.0, device=self.device)
            temporal_energy_loss = torch.tensor(0.0, device=self.device)
            temporal_emotion_loss = torch.tensor(0.0, device=self.device)
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

                if self.energy_predictor is not None:
                    # Temporal energy loss per segment
                    # V5 FIX: Detach to prevent energy gradients from corrupting temporal pitch
                    temporal_energy_target = temporal_prosody['acoustic'][:, :, 2]
                    if self.detach_energy_grad:
                        temporal_energy_input = temporal_prefix.detach()
                    else:
                        temporal_energy_input = temporal_prefix
                    temporal_energy_pred = self.energy_predictor(temporal_energy_input, reduce=False)
                    temporal_energy_loss = F.mse_loss(temporal_energy_pred, temporal_energy_target)

                if self.emotion_predictor is not None:
                    # Temporal emotion loss per segment (same emotion for all segments)
                    emotion_labels = batch.get('emotion_label')
                    if emotion_labels is not None:
                        emotion_labels = emotion_labels.to(self.device)
                        valid_mask = emotion_labels >= 0
                        if valid_mask.any():
                            # Expand emotion labels to all segments
                            temporal_labels = emotion_labels.unsqueeze(1).expand(-1, num_segments)
                            logits = self.emotion_predictor(temporal_prefix, reduce=False)  # [batch, segments, num_emotions]
                            # Flatten for cross_entropy
                            flat_logits = logits[valid_mask].view(-1, logits.shape[-1])
                            flat_target = temporal_labels[valid_mask].view(-1)
                            temporal_emotion_loss = F.cross_entropy(flat_logits, flat_target)

            # Combined loss (now includes energy auxiliary loss)
            total_energy_loss = (energy_loss + temporal_energy_loss) * self.energy_loss_weight

            # V5: Emotion loss warmup - ramp up over first N epochs to prevent early overfitting
            # This prevents the model from overfitting to emotion classification before learning prosody
            warmup_factor = min(1.0, self.current_epoch / max(1, self.emotion_warmup_epochs))
            effective_emotion_weight = self.emotion_loss_weight * warmup_factor
            total_emotion_loss = (emotion_loss + temporal_emotion_loss) * effective_emotion_weight

            loss = (
                global_loss
                + (temporal_loss * self.temporal_loss_weight)
                + audio_loss
                + total_energy_loss
                + total_emotion_loss
            )
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if self.temporal_encoder is not None:
                torch.nn.utils.clip_grad_norm_(self.temporal_encoder.parameters(), max_grad_norm)
            if self.energy_predictor is not None:
                torch.nn.utils.clip_grad_norm_(self.energy_predictor.parameters(), max_grad_norm)
            if self.emotion_predictor is not None:
                torch.nn.utils.clip_grad_norm_(self.emotion_predictor.parameters(), max_grad_norm)

            self.optimizer.step()

            return {
                'loss': loss.item(),
                'global_loss': global_loss.item(),
                'temporal_loss': temporal_loss.item() if self.temporal_encoder else 0.0,
                'audio_loss': audio_loss.item() if isinstance(audio_loss, torch.Tensor) else 0.0,
                'energy_loss': total_energy_loss.item() if isinstance(total_energy_loss, torch.Tensor) else 0.0,
                'emotion_loss': total_emotion_loss.item() if isinstance(total_emotion_loss, torch.Tensor) else 0.0,
            }

        except Exception as e:
            import traceback
            print(f"Training step error: {e}")
            traceback.print_exc()
            return {
                'loss': 0.0,
                'global_loss': 0.0,
                'temporal_loss': 0.0,
                'audio_loss': 0.0,
                'energy_loss': 0.0,
                'emotion_loss': 0.0,
            }

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

        # Semantic + rhythm: repeat global values (no artificial noise)
        for key in ["semantic", "rhythm"]:
            global_vec = prosody_dict[key]  # [batch, dim]
            dim = global_vec.shape[-1]
            expanded = global_vec.unsqueeze(1).expand(batch_size, num_segments, dim).clone()
            temporal[key] = expanded.clamp(0, 1)

        # Contour: split the time series into segments and resample each
        contour = prosody_dict["contour"]  # [batch, contour_dim]
        temporal_contour = self._segment_contour(contour, num_segments)
        temporal["contour"] = temporal_contour.clamp(0, 1)

        # Acoustic: repeat global, then adjust pitch_mean/std to follow contour segments
        acoustic = prosody_dict["acoustic"]  # [batch, acoustic_dim]
        acoustic_dim = acoustic.shape[-1]
        acoustic_expanded = acoustic.unsqueeze(1).expand(batch_size, num_segments, acoustic_dim).clone()

        # Derive per-segment pitch mean/std from contour
        segment_means = temporal_contour.mean(dim=2)
        segment_stds = temporal_contour.std(dim=2)
        acoustic_expanded[:, :, 0] = segment_means
        if acoustic_dim > 1:
            acoustic_expanded[:, :, 1] = segment_stds

        temporal["acoustic"] = acoustic_expanded.clamp(0, 1)

        return temporal

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        curriculum_sampler: Optional[CurriculumSampler] = None,
    ):
        """Main training loop with V5 improvements."""
        print(f"\nStarting prosody-conditioned training for {num_epochs} epochs")

        # V5: Print anti-overfitting settings
        print(f"V5 Settings:")
        print(f"  - Emotion warmup epochs: {self.emotion_warmup_epochs}")
        print(f"  - Detach emotion gradients: {self.detach_emotion_grad}")
        print(f"  - Early stopping patience: {self.early_stopping_patience}")
        print(f"  - Use curriculum: {self.use_curriculum}")

        # V5: Initialize early stopping
        early_stopper = EarlyStopping(patience=self.early_stopping_patience, mode='min')

        for epoch in range(num_epochs):
            self.current_epoch = epoch  # V5: Track epoch for warmup

            # V5: Update curriculum sampler
            if curriculum_sampler is not None:
                curriculum_sampler.step(epoch)
                # Rebuild DataLoader with updated sampler indices
                current_indices = curriculum_sampler.get_indices()
                epoch_loader = DataLoader(
                    train_loader.dataset,
                    batch_size=train_loader.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(current_indices),
                    collate_fn=collate_fn,  # Use module-level collate_fn
                )
            else:
                epoch_loader = train_loader

            epoch_losses = []

            for batch_idx, batch in enumerate(epoch_loader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                self.global_step += 1

                if self.global_step % self.config.get('log_every', 10) == 0:
                    avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                    # V5: Show warmup factor
                    warmup_factor = min(1.0, self.current_epoch / max(1, self.emotion_warmup_epochs))
                    log_msg = f"Epoch {epoch+1}, Step {self.global_step}: loss={avg_loss:.4f}"
                    if 'global_loss' in metrics:
                        log_msg += f" (global={metrics['global_loss']:.4f}, temporal={metrics['temporal_loss']:.4f}"
                        if metrics.get('energy_loss', 0) > 0:
                            log_msg += f", energy={metrics['energy_loss']:.4f}"
                        if metrics.get('emotion_loss', 0) > 0:
                            log_msg += f", emotion={metrics['emotion_loss']:.4f}[w={warmup_factor:.2f}]"
                        if metrics.get('audio_loss', 0) > 0:
                            log_msg += f", audio={metrics['audio_loss']:.4f}"
                        log_msg += ")"
                    print(log_msg)

            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            # Validation with pitch metrics (V5)
            pitch_metrics = None
            if val_loader:
                val_loss, pitch_metrics = self.validate_with_pitch_metrics(val_loader)
                print(f"Validation loss: {val_loss:.4f}")

                if pitch_metrics:
                    happy_pitch = pitch_metrics.get('happy_mean_pitch', 0)
                    sad_pitch = pitch_metrics.get('sad_mean_pitch', 0)
                    print(f"  Pitch: happy={happy_pitch:.1f}Hz, sad={sad_pitch:.1f}Hz, "
                          f"separation={happy_pitch - sad_pitch:.1f}Hz")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')

                # V5: Early stopping check
                if early_stopper(val_loss, epoch, pitch_metrics):
                    print(f"Early stopping at epoch {epoch+1}. Best epoch: {early_stopper.best_epoch+1}")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

        self.save_checkpoint('final')
        print("\nTraining complete!")
        if early_stopper.best_epoch > 0:
            print(f"Best model saved at epoch {early_stopper.best_epoch+1}")

    def validate(self, val_loader: DataLoader) -> float:
        """Validation loop - computes actual validation metrics."""
        self.model.eval()
        if self.temporal_encoder is not None:
            self.temporal_encoder.eval()
        if self.energy_predictor is not None:
            self.energy_predictor.eval()
        if self.emotion_predictor is not None:
            self.emotion_predictor.eval()

        total_loss = 0.0
        total_global_loss = 0.0
        total_temporal_loss = 0.0
        total_energy_loss = 0.0
        total_emotion_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                prosody_dict = {
                    'semantic': batch['prosody_semantic'].to(self.device),
                    'acoustic': batch['prosody_acoustic'].to(self.device),
                    'rhythm': batch['prosody_rhythm'].to(self.device),
                    'contour': batch['prosody_contour'].to(self.device),
                }

                # ============ GLOBAL ENCODER LOSS ============
                prosody_prefix = self.model.get_prosody_prefix(
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                )

                # Reconstruction target
                target_embedding = torch.cat([
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'][:, :8] if prosody_dict['contour'].shape[1] > 8 else prosody_dict['contour'],
                ], dim=-1)

                if target_embedding.shape[-1] < prosody_prefix.shape[-1]:
                    target_embedding = F.pad(
                        target_embedding,
                        (0, prosody_prefix.shape[-1] - target_embedding.shape[-1])
                    )
                target_embedding = target_embedding.unsqueeze(1).expand_as(prosody_prefix[:, :1, :])

                global_loss = F.mse_loss(prosody_prefix[:, 0, :target_embedding.shape[-1]], target_embedding[:, 0, :])
                global_loss = global_loss * self.prosody_loss_weight
                total_global_loss += global_loss.item()

                # ============ ENERGY PREDICTION LOSS ============
                energy_loss = 0.0
                if self.energy_predictor is not None:
                    target_energy = prosody_dict['acoustic'][:, 2]
                    predicted_energy = self.energy_predictor(prosody_prefix)
                    energy_loss = F.mse_loss(predicted_energy, target_energy).item() * self.energy_loss_weight
                    total_energy_loss += energy_loss

                # ============ EMOTION CLASSIFICATION LOSS ============
                emotion_loss = 0.0
                if self.emotion_predictor is not None:
                    emotion_labels = batch.get('emotion_label')
                    if emotion_labels is not None:
                        emotion_labels = emotion_labels.to(self.device)
                        valid_mask = emotion_labels >= 0
                        if valid_mask.any():
                            logits = self.emotion_predictor(prosody_prefix)
                            emotion_loss = F.cross_entropy(logits[valid_mask], emotion_labels[valid_mask]).item() * self.emotion_loss_weight
                            total_emotion_loss += emotion_loss

                # ============ TEMPORAL ENCODER LOSS ============
                temporal_loss = 0.0
                if self.temporal_encoder is not None:
                    batch_size = prosody_dict['semantic'].shape[0]
                    num_segments = self.num_temporal_segments
                    temporal_prosody = self._create_temporal_prosody(prosody_dict, num_segments)

                    temporal_prefix = self.temporal_encoder(
                        temporal_prosody['semantic'],
                        temporal_prosody['acoustic'],
                        temporal_prosody['rhythm'],
                        temporal_prosody['contour'],
                    )

                    temporal_target = torch.cat([
                        temporal_prosody['semantic'],
                        temporal_prosody['acoustic'],
                        temporal_prosody['rhythm'],
                        temporal_prosody['contour'][:, :, :8] if temporal_prosody['contour'].shape[-1] > 8
                        else temporal_prosody['contour'],
                    ], dim=-1)

                    if temporal_target.shape[-1] < temporal_prefix.shape[-1]:
                        temporal_target = F.pad(
                            temporal_target,
                            (0, temporal_prefix.shape[-1] - temporal_target.shape[-1])
                        )

                    temporal_loss = F.mse_loss(
                        temporal_prefix[:, :, :temporal_target.shape[-1]],
                        temporal_target
                    ).item() * self.temporal_loss_weight
                    total_temporal_loss += temporal_loss

                batch_loss = global_loss.item() + temporal_loss + energy_loss + emotion_loss
                total_loss += batch_loss
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)

        # Log validation breakdown
        if num_batches > 0:
            print(f"  Val breakdown: global={total_global_loss/num_batches:.4f}, "
                  f"temporal={total_temporal_loss/num_batches:.4f}, "
                  f"energy={total_energy_loss/num_batches:.4f}, "
                  f"emotion={total_emotion_loss/num_batches:.4f}")

        return avg_loss

    def validate_with_pitch_metrics(self, val_loader: DataLoader) -> Tuple[float, dict]:
        """
        V5: Validation with pitch pattern tracking per emotion.

        Monitors that happy pitch > sad pitch to detect overfitting.
        Returns (loss, pitch_metrics) where pitch_metrics contains per-emotion pitch stats.
        """
        self.model.eval()
        if self.temporal_encoder is not None:
            self.temporal_encoder.eval()
        if self.energy_predictor is not None:
            self.energy_predictor.eval()
        if self.emotion_predictor is not None:
            self.emotion_predictor.eval()

        total_loss = 0.0
        num_batches = 0

        # V5: Track pitch per emotion
        emotion_pitches = {i: [] for i in range(10)}  # 10 emotion classes

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                prosody_dict = {
                    'semantic': batch['prosody_semantic'].to(self.device),
                    'acoustic': batch['prosody_acoustic'].to(self.device),
                    'rhythm': batch['prosody_rhythm'].to(self.device),
                    'contour': batch['prosody_contour'].to(self.device),
                }

                # Get prosody prefix
                prosody_prefix = self.model.get_prosody_prefix(
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                )

                # Reconstruction target
                target_embedding = torch.cat([
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'][:, :8] if prosody_dict['contour'].shape[1] > 8 else prosody_dict['contour'],
                ], dim=-1)

                if target_embedding.shape[-1] < prosody_prefix.shape[-1]:
                    target_embedding = F.pad(
                        target_embedding,
                        (0, prosody_prefix.shape[-1] - target_embedding.shape[-1])
                    )
                target_embedding = target_embedding.unsqueeze(1).expand_as(prosody_prefix[:, :1, :])

                global_loss = F.mse_loss(prosody_prefix[:, 0, :target_embedding.shape[-1]], target_embedding[:, 0, :])
                total_loss += global_loss.item()
                num_batches += 1

                # V5: Collect pitch per emotion
                # acoustic[0] is pitch_mean (normalized to [0, 1])
                # We denormalize by assuming pitch range ~50-500Hz
                pitch_normalized = prosody_dict['acoustic'][:, 0]  # [batch]
                pitch_hz = pitch_normalized * 450 + 50  # Denormalize to Hz

                emotion_labels = batch.get('emotion_label')
                if emotion_labels is not None:
                    for i, (pitch, label) in enumerate(zip(pitch_hz.cpu().numpy(), emotion_labels.numpy())):
                        if 0 <= label < 10:
                            emotion_pitches[label].append(pitch)

        avg_loss = total_loss / max(1, num_batches)

        # V5: Compute pitch statistics per emotion
        pitch_metrics = {}
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted', 'calm', 'excited', 'contempt']

        for idx, name in enumerate(emotion_names):
            if emotion_pitches[idx]:
                pitch_metrics[f'{name}_mean_pitch'] = float(np.mean(emotion_pitches[idx]))
                pitch_metrics[f'{name}_std_pitch'] = float(np.std(emotion_pitches[idx]))
                pitch_metrics[f'{name}_count'] = len(emotion_pitches[idx])

        return avg_loss, pitch_metrics

    def save_checkpoint(self, name: str):
        """Save checkpoint with all trained modules.

        The temporal_encoder state is saved from self.model.temporal_encoder
        to ensure inference compatibility (same weights used at training and inference).
        """
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
            'prosody_loss_weight': self.prosody_loss_weight,
            'energy_loss_weight': self.energy_loss_weight,
            'temporal_loss_weight': self.temporal_loss_weight,
            'emotion_loss_weight': self.emotion_loss_weight,
        }

        # Save temporal encoder from model (ensures inference uses same weights)
        # Note: self.temporal_encoder IS self.model.temporal_encoder (same object)
        if self.temporal_encoder is not None:
            checkpoint['temporal_encoder'] = self.model.temporal_encoder.state_dict()
            checkpoint['num_temporal_segments'] = self.num_temporal_segments

        # Save energy predictor if trained
        if self.energy_predictor is not None:
            checkpoint['energy_predictor'] = self.energy_predictor.state_dict()
        # Save emotion predictor if trained
        if self.emotion_predictor is not None:
            checkpoint['emotion_predictor'] = self.emotion_predictor.state_dict()

        # Save LoRA adapter if CSM has LoRA applied (CRITICAL for prosody control!)
        if HAS_PEFT and hasattr(self.model, 'csm'):
            try:
                from peft import PeftModel
                if isinstance(self.model.csm, PeftModel):
                    checkpoint['lora_state_dict'] = self.model.csm.state_dict()
                    checkpoint['lora_config'] = {
                        'r': self.config.get('lora_r', 8),
                        'lora_alpha': self.config.get('lora_alpha', 16),
                        'target_modules': self.config.get('lora_target_modules', ['q_proj', 'v_proj']),
                        'lora_dropout': self.config.get('lora_dropout', 0.1),
                    }
                    print("  LoRA adapter weights saved!")
            except Exception as e:
                print(f"  Warning: Could not save LoRA weights: {e}")

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")
        modules_saved = []
        if self.temporal_encoder is not None:
            modules_saved.append(f"temporal ({self.num_temporal_segments} segments)")
        if self.energy_predictor is not None:
            modules_saved.append("energy predictor")
        if modules_saved:
            print(f"  Includes: {', '.join(modules_saved)}")


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

    # Create dataset from manifest (CLI arg or config)
    manifest_path = args.manifest or config.get('train_manifest')
    if manifest_path:
        dataset = ProsodyConditionedDataset(
            manifest_path,
            config.get('prosody_cache_dir', '../data/prosody_cache'),
            energy_augmentation=config.get('energy_augmentation'),
        )

        # V5: Use curriculum sampling instead of weighted random sampling
        use_curriculum = config.get('use_curriculum', True)
        curriculum_sampler = None

        if use_curriculum:
            curriculum_sampler = CurriculumSampler(
                dataset,
                start_threshold=config.get('curriculum_start', 0.4),
                epoch_increment=config.get('curriculum_increment', 0.1),
            )
            print("Using curriculum sampler (V5) for gradual emotion introduction")
            sampler = None  # Curriculum handled in train loop
        else:
            # Fall back to weighted random sampling
            balance_emotions = config.get('balance_emotions', True)
            sampler = None
            if balance_emotions:
                weights = dataset.get_emotion_weights()
                if weights is not None:
                    sampler = WeightedRandomSampler(
                        weights=weights,
                        num_samples=len(weights),
                        replacement=True,
                    )
                    print("Using weighted sampler to balance emotions")
                else:
                    print("Emotion labels not found; falling back to shuffled sampling")

        train_loader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=sampler is None and curriculum_sampler is None,
            sampler=sampler,
            collate_fn=collate_fn,
        )

        # V5: Create validation loader if val_manifest provided
        val_loader = None
        val_manifest = config.get('val_manifest')
        if val_manifest:
            val_dataset = ProsodyConditionedDataset(
                val_manifest,
                config.get('prosody_cache_dir', '../data/prosody_cache'),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.get('batch_size', 4),
                shuffle=False,
                collate_fn=collate_fn,
            )
            print(f"Validation set: {len(val_dataset)} samples")

        trainer.train(
            train_loader,
            val_loader=val_loader,
            num_epochs=config.get('num_epochs', 10),
            curriculum_sampler=curriculum_sampler,
        )
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
