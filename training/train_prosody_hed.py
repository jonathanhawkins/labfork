"""
Prosody V6 Training with Hierarchical Emotion Distribution (HED)

This extends the V5 prosody-conditioned training with HED from ICASSP 2024:
"Hierarchical Emotion Prediction and Control in TTS"

Key additions:
1. Phoneme-level emotion features via OpenSMILE
2. Word-level emotion aggregation
3. Utterance-level global emotion
4. HED variance adaptor integrated with prosody encoder

Usage:
    python train_prosody_hed.py --config config/prosody_hed_v6.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import yaml

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'backend'))

# Import base modules
from prosody_conditioning import (
    ProsodyConfig,
    ProsodyEncoder,
    TemporalProsodyEncoder,
    ProsodyControlledCSM,
    extract_prosody_for_conditioning,
)
from keyframe_prosody import get_temporal_prosody_tokens

# Import HED modules
from hierarchical_emotion import (
    HEDConfig,
    HEDPipeline,
    HierarchicalEmotionEncoder,
    HierarchicalEmotionLoss,
    HEDVarianceAdaptor,
    PhonemeAligner,
    OpenSMILEExtractor,
)

try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: PEFT not available. LoRA disabled.")


# =============================================================================
# HED-ENABLED DATASET
# =============================================================================

class HEDProsodyDataset(Dataset):
    """
    Dataset that provides (audio, text, prosody, HED_features) for V6 training.

    Extends the base prosody dataset with hierarchical emotion features.
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        hed_config: HEDConfig,
        max_audio_length_ms: int = 30000,
        sample_rate: int = 24000,
        energy_augmentation: Optional[dict] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.prosody_cache_dir = Path(prosody_cache_dir)
        self.hed_cache_dir = self.prosody_cache_dir / 'hed_features'
        self.max_audio_length = int(max_audio_length_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        self.hed_config = hed_config

        # Energy augmentation config
        self.energy_aug = energy_augmentation or {}
        self.energy_aug_enabled = self.energy_aug.get('enabled', False)
        self.angry_boost_db = self.energy_aug.get('angry_boost_db', 6.0)
        self.sad_reduce_db = self.energy_aug.get('sad_reduce_db', -6.0)
        self.energy_aug_prob = self.energy_aug.get('probability', 0.5)

        # Load manifest
        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples for HED training")

        # Create cache directories
        self.prosody_cache_dir.mkdir(parents=True, exist_ok=True)
        self.hed_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HED components (shared across dataset)
        self.aligner = PhonemeAligner(hed_config)
        self.feature_extractor = OpenSMILEExtractor(hed_config)

        # Prosody analyzer (lazy loaded)
        self._analyzer = None

    def __len__(self):
        return len(self.samples)

    # Emotion label mapping
    EMOTION_TO_IDX = {
        'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
        'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
        'excited': 8, 'contempt': 9,
    }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get audio path
        audio_path = sample.get('audio_path') or sample.get('path') or sample.get('audio')
        if not audio_path:
            raise ValueError(f"Sample {idx} has no audio path")

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]

        # Get text and word timestamps
        text = sample.get('text') or sample.get('transcript', '')
        word_timestamps = sample.get('word_timestamps', None)

        # Get prosody
        prosody = self._get_prosody(sample, idx, audio_path)

        # Get HED features
        hed_features = self._get_hed_features(audio_path, text, word_timestamps, idx)

        # Get emotion label
        emotion_label = self._get_emotion_label(sample)

        # Apply energy augmentation
        if self.energy_aug_enabled:
            import random
            if random.random() < self.energy_aug_prob:
                waveform, prosody = self._apply_energy_augmentation(waveform, prosody, sample)

        return {
            'audio': waveform,
            'text': text,
            'prosody_semantic': prosody['semantic'].squeeze(0),
            'prosody_acoustic': prosody['acoustic'].squeeze(0),
            'prosody_rhythm': prosody['rhythm'].squeeze(0),
            'prosody_contour': prosody['contour'].squeeze(0),
            'emotion_label': emotion_label,
            # HED features
            'hed_phoneme_features': hed_features['phoneme_features'],
            'hed_phoneme_to_word': hed_features['phoneme_to_word'],
            'hed_num_words': hed_features['num_words'],
            'hed_num_phonemes': hed_features['num_phonemes'],
        }

    def _get_emotion_label(self, sample: dict) -> int:
        """Extract emotion label from sample."""
        emotion = sample.get('emotion', '').lower()
        if not emotion:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion', '').lower()
        if not emotion:
            emotions = sample.get('prosody', {}).get('semantic', {}).get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()
        return self.EMOTION_TO_IDX.get(emotion, -1)

    def _get_prosody(self, sample: dict, idx: int, audio_path: str) -> Dict[str, torch.Tensor]:
        """Get prosody from cache or extract."""
        import hashlib
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        cache_path = self.prosody_cache_dir / f"prosody_{path_hash}.pt"

        if cache_path.exists():
            return torch.load(cache_path)

        if 'prosody' in sample:
            prosody = extract_prosody_for_conditioning(sample['prosody'])
            torch.save(prosody, cache_path)
            return prosody

        # Extract prosody
        try:
            from prosody_analyzer import CompleteProsodyAnalyzer
            if self._analyzer is None:
                self._analyzer = CompleteProsodyAnalyzer(use_qwen=True, device="cuda")
            prosody_result = self._analyzer.analyze(audio_path)
            prosody = extract_prosody_for_conditioning(prosody_result.to_dict())
            torch.save(prosody, cache_path)
            return prosody
        except Exception as e:
            print(f"Prosody extraction failed for {audio_path}: {e}")
            config = ProsodyConfig()
            return {
                'semantic': torch.zeros(1, config.semantic_dim),
                'acoustic': torch.zeros(1, config.acoustic_dim),
                'rhythm': torch.zeros(1, config.rhythm_dim),
                'contour': torch.zeros(1, config.contour_dim),
            }

    def _get_hed_features(
        self,
        audio_path: str,
        text: str,
        word_timestamps: Optional[List[Dict]],
        idx: int,
    ) -> Dict:
        """Get HED features from cache or extract."""
        import hashlib
        cache_key = hashlib.md5(f"{audio_path}_{text}".encode()).hexdigest()[:16]
        cache_path = self.hed_cache_dir / f"hed_{cache_key}.pt"

        if cache_path.exists():
            return torch.load(cache_path)

        # Extract HED features
        try:
            # Align phonemes and words
            alignment = self.aligner.align(audio_path, text, word_timestamps)

            # Extract OpenSMILE features per phoneme
            phoneme_features = []
            for phoneme in alignment['phonemes']:
                features = self.feature_extractor.extract_segment(
                    audio_path,
                    phoneme['start_time'],
                    phoneme['end_time'],
                    self.sample_rate,
                )
                phoneme_features.append(features)

            if not phoneme_features:
                phoneme_features = [np.zeros(self.hed_config.opensmile_dim)]

            phoneme_features = torch.tensor(np.stack(phoneme_features), dtype=torch.float32)
            phoneme_to_word = [p['word_idx'] for p in alignment['phonemes']]

            hed_features = {
                'phoneme_features': phoneme_features,
                'phoneme_to_word': phoneme_to_word,
                'num_words': len(alignment['words']),
                'num_phonemes': len(alignment['phonemes']),
            }

            torch.save(hed_features, cache_path)
            return hed_features

        except Exception as e:
            print(f"HED feature extraction failed for {audio_path}: {e}")
            return {
                'phoneme_features': torch.zeros(1, self.hed_config.opensmile_dim),
                'phoneme_to_word': [0],
                'num_words': 1,
                'num_phonemes': 1,
            }

    def _apply_energy_augmentation(
        self,
        waveform: torch.Tensor,
        prosody: Dict[str, torch.Tensor],
        sample: dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply energy augmentation based on emotion."""
        semantic = sample.get('prosody', {}).get('semantic', {})
        emotion = semantic.get('emotion', '').lower()

        if not emotion:
            emotions = semantic.get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()

        if not emotion:
            return waveform, prosody

        gain_db = 0.0
        if emotion in ['angry', 'excited', 'surprised']:
            gain_db = self.angry_boost_db
        elif emotion in ['sad', 'calm', 'fearful']:
            gain_db = self.sad_reduce_db

        if gain_db == 0.0:
            return waveform, prosody

        gain_linear = 10 ** (gain_db / 20.0)
        waveform = torch.clamp(waveform * gain_linear, -1.0, 1.0)

        prosody = {k: v.clone() for k, v in prosody.items()}
        if prosody['acoustic'].shape[-1] > 2:
            intensity_delta = gain_db / 40.0
            prosody['acoustic'][..., 2] = torch.clamp(
                prosody['acoustic'][..., 2] + intensity_delta, 0.0, 1.0
            )

        return waveform, prosody

    def get_emotion_weights(self) -> Optional[List[float]]:
        """Compute per-sample weights to balance emotions."""
        emotions = []
        for sample in self.samples:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion')
            if not emotion and isinstance(semantic.get('emotions'), dict):
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

        max_count = max(counts.values())
        weights = []
        for emotion in emotions:
            if not emotion:
                weights.append(1.0)
            else:
                weights.append(max_count / counts[emotion])

        return weights


def hed_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with HED features (variable length phonemes)."""
    # Pad audio
    audios = [item['audio'] for item in batch]
    max_audio_len = max(a.shape[0] for a in audios)
    audios_padded = torch.stack([
        F.pad(a, (0, max_audio_len - a.shape[0])) for a in audios
    ])

    # Pad HED phoneme features
    max_phonemes = max(item['hed_num_phonemes'] for item in batch)
    phoneme_dim = batch[0]['hed_phoneme_features'].shape[-1]

    phoneme_features_padded = []
    phoneme_masks = []
    phoneme_to_word_padded = []

    for item in batch:
        pf = item['hed_phoneme_features']
        num_p = item['hed_num_phonemes']

        # Pad phoneme features
        if num_p < max_phonemes:
            pf = F.pad(pf, (0, 0, 0, max_phonemes - num_p))
        phoneme_features_padded.append(pf)

        # Create mask
        mask = torch.zeros(max_phonemes, dtype=torch.bool)
        mask[:num_p] = True
        phoneme_masks.append(mask)

        # Pad phoneme_to_word mapping
        p2w = item['hed_phoneme_to_word']
        if len(p2w) < max_phonemes:
            p2w = p2w + [-1] * (max_phonemes - len(p2w))
        phoneme_to_word_padded.append(p2w)

    return {
        'audio': audios_padded,
        'text': [item['text'] for item in batch],
        'prosody_semantic': torch.stack([item['prosody_semantic'] for item in batch]),
        'prosody_acoustic': torch.stack([item['prosody_acoustic'] for item in batch]),
        'prosody_rhythm': torch.stack([item['prosody_rhythm'] for item in batch]),
        'prosody_contour': torch.stack([item['prosody_contour'] for item in batch]),
        'emotion_label': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long),
        # HED
        'hed_phoneme_features': torch.stack(phoneme_features_padded),
        'hed_phoneme_mask': torch.stack(phoneme_masks),
        'hed_phoneme_to_word': phoneme_to_word_padded,
        'hed_num_words': [item['hed_num_words'] for item in batch],
    }


# =============================================================================
# HED TRAINER
# =============================================================================

class HEDProsodyTrainer:
    """
    Trainer for V6 prosody conditioning with HED.

    Combines:
    - ProsodyEncoder (V5): Global prosody prefix tokens
    - TemporalProsodyEncoder (V5): Per-segment prosody tokens
    - HierarchicalEmotionEncoder (V6): Multi-level emotion features
    - HEDVarianceAdaptor (V6): HED -> prosody integration
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = self._setup_device()

        # Build HED config from training config
        hed_dict = config.get('hed', {})
        self.hed_config = HEDConfig(
            opensmile_dim=hed_dict.get('opensmile_dim', 88),
            phoneme_hidden=hed_dict.get('phoneme_hidden', 128),
            word_hidden=hed_dict.get('word_hidden', 256),
            utterance_hidden=hed_dict.get('utterance_hidden', 512),
            output_hidden=hed_dict.get('output_hidden', 2048),
            phoneme_pooling=hed_dict.get('phoneme_pooling', 'attention'),
            word_pooling=hed_dict.get('word_pooling', 'attention'),
            alignment_method=hed_dict.get('alignment_method', 'g2p'),
            dropout=config.get('dropout', 0.1),
        )

        # Prosody config
        self.prosody_config = ProsodyConfig(
            hidden_size=config.get('hidden_size', 2048),
            num_prosody_tokens=config.get('num_prosody_tokens', 4),
            dropout=config.get('dropout', 0.1),
        )

        # V6 settings
        self.use_hed = config.get('use_hed', True)
        self.hed_integration = config.get('hed_integration', 'additive')
        self.hed_scale = config.get('hed_scale', 0.5)

        # V5 settings (retained)
        self.train_temporal = config.get('train_temporal', True)
        self.num_temporal_segments = config.get('num_temporal_segments', 4)
        self.prosody_loss_weight = config.get('prosody_loss_weight', 2.0)
        self.energy_loss_weight = config.get('energy_loss_weight', 1.0)
        self.temporal_loss_weight = config.get('temporal_loss_weight', 1.0)
        self.emotion_loss_weight = config.get('emotion_loss_weight', 1.0)
        self.emotion_warmup_epochs = config.get('emotion_warmup_epochs', 10)
        self.detach_emotion_grad = config.get('detach_emotion_grad', True)
        self.detach_energy_grad = config.get('detach_energy_grad', True)
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        self.current_epoch = 0

        # HED loss settings
        hed_loss_config = config.get('hed_loss', {})
        self.hed_total_weight = config.get('hed_total_weight', 0.5)

        # Setup models
        self.model = self._setup_prosody_model()
        self.temporal_encoder = self._setup_temporal_encoder() if self.train_temporal else None
        self.hed_encoder = self._setup_hed_encoder() if self.use_hed else None
        self.hed_variance_adaptor = self._setup_hed_adaptor() if self.use_hed else None
        self.hed_loss_fn = HierarchicalEmotionLoss(
            self.hed_config,
            **hed_loss_config
        ) if self.use_hed else None

        # Auxiliary heads (from V5)
        self.energy_predictor = self._setup_energy_predictor()
        self.emotion_predictor = self._setup_emotion_predictor()

        self.optimizer = self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _setup_prosody_model(self) -> ProsodyControlledCSM:
        """Setup CSM with prosody conditioning."""
        from transformers import CsmForConditionalGeneration, AutoProcessor

        model_path = self.config['model_path']
        print(f"Loading CSM from: {model_path}")

        csm = CsmForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

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

        model = ProsodyControlledCSM(
            csm,
            self.prosody_config,
            freeze_csm=not self.config.get('train_csm', False),
        )
        model = model.to(self.device)

        prosody_params = sum(p.numel() for p in model.prosody_encoder.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Prosody encoder: {prosody_params:,} params")
        print(f"Trainable (model): {trainable:,} params")

        return model

    def _setup_temporal_encoder(self) -> TemporalProsodyEncoder:
        """Setup temporal prosody encoder."""
        if hasattr(self.model, 'temporal_encoder') and self.model.temporal_encoder is not None:
            temporal_encoder = self.model.temporal_encoder
        else:
            temporal_encoder = TemporalProsodyEncoder(self.prosody_config)
            self.model.temporal_encoder = temporal_encoder
            self.model.enable_temporal = True

        temporal_encoder = temporal_encoder.to(self.device)
        temporal_encoder.init_from_global_encoder(self.model.prosody_encoder)

        print(f"Temporal encoder: {sum(p.numel() for p in temporal_encoder.parameters()):,} params")
        return temporal_encoder

    def _setup_hed_encoder(self) -> HierarchicalEmotionEncoder:
        """Setup HED encoder."""
        print("Setting up HED encoder (V6)...")
        hed_encoder = HierarchicalEmotionEncoder(self.hed_config).to(self.device)
        print(f"HED encoder: {sum(p.numel() for p in hed_encoder.parameters()):,} params")
        return hed_encoder

    def _setup_hed_adaptor(self) -> HEDVarianceAdaptor:
        """Setup HED variance adaptor."""
        print("Setting up HED variance adaptor (V6)...")
        adaptor = HEDVarianceAdaptor(
            self.hed_config,
            prosody_hidden=self.prosody_config.hidden_size,
            num_tokens=self.prosody_config.num_prosody_tokens,
        ).to(self.device)
        print(f"HED adaptor: {sum(p.numel() for p in adaptor.parameters()):,} params")
        return adaptor

    def _setup_energy_predictor(self) -> nn.Module:
        """Setup energy predictor (V5)."""
        from train_prosody_conditioned import EnergyPredictor
        predictor = EnergyPredictor(
            hidden_size=self.prosody_config.hidden_size,
            dropout=self.prosody_config.dropout,
        ).to(self.device)
        return predictor

    def _setup_emotion_predictor(self) -> nn.Module:
        """Setup emotion predictor (V5)."""
        from train_prosody_conditioned import EmotionPredictor
        predictor = EmotionPredictor(
            hidden_size=self.prosody_config.hidden_size,
            num_emotions=self.prosody_config.semantic_dim,
            dropout=self.prosody_config.dropout,
        ).to(self.device)
        return predictor

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for all trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if self.temporal_encoder is not None:
            trainable_params.extend(list(self.temporal_encoder.parameters()))

        if self.hed_encoder is not None:
            trainable_params.extend(list(self.hed_encoder.parameters()))

        if self.hed_variance_adaptor is not None:
            trainable_params.extend(list(self.hed_variance_adaptor.parameters()))

        trainable_params.extend(list(self.energy_predictor.parameters()))
        trainable_params.extend(list(self.emotion_predictor.parameters()))

        total = sum(p.numel() for p in trainable_params)
        print(f"Total trainable: {total:,} params")

        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.get('learning_rate', 5e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with HED."""
        self.model.train()
        if self.temporal_encoder:
            self.temporal_encoder.train()
        if self.hed_encoder:
            self.hed_encoder.train()
        if self.hed_variance_adaptor:
            self.hed_variance_adaptor.train()
        self.energy_predictor.train()
        self.emotion_predictor.train()

        # Move to device
        audio = batch['audio'].to(self.device)
        prosody_dict = {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }
        emotion_labels = batch['emotion_label'].to(self.device)

        self.optimizer.zero_grad()

        try:
            # ============ PROSODY ENCODER (V5) ============
            prosody_prefix = self.model.get_prosody_prefix(
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'],
            )  # [batch, num_tokens, hidden]

            # Prosody reconstruction target
            target_embedding = torch.cat([
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'][:, :8],
            ], dim=-1)

            if target_embedding.shape[-1] < prosody_prefix.shape[-1]:
                target_embedding = F.pad(target_embedding, (0, prosody_prefix.shape[-1] - target_embedding.shape[-1]))

            target_embedding = target_embedding.unsqueeze(1).expand_as(prosody_prefix[:, :1, :])
            global_loss = F.mse_loss(prosody_prefix[:, 0, :target_embedding.shape[-1]], target_embedding[:, 0, :])
            global_loss = global_loss * self.prosody_loss_weight

            # ============ HED ENCODER (V6) ============
            hed_loss_total = torch.tensor(0.0, device=self.device)
            if self.use_hed and self.hed_encoder is not None:
                # Get HED features from batch
                hed_phoneme_features = batch['hed_phoneme_features'].to(self.device)
                hed_phoneme_to_word = batch['hed_phoneme_to_word']
                hed_num_words = max(batch['hed_num_words'])
                hed_phoneme_mask = batch['hed_phoneme_mask'].to(self.device)

                # Forward through HED encoder
                hed_output = self.hed_encoder(
                    hed_phoneme_features,
                    hed_phoneme_to_word,
                    hed_num_words,
                    phoneme_mask=hed_phoneme_mask,
                )

                # Compute HED losses
                hed_losses = self.hed_loss_fn(hed_output, emotion_labels)
                hed_loss_total = hed_losses['total'] * self.hed_total_weight

                # Generate HED variance adaptor tokens
                hed_tokens = self.hed_variance_adaptor(hed_output['combined_embedding'])

                # Integrate HED with prosody prefix
                if self.hed_integration == 'additive':
                    prosody_prefix = prosody_prefix + hed_tokens * self.hed_scale
                elif self.hed_integration == 'concat':
                    prosody_prefix = torch.cat([prosody_prefix, hed_tokens], dim=1)

            # ============ ENERGY PREDICTION (V5) ============
            energy_loss = torch.tensor(0.0, device=self.device)
            target_energy = prosody_dict['acoustic'][:, 2]
            energy_input = prosody_prefix.detach() if self.detach_energy_grad else prosody_prefix
            predicted_energy = self.energy_predictor(energy_input)
            energy_loss = F.mse_loss(predicted_energy, target_energy) * self.energy_loss_weight

            # ============ EMOTION CLASSIFICATION (V5) ============
            emotion_loss = torch.tensor(0.0, device=self.device)
            valid_mask = emotion_labels >= 0
            if valid_mask.any():
                emotion_input = prosody_prefix.detach() if self.detach_emotion_grad else prosody_prefix
                logits = self.emotion_predictor(emotion_input)
                emotion_loss = F.cross_entropy(logits[valid_mask], emotion_labels[valid_mask])

            # Emotion warmup
            warmup_factor = min(1.0, self.current_epoch / max(1, self.emotion_warmup_epochs))
            emotion_loss = emotion_loss * self.emotion_loss_weight * warmup_factor

            # ============ TEMPORAL ENCODER (V5) ============
            temporal_loss = torch.tensor(0.0, device=self.device)
            if self.temporal_encoder is not None:
                temporal_prosody = self._create_temporal_prosody(prosody_dict, self.num_temporal_segments)
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
                    temporal_prosody['contour'][:, :, :8],
                ], dim=-1)

                if temporal_target.shape[-1] < temporal_prefix.shape[-1]:
                    temporal_target = F.pad(temporal_target, (0, temporal_prefix.shape[-1] - temporal_target.shape[-1]))

                temporal_loss = F.mse_loss(
                    temporal_prefix[:, :, :temporal_target.shape[-1]],
                    temporal_target
                ) * self.temporal_loss_weight

            # ============ TOTAL LOSS ============
            loss = global_loss + temporal_loss + energy_loss + emotion_loss + hed_loss_total
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
            if self.temporal_encoder:
                torch.nn.utils.clip_grad_norm_(self.temporal_encoder.parameters(), self.config.get('max_grad_norm', 1.0))
            if self.hed_encoder:
                torch.nn.utils.clip_grad_norm_(self.hed_encoder.parameters(), self.config.get('max_grad_norm', 1.0))
            if self.hed_variance_adaptor:
                torch.nn.utils.clip_grad_norm_(self.hed_variance_adaptor.parameters(), self.config.get('max_grad_norm', 1.0))

            self.optimizer.step()

            return {
                'loss': loss.item(),
                'global_loss': global_loss.item(),
                'temporal_loss': temporal_loss.item() if self.temporal_encoder else 0.0,
                'energy_loss': energy_loss.item(),
                'emotion_loss': emotion_loss.item(),
                'hed_loss': hed_loss_total.item() if self.use_hed else 0.0,
            }

        except Exception as e:
            import traceback
            print(f"Training step error: {e}")
            traceback.print_exc()
            return {k: 0.0 for k in ['loss', 'global_loss', 'temporal_loss', 'energy_loss', 'emotion_loss', 'hed_loss']}

    def _create_temporal_prosody(self, prosody_dict: Dict, num_segments: int) -> Dict[str, torch.Tensor]:
        """Create temporal prosody from global prosody."""
        batch_size = prosody_dict['semantic'].shape[0]
        temporal = {}

        for key in ["semantic", "rhythm"]:
            global_vec = prosody_dict[key]
            dim = global_vec.shape[-1]
            temporal[key] = global_vec.unsqueeze(1).expand(batch_size, num_segments, dim).clone().clamp(0, 1)

        contour = prosody_dict["contour"]
        contour_dim = contour.shape[-1]
        segment_size = max(1, contour_dim // num_segments)
        segments = []
        for s in range(num_segments):
            start = s * segment_size
            end = contour_dim if s == num_segments - 1 else start + segment_size
            segment = contour[:, start:end]
            if segment.shape[1] < 2:
                segment = segment.repeat(1, 2)
            segment = F.interpolate(segment.unsqueeze(1), size=contour_dim, mode="linear", align_corners=False).squeeze(1)
            segments.append(segment)
        temporal["contour"] = torch.stack(segments, dim=1).clamp(0, 1)

        acoustic = prosody_dict["acoustic"]
        acoustic_dim = acoustic.shape[-1]
        acoustic_expanded = acoustic.unsqueeze(1).expand(batch_size, num_segments, acoustic_dim).clone()
        segment_means = temporal["contour"].mean(dim=2)
        segment_stds = temporal["contour"].std(dim=2)
        acoustic_expanded[:, :, 0] = segment_means
        if acoustic_dim > 1:
            acoustic_expanded[:, :, 1] = segment_stds
        temporal["acoustic"] = acoustic_expanded.clamp(0, 1)

        return temporal

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
    ):
        """Main training loop."""
        print(f"\nStarting V6 HED training for {num_epochs} epochs")
        print(f"HED enabled: {self.use_hed}")
        print(f"HED integration: {self.hed_integration}")
        print(f"HED scale: {self.hed_scale}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                self.global_step += 1

                if self.global_step % self.config.get('log_every', 10) == 0:
                    avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                    log_msg = f"Epoch {epoch+1}, Step {self.global_step}: loss={avg_loss:.4f}"
                    log_msg += f" (global={metrics['global_loss']:.4f}"
                    if metrics['temporal_loss'] > 0:
                        log_msg += f", temporal={metrics['temporal_loss']:.4f}"
                    if metrics['hed_loss'] > 0:
                        log_msg += f", hed={metrics['hed_loss']:.4f}"
                    log_msg += f", energy={metrics['energy_loss']:.4f}, emotion={metrics['emotion_loss']:.4f})"
                    print(log_msg)

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}")

            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Validation loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')

            if (epoch + 1) % self.config.get('save_every_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

        self.save_checkpoint('final')
        print("\nV6 HED training complete!")

    def validate(self, val_loader: DataLoader) -> float:
        """Validation loop."""
        self.model.eval()
        if self.temporal_encoder:
            self.temporal_encoder.eval()
        if self.hed_encoder:
            self.hed_encoder.eval()
        if self.hed_variance_adaptor:
            self.hed_variance_adaptor.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                prosody_dict = {
                    'semantic': batch['prosody_semantic'].to(self.device),
                    'acoustic': batch['prosody_acoustic'].to(self.device),
                    'rhythm': batch['prosody_rhythm'].to(self.device),
                    'contour': batch['prosody_contour'].to(self.device),
                }

                prosody_prefix = self.model.get_prosody_prefix(
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                )

                target_embedding = torch.cat([
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'][:, :8],
                ], dim=-1)

                if target_embedding.shape[-1] < prosody_prefix.shape[-1]:
                    target_embedding = F.pad(target_embedding, (0, prosody_prefix.shape[-1] - target_embedding.shape[-1]))

                target_embedding = target_embedding.unsqueeze(1).expand_as(prosody_prefix[:, :1, :])
                loss = F.mse_loss(prosody_prefix[:, 0, :target_embedding.shape[-1]], target_embedding[:, 0, :])

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, name: str):
        """Save checkpoint with all modules."""
        output_dir = Path(self.config.get('output_dir', 'checkpoints/prosody_v6'))
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

        if self.temporal_encoder:
            checkpoint['temporal_encoder'] = self.model.temporal_encoder.state_dict()

        if self.hed_encoder:
            checkpoint['hed_encoder'] = self.hed_encoder.state_dict()
            checkpoint['hed_config'] = {
                'opensmile_dim': self.hed_config.opensmile_dim,
                'phoneme_hidden': self.hed_config.phoneme_hidden,
                'word_hidden': self.hed_config.word_hidden,
                'utterance_hidden': self.hed_config.utterance_hidden,
                'output_hidden': self.hed_config.output_hidden,
            }

        if self.hed_variance_adaptor:
            checkpoint['hed_variance_adaptor'] = self.hed_variance_adaptor.state_dict()

        checkpoint['energy_predictor'] = self.energy_predictor.state_dict()
        checkpoint['emotion_predictor'] = self.emotion_predictor.state_dict()

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/prosody_hed_v6.yaml')
    parser.add_argument('--manifest', type=str, help='Override train manifest')
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}")
        return

    print("=" * 60)
    print("Prosody V6 Training with Hierarchical Emotion Distribution")
    print("=" * 60)

    trainer = HEDProsodyTrainer(config)

    manifest_path = args.manifest or config.get('train_manifest')
    if not manifest_path:
        print("No manifest provided. Use --manifest or set train_manifest in config.")
        return

    # Build HED config
    hed_dict = config.get('hed', {})
    hed_config = HEDConfig(
        opensmile_dim=hed_dict.get('opensmile_dim', 88),
        phoneme_hidden=hed_dict.get('phoneme_hidden', 128),
        word_hidden=hed_dict.get('word_hidden', 256),
        utterance_hidden=hed_dict.get('utterance_hidden', 512),
        output_hidden=hed_dict.get('output_hidden', 2048),
        alignment_method=hed_dict.get('alignment_method', 'g2p'),
    )

    dataset = HEDProsodyDataset(
        manifest_path,
        config.get('prosody_cache_dir', '../data/prosody_cache'),
        hed_config,
        energy_augmentation=config.get('energy_augmentation'),
    )

    # Balanced sampling
    sampler = None
    if config.get('balance_emotions', True):
        weights = dataset.get_emotion_weights()
        if weights:
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            print("Using weighted sampler to balance emotions")

    train_loader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=hed_collate_fn,
    )

    # Validation loader
    val_loader = None
    val_manifest = config.get('val_manifest')
    if val_manifest:
        val_dataset = HEDProsodyDataset(
            val_manifest,
            config.get('prosody_cache_dir', '../data/prosody_cache'),
            hed_config,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=False,
            collate_fn=hed_collate_fn,
        )
        print(f"Validation set: {len(val_dataset)} samples")

    trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 30),
    )


if __name__ == "__main__":
    main()
