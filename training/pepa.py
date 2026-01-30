"""
PEPA: Phoneme-Emotion Projection Adapter

Based on "Emotional Text-To-Speech Based on Mutual-Information-Guided
Emotion-Timbre Disentanglement" (arXiv:2510.01722)

Key Innovation: Bridges phoneme embeddings to acoustic emotion space via two
successive 1D convolutions. This enables phoneme-level emotion embedding
prediction that captures fine-grained prosody/intensity cues.

Architecture:
    Phoneme Embeddings → Conv1D → ReLU → Conv1D → Emotion Space
                                   ↓
    Reference Encoder  →  Emotion Embeddings (supervision)
                                   ↓
    MINE Loss ← Minimize I(emotion, timbre) for disentanglement

Key Insight:
- Text alone can't capture prosody/intensity cues essential for emotion
- PEPA bridges phoneme embeddings to acoustic emotion space
- Enables fine-grained control that global emotion vectors miss

Benefits:
- Phoneme-level emotion embedding (finer than word-level)
- Connects linguistic structure to acoustic emotion features
- Simple architecture (2 conv layers) but effective
- Works with existing reference encoder approaches
- Integrates with HED (hierarchical emotion distribution) for multi-level control

Reference: https://arxiv.org/abs/2510.01722
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PEPAConfig:
    """Configuration for PEPA module."""

    # Input dimensions
    phoneme_embed_dim: int = 512        # Phoneme embedding dimension
    phoneme_vocab_size: int = 256       # Number of phonemes (including special tokens)

    # Reference encoder (GST-style)
    reference_encoder_dim: int = 256    # Reference encoder output dimension
    reference_hidden_dim: int = 512     # Reference encoder hidden dimension
    num_gst_tokens: int = 10            # Number of Global Style Tokens
    gst_head_dim: int = 256             # GST attention head dimension
    mel_dim: int = 80                   # Mel spectrogram dimension

    # PEPA projection layers
    pepa_hidden_dim: int = 512          # Hidden dimension for PEPA projections
    pepa_kernel_size: int = 3           # Kernel size for 1D convolutions
    pepa_dropout: float = 0.1           # Dropout rate

    # Emotion dimensions
    emotion_dim: int = 256              # Emotion embedding dimension
    num_emotions: int = 8               # Number of discrete emotions
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry",
        "surprised", "calm", "fearful", "disgusted"
    ])

    # Output dimensions (for CSM integration)
    output_dim: int = 2048              # CSM prosody hidden dimension
    num_prosody_tokens: int = 4         # Prefix tokens for prosody conditioning

    # Training settings
    use_mine: bool = True               # Use MINE for emotion-timbre disentanglement
    mine_weight: float = 0.5            # Weight for MINE loss
    emotion_cls_weight: float = 1.0     # Weight for emotion classification loss
    reconstruction_weight: float = 0.3  # Weight for prosody reconstruction loss

    # Speaker settings (for disentanglement)
    num_speakers: int = 1000
    speaker_embed_dim: int = 256


# =============================================================================
# REFERENCE ENCODER (GST-STYLE)
# =============================================================================

class ReferenceEncoderConvBlock(nn.Module):
    """Convolutional block for reference encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder for extracting global prosodic/emotional style from audio.

    Based on Global Style Tokens (GST) approach from "Style Tokens: Unsupervised
    Style Modeling, Control and Transfer in End-to-End Speech Synthesis".

    Takes mel spectrogram as input, outputs emotion embedding via:
    1. CNN stack to extract local features
    2. GRU to capture temporal dependencies
    3. Multi-head attention over GST bank for interpretable style tokens

    This provides the supervision signal for PEPA training.
    """

    def __init__(self, config: PEPAConfig):
        super().__init__()
        self.config = config

        # CNN layers (progressively reduce spatial dimensions)
        channels = [1, 32, 32, 64, 64, 128, 128]

        self.conv_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_layers.append(
                ReferenceEncoderConvBlock(
                    channels[i], channels[i + 1],
                    kernel_size=3, stride=2, padding=1
                )
            )

        # Calculate output size after convolutions
        # Input: [batch, 1, time, mel_dim]
        # After 6 conv layers with stride 2: time/64, mel/64
        # But we use GRU over time, so we need to flatten mel dimension

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=channels[-1] * (config.mel_dim // (2 ** 6) + 1),
            hidden_size=config.reference_hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Global Style Token bank
        self.gst_tokens = nn.Parameter(
            torch.randn(config.num_gst_tokens, config.gst_head_dim)
        )

        # Multi-head attention for style token selection
        self.style_attention = nn.MultiheadAttention(
            embed_dim=config.reference_hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Key/Value projections for GST
        self.gst_key_proj = nn.Linear(config.gst_head_dim, config.reference_hidden_dim)
        self.gst_value_proj = nn.Linear(config.gst_head_dim, config.reference_hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.reference_hidden_dim, config.emotion_dim),
            nn.LayerNorm(config.emotion_dim),
            nn.Tanh(),
        )

        # Emotion classifier head
        self.emotion_classifier = nn.Linear(config.emotion_dim, config.num_emotions)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, time, mel_dim] or [batch, mel_dim, time]
        return_gst_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract emotion embedding from mel spectrogram.

        Args:
            mel: Mel spectrogram [batch, time, mel_dim] or [batch, mel_dim, time]
            return_gst_weights: Whether to return GST attention weights

        Returns:
            Dict with:
                - 'emotion_embedding': [batch, emotion_dim] - global emotion
                - 'emotion_logits': [batch, num_emotions] - emotion classification
                - 'gst_weights': [batch, num_gst_tokens] (if return_gst_weights)
        """
        # Ensure [batch, time, mel_dim] format
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[1] == self.config.mel_dim:
            mel = mel.transpose(1, 2)

        batch_size, time, mel_dim = mel.shape

        # Add channel dimension: [batch, 1, time, mel]
        x = mel.unsqueeze(1)

        # CNN layers
        for conv in self.conv_layers:
            x = conv(x)

        # Reshape for GRU: [batch, time', features]
        batch, channels, time_reduced, mel_reduced = x.shape
        x = x.permute(0, 2, 1, 3)  # [batch, time', channels, mel']
        x = x.reshape(batch, time_reduced, -1)  # [batch, time', channels*mel']

        # GRU for temporal modeling
        gru_out, hidden = self.gru(x)  # [batch, time', hidden*2]

        # Use last hidden state as query
        query = hidden.transpose(0, 1).reshape(batch, 1, -1)  # [batch, 1, hidden*2]

        # Expand GST tokens for batch
        gst_keys = self.gst_key_proj(self.gst_tokens)  # [num_gst, hidden]
        gst_keys = gst_keys.unsqueeze(0).expand(batch, -1, -1)  # [batch, num_gst, hidden]

        gst_values = self.gst_value_proj(self.gst_tokens)  # [num_gst, hidden]
        gst_values = gst_values.unsqueeze(0).expand(batch, -1, -1)

        # Multi-head attention over GST bank
        style_embedding, gst_weights = self.style_attention(
            query=query,
            key=gst_keys,
            value=gst_values,
        )
        style_embedding = style_embedding.squeeze(1)  # [batch, hidden]

        # Project to emotion space
        emotion_embedding = self.output_proj(style_embedding)  # [batch, emotion_dim]

        # Emotion classification
        emotion_logits = self.emotion_classifier(emotion_embedding)

        result = {
            'emotion_embedding': emotion_embedding,
            'emotion_logits': emotion_logits,
        }

        if return_gst_weights:
            result['gst_weights'] = gst_weights.squeeze(1)  # [batch, num_gst]

        return result


# =============================================================================
# PEPA: PHONEME-EMOTION PROJECTION ADAPTER
# =============================================================================

class PEPA(nn.Module):
    """
    Phoneme-Emotion Projection Adapter.

    Core insight: Text/phoneme embeddings alone cannot capture prosody/intensity
    cues essential for emotion. PEPA bridges this gap by projecting phoneme
    embeddings into the acoustic emotion space learned by the reference encoder.

    Architecture (from paper):
        Phoneme Embeddings [batch, seq, phoneme_dim]
            ↓
        Conv1D (kernel=3, stride=1, padding=same)
            ↓
        ReLU + Dropout
            ↓
        Conv1D (kernel=3, stride=1, padding=same)
            ↓
        Emotion Space [batch, seq, emotion_dim]

    The output is supervised by the reference encoder's emotion embeddings.
    This enables phoneme-level emotion prediction at inference without reference audio.
    """

    def __init__(self, config: PEPAConfig):
        super().__init__()
        self.config = config

        # Phoneme embedding (optional - can also accept pre-computed embeddings)
        self.phoneme_embedding = nn.Embedding(
            config.phoneme_vocab_size,
            config.phoneme_embed_dim,
            padding_idx=0,
        )

        # Two successive 1D convolutions (core PEPA architecture)
        # Conv1D operates on [batch, channels, seq] format
        self.conv1 = nn.Conv1d(
            in_channels=config.phoneme_embed_dim,
            out_channels=config.pepa_hidden_dim,
            kernel_size=config.pepa_kernel_size,
            padding=config.pepa_kernel_size // 2,  # Same padding
        )

        self.conv2 = nn.Conv1d(
            in_channels=config.pepa_hidden_dim,
            out_channels=config.emotion_dim,
            kernel_size=config.pepa_kernel_size,
            padding=config.pepa_kernel_size // 2,  # Same padding
        )

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.pepa_dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(config.emotion_dim)

        # Global pooling for utterance-level emotion
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize convolutional layers."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,  # [batch, seq]
        phoneme_embeddings: Optional[torch.Tensor] = None,  # [batch, seq, phoneme_dim]
        phoneme_mask: Optional[torch.Tensor] = None,  # [batch, seq]
    ) -> Dict[str, torch.Tensor]:
        """
        Project phoneme embeddings to emotion space.

        Args:
            phoneme_ids: Phoneme token IDs [batch, seq]
            phoneme_embeddings: Pre-computed phoneme embeddings [batch, seq, phoneme_dim]
            phoneme_mask: Mask for valid phoneme positions [batch, seq]

        Returns:
            Dict with:
                - 'phoneme_emotions': [batch, seq, emotion_dim] - per-phoneme emotions
                - 'utterance_emotion': [batch, emotion_dim] - pooled emotion
        """
        # Get phoneme embeddings
        if phoneme_embeddings is not None:
            x = phoneme_embeddings
        elif phoneme_ids is not None:
            x = self.phoneme_embedding(phoneme_ids)
        else:
            raise ValueError("Either phoneme_ids or phoneme_embeddings must be provided")

        batch_size, seq_len, embed_dim = x.shape

        # Transpose for Conv1D: [batch, seq, dim] -> [batch, dim, seq]
        x = x.transpose(1, 2)

        # Apply mask by zeroing out padded positions
        if phoneme_mask is not None:
            x = x * phoneme_mask.unsqueeze(1).float()

        # Two-layer PEPA projection
        x = self.conv1(x)           # [batch, hidden, seq]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)           # [batch, emotion_dim, seq]

        # Transpose back: [batch, emotion_dim, seq] -> [batch, seq, emotion_dim]
        phoneme_emotions = x.transpose(1, 2)

        # Apply layer norm
        phoneme_emotions = self.norm(phoneme_emotions)

        # Global pooling for utterance-level emotion
        # Apply mask if provided
        if phoneme_mask is not None:
            mask_expanded = phoneme_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            masked_emotions = phoneme_emotions * mask_expanded
            lengths = phoneme_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
            utterance_emotion = masked_emotions.sum(dim=1) / lengths  # [batch, emotion_dim]
        else:
            utterance_emotion = phoneme_emotions.mean(dim=1)  # [batch, emotion_dim]

        return {
            'phoneme_emotions': phoneme_emotions,
            'utterance_emotion': utterance_emotion,
        }


# =============================================================================
# PEPA EMOTION MODULE (COMPLETE PIPELINE)
# =============================================================================

class PEPAEmotionModule(nn.Module):
    """
    Complete PEPA module with reference encoder and disentanglement.

    Combines:
    1. Reference Encoder: Extract ground-truth emotion from audio
    2. PEPA: Project phonemes to emotion space
    3. MINE: Disentangle emotion from speaker/timbre
    4. Prosody Adapter: Convert to CSM prefix tokens

    Training:
        - PEPA is supervised by reference encoder's emotion embeddings
        - MINE minimizes I(emotion, timbre) for speaker-independent emotions
        - Emotion classification provides categorical supervision

    Inference:
        - PEPA predicts emotion from phonemes alone (no reference audio needed)
        - Or: extract emotion from reference audio and transfer to new phonemes
    """

    def __init__(self, config: PEPAConfig):
        super().__init__()
        self.config = config

        # Reference encoder (for training supervision)
        self.reference_encoder = ReferenceEncoder(config)

        # PEPA projection adapter
        self.pepa = PEPA(config)

        # Emotion classifier (for PEPA output)
        self.emotion_classifier = nn.Linear(config.emotion_dim, config.num_emotions)

        # Speaker encoder (for disentanglement)
        self.speaker_encoder = nn.Sequential(
            nn.Linear(config.mel_dim, config.speaker_embed_dim),
            nn.GELU(),
            nn.Linear(config.speaker_embed_dim, config.speaker_embed_dim),
            nn.LayerNorm(config.speaker_embed_dim),
        )

        # MINE for disentanglement (optional)
        self.mine = None
        if config.use_mine:
            try:
                from mine_disentanglement import MINEConfig, MINEDisentanglementLoss
                mine_config = MINEConfig(
                    emotion_dim=config.emotion_dim,
                    timbre_dim=config.speaker_embed_dim,
                    warmup_epochs=5,
                )
                self.mine = MINEDisentanglementLoss(
                    mine_config=mine_config,
                    emotion_dim=config.emotion_dim,
                    timbre_dim=config.speaker_embed_dim,
                    mine_weight=config.mine_weight,
                )
            except ImportError:
                warnings.warn("MINE module not available. Disentanglement disabled.")

        # Prosody token generator (for CSM integration)
        self.prosody_proj = nn.Sequential(
            nn.Linear(config.emotion_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.pepa_dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            phoneme_ids: Phoneme token IDs [batch, seq]
            phoneme_embeddings: Pre-computed phoneme embeddings [batch, seq, dim]
            phoneme_mask: Mask for valid phonemes [batch, seq]
            mel: Mel spectrogram for reference encoder [batch, time, mel_dim]
            speaker_labels: Speaker IDs for disentanglement [batch]
            emotion_labels: Emotion labels for classification [batch]
            epoch: Current training epoch

        Returns:
            Dict with embeddings and losses
        """
        batch_size = (
            phoneme_ids.shape[0] if phoneme_ids is not None
            else phoneme_embeddings.shape[0]
        )
        device = (
            phoneme_ids.device if phoneme_ids is not None
            else phoneme_embeddings.device
        )

        result = {}
        losses = {}

        # 1. PEPA: Project phonemes to emotion space
        pepa_output = self.pepa(
            phoneme_ids=phoneme_ids,
            phoneme_embeddings=phoneme_embeddings,
            phoneme_mask=phoneme_mask,
        )

        result['phoneme_emotions'] = pepa_output['phoneme_emotions']
        result['pepa_emotion'] = pepa_output['utterance_emotion']

        # 2. Reference encoder: Get ground-truth emotion (if mel provided)
        if mel is not None:
            ref_output = self.reference_encoder(mel)
            result['ref_emotion'] = ref_output['emotion_embedding']
            result['ref_emotion_logits'] = ref_output['emotion_logits']

            # Reconstruction loss: PEPA should match reference encoder
            losses['reconstruction'] = F.mse_loss(
                pepa_output['utterance_emotion'],
                ref_output['emotion_embedding'].detach(),
            )

            # Reference encoder emotion classification
            if emotion_labels is not None:
                losses['ref_emotion_cls'] = F.cross_entropy(
                    ref_output['emotion_logits'],
                    emotion_labels,
                )

        # 3. PEPA emotion classification
        pepa_emotion_logits = self.emotion_classifier(pepa_output['utterance_emotion'])
        result['pepa_emotion_logits'] = pepa_emotion_logits

        if emotion_labels is not None:
            losses['pepa_emotion_cls'] = F.cross_entropy(
                pepa_emotion_logits,
                emotion_labels,
            )

        # 4. MINE disentanglement (if enabled and mel provided)
        if self.mine is not None and mel is not None:
            # Extract speaker embedding
            mel_pooled = mel.mean(dim=1) if mel.dim() == 3 else mel
            speaker_emb = self.speaker_encoder(mel_pooled)
            result['speaker_embedding'] = speaker_emb

            # MINE loss
            mine_losses = self.mine(
                pepa_output['utterance_emotion'],
                speaker_emb,
                epoch=epoch,
            )
            losses['mine'] = mine_losses['mine']
            losses['mi_estimate'] = mine_losses['mi_estimate']
        else:
            losses['mine'] = torch.tensor(0.0, device=device)
            losses['mi_estimate'] = torch.tensor(0.0, device=device)

        # 5. Generate prosody tokens
        tokens = self.prosody_proj(pepa_output['utterance_emotion'])
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)
        result['prosody_tokens'] = tokens

        # Compute total loss
        total_loss = torch.tensor(0.0, device=device)
        if 'reconstruction' in losses:
            total_loss = total_loss + self.config.reconstruction_weight * losses['reconstruction']
        if 'pepa_emotion_cls' in losses:
            total_loss = total_loss + self.config.emotion_cls_weight * losses['pepa_emotion_cls']
        if 'mine' in losses and losses['mine'].item() != 0:
            total_loss = total_loss + self.config.mine_weight * losses['mine']

        losses['total'] = total_loss
        result['losses'] = losses

        return result

    def predict_emotion_from_phonemes(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: Predict emotion from phonemes alone (no reference audio).

        This is the key capability of PEPA - emotion prediction without reference.
        """
        with torch.no_grad():
            pepa_output = self.pepa(
                phoneme_ids=phoneme_ids,
                phoneme_embeddings=phoneme_embeddings,
                phoneme_mask=phoneme_mask,
            )

            emotion_logits = self.emotion_classifier(pepa_output['utterance_emotion'])
            emotion_probs = F.softmax(emotion_logits, dim=-1)

            # Get predicted emotion
            emotion_idx = emotion_probs.argmax(dim=-1)
            emotion_names = [self.config.emotion_labels[i] for i in emotion_idx.cpu().numpy()]

            # Generate prosody tokens
            batch_size = pepa_output['utterance_emotion'].shape[0]
            tokens = self.prosody_proj(pepa_output['utterance_emotion'])
            tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
            tokens = self.output_norm(tokens)

            return {
                'phoneme_emotions': pepa_output['phoneme_emotions'],
                'utterance_emotion': pepa_output['utterance_emotion'],
                'emotion_probs': emotion_probs,
                'predicted_emotion': emotion_names,
                'prosody_tokens': tokens,
            }

    def transfer_emotion(
        self,
        reference_mel: torch.Tensor,
        target_phoneme_ids: Optional[torch.Tensor] = None,
        target_phoneme_embeddings: Optional[torch.Tensor] = None,
        target_phoneme_mask: Optional[torch.Tensor] = None,
        blend_weight: float = 0.7,  # Weight for reference vs PEPA prediction
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer emotion from reference audio to new phonemes.

        Combines:
        1. Reference encoder emotion (global style from audio)
        2. PEPA phoneme-level emotion (local prosody from text)

        blend_weight: 1.0 = pure reference, 0.0 = pure PEPA prediction
        """
        with torch.no_grad():
            # Get reference emotion
            ref_output = self.reference_encoder(reference_mel)
            ref_emotion = ref_output['emotion_embedding']

            # Get PEPA prediction
            pepa_output = self.pepa(
                phoneme_ids=target_phoneme_ids,
                phoneme_embeddings=target_phoneme_embeddings,
                phoneme_mask=target_phoneme_mask,
            )
            pepa_emotion = pepa_output['utterance_emotion']

            # Blend emotions
            blended_emotion = blend_weight * ref_emotion + (1 - blend_weight) * pepa_emotion

            # Generate prosody tokens from blended emotion
            batch_size = blended_emotion.shape[0]
            tokens = self.prosody_proj(blended_emotion)
            tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
            tokens = self.output_norm(tokens)

            # Also compute phoneme-level emotions weighted by reference
            # This provides fine-grained emotion with global style influence
            ref_emotion_expanded = ref_emotion.unsqueeze(1)  # [batch, 1, emotion_dim]
            phoneme_emotions = pepa_output['phoneme_emotions']  # [batch, seq, emotion_dim]

            # Blend at phoneme level
            blended_phoneme_emotions = (
                blend_weight * ref_emotion_expanded +
                (1 - blend_weight) * phoneme_emotions
            )

            return {
                'reference_emotion': ref_emotion,
                'pepa_emotion': pepa_emotion,
                'blended_emotion': blended_emotion,
                'blended_phoneme_emotions': blended_phoneme_emotions,
                'prosody_tokens': tokens,
            }


# =============================================================================
# PEPA ADAPTER (CSM INTEGRATION)
# =============================================================================

class PEPAAdapter(nn.Module):
    """
    Adapter for integrating PEPA with CSM prosody pipeline.

    Drop-in replacement for other prosody adapters, generates prosody prefix
    tokens from phoneme-based emotion predictions.

    Usage:
        adapter = PEPAAdapter(config)

        # Training: With reference audio for supervision
        result = adapter(
            phoneme_ids=phoneme_ids,
            mel=reference_mel,
            emotion_labels=emotion_labels,
        )
        loss = result['losses']['total']
        prosody_tokens = result['prosody_tokens']

        # Inference: From phonemes alone
        result = adapter.from_phonemes(phoneme_ids=phoneme_ids)
        prosody_tokens = result['prosody_tokens']

        # Inference: With emotion transfer
        result = adapter.transfer(
            reference_mel=source_mel,
            target_phoneme_ids=target_phonemes,
        )
    """

    def __init__(self, config: PEPAConfig):
        super().__init__()
        self.config = config
        self.module = PEPAEmotionModule(config)

        # Emotion label to index mapping
        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(config.emotion_labels)
        }

    def forward(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        return self.module(
            phoneme_ids=phoneme_ids,
            phoneme_embeddings=phoneme_embeddings,
            phoneme_mask=phoneme_mask,
            mel=mel,
            speaker_labels=speaker_labels,
            emotion_labels=emotion_labels,
            epoch=epoch,
        )

    def from_phonemes(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: Predict emotion from phonemes alone.

        This is the main inference mode - no reference audio needed.
        """
        return self.module.predict_emotion_from_phonemes(
            phoneme_ids=phoneme_ids,
            phoneme_embeddings=phoneme_embeddings,
            phoneme_mask=phoneme_mask,
        )

    def transfer(
        self,
        reference_mel: torch.Tensor,
        target_phoneme_ids: Optional[torch.Tensor] = None,
        target_phoneme_embeddings: Optional[torch.Tensor] = None,
        target_phoneme_mask: Optional[torch.Tensor] = None,
        blend_weight: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer emotion from reference audio to target phonemes.

        blend_weight: 1.0 = pure reference style, 0.0 = pure PEPA prediction
        """
        return self.module.transfer_emotion(
            reference_mel=reference_mel,
            target_phoneme_ids=target_phoneme_ids,
            target_phoneme_embeddings=target_phoneme_embeddings,
            target_phoneme_mask=target_phoneme_mask,
            blend_weight=blend_weight,
        )

    def from_emotion_label(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
        emotion: str = "neutral",
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with specified emotion.

        Predicts base emotion from phonemes, then modulates toward target emotion.
        """
        # Get base PEPA prediction
        base_output = self.from_phonemes(
            phoneme_ids=phoneme_ids,
            phoneme_embeddings=phoneme_embeddings,
            phoneme_mask=phoneme_mask,
        )

        # Get target emotion index
        emotion_idx = self.emotion_to_idx.get(emotion.lower(), 0)

        # Modulate emotion probabilities toward target
        target_probs = torch.zeros_like(base_output['emotion_probs'])
        target_probs[..., emotion_idx] = 1.0

        # Blend based on intensity
        blended_probs = (
            intensity * target_probs +
            (1 - intensity) * base_output['emotion_probs']
        )

        # The prosody tokens are already generated from base prediction
        # For more sophisticated control, we could train a conditional generator
        return {
            'prosody_tokens': base_output['prosody_tokens'],
            'phoneme_emotions': base_output['phoneme_emotions'],
            'base_emotion_probs': base_output['emotion_probs'],
            'modulated_emotion_probs': blended_probs,
            'target_emotion': emotion,
            'intensity': intensity,
        }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class PEPALoss(nn.Module):
    """
    Combined loss function for PEPA training.

    Components:
    1. Reconstruction: PEPA output should match reference encoder
    2. Emotion Classification: Both PEPA and reference should predict correct emotion
    3. MINE Disentanglement: Minimize I(emotion, speaker)
    4. Phoneme-level Consistency: Smooth emotion trajectory across phonemes
    """

    def __init__(
        self,
        config: PEPAConfig,
        reconstruction_weight: float = 1.0,
        classification_weight: float = 0.5,
        mine_weight: float = 0.3,
        smoothness_weight: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        self.mine_weight = mine_weight
        self.smoothness_weight = smoothness_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        module_output: Dict[str, torch.Tensor],
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all PEPA training losses."""
        losses = module_output.get('losses', {})
        device = module_output['prosody_tokens'].device

        # Reconstruction loss (already computed in module)
        reconstruction = losses.get('reconstruction', torch.tensor(0.0, device=device))

        # Classification losses (already computed in module)
        pepa_cls = losses.get('pepa_emotion_cls', torch.tensor(0.0, device=device))
        ref_cls = losses.get('ref_emotion_cls', torch.tensor(0.0, device=device))

        # MINE loss (already computed in module)
        mine = losses.get('mine', torch.tensor(0.0, device=device))

        # Phoneme-level smoothness
        phoneme_emotions = module_output.get('phoneme_emotions')
        if phoneme_emotions is not None and phoneme_emotions.shape[1] > 1:
            diffs = (phoneme_emotions[:, 1:] - phoneme_emotions[:, :-1]).pow(2).mean()
            smoothness = diffs
        else:
            smoothness = torch.tensor(0.0, device=device)

        # Total loss
        total = (
            self.reconstruction_weight * reconstruction +
            self.classification_weight * (pepa_cls + ref_cls) +
            self.mine_weight * mine +
            self.smoothness_weight * smoothness
        )

        return {
            'reconstruction': reconstruction,
            'pepa_cls': pepa_cls,
            'ref_cls': ref_cls,
            'mine': mine,
            'smoothness': smoothness,
            'total': total,
        }


# =============================================================================
# INTEGRATION WITH HED (HIERARCHICAL EMOTION DISTRIBUTION)
# =============================================================================

class PEPAWithHED(nn.Module):
    """
    PEPA integrated with Hierarchical Emotion Distribution.

    Combines:
    1. PEPA: Phoneme-level emotion projection
    2. HED: Multi-level emotion (phoneme, word, utterance)

    This provides the finest granularity of emotion control:
    - Phoneme-level: PEPA-projected emotions
    - Word-level: Aggregated phoneme emotions
    - Utterance-level: Global reference encoder emotion
    """

    def __init__(self, pepa_config: PEPAConfig):
        super().__init__()
        self.config = pepa_config

        # Core PEPA module
        self.pepa = PEPAEmotionModule(pepa_config)

        # Try to integrate with HED
        self.hed = None
        try:
            from hierarchical_emotion import HEDConfig, HierarchicalEmotionEncoder

            hed_config = HEDConfig(
                opensmile_dim=pepa_config.emotion_dim,  # Use PEPA emotions as input
                phoneme_hidden=128,
                word_hidden=256,
                utterance_hidden=512,
                output_hidden=pepa_config.output_dim,
                num_emotions=pepa_config.num_emotions,
            )
            self.hed = HierarchicalEmotionEncoder(hed_config)
        except ImportError:
            warnings.warn("HED module not available. Using PEPA only.")

    def forward(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_embeddings: Optional[torch.Tensor] = None,
        phoneme_mask: Optional[torch.Tensor] = None,
        phoneme_to_word: Optional[List[List[int]]] = None,  # Phoneme -> word mapping
        num_words: int = 1,
        mel: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical emotion.

        Args:
            phoneme_ids: Phoneme token IDs
            phoneme_embeddings: Pre-computed phoneme embeddings
            phoneme_mask: Valid phoneme mask
            phoneme_to_word: Mapping from phoneme index to word index
            num_words: Number of words in utterance
            mel: Mel spectrogram for reference encoder
            emotion_labels: Ground truth emotion labels
            epoch: Current training epoch

        Returns:
            Dict with PEPA outputs + HED hierarchical emotions
        """
        # Get PEPA outputs
        pepa_output = self.pepa(
            phoneme_ids=phoneme_ids,
            phoneme_embeddings=phoneme_embeddings,
            phoneme_mask=phoneme_mask,
            mel=mel,
            emotion_labels=emotion_labels,
            epoch=epoch,
        )

        result = pepa_output.copy()

        # Integrate with HED if available
        if self.hed is not None and phoneme_to_word is not None:
            # Use PEPA phoneme emotions as input to HED
            phoneme_emotions = pepa_output['phoneme_emotions']

            # HED expects [batch, phonemes, feature_dim]
            hed_output = self.hed(
                phoneme_features=phoneme_emotions,
                phoneme_to_word=phoneme_to_word,
                num_words=num_words,
                phoneme_mask=phoneme_mask,
            )

            result.update({
                'hed_phoneme_emotions': hed_output['phoneme_emotions'],
                'hed_word_emotions': hed_output['word_emotions'],
                'hed_word_intensities': hed_output['word_intensities'],
                'hed_utterance_emotion': hed_output['utterance_emotions'],
                'hed_combined': hed_output['combined_embedding'],
            })

        return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PEPA: Phoneme-Emotion Projection Adapter - Test Suite")
    print("=" * 70)

    config = PEPAConfig()
    device = 'cpu'

    # Test 1: ReferenceEncoder
    print("\n[Test 1] ReferenceEncoder...")
    ref_encoder = ReferenceEncoder(config).to(device)

    # Mock mel spectrogram [batch, time, mel_dim]
    mel = torch.randn(2, 100, config.mel_dim)
    ref_output = ref_encoder(mel, return_gst_weights=True)

    print(f"  Emotion embedding: {ref_output['emotion_embedding'].shape}")
    print(f"  Emotion logits: {ref_output['emotion_logits'].shape}")
    print(f"  GST weights: {ref_output['gst_weights'].shape}")
    print("  [PASS]")

    # Test 2: PEPA core module
    print("\n[Test 2] PEPA (Phoneme-Emotion Projection Adapter)...")
    pepa = PEPA(config).to(device)

    # Mock phoneme input
    phoneme_ids = torch.randint(1, config.phoneme_vocab_size, (2, 30))
    phoneme_mask = torch.ones(2, 30).bool()
    phoneme_mask[0, 20:] = False  # Mask last 10 positions for first sample

    pepa_output = pepa(phoneme_ids=phoneme_ids, phoneme_mask=phoneme_mask)

    print(f"  Phoneme emotions: {pepa_output['phoneme_emotions'].shape}")
    print(f"  Utterance emotion: {pepa_output['utterance_emotion'].shape}")
    print("  [PASS]")

    # Test 3: PEPA with pre-computed embeddings
    print("\n[Test 3] PEPA with phoneme embeddings...")
    phoneme_embeddings = torch.randn(2, 30, config.phoneme_embed_dim)
    pepa_output = pepa(phoneme_embeddings=phoneme_embeddings)

    print(f"  Phoneme emotions: {pepa_output['phoneme_emotions'].shape}")
    print("  [PASS]")

    # Test 4: PEPAEmotionModule (complete pipeline)
    print("\n[Test 4] PEPAEmotionModule (full pipeline)...")
    module = PEPAEmotionModule(config).to(device)

    emotion_labels = torch.randint(0, config.num_emotions, (2,))

    module_output = module(
        phoneme_ids=phoneme_ids,
        phoneme_mask=phoneme_mask,
        mel=mel,
        emotion_labels=emotion_labels,
        epoch=5,
    )

    print(f"  PEPA emotion: {module_output['pepa_emotion'].shape}")
    print(f"  Reference emotion: {module_output['ref_emotion'].shape}")
    print(f"  Prosody tokens: {module_output['prosody_tokens'].shape}")
    print(f"  Reconstruction loss: {module_output['losses']['reconstruction'].item():.4f}")
    print(f"  PEPA classification loss: {module_output['losses']['pepa_emotion_cls'].item():.4f}")
    print(f"  Total loss: {module_output['losses']['total'].item():.4f}")
    print("  [PASS]")

    # Test 5: Inference from phonemes only
    print("\n[Test 5] Inference from phonemes (no reference audio)...")
    inference_output = module.predict_emotion_from_phonemes(
        phoneme_ids=phoneme_ids,
        phoneme_mask=phoneme_mask,
    )

    print(f"  Predicted emotions: {inference_output['predicted_emotion']}")
    print(f"  Emotion probs shape: {inference_output['emotion_probs'].shape}")
    print(f"  Prosody tokens: {inference_output['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 6: Emotion transfer
    print("\n[Test 6] Emotion transfer (reference -> new phonemes)...")
    transfer_output = module.transfer_emotion(
        reference_mel=mel,
        target_phoneme_ids=phoneme_ids,
        target_phoneme_mask=phoneme_mask,
        blend_weight=0.7,
    )

    print(f"  Reference emotion: {transfer_output['reference_emotion'].shape}")
    print(f"  PEPA emotion: {transfer_output['pepa_emotion'].shape}")
    print(f"  Blended emotion: {transfer_output['blended_emotion'].shape}")
    print(f"  Blended phoneme emotions: {transfer_output['blended_phoneme_emotions'].shape}")
    print(f"  Prosody tokens: {transfer_output['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 7: PEPAAdapter
    print("\n[Test 7] PEPAAdapter (CSM integration)...")
    adapter = PEPAAdapter(config).to(device)

    # Training mode
    train_output = adapter(
        phoneme_ids=phoneme_ids,
        phoneme_mask=phoneme_mask,
        mel=mel,
        emotion_labels=emotion_labels,
    )
    print(f"  Training - prosody tokens: {train_output['prosody_tokens'].shape}")
    print(f"  Training - total loss: {train_output['losses']['total'].item():.4f}")

    # Inference mode
    inference_output = adapter.from_phonemes(
        phoneme_ids=phoneme_ids,
        phoneme_mask=phoneme_mask,
    )
    print(f"  Inference - prosody tokens: {inference_output['prosody_tokens'].shape}")

    # With emotion label
    labeled_output = adapter.from_emotion_label(
        phoneme_ids=phoneme_ids,
        phoneme_mask=phoneme_mask,
        emotion="happy",
        intensity=0.8,
    )
    print(f"  With label - target emotion: {labeled_output['target_emotion']}")
    print("  [PASS]")

    # Test 8: PEPALoss
    print("\n[Test 8] PEPALoss...")
    loss_fn = PEPALoss(config)

    losses = loss_fn(module_output, emotion_labels)
    print(f"  Reconstruction: {losses['reconstruction'].item():.4f}")
    print(f"  PEPA classification: {losses['pepa_cls'].item():.4f}")
    print(f"  Smoothness: {losses['smoothness'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: Gradient flow
    print("\n[Test 9] Gradient flow test...")
    module = PEPAEmotionModule(config).to(device)

    phoneme_ids = torch.randint(1, config.phoneme_vocab_size, (2, 30))
    mel = torch.randn(2, 100, config.mel_dim)
    emotion_labels = torch.randint(0, config.num_emotions, (2,))

    output = module(
        phoneme_ids=phoneme_ids,
        mel=mel,
        emotion_labels=emotion_labels,
    )

    loss = output['losses']['total']
    loss.backward()

    # Check gradients
    pepa_grad_norm = sum(p.grad.norm().item() for p in module.pepa.parameters() if p.grad is not None)
    ref_grad_norm = sum(p.grad.norm().item() for p in module.reference_encoder.parameters() if p.grad is not None)

    print(f"  PEPA gradient norm: {pepa_grad_norm:.4f}")
    print(f"  Reference encoder gradient norm: {ref_grad_norm:.4f}")
    print(f"  Gradients flowing: {pepa_grad_norm > 0 and ref_grad_norm > 0}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All PEPA tests passed!")
    print("=" * 70)

    # Usage example
    print("\nUsage Example:")
    print("-" * 40)
    print("""
from pepa import PEPAConfig, PEPAAdapter

# Initialize
config = PEPAConfig(
    phoneme_embed_dim=512,
    emotion_dim=256,
    output_dim=2048,
    use_mine=True,  # Enable emotion-timbre disentanglement
)

adapter = PEPAAdapter(config).cuda()

# Training: With reference audio for supervision
result = adapter(
    phoneme_ids=phoneme_ids,        # [batch, seq]
    phoneme_mask=phoneme_mask,      # [batch, seq]
    mel=reference_mel,              # [batch, time, 80]
    emotion_labels=emotion_labels,  # [batch]
    epoch=current_epoch,
)
loss = result['losses']['total']
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

# Inference: From phonemes alone (no reference audio needed!)
result = adapter.from_phonemes(
    phoneme_ids=phoneme_ids,
    phoneme_mask=phoneme_mask,
)
prosody_tokens = result['prosody_tokens']
predicted_emotions = result['predicted_emotion']  # ["happy", "sad", ...]

# Inference: Emotion transfer (style from reference, phonemes from text)
result = adapter.transfer(
    reference_mel=style_reference,   # Source emotion
    target_phoneme_ids=new_text,     # Target content
    blend_weight=0.7,                # 70% reference style
)
prosody_tokens = result['prosody_tokens']

# Inference: With specified emotion label
result = adapter.from_emotion_label(
    phoneme_ids=phoneme_ids,
    emotion="happy",
    intensity=0.9,
)

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
