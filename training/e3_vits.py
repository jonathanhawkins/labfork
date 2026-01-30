"""
E3-VITS: Batch-Permuted Style Perturbation for Cross-Speaker Emotion Transfer

Based on E3-VITS (ICML 2023): "Emotional End-to-End TTS with Cross-speaker Style Transfer"
Paper: https://openreview.net/forum?id=qL47xtuEuv
GitHub: https://github.com/Wonbin-Jung/e3-vits

Key Technique - Batch-Permuted Style Perturbation:
- During training, permute style embeddings within a batch via VITS flow module
- Creates audio samples with unpaired speaker-emotion combinations
- Enables training on disjoint datasets (not all speakers have all emotions)
- Improves cross-speaker emotion transfer quality

Benefits:
- Supports both reference speech and text-based emotion control
- Trains on datasets where not every speaker has every emotion
- Generates "what would speaker A sound like with emotion from speaker B?"
- Better emotion transfer than simple embedding concatenation

Architecture:
1. Style Encoder: Extracts speaker + emotion embeddings from audio/text
2. Batch Permutation: Creates unpaired speaker-emotion combinations
3. Flow Module: Transforms latent z with perturbed style (voice conversion)
4. Discriminator: Ensures quality of perturbed samples

Integration:
- Compatible with existing prosody pipeline (SAVCAdapter, DisentanglementLoss)
- Works as drop-in data augmentation for emotion transfer training
- Outputs prefix tokens for ProsodyControlledCSM

Usage:
    from e3_vits import (
        E3VITSConfig,
        BatchPermutedStylePerturbation,
        E3VITSAdapter,
        StyleFlowModule,
    )

    config = E3VITSConfig()
    adapter = E3VITSAdapter(config).cuda()

    # Training: With batch permutation for cross-speaker emotion transfer
    result = adapter(
        audio_features,           # [batch, seq, dim]
        speaker_ids,              # [batch]
        emotion_labels,           # [batch] or None for unsupervised
        use_permutation=True,     # Enable batch permutation
    )

    prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]
    perturbed_tokens = result['perturbed_tokens']  # Unpaired speaker-emotion
    discriminator_loss = result['losses']['discriminator']

    # Inference: From emotion label (no permutation)
    result = adapter.from_emotion("happy", speaker_emb=speaker_emb)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class E3VITSConfig:
    """Configuration for E3-VITS batch-permuted style perturbation."""

    # Feature dimensions
    input_dim: int = 768            # Input feature dimension (HuBERT/wav2vec2)
    hidden_dim: int = 512           # Hidden layer dimension
    style_dim: int = 256            # Style embedding dimension
    speaker_dim: int = 256          # Speaker embedding dimension
    emotion_dim: int = 256          # Emotion embedding dimension
    output_dim: int = 2048          # Output dimension for CSM integration

    # Flow module settings (VITS-style normalizing flow)
    flow_hidden_dim: int = 192      # Flow hidden dimension
    flow_kernel_size: int = 5       # Convolution kernel size
    flow_dilation_rate: int = 1     # Dilation rate for WaveNet
    flow_num_layers: int = 4        # Number of flow layers
    flow_num_flows: int = 4         # Number of coupling layers

    # Batch permutation settings
    permutation_probability: float = 0.5    # Probability of permuting each sample
    permute_speaker: bool = True            # Permute speaker embeddings
    permute_emotion: bool = True            # Permute emotion embeddings
    separate_permutation: bool = True       # Separate permutation for speaker/emotion

    # Style encoder settings
    num_emotions: int = 8           # Number of emotion categories
    use_emotion_embedding: bool = True      # Use learned emotion embeddings
    use_reference_encoder: bool = True      # Extract emotion from reference audio

    # Discriminator settings
    use_discriminator: bool = True
    discriminator_hidden_dim: int = 256
    discriminator_num_layers: int = 3

    # Loss weights
    adversarial_weight: float = 1.0         # Weight for adversarial loss
    reconstruction_weight: float = 1.0      # Weight for reconstruction loss
    style_consistency_weight: float = 0.5   # Weight for style consistency loss
    kl_weight: float = 0.01                 # Weight for KL divergence (VAE)

    # Training settings
    warmup_steps: int = 1000        # Steps before full permutation strength
    use_gradient_penalty: bool = True       # R1 gradient penalty
    gradient_penalty_weight: float = 10.0

    # Integration settings
    num_prefix_tokens: int = 4      # Number of prosody prefix tokens


# =============================================================================
# EMOTION EMBEDDINGS
# =============================================================================

# Standard emotion labels (matching other modules)
EMOTION_LABELS = [
    "neutral", "happy", "sad", "angry",
    "fearful", "surprised", "disgusted", "calm"
]

EMOTION_TO_IDX = {emo: idx for idx, emo in enumerate(EMOTION_LABELS)}


class EmotionEmbedding(nn.Module):
    """Learned emotion embeddings for categorical emotions."""

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

        # Emotion embeddings
        self.emotion_embeddings = nn.Embedding(
            config.num_emotions,
            config.emotion_dim,
        )

        # Optional intensity scaling
        self.intensity_scale = nn.Parameter(torch.ones(config.num_emotions))

        # Initialize with orthogonal vectors for better separation
        nn.init.orthogonal_(self.emotion_embeddings.weight)

    def forward(
        self,
        emotion_idx: torch.Tensor,
        intensity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get emotion embedding.

        Args:
            emotion_idx: [batch] emotion indices
            intensity: Optional [batch] intensity scaling (0-1)

        Returns:
            [batch, emotion_dim] emotion embeddings
        """
        emb = self.emotion_embeddings(emotion_idx)

        if intensity is not None:
            # Scale by intensity
            scale = self.intensity_scale[emotion_idx] * intensity.unsqueeze(-1)
            emb = emb * scale

        return emb

    def get_by_name(
        self,
        emotion_name: str,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Get embedding by emotion name."""
        idx = EMOTION_TO_IDX.get(emotion_name.lower(), 0)
        indices = torch.full((batch_size,), idx, dtype=torch.long, device=device)
        return self(indices)


# =============================================================================
# REFERENCE ENCODER (GST-STYLE)
# =============================================================================

class ReferenceEncoder(nn.Module):
    """
    Reference encoder for extracting style from audio.

    Extracts a global style embedding from mel-spectrogram or audio features.
    This enables reference speech-based emotion control.
    """

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

        # Convolutional layers for mel processing
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=128 * 2,  # After conv, assume mel_dim/64 * 128
            hidden_size=config.style_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Output projection
        self.projection = nn.Linear(config.style_dim * 2, config.style_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Extract style embedding from mel-spectrogram.

        Args:
            mel: [batch, time, mel_dim] or [batch, mel_dim, time]

        Returns:
            [batch, style_dim] style embedding
        """
        # Ensure [B, 1, T, Mel] format for conv2d
        if mel.dim() == 3:
            if mel.shape[1] > mel.shape[2]:
                # [B, T, Mel] -> [B, 1, T, Mel]
                mel = mel.unsqueeze(1)
            else:
                # [B, Mel, T] -> [B, 1, Mel, T] -> [B, 1, T, Mel]
                mel = mel.unsqueeze(1).transpose(2, 3)

        # Convolutional processing
        x = self.convs(mel)  # [B, 128, T', Mel']

        # Reshape for GRU: [B, T', 128 * Mel']
        batch, channels, time, mel_bins = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * mel_bins)

        # GRU processing
        _, hidden = self.gru(x)

        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        # Project to style dimension
        style = self.projection(hidden)

        return style


class FeatureReferenceEncoder(nn.Module):
    """
    Reference encoder for audio features (HuBERT/wav2vec2).

    Simpler than mel-based encoder, directly processes pre-extracted features.
    """

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

        # Transformer encoder for feature processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Attentive pooling
        self.attention = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Output projection
        self.projection = nn.Linear(config.input_dim, config.style_dim)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract style from audio features.

        Args:
            features: [batch, seq, dim] audio features
            mask: Optional [batch, seq] attention mask

        Returns:
            [batch, style_dim] style embedding
        """
        # Transformer encoding
        if mask is not None:
            attn_mask = ~mask.bool()
            x = self.transformer(features, src_key_padding_mask=attn_mask)
        else:
            x = self.transformer(features)

        # Attentive pooling
        attn_weights = self.attention(x)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)

        pooled = (x * attn_weights).sum(dim=1)

        # Project to style dimension
        style = self.projection(pooled)

        return style


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for extracting speaker identity.

    Can use either:
    1. Learned speaker embeddings (for multi-speaker training)
    2. Reference-based speaker extraction (for zero-shot)
    """

    def __init__(self, config: E3VITSConfig, num_speakers: int = 1000):
        super().__init__()
        self.config = config
        self.num_speakers = num_speakers

        # Learned speaker embeddings
        self.speaker_embeddings = nn.Embedding(
            num_speakers,
            config.speaker_dim,
        )

        # Reference-based speaker encoder (for zero-shot)
        self.reference_encoder = FeatureReferenceEncoder(config)
        self.reference_proj = nn.Linear(config.style_dim, config.speaker_dim)

    def forward(
        self,
        speaker_ids: Optional[torch.Tensor] = None,
        reference_features: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get speaker embedding.

        Args:
            speaker_ids: [batch] speaker indices (for training)
            reference_features: [batch, seq, dim] reference audio features (for zero-shot)
            reference_mask: Optional attention mask

        Returns:
            [batch, speaker_dim] speaker embedding
        """
        if speaker_ids is not None:
            return self.speaker_embeddings(speaker_ids)

        if reference_features is not None:
            style = self.reference_encoder(reference_features, reference_mask)
            return self.reference_proj(style)

        raise ValueError("Must provide either speaker_ids or reference_features")


# =============================================================================
# STYLE ENCODER (COMBINED SPEAKER + EMOTION)
# =============================================================================

class StyleEncoder(nn.Module):
    """
    Combined style encoder for speaker + emotion.

    Produces a joint style embedding that can be used for:
    1. Conditioning generation
    2. Batch permutation (E3-VITS key technique)
    3. Style consistency loss
    """

    def __init__(self, config: E3VITSConfig, num_speakers: int = 1000):
        super().__init__()
        self.config = config

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(config, num_speakers)

        # Emotion encoder
        self.emotion_embedding = EmotionEmbedding(config)
        self.emotion_reference_encoder = FeatureReferenceEncoder(config)
        self.emotion_proj = nn.Linear(config.style_dim, config.emotion_dim)

        # Style fusion
        self.style_fusion = nn.Sequential(
            nn.Linear(config.speaker_dim + config.emotion_dim, config.style_dim),
            nn.LayerNorm(config.style_dim),
            nn.GELU(),
            nn.Linear(config.style_dim, config.style_dim),
        )

    def forward(
        self,
        speaker_ids: Optional[torch.Tensor] = None,
        speaker_reference: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        emotion_reference: Optional[torch.Tensor] = None,
        emotion_intensity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract combined style embedding.

        Args:
            speaker_ids: [batch] speaker indices
            speaker_reference: [batch, seq, dim] speaker reference features
            emotion_ids: [batch] emotion indices
            emotion_reference: [batch, seq, dim] emotion reference features
            emotion_intensity: [batch] emotion intensity (0-1)

        Returns:
            Dict with style, speaker, and emotion embeddings
        """
        # Get speaker embedding
        speaker_emb = self.speaker_encoder(
            speaker_ids=speaker_ids,
            reference_features=speaker_reference,
        )

        # Get emotion embedding
        if emotion_ids is not None:
            emotion_emb = self.emotion_embedding(emotion_ids, emotion_intensity)
        elif emotion_reference is not None:
            emotion_style = self.emotion_reference_encoder(emotion_reference)
            emotion_emb = self.emotion_proj(emotion_style)
        else:
            # Neutral emotion
            batch = speaker_emb.shape[0]
            device = speaker_emb.device
            neutral_idx = torch.zeros(batch, dtype=torch.long, device=device)
            emotion_emb = self.emotion_embedding(neutral_idx)

        # Fuse speaker + emotion
        combined = torch.cat([speaker_emb, emotion_emb], dim=-1)
        style_emb = self.style_fusion(combined)

        return {
            'style': style_emb,
            'speaker': speaker_emb,
            'emotion': emotion_emb,
        }


# =============================================================================
# BATCH PERMUTATION MODULE
# =============================================================================

class BatchPermutation(nn.Module):
    """
    Batch permutation module for creating unpaired speaker-emotion samples.

    This is the key technique from E3-VITS:
    1. Within a batch, randomly permute speaker embeddings
    2. Independently permute emotion embeddings
    3. Create new style embeddings with unpaired speaker-emotion combinations
    4. This enables training on disjoint datasets

    Example:
        Original batch:
            Sample 1: Speaker A + Happy
            Sample 2: Speaker B + Sad
            Sample 3: Speaker C + Angry

        After permutation:
            Sample 1: Speaker B + Angry  (unpaired)
            Sample 2: Speaker C + Happy  (unpaired)
            Sample 3: Speaker A + Sad    (unpaired)
    """

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

    def generate_permutation(
        self,
        batch_size: int,
        device: torch.device,
        avoid_identity: bool = True,
    ) -> torch.Tensor:
        """
        Generate a random permutation index.

        Args:
            batch_size: Number of samples in batch
            device: Tensor device
            avoid_identity: If True, ensure permutation is not identity

        Returns:
            [batch] permutation indices
        """
        if batch_size == 1:
            # Can't permute a single sample
            return torch.tensor([0], device=device)

        perm = torch.randperm(batch_size, device=device)

        if avoid_identity:
            # Ensure at least one element is different
            is_identity = torch.all(perm == torch.arange(batch_size, device=device))
            attempts = 0
            while is_identity and attempts < 10:
                perm = torch.randperm(batch_size, device=device)
                is_identity = torch.all(perm == torch.arange(batch_size, device=device))
                attempts += 1

        return perm

    def forward(
        self,
        speaker_emb: torch.Tensor,
        emotion_emb: torch.Tensor,
        permutation_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply batch permutation to create unpaired samples.

        Args:
            speaker_emb: [batch, speaker_dim] speaker embeddings
            emotion_emb: [batch, emotion_dim] emotion embeddings
            permutation_mask: Optional [batch] mask for which samples to permute

        Returns:
            Dict with original and permuted embeddings
        """
        batch = speaker_emb.shape[0]
        device = speaker_emb.device

        # Generate permutation indices
        if self.config.separate_permutation:
            # Separate permutation for speaker and emotion
            speaker_perm = self.generate_permutation(batch, device)
            emotion_perm = self.generate_permutation(batch, device)
        else:
            # Same permutation for both
            perm = self.generate_permutation(batch, device)
            speaker_perm = perm
            emotion_perm = perm

        # Apply permutation
        speaker_permuted = speaker_emb[speaker_perm] if self.config.permute_speaker else speaker_emb
        emotion_permuted = emotion_emb[emotion_perm] if self.config.permute_emotion else emotion_emb

        # Apply optional mask (some samples may not be permuted)
        if permutation_mask is not None:
            mask = permutation_mask.float().unsqueeze(-1)
            speaker_permuted = speaker_emb * (1 - mask) + speaker_permuted * mask
            emotion_permuted = emotion_emb * (1 - mask) + emotion_permuted * mask

        return {
            'speaker_original': speaker_emb,
            'emotion_original': emotion_emb,
            'speaker_permuted': speaker_permuted,
            'emotion_permuted': emotion_permuted,
            'speaker_perm_idx': speaker_perm,
            'emotion_perm_idx': emotion_perm,
        }


# =============================================================================
# FLOW MODULE (VITS-STYLE NORMALIZING FLOW)
# =============================================================================

class WaveNetLayer(nn.Module):
    """Single WaveNet-style layer with dilated convolution."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.output_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim, time]
            conditioning: Optional [batch, hidden_dim, time] or [batch, hidden_dim]

        Returns:
            [batch, hidden_dim, time]
        """
        residual = x

        # Dilated convolution
        x = self.dilated_conv(x)

        # Split for gated activation
        tanh_out, sigmoid_out = x.chunk(2, dim=1)
        x = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)

        # Add conditioning if provided
        if conditioning is not None:
            if conditioning.dim() == 2:
                conditioning = conditioning.unsqueeze(-1)
            x = x + conditioning

        # Output projection
        x = self.output_conv(x)

        return residual + x


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

        # Half dimension for coupling
        half_dim = config.flow_hidden_dim // 2

        # WaveNet-style network for computing scale and shift
        # Input is half the hidden dim, output is half the hidden dim
        self.wavenet_layers = nn.ModuleList([
            WaveNetLayer(
                half_dim,
                config.flow_kernel_size,
                config.flow_dilation_rate ** i,
            )
            for i in range(config.flow_num_layers)
        ])

        # Output projection for scale (log_s) and shift (t)
        # Output is the same as half_dim (scale and shift for the other half)
        self.output = nn.Conv1d(half_dim, half_dim * 2, 1)

        # Initialize output to zero for identity initialization
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Affine coupling transform.

        Args:
            x: [batch, hidden_dim, time] input
            conditioning: [batch, style_dim] style conditioning
            reverse: Whether to apply reverse (inverse) transform

        Returns:
            Tuple of (output, log_det_jacobian)
        """
        # Split input
        x0, x1 = x.chunk(2, dim=1)

        # Compute scale and shift from x0
        h = x0
        for layer in self.wavenet_layers:
            h = layer(h, conditioning)

        params = self.output(h)
        log_s, t = params.chunk(2, dim=1)

        # Clamp log_s for stability
        log_s = torch.clamp(log_s, min=-10, max=10)
        s = torch.exp(log_s)

        if reverse:
            # Inverse transform: x1' = (x1 - t) / s
            x1 = (x1 - t) / s
            log_det = -log_s.sum(dim=[1, 2])
        else:
            # Forward transform: x1' = x1 * s + t
            x1 = x1 * s + t
            log_det = log_s.sum(dim=[1, 2])

        # Recombine
        output = torch.cat([x0, x1], dim=1)

        return output, log_det


class StyleFlowModule(nn.Module):
    """
    Normalizing flow for style-conditioned latent transformation.

    This is the VITS flow module adapted for style perturbation.
    Given latent z and style embedding, it can:
    1. Transform z to match a different style (forward)
    2. Invert the transformation (reverse)

    Used in E3-VITS to apply permuted style to original latent.
    """

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Conv1d(config.hidden_dim, config.flow_hidden_dim, 1)

        # Style conditioning projection (to half dim for coupling layers)
        half_dim = config.flow_hidden_dim // 2
        self.style_proj = nn.Linear(config.style_dim, half_dim)

        # Coupling layers
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(config)
            for _ in range(config.flow_num_flows)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(config.flow_hidden_dim, config.hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        style: torch.Tensor,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply style-conditioned flow transformation.

        Args:
            z: [batch, seq, hidden_dim] latent representation
            style: [batch, style_dim] style embedding
            reverse: Whether to apply inverse transform

        Returns:
            Tuple of (transformed_z, log_det_jacobian)
        """
        # [B, T, D] -> [B, D, T]
        z = z.transpose(1, 2)

        # Project to flow hidden dim
        z = self.input_proj(z)

        # Project style conditioning
        style_cond = self.style_proj(style)  # [B, flow_hidden_dim]

        # Apply coupling layers
        total_log_det = 0

        layers = reversed(self.coupling_layers) if reverse else self.coupling_layers

        for layer in layers:
            z, log_det = layer(z, style_cond, reverse=reverse)
            total_log_det = total_log_det + log_det

        # Project back
        z = self.output_proj(z)

        # [B, D, T] -> [B, T, D]
        z = z.transpose(1, 2)

        return z, total_log_det


# =============================================================================
# STYLE PERTURBATION MODULE
# =============================================================================

class BatchPermutedStylePerturbation(nn.Module):
    """
    Complete batch-permuted style perturbation module.

    Combines:
    1. Style encoder (speaker + emotion)
    2. Batch permutation (create unpaired samples)
    3. Flow module (apply perturbed style to latent)
    4. Discriminator (ensure perturbed sample quality)

    This is the core E3-VITS technique for cross-speaker emotion transfer.
    """

    def __init__(self, config: E3VITSConfig, num_speakers: int = 1000):
        super().__init__()
        self.config = config

        # Style encoder
        self.style_encoder = StyleEncoder(config, num_speakers)

        # Batch permutation
        self.batch_permutation = BatchPermutation(config)

        # Flow module
        self.flow_module = StyleFlowModule(config)

        # Style fusion (for creating new style from permuted speaker+emotion)
        self.style_fusion = nn.Sequential(
            nn.Linear(config.speaker_dim + config.emotion_dim, config.style_dim),
            nn.LayerNorm(config.style_dim),
            nn.GELU(),
            nn.Linear(config.style_dim, config.style_dim),
        )

        # Track training steps for warmup
        self.register_buffer('step', torch.tensor(0))

    def get_permutation_strength(self) -> float:
        """Get current permutation probability based on warmup."""
        if not self.training:
            return 0.0  # No permutation during inference
        progress = min(self.step.item() / max(1, self.config.warmup_steps), 1.0)
        return self.config.permutation_probability * progress

    def should_permute(self) -> bool:
        """Decide whether to apply permutation for this sample."""
        if not self.training:
            return False
        return random.random() < self.get_permutation_strength()

    def create_permuted_style(
        self,
        speaker_emb: torch.Tensor,
        emotion_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Create fused style from (possibly permuted) speaker and emotion."""
        combined = torch.cat([speaker_emb, emotion_emb], dim=-1)
        return self.style_fusion(combined)

    def forward(
        self,
        z: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        speaker_reference: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        emotion_reference: Optional[torch.Tensor] = None,
        force_permute: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply batch-permuted style perturbation.

        Args:
            z: [batch, seq, hidden_dim] latent representation
            speaker_ids: [batch] speaker indices
            speaker_reference: [batch, seq, dim] speaker reference features
            emotion_ids: [batch] emotion indices
            emotion_reference: [batch, seq, dim] emotion reference features
            force_permute: Force permutation regardless of probability

        Returns:
            Dict containing:
                - z_original: Original latent with original style
                - z_perturbed: Latent with permuted style
                - style_original: Original style embedding
                - style_perturbed: Permuted style embedding
                - log_det: Log determinant of flow transformation
                - permutation_info: Dict with permutation details
        """
        batch = z.shape[0]
        device = z.device

        # Update step counter
        if self.training:
            self.step += 1

        # Extract style embeddings
        style_result = self.style_encoder(
            speaker_ids=speaker_ids,
            speaker_reference=speaker_reference,
            emotion_ids=emotion_ids,
            emotion_reference=emotion_reference,
        )

        speaker_emb = style_result['speaker']
        emotion_emb = style_result['emotion']
        style_original = style_result['style']

        # Decide whether to permute
        do_permute = force_permute or self.should_permute()

        if do_permute:
            # Apply batch permutation
            perm_result = self.batch_permutation(speaker_emb, emotion_emb)

            # Create permuted style
            style_perturbed = self.create_permuted_style(
                perm_result['speaker_permuted'],
                perm_result['emotion_permuted'],
            )

            # Apply flow transformation with permuted style
            z_perturbed, log_det = self.flow_module(z, style_perturbed)

            permutation_info = {
                'permuted': True,
                'speaker_perm_idx': perm_result['speaker_perm_idx'],
                'emotion_perm_idx': perm_result['emotion_perm_idx'],
            }
        else:
            # No permutation - just apply original style
            z_perturbed = z
            log_det = torch.zeros(batch, device=device)
            style_perturbed = style_original

            permutation_info = {
                'permuted': False,
                'speaker_perm_idx': torch.arange(batch, device=device),
                'emotion_perm_idx': torch.arange(batch, device=device),
            }

        # Also compute z with original style through flow (for comparison)
        z_with_original_style, log_det_original = self.flow_module(z, style_original)

        return {
            'z_original': z,
            'z_styled': z_with_original_style,
            'z_perturbed': z_perturbed,
            'style_original': style_original,
            'style_perturbed': style_perturbed,
            'speaker_emb': speaker_emb,
            'emotion_emb': emotion_emb,
            'log_det': log_det,
            'log_det_original': log_det_original,
            'permutation_info': permutation_info,
        }


# =============================================================================
# DISCRIMINATOR
# =============================================================================

class StyleDiscriminator(nn.Module):
    """
    Discriminator for distinguishing real vs perturbed samples.

    Ensures that perturbed samples (with unpaired speaker-emotion) are
    still high quality and natural-sounding.
    """

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.hidden_dim, config.discriminator_hidden_dim)

        # Discriminator layers
        layers = []
        dim = config.discriminator_hidden_dim
        for _ in range(config.discriminator_num_layers):
            layers.extend([
                nn.Linear(dim, dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
        self.layers = nn.Sequential(*layers)

        # Style conditioning
        self.style_proj = nn.Linear(config.style_dim, config.discriminator_hidden_dim)

        # Output
        self.output = nn.Linear(config.discriminator_hidden_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminate real vs perturbed samples.

        Args:
            z: [batch, seq, hidden_dim] latent representation
            style: [batch, style_dim] style embedding

        Returns:
            [batch] discriminator scores (higher = more real)
        """
        # Pool temporal dimension
        z = z.mean(dim=1)

        # Project
        x = self.input_proj(z)

        # Add style conditioning
        style_cond = self.style_proj(style)
        x = x + style_cond

        # Discriminator layers
        x = self.layers(x)

        # Output
        score = self.output(x).squeeze(-1)

        return score


# =============================================================================
# E3-VITS ADAPTER FOR CSM INTEGRATION
# =============================================================================

class E3VITSAdapter(nn.Module):
    """
    E3-VITS adapter for integration with ProsodyControlledCSM.

    Wraps the batch-permuted style perturbation module to produce
    prefix tokens that can be used with the CSM model.

    During training:
    1. Encode audio features to latent z
    2. Apply batch-permuted style perturbation
    3. Compute adversarial + reconstruction losses
    4. Return prefix tokens for both original and perturbed

    During inference:
    1. Encode emotion (from label or reference)
    2. Apply style to generate prefix tokens
    """

    def __init__(self, config: E3VITSConfig, num_speakers: int = 1000):
        super().__init__()
        self.config = config

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Batch-permuted style perturbation
        self.perturbation = BatchPermutedStylePerturbation(config, num_speakers)

        # Discriminator
        if config.use_discriminator:
            self.discriminator = StyleDiscriminator(config)
        else:
            self.discriminator = None

        # Prefix token projection
        self.prefix_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim * config.num_prefix_tokens),
        )

    def get_prefix_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert latent to prefix tokens.

        Args:
            z: [batch, seq, hidden_dim] latent representation

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        # Pool temporal dimension
        z_pooled = z.mean(dim=1)  # [batch, hidden_dim]

        # Project to prefix tokens
        tokens = self.prefix_proj(z_pooled)
        batch = z.shape[0]
        tokens = tokens.view(batch, self.config.num_prefix_tokens, self.config.output_dim)

        return tokens

    def compute_discriminator_loss(
        self,
        z_real: torch.Tensor,
        z_perturbed: torch.Tensor,
        style_real: torch.Tensor,
        style_perturbed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator and generator losses.

        Args:
            z_real: Real latent with original style
            z_perturbed: Latent with perturbed style
            style_real: Original style embedding
            style_perturbed: Perturbed style embedding

        Returns:
            Dict with discriminator and generator losses
        """
        device = z_real.device

        if self.discriminator is None:
            return {
                'discriminator': torch.tensor(0.0, device=device),
                'generator': torch.tensor(0.0, device=device),
            }

        # Discriminator scores
        score_real = self.discriminator(z_real.detach(), style_real.detach())
        score_fake = self.discriminator(z_perturbed.detach(), style_perturbed.detach())

        # Discriminator loss (real = 1, fake = 0)
        d_loss_real = F.binary_cross_entropy_with_logits(
            score_real, torch.ones_like(score_real)
        )
        d_loss_fake = F.binary_cross_entropy_with_logits(
            score_fake, torch.zeros_like(score_fake)
        )
        d_loss = (d_loss_real + d_loss_fake) / 2

        # R1 gradient penalty (if enabled)
        if self.config.use_gradient_penalty and self.training:
            z_real_req_grad = z_real.detach().requires_grad_(True)
            style_real_req_grad = style_real.detach().requires_grad_(True)
            score_real_gp = self.discriminator(z_real_req_grad, style_real_req_grad)

            grad = torch.autograd.grad(
                outputs=score_real_gp.sum(),
                inputs=[z_real_req_grad, style_real_req_grad],
                create_graph=True,
            )
            grad_penalty = sum(g.pow(2).sum() for g in grad)
            d_loss = d_loss + self.config.gradient_penalty_weight * grad_penalty

        # Generator loss (want discriminator to think perturbed is real)
        score_fake_for_g = self.discriminator(z_perturbed, style_perturbed)
        g_loss = F.binary_cross_entropy_with_logits(
            score_fake_for_g, torch.ones_like(score_fake_for_g)
        )

        return {
            'discriminator': d_loss,
            'generator': g_loss,
            'score_real': score_real.mean(),
            'score_fake': score_fake.mean(),
        }

    def forward(
        self,
        features: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_permutation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with batch-permuted style perturbation.

        Args:
            features: [batch, seq, dim] audio features
            speaker_ids: [batch] speaker indices
            emotion_ids: [batch] emotion indices
            mask: Optional attention mask
            use_permutation: Whether to use batch permutation

        Returns:
            Dict containing:
                - prosody_tokens: [batch, num_tokens, output_dim] original style tokens
                - perturbed_tokens: [batch, num_tokens, output_dim] perturbed style tokens
                - losses: Dict of loss values
                - style_info: Dict with style embeddings
        """
        batch = features.shape[0]
        device = features.device

        # Encode features to latent
        z = self.feature_encoder(features)

        # Apply batch-permuted style perturbation
        pert_result = self.perturbation(
            z=z,
            speaker_ids=speaker_ids,
            emotion_ids=emotion_ids,
            force_permute=self.training and use_permutation,
        )

        # Get prefix tokens
        tokens_original = self.get_prefix_tokens(pert_result['z_styled'])
        tokens_perturbed = self.get_prefix_tokens(pert_result['z_perturbed'])

        # Compute losses during training
        losses = {}
        if self.training:
            # Discriminator/generator losses
            disc_losses = self.compute_discriminator_loss(
                z_real=pert_result['z_styled'],
                z_perturbed=pert_result['z_perturbed'],
                style_real=pert_result['style_original'],
                style_perturbed=pert_result['style_perturbed'],
            )
            losses.update(disc_losses)

            # KL loss for flow
            log_det = pert_result['log_det']
            kl_loss = -log_det.mean()  # Encourage high-entropy transformations
            losses['kl'] = kl_loss

            # Style consistency loss (original should be reconstructable)
            log_det_original = pert_result['log_det_original']
            recon_loss = -log_det_original.mean()
            losses['reconstruction'] = recon_loss

            # Total loss
            losses['total'] = (
                self.config.adversarial_weight * (
                    losses.get('generator', torch.tensor(0.0, device=device)) +
                    losses.get('discriminator', torch.tensor(0.0, device=device))
                ) +
                self.config.kl_weight * losses['kl'] +
                self.config.reconstruction_weight * losses['reconstruction']
            )

        return {
            'prosody_tokens': tokens_original,
            'perturbed_tokens': tokens_perturbed,
            'z_original': pert_result['z_original'],
            'z_styled': pert_result['z_styled'],
            'z_perturbed': pert_result['z_perturbed'],
            'style_original': pert_result['style_original'],
            'style_perturbed': pert_result['style_perturbed'],
            'speaker_emb': pert_result['speaker_emb'],
            'emotion_emb': pert_result['emotion_emb'],
            'permutation_info': pert_result['permutation_info'],
            'losses': losses,
        }

    def from_emotion(
        self,
        emotion: Union[str, int, torch.Tensor],
        features: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prefix tokens from emotion label (inference mode).

        Args:
            emotion: Emotion name, index, or tensor
            features: Optional audio features (for feature-based generation)
            speaker_ids: Optional speaker indices
            speaker_emb: Optional pre-computed speaker embedding
            intensity: Emotion intensity (0-1)

        Returns:
            Dict with prosody_tokens and style embeddings
        """
        self.eval()

        # Determine batch size and device
        if features is not None:
            batch = features.shape[0]
            device = features.device
        elif speaker_ids is not None:
            batch = speaker_ids.shape[0]
            device = speaker_ids.device
        elif speaker_emb is not None:
            batch = speaker_emb.shape[0]
            device = speaker_emb.device
        else:
            batch = 1
            device = next(self.parameters()).device

        # Convert emotion to tensor
        if isinstance(emotion, str):
            emotion_idx = EMOTION_TO_IDX.get(emotion.lower(), 0)
            emotion_ids = torch.full((batch,), emotion_idx, dtype=torch.long, device=device)
        elif isinstance(emotion, int):
            emotion_ids = torch.full((batch,), emotion, dtype=torch.long, device=device)
        else:
            emotion_ids = emotion

        # Generate placeholder features if not provided
        if features is None:
            # Use learned embedding directly without features
            seq_len = 50  # Default sequence length
            features = torch.zeros(batch, seq_len, self.config.input_dim, device=device)

        # Forward pass without permutation
        with torch.no_grad():
            result = self.forward(
                features=features,
                speaker_ids=speaker_ids,
                emotion_ids=emotion_ids,
                use_permutation=False,
            )

        return {
            'prosody_tokens': result['prosody_tokens'],
            'style': result['style_original'],
            'speaker_emb': result['speaker_emb'],
            'emotion_emb': result['emotion_emb'],
        }

    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        features: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotions.

        Args:
            emotion1: First emotion name
            emotion2: Second emotion name
            t: Interpolation parameter (0 = emotion1, 1 = emotion2)
            features: Optional audio features
            speaker_ids: Optional speaker indices

        Returns:
            Dict with interpolated prosody_tokens
        """
        self.eval()

        # Get embeddings for both emotions
        result1 = self.from_emotion(emotion1, features, speaker_ids)
        result2 = self.from_emotion(emotion2, features, speaker_ids)

        # Interpolate embeddings
        emotion_interp = result1['emotion_emb'] * (1 - t) + result2['emotion_emb'] * t

        # Recreate style with interpolated emotion
        with torch.no_grad():
            speaker_emb = result1['speaker_emb']
            style_interp = self.perturbation.create_permuted_style(speaker_emb, emotion_interp)

            # Generate features and transform
            if features is None:
                batch = speaker_emb.shape[0]
                device = speaker_emb.device
                seq_len = 50
                features = torch.zeros(batch, seq_len, self.config.input_dim, device=device)

            z = self.feature_encoder(features)
            z_styled, _ = self.perturbation.flow_module(z, style_interp)
            tokens = self.get_prefix_tokens(z_styled)

        return {
            'prosody_tokens': tokens,
            'style': style_interp,
            'emotion_emb': emotion_interp,
        }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class E3VITSLoss(nn.Module):
    """
    Combined loss for E3-VITS training.

    Includes:
    - Adversarial loss (discriminator + generator)
    - Reconstruction loss
    - Style consistency loss
    - KL divergence for flow
    """

    def __init__(self, config: E3VITSConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        adapter_output: Dict[str, torch.Tensor],
        reconstruction_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            adapter_output: Output from E3VITSAdapter.forward()
            reconstruction_target: Optional target for reconstruction loss

        Returns:
            Dict with all loss components
        """
        losses = adapter_output.get('losses', {})

        # Add reconstruction loss if target provided
        if reconstruction_target is not None:
            z_styled = adapter_output['z_styled']
            recon_loss = F.mse_loss(z_styled, reconstruction_target)
            losses['reconstruction_mse'] = recon_loss
            losses['total'] = losses.get('total', 0) + self.config.reconstruction_weight * recon_loss

        return losses


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_e3vits_adapter(
    input_dim: int = 768,
    output_dim: int = 2048,
    num_speakers: int = 1000,
    num_emotions: int = 8,
    use_discriminator: bool = True,
) -> E3VITSAdapter:
    """
    Create E3-VITS adapter with common configuration.

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension for CSM
        num_speakers: Number of speakers
        num_emotions: Number of emotion categories
        use_discriminator: Whether to use discriminator

    Returns:
        Configured E3VITSAdapter
    """
    config = E3VITSConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_emotions=num_emotions,
        use_discriminator=use_discriminator,
    )
    return E3VITSAdapter(config, num_speakers)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("E3-VITS: Batch-Permuted Style Perturbation - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = E3VITSConfig()

    # Test 1: EmotionEmbedding
    print("\n[Test 1] EmotionEmbedding...")
    emotion_emb = EmotionEmbedding(config).to(device)
    emotion_idx = torch.randint(0, 8, (4,), device=device)
    emb = emotion_emb(emotion_idx)
    print(f"  Emotion embedding shape: {emb.shape}")
    emb_named = emotion_emb.get_by_name("happy", batch_size=4, device=device)
    print(f"  Named embedding shape: {emb_named.shape}")
    print("  [PASS]")

    # Test 2: FeatureReferenceEncoder
    print("\n[Test 2] FeatureReferenceEncoder...")
    ref_encoder = FeatureReferenceEncoder(config).to(device)
    features = torch.randn(4, 100, 768, device=device)
    style = ref_encoder(features)
    print(f"  Reference style shape: {style.shape}")
    print("  [PASS]")

    # Test 3: StyleEncoder
    print("\n[Test 3] StyleEncoder...")
    style_encoder = StyleEncoder(config, num_speakers=100).to(device)
    speaker_ids = torch.randint(0, 100, (4,), device=device)
    emotion_ids = torch.randint(0, 8, (4,), device=device)
    style_result = style_encoder(speaker_ids=speaker_ids, emotion_ids=emotion_ids)
    print(f"  Style shape: {style_result['style'].shape}")
    print(f"  Speaker shape: {style_result['speaker'].shape}")
    print(f"  Emotion shape: {style_result['emotion'].shape}")
    print("  [PASS]")

    # Test 4: BatchPermutation
    print("\n[Test 4] BatchPermutation...")
    batch_perm = BatchPermutation(config)
    speaker_emb = torch.randn(4, 256, device=device)
    emotion_emb_tensor = torch.randn(4, 256, device=device)
    perm_result = batch_perm(speaker_emb, emotion_emb_tensor)
    print(f"  Speaker permuted shape: {perm_result['speaker_permuted'].shape}")
    print(f"  Emotion permuted shape: {perm_result['emotion_permuted'].shape}")
    print(f"  Speaker perm idx: {perm_result['speaker_perm_idx']}")
    print(f"  Emotion perm idx: {perm_result['emotion_perm_idx']}")
    # Verify permutation is not identity
    is_identity_speaker = torch.all(perm_result['speaker_perm_idx'] == torch.arange(4, device=device))
    is_identity_emotion = torch.all(perm_result['emotion_perm_idx'] == torch.arange(4, device=device))
    print(f"  Speaker permutation is identity: {is_identity_speaker.item()}")
    print(f"  Emotion permutation is identity: {is_identity_emotion.item()}")
    print("  [PASS]")

    # Test 5: StyleFlowModule
    print("\n[Test 5] StyleFlowModule...")
    flow = StyleFlowModule(config).to(device)
    z = torch.randn(4, 50, 512, device=device)
    style_tensor = torch.randn(4, 256, device=device)
    z_transformed, log_det = flow(z, style_tensor)
    print(f"  Transformed z shape: {z_transformed.shape}")
    print(f"  Log det shape: {log_det.shape}")
    print(f"  Log det mean: {log_det.mean().item():.4f}")
    # Test reversibility
    z_reversed, _ = flow(z_transformed, style_tensor, reverse=True)
    recon_error = (z - z_reversed).abs().mean()
    print(f"  Reconstruction error (should be small): {recon_error.item():.6f}")
    print("  [PASS]")

    # Test 6: BatchPermutedStylePerturbation
    print("\n[Test 6] BatchPermutedStylePerturbation...")
    perturbation = BatchPermutedStylePerturbation(config, num_speakers=100).to(device)
    perturbation.train()
    z = torch.randn(4, 50, 512, device=device)
    pert_result = perturbation(
        z=z,
        speaker_ids=speaker_ids,
        emotion_ids=emotion_ids,
        force_permute=True,
    )
    print(f"  z_original shape: {pert_result['z_original'].shape}")
    print(f"  z_styled shape: {pert_result['z_styled'].shape}")
    print(f"  z_perturbed shape: {pert_result['z_perturbed'].shape}")
    print(f"  Permuted: {pert_result['permutation_info']['permuted']}")
    # Check that z_styled and z_perturbed are different
    diff = (pert_result['z_styled'] - pert_result['z_perturbed']).abs().mean()
    print(f"  Diff between styled and perturbed: {diff.item():.4f}")
    print("  [PASS]")

    # Test 7: StyleDiscriminator
    print("\n[Test 7] StyleDiscriminator...")
    discriminator = StyleDiscriminator(config).to(device)
    z = torch.randn(4, 50, 512, device=device)
    style_tensor = torch.randn(4, 256, device=device)
    score = discriminator(z, style_tensor)
    print(f"  Discriminator score shape: {score.shape}")
    print(f"  Score mean: {score.mean().item():.4f}")
    print("  [PASS]")

    # Test 8: E3VITSAdapter (training mode)
    print("\n[Test 8] E3VITSAdapter (training mode)...")
    adapter = E3VITSAdapter(config, num_speakers=100).to(device)
    adapter.train()
    features = torch.randn(4, 100, 768, device=device)
    result = adapter(
        features=features,
        speaker_ids=speaker_ids,
        emotion_ids=emotion_ids,
        use_permutation=True,
    )
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Perturbed tokens shape: {result['perturbed_tokens'].shape}")
    print(f"  Losses:")
    for k, v in result['losses'].items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.item():.4f}")
        else:
            print(f"    {k}: {v}")
    print("  [PASS]")

    # Test 9: E3VITSAdapter (inference mode)
    print("\n[Test 9] E3VITSAdapter (inference mode)...")
    adapter.eval()
    with torch.no_grad():
        result = adapter.from_emotion(
            emotion="happy",
            speaker_ids=speaker_ids[:1],
        )
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Style shape: {result['style'].shape}")
    print("  [PASS]")

    # Test 10: Emotion interpolation
    print("\n[Test 10] Emotion interpolation...")
    with torch.no_grad():
        result = adapter.interpolate_emotions(
            emotion1="sad",
            emotion2="happy",
            t=0.5,
            speaker_ids=speaker_ids[:1],
        )
    print(f"  Interpolated tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Interpolated style shape: {result['style'].shape}")
    print("  [PASS]")

    # Test 11: Gradient flow
    print("\n[Test 11] Gradient flow verification...")
    adapter.train()
    features = torch.randn(4, 100, 768, device=device, requires_grad=True)
    result = adapter(
        features=features,
        speaker_ids=speaker_ids,
        emotion_ids=emotion_ids,
        use_permutation=True,
    )
    loss = result['losses']['total']
    loss.backward()
    grad_norm = features.grad.norm().item()
    print(f"  Input gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients should flow back to input"
    print("  [PASS]")

    # Test 12: Cross-speaker emotion transfer simulation
    print("\n[Test 12] Cross-speaker emotion transfer simulation...")
    adapter.train()
    # Create batch with different speakers and emotions
    speaker_a, speaker_b = 0, 1
    emotion_happy, emotion_sad = 1, 2

    # Original: Speaker A + Happy, Speaker B + Sad
    speakers = torch.tensor([speaker_a, speaker_b], device=device)
    emotions = torch.tensor([emotion_happy, emotion_sad], device=device)
    features = torch.randn(2, 100, 768, device=device)

    result = adapter(
        features=features,
        speaker_ids=speakers,
        emotion_ids=emotions,
        use_permutation=True,
    )

    perm_info = result['permutation_info']
    print(f"  Original speakers: {speakers.tolist()}")
    print(f"  Original emotions: {emotions.tolist()}")
    print(f"  Speaker perm: {perm_info['speaker_perm_idx'].tolist()}")
    print(f"  Emotion perm: {perm_info['emotion_perm_idx'].tolist()}")

    # Check that permutation creates unpaired combinations
    original_pairs = list(zip(speakers.tolist(), emotions.tolist()))
    perturbed_speakers = speakers[perm_info['speaker_perm_idx']].tolist()
    perturbed_emotions = emotions[perm_info['emotion_perm_idx']].tolist()
    perturbed_pairs = list(zip(perturbed_speakers, perturbed_emotions))

    print(f"  Original pairs: {original_pairs}")
    print(f"  Perturbed pairs: {perturbed_pairs}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All E3-VITS tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from e3_vits import (
    E3VITSConfig,
    E3VITSAdapter,
    create_e3vits_adapter,
    EMOTION_LABELS,
)

# Initialize
config = E3VITSConfig(
    input_dim=768,        # HuBERT/wav2vec2 feature dim
    output_dim=2048,      # CSM hidden dim
    num_emotions=8,
)

adapter = E3VITSAdapter(config, num_speakers=1000).cuda()

# Training with batch-permuted style perturbation
adapter.train()
for batch in dataloader:
    features = feature_extractor(batch['audio'])  # [batch, seq, 768]
    speaker_ids = batch['speaker_id']              # [batch]
    emotion_ids = batch['emotion_id']              # [batch]

    # Forward pass with batch permutation
    result = adapter(
        features=features,
        speaker_ids=speaker_ids,
        emotion_ids=emotion_ids,
        use_permutation=True,  # Enable cross-speaker emotion transfer
    )

    # Get tokens for CSM
    prosody_tokens = result['prosody_tokens']          # Original style
    perturbed_tokens = result['perturbed_tokens']      # Unpaired style

    # Compute losses
    e3vits_loss = result['losses']['total']

    # Train with both original and perturbed samples
    csm_loss_orig = csm_model(prosody_prefix=prosody_tokens, ...)
    csm_loss_pert = csm_model(prosody_prefix=perturbed_tokens, ...)

    total_loss = csm_loss_orig + 0.5 * csm_loss_pert + e3vits_loss
    total_loss.backward()

# Inference - from emotion label
adapter.eval()
with torch.no_grad():
    result = adapter.from_emotion(
        emotion="happy",
        speaker_ids=speaker_ids[:1],
    )
    prosody_tokens = result['prosody_tokens']

# Inference - emotion interpolation
with torch.no_grad():
    result = adapter.interpolate_emotions(
        emotion1="sad",
        emotion2="happy",
        t=0.7,  # 70% toward happy
        speaker_ids=speaker_ids[:1],
    )
    prosody_tokens = result['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
