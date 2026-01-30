"""
ProMode: A Stand-Alone Speech Prosody Model for TTS

Based on "ProMode: A Speech Prosody Model Conditioned on Acoustic and Textual Inputs"
(Interspeech 2025) - https://arxiv.org/abs/2508.09389

Key Innovation: Stand-alone prosody model that maps text to prosodic features (F0, energy)
independent of the TTS model. Can be integrated into any downstream TTS system.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ProMode Architecture                                    │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                           ENCODER (Perceiver IO)                                 ││
│  │                                                                                  ││
│  │  Acoustic Features ──[Mask]──┐                                                   ││
│  │                              ├──► Cross-Attention ──► Latent Prosody Embedding  ││
│  │  Text Features ──────[Mask]──┘        ▲                                         ││
│  │                                       │                                          ││
│  │                              Latent Queries (Fixed Length)                       ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                           DUAL DECODERS                                          ││
│  │                                                                                  ││
│  │  Decoder 1 (with text):    Latent + Unmasked Text ──► Predict Masked Acoustics  ││
│  │  Decoder 2 (without text): Latent Only ──► Predict Masked Acoustics             ││
│  │                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐│
│  │                           PROSODY HEADS                                          ││
│  │                                                                                  ││
│  │  F0 Predictor ────────► Predict F0 contour at multiple granularities            ││
│  │  Energy Predictor ────► Predict energy envelope                                  ││
│  │  Duration Predictor ──► Predict phoneme/word durations                          ││
│  │                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────┘

Key Features:
1. Modular prosody prediction - can be swapped into different TTS systems
2. Superior F0 and energy prediction vs baselines on GigaSpeech
3. Improved prosody naturalness in downstream TTS (e.g., FluentSpeech)
4. Can be trained independently on large prosody datasets

Differs from PE-wav2vec (#35) which focuses on LPC residual supervision.
ProMode is a complete stand-alone prosody prediction module with dual decoders
and multi-granularity prediction heads.

Usage:
    from promode import (
        ProModeConfig,
        ProMode,
        ProModeAdapter,
        create_promode_adapter,
    )

    # Initialize
    config = ProModeConfig()
    model = ProMode(config).cuda()

    # Training: masked prediction of acoustics
    losses = model.compute_loss(
        acoustic_features=mel,
        text_features=text_emb,
        f0_target=f0,
        energy_target=energy,
    )

    # Inference: text-to-prosody
    prosody = model.predict_prosody(text_features=text_emb)
    f0_pred = prosody['f0']
    energy_pred = prosody['energy']

    # CSM integration
    adapter = ProModeAdapter(config, model)
    prefix_tokens = adapter(text_emb)  # [batch, 4, 2048]
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProModeConfig:
    """Configuration for ProMode stand-alone prosody model."""

    # Input dimensions
    acoustic_dim: int = 80  # Mel spectrogram channels
    text_dim: int = 512  # Text encoder output dimension (e.g., BERT)

    # Latent prosody embedding
    latent_dim: int = 256  # Dimension of latent prosody queries
    num_latent_queries: int = 32  # Number of latent queries (fixed length output)

    # Perceiver IO encoder
    encoder_num_layers: int = 6
    encoder_num_heads: int = 8
    encoder_ffn_dim: int = 1024
    encoder_dropout: float = 0.1

    # Cross-attention settings
    num_cross_attn_layers: int = 4  # Cross-attention iterations
    cross_attn_heads: int = 8

    # Decoder settings (dual decoders)
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 1024
    decoder_dropout: float = 0.1

    # Masking settings
    acoustic_mask_ratio: float = 0.5  # Mask 50% of acoustic features
    text_mask_ratio: float = 0.3  # Mask 30% of text features
    mask_patch_size: int = 4  # Mask entire patches for temporal coherence

    # Prosody prediction heads
    f0_dim: int = 1  # F0 is 1D per frame
    energy_dim: int = 1  # Energy is 1D per frame
    duration_dim: int = 1  # Duration per phoneme/word

    # Multi-granularity prediction
    granularities: List[str] = field(default_factory=lambda: ['frame', 'phoneme', 'word'])

    # Loss weights
    reconstruction_weight: float = 1.0
    f0_weight: float = 1.0
    energy_weight: float = 0.5
    duration_weight: float = 0.5
    consistency_weight: float = 0.1  # Consistency between two decoders

    # Audio settings
    sample_rate: int = 16000
    hop_length: int = 256  # ~16ms frames at 16kHz
    f0_min: float = 50.0
    f0_max: float = 800.0

    # Training settings
    warmup_steps: int = 5000
    dropout: float = 0.1

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# MASKING UTILITIES
# =============================================================================

class PatchMasker(nn.Module):
    """
    Mask input sequences at patch level for temporal coherence.

    Masks entire contiguous patches rather than individual frames,
    which forces the model to learn longer-range dependencies.
    """

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.mask_patch_size

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim]
        mask_ratio: float,
        return_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply patch-level masking.

        Args:
            x: Input features
            mask_ratio: Fraction of patches to mask
            return_mask: Whether to return the mask

        Returns:
            Masked features and optional mask tensor
        """
        batch_size, seq_len, dim = x.shape
        device = x.device

        # Calculate number of patches
        num_patches = seq_len // self.patch_size
        num_mask = int(num_patches * mask_ratio)

        # Generate random patch indices to mask
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_mask = ids_shuffle[:, :num_mask]

        # Create patch-level mask
        patch_mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_mask)
        patch_mask[batch_indices, ids_mask] = True

        # Expand to frame-level mask
        frame_mask = patch_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        frame_mask = frame_mask.view(batch_size, num_patches * self.patch_size)

        # Pad if necessary
        if seq_len > num_patches * self.patch_size:
            padding = torch.zeros(batch_size, seq_len - num_patches * self.patch_size,
                                 device=device, dtype=torch.bool)
            frame_mask = torch.cat([frame_mask, padding], dim=1)

        # Apply mask (set masked positions to zero)
        x_masked = x.clone()
        x_masked[frame_mask.unsqueeze(-1).expand_as(x)] = 0

        if return_mask:
            return x_masked, frame_mask
        return x_masked, None


# =============================================================================
# PERCEIVER IO ENCODER
# =============================================================================

class PerceiverCrossAttention(nn.Module):
    """
    Cross-attention layer for Perceiver IO.

    Queries attend to both acoustic and text inputs to learn
    joint prosodic representations.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(key_dim)

    def forward(
        self,
        query: torch.Tensor,  # [batch, num_queries, query_dim]
        key_value: torch.Tensor,  # [batch, seq, key_dim]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from latent queries to input features.
        """
        batch_size, num_queries, _ = query.shape
        _, seq_len, _ = key_value.shape

        # Normalize
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)

        # Project
        q = self.q_proj(query)  # [batch, num_queries, dim]
        k = self.k_proj(key_value)  # [batch, seq, dim]
        v = self.v_proj(key_value)  # [batch, seq, dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply padding mask if provided
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, heads, queries, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, num_queries, -1)

        return self.out_proj(out)


class PerceiverSelfAttention(nn.Module):
    """Self-attention layer for processing latent queries."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention on latent queries."""
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        return residual + x


class PerceiverFFN(nn.Module):
    """Feed-forward network for Perceiver."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class ProModeEncoder(nn.Module):
    """
    ProMode Encoder using Perceiver IO architecture.

    Takes partially masked acoustic and text features as input,
    uses cross-attention to produce fixed-length latent prosody embedding.
    """

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config

        # Learnable latent queries (fixed length output)
        self.latent_queries = nn.Parameter(
            torch.randn(1, config.num_latent_queries, config.latent_dim)
        )

        # Input projections
        self.acoustic_proj = nn.Sequential(
            nn.Linear(config.acoustic_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

        # Positional encoding for inputs
        self.acoustic_pos_enc = nn.Parameter(
            torch.randn(1, 2048, config.latent_dim) * 0.02
        )
        self.text_pos_enc = nn.Parameter(
            torch.randn(1, 512, config.latent_dim) * 0.02
        )

        # Cross-attention layers (latent queries attend to inputs)
        self.cross_attn_layers = nn.ModuleList([
            PerceiverCrossAttention(
                query_dim=config.latent_dim,
                key_dim=config.latent_dim,
                num_heads=config.cross_attn_heads,
                dropout=config.encoder_dropout,
            )
            for _ in range(config.num_cross_attn_layers)
        ])

        # Self-attention layers (process latent queries)
        self.self_attn_layers = nn.ModuleList([
            nn.Sequential(
                PerceiverSelfAttention(
                    dim=config.latent_dim,
                    num_heads=config.encoder_num_heads,
                    dropout=config.encoder_dropout,
                ),
                PerceiverFFN(
                    dim=config.latent_dim,
                    ffn_dim=config.encoder_ffn_dim,
                    dropout=config.encoder_dropout,
                ),
            )
            for _ in range(config.encoder_num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.latent_dim)

        # Masker
        self.masker = PatchMasker(config)

    def forward(
        self,
        acoustic_features: torch.Tensor,  # [batch, acoustic_seq, acoustic_dim]
        text_features: torch.Tensor,  # [batch, text_seq, text_dim]
        apply_masking: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode acoustic and text features to fixed-length latent prosody.

        Args:
            acoustic_features: Mel spectrogram or similar acoustic features
            text_features: Text encoder output (e.g., BERT embeddings)
            apply_masking: Whether to apply random masking during training

        Returns:
            Dict with:
                - latent_prosody: [batch, num_queries, latent_dim]
                - acoustic_masked: Masked acoustic features
                - text_masked: Masked text features
                - acoustic_mask: Mask applied to acoustics
                - text_mask: Mask applied to text
        """
        batch_size = acoustic_features.shape[0]
        acoustic_seq_len = acoustic_features.shape[1]
        text_seq_len = text_features.shape[1]
        device = acoustic_features.device

        # Apply masking if training
        if apply_masking:
            acoustic_masked, acoustic_mask = self.masker(
                acoustic_features,
                self.config.acoustic_mask_ratio,
            )
            text_masked, text_mask = self.masker(
                text_features,
                self.config.text_mask_ratio,
            )
        else:
            acoustic_masked = acoustic_features
            text_masked = text_features
            acoustic_mask = None
            text_mask = None

        # Project inputs to latent dimension
        acoustic_proj = self.acoustic_proj(acoustic_masked)  # [batch, acoustic_seq, latent_dim]
        text_proj = self.text_proj(text_masked)  # [batch, text_seq, latent_dim]

        # Add positional encoding
        acoustic_proj = acoustic_proj + self.acoustic_pos_enc[:, :acoustic_seq_len, :]
        text_proj = text_proj + self.text_pos_enc[:, :text_seq_len, :]

        # Concatenate acoustic and text as key-value pairs
        kv_input = torch.cat([acoustic_proj, text_proj], dim=1)

        # Initialize latent queries
        latent = self.latent_queries.expand(batch_size, -1, -1)

        # Alternating cross-attention and self-attention
        for i, (cross_attn, self_attn) in enumerate(
            zip(self.cross_attn_layers, self.self_attn_layers)
        ):
            # Cross-attention: latent queries attend to inputs
            latent = latent + cross_attn(latent, kv_input)

            # Self-attention: process latent queries
            latent = self_attn(latent)

        # Additional self-attention layers if encoder has more
        for i in range(len(self.cross_attn_layers), len(self.self_attn_layers)):
            latent = self.self_attn_layers[i](latent)

        # Final normalization
        latent = self.final_norm(latent)

        return {
            'latent_prosody': latent,
            'acoustic_masked': acoustic_masked,
            'text_masked': text_masked,
            'acoustic_mask': acoustic_mask,
            'text_mask': text_mask,
        }


# =============================================================================
# DUAL DECODERS
# =============================================================================

class ProModeDecoder(nn.Module):
    """
    ProMode Decoder for reconstructing masked acoustic features.

    Two variants:
    - With text: Uses both latent prosody and unmasked text features
    - Without text: Uses only latent prosody (for text-only inference)
    """

    def __init__(
        self,
        config: ProModeConfig,
        use_text_condition: bool = True,
    ):
        super().__init__()
        self.config = config
        self.use_text_condition = use_text_condition

        # Input projection for latent prosody
        self.latent_proj = nn.Linear(config.latent_dim, config.latent_dim)

        # Optional text condition projection
        if use_text_condition:
            self.text_proj = nn.Linear(config.text_dim, config.latent_dim)
            self.text_cross_attn = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.decoder_num_heads,
                dropout=config.decoder_dropout,
                batch_first=True,
            )
            self.text_cross_norm = nn.LayerNorm(config.latent_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.decoder_dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.decoder_num_layers,
        )

        # Upsample from latent queries to acoustic sequence
        self.upsample = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 4),
            nn.GELU(),
            nn.Linear(config.latent_dim * 4, config.latent_dim),
        )

        # Output projection to acoustic dimension
        self.out_proj = nn.Linear(config.latent_dim, config.acoustic_dim)

    def forward(
        self,
        latent_prosody: torch.Tensor,  # [batch, num_queries, latent_dim]
        target_length: int,  # Target acoustic sequence length
        text_features: Optional[torch.Tensor] = None,  # [batch, text_seq, text_dim]
    ) -> torch.Tensor:
        """
        Decode latent prosody to acoustic features.

        Args:
            latent_prosody: Encoded prosody from encoder
            target_length: Target sequence length
            text_features: Optional text features for conditioning

        Returns:
            [batch, target_length, acoustic_dim] reconstructed acoustics
        """
        batch_size = latent_prosody.shape[0]
        num_queries = latent_prosody.shape[1]
        device = latent_prosody.device

        # Project latent prosody
        hidden = self.latent_proj(latent_prosody)

        # Optional text conditioning via cross-attention
        if self.use_text_condition and text_features is not None:
            text_proj = self.text_proj(text_features)
            hidden_norm = self.text_cross_norm(hidden)
            hidden_attn, _ = self.text_cross_attn(
                hidden_norm, text_proj, text_proj
            )
            hidden = hidden + hidden_attn

        # Process through decoder
        hidden = self.decoder(hidden)

        # Upsample to target length
        # Interpolate from num_queries to target_length
        hidden = hidden.transpose(1, 2)  # [batch, latent_dim, num_queries]
        hidden = F.interpolate(
            hidden,
            size=target_length,
            mode='linear',
            align_corners=False,
        )
        hidden = hidden.transpose(1, 2)  # [batch, target_length, latent_dim]

        # Apply upsampling network
        hidden = self.upsample(hidden)

        # Project to acoustic dimension
        output = self.out_proj(hidden)

        return output


# =============================================================================
# PROSODY PREDICTION HEADS
# =============================================================================

class F0Predictor(nn.Module):
    """
    Predicts F0 (fundamental frequency) contour from latent prosody.

    Supports multi-granularity prediction:
    - Frame-level: Detailed F0 per frame (~16ms)
    - Phoneme-level: Average F0 per phoneme
    - Word-level: Average F0 per word
    """

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config

        # Frame-level F0 prediction
        self.frame_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.GELU(),
            nn.Linear(config.latent_dim // 2, config.f0_dim),
        )

        # Voiced/unvoiced classification
        self.vuv_classifier = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.GELU(),
            nn.Linear(config.latent_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Log F0 normalization parameters
        self.f0_min = config.f0_min
        self.f0_max = config.f0_max
        self.log_f0_min = math.log(config.f0_min)
        self.log_f0_max = math.log(config.f0_max)

    def forward(
        self,
        latent_prosody: torch.Tensor,  # [batch, num_queries, latent_dim]
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict F0 contour from latent prosody.

        Returns:
            Dict with:
                - f0: [batch, target_length] predicted F0 in Hz
                - log_f0: [batch, target_length] predicted log F0
                - vuv: [batch, target_length] voiced/unvoiced probability
        """
        batch_size = latent_prosody.shape[0]
        device = latent_prosody.device

        # Upsample latent to target length
        latent_up = latent_prosody.transpose(1, 2)  # [batch, dim, queries]
        latent_up = F.interpolate(
            latent_up,
            size=target_length,
            mode='linear',
            align_corners=False,
        )
        latent_up = latent_up.transpose(1, 2)  # [batch, target_length, dim]

        # Predict log F0 (normalized to 0-1)
        log_f0_norm = self.frame_predictor(latent_up).squeeze(-1)  # [batch, target_length]
        log_f0_norm = torch.sigmoid(log_f0_norm)  # Bound to 0-1

        # Convert to log F0
        log_f0 = log_f0_norm * (self.log_f0_max - self.log_f0_min) + self.log_f0_min

        # Convert to Hz
        f0 = torch.exp(log_f0)

        # Predict voiced/unvoiced
        vuv = self.vuv_classifier(latent_up).squeeze(-1)  # [batch, target_length]

        return {
            'f0': f0,
            'log_f0': log_f0,
            'vuv': vuv,
        }


class EnergyPredictor(nn.Module):
    """
    Predicts energy envelope from latent prosody.
    """

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config

        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.GELU(),
            nn.Linear(config.latent_dim // 2, config.energy_dim),
        )

    def forward(
        self,
        latent_prosody: torch.Tensor,  # [batch, num_queries, latent_dim]
        target_length: int,
    ) -> torch.Tensor:
        """
        Predict energy envelope from latent prosody.

        Returns:
            [batch, target_length] predicted energy (log scale)
        """
        # Upsample latent to target length
        latent_up = latent_prosody.transpose(1, 2)  # [batch, dim, queries]
        latent_up = F.interpolate(
            latent_up,
            size=target_length,
            mode='linear',
            align_corners=False,
        )
        latent_up = latent_up.transpose(1, 2)  # [batch, target_length, dim]

        # Predict energy
        energy = self.predictor(latent_up).squeeze(-1)  # [batch, target_length]

        return energy


class DurationPredictor(nn.Module):
    """
    Predicts duration from latent prosody.

    Outputs duration scaling factors per phoneme/word.
    """

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config

        # Conv-based duration predictor (similar to FastSpeech)
        self.conv = nn.Sequential(
            nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.proj = nn.Linear(config.latent_dim, config.duration_dim)

    def forward(
        self,
        latent_prosody: torch.Tensor,  # [batch, num_queries, latent_dim]
        num_phonemes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict duration from latent prosody.

        Returns:
            [batch, num_phonemes or num_queries] duration scaling factors
        """
        # Apply conv
        hidden = latent_prosody.transpose(1, 2)  # [batch, dim, queries]
        hidden = self.conv(hidden)
        hidden = hidden.transpose(1, 2)  # [batch, queries, dim]

        # Predict duration
        duration = self.proj(hidden).squeeze(-1)  # [batch, queries]
        duration = F.softplus(duration)  # Ensure positive

        # Optionally interpolate to phoneme count
        if num_phonemes is not None and num_phonemes != duration.shape[1]:
            duration = F.interpolate(
                duration.unsqueeze(1),
                size=num_phonemes,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        return duration


# =============================================================================
# MAIN PROMODE MODEL
# =============================================================================

class ProMode(nn.Module):
    """
    ProMode: Stand-alone prosody model for TTS.

    Takes acoustic features and time-aligned text as input,
    produces prosodic features (F0, energy, duration) as output.

    Can be integrated into any downstream TTS system.
    """

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config

        # Encoder (Perceiver IO)
        self.encoder = ProModeEncoder(config)

        # Dual decoders
        self.decoder_with_text = ProModeDecoder(config, use_text_condition=True)
        self.decoder_no_text = ProModeDecoder(config, use_text_condition=False)

        # Prosody prediction heads
        self.f0_predictor = F0Predictor(config)
        self.energy_predictor = EnergyPredictor(config)
        self.duration_predictor = DurationPredictor(config)

    def forward(
        self,
        acoustic_features: torch.Tensor,  # [batch, acoustic_seq, acoustic_dim]
        text_features: torch.Tensor,  # [batch, text_seq, text_dim]
        apply_masking: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and dual decoders.

        Args:
            acoustic_features: Mel spectrogram or similar
            text_features: Text encoder output
            apply_masking: Whether to apply masking (training)

        Returns:
            Dict with encoder and decoder outputs
        """
        target_length = acoustic_features.shape[1]

        # Encode
        encoder_output = self.encoder(
            acoustic_features,
            text_features,
            apply_masking=apply_masking,
        )

        latent_prosody = encoder_output['latent_prosody']

        # Decode with text condition
        recon_with_text = self.decoder_with_text(
            latent_prosody,
            target_length,
            text_features,
        )

        # Decode without text condition
        recon_no_text = self.decoder_no_text(
            latent_prosody,
            target_length,
            None,
        )

        # Predict prosody features
        f0_pred = self.f0_predictor(latent_prosody, target_length)
        energy_pred = self.energy_predictor(latent_prosody, target_length)
        duration_pred = self.duration_predictor(latent_prosody)

        return {
            **encoder_output,
            'recon_with_text': recon_with_text,
            'recon_no_text': recon_no_text,
            'f0_pred': f0_pred,
            'energy_pred': energy_pred,
            'duration_pred': duration_pred,
        }

    def predict_prosody(
        self,
        text_features: torch.Tensor,  # [batch, text_seq, text_dim]
        target_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict prosody from text only (inference mode).

        Uses a placeholder for acoustic features since we don't have
        ground truth acoustics during TTS inference.

        Args:
            text_features: Text encoder output
            target_length: Target acoustic sequence length

        Returns:
            Dict with predicted prosody features
        """
        batch_size = text_features.shape[0]
        text_seq_len = text_features.shape[1]
        device = text_features.device

        if target_length is None:
            # Estimate target length from text (rough approximation)
            target_length = text_seq_len * 4  # Assume ~4 acoustic frames per text token

        # Create placeholder acoustic features
        # During inference, we don't have ground truth acoustics
        # The encoder learns to handle this by the masking strategy
        placeholder_acoustic = torch.zeros(
            batch_size, target_length, self.config.acoustic_dim,
            device=device,
        )

        # Encode with no masking (use full text)
        encoder_output = self.encoder(
            placeholder_acoustic,
            text_features,
            apply_masking=False,
        )

        latent_prosody = encoder_output['latent_prosody']

        # Predict prosody
        f0_pred = self.f0_predictor(latent_prosody, target_length)
        energy_pred = self.energy_predictor(latent_prosody, target_length)
        duration_pred = self.duration_predictor(latent_prosody)

        return {
            'latent_prosody': latent_prosody,
            'f0': f0_pred['f0'],
            'log_f0': f0_pred['log_f0'],
            'vuv': f0_pred['vuv'],
            'energy': energy_pred,
            'duration': duration_pred,
        }

    def compute_loss(
        self,
        acoustic_features: torch.Tensor,  # [batch, acoustic_seq, acoustic_dim]
        text_features: torch.Tensor,  # [batch, text_seq, text_dim]
        f0_target: Optional[torch.Tensor] = None,  # [batch, acoustic_seq]
        energy_target: Optional[torch.Tensor] = None,  # [batch, acoustic_seq]
        vuv_target: Optional[torch.Tensor] = None,  # [batch, acoustic_seq]
        duration_target: Optional[torch.Tensor] = None,  # [batch, num_phonemes]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            acoustic_features: Ground truth mel spectrogram
            text_features: Text encoder output
            f0_target: Ground truth F0 (optional)
            energy_target: Ground truth energy (optional)
            vuv_target: Ground truth voiced/unvoiced (optional)
            duration_target: Ground truth duration (optional)

        Returns:
            Dict with individual losses and total
        """
        # Forward pass
        output = self.forward(acoustic_features, text_features, apply_masking=True)

        losses = {}

        # Reconstruction loss (masked regions only)
        acoustic_mask = output['acoustic_mask']

        if acoustic_mask is not None:
            # Loss only on masked positions
            mask = acoustic_mask.unsqueeze(-1)

            # Decoder with text
            recon_loss_text = F.mse_loss(
                output['recon_with_text'] * mask.float(),
                acoustic_features * mask.float(),
                reduction='sum',
            ) / (mask.sum() + 1e-8)

            # Decoder without text
            recon_loss_no_text = F.mse_loss(
                output['recon_no_text'] * mask.float(),
                acoustic_features * mask.float(),
                reduction='sum',
            ) / (mask.sum() + 1e-8)

            losses['recon_with_text'] = recon_loss_text * self.config.reconstruction_weight
            losses['recon_no_text'] = recon_loss_no_text * self.config.reconstruction_weight

            # Consistency loss between two decoders
            consistency = F.mse_loss(
                output['recon_with_text'],
                output['recon_no_text'],
            )
            losses['consistency'] = consistency * self.config.consistency_weight
        else:
            # Full reconstruction loss
            losses['recon_with_text'] = F.mse_loss(
                output['recon_with_text'],
                acoustic_features,
            ) * self.config.reconstruction_weight
            losses['recon_no_text'] = F.mse_loss(
                output['recon_no_text'],
                acoustic_features,
            ) * self.config.reconstruction_weight

        # F0 loss
        if f0_target is not None:
            f0_pred = output['f0_pred']['f0']
            min_len = min(f0_pred.shape[1], f0_target.shape[1])

            # F0 regression loss (only on voiced frames)
            if vuv_target is not None:
                voiced_mask = vuv_target[:, :min_len] > 0.5
                if voiced_mask.sum() > 0:
                    f0_loss = F.l1_loss(
                        f0_pred[:, :min_len][voiced_mask],
                        f0_target[:, :min_len][voiced_mask],
                    )
                else:
                    f0_loss = torch.tensor(0.0, device=f0_pred.device)
            else:
                f0_loss = F.l1_loss(f0_pred[:, :min_len], f0_target[:, :min_len])

            losses['f0'] = f0_loss * self.config.f0_weight

        # Voiced/unvoiced loss
        if vuv_target is not None:
            vuv_pred = output['f0_pred']['vuv']
            min_len = min(vuv_pred.shape[1], vuv_target.shape[1])
            vuv_loss = F.binary_cross_entropy(
                vuv_pred[:, :min_len],
                vuv_target[:, :min_len].float(),
            )
            losses['vuv'] = vuv_loss * 0.1

        # Energy loss
        if energy_target is not None:
            energy_pred = output['energy_pred']
            min_len = min(energy_pred.shape[1], energy_target.shape[1])
            energy_loss = F.l1_loss(
                energy_pred[:, :min_len],
                energy_target[:, :min_len],
            )
            losses['energy'] = energy_loss * self.config.energy_weight

        # Duration loss
        if duration_target is not None:
            duration_pred = output['duration_pred']
            if duration_pred.shape[1] != duration_target.shape[1]:
                duration_pred = F.interpolate(
                    duration_pred.unsqueeze(1),
                    size=duration_target.shape[1],
                    mode='linear',
                    align_corners=False,
                ).squeeze(1)
            duration_loss = F.mse_loss(duration_pred, duration_target)
            losses['duration'] = duration_loss * self.config.duration_weight

        # Total loss
        losses['total'] = sum(losses.values())

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class ProModeAdapter(nn.Module):
    """
    Adapter for integrating ProMode with CSM pipeline.

    Converts ProMode prosody predictions to prefix tokens compatible
    with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: ProModeConfig,
        model: Optional[ProMode] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.promode = model if model is not None else ProMode(config)

        # Project latent prosody to output dimension
        self.latent_proj = nn.Linear(config.latent_dim, config.output_dim)

        # Project F0 and energy to embedding
        self.f0_proj = nn.Sequential(
            nn.Linear(2, config.output_dim // 4),  # F0 + VUV
            nn.GELU(),
            nn.Linear(config.output_dim // 4, config.output_dim // 4),
        )
        self.energy_proj = nn.Sequential(
            nn.Linear(1, config.output_dim // 4),
            nn.GELU(),
            nn.Linear(config.output_dim // 4, config.output_dim // 4),
        )

        # Combine prosody features
        self.combine = nn.Sequential(
            nn.Linear(config.output_dim + config.output_dim // 2, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

        # Generate prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

        # Attention-based pooling for variable-length to fixed tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_prefix_tokens, config.output_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(
        self,
        text_features: torch.Tensor,  # [batch, text_seq, text_dim]
        acoustic_features: Optional[torch.Tensor] = None,  # Optional for training
        target_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody prefix tokens.

        Args:
            text_features: Text encoder output
            acoustic_features: Optional acoustic features for training
            target_length: Target acoustic sequence length

        Returns:
            Dict with:
                - prosody_tokens: [batch, num_prefix_tokens, output_dim]
                - f0, energy, duration predictions
        """
        batch_size = text_features.shape[0]
        device = text_features.device

        if target_length is None:
            target_length = text_features.shape[1] * 4

        # Get prosody predictions
        if acoustic_features is not None:
            # Training mode: use both acoustic and text
            output = self.promode(
                acoustic_features,
                text_features,
                apply_masking=True,
            )
            latent = output['latent_prosody']
            f0_pred = output['f0_pred']
            energy_pred = output['energy_pred']
        else:
            # Inference mode: text only
            prosody = self.promode.predict_prosody(text_features, target_length)
            latent = prosody['latent_prosody']
            f0_pred = {
                'f0': prosody['f0'],
                'log_f0': prosody['log_f0'],
                'vuv': prosody['vuv'],
            }
            energy_pred = prosody['energy']

        # Project latent prosody
        latent_proj = self.latent_proj(latent)  # [batch, num_queries, output_dim]

        # Create prosody features from F0 and energy (pooled)
        f0_mean = f0_pred['log_f0'].mean(dim=1, keepdim=True)  # [batch, 1]
        vuv_mean = f0_pred['vuv'].mean(dim=1, keepdim=True)  # [batch, 1]
        f0_features = self.f0_proj(torch.cat([f0_mean, vuv_mean], dim=-1))  # [batch, dim//4]

        if isinstance(energy_pred, dict):
            energy_mean = energy_pred.mean(dim=1, keepdim=True)
        else:
            energy_mean = energy_pred.mean(dim=1, keepdim=True)  # [batch, 1]
        energy_features = self.energy_proj(energy_mean)  # [batch, dim//4]

        # Expand to sequence length
        prosody_features = torch.cat([f0_features, energy_features], dim=-1)  # [batch, dim//2]
        prosody_features = prosody_features.unsqueeze(1).expand(-1, latent_proj.shape[1], -1)

        # Combine latent and prosody features
        combined = torch.cat([latent_proj, prosody_features], dim=-1)
        combined = self.combine(combined)  # [batch, num_queries, output_dim]

        # Generate prefix tokens via cross-attention
        queries = self.query_tokens.expand(batch_size, -1, -1)
        tokens, _ = self.cross_attn(queries, combined, combined)
        tokens = self.norm(tokens)

        result = {
            'prosody_tokens': tokens,
            'f0': f0_pred['f0'] if isinstance(f0_pred, dict) else f0_pred,
            'vuv': f0_pred['vuv'] if isinstance(f0_pred, dict) else None,
            'latent_prosody': latent,
        }

        if isinstance(energy_pred, torch.Tensor):
            result['energy'] = energy_pred
        else:
            result['energy'] = energy_pred

        return result

    def from_text(
        self,
        text_features: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from text only (inference).

        This is the main method for downstream TTS integration.
        """
        return self.forward(text_features, None, target_length)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class ProModeLoss(nn.Module):
    """Combined loss for ProMode training."""

    def __init__(self, config: ProModeConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        acoustic_target: torch.Tensor,
        f0_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        vuv_target: Optional[torch.Tensor] = None,
        duration_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Reconstruction losses
        acoustic_mask = model_output.get('acoustic_mask')

        if acoustic_mask is not None:
            mask = acoustic_mask.unsqueeze(-1).float()
            mask_sum = mask.sum() + 1e-8

            recon_text = F.mse_loss(
                model_output['recon_with_text'] * mask,
                acoustic_target * mask,
                reduction='sum',
            ) / mask_sum

            recon_no_text = F.mse_loss(
                model_output['recon_no_text'] * mask,
                acoustic_target * mask,
                reduction='sum',
            ) / mask_sum
        else:
            recon_text = F.mse_loss(
                model_output['recon_with_text'],
                acoustic_target,
            )
            recon_no_text = F.mse_loss(
                model_output['recon_no_text'],
                acoustic_target,
            )

        losses['recon_with_text'] = recon_text * self.config.reconstruction_weight
        losses['recon_no_text'] = recon_no_text * self.config.reconstruction_weight

        # Consistency loss
        consistency = F.mse_loss(
            model_output['recon_with_text'],
            model_output['recon_no_text'],
        )
        losses['consistency'] = consistency * self.config.consistency_weight

        # F0 loss
        if f0_target is not None:
            f0_pred = model_output['f0_pred']['f0']
            min_len = min(f0_pred.shape[1], f0_target.shape[1])

            if vuv_target is not None:
                voiced = vuv_target[:, :min_len] > 0.5
                if voiced.sum() > 0:
                    f0_loss = F.l1_loss(
                        f0_pred[:, :min_len][voiced],
                        f0_target[:, :min_len][voiced],
                    )
                else:
                    f0_loss = torch.tensor(0.0, device=f0_pred.device)
            else:
                f0_loss = F.l1_loss(f0_pred[:, :min_len], f0_target[:, :min_len])

            losses['f0'] = f0_loss * self.config.f0_weight

        # VUV loss
        if vuv_target is not None:
            vuv_pred = model_output['f0_pred']['vuv']
            min_len = min(vuv_pred.shape[1], vuv_target.shape[1])
            vuv_loss = F.binary_cross_entropy(
                vuv_pred[:, :min_len],
                vuv_target[:, :min_len].float(),
            )
            losses['vuv'] = vuv_loss * 0.1

        # Energy loss
        if energy_target is not None:
            energy_pred = model_output['energy_pred']
            min_len = min(energy_pred.shape[1], energy_target.shape[1])
            energy_loss = F.l1_loss(
                energy_pred[:, :min_len],
                energy_target[:, :min_len],
            )
            losses['energy'] = energy_loss * self.config.energy_weight

        # Duration loss
        if duration_target is not None:
            duration_pred = model_output['duration_pred']
            if duration_pred.shape[1] != duration_target.shape[1]:
                duration_pred = F.interpolate(
                    duration_pred.unsqueeze(1),
                    size=duration_target.shape[1],
                    mode='linear',
                    align_corners=False,
                ).squeeze(1)
            duration_loss = F.mse_loss(duration_pred, duration_target)
            losses['duration'] = duration_loss * self.config.duration_weight

        losses['total'] = sum(losses.values())

        return losses


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_promode(config: Optional[ProModeConfig] = None) -> ProMode:
    """Create ProMode model."""
    if config is None:
        config = ProModeConfig()
    return ProMode(config)


def create_promode_adapter(
    config: Optional[ProModeConfig] = None,
    checkpoint: Optional[str] = None,
) -> ProModeAdapter:
    """
    Create ProMode adapter for CSM integration.

    Args:
        config: ProMode configuration
        checkpoint: Path to checkpoint file

    Returns:
        ProModeAdapter ready for inference
    """
    if config is None:
        config = ProModeConfig()

    model = ProMode(config)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded ProMode checkpoint from {checkpoint}")

    return ProModeAdapter(config, model)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ProMode: Stand-Alone Prosody Model for TTS - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ProModeConfig()

    # Test parameters
    batch_size = 2
    acoustic_seq = 200
    text_seq = 50

    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Acoustic seq: {acoustic_seq}")
    print(f"Text seq: {text_seq}")

    # Test 1: PatchMasker
    print("\n[Test 1] PatchMasker...")
    masker = PatchMasker(config).to(device)
    x = torch.randn(batch_size, acoustic_seq, config.acoustic_dim).to(device)
    x_masked, mask = masker(x, mask_ratio=0.5)
    print(f"  Input: {x.shape}")
    print(f"  Masked: {x_masked.shape}")
    print(f"  Mask: {mask.shape}, ratio: {mask.float().mean():.2f}")
    print("  [PASS]")

    # Test 2: Perceiver Cross-Attention
    print("\n[Test 2] Perceiver Cross-Attention...")
    cross_attn = PerceiverCrossAttention(
        query_dim=config.latent_dim,
        key_dim=config.latent_dim,
        num_heads=config.cross_attn_heads,
    ).to(device)

    queries = torch.randn(batch_size, config.num_latent_queries, config.latent_dim).to(device)
    kv = torch.randn(batch_size, acoustic_seq + text_seq, config.latent_dim).to(device)
    attended = cross_attn(queries, kv)
    print(f"  Queries: {queries.shape}")
    print(f"  Key-Value: {kv.shape}")
    print(f"  Attended: {attended.shape}")
    print("  [PASS]")

    # Test 3: ProMode Encoder
    print("\n[Test 3] ProMode Encoder...")
    encoder = ProModeEncoder(config).to(device)

    acoustic = torch.randn(batch_size, acoustic_seq, config.acoustic_dim).to(device)
    text = torch.randn(batch_size, text_seq, config.text_dim).to(device)

    enc_output = encoder(acoustic, text, apply_masking=True)
    print(f"  Acoustic input: {acoustic.shape}")
    print(f"  Text input: {text.shape}")
    print(f"  Latent prosody: {enc_output['latent_prosody'].shape}")
    print(f"  Acoustic mask: {enc_output['acoustic_mask'].shape if enc_output['acoustic_mask'] is not None else 'None'}")
    print("  [PASS]")

    # Test 4: ProMode Decoder
    print("\n[Test 4] ProMode Decoder...")
    decoder_text = ProModeDecoder(config, use_text_condition=True).to(device)
    decoder_no_text = ProModeDecoder(config, use_text_condition=False).to(device)

    latent = enc_output['latent_prosody']
    recon_text = decoder_text(latent, acoustic_seq, text)
    recon_no_text = decoder_no_text(latent, acoustic_seq, None)

    print(f"  Latent: {latent.shape}")
    print(f"  Recon (with text): {recon_text.shape}")
    print(f"  Recon (no text): {recon_no_text.shape}")
    print("  [PASS]")

    # Test 5: Prosody Predictors
    print("\n[Test 5] Prosody Predictors...")
    f0_pred = F0Predictor(config).to(device)
    energy_pred = EnergyPredictor(config).to(device)
    duration_pred = DurationPredictor(config).to(device)

    f0_out = f0_pred(latent, acoustic_seq)
    energy_out = energy_pred(latent, acoustic_seq)
    duration_out = duration_pred(latent)

    print(f"  F0: {f0_out['f0'].shape}, VUV: {f0_out['vuv'].shape}")
    print(f"  Energy: {energy_out.shape}")
    print(f"  Duration: {duration_out.shape}")
    print("  [PASS]")

    # Test 6: Full ProMode
    print("\n[Test 6] Full ProMode...")
    model = ProMode(config).to(device)

    output = model(acoustic, text, apply_masking=True)
    print(f"  Latent prosody: {output['latent_prosody'].shape}")
    print(f"  Recon (text): {output['recon_with_text'].shape}")
    print(f"  Recon (no text): {output['recon_no_text'].shape}")
    print(f"  F0 pred: {output['f0_pred']['f0'].shape}")
    print(f"  Energy pred: {output['energy_pred'].shape}")
    print("  [PASS]")

    # Test 7: Loss Computation
    print("\n[Test 7] Loss Computation...")
    f0_target = torch.rand(batch_size, acoustic_seq).to(device) * 300 + 100
    energy_target = torch.randn(batch_size, acoustic_seq).to(device)
    vuv_target = (torch.rand(batch_size, acoustic_seq) > 0.3).float().to(device)

    losses = model.compute_loss(
        acoustic, text,
        f0_target=f0_target,
        energy_target=energy_target,
        vuv_target=vuv_target,
    )

    print(f"  Recon (text): {losses['recon_with_text'].item():.4f}")
    print(f"  Recon (no text): {losses['recon_no_text'].item():.4f}")
    print(f"  F0 loss: {losses.get('f0', 0):.4f}")
    print(f"  Energy loss: {losses.get('energy', 0):.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 8: Inference (text-only)
    print("\n[Test 8] Inference (text-only)...")
    with torch.no_grad():
        prosody = model.predict_prosody(text, target_length=acoustic_seq)
    print(f"  F0: {prosody['f0'].shape}")
    print(f"  Energy: {prosody['energy'].shape}")
    print(f"  VUV: {prosody['vuv'].shape}")
    print(f"  Duration: {prosody['duration'].shape}")
    print("  [PASS]")

    # Test 9: ProMode Adapter
    print("\n[Test 9] ProMode Adapter...")
    adapter = ProModeAdapter(config, model).to(device)

    # Training mode (with acoustic)
    result = adapter(text, acoustic)
    print(f"  Prosody tokens (train): {result['prosody_tokens'].shape}")

    # Inference mode (text only)
    result = adapter.from_text(text)
    print(f"  Prosody tokens (infer): {result['prosody_tokens'].shape}")
    assert result['prosody_tokens'].shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 10: Backward pass
    print("\n[Test 10] Backward pass...")
    model.zero_grad()
    losses = model.compute_loss(acoustic, text, f0_target=f0_target)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Memory usage
    if device == "cuda":
        print("\n[Memory Usage]")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("All ProMode tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. PERCEIVER IO ENCODER:
       - Latent queries attend to masked acoustic + text
       - Fixed-length prosody embedding regardless of input length
       - Cross-attention for joint prosody representation

    2. DUAL DECODERS:
       - With text: Uses latent + unmasked text for reconstruction
       - Without text: Uses latent only (for text-only inference)
       - Consistency loss between decoders

    3. PROSODY PREDICTION HEADS:
       - F0 predictor with voiced/unvoiced classification
       - Energy predictor
       - Duration predictor

    4. MODULAR INTEGRATION:
       - Can be used with any downstream TTS
       - Tested with FluentSpeech in paper
       - Prefix tokens for CSM integration
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from promode import (
    ProModeConfig,
    ProMode,
    ProModeAdapter,
    create_promode_adapter,
)

# Initialize
config = ProModeConfig()
model = ProMode(config).cuda()

# Training: masked prediction
losses = model.compute_loss(
    acoustic_features=mel,
    text_features=text_emb,
    f0_target=f0,
    energy_target=energy,
)

# Inference: text-to-prosody
prosody = model.predict_prosody(text_features=text_emb)
f0_pred = prosody['f0']
energy_pred = prosody['energy']

# CSM integration
adapter = ProModeAdapter(config, model)
prefix_tokens = adapter.from_text(text_emb)['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
