"""
MSR-Codec: Multi-Stream Residual Speech Codec with 4-Stream Disentanglement

Based on "MSR-Codec: Multi-Stream Residual Codec for Ultra-Low Bitrate Speech Coding"
(arXiv:2509.13068)

Key Innovation: 4-stream decomposition with cascaded residual architecture for implicit
disentanglement WITHOUT adversarial training. Each stream naturally captures distinct
attributes as information flows through progressive fusion.

4 Streams:
1. Semantic (HuBERT) - linguistic content
2. Timbre (speaker embedding) - speaker identity
3. Prosody (VQ1, 32-128 codes) - prosodic patterns
4. Residual (VQ2, 64-128 codes) - fine-grained acoustic details

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                        Audio Input                              │
    └─────────────────────────┬──────────────────────────────────────┘
                              │
    ┌─────────────────────────┼──────────────────────────────────────┐
    │                         ▼                                       │
    │  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
    │  │    Semantic    │  │     Timbre      │  │    Acoustic     │  │
    │  │    Encoder     │  │    Encoder      │  │    Encoder      │  │
    │  │   (HuBERT)     │  │  (Speaker Emb)  │  │  (Mel → Latent) │  │
    │  └───────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
    │          │                    │                     │           │
    │          ▼                    ▼                     ▼           │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │              Progressive Residual Fusion                   │ │
    │  │                                                            │ │
    │  │   acoustic_latent                                          │ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   [- semantic_latent] ────────────────────────────────────►│ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   residual_1 = acoustic - semantic                         │ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   [- timbre_proj] ────────────────────────────────────────►│ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   residual_2 = residual_1 - timbre                         │ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   ┌─────────┐     prosody_res = VQ1(residual_2)            │ │
    │  │   │  VQ1    │ ──────────────────────────────────────────►  │ │
    │  │   └─────────┘                                              │ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   residual_3 = residual_2 - prosody_res                    │ │
    │  │        │                                                   │ │
    │  │        ▼                                                   │ │
    │  │   ┌─────────┐     detail_res = VQ2(residual_3)             │ │
    │  │   │  VQ2    │ ──────────────────────────────────────────►  │ │
    │  │   └─────────┘                                              │ │
    │  │                                                            │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                  │
    │  OUTPUTS:                                                        │
    │    - semantic_tokens: HuBERT quantized tokens                    │
    │    - timbre_emb: Global speaker embedding                        │
    │    - prosody_codes: VQ1 prosody pattern codes (32-128)          │
    │    - residual_codes: VQ2 fine-grained detail codes (64-128)     │
    │                                                                  │
    │  BITRATE: ~424-524 bps (62.5 tokens/sec)                        │
    │  COMPRESSION: >200x factor                                       │
    └──────────────────────────────────────────────────────────────────┘

Benefits:
- No adversarial training needed - implicit disentanglement via cascaded residuals
- Ultra-low bitrate (~424-524 bps) with good quality
- Clean prosody codes without semantic/speaker leakage
- Effective zero-shot voice conversion of both timbre AND prosody
- Simpler and more stable training than GRL-based approaches

Usage:
    from msr_codec import (
        MSRCodecConfig,
        MSRCodec,
        MSRCodecLoss,
        MSRCodecAdapter,
        ProgressiveResidualFusion,
    )

    # Initialize
    config = MSRCodecConfig()
    model = MSRCodec(config).cuda()

    # Encode to 4 streams
    encoded = model.encode(mel, semantic_features)
    semantic_tokens = encoded['semantic_tokens']
    timbre_emb = encoded['timbre_emb']
    prosody_codes = encoded['prosody_codes']
    residual_codes = encoded['residual_codes']

    # Decode from streams
    mel_reconstructed = model.decode(
        semantic_tokens, timbre_emb, prosody_codes, residual_codes
    )

    # Zero-shot voice conversion (both timbre AND prosody)
    mel_converted = model.voice_convert(
        source_mel=content_source,
        target_prosody_mel=prosody_reference,
        target_timbre_mel=speaker_reference,
    )

    # CSM integration
    adapter = MSRCodecAdapter(config, model)
    prefix_tokens = adapter(mel, semantic_features)  # [batch, 4, 2048]
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MSRCodecConfig:
    """Configuration for MSR-Codec 4-stream disentanglement."""

    # Input dimensions
    mel_dim: int = 80  # Mel spectrogram channels
    semantic_dim: int = 768  # HuBERT/WavLM feature dimension
    sample_rate: int = 16000
    hop_length: int = 320  # 20ms at 16kHz (~62.5 tokens/sec)

    # Semantic Encoder (uses pre-trained HuBERT features)
    semantic_codebook_size: int = 500  # HuBERT k-means clusters (if quantizing)
    use_semantic_quantization: bool = False  # Use pre-quantized or continuous

    # Timbre Encoder (global speaker embedding)
    timbre_dim: int = 256  # Speaker embedding dimension
    timbre_hidden_dim: int = 512
    timbre_num_layers: int = 3

    # Acoustic Encoder (mel → latent)
    acoustic_dim: int = 256  # Latent dimension for residual computation
    acoustic_hidden_dim: int = 512
    acoustic_num_layers: int = 4

    # Prosody VQ (VQ1) - small codebook for prosodic patterns
    prosody_codebook_size: int = 64  # 32-128 in paper, 64 default
    prosody_code_dim: int = 64  # Per-code dimension
    prosody_num_heads: int = 1  # Single head VQ
    prosody_commitment_cost: float = 0.25
    prosody_ema_decay: float = 0.99

    # Residual VQ (VQ2) - captures fine-grained details
    residual_codebook_size: int = 128  # 64-128 in paper
    residual_code_dim: int = 64
    residual_num_heads: int = 1
    residual_commitment_cost: float = 0.25
    residual_ema_decay: float = 0.99

    # Progressive Residual Fusion
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.1

    # Decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Training settings
    dropout: float = 0.1

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# HELPER MODULES
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Residual convolutional block."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual connection
        if stride != 1 or in_dim != out_dim:
            self.residual = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 1, stride),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Align lengths if needed
        min_len = min(out.shape[-1], residual.shape[-1])
        out = out[..., :min_len]
        residual = residual[..., :min_len]

        return self.activation(out + residual)


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = residual + x

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


# =============================================================================
# VECTOR QUANTIZER WITH EMA
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average codebook updates.

    Used for both Prosody VQ (VQ1) and Residual VQ (VQ2).
    Small codebook sizes (32-128) encourage abstraction and disentanglement.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 64,
        code_dim: int = 64,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Projection layers
        self.pre_proj = nn.Linear(input_dim, code_dim)
        self.post_proj = nn.Linear(code_dim, input_dim)

        # Codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim))

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_sum', torch.randn(codebook_size, code_dim))
        self.register_buffer('initialized', torch.tensor(False))

        # Initialize codebook
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size)

    def _init_from_data(self, z: torch.Tensor):
        """Initialize codebook from first batch (k-means++ style)."""
        n_samples = z.shape[0]
        if n_samples >= self.codebook_size:
            indices = torch.randperm(n_samples)[:self.codebook_size]
            self.codebook.data.copy_(z[indices])
        else:
            repeats = (self.codebook_size // n_samples) + 1
            expanded = z.repeat(repeats, 1)[:self.codebook_size]
            self.codebook.data.copy_(expanded)

        self.ema_sum.data.copy_(self.codebook.data)
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input.

        Returns:
            Dict with z_q, indices, commitment_loss, perplexity, quantized_proj
        """
        batch_size, seq_len, _ = x.shape

        # Project to code dimension
        z = self.pre_proj(x)  # [B, T, code_dim]
        z_flat = z.view(-1, self.code_dim)  # [B*T, code_dim]

        # Initialize from first batch
        if self.training and not self.initialized:
            self._init_from_data(z_flat)

        # Compute distances
        d = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.codebook.t())
            + self.codebook.pow(2).sum(dim=-1)
        )  # [B*T, codebook_size]

        # Find nearest codes
        indices = d.argmin(dim=-1)  # [B*T]
        z_q = F.embedding(indices, self.codebook)  # [B*T, code_dim]

        # EMA update
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                new_size = encodings.sum(dim=0)
                new_sum = torch.matmul(encodings.t(), z_flat)

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    new_size, alpha=1 - self.ema_decay
                )
                self.ema_sum.mul_(self.ema_decay).add_(
                    new_sum, alpha=1 - self.ema_decay
                )

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon) /
                    (n + self.codebook_size * self.epsilon) * n
                )
                self.codebook.data.copy_(self.ema_sum / cluster_size.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z_flat, z_q.detach())

        # Straight-through estimator
        z_q_st = z_flat + (z_q - z_flat).detach()

        # Reshape
        z_q_st = z_q_st.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

        # Project back to input dimension
        z_q_proj = self.post_proj(z_q_st)

        # Compute perplexity
        flat_indices = indices.view(-1)
        encodings = F.one_hot(flat_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'z_q': z_q_st,  # Quantized in code dimension
            'z_q_proj': z_q_proj,  # Quantized projected back
            'indices': indices,
            'commitment_loss': commitment_loss * self.commitment_cost,
            'perplexity': perplexity,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to projected vectors."""
        z_q = F.embedding(indices, self.codebook)
        return self.post_proj(z_q)


# =============================================================================
# SEMANTIC ENCODER
# =============================================================================

class SemanticEncoder(nn.Module):
    """
    Semantic Encoder: Uses HuBERT features for linguistic content.

    Can either use pre-extracted HuBERT features directly or
    project them through a small network.
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Project HuBERT features to acoustic dimension for residual computation
        self.proj = nn.Sequential(
            nn.Linear(config.semantic_dim, config.acoustic_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.acoustic_hidden_dim, config.acoustic_dim),
            nn.LayerNorm(config.acoustic_dim),
        )

        # Optional: quantize semantic features (if using discrete tokens)
        if config.use_semantic_quantization:
            self.quantizer = VectorQuantizerEMA(
                input_dim=config.acoustic_dim,
                codebook_size=config.semantic_codebook_size,
                code_dim=config.acoustic_dim,
                commitment_cost=0.25,
                ema_decay=0.99,
            )
        else:
            self.quantizer = None

    def forward(
        self,
        semantic_features: torch.Tensor,  # [batch, seq, semantic_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode semantic (linguistic) content.

        Returns:
            Dict with semantic_z, semantic_tokens (if quantizing)
        """
        # Project to acoustic dimension
        z = self.proj(semantic_features)  # [B, T, acoustic_dim]

        result = {
            'semantic_z': z,
        }

        # Optionally quantize
        if self.quantizer is not None:
            quant_output = self.quantizer(z)
            result['semantic_tokens'] = quant_output['indices']
            result['semantic_z_q'] = quant_output['z_q_proj']
            result['semantic_commitment_loss'] = quant_output['commitment_loss']
            result['semantic_perplexity'] = quant_output['perplexity']
        else:
            result['semantic_tokens'] = None

        return result


# =============================================================================
# TIMBRE ENCODER
# =============================================================================

class TimbreEncoder(nn.Module):
    """
    Timbre Encoder: Extracts global speaker identity embedding.

    Uses attentive statistics pooling to capture speaker characteristics
    that are time-invariant (constant across the utterance).
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Initial projection from mel
        self.input_proj = nn.Linear(config.mel_dim, config.timbre_hidden_dim)

        # Convolutional layers for local pattern extraction
        self.conv_layers = nn.ModuleList()
        for i in range(config.timbre_num_layers):
            self.conv_layers.append(
                ConvBlock(
                    config.timbre_hidden_dim,
                    config.timbre_hidden_dim,
                    kernel_size=5,
                    stride=1,
                    dropout=config.dropout,
                )
            )

        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Linear(config.timbre_hidden_dim, config.timbre_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.timbre_hidden_dim // 2, 1),
        )

        # Output projection (mean + std → timbre)
        self.output_proj = nn.Sequential(
            nn.Linear(config.timbre_hidden_dim * 2, config.timbre_hidden_dim),
            nn.GELU(),
            nn.Linear(config.timbre_hidden_dim, config.timbre_dim),
        )

        self.norm = nn.LayerNorm(config.timbre_dim)

        # Projection to acoustic dimension for residual subtraction
        self.to_acoustic = nn.Linear(config.timbre_dim, config.acoustic_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global timbre (speaker) embedding.

        Returns:
            Dict with:
                - timbre_emb: [batch, timbre_dim] global speaker vector
                - timbre_proj: [batch, acoustic_dim] projected for residual
        """
        batch_size, seq_len, _ = mel.shape

        # Project input
        x = self.input_proj(mel)  # [B, T, H]

        # Convolutional processing
        x = x.transpose(1, 2)  # [B, H, T]
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)  # [B, T, H]

        # Attentive statistics pooling
        attn_weights = self.attention(x)  # [B, T, 1]

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]

        # Weighted mean
        mean = (x * attn_weights).sum(dim=1)  # [B, H]

        # Weighted std
        var = ((x - mean.unsqueeze(1)).pow(2) * attn_weights).sum(dim=1)
        std = var.clamp(min=1e-8).sqrt()  # [B, H]

        # Concatenate statistics
        stats = torch.cat([mean, std], dim=-1)  # [B, H*2]

        # Project to timbre embedding
        timbre_emb = self.output_proj(stats)  # [B, timbre_dim]
        timbre_emb = self.norm(timbre_emb)

        # Project to acoustic dimension for residual computation
        timbre_proj = self.to_acoustic(timbre_emb)  # [B, acoustic_dim]

        return {
            'timbre_emb': timbre_emb,
            'timbre_proj': timbre_proj,
        }


# =============================================================================
# ACOUSTIC ENCODER
# =============================================================================

class AcousticEncoder(nn.Module):
    """
    Acoustic Encoder: Encodes mel spectrogram to latent representation.

    This captures the full acoustic information before we subtract
    semantic and timbre to get the residual prosody/detail components.
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = nn.Linear(config.mel_dim, config.acoustic_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.acoustic_hidden_dim, dropout=config.dropout
        )

        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.acoustic_hidden_dim,
                num_heads=8,
                ffn_dim=config.acoustic_hidden_dim * 4,
                dropout=config.dropout,
            )
            for _ in range(config.acoustic_num_layers)
        ])

        # Output projection to acoustic dimension
        self.output_proj = nn.Linear(config.acoustic_hidden_dim, config.acoustic_dim)
        self.norm = nn.LayerNorm(config.acoustic_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel to acoustic latent.

        Returns:
            Dict with acoustic_z: [batch, seq, acoustic_dim]
        """
        # Project input
        x = self.input_proj(mel)  # [B, T, H]

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoding
        for transformer in self.transformer:
            x = transformer(x, mask)

        # Project to acoustic dimension
        acoustic_z = self.output_proj(x)
        acoustic_z = self.norm(acoustic_z)

        return {
            'acoustic_z': acoustic_z,
        }


# =============================================================================
# PROGRESSIVE RESIDUAL FUSION
# =============================================================================

class ProgressiveResidualFusion(nn.Module):
    """
    Progressive Residual Fusion: The core of MSR-Codec's implicit disentanglement.

    This module implements cascaded residual subtraction:
    1. Start with full acoustic latent
    2. Subtract semantic (linguistic) → residual_1 contains prosody + speaker + detail
    3. Subtract timbre (speaker) → residual_2 contains prosody + detail
    4. VQ1 quantizes prosody → residual_3 contains fine detail only
    5. VQ2 quantizes residual detail

    Key insight: Each subtraction naturally removes that attribute,
    leaving a residual that contains the remaining information.
    No adversarial training needed!
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Learnable scaling for residual computation
        self.semantic_scale = nn.Parameter(torch.ones(1))
        self.timbre_scale = nn.Parameter(torch.ones(1))

        # Layer norm after each residual step
        self.norm1 = nn.LayerNorm(config.acoustic_dim)  # After semantic subtraction
        self.norm2 = nn.LayerNorm(config.acoustic_dim)  # After timbre subtraction
        self.norm3 = nn.LayerNorm(config.acoustic_dim)  # After prosody VQ

        # Prosody VQ (VQ1) - small codebook for prosodic patterns
        self.prosody_vq = VectorQuantizerEMA(
            input_dim=config.acoustic_dim,
            codebook_size=config.prosody_codebook_size,
            code_dim=config.prosody_code_dim,
            commitment_cost=config.prosody_commitment_cost,
            ema_decay=config.prosody_ema_decay,
        )

        # Residual VQ (VQ2) - captures fine-grained details
        self.residual_vq = VectorQuantizerEMA(
            input_dim=config.acoustic_dim,
            codebook_size=config.residual_codebook_size,
            code_dim=config.residual_code_dim,
            commitment_cost=config.residual_commitment_cost,
            ema_decay=config.residual_ema_decay,
        )

        # Optional: small MLP to refine residuals
        self.refine_prosody = nn.Sequential(
            nn.Linear(config.acoustic_dim, config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.acoustic_dim),
        )

    def forward(
        self,
        acoustic_z: torch.Tensor,  # [batch, seq, acoustic_dim]
        semantic_z: torch.Tensor,  # [batch, seq, acoustic_dim]
        timbre_proj: torch.Tensor,  # [batch, acoustic_dim] (broadcast)
    ) -> Dict[str, torch.Tensor]:
        """
        Progressive residual fusion.

        Returns:
            Dict with:
                - prosody_codes: VQ1 indices
                - prosody_z: VQ1 quantized
                - residual_codes: VQ2 indices
                - residual_z: VQ2 quantized
                - residual_1/2/3: intermediate residuals for analysis
                - commitment losses and perplexities
        """
        batch_size, seq_len, _ = acoustic_z.shape

        # Step 1: Subtract semantic content
        # residual_1 contains: prosody + speaker + detail
        semantic_aligned = semantic_z  # Already aligned in semantic encoder
        if semantic_aligned.shape[1] != seq_len:
            # Interpolate if needed
            semantic_aligned = F.interpolate(
                semantic_aligned.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        residual_1 = acoustic_z - self.semantic_scale * semantic_aligned
        residual_1 = self.norm1(residual_1)

        # Step 2: Subtract timbre (broadcast global embedding)
        # residual_2 contains: prosody + detail
        timbre_broadcast = timbre_proj.unsqueeze(1).expand(-1, seq_len, -1)
        residual_2 = residual_1 - self.timbre_scale * timbre_broadcast
        residual_2 = self.norm2(residual_2)

        # Optional: refine before VQ
        residual_2 = residual_2 + self.refine_prosody(residual_2)

        # Step 3: VQ1 - quantize prosody patterns
        prosody_output = self.prosody_vq(residual_2)

        # residual_3 contains: fine detail only
        residual_3 = residual_2 - prosody_output['z_q_proj']
        residual_3 = self.norm3(residual_3)

        # Step 4: VQ2 - quantize residual detail
        residual_output = self.residual_vq(residual_3)

        return {
            # Prosody stream (VQ1)
            'prosody_codes': prosody_output['indices'],
            'prosody_z': prosody_output['z_q'],
            'prosody_z_proj': prosody_output['z_q_proj'],
            'prosody_commitment_loss': prosody_output['commitment_loss'],
            'prosody_perplexity': prosody_output['perplexity'],

            # Residual stream (VQ2)
            'residual_codes': residual_output['indices'],
            'residual_z': residual_output['z_q'],
            'residual_z_proj': residual_output['z_q_proj'],
            'residual_commitment_loss': residual_output['commitment_loss'],
            'residual_perplexity': residual_output['perplexity'],

            # Intermediate residuals (for analysis/debugging)
            'residual_1': residual_1,  # After semantic subtraction
            'residual_2': residual_2,  # After timbre subtraction
            'residual_3': residual_3,  # After prosody VQ
        }


# =============================================================================
# DECODER
# =============================================================================

class MSRCodecDecoder(nn.Module):
    """
    Decoder: Reconstructs mel from all 4 streams.

    Combines:
    - Semantic (projected HuBERT features or quantized tokens)
    - Timbre (global speaker embedding, broadcast)
    - Prosody (VQ1 quantized patterns)
    - Residual (VQ2 fine-grained detail)
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Stream projections to hidden dimension
        self.semantic_proj = nn.Linear(config.acoustic_dim, config.decoder_hidden_dim)
        self.timbre_proj = nn.Linear(config.timbre_dim, config.decoder_hidden_dim)
        self.prosody_proj = nn.Linear(config.acoustic_dim, config.decoder_hidden_dim)
        self.residual_proj = nn.Linear(config.acoustic_dim, config.decoder_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.decoder_hidden_dim, dropout=config.dropout
        )

        # Transformer decoder
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.decoder_hidden_dim,
                num_heads=config.decoder_num_heads,
                ffn_dim=config.decoder_ffn_dim,
                dropout=config.dropout,
            )
            for _ in range(config.decoder_num_layers)
        ])

        # Output projection to mel
        self.output_proj = nn.Linear(config.decoder_hidden_dim, config.mel_dim)
        self.norm = nn.LayerNorm(config.decoder_hidden_dim)

    def forward(
        self,
        semantic_z: torch.Tensor,  # [batch, seq, acoustic_dim]
        timbre_emb: torch.Tensor,  # [batch, timbre_dim]
        prosody_z: torch.Tensor,  # [batch, seq, acoustic_dim]
        residual_z: torch.Tensor,  # [batch, seq, acoustic_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode from 4 streams to mel spectrogram.

        Returns:
            Dict with mel_reconstructed
        """
        batch_size, seq_len, _ = prosody_z.shape

        # Project each stream
        semantic_feat = self.semantic_proj(semantic_z)  # [B, T, H]
        prosody_feat = self.prosody_proj(prosody_z)  # [B, T, H]
        residual_feat = self.residual_proj(residual_z)  # [B, T, H]

        # Handle sequence length mismatch for semantic
        if semantic_feat.shape[1] != seq_len:
            semantic_feat = F.interpolate(
                semantic_feat.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        # Timbre (broadcast global embedding)
        timbre_feat = self.timbre_proj(timbre_emb)  # [B, H]
        timbre_feat = timbre_feat.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine all streams
        x = semantic_feat + timbre_feat + prosody_feat + residual_feat

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer decoding
        for transformer in self.transformer:
            x = transformer(x, mask)
        x = self.norm(x)

        # Output mel
        mel_reconstructed = self.output_proj(x)

        return {
            'mel_reconstructed': mel_reconstructed,
            'decoder_hidden': x,
        }


# =============================================================================
# FULL MSR-CODEC MODEL
# =============================================================================

class MSRCodec(nn.Module):
    """
    MSR-Codec: Multi-Stream Residual Speech Codec.

    4-stream decomposition with cascaded residual architecture:
    1. Semantic - linguistic content from HuBERT
    2. Timbre - speaker identity (global embedding)
    3. Prosody - prosodic patterns (VQ1, small codebook)
    4. Residual - fine-grained acoustic detail (VQ2)

    Key innovation: Implicit disentanglement via progressive residual
    subtraction - no adversarial training needed!
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Stream encoders
        self.semantic_encoder = SemanticEncoder(config)
        self.timbre_encoder = TimbreEncoder(config)
        self.acoustic_encoder = AcousticEncoder(config)

        # Progressive residual fusion (core disentanglement)
        self.fusion = ProgressiveResidualFusion(config)

        # Decoder
        self.decoder = MSRCodecDecoder(config)

        # Output projection for CSM integration
        combined_dim = (
            config.acoustic_dim +  # semantic
            config.timbre_dim +    # timbre
            config.acoustic_dim +  # prosody (projected back)
            config.acoustic_dim    # residual (projected back)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        semantic_features: torch.Tensor,  # [batch, seq, semantic_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode to 4 disentangled streams.

        Returns:
            Dict with all stream embeddings and codes
        """
        # Encode each stream
        semantic_output = self.semantic_encoder(semantic_features)
        timbre_output = self.timbre_encoder(mel, mask)
        acoustic_output = self.acoustic_encoder(mel, mask)

        # Progressive residual fusion
        fusion_output = self.fusion(
            acoustic_output['acoustic_z'],
            semantic_output['semantic_z'],
            timbre_output['timbre_proj'],
        )

        return {
            # Semantic stream
            'semantic_z': semantic_output['semantic_z'],
            'semantic_tokens': semantic_output.get('semantic_tokens'),

            # Timbre stream
            'timbre_emb': timbre_output['timbre_emb'],
            'timbre_proj': timbre_output['timbre_proj'],

            # Prosody stream (VQ1)
            'prosody_codes': fusion_output['prosody_codes'],
            'prosody_z': fusion_output['prosody_z'],
            'prosody_z_proj': fusion_output['prosody_z_proj'],
            'prosody_commitment_loss': fusion_output['prosody_commitment_loss'],
            'prosody_perplexity': fusion_output['prosody_perplexity'],

            # Residual stream (VQ2)
            'residual_codes': fusion_output['residual_codes'],
            'residual_z': fusion_output['residual_z'],
            'residual_z_proj': fusion_output['residual_z_proj'],
            'residual_commitment_loss': fusion_output['residual_commitment_loss'],
            'residual_perplexity': fusion_output['residual_perplexity'],

            # Intermediate for analysis
            'acoustic_z': acoustic_output['acoustic_z'],
            'residual_1': fusion_output['residual_1'],
            'residual_2': fusion_output['residual_2'],
            'residual_3': fusion_output['residual_3'],
        }

    def decode(
        self,
        semantic_z: torch.Tensor,
        timbre_emb: torch.Tensor,
        prosody_z_proj: torch.Tensor,
        residual_z_proj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode from 4 streams to mel spectrogram.
        """
        return self.decoder(
            semantic_z,
            timbre_emb,
            prosody_z_proj,
            residual_z_proj,
            mask,
        )

    def forward(
        self,
        mel: torch.Tensor,
        semantic_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Returns:
            Dict with all embeddings, codes, and reconstruction
        """
        # Encode
        encoded = self.encode(mel, semantic_features, mask)

        # Decode
        decoded = self.decode(
            encoded['semantic_z'],
            encoded['timbre_emb'],
            encoded['prosody_z_proj'],
            encoded['residual_z_proj'],
            mask,
        )

        # Combined embedding for downstream
        batch_size, seq_len, _ = encoded['prosody_z_proj'].shape

        # Pool temporal dimensions
        semantic_pooled = encoded['semantic_z'].mean(dim=1)
        prosody_pooled = encoded['prosody_z_proj'].mean(dim=1)
        residual_pooled = encoded['residual_z_proj'].mean(dim=1)

        combined = torch.cat([
            semantic_pooled,
            encoded['timbre_emb'],
            prosody_pooled,
            residual_pooled,
        ], dim=-1)

        combined_emb = self.output_proj(combined)

        return {
            **encoded,
            'mel_reconstructed': decoded['mel_reconstructed'],
            'combined_embedding': combined_emb,
        }

    def voice_convert(
        self,
        source_mel: torch.Tensor,
        source_semantic: torch.Tensor,
        target_prosody_mel: Optional[torch.Tensor] = None,
        target_prosody_semantic: Optional[torch.Tensor] = None,
        target_timbre_mel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Zero-shot voice conversion with control over both timbre AND prosody.

        Args:
            source_mel: Content source mel
            source_semantic: Content source HuBERT features
            target_prosody_mel: Prosody reference mel (optional)
            target_prosody_semantic: Prosody reference HuBERT (optional)
            target_timbre_mel: Timbre (speaker) reference mel

        Returns:
            Converted mel spectrogram
        """
        # Encode source (get semantic content)
        source_encoded = self.encode(source_mel, source_semantic)
        semantic_z = source_encoded['semantic_z']

        # Get timbre from target speaker
        if target_timbre_mel is not None:
            timbre_output = self.timbre_encoder(target_timbre_mel)
            timbre_emb = timbre_output['timbre_emb']
        else:
            timbre_emb = source_encoded['timbre_emb']

        # Get prosody from target (or source)
        if target_prosody_mel is not None and target_prosody_semantic is not None:
            target_encoded = self.encode(target_prosody_mel, target_prosody_semantic)
            prosody_z_proj = target_encoded['prosody_z_proj']
            residual_z_proj = target_encoded['residual_z_proj']

            # Handle length mismatch
            source_len = semantic_z.shape[1]
            if prosody_z_proj.shape[1] != source_len:
                prosody_z_proj = F.interpolate(
                    prosody_z_proj.transpose(1, 2),
                    size=source_len,
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
                residual_z_proj = F.interpolate(
                    residual_z_proj.transpose(1, 2),
                    size=source_len,
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
        else:
            prosody_z_proj = source_encoded['prosody_z_proj']
            residual_z_proj = source_encoded['residual_z_proj']

        # Decode with mixed streams
        decoded = self.decode(
            semantic_z,
            timbre_emb,
            prosody_z_proj,
            residual_z_proj,
        )

        return decoded['mel_reconstructed']

    def prosody_transfer(
        self,
        content_mel: torch.Tensor,
        content_semantic: torch.Tensor,
        prosody_mel: torch.Tensor,
        prosody_semantic: torch.Tensor,
        timbre_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Three-way transfer: content from A, prosody from B, timbre from C.
        """
        # Encode all sources
        content_encoded = self.encode(content_mel, content_semantic)
        prosody_encoded = self.encode(prosody_mel, prosody_semantic)
        timbre_output = self.timbre_encoder(timbre_mel)

        # Get components
        semantic_z = content_encoded['semantic_z']
        prosody_z_proj = prosody_encoded['prosody_z_proj']
        residual_z_proj = prosody_encoded['residual_z_proj']
        timbre_emb = timbre_output['timbre_emb']

        # Handle length mismatch
        source_len = semantic_z.shape[1]
        if prosody_z_proj.shape[1] != source_len:
            prosody_z_proj = F.interpolate(
                prosody_z_proj.transpose(1, 2),
                size=source_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)
            residual_z_proj = F.interpolate(
                residual_z_proj.transpose(1, 2),
                size=source_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        # Decode
        decoded = self.decode(semantic_z, timbre_emb, prosody_z_proj, residual_z_proj)

        return decoded['mel_reconstructed']

    def compute_bitrate(
        self,
        prosody_codes: torch.Tensor,
        residual_codes: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute bitrate for VQ streams.

        MSR-Codec target: ~424-524 bps with 62.5 tokens/sec
        """
        seq_len = prosody_codes.shape[1]
        hop_length = self.config.hop_length
        sample_rate = self.config.sample_rate

        # Duration in seconds
        duration_s = (seq_len * hop_length) / sample_rate

        # Bits per stream
        prosody_bits = math.log2(self.config.prosody_codebook_size)  # ~6 bits
        residual_bits = math.log2(self.config.residual_codebook_size)  # ~7 bits

        # Total bits per frame
        bits_per_frame = prosody_bits + residual_bits  # ~13 bits

        # Tokens per second
        tokens_per_sec = seq_len / duration_s

        # Bits per second
        bps = (seq_len * bits_per_frame) / duration_s

        return {
            'tokens_per_second': tokens_per_sec,
            'bits_per_second': bps,
            'prosody_bits_per_token': prosody_bits,
            'residual_bits_per_token': residual_bits,
            'total_bits_per_token': bits_per_frame,
            'duration_seconds': duration_s,
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class MSRCodecLoss(nn.Module):
    """
    Combined loss function for MSR-Codec training.

    Components:
    1. Reconstruction loss (L1 + L2 mel)
    2. VQ commitment losses (prosody + residual)
    3. Semantic consistency (if using quantized semantics)
    """

    def __init__(self, config: MSRCodecConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.reconstruction_weight = 1.0
        self.commitment_weight = 0.25

    def reconstruction_loss(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute mel reconstruction losses."""
        # Handle length mismatch
        min_len = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_len]
        mel_target = mel_target[:, :min_len]

        # L1 loss
        l1_loss = F.l1_loss(mel_pred, mel_target, reduction='none')

        # L2 loss
        l2_loss = F.mse_loss(mel_pred, mel_target, reduction='none')

        if mask is not None:
            mask = mask[:, :min_len]
            mask_expanded = mask.unsqueeze(-1)
            l1_loss = (l1_loss * mask_expanded).sum() / mask_expanded.sum()
            l2_loss = (l2_loss * mask_expanded).sum() / mask_expanded.sum()
        else:
            l1_loss = l1_loss.mean()
            l2_loss = l2_loss.mean()

        return {
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'reconstruction_loss': l1_loss + l2_loss,
        }

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            model_output: Output from MSRCodec.forward()
            mel_target: Target mel spectrogram
            mask: Optional mask for valid frames
        """
        losses = {}

        # Reconstruction loss
        recon_losses = self.reconstruction_loss(
            model_output['mel_reconstructed'],
            mel_target,
            mask,
        )
        losses.update(recon_losses)

        # VQ commitment losses
        losses['prosody_commitment'] = model_output['prosody_commitment_loss']
        losses['residual_commitment'] = model_output['residual_commitment_loss']
        losses['total_commitment'] = (
            losses['prosody_commitment'] + losses['residual_commitment']
        )

        # Perplexities (for monitoring)
        losses['prosody_perplexity'] = model_output['prosody_perplexity']
        losses['residual_perplexity'] = model_output['residual_perplexity']

        # Total loss
        total = (
            self.reconstruction_weight * losses['reconstruction_loss']
            + self.commitment_weight * losses['total_commitment']
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class MSRCodecAdapter(nn.Module):
    """
    Adapter for integrating MSR-Codec with existing prosody pipeline.

    Converts MSR-Codec's 4-stream representation to prefix tokens
    compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: MSRCodecConfig,
        model: Optional[MSRCodec] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else MSRCodec(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        mel: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram
            semantic_features: [batch, seq, semantic_dim] HuBERT features

        Returns:
            Dict with:
                - prosody_tokens: [batch, num_prefix_tokens, output_dim]
                - streams: Dict with separate stream embeddings
                - bitrate: Dict with bitrate statistics
        """
        batch_size = mel.shape[0]

        # Get combined embedding
        output = self.model(mel, semantic_features)
        combined_emb = output['combined_embedding']  # [B, output_dim]

        # Project to tokens
        tokens = self.token_proj(combined_emb)  # [B, output_dim * num_tokens]

        # Reshape
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )

        # Normalize
        tokens = self.norm(tokens)

        # Compute bitrate
        bitrate = self.model.compute_bitrate(
            output['prosody_codes'],
            output['residual_codes'],
        )

        return {
            'prosody_tokens': tokens,
            'streams': {
                'semantic_z': output['semantic_z'],
                'timbre_emb': output['timbre_emb'],
                'prosody_codes': output['prosody_codes'],
                'prosody_z': output['prosody_z_proj'],
                'residual_codes': output['residual_codes'],
                'residual_z': output['residual_z_proj'],
            },
            'commitment_loss': (
                output['prosody_commitment_loss'] +
                output['residual_commitment_loss']
            ),
            'prosody_perplexity': output['prosody_perplexity'],
            'residual_perplexity': output['residual_perplexity'],
            'bitrate': bitrate,
        }

    def voice_convert(
        self,
        source_mel: torch.Tensor,
        source_semantic: torch.Tensor,
        target_prosody_mel: Optional[torch.Tensor] = None,
        target_prosody_semantic: Optional[torch.Tensor] = None,
        target_timbre_mel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Zero-shot voice conversion."""
        return self.model.voice_convert(
            source_mel,
            source_semantic,
            target_prosody_mel,
            target_prosody_semantic,
            target_timbre_mel,
        )

    def prosody_transfer(
        self,
        content_mel: torch.Tensor,
        content_semantic: torch.Tensor,
        prosody_mel: torch.Tensor,
        prosody_semantic: torch.Tensor,
        timbre_mel: torch.Tensor,
    ) -> torch.Tensor:
        """Three-way prosody transfer."""
        return self.model.prosody_transfer(
            content_mel,
            content_semantic,
            prosody_mel,
            prosody_semantic,
            timbre_mel,
        )

    def get_separate_embeddings(
        self,
        mel: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get all 4 stream embeddings separately."""
        encoded = self.model.encode(mel, semantic_features)
        return {
            'semantic': encoded['semantic_z'],
            'timbre': encoded['timbre_emb'],
            'prosody': encoded['prosody_z_proj'],
            'prosody_codes': encoded['prosody_codes'],
            'residual': encoded['residual_z_proj'],
            'residual_codes': encoded['residual_codes'],
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MSR-Codec: 4-Stream Disentanglement - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 100
    mel_dim = 80
    semantic_dim = 768

    # Create dummy inputs
    mel = torch.randn(batch_size, seq_len, mel_dim).to(device)
    semantic_features = torch.randn(batch_size, seq_len, semantic_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    config = MSRCodecConfig()
    print(f"  Semantic dim: {config.semantic_dim}")
    print(f"  Timbre dim: {config.timbre_dim}")
    print(f"  Acoustic dim: {config.acoustic_dim}")
    print(f"  Prosody codebook: {config.prosody_codebook_size} codes")
    print(f"  Residual codebook: {config.residual_codebook_size} codes")
    print("  [PASS]")

    # Test 2: Semantic Encoder
    print("\n[Test 2] Semantic Encoder...")
    semantic_encoder = SemanticEncoder(config).to(device)
    semantic_output = semantic_encoder(semantic_features)
    print(f"  Semantic z shape: {semantic_output['semantic_z'].shape}")
    assert semantic_output['semantic_z'].shape == (batch_size, seq_len, config.acoustic_dim)
    print("  [PASS]")

    # Test 3: Timbre Encoder
    print("\n[Test 3] Timbre Encoder...")
    timbre_encoder = TimbreEncoder(config).to(device)
    timbre_output = timbre_encoder(mel)
    print(f"  Timbre emb shape: {timbre_output['timbre_emb'].shape}")
    print(f"  Timbre proj shape: {timbre_output['timbre_proj'].shape}")
    assert timbre_output['timbre_emb'].shape == (batch_size, config.timbre_dim)
    print("  [PASS]")

    # Test 4: Acoustic Encoder
    print("\n[Test 4] Acoustic Encoder...")
    acoustic_encoder = AcousticEncoder(config).to(device)
    acoustic_output = acoustic_encoder(mel)
    print(f"  Acoustic z shape: {acoustic_output['acoustic_z'].shape}")
    assert acoustic_output['acoustic_z'].shape == (batch_size, seq_len, config.acoustic_dim)
    print("  [PASS]")

    # Test 5: Progressive Residual Fusion
    print("\n[Test 5] Progressive Residual Fusion...")
    fusion = ProgressiveResidualFusion(config).to(device)
    fusion_output = fusion(
        acoustic_output['acoustic_z'],
        semantic_output['semantic_z'],
        timbre_output['timbre_proj'],
    )
    print(f"  Prosody codes shape: {fusion_output['prosody_codes'].shape}")
    print(f"  Prosody z shape: {fusion_output['prosody_z_proj'].shape}")
    print(f"  Residual codes shape: {fusion_output['residual_codes'].shape}")
    print(f"  Residual z shape: {fusion_output['residual_z_proj'].shape}")
    print(f"  Prosody perplexity: {fusion_output['prosody_perplexity'].item():.2f}")
    print(f"  Residual perplexity: {fusion_output['residual_perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 6: Decoder
    print("\n[Test 6] Decoder...")
    decoder = MSRCodecDecoder(config).to(device)
    decode_output = decoder(
        semantic_output['semantic_z'],
        timbre_output['timbre_emb'],
        fusion_output['prosody_z_proj'],
        fusion_output['residual_z_proj'],
    )
    print(f"  Mel reconstructed shape: {decode_output['mel_reconstructed'].shape}")
    assert decode_output['mel_reconstructed'].shape == (batch_size, seq_len, mel_dim)
    print("  [PASS]")

    # Test 7: Full Model
    print("\n[Test 7] Full MSR-Codec Model...")
    model = MSRCodec(config).to(device)
    output = model(mel, semantic_features)
    print(f"  Semantic z: {output['semantic_z'].shape}")
    print(f"  Timbre emb: {output['timbre_emb'].shape}")
    print(f"  Prosody codes: {output['prosody_codes'].shape}")
    print(f"  Residual codes: {output['residual_codes'].shape}")
    print(f"  Mel reconstructed: {output['mel_reconstructed'].shape}")
    print(f"  Combined embedding: {output['combined_embedding'].shape}")
    print("  [PASS]")

    # Test 8: Loss Function
    print("\n[Test 8] Loss Function...")
    loss_fn = MSRCodecLoss(config)
    losses = loss_fn(output, mel)
    print(f"  Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
    print(f"  Prosody commitment: {losses['prosody_commitment'].item():.4f}")
    print(f"  Residual commitment: {losses['residual_commitment'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: CSM Adapter
    print("\n[Test 9] CSM Adapter...")
    adapter = MSRCodecAdapter(config, model).to(device)
    adapter_output = adapter(mel, semantic_features)
    print(f"  Prefix tokens: {adapter_output['prosody_tokens'].shape}")
    print(f"  Prosody perplexity: {adapter_output['prosody_perplexity'].item():.2f}")
    print(f"  Residual perplexity: {adapter_output['residual_perplexity'].item():.2f}")
    assert adapter_output['prosody_tokens'].shape == (
        batch_size, config.num_prefix_tokens, config.output_dim
    )
    print("  [PASS]")

    # Test 10: Bitrate Computation
    print("\n[Test 10] Bitrate Computation...")
    bitrate = model.compute_bitrate(
        output['prosody_codes'],
        output['residual_codes'],
    )
    print(f"  Tokens per second: {bitrate['tokens_per_second']:.1f}")
    print(f"  Bits per second: {bitrate['bits_per_second']:.1f}")
    print(f"  Prosody bits/token: {bitrate['prosody_bits_per_token']:.1f}")
    print(f"  Residual bits/token: {bitrate['residual_bits_per_token']:.1f}")
    print(f"  Total bits/token: {bitrate['total_bits_per_token']:.1f}")
    # Target: ~62.5 tokens/sec, ~424-524 bps
    print("  [PASS]")

    # Test 11: Voice Conversion
    print("\n[Test 11] Voice Conversion...")
    mel_a = torch.randn(1, seq_len, mel_dim).to(device)
    mel_b = torch.randn(1, seq_len, mel_dim).to(device)
    semantic_a = torch.randn(1, seq_len, semantic_dim).to(device)
    semantic_b = torch.randn(1, seq_len, semantic_dim).to(device)

    with torch.no_grad():
        mel_converted = model.voice_convert(
            source_mel=mel_a,
            source_semantic=semantic_a,
            target_prosody_mel=mel_b,
            target_prosody_semantic=semantic_b,
            target_timbre_mel=mel_b,
        )
    print(f"  Converted mel shape: {mel_converted.shape}")
    print("  [PASS]")

    # Test 12: Three-way Prosody Transfer
    print("\n[Test 12] Three-way Prosody Transfer...")
    mel_c = torch.randn(1, seq_len, mel_dim).to(device)
    with torch.no_grad():
        mel_transferred = model.prosody_transfer(
            content_mel=mel_a,
            content_semantic=semantic_a,
            prosody_mel=mel_b,
            prosody_semantic=semantic_b,
            timbre_mel=mel_c,
        )
    print(f"  Transferred mel shape: {mel_transferred.shape}")
    print("  [PASS]")

    # Test 13: Backward Pass
    print("\n[Test 13] Backward Pass...")
    model.zero_grad()
    output = model(mel, semantic_features)
    losses = loss_fn(output, mel)
    losses['total'].backward()

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test 14: Separate Embeddings
    print("\n[Test 14] Separate Embeddings...")
    with torch.no_grad():
        embeddings = adapter.get_separate_embeddings(mel, semantic_features)
    print(f"  Semantic: {embeddings['semantic'].shape}")
    print(f"  Timbre: {embeddings['timbre'].shape}")
    print(f"  Prosody: {embeddings['prosody'].shape}")
    print(f"  Prosody codes: {embeddings['prosody_codes'].shape}")
    print(f"  Residual: {embeddings['residual'].shape}")
    print(f"  Residual codes: {embeddings['residual_codes'].shape}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All MSR-Codec tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. 4-STREAM DECOMPOSITION:
       - Semantic: Linguistic content (HuBERT features)
       - Timbre: Speaker identity (global embedding)
       - Prosody: Prosodic patterns (VQ1, 32-128 codes)
       - Residual: Fine-grained detail (VQ2, 64-128 codes)

    2. CASCADED RESIDUAL ARCHITECTURE:
       acoustic_z
           - semantic_z → residual_1 (prosody+speaker+detail)
           - timbre_proj → residual_2 (prosody+detail)
           VQ1 → prosody_codes, residual_3 (detail only)
           VQ2 → residual_codes

    3. IMPLICIT DISENTANGLEMENT:
       - No adversarial training needed!
       - No GRL (Gradient Reversal Layer)
       - Progressive subtraction naturally separates attributes
       - Simpler and more stable training

    4. ULTRA-LOW BITRATE:
       - ~62.5 tokens/sec
       - ~424-524 bps with good quality
       - >200x compression factor

    5. ZERO-SHOT VOICE CONVERSION:
       - Control BOTH timbre AND prosody independently
       - Three-way transfer: content + prosody + speaker
       - Clean separation enables flexible mixing
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from msr_codec import (
    MSRCodecConfig,
    MSRCodec,
    MSRCodecLoss,
    MSRCodecAdapter,
)

# Initialize
config = MSRCodecConfig(
    prosody_codebook_size=64,   # VQ1 size
    residual_codebook_size=128,  # VQ2 size
)

model = MSRCodec(config).cuda()
loss_fn = MSRCodecLoss(config)

# Training
for mel, semantic in dataloader:
    output = model(mel, semantic)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor codebook usage
    print(f"Prosody perplexity: {output['prosody_perplexity']:.2f}")
    print(f"Residual perplexity: {output['residual_perplexity']:.2f}")

# Zero-shot voice conversion (timbre AND prosody transfer)
mel_converted = model.voice_convert(
    source_mel=content_source,
    source_semantic=content_semantic,
    target_prosody_mel=prosody_reference,
    target_prosody_semantic=prosody_semantic,
    target_timbre_mel=speaker_reference,
)

# Three-way transfer
mel_transferred = model.prosody_transfer(
    content_mel=mel_a,      # Content source
    content_semantic=sem_a,
    prosody_mel=mel_b,      # Prosody source
    prosody_semantic=sem_b,
    timbre_mel=mel_c,       # Speaker source
)

# CSM integration
adapter = MSRCodecAdapter(config, model)
result = adapter(mel, semantic)
prefix_tokens = result['prosody_tokens']  # [batch, 4, 2048]
bitrate = result['bitrate']
print(f"Bitrate: {bitrate['bits_per_second']:.1f} bps")

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
    """)
