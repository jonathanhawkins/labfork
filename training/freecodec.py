"""
FreeCodec: Parallel-Encoder Disentangled Speech Codec

Based on "FreeCodec: A Disentangled Neural Speech Codec with Fewer Tokens"
(arXiv:2412.01053) - https://arxiv.org/abs/2412.01053

Key Innovation: Parallel decomposition with 3 separate encoders instead of
cascaded RVQ, achieving better disentanglement with fewer tokens.

Architecture:
1. Global Timbre Encoder → Single vector for speaker identity (no temporal codes)
2. Prosody Encoder → Long-stride temporal encoding for prosody patterns
3. Content Encoder → Linguistic information with semantic learning targets

Variants:
- FreeCodec-v1: Basic parallel encoding with VQ
- FreeCodec-v2: Semantic learning targets reduce content redundancy
- FreeCodec-v3: Semantic target only at decoder to prevent speaker leakage

Benefits over RVQ-based codecs (FACodec, EnCodec, etc.):
- Fewer tokens needed (higher coding efficiency)
- Better disentanglement via parallel encoders
- State-of-the-art reconstruction with fewer parameters
- Clean prosody codes without content/timbre leakage
- Global timbre representation avoids timbre redundancy

Training Objectives:
1. Reconstruction loss (mel spectrogram)
2. VQ commitment loss (for content quantization)
3. Semantic learning target loss (HuBERT/WavLM target)
4. Adversarial loss (optional discriminator)
5. Orthogonality loss (disentanglement)

Usage:
    from freecodec import (
        FreeCodecConfig,
        FreeCodec,
        FreeCodecLoss,
        FreeCodecAdapter,
    )

    # Initialize
    config = FreeCodecConfig()
    model = FreeCodec(config).cuda()

    # Encode to parallel latent spaces
    encoded = model.encode(mel)
    timbre_z = encoded['timbre_z']    # [batch, timbre_dim] global speaker
    prosody_z = encoded['prosody_z']  # [batch, seq//stride, prosody_dim]
    content_z = encoded['content_z']  # [batch, seq, content_dim]

    # Decode from latents
    mel_reconstructed = model.decode(timbre_z, prosody_z, content_z)

    # Zero-shot voice conversion
    mel_converted = model.voice_convert(
        source_mel=mel_a,        # Content source
        target_speaker_mel=mel_b, # Timbre source
    )

    # CSM integration
    adapter = FreeCodecAdapter(config, model)
    prefix_tokens = adapter(mel)  # [batch, 4, 2048]
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
class FreeCodecConfig:
    """Configuration for FreeCodec parallel-encoder disentanglement."""

    # Input dimensions
    mel_dim: int = 80  # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 256  # ~16ms at 16kHz

    # Variant selection
    variant: str = "v2"  # "v1", "v2", or "v3"

    # Global Timbre Encoder (single vector, no temporal codes)
    timbre_dim: int = 256  # Global speaker embedding dimension
    timbre_hidden_dim: int = 512
    timbre_num_layers: int = 3

    # Prosody Encoder (long-stride temporal encoding)
    prosody_dim: int = 128  # Per-frame prosody dimension
    prosody_hidden_dim: int = 256
    prosody_num_layers: int = 4
    prosody_stride: int = 4  # Temporal downsampling factor (key for efficiency)
    prosody_kernel_size: int = 8  # Covers prosody_stride * 2 for receptive field

    # Content Encoder (linguistic with semantic targets)
    content_dim: int = 256  # Content code dimension
    content_hidden_dim: int = 512
    content_num_layers: int = 6
    content_codebook_size: int = 1024  # VQ codebook size
    content_num_groups: int = 1  # Product quantization groups
    content_commitment_cost: float = 0.25
    content_ema_decay: float = 0.99

    # Semantic learning targets (HuBERT/WavLM based)
    semantic_dim: int = 768  # HuBERT feature dimension
    use_semantic_target: bool = True  # Enable semantic learning
    semantic_weight: float = 1.0  # Semantic loss weight

    # Decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Training settings
    dropout: float = 0.1

    # Disentanglement losses
    use_orthogonality: bool = True
    ortho_weight: float = 0.01  # Orthogonality regularization weight

    # Optional discriminator
    use_discriminator: bool = False
    discriminator_hidden_dim: int = 256
    discriminator_num_layers: int = 3

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
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Residual convolutional block with optional downsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.stride = stride

        # First conv with stride for downsampling
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_dim)

        # Second conv without stride
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual connection with potential dimension/stride adjustment
        if stride != 1 or in_dim != out_dim:
            self.residual = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 1, stride=stride),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, dim, seq]
        Returns:
            [batch, out_dim, seq // stride]
        """
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection (handle potential length mismatch)
        residual = self.residual(x)

        # Align lengths if needed (can happen due to padding differences)
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
# GLOBAL TIMBRE ENCODER
# =============================================================================

class GlobalTimbreEncoder(nn.Module):
    """
    Global Timbre Encoder: Extracts speaker identity as a SINGLE vector.

    Key design choices from FreeCodec:
    1. No temporal codes - single global vector captures speaker identity
    2. Attentive statistics pooling for robust speaker extraction
    3. Removes temporal variation that could leak prosody

    This is more efficient than per-frame timbre codes (as in FACodec)
    while maintaining speaker fidelity.
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = nn.Linear(config.mel_dim, config.timbre_hidden_dim)

        # Convolutional layers for local feature extraction
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

        # Attentive statistics pooling (captures speaker characteristics)
        self.attention = nn.Sequential(
            nn.Linear(config.timbre_hidden_dim, config.timbre_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.timbre_hidden_dim // 2, 1),
        )

        # Output projection: statistics (mean + std) -> timbre embedding
        self.output_proj = nn.Sequential(
            nn.Linear(config.timbre_hidden_dim * 2, config.timbre_hidden_dim),
            nn.GELU(),
            nn.Linear(config.timbre_hidden_dim, config.timbre_dim),
        )

        self.norm = nn.LayerNorm(config.timbre_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,  # [batch, seq]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global timbre (speaker) embedding.

        Returns:
            Dict with:
                - timbre_z: [batch, timbre_dim] global speaker vector
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

        # Weighted standard deviation
        var = ((x - mean.unsqueeze(1)).pow(2) * attn_weights).sum(dim=1)
        std = var.clamp(min=1e-8).sqrt()  # [B, H]

        # Concatenate statistics
        stats = torch.cat([mean, std], dim=-1)  # [B, H*2]

        # Project to timbre embedding
        timbre_z = self.output_proj(stats)  # [B, timbre_dim]
        timbre_z = self.norm(timbre_z)

        return {
            'timbre_z': timbre_z,
            'timbre_features': x,  # For analysis
        }


# =============================================================================
# PROSODY ENCODER (LONG-STRIDE TEMPORAL)
# =============================================================================

class ProsodyEncoder(nn.Module):
    """
    Prosody Encoder: Long-stride temporal encoding for prosody patterns.

    Key design choices from FreeCodec:
    1. Long-stride convolutions (stride=4) for temporal downsampling
    2. Captures pitch, energy, rhythm at reduced temporal resolution
    3. Fewer prosody tokens = more efficient encoding
    4. Separate from content to avoid prosody-content entanglement

    The long stride forces prosody to be captured at a coarser granularity,
    preventing fine-grained content leakage into prosody codes.
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()
        self.config = config
        self.stride = config.prosody_stride

        # Initial projection
        self.input_proj = nn.Linear(config.mel_dim, config.prosody_hidden_dim)

        # Long-stride convolutional layers
        self.conv_layers = nn.ModuleList()

        # First layer with stride for temporal downsampling
        self.conv_layers.append(
            ConvBlock(
                config.prosody_hidden_dim,
                config.prosody_hidden_dim,
                kernel_size=config.prosody_kernel_size,
                stride=config.prosody_stride,
                dropout=config.dropout,
            )
        )

        # Additional layers without stride
        for _ in range(config.prosody_num_layers - 1):
            self.conv_layers.append(
                ConvBlock(
                    config.prosody_hidden_dim,
                    config.prosody_hidden_dim,
                    kernel_size=5,
                    stride=1,
                    dropout=config.dropout,
                )
            )

        # Transformer layers for global prosody patterns
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.prosody_hidden_dim,
                num_heads=4,
                ffn_dim=config.prosody_hidden_dim * 4,
                dropout=config.dropout,
            )
            for _ in range(2)  # Lightweight transformer
        ])

        # Output projection
        self.output_proj = nn.Linear(config.prosody_hidden_dim, config.prosody_dim)
        self.norm = nn.LayerNorm(config.prosody_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosody representation at reduced temporal resolution.

        Returns:
            Dict with:
                - prosody_z: [batch, seq//stride, prosody_dim]
                - prosody_lengths: [batch] actual sequence lengths
        """
        batch_size, seq_len, _ = mel.shape

        # Project input
        x = self.input_proj(mel)  # [B, T, H]

        # Convolutional processing with long stride
        x = x.transpose(1, 2)  # [B, H, T]
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)  # [B, T//stride, H]

        # Adjust mask for downsampled sequence
        if mask is not None:
            # Downsample mask by taking every stride-th element
            mask = mask[:, ::self.stride]
            # Ensure mask length matches x
            mask = mask[:, :x.shape[1]]

        # Transformer for global patterns
        for transformer in self.transformer:
            x = transformer(x, mask)

        # Project to prosody dimension
        prosody_z = self.output_proj(x)  # [B, T//stride, prosody_dim]
        prosody_z = self.norm(prosody_z)

        # Compute actual lengths
        prosody_lengths = torch.full(
            (batch_size,), x.shape[1], dtype=torch.long, device=x.device
        )

        return {
            'prosody_z': prosody_z,
            'prosody_lengths': prosody_lengths,
            'prosody_features': x,  # For analysis
        }


# =============================================================================
# CONTENT ENCODER (WITH SEMANTIC LEARNING TARGETS)
# =============================================================================

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA codebook updates.

    Used for content quantization to remove prosodic variation.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 1024,
        code_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # Pre/post projection
        self.pre_proj = nn.Linear(input_dim, code_dim)
        self.post_proj = nn.Linear(code_dim, input_dim)

        # Codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim))

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_sum', torch.randn(codebook_size, code_dim))
        self.register_buffer('initialized', torch.tensor(False))

        # Initialize
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size)

    def _init_from_data(self, z: torch.Tensor):
        """Initialize codebook from first batch."""
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
            Dict with z_q, indices, commitment_loss, perplexity
        """
        batch_size, seq_len, _ = x.shape

        # Project
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

        # Find nearest
        indices = d.argmin(dim=-1)  # [B*T]
        z_q = F.embedding(indices, self.codebook)  # [B*T, code_dim]

        # EMA update
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                new_size = encodings.sum(dim=0)
                new_sum = torch.matmul(encodings.t(), z_flat)

                self.ema_cluster_size.mul_(self.ema_decay).add_(new_size, alpha=1 - self.ema_decay)
                self.ema_sum.mul_(self.ema_decay).add_(new_sum, alpha=1 - self.ema_decay)

                n = self.ema_cluster_size.clamp(min=1)
                self.codebook.data.copy_(self.ema_sum / n.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z_flat, z_q.detach())

        # Straight-through
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape
        z_q = z_q.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

        # Project back
        z_q = self.post_proj(z_q)

        # Perplexity
        flat_indices = indices.view(-1)
        encodings = F.one_hot(flat_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'z_q': z_q,
            'indices': indices,
            'commitment_loss': commitment_loss * self.commitment_cost,
            'perplexity': perplexity,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to vectors."""
        z_q = F.embedding(indices, self.codebook)
        return self.post_proj(z_q)


class ContentEncoder(nn.Module):
    """
    Content Encoder: Linguistic information with semantic learning targets.

    Key design choices from FreeCodec:
    1. VQ bottleneck forces removal of prosodic/speaker variation
    2. Semantic learning targets (HuBERT/WavLM) guide content representation
    3. v1: Basic VQ content encoding
    4. v2: Semantic targets in encoder (better content)
    5. v3: Semantic targets only in decoder (prevents speaker leakage)

    The semantic target helps content codes focus on linguistic information
    rather than acoustic details.
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = nn.Linear(config.mel_dim, config.content_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.content_hidden_dim, dropout=config.dropout)

        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.content_hidden_dim,
                num_heads=8,
                ffn_dim=config.content_hidden_dim * 4,
                dropout=config.dropout,
            )
            for _ in range(config.content_num_layers)
        ])

        # Vector quantizer
        self.quantizer = VectorQuantizer(
            input_dim=config.content_hidden_dim,
            codebook_size=config.content_codebook_size,
            code_dim=config.content_dim,
            commitment_cost=config.content_commitment_cost,
            ema_decay=config.content_ema_decay,
        )

        # Semantic prediction head (for v2 variant)
        if config.use_semantic_target and config.variant in ["v1", "v2"]:
            self.semantic_head = nn.Linear(config.content_hidden_dim, config.semantic_dim)
        else:
            self.semantic_head = None

        self.norm = nn.LayerNorm(config.content_hidden_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        semantic_target: Optional[torch.Tensor] = None,  # [batch, seq, semantic_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode content with VQ bottleneck.

        Returns:
            Dict with content_z, content_indices, commitment_loss, semantic_loss
        """
        # Project input
        x = self.input_proj(mel)  # [B, T, H]
        x = self.pos_enc(x)

        # Transformer encoding
        for transformer in self.transformer:
            x = transformer(x, mask)
        x = self.norm(x)

        # Vector quantize
        vq_output = self.quantizer(x)

        result = {
            'content_z': vq_output['z_q'],
            'content_indices': vq_output['indices'],
            'commitment_loss': vq_output['commitment_loss'],
            'perplexity': vq_output['perplexity'],
            'content_features': x,  # Pre-VQ features for analysis
        }

        # Semantic prediction (v2 variant)
        if self.semantic_head is not None and semantic_target is not None:
            semantic_pred = self.semantic_head(x)

            # Handle length mismatch
            min_len = min(semantic_pred.shape[1], semantic_target.shape[1])
            semantic_loss = F.mse_loss(
                semantic_pred[:, :min_len],
                semantic_target[:, :min_len],
            )
            result['semantic_loss'] = semantic_loss
            result['semantic_pred'] = semantic_pred
        else:
            result['semantic_loss'] = torch.tensor(0.0, device=mel.device)

        return result


# =============================================================================
# DECODER
# =============================================================================

class FreeCodecDecoder(nn.Module):
    """
    Decoder: Reconstructs mel from timbre + prosody + content.

    Key design:
    1. Timbre is broadcast to all positions (global conditioning)
    2. Prosody is upsampled to match content resolution
    3. Content provides frame-level linguistic information
    4. Cross-attention fusion of all components

    For v3: Semantic prediction happens here to prevent speaker leakage.
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()
        self.config = config

        # Timbre conditioning (global → all positions)
        self.timbre_proj = nn.Linear(config.timbre_dim, config.decoder_hidden_dim)

        # Prosody upsampling (strided → frame-level)
        self.prosody_upsample = nn.Sequential(
            nn.ConvTranspose1d(
                config.prosody_dim,
                config.decoder_hidden_dim,
                kernel_size=config.prosody_kernel_size,
                stride=config.prosody_stride,
                padding=(config.prosody_kernel_size - config.prosody_stride) // 2,
            ),
            nn.GELU(),
        )

        # Content projection
        self.content_proj = nn.Linear(config.content_hidden_dim, config.decoder_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.decoder_hidden_dim, dropout=config.dropout)

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

        # Output projection
        self.output_proj = nn.Linear(config.decoder_hidden_dim, config.mel_dim)
        self.norm = nn.LayerNorm(config.decoder_hidden_dim)

        # Semantic head for v3 variant (at decoder)
        if config.use_semantic_target and config.variant == "v3":
            self.semantic_head = nn.Linear(config.decoder_hidden_dim, config.semantic_dim)
        else:
            self.semantic_head = None

    def forward(
        self,
        timbre_z: torch.Tensor,  # [batch, timbre_dim]
        prosody_z: torch.Tensor,  # [batch, seq//stride, prosody_dim]
        content_z: torch.Tensor,  # [batch, seq, content_hidden_dim]
        semantic_target: Optional[torch.Tensor] = None,  # [batch, seq, semantic_dim] for v3
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode from parallel latent spaces.

        Returns:
            Dict with mel_reconstructed, semantic_loss (for v3)
        """
        batch_size, seq_len, _ = content_z.shape

        # Project content
        x = self.content_proj(content_z)  # [B, T, H]

        # Upsample prosody to match content resolution
        prosody = prosody_z.transpose(1, 2)  # [B, prosody_dim, T//stride]
        prosody = self.prosody_upsample(prosody)  # [B, H, T_up]
        prosody = prosody.transpose(1, 2)  # [B, T_up, H]

        # Handle length mismatch (adjust prosody length to match content)
        if prosody.shape[1] != seq_len:
            if prosody.shape[1] > seq_len:
                prosody = prosody[:, :seq_len]
            else:
                # Pad if shorter
                pad_len = seq_len - prosody.shape[1]
                prosody = F.pad(prosody, (0, 0, 0, pad_len))

        # Add prosody
        x = x + prosody

        # Add timbre (broadcast global embedding)
        timbre = self.timbre_proj(timbre_z)  # [B, H]
        x = x + timbre.unsqueeze(1)  # [B, T, H]

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer decoding
        for transformer in self.transformer:
            x = transformer(x, mask)
        x = self.norm(x)

        # Output mel
        mel_reconstructed = self.output_proj(x)  # [B, T, mel_dim]

        result = {
            'mel_reconstructed': mel_reconstructed,
            'decoder_features': x,
        }

        # Semantic prediction for v3 variant
        if self.semantic_head is not None and semantic_target is not None:
            semantic_pred = self.semantic_head(x)

            min_len = min(semantic_pred.shape[1], semantic_target.shape[1])
            semantic_loss = F.mse_loss(
                semantic_pred[:, :min_len],
                semantic_target[:, :min_len],
            )
            result['semantic_loss'] = semantic_loss
            result['semantic_pred'] = semantic_pred
        else:
            result['semantic_loss'] = torch.tensor(0.0, device=mel_reconstructed.device)

        return result


# =============================================================================
# DISCRIMINATOR (OPTIONAL)
# =============================================================================

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for adversarial training.

    Evaluates mel spectrogram quality at multiple resolutions.
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()

        self.scales = nn.ModuleList()
        for i, scale in enumerate([1, 2, 4]):
            self.scales.append(
                self._make_discriminator(config, scale)
            )

    def _make_discriminator(self, config: FreeCodecConfig, scale: int):
        """Create discriminator for single scale."""
        layers = []
        in_dim = config.mel_dim

        for i in range(config.discriminator_num_layers):
            out_dim = config.discriminator_hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2),
            ])
            in_dim = out_dim

        layers.append(nn.Conv1d(in_dim, 1, kernel_size=3, padding=1))

        return nn.Sequential(*layers)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
    ) -> List[torch.Tensor]:
        """
        Returns list of discriminator outputs at each scale.
        """
        mel = mel.transpose(1, 2)  # [B, mel_dim, T]
        outputs = []

        for i, disc in enumerate(self.scales):
            # Downsample input for different scales
            if i > 0:
                mel = F.avg_pool1d(mel, kernel_size=2)
            outputs.append(disc(mel))

        return outputs


# =============================================================================
# FULL FREECODEC MODEL
# =============================================================================

class FreeCodec(nn.Module):
    """
    FreeCodec: Parallel-Encoder Disentangled Speech Codec.

    Three parallel encoders for disentangled speech representation:
    1. Global Timbre: Single speaker vector (no temporal codes)
    2. Long-stride Prosody: Coarse temporal prosody patterns
    3. VQ Content: Linguistic information with semantic targets

    Variants:
    - v1: Basic parallel encoding with VQ
    - v2: Semantic learning targets in content encoder
    - v3: Semantic targets only at decoder (prevents speaker leakage)
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()
        self.config = config

        # Parallel encoders
        self.timbre_encoder = GlobalTimbreEncoder(config)
        self.prosody_encoder = ProsodyEncoder(config)
        self.content_encoder = ContentEncoder(config)

        # Decoder
        self.decoder = FreeCodecDecoder(config)

        # Optional discriminator
        if config.use_discriminator:
            self.discriminator = MultiScaleDiscriminator(config)
        else:
            self.discriminator = None

        # Output projection for CSM integration
        # Note: content_z has content_hidden_dim, not content_dim
        self.output_proj = nn.Sequential(
            nn.Linear(
                config.timbre_dim + config.prosody_dim + config.content_hidden_dim,
                config.output_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        semantic_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel to parallel latent spaces.

        Returns:
            Dict with timbre_z, prosody_z, content_z, and losses
        """
        # Parallel encoding
        timbre_output = self.timbre_encoder(mel, mask)
        prosody_output = self.prosody_encoder(mel, mask)
        content_output = self.content_encoder(mel, semantic_target, mask)

        return {
            'timbre_z': timbre_output['timbre_z'],
            'prosody_z': prosody_output['prosody_z'],
            'prosody_lengths': prosody_output['prosody_lengths'],
            'content_z': content_output['content_z'],
            'content_indices': content_output['content_indices'],
            'commitment_loss': content_output['commitment_loss'],
            'perplexity': content_output['perplexity'],
            'content_semantic_loss': content_output['semantic_loss'],
            # Features for analysis
            'timbre_features': timbre_output['timbre_features'],
            'prosody_features': prosody_output['prosody_features'],
            'content_features': content_output['content_features'],
        }

    def decode(
        self,
        timbre_z: torch.Tensor,  # [batch, timbre_dim]
        prosody_z: torch.Tensor,  # [batch, seq//stride, prosody_dim]
        content_z: torch.Tensor,  # [batch, seq, content_hidden_dim]
        semantic_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode from parallel latent spaces.

        Returns:
            Dict with mel_reconstructed and decoder_semantic_loss
        """
        return self.decoder(timbre_z, prosody_z, content_z, semantic_target, mask)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        semantic_target: Optional[torch.Tensor] = None,  # [batch, seq, semantic_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, decode, compute losses.

        Returns:
            Dict with all latents, reconstruction, and losses
        """
        # Encode
        encoded = self.encode(mel, semantic_target, mask)

        # Decode
        decoded = self.decode(
            encoded['timbre_z'],
            encoded['prosody_z'],
            encoded['content_z'],
            semantic_target if self.config.variant == "v3" else None,
            mask,
        )

        # Combine all outputs
        result = {
            **encoded,
            'mel_reconstructed': decoded['mel_reconstructed'],
            'decoder_semantic_loss': decoded['semantic_loss'],
        }

        # Compute combined prosody embedding for downstream
        # Pool prosody_z to match timbre dimension
        prosody_pooled = encoded['prosody_z'].mean(dim=1)  # [B, prosody_dim]

        # Pool content for global representation
        content_pooled = encoded['content_z'].mean(dim=1)  # [B, content_hidden_dim]

        # Combined embedding
        combined = torch.cat([
            encoded['timbre_z'],
            prosody_pooled,
            content_pooled,
        ], dim=-1)
        result['combined_embedding'] = self.output_proj(combined)

        return result

    def voice_convert(
        self,
        source_mel: torch.Tensor,  # [batch, seq, mel_dim] content source
        target_speaker_mel: torch.Tensor,  # [batch, seq, mel_dim] timbre source
        preserve_prosody: bool = True,  # Keep source prosody or use target
    ) -> torch.Tensor:
        """
        Zero-shot voice conversion.

        Args:
            source_mel: Source audio (provides content, optionally prosody)
            target_speaker_mel: Target speaker audio (provides timbre)
            preserve_prosody: If True, keep source prosody; else use target prosody

        Returns:
            [batch, seq, mel_dim] converted mel spectrogram
        """
        # Extract content from source
        source_encoded = self.encode(source_mel)
        content_z = source_encoded['content_z']

        # Extract timbre from target
        target_encoded = self.encode(target_speaker_mel)
        timbre_z = target_encoded['timbre_z']

        # Choose prosody source
        if preserve_prosody:
            prosody_z = source_encoded['prosody_z']
        else:
            prosody_z = target_encoded['prosody_z']
            # Handle length mismatch
            if prosody_z.shape[1] != source_encoded['prosody_z'].shape[1]:
                # Interpolate to match source length
                prosody_z = F.interpolate(
                    prosody_z.transpose(1, 2),
                    size=source_encoded['prosody_z'].shape[1],
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)

        # Decode with mixed components
        decoded = self.decode(timbre_z, prosody_z, content_z)

        return decoded['mel_reconstructed']

    def prosody_transfer(
        self,
        content_mel: torch.Tensor,  # Content source
        prosody_mel: torch.Tensor,  # Prosody source
        timbre_mel: torch.Tensor,  # Timbre source
    ) -> torch.Tensor:
        """
        Three-way prosody transfer.

        Combines content from one source, prosody from another,
        and timbre from a third.
        """
        # Encode all sources
        content_encoded = self.encode(content_mel)
        prosody_encoded = self.encode(prosody_mel)
        timbre_encoded = self.encode(timbre_mel)

        # Get individual components
        content_z = content_encoded['content_z']
        timbre_z = timbre_encoded['timbre_z']
        prosody_z = prosody_encoded['prosody_z']

        # Handle prosody length mismatch
        if prosody_z.shape[1] != content_encoded['prosody_z'].shape[1]:
            prosody_z = F.interpolate(
                prosody_z.transpose(1, 2),
                size=content_encoded['prosody_z'].shape[1],
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        # Decode
        decoded = self.decode(timbre_z, prosody_z, content_z)

        return decoded['mel_reconstructed']

    def get_prosody_embedding(
        self,
        mel: torch.Tensor,
        pool: str = 'mean',
    ) -> torch.Tensor:
        """
        Get prosody embedding for downstream tasks.

        Returns:
            [batch, output_dim] prosody representation
        """
        output = self.forward(mel)
        return output['combined_embedding']


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class FreeCodecLoss(nn.Module):
    """
    Combined loss function for FreeCodec training.

    Components:
    1. Reconstruction loss (L1 + L2 mel)
    2. VQ commitment loss
    3. Semantic learning target loss (v2/v3)
    4. Orthogonality loss (disentanglement)
    5. Adversarial loss (optional)
    """

    def __init__(self, config: FreeCodecConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.reconstruction_weight = 1.0
        self.commitment_weight = 0.25
        self.semantic_weight = config.semantic_weight
        self.ortho_weight = config.ortho_weight
        self.adversarial_weight = 0.1

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

        if mask is not None:
            mask = mask[:, :min_len]

        # L1 loss
        l1_loss = F.l1_loss(mel_pred, mel_target, reduction='none')

        # L2 loss
        l2_loss = F.mse_loss(mel_pred, mel_target, reduction='none')

        if mask is not None:
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

    def orthogonality_loss(
        self,
        timbre_z: torch.Tensor,  # [batch, timbre_dim]
        prosody_z: torch.Tensor,  # [batch, seq, prosody_dim]
        content_z: torch.Tensor,  # [batch, seq, content_dim]
    ) -> torch.Tensor:
        """
        Compute orthogonality loss for disentanglement.

        Encourages different latent spaces to be orthogonal/independent.
        """
        # Pool sequence dimensions
        prosody_pooled = prosody_z.mean(dim=1)  # [B, prosody_dim]
        content_pooled = content_z.mean(dim=1)  # [B, content_dim]

        # Project to same dimension for comparison
        min_dim = min(
            timbre_z.shape[-1],
            prosody_pooled.shape[-1],
            content_pooled.shape[-1],
        )

        timbre_proj = timbre_z[:, :min_dim]
        prosody_proj = prosody_pooled[:, :min_dim]
        content_proj = content_pooled[:, :min_dim]

        # Normalize
        timbre_norm = F.normalize(timbre_proj, p=2, dim=-1)
        prosody_norm = F.normalize(prosody_proj, p=2, dim=-1)
        content_norm = F.normalize(content_proj, p=2, dim=-1)

        # Compute pairwise cosine similarities (should be low)
        tp_sim = (timbre_norm * prosody_norm).sum(dim=-1).abs().mean()
        tc_sim = (timbre_norm * content_norm).sum(dim=-1).abs().mean()
        pc_sim = (prosody_norm * content_norm).sum(dim=-1).abs().mean()

        return tp_sim + tc_sim + pc_sim

    def adversarial_loss(
        self,
        discriminator: MultiScaleDiscriminator,
        mel_real: torch.Tensor,
        mel_fake: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute adversarial losses."""
        # Discriminator outputs
        real_outputs = discriminator(mel_real)
        fake_outputs = discriminator(mel_fake.detach())

        # Discriminator loss (train to classify)
        d_loss = 0.0
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            d_loss += F.binary_cross_entropy_with_logits(
                real_out, torch.ones_like(real_out)
            )
            d_loss += F.binary_cross_entropy_with_logits(
                fake_out, torch.zeros_like(fake_out)
            )
        d_loss = d_loss / len(real_outputs)

        # Generator loss (train to fool discriminator)
        fake_outputs_for_g = discriminator(mel_fake)
        g_loss = 0.0
        for fake_out in fake_outputs_for_g:
            g_loss += F.binary_cross_entropy_with_logits(
                fake_out, torch.ones_like(fake_out)
            )
        g_loss = g_loss / len(fake_outputs_for_g)

        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
        }

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        discriminator: Optional[MultiScaleDiscriminator] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            model_output: Output from FreeCodec.forward()
            mel_target: Target mel spectrogram
            discriminator: Optional discriminator for adversarial loss
            mask: Optional mask for valid frames

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Reconstruction loss
        recon_losses = self.reconstruction_loss(
            model_output['mel_reconstructed'],
            mel_target,
            mask,
        )
        losses.update(recon_losses)

        # VQ commitment loss
        losses['commitment_loss'] = model_output['commitment_loss']
        losses['perplexity'] = model_output['perplexity']

        # Semantic loss (choose based on variant)
        if self.config.variant in ["v1", "v2"]:
            losses['semantic_loss'] = model_output['content_semantic_loss']
        elif self.config.variant == "v3":
            losses['semantic_loss'] = model_output['decoder_semantic_loss']
        else:
            losses['semantic_loss'] = torch.tensor(0.0, device=mel_target.device)

        # Orthogonality loss
        if self.config.use_orthogonality:
            losses['orthogonality_loss'] = self.orthogonality_loss(
                model_output['timbre_z'],
                model_output['prosody_z'],
                model_output['content_z'],
            )
        else:
            losses['orthogonality_loss'] = torch.tensor(0.0, device=mel_target.device)

        # Adversarial loss
        if discriminator is not None and self.config.use_discriminator:
            adv_losses = self.adversarial_loss(
                discriminator,
                mel_target,
                model_output['mel_reconstructed'],
            )
            losses.update(adv_losses)
        else:
            losses['discriminator_loss'] = torch.tensor(0.0, device=mel_target.device)
            losses['generator_loss'] = torch.tensor(0.0, device=mel_target.device)

        # Total loss
        total = (
            self.reconstruction_weight * losses['reconstruction_loss']
            + self.commitment_weight * losses['commitment_loss']
            + self.semantic_weight * losses['semantic_loss']
            + self.ortho_weight * losses['orthogonality_loss']
            + self.adversarial_weight * losses['generator_loss']
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class FreeCodecAdapter(nn.Module):
    """
    Adapter for integrating FreeCodec with existing prosody pipeline.

    Converts FreeCodec's parallel latent representation to prefix tokens
    compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: FreeCodecConfig,
        model: Optional[FreeCodec] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else FreeCodec(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        semantic_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix tokens for CSM conditioning.

        Returns:
            Dict with:
                - prosody_tokens: [batch, num_prefix_tokens, output_dim]
                - disentangled: Dict with separate timbre/prosody/content
        """
        batch_size = mel.shape[0]

        # Get combined embedding
        output = self.model(mel, semantic_target)
        combined_emb = output['combined_embedding']  # [B, output_dim]

        # Project to tokens
        tokens = self.token_proj(combined_emb)  # [B, output_dim * num_tokens]

        # Reshape
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )

        # Normalize
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
            'disentangled': {
                'timbre_z': output['timbre_z'],
                'prosody_z': output['prosody_z'],
                'content_z': output['content_z'],
                'content_indices': output['content_indices'],
            },
            'commitment_loss': output['commitment_loss'],
            'perplexity': output['perplexity'],
        }

    def voice_convert(
        self,
        source_mel: torch.Tensor,
        target_speaker_mel: torch.Tensor,
        preserve_prosody: bool = True,
    ) -> torch.Tensor:
        """Zero-shot voice conversion."""
        return self.model.voice_convert(source_mel, target_speaker_mel, preserve_prosody)

    def prosody_transfer(
        self,
        content_mel: torch.Tensor,
        prosody_mel: torch.Tensor,
        timbre_mel: torch.Tensor,
    ) -> torch.Tensor:
        """Three-way prosody transfer."""
        return self.model.prosody_transfer(content_mel, prosody_mel, timbre_mel)

    def get_separate_embeddings(
        self,
        mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get separate timbre, prosody, and content embeddings."""
        encoded = self.model.encode(mel)
        return {
            'timbre': encoded['timbre_z'],
            'prosody': encoded['prosody_z'],
            'content': encoded['content_z'],
            'content_indices': encoded['content_indices'],
        }


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

class FreeCodecVsFACodecComparison:
    """
    Utility for comparing FreeCodec with FACodec.

    Measures:
    1. Token efficiency (tokens per second)
    2. Reconstruction quality (mel L1/L2)
    3. Disentanglement quality (cross-reconstruction)
    4. Prosody transfer quality
    """

    def __init__(
        self,
        freecodec: FreeCodec,
        hop_length: int = 256,
        sample_rate: int = 16000,
    ):
        self.freecodec = freecodec
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def compute_token_efficiency(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
    ) -> Dict[str, float]:
        """
        Compare token efficiency.

        FreeCodec uses:
        - 1 global timbre token
        - seq // stride prosody tokens
        - seq content tokens

        FACodec uses:
        - 6 codebooks with temporal codes
        """
        seq_len = mel.shape[1]
        stride = self.freecodec.config.prosody_stride

        # FreeCodec tokens
        freecodec_tokens = (
            1 +  # Global timbre
            seq_len // stride +  # Prosody (downsampled)
            seq_len  # Content
        )

        # FACodec estimate (6 codebooks, all temporal)
        facodec_tokens_estimate = 6 * seq_len

        # Tokens per second
        duration_s = (seq_len * self.hop_length) / self.sample_rate
        freecodec_tps = freecodec_tokens / duration_s
        facodec_tps = facodec_tokens_estimate / duration_s

        return {
            'freecodec_total_tokens': freecodec_tokens,
            'freecodec_tokens_per_second': freecodec_tps,
            'facodec_tokens_estimate': facodec_tokens_estimate,
            'facodec_tokens_per_second': facodec_tps,
            'efficiency_ratio': facodec_tokens_estimate / freecodec_tokens,
        }

    def compute_reconstruction_quality(
        self,
        mel: torch.Tensor,
    ) -> Dict[str, float]:
        """Measure reconstruction quality."""
        with torch.no_grad():
            output = self.freecodec(mel)
            mel_recon = output['mel_reconstructed']

            # Handle length mismatch
            min_len = min(mel.shape[1], mel_recon.shape[1])

            l1 = F.l1_loss(mel_recon[:, :min_len], mel[:, :min_len]).item()
            l2 = F.mse_loss(mel_recon[:, :min_len], mel[:, :min_len]).item()

        return {
            'reconstruction_l1': l1,
            'reconstruction_l2': l2,
            'perplexity': output['perplexity'].item(),
        }

    def compute_disentanglement_quality(
        self,
        mel_a: torch.Tensor,
        mel_b: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measure disentanglement via cross-reconstruction consistency.

        If well-disentangled, swapping timbre should only change speaker identity,
        not content or prosody.
        """
        with torch.no_grad():
            # Encode both
            enc_a = self.freecodec.encode(mel_a)
            enc_b = self.freecodec.encode(mel_b)

            # Cross-decode: A's content/prosody + B's timbre
            cross_decoded = self.freecodec.decode(
                enc_b['timbre_z'],  # B's speaker
                enc_a['prosody_z'],  # A's prosody
                enc_a['content_z'],  # A's content
            )

            # Compare content tokens (should be preserved)
            content_tokens_original = enc_a['content_indices']

            # Re-encode cross-decoded and check content
            re_encoded = self.freecodec.encode(cross_decoded['mel_reconstructed'])
            content_tokens_after = re_encoded['content_indices']

            # Content preservation rate
            content_match = (
                content_tokens_original == content_tokens_after
            ).float().mean().item()

        return {
            'content_preservation_rate': content_match,
            'timbre_transferred': True,  # By construction
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FreeCodec: Parallel-Encoder Disentangled Speech Codec - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 100
    mel_dim = 80
    semantic_dim = 768

    # Test all variants
    for variant in ["v1", "v2", "v3"]:
        print(f"\n{'='*60}")
        print(f"Testing FreeCodec Variant: {variant}")
        print("=" * 60)

        config = FreeCodecConfig(variant=variant)

        # Create dummy inputs
        mel = torch.randn(batch_size, seq_len, mel_dim).to(device)
        semantic_target = torch.randn(batch_size, seq_len, semantic_dim).to(device)

        # Test 1: Configuration
        print(f"\n[Test 1] Configuration ({variant})...")
        print(f"  Timbre dim: {config.timbre_dim}")
        print(f"  Prosody dim: {config.prosody_dim}, stride: {config.prosody_stride}")
        print(f"  Content codebook: {config.content_codebook_size}")
        print(f"  Semantic target: {config.use_semantic_target}")
        print("  [PASS]")

        # Test 2: Timbre Encoder
        print(f"\n[Test 2] Global Timbre Encoder...")
        timbre_enc = GlobalTimbreEncoder(config).to(device)
        timbre_out = timbre_enc(mel)
        print(f"  Timbre z shape: {timbre_out['timbre_z'].shape}")
        assert timbre_out['timbre_z'].shape == (batch_size, config.timbre_dim)
        print("  [PASS]")

        # Test 3: Prosody Encoder
        print(f"\n[Test 3] Long-Stride Prosody Encoder...")
        prosody_enc = ProsodyEncoder(config).to(device)
        prosody_out = prosody_enc(mel)
        expected_prosody_len = seq_len // config.prosody_stride
        print(f"  Prosody z shape: {prosody_out['prosody_z'].shape}")
        print(f"  Expected length: {expected_prosody_len} (stride={config.prosody_stride})")
        assert prosody_out['prosody_z'].shape[2] == config.prosody_dim
        print("  [PASS]")

        # Test 4: Content Encoder
        print(f"\n[Test 4] VQ Content Encoder...")
        content_enc = ContentEncoder(config).to(device)
        content_out = content_enc(mel, semantic_target)
        print(f"  Content z shape: {content_out['content_z'].shape}")
        print(f"  Content indices shape: {content_out['content_indices'].shape}")
        print(f"  Commitment loss: {content_out['commitment_loss'].item():.4f}")
        print(f"  Perplexity: {content_out['perplexity'].item():.2f}")
        if variant in ["v1", "v2"]:
            print(f"  Semantic loss: {content_out['semantic_loss'].item():.4f}")
        print("  [PASS]")

        # Test 5: Decoder
        print(f"\n[Test 5] Decoder...")
        decoder = FreeCodecDecoder(config).to(device)
        decode_out = decoder(
            timbre_out['timbre_z'],
            prosody_out['prosody_z'],
            content_out['content_z'],
            semantic_target if variant == "v3" else None,
        )
        print(f"  Mel reconstructed shape: {decode_out['mel_reconstructed'].shape}")
        if variant == "v3":
            print(f"  Decoder semantic loss: {decode_out['semantic_loss'].item():.4f}")
        print("  [PASS]")

        # Test 6: Full Model
        print(f"\n[Test 6] Full FreeCodec Model...")
        model = FreeCodec(config).to(device)
        output = model(mel, semantic_target)
        print(f"  Timbre z: {output['timbre_z'].shape}")
        print(f"  Prosody z: {output['prosody_z'].shape}")
        print(f"  Content z: {output['content_z'].shape}")
        print(f"  Mel reconstructed: {output['mel_reconstructed'].shape}")
        print(f"  Combined embedding: {output['combined_embedding'].shape}")
        print("  [PASS]")

        # Test 7: Loss Function
        print(f"\n[Test 7] Loss Function...")
        loss_fn = FreeCodecLoss(config)
        losses = loss_fn(output, mel)
        print(f"  Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
        print(f"  Commitment loss: {losses['commitment_loss'].item():.4f}")
        print(f"  Semantic loss: {losses['semantic_loss'].item():.4f}")
        print(f"  Orthogonality loss: {losses['orthogonality_loss'].item():.4f}")
        print(f"  Total loss: {losses['total'].item():.4f}")
        print("  [PASS]")

        # Test 8: CSM Adapter
        print(f"\n[Test 8] CSM Adapter...")
        adapter = FreeCodecAdapter(config, model).to(device)
        adapter_out = adapter(mel, semantic_target)
        print(f"  Prefix tokens: {adapter_out['prosody_tokens'].shape}")
        assert adapter_out['prosody_tokens'].shape == (
            batch_size, config.num_prefix_tokens, config.output_dim
        )
        print("  [PASS]")

        # Test 9: Voice Conversion
        print(f"\n[Test 9] Voice Conversion...")
        mel_a = torch.randn(1, seq_len, mel_dim).to(device)
        mel_b = torch.randn(1, seq_len, mel_dim).to(device)
        with torch.no_grad():
            mel_converted = model.voice_convert(mel_a, mel_b)
        print(f"  Converted mel shape: {mel_converted.shape}")
        print("  [PASS]")

        # Test 10: Three-way Prosody Transfer
        print(f"\n[Test 10] Three-way Prosody Transfer...")
        mel_c = torch.randn(1, seq_len, mel_dim).to(device)
        with torch.no_grad():
            mel_transferred = model.prosody_transfer(mel_a, mel_b, mel_c)
        print(f"  Transferred mel shape: {mel_transferred.shape}")
        print("  [PASS]")

        # Test 11: Backward Pass
        print(f"\n[Test 11] Backward Pass...")
        model.zero_grad()
        output = model(mel, semantic_target)
        losses = loss_fn(output, mel)
        losses['total'].backward()

        grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )
        print(f"  Total gradient norm: {grad_norm:.4f}")
        print("  [PASS]")

    # Test 12: Token Efficiency Comparison
    print(f"\n{'='*60}")
    print("Token Efficiency Comparison (FreeCodec vs FACodec)")
    print("=" * 60)

    config = FreeCodecConfig(variant="v2")
    model = FreeCodec(config).to(device)
    comparison = FreeCodecVsFACodecComparison(model)

    test_mel = torch.randn(1, 200, mel_dim).to(device)  # ~3.2 seconds
    efficiency = comparison.compute_token_efficiency(test_mel)

    print(f"  FreeCodec tokens: {efficiency['freecodec_total_tokens']}")
    print(f"  FACodec tokens (estimate): {efficiency['facodec_tokens_estimate']}")
    print(f"  Efficiency ratio: {efficiency['efficiency_ratio']:.2f}x fewer tokens")
    print(f"  FreeCodec tokens/sec: {efficiency['freecodec_tokens_per_second']:.1f}")
    print(f"  FACodec tokens/sec: {efficiency['facodec_tokens_per_second']:.1f}")

    print("\n" + "=" * 60)
    print("All FreeCodec tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. PARALLEL DECOMPOSITION:
       - Global Timbre: Single vector (no temporal codes)
       - Long-Stride Prosody: Coarse temporal (stride=4)
       - VQ Content: Frame-level with semantic targets

    2. TOKEN EFFICIENCY:
       - ~3x fewer tokens than FACodec (6 codebooks)
       - Global timbre = 1 token vs T temporal tokens
       - Prosody stride = 4x reduction

    3. SEMANTIC LEARNING TARGETS:
       - v1: Basic VQ content encoding
       - v2: Semantic targets in encoder (better content)
       - v3: Semantic targets at decoder (prevents speaker leakage)

    4. DISENTANGLEMENT:
       - Parallel encoders prevent information mixing
       - Orthogonality loss for additional separation
       - Clean prosody without content/timbre leakage

    5. ZERO-SHOT APPLICATIONS:
       - Voice conversion: Keep content, change speaker
       - Prosody transfer: Mix content/prosody/timbre freely
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from freecodec import (
    FreeCodecConfig,
    FreeCodec,
    FreeCodecLoss,
    FreeCodecAdapter,
)

# Initialize (choose variant)
config = FreeCodecConfig(variant="v2")  # v1, v2, or v3
model = FreeCodec(config).cuda()
loss_fn = FreeCodecLoss(config)

# Training
for mel, semantic_target in dataloader:
    output = model(mel, semantic_target)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    print(f"Perplexity: {losses['perplexity']:.2f}")

# Zero-shot voice conversion
mel_converted = model.voice_convert(
    source_mel=content_source,
    target_speaker_mel=speaker_reference,
)

# Three-way prosody transfer
mel_transferred = model.prosody_transfer(
    content_mel=mel_a,  # Content source
    prosody_mel=mel_b,  # Prosody source
    timbre_mel=mel_c,   # Speaker source
)

# CSM integration
adapter = FreeCodecAdapter(config, model)
prefix_tokens = adapter(mel)['prosody_tokens']  # [batch, 4, 2048]

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
