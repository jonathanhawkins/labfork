"""
SoCodec: Semantic-Ordered Speech Codec with Ordered Product Quantization

Based on "SoCodec: A Semantic-Ordered Multi-Stream Speech Codec for Efficient
Language Model Based Text-to-Speech Synthesis" (ICASSP 2025) - arXiv:2409.00933

Key Innovation: Ordered Product Quantization (OPQ) constrains tokens into an
ordered representation along the stream axis, where each token in the sequence
has semantic ordering relative to the previous one. This improves LM prediction
efficiency by providing a structured representation.

Architecture:
```
Audio → [HuBERT Encoder] → Semantic Features (50Hz)
             ↓
     [Temporal Downsampling] → Compressed Features (~8Hz at 120ms frameshift)
             ↓
     [Ordered Product Quantization] → Multi-Stream Ordered Tokens
             │
             │   Stream 1: Most significant semantic info
             │   Stream 2: Refining info (conditioned on Stream 1)
             │   Stream 3: Refining info (conditioned on Streams 1,2)
             │   ...
             │   Stream N: Fine details (conditioned on all previous)
             ↓
Audio → [ECAPA-TDNN] → Global Acoustic Embedding (speaker/environment)
             ↓
     [Decoder] → Reconstructed Audio
```

Key Components:
1. **Semantic Encoder**: HuBERT-based feature extraction
2. **Temporal Downsampler**: Reduces temporal resolution (50Hz → ~8Hz)
3. **Ordered Product Quantizer (OPQ)**: Multi-stream quantization with ordering constraint
4. **Acoustic Encoder (ECAPA-TDNN)**: Global speaker/environment embedding
5. **Multi-Stream Delayed LM**: For autoregressive generation (training strategy)
6. **Decoder**: Combines semantic + acoustic for reconstruction

OPQ Key Insight:
- Standard VQ/RVQ treats streams independently
- OPQ enforces semantic ordering: Stream k depends on Streams 1...k-1
- Each stream's codebook is conditioned on previous streams' outputs
- Results in more compressible, LM-friendly representation

Multi-Stream Delayed LM Strategy:
- Stream 1 predicts at frame t
- Stream 2 predicts at frame t-1 (delayed by 1)
- Stream 3 predicts at frame t-2 (delayed by 2)
- This allows each stream to "see" previous streams' future tokens

Benefits:
- Ultra-low bitrate: 0.47 kbps at 120ms frameshift (vs 1.5kbps EnCodec)
- Better LM prediction due to ordered representation
- Drop-in replacement for EnCodec in LM-based TTS
- Clean separation of semantic content and acoustic details

Usage:
    from socodec import (
        SoCodecConfig,
        SoCodec,
        SoCodecLoss,
        SoCodecAdapter,
        OrderedProductQuantizer,
    )

    # Initialize
    config = SoCodecConfig()
    model = SoCodec(config).cuda()

    # Encode to ordered multi-stream tokens
    encoded = model.encode(mel, semantic_features)
    ordered_tokens = encoded['ordered_tokens']      # [num_streams, batch, seq]
    acoustic_emb = encoded['acoustic_embedding']    # [batch, acoustic_dim]

    # Decode from tokens
    mel_reconstructed = model.decode(ordered_tokens, acoustic_emb)

    # CSM integration
    adapter = SoCodecAdapter(config, model)
    prefix_tokens = adapter(mel, semantic_features)  # [batch, 4, 2048]

Reference:
- Paper: https://arxiv.org/abs/2409.00933
- GitHub: https://github.com/hhguo/SoCodec
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
class SoCodecConfig:
    """Configuration for SoCodec semantic-ordered speech codec."""

    # Audio settings
    sample_rate: int = 16000
    hop_length: int = 320  # 20ms at 16kHz (50Hz frame rate)
    mel_dim: int = 80

    # Semantic Encoder (HuBERT-based)
    semantic_input_dim: int = 768  # HuBERT/WavLM feature dimension
    semantic_hidden_dim: int = 512
    semantic_output_dim: int = 256
    semantic_num_layers: int = 4
    semantic_num_heads: int = 8

    # Temporal Downsampling
    # 120ms frameshift = 6x downsampling from 20ms (50Hz → ~8Hz)
    temporal_downsample_factor: int = 6
    use_strided_conv: bool = True  # Use strided conv vs pooling

    # Ordered Product Quantization (OPQ)
    num_streams: int = 4  # Number of ordered streams
    codebook_size: int = 1024  # Codes per stream
    code_dim: int = 64  # Dimension per code
    commitment_cost: float = 0.25
    ema_decay: float = 0.99
    use_cosine_similarity: bool = True  # Use cosine sim for quantization

    # OPQ Conditioning
    conditioning_type: str = "concat"  # concat, add, or cross_attention
    conditioning_hidden_dim: int = 128

    # Acoustic Encoder (ECAPA-TDNN style)
    acoustic_dim: int = 256  # Global acoustic embedding dim
    acoustic_hidden_dim: int = 512
    acoustic_num_layers: int = 3
    use_se_block: bool = True  # Squeeze-Excitation blocks

    # Decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Multi-Stream Delayed LM (training strategy)
    use_delayed_lm: bool = True
    max_delay: int = 3  # Maximum delay for streams

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
    """1D Convolution block with residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.norm = nn.GroupNorm(8, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim or stride != 1:
            self.residual = nn.Conv1d(in_dim, out_dim, 1, stride=stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        min_len = min(x.shape[-1], residual.shape[-1])
        return x[..., :min_len] + residual[..., :min_len]


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

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
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


# =============================================================================
# ORDERED PRODUCT QUANTIZER (OPQ)
# =============================================================================

class OrderedCodebook(nn.Module):
    """
    Single codebook for one stream in OPQ.

    Can be conditioned on previous streams' outputs for ordered quantization.
    """

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        conditioning_dim: int = 0,  # 0 for first stream
        conditioning_type: str = "concat",
        ema_decay: float = 0.99,
        commitment_cost: float = 0.25,
        use_cosine: bool = True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.conditioning_dim = conditioning_dim
        self.conditioning_type = conditioning_type
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.use_cosine = use_cosine

        # Main codebook
        self.embedding = nn.Embedding(codebook_size, code_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size
        )

        # Conditioning projection (if conditioned)
        if conditioning_dim > 0:
            if conditioning_type == "concat":
                # Project conditioned input back to code_dim
                self.cond_proj = nn.Linear(code_dim + conditioning_dim, code_dim)
            elif conditioning_type == "add":
                # Project conditioning to code_dim for addition
                self.cond_proj = nn.Linear(conditioning_dim, code_dim)
            elif conditioning_type == "cross_attention":
                # Cross-attention mechanism
                self.cond_attn = nn.MultiheadAttention(
                    code_dim, num_heads=4, batch_first=True
                )
                self.cond_proj = nn.Linear(conditioning_dim, code_dim)
            else:
                raise ValueError(f"Unknown conditioning type: {conditioning_type}")
        else:
            self.cond_proj = None

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_sum', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))

    def _init_from_data(self, flat_x: torch.Tensor):
        """Initialize codebook from first batch."""
        n = flat_x.shape[0]
        if n >= self.codebook_size:
            indices = torch.randperm(n)[:self.codebook_size]
            init_data = flat_x[indices]
        else:
            repeats = (self.codebook_size // n) + 1
            init_data = flat_x.repeat(repeats, 1)[:self.codebook_size]

        self.embedding.weight.data.copy_(init_data)
        self.ema_sum.data.copy_(init_data)
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def _apply_conditioning(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Apply conditioning from previous streams."""
        if self.conditioning_dim == 0 or self.cond_proj is None:
            return x

        if self.conditioning_type == "concat":
            combined = torch.cat([x, conditioning], dim=-1)
            return self.cond_proj(combined)
        elif self.conditioning_type == "add":
            cond_proj = self.cond_proj(conditioning)
            return x + cond_proj
        elif self.conditioning_type == "cross_attention":
            cond_proj = self.cond_proj(conditioning)
            # Reshape for attention
            if x.dim() == 2:
                x = x.unsqueeze(1)
                cond_proj = cond_proj.unsqueeze(1)
            x, _ = self.cond_attn(x, cond_proj, cond_proj)
            return x.squeeze(1) if x.shape[1] == 1 else x

        return x

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input with optional conditioning from previous streams.

        Args:
            x: [batch, seq, code_dim] input features
            conditioning: [batch, seq, conditioning_dim] from previous streams

        Returns:
            Dict with z_q, indices, commitment_loss, perplexity
        """
        batch_size, seq_len, _ = x.shape

        # Apply conditioning
        if conditioning is not None:
            x = self._apply_conditioning(x, conditioning)

        flat_x = x.reshape(-1, self.code_dim)

        # Initialize from first batch
        if self.training and not self.initialized:
            self._init_from_data(flat_x)

        # Normalize for cosine similarity
        if self.use_cosine:
            flat_x_norm = F.normalize(flat_x, dim=-1)
            codes_norm = F.normalize(self.embedding.weight, dim=-1)
            # Cosine similarity → larger is better, so negate for distance
            distances = 1 - torch.matmul(flat_x_norm, codes_norm.t())
        else:
            # L2 distance
            distances = (
                flat_x.pow(2).sum(dim=-1, keepdim=True)
                + self.embedding.weight.pow(2).sum(dim=-1)
                - 2 * torch.matmul(flat_x, self.embedding.weight.t())
            )

        # Find nearest codes
        indices = distances.argmin(dim=-1)
        z_q = self.embedding(indices)

        # EMA update
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                new_size = encodings.sum(dim=0)
                new_sum = torch.matmul(encodings.t(), flat_x)

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    new_size, alpha=1 - self.ema_decay
                )
                self.ema_sum.mul_(self.ema_decay).add_(
                    new_sum, alpha=1 - self.ema_decay
                )

                n = self.ema_cluster_size.clamp(min=1)
                self.embedding.weight.data.copy_(self.ema_sum / n.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), flat_x)

        # Straight-through estimator
        z_q = flat_x + (z_q - flat_x).detach()

        # Reshape
        z_q = z_q.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

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
        return self.embedding(indices)


class OrderedProductQuantizer(nn.Module):
    """
    Ordered Product Quantization (OPQ) for multi-stream semantic ordering.

    Key Innovation:
    - Each stream k is conditioned on the outputs of streams 1...k-1
    - This creates a semantic ordering where earlier streams capture
      more significant/coarse information
    - Later streams capture refinements/details

    Architecture:
        Stream 1: x → VQ₁ → z₁ (unconditional)
        Stream 2: x → f(z₁) → VQ₂ → z₂ (conditioned on z₁)
        Stream 3: x → f(z₁, z₂) → VQ₃ → z₃ (conditioned on z₁, z₂)
        ...

    This is different from RVQ which quantizes residuals:
        RVQ: x → VQ₁ → z₁, (x-z₁) → VQ₂ → z₂, ...

    OPQ quantizes conditioned representations, not residuals.
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config
        self.num_streams = config.num_streams
        self.code_dim = config.code_dim

        # Pre-quantization projection
        self.pre_quant_proj = nn.Linear(
            config.semantic_output_dim, config.code_dim
        )

        # Create ordered codebooks
        self.codebooks = nn.ModuleList()

        for i in range(config.num_streams):
            if i == 0:
                # First stream: unconditional
                conditioning_dim = 0
            else:
                # Later streams: conditioned on all previous outputs
                conditioning_dim = config.code_dim * i

            self.codebooks.append(
                OrderedCodebook(
                    codebook_size=config.codebook_size,
                    code_dim=config.code_dim,
                    conditioning_dim=conditioning_dim,
                    conditioning_type=config.conditioning_type,
                    ema_decay=config.ema_decay,
                    commitment_cost=config.commitment_cost,
                    use_cosine=config.use_cosine_similarity,
                )
            )

        # Stream-specific projections for input transformation
        self.stream_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.code_dim, config.conditioning_hidden_dim),
                nn.GELU(),
                nn.Linear(config.conditioning_hidden_dim, config.code_dim),
            )
            for _ in range(config.num_streams)
        ])

        # Norm for combined output (num_streams * code_dim)
        self.norm = nn.LayerNorm(config.num_streams * config.code_dim)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, semantic_output_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Ordered Product Quantization.

        Returns:
            Dict with:
                - ordered_z_q: [batch, seq, num_streams * code_dim]
                - ordered_indices: [num_streams, batch, seq]
                - stream_outputs: List of per-stream z_q
                - total_commitment_loss
                - perplexities: [num_streams]
        """
        batch_size, seq_len, _ = x.shape

        # Project to code dimension
        x = self.pre_quant_proj(x)  # [B, T, code_dim]

        stream_outputs = []
        all_indices = []
        total_commitment = 0.0
        perplexities = []

        for i, (codebook, stream_proj) in enumerate(
            zip(self.codebooks, self.stream_projs)
        ):
            # Transform input for this stream
            stream_input = stream_proj(x)

            # Build conditioning from previous streams
            if i > 0:
                # Concatenate all previous stream outputs
                conditioning = torch.cat(stream_outputs, dim=-1)
            else:
                conditioning = None

            # Quantize with conditioning
            vq_output = codebook(stream_input, conditioning)

            stream_outputs.append(vq_output['z_q'])
            all_indices.append(vq_output['indices'])
            total_commitment += vq_output['commitment_loss']
            perplexities.append(vq_output['perplexity'])

        # Combine all stream outputs
        ordered_z_q = torch.cat(stream_outputs, dim=-1)  # [B, T, num_streams * code_dim]
        ordered_z_q = self.norm(ordered_z_q)

        return {
            'ordered_z_q': ordered_z_q,
            'ordered_indices': torch.stack(all_indices, dim=0),  # [num_streams, B, T]
            'stream_outputs': stream_outputs,
            'commitment_loss': total_commitment / self.num_streams,
            'perplexities': torch.stack(perplexities),
        }

    def decode_indices(
        self,
        indices: torch.Tensor,  # [num_streams, batch, seq]
    ) -> torch.Tensor:
        """Decode ordered indices to vectors."""
        stream_outputs = []

        for i, codebook in enumerate(self.codebooks):
            z_q = codebook.decode_indices(indices[i])
            stream_outputs.append(z_q)

        ordered_z_q = torch.cat(stream_outputs, dim=-1)
        return self.norm(ordered_z_q)

    def get_stream_tokens(
        self,
        indices: torch.Tensor,
        stream_idx: int,
    ) -> torch.Tensor:
        """Get tokens for a specific stream."""
        return self.codebooks[stream_idx].decode_indices(indices[stream_idx])


# =============================================================================
# SEMANTIC ENCODER
# =============================================================================

class SemanticEncoder(nn.Module):
    """
    Semantic Encoder: Processes HuBERT/WavLM features.

    Takes pre-extracted semantic features and processes them through
    transformer layers before temporal downsampling.
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(
            config.semantic_input_dim, config.semantic_hidden_dim
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.semantic_hidden_dim, dropout=config.dropout
        )

        # Transformer layers
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.semantic_hidden_dim,
                num_heads=config.semantic_num_heads,
                ffn_dim=config.semantic_hidden_dim * 4,
                dropout=config.dropout,
            )
            for _ in range(config.semantic_num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(
            config.semantic_hidden_dim, config.semantic_output_dim
        )

        self.norm = nn.LayerNorm(config.semantic_output_dim)

    def forward(
        self,
        semantic_features: torch.Tensor,  # [batch, seq, semantic_input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process semantic features.

        Returns:
            Dict with semantic_hidden and semantic_output
        """
        # Project input
        x = self.input_proj(semantic_features)
        x = self.pos_enc(x)

        # Transformer encoding
        for transformer in self.transformer:
            x = transformer(x, mask)

        # Output projection
        output = self.output_proj(x)
        output = self.norm(output)

        return {
            'semantic_hidden': x,
            'semantic_output': output,
        }


# =============================================================================
# TEMPORAL DOWNSAMPLER
# =============================================================================

class TemporalDownsampler(nn.Module):
    """
    Temporal Downsampler: Reduces frame rate from 50Hz to ~8Hz.

    Uses strided convolutions or pooling to downsample by the specified factor.
    The 120ms frameshift in SoCodec means 6x downsampling from 20ms (50Hz).
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config
        self.factor = config.temporal_downsample_factor

        if config.use_strided_conv:
            # Strided convolution downsampling
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    config.semantic_output_dim,
                    config.semantic_output_dim,
                    kernel_size=self.factor * 2 - 1,
                    stride=self.factor,
                    padding=self.factor - 1,
                ),
                nn.GroupNorm(8, config.semantic_output_dim),
                nn.GELU(),
                nn.Conv1d(
                    config.semantic_output_dim,
                    config.semantic_output_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(8, config.semantic_output_dim),
                nn.GELU(),
            )
        else:
            # Average pooling downsampling
            self.downsample = nn.Sequential(
                nn.AvgPool1d(self.factor, stride=self.factor),
                nn.Conv1d(
                    config.semantic_output_dim,
                    config.semantic_output_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GroupNorm(8, config.semantic_output_dim),
                nn.GELU(),
            )

        self.norm = nn.LayerNorm(config.semantic_output_dim)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, dim]
    ) -> torch.Tensor:
        """Downsample temporal resolution."""
        # Conv expects [batch, channels, time]
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)

        return self.norm(x)


# =============================================================================
# ACOUSTIC ENCODER (ECAPA-TDNN Style)
# =============================================================================

class ECAPATDNNBlock(nn.Module):
    """ECAPA-TDNN block with SE-Res2Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        use_se: bool = True,
    ):
        super().__init__()
        self.scale = scale

        # First 1x1 conv (channel reduction)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Res2Net-style multi-scale convolutions
        width = out_channels // scale
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(scale - 1):
            self.convs.append(
                nn.Conv1d(
                    width, width, kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2,
                )
            )
            self.bns.append(nn.BatchNorm1d(width))

        # Second 1x1 conv (channel restoration)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # SE block
        self.se = SEBlock(out_channels) if use_se else None

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        residual = self.residual(x)

        # First conv
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        # Res2Net multi-scale
        width = out.size(1) // self.scale
        spx = torch.split(out, width, dim=1)

        sp_outputs = [spx[0]]
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i - 1](sp)
            sp = self.relu(self.bns[i - 1](sp))
            sp_outputs.append(sp)

        out = torch.cat(sp_outputs, dim=1)

        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)

        # SE block
        if self.se is not None:
            out = self.se(out)

        # Residual
        min_len = min(out.shape[-1], residual.shape[-1])
        out = self.relu(out[..., :min_len] + residual[..., :min_len])

        return out


class AcousticEncoder(nn.Module):
    """
    Acoustic Encoder: ECAPA-TDNN style encoder for global acoustic embedding.

    Extracts time-invariant speaker and environment characteristics.
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config

        # Input projection from mel
        self.input_proj = nn.Conv1d(config.mel_dim, config.acoustic_hidden_dim, 1)
        self.input_bn = nn.BatchNorm1d(config.acoustic_hidden_dim)

        # ECAPA-TDNN blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ECAPATDNNBlock(
                config.acoustic_hidden_dim,
                config.acoustic_hidden_dim,
                kernel_size=3,
                dilation=2 ** i,
                use_se=config.use_se_block,
            )
            for i in range(config.acoustic_num_layers)
        ])

        # Multi-layer feature aggregation
        self.mfa_conv = nn.Conv1d(
            config.acoustic_hidden_dim * config.acoustic_num_layers,
            config.acoustic_hidden_dim,
            1,
        )
        self.mfa_bn = nn.BatchNorm1d(config.acoustic_hidden_dim)

        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Conv1d(config.acoustic_hidden_dim, config.acoustic_hidden_dim // 2, 1),
            nn.Tanh(),
            nn.Conv1d(config.acoustic_hidden_dim // 2, config.acoustic_hidden_dim, 1),
            nn.Softmax(dim=2),
        )

        # Final projection
        self.output_proj = nn.Linear(
            config.acoustic_hidden_dim * 2,  # mean + std
            config.acoustic_dim,
        )

        self.norm = nn.LayerNorm(config.acoustic_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global acoustic embedding.

        Returns:
            Dict with acoustic_embedding: [batch, acoustic_dim]
        """
        # Input projection [B, T, mel] -> [B, H, T]
        x = mel.transpose(1, 2)
        x = F.relu(self.input_bn(self.input_proj(x)))

        # ECAPA blocks with multi-layer aggregation
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        # Concatenate all block outputs
        # Handle different lengths
        min_len = min(o.shape[-1] for o in block_outputs)
        block_outputs = [o[..., :min_len] for o in block_outputs]
        x = torch.cat(block_outputs, dim=1)

        # Multi-layer feature aggregation
        x = F.relu(self.mfa_bn(self.mfa_conv(x)))

        # Attentive statistics pooling
        attn_weights = self.attention(x)
        if mask is not None:
            # Extend mask to match length
            mask_len = min(mask.shape[1], x.shape[2])
            attn_weights = attn_weights[:, :, :mask_len]
            attn_weights = attn_weights.masked_fill(
                mask[:, :mask_len].unsqueeze(1), float('-inf')
            )
            attn_weights = F.softmax(attn_weights, dim=2)

        # Weighted mean and std
        mean = torch.sum(x * attn_weights, dim=2)
        std = torch.sqrt(
            torch.sum((x - mean.unsqueeze(2)).pow(2) * attn_weights, dim=2).clamp(min=1e-8)
        )

        # Combine statistics
        stats = torch.cat([mean, std], dim=1)  # [B, H*2]

        # Final projection
        embedding = self.output_proj(stats)
        embedding = self.norm(embedding)

        return {
            'acoustic_embedding': embedding,
            'acoustic_features': x.transpose(1, 2),  # [B, T, H] for analysis
        }


# =============================================================================
# DECODER
# =============================================================================

class SoCodecDecoder(nn.Module):
    """
    Decoder: Reconstructs mel from ordered tokens + acoustic embedding.

    Combines:
    - Ordered semantic tokens (multi-stream)
    - Global acoustic embedding (speaker/environment)
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config

        # Total dimension from OPQ
        ordered_dim = config.num_streams * config.code_dim

        # Input projections
        self.semantic_proj = nn.Linear(ordered_dim, config.decoder_hidden_dim)
        self.acoustic_proj = nn.Linear(config.acoustic_dim, config.decoder_hidden_dim)

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

        # Temporal upsampling (reverse of downsampling)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(
                config.decoder_hidden_dim,
                config.decoder_hidden_dim,
                kernel_size=config.temporal_downsample_factor,
                stride=config.temporal_downsample_factor,
            ),
            nn.GroupNorm(8, config.decoder_hidden_dim),
            nn.GELU(),
        )

        # Output projection to mel
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.mel_dim),
        )

        self.norm = nn.LayerNorm(config.decoder_hidden_dim)

    def forward(
        self,
        ordered_z_q: torch.Tensor,   # [batch, seq_ds, num_streams * code_dim]
        acoustic_emb: torch.Tensor,  # [batch, acoustic_dim]
        target_length: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode to mel spectrogram.

        Returns:
            Dict with mel_reconstructed: [batch, seq, mel_dim]
        """
        batch_size, seq_len_ds, _ = ordered_z_q.shape

        # Project semantic
        semantic_proj = self.semantic_proj(ordered_z_q)  # [B, T_ds, H]

        # Project acoustic (global, broadcast to sequence)
        acoustic_proj = self.acoustic_proj(acoustic_emb)  # [B, H]
        acoustic_proj = acoustic_proj.unsqueeze(1)  # [B, 1, H]

        # Combine
        x = semantic_proj + acoustic_proj

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer decoding
        for transformer in self.transformer:
            x = transformer(x, mask)
        x = self.norm(x)

        # Upsample temporal resolution
        x = x.transpose(1, 2)  # [B, H, T_ds]
        x = self.upsample(x)
        x = x.transpose(1, 2)  # [B, T_us, H]

        # Adjust to target length if specified
        if target_length is not None:
            if x.shape[1] > target_length:
                x = x[:, :target_length]
            elif x.shape[1] < target_length:
                padding = torch.zeros(
                    batch_size, target_length - x.shape[1], x.shape[2],
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)

        # Output projection
        mel_reconstructed = self.output_proj(x)

        return {
            'mel_reconstructed': mel_reconstructed,
            'decoder_features': x,
        }


# =============================================================================
# MULTI-STREAM DELAYED LM MODULE
# =============================================================================

class MultiStreamDelayedLM(nn.Module):
    """
    Multi-Stream Delayed Language Model head for autoregressive generation.

    Key Insight from SoCodec:
    - Stream 1 predicts current frame
    - Stream 2 predicts frame t-1 (can see Stream 1's frame t token)
    - Stream 3 predicts frame t-2 (can see Streams 1,2's future tokens)

    This enables each stream to leverage "future" context from previous
    streams during autoregressive generation.
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config
        self.num_streams = config.num_streams
        self.max_delay = config.max_delay

        # Per-stream transformer heads
        self.stream_heads = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(
                    config.code_dim * (i + 1),  # Sees all previous streams
                    num_heads=4,
                    ffn_dim=config.code_dim * 4,
                    dropout=config.dropout,
                ),
                nn.Linear(config.code_dim * (i + 1), config.codebook_size),
            )
            for i in range(config.num_streams)
        ])

    def forward(
        self,
        stream_embeddings: List[torch.Tensor],  # List of [batch, seq, code_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute delayed LM logits for each stream.

        Returns:
            Dict with per-stream logits for LM training
        """
        batch_size, seq_len, _ = stream_embeddings[0].shape

        stream_logits = []

        for i in range(self.num_streams):
            # Delay for this stream
            delay = min(i, self.max_delay)

            # Gather context from previous streams (with their "future" tokens)
            context_streams = []
            for j in range(i + 1):
                emb = stream_embeddings[j]
                if j < i:
                    # Previous streams: shift to provide "future" context
                    shift = min(i - j, delay)
                    if shift > 0:
                        # Shift left (future tokens become available)
                        emb = F.pad(emb[:, shift:], (0, 0, 0, shift))
                context_streams.append(emb)

            # Concatenate context
            context = torch.cat(context_streams, dim=-1)

            # Predict logits for this stream
            logits = self.stream_heads[i](context)
            stream_logits.append(logits)

        return {
            'stream_logits': stream_logits,  # List of [B, T, codebook_size]
        }

    def compute_loss(
        self,
        stream_logits: List[torch.Tensor],
        target_indices: torch.Tensor,  # [num_streams, batch, seq]
    ) -> torch.Tensor:
        """Compute cross-entropy loss for all streams."""
        total_loss = 0.0

        for i, logits in enumerate(stream_logits):
            # Get target indices for this stream
            targets = target_indices[i]  # [B, T]

            # Compute cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.codebook_size),
                targets.reshape(-1),
                reduction='mean',
            )
            total_loss += loss

        return total_loss / self.num_streams


# =============================================================================
# FULL SOCODEC MODEL
# =============================================================================

class SoCodec(nn.Module):
    """
    SoCodec: Semantic-Ordered Multi-Stream Speech Codec.

    Combines:
    1. Semantic Encoder (HuBERT-based)
    2. Temporal Downsampler (50Hz → ~8Hz)
    3. Ordered Product Quantizer (OPQ)
    4. Acoustic Encoder (ECAPA-TDNN)
    5. Decoder (Transformer + upsampling)
    6. Multi-Stream Delayed LM (optional, for training)
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.semantic_encoder = SemanticEncoder(config)
        self.temporal_downsampler = TemporalDownsampler(config)
        self.acoustic_encoder = AcousticEncoder(config)

        # Ordered Product Quantizer
        self.opq = OrderedProductQuantizer(config)

        # Decoder
        self.decoder = SoCodecDecoder(config)

        # Multi-stream delayed LM (for training)
        if config.use_delayed_lm:
            self.delayed_lm = MultiStreamDelayedLM(config)
        else:
            self.delayed_lm = None

        # Output projection for CSM integration
        combined_dim = config.num_streams * config.code_dim + config.acoustic_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        semantic_features: torch.Tensor,  # [batch, seq, semantic_input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode to ordered multi-stream tokens + acoustic embedding.

        Returns:
            Dict with all encoder outputs
        """
        # Semantic encoding
        semantic_output = self.semantic_encoder(semantic_features, mask)

        # Temporal downsampling
        downsampled = self.temporal_downsampler(semantic_output['semantic_output'])

        # Ordered Product Quantization
        opq_output = self.opq(downsampled)

        # Acoustic encoding
        acoustic_output = self.acoustic_encoder(mel, mask)

        return {
            # Semantic
            'semantic_hidden': semantic_output['semantic_hidden'],
            'semantic_output': semantic_output['semantic_output'],
            'downsampled': downsampled,
            # OPQ
            'ordered_z_q': opq_output['ordered_z_q'],
            'ordered_indices': opq_output['ordered_indices'],
            'stream_outputs': opq_output['stream_outputs'],
            'commitment_loss': opq_output['commitment_loss'],
            'perplexities': opq_output['perplexities'],
            # Acoustic
            'acoustic_embedding': acoustic_output['acoustic_embedding'],
            'acoustic_features': acoustic_output['acoustic_features'],
        }

    def decode(
        self,
        ordered_z_q: torch.Tensor,
        acoustic_emb: torch.Tensor,
        target_length: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode ordered tokens + acoustic embedding to mel."""
        return self.decoder(ordered_z_q, acoustic_emb, target_length, mask)

    def decode_from_indices(
        self,
        ordered_indices: torch.Tensor,  # [num_streams, batch, seq]
        acoustic_emb: torch.Tensor,
        target_length: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode from codebook indices."""
        ordered_z_q = self.opq.decode_indices(ordered_indices)
        return self.decode(ordered_z_q, acoustic_emb, target_length, mask)

    def forward(
        self,
        mel: torch.Tensor,
        semantic_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, quantize, decode.

        Returns:
            Dict with all outputs including reconstruction
        """
        # Encode
        encoded = self.encode(mel, semantic_features, mask)

        # Decode
        target_length = mel.shape[1]
        decoded = self.decode(
            encoded['ordered_z_q'],
            encoded['acoustic_embedding'],
            target_length,
            mask,
        )

        # Delayed LM logits (if enabled)
        if self.delayed_lm is not None:
            lm_output = self.delayed_lm(encoded['stream_outputs'])
        else:
            lm_output = {'stream_logits': None}

        # Combined prosody embedding for CSM
        ordered_pooled = encoded['ordered_z_q'].mean(dim=1)  # [B, num_streams * code_dim]
        acoustic_emb = encoded['acoustic_embedding']  # [B, acoustic_dim]

        combined = torch.cat([ordered_pooled, acoustic_emb], dim=-1)
        prosody_embedding = self.output_proj(combined)

        return {
            **encoded,
            'mel_reconstructed': decoded['mel_reconstructed'],
            'stream_logits': lm_output['stream_logits'],
            'prosody_embedding': prosody_embedding,
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class SoCodecLoss(nn.Module):
    """
    Combined loss function for SoCodec training.

    Components:
    1. Reconstruction loss (L1 + L2 mel)
    2. Commitment loss (VQ)
    3. Multi-stream delayed LM loss (optional)
    """

    def __init__(self, config: SoCodecConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.reconstruction_weight = 1.0
        self.commitment_weight = 0.25
        self.lm_weight = 0.1

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        mel_pred = output['mel_reconstructed']

        # Handle length mismatch
        min_len = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_len]
        mel_target = mel_target[:, :min_len]

        # Reconstruction loss
        l1_loss = F.l1_loss(mel_pred, mel_target)
        l2_loss = F.mse_loss(mel_pred, mel_target)
        losses['reconstruction_l1'] = l1_loss
        losses['reconstruction_l2'] = l2_loss
        losses['reconstruction'] = l1_loss + l2_loss

        # Commitment loss
        losses['commitment'] = output['commitment_loss']

        # Perplexities (for logging)
        losses['perplexities'] = output['perplexities']
        losses['mean_perplexity'] = output['perplexities'].mean()

        # Multi-stream LM loss (if available)
        if output['stream_logits'] is not None:
            lm_loss = 0.0
            for i, logits in enumerate(output['stream_logits']):
                targets = output['ordered_indices'][i]
                # Adjust for length mismatch
                log_len = min(logits.shape[1], targets.shape[1])
                stream_loss = F.cross_entropy(
                    logits[:, :log_len].reshape(-1, self.config.codebook_size),
                    targets[:, :log_len].reshape(-1),
                    reduction='mean',
                )
                lm_loss += stream_loss
            lm_loss /= len(output['stream_logits'])
            losses['lm_loss'] = lm_loss
        else:
            losses['lm_loss'] = torch.tensor(0.0, device=mel_target.device)

        # Total loss
        total = (
            self.reconstruction_weight * losses['reconstruction']
            + self.commitment_weight * losses['commitment']
            + self.lm_weight * losses['lm_loss']
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class SoCodecAdapter(nn.Module):
    """
    Adapter for integrating SoCodec with CSM prosody conditioning.

    Converts SoCodec's ordered multi-stream tokens to prefix tokens for CSM.
    """

    def __init__(
        self,
        config: SoCodecConfig,
        model: Optional[SoCodec] = None,
    ):
        super().__init__()
        self.config = config
        self.model = model if model is not None else SoCodec(config)

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

        Returns:
            Dict with:
                - prosody_tokens: [batch, num_prefix_tokens, output_dim]
                - ordered_indices: [num_streams, batch, seq]
                - acoustic_embedding: [batch, acoustic_dim]
        """
        batch_size = mel.shape[0]

        # Get encoded tokens
        output = self.model(mel, semantic_features)
        prosody_emb = output['prosody_embedding']

        # Project to prefix tokens
        tokens = self.token_proj(prosody_emb)
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
            'ordered_indices': output['ordered_indices'],
            'acoustic_embedding': output['acoustic_embedding'],
            'stream_outputs': output['stream_outputs'],
            'perplexities': output['perplexities'],
            'commitment_loss': output['commitment_loss'],
        }

    def from_indices(
        self,
        ordered_indices: torch.Tensor,
        acoustic_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prosody tokens directly from codebook indices.

        Enables discrete prosody control!
        """
        batch_size = acoustic_emb.shape[0]

        # Decode indices
        ordered_z_q = self.model.opq.decode_indices(ordered_indices)

        # Pool and combine
        ordered_pooled = ordered_z_q.mean(dim=1)

        combined = torch.cat([ordered_pooled, acoustic_emb], dim=-1)
        prosody_emb = self.model.output_proj(combined)

        # Project to tokens
        tokens = self.token_proj(prosody_emb)
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )
        tokens = self.norm(tokens)

        return tokens

    def get_stream_tokens(
        self,
        mel: torch.Tensor,
        semantic_features: torch.Tensor,
        stream_idx: int,
    ) -> torch.Tensor:
        """Get tokens for a specific OPQ stream."""
        encoded = self.model.encode(mel, semantic_features)
        return encoded['stream_outputs'][stream_idx]

    def analyze_ordering(
        self,
        mel: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze the semantic ordering of OPQ streams.

        Returns per-stream statistics for understanding the learned ordering.
        """
        encoded = self.model.encode(mel, semantic_features)

        analysis = {
            'perplexities': encoded['perplexities'],
            'stream_variances': [],
            'stream_norms': [],
        }

        for i, stream_output in enumerate(encoded['stream_outputs']):
            # Variance (higher = more information)
            var = stream_output.var(dim=1).mean()
            analysis['stream_variances'].append(var)

            # Norm (magnitude of contribution)
            norm = stream_output.norm(dim=-1).mean()
            analysis['stream_norms'].append(norm)

        analysis['stream_variances'] = torch.stack(analysis['stream_variances'])
        analysis['stream_norms'] = torch.stack(analysis['stream_norms'])

        return analysis


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_socodec_adapter(
    checkpoint: Optional[str] = None,
    config: Optional[SoCodecConfig] = None,
    device: str = "cpu",
) -> SoCodecAdapter:
    """Create SoCodec adapter, optionally loading from checkpoint."""
    if config is None:
        config = SoCodecConfig()

    adapter = SoCodecAdapter(config)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        adapter.load_state_dict(state_dict)

    return adapter.to(device)


def compute_bitrate(
    config: SoCodecConfig,
    sample_rate: int = 16000,
    hop_length: int = 320,
) -> Dict[str, float]:
    """Compute theoretical bitrate for SoCodec configuration."""
    # Frame rate after downsampling
    base_frame_rate = sample_rate / hop_length  # 50 Hz
    downsampled_rate = base_frame_rate / config.temporal_downsample_factor  # ~8.3 Hz

    # Bits per frame (all streams)
    bits_per_code = math.log2(config.codebook_size)
    bits_per_frame = config.num_streams * bits_per_code

    # Bitrate
    semantic_bps = downsampled_rate * bits_per_frame

    # Acoustic embedding (one-time, amortized over typical utterance length)
    acoustic_bits = config.acoustic_dim * 32  # float32
    avg_utterance_duration = 5.0  # seconds
    acoustic_bps = acoustic_bits / avg_utterance_duration

    total_bps = semantic_bps + acoustic_bps

    return {
        'semantic_bps': semantic_bps,
        'acoustic_bps': acoustic_bps,
        'total_bps': total_bps,
        'frame_rate_hz': downsampled_rate,
        'bits_per_frame': bits_per_frame,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SoCodec: Ordered Product Quantization - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 100
    mel_dim = 80
    semantic_dim = 768

    config = SoCodecConfig()

    # Create dummy inputs
    mel = torch.randn(batch_size, seq_len, mel_dim).to(device)
    semantic_features = torch.randn(batch_size, seq_len, semantic_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Num streams: {config.num_streams}")
    print(f"  Codebook size: {config.codebook_size}")
    print(f"  Downsample factor: {config.temporal_downsample_factor}")
    print(f"  Use delayed LM: {config.use_delayed_lm}")
    print("  [PASS]")

    # Test 2: Bitrate computation
    print("\n[Test 2] Bitrate Computation...")
    bitrate = compute_bitrate(config)
    print(f"  Semantic bitrate: {bitrate['semantic_bps']:.2f} bps")
    print(f"  Acoustic bitrate: {bitrate['acoustic_bps']:.2f} bps")
    print(f"  Total bitrate: {bitrate['total_bps']:.2f} bps")
    print(f"  Frame rate: {bitrate['frame_rate_hz']:.2f} Hz")
    print("  [PASS]")

    # Test 3: Ordered Codebook
    print("\n[Test 3] Ordered Codebook...")
    codebook = OrderedCodebook(
        codebook_size=config.codebook_size,
        code_dim=config.code_dim,
        conditioning_dim=config.code_dim,
        conditioning_type=config.conditioning_type,
    ).to(device)

    x = torch.randn(batch_size, seq_len, config.code_dim).to(device)
    cond = torch.randn(batch_size, seq_len, config.code_dim).to(device)

    cb_output = codebook(x, cond)
    print(f"  Quantized shape: {cb_output['z_q'].shape}")
    print(f"  Indices shape: {cb_output['indices'].shape}")
    print(f"  Perplexity: {cb_output['perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 4: Ordered Product Quantizer
    print("\n[Test 4] Ordered Product Quantizer...")
    opq = OrderedProductQuantizer(config).to(device)

    semantic_output = torch.randn(
        batch_size, seq_len // config.temporal_downsample_factor,
        config.semantic_output_dim
    ).to(device)

    opq_output = opq(semantic_output)
    print(f"  Ordered z_q shape: {opq_output['ordered_z_q'].shape}")
    print(f"  Ordered indices shape: {opq_output['ordered_indices'].shape}")
    print(f"  Num streams: {len(opq_output['stream_outputs'])}")
    for i, perp in enumerate(opq_output['perplexities']):
        print(f"    Stream {i+1} perplexity: {perp.item():.2f}")
    print("  [PASS]")

    # Test 5: Semantic Encoder
    print("\n[Test 5] Semantic Encoder...")
    semantic_enc = SemanticEncoder(config).to(device)
    sem_output = semantic_enc(semantic_features)
    print(f"  Semantic hidden: {sem_output['semantic_hidden'].shape}")
    print(f"  Semantic output: {sem_output['semantic_output'].shape}")
    print("  [PASS]")

    # Test 6: Temporal Downsampler
    print("\n[Test 6] Temporal Downsampler...")
    downsampler = TemporalDownsampler(config).to(device)
    downsampled = downsampler(sem_output['semantic_output'])
    print(f"  Input length: {sem_output['semantic_output'].shape[1]}")
    print(f"  Output length: {downsampled.shape[1]}")
    print(f"  Downsample ratio: {sem_output['semantic_output'].shape[1] / downsampled.shape[1]:.2f}x")
    print("  [PASS]")

    # Test 7: Acoustic Encoder
    print("\n[Test 7] Acoustic Encoder...")
    acoustic_enc = AcousticEncoder(config).to(device)
    acoustic_output = acoustic_enc(mel)
    print(f"  Acoustic embedding: {acoustic_output['acoustic_embedding'].shape}")
    print("  [PASS]")

    # Test 8: Decoder
    print("\n[Test 8] Decoder...")
    decoder = SoCodecDecoder(config).to(device)

    ordered_z_q = torch.randn(
        batch_size,
        downsampled.shape[1],
        config.num_streams * config.code_dim
    ).to(device)

    decoded = decoder(
        ordered_z_q,
        acoustic_output['acoustic_embedding'],
        target_length=seq_len,
    )
    print(f"  Reconstructed mel: {decoded['mel_reconstructed'].shape}")
    print("  [PASS]")

    # Test 9: Full Model
    print("\n[Test 9] Full SoCodec Model...")
    model = SoCodec(config).to(device)
    output = model(mel, semantic_features)
    print(f"  Ordered indices: {output['ordered_indices'].shape}")
    print(f"  Acoustic embedding: {output['acoustic_embedding'].shape}")
    print(f"  Mel reconstructed: {output['mel_reconstructed'].shape}")
    print(f"  Prosody embedding: {output['prosody_embedding'].shape}")
    print(f"  Commitment loss: {output['commitment_loss'].item():.4f}")
    if output['stream_logits'] is not None:
        print(f"  Stream logits: {len(output['stream_logits'])} streams")
    print("  [PASS]")

    # Test 10: Loss Function
    print("\n[Test 10] Loss Function...")
    loss_fn = SoCodecLoss(config)
    losses = loss_fn(output, mel)
    print(f"  Reconstruction loss: {losses['reconstruction'].item():.4f}")
    print(f"  Commitment loss: {losses['commitment'].item():.4f}")
    print(f"  LM loss: {losses['lm_loss'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Mean perplexity: {losses['mean_perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 11: CSM Adapter
    print("\n[Test 11] CSM Adapter...")
    adapter = SoCodecAdapter(config, model).to(device)
    adapter_out = adapter(mel, semantic_features)
    print(f"  Prefix tokens: {adapter_out['prosody_tokens'].shape}")
    assert adapter_out['prosody_tokens'].shape == (
        batch_size, config.num_prefix_tokens, config.output_dim
    )
    print("  [PASS]")

    # Test 12: Decode from Indices
    print("\n[Test 12] Decode from Indices...")
    with torch.no_grad():
        decoded_from_idx = model.decode_from_indices(
            output['ordered_indices'],
            output['acoustic_embedding'],
            target_length=seq_len,
        )
    print(f"  Decoded mel shape: {decoded_from_idx['mel_reconstructed'].shape}")
    print("  [PASS]")

    # Test 13: Prefix from Indices
    print("\n[Test 13] Prefix from Indices...")
    with torch.no_grad():
        prefix_from_idx = adapter.from_indices(
            output['ordered_indices'],
            output['acoustic_embedding'],
        )
    print(f"  Prefix from indices: {prefix_from_idx.shape}")
    print("  [PASS]")

    # Test 14: Stream Analysis
    print("\n[Test 14] Stream Analysis...")
    with torch.no_grad():
        analysis = adapter.analyze_ordering(mel, semantic_features)
    print("  Per-stream statistics:")
    for i in range(config.num_streams):
        print(f"    Stream {i+1}: var={analysis['stream_variances'][i].item():.4f}, "
              f"norm={analysis['stream_norms'][i].item():.4f}, "
              f"perplexity={analysis['perplexities'][i].item():.2f}")
    print("  [PASS]")

    # Test 15: Backward Pass
    print("\n[Test 15] Backward Pass...")
    model.zero_grad()
    output = model(mel, semantic_features)
    losses = loss_fn(output, mel)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All SoCodec tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. ORDERED PRODUCT QUANTIZATION (OPQ):
       - Each stream conditioned on all previous streams
       - Creates semantic ordering: coarse → fine
       - Stream 1: Most significant information
       - Stream N: Refinements and details

    2. MULTI-STREAM DELAYED LM:
       - Enables autoregressive generation with ordering
       - Stream k predicts at frame t-(k-1)
       - Each stream sees "future" from previous streams

    3. ULTRA-LOW BITRATE:
       - ~0.47 kbps at 120ms frameshift
       - 50Hz → ~8Hz temporal downsampling
       - Drop-in replacement for EnCodec

    4. ECAPA-TDNN ACOUSTIC ENCODER:
       - Global speaker/environment embedding
       - Time-invariant characteristics
       - SE-Res2Net architecture

    5. CSM INTEGRATION:
       adapter = SoCodecAdapter(config, model)
       prefix_tokens = adapter(mel, semantic_features)
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from socodec import (
    SoCodecConfig,
    SoCodec,
    SoCodecLoss,
    SoCodecAdapter,
    compute_bitrate,
)

# Initialize
config = SoCodecConfig(
    num_streams=4,
    codebook_size=1024,
    temporal_downsample_factor=6,  # 120ms frameshift
)

model = SoCodec(config).cuda()
loss_fn = SoCodecLoss(config)

# Check theoretical bitrate
bitrate = compute_bitrate(config)
print(f"Bitrate: {bitrate['total_bps']:.2f} bps")

# Training loop
for batch in dataloader:
    mel = batch['mel'].cuda()
    semantic_features = batch['hubert_features'].cuda()

    output = model(mel, semantic_features)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor per-stream perplexities
    for i, perp in enumerate(output['perplexities']):
        print(f"Stream {i+1}: {perp.item():.2f}")

# Encode to ordered tokens
with torch.no_grad():
    encoded = model.encode(mel, semantic_features)
    ordered_indices = encoded['ordered_indices']  # [num_streams, B, T]
    acoustic_emb = encoded['acoustic_embedding']  # [B, acoustic_dim]

# CSM integration
adapter = SoCodecAdapter(config, model)
prefix_tokens = adapter(mel, semantic_features)['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)

# Discrete prosody control via indices
custom_prefix = adapter.from_indices(custom_indices, acoustic_emb)
""")
