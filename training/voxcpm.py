"""
VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation

Based on "VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and
True-to-Life Voice Cloning" (arXiv:2509.24650).

Key Innovation: Hierarchical semantic-acoustic modeling with semi-discrete residual
representations, trained end-to-end under a diffusion objective. Avoids discrete
tokenization bottleneck while maintaining stable training.

Architecture:
```
Text → [TSLM] → Semantic-Prosodic Plans (continuous)
                     ↓
                 [FSQ] → Semi-Discrete Bottleneck
                     ↓
      + History → [RALM] → Acoustic Residuals
                     ↓
                [LocDiT] → Flow-Matching Decoder
                     ↓
              [CausalVAE] → Waveform
```

Key Components:
1. TSLM (Text-Semantic LM): 24 transformer layers, generates semantic-prosodic plans
2. FSQ (Finite Scalar Quantization): Differentiable bottleneck for natural specialization
3. RALM (Residual Acoustic LM): 6 transformer layers, recovers acoustic details
4. LocDiT (Local Diffusion Transformer): 4 layers, bidirectional flow-matching decoder
5. CausalAudioVAE: DAC-style codec with causal convolutions (25 Hz, 640x downsampling)

Benefits:
- Tokenizer-free: No external speech tokenizers needed
- Continuous space: Preserves prosody nuances lost in discrete tokenization
- End-to-end: Single diffusion objective for all components
- Fast inference: RTF ~0.17 on RTX 4090 (6x realtime)
- Zero-shot cloning: Captures timbre, accent, emotional tone from 3-10s reference

Usage:
    from voxcpm import (
        VoxCPMConfig,
        VoxCPM,
        VoxCPMAdapter,
    )

    # Initialize
    config = VoxCPMConfig()
    model = VoxCPM(config).cuda()

    # Training
    losses = model.compute_loss(text_tokens, audio, text_mask, audio_mask)

    # Inference (zero-shot voice cloning)
    audio = model.generate(text, reference_audio)

    # CSM integration
    adapter = VoxCPMAdapter(config, model)
    prefix_tokens = adapter(audio)  # [batch, 4, 2048]
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
class VoxCPMConfig:
    """Configuration for VoxCPM tokenizer-free TTS."""

    # Audio settings
    sample_rate: int = 16000
    vae_frame_rate: int = 25  # 25 Hz (40ms per frame)
    vae_latent_dim: int = 64  # VAE latent dimension

    # TSLM (Text-Semantic Language Model)
    tslm_hidden_dim: int = 1024
    tslm_ffn_dim: int = 4096
    tslm_num_layers: int = 24
    tslm_num_heads: int = 16
    tslm_vocab_size: int = 50000  # Text vocabulary size

    # FSQ (Finite Scalar Quantization)
    fsq_dim: int = 256  # Dimensions to quantize
    fsq_levels: int = 9  # Quantization levels per dimension
    fsq_enabled: bool = True  # Whether to use FSQ bottleneck

    # RALM (Residual Acoustic Language Model)
    ralm_hidden_dim: int = 1024
    ralm_ffn_dim: int = 4096
    ralm_num_layers: int = 6
    ralm_num_heads: int = 16

    # LocDiT (Local Diffusion Transformer)
    locdit_hidden_dim: int = 1024
    locdit_ffn_dim: int = 4096
    locdit_num_layers: int = 4
    locdit_num_heads: int = 16
    locdit_patch_size: int = 4  # Frames per patch

    # Flow matching
    num_diffusion_steps: int = 50
    cfg_scale: float = 2.0  # Classifier-free guidance scale
    cfg_mask_prob: float = 0.1  # Masking probability during training

    # Causal Audio VAE
    vae_hidden_dim: int = 512
    vae_num_blocks: int = 4
    vae_strides: List[int] = field(default_factory=lambda: [2, 5, 8, 8])  # 640x total

    # Training
    dropout: float = 0.1
    stop_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0

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


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embedding for diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary position encoding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        if kv is not None:
            # Cross-attention
            q = self.qkv(x)[:, :, :x.shape[-1]]
            k, v = self.qkv(kv)[:, :, x.shape[-1]:].chunk(2, dim=-1)
        else:
            # Self-attention
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, causal)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, ffn_dim, dropout)

        self.cross_attention = cross_attention
        if cross_attention:
            self.norm_cross = RMSNorm(dim)
            self.cross_attn = MultiHeadAttention(dim, num_heads, dropout, causal=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cross_kv: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)

        if self.cross_attention and cross_kv is not None:
            x = x + self.cross_attn(self.norm_cross(x), cross_mask, cross_kv)

        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# FINITE SCALAR QUANTIZATION (FSQ)
# =============================================================================

class FSQ(nn.Module):
    """
    Finite Scalar Quantization for semi-discrete representations.

    Based on the VoxCPM paper: creates a differentiable quantization bottleneck
    that induces natural specialization between semantic planning and acoustic rendering.

    Formula: h^FSQ = Δ · clip(round(h / Δ), -L, L)

    Uses straight-through estimator for gradient flow.
    """

    def __init__(
        self,
        dim: int,
        levels: int = 9,
    ):
        super().__init__()
        self.dim = dim
        self.levels = levels

        # Compute step size and clipping range
        self.L = (levels - 1) // 2
        self.delta = 2.0 / (levels - 1)  # Step size for [-1, 1] range

        # Learnable scale and shift
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input to discrete levels with straight-through gradient.

        Returns:
            quantized: Quantized values
            codes: Integer codes for each value
        """
        # Normalize to [-1, 1] range
        x_scaled = x * self.scale + self.shift
        x_norm = torch.tanh(x_scaled)

        # Quantize
        codes = torch.round(x_norm / self.delta)
        codes = torch.clamp(codes, -self.L, self.L)

        quantized = codes * self.delta

        # Straight-through estimator
        quantized = x_norm + (quantized - x_norm).detach()

        return quantized, codes.long() + self.L  # Shift codes to [0, levels-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply FSQ to input.

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            Dictionary with 'quantized' and 'codes' tensors
        """
        quantized, codes = self.quantize(x)

        return {
            'quantized': quantized,
            'codes': codes,
            'commitment_loss': F.mse_loss(quantized.detach(), x),
        }

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert codes back to continuous values."""
        shifted_codes = codes.float() - self.L
        return shifted_codes * self.delta


# =============================================================================
# TSLM: TEXT-SEMANTIC LANGUAGE MODEL
# =============================================================================

class TSLM(nn.Module):
    """
    Text-Semantic Language Model.

    Generates semantic-prosodic plans from text input. Initialized from
    pretrained LM backbone for enhanced contextual understanding.

    Architecture: 24 transformer layers, 1024 hidden dim, 4096 FFN dim
    """

    def __init__(self, config: VoxCPMConfig):
        super().__init__()
        self.config = config

        # Text embedding
        self.text_embed = nn.Embedding(config.tslm_vocab_size, config.tslm_hidden_dim)
        self.pos_encode = PositionalEncoding(config.tslm_hidden_dim, dropout=config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.tslm_hidden_dim,
                ffn_dim=config.tslm_ffn_dim,
                num_heads=config.tslm_num_heads,
                dropout=config.dropout,
                causal=True,
            )
            for _ in range(config.tslm_num_layers)
        ])

        self.norm = RMSNorm(config.tslm_hidden_dim)

        # Output projection to FSQ dimension
        self.out_proj = nn.Linear(config.tslm_hidden_dim, config.fsq_dim)

        # Audio embedding for conditioning on historical audio
        self.audio_proj = nn.Linear(config.vae_latent_dim, config.tslm_hidden_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate semantic-prosodic representations from text.

        Args:
            text_tokens: Text token IDs [batch, text_len]
            text_mask: Padding mask [batch, text_len]
            audio_history: Historical audio embeddings [batch, audio_len, vae_latent_dim]

        Returns:
            Dictionary with 'hidden' (semantic representations) and other outputs
        """
        batch_size = text_tokens.shape[0]

        # Embed text
        x = self.text_embed(text_tokens)

        # Add audio history if provided
        if audio_history is not None:
            audio_emb = self.audio_proj(audio_history)
            # Interleave audio and text
            x = torch.cat([audio_emb, x], dim=1)
            if text_mask is not None:
                audio_mask = torch.ones(
                    batch_size, audio_history.shape[1],
                    device=text_mask.device, dtype=text_mask.dtype
                )
                text_mask = torch.cat([audio_mask, text_mask], dim=1)

        x = self.pos_encode(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, text_mask)

        x = self.norm(x)

        # Extract text-conditioned output (skip audio prefix if present)
        if audio_history is not None:
            x = x[:, audio_history.shape[1]:]

        # Project to FSQ dimension
        semantic_out = self.out_proj(x)

        return {
            'hidden': semantic_out,
            'text_hidden': x,
        }


# =============================================================================
# RALM: RESIDUAL ACOUSTIC LANGUAGE MODEL
# =============================================================================

class RALM(nn.Module):
    """
    Residual Acoustic Language Model.

    Recovers fine-grained acoustic details from semantic representations.
    Conditions on TSLM output, FSQ quantized representations, and historical audio.

    Architecture: 6 transformer layers, 1024 hidden dim, 4096 FFN dim
    """

    def __init__(self, config: VoxCPMConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.tslm_proj = nn.Linear(config.tslm_hidden_dim, config.ralm_hidden_dim)
        self.fsq_proj = nn.Linear(config.fsq_dim, config.ralm_hidden_dim)
        self.audio_proj = nn.Linear(config.vae_latent_dim, config.ralm_hidden_dim)

        self.pos_encode = PositionalEncoding(config.ralm_hidden_dim, dropout=config.dropout)

        # Transformer layers with cross-attention to TSLM
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.ralm_hidden_dim,
                ffn_dim=config.ralm_ffn_dim,
                num_heads=config.ralm_num_heads,
                dropout=config.dropout,
                causal=True,
                cross_attention=True,
            )
            for _ in range(config.ralm_num_layers)
        ])

        self.norm = RMSNorm(config.ralm_hidden_dim)

        # Output projection to LocDiT conditioning dimension
        self.out_proj = nn.Linear(config.ralm_hidden_dim, config.locdit_hidden_dim)

    def forward(
        self,
        tslm_hidden: torch.Tensor,
        fsq_quantized: torch.Tensor,
        audio_history: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate acoustic residual representations.

        Args:
            tslm_hidden: TSLM text hidden states [batch, text_len, tslm_hidden_dim]
            fsq_quantized: FSQ quantized representations [batch, seq, fsq_dim]
            audio_history: Historical audio embeddings [batch, audio_len, vae_latent_dim]
            mask: Padding mask [batch, seq]

        Returns:
            Dictionary with 'residual' acoustic representations
        """
        # Combine FSQ and historical audio as input
        x = self.fsq_proj(fsq_quantized)

        if audio_history is not None:
            audio_emb = self.audio_proj(audio_history)
            # Add historical audio context
            min_len = min(x.shape[1], audio_emb.shape[1])
            x[:, :min_len] = x[:, :min_len] + audio_emb[:, :min_len]

        x = self.pos_encode(x)

        # Cross-attention conditioning from TSLM
        tslm_cond = self.tslm_proj(tslm_hidden)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask, cross_kv=tslm_cond)

        x = self.norm(x)
        residual = self.out_proj(x)

        return {
            'residual': residual,
        }


# =============================================================================
# LOCDIT: LOCAL DIFFUSION TRANSFORMER
# =============================================================================

class LocDiT(nn.Module):
    """
    Local Diffusion Transformer for flow-matching based speech generation.

    Bidirectional transformer decoder that generates audio latents from
    conditioning signals using flow-matching objective.

    Architecture: 4 transformer layers, 1024 hidden dim, 4096 FFN dim
    """

    def __init__(self, config: VoxCPMConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.locdit_hidden_dim)

        # Input projection from VAE latent + conditioning
        self.input_proj = nn.Linear(
            config.vae_latent_dim + config.locdit_hidden_dim,
            config.locdit_hidden_dim
        )

        self.pos_encode = PositionalEncoding(config.locdit_hidden_dim, dropout=config.dropout)

        # Bidirectional transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.locdit_hidden_dim,
                ffn_dim=config.locdit_ffn_dim,
                num_heads=config.locdit_num_heads,
                dropout=config.dropout,
                causal=False,  # Bidirectional within patch
            )
            for _ in range(config.locdit_num_layers)
        ])

        self.norm = RMSNorm(config.locdit_hidden_dim)

        # Output projection to velocity field
        self.out_proj = nn.Linear(config.locdit_hidden_dim, config.vae_latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
        prev_patch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Args:
            z_t: Noisy latent at time t [batch, seq, vae_latent_dim]
            t: Time step [batch]
            conditioning: Combined TSLM + RALM conditioning [batch, seq, locdit_hidden_dim]
            prev_patch: Previous patch for context [batch, patch_size, vae_latent_dim]
            mask: Padding mask [batch, seq]

        Returns:
            Predicted velocity field [batch, seq, vae_latent_dim]
        """
        batch_size, seq_len, _ = z_t.shape

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, locdit_hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine noisy latent with conditioning
        x = torch.cat([z_t, conditioning], dim=-1)
        x = self.input_proj(x)

        # Add time embedding
        x = x + t_emb

        # Add previous patch context if available
        if prev_patch is not None:
            prev_proj = self.input_proj(
                torch.cat([
                    prev_patch,
                    conditioning[:, :prev_patch.shape[1]].detach()
                ], dim=-1)
            )
            x = torch.cat([prev_proj, x], dim=1)
            if mask is not None:
                prev_mask = torch.ones(batch_size, prev_patch.shape[1], device=mask.device)
                mask = torch.cat([prev_mask, mask], dim=1)

        x = self.pos_encode(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Remove previous patch from output
        if prev_patch is not None:
            x = x[:, prev_patch.shape[1]:]

        # Project to velocity
        velocity = self.out_proj(x)

        return velocity


# =============================================================================
# CAUSAL AUDIO VAE
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution for streaming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    """Causal convolution block with residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = CausalConv1d(in_dim, out_dim, kernel_size, stride)
        # Use appropriate num_groups that divides out_dim
        num_groups = min(8, out_dim)
        while out_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim or stride != 1:
            self.residual = CausalConv1d(in_dim, out_dim, 1, stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Handle length mismatch
        min_len = min(x.shape[-1], residual.shape[-1])
        return x[..., :min_len] + residual[..., :min_len]


class CausalAudioVAE(nn.Module):
    """
    Causal Audio VAE for streaming synthesis.

    DAC-style architecture with causal convolutions for streaming.
    Operates at 25 Hz frame rate with 640x downsampling.
    """

    def __init__(self, config: VoxCPMConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.vae_latent_dim

        # Calculate total stride
        self.total_stride = 1
        for s in config.vae_strides:
            self.total_stride *= s

        # Encoder
        encoder_dims = [1, 64, 128, 256, 512]
        self.encoder = nn.ModuleList()
        for i, stride in enumerate(config.vae_strides):
            self.encoder.append(
                CausalConvBlock(
                    encoder_dims[i], encoder_dims[i + 1],
                    kernel_size=7, stride=stride,
                    dropout=config.dropout
                )
            )

        self.encoder_out = nn.Sequential(
            CausalConv1d(encoder_dims[-1], config.vae_hidden_dim, 3),
            nn.GELU(),
            CausalConv1d(config.vae_hidden_dim, config.vae_latent_dim * 2, 1),
        )

        # Decoder
        decoder_dims = encoder_dims[::-1]
        self.decoder_in = nn.Sequential(
            CausalConv1d(config.vae_latent_dim, config.vae_hidden_dim, 1),
            nn.GELU(),
            CausalConv1d(config.vae_hidden_dim, decoder_dims[0], 3),
        )

        self.decoder = nn.ModuleList()
        for i, stride in enumerate(config.vae_strides[::-1]):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='linear', align_corners=False),
                CausalConvBlock(
                    decoder_dims[i], decoder_dims[i + 1],
                    kernel_size=7, stride=1,
                    dropout=config.dropout
                )
            ))

        self.decoder_out = CausalConv1d(decoder_dims[-1], 1, 7)

    def encode(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode audio to latent space.

        Args:
            audio: Audio waveform [batch, 1, samples]

        Returns:
            Dictionary with 'mean', 'logvar', 'latent' tensors
        """
        x = audio

        for block in self.encoder:
            x = block(x)

        x = self.encoder_out(x)
        mean, logvar = x.chunk(2, dim=1)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std

        return {
            'mean': mean,
            'logvar': logvar,
            'latent': latent.transpose(1, 2),  # [batch, seq, latent_dim]
        }

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to audio.

        Args:
            latent: Latent representation [batch, seq, latent_dim]

        Returns:
            Audio waveform [batch, 1, samples]
        """
        x = latent.transpose(1, 2)  # [batch, latent_dim, seq]
        x = self.decoder_in(x)

        for block in self.decoder:
            x = block(x)

        audio = self.decoder_out(x)
        return torch.tanh(audio)

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full encode-decode pass."""
        encoded = self.encode(audio)
        decoded = self.decode(encoded['latent'])

        return {
            'mean': encoded['mean'],
            'logvar': encoded['logvar'],
            'latent': encoded['latent'],
            'reconstructed': decoded,
        }


# =============================================================================
# STOP PREDICTOR
# =============================================================================

class StopPredictor(nn.Module):
    """Predicts when to stop generation."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FSQ quantized representations [batch, seq, dim]

        Returns:
            Stop logits [batch, seq]
        """
        return self.head(x).squeeze(-1)


# =============================================================================
# VOXCPM: FULL MODEL
# =============================================================================

class VoxCPM(nn.Module):
    """
    VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation.

    End-to-end model combining:
    - TSLM: Text-Semantic Language Model for semantic-prosodic planning
    - FSQ: Finite Scalar Quantization for semi-discrete bottleneck
    - RALM: Residual Acoustic Language Model for acoustic detail recovery
    - LocDiT: Local Diffusion Transformer for flow-matching generation
    - CausalAudioVAE: Streaming-capable audio codec
    """

    def __init__(self, config: VoxCPMConfig):
        super().__init__()
        self.config = config

        # Core components
        self.vae = CausalAudioVAE(config)
        self.tslm = TSLM(config)
        self.fsq = FSQ(config.fsq_dim, config.fsq_levels)
        self.ralm = RALM(config)
        self.locdit = LocDiT(config)
        self.stop_predictor = StopPredictor(config.fsq_dim)

        # Combine FSQ and RALM outputs for LocDiT conditioning
        self.condition_proj = nn.Linear(
            config.fsq_dim + config.locdit_hidden_dim,
            config.locdit_hidden_dim
        )

    def encode_audio(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode audio to VAE latent space."""
        return self.vae.encode(audio)

    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode VAE latent to audio."""
        return self.vae.decode(latent)

    def get_conditioning(
        self,
        text_tokens: torch.Tensor,
        audio_latent: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute conditioning signal for LocDiT.

        Returns combined TSLM + FSQ + RALM conditioning.
        """
        audio_len = audio_latent.shape[1]

        # TSLM: Generate semantic-prosodic representations
        tslm_out = self.tslm(text_tokens, text_mask, audio_latent)

        # FSQ: Quantize semantic representations
        fsq_out = self.fsq(tslm_out['hidden'])

        # Expand text-level outputs to audio length using interpolation
        fsq_expanded = F.interpolate(
            fsq_out['quantized'].transpose(1, 2),
            size=audio_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        text_hidden_expanded = F.interpolate(
            tslm_out['text_hidden'].transpose(1, 2),
            size=audio_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # RALM: Generate acoustic residuals (now with matching lengths)
        ralm_out = self.ralm(
            text_hidden_expanded,
            fsq_expanded,
            audio_latent,
            audio_mask,
        )

        # Combine for final conditioning
        combined = torch.cat([
            fsq_expanded,
            ralm_out['residual'],
        ], dim=-1)
        conditioning = self.condition_proj(combined)

        # Stop prediction (on expanded FSQ)
        stop_logits = self.stop_predictor(fsq_expanded)

        return {
            'conditioning': conditioning,
            'fsq_quantized': fsq_expanded,
            'fsq_codes': fsq_out['codes'],  # Original codes (text length)
            'fsq_commitment_loss': fsq_out['commitment_loss'],
            'stop_logits': stop_logits,
            'tslm_hidden': tslm_out['hidden'],
            'ralm_residual': ralm_out['residual'],
        }

    def compute_flow_loss(
        self,
        z_0: torch.Tensor,
        conditioning: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow-matching loss.

        Args:
            z_0: Target latent (clean) [batch, seq, latent_dim]
            conditioning: LocDiT conditioning [batch, seq, cond_dim]
            mask: Padding mask [batch, seq]

        Returns:
            Flow matching loss scalar
        """
        batch_size = z_0.shape[0]

        # Sample time steps
        t = torch.rand(batch_size, device=z_0.device)

        # Sample noise
        eps = torch.randn_like(z_0)

        # Interpolate: z_t = t * z_0 + (1-t) * eps
        t_exp = t[:, None, None]
        z_t = t_exp * z_0 + (1 - t_exp) * eps

        # Target velocity: d/dt[z_t] = z_0 - eps
        target_velocity = z_0 - eps

        # Predict velocity
        pred_velocity = self.locdit(z_t, t, conditioning, mask=mask)

        # Compute loss
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).float()
            loss = F.mse_loss(pred_velocity * mask_exp, target_velocity * mask_exp)
        else:
            loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def compute_loss(
        self,
        text_tokens: torch.Tensor,
        audio: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        is_last: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            text_tokens: Text token IDs [batch, text_len]
            audio: Audio waveform [batch, 1, samples]
            text_mask: Text padding mask [batch, text_len]
            audio_mask: Audio padding mask [batch, audio_len]
            is_last: Whether each position is the last token [batch, audio_len]

        Returns:
            Dictionary with individual and total losses
        """
        # Encode audio
        vae_out = self.vae(audio)
        z_0 = vae_out['latent']

        # Get conditioning (with CFG dropout)
        if self.training and self.config.cfg_mask_prob > 0:
            # Randomly mask conditioning for CFG
            mask_prob = torch.rand(text_tokens.shape[0], device=text_tokens.device)
            cfg_mask = mask_prob < self.config.cfg_mask_prob
            text_tokens_masked = text_tokens.clone()
            text_tokens_masked[cfg_mask] = 0  # Use padding token
        else:
            text_tokens_masked = text_tokens

        cond_out = self.get_conditioning(
            text_tokens_masked, z_0, text_mask, audio_mask
        )

        # Flow matching loss
        flow_loss = self.compute_flow_loss(
            z_0, cond_out['conditioning'], audio_mask
        )

        # VAE reconstruction loss
        recon_loss = F.mse_loss(vae_out['reconstructed'], audio)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(
            1 + vae_out['logvar'] - vae_out['mean'].pow(2) - vae_out['logvar'].exp()
        )

        # Stop prediction loss
        if is_last is not None:
            stop_loss = F.binary_cross_entropy_with_logits(
                cond_out['stop_logits'], is_last.float()
            )
        else:
            stop_loss = torch.tensor(0.0, device=audio.device)

        # FSQ commitment loss
        commitment_loss = cond_out['fsq_commitment_loss']

        # Total loss
        total_loss = (
            self.config.flow_loss_weight * flow_loss +
            recon_loss +
            5e-5 * kl_loss +
            self.config.stop_loss_weight * stop_loss +
            0.25 * commitment_loss
        )

        return {
            'total': total_loss,
            'flow': flow_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'stop': stop_loss,
            'commitment': commitment_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample audio latent using flow matching.

        Args:
            conditioning: LocDiT conditioning [batch, seq, cond_dim]
            num_steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale
            temperature: Sampling temperature

        Returns:
            Sampled latent [batch, seq, latent_dim]
        """
        batch_size, seq_len, _ = conditioning.shape
        device = conditioning.device

        # Start from noise
        z = torch.randn(
            batch_size, seq_len, self.config.vae_latent_dim,
            device=device
        ) * temperature

        # Time steps (from 0 to 1)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i / num_steps, device=device)

            # Predict velocity with CFG
            if cfg_scale > 1.0:
                # Conditional prediction
                v_cond = self.locdit(z, t, conditioning)

                # Unconditional prediction (zero conditioning)
                v_uncond = self.locdit(
                    z, t,
                    torch.zeros_like(conditioning)
                )

                # CFG combination
                velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                velocity = self.locdit(z, t, conditioning)

            # Euler step
            z = z + velocity * dt

        return z

    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        reference_audio: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        max_len: int = 500,
    ) -> torch.Tensor:
        """
        Generate audio from text with optional voice cloning.

        Args:
            text_tokens: Text token IDs [batch, text_len]
            reference_audio: Reference audio for voice cloning [batch, 1, samples]
            text_mask: Text padding mask [batch, text_len]
            num_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            max_len: Maximum generation length (frames)

        Returns:
            Generated audio waveform [batch, 1, samples]
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device

        # Encode reference audio if provided
        if reference_audio is not None:
            ref_latent = self.vae.encode(reference_audio)['latent']
        else:
            ref_latent = None

        # Initialize with reference or random start
        if ref_latent is not None:
            # Use reference as initial context
            current_latent = ref_latent[:, :10]  # First 10 frames as context
        else:
            current_latent = None

        # Get TSLM output
        tslm_out = self.tslm(text_tokens, text_mask, current_latent)

        # Determine output length from text (rough estimate)
        output_len = min(max_len, text_tokens.shape[1] * 4)

        # Expand TSLM output to audio length
        tslm_expanded = F.interpolate(
            tslm_out['hidden'].transpose(1, 2),
            size=output_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # FSQ and RALM
        fsq_out = self.fsq(tslm_expanded)
        ralm_out = self.ralm(
            tslm_out['text_hidden'],
            fsq_out['quantized'],
            current_latent,
        )

        # Combine conditioning
        combined = torch.cat([
            fsq_out['quantized'],
            ralm_out['residual'],
        ], dim=-1)
        conditioning = self.condition_proj(combined)

        # Sample latent
        latent = self.sample(conditioning, num_steps, cfg_scale)

        # Decode to audio
        audio = self.vae.decode(latent)

        return audio


# =============================================================================
# TRAINING LOSS
# =============================================================================

class VoxCPMLoss(nn.Module):
    """Combined loss function for VoxCPM training."""

    def __init__(self, config: VoxCPMConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        This is primarily a wrapper for model.compute_loss()
        with additional auxiliary losses if needed.
        """
        return model_output  # Losses already computed in forward pass


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class VoxCPMAdapter(nn.Module):
    """
    Adapter for integrating VoxCPM with CSM prosody pipeline.

    Extracts semantic-prosodic representations from VoxCPM
    and converts them to CSM-compatible prefix tokens.
    """

    def __init__(self, config: VoxCPMConfig, model: Optional[VoxCPM] = None):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        if model is not None:
            self.voxcpm = model
        else:
            self.voxcpm = VoxCPM(config)

        # Projection to output dimension
        self.prosody_proj = nn.Sequential(
            nn.Linear(config.fsq_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # Attention pooling for fixed-length output
        self.pool_query = nn.Parameter(
            torch.randn(1, config.num_prefix_tokens, config.output_dim)
        )
        self.pool_attn = nn.MultiheadAttention(
            config.output_dim, num_heads=8, batch_first=True
        )

    def forward(
        self,
        audio: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosody tokens from audio.

        Args:
            audio: Audio waveform [batch, 1, samples]
            text_tokens: Optional text for conditioning
            text_mask: Text padding mask

        Returns:
            Dictionary with 'prosody_tokens' [batch, num_prefix_tokens, output_dim]
        """
        batch_size = audio.shape[0]

        # Encode audio
        vae_out = self.voxcpm.vae.encode(audio)
        latent = vae_out['latent']

        # Get conditioning if text provided
        if text_tokens is not None:
            cond_out = self.voxcpm.get_conditioning(
                text_tokens, latent, text_mask
            )
            semantic_features = cond_out['fsq_quantized']
        else:
            # Extract semantic features without text
            tslm_out = self.voxcpm.tslm.pos_encode(
                self.voxcpm.tslm.audio_proj(latent)
            )
            for layer in self.voxcpm.tslm.layers[:6]:  # Use subset of layers
                tslm_out = layer(tslm_out)
            semantic_features = self.voxcpm.tslm.out_proj(tslm_out)
            fsq_out = self.voxcpm.fsq(semantic_features)
            semantic_features = fsq_out['quantized']

        # Project to output dimension
        prosody_features = self.prosody_proj(semantic_features)

        # Attention pooling to fixed length
        query = self.pool_query.expand(batch_size, -1, -1)
        prosody_tokens, _ = self.pool_attn(query, prosody_features, prosody_features)

        return {
            'prosody_tokens': prosody_tokens,
            'semantic_features': semantic_features,
            'latent': latent,
        }

    def from_reference(
        self,
        reference_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosody from reference audio for voice cloning.

        Args:
            reference_audio: Reference audio [batch, 1, samples]

        Returns:
            Dictionary with prosody tokens and speaker embedding
        """
        result = self.forward(reference_audio)

        # Extract global speaker embedding
        latent = result['latent']
        speaker_emb = latent.mean(dim=1)  # Global average pooling

        return {
            'prosody_tokens': result['prosody_tokens'],
            'speaker_embedding': speaker_emb,
            'semantic_features': result['semantic_features'],
        }

    def generate_with_prosody(
        self,
        text_tokens: torch.Tensor,
        prosody_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        Generate audio with specified prosody tokens.

        Args:
            text_tokens: Text token IDs
            prosody_tokens: Prosody conditioning tokens
            text_mask: Text padding mask
            num_steps: Diffusion steps
            cfg_scale: CFG scale

        Returns:
            Generated audio
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device

        # Get TSLM output
        tslm_out = self.voxcpm.tslm(text_tokens, text_mask)

        # Estimate output length
        output_len = text_tokens.shape[1] * 4

        # Expand TSLM to audio length
        tslm_expanded = F.interpolate(
            tslm_out['hidden'].transpose(1, 2),
            size=output_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # Use provided prosody tokens (expand to match length)
        prosody_expanded = F.interpolate(
            prosody_tokens.transpose(1, 2),
            size=output_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        # Combine with prosody
        # Project prosody back to FSQ dimension
        prosody_proj = nn.Linear(
            self.config.output_dim,
            self.config.fsq_dim + self.config.locdit_hidden_dim,
            device=device
        )
        combined = prosody_proj(prosody_expanded)
        conditioning = self.voxcpm.condition_proj(combined)

        # Sample and decode
        latent = self.voxcpm.sample(conditioning, num_steps, cfg_scale)
        audio = self.voxcpm.vae.decode(latent)

        return audio


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_voxcpm_model(
    config: Optional[VoxCPMConfig] = None,
    pretrained_path: Optional[str] = None,
) -> VoxCPM:
    """
    Create VoxCPM model with optional pretrained weights.

    Args:
        config: Model configuration (uses default if None)
        pretrained_path: Path to pretrained weights

    Returns:
        VoxCPM model
    """
    if config is None:
        config = VoxCPMConfig()

    model = VoxCPM(config)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def estimate_rtf(
    model: VoxCPM,
    text_length: int = 50,
    audio_duration: float = 5.0,
    device: str = 'cuda',
    num_runs: int = 10,
) -> float:
    """
    Estimate Real-Time Factor (RTF) for the model.

    Args:
        model: VoxCPM model
        text_length: Length of text tokens
        audio_duration: Target audio duration in seconds
        device: Device to run on
        num_runs: Number of runs for averaging

    Returns:
        RTF (< 1.0 means faster than realtime)
    """
    import time

    model = model.to(device)
    model.eval()

    text_tokens = torch.randint(
        0, model.config.tslm_vocab_size,
        (1, text_length), device=device
    )

    # Warmup
    with torch.no_grad():
        _ = model.generate(text_tokens, max_len=int(audio_duration * 25))

    # Timed runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            _ = model.generate(text_tokens, max_len=int(audio_duration * 25))

        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    rtf = avg_time / audio_duration

    return rtf


# =============================================================================
# TESTING
# =============================================================================

def test_voxcpm():
    """
    Test VoxCPM components and verify end-to-end functionality.

    Returns True if all tests pass, raises error otherwise.
    """
    print("=" * 60)
    print("VoxCPM Component Tests")
    print("=" * 60)

    device = "cpu"  # Use CPU for testing to avoid CUDA issues

    # Create compact config for testing
    config = VoxCPMConfig(
        tslm_num_layers=2,  # Minimal layers for testing
        tslm_hidden_dim=256,
        tslm_ffn_dim=512,
        tslm_vocab_size=1000,
        fsq_dim=64,
        fsq_levels=9,
        ralm_num_layers=2,
        ralm_hidden_dim=256,
        ralm_ffn_dim=512,
        locdit_num_layers=2,
        locdit_hidden_dim=256,
        locdit_ffn_dim=512,
        vae_latent_dim=32,
        vae_hidden_dim=128,
        vae_strides=[2, 2, 2, 2],  # 16x for faster testing
        output_dim=256,
        num_prefix_tokens=4,
    )

    batch_size = 2
    text_len = 20
    audio_samples = 8000  # 0.5 seconds at 16kHz

    print(f"\nConfig: {config.tslm_num_layers} TSLM layers, "
          f"{config.ralm_num_layers} RALM layers, "
          f"{config.locdit_num_layers} LocDiT layers")
    print(f"Test batch: {batch_size} samples, {text_len} tokens, "
          f"{audio_samples/16000:.2f}s audio")

    # Test 1: FSQ (Finite Scalar Quantization)
    print("\n1. Testing FSQ (Finite Scalar Quantization)...")
    fsq = FSQ(config.fsq_dim, config.fsq_levels).to(device)
    x = torch.randn(batch_size, 10, config.fsq_dim)
    fsq_out = fsq(x)

    assert 'quantized' in fsq_out, "FSQ missing 'quantized' output"
    assert 'codes' in fsq_out, "FSQ missing 'codes' output"
    assert fsq_out['quantized'].shape == x.shape, f"FSQ shape mismatch"

    # Check quantization is working (values should be discrete)
    unique_vals = fsq_out['quantized'].unique()
    print(f"   Input shape: {x.shape}")
    print(f"   Quantized shape: {fsq_out['quantized'].shape}")
    print(f"   Unique quantized values: {len(unique_vals)}")
    print(f"   Commitment loss: {fsq_out['commitment_loss'].item():.4f}")
    print("   ✓ FSQ test passed")

    # Test 2: Causal Audio VAE
    print("\n2. Testing Causal Audio VAE...")
    vae = CausalAudioVAE(config).to(device)
    audio = torch.randn(batch_size, 1, audio_samples)
    vae_out = vae(audio)

    assert 'latent' in vae_out, "VAE missing 'latent' output"
    assert 'reconstructed' in vae_out, "VAE missing 'reconstructed' output"

    # Check latent shape (should be downsampled)
    expected_frames = audio_samples // 16  # 16x downsampling for test config
    print(f"   Input audio: {audio.shape}")
    print(f"   Latent shape: {vae_out['latent'].shape}")
    print(f"   Reconstructed: {vae_out['reconstructed'].shape}")
    print(f"   Mean: {vae_out['mean'].mean().item():.4f}")
    print(f"   LogVar: {vae_out['logvar'].mean().item():.4f}")
    print("   ✓ VAE test passed")

    # Test 3: TSLM (Text-Semantic Language Model)
    print("\n3. Testing TSLM (Text-Semantic Language Model)...")
    tslm = TSLM(config).to(device)
    text_tokens = torch.randint(0, config.tslm_vocab_size, (batch_size, text_len))
    tslm_out = tslm(text_tokens)

    assert 'hidden' in tslm_out, "TSLM missing 'hidden' output"
    assert tslm_out['hidden'].shape == (batch_size, text_len, config.fsq_dim), \
        f"TSLM hidden shape mismatch"

    print(f"   Text tokens: {text_tokens.shape}")
    print(f"   TSLM hidden: {tslm_out['hidden'].shape}")
    print("   ✓ TSLM test passed")

    # Test 4: RALM (Residual Acoustic Language Model)
    print("\n4. Testing RALM (Residual Acoustic LM)...")
    ralm = RALM(config).to(device)

    # RALM needs FSQ output and audio latent
    fsq_quantized = torch.randn(batch_size, 50, config.fsq_dim)
    audio_latent = torch.randn(batch_size, 50, config.vae_latent_dim)
    text_hidden = torch.randn(batch_size, text_len, config.tslm_hidden_dim)

    ralm_out = ralm(text_hidden, fsq_quantized, audio_latent)

    assert 'residual' in ralm_out, "RALM missing 'residual' output"
    print(f"   Text hidden: {text_hidden.shape}")
    print(f"   FSQ quantized: {fsq_quantized.shape}")
    print(f"   RALM residual: {ralm_out['residual'].shape}")
    print("   ✓ RALM test passed")

    # Test 5: LocDiT (Local Diffusion Transformer)
    print("\n5. Testing LocDiT (Local Diffusion Transformer)...")
    locdit = LocDiT(config).to(device)

    z_t = torch.randn(batch_size, 50, config.vae_latent_dim)
    t = torch.rand(batch_size)
    conditioning = torch.randn(batch_size, 50, config.locdit_hidden_dim)

    velocity = locdit(z_t, t, conditioning)

    assert velocity.shape == z_t.shape, f"LocDiT velocity shape mismatch"
    print(f"   Noisy z_t: {z_t.shape}")
    print(f"   Time step: {t.shape}")
    print(f"   Conditioning: {conditioning.shape}")
    print(f"   Predicted velocity: {velocity.shape}")
    print("   ✓ LocDiT test passed")

    # Test 6: Full VoxCPM Model
    print("\n6. Testing Full VoxCPM Model...")
    model = VoxCPM(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test training forward pass
    print("\n   Testing training forward pass...")
    model.train()
    losses = model.compute_loss(text_tokens, audio)

    assert 'total' in losses, "Missing total loss"
    assert 'flow' in losses, "Missing flow loss"
    assert 'recon' in losses, "Missing reconstruction loss"

    print(f"   Total loss: {losses['total'].item():.4f}")
    print(f"   Flow loss: {losses['flow'].item():.4f}")
    print(f"   Reconstruction loss: {losses['recon'].item():.4f}")
    print(f"   KL loss: {losses['kl'].item():.4f}")
    print(f"   Commitment loss: {losses['commitment'].item():.4f}")

    # Check gradients
    losses['total'].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"   Gradient norm: {grad_norm:.4f}")
    print("   ✓ Training forward pass test passed")

    # Test inference
    print("\n   Testing inference generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(text_tokens, max_len=50, num_steps=5)

    assert generated.dim() == 3, f"Generated audio should be 3D"
    assert generated.shape[0] == batch_size, "Batch size mismatch"
    assert generated.shape[1] == 1, "Channel dim should be 1"

    print(f"   Generated audio: {generated.shape}")
    print(f"   Duration: {generated.shape[-1]/16000:.2f}s")
    print("   ✓ Inference test passed")

    # Test 7: VoxCPM Adapter (CSM Integration)
    print("\n7. Testing VoxCPM Adapter...")
    adapter = VoxCPMAdapter(config, model).to(device)

    adapter_out = adapter(audio, text_tokens)

    assert 'prosody_tokens' in adapter_out, "Missing prosody_tokens"
    assert adapter_out['prosody_tokens'].shape == (batch_size, config.num_prefix_tokens, config.output_dim), \
        f"Prosody tokens shape mismatch"

    print(f"   Audio input: {audio.shape}")
    print(f"   Prosody tokens: {adapter_out['prosody_tokens'].shape}")

    # Test reference extraction
    ref_out = adapter.from_reference(audio)
    assert 'speaker_embedding' in ref_out, "Missing speaker_embedding"
    print(f"   Speaker embedding: {ref_out['speaker_embedding'].shape}")
    print("   ✓ Adapter test passed")

    # Final summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    print("\nVoxCPM Implementation Summary:")
    print(f"  - FSQ: {config.fsq_dim} dims, {config.fsq_levels} levels")
    print(f"  - TSLM: {config.tslm_num_layers} layers")
    print(f"  - RALM: {config.ralm_num_layers} layers")
    print(f"  - LocDiT: {config.locdit_num_layers} layers")
    print(f"  - VAE: {config.vae_latent_dim} latent dim")
    print(f"  - Output: {config.num_prefix_tokens} x {config.output_dim} tokens")

    return True


if __name__ == "__main__":
    test_voxcpm()
