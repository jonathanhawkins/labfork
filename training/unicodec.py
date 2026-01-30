"""
UniCodec: Universal Speech Token Learning

Based on "UniCodec: Unified Speech Representation Learning via Efficient Tokenization"
(arXiv:2503.12115, IEEE J-STSP 2025).

Key Innovation: Unifies semantic and acoustic tokens into compact universal tokens
that preserve both linguistic AND paralinguistic (prosody) information.

Architecture:
```
Audio → [Global Encoder] → Global Token (speaker/style)
     ↓
     → [Local-Semantic Encoder] → Semantic Tokens (content + prosody)
     ↓
     → [Local-Residual Encoder] → Residual Tokens (acoustic details)
     ↓
     → SSL Distillation Loss (wav2vec2, HuBERT)
     ↓
[Decoder] → Reconstructed Audio
```

Key Technique: Knowledge Distillation from SSL Features
- wav2vec2/HuBERT provide rich self-supervised representations
- Local-semantic encoder learns to predict SSL features
- Preserves paralinguistic information (prosody, emotion) in tokens

Benefits:
- Prosody generation in first stage → more stable acoustic generation
- Compact tokens easy to predict in LMs
- Preserves paralinguistic attributes better than SpeechTokenizer
- Long-term consistency in output quality
- Unified representation for TTS, VC, and emotion control

Two-Stage Training:
1. Token Learning: Train codec with SSL distillation
2. Token Generation: Train LM to generate tokens from text

Usage:
    from unicodec import (
        UniCodecConfig,
        UniCodecEncoder,
        UniversalTokenizer,
        UniCodecLM,
        UniCodecAdapter,
    )

    # Initialize
    config = UniCodecConfig()
    model = UniCodec(config).cuda()

    # Extract universal tokens
    tokens = model.encode(audio)
    global_token = tokens['global_token']       # Speaker/style
    semantic_tokens = tokens['semantic_tokens'] # Content + prosody
    residual_tokens = tokens['residual_tokens'] # Acoustic details

    # Reconstruct from tokens
    audio_recon = model.decode(tokens)

    # CSM integration
    adapter = UniCodecAdapter(config, model)
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
class UniCodecConfig:
    """Configuration for UniCodec universal speech tokenization."""

    # Audio settings
    sample_rate: int = 16000
    hop_length: int = 320  # 20ms at 16kHz (50 Hz token rate)
    mel_dim: int = 80

    # Global encoder (speaker/style)
    global_dim: int = 256
    global_hidden_dim: int = 512
    global_num_layers: int = 3
    global_num_heads: int = 8

    # Local-semantic encoder (content + prosody)
    semantic_dim: int = 256
    semantic_hidden_dim: int = 512
    semantic_num_layers: int = 6
    semantic_num_heads: int = 8
    semantic_codebook_size: int = 1024  # Main semantic codebook
    semantic_num_quantizers: int = 2    # Number of VQ levels

    # Local-residual encoder (acoustic details)
    residual_dim: int = 128
    residual_hidden_dim: int = 256
    residual_num_layers: int = 4
    residual_num_heads: int = 4
    residual_codebook_size: int = 512
    residual_num_quantizers: int = 4  # RVQ levels for residual

    # SSL distillation targets
    ssl_model: str = "wav2vec2"  # wav2vec2 or hubert
    ssl_dim: int = 768           # SSL feature dimension
    ssl_layer: int = 7           # Which SSL layer to use (mid layers have prosody)
    distillation_weight: float = 1.0

    # Decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Prosody-specific settings
    prosody_num_tokens: int = 8    # Prosody tokens per global token
    prosody_hidden_dim: int = 256
    use_prosody_predictor: bool = True  # Predict prosody from text

    # Training
    dropout: float = 0.1
    commitment_cost: float = 0.25
    ema_decay: float = 0.99

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
    """1D Convolution block with residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim, out_dim, kernel_size,
            stride=stride, padding=kernel_size // 2
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

        # Handle length mismatch
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
    """Vector Quantizer with EMA codebook updates."""

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # Codebook
        self.embedding = nn.Embedding(codebook_size, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Quantize input.

        Args:
            x: [batch, seq, code_dim]

        Returns:
            Dict with z_q, indices, commitment_loss, perplexity
        """
        batch_size, seq_len, _ = x.shape
        flat_x = x.reshape(-1, self.code_dim)

        # Initialize from first batch
        if self.training and not self.initialized:
            self._init_from_data(flat_x)

        # Compute distances
        distances = (
            flat_x.pow(2).sum(dim=-1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=-1)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        )

        # Find nearest
        indices = distances.argmin(dim=-1)
        z_q = self.embedding(indices)

        # EMA update
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                new_size = encodings.sum(dim=0)
                new_sum = torch.matmul(encodings.t(), flat_x)

                self.ema_cluster_size.mul_(self.ema_decay).add_(new_size, alpha=1 - self.ema_decay)
                self.ema_sum.mul_(self.ema_decay).add_(new_sum, alpha=1 - self.ema_decay)

                n = self.ema_cluster_size.clamp(min=1)
                self.embedding.weight.data.copy_(self.ema_sum / n.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), flat_x)

        # Straight-through
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


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ) for multi-level quantization."""

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        num_quantizers: int = 4,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers

        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(codebook_size, code_dim, commitment_cost, ema_decay)
            for _ in range(num_quantizers)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Residual quantization.

        Args:
            x: [batch, seq, code_dim]

        Returns:
            Dict with z_q, all_indices, total_commitment_loss, perplexities
        """
        residual = x
        z_q = torch.zeros_like(x)
        all_indices = []
        total_commitment = 0.0
        perplexities = []

        for i, quantizer in enumerate(self.quantizers):
            output = quantizer(residual)
            z_q = z_q + output['z_q']
            residual = residual - output['z_q']
            all_indices.append(output['indices'])
            total_commitment += output['commitment_loss']
            perplexities.append(output['perplexity'])

        return {
            'z_q': z_q,
            'indices': torch.stack(all_indices, dim=0),  # [num_q, batch, seq]
            'commitment_loss': total_commitment / self.num_quantizers,
            'perplexities': torch.stack(perplexities),
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode indices to vectors.

        Args:
            indices: [num_q, batch, seq]

        Returns:
            [batch, seq, code_dim]
        """
        z_q = None
        for i, quantizer in enumerate(self.quantizers):
            q_out = quantizer.decode_indices(indices[i])
            z_q = q_out if z_q is None else z_q + q_out
        return z_q


# =============================================================================
# GLOBAL ENCODER (Speaker/Style)
# =============================================================================

class GlobalEncoder(nn.Module):
    """
    Global Encoder: Extracts speaker/style as a single global token.

    Key Design:
    - Attentive statistics pooling for robust global representation
    - Captures speaker identity and speaking style
    - Single token representation (no temporal codes)
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.mel_dim, config.global_hidden_dim)

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            ConvBlock(config.global_hidden_dim, config.global_hidden_dim, kernel_size=5)
            for _ in range(config.global_num_layers)
        ])

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(config.global_hidden_dim, config.global_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.global_hidden_dim // 2, 1),
        )

        # Output projection (mean + std statistics)
        self.output_proj = nn.Sequential(
            nn.Linear(config.global_hidden_dim * 2, config.global_hidden_dim),
            nn.GELU(),
            nn.Linear(config.global_hidden_dim, config.global_dim),
        )

        self.norm = nn.LayerNorm(config.global_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract global speaker/style token.

        Returns:
            Dict with global_token: [batch, global_dim]
        """
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
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted mean and std
        mean = (x * attn_weights).sum(dim=1)  # [B, H]
        var = ((x - mean.unsqueeze(1)).pow(2) * attn_weights).sum(dim=1)
        std = var.clamp(min=1e-8).sqrt()

        # Combine statistics
        stats = torch.cat([mean, std], dim=-1)  # [B, H*2]

        # Project to global token
        global_token = self.output_proj(stats)
        global_token = self.norm(global_token)

        return {
            'global_token': global_token,
            'features': x,
        }


# =============================================================================
# LOCAL-SEMANTIC ENCODER (Content + Prosody)
# =============================================================================

class LocalSemanticEncoder(nn.Module):
    """
    Local-Semantic Encoder: Extracts content + prosody tokens.

    Key Innovation from UniCodec:
    - Uses SSL distillation to preserve paralinguistic information
    - Mid-layer SSL features contain rich prosody information
    - VQ codebook captures both content AND prosody patterns

    This is the core component that unifies semantic and prosodic information.
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.mel_dim, config.semantic_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.semantic_hidden_dim, dropout=config.dropout)

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

        # Pre-quantization projection
        self.pre_quant_proj = nn.Linear(config.semantic_hidden_dim, config.semantic_dim)

        # RVQ for semantic tokens
        self.quantizer = ResidualVectorQuantizer(
            codebook_size=config.semantic_codebook_size,
            code_dim=config.semantic_dim,
            num_quantizers=config.semantic_num_quantizers,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
        )

        # SSL distillation head
        self.ssl_head = nn.Linear(config.semantic_hidden_dim, config.ssl_dim)

        self.norm = nn.LayerNorm(config.semantic_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        ssl_target: Optional[torch.Tensor] = None,  # [batch, seq, ssl_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract semantic tokens with optional SSL distillation.

        Returns:
            Dict with:
                - semantic_z: Pre-quantization features
                - semantic_z_q: Quantized features
                - semantic_indices: Codebook indices [num_q, batch, seq]
                - ssl_pred: SSL feature prediction
                - distillation_loss: SSL distillation loss
                - commitment_loss, perplexities
        """
        # Project input
        x = self.input_proj(mel)
        x = self.pos_enc(x)

        # Transformer encoding
        for transformer in self.transformer:
            x = transformer(x, mask)

        # SSL prediction (before quantization for richer signal)
        ssl_pred = self.ssl_head(x)

        # Distillation loss
        if ssl_target is not None:
            # Handle length mismatch
            min_len = min(ssl_pred.shape[1], ssl_target.shape[1])
            distillation_loss = F.mse_loss(
                ssl_pred[:, :min_len],
                ssl_target[:, :min_len]
            )
        else:
            distillation_loss = torch.tensor(0.0, device=mel.device)

        # Pre-quantization projection
        z = self.pre_quant_proj(x)

        # Quantize
        vq_output = self.quantizer(z)
        z_q = self.norm(vq_output['z_q'])

        return {
            'semantic_z': z,
            'semantic_z_q': z_q,
            'semantic_indices': vq_output['indices'],
            'ssl_pred': ssl_pred,
            'distillation_loss': distillation_loss,
            'commitment_loss': vq_output['commitment_loss'],
            'perplexities': vq_output['perplexities'],
            'features': x,  # Pre-quantization features for analysis
        }


# =============================================================================
# LOCAL-RESIDUAL ENCODER (Acoustic Details)
# =============================================================================

class LocalResidualEncoder(nn.Module):
    """
    Local-Residual Encoder: Captures acoustic details not in semantic tokens.

    Encodes the residual information after semantic encoding:
    - Fine-grained spectral details
    - Speaker-specific acoustic variations
    - High-frequency information

    Uses multi-level RVQ for progressive refinement.
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Input projection (mel + semantic residual)
        self.input_proj = nn.Linear(config.mel_dim + config.semantic_dim, config.residual_hidden_dim)

        # Convolutional layers for local patterns
        self.conv_layers = nn.ModuleList([
            ConvBlock(config.residual_hidden_dim, config.residual_hidden_dim)
            for _ in range(config.residual_num_layers)
        ])

        # Transformer for context
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.residual_hidden_dim,
                num_heads=config.residual_num_heads,
                ffn_dim=config.residual_hidden_dim * 4,
                dropout=config.dropout,
            )
            for _ in range(2)  # Lightweight transformer
        ])

        # Pre-quantization projection
        self.pre_quant_proj = nn.Linear(config.residual_hidden_dim, config.residual_dim)

        # RVQ for residual tokens
        self.quantizer = ResidualVectorQuantizer(
            codebook_size=config.residual_codebook_size,
            code_dim=config.residual_dim,
            num_quantizers=config.residual_num_quantizers,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
        )

        self.norm = nn.LayerNorm(config.residual_dim)

    def forward(
        self,
        mel: torch.Tensor,        # [batch, seq, mel_dim]
        semantic_z_q: torch.Tensor,  # [batch, seq, semantic_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode residual acoustic information.

        Returns:
            Dict with:
                - residual_z: Pre-quantization features
                - residual_z_q: Quantized features
                - residual_indices: Codebook indices [num_q, batch, seq]
                - commitment_loss, perplexities
        """
        # Handle length mismatch
        min_len = min(mel.shape[1], semantic_z_q.shape[1])
        mel = mel[:, :min_len]
        semantic_z_q = semantic_z_q[:, :min_len]

        # Concatenate mel with semantic (to learn residual)
        x = torch.cat([mel, semantic_z_q], dim=-1)
        x = self.input_proj(x)

        # Convolutional processing
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(1, 2)

        # Transformer for context
        for transformer in self.transformer:
            x = transformer(x, mask[:, :min_len] if mask is not None else None)

        # Pre-quantization
        z = self.pre_quant_proj(x)

        # Quantize
        vq_output = self.quantizer(z)
        z_q = self.norm(vq_output['z_q'])

        return {
            'residual_z': z,
            'residual_z_q': z_q,
            'residual_indices': vq_output['indices'],
            'commitment_loss': vq_output['commitment_loss'],
            'perplexities': vq_output['perplexities'],
        }


# =============================================================================
# DECODER
# =============================================================================

class UniCodecDecoder(nn.Module):
    """
    Decoder: Reconstructs mel spectrogram from all tokens.

    Combines:
    - Global token (speaker/style conditioning)
    - Semantic tokens (content + prosody)
    - Residual tokens (acoustic details)
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.global_proj = nn.Linear(config.global_dim, config.decoder_hidden_dim)
        self.semantic_proj = nn.Linear(config.semantic_dim, config.decoder_hidden_dim)
        self.residual_proj = nn.Linear(config.residual_dim, config.decoder_hidden_dim)

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
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.mel_dim),
        )

        self.norm = nn.LayerNorm(config.decoder_hidden_dim)

    def forward(
        self,
        global_token: torch.Tensor,   # [batch, global_dim]
        semantic_z_q: torch.Tensor,   # [batch, seq, semantic_dim]
        residual_z_q: torch.Tensor,   # [batch, seq, residual_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode tokens to mel spectrogram.

        Returns:
            Dict with mel_reconstructed: [batch, seq, mel_dim]
        """
        batch_size, seq_len, _ = semantic_z_q.shape

        # Handle residual length mismatch
        if residual_z_q.shape[1] != seq_len:
            residual_z_q = F.interpolate(
                residual_z_q.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Project all components
        global_proj = self.global_proj(global_token).unsqueeze(1)  # [B, 1, H]
        semantic_proj = self.semantic_proj(semantic_z_q)  # [B, T, H]
        residual_proj = self.residual_proj(residual_z_q)  # [B, T, H]

        # Combine: semantic + residual + global (broadcast)
        x = semantic_proj + residual_proj + global_proj

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer decoding
        for transformer in self.transformer:
            x = transformer(x, mask)
        x = self.norm(x)

        # Output projection
        mel_reconstructed = self.output_proj(x)

        return {
            'mel_reconstructed': mel_reconstructed,
            'decoder_features': x,
        }


# =============================================================================
# PROSODY PREDICTOR (for LM generation)
# =============================================================================

class ProsodyPredictor(nn.Module):
    """
    Prosody Predictor: Predicts prosody tokens from text.

    Used in the generation stage to predict semantic tokens
    (which contain prosody) from text embeddings.
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Input projection (text embeddings)
        self.input_proj = nn.Linear(config.semantic_hidden_dim, config.prosody_hidden_dim)

        # Transformer for prosody prediction
        self.transformer = nn.ModuleList([
            TransformerBlock(
                config.prosody_hidden_dim,
                num_heads=4,
                ffn_dim=config.prosody_hidden_dim * 4,
                dropout=config.dropout,
            )
            for _ in range(3)
        ])

        # Output: predict prosody token logits
        self.output_proj = nn.Linear(
            config.prosody_hidden_dim,
            config.semantic_codebook_size * config.prosody_num_tokens
        )

        self.norm = nn.LayerNorm(config.prosody_hidden_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, seq, dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict prosody tokens from text.

        Returns:
            Dict with:
                - prosody_logits: [batch, num_tokens, codebook_size]
                - prosody_hidden: [batch, prosody_hidden_dim]
        """
        # Project input
        x = self.input_proj(text_embeddings)

        # Transformer
        for transformer in self.transformer:
            x = transformer(x, mask)
        x = self.norm(x)

        # Pool to fixed-size prosody representation
        prosody_hidden = x.mean(dim=1)  # [B, H]

        # Predict token logits
        logits = self.output_proj(prosody_hidden)  # [B, num_tokens * codebook_size]
        logits = logits.view(
            -1, self.config.prosody_num_tokens, self.config.semantic_codebook_size
        )

        return {
            'prosody_logits': logits,
            'prosody_hidden': prosody_hidden,
        }


# =============================================================================
# FULL UNICODEC MODEL
# =============================================================================

class UniCodec(nn.Module):
    """
    UniCodec: Universal Speech Token Learning.

    Unifies semantic and acoustic tokens into compact universal tokens:
    1. Global Token: Speaker/style (single vector)
    2. Semantic Tokens: Content + prosody (with SSL distillation)
    3. Residual Tokens: Acoustic details

    Key Innovation: SSL distillation preserves paralinguistic information
    in the semantic tokens, enabling prosody generation in the first stage.
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.global_encoder = GlobalEncoder(config)
        self.semantic_encoder = LocalSemanticEncoder(config)
        self.residual_encoder = LocalResidualEncoder(config)

        # Decoder
        self.decoder = UniCodecDecoder(config)

        # Prosody predictor (optional, for LM generation)
        if config.use_prosody_predictor:
            self.prosody_predictor = ProsodyPredictor(config)
        else:
            self.prosody_predictor = None

        # Output projection for CSM integration
        combined_dim = config.global_dim + config.semantic_dim + config.residual_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        ssl_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel to universal tokens.

        Returns:
            Dict with all encoder outputs
        """
        # Global encoding
        global_output = self.global_encoder(mel, mask)

        # Semantic encoding (with SSL distillation)
        semantic_output = self.semantic_encoder(mel, ssl_target, mask)

        # Residual encoding
        residual_output = self.residual_encoder(mel, semantic_output['semantic_z_q'], mask)

        return {
            # Global
            'global_token': global_output['global_token'],
            # Semantic
            'semantic_z': semantic_output['semantic_z'],
            'semantic_z_q': semantic_output['semantic_z_q'],
            'semantic_indices': semantic_output['semantic_indices'],
            'ssl_pred': semantic_output['ssl_pred'],
            'distillation_loss': semantic_output['distillation_loss'],
            'semantic_commitment_loss': semantic_output['commitment_loss'],
            'semantic_perplexities': semantic_output['perplexities'],
            # Residual
            'residual_z': residual_output['residual_z'],
            'residual_z_q': residual_output['residual_z_q'],
            'residual_indices': residual_output['residual_indices'],
            'residual_commitment_loss': residual_output['commitment_loss'],
            'residual_perplexities': residual_output['perplexities'],
        }

    def decode(
        self,
        global_token: torch.Tensor,   # [batch, global_dim]
        semantic_z_q: torch.Tensor,   # [batch, seq, semantic_dim]
        residual_z_q: torch.Tensor,   # [batch, seq, residual_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode tokens to mel spectrogram."""
        return self.decoder(global_token, semantic_z_q, residual_z_q, mask)

    def decode_from_indices(
        self,
        global_token: torch.Tensor,    # [batch, global_dim]
        semantic_indices: torch.Tensor,  # [num_q, batch, seq]
        residual_indices: torch.Tensor,  # [num_q, batch, seq]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode from codebook indices."""
        semantic_z_q = self.semantic_encoder.quantizer.decode_indices(semantic_indices)
        residual_z_q = self.residual_encoder.quantizer.decode_indices(residual_indices)
        return self.decode(global_token, semantic_z_q, residual_z_q, mask)

    def forward(
        self,
        mel: torch.Tensor,
        ssl_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Returns:
            Dict with all tokens, reconstruction, and losses
        """
        # Encode
        encoded = self.encode(mel, ssl_target, mask)

        # Decode
        decoded = self.decode(
            encoded['global_token'],
            encoded['semantic_z_q'],
            encoded['residual_z_q'],
            mask,
        )

        # Combined prosody embedding for CSM
        global_expanded = encoded['global_token']
        semantic_pooled = encoded['semantic_z_q'].mean(dim=1)
        residual_pooled = encoded['residual_z_q'].mean(dim=1)

        combined = torch.cat([global_expanded, semantic_pooled, residual_pooled], dim=-1)
        prosody_embedding = self.output_proj(combined)

        return {
            **encoded,
            'mel_reconstructed': decoded['mel_reconstructed'],
            'prosody_embedding': prosody_embedding,
        }

    def get_prosody_tokens(
        self,
        text_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody token predictions from text.

        Used in LM generation stage.
        """
        if self.prosody_predictor is None:
            raise ValueError("Prosody predictor not enabled in config")
        return self.prosody_predictor(text_embeddings, mask)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class UniCodecLoss(nn.Module):
    """
    Combined loss function for UniCodec training.

    Components:
    1. Reconstruction loss (L1 + L2 mel)
    2. SSL distillation loss (preserve paralinguistic info)
    3. Semantic commitment loss
    4. Residual commitment loss
    5. Prosody prediction loss (optional)
    """

    def __init__(self, config: UniCodecConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.reconstruction_weight = 1.0
        self.distillation_weight = config.distillation_weight
        self.commitment_weight = 0.25

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        ssl_target: Optional[torch.Tensor] = None,
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

        # Distillation loss
        losses['distillation'] = output['distillation_loss']

        # Commitment losses
        losses['semantic_commitment'] = output['semantic_commitment_loss']
        losses['residual_commitment'] = output['residual_commitment_loss']

        # Perplexities (for logging)
        losses['semantic_perplexity'] = output['semantic_perplexities'].mean()
        losses['residual_perplexity'] = output['residual_perplexities'].mean()

        # Total loss
        total = (
            self.reconstruction_weight * losses['reconstruction']
            + self.distillation_weight * losses['distillation']
            + self.commitment_weight * (losses['semantic_commitment'] + losses['residual_commitment'])
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class UniCodecAdapter(nn.Module):
    """
    Adapter for integrating UniCodec with CSM prosody conditioning.

    Converts UniCodec's universal tokens to prefix tokens for CSM.
    """

    def __init__(
        self,
        config: UniCodecConfig,
        model: Optional[UniCodec] = None,
    ):
        super().__init__()
        self.config = config
        self.model = model if model is not None else UniCodec(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        mel: torch.Tensor,
        ssl_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix tokens for CSM conditioning.

        Returns:
            Dict with:
                - prosody_tokens: [batch, num_prefix_tokens, output_dim]
                - encoded: Full encoding results
        """
        batch_size = mel.shape[0]

        # Get universal tokens
        output = self.model(mel, ssl_target)
        prosody_emb = output['prosody_embedding']

        # Project to tokens
        tokens = self.token_proj(prosody_emb)
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
            'global_token': output['global_token'],
            'semantic_indices': output['semantic_indices'],
            'residual_indices': output['residual_indices'],
            'distillation_loss': output['distillation_loss'],
            'commitment_loss': (
                output['semantic_commitment_loss'] + output['residual_commitment_loss']
            ) / 2,
        }

    def from_indices(
        self,
        global_token: torch.Tensor,
        semantic_indices: torch.Tensor,
        residual_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prosody tokens directly from codebook indices.

        Enables discrete prosody control!
        """
        batch_size = global_token.shape[0]

        # Decode indices
        semantic_z_q = self.model.semantic_encoder.quantizer.decode_indices(semantic_indices)
        residual_z_q = self.model.residual_encoder.quantizer.decode_indices(residual_indices)

        # Pool and combine
        semantic_pooled = semantic_z_q.mean(dim=1)
        residual_pooled = residual_z_q.mean(dim=1)

        combined = torch.cat([global_token, semantic_pooled, residual_pooled], dim=-1)
        prosody_emb = self.model.output_proj(combined)

        # Project to tokens
        tokens = self.token_proj(prosody_emb)
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)
        tokens = self.norm(tokens)

        return tokens

    def get_universal_tokens(
        self,
        mel: torch.Tensor,
        ssl_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get all universal token representations."""
        encoded = self.model.encode(mel, ssl_target)
        return {
            'global': encoded['global_token'],
            'semantic': encoded['semantic_indices'],
            'residual': encoded['residual_indices'],
        }


# =============================================================================
# SSL FEATURE EXTRACTOR WRAPPER
# =============================================================================

class SSLFeatureExtractor(nn.Module):
    """
    Wrapper for extracting SSL features (wav2vec2/HuBERT).

    Used to get distillation targets for training.
    """

    def __init__(
        self,
        model_name: str = "wav2vec2",
        target_layer: int = 7,
        freeze: bool = True,
    ):
        super().__init__()
        self.target_layer = target_layer
        self.freeze = freeze
        self._model = None
        self._model_name = model_name

    def _load_model(self, device):
        """Lazy-load SSL model."""
        if self._model is not None:
            return

        try:
            from transformers import Wav2Vec2Model, HubertModel

            if self._model_name == "wav2vec2":
                self._model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                ).to(device)
            else:
                self._model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960"
                ).to(device)

            if self.freeze:
                for param in self._model.parameters():
                    param.requires_grad = False

            self._model.eval()

        except ImportError:
            print("Warning: transformers not installed. SSL extraction disabled.")
            self._model = None

    def forward(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract SSL features from audio.

        Args:
            audio: [batch, samples] at 16kHz

        Returns:
            [batch, seq, 768] SSL features from target layer
        """
        self._load_model(audio.device)

        if self._model is None:
            return None

        with torch.no_grad():
            outputs = self._model(audio, output_hidden_states=True)
            # Get features from target layer
            hidden_states = outputs.hidden_states[self.target_layer]

        return hidden_states


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UniCodec: Universal Speech Token Learning - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 100
    mel_dim = 80
    ssl_dim = 768

    config = UniCodecConfig()

    # Create dummy inputs
    mel = torch.randn(batch_size, seq_len, mel_dim).to(device)
    ssl_target = torch.randn(batch_size, seq_len, ssl_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Global dim: {config.global_dim}")
    print(f"  Semantic codebook: {config.semantic_codebook_size} x {config.semantic_num_quantizers}")
    print(f"  Residual codebook: {config.residual_codebook_size} x {config.residual_num_quantizers}")
    print(f"  SSL model: {config.ssl_model}, layer {config.ssl_layer}")
    print("  [PASS]")

    # Test 2: Global Encoder
    print("\n[Test 2] Global Encoder...")
    global_enc = GlobalEncoder(config).to(device)
    global_out = global_enc(mel)
    print(f"  Global token shape: {global_out['global_token'].shape}")
    assert global_out['global_token'].shape == (batch_size, config.global_dim)
    print("  [PASS]")

    # Test 3: Semantic Encoder
    print("\n[Test 3] Semantic Encoder...")
    semantic_enc = LocalSemanticEncoder(config).to(device)
    semantic_out = semantic_enc(mel, ssl_target)
    print(f"  Semantic z shape: {semantic_out['semantic_z'].shape}")
    print(f"  Semantic z_q shape: {semantic_out['semantic_z_q'].shape}")
    print(f"  Semantic indices shape: {semantic_out['semantic_indices'].shape}")
    print(f"  Distillation loss: {semantic_out['distillation_loss'].item():.4f}")
    print(f"  Commitment loss: {semantic_out['commitment_loss'].item():.4f}")
    print("  [PASS]")

    # Test 4: Residual Encoder
    print("\n[Test 4] Residual Encoder...")
    residual_enc = LocalResidualEncoder(config).to(device)
    residual_out = residual_enc(mel, semantic_out['semantic_z_q'])
    print(f"  Residual z shape: {residual_out['residual_z'].shape}")
    print(f"  Residual z_q shape: {residual_out['residual_z_q'].shape}")
    print(f"  Residual indices shape: {residual_out['residual_indices'].shape}")
    print("  [PASS]")

    # Test 5: Decoder
    print("\n[Test 5] Decoder...")
    decoder = UniCodecDecoder(config).to(device)
    decoded = decoder(
        global_out['global_token'],
        semantic_out['semantic_z_q'],
        residual_out['residual_z_q'],
    )
    print(f"  Mel reconstructed shape: {decoded['mel_reconstructed'].shape}")
    print("  [PASS]")

    # Test 6: Full Model
    print("\n[Test 6] Full UniCodec Model...")
    model = UniCodec(config).to(device)
    output = model(mel, ssl_target)
    print(f"  Global token: {output['global_token'].shape}")
    print(f"  Semantic indices: {output['semantic_indices'].shape}")
    print(f"  Residual indices: {output['residual_indices'].shape}")
    print(f"  Mel reconstructed: {output['mel_reconstructed'].shape}")
    print(f"  Prosody embedding: {output['prosody_embedding'].shape}")
    print(f"  Distillation loss: {output['distillation_loss'].item():.4f}")
    print("  [PASS]")

    # Test 7: Loss Function
    print("\n[Test 7] Loss Function...")
    loss_fn = UniCodecLoss(config)
    losses = loss_fn(output, mel, ssl_target)
    print(f"  Reconstruction loss: {losses['reconstruction'].item():.4f}")
    print(f"  Distillation loss: {losses['distillation'].item():.4f}")
    print(f"  Semantic commitment: {losses['semantic_commitment'].item():.4f}")
    print(f"  Semantic perplexity: {losses['semantic_perplexity'].item():.2f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 8: CSM Adapter
    print("\n[Test 8] CSM Adapter...")
    adapter = UniCodecAdapter(config, model).to(device)
    adapter_out = adapter(mel, ssl_target)
    print(f"  Prefix tokens: {adapter_out['prosody_tokens'].shape}")
    assert adapter_out['prosody_tokens'].shape == (
        batch_size, config.num_prefix_tokens, config.output_dim
    )
    print("  [PASS]")

    # Test 9: Decode from Indices
    print("\n[Test 9] Decode from Indices...")
    with torch.no_grad():
        decoded_from_idx = model.decode_from_indices(
            output['global_token'],
            output['semantic_indices'],
            output['residual_indices'],
        )
    print(f"  Decoded mel shape: {decoded_from_idx['mel_reconstructed'].shape}")
    print("  [PASS]")

    # Test 10: Prefix from Indices
    print("\n[Test 10] Prefix from Indices...")
    with torch.no_grad():
        prefix_from_idx = adapter.from_indices(
            output['global_token'],
            output['semantic_indices'],
            output['residual_indices'],
        )
    print(f"  Prefix from indices: {prefix_from_idx.shape}")
    print("  [PASS]")

    # Test 11: Prosody Predictor
    print("\n[Test 11] Prosody Predictor...")
    if model.prosody_predictor is not None:
        text_emb = torch.randn(batch_size, seq_len, config.semantic_hidden_dim).to(device)
        prosody_pred = model.get_prosody_tokens(text_emb)
        print(f"  Prosody logits shape: {prosody_pred['prosody_logits'].shape}")
    else:
        print("  Prosody predictor not enabled")
    print("  [PASS]")

    # Test 12: Backward Pass
    print("\n[Test 12] Backward Pass...")
    model.zero_grad()
    output = model(mel, ssl_target)
    losses = loss_fn(output, mel, ssl_target)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test 13: Token Efficiency
    print("\n[Test 13] Token Efficiency...")
    total_semantic_tokens = config.semantic_num_quantizers * seq_len
    total_residual_tokens = config.residual_num_quantizers * seq_len
    total_tokens = 1 + total_semantic_tokens + total_residual_tokens  # 1 global
    print(f"  Global tokens: 1")
    print(f"  Semantic tokens: {total_semantic_tokens} ({config.semantic_num_quantizers} levels)")
    print(f"  Residual tokens: {total_residual_tokens} ({config.residual_num_quantizers} levels)")
    print(f"  Total tokens: {total_tokens}")

    # Compare to naive approach (6 codebooks all temporal)
    naive_tokens = 6 * seq_len
    efficiency = naive_tokens / total_tokens
    print(f"  Naive approach: {naive_tokens} tokens")
    print(f"  Efficiency ratio: {efficiency:.2f}x")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All UniCodec tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. UNIVERSAL TOKEN LEARNING:
       - Global Token: Single vector for speaker/style
       - Semantic Tokens: Content + prosody (SSL-distilled)
       - Residual Tokens: Acoustic details

    2. SSL DISTILLATION:
       - wav2vec2/HuBERT features preserve paralinguistic info
       - Mid-layer features rich in prosody/emotion
       - Enables prosody generation in first stage

    3. COMPRESSION-GENERATION FRAMEWORK:
       - Low-bitrate neural codec
       - Compact tokens easy to predict in LMs
       - Long-term consistency in output quality

    4. PROSODY PRESERVATION:
       - Better than SpeechTokenizer for paralinguistic info
       - Prosody embedded in semantic tokens
       - Enables emotion/style transfer

    5. CSM INTEGRATION:
       adapter = UniCodecAdapter(config, model)
       prefix_tokens = adapter(mel)  # [batch, 4, 2048]
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from unicodec import (
    UniCodecConfig,
    UniCodec,
    UniCodecLoss,
    UniCodecAdapter,
    SSLFeatureExtractor,
)

# Initialize
config = UniCodecConfig()
model = UniCodec(config).cuda()
loss_fn = UniCodecLoss(config)

# SSL feature extractor for distillation targets
ssl_extractor = SSLFeatureExtractor(model_name="wav2vec2", target_layer=7)

# Training loop
for batch in dataloader:
    mel = batch['mel'].cuda()
    audio = batch['audio'].cuda()

    # Extract SSL features as distillation targets
    ssl_target = ssl_extractor(audio)

    # Forward pass
    output = model(mel, ssl_target)
    losses = loss_fn(output, mel, ssl_target)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor
    print(f"Recon: {losses['reconstruction']:.4f}")
    print(f"Distill: {losses['distillation']:.4f}")
    print(f"Semantic perplexity: {losses['semantic_perplexity']:.2f}")

# Extract universal tokens
with torch.no_grad():
    encoded = model.encode(mel, ssl_target)
    global_token = encoded['global_token']       # Speaker/style
    semantic_idx = encoded['semantic_indices']   # Content + prosody
    residual_idx = encoded['residual_indices']   # Acoustic details

# CSM integration
adapter = UniCodecAdapter(config, model)
prefix_tokens = adapter(mel)['prosody_tokens']  # [batch, 4, 2048]

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)

# Direct control via indices
prefix_from_idx = adapter.from_indices(global_token, semantic_idx, residual_idx)
""")
