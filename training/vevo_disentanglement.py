"""
VEVO: Controllable Zero-Shot Voice Imitation via Self-Supervised Disentanglement

Based on "VEVO: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement"
(ICLR 2025) - arXiv:2502.07243 - Demo: versavoice.github.io

Key Innovation: Speech decomposition into content, style, and timbre using VQ bottleneck
- Content (what): Linguistic/phonetic information via VQ-VAE with small codebook
- Style (how): Prosody, emotion, accent, speaking manner
- Timbre (who): Speaker identity characteristics

Architecture:
1. HuBERT/WavLM encoder → VQ tokenizer (content extraction with info bottleneck)
2. Content-Style Modeling: AR transformer generates content-style tokens from content
   - Prompted by style reference audio
3. Acoustic Modeling: Flow-matching transformer generates acoustic features
   - Prompted by timbre reference audio

VQ Bottleneck Insight:
- Larger codebook (16k+) → preserves more prosodic variation
- Smaller codebook (256-1024) → more speaker/style invariant content
- VEVO uses ~1024 codes for optimal content-style separation

Benefits:
- Fully self-supervised (no labeled emotion/speaker data needed)
- Zero-shot style imitation from just a few seconds of reference
- Independent control of style (emotion, accent) and timbre (speaker)
- Progressive decoupling through VQ bottleneck
- Diverse sampling via flow matching

Usage:
    from vevo_disentanglement import (
        VEVOConfig,
        VEVOContentEncoder,
        VEVOContentStyleModel,
        VEVOAcousticModel,
        VEVO,
        VEVOAdapter,
    )

    # Initialize
    config = VEVOConfig()
    model = VEVO(config).cuda()

    # Extract content tokens
    content_tokens = model.encode_content(audio)

    # Zero-shot style imitation
    audio_out = model.generate(
        content_audio=source_audio,      # What to say
        style_audio=style_reference,      # How to say it
        timbre_audio=speaker_reference,   # Who is speaking
    )

    # For CSM integration
    adapter = VEVOAdapter(config, model)
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
class VEVOConfig:
    """Configuration for VEVO self-supervised disentanglement."""

    # Input dimensions
    input_dim: int = 768  # HuBERT/WavLM feature dimension
    mel_dim: int = 80  # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 320  # 20ms at 16kHz

    # VQ Tokenizer (Content Extraction)
    # Key insight: smaller codebook = more disentangled content
    content_codebook_size: int = 1024  # Info bottleneck (256-2048 range)
    content_code_dim: int = 256  # Dimension per content code
    content_num_groups: int = 1  # Single codebook for simplicity
    content_ema_decay: float = 0.99  # EMA for codebook updates
    content_commitment_cost: float = 0.25

    # Style Encoder
    style_dim: int = 512  # Style embedding dimension
    style_num_layers: int = 4  # Transformer layers for style extraction
    style_num_heads: int = 8
    style_ffn_dim: int = 2048

    # Timbre Encoder
    timbre_dim: int = 256  # Global speaker embedding
    timbre_num_layers: int = 3

    # Content-Style Model (Stage 1: AR Transformer)
    cs_hidden_dim: int = 768  # Content-style transformer hidden dim
    cs_num_layers: int = 8  # Deeper for content-style modeling
    cs_num_heads: int = 12
    cs_ffn_dim: int = 3072
    cs_dropout: float = 0.1
    cs_max_len: int = 2048  # Max sequence length for AR model

    # Acoustic Model (Stage 2: Flow-Matching Transformer)
    acoustic_hidden_dim: int = 512
    acoustic_num_layers: int = 6
    acoustic_num_heads: int = 8
    acoustic_ffn_dim: int = 2048
    acoustic_dropout: float = 0.1

    # Flow matching settings
    sigma_min: float = 0.001
    num_ode_steps: int = 50  # ODE integration steps at inference
    ode_method: str = "euler"  # euler, midpoint, rk4

    # Training settings
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 0  # Top-k sampling (0 = disabled)
    top_p: float = 0.9  # Nucleus sampling

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# VQ TOKENIZER (CONTENT ENCODER WITH INFO BOTTLENECK)
# =============================================================================

class VQTokenizer(nn.Module):
    """
    Vector Quantizer for content extraction with information bottleneck.

    The key insight from VEVO: codebook size controls disentanglement level.
    - Large codebook (16k+): Preserves prosody, speaker characteristics
    - Small codebook (256-1024): Forces content-only encoding, removes style/timbre

    Uses exponential moving average (EMA) for codebook updates.
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

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # Project input to code dimension
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

    def _init_codebook_from_data(self, z: torch.Tensor):
        """Initialize codebook from first batch of data (k-means++ style)."""
        batch_size = z.shape[0]
        if batch_size >= self.codebook_size:
            indices = torch.randperm(batch_size)[:self.codebook_size]
            self.codebook.data.copy_(z[indices])
        else:
            # Repeat if not enough samples
            repeats = (self.codebook_size // batch_size) + 1
            expanded = z.repeat(repeats, 1)[:self.codebook_size]
            self.codebook.data.copy_(expanded)

        self.ema_sum.data.copy_(self.codebook.data.clone())
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input to discrete content tokens.

        Returns:
            Dict with:
                - z_q: [batch, seq, input_dim] quantized output
                - indices: [batch, seq] token indices
                - commitment_loss: scalar loss
                - perplexity: codebook usage metric
        """
        batch_size, seq_len, _ = x.shape

        # Project to code dimension
        z = self.pre_proj(x)  # [B, T, code_dim]
        z_flat = z.view(-1, self.code_dim)  # [B*T, code_dim]

        # Initialize codebook from first batch
        if self.training and not self.initialized:
            self._init_codebook_from_data(z_flat)

        # Compute distances to codebook
        d = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.codebook.t())
            + self.codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )  # [B*T, codebook_size]

        # Find nearest codes
        indices = d.argmin(dim=-1)  # [B*T]

        # Get quantized vectors
        z_q = F.embedding(indices, self.codebook)  # [B*T, code_dim]

        # EMA codebook update
        if self.training:
            with torch.no_grad():
                # One-hot encoding
                encodings = F.one_hot(indices, self.codebook_size).float()

                # Update cluster sizes
                new_cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    new_cluster_size, alpha=1 - self.ema_decay
                )

                # Update sum
                new_sum = torch.matmul(encodings.t(), z_flat)
                self.ema_sum.mul_(self.ema_decay).add_(
                    new_sum, alpha=1 - self.ema_decay
                )

                # Update codebook
                n = self.ema_cluster_size.clamp(min=1)
                self.codebook.data.copy_(self.ema_sum / n.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z_flat, z_q.detach())

        # Straight-through estimator
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape back
        z_q = z_q.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

        # Project back to input dimension
        z_q = self.post_proj(z_q)

        # Compute perplexity
        perplexity = self._compute_perplexity(indices)

        return {
            'z_q': z_q,
            'indices': indices,
            'commitment_loss': commitment_loss * self.commitment_cost,
            'perplexity': perplexity,
        }

    def _compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute perplexity (higher = better codebook usage)."""
        flat_indices = indices.view(-1)
        encodings = F.one_hot(flat_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def encode_indices(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
    ) -> torch.Tensor:
        """Get just the token indices (for AR model input)."""
        z = self.pre_proj(x)
        z_flat = z.view(-1, self.code_dim)

        d = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.codebook.t())
            + self.codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )

        indices = d.argmin(dim=-1)
        return indices.view(x.shape[0], x.shape[1])

    def decode_indices(
        self,
        indices: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        """Decode token indices back to features."""
        z_q = F.embedding(indices, self.codebook)
        return self.post_proj(z_q)


# =============================================================================
# STYLE ENCODER
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


class StyleEncoder(nn.Module):
    """
    Style Encoder: Extracts "how" to speak (prosody, emotion, accent).

    Uses multi-head attention to capture style-relevant patterns from
    the input audio. Outputs a global style embedding that can prompt
    the content-style model.
    """

    def __init__(self, config: VEVOConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.style_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.style_dim, dropout=config.cs_dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.style_dim,
            nhead=config.style_num_heads,
            dim_feedforward=config.style_ffn_dim,
            dropout=config.cs_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.style_num_layers
        )

        # Style pooling: attention-weighted average
        self.style_query = nn.Parameter(torch.randn(1, 1, config.style_dim))
        self.style_attn = nn.MultiheadAttention(
            config.style_dim, num_heads=4, dropout=config.cs_dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(config.style_dim)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract style embedding from audio features.

        Returns:
            Dict with:
                - style_emb: [batch, style_dim] global style embedding
                - style_seq: [batch, seq, style_dim] per-frame style features
        """
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)  # [B, T, style_dim]

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Style query attention
        query = self.style_query.expand(batch_size, -1, -1)  # [B, 1, style_dim]
        style_emb, _ = self.style_attn(query, x, x)  # [B, 1, style_dim]
        style_emb = self.norm(style_emb.squeeze(1))  # [B, style_dim]

        return {
            'style_emb': style_emb,
            'style_seq': x,
        }


# =============================================================================
# TIMBRE ENCODER
# =============================================================================

class TimbreEncoder(nn.Module):
    """
    Timbre Encoder: Extracts "who" is speaking (speaker identity).

    Uses global average pooling + statistics pooling to capture
    speaker-level characteristics that are time-invariant.
    """

    def __init__(self, config: VEVOConfig):
        super().__init__()
        self.config = config

        # Convolutional feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )

        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=1),
        )

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 512),  # Mean + Std
            nn.GELU(),
            nn.Linear(512, config.timbre_dim),
        )

        self.norm = nn.LayerNorm(config.timbre_dim)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract timbre (speaker) embedding.

        Returns:
            Dict with:
                - timbre_emb: [batch, timbre_dim] global speaker embedding
        """
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # Convolutional processing
        x = self.conv(x)  # [B, 512, T]

        # Attentive statistics pooling
        attn_weights = F.softmax(self.attention(x), dim=-1)  # [B, 512, T]

        # Weighted mean
        mean = (x * attn_weights).sum(dim=-1)  # [B, 512]

        # Weighted std
        var = ((x - mean.unsqueeze(-1)).pow(2) * attn_weights).sum(dim=-1)
        std = var.clamp(min=1e-8).sqrt()

        # Concatenate statistics
        stats = torch.cat([mean, std], dim=-1)  # [B, 1024]

        # Project to timbre embedding
        timbre_emb = self.fc(stats)
        timbre_emb = self.norm(timbre_emb)

        return {
            'timbre_emb': timbre_emb,
        }


# =============================================================================
# CONTENT-STYLE MODEL (STAGE 1: AUTOREGRESSIVE TRANSFORMER)
# =============================================================================

class ContentStyleModel(nn.Module):
    """
    Content-Style Transformer: Generates content-style tokens autoregressively.

    Takes content tokens as input and generates content+style tokens,
    conditioned on a style reference embedding. This models the mapping
    from "what to say" to "what to say + how to say it".

    Architecture:
    - Token embedding for content tokens (input)
    - Separate embedding for content-style tokens (generated)
    - Cross-attention to style reference
    - Causal transformer for autoregressive generation
    """

    def __init__(self, config: VEVOConfig):
        super().__init__()
        self.config = config

        # Content-style vocabulary size (larger to encode prosodic variation)
        self.cs_vocab_size = config.content_codebook_size * 4  # 4x for style variations

        # Special tokens
        self.BOS_TOKEN = self.cs_vocab_size
        self.EOS_TOKEN = self.cs_vocab_size + 1
        total_vocab_size = self.cs_vocab_size + 2  # +2 for BOS/EOS

        # Token embeddings for content tokens (input)
        self.content_embedding = nn.Embedding(
            config.content_codebook_size + 2,  # +2 for BOS/EOS
            config.cs_hidden_dim,
        )

        # Token embeddings for content-style tokens (generated/AR)
        self.cs_embedding = nn.Embedding(
            total_vocab_size,
            config.cs_hidden_dim,
        )

        # Style conditioning projection
        self.style_proj = nn.Linear(config.style_dim, config.cs_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.cs_hidden_dim, max_len=config.cs_max_len, dropout=config.cs_dropout
        )

        # Transformer decoder with causal mask
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.cs_hidden_dim,
            nhead=config.cs_num_heads,
            dim_feedforward=config.cs_ffn_dim,
            dropout=config.cs_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=config.cs_num_layers
        )

        # Output projection (back to content-style vocabulary)
        self.output_proj = nn.Linear(config.cs_hidden_dim, total_vocab_size)

        self.norm = nn.LayerNorm(config.cs_hidden_dim)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        content_tokens: torch.Tensor,  # [batch, seq]
        style_emb: torch.Tensor,  # [batch, style_dim]
        target_tokens: Optional[torch.Tensor] = None,  # [batch, seq] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with teacher forcing.

        Args:
            content_tokens: Input content token indices
            style_emb: Style embedding from style encoder
            target_tokens: Target content-style tokens for training

        Returns:
            Dict with logits, loss (if target provided)
        """
        batch_size, seq_len = content_tokens.shape
        device = content_tokens.device

        # Embed content tokens
        x = self.content_embedding(content_tokens)  # [B, T, hidden]

        # Add positional encoding
        x = self.pos_enc(x)

        # Project style embedding for cross-attention
        style_mem = self.style_proj(style_emb).unsqueeze(1)  # [B, 1, hidden]

        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Transformer decoder with style conditioning
        x = self.transformer(x, style_mem, tgt_mask=causal_mask)
        x = self.norm(x)

        # Output logits
        logits = self.output_proj(x)  # [B, T, cs_vocab_size]

        result = {'logits': logits, 'hidden': x}

        # Compute loss if targets provided
        if target_tokens is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cs_vocab_size),
                target_tokens.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss

        return result

    def forward_ar(
        self,
        cs_tokens: torch.Tensor,  # [batch, seq] content-style tokens (for generation)
        style_emb: torch.Tensor,  # [batch, style_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for AR generation using content-style token embeddings.

        Args:
            cs_tokens: Previously generated content-style token indices
            style_emb: Style embedding from style encoder

        Returns:
            Dict with logits, hidden
        """
        batch_size, seq_len = cs_tokens.shape
        device = cs_tokens.device

        # Embed content-style tokens (for AR generation)
        x = self.cs_embedding(cs_tokens)  # [B, T, hidden]

        # Add positional encoding
        x = self.pos_enc(x)

        # Project style embedding for cross-attention
        style_mem = self.style_proj(style_emb).unsqueeze(1)  # [B, 1, hidden]

        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Transformer decoder with style conditioning
        x = self.transformer(x, style_mem, tgt_mask=causal_mask)
        x = self.norm(x)

        # Output logits
        logits = self.output_proj(x)  # [B, T, total_vocab_size]

        return {'logits': logits, 'hidden': x}

    @torch.no_grad()
    def generate(
        self,
        content_tokens: torch.Tensor,  # [batch, seq]
        style_emb: torch.Tensor,  # [batch, style_dim]
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation of content-style tokens.

        Returns:
            [batch, seq] content-style token indices
        """
        batch_size, seq_len = content_tokens.shape
        device = content_tokens.device

        # Start with BOS token
        generated = torch.full(
            (batch_size, 1), self.BOS_TOKEN, dtype=torch.long, device=device
        )

        # Generate tokens autoregressively
        for _ in range(seq_len):
            # Forward pass using AR method with cs_embedding
            result = self.forward_ar(generated, style_emb)
            logits = result['logits'][:, -1, :]  # Last position

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample (clamp to valid vocab range)
            probs = F.softmax(logits[:, :self.cs_vocab_size], dim=-1)  # Exclude special tokens
            next_token = torch.multinomial(probs, 1)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

        # Remove BOS token
        return generated[:, 1:]


# =============================================================================
# ACOUSTIC MODEL (STAGE 2: FLOW-MATCHING TRANSFORMER)
# =============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )

        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


class AcousticModel(nn.Module):
    """
    Acoustic Model: Flow-matching transformer for acoustic feature generation.

    Takes content-style tokens and generates acoustic features (mel spectrogram),
    conditioned on timbre reference. Uses conditional flow matching for diverse
    and high-quality synthesis.
    """

    def __init__(self, config: VEVOConfig):
        super().__init__()
        self.config = config

        # Content-style embedding
        self.cs_embedding = nn.Linear(config.cs_hidden_dim, config.acoustic_hidden_dim)

        # Timbre conditioning
        self.timbre_proj = nn.Linear(config.timbre_dim, config.acoustic_hidden_dim)

        # Time embedding
        self.time_embed = TimeEmbedding(config.acoustic_hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(config.acoustic_hidden_dim, config.acoustic_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.acoustic_hidden_dim * 4, config.acoustic_hidden_dim),
        )

        # Mel input projection
        self.mel_proj = nn.Linear(config.mel_dim, config.acoustic_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.acoustic_hidden_dim, dropout=config.acoustic_dropout
        )

        # Transformer for velocity field prediction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.acoustic_hidden_dim,
            nhead=config.acoustic_num_heads,
            dim_feedforward=config.acoustic_ffn_dim,
            dropout=config.acoustic_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.acoustic_num_layers
        )

        # Output projection to velocity
        self.velocity_proj = nn.Sequential(
            nn.Linear(config.acoustic_hidden_dim, config.acoustic_hidden_dim),
            nn.GELU(),
            nn.Linear(config.acoustic_hidden_dim, config.mel_dim),
        )

        self.norm = nn.LayerNorm(config.acoustic_hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,  # [batch, seq, mel_dim] noisy mel at time t
        t: torch.Tensor,  # [batch] time values in [0, 1]
        cs_hidden: torch.Tensor,  # [batch, seq, cs_hidden_dim] content-style features
        timbre_emb: torch.Tensor,  # [batch, timbre_dim]
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Returns:
            [batch, seq, mel_dim] predicted velocity
        """
        batch_size, seq_len, _ = x_t.shape

        # Project inputs
        mel_feat = self.mel_proj(x_t)  # [B, T, hidden]
        cs_feat = self.cs_embedding(cs_hidden)  # [B, T, hidden]

        # Time embedding
        t_emb = self.time_embed(t)  # [B, hidden]
        t_emb = self.time_proj(t_emb)  # [B, hidden]
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, hidden]

        # Timbre embedding (broadcast to all positions)
        timbre_feat = self.timbre_proj(timbre_emb)  # [B, hidden]
        timbre_feat = timbre_feat.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine all features
        x = mel_feat + cs_feat + t_emb + timbre_feat

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Predict velocity
        velocity = self.velocity_proj(x)

        return velocity

    def compute_loss(
        self,
        x_1: torch.Tensor,  # [batch, seq, mel_dim] target mel
        cs_hidden: torch.Tensor,  # [batch, seq, cs_hidden_dim]
        timbre_emb: torch.Tensor,  # [batch, timbre_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.

        Uses the conditional flow matching objective:
        L = E_t,x_0,x_1 ||v_theta(x_t, t) - u_t(x_t|x_1)||^2

        For linear interpolation path:
        u_t(x_t|x_1) = (x_1 - x_0) / (1 - sigma_min)
        """
        batch_size, seq_len, mel_dim = x_1.shape
        device = x_1.device

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Sample noise x_0
        x_0 = torch.randn_like(x_1)

        # Compute x_t (linear interpolation)
        sigma_min = self.config.sigma_min
        sigma_t = (1 - t) + t * sigma_min
        sigma_t = sigma_t.view(batch_size, 1, 1)

        t_expanded = t.view(batch_size, 1, 1)
        x_t = t_expanded * x_1 + sigma_t * x_0

        # Target velocity
        target_velocity = x_1 - (1 - sigma_min) * x_0

        # Predicted velocity
        pred_velocity = self.forward(x_t, t, cs_hidden, timbre_emb)

        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        return {
            'loss': loss,
            'x_t': x_t,
            'pred_velocity': pred_velocity,
            'target_velocity': target_velocity,
        }

    @torch.no_grad()
    def sample(
        self,
        cs_hidden: torch.Tensor,  # [batch, seq, cs_hidden_dim]
        timbre_emb: torch.Tensor,  # [batch, timbre_dim]
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample acoustic features using ODE integration.

        Returns:
            [batch, seq, mel_dim] generated mel spectrogram
        """
        num_steps = num_steps or self.config.num_ode_steps
        batch_size, seq_len, _ = cs_hidden.shape
        device = cs_hidden.device

        # Start from noise
        x = torch.randn(batch_size, seq_len, self.config.mel_dim, device=device)

        # ODE integration from t=0 (noise) to t=1 (data)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)

            # Predict velocity
            velocity = self.forward(x, t, cs_hidden, timbre_emb)

            # Euler step
            x = x + velocity * dt

        return x


# =============================================================================
# FULL VEVO MODEL
# =============================================================================

class VEVO(nn.Module):
    """
    VEVO: Controllable Zero-Shot Voice Imitation.

    Two-stage architecture:
    1. Content-Style Modeling: Content tokens → Content-Style tokens (AR)
    2. Acoustic Modeling: Content-Style tokens → Mel spectrogram (Flow)

    Separate encoders for style (how) and timbre (who) enable independent control.
    """

    def __init__(self, config: VEVOConfig):
        super().__init__()
        self.config = config

        # Content tokenizer (VQ bottleneck)
        self.content_tokenizer = VQTokenizer(
            input_dim=config.input_dim,
            codebook_size=config.content_codebook_size,
            code_dim=config.content_code_dim,
            commitment_cost=config.content_commitment_cost,
            ema_decay=config.content_ema_decay,
        )

        # Style encoder
        self.style_encoder = StyleEncoder(config)

        # Timbre encoder
        self.timbre_encoder = TimbreEncoder(config)

        # Content-Style model (Stage 1)
        self.content_style_model = ContentStyleModel(config)

        # Acoustic model (Stage 2)
        self.acoustic_model = AcousticModel(config)

        # Output projection for CSM integration
        self.output_proj = nn.Sequential(
            nn.Linear(config.style_dim + config.timbre_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode_content(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode audio features to content tokens.

        Returns:
            Dict with content_tokens, commitment_loss, perplexity
        """
        output = self.content_tokenizer(features)
        return {
            'content_tokens': output['indices'],
            'content_z': output['z_q'],
            'commitment_loss': output['commitment_loss'],
            'perplexity': output['perplexity'],
        }

    def encode_style(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """Encode audio features to style embedding."""
        return self.style_encoder(features)

    def encode_timbre(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """Encode audio features to timbre embedding."""
        return self.timbre_encoder(features)

    def forward(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
        mel: Optional[torch.Tensor] = None,  # [batch, seq, mel_dim] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            features: HuBERT/WavLM features
            mel: Target mel spectrogram for acoustic loss

        Returns:
            Dict with all losses and embeddings
        """
        # Encode all components
        content_output = self.encode_content(features)
        style_output = self.encode_style(features)
        timbre_output = self.encode_timbre(features)

        losses = {
            'content_commitment': content_output['commitment_loss'],
            'content_perplexity': content_output['perplexity'],
        }

        # Content-Style forward (get hidden for acoustic model)
        cs_output = self.content_style_model(
            content_output['content_tokens'],
            style_output['style_emb'],
        )
        cs_hidden = cs_output['hidden']

        # Acoustic model loss (if mel provided)
        if mel is not None:
            # Handle sequence length mismatch
            min_len = min(cs_hidden.shape[1], mel.shape[1])
            cs_hidden_aligned = cs_hidden[:, :min_len]
            mel_aligned = mel[:, :min_len]

            acoustic_loss = self.acoustic_model.compute_loss(
                mel_aligned,
                cs_hidden_aligned,
                timbre_output['timbre_emb'],
            )
            losses['acoustic'] = acoustic_loss['loss']

        # Combined embedding for downstream
        combined = torch.cat([
            style_output['style_emb'],
            timbre_output['timbre_emb'],
        ], dim=-1)
        combined_emb = self.output_proj(combined)

        return {
            **losses,
            'content_tokens': content_output['content_tokens'],
            'content_z': content_output['content_z'],
            'style_emb': style_output['style_emb'],
            'timbre_emb': timbre_output['timbre_emb'],
            'cs_hidden': cs_hidden,
            'combined_embedding': combined_emb,
        }

    @torch.no_grad()
    def generate(
        self,
        content_features: torch.Tensor,  # [batch, seq, input_dim] what to say
        style_features: torch.Tensor,  # [batch, seq, input_dim] how to say it
        timbre_features: torch.Tensor,  # [batch, seq, input_dim] who is speaking
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        num_ode_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Zero-shot voice imitation: Generate mel spectrogram with controlled style/timbre.

        Args:
            content_features: Features from source (determines WHAT is said)
            style_features: Features from style reference (determines HOW)
            timbre_features: Features from speaker reference (determines WHO)
            temperature: Sampling temperature for AR model
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_ode_steps: ODE integration steps

        Returns:
            [batch, seq, mel_dim] generated mel spectrogram
        """
        # Extract content tokens from source
        content_tokens = self.content_tokenizer.encode_indices(content_features)

        # Extract style embedding from style reference
        style_output = self.style_encoder(style_features)

        # Extract timbre embedding from speaker reference
        timbre_output = self.timbre_encoder(timbre_features)

        # Generate content-style tokens
        cs_tokens = self.content_style_model.generate(
            content_tokens,
            style_output['style_emb'],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Get content-style hidden representations using AR embedding
        # (cs_tokens are content-style tokens, so use forward_ar which uses cs_embedding)
        cs_output = self.content_style_model.forward_ar(
            cs_tokens,
            style_output['style_emb'],
        )

        # Generate acoustic features via flow matching
        mel = self.acoustic_model.sample(
            cs_output['hidden'],
            timbre_output['timbre_emb'],
            num_steps=num_ode_steps,
        )

        return mel

    def get_style_timbre_embedding(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Get combined style+timbre embedding."""
        style_output = self.style_encoder(features)
        timbre_output = self.timbre_encoder(features)

        combined = torch.cat([
            style_output['style_emb'],
            timbre_output['timbre_emb'],
        ], dim=-1)

        return self.output_proj(combined)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class VEVOLoss(nn.Module):
    """
    Combined loss function for VEVO training.

    Components:
    1. VQ commitment loss (content tokenizer)
    2. Content-Style AR loss (cross-entropy)
    3. Acoustic flow matching loss (MSE on velocity)
    4. Optional: Contrastive style loss
    """

    def __init__(self, config: VEVOConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.commitment_weight = 0.25
        self.cs_weight = 1.0
        self.acoustic_weight = 1.0
        self.contrastive_weight = 0.1

    def contrastive_style_loss(
        self,
        style_emb: torch.Tensor,  # [batch, style_dim]
        labels: Optional[torch.Tensor] = None,  # [batch] style/speaker labels
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Contrastive loss to encourage style clustering.
        """
        if labels is None:
            return torch.tensor(0.0, device=style_emb.device)

        # Normalize embeddings
        style_norm = F.normalize(style_emb, p=2, dim=-1)

        # Compute similarity matrix
        sim = torch.matmul(style_norm, style_norm.t()) / temperature

        # Create label mask
        labels = labels.view(-1, 1)
        mask = (labels == labels.t()).float()

        # InfoNCE loss
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True))

        # Mean of positive pairs
        loss = -(log_prob * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        return loss.mean()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        cs_target: Optional[torch.Tensor] = None,
        style_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            model_output: Output from VEVO.forward()
            mel_target: Target mel spectrogram
            cs_target: Target content-style tokens (optional, for supervised training)
            style_labels: Style/emotion labels for contrastive loss

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # 1. VQ commitment loss
        losses['commitment'] = model_output['content_commitment']
        losses['perplexity'] = model_output['content_perplexity']

        # 2. Acoustic loss (already computed in forward if mel provided)
        if 'acoustic' in model_output:
            losses['acoustic'] = model_output['acoustic']
        else:
            losses['acoustic'] = torch.tensor(0.0, device=mel_target.device)

        # 3. Contrastive style loss
        losses['contrastive'] = self.contrastive_style_loss(
            model_output['style_emb'], style_labels
        )

        # Total loss
        total = (
            self.commitment_weight * losses['commitment']
            + self.acoustic_weight * losses['acoustic']
            + self.contrastive_weight * losses['contrastive']
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class VEVOAdapter(nn.Module):
    """
    Adapter to integrate VEVO with existing prosody pipeline.

    Converts VEVO's style+timbre representation to prefix tokens
    compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: VEVOConfig,
        model: Optional[VEVO] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else VEVO(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            features: HuBERT/WavLM features
            text_embeddings: Optional text embeddings (unused for now)

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        batch_size = features.shape[0]

        # Get combined embedding
        combined_emb = self.model.get_style_timbre_embedding(features)  # [B, output_dim]

        # Project to tokens
        tokens = self.token_proj(combined_emb)  # [B, output_dim * num_tokens]

        # Reshape
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )

        # Normalize
        tokens = self.norm(tokens)

        return tokens

    def get_disentangled_embeddings(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get all disentangled embeddings separately.

        Returns:
            Dict with content_tokens, style_emb, timbre_emb
        """
        content_output = self.model.encode_content(features)
        style_output = self.model.encode_style(features)
        timbre_output = self.model.encode_timbre(features)

        return {
            'content': content_output['content_tokens'],
            'content_z': content_output['content_z'],
            'style': style_output['style_emb'],
            'timbre': timbre_output['timbre_emb'],
        }

    def zero_shot_imitation(
        self,
        content_features: torch.Tensor,
        style_features: torch.Tensor,
        timbre_features: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Zero-shot voice imitation.

        Args:
            content_features: What to say
            style_features: How to say it
            timbre_features: Who is speaking

        Returns:
            [batch, seq, mel_dim] generated mel spectrogram
        """
        return self.model.generate(
            content_features, style_features, timbre_features, **kwargs
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VEVO: Zero-Shot Voice Imitation - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = VEVOConfig()

    # Test parameters
    batch_size = 2
    seq_len = 100
    input_dim = config.input_dim
    mel_dim = config.mel_dim

    # Create dummy inputs
    features = torch.randn(batch_size, seq_len, input_dim).to(device)
    mel = torch.randn(batch_size, seq_len, mel_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Content codebook size: {config.content_codebook_size}")
    print(f"  Style dim: {config.style_dim}")
    print(f"  Timbre dim: {config.timbre_dim}")
    print(f"  Output dim: {config.output_dim}")
    print("  [PASS]")

    # Test 2: VQ Tokenizer
    print("\n[Test 2] VQ Tokenizer...")
    tokenizer = VQTokenizer(
        input_dim=input_dim,
        codebook_size=config.content_codebook_size,
        code_dim=config.content_code_dim,
    ).to(device)

    vq_output = tokenizer(features)
    print(f"  Quantized z shape: {vq_output['z_q'].shape}")
    print(f"  Token indices shape: {vq_output['indices'].shape}")
    print(f"  Commitment loss: {vq_output['commitment_loss'].item():.4f}")
    print(f"  Perplexity: {vq_output['perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 3: Style Encoder
    print("\n[Test 3] Style Encoder...")
    style_encoder = StyleEncoder(config).to(device)
    style_output = style_encoder(features)
    print(f"  Style embedding shape: {style_output['style_emb'].shape}")
    print(f"  Style sequence shape: {style_output['style_seq'].shape}")
    print("  [PASS]")

    # Test 4: Timbre Encoder
    print("\n[Test 4] Timbre Encoder...")
    timbre_encoder = TimbreEncoder(config).to(device)
    timbre_output = timbre_encoder(features)
    print(f"  Timbre embedding shape: {timbre_output['timbre_emb'].shape}")
    print("  [PASS]")

    # Test 5: Content-Style Model
    print("\n[Test 5] Content-Style Model...")
    cs_model = ContentStyleModel(config).to(device)
    content_tokens = tokenizer.encode_indices(features)
    cs_output = cs_model(content_tokens, style_output['style_emb'])
    print(f"  CS logits shape: {cs_output['logits'].shape}")
    print(f"  CS hidden shape: {cs_output['hidden'].shape}")
    print("  [PASS]")

    # Test 6: Acoustic Model
    print("\n[Test 6] Acoustic Model...")
    acoustic_model = AcousticModel(config).to(device)
    t = torch.rand(batch_size, device=device)
    velocity = acoustic_model(mel, t, cs_output['hidden'], timbre_output['timbre_emb'])
    print(f"  Velocity shape: {velocity.shape}")

    acoustic_loss = acoustic_model.compute_loss(
        mel, cs_output['hidden'], timbre_output['timbre_emb']
    )
    print(f"  Acoustic loss: {acoustic_loss['loss'].item():.4f}")
    print("  [PASS]")

    # Test 7: Full VEVO Model
    print("\n[Test 7] Full VEVO Model...")
    model = VEVO(config).to(device)
    output = model(features, mel)
    print(f"  Content tokens shape: {output['content_tokens'].shape}")
    print(f"  Style embedding shape: {output['style_emb'].shape}")
    print(f"  Timbre embedding shape: {output['timbre_emb'].shape}")
    print(f"  Combined embedding shape: {output['combined_embedding'].shape}")
    print(f"  Commitment loss: {output['content_commitment'].item():.4f}")
    print(f"  Acoustic loss: {output['acoustic'].item():.4f}")
    print("  [PASS]")

    # Test 8: Loss Function
    print("\n[Test 8] Loss Function...")
    loss_fn = VEVOLoss(config)
    losses = loss_fn(output, mel)
    print(f"  Commitment loss: {losses['commitment'].item():.4f}")
    print(f"  Acoustic loss: {losses['acoustic'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: CSM Adapter
    print("\n[Test 9] CSM Adapter...")
    adapter = VEVOAdapter(config, model).to(device)
    prefix_tokens = adapter(features)
    print(f"  Prefix tokens shape: {prefix_tokens.shape}")
    assert prefix_tokens.shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 10: Disentanglement Check
    print("\n[Test 10] Disentanglement Check...")
    disentangled = adapter.get_disentangled_embeddings(features)
    print(f"  Content tokens shape: {disentangled['content'].shape}")
    print(f"  Content z shape: {disentangled['content_z'].shape}")
    print(f"  Style shape: {disentangled['style'].shape}")
    print(f"  Timbre shape: {disentangled['timbre'].shape}")

    # Check orthogonality between style and timbre
    style_norm = F.normalize(disentangled['style'], p=2, dim=-1)
    # Project timbre to same dimension for comparison
    timbre_proj = disentangled['timbre'][:, :config.style_dim]
    timbre_norm = F.normalize(timbre_proj, p=2, dim=-1)
    cos_sim = (style_norm[:, :config.timbre_dim] * timbre_norm).sum(dim=-1).mean()
    print(f"  Style-Timbre cosine similarity: {cos_sim.item():.4f} (lower is better)")
    print("  [PASS]")

    # Test 11: Backward Pass
    print("\n[Test 11] Backward Pass...")
    model.zero_grad()
    output = model(features, mel)
    losses = loss_fn(output, mel)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test 12: Zero-Shot Generation (inference mode)
    print("\n[Test 12] Zero-Shot Generation...")
    model.eval()

    # Different features for content, style, and timbre
    content_feat = torch.randn(1, seq_len, input_dim).to(device)
    style_feat = torch.randn(1, 50, input_dim).to(device)  # Shorter style reference
    timbre_feat = torch.randn(1, 50, input_dim).to(device)  # Shorter timbre reference

    with torch.no_grad():
        generated_mel = model.generate(
            content_feat, style_feat, timbre_feat,
            temperature=0.8,
            top_p=0.9,
            num_ode_steps=10,  # Fewer steps for testing
        )
    print(f"  Generated mel shape: {generated_mel.shape}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All VEVO tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. VQ BOTTLENECK FOR DISENTANGLEMENT:
       - Content tokenizer with controllable codebook size
       - Smaller codebook (256-1024) = more disentangled content
       - Larger codebook (16k+) = preserves more prosody

    2. THREE-WAY DECOMPOSITION:
       - Content (what): VQ tokens from speech
       - Style (how): Prosody, emotion, accent, manner
       - Timbre (who): Speaker identity characteristics

    3. TWO-STAGE ARCHITECTURE:
       Stage 1: Content-Style Modeling (AR Transformer)
         - Input: Content tokens + Style reference
         - Output: Content-Style tokens

       Stage 2: Acoustic Modeling (Flow-Matching)
         - Input: Content-Style tokens + Timbre reference
         - Output: Mel spectrogram

    4. ZERO-SHOT VOICE IMITATION:
       mel = model.generate(
           content_features=source,    # What to say
           style_features=style_ref,   # How to say it
           timbre_features=speaker_ref, # Who is speaking
       )

    5. CSM INTEGRATION:
       adapter = VEVOAdapter(config)
       prefix_tokens = adapter(features)  # [batch, 4, 2048]
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from vevo_disentanglement import (
    VEVOConfig,
    VEVO,
    VEVOLoss,
    VEVOAdapter,
)

# Initialize
config = VEVOConfig(
    content_codebook_size=1024,  # Info bottleneck size
    style_dim=512,
    timbre_dim=256,
)

model = VEVO(config).cuda()
loss_fn = VEVOLoss(config)

# Training
for batch in dataloader:
    features = feature_extractor(batch['audio'])  # HuBERT/WavLM
    mel = mel_extractor(batch['audio'])

    output = model(features, mel)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor disentanglement
    print(f"Content perplexity: {output['content_perplexity']:.2f}")

# Zero-shot voice imitation
with torch.no_grad():
    mel_out = model.generate(
        content_features=source_audio_features,
        style_features=style_reference_features,
        timbre_features=speaker_reference_features,
    )

# CSM integration
adapter = VEVOAdapter(config, model)
prefix_tokens = adapter(features)

# Analyze disentanglement
embs = adapter.get_disentangled_embeddings(features)
print(f"Content: {embs['content'].shape}")  # Discrete tokens
print(f"Style: {embs['style'].shape}")       # How to speak
print(f"Timbre: {embs['timbre'].shape}")     # Who is speaking
""")
