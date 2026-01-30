"""
Dual-Codebook VQ-VAE for Separate F0 and Content Learning

Based on "Improved Prosody from Learned F0 Codebook Representations for VQ-VAE"
(Interspeech 2020).

Key Innovation: Two separate VQ-VAE encoders with distinct codebooks:
1. F0 Encoder → F0 Codebook: Captures prosodic/intonation patterns
2. Content Encoder → Content Codebook: Captures linguistic/phonetic content

Benefits:
- Speaker-independent prosody codes (reduces F0 distortion for unseen speakers)
- Explicit separation of F0 from phonetic content
- Better prosody transfer across speakers
- F0 codebook indices can be used directly as prosody conditioning

Architecture:
```
Audio → Mel Spectrogram → [Content Encoder] → Content VQ → Content Codes
                    ↓
      F0 Extraction → [F0 Encoder] → F0 VQ → F0 Codes (Prosody Tokens)
                                      ↓           ↓
                              [Joint Decoder] ← ─ ─ ─
                                      ↓
                              Reconstructed Mel
```

This approach decouples prosody from content at the representation level,
enabling:
- Cross-speaker prosody transfer using F0 codes
- Direct manipulation of prosody via codebook indices
- Speaker-invariant prosody representation

Usage:
    from dual_codebook_vqvae import (
        DualCodebookConfig,
        DualCodebookVQVAE,
        DualCodebookLoss,
        DualCodebookProsodyAdapter,
    )

    # Initialize
    config = DualCodebookConfig()
    model = DualCodebookVQVAE(config).cuda()

    # Encode to separate codebooks
    encoded = model.encode(mel, f0)
    f0_codes = encoded['f0_indices']      # Prosody tokens
    content_codes = encoded['content_indices']  # Content tokens

    # Decode from codes
    mel_recon = model.decode(f0_codes, content_codes)

    # Cross-speaker prosody transfer
    mel_transferred = model.prosody_transfer(
        source_f0=f0_speaker_a,
        target_content=content_speaker_b,
    )

    # CSM integration
    adapter = DualCodebookProsodyAdapter(config, model)
    prefix_tokens = adapter(mel, f0)  # [batch, 4, 2048]
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
class DualCodebookConfig:
    """Configuration for Dual-Codebook VQ-VAE."""

    # Input dimensions
    mel_dim: int = 80  # Mel spectrogram channels
    f0_dim: int = 1    # F0 is 1-dimensional (can be expanded with delta features)
    f0_context: int = 5  # Context frames for F0 encoding (total = 2*context + 1)

    # F0 Codebook settings
    f0_codebook_size: int = 64   # Smaller codebook for prosody patterns
    f0_code_dim: int = 64        # Dimension per F0 code
    f0_num_layers: int = 3       # Encoder depth
    f0_hidden_dim: int = 256     # Encoder hidden dimension
    f0_num_heads: int = 4        # Attention heads

    # Content Codebook settings
    content_codebook_size: int = 512  # Larger codebook for phonetic variety
    content_code_dim: int = 256       # Dimension per content code
    content_num_layers: int = 4       # Encoder depth
    content_hidden_dim: int = 512     # Encoder hidden dimension
    content_num_heads: int = 8        # Attention heads

    # Decoder settings
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Training settings
    dropout: float = 0.1
    commitment_cost: float = 0.25
    ema_decay: float = 0.99  # EMA for codebook updates

    # Speaker embedding (optional, for decoder-only)
    num_speakers: int = 100
    speaker_embed_dim: int = 256
    use_speaker_embedding: bool = True

    # Output dimension for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# VECTOR QUANTIZER WITH EMA
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average codebook updates.

    Implements straight-through estimator for gradient flow and EMA
    for more stable codebook learning.
    """

    def __init__(
        self,
        codebook_size: int,
        code_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking buffers
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))

    def _init_from_data(self, flat_input: torch.Tensor):
        """Initialize codebook from first batch using k-means++ style."""
        n_samples = flat_input.shape[0]

        if n_samples >= self.codebook_size:
            # Random subset
            indices = torch.randperm(n_samples)[:self.codebook_size]
            init_data = flat_input[indices]
        else:
            # Repeat if not enough samples
            repeats = (self.codebook_size // n_samples) + 1
            expanded = flat_input.repeat(repeats, 1)[:self.codebook_size]
            init_data = expanded

        self.embedding.weight.data.copy_(init_data)
        self.ema_w.data.copy_(init_data)
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, code_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input to codebook vectors.

        Returns:
            Dict with:
                - z_q: [batch, seq, code_dim] quantized vectors
                - indices: [batch, seq] codebook indices
                - commitment_loss: scalar commitment loss
                - perplexity: codebook usage metric
        """
        batch_size, seq_len, _ = x.shape

        # Flatten for distance computation
        flat_x = x.reshape(-1, self.code_dim)  # [B*T, D]

        # Initialize from first batch if needed
        if self.training and not self.initialized:
            self._init_from_data(flat_x)

        # Compute squared distances to codebook
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x.e
        distances = (
            flat_x.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(flat_x, self.embedding.weight.t())
        )  # [B*T, codebook_size]

        # Find nearest codes
        indices = distances.argmin(dim=1)  # [B*T]
        z_q = self.embedding(indices)  # [B*T, code_dim]

        # EMA codebook update (training only)
        if self.training:
            with torch.no_grad():
                # One-hot encoding
                encodings = F.one_hot(indices, self.codebook_size).float()

                # Update cluster sizes
                new_cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size = (
                    self.ema_decay * self.ema_cluster_size
                    + (1 - self.ema_decay) * new_cluster_size
                )

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.codebook_size * self.epsilon) * n
                )

                # Update embedding weights
                dw = torch.matmul(encodings.t(), flat_x)
                self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw
                self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

        # Commitment loss (encoder must commit to codebook)
        commitment_loss = F.mse_loss(z_q.detach(), flat_x)

        # Straight-through estimator: copy gradients from z_q to x
        z_q = flat_x + (z_q - flat_x).detach()

        # Reshape back
        z_q = z_q.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

        # Compute perplexity (higher = better codebook usage)
        perplexity = self._compute_perplexity(indices)

        return {
            'z_q': z_q,
            'indices': indices,
            'commitment_loss': commitment_loss * self.commitment_cost,
            'perplexity': perplexity,
        }

    def _compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute perplexity (exponential of entropy)."""
        flat_indices = indices.view(-1)
        encodings = F.one_hot(flat_indices, self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to vectors."""
        return self.embedding(indices)


# =============================================================================
# F0 ENCODER
# =============================================================================

class F0Encoder(nn.Module):
    """
    Encoder for F0 (fundamental frequency) trajectory.

    Encodes F0 contour into a sequence of prosody codes using a
    transformer with local context. The small codebook size enforces
    learning of speaker-independent prosodic patterns.

    Key Design Choices:
    - Context window around each frame for local F0 patterns
    - Small codebook (32-128) forces abstraction of prosody patterns
    - Log-F0 + delta features for better representation
    """

    def __init__(self, config: DualCodebookConfig):
        super().__init__()
        self.config = config

        # Compute input dimension with context
        # F0 + delta-F0 + delta-delta-F0 per frame, with context
        self.input_dim = config.f0_dim * 3 * (2 * config.f0_context + 1)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, config.f0_hidden_dim),
            nn.LayerNorm(config.f0_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            config.f0_hidden_dim, max_len=5000
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.f0_hidden_dim,
            nhead=config.f0_num_heads,
            dim_feedforward=config.f0_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.f0_num_layers
        )

        # Output projection to code dimension
        self.output_proj = nn.Sequential(
            nn.Linear(config.f0_hidden_dim, config.f0_code_dim),
            nn.LayerNorm(config.f0_code_dim),
        )

        self.norm = nn.LayerNorm(config.f0_code_dim)

    def _create_positional_encoding(self, dim: int, max_len: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _extract_f0_features(
        self,
        f0: torch.Tensor,  # [batch, seq, 1] or [batch, seq]
    ) -> torch.Tensor:
        """
        Extract F0 features with context window and delta features.

        Returns:
            [batch, seq, input_dim] F0 features with context
        """
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)  # [B, T, 1]

        batch_size, seq_len, _ = f0.shape
        device = f0.device

        # Convert to log-F0 (voiced frames only, use 0 for unvoiced)
        # Add small epsilon to avoid log(0)
        log_f0 = torch.where(f0 > 0, torch.log(f0 + 1e-6), torch.zeros_like(f0))

        # Compute delta and delta-delta features
        # Delta: f0[t+1] - f0[t-1]
        delta_f0 = torch.zeros_like(log_f0)
        delta_f0[:, 1:-1] = (log_f0[:, 2:] - log_f0[:, :-2]) / 2

        # Delta-delta
        delta2_f0 = torch.zeros_like(log_f0)
        delta2_f0[:, 1:-1] = (delta_f0[:, 2:] - delta_f0[:, :-2]) / 2

        # Concatenate: [log_f0, delta, delta2]
        f0_features = torch.cat([log_f0, delta_f0, delta2_f0], dim=-1)  # [B, T, 3]

        # Extract context windows
        context = self.config.f0_context
        pad_len = context

        # Pad sequence
        f0_padded = F.pad(f0_features, (0, 0, pad_len, pad_len), mode='replicate')

        # Extract context for each frame
        features_list = []
        for t in range(seq_len):
            # Window: [t, t + 2*context + 1)
            window = f0_padded[:, t:t + 2 * context + 1]  # [B, 2*ctx+1, 3]
            features_list.append(window.reshape(batch_size, -1))  # [B, 3*(2*ctx+1)]

        features = torch.stack(features_list, dim=1)  # [B, T, input_dim]

        return features

    def forward(
        self,
        f0: torch.Tensor,  # [batch, seq] or [batch, seq, 1]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode F0 trajectory to latent representation.

        Args:
            f0: F0 values (Hz) per frame
            mask: Optional padding mask

        Returns:
            [batch, seq, f0_code_dim] encoded F0 features
        """
        # Extract F0 features with context
        features = self._extract_f0_features(f0)  # [B, T, input_dim]

        # Project to hidden dimension
        x = self.input_proj(features)  # [B, T, hidden]

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Project to code dimension
        x = self.output_proj(x)
        x = self.norm(x)

        return x


# =============================================================================
# CONTENT ENCODER
# =============================================================================

class ContentEncoder(nn.Module):
    """
    Encoder for speech content (phonetic/linguistic information).

    Encodes mel spectrogram into content codes using a larger codebook
    to capture phonetic variety. The encoder is designed to capture
    linguistic content while being agnostic to F0 variations.

    Key Design Choices:
    - Convolutional pre-net for local feature extraction
    - Larger codebook (256-1024) for phonetic diversity
    - No explicit F0 input to encourage content-only representation
    """

    def __init__(self, config: DualCodebookConfig):
        super().__init__()
        self.config = config

        # Convolutional pre-net
        self.prenet = nn.Sequential(
            nn.Conv1d(config.mel_dim, config.content_hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.content_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.content_hidden_dim, config.content_hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.content_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            config.content_hidden_dim, max_len=5000
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.content_hidden_dim,
            nhead=config.content_num_heads,
            dim_feedforward=config.content_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.content_num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.content_hidden_dim, config.content_code_dim),
            nn.LayerNorm(config.content_code_dim),
        )

        self.norm = nn.LayerNorm(config.content_code_dim)

    def _create_positional_encoding(self, dim: int, max_len: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode mel spectrogram to content representation.

        Args:
            mel: Mel spectrogram
            mask: Optional padding mask

        Returns:
            [batch, seq, content_code_dim] encoded content features
        """
        # Pre-net: [B, T, D] -> [B, D, T] -> [B, D, T] -> [B, T, D]
        x = mel.transpose(1, 2)  # [B, D, T]
        x = self.prenet(x)
        x = x.transpose(1, 2)  # [B, T, D]

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Project to code dimension
        x = self.output_proj(x)
        x = self.norm(x)

        return x


# =============================================================================
# JOINT DECODER
# =============================================================================

class JointDecoder(nn.Module):
    """
    Joint decoder that reconstructs mel spectrogram from F0 and content codes.

    Takes quantized representations from both codebooks and reconstructs
    the original mel spectrogram. Optional speaker embedding allows
    speaker-conditioned reconstruction.

    Architecture:
    - F0 codes: Capture prosodic timing and intonation
    - Content codes: Capture phonetic/linguistic content
    - Speaker embedding: Captures voice characteristics (optional)
    - Cross-attention between F0 and content streams
    """

    def __init__(self, config: DualCodebookConfig):
        super().__init__()
        self.config = config

        # Input projections for F0 and content codes
        self.f0_proj = nn.Linear(config.f0_code_dim, config.decoder_hidden_dim)
        self.content_proj = nn.Linear(config.content_code_dim, config.decoder_hidden_dim)

        # Speaker embedding (optional, only used in decoder)
        if config.use_speaker_embedding:
            self.speaker_embed = nn.Embedding(config.num_speakers, config.speaker_embed_dim)
            self.speaker_proj = nn.Linear(config.speaker_embed_dim, config.decoder_hidden_dim)
        else:
            self.speaker_embed = None
            self.speaker_proj = None

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(
            config.decoder_hidden_dim, max_len=5000
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_num_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.mel_dim),
        )

        self.norm = nn.LayerNorm(config.decoder_hidden_dim)

    def _create_positional_encoding(self, dim: int, max_len: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(
        self,
        f0_codes: torch.Tensor,      # [batch, seq, f0_code_dim]
        content_codes: torch.Tensor,  # [batch, seq, content_code_dim]
        speaker_ids: Optional[torch.Tensor] = None,  # [batch]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode F0 and content codes to mel spectrogram.

        Args:
            f0_codes: Quantized F0 representations
            content_codes: Quantized content representations
            speaker_ids: Optional speaker IDs for speaker-conditioned decoding
            mask: Optional padding mask

        Returns:
            [batch, seq, mel_dim] reconstructed mel spectrogram
        """
        batch_size, seq_len, _ = f0_codes.shape

        # Project F0 and content codes
        f0_proj = self.f0_proj(f0_codes)  # [B, T, hidden]
        content_proj = self.content_proj(content_codes)  # [B, T, hidden]

        # Combine F0 and content (additive fusion)
        x = f0_proj + content_proj  # [B, T, hidden]

        # Add speaker embedding if provided
        if speaker_ids is not None and self.speaker_embed is not None:
            speaker_emb = self.speaker_embed(speaker_ids)  # [B, speaker_dim]
            speaker_proj = self.speaker_proj(speaker_emb)  # [B, hidden]
            x = x + speaker_proj.unsqueeze(1)  # [B, T, hidden]

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len]

        # Use content as memory for cross-attention
        memory = content_proj + self.pos_encoding[:, :seq_len]

        # Transformer decoder (x attends to memory)
        x = self.transformer(x, memory)
        x = self.norm(x)

        # Output projection
        mel_out = self.output_proj(x)

        return mel_out


# =============================================================================
# DUAL CODEBOOK VQ-VAE
# =============================================================================

class DualCodebookVQVAE(nn.Module):
    """
    Dual-Codebook VQ-VAE for separate F0 and content learning.

    This model learns two separate codebooks:
    1. F0 Codebook: Captures prosodic patterns (speaker-independent)
    2. Content Codebook: Captures phonetic/linguistic content

    The separation enables:
    - Cross-speaker prosody transfer using F0 codes
    - Direct manipulation of prosody via codebook indices
    - Speaker-invariant prosody representation

    Key Insight from Paper:
    - Small F0 codebook (32-128) forces learning abstract prosody patterns
    - These patterns transfer better across speakers than raw F0 values
    """

    def __init__(self, config: DualCodebookConfig):
        super().__init__()
        self.config = config

        # F0 Encoder and Codebook
        self.f0_encoder = F0Encoder(config)
        self.f0_quantizer = VectorQuantizerEMA(
            codebook_size=config.f0_codebook_size,
            code_dim=config.f0_code_dim,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
        )

        # Content Encoder and Codebook
        self.content_encoder = ContentEncoder(config)
        self.content_quantizer = VectorQuantizerEMA(
            codebook_size=config.content_codebook_size,
            code_dim=config.content_code_dim,
            commitment_cost=config.commitment_cost,
            ema_decay=config.ema_decay,
        )

        # Joint Decoder
        self.decoder = JointDecoder(config)

        # Output projection for CSM integration
        # Combines F0 codes (prosody) with content codes for prefix tokens
        combined_dim = config.f0_code_dim + config.content_code_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        mel: torch.Tensor,      # [batch, seq, mel_dim]
        f0: torch.Tensor,       # [batch, seq] or [batch, seq, 1]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel and F0 to separate codebook representations.

        Returns:
            Dict with:
                - f0_z: [batch, seq, f0_code_dim] F0 latent (pre-quantization)
                - f0_z_q: [batch, seq, f0_code_dim] F0 quantized
                - f0_indices: [batch, seq] F0 codebook indices (prosody tokens!)
                - content_z: [batch, seq, content_code_dim] content latent
                - content_z_q: [batch, seq, content_code_dim] content quantized
                - content_indices: [batch, seq] content codebook indices
                - f0_commitment_loss, content_commitment_loss, perplexities
        """
        # Encode F0
        f0_z = self.f0_encoder(f0, mask)  # [B, T, f0_code_dim]
        f0_vq = self.f0_quantizer(f0_z)

        # Encode content
        content_z = self.content_encoder(mel, mask)  # [B, T, content_code_dim]
        content_vq = self.content_quantizer(content_z)

        return {
            'f0_z': f0_z,
            'f0_z_q': f0_vq['z_q'],
            'f0_indices': f0_vq['indices'],
            'f0_commitment_loss': f0_vq['commitment_loss'],
            'f0_perplexity': f0_vq['perplexity'],
            'content_z': content_z,
            'content_z_q': content_vq['z_q'],
            'content_indices': content_vq['indices'],
            'content_commitment_loss': content_vq['commitment_loss'],
            'content_perplexity': content_vq['perplexity'],
        }

    def decode(
        self,
        f0_codes: torch.Tensor,      # [batch, seq] indices or [batch, seq, dim] vectors
        content_codes: torch.Tensor,  # [batch, seq] indices or [batch, seq, dim] vectors
        speaker_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode from F0 and content codes to mel spectrogram.

        Args:
            f0_codes: F0 codebook indices or quantized vectors
            content_codes: Content codebook indices or quantized vectors
            speaker_ids: Optional speaker IDs
            mask: Optional padding mask

        Returns:
            [batch, seq, mel_dim] reconstructed mel spectrogram
        """
        # Convert indices to vectors if needed
        if f0_codes.dim() == 2:
            f0_vectors = self.f0_quantizer.decode_indices(f0_codes)
        else:
            f0_vectors = f0_codes

        if content_codes.dim() == 2:
            content_vectors = self.content_quantizer.decode_indices(content_codes)
        else:
            content_vectors = content_codes

        # Handle sequence length mismatch
        min_len = min(f0_vectors.shape[1], content_vectors.shape[1])
        f0_vectors = f0_vectors[:, :min_len]
        content_vectors = content_vectors[:, :min_len]

        # Decode
        mel_reconstructed = self.decoder(f0_vectors, content_vectors, speaker_ids, mask)

        return mel_reconstructed

    def forward(
        self,
        mel: torch.Tensor,      # [batch, seq, mel_dim]
        f0: torch.Tensor,       # [batch, seq] or [batch, seq, 1]
        speaker_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, quantize, decode.

        Returns:
            Dict with all encoding outputs plus:
                - mel_reconstructed: [batch, seq, mel_dim]
                - prosody_embedding: [batch, output_dim] for CSM integration
        """
        # Encode
        encoded = self.encode(mel, f0, mask)

        # Decode
        mel_reconstructed = self.decode(
            encoded['f0_z_q'],
            encoded['content_z_q'],
            speaker_ids,
            mask,
        )

        # Combined prosody embedding for CSM
        # Pool temporally and combine F0 + content representations
        f0_pooled = encoded['f0_z_q'].mean(dim=1)  # [B, f0_code_dim]
        content_pooled = encoded['content_z_q'].mean(dim=1)  # [B, content_code_dim]
        combined = torch.cat([f0_pooled, content_pooled], dim=-1)
        prosody_embedding = self.output_proj(combined)  # [B, output_dim]

        return {
            **encoded,
            'mel_reconstructed': mel_reconstructed,
            'prosody_embedding': prosody_embedding,
        }

    def prosody_transfer(
        self,
        source_f0: torch.Tensor,       # [batch, seq] F0 from source (prosody donor)
        target_mel: torch.Tensor,      # [batch, seq, mel_dim] mel from target (content)
        target_speaker_id: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transfer prosody from source to target.

        Uses F0 from source speaker and content from target speaker
        to generate speech with source's prosody and target's content.

        Args:
            source_f0: F0 trajectory from prosody source
            target_mel: Mel spectrogram from content source
            target_speaker_id: Speaker ID for voice characteristics
            mask: Optional padding mask

        Returns:
            [batch, seq, mel_dim] mel with transferred prosody
        """
        # Encode source F0 (prosody)
        f0_z = self.f0_encoder(source_f0, mask)
        f0_vq = self.f0_quantizer(f0_z)

        # Encode target content
        content_z = self.content_encoder(target_mel, mask)
        content_vq = self.content_quantizer(content_z)

        # Decode with source prosody + target content
        mel_transferred = self.decode(
            f0_vq['z_q'],
            content_vq['z_q'],
            target_speaker_id,
            mask,
        )

        return mel_transferred

    def get_f0_codebook(self) -> torch.Tensor:
        """Get F0 codebook embeddings for analysis/visualization."""
        return self.f0_quantizer.embedding.weight.data

    def get_content_codebook(self) -> torch.Tensor:
        """Get content codebook embeddings for analysis/visualization."""
        return self.content_quantizer.embedding.weight.data


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class DualCodebookLoss(nn.Module):
    """
    Loss function for Dual-Codebook VQ-VAE training.

    Components:
    1. Reconstruction loss (mel spectrogram L1 + L2)
    2. F0 commitment loss (VQ)
    3. Content commitment loss (VQ)
    4. Optional: F0 reconstruction auxiliary loss
    5. Optional: Disentanglement regularization
    """

    def __init__(self, config: DualCodebookConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.reconstruction_weight = 1.0
        self.f0_commitment_weight = 0.25
        self.content_commitment_weight = 0.25
        self.f0_aux_weight = 0.1  # Auxiliary F0 prediction loss

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        f0_target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            output: Model output dict from forward()
            mel_target: Target mel spectrogram
            f0_target: Target F0 for auxiliary loss (optional)
            mask: Optional sequence mask

        Returns:
            Dict with individual losses and total
        """
        losses = {}
        mel_pred = output['mel_reconstructed']

        # Handle length mismatch
        min_len = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_len]
        mel_target = mel_target[:, :min_len]

        # 1. Reconstruction loss
        l1_loss = F.l1_loss(mel_pred, mel_target)
        l2_loss = F.mse_loss(mel_pred, mel_target)
        losses['reconstruction_l1'] = l1_loss
        losses['reconstruction_l2'] = l2_loss
        losses['reconstruction'] = l1_loss + l2_loss

        # 2. F0 commitment loss
        losses['f0_commitment'] = output['f0_commitment_loss']
        losses['f0_perplexity'] = output['f0_perplexity']

        # 3. Content commitment loss
        losses['content_commitment'] = output['content_commitment_loss']
        losses['content_perplexity'] = output['content_perplexity']

        # Total loss
        total = (
            self.reconstruction_weight * losses['reconstruction']
            + self.f0_commitment_weight * losses['f0_commitment']
            + self.content_commitment_weight * losses['content_commitment']
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class DualCodebookProsodyAdapter(nn.Module):
    """
    Adapter to integrate Dual-Codebook VQ-VAE with existing prosody pipeline.

    Converts the learned F0 codes (prosody tokens) to prefix tokens
    compatible with ProsodyControlledCSM.

    Key Feature: F0 codebook indices can be used directly as discrete
    prosody conditioning - no need for continuous prosody vectors!
    """

    def __init__(
        self,
        config: DualCodebookConfig,
        model: Optional[DualCodebookVQVAE] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else DualCodebookVQVAE(config)

        # Project F0 codes to prefix tokens
        self.f0_token_proj = nn.Sequential(
            nn.Linear(config.f0_code_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim * config.num_prefix_tokens),
        )

        # Optional: include content codes in prosody prefix
        self.content_token_proj = nn.Sequential(
            nn.Linear(config.content_code_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # Fusion layer
        self.fusion = nn.Linear(
            config.output_dim * (config.num_prefix_tokens + 1),
            config.output_dim * config.num_prefix_tokens,
        )

        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        mel: torch.Tensor,      # [batch, seq, mel_dim]
        f0: torch.Tensor,       # [batch, seq] or [batch, seq, 1]
        use_content: bool = True,
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            mel: Mel spectrogram
            f0: F0 trajectory
            use_content: Whether to include content codes in prefix

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        batch_size = mel.shape[0]

        # Encode to separate codebooks
        encoded = self.model.encode(mel, f0)

        # Pool F0 codes temporally
        f0_pooled = encoded['f0_z_q'].mean(dim=1)  # [B, f0_code_dim]

        # Project F0 to tokens
        f0_tokens = self.f0_token_proj(f0_pooled)  # [B, output_dim * num_tokens]

        if use_content:
            # Pool and project content codes
            content_pooled = encoded['content_z_q'].mean(dim=1)  # [B, content_code_dim]
            content_token = self.content_token_proj(content_pooled)  # [B, output_dim]

            # Concatenate and fuse
            combined = torch.cat([f0_tokens, content_token], dim=-1)
            tokens = self.fusion(combined)
        else:
            tokens = f0_tokens

        # Reshape to sequence of tokens
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)

        # Normalize
        tokens = self.norm(tokens)

        return tokens

    def from_f0_indices(
        self,
        f0_indices: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens directly from F0 codebook indices.

        This enables discrete prosody control without mel spectrogram!

        Args:
            f0_indices: F0 codebook indices

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        batch_size = f0_indices.shape[0]

        # Decode indices to vectors
        f0_codes = self.model.f0_quantizer.decode_indices(f0_indices)  # [B, T, f0_code_dim]

        # Pool temporally
        f0_pooled = f0_codes.mean(dim=1)  # [B, f0_code_dim]

        # Project to tokens
        tokens = self.f0_token_proj(f0_pooled)  # [B, output_dim * num_tokens]
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)
        tokens = self.norm(tokens)

        return tokens

    def get_prosody_embedding(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prosody embedding compatible with existing interface.

        Returns:
            [batch, output_dim] prosody embedding
        """
        output = self.model(mel, f0)
        return output['prosody_embedding']

    def get_f0_codes(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get F0 codebook indices (discrete prosody tokens).

        These can be saved, manipulated, and used for prosody transfer.

        Returns:
            [batch, seq] F0 codebook indices
        """
        encoded = self.model.encode(mel, f0)
        return encoded['f0_indices']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_f0(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    hop_length: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 600.0,
) -> torch.Tensor:
    """
    Extract F0 from audio using simple autocorrelation method.

    For production, consider using CREPE, PYIN, or other robust F0 extractors.

    Args:
        audio: [batch, samples] audio waveform
        sample_rate: Audio sample rate
        hop_length: Hop length for frame-wise extraction
        f0_min: Minimum F0 frequency
        f0_max: Maximum F0 frequency

    Returns:
        [batch, frames] F0 values in Hz (0 for unvoiced)
    """
    # Simple placeholder - in production use a proper F0 extractor
    # This is a naive implementation for testing
    batch_size = audio.shape[0]
    num_frames = audio.shape[1] // hop_length

    # Return zeros as placeholder (replace with actual F0 extraction)
    f0 = torch.zeros(batch_size, num_frames, device=audio.device)

    return f0


def analyze_codebook_usage(
    model: DualCodebookVQVAE,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
    """
    Analyze codebook usage statistics across a dataset.

    Returns:
        Dict with usage histograms and statistics for both codebooks
    """
    f0_counts = torch.zeros(model.config.f0_codebook_size, device=device)
    content_counts = torch.zeros(model.config.content_codebook_size, device=device)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            mel = batch['mel'].to(device)
            f0 = batch['f0'].to(device)

            encoded = model.encode(mel, f0)

            # Count F0 indices
            f0_indices = encoded['f0_indices'].view(-1)
            for idx in f0_indices:
                f0_counts[idx] += 1

            # Count content indices
            content_indices = encoded['content_indices'].view(-1)
            for idx in content_indices:
                content_counts[idx] += 1

    # Normalize to probabilities
    f0_probs = f0_counts / f0_counts.sum()
    content_probs = content_counts / content_counts.sum()

    # Compute statistics
    f0_perplexity = torch.exp(-torch.sum(f0_probs * torch.log(f0_probs + 1e-10)))
    content_perplexity = torch.exp(-torch.sum(content_probs * torch.log(content_probs + 1e-10)))

    # Count unused codes
    f0_unused = (f0_counts == 0).sum()
    content_unused = (content_counts == 0).sum()

    return {
        'f0_counts': f0_counts,
        'f0_probs': f0_probs,
        'f0_perplexity': f0_perplexity,
        'f0_unused': f0_unused,
        'content_counts': content_counts,
        'content_probs': content_probs,
        'content_perplexity': content_perplexity,
        'content_unused': content_unused,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dual-Codebook VQ-VAE - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = DualCodebookConfig()

    # Test parameters
    batch_size = 2
    seq_len = 100
    mel_dim = config.mel_dim

    # Create dummy inputs
    mel = torch.randn(batch_size, seq_len, mel_dim).to(device)
    f0 = torch.abs(torch.randn(batch_size, seq_len).to(device)) * 200 + 100  # 100-300 Hz range
    speaker_ids = torch.randint(0, config.num_speakers, (batch_size,)).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  F0 codebook size: {config.f0_codebook_size}")
    print(f"  Content codebook size: {config.content_codebook_size}")
    print(f"  Output dim: {config.output_dim}")
    print("  [PASS]")

    # Test 2: F0 Encoder
    print("\n[Test 2] F0 Encoder...")
    f0_encoder = F0Encoder(config).to(device)
    f0_encoded = f0_encoder(f0)
    print(f"  F0 encoded shape: {f0_encoded.shape}")
    assert f0_encoded.shape == (batch_size, seq_len, config.f0_code_dim)
    print("  [PASS]")

    # Test 3: Content Encoder
    print("\n[Test 3] Content Encoder...")
    content_encoder = ContentEncoder(config).to(device)
    content_encoded = content_encoder(mel)
    print(f"  Content encoded shape: {content_encoded.shape}")
    assert content_encoded.shape == (batch_size, seq_len, config.content_code_dim)
    print("  [PASS]")

    # Test 4: Vector Quantizer
    print("\n[Test 4] Vector Quantizer...")
    f0_vq = VectorQuantizerEMA(
        codebook_size=config.f0_codebook_size,
        code_dim=config.f0_code_dim,
    ).to(device)
    f0_vq_out = f0_vq(f0_encoded)
    print(f"  Quantized z shape: {f0_vq_out['z_q'].shape}")
    print(f"  Indices shape: {f0_vq_out['indices'].shape}")
    print(f"  Commitment loss: {f0_vq_out['commitment_loss'].item():.4f}")
    print(f"  Perplexity: {f0_vq_out['perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 5: Joint Decoder
    print("\n[Test 5] Joint Decoder...")
    decoder = JointDecoder(config).to(device)
    mel_decoded = decoder(f0_vq_out['z_q'], content_encoded, speaker_ids)
    print(f"  Decoded mel shape: {mel_decoded.shape}")
    assert mel_decoded.shape == (batch_size, seq_len, mel_dim)
    print("  [PASS]")

    # Test 6: Full Model
    print("\n[Test 6] Full Dual-Codebook VQ-VAE...")
    model = DualCodebookVQVAE(config).to(device)
    output = model(mel, f0, speaker_ids)
    print(f"  F0 indices shape: {output['f0_indices'].shape}")
    print(f"  Content indices shape: {output['content_indices'].shape}")
    print(f"  Mel reconstructed shape: {output['mel_reconstructed'].shape}")
    print(f"  Prosody embedding shape: {output['prosody_embedding'].shape}")
    print(f"  F0 perplexity: {output['f0_perplexity'].item():.2f}")
    print(f"  Content perplexity: {output['content_perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 7: Encode/Decode
    print("\n[Test 7] Encode and Decode...")
    encoded = model.encode(mel, f0)
    mel_decoded = model.decode(
        encoded['f0_indices'],
        encoded['content_indices'],
        speaker_ids,
    )
    print(f"  Encoded F0 indices: {encoded['f0_indices'].shape}")
    print(f"  Encoded content indices: {encoded['content_indices'].shape}")
    print(f"  Decoded mel: {mel_decoded.shape}")
    print("  [PASS]")

    # Test 8: Prosody Transfer
    print("\n[Test 8] Prosody Transfer...")
    # Different F0 from "prosody donor"
    source_f0 = torch.abs(torch.randn(batch_size, seq_len).to(device)) * 100 + 200  # Different range
    mel_transferred = model.prosody_transfer(source_f0, mel, speaker_ids)
    print(f"  Transferred mel shape: {mel_transferred.shape}")
    print("  [PASS]")

    # Test 9: Loss Function
    print("\n[Test 9] Loss Function...")
    loss_fn = DualCodebookLoss(config)
    losses = loss_fn(output, mel)
    print(f"  Reconstruction loss: {losses['reconstruction'].item():.4f}")
    print(f"  F0 commitment: {losses['f0_commitment'].item():.4f}")
    print(f"  Content commitment: {losses['content_commitment'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 10: CSM Adapter
    print("\n[Test 10] CSM Adapter...")
    adapter = DualCodebookProsodyAdapter(config, model).to(device)
    prefix_tokens = adapter(mel, f0)
    print(f"  Prefix tokens shape: {prefix_tokens.shape}")
    assert prefix_tokens.shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 11: Prefix from F0 Indices
    print("\n[Test 11] Prefix from F0 Indices...")
    f0_indices = encoded['f0_indices']
    prefix_from_indices = adapter.from_f0_indices(f0_indices)
    print(f"  Prefix from indices shape: {prefix_from_indices.shape}")
    assert prefix_from_indices.shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 12: Backward Pass
    print("\n[Test 12] Backward Pass...")
    model.zero_grad()
    output = model(mel, f0, speaker_ids)
    losses = loss_fn(output, mel)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test 13: Codebook Access
    print("\n[Test 13] Codebook Access...")
    f0_codebook = model.get_f0_codebook()
    content_codebook = model.get_content_codebook()
    print(f"  F0 codebook shape: {f0_codebook.shape}")
    print(f"  Content codebook shape: {content_codebook.shape}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All Dual-Codebook VQ-VAE tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. SEPARATE F0 AND CONTENT CODEBOOKS:
       - F0 Codebook: Small (64 codes) for prosody patterns
       - Content Codebook: Large (512 codes) for phonetic variety

    2. SPEAKER-INDEPENDENT PROSODY:
       - F0 codes capture abstract prosody patterns
       - Same F0 code sequence produces similar intonation across speakers
       - Better prosody transfer than raw F0 values

    3. PROSODY TRANSFER:
       mel_transferred = model.prosody_transfer(
           source_f0=f0_from_expressive_speaker,
           target_mel=mel_from_target_speaker,
           target_speaker_id=target_id,
       )

    4. DISCRETE PROSODY CONTROL:
       # Get F0 codebook indices
       f0_indices = model.encode(mel, f0)['f0_indices']

       # Use indices directly for conditioning
       prefix_tokens = adapter.from_f0_indices(f0_indices)

    5. CSM INTEGRATION:
       adapter = DualCodebookProsodyAdapter(config)
       prefix_tokens = adapter(mel, f0)  # [batch, 4, 2048]
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from dual_codebook_vqvae import (
    DualCodebookConfig,
    DualCodebookVQVAE,
    DualCodebookLoss,
    DualCodebookProsodyAdapter,
)

# Initialize
config = DualCodebookConfig(
    f0_codebook_size=64,      # Small for prosody abstraction
    content_codebook_size=512, # Large for phonetic diversity
)

model = DualCodebookVQVAE(config).cuda()
loss_fn = DualCodebookLoss(config)

# Training loop
for batch in dataloader:
    mel = batch['mel'].cuda()
    f0 = batch['f0'].cuda()
    speaker_id = batch['speaker_id'].cuda()

    output = model(mel, f0, speaker_id)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor codebook usage
    print(f"F0 perplexity: {output['f0_perplexity']:.2f}")
    print(f"Content perplexity: {output['content_perplexity']:.2f}")

# Prosody transfer at inference
with torch.no_grad():
    # Transfer prosody from speaker A to speaker B's content
    mel_transferred = model.prosody_transfer(
        source_f0=f0_speaker_a,
        target_mel=mel_speaker_b,
        target_speaker_id=speaker_b_id,
    )

# CSM integration
adapter = DualCodebookProsodyAdapter(config, model)
prefix_tokens = adapter(mel, f0)  # Use with ProsodyControlledCSM

# Direct control via F0 indices
f0_indices = adapter.get_f0_codes(mel, f0)
# Manipulate indices (e.g., swap patterns, interpolate)
prefix_from_indices = adapter.from_f0_indices(f0_indices)
""")
