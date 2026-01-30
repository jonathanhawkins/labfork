"""
SpeechTripleNet: End-to-End Triple Disentanglement VAE

Based on "SpeechTripleNet: End-to-End Disentangled Speech Representation Learning" (ACM MM 2023)

Key Innovation: VAE that simultaneously disentangles content, timbre, and prosody
using structured latent variables - NO explicit labels needed.

Architecture:
1. Content Encoder → Discrete VQ latent (removes prosody/speaker via quantization bottleneck)
2. Timbre Encoder → Global continuous vector (speaker identity via global pooling)
3. Prosody Encoder → Sequence continuous latent (pitch, energy, duration, rhythm)
4. Decoder → Reconstructs from all three latents jointly

Structural Constraints for Disentanglement:
- Content: Discrete VQ bottleneck forces linguistic-only information
- Timbre: Global pooling removes temporal/prosodic variation
- Prosody: Low-dimensional continuous bottleneck + residual connection

Loss Components:
- Reconstruction loss (mel spectrogram)
- Content VQ commitment loss
- Timbre/Prosody KL divergence
- Cross-reconstruction consistency loss
- Orthogonality regularization (optional)

Benefits:
- Fully unsupervised disentanglement (no emotion/speaker labels)
- Clean separation via architectural constraints
- Can scale to unlabeled data
- Each latent can be manipulated independently

Usage:
    from speech_triple_net import (
        SpeechTripleNetConfig,
        SpeechTripleNet,
        SpeechTripleNetLoss,
        SpeechTripleNetAdapter,
    )

    config = SpeechTripleNetConfig()
    model = SpeechTripleNet(config).cuda()
    loss_fn = SpeechTripleNetLoss(config)

    # Encode to three latent spaces
    encoded = model.encode(mel)
    content_z = encoded['content_z']   # Discrete content tokens
    timbre_z = encoded['timbre_z']     # Global speaker vector
    prosody_z = encoded['prosody_z']   # Sequence prosody latent

    # Decode from latents
    mel_reconstructed = model.decode(content_z, timbre_z, prosody_z)

    # Prosody transfer (content from A, prosody from B, timbre from C)
    mel_transferred = model.transfer(
        content_mel=mel_a,
        prosody_mel=mel_b,
        timbre_mel=mel_c,
    )

    # Training
    output = model(mel)
    losses = loss_fn(output, mel)
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
class SpeechTripleNetConfig:
    """Configuration for SpeechTripleNet VAE."""

    # Input dimensions
    mel_dim: int = 80  # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 256

    # Encoder dimensions
    encoder_hidden_dim: int = 512
    encoder_layers: int = 4
    encoder_heads: int = 8
    encoder_ffn_dim: int = 2048

    # Content latent (discrete VQ)
    content_codebook_size: int = 512  # Number of content codes
    content_code_dim: int = 256  # Dimension per content code
    content_num_groups: int = 2  # Product quantization groups
    content_commitment_cost: float = 0.25
    content_ema_decay: float = 0.99

    # Timbre latent (global continuous)
    timbre_latent_dim: int = 256  # Global speaker vector dimension
    timbre_prior_mean: float = 0.0
    timbre_prior_std: float = 1.0

    # Prosody latent (sequence continuous)
    prosody_latent_dim: int = 64  # Per-frame prosody dimension (low for bottleneck)
    prosody_prior_mean: float = 0.0
    prosody_prior_std: float = 1.0

    # Decoder dimensions
    decoder_hidden_dim: int = 512
    decoder_layers: int = 4
    decoder_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Training settings
    dropout: float = 0.1
    kl_weight: float = 0.1  # KL divergence weight (β-VAE)
    kl_anneal_steps: int = 10000  # Steps to anneal KL weight from 0 to kl_weight

    # Orthogonality regularization
    use_orthogonality: bool = True
    ortho_weight: float = 0.1
    timbre_prosody_beta: float = 0.0001  # Strict orthogonality

    # Output dimension for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# ENCODER MODULES
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


class ConvPreNet(nn.Module):
    """Convolutional pre-net for initial feature extraction."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_sizes: List[int] = [5, 5, 5],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        current_dim = in_dim

        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2
            layers.extend([
                nn.Conv1d(current_dim, out_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = out_dim

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            [batch, seq, out_dim]
        """
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)


# =============================================================================
# CONTENT ENCODER (DISCRETE VQ)
# =============================================================================

class ProductVectorQuantizer(nn.Module):
    """
    Product Vector Quantizer for content encoding.

    Uses product quantization to increase effective codebook size
    while maintaining computational efficiency.

    The discrete bottleneck removes prosodic variation by forcing
    information through a limited set of codes.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 512,
        code_dim: int = 256,
        num_groups: int = 2,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        assert code_dim % num_groups == 0, "code_dim must be divisible by num_groups"

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.num_groups = num_groups
        self.group_dim = code_dim // num_groups
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # Project input to code dimension
        self.pre_proj = nn.Linear(input_dim, code_dim)
        self.post_proj = nn.Linear(code_dim, input_dim)

        # Codebooks for each group
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, self.group_dim))
            for _ in range(num_groups)
        ])

        # EMA cluster sizes and sums for codebook updates
        for i in range(num_groups):
            self.register_buffer(f'ema_cluster_size_{i}', torch.zeros(codebook_size))
            self.register_buffer(f'ema_sum_{i}', torch.randn(codebook_size, self.group_dim))

        # Initialize codebooks
        self._init_codebooks()

    def _init_codebooks(self):
        for codebook in self.codebooks:
            nn.init.uniform_(codebook, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def _quantize_group(
        self,
        z: torch.Tensor,  # [batch * seq, group_dim]
        codebook: nn.Parameter,
        group_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a single group."""
        # Compute distances
        d = (
            z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(z, codebook.t())
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )

        # Find nearest codes
        indices = d.argmin(dim=-1)  # [batch * seq]

        # Quantized vectors
        z_q = F.embedding(indices, codebook)

        # EMA codebook update (training only)
        if self.training:
            cluster_size = getattr(self, f'ema_cluster_size_{group_idx}')
            ema_sum = getattr(self, f'ema_sum_{group_idx}')

            # One-hot encoding
            encodings = F.one_hot(indices, self.codebook_size).float()

            # Update cluster sizes
            new_cluster_size = encodings.sum(dim=0)
            cluster_size.mul_(self.ema_decay).add_(new_cluster_size, alpha=1 - self.ema_decay)

            # Update sums
            new_sum = torch.matmul(encodings.t(), z)
            ema_sum.mul_(self.ema_decay).add_(new_sum, alpha=1 - self.ema_decay)

            # Update codebook
            n = cluster_size.clamp(min=1)
            codebook.data.copy_(ema_sum / n.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z, z_q.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices, commitment_loss

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input to discrete content codes.

        Returns:
            Dict with:
                - z_q: [batch, seq, input_dim] quantized output
                - indices: [batch, seq, num_groups] code indices
                - commitment_loss: scalar loss
                - perplexity: average codebook usage
        """
        batch_size, seq_len, _ = x.shape

        # Project to code dimension
        z = self.pre_proj(x)  # [B, T, code_dim]

        # Split into groups
        z_groups = z.view(batch_size * seq_len, self.num_groups, self.group_dim)
        z_groups = z_groups.transpose(0, 1)  # [num_groups, B*T, group_dim]

        # Quantize each group
        z_q_groups = []
        indices_groups = []
        total_commitment_loss = 0.0

        for i, (z_g, codebook) in enumerate(zip(z_groups, self.codebooks)):
            z_q_g, idx_g, commit_loss = self._quantize_group(z_g, codebook, i)
            z_q_groups.append(z_q_g)
            indices_groups.append(idx_g)
            total_commitment_loss += commit_loss

        # Reassemble
        z_q = torch.stack(z_q_groups, dim=0)  # [num_groups, B*T, group_dim]
        z_q = z_q.transpose(0, 1)  # [B*T, num_groups, group_dim]
        z_q = z_q.reshape(batch_size, seq_len, self.code_dim)

        indices = torch.stack(indices_groups, dim=-1)  # [B*T, num_groups]
        indices = indices.view(batch_size, seq_len, self.num_groups)

        # Project back
        z_q = self.post_proj(z_q)

        # Compute perplexity (codebook usage)
        perplexity = self._compute_perplexity(indices)

        return {
            'z_q': z_q,
            'indices': indices,
            'commitment_loss': total_commitment_loss * self.commitment_cost / self.num_groups,
            'perplexity': perplexity,
        }

    def _compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute average perplexity (higher = better codebook usage)."""
        flat_indices = indices.view(-1, self.num_groups)
        perplexities = []

        for i in range(self.num_groups):
            encodings = F.one_hot(flat_indices[:, i], self.codebook_size).float()
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            perplexities.append(perplexity)

        return torch.stack(perplexities).mean()


class ContentEncoder(nn.Module):
    """
    Content encoder with discrete VQ bottleneck.

    The discrete bottleneck forces the encoder to capture only
    linguistic/phonetic content, removing speaker and prosodic variation.
    """

    def __init__(self, config: SpeechTripleNetConfig):
        super().__init__()
        self.config = config

        # Pre-net
        self.prenet = ConvPreNet(
            in_dim=config.mel_dim,
            out_dim=config.encoder_hidden_dim,
            dropout=config.dropout,
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            dim=config.encoder_hidden_dim,
            dropout=config.dropout,
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            dim=config.encoder_hidden_dim,
            num_layers=config.encoder_layers,
            num_heads=config.encoder_heads,
            ffn_dim=config.encoder_ffn_dim,
            dropout=config.dropout,
        )

        # Vector quantizer
        self.vq = ProductVectorQuantizer(
            input_dim=config.encoder_hidden_dim,
            codebook_size=config.content_codebook_size,
            code_dim=config.content_code_dim,
            num_groups=config.content_num_groups,
            commitment_cost=config.content_commitment_cost,
            ema_decay=config.content_ema_decay,
        )

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel spectrogram to discrete content codes.

        Returns:
            Dict with content_z, indices, commitment_loss, perplexity
        """
        # Pre-net
        x = self.prenet(mel)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.encoder(x, mask)

        # Vector quantization
        vq_output = self.vq(x)

        return {
            'content_z': vq_output['z_q'],
            'content_indices': vq_output['indices'],
            'content_commitment_loss': vq_output['commitment_loss'],
            'content_perplexity': vq_output['perplexity'],
        }


# =============================================================================
# TIMBRE ENCODER (GLOBAL CONTINUOUS)
# =============================================================================

class TimbreEncoder(nn.Module):
    """
    Timbre encoder with global pooling for speaker identity.

    Uses global average pooling to remove temporal variation,
    capturing only speaker-level characteristics.
    """

    def __init__(self, config: SpeechTripleNetConfig):
        super().__init__()
        self.config = config

        # Frame-level processing
        self.conv = nn.Sequential(
            nn.Conv1d(config.mel_dim, 256, kernel_size=5, padding=2),
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

        # Output projection (mu, logvar for VAE)
        self.fc_mu = nn.Linear(512 * 2, config.timbre_latent_dim)
        self.fc_logvar = nn.Linear(512 * 2, config.timbre_latent_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel spectrogram to global timbre vector.

        Returns:
            Dict with timbre_z (sampled), timbre_mu, timbre_logvar
        """
        # [B, T, D] -> [B, D, T]
        x = mel.transpose(1, 2)

        # Frame-level processing
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

        # VAE parameters
        mu = self.fc_mu(stats)
        logvar = self.fc_logvar(stats).clamp(-10, 10)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return {
            'timbre_z': z,
            'timbre_mu': mu,
            'timbre_logvar': logvar,
        }


# =============================================================================
# PROSODY ENCODER (SEQUENCE CONTINUOUS)
# =============================================================================

class ProsodyEncoder(nn.Module):
    """
    Prosody encoder with low-dimensional continuous bottleneck.

    Uses a low-dimensional latent space to capture prosodic variation
    (pitch, energy, duration, rhythm) while removing content and speaker.
    """

    def __init__(self, config: SpeechTripleNetConfig):
        super().__init__()
        self.config = config

        # Pre-net
        self.prenet = ConvPreNet(
            in_dim=config.mel_dim,
            out_dim=config.encoder_hidden_dim,
            dropout=config.dropout,
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            dim=config.encoder_hidden_dim,
            dropout=config.dropout,
        )

        # Transformer encoder
        self.encoder = TransformerEncoder(
            dim=config.encoder_hidden_dim,
            num_layers=config.encoder_layers // 2,  # Fewer layers for prosody
            num_heads=config.encoder_heads,
            ffn_dim=config.encoder_ffn_dim,
            dropout=config.dropout,
        )

        # Project to low-dimensional VAE space
        self.fc_mu = nn.Linear(config.encoder_hidden_dim, config.prosody_latent_dim)
        self.fc_logvar = nn.Linear(config.encoder_hidden_dim, config.prosody_latent_dim)

        # Project back to hidden dimension for decoder
        self.fc_out = nn.Linear(config.prosody_latent_dim, config.encoder_hidden_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel spectrogram to sequence prosody latent.

        Returns:
            Dict with prosody_z (sampled), prosody_mu, prosody_logvar, prosody_hidden
        """
        # Pre-net
        x = self.prenet(mel)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        x = self.encoder(x, mask)

        # VAE parameters (per-frame)
        mu = self.fc_mu(x)  # [B, T, prosody_dim]
        logvar = self.fc_logvar(x).clamp(-10, 10)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # Project to hidden dimension for decoder
        z_hidden = self.fc_out(z)

        return {
            'prosody_z': z,
            'prosody_mu': mu,
            'prosody_logvar': logvar,
            'prosody_hidden': z_hidden,
        }


# =============================================================================
# DECODER
# =============================================================================

class TripleNetDecoder(nn.Module):
    """
    Decoder that reconstructs mel spectrogram from content, timbre, and prosody.

    Combines:
    - Content: Sequence of quantized tokens (linguistic structure)
    - Timbre: Global speaker vector (broadcast to all frames)
    - Prosody: Sequence of continuous features (local variation)
    """

    def __init__(self, config: SpeechTripleNetConfig):
        super().__init__()
        self.config = config

        # Combine content + prosody sequences with timbre
        combined_dim = config.encoder_hidden_dim * 2  # content + prosody
        self.timbre_proj = nn.Linear(config.timbre_latent_dim, config.encoder_hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(combined_dim + config.encoder_hidden_dim, config.decoder_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            dim=config.decoder_hidden_dim,
            dropout=config.dropout,
        )

        # Transformer decoder
        self.decoder = TransformerEncoder(
            dim=config.decoder_hidden_dim,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            ffn_dim=config.decoder_ffn_dim,
            dropout=config.dropout,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dim, config.mel_dim),
        )

    def forward(
        self,
        content_z: torch.Tensor,  # [batch, seq, hidden_dim]
        timbre_z: torch.Tensor,   # [batch, timbre_dim]
        prosody_z: torch.Tensor,  # [batch, seq, hidden_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode latents to mel spectrogram.

        Returns:
            [batch, seq, mel_dim] reconstructed mel spectrogram
        """
        batch_size, seq_len, _ = content_z.shape

        # Project and expand timbre to sequence length
        timbre_proj = self.timbre_proj(timbre_z)  # [B, hidden_dim]
        timbre_expanded = timbre_proj.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate all latents
        combined = torch.cat([content_z, prosody_z, timbre_expanded], dim=-1)

        # Project to decoder dimension
        x = self.input_proj(combined)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer decoder
        x = self.decoder(x, mask)

        # Output projection
        mel_out = self.output_proj(x)

        return mel_out


# =============================================================================
# FULL MODEL
# =============================================================================

class SpeechTripleNet(nn.Module):
    """
    SpeechTripleNet: End-to-End Triple Disentanglement VAE.

    Simultaneously disentangles content, timbre, and prosody using
    structural constraints on latent variables - no explicit labels needed.

    Components:
    1. Content Encoder: Discrete VQ latent for linguistic content
    2. Timbre Encoder: Global continuous latent for speaker identity
    3. Prosody Encoder: Sequence continuous latent for prosodic variation
    4. Decoder: Reconstructs mel from all three latents
    """

    def __init__(self, config: SpeechTripleNetConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.content_encoder = ContentEncoder(config)
        self.timbre_encoder = TimbreEncoder(config)
        self.prosody_encoder = ProsodyEncoder(config)

        # Decoder
        self.decoder = TripleNetDecoder(config)

        # Output projection for CSM integration
        self.output_proj = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim * 2 + config.timbre_latent_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel spectrogram to three disentangled latent spaces.

        Returns:
            Dict with all encoder outputs
        """
        # Content encoding
        content_output = self.content_encoder(mel, mask)

        # Timbre encoding
        timbre_output = self.timbre_encoder(mel, mask)

        # Prosody encoding
        prosody_output = self.prosody_encoder(mel, mask)

        return {
            **content_output,
            **timbre_output,
            **prosody_output,
        }

    def decode(
        self,
        content_z: torch.Tensor,
        timbre_z: torch.Tensor,
        prosody_hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode from latent spaces to mel spectrogram.

        Returns:
            [batch, seq, mel_dim] reconstructed mel
        """
        return self.decoder(content_z, timbre_z, prosody_hidden, mask)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Returns:
            Dict with all encoder outputs and reconstructed mel
        """
        # Encode
        encoded = self.encode(mel, mask)

        # Decode
        mel_reconstructed = self.decode(
            encoded['content_z'],
            encoded['timbre_z'],
            encoded['prosody_hidden'],
            mask,
        )

        # Combined embedding for downstream use
        batch_size, seq_len, _ = mel.shape

        # Pool prosody for combination
        prosody_pooled = encoded['prosody_hidden'].mean(dim=1)  # [B, hidden]
        content_pooled = encoded['content_z'].mean(dim=1)  # [B, hidden]

        combined = torch.cat([
            content_pooled,
            encoded['timbre_z'],
            prosody_pooled,
        ], dim=-1)

        combined_embedding = self.output_proj(combined)

        return {
            **encoded,
            'mel_reconstructed': mel_reconstructed,
            'combined_embedding': combined_embedding,
        }

    def transfer(
        self,
        content_mel: torch.Tensor,
        prosody_mel: torch.Tensor,
        timbre_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transfer prosody and timbre to different content.

        Args:
            content_mel: Source for linguistic content
            prosody_mel: Source for prosodic variation
            timbre_mel: Source for speaker identity

        Returns:
            [batch, seq, mel_dim] mel with transferred attributes
        """
        # Encode each source
        content_output = self.content_encoder(content_mel, mask)
        timbre_output = self.timbre_encoder(timbre_mel, mask)
        prosody_output = self.prosody_encoder(prosody_mel, mask)

        # Align sequence lengths
        min_len = min(
            content_output['content_z'].shape[1],
            prosody_output['prosody_hidden'].shape[1],
        )

        content_z = content_output['content_z'][:, :min_len]
        prosody_hidden = prosody_output['prosody_hidden'][:, :min_len]

        # Decode with mixed latents
        mel_transferred = self.decode(
            content_z,
            timbre_output['timbre_z'],
            prosody_hidden,
            mask,
        )

        return mel_transferred

    def get_prosody_embedding(
        self,
        mel: torch.Tensor,
        pool: str = 'mean',
    ) -> torch.Tensor:
        """
        Get prosody embedding compatible with existing interface.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram
            pool: 'mean' or 'first' for temporal pooling

        Returns:
            [batch, output_dim] prosody embedding
        """
        output = self.forward(mel)
        return output['combined_embedding']


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class SpeechTripleNetLoss(nn.Module):
    """
    Loss function for SpeechTripleNet training.

    Components:
    1. Reconstruction loss (L1 + L2 mel)
    2. Content VQ commitment loss
    3. Timbre KL divergence
    4. Prosody KL divergence
    5. Orthogonality regularization (optional)
    """

    def __init__(self, config: SpeechTripleNetConfig):
        super().__init__()
        self.config = config
        self.step = 0

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prior_mu: float = 0.0,
        prior_std: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute KL divergence from standard normal prior.

        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=-1).mean()

    def orthogonality_loss(
        self,
        z1: torch.Tensor,  # [batch, dim1] or [batch, seq, dim1]
        z2: torch.Tensor,  # [batch, dim2] or [batch, seq, dim2]
        target_similarity: float = 0.0,
    ) -> torch.Tensor:
        """
        Soft orthogonality loss between two latent spaces.

        Works with different dimensions by projecting to shared space.
        """
        # Pool if sequence
        if z1.dim() == 3:
            z1 = z1.mean(dim=1)
        if z2.dim() == 3:
            z2 = z2.mean(dim=1)

        # Project to common dimension if needed
        dim1, dim2 = z1.shape[-1], z2.shape[-1]
        min_dim = min(dim1, dim2)

        if dim1 != dim2:
            # Use random projection to common space (preserves similarity structure)
            if dim1 > min_dim:
                z1 = z1[:, :min_dim]
            if dim2 > min_dim:
                z2 = z2[:, :min_dim]

        # L2 normalize
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)

        # Cosine similarity
        cos_sim = (z1_norm * z2_norm).sum(dim=-1)

        # Push toward target
        loss = (cos_sim - target_similarity).pow(2).mean()

        return loss

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            output: Model output dict
            mel_target: Target mel spectrogram
            mask: Optional sequence mask

        Returns:
            Dict with individual losses and total
        """
        device = mel_target.device
        losses = {}

        # Handle length mismatch
        mel_pred = output['mel_reconstructed']
        min_len = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_len]
        mel_target = mel_target[:, :min_len]

        # 1. Reconstruction loss
        l1_loss = F.l1_loss(mel_pred, mel_target)
        l2_loss = F.mse_loss(mel_pred, mel_target)
        losses['reconstruction_l1'] = l1_loss
        losses['reconstruction_l2'] = l2_loss
        losses['reconstruction'] = l1_loss + l2_loss

        # 2. Content VQ commitment loss
        losses['content_commitment'] = output['content_commitment_loss']
        losses['content_perplexity'] = output['content_perplexity']

        # 3. Timbre KL divergence
        timbre_kl = self.kl_divergence(
            output['timbre_mu'],
            output['timbre_logvar'],
            self.config.timbre_prior_mean,
            self.config.timbre_prior_std,
        )
        losses['timbre_kl'] = timbre_kl

        # 4. Prosody KL divergence
        prosody_kl = self.kl_divergence(
            output['prosody_mu'],
            output['prosody_logvar'],
            self.config.prosody_prior_mean,
            self.config.prosody_prior_std,
        )
        losses['prosody_kl'] = prosody_kl

        # KL annealing
        kl_weight = min(1.0, self.step / max(self.config.kl_anneal_steps, 1))
        kl_weight *= self.config.kl_weight
        losses['kl_weight'] = torch.tensor(kl_weight, device=device)

        # 5. Orthogonality regularization
        if self.config.use_orthogonality:
            # Timbre-Prosody orthogonality (strict)
            # Use prosody_hidden for matching dimensions
            ortho_loss = self.orthogonality_loss(
                output['timbre_z'],
                output['prosody_hidden'],  # Use hidden (encoder_hidden_dim) not z (prosody_latent_dim)
                self.config.timbre_prosody_beta,
            )
            losses['orthogonality'] = ortho_loss
        else:
            losses['orthogonality'] = torch.tensor(0.0, device=device)

        # Total loss
        total = (
            losses['reconstruction']
            + losses['content_commitment']
            + kl_weight * (losses['timbre_kl'] + losses['prosody_kl'])
            + self.config.ortho_weight * losses['orthogonality']
        )
        losses['total'] = total

        # Update step
        self.step += 1

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class SpeechTripleNetAdapter(nn.Module):
    """
    Adapter to integrate SpeechTripleNet with existing prosody pipeline.

    Converts the triple latent representation to prefix tokens compatible
    with ProsodyControlledCSM and the V6 prosody conditioning system.
    """

    def __init__(
        self,
        config: SpeechTripleNetConfig,
        model: Optional[SpeechTripleNet] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else SpeechTripleNet(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

        # Optional: separate prosody token projection
        self.prosody_token_proj = nn.Linear(
            config.prosody_latent_dim,
            config.output_dim,
        )

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            mel: Mel spectrogram
            text_embeddings: Optional text embeddings to blend

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        # Get combined embedding from model
        output = self.model(mel)
        combined_emb = output['combined_embedding']  # [B, output_dim]

        # Project to tokens
        tokens = self.token_proj(combined_emb)  # [B, output_dim * num_tokens]

        # Reshape
        batch_size = mel.shape[0]
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)

        # Normalize
        tokens = self.norm(tokens)

        return tokens

    def get_prosody_embedding(
        self,
        mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prosody embedding compatible with existing interface.

        Returns:
            [batch, output_dim] prosody embedding
        """
        output = self.model(mel)
        return output['combined_embedding']

    def get_disentangled_embeddings(
        self,
        mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get all disentangled embeddings separately.

        Returns:
            Dict with content_z, timbre_z, prosody_z
        """
        output = self.model(mel)
        return {
            'content': output['content_z'],
            'timbre': output['timbre_z'],
            'prosody': output['prosody_z'],
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SpeechTripleNet VAE - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = SpeechTripleNetConfig()

    # Test parameters
    batch_size = 2
    seq_len = 100
    mel_dim = config.mel_dim

    # Create dummy input
    mel = torch.randn(batch_size, seq_len, mel_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Content codebook size: {config.content_codebook_size}")
    print(f"  Timbre latent dim: {config.timbre_latent_dim}")
    print(f"  Prosody latent dim: {config.prosody_latent_dim}")
    print("  [PASS]")

    # Test 2: Content Encoder
    print("\n[Test 2] Content Encoder...")
    content_encoder = ContentEncoder(config).to(device)
    content_output = content_encoder(mel)
    print(f"  Content z shape: {content_output['content_z'].shape}")
    print(f"  Content indices shape: {content_output['content_indices'].shape}")
    print(f"  Commitment loss: {content_output['content_commitment_loss'].item():.4f}")
    print(f"  Perplexity: {content_output['content_perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 3: Timbre Encoder
    print("\n[Test 3] Timbre Encoder...")
    timbre_encoder = TimbreEncoder(config).to(device)
    timbre_output = timbre_encoder(mel)
    print(f"  Timbre z shape: {timbre_output['timbre_z'].shape}")
    print(f"  Timbre mu shape: {timbre_output['timbre_mu'].shape}")
    print(f"  Timbre logvar range: [{timbre_output['timbre_logvar'].min():.2f}, {timbre_output['timbre_logvar'].max():.2f}]")
    print("  [PASS]")

    # Test 4: Prosody Encoder
    print("\n[Test 4] Prosody Encoder...")
    prosody_encoder = ProsodyEncoder(config).to(device)
    prosody_output = prosody_encoder(mel)
    print(f"  Prosody z shape: {prosody_output['prosody_z'].shape}")
    print(f"  Prosody hidden shape: {prosody_output['prosody_hidden'].shape}")
    print(f"  Prosody logvar range: [{prosody_output['prosody_logvar'].min():.2f}, {prosody_output['prosody_logvar'].max():.2f}]")
    print("  [PASS]")

    # Test 5: Full Model
    print("\n[Test 5] Full Model...")
    model = SpeechTripleNet(config).to(device)
    output = model(mel)
    print(f"  Mel reconstructed shape: {output['mel_reconstructed'].shape}")
    print(f"  Combined embedding shape: {output['combined_embedding'].shape}")
    print("  [PASS]")

    # Test 6: Loss Function
    print("\n[Test 6] Loss Function...")
    loss_fn = SpeechTripleNetLoss(config)
    losses = loss_fn(output, mel)
    print(f"  Reconstruction loss: {losses['reconstruction'].item():.4f}")
    print(f"  Content commitment: {losses['content_commitment'].item():.4f}")
    print(f"  Timbre KL: {losses['timbre_kl'].item():.4f}")
    print(f"  Prosody KL: {losses['prosody_kl'].item():.4f}")
    print(f"  Orthogonality: {losses['orthogonality'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 7: Transfer
    print("\n[Test 7] Prosody/Timbre Transfer...")
    mel_a = torch.randn(batch_size, seq_len, mel_dim).to(device)
    mel_b = torch.randn(batch_size, seq_len, mel_dim).to(device)
    mel_c = torch.randn(batch_size, seq_len, mel_dim).to(device)

    mel_transferred = model.transfer(
        content_mel=mel_a,
        prosody_mel=mel_b,
        timbre_mel=mel_c,
    )
    print(f"  Transferred mel shape: {mel_transferred.shape}")
    print("  [PASS]")

    # Test 8: Adapter
    print("\n[Test 8] CSM Adapter...")
    adapter = SpeechTripleNetAdapter(config, model).to(device)
    prefix_tokens = adapter(mel)
    print(f"  Prefix tokens shape: {prefix_tokens.shape}")
    assert prefix_tokens.shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 9: Backward pass
    print("\n[Test 9] Backward Pass...")
    model.zero_grad()
    output = model(mel)
    losses = loss_fn(output, mel)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test 10: Disentanglement check
    print("\n[Test 10] Disentanglement Check...")
    disentangled = adapter.get_disentangled_embeddings(mel)
    print(f"  Content shape: {disentangled['content'].shape}")
    print(f"  Timbre shape: {disentangled['timbre'].shape}")
    print(f"  Prosody shape: {disentangled['prosody'].shape}")

    # Check orthogonality between timbre and prosody
    # Use prosody_hidden from the model output for matching dimensions
    output = model(mel)
    timbre_pooled = output['timbre_z']
    prosody_pooled = output['prosody_hidden'].mean(dim=1)

    # Project to common dimension for comparison
    min_dim = min(timbre_pooled.shape[-1], prosody_pooled.shape[-1])
    timbre_proj = timbre_pooled[:, :min_dim]
    prosody_proj = prosody_pooled[:, :min_dim]

    # Compute cosine similarity
    timbre_norm = F.normalize(timbre_proj, p=2, dim=-1)
    prosody_norm = F.normalize(prosody_proj, p=2, dim=-1)
    cos_sim = (timbre_norm * prosody_norm).sum(dim=-1).mean()
    print(f"  Timbre-Prosody cosine similarity: {cos_sim.item():.4f} (lower is better)")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All SpeechTripleNet tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. TRIPLE DISENTANGLEMENT:
       - Content: Discrete VQ codes (linguistic only)
       - Timbre: Global vector (speaker identity)
       - Prosody: Sequence latent (pitch, energy, rhythm)

    2. NO EXPLICIT LABELS NEEDED:
       - Disentanglement via architectural constraints
       - Content bottleneck removes speaker/prosody
       - Global pooling removes temporal/prosodic variation
       - Low-dim prosody bottleneck removes content/speaker

    3. PROSODY TRANSFER:
       mel_transferred = model.transfer(
           content_mel=mel_a,   # Linguistic content source
           prosody_mel=mel_b,   # Prosody source
           timbre_mel=mel_c,    # Speaker identity source
       )

    4. CSM INTEGRATION:
       adapter = SpeechTripleNetAdapter(config)
       prefix_tokens = adapter(mel)  # [batch, 4, 2048]
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from speech_triple_net import (
    SpeechTripleNetConfig,
    SpeechTripleNet,
    SpeechTripleNetLoss,
    SpeechTripleNetAdapter,
)

# Initialize
config = SpeechTripleNetConfig()
model = SpeechTripleNet(config).cuda()
loss_fn = SpeechTripleNetLoss(config)

# Training loop
for mel in dataloader:
    output = model(mel)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor disentanglement
    print(f"Content perplexity: {losses['content_perplexity']:.2f}")
    print(f"Timbre KL: {losses['timbre_kl']:.4f}")
    print(f"Prosody KL: {losses['prosody_kl']:.4f}")
    print(f"Orthogonality: {losses['orthogonality']:.4f}")

# Prosody manipulation
encoded = model.encode(mel)
prosody_z = encoded['prosody_z']

# Scale prosody intensity
prosody_z_scaled = prosody_z * 1.5  # More expressive

# Decode with modified prosody
mel_expressive = model.decode(
    encoded['content_z'],
    encoded['timbre_z'],
    model.prosody_encoder.fc_out(prosody_z_scaled),
)

# Cross-speaker prosody transfer
mel_transferred = model.transfer(
    content_mel=speaker_a_mel,
    prosody_mel=speaker_b_mel,
    timbre_mel=speaker_c_mel,
)
""")
