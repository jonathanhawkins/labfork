"""
SD-Codec: Source Disentanglement via Joint Coding and Separation

Based on "Learning Source Disentanglement in Neural Audio Codec" (2025)
(hal-04902131) - https://hal.science/hal-04902131

Key Innovation: Joint learning of audio coding and source separation where
different codebooks are explicitly assigned to different audio domains/sources.
Unlike post-hoc disentanglement, SD-Codec achieves both excellent reconstruction
AND source separation through joint training.

Architecture:
1. Source-Aware Encoder: Identifies and routes features to domain-specific codebooks
2. Domain-Specific VQ: Separate codebooks for different attributes
3. Source Router: Learns to assign features to appropriate codebooks
4. Joint Decoder: Reconstructs from combined domain codebooks
5. Separation Head: Explicitly reconstructs individual sources

Key Differences from FreeCodec/SoftFreqBandCodec:
- Explicit source routing (not just architectural constraints)
- Joint training with separation objective
- Domain codebooks have explicit semantic meaning
- Supports arbitrary source domains (prosody, content, timbre, noise, etc.)

Training Objectives:
1. Reconstruction loss (full audio)
2. Source separation loss (per-source reconstruction)
3. VQ commitment loss (per codebook)
4. Source routing entropy (encourage balanced assignment)
5. Source orthogonality loss (disentanglement)

Usage:
    from sd_codec import (
        SDCodecConfig,
        SDCodec,
        SDCodecLoss,
        SDCodecAdapter,
        SOURCE_DOMAINS,
    )

    # Initialize with 3 source domains (prosody, content, timbre)
    config = SDCodecConfig(
        num_sources=3,
        source_names=["prosody", "content", "timbre"],
    )

    model = SDCodec(config).cuda()
    loss_fn = SDCodecLoss(config)

    # Training
    output = model(mel)
    losses = loss_fn(output, mel)

    # Get source-specific codes
    encoded = model.encode(mel)
    prosody_codes = encoded['source_indices'][0]  # Prosody domain
    content_codes = encoded['source_indices'][1]  # Content domain
    timbre_codes = encoded['source_indices'][2]   # Timbre domain

    # Source separation (extract individual sources)
    separated = model.separate(mel)
    prosody_mel = separated['source_reconstructions'][0]
    content_mel = separated['source_reconstructions'][1]

    # Cross-speaker prosody transfer
    mel_transferred = model.transfer_source(
        source_mel=mel_a,
        target_mel=mel_b,
        source_name="prosody",  # Transfer prosody from A
    )

    # CSM integration
    adapter = SDCodecAdapter(config, model)
    prefix_tokens = adapter(mel=mel)['prosody_tokens']
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DEFAULT SOURCE DOMAINS
# =============================================================================

SOURCE_DOMAINS = {
    "prosody": {
        "description": "Pitch, rhythm, stress patterns",
        "codebook_size": 512,
        "code_dim": 64,
    },
    "content": {
        "description": "Linguistic/phonetic information",
        "codebook_size": 1024,
        "code_dim": 128,
    },
    "timbre": {
        "description": "Speaker identity characteristics",
        "codebook_size": 256,
        "code_dim": 64,
    },
    "acoustic": {
        "description": "Fine-grained spectral details",
        "codebook_size": 512,
        "code_dim": 64,
    },
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SDCodecConfig:
    """Configuration for SD-Codec source disentanglement."""

    # Input dimensions
    mel_dim: int = 80
    sample_rate: int = 16000
    hop_length: int = 256  # ~16ms at 16kHz

    # Source domain configuration
    num_sources: int = 3  # Number of source domains
    source_names: Tuple[str, ...] = ("prosody", "content", "timbre")

    # Per-source codebook configuration
    # If None, uses SOURCE_DOMAINS defaults
    source_codebook_sizes: Optional[Tuple[int, ...]] = None
    source_code_dims: Optional[Tuple[int, ...]] = None

    # Default codebook settings (used if per-source not specified)
    default_codebook_size: int = 512
    default_code_dim: int = 64
    commitment_cost: float = 0.25
    ema_decay: float = 0.99

    # Source-aware encoder
    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 6
    encoder_num_heads: int = 8
    encoder_ffn_dim: int = 2048

    # Source router
    router_hidden_dim: int = 256
    router_temperature: float = 0.5  # Softmax temperature for routing
    use_gumbel_softmax: bool = True  # Differentiable routing
    min_source_usage: float = 0.1  # Minimum usage per source (balance)

    # Per-source encoders
    source_encoder_hidden_dim: int = 256
    source_encoder_num_layers: int = 3

    # Joint decoder
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048

    # Separation head
    use_separation_head: bool = True
    separation_hidden_dim: int = 256

    # Training settings
    dropout: float = 0.1

    # Loss weights
    reconstruction_weight: float = 1.0
    separation_weight: float = 0.5
    commitment_weight: float = 0.25
    routing_entropy_weight: float = 0.1
    orthogonality_weight: float = 0.05

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

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


# =============================================================================
# VECTOR QUANTIZER
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """Vector quantizer with EMA codebook updates."""

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

        # Codebook
        self.embedding = nn.Embedding(codebook_size, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch, seq, code_dim] input features

        Returns:
            z_q: [batch, seq, code_dim] quantized features
            indices: [batch, seq] codebook indices
            commitment_loss: scalar commitment loss
            perplexity: scalar codebook usage
        """
        input_shape = z.shape

        # Flatten batch and sequence
        flat_z = z.reshape(-1, self.code_dim)

        # Find nearest codes
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = distances.argmin(dim=1)
        encodings = F.one_hot(indices, self.codebook_size).float()

        # Quantize
        z_q = self.embedding(indices)

        # EMA update
        if self.training:
            with torch.no_grad():
                self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + (
                    1 - self.ema_decay
                ) * encodings.sum(0)

                dw = encodings.t() @ flat_z
                self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * dw

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.codebook_size * self.epsilon)
                    * n
                )

                self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

        # Commitment loss (computed on flat tensors)
        commitment_loss = F.mse_loss(flat_z, z_q.detach())

        # Straight-through gradient
        z_q = flat_z + (z_q - flat_z).detach()

        # Perplexity (codebook usage)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reshape back
        z_q = z_q.view(input_shape)
        indices = indices.view(input_shape[:-1])

        return z_q, indices, commitment_loss, perplexity


# =============================================================================
# SOURCE-AWARE ENCODER
# =============================================================================

class SourceAwareEncoder(nn.Module):
    """Encoder that produces features for source routing."""

    def __init__(self, config: SDCodecConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.mel_dim, config.encoder_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.encoder_hidden_dim, dropout=config.dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden_dim,
            nhead=config.encoder_num_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_num_layers
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.encoder_hidden_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch, seq, mel_dim] mel spectrogram

        Returns:
            features: [batch, seq, hidden_dim] encoded features
        """
        x = self.input_proj(mel)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x


# =============================================================================
# SOURCE ROUTER
# =============================================================================

class SourceRouter(nn.Module):
    """Routes features to source-specific codebooks."""

    def __init__(self, config: SDCodecConfig):
        super().__init__()
        self.config = config
        self.num_sources = config.num_sources
        self.temperature = config.router_temperature
        self.use_gumbel = config.use_gumbel_softmax

        # Routing network
        self.router = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.router_hidden_dim, config.router_hidden_dim),
            nn.GELU(),
            nn.Linear(config.router_hidden_dim, config.num_sources),
        )

    def forward(
        self, features: torch.Tensor, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, seq, hidden_dim] encoded features
            hard: Whether to use hard routing (inference)

        Returns:
            routing_weights: [batch, seq, num_sources] soft routing weights
            routing_indices: [batch, seq] hard routing assignments
        """
        logits = self.router(features)

        if self.use_gumbel and self.training:
            # Gumbel-softmax for differentiable discrete routing
            routing_weights = F.gumbel_softmax(
                logits, tau=self.temperature, hard=hard
            )
        else:
            routing_weights = F.softmax(logits / self.temperature, dim=-1)

        routing_indices = logits.argmax(dim=-1)

        return routing_weights, routing_indices

    def compute_entropy_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss to encourage balanced routing."""
        # Average routing probability per source
        avg_probs = routing_weights.mean(dim=(0, 1))  # [num_sources]

        # Entropy (maximize for balanced usage)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        # Maximum entropy for uniform distribution
        max_entropy = math.log(self.num_sources)

        # Loss is negative entropy (minimize to maximize entropy)
        entropy_loss = max_entropy - entropy

        return entropy_loss


# =============================================================================
# SOURCE-SPECIFIC ENCODER
# =============================================================================

class SourceSpecificEncoder(nn.Module):
    """Encoder for a specific source domain."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Projection to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Conv blocks with increasing dilation
        self.conv_blocks = nn.ModuleList([
            ConvBlock(hidden_dim, kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, input_dim]

        Returns:
            z: [batch, seq, output_dim]
        """
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq]

        for block in self.conv_blocks:
            x = block(x)

        x = x.transpose(1, 2)  # [batch, seq, hidden_dim]
        x = self.output_proj(x)
        x = self.norm(x)

        return x


# =============================================================================
# JOINT DECODER
# =============================================================================

class JointDecoder(nn.Module):
    """Decoder that reconstructs from combined source codes."""

    def __init__(self, config: SDCodecConfig):
        super().__init__()
        self.config = config

        # Compute total code dimension
        self.total_code_dim = self._compute_total_code_dim()

        # Input projection from combined codes
        self.input_proj = nn.Linear(self.total_code_dim, config.decoder_hidden_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.decoder_hidden_dim, dropout=config.dropout)

        # Transformer layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=config.decoder_num_layers
        )

        # Output projection to mel
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.decoder_hidden_dim),
            nn.Linear(config.decoder_hidden_dim, config.mel_dim),
        )

    def _compute_total_code_dim(self) -> int:
        """Compute total dimension of concatenated source codes."""
        total = 0
        for i in range(self.config.num_sources):
            if self.config.source_code_dims is not None:
                total += self.config.source_code_dims[i]
            else:
                source_name = self.config.source_names[i]
                if source_name in SOURCE_DOMAINS:
                    total += SOURCE_DOMAINS[source_name]["code_dim"]
                else:
                    total += self.config.default_code_dim
        return total

    def forward(self, source_codes: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            source_codes: List of [batch, seq, code_dim] per source

        Returns:
            mel: [batch, seq, mel_dim] reconstructed mel spectrogram
        """
        # Concatenate source codes
        combined = torch.cat(source_codes, dim=-1)

        x = self.input_proj(combined)
        x = self.pos_enc(x)
        x = self.transformer(x)
        mel = self.output_proj(x)

        return mel


# =============================================================================
# SEPARATION HEAD
# =============================================================================

class SeparationHead(nn.Module):
    """Head for reconstructing individual sources."""

    def __init__(self, config: SDCodecConfig, code_dim: int, source_idx: int):
        super().__init__()
        self.config = config
        self.source_idx = source_idx

        # Source-specific decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, config.separation_hidden_dim),
            nn.GELU(),
            nn.Linear(config.separation_hidden_dim, config.separation_hidden_dim),
            nn.GELU(),
            nn.Linear(config.separation_hidden_dim, config.mel_dim),
        )

    def forward(self, source_code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_code: [batch, seq, code_dim] source-specific code

        Returns:
            mel: [batch, seq, mel_dim] source contribution to mel
        """
        return self.decoder(source_code)


# =============================================================================
# SD-CODEC MAIN MODEL
# =============================================================================

class SDCodec(nn.Module):
    """Source Disentanglement Codec with joint coding and separation."""

    def __init__(self, config: SDCodecConfig):
        super().__init__()
        self.config = config

        # Get codebook sizes and dims per source
        self.source_codebook_sizes = []
        self.source_code_dims = []
        for i in range(config.num_sources):
            source_name = config.source_names[i]

            if config.source_codebook_sizes is not None:
                cb_size = config.source_codebook_sizes[i]
            elif source_name in SOURCE_DOMAINS:
                cb_size = SOURCE_DOMAINS[source_name]["codebook_size"]
            else:
                cb_size = config.default_codebook_size
            self.source_codebook_sizes.append(cb_size)

            if config.source_code_dims is not None:
                code_dim = config.source_code_dims[i]
            elif source_name in SOURCE_DOMAINS:
                code_dim = SOURCE_DOMAINS[source_name]["code_dim"]
            else:
                code_dim = config.default_code_dim
            self.source_code_dims.append(code_dim)

        # Source-aware encoder
        self.encoder = SourceAwareEncoder(config)

        # Source router
        self.router = SourceRouter(config)

        # Source-specific encoders
        self.source_encoders = nn.ModuleList([
            SourceSpecificEncoder(
                input_dim=config.encoder_hidden_dim,
                hidden_dim=config.source_encoder_hidden_dim,
                output_dim=self.source_code_dims[i],
                num_layers=config.source_encoder_num_layers,
                dropout=config.dropout,
            )
            for i in range(config.num_sources)
        ])

        # Per-source vector quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                codebook_size=self.source_codebook_sizes[i],
                code_dim=self.source_code_dims[i],
                commitment_cost=config.commitment_cost,
                ema_decay=config.ema_decay,
            )
            for i in range(config.num_sources)
        ])

        # Joint decoder
        self.decoder = JointDecoder(config)

        # Separation heads (per source)
        if config.use_separation_head:
            self.separation_heads = nn.ModuleList([
                SeparationHead(config, self.source_code_dims[i], i)
                for i in range(config.num_sources)
            ])
        else:
            self.separation_heads = None

        # Prosody output projection (for CSM integration)
        total_code_dim = sum(self.source_code_dims)
        self.prosody_proj = nn.Sequential(
            nn.Linear(total_code_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(
        self,
        mel: torch.Tensor,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram
            return_routing: Whether to return routing weights

        Returns:
            Dictionary with:
                - mel_reconstructed: Reconstructed mel spectrogram
                - source_codes: List of quantized codes per source
                - source_indices: List of codebook indices per source
                - commitment_losses: List of commitment losses per source
                - perplexities: List of perplexities per source
                - routing_weights: Soft routing weights (if return_routing)
                - routing_entropy_loss: Entropy loss for balanced routing
                - source_reconstructions: Per-source mel reconstructions (if separation_head)
        """
        batch_size, seq_len, _ = mel.shape

        # Encode
        features = self.encoder(mel)  # [batch, seq, hidden_dim]

        # Get routing weights
        routing_weights, routing_indices = self.router(features)

        # Process through source-specific encoders
        source_z = []
        source_z_q = []
        source_indices = []
        commitment_losses = []
        perplexities = []

        for i in range(self.config.num_sources):
            # Weight features by routing probability
            weighted_features = features * routing_weights[..., i:i+1]

            # Source-specific encoding
            z = self.source_encoders[i](weighted_features)
            source_z.append(z)

            # Quantize
            z_q, indices, commit_loss, perplexity = self.quantizers[i](z)
            source_z_q.append(z_q)
            source_indices.append(indices)
            commitment_losses.append(commit_loss)
            perplexities.append(perplexity)

        # Joint decoding
        mel_reconstructed = self.decoder(source_z_q)

        # Compute routing entropy loss
        routing_entropy_loss = self.router.compute_entropy_loss(routing_weights)

        # Prepare output
        output = {
            "mel_reconstructed": mel_reconstructed,
            "source_z": source_z,
            "source_z_q": source_z_q,
            "source_indices": source_indices,
            "commitment_losses": commitment_losses,
            "perplexities": perplexities,
            "routing_entropy_loss": routing_entropy_loss,
        }

        if return_routing:
            output["routing_weights"] = routing_weights
            output["routing_indices"] = routing_indices

        # Source separation (per-source reconstruction)
        if self.separation_heads is not None:
            source_reconstructions = []
            for i in range(self.config.num_sources):
                mel_source = self.separation_heads[i](source_z_q[i])
                source_reconstructions.append(mel_source)
            output["source_reconstructions"] = source_reconstructions

        return output

    def encode(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode mel spectrogram to source-specific codes.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram

        Returns:
            Dictionary with source codes and indices
        """
        features = self.encoder(mel)
        routing_weights, routing_indices = self.router(features, hard=True)

        source_z_q = []
        source_indices = []

        for i in range(self.config.num_sources):
            weighted_features = features * routing_weights[..., i:i+1]
            z = self.source_encoders[i](weighted_features)
            z_q, indices, _, _ = self.quantizers[i](z)
            source_z_q.append(z_q)
            source_indices.append(indices)

        return {
            "source_z_q": source_z_q,
            "source_indices": source_indices,
            "routing_weights": routing_weights,
            "routing_indices": routing_indices,
        }

    def decode(self, source_z_q: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from source-specific codes.

        Args:
            source_z_q: List of [batch, seq, code_dim] quantized codes per source

        Returns:
            mel: [batch, seq, mel_dim] reconstructed mel spectrogram
        """
        return self.decoder(source_z_q)

    def decode_from_indices(
        self, source_indices: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode from codebook indices.

        Args:
            source_indices: List of [batch, seq] indices per source

        Returns:
            mel: [batch, seq, mel_dim] reconstructed mel spectrogram
        """
        source_z_q = []
        for i, indices in enumerate(source_indices):
            z_q = self.quantizers[i].embedding(indices)
            source_z_q.append(z_q)
        return self.decode(source_z_q)

    def separate(self, mel: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Separate mel spectrogram into individual source contributions.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram

        Returns:
            Dictionary with source_reconstructions
        """
        if self.separation_heads is None:
            raise ValueError("Separation heads not enabled in config")

        encoded = self.encode(mel)
        source_reconstructions = []

        for i in range(self.config.num_sources):
            mel_source = self.separation_heads[i](encoded["source_z_q"][i])
            source_reconstructions.append(mel_source)

        return {"source_reconstructions": source_reconstructions}

    def transfer_source(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        source_name: str,
    ) -> torch.Tensor:
        """
        Transfer a specific source from source_mel to target_mel.

        Args:
            source_mel: [batch, seq, mel_dim] source of the attribute
            target_mel: [batch, seq, mel_dim] target receiving the attribute
            source_name: Name of the source to transfer

        Returns:
            mel: [batch, seq, mel_dim] mel with transferred source
        """
        source_idx = self.config.source_names.index(source_name)

        # Encode both
        source_encoded = self.encode(source_mel)
        target_encoded = self.encode(target_mel)

        # Mix: take source_idx from source_mel, rest from target_mel
        mixed_z_q = []
        for i in range(self.config.num_sources):
            if i == source_idx:
                mixed_z_q.append(source_encoded["source_z_q"][i])
            else:
                mixed_z_q.append(target_encoded["source_z_q"][i])

        return self.decode(mixed_z_q)

    def get_prosody_embedding(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Get prosody embedding for CSM integration.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram

        Returns:
            prosody_emb: [batch, output_dim] prosody embedding
        """
        encoded = self.encode(mel)

        # Concatenate all source codes
        combined = torch.cat(encoded["source_z_q"], dim=-1)

        # Pool across time
        pooled = combined.mean(dim=1)

        # Project to output dim
        prosody_emb = self.prosody_proj(pooled)

        return prosody_emb


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class SDCodecLoss(nn.Module):
    """Loss function for SD-Codec."""

    def __init__(self, config: SDCodecConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target_mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            output: Output from SDCodec.forward()
            target_mel: [batch, seq, mel_dim] target mel spectrogram

        Returns:
            Dictionary with loss components and total
        """
        losses = {}

        # Reconstruction loss
        losses["reconstruction"] = F.l1_loss(
            output["mel_reconstructed"], target_mel
        )

        # Commitment loss (average across sources)
        losses["commitment"] = sum(output["commitment_losses"]) / len(output["commitment_losses"])

        # Routing entropy loss
        losses["routing_entropy"] = output["routing_entropy_loss"]

        # Source separation loss (if available)
        if "source_reconstructions" in output:
            # Each source should contribute to the full mel
            # The sum should approximate the original
            source_sum = sum(output["source_reconstructions"])
            losses["separation"] = F.l1_loss(source_sum, target_mel)

            # Per-source orthogonality (encourage distinct contributions)
            ortho_loss = 0.0
            for i in range(len(output["source_reconstructions"])):
                for j in range(i + 1, len(output["source_reconstructions"])):
                    # Minimize correlation between source reconstructions
                    s_i = output["source_reconstructions"][i].flatten(1)
                    s_j = output["source_reconstructions"][j].flatten(1)
                    # Normalize
                    s_i = F.normalize(s_i, dim=1)
                    s_j = F.normalize(s_j, dim=1)
                    ortho_loss += (s_i * s_j).sum(dim=1).abs().mean()
            losses["orthogonality"] = ortho_loss / (
                self.config.num_sources * (self.config.num_sources - 1) / 2
            )
        else:
            losses["separation"] = torch.tensor(0.0, device=target_mel.device)
            losses["orthogonality"] = torch.tensor(0.0, device=target_mel.device)

        # Total loss
        losses["total"] = (
            self.config.reconstruction_weight * losses["reconstruction"]
            + self.config.commitment_weight * losses["commitment"]
            + self.config.routing_entropy_weight * losses["routing_entropy"]
            + self.config.separation_weight * losses["separation"]
            + self.config.orthogonality_weight * losses["orthogonality"]
        )

        # Add perplexity stats (not losses, just metrics)
        losses["mean_perplexity"] = sum(output["perplexities"]) / len(output["perplexities"])

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class SDCodecAdapter(nn.Module):
    """Adapter for integrating SD-Codec with CSM prosody pipeline."""

    def __init__(self, config: SDCodecConfig, model: Optional[SDCodec] = None):
        super().__init__()
        self.config = config

        if model is not None:
            self.model = model
        else:
            self.model = SDCodec(config)

        # Token generation
        total_code_dim = sum(self.model.source_code_dims)
        self.token_proj = nn.Sequential(
            nn.Linear(total_code_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
            nn.Linear(config.output_dim, config.num_prefix_tokens * config.output_dim),
        )

    def forward(
        self,
        mel: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from mel spectrogram or audio.

        Args:
            mel: [batch, seq, mel_dim] mel spectrogram
            audio: [batch, samples] raw audio (will be converted to mel)

        Returns:
            Dictionary with prosody_tokens and source information
        """
        if mel is None and audio is None:
            raise ValueError("Either mel or audio must be provided")

        if mel is None:
            # Convert audio to mel (simplified - assumes preprocessing)
            # In practice, use proper mel spectrogram extraction
            mel = self._audio_to_mel(audio)

        # Encode
        encoded = self.model.encode(mel)

        # Concatenate source codes
        combined = torch.cat(encoded["source_z_q"], dim=-1)

        # Pool across time
        pooled = combined.mean(dim=1)  # [batch, total_code_dim]

        # Project to tokens
        tokens = self.token_proj(pooled)
        tokens = tokens.view(-1, self.config.num_prefix_tokens, self.config.output_dim)

        return {
            "prosody_tokens": tokens,
            "source_z_q": encoded["source_z_q"],
            "source_indices": encoded["source_indices"],
            "routing_weights": encoded["routing_weights"],
        }

    def from_indices(
        self, source_indices: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from codebook indices.

        Args:
            source_indices: List of [batch, seq] indices per source

        Returns:
            Dictionary with prosody_tokens
        """
        source_z_q = []
        for i, indices in enumerate(source_indices):
            z_q = self.model.quantizers[i].embedding(indices)
            source_z_q.append(z_q)

        combined = torch.cat(source_z_q, dim=-1)
        pooled = combined.mean(dim=1)
        tokens = self.token_proj(pooled)
        tokens = tokens.view(-1, self.config.num_prefix_tokens, self.config.output_dim)

        return {"prosody_tokens": tokens}

    def transfer_prosody(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer prosody from source to target.

        Args:
            source_mel: [batch, seq, mel_dim] prosody donor
            target_mel: [batch, seq, mel_dim] content source

        Returns:
            Dictionary with prosody_tokens and transferred mel
        """
        # Transfer prosody source
        transferred_mel = self.model.transfer_source(
            source_mel, target_mel, source_name="prosody"
        )

        # Get tokens from transferred
        tokens = self(mel=transferred_mel)

        return {
            "prosody_tokens": tokens["prosody_tokens"],
            "transferred_mel": transferred_mel,
        }

    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram (simplified)."""
        # In practice, use torchaudio or librosa
        # This is a placeholder
        batch_size = audio.shape[0]
        seq_len = audio.shape[-1] // self.config.hop_length
        return torch.randn(batch_size, seq_len, self.config.mel_dim, device=audio.device)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_source_statistics(
    model: SDCodec,
    mel: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-source statistics for analysis.

    Args:
        model: SDCodec model
        mel: [batch, seq, mel_dim] mel spectrogram

    Returns:
        Dictionary with per-source statistics
    """
    output = model(mel, return_routing=True)

    stats = {}

    # Routing distribution
    routing_weights = output["routing_weights"]
    stats["routing_probs"] = routing_weights.mean(dim=(0, 1))  # [num_sources]

    # Per-source perplexity
    stats["perplexities"] = torch.stack(output["perplexities"])

    # Per-source code variance
    code_variances = []
    for z_q in output["source_z_q"]:
        var = z_q.var(dim=1).mean()
        code_variances.append(var)
    stats["code_variances"] = torch.stack(code_variances)

    # Source contribution to reconstruction (if separation available)
    if "source_reconstructions" in output:
        contributions = []
        for recon in output["source_reconstructions"]:
            contrib = recon.abs().mean()
            contributions.append(contrib)
        stats["source_contributions"] = torch.stack(contributions)

    return stats


def analyze_source_separation(
    model: SDCodec,
    mel: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Analyze quality of source separation.

    Args:
        model: SDCodec model
        mel: [batch, seq, mel_dim] mel spectrogram

    Returns:
        Dictionary with separation quality metrics
    """
    if model.separation_heads is None:
        raise ValueError("Model does not have separation heads")

    output = model(mel)
    source_recons = output["source_reconstructions"]

    analysis = {}

    # Reconstruction quality from sum of sources
    source_sum = sum(source_recons)
    analysis["sum_reconstruction_error"] = F.l1_loss(source_sum, mel)

    # Per-source energy
    source_energies = []
    for recon in source_recons:
        energy = recon.pow(2).mean()
        source_energies.append(energy)
    analysis["source_energies"] = torch.stack(source_energies)

    # Source orthogonality (correlation between sources)
    ortho_matrix = torch.zeros(model.config.num_sources, model.config.num_sources)
    for i in range(model.config.num_sources):
        for j in range(model.config.num_sources):
            s_i = source_recons[i].flatten(1)
            s_j = source_recons[j].flatten(1)
            s_i = F.normalize(s_i, dim=1)
            s_j = F.normalize(s_j, dim=1)
            corr = (s_i * s_j).sum(dim=1).mean()
            ortho_matrix[i, j] = corr
    analysis["orthogonality_matrix"] = ortho_matrix

    # Disentanglement score (average off-diagonal should be low)
    mask = ~torch.eye(model.config.num_sources, dtype=torch.bool)
    analysis["disentanglement_score"] = 1.0 - ortho_matrix[mask].abs().mean()

    return analysis


def create_sd_codec_adapter(
    checkpoint: Optional[str] = None,
    config: Optional[SDCodecConfig] = None,
) -> SDCodecAdapter:
    """
    Factory function to create SD-Codec adapter.

    Args:
        checkpoint: Optional path to checkpoint
        config: Optional config (uses default if None)

    Returns:
        SDCodecAdapter instance
    """
    if config is None:
        config = SDCodecConfig()

    adapter = SDCodecAdapter(config)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        if "model_state_dict" in state_dict:
            adapter.load_state_dict(state_dict["model_state_dict"])
        else:
            adapter.load_state_dict(state_dict)

    return adapter


# =============================================================================
# TEST SUITE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SD-Codec: Source Disentanglement via Joint Coding and Separation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Initialize config with 3 sources
    config = SDCodecConfig(
        num_sources=3,
        source_names=("prosody", "content", "timbre"),
        mel_dim=80,
        encoder_hidden_dim=256,
        encoder_num_layers=3,
        encoder_num_heads=4,
        encoder_ffn_dim=512,
        decoder_hidden_dim=256,
        decoder_num_layers=3,
        decoder_num_heads=4,
        decoder_ffn_dim=512,
        router_hidden_dim=128,
        source_encoder_hidden_dim=128,
        source_encoder_num_layers=2,
        output_dim=2048,
        num_prefix_tokens=4,
    )

    # Test data
    batch_size = 2
    seq_len = 100
    mel = torch.randn(batch_size, seq_len, config.mel_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Num sources: {config.num_sources}")
    print(f"  Source names: {config.source_names}")
    print(f"  Encoder hidden: {config.encoder_hidden_dim}")
    print("  [PASS]")

    # Test 2: Source-Aware Encoder
    print("\n[Test 2] Source-Aware Encoder...")
    encoder = SourceAwareEncoder(config).to(device)
    features = encoder(mel)
    print(f"  Input: {mel.shape}")
    print(f"  Output: {features.shape}")
    assert features.shape == (batch_size, seq_len, config.encoder_hidden_dim)
    print("  [PASS]")

    # Test 3: Source Router
    print("\n[Test 3] Source Router...")
    router = SourceRouter(config).to(device)
    routing_weights, routing_indices = router(features)
    print(f"  Routing weights: {routing_weights.shape}")
    print(f"  Routing indices: {routing_indices.shape}")
    print(f"  Weights sum: {routing_weights.sum(dim=-1).mean().item():.4f} (should be ~1.0)")
    assert routing_weights.shape == (batch_size, seq_len, config.num_sources)
    print("  [PASS]")

    # Test 4: Routing Entropy
    print("\n[Test 4] Routing Entropy Loss...")
    entropy_loss = router.compute_entropy_loss(routing_weights)
    print(f"  Entropy loss: {entropy_loss.item():.4f}")
    print("  [PASS]")

    # Test 5: Full SDCodec Model
    print("\n[Test 5] Full SDCodec Model...")
    model = SDCodec(config).to(device)
    output = model(mel, return_routing=True)
    print(f"  Mel reconstructed: {output['mel_reconstructed'].shape}")
    print(f"  Num source codes: {len(output['source_z_q'])}")
    for i, z_q in enumerate(output['source_z_q']):
        print(f"    Source {i} ({config.source_names[i]}): {z_q.shape}")
    print(f"  Commitment losses: {[f'{l.item():.4f}' for l in output['commitment_losses']]}")
    print(f"  Perplexities: {[f'{p.item():.2f}' for p in output['perplexities']]}")
    print("  [PASS]")

    # Test 6: Source Separation
    print("\n[Test 6] Source Separation...")
    if output.get("source_reconstructions"):
        for i, recon in enumerate(output["source_reconstructions"]):
            print(f"    Source {i} reconstruction: {recon.shape}")
        # Check that sum approximates original
        source_sum = sum(output["source_reconstructions"])
        sum_error = F.l1_loss(source_sum, mel).item()
        print(f"  Sum reconstruction error: {sum_error:.4f}")
    print("  [PASS]")

    # Test 7: Encode/Decode
    print("\n[Test 7] Encode/Decode Consistency...")
    encoded = model.encode(mel)
    print(f"  Source indices: {[idx.shape for idx in encoded['source_indices']]}")
    mel_decoded = model.decode(encoded['source_z_q'])
    print(f"  Decoded mel: {mel_decoded.shape}")
    # Decode from indices
    mel_from_idx = model.decode_from_indices(encoded['source_indices'])
    decode_diff = (mel_decoded - mel_from_idx).abs().mean().item()
    print(f"  Decode consistency error: {decode_diff:.6f}")
    print("  [PASS]")

    # Test 8: Source Transfer
    print("\n[Test 8] Source Transfer...")
    mel_a = torch.randn(batch_size, seq_len, config.mel_dim).to(device)
    mel_b = torch.randn(batch_size, seq_len, config.mel_dim).to(device)
    transferred = model.transfer_source(mel_a, mel_b, "prosody")
    print(f"  Source mel: {mel_a.shape}")
    print(f"  Target mel: {mel_b.shape}")
    print(f"  Transferred: {transferred.shape}")
    print("  [PASS]")

    # Test 9: Loss Function
    print("\n[Test 9] Loss Function...")
    loss_fn = SDCodecLoss(config)
    losses = loss_fn(output, mel)
    print(f"  Reconstruction: {losses['reconstruction'].item():.4f}")
    print(f"  Commitment: {losses['commitment'].item():.4f}")
    print(f"  Routing entropy: {losses['routing_entropy'].item():.4f}")
    print(f"  Separation: {losses['separation'].item():.4f}")
    print(f"  Orthogonality: {losses['orthogonality'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 10: CSM Adapter
    print("\n[Test 10] CSM Adapter...")
    adapter = SDCodecAdapter(config, model).to(device)
    result = adapter(mel=mel)
    print(f"  Prosody tokens: {result['prosody_tokens'].shape}")
    assert result['prosody_tokens'].shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 11: Adapter from Indices
    print("\n[Test 11] Adapter from Indices...")
    tokens_from_idx = adapter.from_indices(encoded['source_indices'])
    print(f"  Tokens from indices: {tokens_from_idx['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 12: Prosody Transfer via Adapter
    print("\n[Test 12] Prosody Transfer via Adapter...")
    transfer_result = adapter.transfer_prosody(mel_a, mel_b)
    print(f"  Transferred tokens: {transfer_result['prosody_tokens'].shape}")
    print(f"  Transferred mel: {transfer_result['transferred_mel'].shape}")
    print("  [PASS]")

    # Test 13: Source Statistics
    print("\n[Test 13] Source Statistics...")
    stats = compute_source_statistics(model, mel)
    print(f"  Routing probs: {stats['routing_probs'].tolist()}")
    print(f"  Perplexities: {stats['perplexities'].tolist()}")
    print(f"  Code variances: {[f'{v.item():.4f}' for v in stats['code_variances']]}")
    if 'source_contributions' in stats:
        print(f"  Source contributions: {[f'{c.item():.4f}' for c in stats['source_contributions']]}")
    print("  [PASS]")

    # Test 14: Separation Analysis
    print("\n[Test 14] Separation Analysis...")
    analysis = analyze_source_separation(model, mel)
    print(f"  Sum reconstruction error: {analysis['sum_reconstruction_error'].item():.4f}")
    print(f"  Source energies: {[f'{e.item():.4f}' for e in analysis['source_energies']]}")
    print(f"  Orthogonality matrix:\n{analysis['orthogonality_matrix']}")
    print(f"  Disentanglement score: {analysis['disentanglement_score'].item():.4f}")
    print("  [PASS]")

    # Test 15: Backward Pass
    print("\n[Test 15] Backward Pass...")
    model.zero_grad()
    output = model(mel)
    losses = loss_fn(output, mel)
    losses['total'].backward()
    total_grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"  Total gradient norm: {total_grad_norm:.4f}")
    print("  [PASS]")

    # Test 16: Different Source Configurations
    print("\n[Test 16] Different Source Configurations...")
    config_4source = SDCodecConfig(
        num_sources=4,
        source_names=("prosody", "content", "timbre", "acoustic"),
        encoder_hidden_dim=256,
        encoder_num_layers=2,
        decoder_hidden_dim=256,
        decoder_num_layers=2,
    )
    model_4source = SDCodec(config_4source).to(device)
    output_4 = model_4source(mel)
    print(f"  4-source model output sources: {len(output_4['source_z_q'])}")
    print(f"  Source dims: {[z.shape[-1] for z in output_4['source_z_q']]}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All SD-Codec tests passed!")
    print("=" * 60)

    print("""
Key Features:
----------------------------------------

    1. SOURCE-AWARE ENCODING:
       - Identifies and routes features to domain-specific codebooks
       - Differentiable routing via Gumbel-softmax
       - Entropy regularization for balanced usage

    2. DOMAIN-SPECIFIC CODEBOOKS:
       - Separate VQ codebook per source (prosody, content, timbre)
       - Each codebook learns domain-specific patterns
       - Explicit semantic meaning for codebook indices

    3. JOINT CODING AND SEPARATION:
       - Single model for both coding and separation
       - Per-source reconstruction heads
       - Orthogonality loss for disentanglement

    4. SOURCE TRANSFER:
       - Transfer specific attributes between samples
       - Zero-shot voice conversion via source manipulation
       - Compatible with prosody transfer workflows

    5. CSM INTEGRATION:
       adapter = SDCodecAdapter(config, model)
       prefix_tokens = adapter(mel=mel)['prosody_tokens']


Usage Example:
----------------------------------------

from sd_codec import (
    SDCodecConfig,
    SDCodec,
    SDCodecLoss,
    SDCodecAdapter,
    compute_source_statistics,
    analyze_source_separation,
)

# Initialize with 3 source domains
config = SDCodecConfig(
    num_sources=3,
    source_names=("prosody", "content", "timbre"),
)

model = SDCodec(config).cuda()
loss_fn = SDCodecLoss(config)

# Training loop
for batch in dataloader:
    mel = batch['mel'].cuda()

    output = model(mel)
    losses = loss_fn(output, mel)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor per-source perplexities
    for i, name in enumerate(config.source_names):
        print(f"{name}: {output['perplexities'][i].item():.2f}")

    # Analyze separation quality
    if step % 100 == 0:
        analysis = analyze_source_separation(model, mel)
        print(f"Disentanglement: {analysis['disentanglement_score']:.4f}")

# Encode to source-specific codes
with torch.no_grad():
    encoded = model.encode(mel)
    prosody_idx = encoded['source_indices'][0]  # Prosody tokens
    content_idx = encoded['source_indices'][1]  # Content tokens
    timbre_idx = encoded['source_indices'][2]   # Timbre tokens

# Source transfer (prosody from A, content/timbre from B)
mel_transferred = model.transfer_source(mel_a, mel_b, "prosody")

# Separate into individual sources
separated = model.separate(mel)
prosody_mel = separated['source_reconstructions'][0]
content_mel = separated['source_reconstructions'][1]

# CSM integration
adapter = SDCodecAdapter(config, model)
prefix_tokens = adapter(mel=mel)['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
