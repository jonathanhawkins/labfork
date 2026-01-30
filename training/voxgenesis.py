"""
VoxGenesis: Latent Speaker Manifold for Interpretable Pitch/Emotion Control

Based on "VoxGenesis: Interpretable Voice Synthesis and Manipulation"
(March 2024) - arXiv:2403.00529

Key Innovation:
Transform Gaussian distribution → speech distribution conditioned on semantic tokens.
This forces learning of speaker distribution disentangled from semantic content.

Architecture:
- Mapping Network (M): 7-layer FF network (Style-GAN inspired) transforms
  isotropic Gaussian to non-isotropic speaker latent space
- Semantic Conditioned Transformation (T): Conditions latent on semantic tokens
- Shared Embedding (e): Processes semantic tokens for both G and D
- Deconvolution Network (f): Generates audio features from transformed latent

Generator: G(z|Y) = f(T(M(z), Y) + e(Y))

Interpretable Directions:
- Apply PCA/SVD to M(z) outputs to discover human-interpretable directions
- Leading PCs capture inter-speaker variations (gender, speaker identity)
- Later PCs capture intra-speaker nuances (emotion, pitch, tone)
- Voice editing: w' = M(z) + s · v_n (move along PC direction with scale s)

Benefits:
- Novel speaker generation by sampling from Gaussian
- Zero-shot voice editing via latent manipulation
- Interpretable control axes discovered unsupervised
- Clean separation of content vs speaker characteristics

Usage:
    from voxgenesis import (
        VoxGenesisConfig,
        VoxGenesis,
        VoxGenesisAdapter,
        LatentDirectionDiscovery,
    )

    config = VoxGenesisConfig()
    model = VoxGenesis(config).cuda()

    # Encode speaker to latent space
    z = model.encode_speaker(speaker_features)

    # Discover interpretable directions
    discovery = LatentDirectionDiscovery(model)
    directions = discovery.discover(speaker_dataset)

    # Voice editing (e.g., increase pitch)
    z_edited = z + 0.5 * directions['pitch']
    edited_audio = model.generate(z_edited, semantic_tokens)

    # Novel speaker generation
    z_novel = torch.randn(1, config.latent_dim)
    novel_audio = model.generate(z_novel, semantic_tokens)

    # CSM integration
    adapter = VoxGenesisAdapter(config, model)
    prefix_tokens = adapter(features)  # [batch, 4, 2048]
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VoxGenesisConfig:
    """Configuration for VoxGenesis latent speaker manifold."""

    # Input dimensions
    input_dim: int = 768  # HuBERT/WavLM feature dimension
    semantic_dim: int = 768  # Semantic token dimension (HuBERT)
    mel_dim: int = 80  # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 320  # 20ms at 16kHz

    # Latent space
    latent_dim: int = 512  # Speaker latent dimension (z)
    mapping_dim: int = 512  # Mapping network output dimension (w)
    num_mapping_layers: int = 7  # Style-GAN style mapping depth

    # Semantic conditioning
    semantic_hidden_dim: int = 512
    semantic_num_layers: int = 4
    semantic_num_heads: int = 8

    # Transformation network (T)
    transform_hidden_dim: int = 512
    transform_num_layers: int = 4

    # Deconvolution network (f)
    deconv_hidden_dim: int = 512
    deconv_num_layers: int = 8
    deconv_upsample_rates: List[int] = field(
        default_factory=lambda: [10, 4, 2, 2]
    )  # Total 160x upsampling for HuBERT hop

    # Discriminator
    disc_hidden_dim: int = 512
    disc_num_layers: int = 4

    # Training settings
    dropout: float = 0.1
    use_spectral_norm: bool = True

    # GAN training
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4
    gan_feature_matching_weight: float = 10.0
    gan_disc_weight: float = 1.0

    # Gaussian constraint (KL divergence to prior)
    kl_weight: float = 0.01
    kl_anneal_steps: int = 10000

    # Direction discovery
    num_directions: int = 32  # Number of PCA directions to extract
    direction_regularization: float = 0.01

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# MAPPING NETWORK (Style-GAN inspired)
# =============================================================================

class MappingNetwork(nn.Module):
    """
    7-layer Mapping Network (M) from Style-GAN.

    Transforms isotropic Gaussian z → non-isotropic w space.
    This enables more representative latent codes and captures
    speaker distribution structure.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        mapping_dim: int = 512,
        num_layers: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.mapping_dim = mapping_dim
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else mapping_dim
            layers.extend([
                nn.Linear(in_dim, mapping_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(mapping_dim)

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform latent z to mapping space w.

        Args:
            z: [batch, latent_dim] samples from N(0, I)

        Returns:
            w: [batch, mapping_dim] transformed latent codes
        """
        w = self.network(z)
        w = self.norm(w)
        return w


# =============================================================================
# SEMANTIC EMBEDDING
# =============================================================================

class SemanticEmbedding(nn.Module):
    """
    Shared Embedding Layer (e) for semantic tokens.

    Processes semantic tokens (from HuBERT) to provide content
    information to both generator and discriminator.
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.semantic_dim, config.semantic_hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            self._create_sinusoidal_positions(4096, config.semantic_hidden_dim),
            requires_grad=False
        )

        # Transformer layers for contextual embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.semantic_hidden_dim,
            nhead=config.semantic_num_heads,
            dim_feedforward=config.semantic_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.semantic_num_layers
        )

        self.norm = nn.LayerNorm(config.semantic_hidden_dim)

    def _create_sinusoidal_positions(self, max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(
        self,
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Embed semantic tokens.

        Returns:
            Dict with:
                - embedding: [batch, seq, hidden_dim] contextual embeddings
                - global_emb: [batch, hidden_dim] pooled global embedding
        """
        batch_size, seq_len = semantic_tokens.shape[:2]

        # Project input
        x = self.input_proj(semantic_tokens)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        # Global pooling (mean over sequence)
        if mask is not None:
            # Mask padded positions
            mask_expanded = ~mask.unsqueeze(-1)
            x_masked = x * mask_expanded.float()
            global_emb = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            global_emb = x.mean(dim=1)

        return {
            'embedding': x,
            'global_emb': global_emb,
        }


# =============================================================================
# SEMANTIC CONDITIONED TRANSFORMATION
# =============================================================================

class SemanticConditionedTransformation(nn.Module):
    """
    Semantic Conditioned Transformation (T).

    Conditionally transforms mapping codes based on semantic information.
    This enables semantic-specific speaker attribute modifications.

    T(M(z), Y) combines speaker latent with content conditioning.
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        # Mapping code projection
        self.w_proj = nn.Linear(config.mapping_dim, config.transform_hidden_dim)

        # Semantic conditioning projection
        self.semantic_proj = nn.Linear(config.semantic_hidden_dim, config.transform_hidden_dim)

        # Cross-attention for semantic conditioning
        self.cross_attn = nn.MultiheadAttention(
            config.transform_hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward transformation layers
        layers = []
        for i in range(config.transform_num_layers):
            layers.extend([
                nn.Linear(config.transform_hidden_dim, config.transform_hidden_dim),
                nn.LayerNorm(config.transform_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
        self.transform_layers = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(config.transform_hidden_dim, config.deconv_hidden_dim)

    def forward(
        self,
        w: torch.Tensor,  # [batch, mapping_dim] transformed latent
        semantic_emb: torch.Tensor,  # [batch, seq, semantic_hidden_dim]
    ) -> torch.Tensor:
        """
        Condition mapping code on semantic content.

        Args:
            w: Speaker mapping code from mapping network
            semantic_emb: Semantic embeddings from SemanticEmbedding

        Returns:
            [batch, seq, deconv_hidden_dim] conditioned features
        """
        batch_size, seq_len = semantic_emb.shape[:2]

        # Project mapping code and expand to sequence
        w_proj = self.w_proj(w)  # [batch, hidden]
        w_expanded = w_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, hidden]

        # Project semantic embeddings
        semantic_proj = self.semantic_proj(semantic_emb)  # [batch, seq, hidden]

        # Cross-attention: w attends to semantic
        query = w_expanded
        key = semantic_proj
        value = semantic_proj

        attended, _ = self.cross_attn(query, key, value)

        # Combine with additive connection
        combined = w_expanded + attended + semantic_proj

        # Transform through FF layers
        transformed = self.transform_layers(combined)

        # Output projection
        output = self.output_proj(transformed)

        return output


# =============================================================================
# DECONVOLUTION NETWORK
# =============================================================================

class ResidualBlock1D(nn.Module):
    """1D Residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

        if use_spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)

        self.block = nn.Sequential(
            conv1,
            nn.LeakyReLU(0.2, inplace=True),
            conv2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeconvolutionNetwork(nn.Module):
    """
    Deconvolution Network (f) for audio generation.

    Upsamples conditioned features to audio sample rate.
    Inspired by HiFi-GAN generator structure.
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Conv1d(config.deconv_hidden_dim, config.deconv_hidden_dim, 1)

        # Upsampling blocks
        self.upsamples = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        current_channels = config.deconv_hidden_dim
        for i, rate in enumerate(config.deconv_upsample_rates):
            out_channels = current_channels // 2 if i < len(config.deconv_upsample_rates) - 1 else config.deconv_hidden_dim // 4

            # Upsampling conv transpose
            conv_t = nn.ConvTranspose1d(
                current_channels, out_channels,
                kernel_size=rate * 2, stride=rate,
                padding=rate // 2,
            )
            if config.use_spectral_norm:
                conv_t = nn.utils.spectral_norm(conv_t)
            self.upsamples.append(conv_t)

            # Residual blocks with multi-scale dilations
            res_blocks = nn.ModuleList()
            for dilation in [1, 3, 5]:
                res_blocks.append(
                    ResidualBlock1D(out_channels, dilation=dilation, use_spectral_norm=config.use_spectral_norm)
                )
            self.residual_blocks.append(res_blocks)

            current_channels = out_channels

        # Output projection to mel spectrogram
        self.output_proj = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(current_channels, config.mel_dim, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate mel spectrogram from conditioned features.

        Args:
            x: [batch, seq, hidden_dim] conditioned features

        Returns:
            [batch, time, mel_dim] mel spectrogram
        """
        # [batch, seq, dim] -> [batch, dim, seq]
        x = x.transpose(1, 2)

        x = self.input_proj(x)

        for upsample, res_blocks in zip(self.upsamples, self.residual_blocks):
            x = F.leaky_relu(x, 0.2)
            x = upsample(x)

            # Apply residual blocks and sum
            xs = None
            for res_block in res_blocks:
                if xs is None:
                    xs = res_block(x)
                else:
                    xs = xs + res_block(x)
            x = xs / len(res_blocks)

        x = self.output_proj(x)

        # [batch, mel_dim, time] -> [batch, time, mel_dim]
        x = x.transpose(1, 2)

        return x


# =============================================================================
# DISCRIMINATOR
# =============================================================================

class VoxGenesisDiscriminator(nn.Module):
    """
    Discriminator for VoxGenesis GAN training.

    Receives semantic information to prevent generator from
    producing unintelligible speech.
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        # Mel input processing
        self.mel_conv = nn.Sequential(
            nn.Conv1d(config.mel_dim, config.disc_hidden_dim, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Semantic embedding (shared with generator)
        self.semantic_embed = SemanticEmbedding(config)

        # Fusion layer
        self.fusion = nn.Linear(
            config.disc_hidden_dim + config.semantic_hidden_dim,
            config.disc_hidden_dim
        )

        # Discriminator layers
        layers = []
        current_dim = config.disc_hidden_dim
        for i in range(config.disc_num_layers):
            stride = 2 if i < config.disc_num_layers - 1 else 1
            out_dim = min(current_dim * 2, 1024)

            conv = nn.Conv1d(current_dim, out_dim, kernel_size=4, stride=stride, padding=2)
            if config.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)

            layers.extend([
                conv,
                nn.LeakyReLU(0.2, inplace=True),
            ])
            current_dim = out_dim

        self.disc_layers = nn.ModuleList([nn.Sequential(layers[i:i+2]) for i in range(0, len(layers), 2)])

        # Output head
        self.output_conv = nn.Conv1d(current_dim, 1, kernel_size=3, padding=1)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, time, mel_dim]
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Discriminate real vs fake mel spectrograms.

        Args:
            mel: Mel spectrogram
            semantic_tokens: HuBERT semantic tokens
            return_features: Whether to return intermediate features

        Returns:
            Dict with 'logits' and optionally 'features' for feature matching
        """
        # Process mel
        mel_t = mel.transpose(1, 2)  # [batch, mel_dim, time]
        x = self.mel_conv(mel_t)

        # Get semantic embedding
        semantic_output = self.semantic_embed(semantic_tokens)
        semantic_global = semantic_output['global_emb']  # [batch, hidden]

        # Global average pool mel features for fusion
        mel_global = x.mean(dim=-1)  # [batch, hidden]

        # Fuse mel and semantic
        fused = torch.cat([mel_global, semantic_global], dim=-1)
        fused = self.fusion(fused)  # [batch, hidden]

        # Expand back to sequence
        fused = fused.unsqueeze(-1).expand(-1, -1, x.shape[-1])

        # Add fused semantic info to mel features
        x = x + fused

        # Discriminator layers
        features = []
        for layer in self.disc_layers:
            x = layer(x)
            if return_features:
                features.append(x)

        logits = self.output_conv(x).squeeze(1)  # [batch, time]

        result = {'logits': logits}
        if return_features:
            result['features'] = features

        return result


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class SpeakerEncoder(nn.Module):
    """
    Speaker Encoder for extracting speaker embeddings from audio.

    Maps input features to latent z space with Gaussian constraint.
    """

    def __init__(self, config: VoxGenesisConfig):
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

        # Output to mean and log variance
        self.fc_mu = nn.Linear(512 * 2, config.latent_dim)
        self.fc_logvar = nn.Linear(512 * 2, config.latent_dim)

        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode speaker to latent space.

        Returns:
            Dict with:
                - z: [batch, latent_dim] sampled latent (with reparameterization)
                - mu: [batch, latent_dim] mean
                - logvar: [batch, latent_dim] log variance
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

        # Output mean and log variance
        mu = self.fc_mu(stats)
        logvar = self.fc_logvar(stats)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        z = self.norm(z)

        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
        }


# =============================================================================
# LATENT DIRECTION DISCOVERY
# =============================================================================

class LatentDirectionDiscovery(nn.Module):
    """
    Discover interpretable directions in the latent speaker manifold via PCA.

    Leading principal components capture inter-speaker variations (gender, speaker identity).
    Later components capture intra-speaker nuances (emotion, pitch, tone).

    Voice editing: w' = M(z) + s · v_n (scale s along direction v_n)
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        # Store discovered directions as buffers
        self.register_buffer('directions', torch.zeros(config.num_directions, config.mapping_dim))
        self.register_buffer('eigenvalues', torch.zeros(config.num_directions))
        self.register_buffer('mean', torch.zeros(config.mapping_dim))
        self.register_buffer('discovered', torch.tensor(False))

        # Direction labels (will be populated after discovery)
        self.direction_labels: Dict[str, int] = {}

    def discover_from_samples(
        self,
        mapping_network: MappingNetwork,
        num_samples: int = 10000,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Discover directions by sampling from Gaussian and applying PCA to M(z).

        Args:
            mapping_network: Trained mapping network M
            num_samples: Number of samples for PCA

        Returns:
            Dict with directions, eigenvalues, explained_variance_ratio
        """
        if device is None:
            device = next(mapping_network.parameters()).device

        # Sample from Gaussian and transform through mapping network
        with torch.no_grad():
            z_samples = torch.randn(num_samples, self.config.latent_dim, device=device)
            w_samples = mapping_network(z_samples)  # [N, mapping_dim]

        # Compute mean
        mean = w_samples.mean(dim=0)
        w_centered = w_samples - mean

        # SVD for PCA
        U, S, Vt = torch.linalg.svd(w_centered, full_matrices=False)

        # Store top directions
        num_dirs = min(self.config.num_directions, Vt.shape[0])
        self.directions.copy_(Vt[:num_dirs])
        self.eigenvalues.copy_(S[:num_dirs])
        self.mean.copy_(mean)
        self.discovered.fill_(True)

        # Compute explained variance ratio
        total_var = (S ** 2).sum()
        explained_var_ratio = (S[:num_dirs] ** 2) / total_var

        return {
            'directions': self.directions.clone(),
            'eigenvalues': self.eigenvalues.clone(),
            'explained_variance_ratio': explained_var_ratio,
            'mean': mean,
        }

    def discover_from_dataset(
        self,
        mapping_network: MappingNetwork,
        speaker_encoder: SpeakerEncoder,
        dataloader,
        max_samples: int = 10000,
    ) -> Dict[str, torch.Tensor]:
        """
        Discover directions from actual speaker embeddings in dataset.

        Args:
            mapping_network: Trained mapping network M
            speaker_encoder: Trained speaker encoder
            dataloader: DataLoader with audio features
            max_samples: Maximum number of samples to use

        Returns:
            Dict with directions, eigenvalues, explained_variance_ratio
        """
        device = next(mapping_network.parameters()).device

        w_samples = []
        total_samples = 0

        mapping_network.eval()
        speaker_encoder.eval()

        with torch.no_grad():
            for batch in dataloader:
                if total_samples >= max_samples:
                    break

                features = batch['features'].to(device)
                encoder_output = speaker_encoder(features)
                z = encoder_output['z']
                w = mapping_network(z)

                w_samples.append(w.cpu())
                total_samples += w.shape[0]

        w_samples = torch.cat(w_samples, dim=0)[:max_samples].to(device)

        # Rest is same as discover_from_samples
        mean = w_samples.mean(dim=0)
        w_centered = w_samples - mean

        U, S, Vt = torch.linalg.svd(w_centered, full_matrices=False)

        num_dirs = min(self.config.num_directions, Vt.shape[0])
        self.directions.copy_(Vt[:num_dirs])
        self.eigenvalues.copy_(S[:num_dirs])
        self.mean.copy_(mean)
        self.discovered.fill_(True)

        total_var = (S ** 2).sum()
        explained_var_ratio = (S[:num_dirs] ** 2) / total_var

        return {
            'directions': self.directions.clone(),
            'eigenvalues': self.eigenvalues.clone(),
            'explained_variance_ratio': explained_var_ratio,
            'mean': mean,
        }

    def label_direction(
        self,
        direction_idx: int,
        label: str,
    ):
        """Label a discovered direction for semantic control."""
        assert 0 <= direction_idx < self.config.num_directions
        self.direction_labels[label] = direction_idx

    def manipulate(
        self,
        w: torch.Tensor,  # [batch, mapping_dim]
        direction_idx: int,
        scale: float,
    ) -> torch.Tensor:
        """
        Manipulate latent code along a direction.

        Args:
            w: Original mapping code
            direction_idx: Index of direction to manipulate
            scale: Scale factor (positive or negative)

        Returns:
            Modified w' = w + scale * v_n
        """
        assert self.discovered, "Directions not yet discovered. Call discover_* first."
        direction = self.directions[direction_idx]
        return w + scale * direction

    def manipulate_by_label(
        self,
        w: torch.Tensor,
        label: str,
        scale: float,
    ) -> torch.Tensor:
        """
        Manipulate by semantic label (e.g., 'pitch', 'gender', 'emotion').

        Args:
            w: Original mapping code
            label: Semantic label of direction
            scale: Scale factor

        Returns:
            Modified w'
        """
        assert label in self.direction_labels, f"Unknown label: {label}. Available: {list(self.direction_labels.keys())}"
        direction_idx = self.direction_labels[label]
        return self.manipulate(w, direction_idx, scale)

    def interpolate_directions(
        self,
        w: torch.Tensor,
        direction_scales: Dict[int, float],
    ) -> torch.Tensor:
        """
        Apply multiple direction manipulations simultaneously.

        Args:
            w: Original mapping code
            direction_scales: Dict mapping direction_idx -> scale

        Returns:
            Modified w with all manipulations applied
        """
        w_modified = w.clone()
        for direction_idx, scale in direction_scales.items():
            w_modified = self.manipulate(w_modified, direction_idx, scale)
        return w_modified

    def get_direction_info(self) -> Dict[str, any]:
        """Get information about discovered directions."""
        if not self.discovered:
            return {'discovered': False}

        total_var = (self.eigenvalues ** 2).sum()
        explained_var = (self.eigenvalues ** 2) / total_var

        return {
            'discovered': True,
            'num_directions': self.config.num_directions,
            'explained_variance_ratio': explained_var.cpu().numpy(),
            'cumulative_variance': explained_var.cumsum(dim=0).cpu().numpy(),
            'labels': self.direction_labels,
        }


# =============================================================================
# FULL VOXGENESIS MODEL
# =============================================================================

class VoxGenesis(nn.Module):
    """
    VoxGenesis: Latent Speaker Manifold for Interpretable Control.

    Generator: G(z|Y) = f(T(M(z), Y) + e(Y))

    Components:
    - Speaker Encoder: Audio → z (speaker latent)
    - Mapping Network M: z → w (non-isotropic latent)
    - Semantic Embedding e: Y → semantic features
    - Transformation T: Conditions w on semantics
    - Deconvolution f: Generates mel spectrogram
    - Direction Discovery: PCA-based interpretable axes
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        # Components
        self.speaker_encoder = SpeakerEncoder(config)
        self.mapping_network = MappingNetwork(
            latent_dim=config.latent_dim,
            mapping_dim=config.mapping_dim,
            num_layers=config.num_mapping_layers,
            dropout=config.dropout,
        )
        self.semantic_embedding = SemanticEmbedding(config)
        self.transformation = SemanticConditionedTransformation(config)
        self.deconvolution = DeconvolutionNetwork(config)
        self.discriminator = VoxGenesisDiscriminator(config)

        # Direction discovery
        self.direction_discovery = LatentDirectionDiscovery(config)

        # Output projection for CSM integration
        self.output_proj = nn.Sequential(
            nn.Linear(config.mapping_dim + config.semantic_hidden_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode_speaker(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """Encode speaker features to latent space."""
        return self.speaker_encoder(features)

    def generate_from_z(
        self,
        z: torch.Tensor,  # [batch, latent_dim]
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate mel spectrogram from latent z and semantic tokens.

        G(z|Y) = f(T(M(z), Y) + e(Y))
        """
        # Mapping network: z → w
        w = self.mapping_network(z)

        # Semantic embedding: Y → e(Y)
        semantic_output = self.semantic_embedding(semantic_tokens)
        semantic_emb = semantic_output['embedding']

        # Transformation: T(M(z), Y)
        transformed = self.transformation(w, semantic_emb)

        # Add semantic embedding
        conditioned = transformed + semantic_emb

        # Deconvolution: f(...)
        mel = self.deconvolution(conditioned)

        return {
            'mel': mel,
            'w': w,
            'semantic_emb': semantic_emb,
            'transformed': transformed,
        }

    def generate_from_features(
        self,
        speaker_features: torch.Tensor,  # [batch, seq, input_dim]
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
    ) -> Dict[str, torch.Tensor]:
        """Generate mel from speaker audio features and semantic tokens."""
        encoder_output = self.encode_speaker(speaker_features)
        z = encoder_output['z']

        gen_output = self.generate_from_z(z, semantic_tokens)
        gen_output.update({
            'z': z,
            'mu': encoder_output['mu'],
            'logvar': encoder_output['logvar'],
        })

        return gen_output

    def generate_novel_speaker(
        self,
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
        num_speakers: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate speech with novel speakers by sampling from Gaussian.

        Args:
            semantic_tokens: Semantic tokens determining content
            num_speakers: Number of novel speakers to generate

        Returns:
            Dict with generated mels for each novel speaker
        """
        device = semantic_tokens.device
        batch_size = semantic_tokens.shape[0]

        results = []
        for _ in range(num_speakers):
            z = torch.randn(batch_size, self.config.latent_dim, device=device)
            gen_output = self.generate_from_z(z, semantic_tokens)
            results.append(gen_output)

        return {
            'generations': results,
            'z_samples': [r['w'] for r in results],
        }

    def voice_conversion(
        self,
        source_semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim] (content)
        target_speaker_features: torch.Tensor,  # [batch, seq, input_dim] (speaker)
    ) -> Dict[str, torch.Tensor]:
        """
        Voice conversion: source content + target speaker.

        Y_a→b = G(z_b | Y_a)
        """
        # Get target speaker latent
        encoder_output = self.encode_speaker(target_speaker_features)
        z_target = encoder_output['z']

        # Generate with source content and target speaker
        return self.generate_from_z(z_target, source_semantic_tokens)

    def voice_editing(
        self,
        speaker_features: torch.Tensor,  # [batch, seq, input_dim]
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
        direction_idx: int,
        scale: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Edit voice by manipulating latent along discovered direction.

        Args:
            speaker_features: Source speaker features
            semantic_tokens: Semantic tokens
            direction_idx: Which direction to manipulate
            scale: How much to move along direction

        Returns:
            Dict with original and edited generation
        """
        # Encode speaker
        encoder_output = self.encode_speaker(speaker_features)
        z = encoder_output['z']
        w = self.mapping_network(z)

        # Original generation
        original = self.generate_from_z(z, semantic_tokens)

        # Edited latent
        w_edited = self.direction_discovery.manipulate(w, direction_idx, scale)

        # Generate from edited w (bypass mapping network)
        semantic_output = self.semantic_embedding(semantic_tokens)
        semantic_emb = semantic_output['embedding']
        transformed = self.transformation(w_edited, semantic_emb)
        conditioned = transformed + semantic_emb
        mel_edited = self.deconvolution(conditioned)

        return {
            'original_mel': original['mel'],
            'edited_mel': mel_edited,
            'original_w': w,
            'edited_w': w_edited,
        }

    def forward(
        self,
        speaker_features: torch.Tensor,  # [batch, seq, input_dim]
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
        mel_target: Optional[torch.Tensor] = None,  # [batch, time, mel_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Returns all outputs needed for computing losses.
        """
        # Generate
        gen_output = self.generate_from_features(speaker_features, semantic_tokens)

        result = {
            **gen_output,
        }

        # Discriminator forward if target provided
        if mel_target is not None:
            # Real samples
            disc_real = self.discriminator(mel_target, semantic_tokens, return_features=True)

            # Fake samples
            disc_fake = self.discriminator(gen_output['mel'], semantic_tokens, return_features=True)

            result.update({
                'disc_real': disc_real,
                'disc_fake': disc_fake,
            })

        return result

    def get_combined_embedding(
        self,
        speaker_features: torch.Tensor,
        semantic_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Get combined speaker+semantic embedding for downstream tasks."""
        encoder_output = self.encode_speaker(speaker_features)
        w = self.mapping_network(encoder_output['z'])

        semantic_output = self.semantic_embedding(semantic_tokens)
        semantic_global = semantic_output['global_emb']

        combined = torch.cat([w, semantic_global], dim=-1)
        return self.output_proj(combined)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class VoxGenesisLoss(nn.Module):
    """
    Combined loss function for VoxGenesis training.

    Components:
    1. GAN adversarial loss
    2. Feature matching loss
    3. KL divergence to Gaussian prior
    4. Mel reconstruction loss (optional auxiliary)
    """

    def __init__(self, config: VoxGenesisConfig):
        super().__init__()
        self.config = config

        self.gan_weight = config.gan_disc_weight
        self.fm_weight = config.gan_feature_matching_weight
        self.kl_weight = config.kl_weight
        self.recon_weight = 1.0  # Auxiliary reconstruction

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def adversarial_loss(
        self,
        disc_real: Dict[str, torch.Tensor],
        disc_fake: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial losses for generator and discriminator.

        Returns:
            (generator_loss, discriminator_loss)
        """
        real_logits = disc_real['logits']
        fake_logits = disc_fake['logits']

        # Discriminator loss (real = 1, fake = 0)
        d_real = F.mse_loss(real_logits, torch.ones_like(real_logits))
        d_fake = F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        d_loss = (d_real + d_fake) / 2

        # Generator loss (fool discriminator: fake = 1)
        g_loss = F.mse_loss(fake_logits, torch.ones_like(fake_logits))

        return g_loss, d_loss

    def feature_matching_loss(
        self,
        disc_real: Dict[str, torch.Tensor],
        disc_fake: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Feature matching loss between real and fake features."""
        if 'features' not in disc_real or 'features' not in disc_fake:
            return torch.tensor(0.0)

        loss = 0.0
        for real_feat, fake_feat in zip(disc_real['features'], disc_fake['features']):
            loss = loss + self.l1_loss(fake_feat, real_feat.detach())

        return loss / len(disc_real['features'])

    def kl_divergence_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence from posterior to N(0, I)."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def reconstruction_loss(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
    ) -> torch.Tensor:
        """L1 reconstruction loss on mel spectrogram."""
        # Handle length mismatch
        min_len = min(mel_pred.shape[1], mel_target.shape[1])
        return self.l1_loss(mel_pred[:, :min_len], mel_target[:, :min_len])

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            model_output: Output from VoxGenesis.forward()
            mel_target: Target mel spectrogram
            step: Training step for KL annealing

        Returns:
            Dict with individual losses and totals
        """
        losses = {}

        # 1. Adversarial loss
        if 'disc_real' in model_output and 'disc_fake' in model_output:
            g_adv, d_loss = self.adversarial_loss(
                model_output['disc_real'],
                model_output['disc_fake']
            )
            losses['g_adv'] = g_adv
            losses['d_loss'] = d_loss

            # 2. Feature matching loss
            fm_loss = self.feature_matching_loss(
                model_output['disc_real'],
                model_output['disc_fake']
            )
            losses['fm'] = fm_loss
        else:
            losses['g_adv'] = torch.tensor(0.0, device=mel_target.device)
            losses['d_loss'] = torch.tensor(0.0, device=mel_target.device)
            losses['fm'] = torch.tensor(0.0, device=mel_target.device)

        # 3. KL divergence with annealing
        kl_weight = min(1.0, step / self.config.kl_anneal_steps) * self.kl_weight
        kl_loss = self.kl_divergence_loss(model_output['mu'], model_output['logvar'])
        losses['kl'] = kl_loss
        losses['kl_weighted'] = kl_weight * kl_loss

        # 4. Reconstruction loss
        recon_loss = self.reconstruction_loss(model_output['mel'], mel_target)
        losses['recon'] = recon_loss

        # Total generator loss
        losses['g_total'] = (
            self.gan_weight * losses['g_adv']
            + self.fm_weight * losses['fm']
            + losses['kl_weighted']
            + self.recon_weight * losses['recon']
        )

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class VoxGenesisAdapter(nn.Module):
    """
    Adapter to integrate VoxGenesis with existing prosody pipeline.

    Converts VoxGenesis speaker+semantic representation to prefix tokens
    compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: VoxGenesisConfig,
        model: Optional[VoxGenesis] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else VoxGenesis(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        speaker_features: torch.Tensor,  # [batch, seq, input_dim]
        semantic_tokens: Optional[torch.Tensor] = None,  # [batch, seq, semantic_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            speaker_features: HuBERT/WavLM features from speaker audio
            semantic_tokens: Optional semantic tokens (if None, uses speaker_features)

        Returns:
            Dict with prosody_tokens and intermediate values
        """
        batch_size = speaker_features.shape[0]

        # Use speaker features as semantic tokens if not provided
        if semantic_tokens is None:
            semantic_tokens = speaker_features

        # Get combined embedding
        combined_emb = self.model.get_combined_embedding(speaker_features, semantic_tokens)

        # Project to tokens
        tokens = self.token_proj(combined_emb)  # [B, output_dim * num_tokens]

        # Reshape
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )

        # Normalize
        tokens = self.norm(tokens)

        # Get additional info for potential manipulation
        encoder_output = self.model.encode_speaker(speaker_features)
        w = self.model.mapping_network(encoder_output['z'])

        return {
            'prosody_tokens': tokens,
            'z': encoder_output['z'],
            'w': w,
            'mu': encoder_output['mu'],
            'logvar': encoder_output['logvar'],
        }

    def from_latent(
        self,
        z: torch.Tensor,  # [batch, latent_dim]
        semantic_tokens: torch.Tensor,  # [batch, seq, semantic_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from explicit latent z.

        Useful for:
        - Novel speaker generation (z sampled from N(0,I))
        - Voice editing (z manipulated along directions)
        """
        batch_size = z.shape[0]

        w = self.model.mapping_network(z)
        semantic_output = self.model.semantic_embedding(semantic_tokens)
        semantic_global = semantic_output['global_emb']

        combined = torch.cat([w, semantic_global], dim=-1)
        combined_emb = self.model.output_proj(combined)

        tokens = self.token_proj(combined_emb)
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
            'w': w,
        }

    def novel_speaker_tokens(
        self,
        semantic_tokens: torch.Tensor,
        num_speakers: int = 1,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate prosody tokens for novel speakers."""
        device = semantic_tokens.device
        batch_size = semantic_tokens.shape[0]

        results = []
        for _ in range(num_speakers):
            z = torch.randn(batch_size, self.config.latent_dim, device=device)
            result = self.from_latent(z, semantic_tokens)
            results.append(result)

        return results

    def edit_speaker_tokens(
        self,
        speaker_features: torch.Tensor,
        semantic_tokens: torch.Tensor,
        direction_idx: int,
        scale: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Edit speaker characteristics along discovered direction.

        Args:
            speaker_features: Source speaker audio features
            semantic_tokens: Semantic tokens
            direction_idx: Direction to manipulate
            scale: Scale of manipulation

        Returns:
            Original and edited prosody tokens
        """
        batch_size = speaker_features.shape[0]

        # Get original
        encoder_output = self.model.encode_speaker(speaker_features)
        z = encoder_output['z']
        w = self.model.mapping_network(z)

        # Edit
        w_edited = self.model.direction_discovery.manipulate(w, direction_idx, scale)

        # Generate tokens for both
        semantic_output = self.model.semantic_embedding(semantic_tokens)
        semantic_global = semantic_output['global_emb']

        # Original
        combined = torch.cat([w, semantic_global], dim=-1)
        combined_emb = self.model.output_proj(combined)
        tokens_orig = self.token_proj(combined_emb)
        tokens_orig = tokens_orig.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)
        tokens_orig = self.norm(tokens_orig)

        # Edited
        combined_edited = torch.cat([w_edited, semantic_global], dim=-1)
        combined_emb_edited = self.model.output_proj(combined_edited)
        tokens_edited = self.token_proj(combined_emb_edited)
        tokens_edited = tokens_edited.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)
        tokens_edited = self.norm(tokens_edited)

        return {
            'original_tokens': tokens_orig,
            'edited_tokens': tokens_edited,
            'original_w': w,
            'edited_w': w_edited,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_voxgenesis_adapter(
    checkpoint: Optional[str] = None,
    config: Optional[VoxGenesisConfig] = None,
    device: str = "cpu",
) -> VoxGenesisAdapter:
    """Create VoxGenesis adapter, optionally loading from checkpoint."""
    if config is None:
        config = VoxGenesisConfig()

    model = VoxGenesis(config)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

    adapter = VoxGenesisAdapter(config, model)
    return adapter.to(device)


def discover_directions(
    model: VoxGenesis,
    num_samples: int = 10000,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Discover interpretable directions from trained model."""
    model.eval()
    return model.direction_discovery.discover_from_samples(
        model.mapping_network,
        num_samples=num_samples,
        device=torch.device(device),
    )


def label_directions_from_analysis(
    model: VoxGenesis,
    suggested_labels: Optional[Dict[int, str]] = None,
):
    """
    Label directions based on analysis.

    Default labels based on VoxGenesis paper findings:
    - PC0-1: Inter-speaker (gender, identity)
    - PC2-5: Intra-speaker (pitch, tone, emotion)
    """
    if suggested_labels is None:
        suggested_labels = {
            0: 'gender',
            1: 'speaker_identity',
            2: 'pitch',
            3: 'tone',
            4: 'emotion',
            5: 'speaking_rate',
        }

    for idx, label in suggested_labels.items():
        if idx < model.config.num_directions:
            model.direction_discovery.label_direction(idx, label)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VoxGenesis: Latent Speaker Manifold - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = VoxGenesisConfig()

    # Test parameters
    batch_size = 2
    seq_len = 100
    input_dim = config.input_dim
    semantic_dim = config.semantic_dim
    mel_dim = config.mel_dim

    # Create dummy inputs
    speaker_features = torch.randn(batch_size, seq_len, input_dim).to(device)
    semantic_tokens = torch.randn(batch_size, seq_len, semantic_dim).to(device)
    mel_target = torch.randn(batch_size, seq_len * 4, mel_dim).to(device)  # Upsampled length

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Mapping dim: {config.mapping_dim}")
    print(f"  Mapping layers: {config.num_mapping_layers}")
    print(f"  Num directions: {config.num_directions}")
    print("  [PASS]")

    # Test 2: Mapping Network
    print("\n[Test 2] Mapping Network...")
    mapping = MappingNetwork(
        latent_dim=config.latent_dim,
        mapping_dim=config.mapping_dim,
        num_layers=config.num_mapping_layers,
    ).to(device)

    z = torch.randn(batch_size, config.latent_dim).to(device)
    w = mapping(z)
    print(f"  Input z shape: {z.shape}")
    print(f"  Output w shape: {w.shape}")
    assert w.shape == (batch_size, config.mapping_dim)
    print("  [PASS]")

    # Test 3: Semantic Embedding
    print("\n[Test 3] Semantic Embedding...")
    semantic_embed = SemanticEmbedding(config).to(device)
    sem_output = semantic_embed(semantic_tokens)
    print(f"  Embedding shape: {sem_output['embedding'].shape}")
    print(f"  Global embedding shape: {sem_output['global_emb'].shape}")
    print("  [PASS]")

    # Test 4: Speaker Encoder
    print("\n[Test 4] Speaker Encoder...")
    speaker_enc = SpeakerEncoder(config).to(device)
    enc_output = speaker_enc(speaker_features)
    print(f"  z shape: {enc_output['z'].shape}")
    print(f"  mu shape: {enc_output['mu'].shape}")
    print(f"  logvar shape: {enc_output['logvar'].shape}")
    print("  [PASS]")

    # Test 5: Transformation Network
    print("\n[Test 5] Transformation Network...")
    transform = SemanticConditionedTransformation(config).to(device)
    transformed = transform(w, sem_output['embedding'])
    print(f"  Transformed shape: {transformed.shape}")
    print("  [PASS]")

    # Test 6: Deconvolution Network
    print("\n[Test 6] Deconvolution Network...")
    deconv = DeconvolutionNetwork(config).to(device)
    mel_gen = deconv(transformed)
    print(f"  Generated mel shape: {mel_gen.shape}")
    print("  [PASS]")

    # Test 7: Full VoxGenesis Model
    print("\n[Test 7] Full VoxGenesis Model...")
    model = VoxGenesis(config).to(device)
    output = model(speaker_features, semantic_tokens, mel_target)
    print(f"  Generated mel shape: {output['mel'].shape}")
    print(f"  z shape: {output['z'].shape}")
    print(f"  w shape: {output['w'].shape}")
    print("  [PASS]")

    # Test 8: Loss Function
    print("\n[Test 8] Loss Function...")
    loss_fn = VoxGenesisLoss(config)
    losses = loss_fn(output, mel_target, step=1000)
    print(f"  Generator adversarial loss: {losses['g_adv'].item():.4f}")
    print(f"  Discriminator loss: {losses['d_loss'].item():.4f}")
    print(f"  Feature matching loss: {losses['fm'].item():.4f}")
    print(f"  KL loss: {losses['kl'].item():.4f}")
    print(f"  Reconstruction loss: {losses['recon'].item():.4f}")
    print(f"  Total generator loss: {losses['g_total'].item():.4f}")
    print("  [PASS]")

    # Test 9: Direction Discovery
    print("\n[Test 9] Direction Discovery...")
    directions = model.direction_discovery.discover_from_samples(
        model.mapping_network,
        num_samples=500,  # Smaller for testing
        device=torch.device(device),
    )
    print(f"  Discovered {config.num_directions} directions")
    print(f"  Top 5 explained variance: {directions['explained_variance_ratio'][:5].cpu().numpy()}")
    print(f"  Cumulative variance (top 5): {directions['explained_variance_ratio'][:5].sum().item():.4f}")

    # Label some directions
    label_directions_from_analysis(model)
    print(f"  Labeled directions: {model.direction_discovery.direction_labels}")
    print("  [PASS]")

    # Test 10: Voice Editing
    print("\n[Test 10] Voice Editing...")
    edit_output = model.voice_editing(
        speaker_features, semantic_tokens,
        direction_idx=2,  # pitch
        scale=0.5,
    )
    print(f"  Original mel shape: {edit_output['original_mel'].shape}")
    print(f"  Edited mel shape: {edit_output['edited_mel'].shape}")
    print(f"  Original w shape: {edit_output['original_w'].shape}")
    print(f"  Edited w shape: {edit_output['edited_w'].shape}")

    # Check that editing actually changed something
    w_diff = (edit_output['edited_w'] - edit_output['original_w']).abs().mean()
    print(f"  W difference: {w_diff.item():.4f}")
    assert w_diff > 0, "Editing should change w"
    print("  [PASS]")

    # Test 11: Novel Speaker Generation
    print("\n[Test 11] Novel Speaker Generation...")
    novel_output = model.generate_novel_speaker(semantic_tokens, num_speakers=3)
    print(f"  Generated {len(novel_output['generations'])} novel speakers")
    for i, gen in enumerate(novel_output['generations']):
        print(f"  Speaker {i}: mel shape {gen['mel'].shape}")
    print("  [PASS]")

    # Test 12: CSM Adapter
    print("\n[Test 12] CSM Adapter...")
    adapter = VoxGenesisAdapter(config, model).to(device)
    adapter_output = adapter(speaker_features, semantic_tokens)
    print(f"  Prosody tokens shape: {adapter_output['prosody_tokens'].shape}")
    assert adapter_output['prosody_tokens'].shape == (batch_size, config.num_prefix_tokens, config.output_dim)

    # Test edit through adapter
    edit_adapter_output = adapter.edit_speaker_tokens(
        speaker_features, semantic_tokens,
        direction_idx=2,
        scale=0.5,
    )
    print(f"  Original tokens shape: {edit_adapter_output['original_tokens'].shape}")
    print(f"  Edited tokens shape: {edit_adapter_output['edited_tokens'].shape}")
    print("  [PASS]")

    # Test 13: Novel Speaker Tokens
    print("\n[Test 13] Novel Speaker Tokens...")
    novel_tokens = adapter.novel_speaker_tokens(semantic_tokens, num_speakers=2)
    print(f"  Generated tokens for {len(novel_tokens)} novel speakers")
    for i, tokens in enumerate(novel_tokens):
        print(f"  Speaker {i}: tokens shape {tokens['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 14: Backward Pass
    print("\n[Test 14] Backward Pass...")
    model.zero_grad()
    output = model(speaker_features, semantic_tokens, mel_target)
    losses = loss_fn(output, mel_target, step=1000)
    losses['g_total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All VoxGenesis tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. LATENT SPEAKER MANIFOLD:
       - Mapping Network M: z ~ N(0,I) → w (non-isotropic)
       - 7-layer FF network (Style-GAN inspired)
       - Transforms Gaussian to speaker distribution

    2. GENERATOR ARCHITECTURE:
       G(z|Y) = f(T(M(z), Y) + e(Y))
       - e: Semantic embedding (content)
       - M: Mapping network (speaker)
       - T: Semantic-conditioned transformation
       - f: Deconvolution network (synthesis)

    3. INTERPRETABLE DIRECTIONS VIA PCA:
       - Apply SVD to M(z) samples
       - Leading PCs: Inter-speaker (gender, identity)
       - Later PCs: Intra-speaker (pitch, tone, emotion)
       - Voice editing: w' = w + scale * v_n

    4. NOVEL SPEAKER GENERATION:
       - Sample z ~ N(0, I)
       - Generate speech with new speaker characteristics
       - No reference audio needed

    5. VOICE CONVERSION:
       Y_a→b = G(z_b | Y_a)
       - Content from source (semantic tokens)
       - Speaker from target (encoded z)

    6. CSM INTEGRATION:
       adapter = VoxGenesisAdapter(config)
       tokens = adapter(features)  # [batch, 4, 2048]
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from voxgenesis import (
    VoxGenesisConfig,
    VoxGenesis,
    VoxGenesisAdapter,
    discover_directions,
    label_directions_from_analysis,
)

# Initialize
config = VoxGenesisConfig()
model = VoxGenesis(config).cuda()

# Train model...

# Discover interpretable directions
directions = discover_directions(model, num_samples=10000)
print(f"Explained variance: {directions['explained_variance_ratio'][:5]}")

# Label directions (based on analysis or manual inspection)
label_directions_from_analysis(model, {
    0: 'gender',
    1: 'speaker_identity',
    2: 'pitch',
    3: 'tone',
    4: 'emotion',
})

# Voice editing: increase pitch
edit_output = model.voice_editing(
    speaker_features, semantic_tokens,
    direction_idx=2,  # 'pitch'
    scale=0.8,
)
edited_mel = edit_output['edited_mel']

# Or by label
w = model.mapping_network(model.speaker_encoder(features)['z'])
w_high_pitch = model.direction_discovery.manipulate_by_label(w, 'pitch', 0.8)

# Novel speaker generation
novel = model.generate_novel_speaker(semantic_tokens, num_speakers=5)

# Voice conversion
converted = model.voice_conversion(source_semantic, target_speaker_features)

# CSM integration
adapter = VoxGenesisAdapter(config, model)
prefix_tokens = adapter(features)['prosody_tokens']  # [batch, 4, 2048]

# Edit through adapter
edited = adapter.edit_speaker_tokens(
    features, semantic_tokens,
    direction_idx=4,  # 'emotion'
    scale=1.5,
)
edited_tokens = edited['edited_tokens']
""")
