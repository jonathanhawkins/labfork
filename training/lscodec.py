"""
LSCodec: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec

Based on "LSCodec: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec"
(Interspeech 2025, arXiv:2410.15764).

Key Innovation: Three-stage unsupervised training with time stretching speaker
perturbation that creates a continuous information bottleneck before vector
quantization, producing speaker-decoupled discrete speech tokens.

Architecture:
```
                Time Stretching Perturbation
                         ↓
Audio (content) → [CNN Encoder] → μ, σ² (VAE posterior)
                         ↓
              [Gaussian Sampling / VQ]
                         ↓
Reference Audio → [WavLM] → Timbre Features
                         ↓
              [Conformer Decoder with Cross-Attention]
                         ↓
                  Mel + SSL Tokens
```

Three Training Stages:
1. Stage 1 (VAE): Speech VAE with speaker perturbation for continuous disentanglement
   - KL divergence + reconstruction + SSL token prediction

2. Stage 2 (VQ-VAE): Add VQ layer to quantize VAE outputs
   - Commitment loss + reconstruction + SSL token prediction

3. Stage 3 (Vocoder): Token vocoder for waveform synthesis
   - Discrete tokens + WavLM timbre → waveform

Key Technique - Time Stretching Perturbation:
- Speed up by factor β ∈ [0.8, 1.2] (changes pitch/timbre)
- WSOLA pitch-preserving tempo restoration to original duration
- Creates time-aligned pairs: perturbed content, original timbre
- Forces bottleneck to encode only content/prosody (speaker stripped)

Bitrate Configurations:
- LSCodec-50Hz: V=300, 0.45 kbps
- LSCodec-25Hz: V=1024, 0.25 kbps

Benefits:
- Ultra-low bitrate (0.25-0.45 kbps) with single codebook
- Speaker-decoupled tokens reduce TTS modeling burden
- Unsupervised training (no speaker labels needed)
- Excellent voice conversion from speaker disentanglement
- Compatible with LM-based speech generation

Usage:
    from lscodec import (
        LSCodecConfig,
        LSCodec,
        LSCodecLoss,
        LSCodecAdapter,
        TimeStretchingPerturbation,
    )

    # Initialize
    config = LSCodecConfig(frame_rate=50, vocab_size=300)  # 0.45 kbps
    model = LSCodec(config).cuda()

    # Training Stage 1: VAE
    output = model.forward_vae(mel, reference_mel)
    losses = loss_fn.compute_vae_loss(output, mel_target, ssl_target)

    # Training Stage 2: VQ-VAE
    output = model.forward_vqvae(mel, reference_mel)
    losses = loss_fn.compute_vqvae_loss(output, mel_target, ssl_target)

    # Training Stage 3: Token Vocoder
    tokens = model.encode_tokens(mel)
    audio = model.vocoder(tokens, reference_audio)

    # Inference
    tokens = model.encode(mel)['indices']  # Speaker-decoupled tokens
    mel_recon = model.decode(tokens, reference_mel)

    # Voice Conversion
    mel_vc = model.voice_convert(source_mel, target_speaker_mel)

    # CSM Integration
    adapter = LSCodecAdapter(config, model)
    prefix_tokens = adapter(mel)['prosody_tokens']  # [batch, 4, 2048]
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LSCodecConfig:
    """Configuration for LSCodec speaker-decoupled speech codec."""

    # Audio settings
    sample_rate: int = 16000
    hop_length: int = 320  # For 50Hz: 320, for 25Hz: 640
    mel_dim: int = 80  # Mel spectrogram bins (cepstral normalized)

    # Frame rate and vocabulary (determines bitrate)
    # LSCodec-50Hz: V=300, 0.45 kbps
    # LSCodec-25Hz: V=1024, 0.25 kbps
    frame_rate: int = 50  # 25 or 50 Hz
    vocab_size: int = 300  # V: codebook size (300 or 1024)

    # CNN Encoder
    encoder_hidden_dim: int = 512
    encoder_num_blocks: int = 11  # 11 for 50Hz, 12 for 25Hz
    encoder_output_dim: int = 128  # Split into μ (64) and σ² (64)
    code_dim: int = 64  # VQ code dimension (first 64 dims of encoder output)

    # Conformer Decoder
    decoder_num_blocks: int = 2
    decoder_attention_dim: int = 184
    decoder_num_heads: int = 2
    decoder_ffn_dim: int = 512

    # WavLM Timbre Extractor
    wavlm_layer: int = 6  # WavLM-Large layer for timbre embeddings
    wavlm_dim: int = 1024  # WavLM hidden dimension

    # Prompt Prenet (for WavLM embeddings)
    prenet_hidden_dims: Tuple[int, ...] = (128, 256, 512)

    # SSL Token Prediction
    ssl_num_clusters: int = 2048  # K-means clusters for SSL tokens

    # Training - Perturbation
    beta_min: float = 0.8  # Min stretching factor
    beta_max: float = 1.2  # Max stretching factor

    # Training - Loss Weights
    gamma_kl: float = 60.0
    gamma_recon: float = 60.0
    gamma_idx: float = 2.0
    gamma_cmt: float = 1.0

    # Training - VQ
    ema_decay: float = 0.99  # EMA codebook update weight
    code_expiration_delay: int = 5000  # Steps before resetting unused codes

    # Training - General
    training_epochs_per_stage: int = 200
    min_utterance_seconds: float = 6.0  # Minimum utterance length
    prompt_ratio_min: float = 0.33  # Prompt = 1/3 to 1/2 of duration
    prompt_ratio_max: float = 0.5

    dropout: float = 0.1

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# TIME STRETCHING PERTURBATION
# =============================================================================

class TimeStretchingPerturbation(nn.Module):
    """
    Time Stretching Speaker Perturbation for unsupervised speaker disentanglement.

    Algorithm:
    1. Speed up audio by factor β ∈ [0.8, 1.2] (changes pitch and formants)
    2. Apply WSOLA pitch-preserving tempo restoration to original duration

    This creates time-aligned perturbed speech where:
    - Content and relative pitch patterns are preserved
    - Absolute pitch and timbre (formants) are altered

    The perturbation forces the VAE bottleneck to encode only content/prosody
    information, as the decoder receives timbre from an unperturbed reference.

    Note: For efficiency, this implementation uses spectral methods.
    For production, use librosa or SoX for higher quality.
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config
        self.beta_min = config.beta_min
        self.beta_max = config.beta_max

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples] or [batch, 1, samples]
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply time stretching perturbation.

        Args:
            audio: Input audio waveform
            beta: Stretching factor (random if None)

        Returns:
            Tuple of (perturbed_audio, beta_used)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        batch_size, num_samples = audio.shape

        # Sample beta if not provided
        if beta is None:
            beta = torch.empty(1).uniform_(self.beta_min, self.beta_max).item()

        # For simplicity, use resampling-based approach
        # In production, use librosa.effects.time_stretch + pitch_shift
        perturbed = self._time_stretch_resample(audio, beta)

        return perturbed, beta

    def _time_stretch_resample(
        self,
        audio: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """
        Simple time stretch via resampling.

        Note: This is a simplified implementation. For full quality:
        1. Use librosa.effects.time_stretch (WSOLA)
        2. Use SoX tempo and pitch effects
        """
        batch_size, num_samples = audio.shape
        device = audio.device

        # Step 1: Speed up by beta (changes pitch)
        # Resample to shorter duration
        new_length = int(num_samples / beta)

        # Use interpolation for resampling
        audio_scaled = F.interpolate(
            audio.unsqueeze(1),  # [B, 1, T]
            size=new_length,
            mode='linear',
            align_corners=False,
        ).squeeze(1)  # [B, T_new]

        # Step 2: Pitch-preserving stretch back to original duration
        # In reality, WSOLA preserves pitch while changing tempo
        # Here we approximate with interpolation (which shifts pitch back)
        perturbed = F.interpolate(
            audio_scaled.unsqueeze(1),
            size=num_samples,
            mode='linear',
            align_corners=False,
        ).squeeze(1)

        return perturbed

    def perturb_mel(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply perturbation directly to mel spectrogram.

        This is a faster approximation that operates in frequency domain.
        """
        batch_size, seq_len, mel_dim = mel.shape

        if beta is None:
            beta = torch.empty(1).uniform_(self.beta_min, self.beta_max).item()

        # Perturbation in mel domain:
        # 1. Time scaling changes pitch → shift mel bins
        # 2. Formant shift → scale frequency axis

        # Frequency axis shift (simulates pitch change)
        shift_bins = int((beta - 1.0) * mel_dim * 0.5)

        if shift_bins != 0:
            if shift_bins > 0:
                # Shift up (higher pitch)
                perturbed = F.pad(mel[:, :, shift_bins:], (0, shift_bins), mode='replicate')
            else:
                # Shift down (lower pitch)
                perturbed = F.pad(mel[:, :, :shift_bins], (-shift_bins, 0), mode='replicate')
        else:
            perturbed = mel

        # Add small noise to simulate perturbation artifacts
        noise = torch.randn_like(perturbed) * 0.02 * (abs(beta - 1.0))
        perturbed = perturbed + noise

        return perturbed, beta


# =============================================================================
# CNN ENCODER
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual convolutional block for CNN encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Align lengths
        min_len = min(out.shape[-1], residual.shape[-1])
        out = out[..., :min_len] + residual[..., :min_len]

        return self.activation(out)


class CNNEncoder(nn.Module):
    """
    CNN Encoder for LSCodec.

    Compresses input mel spectrogram to Gaussian posterior parameters (μ, σ²).
    Architecture follows paper: 11-12 residual blocks, 512 hidden dim,
    output 128 dim (split into μ and σ²).
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = nn.Conv1d(config.mel_dim, config.encoder_hidden_dim, 1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(config.encoder_num_blocks):
            # Downsample every few blocks for temporal compression
            stride = 2 if (i == 2 or i == 5) else 1
            self.blocks.append(
                ResidualBlock(
                    config.encoder_hidden_dim,
                    config.encoder_hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    dropout=config.dropout,
                )
            )

        # Output projection: 128 dim (64 for μ, 64 for σ²)
        self.output_proj = nn.Conv1d(
            config.encoder_hidden_dim,
            config.encoder_output_dim,  # 128
            kernel_size=1,
        )

        self.norm = nn.LayerNorm(config.encoder_output_dim)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode mel to VAE posterior parameters.

        Args:
            mel: [batch, seq, mel_dim]

        Returns:
            Dict with mu, log_var, z (sampled latent)
        """
        # Transpose for conv: [B, T, D] -> [B, D, T]
        x = mel.transpose(1, 2)

        # Encode
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)

        # Back to [B, T, D]
        x = x.transpose(1, 2)
        x = self.norm(x)

        # Split into mu and log_var
        code_dim = self.config.code_dim  # 64
        mu = x[..., :code_dim]
        log_var = x[..., code_dim:code_dim*2]

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean at inference

        return {
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'encoder_features': x,
        }


# =============================================================================
# PROMPT PRENET (WavLM TIMBRE PROCESSOR)
# =============================================================================

class PromptPrenet(nn.Module):
    """
    Prompt Prenet for processing WavLM timbre embeddings.

    Four CNN blocks with scaled residual connections.
    Hidden dims: 128 → 256 → 512
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()

        # Input projection from WavLM dim
        self.input_proj = nn.Linear(config.wavlm_dim, config.prenet_hidden_dims[0])

        # CNN blocks with increasing dimensions
        dims = list(config.prenet_hidden_dims)
        self.blocks = nn.ModuleList()

        for i in range(len(dims)):
            in_dim = dims[i-1] if i > 0 else dims[0]
            out_dim = dims[i]
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )

        # Add one more block to reach decoder attention dim
        self.output_proj = nn.Linear(dims[-1], config.decoder_attention_dim)
        self.norm = nn.LayerNorm(config.decoder_attention_dim)

    def forward(self, wavlm_features: torch.Tensor) -> torch.Tensor:
        """
        Process WavLM features for cross-attention.

        Args:
            wavlm_features: [batch, seq, wavlm_dim]

        Returns:
            [batch, seq, attention_dim]
        """
        # Project input
        x = self.input_proj(wavlm_features)  # [B, T, H]

        # CNN processing
        x = x.transpose(1, 2)  # [B, H, T]
        for block in self.blocks:
            residual = x
            x = block(x)
            # Scaled residual if dimensions match
            if x.shape == residual.shape:
                x = x + 0.5 * residual
        x = x.transpose(1, 2)  # [B, T, H]

        # Output projection
        x = self.output_proj(x)
        x = self.norm(x)

        return x


# =============================================================================
# CONFORMER DECODER
# =============================================================================

class PositionAgnosticCrossAttention(nn.Module):
    """
    Position-agnostic cross-attention for timbre injection.

    This ensures the reference prompt provides ONLY timbre information,
    not positional/temporal cues that could leak content.
    """

    def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,  # [B, T_q, D] from decoder
        key_value: torch.Tensor,  # [B, T_kv, D] from reference
    ) -> torch.Tensor:
        """
        Position-agnostic cross-attention.

        No positional encoding is added to key/value, ensuring
        the reference only provides timbre information.
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key_value.shape[1]

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # Reshape for multi-head attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores (no positional bias)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.dim)

        return self.out_proj(out)


class ConformerBlock(nn.Module):
    """Single Conformer block with position-agnostic cross-attention."""

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        dim = config.decoder_attention_dim

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, config.decoder_num_heads,
            dropout=config.dropout, batch_first=True
        )

        # Position-agnostic cross-attention for timbre
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = PositionAgnosticCrossAttention(
            dim, config.decoder_num_heads, config.dropout
        )

        # Convolution module
        self.norm3 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(config.dropout),
        )

        # Feed-forward
        self.norm4 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_ffn_dim, dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        timbre: torch.Tensor,  # [B, T_ref, D] reference timbre features
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = residual + x

        # Cross-attention to timbre (position-agnostic)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, timbre)
        x = residual + x

        # Convolution module
        residual = x
        x = self.norm3(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, T, D]
        x = residual + x

        # Feed-forward
        residual = x
        x = self.norm4(x)
        x = residual + self.ffn(x)

        return x


class ConformerDecoder(nn.Module):
    """
    Conformer Decoder for LSCodec.

    Reconstructs mel spectrogram and predicts SSL tokens from:
    - Latent z from encoder (content/prosody)
    - Timbre features from WavLM reference (speaker)
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.code_dim, config.decoder_attention_dim)

        # Positional encoding for decoder input (content has position)
        self.pos_encoding = self._create_positional_encoding(
            config.decoder_attention_dim, max_len=5000
        )

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(config)
            for _ in range(config.decoder_num_blocks)
        ])

        # Output heads
        # Mel prediction
        self.mel_head = nn.Linear(config.decoder_attention_dim, config.mel_dim)

        # SSL token prediction
        self.ssl_head = nn.Linear(config.decoder_attention_dim, config.ssl_num_clusters)

        self.norm = nn.LayerNorm(config.decoder_attention_dim)

    def _create_positional_encoding(self, dim: int, max_len: int = 5000) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, max_len, dim]

    def forward(
        self,
        z: torch.Tensor,  # [B, T, code_dim] latent from encoder
        timbre_features: torch.Tensor,  # [B, T_ref, D] from PromptPrenet
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to mel and SSL tokens.

        Returns:
            Dict with mel_pred, ssl_logits
        """
        # Project input
        x = self.input_proj(z)  # [B, T, D]

        # Add positional encoding
        pe = self.pos_encoding[:, :x.shape[1]].to(x.device)
        x = x + pe

        # Process through Conformer blocks
        for block in self.blocks:
            x = block(x, timbre_features, mask)

        x = self.norm(x)

        # Output predictions
        mel_pred = self.mel_head(x)
        ssl_logits = self.ssl_head(x)

        return {
            'mel_pred': mel_pred,
            'ssl_logits': ssl_logits,
            'decoder_features': x,
        }


# =============================================================================
# VECTOR QUANTIZER
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with EMA codebook updates.

    Used in Stage 2 to discretize the continuous VAE latent space.
    Includes codebook reinitialization for unused codes.
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.code_dim = config.code_dim
        self.ema_decay = config.ema_decay
        self.expiration_delay = config.code_expiration_delay

        # Codebook
        self.codebook = nn.Parameter(torch.randn(config.vocab_size, config.code_dim))
        nn.init.uniform_(self.codebook, -1.0 / config.vocab_size, 1.0 / config.vocab_size)

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(config.vocab_size))
        self.register_buffer('ema_sum', torch.randn(config.vocab_size, config.code_dim))
        self.register_buffer('code_age', torch.zeros(config.vocab_size))
        self.register_buffer('initialized', torch.tensor(False))

    def initialize_from_kmeans(self, data: torch.Tensor):
        """
        Initialize codebook from k-means on VAE means.

        This is the V-centroid initialization from the paper.
        """
        flat_data = data.reshape(-1, self.code_dim)
        n_samples = flat_data.shape[0]

        if n_samples >= self.vocab_size:
            # Simple k-means-like initialization
            # Select V random samples as initial centroids
            indices = torch.randperm(n_samples)[:self.vocab_size]
            self.codebook.data.copy_(flat_data[indices])
        else:
            # Repeat if not enough samples
            repeats = (self.vocab_size // n_samples) + 1
            expanded = flat_data.repeat(repeats, 1)[:self.vocab_size]
            self.codebook.data.copy_(expanded)

        self.ema_sum.data.copy_(self.codebook.data)
        self.ema_cluster_size.fill_(1.0)
        self.code_age.zero_()
        self.initialized.fill_(True)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Quantize continuous latent to discrete tokens.

        Args:
            z: [batch, seq, code_dim]

        Returns:
            Dict with z_q, indices, commitment_loss, perplexity
        """
        batch_size, seq_len, _ = z.shape
        flat_z = z.reshape(-1, self.code_dim)

        # Compute distances to codebook
        distances = (
            flat_z.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(flat_z, self.codebook.t())
            + self.codebook.pow(2).sum(dim=-1)
        )

        # Find nearest codes
        indices = distances.argmin(dim=-1)  # [B*T]
        z_q = F.embedding(indices, self.codebook)  # [B*T, code_dim]

        # EMA updates during training
        if self.training:
            with torch.no_grad():
                # Count usage
                encodings = F.one_hot(indices, self.vocab_size).float()
                new_cluster_size = encodings.sum(dim=0)
                new_cluster_sum = torch.matmul(encodings.t(), flat_z)

                # Update EMA
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    new_cluster_size, alpha=1 - self.ema_decay
                )
                self.ema_sum.mul_(self.ema_decay).add_(
                    new_cluster_sum, alpha=1 - self.ema_decay
                )

                # Update codebook
                n = self.ema_cluster_size.clamp(min=1)
                self.codebook.data.copy_(self.ema_sum / n.unsqueeze(-1))

                # Track code age (for reinitialization of unused codes)
                used_mask = new_cluster_size > 0
                self.code_age[used_mask] = 0
                self.code_age[~used_mask] += 1

                # Reinitialize expired codes
                expired_mask = self.code_age > self.expiration_delay
                if expired_mask.any():
                    n_expired = expired_mask.sum().item()
                    # Replace with random samples from current batch
                    random_indices = torch.randperm(flat_z.shape[0])[:n_expired]
                    self.codebook.data[expired_mask] = flat_z[random_indices]
                    self.code_age[expired_mask] = 0

        # Commitment loss
        commitment_loss = F.mse_loss(z_q.detach(), flat_z)

        # Straight-through estimator
        z_q = flat_z + (z_q - flat_z).detach()

        # Reshape
        z_q = z_q.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

        # Compute perplexity
        flat_indices = indices.view(-1)
        encodings = F.one_hot(flat_indices, self.vocab_size).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'z_q': z_q,
            'indices': indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to continuous vectors."""
        return F.embedding(indices, self.codebook)


# =============================================================================
# WAVLM TIMBRE EXTRACTOR (MOCK)
# =============================================================================

class WavLMTimbreExtractor(nn.Module):
    """
    WavLM-based timbre feature extractor.

    Extracts features from layer 6 of WavLM-Large for timbre information.

    Note: This is a mock implementation. For production, use:
    ```python
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    ```
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config
        self.target_layer = config.wavlm_layer

        # Mock feature extractor (replace with real WavLM)
        self.mock_extractor = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(1024, config.wavlm_dim, kernel_size=3, stride=2, padding=1),
        )

        # Feature projection
        self.proj = nn.Linear(config.wavlm_dim, config.wavlm_dim)
        self.norm = nn.LayerNorm(config.wavlm_dim)

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples] or [batch, 1, samples]
    ) -> torch.Tensor:
        """
        Extract WavLM features for timbre.

        Returns:
            [batch, seq, wavlm_dim]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]

        # Extract features (mock)
        features = self.mock_extractor(audio)  # [B, D, T']
        features = features.transpose(1, 2)  # [B, T', D]

        # Project and normalize
        features = self.proj(features)
        features = self.norm(features)

        return features

    def extract_from_mel(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
    ) -> torch.Tensor:
        """
        Extract pseudo-timbre features from mel spectrogram.

        Useful when only mel is available (training with precomputed features).
        """
        # Simple linear projection as approximation
        batch_size, seq_len, mel_dim = mel.shape

        # Learnable projection from mel to WavLM-like features
        if not hasattr(self, 'mel_proj'):
            self.mel_proj = nn.Linear(mel_dim, self.config.wavlm_dim).to(mel.device)

        features = self.mel_proj(mel)
        features = self.norm(features)

        return features


# =============================================================================
# TOKEN VOCODER
# =============================================================================

class TokenVocoder(nn.Module):
    """
    CTX-vec2wav-style Token Vocoder for LSCodec.

    Converts discrete LSCodec tokens + WavLM timbre to waveform.
    This is Stage 3 of the training pipeline.

    Note: Full implementation would use HiFi-GAN or similar.
    This is a simplified version for demonstration.
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.decoder_attention_dim)

        # Timbre processor (from WavLM)
        self.timbre_proj = nn.Linear(config.wavlm_dim, config.decoder_attention_dim)

        # Cross-attention for timbre injection
        self.cross_attn = PositionAgnosticCrossAttention(
            config.decoder_attention_dim,
            config.decoder_num_heads,
            config.dropout,
        )

        # Upsampling to waveform rate
        upsample_rates = [5, 4, 4, 2]  # Total: 160x for 50Hz tokens
        self.upsamples = nn.ModuleList()
        channels = config.decoder_attention_dim

        for rate in upsample_rates:
            self.upsamples.append(
                nn.Sequential(
                    nn.ConvTranspose1d(channels, channels // 2, rate * 2, rate, rate // 2),
                    nn.GELU(),
                    nn.Conv1d(channels // 2, channels // 2, 3, padding=1),
                    nn.GELU(),
                )
            )
            channels = channels // 2

        # Final output (to waveform)
        self.output_conv = nn.Conv1d(channels, 1, 7, padding=3)
        self.tanh = nn.Tanh()

    def forward(
        self,
        tokens: torch.Tensor,  # [batch, seq] discrete tokens
        timbre_features: torch.Tensor,  # [batch, seq_ref, wavlm_dim]
    ) -> torch.Tensor:
        """
        Generate waveform from tokens and timbre.

        Returns:
            [batch, 1, samples] audio waveform
        """
        # Embed tokens
        x = self.token_embed(tokens)  # [B, T, D]

        # Project timbre
        timbre = self.timbre_proj(timbre_features)  # [B, T_ref, D]

        # Cross-attention for timbre injection
        x = x + self.cross_attn(x, timbre)

        # Upsample to waveform rate
        x = x.transpose(1, 2)  # [B, D, T]
        for upsample in self.upsamples:
            x = upsample(x)

        # Output waveform
        audio = self.output_conv(x)  # [B, 1, samples]
        audio = self.tanh(audio)

        return audio


# =============================================================================
# FULL LSCODEC MODEL
# =============================================================================

class LSCodec(nn.Module):
    """
    LSCodec: Low-Bitrate and Speaker-Decoupled Discrete Speech Codec.

    Three-stage training:
    1. VAE: Train encoder-decoder with KL divergence and speaker perturbation
    2. VQ-VAE: Add vector quantization, replace KL with commitment loss
    3. Vocoder: Train token vocoder for waveform synthesis

    At inference:
    - Encode: mel → discrete speaker-decoupled tokens
    - Decode: tokens + reference speaker → mel/waveform
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config

        # Speaker perturbation
        self.perturbation = TimeStretchingPerturbation(config)

        # Encoder
        self.encoder = CNNEncoder(config)

        # Vector quantizer (used in Stage 2+)
        self.vq = VectorQuantizerEMA(config)

        # WavLM timbre extractor
        self.wavlm = WavLMTimbreExtractor(config)

        # Prompt prenet
        self.prenet = PromptPrenet(config)

        # Decoder
        self.decoder = ConformerDecoder(config)

        # Token vocoder (Stage 3)
        self.vocoder = TokenVocoder(config)

        # Track current training stage
        self.register_buffer('current_stage', torch.tensor(1))

    def set_stage(self, stage: int):
        """Set training stage (1=VAE, 2=VQ-VAE, 3=Vocoder)."""
        assert stage in [1, 2, 3]
        self.current_stage.fill_(stage)

        # Initialize VQ from VAE means when transitioning to Stage 2
        if stage == 2 and not self.vq.initialized:
            warnings.warn(
                "VQ not initialized. Call vq.initialize_from_kmeans() with "
                "VAE means before Stage 2 training."
            )

    def encode(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
        return_continuous: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode mel to discrete tokens.

        Args:
            mel: Input mel spectrogram
            return_continuous: Also return continuous VAE latent

        Returns:
            Dict with indices, and optionally z (continuous latent)
        """
        # CNN encoder
        enc_out = self.encoder(mel)
        z = enc_out['z']

        # Vector quantize
        vq_out = self.vq(z)

        result = {
            'indices': vq_out['indices'],
            'perplexity': vq_out['perplexity'],
        }

        if return_continuous:
            result['z'] = z
            result['mu'] = enc_out['mu']
            result['log_var'] = enc_out['log_var']

        return result

    def decode(
        self,
        indices: torch.Tensor,  # [batch, seq] discrete tokens
        reference_mel: torch.Tensor,  # [batch, seq_ref, mel_dim] reference speaker
    ) -> Dict[str, torch.Tensor]:
        """
        Decode tokens to mel using reference timbre.

        Args:
            indices: Discrete LSCodec tokens
            reference_mel: Reference mel for speaker timbre

        Returns:
            Dict with mel_pred, ssl_logits
        """
        # Get continuous vectors from tokens
        z_q = self.vq.decode_indices(indices)

        # Extract timbre from reference
        timbre_features = self.wavlm.extract_from_mel(reference_mel)
        timbre_processed = self.prenet(timbre_features)

        # Decode
        dec_out = self.decoder(z_q, timbre_processed)

        return dec_out

    def forward_vae(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim] content mel (perturbed)
        reference_mel: torch.Tensor,  # [batch, seq_ref, mel_dim] reference (original)
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1: VAE forward pass.

        Args:
            mel: Perturbed content mel spectrogram
            reference_mel: Original (unperturbed) mel for timbre

        Returns:
            Dict with all VAE outputs
        """
        # Encode
        enc_out = self.encoder(mel)
        z = enc_out['z']  # Sampled latent

        # Extract timbre from reference
        timbre_features = self.wavlm.extract_from_mel(reference_mel)
        timbre_processed = self.prenet(timbre_features)

        # Decode
        dec_out = self.decoder(z, timbre_processed)

        return {
            'mu': enc_out['mu'],
            'log_var': enc_out['log_var'],
            'z': z,
            'mel_pred': dec_out['mel_pred'],
            'ssl_logits': dec_out['ssl_logits'],
        }

    def forward_vqvae(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim] content mel (perturbed)
        reference_mel: torch.Tensor,  # [batch, seq_ref, mel_dim] reference
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2: VQ-VAE forward pass.

        Args:
            mel: Perturbed content mel spectrogram
            reference_mel: Original mel for timbre

        Returns:
            Dict with all VQ-VAE outputs including tokens
        """
        # Encode to continuous
        enc_out = self.encoder(mel)
        z = enc_out['z']

        # Vector quantize
        vq_out = self.vq(z)
        z_q = vq_out['z_q']

        # Extract timbre from reference
        timbre_features = self.wavlm.extract_from_mel(reference_mel)
        timbre_processed = self.prenet(timbre_features)

        # Decode
        dec_out = self.decoder(z_q, timbre_processed)

        return {
            'mu': enc_out['mu'],
            'log_var': enc_out['log_var'],
            'z': z,
            'z_q': z_q,
            'indices': vq_out['indices'],
            'commitment_loss': vq_out['commitment_loss'],
            'perplexity': vq_out['perplexity'],
            'mel_pred': dec_out['mel_pred'],
            'ssl_logits': dec_out['ssl_logits'],
        }

    def forward(
        self,
        mel: torch.Tensor,
        reference_mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass based on current training stage.
        """
        stage = self.current_stage.item()

        if stage == 1:
            return self.forward_vae(mel, reference_mel)
        else:
            return self.forward_vqvae(mel, reference_mel)

    def voice_convert(
        self,
        source_mel: torch.Tensor,  # [batch, seq, mel_dim] content source
        target_mel: torch.Tensor,  # [batch, seq_ref, mel_dim] target speaker
    ) -> torch.Tensor:
        """
        Zero-shot voice conversion.

        Encodes source mel to speaker-decoupled tokens, then decodes
        with target speaker's timbre.

        Args:
            source_mel: Source audio mel (provides content)
            target_mel: Target speaker mel (provides timbre)

        Returns:
            Converted mel spectrogram
        """
        # Encode source to tokens
        enc_result = self.encode(source_mel)
        tokens = enc_result['indices']

        # Decode with target timbre
        dec_result = self.decode(tokens, target_mel)

        return dec_result['mel_pred']

    def generate_waveform(
        self,
        tokens: torch.Tensor,  # [batch, seq] discrete tokens
        reference_audio: torch.Tensor,  # [batch, samples] reference speaker
    ) -> torch.Tensor:
        """
        Generate waveform from tokens (Stage 3).

        Args:
            tokens: Discrete LSCodec tokens
            reference_audio: Reference audio for timbre

        Returns:
            [batch, 1, samples] synthesized waveform
        """
        # Extract timbre from reference audio
        timbre_features = self.wavlm(reference_audio)

        # Generate waveform
        audio = self.vocoder(tokens, timbre_features)

        return audio

    def compute_bitrate(self) -> float:
        """Compute codec bitrate in kbps."""
        frame_rate = self.config.frame_rate
        vocab_size = self.config.vocab_size

        # Q × F × ceil(log2(V))
        bits_per_second = frame_rate * math.ceil(math.log2(vocab_size))
        kbps = bits_per_second / 1000

        return kbps


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class LSCodecLoss(nn.Module):
    """
    Loss functions for LSCodec three-stage training.

    Stage 1 (VAE):
        γKL * KL + γrecon * L1_recon + γidx * CE_ssl

    Stage 2 (VQ-VAE):
        γcmt * commitment + γrecon * L1_recon + γidx * CE_ssl

    Stage 3 (Vocoder):
        Multi-resolution STFT loss + adversarial loss
    """

    def __init__(self, config: LSCodecConfig):
        super().__init__()
        self.config = config

    def kl_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence to standard normal.

        KL = (1/T) Σ DKL(N(μt, σt²) || N(0, 1))
        """
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kl.mean()

    def reconstruction_loss(
        self,
        mel_pred: torch.Tensor,
        mel_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """L1 reconstruction loss on mel spectrogram."""
        # Handle length mismatch
        min_len = min(mel_pred.shape[1], mel_target.shape[1])
        mel_pred = mel_pred[:, :min_len]
        mel_target = mel_target[:, :min_len]

        loss = F.l1_loss(mel_pred, mel_target, reduction='none')

        if mask is not None:
            mask = mask[:, :min_len].unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def ssl_token_loss(
        self,
        ssl_logits: torch.Tensor,
        ssl_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-entropy loss for SSL token prediction."""
        # Handle length mismatch
        min_len = min(ssl_logits.shape[1], ssl_target.shape[1])
        ssl_logits = ssl_logits[:, :min_len]
        ssl_target = ssl_target[:, :min_len]

        # ssl_target should be token indices [batch, seq]
        if ssl_target.dim() == 3:
            # If given as embeddings, would need k-means assignment
            # For simplicity, assume indices are provided
            ssl_target = ssl_target.argmax(dim=-1)

        loss = F.cross_entropy(
            ssl_logits.reshape(-1, ssl_logits.shape[-1]),
            ssl_target.reshape(-1).long(),
            reduction='none',
        )
        loss = loss.view(ssl_logits.shape[:2])

        if mask is not None:
            mask = mask[:, :min_len]
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def compute_vae_loss(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        ssl_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 1 VAE loss.

        Args:
            output: Output from forward_vae()
            mel_target: Target mel (unperturbed)
            ssl_target: Target SSL token indices

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # KL divergence
        losses['kl'] = self.kl_loss(output['mu'], output['log_var'])

        # Reconstruction
        losses['recon'] = self.reconstruction_loss(
            output['mel_pred'], mel_target, mask
        )

        # SSL token prediction
        losses['ssl'] = self.ssl_token_loss(
            output['ssl_logits'], ssl_target, mask
        )

        # Total loss
        losses['total'] = (
            self.config.gamma_kl * losses['kl']
            + self.config.gamma_recon * losses['recon']
            + self.config.gamma_idx * losses['ssl']
        )

        return losses

    def compute_vqvae_loss(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
        ssl_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 2 VQ-VAE loss.

        Args:
            output: Output from forward_vqvae()
            mel_target: Target mel
            ssl_target: Target SSL token indices

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Commitment loss (replaces KL)
        losses['commitment'] = output['commitment_loss']
        losses['perplexity'] = output['perplexity']

        # Reconstruction
        losses['recon'] = self.reconstruction_loss(
            output['mel_pred'], mel_target, mask
        )

        # SSL token prediction
        losses['ssl'] = self.ssl_token_loss(
            output['ssl_logits'], ssl_target, mask
        )

        # Total loss
        losses['total'] = (
            self.config.gamma_cmt * losses['commitment']
            + self.config.gamma_recon * losses['recon']
            + self.config.gamma_idx * losses['ssl']
        )

        return losses


# =============================================================================
# CSM ADAPTER
# =============================================================================

class LSCodecAdapter(nn.Module):
    """
    Adapter for integrating LSCodec with CSM prosody pipeline.

    Converts LSCodec's speaker-decoupled tokens to prefix tokens
    compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: LSCodecConfig,
        model: Optional[LSCodec] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else LSCodec(config)

        # Project VQ codes to output dim
        self.embed_proj = nn.Sequential(
            nn.Linear(config.code_dim, config.output_dim // 2),
            nn.GELU(),
            nn.Linear(config.output_dim // 2, config.output_dim),
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim // 4),
            nn.Tanh(),
            nn.Linear(config.output_dim // 4, 1),
        )

        # Token projection
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )

        self.norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, seq, mel_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix tokens from LSCodec encoding.

        Returns:
            Dict with:
                - prosody_tokens: [batch, num_prefix_tokens, output_dim]
                - indices: [batch, seq] discrete tokens
                - perplexity: codebook usage
        """
        batch_size = mel.shape[0]

        # Encode to tokens
        enc_result = self.model.encode(mel, return_continuous=True)
        z = enc_result['z']  # [B, T, code_dim]

        # Project to output dim
        features = self.embed_proj(z)  # [B, T, output_dim]

        # Attention pooling
        attn_weights = self.attention(features)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (features * attn_weights).sum(dim=1)  # [B, output_dim]

        # Project to tokens
        tokens = self.token_proj(pooled)  # [B, output_dim * num_tokens]
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
            'indices': enc_result['indices'],
            'perplexity': enc_result['perplexity'],
            'z': z,
        }

    def from_tokens(
        self,
        indices: torch.Tensor,  # [batch, seq]
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix from pre-computed tokens.

        Useful when tokens are already extracted or for generation.
        """
        batch_size = indices.shape[0]

        # Decode tokens to continuous
        z = self.model.vq.decode_indices(indices)  # [B, T, code_dim]

        # Project
        features = self.embed_proj(z)

        # Attention pooling
        attn_weights = F.softmax(self.attention(features), dim=1)
        pooled = (features * attn_weights).sum(dim=1)

        # Project to tokens
        tokens = self.token_proj(pooled)
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
        }

    def voice_convert(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-shot voice conversion."""
        return self.model.voice_convert(source_mel, target_mel)


# =============================================================================
# UTILITIES
# =============================================================================

def create_lscodec_50hz(pretrained: Optional[str] = None) -> LSCodec:
    """Create LSCodec-50Hz (0.45 kbps)."""
    config = LSCodecConfig(
        frame_rate=50,
        vocab_size=300,
        encoder_num_blocks=11,
        hop_length=320,
    )
    model = LSCodec(config)

    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


def create_lscodec_25hz(pretrained: Optional[str] = None) -> LSCodec:
    """Create LSCodec-25Hz (0.25 kbps)."""
    config = LSCodecConfig(
        frame_rate=25,
        vocab_size=1024,
        encoder_num_blocks=12,
        hop_length=640,
    )
    model = LSCodec(config)

    if pretrained:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LSCodec: Low-Bitrate Speaker-Decoupled Speech Codec")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    seq_len = 100
    ref_len = 50
    mel_dim = 80

    # Test both configurations
    for frame_rate, vocab_size in [(50, 300), (25, 1024)]:
        print(f"\n{'='*60}")
        print(f"Testing LSCodec-{frame_rate}Hz (V={vocab_size})")
        print("=" * 60)

        config = LSCodecConfig(
            frame_rate=frame_rate,
            vocab_size=vocab_size,
            encoder_num_blocks=11 if frame_rate == 50 else 12,
        )

        # Create dummy inputs
        mel = torch.randn(batch_size, seq_len, mel_dim).to(device)
        reference_mel = torch.randn(batch_size, ref_len, mel_dim).to(device)
        ssl_target = torch.randint(0, config.ssl_num_clusters, (batch_size, seq_len)).to(device)

        # Test 1: Configuration
        print(f"\n[Test 1] Configuration...")
        print(f"  Frame rate: {config.frame_rate} Hz")
        print(f"  Vocabulary: {config.vocab_size}")
        bitrate = config.frame_rate * math.ceil(math.log2(config.vocab_size)) / 1000
        print(f"  Bitrate: {bitrate:.2f} kbps")
        print("  [PASS]")

        # Test 2: Time Stretching Perturbation
        print(f"\n[Test 2] Time Stretching Perturbation...")
        perturb = TimeStretchingPerturbation(config).to(device)
        perturbed_mel, beta = perturb.perturb_mel(mel)
        print(f"  Input shape: {mel.shape}")
        print(f"  Output shape: {perturbed_mel.shape}")
        print(f"  Beta: {beta:.3f}")
        print("  [PASS]")

        # Test 3: CNN Encoder
        print(f"\n[Test 3] CNN Encoder...")
        encoder = CNNEncoder(config).to(device)
        enc_out = encoder(mel)
        print(f"  μ shape: {enc_out['mu'].shape}")
        print(f"  log_var shape: {enc_out['log_var'].shape}")
        print(f"  z shape: {enc_out['z'].shape}")
        print("  [PASS]")

        # Test 4: Vector Quantizer
        print(f"\n[Test 4] Vector Quantizer...")
        vq = VectorQuantizerEMA(config).to(device)
        # Initialize from encoder output
        vq.initialize_from_kmeans(enc_out['mu'])
        vq_out = vq(enc_out['z'])
        print(f"  z_q shape: {vq_out['z_q'].shape}")
        print(f"  indices shape: {vq_out['indices'].shape}")
        print(f"  Perplexity: {vq_out['perplexity'].item():.2f}")
        print("  [PASS]")

        # Test 5: WavLM Extractor
        print(f"\n[Test 5] WavLM Timbre Extractor...")
        wavlm = WavLMTimbreExtractor(config).to(device)
        timbre = wavlm.extract_from_mel(reference_mel)
        print(f"  Timbre shape: {timbre.shape}")
        print("  [PASS]")

        # Test 6: Prompt Prenet
        print(f"\n[Test 6] Prompt Prenet...")
        prenet = PromptPrenet(config).to(device)
        timbre_proc = prenet(timbre)
        print(f"  Processed timbre shape: {timbre_proc.shape}")
        print("  [PASS]")

        # Test 7: Conformer Decoder
        print(f"\n[Test 7] Conformer Decoder...")
        decoder = ConformerDecoder(config).to(device)
        dec_out = decoder(vq_out['z_q'], timbre_proc)
        print(f"  mel_pred shape: {dec_out['mel_pred'].shape}")
        print(f"  ssl_logits shape: {dec_out['ssl_logits'].shape}")
        print("  [PASS]")

        # Test 8: Full Model - Stage 1 (VAE)
        print(f"\n[Test 8] Full Model - Stage 1 (VAE)...")
        model = LSCodec(config).to(device)
        model.set_stage(1)
        output = model.forward_vae(perturbed_mel, reference_mel)
        print(f"  mu shape: {output['mu'].shape}")
        print(f"  mel_pred shape: {output['mel_pred'].shape}")
        print("  [PASS]")

        # Test 9: Loss - Stage 1
        print(f"\n[Test 9] Loss - Stage 1 (VAE)...")
        loss_fn = LSCodecLoss(config)
        losses = loss_fn.compute_vae_loss(output, mel, ssl_target)
        print(f"  KL loss: {losses['kl'].item():.4f}")
        print(f"  Recon loss: {losses['recon'].item():.4f}")
        print(f"  SSL loss: {losses['ssl'].item():.4f}")
        print(f"  Total: {losses['total'].item():.4f}")
        print("  [PASS]")

        # Test 10: Full Model - Stage 2 (VQ-VAE)
        print(f"\n[Test 10] Full Model - Stage 2 (VQ-VAE)...")
        model.vq.initialize_from_kmeans(output['mu'])
        model.set_stage(2)
        output = model.forward_vqvae(perturbed_mel, reference_mel)
        print(f"  indices shape: {output['indices'].shape}")
        print(f"  perplexity: {output['perplexity'].item():.2f}")
        print("  [PASS]")

        # Test 11: Loss - Stage 2
        print(f"\n[Test 11] Loss - Stage 2 (VQ-VAE)...")
        losses = loss_fn.compute_vqvae_loss(output, mel, ssl_target)
        print(f"  Commitment loss: {losses['commitment'].item():.4f}")
        print(f"  Recon loss: {losses['recon'].item():.4f}")
        print(f"  SSL loss: {losses['ssl'].item():.4f}")
        print(f"  Total: {losses['total'].item():.4f}")
        print("  [PASS]")

        # Test 12: Encode/Decode
        print(f"\n[Test 12] Encode/Decode...")
        with torch.no_grad():
            enc_result = model.encode(mel)
            dec_result = model.decode(enc_result['indices'], reference_mel)
        print(f"  Encoded tokens: {enc_result['indices'].shape}")
        print(f"  Decoded mel: {dec_result['mel_pred'].shape}")
        print("  [PASS]")

        # Test 13: Voice Conversion
        print(f"\n[Test 13] Voice Conversion...")
        source_mel = torch.randn(1, seq_len, mel_dim).to(device)
        target_mel = torch.randn(1, ref_len, mel_dim).to(device)
        with torch.no_grad():
            converted = model.voice_convert(source_mel, target_mel)
        print(f"  Source: {source_mel.shape}")
        print(f"  Target speaker: {target_mel.shape}")
        print(f"  Converted: {converted.shape}")
        print("  [PASS]")

        # Test 14: CSM Adapter
        print(f"\n[Test 14] CSM Adapter...")
        adapter = LSCodecAdapter(config, model).to(device)
        adapter_out = adapter(mel)
        print(f"  Prosody tokens: {adapter_out['prosody_tokens'].shape}")
        assert adapter_out['prosody_tokens'].shape == (
            batch_size, config.num_prefix_tokens, config.output_dim
        )
        print("  [PASS]")

        # Test 15: Backward Pass
        print(f"\n[Test 15] Backward Pass...")
        model.train()
        model.zero_grad()
        output = model.forward_vqvae(perturbed_mel, reference_mel)
        losses = loss_fn.compute_vqvae_loss(output, mel, ssl_target)
        losses['total'].backward()
        grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )
        print(f"  Total gradient norm: {grad_norm:.4f}")
        print("  [PASS]")

    print("\n" + "=" * 60)
    print("All LSCodec tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. TIME STRETCHING PERTURBATION:
       - Speed up by β ∈ [0.8, 1.2] to alter pitch/timbre
       - WSOLA restoration preserves content/prosody
       - Creates time-aligned perturbed training pairs

    2. THREE-STAGE TRAINING:
       - Stage 1 (VAE): KL + recon + SSL token prediction
       - Stage 2 (VQ-VAE): Commitment + recon + SSL prediction
       - Stage 3 (Vocoder): Token → waveform synthesis

    3. SPEAKER DISENTANGLEMENT:
       - Continuous bottleneck (VAE) before discretization
       - Position-agnostic cross-attention for timbre
       - Reference provides ONLY timbre information

    4. ULTRA-LOW BITRATE:
       - LSCodec-50Hz: V=300, 0.45 kbps
       - LSCodec-25Hz: V=1024, 0.25 kbps
       - Single codebook (vs 6+ in FACodec)

    5. ZERO-SHOT VOICE CONVERSION:
       - Encode source → speaker-decoupled tokens
       - Decode with target speaker's timbre
       - No paired training data needed
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from lscodec import (
    LSCodecConfig,
    LSCodec,
    LSCodecLoss,
    LSCodecAdapter,
    create_lscodec_50hz,
    create_lscodec_25hz,
)

# Option 1: Create specific configuration
config = LSCodecConfig(frame_rate=50, vocab_size=300)
model = LSCodec(config).cuda()

# Option 2: Use factory function
model = create_lscodec_50hz().cuda()  # 0.45 kbps
# model = create_lscodec_25hz().cuda()  # 0.25 kbps

loss_fn = LSCodecLoss(config)

# Stage 1: Train VAE
model.set_stage(1)
for mel, reference_mel, ssl_target in dataloader:
    # Apply speaker perturbation
    perturbed_mel, beta = model.perturbation.perturb_mel(mel)

    output = model.forward_vae(perturbed_mel, reference_mel)
    losses = loss_fn.compute_vae_loss(output, mel, ssl_target)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

# Stage 2: Train VQ-VAE (after VAE converges)
# Initialize VQ from VAE means
with torch.no_grad():
    all_means = []
    for mel, *_ in dataloader:
        enc = model.encoder(mel)
        all_means.append(enc['mu'])
    all_means = torch.cat(all_means, dim=0)
    model.vq.initialize_from_kmeans(all_means)

model.set_stage(2)
for mel, reference_mel, ssl_target in dataloader:
    perturbed_mel, beta = model.perturbation.perturb_mel(mel)

    output = model.forward_vqvae(perturbed_mel, reference_mel)
    losses = loss_fn.compute_vqvae_loss(output, mel, ssl_target)

    losses['total'].backward()
    optimizer.step()

# Inference: Encode to speaker-decoupled tokens
tokens = model.encode(mel)['indices']

# Voice Conversion
converted_mel = model.voice_convert(source_mel, target_speaker_mel)

# CSM Integration
adapter = LSCodecAdapter(config, model)
prefix_tokens = adapter(mel)['prosody_tokens']  # [batch, 4, 2048]

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
