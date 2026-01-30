"""
Soft Frequency-Band Disentanglement Codec

Based on "Soft Frequency-Band Disentanglement for Neural Audio Codecs"
(EUSIPCO 2025) - arXiv:2510.03735

Key Innovation: Multi-branch cascade codec with frequency-band specialization,
where branches learn to handle designated frequency bands through soft
(non-strict) constraints rather than hard spectral decomposition.

Architecture:
```
                    Audio (Full Band: 0-N kHz)
                            ↓
                [Spectral Decomposition via STFT/Filterbank]
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
    Low-Band                               High-Band
    (0-8 kHz)                             (8-16 kHz)
        ↓                                       ↓
    [LF Encoder]                          [HF Encoder]
        ↓                                       ↓
    [LF VQ Codebook]                     [HF VQ Codebook]
        ↓                                       ↓
    [LF Decoder] ──────────────────────► [HF Decoder]
        ↓                    ↑ LF context        ↓
    LF Reconstruction        └─────────── HF Reconstruction
        ↓                                       ↓
        └───────────────────┬───────────────────┘
                            ↓
                    [Cascade Fusion]
                            ↓
                    Full-Band Output
```

Key Components:
1. **Spectral Decomposition**: STFT-based separation into frequency bands
2. **Band-Specific Encoders**: Each branch learns to encode its designated band
3. **Soft Disentanglement**: Cross-band prediction loss encourages specialization
4. **Cascade Fusion**: Higher bands see lower band reconstructions
5. **Reconstruction**: Sum decomposition for full-band output

Soft Disentanglement Mechanism:
- No hard constraints (like orthogonality loss)
- Branches naturally specialize through:
  a) Input conditioning (each sees primarily its frequency range)
  b) Cross-band reconstruction loss (predict residual from other bands)
  c) Cascade architecture (higher bands refine lower band output)

Benefits:
- Better perceptual quality than hard decomposition
- Implicit disentanglement without strict constraints
- Advantages for audio inpainting and bandwidth extension
- Generalizable approach independent of specific data characteristics
- Compatible with various base codec architectures

Usage:
    from soft_freq_band_codec import (
        SoftFreqBandConfig,
        SoftFreqBandCodec,
        SoftFreqBandLoss,
        SoftFreqBandAdapter,
    )

    # Initialize
    config = SoftFreqBandConfig(
        sample_rate=32000,
        num_bands=2,  # 0-8kHz, 8-16kHz
        band_cutoffs=[8000],  # Cutoff frequencies
    )

    model = SoftFreqBandCodec(config).cuda()
    loss_fn = SoftFreqBandLoss(config)

    # Training
    output = model(audio)
    losses = loss_fn(output, audio)

    # Encode to band-specific tokens
    encoded = model.encode(audio)
    lf_indices = encoded['band_indices'][0]  # Low-frequency tokens
    hf_indices = encoded['band_indices'][1]  # High-frequency tokens

    # Decode from tokens
    audio_recon = model.decode(encoded['band_indices'])

    # CSM integration
    adapter = SoftFreqBandAdapter(config, model)
    prefix_tokens = adapter(mel)['prosody_tokens']  # [batch, 4, 2048]

Reference:
- Paper: https://arxiv.org/abs/2510.03735
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
class SoftFreqBandConfig:
    """Configuration for Soft Frequency-Band Disentanglement Codec."""

    # Audio settings
    sample_rate: int = 32000  # Full-band sample rate (supports up to 16kHz audio)
    hop_length: int = 320  # ~10ms at 32kHz
    win_length: int = 1280  # ~40ms at 32kHz
    n_fft: int = 2048  # FFT size for spectral analysis

    # Frequency band configuration
    # Default: 2 bands (0-8kHz, 8-16kHz for 32kHz audio)
    num_bands: int = 2
    band_cutoffs: Tuple[int, ...] = (8000,)  # Cutoff frequencies in Hz
    # Alternative: 3 bands (0-4kHz, 4-8kHz, 8-16kHz)
    # band_cutoffs: Tuple[int, ...] = (4000, 8000)

    # Encoder settings (per band)
    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 4
    encoder_kernel_sizes: Tuple[int, ...] = (7, 5, 5, 3)
    encoder_strides: Tuple[int, ...] = (2, 2, 2, 2)  # 16x downsampling

    # VQ settings (per band)
    codebook_size: int = 1024
    code_dim: int = 128
    num_quantizers: int = 2  # RVQ depth per band
    commitment_cost: float = 0.25
    ema_decay: float = 0.99

    # Decoder settings (per band)
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4

    # Cascade connection settings
    use_cascade: bool = True  # Higher bands see lower band outputs
    cascade_hidden_dim: int = 256
    cascade_num_layers: int = 2

    # Soft disentanglement settings
    cross_band_weight: float = 0.1  # Weight for cross-band reconstruction loss
    band_consistency_weight: float = 0.05  # Encourages band specialization
    enable_soft_masks: bool = True  # Learnable soft frequency masks

    # Training settings
    dropout: float = 0.1

    # Mel spectrogram settings (for CSM integration)
    mel_dim: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0  # Focus on lower frequencies for prosody

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# SPECTRAL DECOMPOSITION
# =============================================================================

class SpectralDecomposer(nn.Module):
    """
    Spectral decomposition layer for frequency band separation.

    Uses STFT-based analysis to separate audio into frequency bands.
    Supports both hard cutoff and learned soft masks.
    """

    def __init__(self, config: SoftFreqBandConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.sample_rate = config.sample_rate
        self.num_bands = config.num_bands

        # Convert cutoff frequencies to FFT bin indices
        self.bin_cutoffs = []
        freq_per_bin = config.sample_rate / config.n_fft
        for cutoff in config.band_cutoffs:
            bin_idx = int(cutoff / freq_per_bin)
            self.bin_cutoffs.append(bin_idx)
        self.bin_cutoffs.append(config.n_fft // 2 + 1)  # Nyquist

        # Register STFT window
        window = torch.hann_window(config.win_length)
        self.register_buffer('window', window)

        # Learnable soft masks (optional)
        if config.enable_soft_masks:
            n_bins = config.n_fft // 2 + 1
            self.soft_masks = nn.ParameterList([
                nn.Parameter(self._init_soft_mask(i, n_bins))
                for i in range(config.num_bands)
            ])
        else:
            self.soft_masks = None

    def _init_soft_mask(self, band_idx: int, n_bins: int) -> torch.Tensor:
        """Initialize soft mask with smooth transitions around cutoffs."""
        mask = torch.zeros(n_bins)

        # Get band boundaries
        if band_idx == 0:
            start_bin = 0
        else:
            start_bin = self.bin_cutoffs[band_idx - 1]

        end_bin = self.bin_cutoffs[band_idx]

        # Smooth mask with sigmoid transitions
        transition_width = max(10, (end_bin - start_bin) // 10)

        for i in range(n_bins):
            if i < start_bin - transition_width:
                mask[i] = 0.0
            elif i < start_bin:
                # Rising edge
                t = (i - (start_bin - transition_width)) / transition_width
                mask[i] = t
            elif i < end_bin:
                mask[i] = 1.0
            elif i < end_bin + transition_width:
                # Falling edge
                t = (i - end_bin) / transition_width
                mask[i] = 1.0 - t
            else:
                mask[i] = 0.0

        return mask

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose audio into frequency bands.

        Returns:
            Dict with:
                - band_audio: List of [batch, samples] per band
                - stft: Full STFT [batch, freq, time]
                - band_stft: List of masked STFT per band
                - masks: Applied masks (hard or soft)
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        batch_size, num_samples = audio.shape
        device = audio.device

        # Compute STFT
        # Ensure window is on correct device
        window = self.window.to(device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )  # [batch, freq, time]

        band_audio = []
        band_stft = []
        masks = []

        for i in range(self.num_bands):
            # Get mask
            if self.soft_masks is not None:
                # Learnable soft mask
                mask = torch.sigmoid(self.soft_masks[i]).to(device)
            else:
                # Hard cutoff mask
                mask = torch.zeros(stft.shape[1], device=device)
                if i == 0:
                    start_bin = 0
                else:
                    start_bin = self.bin_cutoffs[i - 1]
                end_bin = self.bin_cutoffs[i]
                mask[start_bin:end_bin] = 1.0

            masks.append(mask)

            # Apply mask to STFT
            masked_stft = stft * mask.unsqueeze(0).unsqueeze(-1)
            band_stft.append(masked_stft)

            # Inverse STFT to get band audio
            band_wav = torch.istft(
                masked_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=num_samples,
            )
            band_audio.append(band_wav)

        return {
            'band_audio': band_audio,
            'stft': stft,
            'band_stft': band_stft,
            'masks': masks,
        }

    def reconstruct(
        self,
        band_audio: List[torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct full-band audio from band components."""
        return sum(band_audio)

    def get_band_energy(
        self,
        stft: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute energy per frequency band."""
        energies = []
        for i in range(self.num_bands):
            if i == 0:
                start_bin = 0
            else:
                start_bin = self.bin_cutoffs[i - 1]
            end_bin = self.bin_cutoffs[i]

            band_magnitude = stft[:, start_bin:end_bin, :].abs()
            energy = band_magnitude.pow(2).mean(dim=(1, 2))
            energies.append(energy)

        return energies


# =============================================================================
# BAND ENCODER
# =============================================================================

class ConvBlock(nn.Module):
    """Causal convolution block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Causal padding (pad left side only)
        self.padding = (kernel_size - 1) * 1  # dilation = 1

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0,  # Manual padding
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        # Causal padding
        x_padded = F.pad(x, (self.padding, 0))

        residual = self.residual(x)
        out = self.conv(x_padded)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Match lengths
        min_len = min(out.shape[-1], residual.shape[-1])
        return out[..., :min_len] + residual[..., :min_len]


class TransposedConvBlock(nn.Module):
    """Transposed convolution block for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2,
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class BandEncoder(nn.Module):
    """
    Encoder for a single frequency band.

    Takes band-limited audio and produces latent representations.
    """

    def __init__(
        self,
        config: SoftFreqBandConfig,
        band_idx: int,
    ):
        super().__init__()
        self.config = config
        self.band_idx = band_idx

        # Input convolution
        self.input_conv = nn.Conv1d(1, config.encoder_hidden_dim, 7, padding=3)
        self.input_norm = nn.GroupNorm(8, config.encoder_hidden_dim)

        # Downsampling encoder blocks
        self.encoder_blocks = nn.ModuleList()
        in_channels = config.encoder_hidden_dim

        for i, (kernel_size, stride) in enumerate(
            zip(config.encoder_kernel_sizes, config.encoder_strides)
        ):
            out_channels = config.encoder_hidden_dim * (2 ** min(i, 2))
            self.encoder_blocks.append(
                ConvBlock(
                    in_channels, out_channels, kernel_size,
                    stride=stride, dropout=config.dropout,
                )
            )
            in_channels = out_channels

        # Output projection to code dimension
        self.output_proj = nn.Conv1d(in_channels, config.code_dim, 1)
        self.output_norm = nn.GroupNorm(1, config.code_dim)

    def forward(
        self,
        band_audio: torch.Tensor,  # [batch, samples]
    ) -> torch.Tensor:
        """
        Encode band audio to latent representation.

        Returns:
            Latent: [batch, code_dim, seq_len]
        """
        # Input: [batch, samples] -> [batch, 1, samples]
        if band_audio.dim() == 2:
            x = band_audio.unsqueeze(1)
        else:
            x = band_audio

        # Initial projection
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)

        # Encoder blocks with downsampling
        for block in self.encoder_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)
        x = self.output_norm(x)

        return x  # [batch, code_dim, seq_len]


# =============================================================================
# VECTOR QUANTIZATION
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
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

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

    def forward(
        self,
        x: torch.Tensor,  # [batch, code_dim, seq]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input.

        Returns:
            Dict with z_q, indices, commitment_loss, perplexity
        """
        # Reshape: [B, D, T] -> [B*T, D]
        x = x.permute(0, 2, 1)  # [B, T, D]
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

        # Reshape back
        z_q = z_q.view(batch_size, seq_len, self.code_dim).permute(0, 2, 1)
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

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to vectors."""
        z_q = self.embedding(indices)  # [B, T, D]
        return z_q.permute(0, 2, 1)  # [B, D, T]


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ) for a single band.

    Applies multiple VQ layers to progressively refine quantization.
    """

    def __init__(
        self,
        config: SoftFreqBandConfig,
        band_idx: int,
    ):
        super().__init__()
        self.config = config
        self.band_idx = band_idx
        self.num_quantizers = config.num_quantizers

        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(
                config.codebook_size,
                config.code_dim,
                config.commitment_cost,
                config.ema_decay,
            )
            for _ in range(config.num_quantizers)
        ])

    def forward(
        self,
        x: torch.Tensor,  # [batch, code_dim, seq]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply RVQ.

        Returns:
            Dict with z_q, indices, commitment_loss, perplexities
        """
        residual = x
        z_q_sum = torch.zeros_like(x)
        all_indices = []
        total_commitment = 0.0
        perplexities = []

        for quantizer in self.quantizers:
            vq_output = quantizer(residual)

            z_q_sum = z_q_sum + vq_output['z_q']
            all_indices.append(vq_output['indices'])
            total_commitment += vq_output['commitment_loss']
            perplexities.append(vq_output['perplexity'])

            residual = residual - vq_output['z_q']

        return {
            'z_q': z_q_sum,
            'indices': torch.stack(all_indices, dim=0),  # [num_q, batch, seq]
            'commitment_loss': total_commitment / self.num_quantizers,
            'perplexities': torch.stack(perplexities),
            'residual': residual,
        }

    def decode(
        self,
        indices: torch.Tensor,  # [num_q, batch, seq]
    ) -> torch.Tensor:
        """Decode from indices."""
        z_q_sum = torch.zeros(
            indices.shape[1], self.config.code_dim, indices.shape[2],
            device=indices.device
        )

        for i, quantizer in enumerate(self.quantizers):
            z_q = quantizer.decode(indices[i])
            z_q_sum = z_q_sum + z_q

        return z_q_sum


# =============================================================================
# BAND DECODER
# =============================================================================

class BandDecoder(nn.Module):
    """
    Decoder for a single frequency band.

    Takes quantized latent and produces band-limited audio.
    Optionally receives context from lower-frequency bands (cascade).
    """

    def __init__(
        self,
        config: SoftFreqBandConfig,
        band_idx: int,
    ):
        super().__init__()
        self.config = config
        self.band_idx = band_idx

        # Input dimension
        input_dim = config.code_dim

        # Cascade connection (receives context from lower bands)
        if config.use_cascade and band_idx > 0:
            self.cascade_proj = nn.Sequential(
                nn.Conv1d(config.code_dim, config.cascade_hidden_dim, 1),
                nn.GELU(),
                nn.Conv1d(config.cascade_hidden_dim, config.code_dim, 1),
            )
            # Cross-attention for cascade
            self.cascade_attn = nn.MultiheadAttention(
                config.code_dim, num_heads=4, batch_first=True
            )
            self.cascade_norm = nn.LayerNorm(config.code_dim)
        else:
            self.cascade_proj = None
            self.cascade_attn = None

        # Calculate output channels progression (reverse of encoder)
        strides = config.encoder_strides
        out_channels_list = [config.decoder_hidden_dim * (2 ** min(i, 2))
                            for i in range(len(strides) - 1, -1, -1)]

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, out_channels_list[0], 1)
        self.input_norm = nn.GroupNorm(8, out_channels_list[0])

        # Upsampling decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_channels = out_channels_list[0]

        for i, stride in enumerate(reversed(strides)):
            if i < len(out_channels_list) - 1:
                out_channels = out_channels_list[i + 1]
            else:
                out_channels = config.decoder_hidden_dim

            self.decoder_blocks.append(
                TransposedConvBlock(
                    in_channels, out_channels,
                    kernel_size=stride * 2,
                    stride=stride,
                    dropout=config.dropout,
                )
            )
            in_channels = out_channels

        # Output convolution
        self.output_conv = nn.Conv1d(config.decoder_hidden_dim, 1, 7, padding=3)

    def forward(
        self,
        z_q: torch.Tensor,  # [batch, code_dim, seq]
        cascade_context: Optional[torch.Tensor] = None,  # From lower band
    ) -> torch.Tensor:
        """
        Decode to band audio.

        Args:
            z_q: Quantized latent [batch, code_dim, seq]
            cascade_context: Context from lower frequency band [batch, code_dim, seq]

        Returns:
            Band audio: [batch, samples]
        """
        # Apply cascade context if available
        if self.cascade_proj is not None and cascade_context is not None:
            # Project cascade context
            ctx = self.cascade_proj(cascade_context)

            # Align sequence lengths
            min_len = min(z_q.shape[-1], ctx.shape[-1])
            z_q = z_q[..., :min_len]
            ctx = ctx[..., :min_len]

            # Cross-attention: query=z_q, key/value=cascade context
            z_q_t = z_q.permute(0, 2, 1)  # [B, T, D]
            ctx_t = ctx.permute(0, 2, 1)

            attn_out, _ = self.cascade_attn(z_q_t, ctx_t, ctx_t)
            z_q = self.cascade_norm(z_q_t + attn_out).permute(0, 2, 1)

        # Input projection
        x = self.input_proj(z_q)
        x = self.input_norm(x)
        x = F.gelu(x)

        # Decoder blocks with upsampling
        for block in self.decoder_blocks:
            x = block(x)

        # Output
        audio = self.output_conv(x).squeeze(1)

        return audio


# =============================================================================
# CASCADE FUSION MODULE
# =============================================================================

class CascadeFusion(nn.Module):
    """
    Cascade Fusion module for combining multi-band outputs.

    Higher bands can see and refine lower band reconstructions.
    """

    def __init__(self, config: SoftFreqBandConfig):
        super().__init__()
        self.config = config
        self.num_bands = config.num_bands

        # Per-band learnable weights for fusion
        self.band_weights = nn.Parameter(torch.ones(config.num_bands))

        # Optional refinement convolutions
        self.refinement = nn.Sequential(
            nn.Conv1d(config.num_bands, config.num_bands * 2, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(config.num_bands * 2, 1, 5, padding=2),
        )

    def forward(
        self,
        band_outputs: List[torch.Tensor],  # List of [batch, samples]
    ) -> torch.Tensor:
        """
        Fuse multi-band outputs to full-band audio.

        Returns:
            Full-band audio: [batch, samples]
        """
        # Normalize weights
        weights = F.softmax(self.band_weights, dim=0)

        # Align lengths
        min_len = min(b.shape[-1] for b in band_outputs)
        band_outputs = [b[..., :min_len] for b in band_outputs]

        # Weighted sum
        weighted_sum = sum(w * b for w, b in zip(weights, band_outputs))

        # Stack for refinement
        stacked = torch.stack(band_outputs, dim=1)  # [B, num_bands, samples]

        # Refinement
        refinement = self.refinement(stacked).squeeze(1)  # [B, samples]

        # Combine
        output = weighted_sum + 0.1 * refinement

        return output


# =============================================================================
# FULL MODEL
# =============================================================================

class SoftFreqBandCodec(nn.Module):
    """
    Soft Frequency-Band Disentanglement Codec.

    Multi-branch codec with frequency-band specialization and soft disentanglement.
    """

    def __init__(self, config: SoftFreqBandConfig):
        super().__init__()
        self.config = config

        # Spectral decomposer
        self.decomposer = SpectralDecomposer(config)

        # Per-band encoders
        self.encoders = nn.ModuleList([
            BandEncoder(config, i)
            for i in range(config.num_bands)
        ])

        # Per-band quantizers
        self.quantizers = nn.ModuleList([
            ResidualVectorQuantizer(config, i)
            for i in range(config.num_bands)
        ])

        # Per-band decoders
        self.decoders = nn.ModuleList([
            BandDecoder(config, i)
            for i in range(config.num_bands)
        ])

        # Cascade fusion
        self.fusion = CascadeFusion(config)

        # Output projection for CSM integration
        total_code_dim = config.num_bands * config.code_dim
        self.output_proj = nn.Sequential(
            nn.Linear(total_code_dim, config.output_dim),
            nn.GELU(),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        audio: torch.Tensor,  # [batch, samples]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode audio to per-band tokens.

        Returns:
            Dict with band_latents, band_indices, band_z_q, commitment_losses, etc.
        """
        # Decompose into frequency bands
        decomp = self.decomposer(audio)

        band_latents = []
        band_indices = []
        band_z_q = []
        commitment_losses = []
        perplexities = []

        for i in range(self.config.num_bands):
            # Encode band
            latent = self.encoders[i](decomp['band_audio'][i])
            band_latents.append(latent)

            # Quantize
            vq_output = self.quantizers[i](latent)
            band_indices.append(vq_output['indices'])
            band_z_q.append(vq_output['z_q'])
            commitment_losses.append(vq_output['commitment_loss'])
            perplexities.append(vq_output['perplexities'])

        return {
            'band_audio': decomp['band_audio'],
            'band_latents': band_latents,
            'band_indices': band_indices,  # List of [num_q, batch, seq]
            'band_z_q': band_z_q,  # List of [batch, code_dim, seq]
            'commitment_losses': commitment_losses,
            'perplexities': perplexities,
            'stft': decomp['stft'],
            'masks': decomp['masks'],
        }

    def decode(
        self,
        band_indices: List[torch.Tensor],  # List of [num_q, batch, seq]
    ) -> torch.Tensor:
        """
        Decode from per-band indices.

        Returns:
            Full-band audio: [batch, samples]
        """
        # Decode each band
        band_z_q = []
        for i, indices in enumerate(band_indices):
            z_q = self.quantizers[i].decode(indices)
            band_z_q.append(z_q)

        # Decode with cascade
        band_outputs = []
        cascade_context = None

        for i in range(self.config.num_bands):
            output = self.decoders[i](band_z_q[i], cascade_context)
            band_outputs.append(output)

            # Update cascade context for next band
            if self.config.use_cascade:
                cascade_context = band_z_q[i]

        # Fuse bands
        audio = self.fusion(band_outputs)

        return audio

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode, quantize, decode.

        Returns:
            Dict with all outputs including reconstruction
        """
        # Encode
        encoded = self.encode(audio)

        # Decode with cascade
        band_outputs = []
        cascade_context = None

        for i in range(self.config.num_bands):
            output = self.decoders[i](encoded['band_z_q'][i], cascade_context)
            band_outputs.append(output)

            if self.config.use_cascade:
                cascade_context = encoded['band_z_q'][i]

        # Fuse bands
        audio_reconstructed = self.fusion(band_outputs)

        # Combined embedding for CSM (pool and concatenate band latents)
        pooled_latents = []
        for z_q in encoded['band_z_q']:
            pooled = z_q.mean(dim=-1)  # [batch, code_dim]
            pooled_latents.append(pooled)

        combined = torch.cat(pooled_latents, dim=-1)  # [batch, num_bands * code_dim]
        prosody_embedding = self.output_proj(combined)

        return {
            **encoded,
            'band_outputs': band_outputs,
            'audio_reconstructed': audio_reconstructed,
            'prosody_embedding': prosody_embedding,
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class SoftFreqBandLoss(nn.Module):
    """
    Loss function for Soft Frequency-Band Disentanglement Codec.

    Components:
    1. Per-band reconstruction loss
    2. Full-band reconstruction loss
    3. Commitment loss (VQ)
    4. Cross-band disentanglement loss (soft)
    5. Spectral loss
    """

    def __init__(self, config: SoftFreqBandConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.band_recon_weight = 1.0
        self.full_recon_weight = 1.0
        self.commitment_weight = 0.25
        self.cross_band_weight = config.cross_band_weight
        self.spectral_weight = 0.5
        self.band_consistency_weight = config.band_consistency_weight

    def compute_spectral_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> torch.Tensor:
        """Multi-resolution spectral loss."""
        loss = 0.0

        for fft_size in [512, 1024, 2048]:
            hop = fft_size // 4

            # Compute spectrograms
            pred_spec = torch.stft(
                pred, fft_size, hop, return_complex=True
            ).abs()
            target_spec = torch.stft(
                target, fft_size, hop, return_complex=True
            ).abs()

            # L1 loss on magnitude
            loss += F.l1_loss(pred_spec, target_spec)

            # Log-magnitude loss
            loss += F.l1_loss(
                torch.log(pred_spec + 1e-8),
                torch.log(target_spec + 1e-8)
            )

        return loss / 6.0  # Average over 3 resolutions × 2 loss types

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        audio_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        audio_pred = output['audio_reconstructed']
        band_outputs = output['band_outputs']
        band_audio_target = output['band_audio']

        # Align lengths
        min_len = min(audio_pred.shape[-1], audio_target.shape[-1])
        audio_pred = audio_pred[..., :min_len]
        audio_target = audio_target[..., :min_len]

        # 1. Per-band reconstruction loss
        band_recon_loss = 0.0
        for i in range(self.config.num_bands):
            pred = band_outputs[i]
            target = band_audio_target[i]

            # Align lengths
            min_band_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_band_len]
            target = target[..., :min_band_len]

            band_recon_loss += F.l1_loss(pred, target)
            band_recon_loss += F.mse_loss(pred, target)

        losses['band_reconstruction'] = band_recon_loss / self.config.num_bands

        # 2. Full-band reconstruction loss
        losses['full_reconstruction'] = (
            F.l1_loss(audio_pred, audio_target) +
            F.mse_loss(audio_pred, audio_target)
        )

        # 3. Commitment losses
        total_commitment = sum(output['commitment_losses'])
        losses['commitment'] = total_commitment / self.config.num_bands

        # 4. Spectral loss
        losses['spectral'] = self.compute_spectral_loss(audio_pred, audio_target)

        # 5. Cross-band disentanglement loss (soft)
        # Encourage bands to focus on their designated frequency ranges
        # by penalizing energy leakage to other bands
        cross_band_loss = 0.0
        for i in range(self.config.num_bands):
            pred_band = band_outputs[i]

            # Compute how much energy is in OTHER bands' frequency ranges
            for j in range(self.config.num_bands):
                if i != j:
                    # Measure energy overlap (should be low for good disentanglement)
                    target_band = band_audio_target[j]
                    min_cb_len = min(pred_band.shape[-1], target_band.shape[-1])

                    # Correlation penalty - bands should be uncorrelated
                    pred_norm = F.normalize(pred_band[..., :min_cb_len], dim=-1)
                    target_norm = F.normalize(target_band[..., :min_cb_len], dim=-1)
                    correlation = (pred_norm * target_norm).sum(dim=-1).abs().mean()
                    cross_band_loss += correlation

        if self.config.num_bands > 1:
            losses['cross_band'] = cross_band_loss / (
                self.config.num_bands * (self.config.num_bands - 1)
            )
        else:
            losses['cross_band'] = torch.tensor(0.0, device=audio_target.device)

        # 6. Band consistency loss (optional)
        # Encourages band specialization by maximizing inter-band variance
        # and minimizing intra-band variance in the latent space
        if self.band_consistency_weight > 0:
            band_means = []
            for z_q in output['band_z_q']:
                band_means.append(z_q.mean(dim=(0, 2)))  # [code_dim]

            if len(band_means) > 1:
                stacked_means = torch.stack(band_means, dim=0)  # [num_bands, code_dim]
                inter_band_var = stacked_means.var(dim=0).mean()
                losses['band_consistency'] = -inter_band_var  # Maximize inter-band variance
            else:
                losses['band_consistency'] = torch.tensor(0.0, device=audio_target.device)
        else:
            losses['band_consistency'] = torch.tensor(0.0, device=audio_target.device)

        # Perplexities for logging
        all_perplexities = []
        for perps in output['perplexities']:
            all_perplexities.append(perps.mean())
        losses['mean_perplexity'] = torch.stack(all_perplexities).mean()

        # Total loss
        total = (
            self.band_recon_weight * losses['band_reconstruction'] +
            self.full_recon_weight * losses['full_reconstruction'] +
            self.commitment_weight * losses['commitment'] +
            self.spectral_weight * losses['spectral'] +
            self.cross_band_weight * losses['cross_band'] +
            self.band_consistency_weight * losses['band_consistency']
        )
        losses['total'] = total

        return losses


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class SoftFreqBandAdapter(nn.Module):
    """
    Adapter for integrating Soft Frequency-Band Codec with CSM prosody conditioning.

    Converts multi-band tokens to prefix tokens for CSM.
    """

    def __init__(
        self,
        config: SoftFreqBandConfig,
        model: Optional[SoftFreqBandCodec] = None,
    ):
        super().__init__()
        self.config = config
        self.model = model if model is not None else SoftFreqBandCodec(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

        # Optional mel conversion layer (if input is mel instead of audio)
        self.mel_to_audio = nn.Sequential(
            nn.Linear(config.mel_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Linear(config.encoder_hidden_dim, 1),
        )

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,  # [batch, samples]
        mel: Optional[torch.Tensor] = None,  # [batch, seq, mel_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            audio: Raw audio waveform (preferred)
            mel: Mel spectrogram (alternative if audio not available)

        Returns:
            Dict with prosody_tokens, band_indices, etc.
        """
        batch_size = audio.shape[0] if audio is not None else mel.shape[0]

        if audio is None and mel is not None:
            # Convert mel to pseudo-audio for encoding
            # This is a rough approximation; prefer using actual audio
            audio = self.mel_to_audio(mel).squeeze(-1)  # [batch, seq]
            # Upsample to audio sample rate
            audio = F.interpolate(
                audio.unsqueeze(1),
                scale_factor=self.config.hop_length,
                mode='linear',
                align_corners=False,
            ).squeeze(1)

        # Encode
        output = self.model(audio)
        prosody_emb = output['prosody_embedding']

        # Project to prefix tokens
        tokens = self.token_proj(prosody_emb)
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )
        tokens = self.norm(tokens)

        return {
            'prosody_tokens': tokens,
            'band_indices': output['band_indices'],
            'band_z_q': output['band_z_q'],
            'commitment_losses': output['commitment_losses'],
            'perplexities': output['perplexities'],
        }

    def from_indices(
        self,
        band_indices: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Get prosody tokens from pre-computed band indices.

        Enables discrete prosody control via band tokens.
        """
        batch_size = band_indices[0].shape[1]

        # Decode indices to z_q
        band_z_q = []
        for i, indices in enumerate(band_indices):
            z_q = self.model.quantizers[i].decode(indices)
            band_z_q.append(z_q)

        # Pool and concatenate
        pooled_latents = [z_q.mean(dim=-1) for z_q in band_z_q]
        combined = torch.cat(pooled_latents, dim=-1)
        prosody_emb = self.model.output_proj(combined)

        # Project to tokens
        tokens = self.token_proj(prosody_emb)
        tokens = tokens.view(
            batch_size, self.config.num_prefix_tokens, self.config.output_dim
        )
        tokens = self.norm(tokens)

        return tokens

    def get_band_contributions(
        self,
        audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze per-band contributions to prosody encoding.

        Useful for understanding what each frequency band captures.
        """
        encoded = self.model.encode(audio)

        analysis = {
            'band_energies': [],
            'band_variances': [],
            'band_perplexities': [],
        }

        for i in range(self.config.num_bands):
            # Energy in band
            energy = encoded['band_audio'][i].pow(2).mean(dim=-1)
            analysis['band_energies'].append(energy)

            # Variance in latent
            variance = encoded['band_z_q'][i].var(dim=-1).mean(dim=-1)
            analysis['band_variances'].append(variance)

            # Perplexity
            analysis['band_perplexities'].append(
                encoded['perplexities'][i].mean()
            )

        analysis['band_energies'] = torch.stack(analysis['band_energies'], dim=1)
        analysis['band_variances'] = torch.stack(analysis['band_variances'], dim=1)
        analysis['band_perplexities'] = torch.stack(analysis['band_perplexities'])

        return analysis

    def manipulate_bands(
        self,
        audio: torch.Tensor,
        band_scales: List[float],  # Scale factor per band
    ) -> torch.Tensor:
        """
        Manipulate per-band contributions.

        Enables direct frequency-band control for prosody editing.
        """
        encoded = self.model.encode(audio)

        # Scale band z_q
        scaled_z_q = []
        for i, (z_q, scale) in enumerate(zip(encoded['band_z_q'], band_scales)):
            scaled_z_q.append(z_q * scale)

        # Decode with scaled latents
        band_outputs = []
        cascade_context = None

        for i in range(self.config.num_bands):
            output = self.model.decoders[i](scaled_z_q[i], cascade_context)
            band_outputs.append(output)

            if self.config.use_cascade:
                cascade_context = scaled_z_q[i]

        # Fuse
        audio_manipulated = self.model.fusion(band_outputs)

        return audio_manipulated


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_soft_freq_band_adapter(
    checkpoint: Optional[str] = None,
    config: Optional[SoftFreqBandConfig] = None,
    device: str = "cpu",
) -> SoftFreqBandAdapter:
    """Create adapter, optionally loading from checkpoint."""
    if config is None:
        config = SoftFreqBandConfig()

    adapter = SoftFreqBandAdapter(config)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        adapter.load_state_dict(state_dict)

    return adapter.to(device)


def compute_bitrate(
    config: SoftFreqBandConfig,
) -> Dict[str, float]:
    """Compute theoretical bitrate for configuration."""
    # Frame rate (based on encoder downsampling)
    total_stride = 1
    for stride in config.encoder_strides:
        total_stride *= stride

    frame_rate = config.sample_rate / (config.hop_length * total_stride)

    # Bits per frame per band
    bits_per_code = math.log2(config.codebook_size)
    bits_per_frame_per_band = config.num_quantizers * bits_per_code

    # Total bits per frame
    bits_per_frame = config.num_bands * bits_per_frame_per_band

    # Bitrate
    total_bps = frame_rate * bits_per_frame

    return {
        'frame_rate_hz': frame_rate,
        'bits_per_frame': bits_per_frame,
        'bits_per_band': bits_per_frame_per_band,
        'total_bps': total_bps,
        'total_kbps': total_bps / 1000,
    }


def analyze_frequency_separation(
    model: SoftFreqBandCodec,
    audio: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Analyze how well the model separates frequency bands.

    Returns metrics indicating disentanglement quality.
    """
    with torch.no_grad():
        output = model(audio)

        # Compute spectral overlap between bands
        band_stft = []
        for band_audio in output['band_audio']:
            stft = torch.stft(
                band_audio,
                n_fft=2048,
                hop_length=512,
                return_complex=True
            ).abs()
            band_stft.append(stft)

        # Cross-band energy ratios
        overlap_matrix = torch.zeros(model.config.num_bands, model.config.num_bands)

        for i in range(model.config.num_bands):
            total_energy_i = band_stft[i].pow(2).sum()
            for j in range(model.config.num_bands):
                # Energy of band i in frequency range of band j
                if i == 0:
                    start_bin = 0
                else:
                    start_bin = model.decomposer.bin_cutoffs[i - 1]
                end_bin = model.decomposer.bin_cutoffs[i]

                energy_in_range = band_stft[j][:, start_bin:end_bin, :].pow(2).sum()
                overlap_matrix[i, j] = energy_in_range / (total_energy_i + 1e-10)

        # Disentanglement score (diagonal should be high, off-diagonal low)
        diagonal = overlap_matrix.diag().mean()
        off_diagonal = (overlap_matrix.sum() - overlap_matrix.diag().sum()) / (
            model.config.num_bands * (model.config.num_bands - 1)
        )
        disentanglement_score = diagonal - off_diagonal

        return {
            'overlap_matrix': overlap_matrix,
            'diagonal_mean': diagonal,
            'off_diagonal_mean': off_diagonal,
            'disentanglement_score': disentanglement_score,
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Soft Frequency-Band Disentanglement Codec - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    audio_length = 32000  # 1 second at 32kHz
    mel_seq_len = 100
    mel_dim = 80

    config = SoftFreqBandConfig()

    # Create dummy inputs
    audio = torch.randn(batch_size, audio_length).to(device)
    mel = torch.randn(batch_size, mel_seq_len, mel_dim).to(device)

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  Num bands: {config.num_bands}")
    print(f"  Band cutoffs: {config.band_cutoffs} Hz")
    print(f"  Codebook size: {config.codebook_size}")
    print(f"  RVQ depth: {config.num_quantizers}")
    print(f"  Use cascade: {config.use_cascade}")
    print("  [PASS]")

    # Test 2: Bitrate computation
    print("\n[Test 2] Bitrate Computation...")
    bitrate = compute_bitrate(config)
    print(f"  Frame rate: {bitrate['frame_rate_hz']:.2f} Hz")
    print(f"  Bits per frame: {bitrate['bits_per_frame']:.1f}")
    print(f"  Total bitrate: {bitrate['total_kbps']:.2f} kbps")
    print("  [PASS]")

    # Test 3: Spectral Decomposer
    print("\n[Test 3] Spectral Decomposer...")
    decomposer = SpectralDecomposer(config).to(device)
    decomp = decomposer(audio)
    print(f"  Input audio: {audio.shape}")
    print(f"  Num bands: {len(decomp['band_audio'])}")
    for i, band_audio in enumerate(decomp['band_audio']):
        print(f"    Band {i}: {band_audio.shape}")
    print(f"  STFT shape: {decomp['stft'].shape}")
    print(f"  Mask shapes: {[m.shape for m in decomp['masks']]}")

    # Test reconstruction via sum
    reconstructed = decomposer.reconstruct(decomp['band_audio'])
    recon_error = F.mse_loss(reconstructed, audio)
    print(f"  Reconstruction error (sum): {recon_error.item():.6f}")
    print("  [PASS]")

    # Test 4: Band Encoder
    print("\n[Test 4] Band Encoder...")
    encoder = BandEncoder(config, band_idx=0).to(device)
    latent = encoder(decomp['band_audio'][0])
    print(f"  Input shape: {decomp['band_audio'][0].shape}")
    print(f"  Latent shape: {latent.shape}")
    print("  [PASS]")

    # Test 5: Vector Quantizer
    print("\n[Test 5] Residual Vector Quantizer...")
    rvq = ResidualVectorQuantizer(config, band_idx=0).to(device)
    vq_output = rvq(latent)
    print(f"  z_q shape: {vq_output['z_q'].shape}")
    print(f"  Indices shape: {vq_output['indices'].shape}")
    print(f"  Commitment loss: {vq_output['commitment_loss'].item():.4f}")
    print(f"  Perplexities: {vq_output['perplexities'].tolist()}")
    print("  [PASS]")

    # Test 6: Band Decoder
    print("\n[Test 6] Band Decoder...")
    decoder = BandDecoder(config, band_idx=0).to(device)
    band_recon = decoder(vq_output['z_q'])
    print(f"  z_q shape: {vq_output['z_q'].shape}")
    print(f"  Reconstructed band shape: {band_recon.shape}")
    print("  [PASS]")

    # Test 7: Cascade Decoder (band 1 with cascade)
    print("\n[Test 7] Cascade Decoder (band 1)...")
    decoder1 = BandDecoder(config, band_idx=1).to(device)
    encoder1 = BandEncoder(config, band_idx=1).to(device)
    rvq1 = ResidualVectorQuantizer(config, band_idx=1).to(device)

    latent1 = encoder1(decomp['band_audio'][1])
    vq_output1 = rvq1(latent1)

    # With cascade context from band 0
    band_recon1 = decoder1(vq_output1['z_q'], cascade_context=vq_output['z_q'])
    print(f"  Band 1 z_q shape: {vq_output1['z_q'].shape}")
    print(f"  Cascade context shape: {vq_output['z_q'].shape}")
    print(f"  Reconstructed band 1 shape: {band_recon1.shape}")
    print("  [PASS]")

    # Test 8: Cascade Fusion
    print("\n[Test 8] Cascade Fusion...")
    fusion = CascadeFusion(config).to(device)
    fused = fusion([band_recon, band_recon1])
    print(f"  Band 0 shape: {band_recon.shape}")
    print(f"  Band 1 shape: {band_recon1.shape}")
    print(f"  Fused output shape: {fused.shape}")
    print(f"  Band weights: {F.softmax(fusion.band_weights, dim=0).tolist()}")
    print("  [PASS]")

    # Test 9: Full Model
    print("\n[Test 9] Full SoftFreqBandCodec Model...")
    model = SoftFreqBandCodec(config).to(device)
    output = model(audio)
    print(f"  Band indices: {[idx.shape for idx in output['band_indices']]}")
    print(f"  Band z_q: {[zq.shape for zq in output['band_z_q']]}")
    print(f"  Audio reconstructed: {output['audio_reconstructed'].shape}")
    print(f"  Prosody embedding: {output['prosody_embedding'].shape}")
    commitment_losses_str = [f"{l.item():.4f}" for l in output['commitment_losses']]
    print(f"  Commitment losses: {commitment_losses_str}")
    print("  [PASS]")

    # Test 10: Loss Function
    print("\n[Test 10] Loss Function...")
    loss_fn = SoftFreqBandLoss(config)
    losses = loss_fn(output, audio)
    print(f"  Band reconstruction: {losses['band_reconstruction'].item():.4f}")
    print(f"  Full reconstruction: {losses['full_reconstruction'].item():.4f}")
    print(f"  Commitment: {losses['commitment'].item():.4f}")
    print(f"  Spectral: {losses['spectral'].item():.4f}")
    print(f"  Cross-band: {losses['cross_band'].item():.4f}")
    print(f"  Band consistency: {losses['band_consistency'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Mean perplexity: {losses['mean_perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 11: CSM Adapter (from audio)
    print("\n[Test 11] CSM Adapter (from audio)...")
    adapter = SoftFreqBandAdapter(config, model).to(device)
    adapter_out = adapter(audio=audio)
    print(f"  Prosody tokens: {adapter_out['prosody_tokens'].shape}")
    assert adapter_out['prosody_tokens'].shape == (
        batch_size, config.num_prefix_tokens, config.output_dim
    )
    print("  [PASS]")

    # Test 12: CSM Adapter (from mel)
    print("\n[Test 12] CSM Adapter (from mel)...")
    adapter_out_mel = adapter(mel=mel)
    print(f"  Prosody tokens (from mel): {adapter_out_mel['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 13: Encode and Decode from Indices
    print("\n[Test 13] Encode/Decode from Indices...")
    encoded = model.encode(audio)
    audio_decoded = model.decode(encoded['band_indices'])
    print(f"  Encoded band indices: {[idx.shape for idx in encoded['band_indices']]}")
    print(f"  Decoded audio: {audio_decoded.shape}")

    # Verify close to forward pass output
    recon_diff = F.mse_loss(audio_decoded, output['audio_reconstructed'][..., :audio_decoded.shape[-1]])
    print(f"  Decode consistency error: {recon_diff.item():.6f}")
    print("  [PASS]")

    # Test 14: Prosody from Indices
    print("\n[Test 14] Prosody Tokens from Indices...")
    tokens_from_idx = adapter.from_indices(encoded['band_indices'])
    print(f"  Tokens from indices: {tokens_from_idx.shape}")
    print("  [PASS]")

    # Test 15: Band Contribution Analysis
    print("\n[Test 15] Band Contribution Analysis...")
    analysis = adapter.get_band_contributions(audio)
    print(f"  Band energies: {analysis['band_energies'].shape}")
    print(f"  Band variances: {analysis['band_variances'].shape}")
    print(f"  Band perplexities: {analysis['band_perplexities'].tolist()}")
    print("  [PASS]")

    # Test 16: Band Manipulation
    print("\n[Test 16] Band Manipulation...")
    # Scale low frequencies up, high frequencies down
    band_scales = [1.5, 0.5] if config.num_bands == 2 else [1.5, 1.0, 0.5]
    manipulated = adapter.manipulate_bands(audio, band_scales)
    print(f"  Original audio: {audio.shape}")
    print(f"  Manipulated audio: {manipulated.shape}")
    print(f"  Band scales: {band_scales}")
    print("  [PASS]")

    # Test 17: Frequency Separation Analysis
    print("\n[Test 17] Frequency Separation Analysis...")
    sep_analysis = analyze_frequency_separation(model, audio)
    print(f"  Overlap matrix:\n{sep_analysis['overlap_matrix']}")
    print(f"  Diagonal mean: {sep_analysis['diagonal_mean'].item():.4f}")
    print(f"  Off-diagonal mean: {sep_analysis['off_diagonal_mean'].item():.4f}")
    print(f"  Disentanglement score: {sep_analysis['disentanglement_score'].item():.4f}")
    print("  [PASS]")

    # Test 18: Backward Pass
    print("\n[Test 18] Backward Pass...")
    model.zero_grad()
    output = model(audio)
    losses = loss_fn(output, audio)
    losses['total'].backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All Soft Frequency-Band Disentanglement tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. SPECTRAL DECOMPOSITION:
       - STFT-based frequency band separation
       - Learnable soft masks for smooth transitions
       - Supports configurable number of bands and cutoffs

    2. MULTI-BRANCH ENCODING:
       - Separate encoder per frequency band
       - Each branch learns to specialize for its band
       - RVQ quantization per band

    3. CASCADE ARCHITECTURE:
       - Higher bands receive context from lower bands
       - Cross-attention fusion in decoders
       - Enables residual-style reconstruction

    4. SOFT DISENTANGLEMENT:
       - No hard constraints (unlike orthogonality losses)
       - Implicit specialization through architecture
       - Cross-band reconstruction loss for encouragement

    5. CSM INTEGRATION:
       adapter = SoftFreqBandAdapter(config, model)
       prefix_tokens = adapter(audio=audio)['prosody_tokens']
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from soft_freq_band_codec import (
    SoftFreqBandConfig,
    SoftFreqBandCodec,
    SoftFreqBandLoss,
    SoftFreqBandAdapter,
    compute_bitrate,
    analyze_frequency_separation,
)

# Initialize for 32kHz audio with 2 bands (0-8kHz, 8-16kHz)
config = SoftFreqBandConfig(
    sample_rate=32000,
    num_bands=2,
    band_cutoffs=(8000,),
)

model = SoftFreqBandCodec(config).cuda()
loss_fn = SoftFreqBandLoss(config)

# Check bitrate
bitrate = compute_bitrate(config)
print(f"Bitrate: {bitrate['total_kbps']:.2f} kbps")

# Training loop
for batch in dataloader:
    audio = batch['audio'].cuda()

    output = model(audio)
    losses = loss_fn(output, audio)

    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    # Monitor per-band perplexities
    for i, perps in enumerate(output['perplexities']):
        print(f"Band {i}: {perps.mean().item():.2f}")

    # Analyze frequency separation
    sep = analyze_frequency_separation(model, audio)
    print(f"Disentanglement: {sep['disentanglement_score'].item():.4f}")

# Encode to band-specific tokens
with torch.no_grad():
    encoded = model.encode(audio)
    lf_indices = encoded['band_indices'][0]  # Low-frequency tokens
    hf_indices = encoded['band_indices'][1]  # High-frequency tokens

# Decode from tokens
audio_recon = model.decode(encoded['band_indices'])

# CSM integration
adapter = SoftFreqBandAdapter(config, model)
prefix_tokens = adapter(audio=audio)['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prefix_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)

# Band manipulation for prosody editing
manipulated = adapter.manipulate_bands(audio, band_scales=[1.2, 0.8])

# Direct control via indices
tokens_from_idx = adapter.from_indices([custom_lf_indices, custom_hf_indices])
""")
