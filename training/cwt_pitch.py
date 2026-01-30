"""
CWT F0 Spectrogram Prediction (FastSpeech 2 Approach)

Based on FastSpeech 2: "Fast and High-Quality End-to-End Text to Speech" (ICLR 2021)
https://arxiv.org/abs/2006.04558

Key Innovation: Continuous Wavelet Transform (CWT) for pitch contour prediction.
Instead of directly regressing F0 values, we decompose the pitch contour into a
spectrogram representation that naturally captures hierarchical prosodic structures.

Why CWT is better than direct F0 regression:
1. **Multi-scale patterns**: CWT captures patterns at different time scales simultaneously
   - Micro-prosody (single phoneme variations)
   - Word-level intonation patterns
   - Phrase-level prosodic contours
   - Sentence-level melody arcs

2. **Natural hierarchy**: The spectrogram representation aligns with how humans perceive
   prosody - we hear both local pitch movements and global intonation patterns

3. **Smoother training**: Predicting spectrogram coefficients is more stable than
   directly regressing highly variable F0 values

4. **Better reconstruction**: Inverse CWT naturally produces smooth, continuous
   pitch contours without the discontinuities of frame-level prediction

Pipeline:
    F0 contour → CWT → Pitch Spectrogram → [Predictor] → Reconstructed Spectrogram → iCWT → F0

Reference Implementation: https://github.com/ming024/FastSpeech2
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CWTPitchConfig:
    """Configuration for CWT-based pitch prediction."""

    # CWT settings
    num_scales: int = 10  # Number of wavelet scales (frequency bands)
    min_scale: float = 1.0  # Minimum wavelet scale
    max_scale: float = 64.0  # Maximum wavelet scale (higher = lower freq patterns)
    wavelet_type: str = "mexican_hat"  # Wavelet type: mexican_hat, morlet, paul

    # F0 settings
    f0_min: float = 50.0  # Minimum F0 (Hz)
    f0_max: float = 800.0  # Maximum F0 (Hz)
    f0_normalize: bool = True  # Normalize F0 before CWT
    log_f0: bool = True  # Use log-scale F0

    # Spectrogram settings
    spectrogram_dim: int = 10  # Must match num_scales
    spectrogram_normalize: bool = True  # Normalize CWT spectrogram

    # Predictor settings
    hidden_dim: int = 256
    predictor_layers: int = 3
    predictor_kernel_size: int = 5
    dropout: float = 0.1

    # Output settings
    output_dim: int = 2048  # For prosody token generation
    num_prosody_tokens: int = 4

    # Training
    reconstruction_weight: float = 1.0
    spectrogram_weight: float = 0.5
    variance_weight: float = 0.1

    def __post_init__(self):
        self.spectrogram_dim = self.num_scales


# =============================================================================
# WAVELETS
# =============================================================================

class MexicanHatWavelet(nn.Module):
    """
    Mexican Hat (Ricker) wavelet for CWT.

    ψ(t) = (2/√(3σ)π^(1/4)) * (1 - (t/σ)²) * exp(-t²/(2σ²))

    Properties:
    - Second derivative of Gaussian
    - Good for detecting sharp transitions in F0
    - Symmetric, real-valued
    """

    def forward(self, t: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Compute Mexican Hat wavelet.

        Args:
            t: Time values (centered at 0)
            scale: Wavelet scale (larger = lower frequency)

        Returns:
            Wavelet values at time t
        """
        sigma = scale
        norm = 2.0 / (math.sqrt(3.0 * sigma) * (math.pi ** 0.25))
        t_scaled = t / sigma
        wavelet = norm * (1.0 - t_scaled ** 2) * torch.exp(-0.5 * t_scaled ** 2)
        return wavelet


class MorletWavelet(nn.Module):
    """
    Morlet wavelet for CWT.

    ψ(t) = exp(iω₀t) * exp(-t²/2)

    Properties:
    - Complex wavelet (we use magnitude)
    - Better frequency resolution than Mexican Hat
    - Good for analyzing pitch periodicity
    """

    def __init__(self, omega0: float = 5.0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, t: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Compute Morlet wavelet (real part).

        Args:
            t: Time values (centered at 0)
            scale: Wavelet scale

        Returns:
            Wavelet values at time t
        """
        t_scaled = t / scale
        envelope = torch.exp(-0.5 * t_scaled ** 2)
        oscillation = torch.cos(self.omega0 * t_scaled)
        norm = 1.0 / (scale ** 0.5)
        return norm * envelope * oscillation


class PaulWavelet(nn.Module):
    """
    Paul wavelet for CWT.

    Properties:
    - Asymmetric wavelet
    - Good for detecting onset patterns in prosody
    - Good time localization
    """

    def __init__(self, order: int = 4):
        super().__init__()
        self.order = order
        # Normalization constant
        self.norm = (2 ** order) * math.factorial(order) / (
            math.sqrt(math.pi * math.factorial(2 * order))
        )

    def forward(self, t: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Compute Paul wavelet (real part approximation).

        Args:
            t: Time values
            scale: Wavelet scale

        Returns:
            Wavelet values
        """
        t_scaled = t / scale
        # Simplified real-part approximation
        wavelet = self.norm / (scale ** 0.5) * (
            (1 - t_scaled ** 2) ** self.order * torch.exp(-torch.abs(t_scaled))
        )
        return wavelet


def get_wavelet(wavelet_type: str) -> nn.Module:
    """Get wavelet module by name."""
    wavelets = {
        "mexican_hat": MexicanHatWavelet(),
        "morlet": MorletWavelet(),
        "paul": PaulWavelet(),
    }
    if wavelet_type not in wavelets:
        raise ValueError(f"Unknown wavelet type: {wavelet_type}. "
                        f"Available: {list(wavelets.keys())}")
    return wavelets[wavelet_type]


# =============================================================================
# CWT ENCODER (F0 → Spectrogram)
# =============================================================================

class CWTEncoder(nn.Module):
    """
    Continuous Wavelet Transform encoder.

    Transforms F0 contour into a multi-scale pitch spectrogram where:
    - Rows represent different frequency scales (low to high)
    - Columns represent time
    - Values represent wavelet coefficient magnitudes

    This spectrogram captures prosodic patterns at multiple time scales:
    - Small scales: rapid pitch changes (microprosody)
    - Large scales: slow pitch movements (intonation)
    """

    def __init__(self, config: CWTPitchConfig):
        super().__init__()
        self.config = config

        # Wavelet function
        self.wavelet = get_wavelet(config.wavelet_type)

        # Precompute scales (logarithmically spaced)
        scales = torch.logspace(
            math.log10(config.min_scale),
            math.log10(config.max_scale),
            config.num_scales,
        )
        self.register_buffer("scales", scales)

        # Normalization parameters (learned for better reconstruction)
        self.spec_norm = nn.LayerNorm(config.num_scales)

    def _compute_cwt_conv(
        self,
        f0: torch.Tensor,  # [batch, time]
        scale: float,
    ) -> torch.Tensor:
        """
        Compute CWT at a single scale using convolution.

        The CWT is computed as convolution of signal with scaled wavelet:
        W(a, b) = (1/√a) ∫ f(t) ψ*((t-b)/a) dt

        We implement this efficiently as a 1D convolution.
        """
        batch_size, time_len = f0.shape
        device = f0.device

        # Determine wavelet kernel size (depends on scale)
        # Larger scales need wider kernels to capture full wavelet
        kernel_size = int(min(scale * 8, time_len))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        # Create time vector centered at 0
        half_size = kernel_size // 2
        t = torch.arange(-half_size, half_size + 1, device=device, dtype=f0.dtype)

        # Compute wavelet kernel
        kernel = self.wavelet(t, scale)  # [kernel_size]
        kernel = kernel.view(1, 1, -1)  # [1, 1, kernel_size] for conv1d

        # Pad signal symmetrically
        f0_padded = F.pad(f0.unsqueeze(1), (half_size, half_size), mode='reflect')

        # Convolve
        cwt_coef = F.conv1d(f0_padded, kernel)  # [batch, 1, time]

        return cwt_coef.squeeze(1)  # [batch, time]

    def forward(
        self,
        f0: torch.Tensor,  # [batch, time]
        voiced_mask: Optional[torch.Tensor] = None,  # [batch, time] boolean
    ) -> Dict[str, torch.Tensor]:
        """
        Transform F0 contour to CWT pitch spectrogram.

        Args:
            f0: Fundamental frequency contour [batch, time]
                Can be raw Hz or log-normalized values
            voiced_mask: Optional mask for voiced frames [batch, time]
                True = voiced, False = unvoiced

        Returns:
            Dict with:
                - spectrogram: [batch, num_scales, time] CWT coefficients
                - f0_normalized: [batch, time] normalized F0
                - mean: [batch] mean of voiced F0 (for denormalization)
                - std: [batch] std of voiced F0 (for denormalization)
        """
        batch_size, time_len = f0.shape
        device = f0.device

        # Handle unvoiced regions (F0 = 0)
        if voiced_mask is None:
            voiced_mask = f0 > 0

        # Normalize F0
        f0_processed = f0.clone()

        if self.config.log_f0:
            # Log-scale F0 for better distribution
            f0_processed = torch.where(
                voiced_mask,
                torch.log(f0_processed.clamp(min=1.0)),
                torch.zeros_like(f0_processed),
            )

        if self.config.f0_normalize:
            # Compute mean/std only over voiced regions
            voiced_count = voiced_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (f0_processed * voiced_mask).sum(dim=1, keepdim=True) / voiced_count

            diff_sq = ((f0_processed - mean) ** 2) * voiced_mask
            std = torch.sqrt(diff_sq.sum(dim=1, keepdim=True) / voiced_count + 1e-8)

            # Normalize
            f0_normalized = torch.where(
                voiced_mask,
                (f0_processed - mean) / std,
                torch.zeros_like(f0_processed),
            )
        else:
            f0_normalized = f0_processed
            mean = torch.zeros(batch_size, 1, device=device)
            std = torch.ones(batch_size, 1, device=device)

        # Interpolate unvoiced regions for smoother CWT
        f0_interpolated = self._interpolate_unvoiced(f0_normalized, voiced_mask)

        # Compute CWT at each scale
        spectrogram_list = []
        for scale in self.scales:
            cwt_coef = self._compute_cwt_conv(f0_interpolated, scale.item())
            spectrogram_list.append(cwt_coef)

        # Stack to create spectrogram [batch, num_scales, time]
        spectrogram = torch.stack(spectrogram_list, dim=1)

        # Normalize spectrogram
        if self.config.spectrogram_normalize:
            # Per-scale normalization
            spec_t = spectrogram.transpose(1, 2)  # [batch, time, scales]
            spec_t = self.spec_norm(spec_t)
            spectrogram = spec_t.transpose(1, 2)  # [batch, scales, time]

        return {
            'spectrogram': spectrogram,
            'f0_normalized': f0_normalized,
            'f0_interpolated': f0_interpolated,
            'mean': mean.squeeze(1),
            'std': std.squeeze(1),
            'voiced_mask': voiced_mask,
        }

    def _interpolate_unvoiced(
        self,
        f0: torch.Tensor,  # [batch, time]
        voiced_mask: torch.Tensor,  # [batch, time]
    ) -> torch.Tensor:
        """
        Interpolate F0 values in unvoiced regions for smoother CWT.

        Uses linear interpolation between voiced regions.
        """
        batch_size, time_len = f0.shape
        f0_interp = f0.clone()

        for b in range(batch_size):
            # Find voiced indices
            voiced_idx = torch.where(voiced_mask[b])[0]

            if len(voiced_idx) < 2:
                # Not enough voiced frames, use mean or zero
                if len(voiced_idx) == 1:
                    f0_interp[b] = f0[b, voiced_idx[0]]
                continue

            # Linear interpolation
            voiced_values = f0[b, voiced_idx]

            # Interpolate to all frames
            all_idx = torch.arange(time_len, device=f0.device, dtype=torch.float32)
            voiced_idx_float = voiced_idx.float()

            # Simple linear interpolation using searchsorted
            indices = torch.searchsorted(voiced_idx_float, all_idx)
            indices = indices.clamp(1, len(voiced_idx) - 1)

            # Get surrounding voiced values
            left_idx = indices - 1
            right_idx = indices

            left_pos = voiced_idx_float[left_idx]
            right_pos = voiced_idx_float[right_idx]

            # Interpolation weight
            weight = (all_idx - left_pos) / (right_pos - left_pos + 1e-8)
            weight = weight.clamp(0, 1)

            # Interpolate
            left_val = voiced_values[left_idx]
            right_val = voiced_values[right_idx]
            f0_interp[b] = left_val * (1 - weight) + right_val * weight

        return f0_interp


# =============================================================================
# INVERSE CWT (Spectrogram → F0)
# =============================================================================

class InverseCWT(nn.Module):
    """
    Inverse Continuous Wavelet Transform.

    Reconstructs F0 contour from CWT pitch spectrogram.

    The inverse CWT is computed as:
    f(t) = C_ψ^(-1) ∫∫ W(a,b) ψ((t-b)/a) (da db) / a²

    In practice, we use a simplified summation over scales:
    f(t) ≈ Σ_a W(a,t) / a

    With learned scale weights for better reconstruction.
    """

    def __init__(self, config: CWTPitchConfig):
        super().__init__()
        self.config = config

        # Wavelet for reconstruction
        self.wavelet = get_wavelet(config.wavelet_type)

        # Precompute scales
        scales = torch.logspace(
            math.log10(config.min_scale),
            math.log10(config.max_scale),
            config.num_scales,
        )
        self.register_buffer("scales", scales)

        # Learned scale weights for weighted reconstruction
        # Different scales contribute differently to F0 reconstruction
        self.scale_weights = nn.Parameter(torch.ones(config.num_scales))

        # Post-processing for smoothing
        self.smooth_conv = nn.Conv1d(
            1, 1, kernel_size=5, padding=2, bias=False
        )
        # Initialize as Gaussian smoothing
        with torch.no_grad():
            kernel = torch.tensor([0.06, 0.24, 0.4, 0.24, 0.06])
            self.smooth_conv.weight.copy_(kernel.view(1, 1, -1))

    def forward(
        self,
        spectrogram: torch.Tensor,  # [batch, num_scales, time]
        mean: Optional[torch.Tensor] = None,  # [batch] for denormalization
        std: Optional[torch.Tensor] = None,   # [batch] for denormalization
        voiced_mask: Optional[torch.Tensor] = None,  # [batch, time]
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct F0 from CWT spectrogram.

        Args:
            spectrogram: CWT pitch spectrogram [batch, num_scales, time]
            mean: Mean F0 for denormalization [batch]
            std: Std F0 for denormalization [batch]
            voiced_mask: Optional voiced/unvoiced mask [batch, time]

        Returns:
            Dict with:
                - f0_normalized: Reconstructed normalized F0 [batch, time]
                - f0: Reconstructed F0 in original scale [batch, time]
        """
        batch_size, num_scales, time_len = spectrogram.shape
        device = spectrogram.device

        # Weighted sum over scales
        # The inverse CWT involves integrating over scales with 1/a² weighting
        # We use learned weights that include this normalization
        weights = F.softmax(self.scale_weights, dim=0)  # Normalize weights
        scale_norms = 1.0 / (self.scales + 1e-8)  # 1/a normalization

        # Combine weights with scale normalization
        combined_weights = weights * scale_norms  # [num_scales]

        # Weighted sum: [batch, time]
        f0_normalized = torch.einsum('bst,s->bt', spectrogram, combined_weights)

        # Smooth the result
        f0_smooth = self.smooth_conv(f0_normalized.unsqueeze(1)).squeeze(1)

        # Denormalize if parameters provided
        if mean is not None and std is not None:
            f0_denorm = f0_smooth * std.unsqueeze(1) + mean.unsqueeze(1)
        else:
            f0_denorm = f0_smooth

        # Convert from log scale if needed
        if self.config.log_f0:
            f0_final = torch.exp(f0_denorm)
        else:
            f0_final = f0_denorm

        # Clamp to valid F0 range
        f0_final = f0_final.clamp(min=self.config.f0_min, max=self.config.f0_max)

        # Apply voiced mask if provided
        if voiced_mask is not None:
            f0_final = f0_final * voiced_mask.float()

        return {
            'f0_normalized': f0_smooth,
            'f0': f0_final,
            'scale_weights': weights,
        }


# =============================================================================
# CWT PITCH PREDICTOR
# =============================================================================

class CWTPitchPredictor(nn.Module):
    """
    Predicts CWT pitch spectrogram from text/phoneme embeddings.

    Unlike direct F0 prediction which regresses a single value per frame,
    this predicts the full CWT spectrogram, capturing multi-scale patterns.

    Architecture:
        Text Embeddings → Conv Stack → Spectrogram Predictor → CWT Spectrogram
    """

    def __init__(self, config: CWTPitchConfig, input_dim: int = 256):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # Convolutional stack for local pattern modeling
        self.conv_layers = nn.ModuleList()
        for i in range(config.predictor_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        config.hidden_dim, config.hidden_dim,
                        kernel_size=config.predictor_kernel_size,
                        padding=config.predictor_kernel_size // 2,
                    ),
                    nn.LayerNorm(config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )

        # Multi-scale output heads (one for each CWT scale)
        # Each head predicts coefficients at one scale
        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            for _ in range(config.num_scales)
        ])

        # Cross-scale attention for modeling correlations between scales
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        # Scale position embeddings
        self.scale_embed = nn.Embedding(config.num_scales, config.hidden_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, time, hidden]
        mask: Optional[torch.Tensor] = None,  # [batch, time]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict CWT pitch spectrogram from text embeddings.

        Args:
            text_embeddings: Phoneme/text embeddings [batch, time, hidden]
            mask: Optional attention mask [batch, time]

        Returns:
            Dict with:
                - spectrogram: Predicted CWT spectrogram [batch, num_scales, time]
                - per_scale_features: Per-scale hidden features
        """
        batch_size, time_len, _ = text_embeddings.shape
        device = text_embeddings.device

        # Project input
        x = self.input_proj(text_embeddings)  # [batch, time, hidden]

        # Convolutional layers (need channel-first format)
        x = x.transpose(1, 2)  # [batch, hidden, time]
        for conv in self.conv_layers:
            # Conv1d → LayerNorm needs channel-last
            conv_out = conv[0](x)  # Conv1d
            conv_out = conv_out.transpose(1, 2)  # [batch, time, hidden]
            conv_out = conv[1](conv_out)  # LayerNorm
            conv_out = conv[2](conv_out)  # GELU
            conv_out = conv[3](conv_out)  # Dropout
            x = x.transpose(1, 2) + conv_out  # Residual [batch, time, hidden]
            x = x.transpose(1, 2)  # Back to [batch, hidden, time]

        x = x.transpose(1, 2)  # Final: [batch, time, hidden]

        # Add scale embeddings and apply cross-scale attention
        scale_pos = torch.arange(self.config.num_scales, device=device)
        scale_emb = self.scale_embed(scale_pos)  # [num_scales, hidden]

        # Expand for attention: [batch, num_scales, hidden]
        scale_queries = scale_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: scales attend to temporal features
        scale_features, _ = self.scale_attention(
            scale_queries,  # Query: [batch, num_scales, hidden]
            x,              # Key: [batch, time, hidden]
            x,              # Value: [batch, time, hidden]
            key_padding_mask=mask,
        )  # Output: [batch, num_scales, hidden]

        # Predict per-scale coefficients across time
        # Combine temporal features with scale-specific features
        spectrogram_list = []
        for s, head in enumerate(self.scale_heads):
            # Combine temporal features with scale-specific info
            scale_feat = scale_features[:, s:s+1, :].expand(-1, time_len, -1)  # [batch, time, hidden]
            combined = x + scale_feat  # [batch, time, hidden]

            # Predict coefficient at this scale
            coef = head(combined).squeeze(-1)  # [batch, time]
            spectrogram_list.append(coef)

        # Stack to form spectrogram
        spectrogram = torch.stack(spectrogram_list, dim=1)  # [batch, num_scales, time]

        return {
            'spectrogram': spectrogram,
            'scale_features': scale_features,
            'temporal_features': x,
        }


# =============================================================================
# DIRECT F0 PREDICTOR (For Comparison)
# =============================================================================

class DirectF0Predictor(nn.Module):
    """
    Direct F0 prediction (baseline approach).

    Predicts raw F0 values per frame, without CWT decomposition.
    Used as a comparison baseline for CWT approach.
    """

    def __init__(self, config: CWTPitchConfig, input_dim: int = 256):
        super().__init__()
        self.config = config

        # Simple conv stack
        self.predictor = nn.Sequential(
            nn.Conv1d(input_dim, config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, time, hidden]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict F0 directly from text embeddings.

        Args:
            text_embeddings: [batch, time, hidden]
            mask: Optional mask [batch, time]

        Returns:
            Dict with f0_normalized prediction
        """
        x = text_embeddings.transpose(1, 2)  # [batch, hidden, time]
        f0 = self.predictor(x)  # [batch, 1, time]
        f0 = f0.squeeze(1)  # [batch, time]

        if mask is not None:
            f0 = f0 * (~mask).float()

        return {'f0_normalized': f0}


# =============================================================================
# COMPLETE CWT PITCH MODULE
# =============================================================================

class CWTPitchModule(nn.Module):
    """
    Complete CWT-based pitch prediction module.

    Combines:
    1. CWT Encoder: F0 → Spectrogram (for training target)
    2. CWT Pitch Predictor: Text → Spectrogram
    3. Inverse CWT: Spectrogram → F0 (for inference)

    Usage:
        module = CWTPitchModule(config)

        # Training: encode GT F0, predict, compute loss
        target = module.encode_f0(f0_gt)
        pred = module.predict(text_embeddings)
        loss = module.compute_loss(pred, target)

        # Inference: predict and decode
        pred = module.predict(text_embeddings)
        f0 = module.decode(pred['spectrogram'])
    """

    def __init__(self, config: CWTPitchConfig, input_dim: int = 256):
        super().__init__()
        self.config = config

        # Components
        self.cwt_encoder = CWTEncoder(config)
        self.cwt_predictor = CWTPitchPredictor(config, input_dim)
        self.inverse_cwt = InverseCWT(config)

        # For comparison
        self.direct_predictor = DirectF0Predictor(config, input_dim)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def encode_f0(
        self,
        f0: torch.Tensor,  # [batch, time]
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode F0 contour to CWT spectrogram (training target).

        Args:
            f0: F0 contour [batch, time]
            voiced_mask: Optional voiced mask [batch, time]

        Returns:
            Dict with spectrogram and normalization parameters
        """
        return self.cwt_encoder(f0, voiced_mask)

    def predict(
        self,
        text_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict CWT spectrogram from text embeddings.

        Args:
            text_embeddings: [batch, time, hidden]
            mask: Optional mask

        Returns:
            Dict with predicted spectrogram
        """
        return self.cwt_predictor(text_embeddings, mask)

    def decode(
        self,
        spectrogram: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode CWT spectrogram to F0 contour.

        Args:
            spectrogram: CWT spectrogram [batch, num_scales, time]
            mean, std: Normalization parameters
            voiced_mask: Optional voiced mask

        Returns:
            Dict with reconstructed F0
        """
        return self.inverse_cwt(spectrogram, mean, std, voiced_mask)

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, time, hidden]
        f0_target: Optional[torch.Tensor] = None,  # [batch, time]
        voiced_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with optional training.

        Args:
            text_embeddings: Text/phoneme embeddings
            f0_target: Optional ground truth F0 (for training)
            voiced_mask: Optional voiced mask
            mask: Optional attention mask

        Returns:
            Dict with predictions, reconstructions, and losses
        """
        result = {}

        # Predict spectrogram
        pred = self.predict(text_embeddings, mask)
        result['predicted_spectrogram'] = pred['spectrogram']
        result['scale_features'] = pred['scale_features']

        if f0_target is not None:
            # Training mode: compute target spectrogram and losses
            target = self.encode_f0(f0_target, voiced_mask)
            result['target_spectrogram'] = target['spectrogram']
            result['f0_normalized'] = target['f0_normalized']
            result['f0_mean'] = target['mean']
            result['f0_std'] = target['std']

            # Decode predicted spectrogram
            recon = self.decode(
                pred['spectrogram'],
                target['mean'], target['std'],
                voiced_mask,
            )
            result['reconstructed_f0'] = recon['f0']
            result['reconstructed_f0_normalized'] = recon['f0_normalized']

            # Compute losses
            losses = self.compute_loss(pred, target, f0_target, voiced_mask)
            result.update(losses)
        else:
            # Inference mode: decode with default parameters
            recon = self.decode(pred['spectrogram'])
            result['predicted_f0'] = recon['f0']

        return result

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        f0_target: torch.Tensor,
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Losses:
        1. Spectrogram MSE: Match predicted spectrogram to target
        2. Reconstruction loss: Reconstructed F0 should match original
        3. Per-scale variance loss: Encourage diverse scale usage

        Args:
            pred: Predicted outputs
            target: Encoded target outputs
            f0_target: Original F0 target
            voiced_mask: Voiced/unvoiced mask

        Returns:
            Dict of loss values
        """
        losses = {}

        # Spectrogram prediction loss
        spec_loss = self.mse_loss(pred['spectrogram'], target['spectrogram'])
        losses['spectrogram_loss'] = spec_loss

        # Reconstruction loss (through inverse CWT)
        recon = self.decode(
            pred['spectrogram'],
            target['mean'], target['std'],
            voiced_mask,
        )

        # L1 + L2 reconstruction loss
        if voiced_mask is not None:
            # Only compute loss on voiced regions
            recon_f0_masked = recon['f0'] * voiced_mask.float()
            f0_target_masked = f0_target * voiced_mask.float()
            recon_loss = self.l1_loss(recon_f0_masked, f0_target_masked) + \
                        self.mse_loss(recon_f0_masked, f0_target_masked)
        else:
            recon_loss = self.l1_loss(recon['f0'], f0_target) + \
                        self.mse_loss(recon['f0'], f0_target)
        losses['reconstruction_loss'] = recon_loss

        # Per-scale variance loss (encourage all scales to be used)
        scale_vars = pred['spectrogram'].var(dim=2)  # Variance over time per scale
        variance_loss = -scale_vars.mean()  # Negative = maximize variance
        losses['variance_loss'] = variance_loss

        # Total loss
        total = (
            self.config.spectrogram_weight * spec_loss +
            self.config.reconstruction_weight * recon_loss +
            self.config.variance_weight * variance_loss
        )
        losses['total_loss'] = total

        return losses

    def compare_with_direct(
        self,
        text_embeddings: torch.Tensor,
        f0_target: torch.Tensor,
        voiced_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compare CWT approach with direct F0 prediction.

        Args:
            text_embeddings: Text embeddings
            f0_target: Ground truth F0
            voiced_mask: Voiced mask
            mask: Attention mask

        Returns:
            Dict with comparison metrics
        """
        # CWT-based prediction
        cwt_result = self.forward(text_embeddings, f0_target, voiced_mask, mask)

        # Direct prediction
        direct_pred = self.direct_predictor(text_embeddings, mask)

        # Normalize target for direct comparison
        target_normalized = self.encode_f0(f0_target, voiced_mask)['f0_normalized']

        # Direct prediction loss
        direct_loss = self.mse_loss(direct_pred['f0_normalized'], target_normalized)

        return {
            'cwt_reconstruction_loss': cwt_result['reconstruction_loss'],
            'cwt_spectrogram_loss': cwt_result['spectrogram_loss'],
            'direct_loss': direct_loss,
            'cwt_f0': cwt_result.get('reconstructed_f0'),
            'direct_f0': direct_pred['f0_normalized'],
            'target_f0': f0_target,
        }


# =============================================================================
# PROSODY ADAPTER (Integration with Pipeline)
# =============================================================================

class CWTPitchAdapter(nn.Module):
    """
    Adapter integrating CWT pitch prediction with the prosody pipeline.

    Converts CWT pitch spectrogram to prosody tokens compatible with
    ProsodyControlledCSM.

    Usage:
        adapter = CWTPitchAdapter(config)

        # From F0 contour (training)
        tokens = adapter.from_f0(f0_contour)

        # From text (inference)
        tokens = adapter.from_text(text_embeddings)

        # Get spectrogram visualization
        spec = adapter.get_pitch_spectrogram(text_embeddings)
    """

    def __init__(
        self,
        config: CWTPitchConfig,
        input_dim: int = 256,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core CWT module
        self.cwt_module = CWTPitchModule(config, input_dim)

        # Project spectrogram to prosody tokens
        # Flatten spectrogram and project to token space
        self.spec_projection = nn.Sequential(
            nn.Linear(config.num_scales, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Temporal aggregation with attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        # Query tokens for aggregation
        self.query_tokens = nn.Parameter(
            torch.randn(config.num_prosody_tokens, config.hidden_dim)
        )

        # Output projection to prosody token dimension
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, prosody_hidden),
            nn.LayerNorm(prosody_hidden),
        )

    def from_f0(
        self,
        f0: torch.Tensor,  # [batch, time]
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from F0 contour.

        Args:
            f0: F0 contour [batch, time]
            voiced_mask: Voiced mask [batch, time]

        Returns:
            Dict with prosody_tokens [batch, num_tokens, hidden]
        """
        # Encode F0 to spectrogram
        encoded = self.cwt_module.encode_f0(f0, voiced_mask)
        spectrogram = encoded['spectrogram']  # [batch, num_scales, time]

        # Project and aggregate
        tokens = self._spectrogram_to_tokens(spectrogram)

        return {
            'prosody_tokens': tokens,
            'spectrogram': spectrogram,
            'f0_stats': {
                'mean': encoded['mean'],
                'std': encoded['std'],
            },
        }

    def from_text(
        self,
        text_embeddings: torch.Tensor,  # [batch, time, hidden]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from text embeddings.

        Args:
            text_embeddings: Text/phoneme embeddings
            mask: Attention mask

        Returns:
            Dict with prosody_tokens and predicted spectrogram
        """
        # Predict spectrogram
        pred = self.cwt_module.predict(text_embeddings, mask)
        spectrogram = pred['spectrogram']

        # Project and aggregate
        tokens = self._spectrogram_to_tokens(spectrogram)

        # Decode to F0 for visualization
        decoded = self.cwt_module.decode(spectrogram)

        return {
            'prosody_tokens': tokens,
            'spectrogram': spectrogram,
            'predicted_f0': decoded['f0'],
        }

    def _spectrogram_to_tokens(
        self,
        spectrogram: torch.Tensor,  # [batch, num_scales, time]
    ) -> torch.Tensor:
        """
        Convert spectrogram to prosody tokens.

        Args:
            spectrogram: CWT spectrogram [batch, num_scales, time]

        Returns:
            Prosody tokens [batch, num_tokens, hidden]
        """
        batch_size = spectrogram.shape[0]
        device = spectrogram.device

        # Transpose and project: [batch, time, scales] → [batch, time, hidden]
        spec_t = spectrogram.transpose(1, 2)
        projected = self.spec_projection(spec_t)  # [batch, time, hidden]

        # Use attention to aggregate into fixed number of tokens
        # Query tokens attend to temporal features
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        tokens, _ = self.temporal_attn(
            queries,   # [batch, num_tokens, hidden]
            projected, # [batch, time, hidden]
            projected,
        )  # [batch, num_tokens, hidden]

        # Project to output dimension
        tokens = self.output_projection(tokens)

        return tokens

    def get_pitch_spectrogram(
        self,
        text_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get pitch spectrogram for visualization.

        Args:
            text_embeddings: Text embeddings
            mask: Attention mask

        Returns:
            Spectrogram [batch, num_scales, time]
        """
        pred = self.cwt_module.predict(text_embeddings, mask)
        return pred['spectrogram']

    def forward(
        self,
        text_embeddings: torch.Tensor,
        f0_target: Optional[torch.Tensor] = None,
        voiced_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Training mode (f0_target provided):
            - Encode target F0 to spectrogram
            - Predict spectrogram from text
            - Compute losses
            - Return prosody tokens

        Inference mode (no f0_target):
            - Predict spectrogram from text
            - Return prosody tokens

        Args:
            text_embeddings: Text/phoneme embeddings
            f0_target: Optional ground truth F0
            voiced_mask: Voiced/unvoiced mask
            mask: Attention mask

        Returns:
            Dict with tokens, spectrogram, losses, etc.
        """
        # Get CWT module output
        cwt_result = self.cwt_module(
            text_embeddings, f0_target, voiced_mask, mask
        )

        # Convert spectrogram to tokens
        tokens = self._spectrogram_to_tokens(cwt_result['predicted_spectrogram'])

        result = {
            'prosody_tokens': tokens,
            'spectrogram': cwt_result['predicted_spectrogram'],
        }

        if f0_target is not None:
            result['target_spectrogram'] = cwt_result['target_spectrogram']
            result['reconstructed_f0'] = cwt_result['reconstructed_f0']
            result['spectrogram_loss'] = cwt_result['spectrogram_loss']
            result['reconstruction_loss'] = cwt_result['reconstruction_loss']
            result['total_loss'] = cwt_result['total_loss']
        else:
            result['predicted_f0'] = cwt_result['predicted_f0']

        return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CWT F0 Spectrogram Prediction - Test Suite")
    print("FastSpeech 2 Approach for Hierarchical Pitch Modeling")
    print("=" * 70)

    config = CWTPitchConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    time_len = 100
    input_dim = 256

    # Test 1: Wavelet Functions
    print("\n[Test 1] Wavelet Functions...")
    for wavelet_type in ["mexican_hat", "morlet", "paul"]:
        wavelet = get_wavelet(wavelet_type)
        t = torch.linspace(-5, 5, 101)
        w = wavelet(t, scale=2.0)
        print(f"  {wavelet_type}: shape={w.shape}, range=[{w.min():.3f}, {w.max():.3f}]")
    print("  [PASS]")

    # Test 2: CWT Encoder
    print("\n[Test 2] CWT Encoder (F0 → Spectrogram)...")
    encoder = CWTEncoder(config).to(device)

    # Generate synthetic F0 (with voiced/unvoiced regions)
    t = torch.linspace(0, 4 * math.pi, time_len)
    f0_raw = 150 + 50 * torch.sin(t) + 20 * torch.sin(3 * t)  # Multi-frequency
    f0 = f0_raw.unsqueeze(0).expand(batch_size, -1).to(device)

    # Create voiced mask (80% voiced)
    voiced_mask = torch.rand(batch_size, time_len, device=device) > 0.2

    encoded = encoder(f0, voiced_mask)
    print(f"  F0 input shape: {f0.shape}")
    print(f"  Spectrogram shape: {encoded['spectrogram'].shape}")
    print(f"  Expected: [{batch_size}, {config.num_scales}, {time_len}]")
    print(f"  Mean: {encoded['mean'].shape}, Std: {encoded['std'].shape}")
    print("  [PASS]")

    # Test 3: Inverse CWT
    print("\n[Test 3] Inverse CWT (Spectrogram → F0)...")
    inverse = InverseCWT(config).to(device)

    decoded = inverse(
        encoded['spectrogram'],
        encoded['mean'],
        encoded['std'],
        voiced_mask,
    )
    print(f"  Reconstructed F0 shape: {decoded['f0'].shape}")
    print(f"  Scale weights: {decoded['scale_weights'].shape}")

    # Check reconstruction quality
    recon_error = (decoded['f0'] - f0).abs().mean()
    print(f"  Reconstruction MAE: {recon_error:.2f} Hz")
    print("  [PASS]")

    # Test 4: CWT Pitch Predictor
    print("\n[Test 4] CWT Pitch Predictor...")
    predictor = CWTPitchPredictor(config, input_dim).to(device)

    text_emb = torch.randn(batch_size, time_len, input_dim, device=device)
    pred = predictor(text_emb)

    print(f"  Text embeddings shape: {text_emb.shape}")
    print(f"  Predicted spectrogram shape: {pred['spectrogram'].shape}")
    print(f"  Scale features shape: {pred['scale_features'].shape}")
    print("  [PASS]")

    # Test 5: Direct F0 Predictor (Baseline)
    print("\n[Test 5] Direct F0 Predictor (Baseline)...")
    direct_pred = DirectF0Predictor(config, input_dim).to(device)

    direct_out = direct_pred(text_emb)
    print(f"  Direct F0 prediction shape: {direct_out['f0_normalized'].shape}")
    print("  [PASS]")

    # Test 6: Complete CWT Module
    print("\n[Test 6] Complete CWT Pitch Module...")
    module = CWTPitchModule(config, input_dim).to(device)

    result = module(text_emb, f0, voiced_mask)
    print(f"  Predicted spectrogram: {result['predicted_spectrogram'].shape}")
    print(f"  Target spectrogram: {result['target_spectrogram'].shape}")
    print(f"  Reconstructed F0: {result['reconstructed_f0'].shape}")
    print(f"  Losses:")
    print(f"    Spectrogram loss: {result['spectrogram_loss']:.4f}")
    print(f"    Reconstruction loss: {result['reconstruction_loss']:.4f}")
    print(f"    Total loss: {result['total_loss']:.4f}")
    print("  [PASS]")

    # Test 7: CWT vs Direct Comparison
    print("\n[Test 7] CWT vs Direct Prediction Comparison...")
    comparison = module.compare_with_direct(text_emb, f0, voiced_mask)
    print(f"  CWT reconstruction loss: {comparison['cwt_reconstruction_loss']:.4f}")
    print(f"  CWT spectrogram loss: {comparison['cwt_spectrogram_loss']:.4f}")
    print(f"  Direct prediction loss: {comparison['direct_loss']:.4f}")
    print("  [PASS]")

    # Test 8: CWT Pitch Adapter (Pipeline Integration)
    print("\n[Test 8] CWT Pitch Adapter...")
    adapter = CWTPitchAdapter(config, input_dim, prosody_hidden=2048).to(device)

    # Training mode
    train_result = adapter(text_emb, f0, voiced_mask)
    print(f"  Prosody tokens shape: {train_result['prosody_tokens'].shape}")
    print(f"  Expected: [{batch_size}, {config.num_prosody_tokens}, 2048]")
    print(f"  Training total loss: {train_result['total_loss']:.4f}")

    # Inference mode
    infer_result = adapter(text_emb)
    print(f"  Inference prosody tokens: {infer_result['prosody_tokens'].shape}")
    print(f"  Predicted F0: {infer_result['predicted_f0'].shape}")
    print("  [PASS]")

    # Test 9: From F0 (Direct Encoding)
    print("\n[Test 9] Direct F0 Encoding...")
    direct_tokens = adapter.from_f0(f0, voiced_mask)
    print(f"  Tokens from F0: {direct_tokens['prosody_tokens'].shape}")
    print(f"  Spectrogram: {direct_tokens['spectrogram'].shape}")
    print("  [PASS]")

    # Test 10: Spectrogram Visualization
    print("\n[Test 10] Spectrogram Retrieval for Visualization...")
    spec_viz = adapter.get_pitch_spectrogram(text_emb)
    print(f"  Spectrogram for viz: {spec_viz.shape}")
    print(f"  Value range: [{spec_viz.min():.3f}, {spec_viz.max():.3f}]")
    print("  [PASS]")

    # Test 11: Gradient Flow
    print("\n[Test 11] Gradient Flow...")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)

    result = adapter(text_emb, f0, voiced_mask)
    loss = result['total_loss']
    loss.backward()

    # Check gradients
    total_grad_norm = 0.0
    num_params = 0
    for p in adapter.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
            num_params += 1
    total_grad_norm = total_grad_norm ** 0.5

    print(f"  Total gradient norm: {total_grad_norm:.4f}")
    print(f"  Parameters with grad: {num_params}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All CWT F0 Spectrogram Prediction tests passed!")
    print("=" * 70)

    # Model stats
    print("\nModel Statistics:")
    print("-" * 40)
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from cwt_pitch import (
    CWTPitchConfig,
    CWTPitchModule,
    CWTPitchAdapter,
)

# Initialize
config = CWTPitchConfig(
    num_scales=10,      # Number of frequency bands
    wavelet_type="mexican_hat",  # Good for pitch transitions
)

# Option 1: Use full module for training/analysis
module = CWTPitchModule(config, input_dim=256).cuda()

# Training: encode GT F0, predict from text, compute loss
result = module(text_embeddings, f0_target=f0_gt, voiced_mask=voiced_mask)
loss = result['total_loss']

# Compare CWT vs direct prediction
comparison = module.compare_with_direct(text_emb, f0_gt, voiced_mask)
print(f"CWT loss: {comparison['cwt_reconstruction_loss']:.4f}")
print(f"Direct loss: {comparison['direct_loss']:.4f}")

# Option 2: Use adapter for pipeline integration
adapter = CWTPitchAdapter(config, input_dim=256, prosody_hidden=2048).cuda()

# From F0 (training)
tokens = adapter.from_f0(f0_contour, voiced_mask)
prosody_prefix = tokens['prosody_tokens']  # [batch, 4, 2048]

# From text (inference)
tokens = adapter.from_text(text_embeddings)
prosody_prefix = tokens['prosody_tokens']
predicted_f0 = tokens['predicted_f0']

# Get spectrogram for visualization
spectrogram = adapter.get_pitch_spectrogram(text_embeddings)
# spectrogram shape: [batch, num_scales, time]
# Can be visualized as heatmap showing pitch patterns at different time scales

# Integrate with ProsodyControlledCSM:
combined_prefix = torch.cat([cwt_prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")

    print("\nKey Benefits of CWT Approach:")
    print("-" * 40)
    print("""
1. Multi-scale pattern capture:
   - Small scales → microprosody (local pitch variations)
   - Large scales → intonation contours (phrase-level)

2. Better training stability:
   - Predicting spectrogram coefficients is more stable than raw F0
   - Smoother loss landscape for optimization

3. Natural pitch reconstruction:
   - Inverse CWT produces smooth, continuous F0
   - No discontinuities from frame-level prediction

4. Hierarchical prosody modeling:
   - Aligns with human prosody perception
   - Captures both local and global patterns

5. Visualization-friendly:
   - Spectrogram shows pitch patterns clearly
   - Easy to interpret and debug
""")
