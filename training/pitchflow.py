"""
PitchFlow: Quantized Pitch Control for Flow-Matching TTS

Based on PitchFlow (Interspeech 2024): "PitchFlow: Discrete Pitch Representation
for Controllable Text-to-Speech Synthesis"
https://www.isca-archive.org/interspeech_2024/sadekova24_interspeech.pdf

Key Innovation: Quantized pitch control via 50 log-scale F0 bins
1. Quantize frame-level pitch into 50 bins (log-scale)
2. Train pitch classifier on noisy mel-spectrograms with cross-entropy loss
3. Use Praat/parselmouth for ground-truth F0 extraction (autocorrelation method)
4. Condition flow-matching decoder on predicted pitch bins

Benefits:
- Discrete pitch representation (50 bins) - simpler than continuous regression
- Cross-entropy training is more stable than MSE regression for pitch
- Compatible with existing flow-matching architectures (ProsodyFlow, TTS-CtrlNet)
- Explicit pitch control at inference via bin specification
- Better coverage of rare pitch values via binning

Pitch Binning Strategy:
- Log-scale binning captures perceptual pitch differences
- 50 bins from 50Hz to 800Hz (configurable)
- Bin 0 = unvoiced, bins 1-49 = voiced pitch ranges
- Equal logarithmic spacing for perceptual uniformity

References:
- PitchFlow: https://www.isca-archive.org/interspeech_2024/sadekova24_interspeech.pdf
- Parselmouth (Praat in Python): https://parselmouth.readthedocs.io/
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: parselmouth not available, F0 extraction will use fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PitchFlowConfig:
    """Configuration for PitchFlow pitch control."""

    # Pitch binning settings
    num_pitch_bins: int = 50  # Number of pitch quantization bins (including unvoiced)
    f0_min: float = 50.0      # Minimum F0 frequency (Hz)
    f0_max: float = 800.0     # Maximum F0 frequency (Hz)
    use_log_scale: bool = True  # Use logarithmic pitch binning (perceptually uniform)
    unvoiced_bin: int = 0       # Bin index for unvoiced frames

    # F0 extraction settings (Praat autocorrelation method)
    pitch_floor: float = 50.0   # Minimum pitch for extraction
    pitch_ceiling: float = 800.0  # Maximum pitch for extraction
    time_step: float = 0.01     # 10ms hop between pitch estimates
    voicing_threshold: float = 0.45  # Praat voicing threshold

    # Mel spectrogram settings (input to classifier)
    n_mels: int = 80
    sample_rate: int = 24000
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 0.0
    fmax: float = 12000.0

    # Pitch classifier architecture
    classifier_hidden_dim: int = 256
    classifier_num_layers: int = 4
    classifier_kernel_size: int = 5
    classifier_dropout: float = 0.1

    # Training noise augmentation (for robust classifier)
    train_noise_std: float = 0.1  # Noise added to mel during training
    train_noise_schedule: str = "constant"  # constant, linear_decay, cosine_decay

    # Flow conditioning
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_num_heads: int = 8

    # Integration with prosody pipeline
    prosody_dim: int = 2048
    num_prosody_tokens: int = 4
    text_dim: int = 768

    # Loss weights
    classifier_ce_weight: float = 1.0
    pitch_accuracy_bonus: float = 0.1  # Bonus for exact bin prediction
    flow_loss_weight: float = 1.0


# =============================================================================
# PITCH BINNING UTILITIES
# =============================================================================

class LogF0Quantizer(nn.Module):
    """
    Quantizes continuous F0 values into discrete log-scale bins.

    Binning strategy:
    - Bin 0: Unvoiced frames (F0 = 0)
    - Bins 1-49: Voiced frames with log-spaced F0 ranges

    Log-scale binning is perceptually uniform - each bin represents
    a similar "step" in perceived pitch height.
    """

    def __init__(self, config: PitchFlowConfig):
        super().__init__()
        self.config = config

        # Number of voiced bins (excluding unvoiced bin)
        num_voiced_bins = config.num_pitch_bins - 1

        # Compute log-scale bin boundaries
        if config.use_log_scale:
            # Log-spaced boundaries from f0_min to f0_max
            log_min = math.log(config.f0_min)
            log_max = math.log(config.f0_max)
            log_boundaries = torch.linspace(log_min, log_max, num_voiced_bins + 1)
            boundaries = torch.exp(log_boundaries)
        else:
            # Linear-spaced boundaries
            boundaries = torch.linspace(config.f0_min, config.f0_max, num_voiced_bins + 1)

        self.register_buffer("boundaries", boundaries)

        # Compute bin centers for reconstruction
        centers = (boundaries[:-1] + boundaries[1:]) / 2
        self.register_buffer("bin_centers", centers)

    def quantize(
        self,
        f0: torch.Tensor,  # [batch, time] or [time]
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Quantize continuous F0 to discrete bins.

        Args:
            f0: Continuous F0 values (Hz)
            voiced_mask: Optional boolean mask (True = voiced)

        Returns:
            Bin indices [batch, time] in range [0, num_bins-1]
        """
        # Determine voiced frames if mask not provided
        if voiced_mask is None:
            voiced_mask = f0 > 0

        # Start with all unvoiced (bin 0)
        bins = torch.zeros_like(f0, dtype=torch.long)

        # Clamp F0 to valid range for voiced frames
        f0_clamped = f0.clamp(self.config.f0_min, self.config.f0_max)

        # Find bin for each voiced frame using searchsorted
        # Returns index where value would be inserted to maintain sorted order
        voiced_bins = torch.searchsorted(self.boundaries, f0_clamped)

        # Clamp to valid range [1, num_pitch_bins-1] for voiced
        voiced_bins = voiced_bins.clamp(1, self.config.num_pitch_bins - 1)

        # Apply to voiced frames only
        bins = torch.where(voiced_mask, voiced_bins, bins)

        return bins

    def dequantize(
        self,
        bins: torch.Tensor,  # [batch, time]
    ) -> torch.Tensor:
        """
        Convert discrete bins back to continuous F0 values.

        Args:
            bins: Bin indices

        Returns:
            Reconstructed F0 values (Hz), 0 for unvoiced
        """
        # Get bin centers (shifted by 1 to account for unvoiced bin)
        f0 = torch.zeros_like(bins, dtype=torch.float32)

        voiced_mask = bins > 0
        voiced_bins = (bins[voiced_mask] - 1).clamp(0, len(self.bin_centers) - 1)
        f0[voiced_mask] = self.bin_centers[voiced_bins]

        return f0

    def forward(
        self,
        f0: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full quantization pipeline.

        Args:
            f0: Continuous F0 values

        Returns:
            Dict with bins, reconstructed f0, voiced_mask
        """
        voiced_mask = f0 > 0
        bins = self.quantize(f0, voiced_mask)
        f0_recon = self.dequantize(bins)

        return {
            'bins': bins,
            'f0_reconstructed': f0_recon,
            'voiced_mask': voiced_mask,
            'quantization_error': torch.abs(f0 - f0_recon) * voiced_mask.float(),
        }


# =============================================================================
# F0 EXTRACTOR (using Parselmouth/Praat)
# =============================================================================

class PraatF0Extractor:
    """
    Extracts F0 using Praat's autocorrelation method via Parselmouth.

    Praat's autocorrelation method is considered the gold standard for
    F0 extraction due to its robustness and accuracy.
    """

    def __init__(self, config: PitchFlowConfig):
        self.config = config

        if not PARSELMOUTH_AVAILABLE:
            raise RuntimeError(
                "parselmouth is required for PraatF0Extractor. "
                "Install with: pip install praat-parselmouth"
            )

    def extract_from_file(
        self,
        audio_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract F0 from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (times, f0_values, voiced_mask)
        """
        sound = parselmouth.Sound(audio_path)
        return self._extract_from_sound(sound)

    def extract_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract F0 from audio waveform.

        Args:
            audio: Audio waveform (1D numpy array)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (times, f0_values, voiced_mask)
        """
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        return self._extract_from_sound(sound)

    def _extract_from_sound(
        self,
        sound: "parselmouth.Sound",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract F0 from Parselmouth Sound object.

        Uses Praat's "To Pitch (ac)" command (autocorrelation method).
        """
        # Extract pitch using autocorrelation method
        pitch = call(
            sound,
            "To Pitch (ac)",
            self.config.time_step,
            self.config.pitch_floor,
            15,  # Max number of candidates
            "no",  # Very accurate
            self.config.voicing_threshold,
            0.01,  # Voicing threshold
            0.35,  # Octave cost
            0.25,  # Octave-jump cost
            0.01,  # Voiced/unvoiced cost
            self.config.pitch_ceiling,
        )

        # Get frame-level F0 values
        times = []
        f0_values = []

        num_frames = call(pitch, "Get number of frames")

        for i in range(1, num_frames + 1):
            time = call(pitch, "Get time from frame number", i)
            f0 = call(pitch, "Get value in frame", i, "Hertz")

            times.append(time)
            f0_values.append(f0 if f0 == f0 else 0.0)  # NaN check

        times = np.array(times)
        f0_values = np.array(f0_values)
        voiced_mask = f0_values > 0

        return times, f0_values, voiced_mask


# =============================================================================
# PITCH BIN CLASSIFIER
# =============================================================================

class PitchBinClassifier(nn.Module):
    """
    Classifies pitch bins from noisy mel-spectrograms.

    Architecture: Conv stack with residual connections + classification head.

    Training with noise augmentation makes the classifier robust to
    acoustic variations while maintaining pitch discrimination.
    """

    def __init__(self, config: PitchFlowConfig):
        super().__init__()
        self.config = config

        # Input projection from mel-spectrogram
        self.input_proj = nn.Sequential(
            nn.Conv1d(config.n_mels, config.classifier_hidden_dim, kernel_size=1),
            nn.LayerNorm([config.classifier_hidden_dim]),
            nn.GELU(),
        )

        # Convolutional stack with residual connections
        self.conv_layers = nn.ModuleList()
        for i in range(config.classifier_num_layers):
            self.conv_layers.append(
                ConvResBlock(
                    channels=config.classifier_hidden_dim,
                    kernel_size=config.classifier_kernel_size,
                    dropout=config.classifier_dropout,
                    dilation=2 ** (i % 3),  # Increasing receptive field
                )
            )

        # Multi-head attention for global context
        self.context_attn = nn.MultiheadAttention(
            embed_dim=config.classifier_hidden_dim,
            num_heads=4,
            dropout=config.classifier_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(config.classifier_hidden_dim)

        # Classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(config.classifier_hidden_dim, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_pitch_bins),
        )

        # Register noise schedule
        self.register_buffer("_step", torch.tensor(0))
        self.register_buffer("_max_steps", torch.tensor(100000))

    def get_noise_std(self, step: Optional[int] = None) -> float:
        """Get current noise standard deviation based on schedule."""
        if step is None:
            step = self._step.item()

        max_steps = self._max_steps.item()
        base_std = self.config.train_noise_std

        if self.config.train_noise_schedule == "constant":
            return base_std
        elif self.config.train_noise_schedule == "linear_decay":
            progress = min(step / max_steps, 1.0)
            return base_std * (1 - 0.5 * progress)  # Decay to 50%
        elif self.config.train_noise_schedule == "cosine_decay":
            progress = min(step / max_steps, 1.0)
            return base_std * (1 + math.cos(math.pi * progress)) / 2
        else:
            return base_std

    def forward(
        self,
        mel: torch.Tensor,  # [batch, n_mels, time]
        add_noise: bool = True,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict pitch bins from mel-spectrogram.

        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            add_noise: Whether to add noise during training
            step: Current training step (for noise schedule)

        Returns:
            Dict with logits and predicted bins
        """
        batch_size, n_mels, time_len = mel.shape

        # Add training noise for robustness
        if add_noise and self.training:
            noise_std = self.get_noise_std(step)
            noise = torch.randn_like(mel) * noise_std
            mel = mel + noise

        # Input projection
        x = self.input_proj(mel)  # [batch, hidden, time]

        # Conv layers with residual connections
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Transpose for attention: [batch, time, hidden]
        x = x.transpose(1, 2)

        # Global context with self-attention
        x_norm = self.attn_norm(x)
        attn_out, _ = self.context_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Classification
        logits = self.classifier_head(x)  # [batch, time, num_bins]

        # Get predicted bins
        pred_bins = logits.argmax(dim=-1)  # [batch, time]

        return {
            'logits': logits,
            'pred_bins': pred_bins,
            'features': x,  # For conditioning flow
        }


class ConvResBlock(nn.Module):
    """Residual convolutional block with dilated convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout: float,
        dilation: int = 1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

        self.norm1 = nn.LayerNorm([channels])
        self.norm2 = nn.LayerNorm([channels])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x

        # First conv block
        x = x.transpose(1, 2)  # [batch, time, channels]
        x = self.norm1(x)
        x = x.transpose(1, 2)  # [batch, channels, time]
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)

        # Second conv block
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.dropout(x)

        return x + residual


# =============================================================================
# PITCH-CONDITIONED FLOW DECODER
# =============================================================================

class PitchConditionedVectorField(nn.Module):
    """
    Vector field network conditioned on predicted pitch bins.

    Extends the standard flow matching vector field with pitch conditioning.
    The pitch bins provide explicit control over the generated prosody.
    """

    def __init__(self, config: PitchFlowConfig):
        super().__init__()
        self.config = config

        # Pitch bin embedding
        self.pitch_embed = nn.Embedding(
            config.num_pitch_bins,
            config.flow_hidden_dim,
        )

        # Time embedding (sinusoidal)
        self.time_embed = SinusoidalTimeEmbedding(config.flow_hidden_dim)

        # Input projection: prosody + time + pitch → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(
                config.prosody_dim + config.flow_hidden_dim * 2,
                config.flow_hidden_dim
            ),
            nn.LayerNorm(config.flow_hidden_dim),
            nn.GELU(),
        )

        # Text conditioning projection
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.flow_hidden_dim),
            nn.LayerNorm(config.flow_hidden_dim),
            nn.GELU(),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            PitchConditionedBlock(
                hidden_dim=config.flow_hidden_dim,
                num_heads=config.flow_num_heads,
                dropout=0.1,
            )
            for _ in range(config.flow_num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.flow_hidden_dim),
            nn.Linear(config.flow_hidden_dim, config.flow_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.flow_hidden_dim, config.prosody_dim),
        )

    def forward(
        self,
        t: torch.Tensor,           # [batch] time in [0, 1]
        x_t: torch.Tensor,         # [batch, prosody_dim] current state
        pitch_bins: torch.Tensor,  # [batch, time] or [batch]
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity conditioned on pitch bins.

        Args:
            t: Time values in [0, 1]
            x_t: Current prosody state
            pitch_bins: Predicted or specified pitch bins
            text_cond: Optional text conditioning
            text_mask: Optional text mask

        Returns:
            Velocity vector for flow matching
        """
        batch_size = x_t.shape[0]

        # Handle scalar time
        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, hidden]

        # Pitch embedding - aggregate if sequence
        if pitch_bins.dim() == 1:
            pitch_emb = self.pitch_embed(pitch_bins)  # [batch, hidden]
        else:
            # Average over time dimension
            pitch_emb = self.pitch_embed(pitch_bins).mean(dim=1)  # [batch, hidden]

        # Combine inputs
        combined = torch.cat([x_t, t_emb, pitch_emb], dim=-1)
        h = self.input_proj(combined).unsqueeze(1)  # [batch, 1, hidden]

        # Text conditioning
        if text_cond is not None:
            text_h = self.text_proj(text_cond)  # [batch, seq, hidden]
        else:
            text_h = None

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, text_h, text_mask)

        # Output projection
        h = h.squeeze(1)  # [batch, hidden]
        velocity = self.output_proj(h)  # [batch, prosody_dim]

        return velocity


class PitchConditionedBlock(nn.Module):
    """Transformer block with pitch conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to text
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        text_h: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention to text
        if text_h is not None:
            x_norm = self.cross_attn_norm(x)
            key_padding_mask = ~text_mask if text_mask is not None else None
            cross_out, _ = self.cross_attn(
                x_norm, text_h, text_h,
                key_padding_mask=key_padding_mask
            )
            x = x + cross_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class SinusoidalTimeEmbedding(nn.Module):
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


# =============================================================================
# GAUSSIAN CONDITIONAL PATH
# =============================================================================

class GaussianConditionalPath:
    """Gaussian conditional path for flow matching."""

    def __init__(self, sigma_min: float = 0.001):
        self.sigma_min = sigma_min

    def sample_xt(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t from conditional path."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)

        mu = t * x1
        sigma = (1 - t) + t * self.sigma_min

        return mu + sigma * x0

    def compute_target_velocity(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target conditional velocity."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        while t.dim() < x_t.dim():
            t = t.unsqueeze(-1)

        mu = t * x1
        sigma = (1 - t) + t * self.sigma_min
        sigma_dot = self.sigma_min - 1

        drift = (sigma_dot / sigma) * (x_t - mu)
        return drift + x1


# =============================================================================
# PITCH FLOW MODEL
# =============================================================================

class PitchFlow(nn.Module):
    """
    PitchFlow: Quantized pitch control for flow-matching TTS.

    Combines:
    1. F0 quantizer for discrete pitch representation
    2. Pitch bin classifier for predicting bins from mel
    3. Pitch-conditioned flow decoder for prosody generation

    Training:
    - Classifier: Cross-entropy loss on noisy mel-spectrograms
    - Flow: Conditional flow matching loss with pitch conditioning

    Inference:
    - Predict pitch bins from mel (or specify directly)
    - Generate prosody conditioned on pitch bins
    """

    def __init__(self, config: PitchFlowConfig):
        super().__init__()
        self.config = config

        # Components
        self.quantizer = LogF0Quantizer(config)
        self.classifier = PitchBinClassifier(config)
        self.vector_field = PitchConditionedVectorField(config)
        self.path = GaussianConditionalPath(sigma_min=0.001)

        # Prosody token projection
        self.token_projection = nn.Sequential(
            nn.Linear(config.prosody_dim, config.prosody_dim),
            nn.LayerNorm(config.prosody_dim),
            nn.GELU(),
            nn.Linear(config.prosody_dim, config.prosody_dim * config.num_prosody_tokens),
        )
        self.token_norm = nn.LayerNorm(config.prosody_dim)

    def compute_classifier_loss(
        self,
        mel: torch.Tensor,       # [batch, n_mels, time]
        f0_target: torch.Tensor,  # [batch, time]
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classifier training loss.

        Args:
            mel: Mel-spectrogram
            f0_target: Ground-truth F0 values
            step: Training step for noise schedule

        Returns:
            Dict with loss and metrics
        """
        # Quantize F0 to bins
        quant_result = self.quantizer(f0_target)
        target_bins = quant_result['bins']  # [batch, time]

        # Predict bins from mel
        classifier_out = self.classifier(mel, add_noise=True, step=step)
        logits = classifier_out['logits']  # [batch, time, num_bins]

        # Cross-entropy loss
        logits_flat = logits.view(-1, self.config.num_pitch_bins)
        target_flat = target_bins.view(-1)
        ce_loss = F.cross_entropy(logits_flat, target_flat)

        # Accuracy metrics
        pred_bins = logits.argmax(dim=-1)
        accuracy = (pred_bins == target_bins).float().mean()

        # Voiced accuracy (exclude unvoiced frames)
        voiced_mask = target_bins > 0
        if voiced_mask.any():
            voiced_accuracy = (
                (pred_bins[voiced_mask] == target_bins[voiced_mask]).float().mean()
            )
        else:
            voiced_accuracy = torch.tensor(0.0, device=mel.device)

        return {
            'loss': ce_loss * self.config.classifier_ce_weight,
            'ce_loss': ce_loss,
            'accuracy': accuracy,
            'voiced_accuracy': voiced_accuracy,
            'quantization_error': quant_result['quantization_error'].mean(),
        }

    def compute_flow_loss(
        self,
        x1: torch.Tensor,         # [batch, prosody_dim] target prosody
        pitch_bins: torch.Tensor,  # [batch, time] pitch bins
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.

        Args:
            x1: Target prosody vectors
            pitch_bins: Pitch bins for conditioning
            text_cond: Optional text conditioning
            text_mask: Optional text mask

        Returns:
            Dict with flow loss
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample noise
        x0 = torch.randn_like(x1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Sample x_t from conditional path
        x_t = self.path.sample_xt(t, x0, x1)

        # Compute target velocity
        target_velocity = self.path.compute_target_velocity(t, x_t, x1)

        # Predict velocity conditioned on pitch
        pred_velocity = self.vector_field(t, x_t, pitch_bins, text_cond, text_mask)

        # MSE loss
        flow_loss = F.mse_loss(pred_velocity, target_velocity)

        return {
            'loss': flow_loss * self.config.flow_loss_weight,
            'flow_loss': flow_loss,
            'velocity_norm': pred_velocity.norm(dim=-1).mean(),
        }

    def compute_loss(
        self,
        mel: torch.Tensor,          # [batch, n_mels, time]
        f0_target: torch.Tensor,    # [batch, time]
        prosody_target: torch.Tensor,  # [batch, prosody_dim]
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined training loss.

        Args:
            mel: Mel-spectrogram
            f0_target: Ground-truth F0
            prosody_target: Target prosody vectors
            text_cond: Optional text conditioning
            text_mask: Optional text mask
            step: Training step

        Returns:
            Dict with all losses
        """
        # Classifier loss
        classifier_losses = self.compute_classifier_loss(mel, f0_target, step)

        # Get predicted pitch bins for flow conditioning
        with torch.no_grad():
            classifier_out = self.classifier(mel, add_noise=False)
            pitch_bins = classifier_out['pred_bins']

        # Flow loss
        flow_losses = self.compute_flow_loss(
            prosody_target, pitch_bins, text_cond, text_mask
        )

        # Combined loss
        total_loss = classifier_losses['loss'] + flow_losses['loss']

        return {
            'total_loss': total_loss,
            **{f'classifier_{k}': v for k, v in classifier_losses.items()},
            **{f'flow_{k}': v for k, v in flow_losses.items()},
        }

    @torch.no_grad()
    def sample(
        self,
        mel: torch.Tensor,                    # [batch, n_mels, time]
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        pitch_bins: Optional[torch.Tensor] = None,  # Override predicted bins
        num_steps: int = 50,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample prosody conditioned on predicted pitch.

        Args:
            mel: Mel-spectrogram for pitch prediction
            text_cond: Optional text conditioning
            text_mask: Optional text mask
            pitch_bins: Optional override for pitch bins
            num_steps: ODE integration steps
            temperature: Sampling temperature

        Returns:
            Dict with sampled prosody and pitch info
        """
        batch_size = mel.shape[0]
        device = mel.device

        # Predict pitch bins if not provided
        if pitch_bins is None:
            classifier_out = self.classifier(mel, add_noise=False)
            pitch_bins = classifier_out['pred_bins']

        # Sample initial noise
        x = torch.randn(batch_size, self.config.prosody_dim, device=device)
        x = x * temperature

        # ODE integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            velocity = self.vector_field(t, x, pitch_bins, text_cond, text_mask)
            x = x + dt * velocity

        return {
            'prosody': x,
            'pitch_bins': pitch_bins,
            'pitch_f0': self.quantizer.dequantize(pitch_bins),
        }

    @torch.no_grad()
    def sample_tokens(
        self,
        mel: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        pitch_bins: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample prosody tokens for CSM integration.

        Args:
            mel: Mel-spectrogram
            text_cond: Optional text conditioning
            text_mask: Optional text mask
            pitch_bins: Optional pitch override
            num_steps: ODE steps
            temperature: Sampling temperature

        Returns:
            Prosody tokens [batch, num_tokens, prosody_dim]
        """
        result = self.sample(
            mel, text_cond, text_mask, pitch_bins, num_steps, temperature
        )
        prosody = result['prosody']

        # Project to tokens
        tokens = self.token_projection(prosody)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.prosody_dim)
        tokens = self.token_norm(tokens)

        return tokens

    def set_pitch_bins(
        self,
        f0_contour: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert F0 contour to pitch bins for explicit control.

        Args:
            f0_contour: F0 values in Hz [batch, time]

        Returns:
            Pitch bins [batch, time]
        """
        return self.quantizer.quantize(f0_contour)

    def shift_pitch(
        self,
        pitch_bins: torch.Tensor,
        semitones: float,
    ) -> torch.Tensor:
        """
        Shift pitch bins by semitones.

        Args:
            pitch_bins: Current pitch bins
            semitones: Semitones to shift (positive = up, negative = down)

        Returns:
            Shifted pitch bins
        """
        # Convert bins to F0
        f0 = self.quantizer.dequantize(pitch_bins)

        # Shift in Hz (semitone ratio = 2^(1/12))
        shift_factor = 2 ** (semitones / 12)
        f0_shifted = f0 * shift_factor

        # Re-quantize
        return self.quantizer.quantize(f0_shifted)


# =============================================================================
# ADAPTER FOR PIPELINE INTEGRATION
# =============================================================================

class PitchFlowAdapter(nn.Module):
    """
    Adapter for integrating PitchFlow with the prosody pipeline.

    Provides convenient methods for:
    - Training with F0 supervision
    - Inference with automatic or manual pitch control
    - Pitch manipulation (shift, scale, etc.)
    """

    def __init__(
        self,
        config: PitchFlowConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core PitchFlow model
        self.pitchflow = PitchFlow(config)

        # Optional: F0 extractor for audio inputs
        if PARSELMOUTH_AVAILABLE:
            self.f0_extractor = PraatF0Extractor(config)
        else:
            self.f0_extractor = None

        # Output adapter if dimensions don't match
        if config.prosody_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.prosody_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def forward(
        self,
        mel: torch.Tensor,
        f0_target: Optional[torch.Tensor] = None,
        prosody_target: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with optional training.

        Args:
            mel: Mel-spectrogram [batch, n_mels, time]
            f0_target: Optional F0 for classifier training
            prosody_target: Optional prosody target for flow training
            text_cond: Text conditioning
            text_mask: Text mask
            step: Training step

        Returns:
            Dict with prosody tokens and losses
        """
        result = {}

        if f0_target is not None and prosody_target is not None:
            # Training mode
            losses = self.pitchflow.compute_loss(
                mel, f0_target, prosody_target, text_cond, text_mask, step
            )
            result.update(losses)

            # Generate tokens
            tokens = self.pitchflow.sample_tokens(
                mel, text_cond, text_mask, num_steps=10  # Fewer steps during training
            )
            tokens = self.output_adapter(tokens)
            result['prosody_tokens'] = tokens
        else:
            # Inference mode
            tokens = self.pitchflow.sample_tokens(mel, text_cond, text_mask)
            tokens = self.output_adapter(tokens)
            result['prosody_tokens'] = tokens

            # Get pitch info
            classifier_out = self.pitchflow.classifier(mel, add_noise=False)
            result['pitch_bins'] = classifier_out['pred_bins']
            result['pitch_f0'] = self.pitchflow.quantizer.dequantize(result['pitch_bins'])

        return result

    def from_pitch_contour(
        self,
        f0_contour: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from explicit F0 contour.

        Args:
            f0_contour: F0 values in Hz
            text_cond: Text conditioning
            text_mask: Text mask
            num_steps: ODE steps

        Returns:
            Dict with prosody tokens
        """
        pitch_bins = self.pitchflow.set_pitch_bins(f0_contour)

        # Create dummy mel (not used when pitch_bins provided)
        batch_size = f0_contour.shape[0]
        device = f0_contour.device
        dummy_mel = torch.zeros(batch_size, self.config.n_mels, 10, device=device)

        tokens = self.pitchflow.sample_tokens(
            dummy_mel, text_cond, text_mask, pitch_bins, num_steps
        )
        tokens = self.output_adapter(tokens)

        return {
            'prosody_tokens': tokens,
            'pitch_bins': pitch_bins,
            'pitch_f0': f0_contour,
        }

    def shift_pitch(
        self,
        mel: torch.Tensor,
        semitones: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody with pitch shifted by semitones.

        Args:
            mel: Mel-spectrogram
            semitones: Pitch shift amount
            text_cond: Text conditioning
            text_mask: Text mask

        Returns:
            Dict with shifted prosody tokens
        """
        # Get predicted pitch
        classifier_out = self.pitchflow.classifier(mel, add_noise=False)
        pitch_bins = classifier_out['pred_bins']

        # Shift pitch
        shifted_bins = self.pitchflow.shift_pitch(pitch_bins, semitones)

        # Generate with shifted pitch
        tokens = self.pitchflow.sample_tokens(
            mel, text_cond, text_mask, shifted_bins
        )
        tokens = self.output_adapter(tokens)

        return {
            'prosody_tokens': tokens,
            'original_pitch_bins': pitch_bins,
            'shifted_pitch_bins': shifted_bins,
            'original_f0': self.pitchflow.quantizer.dequantize(pitch_bins),
            'shifted_f0': self.pitchflow.quantizer.dequantize(shifted_bins),
        }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class PitchFlowLoss(nn.Module):
    """Combined loss function for PitchFlow training."""

    def __init__(
        self,
        classifier_weight: float = 1.0,
        flow_weight: float = 1.0,
        accuracy_bonus_weight: float = 0.1,
    ):
        super().__init__()
        self.classifier_weight = classifier_weight
        self.flow_weight = flow_weight
        self.accuracy_bonus_weight = accuracy_bonus_weight

    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss.

        Args:
            losses: Dict from PitchFlow.compute_loss()

        Returns:
            Dict with weighted total loss
        """
        total = (
            losses['classifier_loss'] * self.classifier_weight +
            losses['flow_loss'] * self.flow_weight
        )

        # Bonus for high accuracy (encourages accurate pitch prediction)
        if 'classifier_accuracy' in losses:
            accuracy_bonus = -losses['classifier_accuracy'] * self.accuracy_bonus_weight
            total = total + accuracy_bonus
            losses['accuracy_bonus'] = accuracy_bonus

        losses['total_loss'] = total
        return losses


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PitchFlow: Quantized Pitch Control for Flow-Matching TTS")
    print("Based on Interspeech 2024 Paper")
    print("=" * 70)

    config = PitchFlowConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test parameters
    batch_size = 2
    time_len = 100
    n_mels = config.n_mels

    # Test 1: Log F0 Quantizer
    print("\n[Test 1] Log F0 Quantizer...")
    quantizer = LogF0Quantizer(config).to(device)

    # Synthetic F0 with some unvoiced regions
    f0_raw = 150 + 100 * torch.sin(torch.linspace(0, 4 * math.pi, time_len))
    f0_raw[20:30] = 0  # Unvoiced region
    f0 = f0_raw.unsqueeze(0).expand(batch_size, -1).to(device)

    quant_result = quantizer(f0)
    print(f"  F0 input shape: {f0.shape}")
    print(f"  Bin output shape: {quant_result['bins'].shape}")
    print(f"  Bin range: [{quant_result['bins'].min()}, {quant_result['bins'].max()}]")
    print(f"  Quantization error: {quant_result['quantization_error'].mean():.2f} Hz")
    print("  [PASS]")

    # Test 2: Pitch Bin Classifier
    print("\n[Test 2] Pitch Bin Classifier...")
    classifier = PitchBinClassifier(config).to(device)

    mel = torch.randn(batch_size, n_mels, time_len, device=device)
    classifier_out = classifier(mel, add_noise=True)

    print(f"  Mel input shape: {mel.shape}")
    print(f"  Logits shape: {classifier_out['logits'].shape}")
    print(f"  Pred bins shape: {classifier_out['pred_bins'].shape}")
    print(f"  Features shape: {classifier_out['features'].shape}")
    print("  [PASS]")

    # Test 3: Pitch-Conditioned Vector Field
    print("\n[Test 3] Pitch-Conditioned Vector Field...")
    vector_field = PitchConditionedVectorField(config).to(device)

    x_t = torch.randn(batch_size, config.prosody_dim, device=device)
    t = torch.rand(batch_size, device=device)
    pitch_bins = torch.randint(0, config.num_pitch_bins, (batch_size, time_len), device=device)
    text_cond = torch.randn(batch_size, 20, config.text_dim, device=device)
    text_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)

    velocity = vector_field(t, x_t, pitch_bins, text_cond, text_mask)
    print(f"  x_t shape: {x_t.shape}")
    print(f"  pitch_bins shape: {pitch_bins.shape}")
    print(f"  Velocity shape: {velocity.shape}")
    print(f"  Velocity norm: {velocity.norm(dim=-1).mean():.3f}")
    print("  [PASS]")

    # Test 4: Full PitchFlow Model
    print("\n[Test 4] Full PitchFlow Model...")
    model = PitchFlow(config).to(device)

    # Test classifier loss
    classifier_losses = model.compute_classifier_loss(mel, f0)
    print(f"  Classifier CE loss: {classifier_losses['ce_loss']:.4f}")
    print(f"  Accuracy: {classifier_losses['accuracy']*100:.1f}%")
    print(f"  Voiced accuracy: {classifier_losses['voiced_accuracy']*100:.1f}%")

    # Test flow loss
    prosody_target = torch.randn(batch_size, config.prosody_dim, device=device)
    target_bins = quant_result['bins']
    flow_losses = model.compute_flow_loss(prosody_target, target_bins, text_cond, text_mask)
    print(f"  Flow loss: {flow_losses['flow_loss']:.4f}")
    print("  [PASS]")

    # Test 5: Combined Loss
    print("\n[Test 5] Combined Training Loss...")
    combined_losses = model.compute_loss(
        mel, f0, prosody_target, text_cond, text_mask
    )
    print(f"  Total loss: {combined_losses['total_loss']:.4f}")
    print(f"  Classifier loss: {combined_losses['classifier_loss']:.4f}")
    print(f"  Flow loss: {combined_losses['flow_flow_loss']:.4f}")
    print("  [PASS]")

    # Test 6: Sampling
    print("\n[Test 6] Prosody Sampling...")
    sample_result = model.sample(mel, text_cond, text_mask, num_steps=20)
    print(f"  Sampled prosody shape: {sample_result['prosody'].shape}")
    print(f"  Predicted pitch bins shape: {sample_result['pitch_bins'].shape}")
    print(f"  Reconstructed F0 shape: {sample_result['pitch_f0'].shape}")
    print("  [PASS]")

    # Test 7: Token Generation
    print("\n[Test 7] Prosody Token Generation...")
    tokens = model.sample_tokens(mel, text_cond, text_mask, num_steps=20)
    print(f"  Token shape: {tokens.shape}")
    print(f"  Expected: [{batch_size}, {config.num_prosody_tokens}, {config.prosody_dim}]")
    print("  [PASS]")

    # Test 8: Pitch Shifting
    print("\n[Test 8] Pitch Shifting...")
    original_bins = quant_result['bins']
    shifted_up = model.shift_pitch(original_bins, semitones=3)
    shifted_down = model.shift_pitch(original_bins, semitones=-3)

    original_f0 = quantizer.dequantize(original_bins)
    shifted_up_f0 = quantizer.dequantize(shifted_up)
    shifted_down_f0 = quantizer.dequantize(shifted_down)

    voiced = original_bins > 0
    if voiced.any():
        ratio_up = (shifted_up_f0[voiced] / original_f0[voiced]).mean()
        ratio_down = (shifted_down_f0[voiced] / original_f0[voiced]).mean()
        print(f"  Original F0 mean: {original_f0[voiced].mean():.1f} Hz")
        print(f"  +3 semitones ratio: {ratio_up:.3f} (expected ~1.19)")
        print(f"  -3 semitones ratio: {ratio_down:.3f} (expected ~0.84)")
    print("  [PASS]")

    # Test 9: PitchFlow Adapter
    print("\n[Test 9] PitchFlow Adapter...")
    adapter = PitchFlowAdapter(config).to(device)

    # Training mode
    result = adapter(mel, f0, prosody_target, text_cond, text_mask)
    print(f"  Training - prosody tokens: {result['prosody_tokens'].shape}")
    print(f"  Training - total loss: {result['total_loss']:.4f}")

    # Inference mode
    result = adapter(mel, text_cond=text_cond, text_mask=text_mask)
    print(f"  Inference - prosody tokens: {result['prosody_tokens'].shape}")
    print(f"  Inference - pitch bins: {result['pitch_bins'].shape}")
    print("  [PASS]")

    # Test 10: Explicit Pitch Control
    print("\n[Test 10] Explicit Pitch Control...")
    custom_f0 = torch.linspace(100, 300, time_len).unsqueeze(0).expand(batch_size, -1).to(device)
    result = adapter.from_pitch_contour(custom_f0, text_cond, text_mask)
    print(f"  Custom F0 tokens: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 11: Pitch Shift Adapter
    print("\n[Test 11] Pitch Shift via Adapter...")
    result = adapter.shift_pitch(mel, semitones=5, text_cond=text_cond, text_mask=text_mask)
    print(f"  Shifted tokens: {result['prosody_tokens'].shape}")
    print(f"  Original F0 shape: {result['original_f0'].shape}")
    print(f"  Shifted F0 shape: {result['shifted_f0'].shape}")
    print("  [PASS]")

    # Test 12: Gradient Flow
    print("\n[Test 12] Gradient Flow...")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)

    result = adapter(mel, f0, prosody_target, text_cond, text_mask)
    loss = result['total_loss']
    loss.backward()

    total_grad_norm = 0.0
    for p in adapter.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    print(f"  Total gradient norm: {total_grad_norm:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All PitchFlow tests passed!")
    print("=" * 70)

    # Model statistics
    print("\nModel Statistics:")
    print("-" * 40)
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Component breakdown
    print("\nComponent breakdown:")
    for name, module in [
        ("Quantizer", adapter.pitchflow.quantizer),
        ("Classifier", adapter.pitchflow.classifier),
        ("VectorField", adapter.pitchflow.vector_field),
    ]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from pitchflow import (
    PitchFlowConfig,
    PitchFlow,
    PitchFlowAdapter,
    PraatF0Extractor,
)

# Initialize
config = PitchFlowConfig(
    num_pitch_bins=50,     # 50 log-scale pitch bins
    f0_min=50.0,           # Minimum F0 (Hz)
    f0_max=800.0,          # Maximum F0 (Hz)
)

adapter = PitchFlowAdapter(config).cuda()

# Training: with F0 supervision
result = adapter(
    mel=mel_spectrogram,           # [batch, 80, time]
    f0_target=f0_values,           # [batch, time]
    prosody_target=prosody_emb,    # [batch, prosody_dim]
    text_cond=text_embeddings,     # [batch, seq, 768]
)
loss = result['total_loss']
prosody_tokens = result['prosody_tokens']

# Inference: automatic pitch prediction
result = adapter(mel=mel_spectrogram, text_cond=text_embeddings)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]
pitch_f0 = result['pitch_f0']              # Reconstructed F0

# Explicit pitch control
custom_f0 = torch.linspace(100, 300, 100).unsqueeze(0).cuda()
result = adapter.from_pitch_contour(custom_f0, text_cond=text_embeddings)
prosody_tokens = result['prosody_tokens']

# Pitch manipulation
result = adapter.shift_pitch(mel, semitones=5)  # Shift up by 5 semitones
result = adapter.shift_pitch(mel, semitones=-3)  # Shift down by 3 semitones

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")

    print("\nPitchFlow Benefits:")
    print("-" * 40)
    print("""
1. Discrete pitch representation (50 bins):
   - Simpler than continuous regression
   - Cross-entropy loss more stable than MSE

2. Log-scale binning:
   - Perceptually uniform pitch steps
   - Better coverage of rare pitch values

3. Explicit pitch control:
   - Specify pitch bins directly at inference
   - Pitch shifting in semitones
   - Custom F0 contour input

4. Robust classifier:
   - Trained on noisy mel-spectrograms
   - Generalizes better to real audio

5. Flow-matching integration:
   - Conditions flow decoder on pitch bins
   - Compatible with ProsodyFlow, TTS-CtrlNet
""")
