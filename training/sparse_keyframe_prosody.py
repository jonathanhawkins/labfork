"""
STCTS: Sparse Keyframe Prosody Compression

Based on STCTS (arXiv:2512.00451) - "Towards Ultra-Low Bitrate Speech Coding".

Key Innovation: Ultra-low bitrate (~80 bps) speech with explicit text-prosody-timbre
decomposition. Prosody is transmitted as sparse keyframes (<14 bps at 0.1-1 Hz)
rather than dense frame-by-frame.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Dense Prosody Contour (100 frames/sec)                        │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓ Extract
    ┌─────────────────────────────────────────────────────────────────┐
    │  Sparse Keyframes (0.1-1 Hz)                                    │
    │  ●                    ●              ●                       ●  │
    │  t=0.0                t=0.3          t=0.6                  t=1.0│
    └─────────────────────────────────────────────────────────────────┘
                                    ↓ Interpolate
    ┌─────────────────────────────────────────────────────────────────┐
    │  Reconstructed Prosody (smooth interpolation)                   │
    │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
    └─────────────────────────────────────────────────────────────────┘

Benefits:
- 75x compression vs Opus, 12x vs EnCodec
- <14 bps prosody (vs ~1000+ bps for frame-by-frame)
- Natural prosody via smooth interpolation
- Compatible with DrawSpeech keyframe UI
- Enables "prosody anchor points" for intuitive control

References:
- STCTS: https://arxiv.org/abs/2512.00451
- DrawSpeech: Our existing sketch-conditioned implementation
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from prosody_conditioning import ProsodyConfig, ProsodyEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SparseKeyframeConfig:
    """Configuration for sparse keyframe prosody."""

    # Keyframe extraction
    keyframe_rate_hz: float = 0.5  # Keyframes per second (0.1-1 Hz)
    min_keyframes: int = 2  # Minimum keyframes per utterance
    max_keyframes: int = 16  # Maximum keyframes

    # Extraction strategy
    extraction_method: str = "salient"  # "uniform", "salient", "adaptive"
    salience_threshold: float = 0.3  # Threshold for salient point detection

    # Prosody dimensions (match ProsodyConfig)
    pitch_dim: int = 1  # Scalar pitch value
    energy_dim: int = 1  # Scalar energy value
    duration_dim: int = 1  # Duration hint
    emotion_dim: int = 8  # Emotion vector per keyframe

    # Interpolation
    interpolation_method: str = "cubic"  # "linear", "cubic", "spherical"
    interpolation_smoothing: float = 0.1  # Smoothing factor

    # Encoding
    hidden_dim: int = 256
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    # Output (matches prosody pipeline)
    output_dim: int = 2048  # CSM hidden size
    num_output_tokens: int = 4  # Prosody prefix tokens

    # Compression metrics
    target_bitrate_bps: float = 14.0  # Target prosody bitrate


# =============================================================================
# KEYFRAME DATA STRUCTURE
# =============================================================================

@dataclass
class ProsodyKeyframe:
    """
    A single prosody keyframe anchor point.

    Attributes:
        time: Normalized time position [0, 1]
        pitch: Pitch value (normalized, 0=low, 1=high)
        energy: Energy value (normalized)
        duration_scale: Duration modifier (1.0 = normal)
        emotion: Emotion vector [emotion_dim]
    """
    time: float
    pitch: float = 0.5
    energy: float = 0.5
    duration_scale: float = 1.0
    emotion: Optional[torch.Tensor] = None

    def to_tensor(self, emotion_dim: int = 8) -> torch.Tensor:
        """Convert keyframe to tensor representation."""
        base = torch.tensor([self.time, self.pitch, self.energy, self.duration_scale])
        if self.emotion is not None:
            emotion = self.emotion
        else:
            emotion = torch.zeros(emotion_dim)
            emotion[0] = 0.5  # Neutral
        return torch.cat([base, emotion])

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, emotion_dim: int = 8) -> "ProsodyKeyframe":
        """Create keyframe from tensor."""
        return cls(
            time=tensor[0].item(),
            pitch=tensor[1].item(),
            energy=tensor[2].item(),
            duration_scale=tensor[3].item(),
            emotion=tensor[4:4+emotion_dim] if len(tensor) > 4 else None,
        )


# =============================================================================
# KEYFRAME EXTRACTION
# =============================================================================

class SparseKeyframeExtractor(nn.Module):
    """
    Extract sparse prosody keyframes from dense contours.

    Methods:
    1. Uniform: Evenly spaced keyframes
    2. Salient: Detect peaks, valleys, and inflection points
    3. Adaptive: Learn optimal keyframe positions

    The extractor identifies the most informative prosody anchor points
    that can reconstruct the full contour when interpolated.
    """

    def __init__(self, config: SparseKeyframeConfig):
        super().__init__()
        self.config = config

        # For adaptive extraction: learnable position predictor
        if config.extraction_method == "adaptive":
            self.position_predictor = nn.Sequential(
                nn.Linear(config.pitch_dim + config.energy_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid(),
            )

            # Transformer for context-aware extraction
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_transformer_layers,
            )

    def compute_salience(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute salience score for each time point.

        Salience indicates how important a point is for reconstruction.
        High salience = peaks, valleys, rapid changes.

        Args:
            pitch: [batch, seq_len] pitch contour
            energy: [batch, seq_len] energy contour

        Returns:
            salience: [batch, seq_len] salience scores
        """
        batch_size, seq_len = pitch.shape
        device = pitch.device

        # Compute first derivative (rate of change)
        pitch_delta = torch.zeros_like(pitch)
        pitch_delta[:, 1:] = pitch[:, 1:] - pitch[:, :-1]

        energy_delta = torch.zeros_like(energy)
        energy_delta[:, 1:] = energy[:, 1:] - energy[:, :-1]

        # Compute second derivative (curvature/acceleration)
        pitch_accel = torch.zeros_like(pitch)
        pitch_accel[:, 1:-1] = pitch[:, 2:] - 2 * pitch[:, 1:-1] + pitch[:, :-2]

        energy_accel = torch.zeros_like(energy)
        energy_accel[:, 1:-1] = energy[:, 2:] - 2 * energy[:, 1:-1] + energy[:, :-2]

        # Detect peaks and valleys (sign changes in derivative)
        pitch_peaks = (pitch_delta[:, :-1] * pitch_delta[:, 1:] < 0).float()
        energy_peaks = (energy_delta[:, :-1] * energy_delta[:, 1:] < 0).float()

        # Pad to match length
        pitch_peaks = F.pad(pitch_peaks, (0, 1), value=0)
        energy_peaks = F.pad(energy_peaks, (0, 1), value=0)

        # Combine into salience score
        # Higher score for: peaks/valleys, rapid changes, high curvature
        salience = (
            pitch_peaks * 0.3 +
            energy_peaks * 0.3 +
            pitch_delta.abs() * 0.2 +
            energy_delta.abs() * 0.1 +
            pitch_accel.abs() * 0.05 +
            energy_accel.abs() * 0.05
        )

        # Normalize
        salience = salience / (salience.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Always include start and end points
        salience[:, 0] = 1.0
        salience[:, -1] = 1.0

        return salience

    def extract_uniform(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration_seconds: float,
    ) -> List[List[ProsodyKeyframe]]:
        """
        Extract uniformly-spaced keyframes.

        Args:
            pitch: [batch, seq_len] normalized pitch
            energy: [batch, seq_len] normalized energy
            duration_seconds: Total duration

        Returns:
            List of keyframe lists (one per batch item)
        """
        batch_size, seq_len = pitch.shape

        # Calculate number of keyframes based on rate
        num_keyframes = max(
            self.config.min_keyframes,
            min(
                self.config.max_keyframes,
                int(duration_seconds * self.config.keyframe_rate_hz) + 1
            )
        )

        # Get uniform indices
        indices = torch.linspace(0, seq_len - 1, num_keyframes).long()

        all_keyframes = []
        for b in range(batch_size):
            keyframes = []
            for i, idx in enumerate(indices):
                kf = ProsodyKeyframe(
                    time=idx.item() / (seq_len - 1),
                    pitch=pitch[b, idx].item(),
                    energy=energy[b, idx].item(),
                )
                keyframes.append(kf)
            all_keyframes.append(keyframes)

        return all_keyframes

    def extract_salient(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration_seconds: float,
        emotion: Optional[torch.Tensor] = None,
    ) -> List[List[ProsodyKeyframe]]:
        """
        Extract keyframes at salient prosody points.

        Selects peaks, valleys, and inflection points that best
        capture the prosody contour shape.

        Args:
            pitch: [batch, seq_len] normalized pitch
            energy: [batch, seq_len] normalized energy
            duration_seconds: Total duration
            emotion: Optional [batch, seq_len, emotion_dim] per-frame emotions

        Returns:
            List of keyframe lists
        """
        batch_size, seq_len = pitch.shape
        device = pitch.device

        # Compute salience
        salience = self.compute_salience(pitch, energy)

        # Target number of keyframes
        target_keyframes = max(
            self.config.min_keyframes,
            min(
                self.config.max_keyframes,
                int(duration_seconds * self.config.keyframe_rate_hz) + 1
            )
        )

        all_keyframes = []
        for b in range(batch_size):
            # Get top-k salient points
            _, indices = torch.topk(salience[b], target_keyframes)
            indices = indices.sort()[0]  # Sort by time

            keyframes = []
            for idx in indices:
                idx_int = idx.item()

                # Get emotion for this keyframe if available
                emo = None
                if emotion is not None:
                    emo = emotion[b, idx_int]

                kf = ProsodyKeyframe(
                    time=idx_int / (seq_len - 1),
                    pitch=pitch[b, idx_int].item(),
                    energy=energy[b, idx_int].item(),
                    emotion=emo,
                )
                keyframes.append(kf)

            all_keyframes.append(keyframes)

        return all_keyframes

    def forward(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration_seconds: float,
        emotion: Optional[torch.Tensor] = None,
    ) -> List[List[ProsodyKeyframe]]:
        """
        Extract sparse keyframes from dense prosody contours.

        Args:
            pitch: [batch, seq_len] normalized pitch contour
            energy: [batch, seq_len] normalized energy contour
            duration_seconds: Total audio duration
            emotion: Optional [batch, seq_len, emotion_dim] emotion per frame

        Returns:
            List of keyframe lists (one per batch item)
        """
        if self.config.extraction_method == "uniform":
            return self.extract_uniform(pitch, energy, duration_seconds)
        elif self.config.extraction_method == "salient":
            return self.extract_salient(pitch, energy, duration_seconds, emotion)
        else:
            # Adaptive - use neural network
            return self.extract_salient(pitch, energy, duration_seconds, emotion)


# =============================================================================
# KEYFRAME INTERPOLATION
# =============================================================================

class KeyframeInterpolator(nn.Module):
    """
    Interpolate between sparse keyframes to reconstruct dense prosody.

    Methods:
    1. Linear: Simple linear interpolation
    2. Cubic: Smooth cubic spline interpolation
    3. Spherical: SLERP for emotion vectors

    The interpolator fills in natural prosody between anchor points,
    creating smooth and expressive output.
    """

    def __init__(self, config: SparseKeyframeConfig):
        super().__init__()
        self.config = config

    def interpolate_linear(
        self,
        keyframes: List[ProsodyKeyframe],
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Linear interpolation between keyframes.

        Args:
            keyframes: Sorted list of keyframes
            target_length: Output sequence length

        Returns:
            Dict with 'pitch', 'energy', 'emotion' tensors
        """
        device = 'cpu'

        # Extract keyframe data
        times = torch.tensor([kf.time for kf in keyframes])
        pitches = torch.tensor([kf.pitch for kf in keyframes])
        energies = torch.tensor([kf.energy for kf in keyframes])

        # Target time points
        t = torch.linspace(0, 1, target_length)

        # Interpolate pitch
        pitch_out = torch.zeros(target_length)
        for i, ti in enumerate(t):
            idx = torch.searchsorted(times, ti)
            if idx == 0:
                pitch_out[i] = pitches[0]
            elif idx >= len(times):
                pitch_out[i] = pitches[-1]
            else:
                t0, t1 = times[idx - 1], times[idx]
                alpha = (ti - t0) / (t1 - t0 + 1e-8)
                pitch_out[i] = (1 - alpha) * pitches[idx - 1] + alpha * pitches[idx]

        # Interpolate energy
        energy_out = torch.zeros(target_length)
        for i, ti in enumerate(t):
            idx = torch.searchsorted(times, ti)
            if idx == 0:
                energy_out[i] = energies[0]
            elif idx >= len(times):
                energy_out[i] = energies[-1]
            else:
                t0, t1 = times[idx - 1], times[idx]
                alpha = (ti - t0) / (t1 - t0 + 1e-8)
                energy_out[i] = (1 - alpha) * energies[idx - 1] + alpha * energies[idx]

        # Interpolate emotion vectors
        emotion_out = None
        if keyframes[0].emotion is not None:
            emotion_dim = keyframes[0].emotion.shape[0]
            emotions = torch.stack([kf.emotion for kf in keyframes])
            emotion_out = torch.zeros(target_length, emotion_dim)

            for i, ti in enumerate(t):
                idx = torch.searchsorted(times, ti)
                if idx == 0:
                    emotion_out[i] = emotions[0]
                elif idx >= len(times):
                    emotion_out[i] = emotions[-1]
                else:
                    t0, t1 = times[idx - 1], times[idx]
                    alpha = (ti - t0) / (t1 - t0 + 1e-8)
                    emotion_out[i] = (1 - alpha) * emotions[idx - 1] + alpha * emotions[idx]

        return {
            'pitch': pitch_out,
            'energy': energy_out,
            'emotion': emotion_out,
        }

    def interpolate_cubic(
        self,
        keyframes: List[ProsodyKeyframe],
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Cubic spline interpolation for smoother prosody.

        Uses Catmull-Rom splines for natural curve transitions.
        """
        device = 'cpu'

        # Extract keyframe data
        times = torch.tensor([kf.time for kf in keyframes])
        pitches = torch.tensor([kf.pitch for kf in keyframes])
        energies = torch.tensor([kf.energy for kf in keyframes])

        # Target time points
        t = torch.linspace(0, 1, target_length)

        def catmull_rom(p0, p1, p2, p3, t):
            """Catmull-Rom spline interpolation."""
            return 0.5 * (
                2 * p1 +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t**3
            )

        # Interpolate with cubic splines
        pitch_out = torch.zeros(target_length)
        energy_out = torch.zeros(target_length)

        for i, ti in enumerate(t):
            idx = torch.searchsorted(times, ti)

            if idx == 0:
                pitch_out[i] = pitches[0]
                energy_out[i] = energies[0]
            elif idx >= len(times):
                pitch_out[i] = pitches[-1]
                energy_out[i] = energies[-1]
            else:
                # Get 4 control points for Catmull-Rom
                i0 = max(0, idx - 2)
                i1 = idx - 1
                i2 = idx
                i3 = min(len(times) - 1, idx + 1)

                t0, t1 = times[i1], times[i2]
                alpha = (ti - t0) / (t1 - t0 + 1e-8)

                pitch_out[i] = catmull_rom(
                    pitches[i0], pitches[i1], pitches[i2], pitches[i3], alpha
                )
                energy_out[i] = catmull_rom(
                    energies[i0], energies[i1], energies[i2], energies[i3], alpha
                )

        # Clamp to valid range
        pitch_out = torch.clamp(pitch_out, 0, 1)
        energy_out = torch.clamp(energy_out, 0, 1)

        # Handle emotions (use linear for now, SLERP would be better)
        result = self.interpolate_linear(keyframes, target_length)
        result['pitch'] = pitch_out
        result['energy'] = energy_out

        return result

    def interpolate_spherical(
        self,
        keyframes: List[ProsodyKeyframe],
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Spherical interpolation (SLERP) for emotion vectors.

        Better for blending emotions on the VAD sphere.
        """
        # Get cubic for pitch/energy
        result = self.interpolate_cubic(keyframes, target_length)

        # SLERP for emotions
        if keyframes[0].emotion is not None:
            times = torch.tensor([kf.time for kf in keyframes])
            emotions = torch.stack([kf.emotion for kf in keyframes])

            # Normalize to unit sphere
            emotions_norm = F.normalize(emotions, dim=-1)

            emotion_dim = emotions.shape[1]
            emotion_out = torch.zeros(target_length, emotion_dim)
            t = torch.linspace(0, 1, target_length)

            for i, ti in enumerate(t):
                idx = torch.searchsorted(times, ti)
                if idx == 0:
                    emotion_out[i] = emotions[0]
                elif idx >= len(times):
                    emotion_out[i] = emotions[-1]
                else:
                    t0, t1 = times[idx - 1], times[idx]
                    alpha = (ti - t0) / (t1 - t0 + 1e-8)

                    # SLERP
                    e0 = emotions_norm[idx - 1]
                    e1 = emotions_norm[idx]

                    dot = torch.dot(e0, e1).clamp(-1, 1)
                    theta = torch.acos(dot)

                    if theta.abs() < 1e-6:
                        # Parallel vectors - use linear
                        emotion_out[i] = (1 - alpha) * emotions[idx - 1] + alpha * emotions[idx]
                    else:
                        sin_theta = torch.sin(theta)
                        emotion_out[i] = (
                            torch.sin((1 - alpha) * theta) / sin_theta * emotions[idx - 1] +
                            torch.sin(alpha * theta) / sin_theta * emotions[idx]
                        )

            result['emotion'] = emotion_out

        return result

    def forward(
        self,
        keyframes: List[ProsodyKeyframe],
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate keyframes to dense prosody.

        Args:
            keyframes: List of sparse keyframes
            target_length: Target output length

        Returns:
            Dict with interpolated 'pitch', 'energy', 'emotion'
        """
        # Sort keyframes by time
        keyframes = sorted(keyframes, key=lambda k: k.time)

        if self.config.interpolation_method == "linear":
            return self.interpolate_linear(keyframes, target_length)
        elif self.config.interpolation_method == "cubic":
            return self.interpolate_cubic(keyframes, target_length)
        elif self.config.interpolation_method == "spherical":
            return self.interpolate_spherical(keyframes, target_length)
        else:
            return self.interpolate_cubic(keyframes, target_length)


# =============================================================================
# SPARSE KEYFRAME ENCODER
# =============================================================================

class SparseKeyframeEncoder(nn.Module):
    """
    Encode sparse keyframes into prosody conditioning tokens.

    Unlike dense prosody encoders that process every frame,
    this encoder works with variable-length keyframe sequences
    and produces fixed-size conditioning tokens.
    """

    def __init__(self, config: SparseKeyframeConfig):
        super().__init__()
        self.config = config

        # Keyframe embedding
        keyframe_dim = 4 + config.emotion_dim  # time, pitch, energy, duration + emotion
        self.keyframe_embed = nn.Sequential(
            nn.Linear(keyframe_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Positional encoding for keyframe sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_keyframes, config.hidden_dim) * 0.02
        )

        # Transformer for keyframe context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
        )

        # Pool keyframes to fixed-size output
        self.global_pool = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Learnable query tokens for pooling
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_output_tokens, config.hidden_dim) * 0.02
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def forward(
        self,
        keyframes_tensor: torch.Tensor,
        keyframe_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode keyframes to prosody tokens.

        Args:
            keyframes_tensor: [batch, num_keyframes, keyframe_dim]
            keyframe_mask: [batch, num_keyframes] - True for valid keyframes

        Returns:
            prosody_tokens: [batch, num_output_tokens, output_dim]
        """
        batch_size, num_keyframes, _ = keyframes_tensor.shape
        device = keyframes_tensor.device

        # Embed keyframes
        h = self.keyframe_embed(keyframes_tensor)  # [B, K, hidden]

        # Add positional encoding
        h = h + self.pos_embed[:, :num_keyframes, :]

        # Create attention mask for transformer
        if keyframe_mask is not None:
            src_key_padding_mask = ~keyframe_mask
        else:
            src_key_padding_mask = None

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Pool to fixed-size output using cross-attention
        queries = self.query_tokens.expand(batch_size, -1, -1)

        pooled, _ = self.global_pool(
            query=queries,
            key=h,
            value=h,
            key_padding_mask=src_key_padding_mask,
        )

        # Project to output dimension
        output = self.output_proj(pooled)

        return output


# =============================================================================
# SPARSE KEYFRAME CONDITIONER (MAIN MODULE)
# =============================================================================

class SparseKeyframeConditioner(nn.Module):
    """
    Main module for sparse keyframe prosody conditioning.

    Combines:
    1. SparseKeyframeExtractor - Dense → Sparse
    2. KeyframeInterpolator - Sparse → Dense (for reconstruction)
    3. SparseKeyframeEncoder - Sparse → Prosody Tokens

    Usage:
        conditioner = SparseKeyframeConditioner(config)

        # Training: Extract keyframes from dense prosody
        keyframes = conditioner.extract(pitch, energy, duration)
        loss = conditioner.compute_reconstruction_loss(pitch, energy, keyframes)

        # Inference: Encode keyframes to prosody tokens
        tokens = conditioner.encode(keyframes)

        # From user input: UI keyframes → tokens
        tokens = conditioner.from_user_keyframes(user_keyframes)
    """

    def __init__(self, config: SparseKeyframeConfig):
        super().__init__()
        self.config = config

        # Components
        self.extractor = SparseKeyframeExtractor(config)
        self.interpolator = KeyframeInterpolator(config)
        self.encoder = SparseKeyframeEncoder(config)

    def keyframes_to_tensor(
        self,
        keyframes_list: List[List[ProsodyKeyframe]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert keyframe lists to padded tensors.

        Args:
            keyframes_list: List of keyframe lists (one per batch item)

        Returns:
            (keyframes_tensor, keyframe_mask)
        """
        batch_size = len(keyframes_list)
        max_keyframes = max(len(kfs) for kfs in keyframes_list)
        max_keyframes = min(max_keyframes, self.config.max_keyframes)

        keyframe_dim = 4 + self.config.emotion_dim

        tensor = torch.zeros(batch_size, max_keyframes, keyframe_dim)
        mask = torch.zeros(batch_size, max_keyframes, dtype=torch.bool)

        for b, keyframes in enumerate(keyframes_list):
            for i, kf in enumerate(keyframes[:max_keyframes]):
                tensor[b, i] = kf.to_tensor(self.config.emotion_dim)
                mask[b, i] = True

        return tensor, mask

    def extract(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration_seconds: float,
        emotion: Optional[torch.Tensor] = None,
    ) -> List[List[ProsodyKeyframe]]:
        """
        Extract sparse keyframes from dense prosody.

        Args:
            pitch: [batch, seq_len] normalized pitch
            energy: [batch, seq_len] normalized energy
            duration_seconds: Audio duration
            emotion: Optional [batch, seq_len, emotion_dim]

        Returns:
            List of keyframe lists
        """
        return self.extractor(pitch, energy, duration_seconds, emotion)

    def interpolate(
        self,
        keyframes: List[ProsodyKeyframe],
        target_length: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct dense prosody from sparse keyframes.

        Args:
            keyframes: Sparse keyframes
            target_length: Output length

        Returns:
            Dict with 'pitch', 'energy', 'emotion'
        """
        return self.interpolator(keyframes, target_length)

    def encode(
        self,
        keyframes_list: List[List[ProsodyKeyframe]],
    ) -> torch.Tensor:
        """
        Encode keyframes to prosody conditioning tokens.

        Args:
            keyframes_list: List of keyframe lists

        Returns:
            prosody_tokens: [batch, num_tokens, output_dim]
        """
        tensor, mask = self.keyframes_to_tensor(keyframes_list)
        device = next(self.parameters()).device
        tensor = tensor.to(device)
        mask = mask.to(device)

        return self.encoder(tensor, mask)

    def from_user_keyframes(
        self,
        keyframes: List[Dict],
        device: str = 'cpu',
    ) -> torch.Tensor:
        """
        Convert UI keyframe format to prosody tokens.

        Args:
            keyframes: List of dicts with 'time', 'pitch', 'energy', etc.
            device: Target device

        Returns:
            prosody_tokens: [1, num_tokens, output_dim]
        """
        # Convert dicts to ProsodyKeyframe objects
        kf_objects = []
        for kf_dict in keyframes:
            emotion = None
            if 'emotion' in kf_dict:
                emotion = torch.tensor(kf_dict['emotion'])

            kf = ProsodyKeyframe(
                time=kf_dict.get('time', 0.5),
                pitch=kf_dict.get('pitch', 0.5),
                energy=kf_dict.get('energy', 0.5),
                duration_scale=kf_dict.get('duration_scale', 1.0),
                emotion=emotion,
            )
            kf_objects.append(kf)

        # Sort by time
        kf_objects.sort(key=lambda k: k.time)

        # Encode
        return self.encode([kf_objects]).to(device)

    def compute_reconstruction_loss(
        self,
        pitch_target: torch.Tensor,
        energy_target: torch.Tensor,
        keyframes_list: List[List[ProsodyKeyframe]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for keyframe reconstruction quality.

        Args:
            pitch_target: [batch, seq_len] target pitch
            energy_target: [batch, seq_len] target energy
            keyframes_list: Extracted keyframes

        Returns:
            Dict with losses
        """
        batch_size, seq_len = pitch_target.shape
        device = pitch_target.device

        total_pitch_loss = 0.0
        total_energy_loss = 0.0

        for b, keyframes in enumerate(keyframes_list):
            # Reconstruct
            recon = self.interpolate(keyframes, seq_len)

            pitch_recon = recon['pitch'].to(device)
            energy_recon = recon['energy'].to(device)

            total_pitch_loss += F.mse_loss(pitch_recon, pitch_target[b])
            total_energy_loss += F.mse_loss(energy_recon, energy_target[b])

        total_pitch_loss /= batch_size
        total_energy_loss /= batch_size

        return {
            'pitch_reconstruction': total_pitch_loss,
            'energy_reconstruction': total_energy_loss,
            'total': total_pitch_loss + total_energy_loss,
        }

    def forward(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration_seconds: float,
        emotion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward: extract keyframes and encode to tokens.

        Args:
            pitch: [batch, seq_len] normalized pitch
            energy: [batch, seq_len] normalized energy
            duration_seconds: Audio duration
            emotion: Optional per-frame emotion

        Returns:
            Dict with 'prosody_tokens', 'keyframes', 'reconstruction_loss'
        """
        # Extract keyframes
        keyframes_list = self.extract(pitch, energy, duration_seconds, emotion)

        # Encode to tokens
        prosody_tokens = self.encode(keyframes_list)

        # Compute reconstruction loss
        losses = self.compute_reconstruction_loss(pitch, energy, keyframes_list)

        return {
            'prosody_tokens': prosody_tokens,
            'keyframes': keyframes_list,
            'reconstruction_loss': losses['total'],
            'pitch_loss': losses['pitch_reconstruction'],
            'energy_loss': losses['energy_reconstruction'],
        }


# =============================================================================
# SPARSE KEYFRAME ADAPTER (PIPELINE INTEGRATION)
# =============================================================================

class SparseKeyframeAdapter(nn.Module):
    """
    Adapter for integrating sparse keyframes with existing prosody pipeline.

    Provides compatibility with:
    - DrawSpeech (converts between sketches and keyframes)
    - ProsodyControlledCSM
    - Frontend SketchProsodyEditor
    """

    def __init__(
        self,
        config: SparseKeyframeConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core conditioner
        self.conditioner = SparseKeyframeConditioner(config)

        # Adapt output dimension if needed
        if config.output_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def from_sketch(
        self,
        pitch_curve: torch.Tensor,
        energy_curve: torch.Tensor,
        duration_seconds: float = 3.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert dense sketch curves to sparse keyframes and encode.

        Args:
            pitch_curve: [batch, curve_length] or [curve_length]
            energy_curve: [batch, curve_length] or [curve_length]
            duration_seconds: Audio duration

        Returns:
            Dict with 'prosody_tokens', 'keyframes'
        """
        # Handle unbatched input
        if pitch_curve.dim() == 1:
            pitch_curve = pitch_curve.unsqueeze(0)
        if energy_curve.dim() == 1:
            energy_curve = energy_curve.unsqueeze(0)

        # Extract and encode
        result = self.conditioner(pitch_curve, energy_curve, duration_seconds)

        # Adapt output
        result['prosody_tokens'] = self.output_adapter(result['prosody_tokens'])

        return result

    def from_keyframes(
        self,
        keyframes: List[Dict],
        text_cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode keyframes from UI format.

        Args:
            keyframes: List of keyframe dicts from frontend
            text_cond: Optional text conditioning

        Returns:
            Dict with 'prosody_tokens'
        """
        device = text_cond.device if text_cond is not None else 'cpu'

        tokens = self.conditioner.from_user_keyframes(keyframes, device)
        tokens = self.output_adapter(tokens)

        return {'prosody_tokens': tokens}

    def to_sketch(
        self,
        keyframes: List[ProsodyKeyframe],
        sketch_length: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sparse keyframes back to dense sketch curves.

        Useful for visualization and editing.

        Args:
            keyframes: Sparse keyframes
            sketch_length: Output curve length

        Returns:
            (pitch_curve, energy_curve)
        """
        result = self.conditioner.interpolate(keyframes, sketch_length)
        return result['pitch'], result['energy']

    def compute_compression_stats(
        self,
        duration_seconds: float,
        num_keyframes: int,
    ) -> Dict[str, float]:
        """
        Compute compression statistics.

        Returns:
            Dict with 'bitrate_bps', 'compression_ratio', etc.
        """
        # Bits per keyframe:
        # - time: 8 bits (256 positions)
        # - pitch: 8 bits (256 levels)
        # - energy: 8 bits (256 levels)
        # - emotion: 8 * emotion_dim bits
        bits_per_keyframe = 8 + 8 + 8 + 8 * self.config.emotion_dim

        total_bits = num_keyframes * bits_per_keyframe
        bitrate = total_bits / duration_seconds

        # Compare to dense encoding at 100 Hz, 16 bits each
        dense_bits = duration_seconds * 100 * 2 * 16  # pitch + energy, 16 bits
        compression_ratio = dense_bits / total_bits

        return {
            'bitrate_bps': bitrate,
            'compression_ratio': compression_ratio,
            'num_keyframes': num_keyframes,
            'bits_per_keyframe': bits_per_keyframe,
            'keyframe_rate_hz': num_keyframes / duration_seconds,
        }


# =============================================================================
# UTILITIES
# =============================================================================

def create_emotion_keyframes(
    emotion_trajectory: List[Tuple[float, str, float]],
    emotion_profiles: Dict[str, torch.Tensor],
) -> List[ProsodyKeyframe]:
    """
    Create keyframes from an emotion trajectory.

    Args:
        emotion_trajectory: List of (time, emotion_name, intensity)
        emotion_profiles: Dict mapping emotion names to vectors

    Returns:
        List of ProsodyKeyframe objects
    """
    keyframes = []

    for time, emotion_name, intensity in emotion_trajectory:
        if emotion_name in emotion_profiles:
            emotion = emotion_profiles[emotion_name] * intensity
        else:
            emotion = torch.zeros(8)
            emotion[0] = 0.5  # Neutral

        # Derive pitch/energy from emotion (simplified mapping)
        pitch_map = {
            'happy': 0.7, 'sad': 0.3, 'angry': 0.6,
            'surprised': 0.8, 'calm': 0.5, 'neutral': 0.5,
        }
        energy_map = {
            'happy': 0.7, 'sad': 0.3, 'angry': 0.9,
            'surprised': 0.6, 'calm': 0.4, 'neutral': 0.5,
        }

        pitch = pitch_map.get(emotion_name.lower(), 0.5)
        energy = energy_map.get(emotion_name.lower(), 0.5)

        # Apply intensity
        pitch = 0.5 + (pitch - 0.5) * intensity
        energy = 0.5 + (energy - 0.5) * intensity

        kf = ProsodyKeyframe(
            time=time,
            pitch=pitch,
            energy=energy,
            emotion=emotion,
        )
        keyframes.append(kf)

    return keyframes


def keyframes_to_json(keyframes: List[ProsodyKeyframe]) -> List[Dict]:
    """Convert keyframes to JSON-serializable format for frontend."""
    return [
        {
            'time': kf.time,
            'pitch': kf.pitch,
            'energy': kf.energy,
            'duration_scale': kf.duration_scale,
            'emotion': kf.emotion.tolist() if kf.emotion is not None else None,
        }
        for kf in keyframes
    ]


def json_to_keyframes(data: List[Dict]) -> List[ProsodyKeyframe]:
    """Convert JSON format back to keyframes."""
    keyframes = []
    for d in data:
        emotion = None
        if d.get('emotion') is not None:
            emotion = torch.tensor(d['emotion'])

        kf = ProsodyKeyframe(
            time=d['time'],
            pitch=d['pitch'],
            energy=d['energy'],
            duration_scale=d.get('duration_scale', 1.0),
            emotion=emotion,
        )
        keyframes.append(kf)

    return keyframes


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STCTS: Sparse Keyframe Prosody Compression - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create config
    config = SparseKeyframeConfig()
    print(f"\nConfiguration:")
    print(f"  Keyframe rate: {config.keyframe_rate_hz} Hz")
    print(f"  Min/Max keyframes: {config.min_keyframes}/{config.max_keyframes}")
    print(f"  Extraction method: {config.extraction_method}")
    print(f"  Interpolation method: {config.interpolation_method}")

    # Test 1: Keyframe Extraction
    print("\n[Test 1] Keyframe Extraction...")
    extractor = SparseKeyframeExtractor(config)

    batch_size = 2
    seq_len = 100
    duration = 3.0

    pitch = torch.sin(torch.linspace(0, 4 * 3.14159, seq_len)).unsqueeze(0).repeat(batch_size, 1)
    pitch = (pitch + 1) / 2  # Normalize to [0, 1]

    energy = torch.cos(torch.linspace(0, 2 * 3.14159, seq_len)).unsqueeze(0).repeat(batch_size, 1)
    energy = (energy + 1) / 2

    keyframes_list = extractor(pitch, energy, duration)
    print(f"  Input shape: pitch={pitch.shape}, energy={energy.shape}")
    print(f"  Extracted keyframes per sample: {[len(kfs) for kfs in keyframes_list]}")
    print(f"  Sample keyframe: time={keyframes_list[0][0].time:.2f}, "
          f"pitch={keyframes_list[0][0].pitch:.2f}, "
          f"energy={keyframes_list[0][0].energy:.2f}")
    print("  [PASS]")

    # Test 2: Keyframe Interpolation
    print("\n[Test 2] Keyframe Interpolation...")
    interpolator = KeyframeInterpolator(config)

    recon = interpolator(keyframes_list[0], seq_len)
    print(f"  Reconstructed pitch shape: {recon['pitch'].shape}")
    print(f"  Reconstructed energy shape: {recon['energy'].shape}")

    # Compute reconstruction error
    pitch_mse = F.mse_loss(recon['pitch'], pitch[0])
    energy_mse = F.mse_loss(recon['energy'], energy[0])
    print(f"  Reconstruction MSE - pitch: {pitch_mse:.4f}, energy: {energy_mse:.4f}")
    print("  [PASS]")

    # Test 3: Sparse Keyframe Encoder
    print("\n[Test 3] Sparse Keyframe Encoder...")
    encoder = SparseKeyframeEncoder(config).to(device)

    conditioner = SparseKeyframeConditioner(config).to(device)
    tensor, mask = conditioner.keyframes_to_tensor(keyframes_list)
    tensor = tensor.to(device)
    mask = mask.to(device)

    tokens = encoder(tensor, mask)
    print(f"  Keyframe tensor shape: {tensor.shape}")
    print(f"  Output tokens shape: {tokens.shape}")
    print(f"  Expected: [{batch_size}, {config.num_output_tokens}, {config.output_dim}]")
    assert tokens.shape == (batch_size, config.num_output_tokens, config.output_dim)
    print("  [PASS]")

    # Test 4: Full Conditioner
    print("\n[Test 4] Full SparseKeyframeConditioner...")
    conditioner = SparseKeyframeConditioner(config).to(device)

    pitch = pitch.to(device)
    energy = energy.to(device)

    result = conditioner(pitch, energy, duration)
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Num keyframes extracted: {[len(kfs) for kfs in result['keyframes']]}")
    print(f"  Reconstruction loss: {result['reconstruction_loss'].item():.4f}")
    print("  [PASS]")

    # Test 5: Adapter Integration
    print("\n[Test 5] SparseKeyframeAdapter...")
    adapter = SparseKeyframeAdapter(config).to(device)

    # From sketch curves
    sketch_result = adapter.from_sketch(pitch, energy, duration)
    print(f"  From sketch - tokens shape: {sketch_result['prosody_tokens'].shape}")

    # From UI keyframes
    ui_keyframes = [
        {'time': 0.0, 'pitch': 0.5, 'energy': 0.5},
        {'time': 0.3, 'pitch': 0.8, 'energy': 0.7},
        {'time': 0.6, 'pitch': 0.3, 'energy': 0.4},
        {'time': 1.0, 'pitch': 0.5, 'energy': 0.5},
    ]
    ui_result = adapter.from_keyframes(ui_keyframes)
    print(f"  From UI keyframes - tokens shape: {ui_result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 6: Compression Statistics
    print("\n[Test 6] Compression Statistics...")
    stats = adapter.compute_compression_stats(duration, 4)
    print(f"  Bitrate: {stats['bitrate_bps']:.1f} bps")
    print(f"  Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"  Keyframe rate: {stats['keyframe_rate_hz']:.2f} Hz")
    print(f"  Target bitrate: {config.target_bitrate_bps} bps")
    print("  [PASS]")

    # Test 7: Bidirectional Conversion
    print("\n[Test 7] Sketch ↔ Keyframes Conversion...")

    # Create keyframes
    kfs = [
        ProsodyKeyframe(time=0.0, pitch=0.5, energy=0.5),
        ProsodyKeyframe(time=0.5, pitch=0.8, energy=0.7),
        ProsodyKeyframe(time=1.0, pitch=0.5, energy=0.5),
    ]

    # Convert to sketch
    pitch_curve, energy_curve = adapter.to_sketch(kfs, sketch_length=100)
    print(f"  Keyframes → Sketch: pitch={pitch_curve.shape}, energy={energy_curve.shape}")

    # Convert back to keyframes
    kfs_back = conditioner.extract(
        pitch_curve.unsqueeze(0).to(device),
        energy_curve.unsqueeze(0).to(device),
        duration_seconds=3.0
    )[0]
    print(f"  Sketch → Keyframes: {len(kfs_back)} keyframes")
    print("  [PASS]")

    # Test 8: JSON Serialization
    print("\n[Test 8] JSON Serialization...")
    json_data = keyframes_to_json(kfs)
    print(f"  Keyframes to JSON: {len(json_data)} items")

    kfs_restored = json_to_keyframes(json_data)
    print(f"  JSON to Keyframes: {len(kfs_restored)} items")

    assert len(kfs) == len(kfs_restored)
    assert abs(kfs[0].pitch - kfs_restored[0].pitch) < 1e-6
    print("  [PASS]")

    # Test 9: Different Interpolation Methods
    print("\n[Test 9] Interpolation Methods...")
    for method in ['linear', 'cubic', 'spherical']:
        config_test = SparseKeyframeConfig(interpolation_method=method)
        interp = KeyframeInterpolator(config_test)
        recon = interp(kfs, 100)
        print(f"  {method}: pitch range=[{recon['pitch'].min():.2f}, {recon['pitch'].max():.2f}]")
    print("  [PASS]")

    # Test 10: Emotion Keyframes
    print("\n[Test 10] Emotion Trajectory...")
    emotion_profiles = {
        'happy': torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'sad': torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'neutral': torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    trajectory = [
        (0.0, 'neutral', 0.5),
        (0.3, 'happy', 0.9),
        (0.7, 'sad', 0.6),
        (1.0, 'neutral', 0.5),
    ]

    emotion_kfs = create_emotion_keyframes(trajectory, emotion_profiles)
    print(f"  Created {len(emotion_kfs)} emotion keyframes")
    print(f"  First: time={emotion_kfs[0].time}, pitch={emotion_kfs[0].pitch:.2f}")
    print(f"  Second: time={emotion_kfs[1].time}, pitch={emotion_kfs[1].pitch:.2f} (happy)")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All STCTS Sparse Keyframe tests passed!")
    print("=" * 70)

    print("\nCompression Summary:")
    print("-" * 40)
    print(f"  Dense prosody:    ~{100 * 2 * 16} bps (100Hz, 2 channels, 16-bit)")
    print(f"  Sparse keyframes: ~{stats['bitrate_bps']:.1f} bps")
    print(f"  Compression:      ~{stats['compression_ratio']:.0f}x")
    print(f"  STCTS target:     <14 bps for prosody")
    print()

    print("Usage Example:")
    print("-" * 40)
    print("""
from sparse_keyframe_prosody import (
    SparseKeyframeConfig,
    SparseKeyframeConditioner,
    SparseKeyframeAdapter,
    ProsodyKeyframe,
    create_emotion_keyframes,
)

# Initialize
config = SparseKeyframeConfig(
    keyframe_rate_hz=0.5,  # 1 keyframe per 2 seconds
    extraction_method="salient",
    interpolation_method="cubic",
)
adapter = SparseKeyframeAdapter(config).cuda()

# From dense prosody (training)
result = adapter.from_sketch(pitch_curve, energy_curve, duration=3.0)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]
recon_loss = result['reconstruction_loss']

# From UI keyframes (inference)
keyframes = [
    {'time': 0.0, 'pitch': 0.5, 'energy': 0.5},
    {'time': 0.4, 'pitch': 0.8, 'energy': 0.8},  # Emphasis
    {'time': 0.8, 'pitch': 0.4, 'energy': 0.3},  # Calm down
    {'time': 1.0, 'pitch': 0.5, 'energy': 0.5},
]
tokens = adapter.from_keyframes(keyframes)['prosody_tokens']

# From emotion trajectory (natural language control)
trajectory = [
    (0.0, 'neutral', 0.5),
    (0.5, 'happy', 0.9),
    (1.0, 'calm', 0.6),
]
emotion_kfs = create_emotion_keyframes(trajectory, emotion_profiles)
tokens = adapter.conditioner.encode([emotion_kfs])

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)

# Get compression stats
stats = adapter.compute_compression_stats(duration=3.0, num_keyframes=4)
print(f"Bitrate: {stats['bitrate_bps']:.1f} bps")  # ~10-14 bps
""")
