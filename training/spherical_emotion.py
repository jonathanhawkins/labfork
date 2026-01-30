"""
Spherical Emotion Vectors (EmoSphere-TTS approach) for V6

Based on EmoSphere-TTS (2024): "EmoSphere-TTS: Emotional Style and Intensity Modeling via
Spherical Emotion Vector for Controllable Emotional Text-to-Speech"
https://arxiv.org/abs/2406.07803

Key Innovation: Represent emotions as vectors on a sphere using VAD (Valence-Arousal-Dominance):
- Origin = neutral state
- Direction = emotion type (happy, sad, angry, etc.)
- Magnitude = emotion intensity

Benefits for V6:
1. Continuous intensity control via single α parameter
2. Intuitive emotion blending/interpolation
3. More natural emotion transitions than discrete labels
4. Compatible with existing prosody conditioning pipeline

Implementation:
1. Map discrete emotions to VAD coordinates based on psychological research
2. Convert VAD (Cartesian) to spherical coordinates (r, θ, φ)
3. Use learnable emotion embeddings anchored to VAD prototypes
4. Enable smooth interpolation between any two emotions
5. Scale by α for intensity control (α=0 is neutral, α=1 is full intensity)

Reference VAD values from Russell's Circumplex Model and Mehrabian's PAD space.
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
class SphericalEmotionConfig:
    """Configuration for Spherical Emotion Vectors."""

    # VAD dimensions
    vad_dim: int = 3  # Valence, Arousal, Dominance

    # Spherical embedding settings
    embedding_dim: int = 256  # Dimension of learned emotion embeddings
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 2048  # Output to match prosody encoder

    # Number of emotion prototypes
    num_emotions: int = 8  # neutral, happy, sad, angry, surprised, calm, fearful, disgusted

    # Training settings
    dropout: float = 0.1
    use_learnable_prototypes: bool = True  # Learn refined VAD positions

    # Intensity settings
    default_intensity: float = 0.7  # Default α when not specified
    min_intensity: float = 0.0  # Neutral
    max_intensity: float = 2.0  # Allow exaggerated emotion (EASV supports >1.0)

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate

    # ECE-TTS EASV (Emotion-Adaptive Spherical Vector) settings
    # When enabled, uses EASV formula: neutral + intensity * (emotion - neutral)
    # This provides better intensity control with neutral-at-zero property
    use_easv: bool = True  # Use EASV formula instead of basic scaling
    easv_intensity_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 2.0))


# =============================================================================
# VAD EMOTION PROTOTYPES
# =============================================================================

# VAD (Valence-Arousal-Dominance) coordinates for emotions
# Based on Russell's Circumplex Model and empirical studies
# Valence: Positive (+) to Negative (-)
# Arousal: High (+) to Low (-)
# Dominance: Dominant (+) to Submissive (-)
# Normalized to [-1, 1] range

VAD_PROTOTYPES = {
    "neutral": (0.0, 0.0, 0.0),      # Origin - baseline state
    "happy": (0.8, 0.6, 0.6),         # Positive, moderately aroused, slightly dominant
    "sad": (-0.6, -0.4, -0.5),        # Negative, low arousal, submissive
    "angry": (-0.5, 0.8, 0.7),        # Negative, high arousal, dominant
    "surprised": (0.3, 0.8, 0.2),     # Slightly positive, high arousal, neutral dominance
    "calm": (0.4, -0.5, 0.3),         # Positive, low arousal, slightly dominant
    "fearful": (-0.7, 0.7, -0.7),     # Negative, high arousal, submissive
    "disgusted": (-0.6, 0.3, 0.4),    # Negative, moderate arousal, slightly dominant
    "excited": (0.7, 0.9, 0.5),       # Very positive, very high arousal, dominant
    "bored": (-0.2, -0.7, -0.2),      # Slightly negative, very low arousal, submissive
    "tender": (0.7, -0.2, -0.2),      # Positive, low arousal, slightly submissive
    "anxious": (-0.4, 0.6, -0.4),     # Negative, high arousal, submissive
}

EMOTION_TO_IDX = {e: i for i, e in enumerate(VAD_PROTOTYPES.keys())}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}

# Core emotions for the 8-emotion setup
CORE_EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "calm", "fearful", "disgusted"]


# =============================================================================
# COORDINATE CONVERSION UTILITIES
# =============================================================================

def cartesian_to_spherical(
    x: torch.Tensor,  # [..., 3] - VAD coordinates (V, A, D)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert Cartesian VAD coordinates to spherical coordinates.

    Args:
        x: Tensor of shape [..., 3] with (Valence, Arousal, Dominance)

    Returns:
        r: Magnitude/radius (emotion intensity)
        theta: Polar angle [0, π] (related to dominance axis)
        phi: Azimuthal angle [0, 2π] (related to valence-arousal plane)
    """
    v, a, d = x[..., 0], x[..., 1], x[..., 2]

    # Radius (magnitude) - emotion intensity
    r = torch.sqrt(v**2 + a**2 + d**2 + 1e-8)

    # Polar angle theta (from positive D-axis)
    theta = torch.acos(torch.clamp(d / (r + 1e-8), -1.0, 1.0))

    # Azimuthal angle phi (in V-A plane, from positive V-axis)
    phi = torch.atan2(a, v + 1e-8)

    return r, theta, phi


def spherical_to_cartesian(
    r: torch.Tensor,      # [...] - Magnitude
    theta: torch.Tensor,  # [...] - Polar angle
    phi: torch.Tensor,    # [...] - Azimuthal angle
) -> torch.Tensor:
    """
    Convert spherical coordinates to Cartesian VAD coordinates.

    Args:
        r: Magnitude/radius (emotion intensity)
        theta: Polar angle [0, π]
        phi: Azimuthal angle [0, 2π]

    Returns:
        Tensor of shape [..., 3] with (Valence, Arousal, Dominance)
    """
    v = r * torch.sin(theta) * torch.cos(phi)
    a = r * torch.sin(theta) * torch.sin(phi)
    d = r * torch.cos(theta)

    return torch.stack([v, a, d], dim=-1)


def normalize_vad(vad: torch.Tensor) -> torch.Tensor:
    """Normalize VAD vector to unit sphere (direction only)."""
    return F.normalize(vad, p=2, dim=-1)


def scale_vad(vad: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale VAD vector by intensity factor α."""
    return vad * alpha


# =============================================================================
# ECE-TTS EMOTION-ADAPTIVE SPHERICAL VECTORS (EASV)
# =============================================================================

# Based on ECE-TTS (2025): Emotion-Controllable Expressive TTS
# Key insight: Transform VAD to spherical vectors that enable precise emotion
# control via simple arithmetic operations, without additional modules.
#
# EASV Formula: emotion_scaled = neutral + intensity * (emotion - neutral)
#
# Benefits over basic scaling (emotion * intensity):
# - intensity=0: Always returns neutral (origin point)
# - intensity=1: Returns full emotion prototype
# - intensity>1: Exaggerates emotion beyond prototype
# - intensity<1: Weakens emotion toward neutral
# - Enables continuous, linear interpolation from neutral to exaggerated

def easv_scale(
    emotion_vad: torch.Tensor,
    intensity: torch.Tensor,
    neutral_vad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply Emotion-Adaptive Spherical Vector (EASV) scaling from ECE-TTS.

    This formula produces more natural intensity control than simple multiplication:
    emotion_scaled = neutral + intensity * (emotion - neutral)

    Args:
        emotion_vad: Emotion VAD coordinates [batch, 3] or [3]
        intensity: Intensity factor [batch] or scalar, range [0.0, 2.0]
        neutral_vad: Optional neutral VAD (default: origin [0,0,0])

    Returns:
        EASV-scaled VAD coordinates

    Example:
        >>> vad_happy = torch.tensor([0.8, 0.6, 0.6])
        >>> easv_scale(vad_happy, 0.0)  # Returns neutral [0, 0, 0]
        >>> easv_scale(vad_happy, 0.5)  # Weak happy [0.4, 0.3, 0.3]
        >>> easv_scale(vad_happy, 1.0)  # Full happy [0.8, 0.6, 0.6]
        >>> easv_scale(vad_happy, 1.5)  # Exaggerated [1.2, 0.9, 0.9]
    """
    device = emotion_vad.device

    # Default neutral is origin
    if neutral_vad is None:
        if emotion_vad.dim() == 1:
            neutral_vad = torch.zeros(3, device=device)
        else:
            neutral_vad = torch.zeros(1, 3, device=device)

    # Ensure intensity has proper shape
    if isinstance(intensity, (int, float)):
        intensity = torch.tensor(intensity, device=device)

    if intensity.dim() == 0:
        intensity = intensity.unsqueeze(0)

    # EASV formula: neutral + intensity * (emotion - neutral)
    if emotion_vad.dim() == 1:
        # Single emotion
        delta = emotion_vad - neutral_vad
        return neutral_vad + intensity[0] * delta
    else:
        # Batch of emotions
        if neutral_vad.dim() == 1:
            neutral_vad = neutral_vad.unsqueeze(0).expand_as(emotion_vad)
        delta = emotion_vad - neutral_vad
        return neutral_vad + intensity.unsqueeze(-1) * delta


def easv_interpolate(
    emotion1_vad: torch.Tensor,
    emotion2_vad: torch.Tensor,
    t: float,
    intensity: float = 1.0,
) -> torch.Tensor:
    """
    Interpolate between two emotions using EASV in spherical space.

    The interpolation happens in direction space (unit sphere), then
    intensity scaling is applied.

    Args:
        emotion1_vad: First emotion VAD
        emotion2_vad: Second emotion VAD
        t: Interpolation factor [0, 1] (0=emotion1, 1=emotion2)
        intensity: Intensity scaling for the interpolated result

    Returns:
        Interpolated and intensity-scaled VAD
    """
    # First interpolate direction (SLERP)
    v1_norm = F.normalize(emotion1_vad, dim=-1, eps=1e-8)
    v2_norm = F.normalize(emotion2_vad, dim=-1, eps=1e-8)

    dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)

    sin_omega = torch.sin(omega)
    nearly_parallel = sin_omega.abs() < 1e-6

    s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
    s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

    direction = s1 * v1_norm + s2 * v2_norm

    # Fall back to LERP for parallel vectors
    lerp_direction = (1 - t) * v1_norm + t * v2_norm
    direction = torch.where(nearly_parallel, lerp_direction, direction)
    direction = F.normalize(direction, dim=-1, eps=1e-8)

    # Interpolate magnitudes
    mag1 = torch.norm(emotion1_vad, dim=-1, keepdim=True)
    mag2 = torch.norm(emotion2_vad, dim=-1, keepdim=True)
    mag = (1 - t) * mag1 + t * mag2

    # Apply EASV intensity scaling from neutral
    result = direction * mag
    return easv_scale(result, torch.tensor(intensity))


def easv_blend(
    emotions_vad: torch.Tensor,  # [num_emotions, 3]
    weights: torch.Tensor,       # [num_emotions]
    intensity: float = 1.0,
) -> torch.Tensor:
    """
    Blend multiple emotions using EASV-aware weighted combination.

    Args:
        emotions_vad: VAD coordinates for each emotion [num_emotions, 3]
        weights: Blend weights (will be normalized) [num_emotions]
        intensity: Final intensity scaling

    Returns:
        Blended VAD with intensity scaling
    """
    # Normalize weights
    weights = weights / (weights.sum() + 1e-8)

    # Weighted combination
    blended = torch.sum(emotions_vad * weights.unsqueeze(-1), dim=0)

    # Apply EASV intensity
    return easv_scale(blended, torch.tensor(intensity))


def compute_easv_gradient(
    emotion_vad: torch.Tensor,
    neutral_vad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the EASV gradient (direction from neutral to emotion).

    This is useful for:
    - Understanding the "emotion direction" in VAD space
    - Arithmetic emotion manipulation
    - Visualizing emotion relationships

    Args:
        emotion_vad: Emotion VAD coordinates
        neutral_vad: Neutral VAD (default: origin)

    Returns:
        Gradient vector (emotion - neutral)
    """
    if neutral_vad is None:
        neutral_vad = torch.zeros_like(emotion_vad)
    return emotion_vad - neutral_vad


# =============================================================================
# SPHERICAL EMOTION ENCODER
# =============================================================================

class SphericalEmotionEncoder(nn.Module):
    """
    Encodes emotions using spherical VAD representation.

    The encoder learns to:
    1. Map VAD coordinates to high-dimensional embeddings
    2. Refine emotion prototype positions (if learnable)
    3. Generate intensity-controllable emotion vectors

    Architecture:
        VAD → Spherical Coords → Positional Encoding → MLP → Emotion Embedding
    """

    def __init__(self, config: SphericalEmotionConfig):
        super().__init__()
        self.config = config

        # Initialize VAD prototypes (can be learned or fixed)
        vad_values = torch.tensor([
            VAD_PROTOTYPES[e] for e in CORE_EMOTIONS
        ], dtype=torch.float32)

        if config.use_learnable_prototypes:
            # Learn refined positions while staying close to prototypes
            self.vad_prototypes = nn.Parameter(vad_values)
        else:
            self.register_buffer('vad_prototypes', vad_values)

        # Spherical coordinate encoder
        # Input: (r, theta, phi, sin(theta), cos(theta), sin(phi), cos(phi)) = 7
        spherical_input_dim = 7

        self.spherical_encoder = nn.Sequential(
            nn.Linear(spherical_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
        )

        # VAD direct encoder (parallel path)
        self.vad_encoder = nn.Sequential(
            nn.Linear(config.vad_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Output projection to prosody-compatible dimension
        self.output_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

        # Intensity predictor (learns to predict appropriate intensity)
        self.intensity_predictor = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def get_vad_for_emotion(
        self,
        emotion: Union[str, int, torch.Tensor],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Get VAD coordinates for an emotion.

        Args:
            emotion: Emotion name (str), index (int), or one-hot tensor
            device: Target device

        Returns:
            VAD tensor of shape [3] or [batch, 3]
        """
        if device is None:
            device = self.vad_prototypes.device

        if isinstance(emotion, str):
            emotion_lower = emotion.lower()
            # Check if it's a core emotion (has learnable prototype)
            if emotion_lower in CORE_EMOTIONS:
                idx = CORE_EMOTIONS.index(emotion_lower)
                return self.vad_prototypes[idx].to(device)
            # Check if it's an extended emotion (use static VAD values)
            elif emotion_lower in VAD_PROTOTYPES:
                vad = VAD_PROTOTYPES[emotion_lower]
                return torch.tensor(vad, dtype=torch.float32, device=device)
            else:
                # Unknown emotion - return neutral
                return self.vad_prototypes[0].to(device)
        elif isinstance(emotion, int):
            if emotion < self.config.num_emotions:
                return self.vad_prototypes[emotion].to(device)
            else:
                return self.vad_prototypes[0].to(device)
        elif isinstance(emotion, torch.Tensor):
            if emotion.dim() == 0:
                idx = emotion.long().item()
                if idx < self.config.num_emotions:
                    return self.vad_prototypes[idx].to(device)
                return self.vad_prototypes[0].to(device)
            elif emotion.dim() == 1 and emotion.shape[0] == self.config.num_emotions:
                # One-hot or soft weights: weighted sum of prototypes
                return torch.matmul(emotion.unsqueeze(0), self.vad_prototypes).squeeze(0)
            elif emotion.dim() == 2:
                # Batch of one-hot/soft weights
                return torch.matmul(emotion, self.vad_prototypes)
            else:
                # Assume already VAD coordinates
                return emotion.to(device)
        else:
            raise ValueError(f"Unsupported emotion type: {type(emotion)}")

    def _encode_spherical(self, vad: torch.Tensor) -> torch.Tensor:
        """
        Encode VAD coordinates via spherical representation.

        Args:
            vad: Tensor of shape [batch, 3]

        Returns:
            Spherical features of shape [batch, embedding_dim]
        """
        r, theta, phi = cartesian_to_spherical(vad)

        # Create rich spherical features
        spherical_features = torch.stack([
            r,
            theta,
            phi,
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(phi),
            torch.cos(phi),
        ], dim=-1)

        return self.spherical_encoder(spherical_features)

    def forward(
        self,
        vad: torch.Tensor,           # [batch, 3] VAD coordinates
        intensity: Optional[torch.Tensor] = None,  # [batch] or scalar, α parameter
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion from VAD coordinates.

        Args:
            vad: VAD coordinates [batch, 3]
            intensity: Optional intensity scaling factor α. If None, uses default.

        Returns:
            Dict with:
                - 'embedding': [batch, embedding_dim] emotion embedding
                - 'tokens': [batch, num_tokens, output_dim] prosody prefix tokens
                - 'intensity': [batch] predicted/applied intensity
                - 'spherical': (r, theta, phi) tuple
        """
        batch_size = vad.shape[0]
        device = vad.device

        # Apply intensity scaling
        if intensity is None:
            intensity = torch.full((batch_size,), self.config.default_intensity, device=device)
        elif isinstance(intensity, (int, float)):
            intensity = torch.full((batch_size,), float(intensity), device=device)

        # Scale VAD by intensity using EASV or basic scaling
        if self.config.use_easv:
            # ECE-TTS EASV formula: neutral + intensity * (emotion - neutral)
            # This gives proper neutral-at-zero behavior
            vad_scaled = easv_scale(vad, intensity)
        else:
            # Basic scaling: just multiply by intensity
            vad_scaled = vad * intensity.unsqueeze(-1)

        # Encode via both paths
        spherical_features = self._encode_spherical(vad_scaled)
        vad_features = self.vad_encoder(vad_scaled)

        # Fuse representations
        combined = torch.cat([spherical_features, vad_features], dim=-1)
        embedding = self.fusion(combined)

        # Predict intensity from embedding (for analysis/loss)
        predicted_intensity = self.intensity_predictor(embedding).squeeze(-1)

        # Project to prosody tokens
        tokens = self.output_projection(embedding)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        # Get spherical coordinates for analysis
        r, theta, phi = cartesian_to_spherical(vad_scaled)

        return {
            'embedding': embedding,
            'tokens': tokens,
            'intensity': intensity,
            'predicted_intensity': predicted_intensity,
            'spherical': (r, theta, phi),
            'vad_scaled': vad_scaled,
        }

    def encode_emotion(
        self,
        emotion: Union[str, int, torch.Tensor],
        intensity: float = None,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method to encode an emotion by name/index.

        Args:
            emotion: Emotion name, index, or tensor
            intensity: Intensity factor α (0 = neutral, 1 = full)
            batch_size: Batch size if emotion is scalar

        Returns:
            Same as forward()
        """
        vad = self.get_vad_for_emotion(emotion)
        if vad.dim() == 1:
            vad = vad.unsqueeze(0).expand(batch_size, -1)

        if intensity is None:
            intensity = self.config.default_intensity

        return self.forward(vad, intensity)


# =============================================================================
# EMOTION INTERPOLATION
# =============================================================================

class EmotionInterpolator:
    """
    Interpolates between emotions in VAD space.

    Supports:
    1. Linear interpolation (LERP)
    2. Spherical linear interpolation (SLERP)
    3. Multi-emotion blending
    4. Intensity-aware transitions
    """

    @staticmethod
    def lerp(
        vad1: torch.Tensor,
        vad2: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Linear interpolation between two VAD vectors.

        Args:
            vad1: Source VAD [batch, 3] or [3]
            vad2: Target VAD [batch, 3] or [3]
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated VAD
        """
        return vad1 * (1 - t) + vad2 * t

    @staticmethod
    def slerp(
        vad1: torch.Tensor,
        vad2: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Spherical linear interpolation (great arc path on emotion sphere).

        This produces more natural emotion transitions than linear interpolation,
        especially for emotions that are far apart in VAD space.

        Args:
            vad1: Source VAD [batch, 3] or [3]
            vad2: Target VAD [batch, 3] or [3]
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated VAD on the sphere
        """
        # Normalize to unit sphere (direction only)
        v1_norm = F.normalize(vad1, dim=-1, eps=1e-8)
        v2_norm = F.normalize(vad2, dim=-1, eps=1e-8)

        # Compute angle between vectors
        dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        omega = torch.acos(dot)

        # Handle nearly parallel vectors
        sin_omega = torch.sin(omega)
        nearly_parallel = sin_omega.abs() < 1e-6

        # SLERP formula
        s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
        s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

        result = s1 * v1_norm + s2 * v2_norm

        # Fall back to LERP for nearly parallel vectors
        lerp_result = EmotionInterpolator.lerp(v1_norm, v2_norm, t)
        result = torch.where(nearly_parallel, lerp_result, result)

        # Interpolate magnitudes linearly
        mag1 = torch.norm(vad1, dim=-1, keepdim=True)
        mag2 = torch.norm(vad2, dim=-1, keepdim=True)
        mag = mag1 * (1 - t) + mag2 * t

        return result * mag

    @staticmethod
    def blend_emotions(
        emotions: List[Tuple[str, float]],
        encoder: SphericalEmotionEncoder,
    ) -> torch.Tensor:
        """
        Blend multiple emotions with weights.

        Args:
            emotions: List of (emotion_name, weight) tuples
            encoder: SphericalEmotionEncoder for VAD lookup

        Returns:
            Blended VAD coordinates
        """
        # Normalize weights
        total_weight = sum(w for _, w in emotions)
        if total_weight == 0:
            return encoder.get_vad_for_emotion("neutral")

        # Weighted sum of VAD vectors
        blended = torch.zeros(3, device=encoder.vad_prototypes.device)
        for emotion, weight in emotions:
            vad = encoder.get_vad_for_emotion(emotion)
            blended += vad * (weight / total_weight)

        return blended

    @staticmethod
    def create_trajectory(
        emotions: List[Tuple[str, float]],
        encoder: SphericalEmotionEncoder,
        num_steps: int = 10,
        use_slerp: bool = True,
    ) -> torch.Tensor:
        """
        Create emotion trajectory through multiple waypoints.

        Args:
            emotions: List of (emotion_name, duration_ratio) tuples
            encoder: SphericalEmotionEncoder for VAD lookup
            num_steps: Total number of trajectory steps
            use_slerp: Use spherical interpolation

        Returns:
            Trajectory tensor [num_steps, 3]
        """
        if len(emotions) < 2:
            vad = encoder.get_vad_for_emotion(emotions[0][0] if emotions else "neutral")
            return vad.unsqueeze(0).expand(num_steps, -1)

        # Calculate step allocation per segment
        total_duration = sum(d for _, d in emotions[:-1])  # Last emotion is endpoint
        steps_per_segment = []
        remaining_steps = num_steps

        for i, (_, duration) in enumerate(emotions[:-1]):
            if i == len(emotions) - 2:
                segment_steps = remaining_steps
            else:
                segment_steps = max(1, int(num_steps * duration / total_duration))
            steps_per_segment.append(segment_steps)
            remaining_steps -= segment_steps

        # Build trajectory
        trajectory = []
        for i, ((emotion1, _), segment_steps) in enumerate(zip(emotions[:-1], steps_per_segment)):
            emotion2 = emotions[i + 1][0]
            vad1 = encoder.get_vad_for_emotion(emotion1)
            vad2 = encoder.get_vad_for_emotion(emotion2)

            for s in range(segment_steps):
                t = s / max(1, segment_steps - 1) if segment_steps > 1 else 0
                if use_slerp:
                    interp = EmotionInterpolator.slerp(vad1.unsqueeze(0), vad2.unsqueeze(0), t)
                else:
                    interp = EmotionInterpolator.lerp(vad1, vad2, t)
                trajectory.append(interp.squeeze(0) if interp.dim() > 1 else interp)

        return torch.stack(trajectory)


# =============================================================================
# EMOTION CLASSIFICATION FROM VAD
# =============================================================================

class VADEmotionClassifier(nn.Module):
    """
    Classifies discrete emotion from VAD coordinates.

    Used for:
    1. Validating VAD predictions
    2. Converting continuous VAD to discrete labels
    3. Training with emotion classification loss
    """

    def __init__(self, config: SphericalEmotionConfig):
        super().__init__()
        self.config = config

        # Learnable prototype matching
        self.classifier = nn.Sequential(
            nn.Linear(config.vad_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_emotions),
        )

        # Register prototypes for distance-based classification
        vad_values = torch.tensor([
            VAD_PROTOTYPES[e] for e in CORE_EMOTIONS
        ], dtype=torch.float32)
        self.register_buffer('prototypes', vad_values)

    def forward(self, vad: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify emotion from VAD.

        Args:
            vad: VAD coordinates [batch, 3]

        Returns:
            Dict with logits, probabilities, and nearest prototype
        """
        # Neural classifier
        logits = self.classifier(vad)
        probs = F.softmax(logits, dim=-1)

        # Distance-based classification
        vad_expanded = vad.unsqueeze(1)  # [batch, 1, 3]
        proto_expanded = self.prototypes.unsqueeze(0)  # [1, num_emotions, 3]
        distances = torch.norm(vad_expanded - proto_expanded, dim=-1)  # [batch, num_emotions]
        distance_probs = F.softmax(-distances, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'distance_probs': distance_probs,
            'distances': distances,
            'predicted': torch.argmax(probs, dim=-1),
        }


# =============================================================================
# VAD EXTRACTION FROM AUDIO (PLACEHOLDER)
# =============================================================================

class VADExtractor(nn.Module):
    """
    Extracts VAD (Valence-Arousal-Dominance) coordinates from audio.

    This is a placeholder for integration with pretrained emotion recognizers.
    In practice, you would use:
    - wav2vec2-large-emotion
    - HuBERT emotion models
    - SER (Speech Emotion Recognition) models

    The extracted discrete emotions are converted to VAD coordinates using
    the VAD_PROTOTYPES mapping.
    """

    def __init__(self, config: SphericalEmotionConfig):
        super().__init__()
        self.config = config

        # Register prototype lookup
        vad_values = torch.tensor([
            VAD_PROTOTYPES[e] for e in CORE_EMOTIONS
        ], dtype=torch.float32)
        self.register_buffer('vad_lookup', vad_values)

        # Placeholder projection from audio features
        # In practice, replace with actual emotion recognizer
        self.emotion_head = nn.Linear(768, config.num_emotions)  # 768 = typical wav2vec2 dim

        # Direct VAD regression head (alternative to classification)
        self.vad_head = nn.Sequential(
            nn.Linear(768, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 3),
            nn.Tanh(),  # Bound to [-1, 1]
        )

    def from_emotion_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert emotion classification logits to VAD coordinates.

        Args:
            logits: [batch, num_emotions] emotion logits

        Returns:
            VAD coordinates [batch, 3] as weighted sum of prototypes
        """
        probs = F.softmax(logits, dim=-1)
        return torch.matmul(probs, self.vad_lookup)

    def forward(
        self,
        audio_features: torch.Tensor,  # [batch, seq_len, 768] from wav2vec2/HuBERT
    ) -> Dict[str, torch.Tensor]:
        """
        Extract VAD from audio features.

        Args:
            audio_features: Audio encoder features [batch, seq_len, 768]

        Returns:
            Dict with VAD coordinates and emotion predictions
        """
        # Pool over time dimension
        pooled = audio_features.mean(dim=1)  # [batch, 768]

        # Classification path
        emotion_logits = self.emotion_head(pooled)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        vad_from_class = torch.matmul(emotion_probs, self.vad_lookup)

        # Direct regression path
        vad_direct = self.vad_head(pooled)

        # Combine (can weight based on confidence)
        vad_combined = 0.7 * vad_from_class + 0.3 * vad_direct

        return {
            'vad': vad_combined,
            'vad_from_classification': vad_from_class,
            'vad_direct': vad_direct,
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
        }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SphericalEmotionLoss(nn.Module):
    """
    Loss functions for training spherical emotion encoder.

    Components:
    1. VAD reconstruction loss (for autoencoder training)
    2. Emotion classification loss (for discrete labels)
    3. Intensity prediction loss
    4. Prototype separation loss (encourage distinct emotions)
    5. Spherical regularization (keep on sphere surface)
    """

    def __init__(
        self,
        config: SphericalEmotionConfig,
        vad_weight: float = 1.0,
        classification_weight: float = 0.5,
        intensity_weight: float = 0.3,
        separation_weight: float = 0.1,
        spherical_weight: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.vad_weight = vad_weight
        self.classification_weight = classification_weight
        self.intensity_weight = intensity_weight
        self.separation_weight = separation_weight
        self.spherical_weight = spherical_weight

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        encoder_output: Dict[str, torch.Tensor],
        target_vad: Optional[torch.Tensor] = None,
        target_emotion: Optional[torch.Tensor] = None,
        target_intensity: Optional[torch.Tensor] = None,
        classifier_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for spherical emotion training.

        Args:
            encoder_output: Output from SphericalEmotionEncoder
            target_vad: Ground truth VAD coordinates [batch, 3]
            target_emotion: Ground truth emotion indices [batch]
            target_intensity: Ground truth intensity [batch]
            classifier_output: Output from VADEmotionClassifier

        Returns:
            Dict with individual losses and total
        """
        losses = {}
        device = encoder_output['embedding'].device

        # VAD reconstruction loss
        if target_vad is not None:
            losses['vad'] = self.mse_loss(encoder_output['vad_scaled'], target_vad)
        else:
            losses['vad'] = torch.tensor(0.0, device=device)

        # Emotion classification loss
        if target_emotion is not None and classifier_output is not None:
            losses['classification'] = self.ce_loss(
                classifier_output['logits'],
                target_emotion
            )
        else:
            losses['classification'] = torch.tensor(0.0, device=device)

        # Intensity prediction loss
        if target_intensity is not None:
            losses['intensity'] = self.mse_loss(
                encoder_output['predicted_intensity'],
                target_intensity
            )
        else:
            losses['intensity'] = torch.tensor(0.0, device=device)

        # Prototype separation loss (encourage distinct emotion clusters)
        # Only applies if encoder has learnable prototypes
        if hasattr(self, 'config') and self.config.use_learnable_prototypes:
            # Compute pairwise distances between prototypes
            # Note: Access encoder through the training loop
            losses['separation'] = torch.tensor(0.0, device=device)
        else:
            losses['separation'] = torch.tensor(0.0, device=device)

        # Spherical regularization (encourage unit magnitude for direction)
        r = encoder_output['spherical'][0]
        losses['spherical'] = self.mse_loss(r, torch.ones_like(r))

        # Total loss
        total = (
            losses['vad'] * self.vad_weight +
            losses['classification'] * self.classification_weight +
            losses['intensity'] * self.intensity_weight +
            losses['separation'] * self.separation_weight +
            losses['spherical'] * self.spherical_weight
        )
        losses['total'] = total

        return losses


# =============================================================================
# INTEGRATION WITH PROSODY CONDITIONING
# =============================================================================

class SphericalEmotionAdapter(nn.Module):
    """
    Adapts spherical emotion embeddings for the prosody conditioning pipeline.

    This module bridges SphericalEmotionEncoder with ProsodyControlledCSM,
    enabling emotion control via VAD/spherical coordinates.

    Usage with existing prosody system:
        adapter = SphericalEmotionAdapter(config)

        # Get emotion tokens
        vad = adapter.get_vad_for_emotion("happy")
        output = adapter.forward(vad, intensity=0.8)

        # Use as prosody prefix
        emotion_prefix = output['tokens']  # [batch, num_tokens, hidden]
        # Combine with other prosody conditioning...
    """

    def __init__(
        self,
        emotion_config: SphericalEmotionConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.emotion_config = emotion_config

        # Core emotion encoder
        self.encoder = SphericalEmotionEncoder(emotion_config)

        # Classifier for analysis
        self.classifier = VADEmotionClassifier(emotion_config)

        # Adapter to prosody hidden dimension (if different)
        if emotion_config.output_dim != prosody_hidden:
            self.prosody_adapter = nn.Sequential(
                nn.Linear(emotion_config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.prosody_adapter = nn.Identity()

    def forward(
        self,
        vad: torch.Tensor,
        intensity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion and adapt for prosody conditioning.
        """
        encoder_output = self.encoder(vad, intensity)

        # Adapt tokens to prosody dimension
        tokens = encoder_output['tokens']
        adapted_tokens = self.prosody_adapter(tokens)

        # Get classification for analysis
        classifier_output = self.classifier(encoder_output['vad_scaled'])

        return {
            **encoder_output,
            'prosody_tokens': adapted_tokens,
            'classification': classifier_output,
        }

    def encode_emotion(
        self,
        emotion: Union[str, int, torch.Tensor],
        intensity: float = None,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Encode emotion by name/index."""
        vad = self.encoder.get_vad_for_emotion(emotion)
        if vad.dim() == 1:
            vad = vad.unsqueeze(0).expand(batch_size, -1)
        return self.forward(vad, intensity)

    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = None,
        use_slerp: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotions.

        Args:
            emotion1: Source emotion name
            emotion2: Target emotion name
            t: Interpolation factor [0, 1]
            intensity: Intensity scaling
            use_slerp: Use spherical interpolation

        Returns:
            Encoded interpolated emotion
        """
        vad1 = self.encoder.get_vad_for_emotion(emotion1).unsqueeze(0)
        vad2 = self.encoder.get_vad_for_emotion(emotion2).unsqueeze(0)

        if use_slerp:
            vad_interp = EmotionInterpolator.slerp(vad1, vad2, t)
        else:
            vad_interp = EmotionInterpolator.lerp(vad1, vad2, t)

        return self.forward(vad_interp, intensity)

    def blend_emotions(
        self,
        emotions: List[Tuple[str, float]],
        intensity: float = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Blend multiple emotions with weights.

        Args:
            emotions: List of (emotion_name, weight) tuples
            intensity: Overall intensity

        Returns:
            Encoded blended emotion
        """
        vad = EmotionInterpolator.blend_emotions(emotions, self.encoder)
        return self.forward(vad.unsqueeze(0), intensity)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_emotion_vad(emotion: str) -> Tuple[float, float, float]:
    """Get VAD coordinates for an emotion name."""
    return VAD_PROTOTYPES.get(emotion.lower(), VAD_PROTOTYPES["neutral"])


def vad_to_emotion_name(vad: Union[torch.Tensor, Tuple[float, float, float]]) -> str:
    """Find nearest emotion name for VAD coordinates."""
    if isinstance(vad, torch.Tensor):
        vad = (vad[0].item(), vad[1].item(), vad[2].item())

    min_dist = float('inf')
    nearest = "neutral"

    for emotion, proto in VAD_PROTOTYPES.items():
        dist = sum((a - b) ** 2 for a, b in zip(vad, proto)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = emotion

    return nearest


def create_emotion_trajectory_from_script(
    script: List[Dict[str, Union[str, float]]],
    encoder: SphericalEmotionEncoder = None,
) -> torch.Tensor:
    """
    Create emotion trajectory from a script format.

    Args:
        script: List of dicts with 'emotion', 'time', 'intensity' keys
        encoder: Optional encoder for VAD lookup

    Returns:
        Trajectory tensor [num_segments, 3] of VAD coordinates
    """
    if not script:
        return torch.zeros(1, 3)

    # Sort by time
    sorted_script = sorted(script, key=lambda x: x.get('time', 0))

    # Build VAD sequence
    vad_sequence = []
    for item in sorted_script:
        emotion = item.get('emotion', 'neutral')
        intensity = item.get('intensity', 0.7)

        vad = torch.tensor(VAD_PROTOTYPES.get(emotion, VAD_PROTOTYPES["neutral"]))
        vad = vad * intensity
        vad_sequence.append(vad)

    return torch.stack(vad_sequence)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Spherical Emotion Vectors (EmoSphere-TTS) - Test Suite")
    print("=" * 70)

    config = SphericalEmotionConfig()

    # Test 1: VAD Prototypes
    print("\n[Test 1] VAD Prototypes...")
    for emotion, vad in list(VAD_PROTOTYPES.items())[:5]:
        print(f"  {emotion:12s}: V={vad[0]:+.2f}, A={vad[1]:+.2f}, D={vad[2]:+.2f}")
    print("  [PASS]")

    # Test 2: Coordinate Conversion
    print("\n[Test 2] Cartesian ↔ Spherical Conversion...")
    test_vad = torch.tensor([[0.5, 0.6, 0.3], [-0.4, 0.7, -0.5]])
    r, theta, phi = cartesian_to_spherical(test_vad)
    recovered = spherical_to_cartesian(r, theta, phi)
    error = (test_vad - recovered).abs().max().item()
    print(f"  Original VAD: {test_vad[0].tolist()}")
    print(f"  Spherical: r={r[0]:.3f}, θ={theta[0]:.3f}, φ={phi[0]:.3f}")
    print(f"  Recovered VAD: {recovered[0].tolist()}")
    print(f"  Max reconstruction error: {error:.6f}")
    assert error < 1e-5, "Coordinate conversion error too high!"
    print("  [PASS]")

    # Test 3: SphericalEmotionEncoder
    print("\n[Test 3] SphericalEmotionEncoder...")
    encoder = SphericalEmotionEncoder(config)

    batch_size = 2
    vad_input = torch.randn(batch_size, 3)
    intensity = torch.tensor([0.5, 0.9])

    output = encoder(vad_input, intensity)
    print(f"  Input VAD shape: {vad_input.shape}")
    print(f"  Embedding shape: {output['embedding'].shape}")
    print(f"  Tokens shape: {output['tokens'].shape}")
    print(f"  Predicted intensity: {output['predicted_intensity'].tolist()}")
    print("  [PASS]")

    # Test 4: Encode by emotion name
    print("\n[Test 4] Encode by emotion name...")
    for emotion in ["happy", "sad", "angry"]:
        result = encoder.encode_emotion(emotion, intensity=0.8)
        r_val = result['spherical'][0][0].item()
        print(f"  {emotion:8s}: embedding norm={result['embedding'].norm():.3f}, r={r_val:.3f}")
    print("  [PASS]")

    # Test 5: Emotion Interpolation
    print("\n[Test 5] Emotion Interpolation...")
    vad1 = encoder.get_vad_for_emotion("happy").unsqueeze(0)
    vad2 = encoder.get_vad_for_emotion("sad").unsqueeze(0)

    print(f"  Happy VAD: {vad1.squeeze().tolist()}")
    print(f"  Sad VAD:   {vad2.squeeze().tolist()}")

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lerp_result = EmotionInterpolator.lerp(vad1, vad2, t)
        slerp_result = EmotionInterpolator.slerp(vad1, vad2, t)
        print(f"  t={t:.2f}: LERP={lerp_result.squeeze().tolist()}, SLERP={slerp_result.squeeze().tolist()}")
    print("  [PASS]")

    # Test 6: Multi-emotion Blending
    print("\n[Test 6] Multi-emotion Blending...")
    blend = EmotionInterpolator.blend_emotions(
        [("happy", 0.6), ("surprised", 0.3), ("calm", 0.1)],
        encoder
    )
    nearest = vad_to_emotion_name(blend)
    print(f"  Blended VAD: {blend.tolist()}")
    print(f"  Nearest emotion: {nearest}")
    print("  [PASS]")

    # Test 7: VAD Classifier
    print("\n[Test 7] VAD Emotion Classifier...")
    classifier = VADEmotionClassifier(config)
    vad_test = encoder.get_vad_for_emotion("angry").unsqueeze(0)
    class_output = classifier(vad_test)
    pred_idx = class_output['predicted'].item()
    pred_emotion = CORE_EMOTIONS[pred_idx]
    print(f"  Input VAD (angry): {vad_test.squeeze().tolist()}")
    print(f"  Predicted emotion: {pred_emotion} (idx={pred_idx})")
    print(f"  Probabilities: {dict(zip(CORE_EMOTIONS, class_output['probs'].squeeze().tolist()))}")
    print("  [PASS]")

    # Test 8: SphericalEmotionAdapter
    print("\n[Test 8] SphericalEmotionAdapter (prosody integration)...")
    adapter = SphericalEmotionAdapter(config)

    # Test single emotion
    result = adapter.encode_emotion("surprised", intensity=0.7)
    print(f"  Emotion: surprised")
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")

    # Test interpolation
    interp_result = adapter.interpolate_emotions("calm", "angry", t=0.5, intensity=0.8)
    print(f"  Interpolated (calm→angry, t=0.5) tokens shape: {interp_result['prosody_tokens'].shape}")

    # Test blending
    blend_result = adapter.blend_emotions([("happy", 0.5), ("surprised", 0.5)], intensity=0.9)
    print(f"  Blended (happy+surprised) tokens shape: {blend_result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 9: Emotion Trajectory
    print("\n[Test 9] Emotion Trajectory...")
    trajectory = EmotionInterpolator.create_trajectory(
        emotions=[("neutral", 1.0), ("happy", 1.0), ("surprised", 1.0)],
        encoder=encoder,
        num_steps=10,
        use_slerp=True
    )
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  First point: {trajectory[0].tolist()}")
    print(f"  Mid point: {trajectory[5].tolist()}")
    print(f"  Last point: {trajectory[-1].tolist()}")
    print("  [PASS]")

    # Test 10: Loss Function
    print("\n[Test 10] Loss Function...")
    loss_fn = SphericalEmotionLoss(config)

    target_vad = torch.randn(batch_size, 3)
    target_emotion = torch.randint(0, config.num_emotions, (batch_size,))
    target_intensity = torch.rand(batch_size)

    encoder_output = encoder(vad_input, intensity)
    classifier_output = classifier(encoder_output['vad_scaled'])

    losses = loss_fn(
        encoder_output,
        target_vad=target_vad,
        target_emotion=target_emotion,
        target_intensity=target_intensity,
        classifier_output=classifier_output,
    )

    print(f"  VAD loss: {losses['vad'].item():.4f}")
    print(f"  Classification loss: {losses['classification'].item():.4f}")
    print(f"  Intensity loss: {losses['intensity'].item():.4f}")
    print(f"  Spherical loss: {losses['spherical'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All Spherical Emotion tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from spherical_emotion import (
    SphericalEmotionConfig, SphericalEmotionAdapter, EmotionInterpolator
)

# Initialize
config = SphericalEmotionConfig()
adapter = SphericalEmotionAdapter(config)

# Single emotion control
result = adapter.encode_emotion("happy", intensity=0.8)
prosody_tokens = result['prosody_tokens']  # [1, 4, 2048]

# Emotion interpolation (blend happy → sad)
result = adapter.interpolate_emotions("happy", "sad", t=0.5, intensity=0.7)

# Multi-emotion blending
result = adapter.blend_emotions([
    ("happy", 0.5),
    ("excited", 0.3),
    ("surprised", 0.2),
], intensity=0.9)

# Create emotion trajectory for temporal control
trajectory = EmotionInterpolator.create_trajectory(
    emotions=[("neutral", 0.3), ("happy", 0.5), ("surprised", 0.2)],
    encoder=adapter.encoder,
    num_steps=10,
)

# Use with ProsodyControlledCSM:
# combined_prefix = torch.cat([prosody_prefix, emotion_tokens], dim=1)
""")
