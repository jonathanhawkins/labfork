"""
UDDETTS: Unified Dimensional Emotion Control via ADV Space

Based on UDDETTS (arXiv:2505.10599):
"UDDETTS: Unified Dimensional Emotion Text-to-Speech with ADV Control"

Key Innovations:
1. ADV Space (Arousal-Dominance-Valence): Interpretable 3D emotion space
   - Arousal: Activation level (calm ↔ excited)
   - Dominance: Control level (submissive ↔ dominant)
   - Valence: Pleasure level (negative ↔ positive)

2. Nonlinear ADV Quantization: Fine-grained control along each dimension
   - Quantize continuous ADV into discrete levels
   - Nonlinear mapping for better control at certain intensity ranges
   - Enables discrete label ↔ continuous ADV bidirectional mapping

3. Semi-Supervised Training: Unified training on mixed annotation types
   - Datasets with discrete emotion labels
   - Datasets with continuous ADV annotations
   - Mixed annotation loss for unified learning

4. OT-CFM Decoder: Optimal Transport Conditional Flow Matching
   - Flow-matching generation conditioned on ADV
   - OT coupling for straighter flows and faster inference

5. Integration with Existing VAD: Seamless mapping between representations
   - ADV and VAD are equivalent (just reordered dimensions)
   - Compatible with existing SphericalEmotionAdapter

Benefits:
- Linear emotion control along 3 interpretable dimensions
- Mixing discrete labels with continuous ADV values
- First LLM-based TTS with dimensional ADV control
- Semi-supervised training leverages diverse emotional datasets

Reference VAD/ADV values from Russell's Circumplex Model and Mehrabian's PAD space.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UDDETTSConfig:
    """Configuration for UDDETTS ADV-space unified emotion control."""

    # ADV dimensions
    adv_dim: int = 3  # Arousal, Dominance, Valence

    # Model dimensions
    input_dim: int = 768  # Input feature dimension (wav2vec2/HuBERT)
    emotion_dim: int = 256  # Emotion embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 2048  # Output to match prosody encoder (CSM)

    # Nonlinear ADV quantization
    num_quantization_levels: int = 16  # Levels per ADV dimension
    use_nonlinear_quantization: bool = True  # Apply nonlinear mapping
    quantization_temperature: float = 0.1  # Softmax temperature for soft quantization

    # Discrete emotion mapping
    num_discrete_emotions: int = 8  # Number of discrete emotion categories
    discrete_emotions: Tuple[str, ...] = (
        "neutral", "happy", "sad", "angry",
        "surprised", "calm", "fearful", "disgusted"
    )

    # Semi-supervised training
    use_semi_supervised: bool = True
    discrete_loss_weight: float = 1.0
    adv_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.5  # ADV ↔ discrete consistency
    kl_loss_weight: float = 0.1  # KL between predicted and prior

    # OT-CFM settings
    use_ot_cfm: bool = True
    sigma_min: float = 0.001  # Minimum std at t=1
    ot_reg: float = 0.05  # Sinkhorn regularization
    num_flow_steps: int = 50  # ODE integration steps at inference
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_num_heads: int = 8

    # Training settings
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate


# =============================================================================
# ADV EMOTION PROTOTYPES
# =============================================================================

# ADV (Arousal-Dominance-Valence) coordinates for emotions
# Arousal: Activation level (-1 to +1)
# Dominance: Control level (-1 to +1)
# Valence: Pleasure level (-1 to +1)
#
# Note: ADV and VAD contain the same dimensions, just reordered:
#   ADV = (Arousal, Dominance, Valence)
#   VAD = (Valence, Arousal, Dominance)

ADV_PROTOTYPES = {
    "neutral": (0.0, 0.0, 0.0),      # Origin - baseline state
    "happy": (0.6, 0.6, 0.8),         # High arousal, dominant, positive
    "sad": (-0.4, -0.5, -0.6),        # Low arousal, submissive, negative
    "angry": (0.8, 0.7, -0.5),        # High arousal, dominant, negative
    "surprised": (0.8, 0.2, 0.3),     # High arousal, neutral dominance, slightly positive
    "calm": (-0.5, 0.3, 0.4),         # Low arousal, slightly dominant, positive
    "fearful": (0.7, -0.7, -0.7),     # High arousal, submissive, negative
    "disgusted": (0.3, 0.4, -0.6),    # Moderate arousal, slightly dominant, negative
    "excited": (0.9, 0.5, 0.7),       # Very high arousal, dominant, positive
    "bored": (-0.7, -0.2, -0.2),      # Very low arousal, slightly submissive, negative
    "tender": (-0.2, -0.2, 0.7),      # Low arousal, slightly submissive, positive
    "anxious": (0.6, -0.4, -0.4),     # High arousal, submissive, negative
    "content": (-0.3, 0.2, 0.6),      # Low arousal, slightly dominant, positive
    "proud": (0.4, 0.8, 0.7),         # Moderate arousal, very dominant, positive
    "ashamed": (-0.2, -0.8, -0.5),    # Low arousal, very submissive, negative
}

# Core emotions (8 for primary classification)
CORE_EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "calm", "fearful", "disgusted"]

EMOTION_TO_IDX = {e: i for i, e in enumerate(CORE_EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}


# =============================================================================
# ADV-VAD CONVERSION UTILITIES
# =============================================================================

def adv_to_vad(adv: torch.Tensor) -> torch.Tensor:
    """
    Convert ADV coordinates to VAD coordinates.

    ADV = (Arousal, Dominance, Valence)
    VAD = (Valence, Arousal, Dominance)

    Args:
        adv: Tensor [..., 3] with (A, D, V)

    Returns:
        Tensor [..., 3] with (V, A, D)
    """
    a, d, v = adv[..., 0], adv[..., 1], adv[..., 2]
    return torch.stack([v, a, d], dim=-1)


def vad_to_adv(vad: torch.Tensor) -> torch.Tensor:
    """
    Convert VAD coordinates to ADV coordinates.

    VAD = (Valence, Arousal, Dominance)
    ADV = (Arousal, Dominance, Valence)

    Args:
        vad: Tensor [..., 3] with (V, A, D)

    Returns:
        Tensor [..., 3] with (A, D, V)
    """
    v, a, d = vad[..., 0], vad[..., 1], vad[..., 2]
    return torch.stack([a, d, v], dim=-1)


def get_adv_for_emotion(emotion: str) -> Tuple[float, float, float]:
    """Get ADV coordinates for an emotion name."""
    return ADV_PROTOTYPES.get(emotion.lower(), ADV_PROTOTYPES["neutral"])


def adv_to_emotion_name(adv: Union[torch.Tensor, Tuple[float, float, float]]) -> str:
    """Find nearest emotion name for ADV coordinates."""
    if isinstance(adv, torch.Tensor):
        adv = (adv[0].item(), adv[1].item(), adv[2].item())

    min_dist = float('inf')
    nearest = "neutral"

    for emotion, proto in ADV_PROTOTYPES.items():
        dist = sum((a - b) ** 2 for a, b in zip(adv, proto)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = emotion

    return nearest


# =============================================================================
# NONLINEAR ADV QUANTIZATION
# =============================================================================

class NonlinearADVQuantizer(nn.Module):
    """
    Nonlinear ADV Quantization for fine-grained emotion control.

    Key features:
    1. Quantizes continuous ADV into discrete levels
    2. Uses nonlinear mapping for better control at certain ranges
    3. Supports soft quantization (differentiable) during training
    4. Hard quantization for inference

    The nonlinear mapping uses tanh-based warping to concentrate
    quantization bins near neutral (origin) for finer control.
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config
        self.num_levels = config.num_quantization_levels
        self.temperature = config.quantization_temperature

        # Learnable quantization centroids per dimension
        # Initialize uniformly in [-1, 1]
        centroids = torch.linspace(-1, 1, self.num_levels)
        self.register_buffer('base_centroids', centroids)

        # Learnable offset for nonlinear warping
        if config.use_nonlinear_quantization:
            self.warp_scale = nn.Parameter(torch.ones(3))  # Per-dimension scale
            self.warp_bias = nn.Parameter(torch.zeros(3))  # Per-dimension bias

        # Learnable embeddings for each quantization level per dimension
        self.level_embeddings = nn.Parameter(
            torch.randn(3, self.num_levels, config.emotion_dim // 3) * 0.02
        )

    def _apply_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nonlinear warping to concentrate bins near origin."""
        if not self.config.use_nonlinear_quantization:
            return x

        # Tanh-based warping: more bins near center
        scale = self.warp_scale.view(1, 3)
        bias = self.warp_bias.view(1, 3)
        warped = torch.tanh(scale * x + bias)
        return warped

    def _inverse_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of nonlinear warping."""
        if not self.config.use_nonlinear_quantization:
            return x

        scale = self.warp_scale.view(1, 3)
        bias = self.warp_bias.view(1, 3)
        # atanh with clamping for stability
        x_clamped = x.clamp(-0.999, 0.999)
        unwarped = (torch.atanh(x_clamped) - bias) / (scale + 1e-8)
        return unwarped

    def get_centroids(self) -> torch.Tensor:
        """Get current centroids after warping."""
        centroids = self.base_centroids.unsqueeze(0).expand(3, -1)  # [3, num_levels]
        if self.config.use_nonlinear_quantization:
            # Warp each dimension's centroids
            warped = []
            for dim in range(3):
                c = centroids[dim:dim+1, :].t()  # [num_levels, 1]
                w = self._apply_warp(c)
                warped.append(w.squeeze(-1))
            centroids = torch.stack(warped, dim=0)  # [3, num_levels]
        return centroids

    def quantize_soft(self, adv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Soft quantization (differentiable) for training.

        Args:
            adv: ADV coordinates [batch, 3]

        Returns:
            Dict with quantized values, indices, and embeddings
        """
        batch_size = adv.shape[0]
        device = adv.device

        # Get centroids
        centroids = self.get_centroids()  # [3, num_levels]

        # Compute distances to all centroids per dimension
        # adv: [batch, 3], centroids: [3, num_levels]
        adv_expanded = adv.unsqueeze(-1)  # [batch, 3, 1]
        centroids_expanded = centroids.unsqueeze(0)  # [1, 3, num_levels]

        distances = (adv_expanded - centroids_expanded).abs()  # [batch, 3, num_levels]

        # Soft assignment via softmax
        soft_weights = F.softmax(-distances / self.temperature, dim=-1)  # [batch, 3, num_levels]

        # Quantized values (weighted sum of centroids)
        quantized = (soft_weights * centroids_expanded).sum(dim=-1)  # [batch, 3]

        # Get embeddings
        # soft_weights: [batch, 3, num_levels]
        # level_embeddings: [3, num_levels, embed_dim/3]
        embeddings_list = []
        embed_dim_per = self.config.emotion_dim // 3
        for dim in range(3):
            weights = soft_weights[:, dim, :]  # [batch, num_levels]
            embs = self.level_embeddings[dim]  # [num_levels, embed_dim/3]
            weighted_emb = torch.matmul(weights, embs)  # [batch, embed_dim/3]
            embeddings_list.append(weighted_emb)
        embeddings = torch.cat(embeddings_list, dim=-1)  # [batch, emotion_dim]

        # Hard indices (for analysis)
        hard_indices = soft_weights.argmax(dim=-1)  # [batch, 3]

        return {
            'quantized': quantized,
            'soft_weights': soft_weights,
            'hard_indices': hard_indices,
            'embeddings': embeddings,
            'centroids': centroids,
        }

    def quantize_hard(self, adv: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hard quantization for inference.

        Args:
            adv: ADV coordinates [batch, 3]

        Returns:
            Dict with quantized values, indices, and embeddings
        """
        batch_size = adv.shape[0]
        device = adv.device

        # Get centroids
        centroids = self.get_centroids()  # [3, num_levels]

        # Find nearest centroid per dimension
        adv_expanded = adv.unsqueeze(-1)  # [batch, 3, 1]
        centroids_expanded = centroids.unsqueeze(0)  # [1, 3, num_levels]

        distances = (adv_expanded - centroids_expanded).abs()  # [batch, 3, num_levels]
        indices = distances.argmin(dim=-1)  # [batch, 3]

        # Get quantized values
        quantized = torch.zeros_like(adv)
        for dim in range(3):
            quantized[:, dim] = centroids[dim, indices[:, dim]]

        # Get embeddings
        embeddings_list = []
        for dim in range(3):
            idx = indices[:, dim]  # [batch]
            embs = self.level_embeddings[dim][idx]  # [batch, embed_dim/3]
            embeddings_list.append(embs)
        embeddings = torch.cat(embeddings_list, dim=-1)  # [batch, emotion_dim]

        return {
            'quantized': quantized,
            'hard_indices': indices,
            'embeddings': embeddings,
            'centroids': centroids,
        }

    def forward(
        self,
        adv: torch.Tensor,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize ADV coordinates.

        Args:
            adv: ADV coordinates [batch, 3]
            hard: Use hard quantization (default: soft during training)

        Returns:
            Dict with quantized values and embeddings
        """
        if hard or not self.training:
            return self.quantize_hard(adv)
        else:
            return self.quantize_soft(adv)

    def indices_to_adv(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert quantization indices back to ADV coordinates.

        Args:
            indices: Quantization indices [batch, 3]

        Returns:
            ADV coordinates [batch, 3]
        """
        centroids = self.get_centroids()  # [3, num_levels]
        adv = torch.zeros_like(indices, dtype=torch.float)
        for dim in range(3):
            adv[:, dim] = centroids[dim, indices[:, dim]]
        return adv


# =============================================================================
# DISCRETE-ADV MAPPING
# =============================================================================

class DiscreteADVMapper(nn.Module):
    """
    Bidirectional mapping between discrete emotion labels and ADV coordinates.

    Features:
    1. Map discrete labels to ADV (one-hot → ADV)
    2. Map ADV to discrete distribution (ADV → softmax over emotions)
    3. Learnable prototype refinement
    4. Consistency loss between mappings
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # Initialize ADV prototypes (learnable)
        adv_values = torch.tensor([
            ADV_PROTOTYPES[e] for e in CORE_EMOTIONS
        ], dtype=torch.float32)
        self.adv_prototypes = nn.Parameter(adv_values)

        # Discrete → ADV MLP (for soft mapping)
        self.discrete_to_adv = nn.Sequential(
            nn.Linear(config.num_discrete_emotions, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 3),
            nn.Tanh(),  # Bound to [-1, 1]
        )

        # ADV → Discrete classifier
        self.adv_to_discrete = nn.Sequential(
            nn.Linear(3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.num_discrete_emotions),
        )

    def get_adv_for_emotion(
        self,
        emotion: Union[str, int, torch.Tensor],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Get ADV coordinates for a discrete emotion.

        Args:
            emotion: Emotion name, index, or one-hot tensor

        Returns:
            ADV coordinates [3] or [batch, 3]
        """
        if device is None:
            device = self.adv_prototypes.device

        if isinstance(emotion, str):
            emotion_lower = emotion.lower()
            if emotion_lower in EMOTION_TO_IDX:
                idx = EMOTION_TO_IDX[emotion_lower]
                return self.adv_prototypes[idx]
            elif emotion_lower in ADV_PROTOTYPES:
                return torch.tensor(ADV_PROTOTYPES[emotion_lower], device=device)
            else:
                return self.adv_prototypes[0]  # neutral
        elif isinstance(emotion, int):
            return self.adv_prototypes[emotion]
        elif isinstance(emotion, torch.Tensor):
            if emotion.dim() == 0:
                return self.adv_prototypes[emotion.long()]
            elif emotion.dim() == 1 and emotion.shape[0] == self.config.num_discrete_emotions:
                # One-hot or soft weights
                return torch.matmul(emotion, self.adv_prototypes)
            elif emotion.dim() == 2:
                # Batch of weights
                return torch.matmul(emotion, self.adv_prototypes)
            else:
                raise ValueError(f"Invalid emotion tensor shape: {emotion.shape}")
        else:
            raise ValueError(f"Invalid emotion type: {type(emotion)}")

    def discrete_to_adv_coords(
        self,
        emotion_probs: torch.Tensor,  # [batch, num_emotions]
        use_mlp: bool = True,
    ) -> torch.Tensor:
        """
        Map discrete emotion distribution to ADV coordinates.

        Args:
            emotion_probs: Emotion probabilities [batch, num_emotions]
            use_mlp: Use MLP for soft mapping (else use prototype lookup)

        Returns:
            ADV coordinates [batch, 3]
        """
        if use_mlp:
            return self.discrete_to_adv(emotion_probs)
        else:
            return torch.matmul(emotion_probs, self.adv_prototypes)

    def adv_to_discrete_logits(
        self,
        adv: torch.Tensor,  # [batch, 3]
    ) -> torch.Tensor:
        """
        Map ADV coordinates to discrete emotion logits.

        Args:
            adv: ADV coordinates [batch, 3]

        Returns:
            Emotion logits [batch, num_emotions]
        """
        return self.adv_to_discrete(adv)

    def adv_to_nearest_emotion(
        self,
        adv: torch.Tensor,  # [batch, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find nearest discrete emotion for ADV coordinates.

        Args:
            adv: ADV coordinates [batch, 3]

        Returns:
            Tuple of (emotion_indices, distances)
        """
        # Distance to all prototypes
        adv_expanded = adv.unsqueeze(1)  # [batch, 1, 3]
        proto_expanded = self.adv_prototypes.unsqueeze(0)  # [1, num_emotions, 3]
        distances = torch.norm(adv_expanded - proto_expanded, dim=-1)  # [batch, num_emotions]

        # Find nearest
        min_distances, indices = distances.min(dim=-1)

        return indices, min_distances

    def forward(
        self,
        adv: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        emotion_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Bidirectional mapping.

        Args:
            adv: ADV coordinates [batch, 3] (optional)
            emotion_labels: Discrete emotion indices [batch] (optional)
            emotion_probs: Emotion probabilities [batch, num_emotions] (optional)

        Returns:
            Dict with mapped values
        """
        result = {}

        # ADV → Discrete
        if adv is not None:
            logits = self.adv_to_discrete_logits(adv)
            probs = F.softmax(logits, dim=-1)
            nearest_idx, nearest_dist = self.adv_to_nearest_emotion(adv)
            result['discrete_logits'] = logits
            result['discrete_probs'] = probs
            result['nearest_emotion'] = nearest_idx
            result['nearest_distance'] = nearest_dist

        # Discrete → ADV
        if emotion_labels is not None:
            # One-hot encoding
            one_hot = F.one_hot(emotion_labels, self.config.num_discrete_emotions).float()
            adv_from_label = self.discrete_to_adv_coords(one_hot)
            adv_from_proto = torch.matmul(one_hot, self.adv_prototypes)
            result['adv_from_label'] = adv_from_label
            result['adv_from_proto'] = adv_from_proto

        if emotion_probs is not None:
            adv_from_probs = self.discrete_to_adv_coords(emotion_probs)
            result['adv_from_probs'] = adv_from_probs

        return result


# =============================================================================
# ADV ENCODER
# =============================================================================

class ADVEncoder(nn.Module):
    """
    Encodes ADV coordinates to emotion embeddings.

    Features:
    1. Nonlinear quantization for fine-grained control
    2. Multi-layer encoding with residual connections
    3. Integrates quantization embeddings with raw ADV
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # Quantizer
        self.quantizer = NonlinearADVQuantizer(config)

        # Raw ADV encoder
        self.adv_encoder = nn.Sequential(
            nn.Linear(config.adv_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

        # Fusion layer (combines quantized embeddings + raw encoding)
        self.fusion = nn.Sequential(
            nn.Linear(config.emotion_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

        # Output projection to prosody tokens
        self.output_projection = nn.Sequential(
            nn.Linear(config.emotion_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        adv: torch.Tensor,  # [batch, 3]
        return_quantization: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode ADV coordinates.

        Args:
            adv: ADV coordinates [batch, 3]
            return_quantization: Include quantization details in output

        Returns:
            Dict with embeddings and prosody tokens
        """
        batch_size = adv.shape[0]

        # Quantize ADV
        quant_result = self.quantizer(adv)
        quant_embedding = quant_result['embeddings']  # [batch, emotion_dim]

        # Encode raw ADV
        raw_embedding = self.adv_encoder(adv)  # [batch, emotion_dim]

        # Fuse embeddings
        fused = self.fusion(torch.cat([quant_embedding, raw_embedding], dim=-1))

        # Project to prosody tokens
        tokens = self.output_projection(fused)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        result = {
            'embedding': fused,
            'tokens': tokens,
            'quantized_adv': quant_result['quantized'],
        }

        if return_quantization:
            result['quantization'] = quant_result

        return result


# =============================================================================
# OT-CFM FLOW MATCHING
# =============================================================================

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


class OptimalTransportCoupling:
    """
    Optimal Transport coupling for OT-CFM.

    Uses Sinkhorn algorithm to find optimal pairing between
    noise samples and data samples for straighter flows.
    """

    def __init__(self, reg: float = 0.05, max_iter: int = 50):
        self.reg = reg
        self.max_iter = max_iter

    def compute_coupling(
        self,
        x0: torch.Tensor,  # [batch, dim] noise
        x1: torch.Tensor,  # [batch, dim] data
    ) -> torch.Tensor:
        """
        Compute OT coupling matrix using Sinkhorn algorithm.

        Returns:
            Coupling matrix [batch, batch]
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Cost matrix (squared Euclidean distance)
        cost = torch.cdist(x0, x1, p=2) ** 2  # [batch, batch]

        # Normalize cost
        cost = cost / (cost.max() + 1e-8)

        # Sinkhorn algorithm
        K = torch.exp(-cost / self.reg)

        # Initialize dual variables
        u = torch.ones(batch_size, device=device) / batch_size
        v = torch.ones(batch_size, device=device) / batch_size

        for _ in range(self.max_iter):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.t() @ u + 1e-8)

        # Coupling matrix
        coupling = torch.diag(u) @ K @ torch.diag(v)

        return coupling

    def sample_coupling(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample paired points using OT coupling.

        Returns:
            Paired (x0, x1) with optimal matching
        """
        coupling = self.compute_coupling(x0, x1)

        # Sample pairs according to coupling
        batch_size = x0.shape[0]

        # Use row-wise argmax for deterministic matching
        indices = coupling.argmax(dim=1)

        # Reorder x1 to match x0
        x1_matched = x1[indices]

        return x0, x1_matched


class VelocityNetwork(nn.Module):
    """
    Velocity prediction network for flow matching.

    Predicts the velocity field v(x_t, t, c) where:
    - x_t: Current state
    - t: Time
    - c: Conditioning (ADV embedding)
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.flow_hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(config.output_dim, config.flow_hidden_dim)

        # Condition projection
        self.cond_proj = nn.Linear(config.emotion_dim, config.flow_hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.flow_hidden_dim,
            nhead=config.flow_num_heads,
            dim_feedforward=config.flow_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.flow_num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.flow_hidden_dim, config.output_dim)

    def forward(
        self,
        x_t: torch.Tensor,     # [batch, tokens, dim]
        t: torch.Tensor,       # [batch]
        condition: torch.Tensor,  # [batch, emotion_dim]
    ) -> torch.Tensor:
        """
        Predict velocity at (x_t, t, condition).

        Returns:
            Velocity [batch, tokens, dim]
        """
        batch_size, num_tokens, dim = x_t.shape

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, hidden_dim]

        # Condition embedding
        c_emb = self.cond_proj(condition)  # [batch, hidden_dim]

        # Combined conditioning
        cond_emb = t_emb + c_emb  # [batch, hidden_dim]

        # Project input
        h = self.input_proj(x_t)  # [batch, tokens, hidden_dim]

        # Add conditioning (broadcast to all tokens)
        h = h + cond_emb.unsqueeze(1)

        # Transformer
        h = self.transformer(h)

        # Output
        velocity = self.output_proj(h)  # [batch, tokens, dim]

        return velocity


class OTCFMDecoder(nn.Module):
    """
    Optimal Transport Conditional Flow Matching Decoder.

    Generates prosody tokens from ADV conditioning using flow matching.
    Uses OT coupling for better training dynamics.
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config
        self.sigma_min = config.sigma_min

        # Velocity network
        self.velocity_net = VelocityNetwork(config)

        # OT coupling
        self.ot_coupling = OptimalTransportCoupling(reg=config.ot_reg)

    def compute_flow_loss(
        self,
        x1: torch.Tensor,        # [batch, tokens, dim] target
        condition: torch.Tensor,  # [batch, emotion_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.

        Args:
            x1: Target prosody tokens
            condition: ADV emotion embedding

        Returns:
            Dict with flow loss and diagnostics
        """
        batch_size, num_tokens, dim = x1.shape
        device = x1.device

        # Sample noise
        x0 = torch.randn_like(x1)

        # Apply OT coupling (operate on flattened tokens)
        if self.config.use_ot_cfm:
            x0_flat = x0.view(batch_size, -1)
            x1_flat = x1.view(batch_size, -1)
            x0_flat, x1_flat = self.ot_coupling.sample_coupling(x0_flat, x1_flat)
            x0 = x0_flat.view(batch_size, num_tokens, dim)
            x1 = x1_flat.view(batch_size, num_tokens, dim)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Compute x_t and target velocity
        sigma_t = (1 - t) + t * self.sigma_min
        sigma_t = sigma_t.view(batch_size, 1, 1)
        t_exp = t.view(batch_size, 1, 1)

        x_t = t_exp * x1 + sigma_t * x0

        # Target velocity (derivative of x_t w.r.t. t)
        target_velocity = x1 - x0

        # Predict velocity
        pred_velocity = self.velocity_net(x_t, t, condition)

        # Loss (MSE on velocity)
        loss = F.mse_loss(pred_velocity, target_velocity)

        return {
            'loss': loss,
            't': t,
            'pred_velocity': pred_velocity,
            'target_velocity': target_velocity,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,  # [batch, emotion_dim]
        num_tokens: int = None,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        Sample prosody tokens using ODE integration.

        Args:
            condition: ADV emotion embedding
            num_tokens: Number of tokens to generate
            num_steps: ODE integration steps

        Returns:
            Generated prosody tokens [batch, tokens, dim]
        """
        if num_tokens is None:
            num_tokens = self.config.num_prosody_tokens
        if num_steps is None:
            num_steps = self.config.num_flow_steps

        batch_size = condition.shape[0]
        device = condition.device

        # Start from noise
        x = torch.randn(batch_size, num_tokens, self.config.output_dim, device=device)

        # ODE integration (Euler method)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            velocity = self.velocity_net(x, t, condition)
            x = x + dt * velocity

        return x


# =============================================================================
# SEMI-SUPERVISED LOSS
# =============================================================================

class SemiSupervisedLoss(nn.Module):
    """
    Semi-supervised loss for unified training on mixed annotation types.

    Handles:
    1. Samples with discrete emotion labels
    2. Samples with continuous ADV annotations
    3. Samples with both
    4. Samples with neither (unsupervised)
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        # Model outputs
        discrete_logits: Optional[torch.Tensor] = None,  # [batch, num_emotions]
        predicted_adv: Optional[torch.Tensor] = None,    # [batch, 3]
        adv_from_discrete: Optional[torch.Tensor] = None,  # [batch, 3]

        # Ground truth (may be partially available)
        target_emotion: Optional[torch.Tensor] = None,   # [batch] or None where unavailable
        target_adv: Optional[torch.Tensor] = None,       # [batch, 3] or None where unavailable
        emotion_mask: Optional[torch.Tensor] = None,     # [batch] bool mask
        adv_mask: Optional[torch.Tensor] = None,         # [batch] bool mask
    ) -> Dict[str, torch.Tensor]:
        """
        Compute semi-supervised loss.

        Args:
            discrete_logits: Predicted emotion logits
            predicted_adv: Predicted ADV from model
            adv_from_discrete: ADV mapped from discrete prediction
            target_emotion: Ground truth discrete emotions (where available)
            target_adv: Ground truth ADV (where available)
            emotion_mask: Mask for samples with emotion labels
            adv_mask: Mask for samples with ADV labels

        Returns:
            Dict with individual losses and total
        """
        device = discrete_logits.device if discrete_logits is not None else predicted_adv.device
        batch_size = discrete_logits.shape[0] if discrete_logits is not None else predicted_adv.shape[0]

        losses = {}

        # Initialize masks if not provided
        if emotion_mask is None:
            emotion_mask = target_emotion is not None
            if emotion_mask:
                emotion_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        if adv_mask is None:
            adv_mask = target_adv is not None
            if adv_mask:
                adv_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Discrete emotion loss
        if discrete_logits is not None and target_emotion is not None and emotion_mask.any():
            ce_loss = self.ce_loss(discrete_logits, target_emotion)  # [batch]
            ce_loss = (ce_loss * emotion_mask.float()).sum() / (emotion_mask.sum() + 1e-8)
            losses['discrete'] = ce_loss * self.config.discrete_loss_weight
        else:
            losses['discrete'] = torch.tensor(0.0, device=device)

        # ADV regression loss
        if predicted_adv is not None and target_adv is not None and adv_mask.any():
            mse_loss = self.mse_loss(predicted_adv, target_adv).mean(dim=-1)  # [batch]
            mse_loss = (mse_loss * adv_mask.float()).sum() / (adv_mask.sum() + 1e-8)
            losses['adv'] = mse_loss * self.config.adv_loss_weight
        else:
            losses['adv'] = torch.tensor(0.0, device=device)

        # Consistency loss (ADV ↔ Discrete should match)
        if adv_from_discrete is not None and predicted_adv is not None:
            # For samples with both labels, ADV prediction should match
            both_mask = emotion_mask & adv_mask
            if both_mask.any():
                consistency = self.mse_loss(adv_from_discrete, target_adv).mean(dim=-1)
                consistency = (consistency * both_mask.float()).sum() / (both_mask.sum() + 1e-8)
                losses['consistency'] = consistency * self.config.consistency_loss_weight
            else:
                losses['consistency'] = torch.tensor(0.0, device=device)
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)

        # Total
        losses['total'] = losses['discrete'] + losses['adv'] + losses['consistency']

        return losses


# =============================================================================
# UDDETTS FULL MODEL
# =============================================================================

class UDDETTS(nn.Module):
    """
    UDDETTS: Unified Dimensional Emotion TTS.

    Full model combining:
    1. ADV encoding with nonlinear quantization
    2. Discrete-ADV bidirectional mapping
    3. OT-CFM flow matching decoder
    4. Semi-supervised training support
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # Components
        self.adv_encoder = ADVEncoder(config)
        self.discrete_mapper = DiscreteADVMapper(config)

        # OT-CFM decoder (optional)
        if config.use_ot_cfm:
            self.flow_decoder = OTCFMDecoder(config)

        # Semi-supervised loss
        self.semi_supervised_loss = SemiSupervisedLoss(config)

    def encode_adv(
        self,
        adv: torch.Tensor,
        return_quantization: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode ADV coordinates to emotion embedding."""
        return self.adv_encoder(adv, return_quantization)

    def encode_discrete(
        self,
        emotion_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode discrete emotion labels.

        First maps to ADV, then encodes.
        """
        # Get ADV from discrete
        mapping = self.discrete_mapper(emotion_labels=emotion_labels)
        adv = mapping['adv_from_proto']  # Use prototype lookup

        # Encode ADV
        result = self.adv_encoder(adv)
        result['adv_from_discrete'] = adv
        result['mapping'] = mapping

        return result

    def forward(
        self,
        adv: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        emotion_mask: Optional[torch.Tensor] = None,
        adv_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with flexible input (ADV or discrete or both).

        Args:
            adv: ADV coordinates [batch, 3] (optional)
            emotion_labels: Discrete emotion labels [batch] (optional)
            target_tokens: Target prosody tokens for flow loss [batch, tokens, dim]
            emotion_mask: Mask for samples with emotion labels
            adv_mask: Mask for samples with ADV labels

        Returns:
            Dict with embeddings, tokens, and losses
        """
        result = {}
        device = adv.device if adv is not None else emotion_labels.device

        # Handle different input combinations
        if adv is not None and emotion_labels is not None:
            # Both provided - use ADV as primary
            adv_result = self.adv_encoder(adv, return_quantization=True)
            mapping = self.discrete_mapper(adv=adv, emotion_labels=emotion_labels)
            result['embedding'] = adv_result['embedding']
            result['tokens'] = adv_result['tokens']
            result['quantization'] = adv_result['quantization']
            result['adv_from_discrete'] = mapping['adv_from_proto']
        elif adv is not None:
            # Only ADV
            adv_result = self.adv_encoder(adv, return_quantization=True)
            mapping = self.discrete_mapper(adv=adv)
            result['embedding'] = adv_result['embedding']
            result['tokens'] = adv_result['tokens']
            result['quantization'] = adv_result['quantization']
        elif emotion_labels is not None:
            # Only discrete labels
            disc_result = self.encode_discrete(emotion_labels)
            result['embedding'] = disc_result['embedding']
            result['tokens'] = disc_result['tokens']
            mapping = self.discrete_mapper(emotion_labels=emotion_labels)
            adv = disc_result['adv_from_discrete']
            result['adv_from_discrete'] = adv
        else:
            raise ValueError("Either adv or emotion_labels must be provided")

        result['mapping'] = mapping

        # Compute losses
        if target_tokens is not None and self.config.use_ot_cfm:
            flow_result = self.flow_decoder.compute_flow_loss(
                target_tokens,
                result['embedding']
            )
            result['flow_loss'] = flow_result['loss']

        # Semi-supervised loss
        if emotion_labels is not None or adv is not None:
            discrete_logits = mapping.get('discrete_logits', None)
            predicted_adv = result.get('quantization', {}).get('quantized', adv)
            adv_from_discrete = result.get('adv_from_discrete', None)

            semi_losses = self.semi_supervised_loss(
                discrete_logits=discrete_logits,
                predicted_adv=predicted_adv,
                adv_from_discrete=adv_from_discrete,
                target_emotion=emotion_labels,
                target_adv=adv,
                emotion_mask=emotion_mask,
                adv_mask=adv_mask,
            )
            result['semi_supervised_loss'] = semi_losses

        return result

    @torch.no_grad()
    def generate(
        self,
        adv: Optional[torch.Tensor] = None,
        emotion: Optional[str] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        Generate prosody tokens from ADV or emotion.

        Args:
            adv: ADV coordinates [batch, 3]
            emotion: Emotion name (str)
            emotion_labels: Discrete emotion labels [batch]
            intensity: Scale factor for ADV
            num_steps: ODE integration steps

        Returns:
            Prosody tokens [batch, tokens, dim]
        """
        # Get ADV from input
        if adv is not None:
            adv = adv * intensity
        elif emotion is not None:
            adv = self.discrete_mapper.get_adv_for_emotion(emotion)
            adv = adv.unsqueeze(0) * intensity
        elif emotion_labels is not None:
            mapping = self.discrete_mapper(emotion_labels=emotion_labels)
            adv = mapping['adv_from_proto'] * intensity
        else:
            raise ValueError("Must provide adv, emotion, or emotion_labels")

        # Encode ADV
        enc_result = self.adv_encoder(adv)

        if self.config.use_ot_cfm:
            # Use flow decoder
            tokens = self.flow_decoder.sample(
                enc_result['embedding'],
                num_steps=num_steps,
            )
        else:
            # Use encoder output directly
            tokens = enc_result['tokens']

        return tokens


# =============================================================================
# UDDETTS ADAPTER (CSM Integration)
# =============================================================================

class UDDETTSAdapter(nn.Module):
    """
    Adapter for integrating UDDETTS with the CSM prosody pipeline.

    Provides convenient methods for:
    1. Encoding emotions from ADV or discrete labels
    2. Interpolating between emotions
    3. Generating prosody tokens for CSM
    4. Converting between ADV and VAD representations
    """

    def __init__(
        self,
        config: UDDETTSConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config
        self.prosody_hidden = prosody_hidden

        # Core model
        self.model = UDDETTS(config)

        # Adapter layer if dimensions don't match
        if config.output_dim != prosody_hidden:
            self.prosody_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.prosody_adapter = nn.Identity()

    def from_adv(
        self,
        adv: torch.Tensor,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from ADV coordinates.

        Args:
            adv: ADV coordinates [batch, 3] or [3]
            intensity: Scale factor

        Returns:
            Dict with prosody tokens and metadata
        """
        if adv.dim() == 1:
            adv = adv.unsqueeze(0)

        adv = adv * intensity

        result = self.model.encode_adv(adv, return_quantization=True)

        # Adapt to prosody dimension
        tokens = self.prosody_adapter(result['tokens'])

        return {
            'prosody_tokens': tokens,
            'embedding': result['embedding'],
            'quantized_adv': result['quantized_adv'],
            'quantization': result.get('quantization'),
        }

    def from_vad(
        self,
        vad: torch.Tensor,
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from VAD coordinates (converted to ADV).

        Args:
            vad: VAD coordinates [batch, 3] or [3]
            intensity: Scale factor

        Returns:
            Dict with prosody tokens and metadata
        """
        adv = vad_to_adv(vad)
        return self.from_adv(adv, intensity)

    def from_emotion(
        self,
        emotion: Union[str, int, torch.Tensor],
        intensity: float = 1.0,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from discrete emotion.

        Args:
            emotion: Emotion name, index, or label tensor
            intensity: Scale factor
            batch_size: Batch size if emotion is scalar

        Returns:
            Dict with prosody tokens and metadata
        """
        if isinstance(emotion, str):
            adv = self.model.discrete_mapper.get_adv_for_emotion(emotion)
            adv = adv.unsqueeze(0).expand(batch_size, -1)
        elif isinstance(emotion, int):
            adv = self.model.discrete_mapper.adv_prototypes[emotion]
            adv = adv.unsqueeze(0).expand(batch_size, -1)
        else:
            result = self.model.encode_discrete(emotion)
            adv = result['adv_from_discrete']

        return self.from_adv(adv, intensity)

    def interpolate(
        self,
        adv1: torch.Tensor,
        adv2: torch.Tensor,
        t: float,
        method: str = "slerp",
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two ADV coordinates.

        Args:
            adv1: Source ADV [batch, 3] or [3]
            adv2: Target ADV [batch, 3] or [3]
            t: Interpolation factor [0, 1]
            method: "lerp" or "slerp"

        Returns:
            Dict with interpolated prosody tokens
        """
        if adv1.dim() == 1:
            adv1 = adv1.unsqueeze(0)
        if adv2.dim() == 1:
            adv2 = adv2.unsqueeze(0)

        if method == "slerp":
            # Spherical interpolation
            adv1_norm = F.normalize(adv1, dim=-1)
            adv2_norm = F.normalize(adv2, dim=-1)

            dot = (adv1_norm * adv2_norm).sum(dim=-1, keepdim=True)
            omega = torch.acos(dot.clamp(-0.999, 0.999))
            sin_omega = torch.sin(omega)

            s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
            s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

            direction = s1 * adv1_norm + s2 * adv2_norm

            # Interpolate magnitudes
            mag1 = adv1.norm(dim=-1, keepdim=True)
            mag2 = adv2.norm(dim=-1, keepdim=True)
            mag = (1 - t) * mag1 + t * mag2

            adv_interp = direction * mag
        else:
            # Linear interpolation
            adv_interp = (1 - t) * adv1 + t * adv2

        return self.from_adv(adv_interp)

    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 1.0,
        method: str = "slerp",
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two named emotions.

        Args:
            emotion1: Source emotion name
            emotion2: Target emotion name
            t: Interpolation factor [0, 1]
            intensity: Overall intensity
            method: "lerp" or "slerp"

        Returns:
            Dict with interpolated prosody tokens
        """
        adv1 = self.model.discrete_mapper.get_adv_for_emotion(emotion1)
        adv2 = self.model.discrete_mapper.get_adv_for_emotion(emotion2)

        result = self.interpolate(adv1, adv2, t, method)

        # Apply intensity
        result['prosody_tokens'] = result['prosody_tokens'] * intensity

        return result

    def blend_emotions(
        self,
        emotions: List[Tuple[str, float]],
        intensity: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Blend multiple emotions with weights.

        Args:
            emotions: List of (emotion_name, weight) tuples
            intensity: Overall intensity

        Returns:
            Dict with blended prosody tokens
        """
        # Normalize weights
        total_weight = sum(w for _, w in emotions)
        if total_weight == 0:
            total_weight = 1.0

        # Weighted sum of ADV coordinates
        device = self.model.discrete_mapper.adv_prototypes.device
        blended_adv = torch.zeros(3, device=device)

        for emotion, weight in emotions:
            adv = self.model.discrete_mapper.get_adv_for_emotion(emotion)
            blended_adv += adv * (weight / total_weight)

        return self.from_adv(blended_adv.unsqueeze(0), intensity)

    def forward(
        self,
        adv: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (training mode).

        Args:
            adv: ADV coordinates [batch, 3]
            emotion_labels: Discrete emotion labels [batch]
            target_tokens: Target prosody tokens for flow loss

        Returns:
            Dict with tokens and losses
        """
        result = self.model(
            adv=adv,
            emotion_labels=emotion_labels,
            target_tokens=target_tokens,
            **kwargs,
        )

        # Adapt tokens
        result['prosody_tokens'] = self.prosody_adapter(result['tokens'])

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_uddetts_adapter(
    config: Optional[UDDETTSConfig] = None,
    prosody_hidden: int = 2048,
) -> UDDETTSAdapter:
    """Create UDDETTS adapter with default configuration."""
    if config is None:
        config = UDDETTSConfig()
    return UDDETTSAdapter(config, prosody_hidden)


def adv_to_description(adv: torch.Tensor, threshold: float = 0.3) -> str:
    """
    Convert ADV coordinates to natural language description.

    Args:
        adv: ADV coordinates [3]
        threshold: Threshold for describing dimensions

    Returns:
        Natural language description
    """
    if isinstance(adv, torch.Tensor):
        a, d, v = adv[0].item(), adv[1].item(), adv[2].item()
    else:
        a, d, v = adv

    parts = []

    # Arousal
    if abs(a) > threshold:
        if a > 0.6:
            parts.append("very energetic")
        elif a > 0.3:
            parts.append("energetic")
        elif a < -0.6:
            parts.append("very calm")
        elif a < -0.3:
            parts.append("calm")

    # Dominance
    if abs(d) > threshold:
        if d > 0.6:
            parts.append("assertive")
        elif d > 0.3:
            parts.append("confident")
        elif d < -0.6:
            parts.append("submissive")
        elif d < -0.3:
            parts.append("hesitant")

    # Valence
    if abs(v) > threshold:
        if v > 0.6:
            parts.append("very positive")
        elif v > 0.3:
            parts.append("positive")
        elif v < -0.6:
            parts.append("very negative")
        elif v < -0.3:
            parts.append("negative")

    if not parts:
        return "neutral"

    return ", ".join(parts)


def description_to_adv(description: str) -> torch.Tensor:
    """
    Parse natural language description to ADV coordinates.

    Args:
        description: Natural language emotion description

    Returns:
        ADV coordinates [3]
    """
    description = description.lower()
    adv = torch.zeros(3)

    # Arousal keywords
    if any(w in description for w in ["very energetic", "excited", "enthusiastic", "hyper"]):
        adv[0] = 0.9
    elif any(w in description for w in ["energetic", "lively", "animated"]):
        adv[0] = 0.6
    elif any(w in description for w in ["very calm", "relaxed", "peaceful", "serene"]):
        adv[0] = -0.8
    elif any(w in description for w in ["calm", "quiet", "subdued"]):
        adv[0] = -0.5

    # Dominance keywords
    if any(w in description for w in ["assertive", "commanding", "authoritative"]):
        adv[1] = 0.8
    elif any(w in description for w in ["confident", "assured"]):
        adv[1] = 0.5
    elif any(w in description for w in ["submissive", "meek", "timid"]):
        adv[1] = -0.7
    elif any(w in description for w in ["hesitant", "uncertain"]):
        adv[1] = -0.4

    # Valence keywords
    if any(w in description for w in ["very positive", "joyful", "elated", "ecstatic"]):
        adv[2] = 0.9
    elif any(w in description for w in ["positive", "happy", "pleased", "content"]):
        adv[2] = 0.6
    elif any(w in description for w in ["very negative", "miserable", "devastated", "furious"]):
        adv[2] = -0.8
    elif any(w in description for w in ["negative", "sad", "upset", "angry"]):
        adv[2] = -0.5

    return adv


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UDDETTS: Unified Dimensional Emotion TTS - Test Suite")
    print("=" * 70)

    config = UDDETTSConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: ADV Prototypes
    print("\n[Test 1] ADV Prototypes...")
    for emotion, adv in list(ADV_PROTOTYPES.items())[:5]:
        print(f"  {emotion:12s}: A={adv[0]:+.2f}, D={adv[1]:+.2f}, V={adv[2]:+.2f}")
    print("  [PASS]")

    # Test 2: ADV-VAD Conversion
    print("\n[Test 2] ADV ↔ VAD Conversion...")
    test_adv = torch.tensor([[0.6, 0.4, 0.8], [-0.3, -0.5, -0.6]])
    test_vad = adv_to_vad(test_adv)
    recovered = vad_to_adv(test_vad)
    error = (test_adv - recovered).abs().max().item()
    print(f"  Original ADV: {test_adv[0].tolist()}")
    print(f"  Converted VAD: {test_vad[0].tolist()}")
    print(f"  Recovered ADV: {recovered[0].tolist()}")
    print(f"  Max error: {error:.6f}")
    assert error < 1e-6, "Conversion error!"
    print("  [PASS]")

    # Test 3: Nonlinear Quantizer
    print("\n[Test 3] Nonlinear ADV Quantizer...")
    quantizer = NonlinearADVQuantizer(config).to(device)
    adv_input = torch.randn(4, 3, device=device).tanh()  # Bounded ADV
    quant_result = quantizer(adv_input)
    print(f"  Input ADV shape: {adv_input.shape}")
    print(f"  Quantized ADV shape: {quant_result['quantized'].shape}")
    print(f"  Hard indices shape: {quant_result['hard_indices'].shape}")
    print(f"  Embeddings shape: {quant_result['embeddings'].shape}")
    print(f"  Number of levels: {config.num_quantization_levels}")
    print("  [PASS]")

    # Test 4: Discrete-ADV Mapper
    print("\n[Test 4] Discrete-ADV Mapper...")
    mapper = DiscreteADVMapper(config).to(device)
    emotion_labels = torch.tensor([0, 1, 2, 3], device=device)  # neutral, happy, sad, angry
    mapping = mapper(emotion_labels=emotion_labels)
    print(f"  Emotion labels: {emotion_labels.tolist()}")
    print(f"  ADV from proto: {mapping['adv_from_proto'].tolist()}")
    print(f"  ADV from label (MLP): {mapping['adv_from_label'].tolist()}")
    print("  [PASS]")

    # Test 5: ADV Encoder
    print("\n[Test 5] ADV Encoder...")
    encoder = ADVEncoder(config).to(device)
    enc_result = encoder(adv_input, return_quantization=True)
    print(f"  Input ADV shape: {adv_input.shape}")
    print(f"  Embedding shape: {enc_result['embedding'].shape}")
    print(f"  Tokens shape: {enc_result['tokens'].shape}")
    print(f"  Quantized ADV: {enc_result['quantized_adv'].shape}")
    print("  [PASS]")

    # Test 6: OT-CFM Decoder
    print("\n[Test 6] OT-CFM Flow Decoder...")
    flow_decoder = OTCFMDecoder(config).to(device)

    # Training loss
    target_tokens = torch.randn(4, config.num_prosody_tokens, config.output_dim, device=device)
    condition = torch.randn(4, config.emotion_dim, device=device)
    flow_result = flow_decoder.compute_flow_loss(target_tokens, condition)
    print(f"  Flow loss: {flow_result['loss'].item():.4f}")

    # Sampling
    sampled = flow_decoder.sample(condition, num_steps=10)
    print(f"  Sampled tokens shape: {sampled.shape}")
    print("  [PASS]")

    # Test 7: Full UDDETTS Model
    print("\n[Test 7] Full UDDETTS Model...")
    model = UDDETTS(config).to(device)

    # With ADV input
    result_adv = model(adv=adv_input, target_tokens=target_tokens)
    print(f"  With ADV - Tokens shape: {result_adv['tokens'].shape}")
    print(f"  Flow loss: {result_adv.get('flow_loss', 'N/A')}")

    # With discrete input
    result_disc = model(emotion_labels=emotion_labels, target_tokens=target_tokens)
    print(f"  With discrete - Tokens shape: {result_disc['tokens'].shape}")

    # With both inputs
    result_both = model(adv=adv_input, emotion_labels=emotion_labels, target_tokens=target_tokens)
    print(f"  With both - Tokens shape: {result_both['tokens'].shape}")
    print("  [PASS]")

    # Test 8: UDDETTSAdapter
    print("\n[Test 8] UDDETTSAdapter (CSM Integration)...")
    adapter = UDDETTSAdapter(config).to(device)

    # From ADV
    result = adapter.from_adv(adv_input[0], intensity=0.8)
    print(f"  From ADV - Prosody tokens: {result['prosody_tokens'].shape}")

    # From emotion name
    result = adapter.from_emotion("happy", intensity=0.9)
    print(f"  From emotion - Prosody tokens: {result['prosody_tokens'].shape}")

    # Interpolation
    result = adapter.interpolate_emotions("happy", "sad", t=0.5)
    print(f"  Interpolated - Prosody tokens: {result['prosody_tokens'].shape}")

    # Blending
    result = adapter.blend_emotions([("happy", 0.6), ("surprised", 0.4)], intensity=0.8)
    print(f"  Blended - Prosody tokens: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 9: Description Conversion
    print("\n[Test 9] Description ↔ ADV Conversion...")
    adv = torch.tensor([0.7, 0.5, 0.8])
    desc = adv_to_description(adv)
    print(f"  ADV {adv.tolist()} → '{desc}'")

    desc2 = "energetic, confident, positive"
    adv2 = description_to_adv(desc2)
    print(f"  '{desc2}' → ADV {adv2.tolist()}")
    print("  [PASS]")

    # Test 10: Semi-supervised Loss
    print("\n[Test 10] Semi-supervised Loss...")
    ssl_loss = SemiSupervisedLoss(config)

    discrete_logits = torch.randn(4, config.num_discrete_emotions, device=device)
    predicted_adv = torch.randn(4, 3, device=device).tanh()
    target_emotion = torch.tensor([0, 1, 2, 3], device=device)
    target_adv = torch.randn(4, 3, device=device).tanh()
    emotion_mask = torch.tensor([True, True, False, True], device=device)
    adv_mask = torch.tensor([True, False, True, True], device=device)

    losses = ssl_loss(
        discrete_logits=discrete_logits,
        predicted_adv=predicted_adv,
        target_emotion=target_emotion,
        target_adv=target_adv,
        emotion_mask=emotion_mask,
        adv_mask=adv_mask,
    )
    print(f"  Discrete loss: {losses['discrete'].item():.4f}")
    print(f"  ADV loss: {losses['adv'].item():.4f}")
    print(f"  Consistency loss: {losses['consistency'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All UDDETTS tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from uddetts import (
    UDDETTSConfig,
    UDDETTSAdapter,
    create_uddetts_adapter,
    adv_to_vad,
    vad_to_adv,
    adv_to_description,
)

# Initialize
config = UDDETTSConfig()
adapter = UDDETTSAdapter(config).cuda()

# Option 1: From ADV coordinates (Arousal, Dominance, Valence)
adv = torch.tensor([[0.6, 0.4, 0.8]])  # energetic, confident, positive
result = adapter.from_adv(adv, intensity=0.9)
prosody_tokens = result['prosody_tokens']  # [1, 4, 2048]

# Option 2: From VAD (compatible with SphericalEmotionAdapter)
vad = torch.tensor([[0.8, 0.6, 0.4]])  # (V, A, D)
result = adapter.from_vad(vad)

# Option 3: From discrete emotion
result = adapter.from_emotion("happy", intensity=0.8)

# Option 4: Interpolate between emotions
result = adapter.interpolate_emotions("calm", "angry", t=0.5)

# Option 5: Blend multiple emotions
result = adapter.blend_emotions([
    ("happy", 0.5),
    ("surprised", 0.3),
    ("calm", 0.2),
], intensity=0.85)

# Get description of ADV
desc = adv_to_description(torch.tensor([0.7, 0.5, 0.8]))
# → "very energetic, confident, very positive"

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
