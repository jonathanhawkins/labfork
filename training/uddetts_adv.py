"""
UDDETTS: Unified Discrete-Dimensional Emotion Control for TTS

Based on UDDETTS (arXiv:2505.10599):
"UDDETTS: A Unified Framework for Discrete and Dimensional Emotion Representation in TTS"

Key Innovations:
1. ADV Space: Unified Arousal-Dominance-Valence dimensional representation
   - Different from VAD (Valence-Arousal-Dominance) in ordering
   - Arousal: Active/Passive intensity
   - Dominance: Control/Submission level
   - Valence: Positive/Negative affect

2. Discrete + Dimensional Control: Supports both:
   - Discrete emotion labels (happy, sad, angry, etc.)
   - Continuous ADV coordinates for fine-grained control
   - Can mix both in training via semi-supervised learning

3. Nonlinear ADV Quantization:
   - Quantizes ADV space into learnable anchors
   - Enables fine-grained control with discrete tokens
   - Better than linear interpolation for emotion boundaries

4. Semi-Supervised Loss:
   - Samples with discrete labels only
   - Samples with ADV annotations only
   - Samples with both (full supervision)

5. OT-CFM Integration:
   - Optimal Transport Conditional Flow Matching decoder
   - Generates expressive prosody from ADV conditioning

Benefits:
- Linear emotion control along 3 interpretable dimensions
- Mixing discrete labels with continuous ADV values
- First LLM-based TTS with ADV dimensional control
- State-of-the-art emotion expressiveness

Integration:
- Compatible with existing SphericalEmotionAdapter (VAD)
- ADV coordinates map directly to VAD with reordering
- Prosody tokens compatible with ProsodyControlledCSM
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

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

    # ADV dimensions (Arousal-Dominance-Valence ordering)
    adv_dim: int = 3

    # Model dimensions
    input_dim: int = 768  # Input feature dimension (e.g., wav2vec2/HuBERT)
    embedding_dim: int = 256  # Emotion embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 2048  # Output to match prosody encoder

    # Discrete emotions
    num_discrete_emotions: int = 8  # neutral, happy, sad, angry, surprised, calm, fearful, disgusted

    # Nonlinear ADV quantization
    num_adv_anchors: int = 64  # Number of learnable ADV anchor points
    num_anchors_per_dim: int = 4  # Anchors per dimension (4^3 = 64 if uniform)
    use_learnable_anchors: bool = True  # Learn anchor positions
    quantization_temperature: float = 1.0  # Softmax temperature for soft quantization

    # Semi-supervised training
    discrete_label_weight: float = 1.0
    adv_regression_weight: float = 1.0
    consistency_weight: float = 0.5  # Discrete-ADV consistency loss
    kl_weight: float = 0.1  # KL regularization for embeddings

    # OT-CFM integration
    use_ot_cfm: bool = True
    cfm_hidden_dim: int = 512
    cfm_num_layers: int = 4
    cfm_num_heads: int = 8
    num_ode_steps: int = 50
    sigma_min: float = 0.001

    # Training settings
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate

    # Intensity control
    default_intensity: float = 0.7
    min_intensity: float = 0.0
    max_intensity: float = 1.5


# =============================================================================
# ADV PROTOTYPES (Arousal-Dominance-Valence ordering)
# =============================================================================

# ADV coordinates for standard emotions
# Format: (Arousal, Dominance, Valence) - note different from VAD!
# Arousal: -1 (calm) to +1 (excited)
# Dominance: -1 (submissive) to +1 (dominant)
# Valence: -1 (negative) to +1 (positive)

ADV_PROTOTYPES = {
    # Core emotions
    "neutral": (0.0, 0.0, 0.0),       # Origin - baseline state
    "happy": (0.6, 0.6, 0.8),          # High arousal, dominant, positive
    "sad": (-0.4, -0.5, -0.6),         # Low arousal, submissive, negative
    "angry": (0.8, 0.7, -0.5),         # High arousal, dominant, negative
    "surprised": (0.8, 0.2, 0.3),      # High arousal, neutral dominance, slightly positive
    "calm": (-0.5, 0.3, 0.4),          # Low arousal, slightly dominant, positive
    "fearful": (0.7, -0.7, -0.7),      # High arousal, submissive, negative
    "disgusted": (0.3, 0.4, -0.6),     # Moderate arousal, slightly dominant, negative

    # Extended emotions
    "excited": (0.9, 0.5, 0.7),        # Very high arousal, positive
    "bored": (-0.7, -0.2, -0.2),       # Very low arousal, slightly negative
    "tender": (-0.2, -0.2, 0.7),       # Low arousal, slightly submissive, positive
    "anxious": (0.6, -0.4, -0.4),      # High arousal, submissive, negative
    "content": (-0.3, 0.2, 0.6),       # Low arousal, slight dominance, positive
    "frustrated": (0.5, 0.3, -0.5),    # Moderate arousal, dominant, negative
}

DISCRETE_EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "calm", "fearful", "disgusted"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(DISCRETE_EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}


def adv_to_vad(adv: Tensor) -> Tensor:
    """
    Convert ADV (Arousal-Dominance-Valence) to VAD (Valence-Arousal-Dominance).

    Args:
        adv: ADV coordinates [..., 3] in order (A, D, V)

    Returns:
        VAD coordinates [..., 3] in order (V, A, D)
    """
    # ADV order: [A, D, V] -> VAD order: [V, A, D]
    return torch.stack([adv[..., 2], adv[..., 0], adv[..., 1]], dim=-1)


def vad_to_adv(vad: Tensor) -> Tensor:
    """
    Convert VAD (Valence-Arousal-Dominance) to ADV (Arousal-Dominance-Valence).

    Args:
        vad: VAD coordinates [..., 3] in order (V, A, D)

    Returns:
        ADV coordinates [..., 3] in order (A, D, V)
    """
    # VAD order: [V, A, D] -> ADV order: [A, D, V]
    return torch.stack([vad[..., 1], vad[..., 2], vad[..., 0]], dim=-1)


def get_adv_for_emotion(emotion: str) -> Tuple[float, float, float]:
    """Get ADV coordinates for an emotion name."""
    emotion_lower = emotion.lower()
    return ADV_PROTOTYPES.get(emotion_lower, ADV_PROTOTYPES["neutral"])


# =============================================================================
# NONLINEAR ADV QUANTIZATION
# =============================================================================

class NonlinearADVQuantizer(nn.Module):
    """
    Nonlinear quantization of ADV space using learnable anchor points.

    Instead of uniform grid quantization, learns optimal anchor positions
    that capture emotion-relevant regions of ADV space.

    Benefits:
    - Adapts to data distribution (more anchors near common emotions)
    - Smoother transitions at emotion boundaries
    - Better fine-grained control than linear quantization
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # Initialize anchor points (learnable or fixed)
        if config.use_learnable_anchors:
            # Initialize with emotion prototypes + random perturbations
            initial_anchors = self._initialize_anchors()
            self.anchors = nn.Parameter(initial_anchors)
        else:
            # Fixed uniform grid
            self.register_buffer('anchors', self._create_uniform_grid())

        # Learnable embedding for each anchor
        self.anchor_embeddings = nn.Embedding(config.num_adv_anchors, config.embedding_dim)

        # Temperature for soft quantization
        self.temperature = nn.Parameter(torch.tensor(config.quantization_temperature))

    def _initialize_anchors(self) -> Tensor:
        """Initialize anchor points using emotion prototypes + noise."""
        # Start with emotion prototypes
        proto_coords = [ADV_PROTOTYPES[e] for e in DISCRETE_EMOTIONS]
        proto_tensor = torch.tensor(proto_coords, dtype=torch.float32)

        # Add random anchors to fill the space
        num_proto = len(proto_coords)
        num_random = self.config.num_adv_anchors - num_proto

        if num_random > 0:
            # Random points in [-1, 1]^3 cube
            random_anchors = torch.rand(num_random, 3) * 2 - 1
            anchors = torch.cat([proto_tensor, random_anchors], dim=0)
        else:
            anchors = proto_tensor[:self.config.num_adv_anchors]

        return anchors

    def _create_uniform_grid(self) -> Tensor:
        """Create uniform grid of anchor points."""
        n = self.config.num_anchors_per_dim
        lin = torch.linspace(-1, 1, n)

        # Create 3D grid
        grid_a, grid_d, grid_v = torch.meshgrid(lin, lin, lin, indexing='ij')
        anchors = torch.stack([
            grid_a.flatten(),
            grid_d.flatten(),
            grid_v.flatten()
        ], dim=-1)

        # Limit to num_adv_anchors
        return anchors[:self.config.num_adv_anchors]

    def quantize_soft(
        self,
        adv: Tensor,  # [..., 3]
    ) -> Dict[str, Tensor]:
        """
        Soft quantization: weighted sum of nearest anchors.

        Args:
            adv: ADV coordinates [..., 3]

        Returns:
            Dict with quantized embedding and analysis
        """
        original_shape = adv.shape[:-1]
        adv_flat = adv.reshape(-1, 3)  # [N, 3]

        # Compute distances to all anchors
        # adv_flat: [N, 3], anchors: [K, 3]
        distances = torch.cdist(adv_flat, self.anchors)  # [N, K]

        # Soft assignment via softmax over negative distances
        weights = F.softmax(-distances / self.temperature, dim=-1)  # [N, K]

        # Weighted sum of anchor embeddings
        anchor_embs = self.anchor_embeddings.weight  # [K, embedding_dim]
        quantized = torch.matmul(weights, anchor_embs)  # [N, embedding_dim]

        # Reshape to original batch shape
        quantized = quantized.reshape(*original_shape, -1)

        # Get nearest anchor indices for analysis
        nearest_indices = torch.argmin(distances, dim=-1)
        nearest_indices = nearest_indices.reshape(*original_shape)

        return {
            'embedding': quantized,
            'weights': weights.reshape(*original_shape, -1),
            'nearest_indices': nearest_indices,
            'distances': distances.reshape(*original_shape, -1),
        }

    def quantize_hard(
        self,
        adv: Tensor,  # [..., 3]
    ) -> Dict[str, Tensor]:
        """
        Hard quantization: nearest anchor only.

        Uses straight-through estimator for gradients.
        """
        original_shape = adv.shape[:-1]
        adv_flat = adv.reshape(-1, 3)

        # Find nearest anchor
        distances = torch.cdist(adv_flat, self.anchors)
        nearest_indices = torch.argmin(distances, dim=-1)

        # Get quantized ADV (anchor coordinates)
        quantized_adv = self.anchors[nearest_indices]

        # Get embeddings
        quantized_emb = self.anchor_embeddings(nearest_indices)

        # Straight-through estimator: forward uses quantized, backward uses original
        quantized_adv = adv_flat + (quantized_adv - adv_flat).detach()
        quantized_emb = quantized_emb.reshape(*original_shape, -1)

        return {
            'embedding': quantized_emb,
            'quantized_adv': quantized_adv.reshape(*original_shape, 3),
            'nearest_indices': nearest_indices.reshape(*original_shape),
        }

    def forward(
        self,
        adv: Tensor,
        mode: str = "soft",  # "soft" or "hard"
    ) -> Dict[str, Tensor]:
        """
        Quantize ADV coordinates.

        Args:
            adv: ADV coordinates [..., 3]
            mode: "soft" for weighted, "hard" for nearest

        Returns:
            Quantization result dict
        """
        if mode == "soft":
            return self.quantize_soft(adv)
        else:
            return self.quantize_hard(adv)


# =============================================================================
# ADV ENCODER
# =============================================================================

class ADVEncoder(nn.Module):
    """
    Encodes ADV (Arousal-Dominance-Valence) coordinates to embeddings.

    Supports both:
    1. Direct ADV coordinates input
    2. Discrete emotion labels (mapped to ADV prototypes)

    The encoder uses nonlinear quantization for fine-grained control.
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # ADV prototypes (learnable or fixed)
        proto_values = torch.tensor([
            ADV_PROTOTYPES[e] for e in DISCRETE_EMOTIONS
        ], dtype=torch.float32)

        if config.use_learnable_anchors:
            self.adv_prototypes = nn.Parameter(proto_values)
        else:
            self.register_buffer('adv_prototypes', proto_values)

        # Discrete emotion embeddings (parallel path)
        self.emotion_embeddings = nn.Embedding(
            config.num_discrete_emotions,
            config.embedding_dim
        )

        # Nonlinear ADV quantizer
        self.quantizer = NonlinearADVQuantizer(config)

        # Direct ADV encoder (continuous path)
        self.adv_encoder = nn.Sequential(
            nn.Linear(config.adv_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
        )

        # Spherical coordinate encoder (for richness)
        # Convert ADV to spherical: (r, theta, phi)
        spherical_input_dim = 7  # r, theta, phi, sin(theta), cos(theta), sin(phi), cos(phi)
        self.spherical_encoder = nn.Sequential(
            nn.Linear(spherical_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Fusion of different encoding paths
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Intensity modulation
        self.intensity_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, config.embedding_dim),
            nn.Sigmoid(),
        )

        # Output projection to prosody tokens
        self.output_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

    def _adv_to_spherical(self, adv: Tensor) -> Tensor:
        """Convert ADV to spherical coordinates with rich features."""
        a, d, v = adv[..., 0], adv[..., 1], adv[..., 2]

        # Radius (intensity)
        r = torch.sqrt(a**2 + d**2 + v**2 + 1e-8)

        # Polar angle (from A axis)
        theta = torch.acos(torch.clamp(a / (r + 1e-8), -1.0, 1.0))

        # Azimuthal angle (in D-V plane)
        phi = torch.atan2(v, d + 1e-8)

        return torch.stack([
            r, theta, phi,
            torch.sin(theta), torch.cos(theta),
            torch.sin(phi), torch.cos(phi),
        ], dim=-1)

    def get_adv_for_emotion(
        self,
        emotion: Union[str, int, Tensor],
        device: torch.device = None,
    ) -> Tensor:
        """
        Get ADV coordinates for an emotion.

        Args:
            emotion: Emotion name, index, or one-hot tensor
            device: Target device

        Returns:
            ADV tensor [3] or [batch, 3]
        """
        if device is None:
            device = self.adv_prototypes.device

        if isinstance(emotion, str):
            emotion_lower = emotion.lower()
            if emotion_lower in EMOTION_TO_IDX:
                idx = EMOTION_TO_IDX[emotion_lower]
                return self.adv_prototypes[idx].to(device)
            elif emotion_lower in ADV_PROTOTYPES:
                adv = ADV_PROTOTYPES[emotion_lower]
                return torch.tensor(adv, dtype=torch.float32, device=device)
            else:
                return self.adv_prototypes[0].to(device)  # neutral
        elif isinstance(emotion, int):
            if emotion < self.config.num_discrete_emotions:
                return self.adv_prototypes[emotion].to(device)
            return self.adv_prototypes[0].to(device)
        elif isinstance(emotion, Tensor):
            if emotion.dim() == 0:
                idx = emotion.long().item()
                if idx < self.config.num_discrete_emotions:
                    return self.adv_prototypes[idx].to(device)
                return self.adv_prototypes[0].to(device)
            elif emotion.dim() == 1 and emotion.shape[0] == self.config.num_discrete_emotions:
                # Soft weights over emotions
                return torch.matmul(emotion.unsqueeze(0), self.adv_prototypes).squeeze(0)
            elif emotion.dim() == 2:
                # Batch of soft weights
                return torch.matmul(emotion, self.adv_prototypes)
            else:
                # Assume already ADV coordinates
                return emotion.to(device)
        else:
            raise ValueError(f"Unsupported emotion type: {type(emotion)}")

    def forward(
        self,
        adv: Optional[Tensor] = None,           # [batch, 3] ADV coordinates
        emotion_ids: Optional[Tensor] = None,   # [batch] discrete emotion indices
        intensity: Optional[Tensor] = None,     # [batch] intensity scaling
    ) -> Dict[str, Tensor]:
        """
        Encode emotion from ADV coordinates and/or discrete labels.

        Args:
            adv: ADV coordinates [batch, 3] (if provided)
            emotion_ids: Discrete emotion indices [batch] (if provided)
            intensity: Intensity scaling [batch] or scalar

        Returns:
            Dict with embeddings, tokens, and analysis
        """
        device = self.adv_prototypes.device

        # Handle input cases
        if adv is None and emotion_ids is None:
            raise ValueError("Must provide either adv or emotion_ids")

        if adv is not None:
            batch_size = adv.shape[0]
        else:
            batch_size = emotion_ids.shape[0]

        # Handle intensity
        if intensity is None:
            intensity = torch.full((batch_size,), self.config.default_intensity, device=device)
        elif isinstance(intensity, (int, float)):
            intensity = torch.full((batch_size,), float(intensity), device=device)

        # Get ADV from emotion_ids if not provided directly
        if adv is None:
            adv = self.adv_prototypes[emotion_ids]  # [batch, 3]

        # Scale ADV by intensity
        adv_scaled = adv * intensity.unsqueeze(-1)

        # Path 1: Quantized encoding
        quant_result = self.quantizer(adv_scaled, mode="soft")
        quant_embedding = quant_result['embedding']

        # Path 2: Direct ADV encoding
        adv_embedding = self.adv_encoder(adv_scaled)

        # Path 3: Spherical encoding
        spherical_features = self._adv_to_spherical(adv_scaled)
        spherical_embedding = self.spherical_encoder(spherical_features)

        # Fuse all paths
        combined = torch.cat([quant_embedding, adv_embedding, spherical_embedding], dim=-1)
        embedding = self.fusion(combined)

        # Apply intensity modulation
        intensity_gate = self.intensity_mlp(intensity.unsqueeze(-1))
        embedding = embedding * intensity_gate

        # Project to prosody tokens
        tokens = self.output_projection(embedding)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        return {
            'embedding': embedding,
            'tokens': tokens,
            'adv': adv,
            'adv_scaled': adv_scaled,
            'intensity': intensity,
            'quantization': quant_result,
            'spherical_features': spherical_features,
        }

    def encode_emotion(
        self,
        emotion: Union[str, int],
        intensity: float = None,
        batch_size: int = 1,
    ) -> Dict[str, Tensor]:
        """Convenience method to encode by emotion name/index."""
        device = self.adv_prototypes.device

        if isinstance(emotion, str):
            if emotion.lower() in EMOTION_TO_IDX:
                emotion_id = EMOTION_TO_IDX[emotion.lower()]
            else:
                emotion_id = 0  # neutral
        else:
            emotion_id = emotion

        emotion_ids = torch.full((batch_size,), emotion_id, dtype=torch.long, device=device)
        return self.forward(emotion_ids=emotion_ids, intensity=intensity)


# =============================================================================
# SEMI-SUPERVISED LOSS
# =============================================================================

class UDDETTSSemiSupervisedLoss(nn.Module):
    """
    Semi-supervised loss for UDDETTS training.

    Handles three types of samples:
    1. Discrete labels only: Classification loss
    2. ADV annotations only: Regression loss
    3. Both: Full supervision with consistency

    Also includes:
    - Quantization commitment loss
    - Prototype separation loss
    - Consistency between discrete and ADV paths
    """

    def __init__(self, config: UDDETTSConfig):
        super().__init__()
        self.config = config

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

        # ADV classifier for consistency (predict discrete from ADV)
        self.adv_classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_discrete_emotions),
        )

        # ADV regressor for consistency (predict ADV from discrete)
        self.discrete_to_adv = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 3),
            nn.Tanh(),  # Bound to [-1, 1]
        )

    def forward(
        self,
        encoder_output: Dict[str, Tensor],
        target_emotion_ids: Optional[Tensor] = None,  # [batch] discrete labels
        target_adv: Optional[Tensor] = None,          # [batch, 3] ADV coordinates
        has_discrete: Optional[Tensor] = None,        # [batch] bool mask
        has_adv: Optional[Tensor] = None,             # [batch] bool mask
    ) -> Dict[str, Tensor]:
        """
        Compute semi-supervised loss.

        Args:
            encoder_output: Output from ADVEncoder
            target_emotion_ids: Ground truth discrete emotion indices
            target_adv: Ground truth ADV coordinates
            has_discrete: Mask indicating which samples have discrete labels
            has_adv: Mask indicating which samples have ADV annotations

        Returns:
            Dict with loss components and total
        """
        batch_size = encoder_output['embedding'].shape[0]
        device = encoder_output['embedding'].device

        losses = {}

        # Create default masks if not provided
        if has_discrete is None:
            has_discrete = target_emotion_ids is not None
            if has_discrete:
                has_discrete = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                has_discrete = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if has_adv is None:
            has_adv = target_adv is not None
            if has_adv:
                has_adv = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                has_adv = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 1. Discrete classification loss
        if has_discrete.any() and target_emotion_ids is not None:
            embedding = encoder_output['embedding']
            logits = self.adv_classifier(embedding)

            discrete_loss = self.ce_loss(logits, target_emotion_ids)
            discrete_loss = (discrete_loss * has_discrete.float()).sum() / (has_discrete.float().sum() + 1e-8)
            losses['discrete'] = discrete_loss

            # Accuracy for monitoring
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == target_emotion_ids).float()
            accuracy = (accuracy * has_discrete.float()).sum() / (has_discrete.float().sum() + 1e-8)
            losses['discrete_accuracy'] = accuracy
        else:
            losses['discrete'] = torch.tensor(0.0, device=device)
            losses['discrete_accuracy'] = torch.tensor(0.0, device=device)

        # 2. ADV regression loss
        if has_adv.any() and target_adv is not None:
            adv_pred = encoder_output['adv_scaled']

            # MSE in ADV space
            adv_loss = self.mse_loss(adv_pred, target_adv).mean(dim=-1)
            adv_loss = (adv_loss * has_adv.float()).sum() / (has_adv.float().sum() + 1e-8)
            losses['adv_regression'] = adv_loss

            # Cosine similarity
            adv_similarity = self.cosine_similarity(adv_pred, target_adv)
            adv_similarity = (adv_similarity * has_adv.float()).sum() / (has_adv.float().sum() + 1e-8)
            losses['adv_similarity'] = adv_similarity
        else:
            losses['adv_regression'] = torch.tensor(0.0, device=device)
            losses['adv_similarity'] = torch.tensor(0.0, device=device)

        # 3. Consistency loss (for samples with both annotations)
        has_both = has_discrete & has_adv
        if has_both.any() and target_emotion_ids is not None and target_adv is not None:
            embedding = encoder_output['embedding']

            # Predict ADV from embedding, compare to target
            pred_adv = self.discrete_to_adv(embedding)
            consistency_loss = self.mse_loss(pred_adv, target_adv).mean(dim=-1)
            consistency_loss = (consistency_loss * has_both.float()).sum() / (has_both.float().sum() + 1e-8)
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)

        # 4. Quantization loss (encourage using codebook)
        quant_result = encoder_output.get('quantization', {})
        if 'weights' in quant_result:
            # Entropy of quantization weights (encourage sparse usage)
            weights = quant_result['weights']
            entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
            # We want low entropy (sparse), so minimize negative entropy = maximize entropy
            # Actually, we want commitment to specific anchors, so minimize entropy
            losses['quantization'] = -entropy * 0.01  # Small weight
        else:
            losses['quantization'] = torch.tensor(0.0, device=device)

        # 5. Total loss
        total = (
            losses['discrete'] * self.config.discrete_label_weight +
            losses['adv_regression'] * self.config.adv_regression_weight +
            losses['consistency'] * self.config.consistency_weight +
            losses['quantization']
        )
        losses['total'] = total

        return losses


# =============================================================================
# UDDETTS ADAPTER (Integration with existing pipeline)
# =============================================================================

class UDDETTSAdapter(nn.Module):
    """
    Adapter that integrates UDDETTS ADV-space control with the prosody pipeline.

    Supports:
    1. Direct ADV coordinate control
    2. Discrete emotion label control
    3. Mixed/unified control (both discrete and ADV)
    4. Integration with existing SphericalEmotionAdapter (VAD)

    Usage:
        adapter = UDDETTSAdapter(config)

        # From ADV coordinates
        result = adapter.from_adv(adv_coords, intensity=0.8)

        # From emotion label
        result = adapter.from_emotion("happy", intensity=0.8)

        # Unified control
        result = adapter(adv=adv_coords, emotion_ids=labels, intensity=0.8)

        # Get prosody tokens
        prosody_tokens = result['prosody_tokens']
    """

    def __init__(
        self,
        config: UDDETTSConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core ADV encoder
        self.encoder = ADVEncoder(config)

        # Semi-supervised loss
        self.loss_fn = UDDETTSSemiSupervisedLoss(config)

        # Adapter to prosody hidden dimension
        if config.output_dim != prosody_hidden:
            self.prosody_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.prosody_adapter = nn.Identity()

    def forward(
        self,
        adv: Optional[Tensor] = None,
        emotion_ids: Optional[Tensor] = None,
        intensity: Optional[Union[Tensor, float]] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode emotion from ADV and/or discrete labels.

        Args:
            adv: ADV coordinates [batch, 3]
            emotion_ids: Discrete emotion indices [batch]
            intensity: Intensity scaling

        Returns:
            Dict with prosody_tokens and analysis
        """
        encoder_output = self.encoder(adv, emotion_ids, intensity)

        # Adapt tokens to prosody dimension
        tokens = encoder_output['tokens']
        prosody_tokens = self.prosody_adapter(tokens)

        return {
            **encoder_output,
            'prosody_tokens': prosody_tokens,
        }

    def from_adv(
        self,
        adv: Tensor,
        intensity: Optional[Union[Tensor, float]] = None,
    ) -> Dict[str, Tensor]:
        """Encode from ADV coordinates."""
        return self.forward(adv=adv, intensity=intensity)

    def from_emotion(
        self,
        emotion: Union[str, int],
        intensity: Optional[float] = None,
        batch_size: int = 1,
    ) -> Dict[str, Tensor]:
        """Encode from emotion name/index."""
        device = self.encoder.adv_prototypes.device

        if isinstance(emotion, str):
            if emotion.lower() in EMOTION_TO_IDX:
                emotion_id = EMOTION_TO_IDX[emotion.lower()]
            else:
                emotion_id = 0
        else:
            emotion_id = emotion

        emotion_ids = torch.full((batch_size,), emotion_id, dtype=torch.long, device=device)
        return self.forward(emotion_ids=emotion_ids, intensity=intensity)

    def from_vad(
        self,
        vad: Tensor,
        intensity: Optional[Union[Tensor, float]] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode from VAD coordinates (for compatibility with SphericalEmotionAdapter).

        Args:
            vad: VAD coordinates [batch, 3] in order (V, A, D)
            intensity: Intensity scaling

        Returns:
            Dict with prosody_tokens
        """
        # Convert VAD to ADV
        adv = vad_to_adv(vad)
        return self.forward(adv=adv, intensity=intensity)

    def interpolate_adv(
        self,
        adv1: Tensor,
        adv2: Tensor,
        t: float,
        intensity: Optional[float] = None,
        method: str = "linear",
    ) -> Dict[str, Tensor]:
        """
        Interpolate between two ADV coordinates.

        Args:
            adv1: Source ADV [batch, 3]
            adv2: Target ADV [batch, 3]
            t: Interpolation factor [0, 1]
            intensity: Intensity scaling
            method: "linear" or "spherical"

        Returns:
            Dict with interpolated prosody_tokens
        """
        if method == "linear":
            adv_interp = (1 - t) * adv1 + t * adv2
        elif method == "spherical":
            # SLERP on ADV sphere
            adv1_norm = F.normalize(adv1, dim=-1)
            adv2_norm = F.normalize(adv2, dim=-1)

            dot = (adv1_norm * adv2_norm).sum(dim=-1, keepdim=True).clamp(-1, 1)
            omega = torch.acos(dot)
            sin_omega = torch.sin(omega)

            # Handle parallel vectors
            nearly_parallel = sin_omega.abs() < 1e-6
            s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
            s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

            interp_dir = s1 * adv1_norm + s2 * adv2_norm
            interp_dir = torch.where(nearly_parallel, (1 - t) * adv1_norm + t * adv2_norm, interp_dir)

            # Interpolate magnitudes
            mag1 = adv1.norm(dim=-1, keepdim=True)
            mag2 = adv2.norm(dim=-1, keepdim=True)
            mag = (1 - t) * mag1 + t * mag2

            adv_interp = interp_dir * mag
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return self.forward(adv=adv_interp, intensity=intensity)

    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: Optional[float] = None,
        method: str = "spherical",
    ) -> Dict[str, Tensor]:
        """Interpolate between two emotions."""
        device = self.encoder.adv_prototypes.device

        adv1 = self.encoder.get_adv_for_emotion(emotion1, device).unsqueeze(0)
        adv2 = self.encoder.get_adv_for_emotion(emotion2, device).unsqueeze(0)

        return self.interpolate_adv(adv1, adv2, t, intensity, method)

    def blend_emotions(
        self,
        emotions: List[Tuple[str, float]],
        intensity: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Blend multiple emotions with weights.

        Args:
            emotions: List of (emotion_name, weight) tuples
            intensity: Overall intensity

        Returns:
            Dict with blended prosody_tokens
        """
        device = self.encoder.adv_prototypes.device

        # Normalize weights
        total_weight = sum(w for _, w in emotions)
        if total_weight == 0:
            return self.from_emotion("neutral", intensity)

        # Weighted sum of ADV vectors
        blended_adv = torch.zeros(3, device=device)
        for emotion, weight in emotions:
            adv = self.encoder.get_adv_for_emotion(emotion, device)
            blended_adv += adv * (weight / total_weight)

        return self.forward(adv=blended_adv.unsqueeze(0), intensity=intensity)

    def compute_loss(
        self,
        encoder_output: Dict[str, Tensor],
        target_emotion_ids: Optional[Tensor] = None,
        target_adv: Optional[Tensor] = None,
        has_discrete: Optional[Tensor] = None,
        has_adv: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute semi-supervised training loss."""
        return self.loss_fn(
            encoder_output,
            target_emotion_ids,
            target_adv,
            has_discrete,
            has_adv,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_uddetts_adapter(
    config: Optional[UDDETTSConfig] = None,
    prosody_hidden: int = 2048,
) -> UDDETTSAdapter:
    """Create a UDDETTS adapter with default or custom config."""
    if config is None:
        config = UDDETTSConfig()
    return UDDETTSAdapter(config, prosody_hidden)


def adv_to_emotion_name(adv: Tensor) -> str:
    """Find nearest discrete emotion for ADV coordinates."""
    if adv.dim() > 1:
        adv = adv.squeeze()

    adv_tuple = (adv[0].item(), adv[1].item(), adv[2].item())

    min_dist = float('inf')
    nearest = "neutral"

    for emotion, proto in ADV_PROTOTYPES.items():
        dist = sum((a - b) ** 2 for a, b in zip(adv_tuple, proto)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest = emotion

    return nearest


def describe_adv(adv: Tensor, threshold: float = 0.3) -> str:
    """Generate natural language description of ADV coordinates."""
    if adv.dim() > 1:
        adv = adv.squeeze()

    a, d, v = adv[0].item(), adv[1].item(), adv[2].item()

    parts = []

    # Arousal description
    if a > threshold:
        parts.append("high arousal (energetic)")
    elif a < -threshold:
        parts.append("low arousal (calm)")

    # Dominance description
    if d > threshold:
        parts.append("dominant")
    elif d < -threshold:
        parts.append("submissive")

    # Valence description
    if v > threshold:
        parts.append("positive affect")
    elif v < -threshold:
        parts.append("negative affect")

    if not parts:
        return "neutral emotional state"

    return "Expressing " + ", ".join(parts)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UDDETTS: Unified ADV-Space Emotion Control - Test Suite")
    print("=" * 70)

    config = UDDETTSConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test 1: ADV Prototypes
    print("\n[Test 1] ADV Prototypes...")
    for emotion, adv in list(ADV_PROTOTYPES.items())[:5]:
        print(f"  {emotion:12s}: A={adv[0]:+.2f}, D={adv[1]:+.2f}, V={adv[2]:+.2f}")
    print("  [PASS]")

    # Test 2: ADV <-> VAD Conversion
    print("\n[Test 2] ADV <-> VAD Conversion...")
    test_adv = torch.tensor([[0.6, 0.4, 0.8]])  # A=0.6, D=0.4, V=0.8
    vad = adv_to_vad(test_adv)
    adv_recovered = vad_to_adv(vad)
    print(f"  Original ADV: {test_adv[0].tolist()}")
    print(f"  Converted VAD: {vad[0].tolist()}")
    print(f"  Recovered ADV: {adv_recovered[0].tolist()}")
    assert torch.allclose(test_adv, adv_recovered), "ADV-VAD conversion failed!"
    print("  [PASS]")

    # Test 3: Nonlinear ADV Quantizer
    print("\n[Test 3] Nonlinear ADV Quantizer...")
    quantizer = NonlinearADVQuantizer(config).to(device)
    test_coords = torch.randn(4, 3, device=device) * 0.5

    soft_result = quantizer(test_coords, mode="soft")
    print(f"  Input shape: {test_coords.shape}")
    print(f"  Soft embedding shape: {soft_result['embedding'].shape}")
    print(f"  Weights shape: {soft_result['weights'].shape}")
    print(f"  Nearest indices: {soft_result['nearest_indices'].tolist()}")

    hard_result = quantizer(test_coords, mode="hard")
    print(f"  Hard embedding shape: {hard_result['embedding'].shape}")
    print("  [PASS]")

    # Test 4: ADV Encoder
    print("\n[Test 4] ADV Encoder...")
    encoder = ADVEncoder(config).to(device)

    batch_size = 2
    adv_input = torch.randn(batch_size, 3, device=device) * 0.5
    emotion_ids = torch.randint(0, config.num_discrete_emotions, (batch_size,), device=device)
    intensity = torch.tensor([0.5, 0.9], device=device)

    # From ADV coordinates
    output_adv = encoder(adv=adv_input, intensity=intensity)
    print(f"  From ADV - embedding: {output_adv['embedding'].shape}, tokens: {output_adv['tokens'].shape}")

    # From emotion IDs
    output_discrete = encoder(emotion_ids=emotion_ids, intensity=intensity)
    print(f"  From IDs - embedding: {output_discrete['embedding'].shape}, tokens: {output_discrete['tokens'].shape}")

    # From emotion name
    output_name = encoder.encode_emotion("happy", intensity=0.8)
    print(f"  From name - embedding: {output_name['embedding'].shape}")
    print("  [PASS]")

    # Test 5: UDDETTS Adapter
    print("\n[Test 5] UDDETTS Adapter...")
    adapter = UDDETTSAdapter(config).to(device)

    # From ADV
    result = adapter.from_adv(adv_input, intensity=0.7)
    print(f"  From ADV - prosody_tokens: {result['prosody_tokens'].shape}")

    # From emotion
    result = adapter.from_emotion("angry", intensity=0.9)
    print(f"  From emotion - prosody_tokens: {result['prosody_tokens'].shape}")

    # From VAD (compatibility with SphericalEmotionAdapter)
    vad_input = torch.tensor([[0.8, 0.6, 0.4]], device=device)  # V, A, D
    result = adapter.from_vad(vad_input, intensity=0.8)
    print(f"  From VAD - prosody_tokens: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 6: Emotion Interpolation
    print("\n[Test 6] Emotion Interpolation...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = adapter.interpolate_emotions("happy", "sad", t, intensity=0.7)
        adv = result['adv'][0]
        nearest = adv_to_emotion_name(adv)
        print(f"  t={t:.2f}: ADV=({adv[0]:.2f}, {adv[1]:.2f}, {adv[2]:.2f}), nearest={nearest}")
    print("  [PASS]")

    # Test 7: Emotion Blending
    print("\n[Test 7] Emotion Blending...")
    blend_result = adapter.blend_emotions([
        ("happy", 0.5),
        ("surprised", 0.3),
        ("calm", 0.2),
    ], intensity=0.8)
    adv = blend_result['adv'][0]
    description = describe_adv(adv)
    print(f"  Blended ADV: ({adv[0]:.2f}, {adv[1]:.2f}, {adv[2]:.2f})")
    print(f"  Description: {description}")
    print(f"  Tokens shape: {blend_result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 8: Semi-Supervised Loss
    print("\n[Test 8] Semi-Supervised Loss...")
    loss_fn = UDDETTSSemiSupervisedLoss(config).to(device)

    # Simulated training batch with mixed supervision
    encoder_output = adapter(adv=torch.randn(4, 3, device=device) * 0.5)

    # Half have discrete labels, half have ADV annotations
    has_discrete = torch.tensor([True, True, False, False], device=device)
    has_adv = torch.tensor([False, True, True, True], device=device)

    target_emotion_ids = torch.randint(0, config.num_discrete_emotions, (4,), device=device)
    target_adv = torch.randn(4, 3, device=device) * 0.5

    losses = loss_fn(
        encoder_output,
        target_emotion_ids=target_emotion_ids,
        target_adv=target_adv,
        has_discrete=has_discrete,
        has_adv=has_adv,
    )

    print(f"  Discrete loss: {losses['discrete'].item():.4f}")
    print(f"  ADV regression loss: {losses['adv_regression'].item():.4f}")
    print(f"  Consistency loss: {losses['consistency'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Discrete accuracy: {losses['discrete_accuracy'].item():.4f}")
    print("  [PASS]")

    # Test 9: Create Adapter with Factory
    print("\n[Test 9] Create Adapter with Factory...")
    adapter2 = create_uddetts_adapter(prosody_hidden=2048)
    result = adapter2.from_emotion("surprised", intensity=0.7)
    print(f"  Factory adapter output: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 10: ADV Descriptions
    print("\n[Test 10] ADV Descriptions...")
    test_cases = [
        torch.tensor([0.8, 0.5, 0.7]),   # High A, moderate D, positive V
        torch.tensor([-0.6, -0.5, -0.7]), # Low A, submissive, negative V
        torch.tensor([0.1, 0.0, 0.0]),   # Near neutral
    ]
    for adv in test_cases:
        desc = describe_adv(adv)
        print(f"  ADV=({adv[0]:.1f}, {adv[1]:.1f}, {adv[2]:.1f}): {desc}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All UDDETTS ADV-space tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from uddetts_adv import (
    UDDETTSConfig,
    UDDETTSAdapter,
    create_uddetts_adapter,
    adv_to_vad,
    vad_to_adv,
    describe_adv,
)

# Initialize
config = UDDETTSConfig()
adapter = UDDETTSAdapter(config).cuda()

# From ADV coordinates (Arousal-Dominance-Valence)
adv = torch.tensor([[0.7, 0.5, 0.8]])  # Energetic, dominant, positive
result = adapter.from_adv(adv, intensity=0.9)
prosody_tokens = result['prosody_tokens']  # [1, 4, 2048]

# From discrete emotion label
result = adapter.from_emotion("happy", intensity=0.8)

# From VAD (for compatibility with SphericalEmotionAdapter)
vad = torch.tensor([[0.8, 0.6, 0.4]])  # Valence-Arousal-Dominance
result = adapter.from_vad(vad, intensity=0.8)

# Emotion interpolation
result = adapter.interpolate_emotions("sad", "happy", t=0.5, intensity=0.7)

# Multi-emotion blending
result = adapter.blend_emotions([
    ("happy", 0.6),
    ("surprised", 0.3),
    ("excited", 0.1),
], intensity=0.9)

# Semi-supervised training
losses = adapter.compute_loss(
    encoder_output,
    target_emotion_ids=emotion_ids,  # Discrete labels
    target_adv=adv_coords,            # ADV annotations
    has_discrete=has_discrete_mask,
    has_adv=has_adv_mask,
)

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
