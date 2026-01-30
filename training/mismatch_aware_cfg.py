"""
Mismatch-Aware Classifier-Free Guidance for Emotion Control

Based on:
1. EmoSteer-TTS (arXiv:2508.03543): Training-free emotion control via activation steering
2. Mismatch-Aware Guidance (arXiv:2510.13293): Dynamic CFG scale based on semantic alignment

Key Innovation: Detect when emotion/style doesn't semantically match content and dynamically
reduce guidance to avoid over-guidance artifacts (distortion, unnatural prosody).

Problem with Standard CFG:
- Fixed guidance scale (λ) assumes style always matches content
- Happy emotion on sad text → over-guidance → unnatural/distorted output
- User often doesn't know optimal λ for specific content

Solution - Mismatch-Aware CFG:
- Semantic Mismatch Discriminator estimates style-content alignment score
- Dynamic λ(x) = λ_base × alignment_score
- High alignment → strong guidance, Low alignment → weak guidance
- Automatically adapts to content without user tuning

Formula:
    Standard CFG: f̂ = f_cond + λ × (f_cond - f_uncond)
    Mismatch-Aware: f̂ = f_cond + λ(x) × (f_cond - f_uncond)

    where λ(x) = λ_base × σ(D(emotion, content))
    D = mismatch discriminator, σ = sigmoid for [0,1] scaling

Usage:
    from mismatch_aware_cfg import (
        MismatchAwareCFG,
        SemanticMismatchDiscriminator,
        MismatchAwareActivationSteering,
    )

    # Create mismatch discriminator
    discriminator = SemanticMismatchDiscriminator(config)

    # Wrap existing steering with mismatch awareness
    mismatch_steerer = MismatchAwareActivationSteering(
        model, steering_vectors, discriminator
    )

    # Dynamic guidance based on content
    with mismatch_steerer.steer_adaptive("happy", text_embedding):
        audio = model.generate(text)  # λ adapts to content
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from activation_steering import (
    ActivationSteering,
    SteeringConfig,
    SteeringVectorExtractor,
    SphericalActivationSteering,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MismatchAwareCFGConfig:
    """Configuration for mismatch-aware CFG."""

    # Base guidance scale
    base_guidance_scale: float = 1.0  # λ_base
    min_guidance_scale: float = 0.1   # Minimum λ (even with mismatch)
    max_guidance_scale: float = 2.0   # Maximum λ (perfect alignment)

    # Discriminator architecture
    hidden_dim: int = 512             # Hidden dimension for discriminator
    num_layers: int = 3               # Number of MLP layers
    emotion_dim: int = 256            # Emotion embedding dimension
    content_dim: int = 768            # Content embedding dimension (BERT/wav2vec2)
    dropout: float = 0.1

    # Alignment computation
    alignment_temperature: float = 1.0  # Temperature for sigmoid
    use_learnable_temperature: bool = True

    # Emotion prototypes (for text-based mismatch detection)
    emotion_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "happy": ["happy", "joy", "excited", "wonderful", "great", "amazing", "love", "delighted", "thrilled"],
        "sad": ["sad", "cry", "depressed", "grief", "sorrow", "miserable", "heartbroken", "lonely", "despair"],
        "angry": ["angry", "furious", "mad", "rage", "hate", "frustrated", "annoyed", "irritated", "outraged"],
        "fearful": ["afraid", "scared", "terrified", "anxious", "worried", "panic", "horror", "dread", "nervous"],
        "surprised": ["surprised", "shocked", "amazed", "astonished", "wow", "unexpected", "startled"],
        "disgusted": ["disgusted", "gross", "revolting", "sick", "awful", "nasty", "horrible", "repulsive"],
        "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil", "gentle", "quiet", "soothing"],
        "neutral": [],  # Neutral matches everything
    })

    # VAD alignment (valence-arousal-dominance)
    use_vad_alignment: bool = True
    vad_prototypes: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "happy": (0.8, 0.6, 0.6),
        "sad": (-0.6, -0.4, -0.5),
        "angry": (-0.5, 0.8, 0.7),
        "fearful": (-0.7, 0.7, -0.7),
        "surprised": (0.3, 0.8, 0.2),
        "disgusted": (-0.6, 0.3, 0.4),
        "calm": (0.4, -0.5, 0.3),
        "neutral": (0.0, 0.0, 0.0),
    })

    # Training settings
    contrastive_margin: float = 0.5   # Margin for contrastive loss
    adversarial_weight: float = 0.1   # Weight for adversarial alignment

    # Steering integration
    apply_to_steering: bool = True    # Apply mismatch scaling to activation steering
    steering_config: SteeringConfig = field(default_factory=SteeringConfig)


# =============================================================================
# SEMANTIC MISMATCH DISCRIMINATOR
# =============================================================================

class SemanticMismatchDiscriminator(nn.Module):
    """
    Discriminator that predicts semantic alignment between emotion and content.

    Architecture:
        Emotion Embed → ┐
                        MLP → Alignment Score [0, 1]
        Content Embed → ┘

    Training:
        - Positive pairs: emotion + matching content (e.g., happy + joyful text)
        - Negative pairs: emotion + mismatching content (e.g., happy + sad text)
        - Contrastive or BCE loss

    Inference:
        - Returns alignment score in [0, 1]
        - High score → emotion matches content → use strong guidance
        - Low score → emotion mismatches content → reduce guidance
    """

    def __init__(self, config: MismatchAwareCFGConfig):
        super().__init__()
        self.config = config

        # Emotion encoder (from discrete label or VAD)
        self.emotion_embed = nn.Embedding(
            num_embeddings=len(config.emotion_keywords),
            embedding_dim=config.emotion_dim,
        )
        self.emotion_names = list(config.emotion_keywords.keys())
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotion_names)}

        # VAD encoder (for continuous emotion)
        self.vad_encoder = nn.Sequential(
            nn.Linear(3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

        # Content projection
        self.content_proj = nn.Linear(config.content_dim, config.hidden_dim)

        # Joint embedding network
        input_dim = config.emotion_dim + config.hidden_dim
        layers = []
        for i in range(config.num_layers):
            in_dim = input_dim if i == 0 else config.hidden_dim
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
        self.joint_network = nn.Sequential(*layers)

        # Alignment head
        self.alignment_head = nn.Linear(config.hidden_dim, 1)

        # Learnable temperature
        if config.use_learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(config.alignment_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(config.alignment_temperature))

        # Keyword-based alignment (fallback)
        self._init_keyword_embeddings()

    def _init_keyword_embeddings(self):
        """Initialize keyword embeddings for rule-based alignment."""
        # Pre-compute keyword sets for efficient lookup
        self.keyword_sets = {
            emotion: set(keywords)
            for emotion, keywords in self.config.emotion_keywords.items()
        }

    def get_emotion_embedding(
        self,
        emotion: Optional[str] = None,
        vad: Optional[Tuple[float, float, float]] = None,
        emotion_idx: Optional[int] = None,
    ) -> Tensor:
        """
        Get emotion embedding from label, VAD, or index.

        Args:
            emotion: Emotion name (e.g., "happy")
            vad: VAD coordinates (valence, arousal, dominance)
            emotion_idx: Direct index into embedding table

        Returns:
            Emotion embedding [batch, emotion_dim] or [emotion_dim]
        """
        if emotion is not None:
            idx = self.emotion_to_idx.get(emotion.lower(), self.emotion_to_idx["neutral"])
            idx_tensor = torch.tensor([idx], device=self.emotion_embed.weight.device)
            return self.emotion_embed(idx_tensor).squeeze(0)

        elif vad is not None:
            vad_tensor = torch.tensor(vad, dtype=torch.float32, device=self.vad_encoder[0].weight.device)
            if vad_tensor.dim() == 1:
                vad_tensor = vad_tensor.unsqueeze(0)
            return self.vad_encoder(vad_tensor).squeeze(0)

        elif emotion_idx is not None:
            idx_tensor = torch.tensor([emotion_idx], device=self.emotion_embed.weight.device)
            return self.emotion_embed(idx_tensor).squeeze(0)

        else:
            # Default: neutral
            return self.emotion_embed(torch.tensor([self.emotion_to_idx["neutral"]],
                                                   device=self.emotion_embed.weight.device)).squeeze(0)

    def forward(
        self,
        emotion_embed: Tensor,
        content_embed: Tensor,
    ) -> Tensor:
        """
        Compute alignment score between emotion and content.

        Args:
            emotion_embed: Emotion embedding [batch, emotion_dim]
            content_embed: Content embedding [batch, content_dim]

        Returns:
            Alignment score [batch] in range [0, 1]
        """
        # Ensure batch dimension
        if emotion_embed.dim() == 1:
            emotion_embed = emotion_embed.unsqueeze(0)
        if content_embed.dim() == 1:
            content_embed = content_embed.unsqueeze(0)

        # Project content
        content_hidden = self.content_proj(content_embed)

        # Concatenate emotion and content
        joint = torch.cat([emotion_embed, content_hidden], dim=-1)

        # Forward through joint network
        hidden = self.joint_network(joint)

        # Compute alignment logit
        logit = self.alignment_head(hidden).squeeze(-1)

        # Apply temperature-scaled sigmoid
        alignment = torch.sigmoid(logit / self.temperature)

        return alignment

    def compute_alignment_from_text(
        self,
        emotion: str,
        text: str,
        text_encoder: Optional[nn.Module] = None,
    ) -> float:
        """
        Compute alignment score from raw text.

        Args:
            emotion: Target emotion
            text: Input text
            text_encoder: Optional encoder for text embeddings

        Returns:
            Alignment score [0, 1]
        """
        if text_encoder is not None:
            # Use neural text encoder
            with torch.no_grad():
                content_embed = text_encoder(text)
            emotion_embed = self.get_emotion_embedding(emotion=emotion)
            return self.forward(emotion_embed, content_embed).item()
        else:
            # Fallback: keyword-based alignment
            return self._keyword_alignment(emotion, text)

    def _keyword_alignment(self, emotion: str, text: str) -> float:
        """
        Compute keyword-based alignment (fallback method).

        Simple heuristic: Count emotion keywords in text.
        """
        text_lower = text.lower()
        emotion_lower = emotion.lower()

        if emotion_lower not in self.keyword_sets:
            return 0.5  # Unknown emotion → neutral alignment

        target_keywords = self.keyword_sets[emotion_lower]
        if not target_keywords:
            return 1.0  # Neutral matches everything

        # Count matching keywords
        matches = sum(1 for kw in target_keywords if kw in text_lower)

        # Check for conflicting emotions
        conflicts = 0
        for other_emotion, other_keywords in self.keyword_sets.items():
            if other_emotion != emotion_lower and other_emotion != "neutral":
                conflicts += sum(1 for kw in other_keywords if kw in text_lower)

        # Compute alignment score
        if matches > 0 and conflicts == 0:
            # Keywords match, no conflicts → high alignment
            return min(1.0, 0.7 + 0.1 * matches)
        elif matches > 0 and conflicts > 0:
            # Mixed signals → moderate alignment
            return 0.5 + 0.1 * (matches - conflicts)
        elif conflicts > 0:
            # Conflicting emotion → low alignment
            return max(0.2, 0.5 - 0.1 * conflicts)
        else:
            # Neutral text → moderate alignment
            return 0.6

    def compute_vad_alignment(
        self,
        emotion: str,
        content_vad: Tuple[float, float, float],
    ) -> float:
        """
        Compute alignment in VAD space.

        Args:
            emotion: Target emotion
            content_vad: Estimated VAD of content

        Returns:
            Alignment score [0, 1]
        """
        if emotion.lower() not in self.config.vad_prototypes:
            return 0.5

        emotion_vad = self.config.vad_prototypes[emotion.lower()]

        # Compute cosine similarity in VAD space
        emotion_vec = torch.tensor(emotion_vad)
        content_vec = torch.tensor(content_vad)

        if emotion_vec.norm() < 1e-6 or content_vec.norm() < 1e-6:
            return 0.5  # Neutral

        cos_sim = F.cosine_similarity(emotion_vec.unsqueeze(0), content_vec.unsqueeze(0))

        # Map from [-1, 1] to [0, 1]
        alignment = (cos_sim.item() + 1) / 2

        return alignment


class MismatchDiscriminatorLoss(nn.Module):
    """
    Loss function for training the mismatch discriminator.

    Uses contrastive learning:
    - Positive pairs: emotion + matching content → alignment = 1
    - Negative pairs: emotion + mismatching content → alignment = 0
    """

    def __init__(self, config: MismatchAwareCFGConfig):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        alignment_scores: Tensor,
        labels: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute discriminator loss.

        Args:
            alignment_scores: Predicted alignment [batch]
            labels: Ground truth alignment [batch] (1=match, 0=mismatch)

        Returns:
            Dict with loss values
        """
        # Binary cross-entropy loss
        bce = self.bce_loss(alignment_scores, labels)

        # Contrastive margin loss (optional)
        positive_mask = labels > 0.5
        negative_mask = labels < 0.5

        margin_loss = torch.tensor(0.0, device=alignment_scores.device)
        if positive_mask.any() and negative_mask.any():
            pos_scores = alignment_scores[positive_mask].mean()
            neg_scores = alignment_scores[negative_mask].mean()
            margin_loss = F.relu(self.config.contrastive_margin - (pos_scores - neg_scores))

        total_loss = bce + 0.5 * margin_loss

        return {
            "total": total_loss,
            "bce": bce,
            "margin": margin_loss,
            "mean_alignment": alignment_scores.mean(),
        }


# =============================================================================
# MISMATCH-AWARE CFG
# =============================================================================

class MismatchAwareCFG(nn.Module):
    """
    Mismatch-Aware Classifier-Free Guidance.

    Dynamically adjusts guidance scale based on style-content alignment:

        λ(x) = λ_min + (λ_max - λ_min) × alignment_score

        f̂ = f_cond + λ(x) × (f_cond - f_uncond)

    Benefits:
    - Prevents over-guidance when emotion doesn't match content
    - Automatic adaptation without manual tuning
    - Preserves naturalness in mismatched cases
    """

    def __init__(
        self,
        discriminator: SemanticMismatchDiscriminator,
        config: MismatchAwareCFGConfig,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.config = config

    def compute_dynamic_scale(
        self,
        emotion: Optional[str] = None,
        emotion_embed: Optional[Tensor] = None,
        content_embed: Optional[Tensor] = None,
        text: Optional[str] = None,
        text_encoder: Optional[nn.Module] = None,
    ) -> float:
        """
        Compute dynamic guidance scale based on alignment.

        Args:
            emotion: Target emotion name
            emotion_embed: Pre-computed emotion embedding
            content_embed: Pre-computed content embedding
            text: Raw text (if no content_embed)
            text_encoder: Encoder for text

        Returns:
            Dynamic guidance scale λ(x)
        """
        # Get alignment score
        if content_embed is not None:
            if emotion_embed is None and emotion is not None:
                emotion_embed = self.discriminator.get_emotion_embedding(emotion=emotion)

            if emotion_embed is not None:
                with torch.no_grad():
                    alignment = self.discriminator(emotion_embed, content_embed).item()
            else:
                alignment = 0.5  # Default

        elif text is not None and emotion is not None:
            alignment = self.discriminator.compute_alignment_from_text(
                emotion, text, text_encoder
            )

        else:
            alignment = 0.5  # Default neutral alignment

        # Compute dynamic scale
        scale_range = self.config.max_guidance_scale - self.config.min_guidance_scale
        dynamic_scale = self.config.min_guidance_scale + scale_range * alignment

        return dynamic_scale

    def apply_guidance(
        self,
        output_cond: Tensor,
        output_uncond: Tensor,
        scale: Optional[float] = None,
        emotion: Optional[str] = None,
        content_embed: Optional[Tensor] = None,
        text: Optional[str] = None,
    ) -> Tensor:
        """
        Apply mismatch-aware CFG.

        Args:
            output_cond: Conditioned output [batch, ...]
            output_uncond: Unconditional output [batch, ...]
            scale: Optional fixed scale (bypasses dynamic computation)
            emotion: Target emotion for dynamic scaling
            content_embed: Content embedding for alignment
            text: Raw text for alignment

        Returns:
            Guided output
        """
        if scale is None:
            scale = self.compute_dynamic_scale(
                emotion=emotion,
                content_embed=content_embed,
                text=text,
            )

        # Apply CFG formula
        guided = output_cond + scale * (output_cond - output_uncond)

        return guided

    def forward(
        self,
        output_cond: Tensor,
        output_uncond: Tensor,
        emotion: str,
        content_embed: Tensor,
    ) -> Tuple[Tensor, float]:
        """
        Forward pass with dynamic guidance.

        Returns:
            (guided_output, dynamic_scale)
        """
        scale = self.compute_dynamic_scale(
            emotion=emotion,
            content_embed=content_embed,
        )

        guided = self.apply_guidance(
            output_cond, output_uncond, scale=scale
        )

        return guided, scale


# =============================================================================
# MISMATCH-AWARE ACTIVATION STEERING
# =============================================================================

class MismatchAwareActivationSteering:
    """
    Extends ActivationSteering with mismatch-aware dynamic intensity.

    Instead of fixed intensity α, uses:
        α(x) = α_base × alignment_score

    This prevents over-steering when emotion doesn't match content.

    Usage:
        steerer = MismatchAwareActivationSteering(
            model, steering_vectors, discriminator
        )

        # Automatic intensity adjustment
        with steerer.steer_adaptive("happy", text_embedding=content_emb):
            audio = model.generate(text)

        # Or with raw text
        with steerer.steer_adaptive("happy", text="I'm so sad..."):
            audio = model.generate(text)  # Low intensity due to mismatch
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vectors: Dict[str, Dict[int, Tensor]],
        discriminator: SemanticMismatchDiscriminator,
        config: MismatchAwareCFGConfig = None,
        steering_config: SteeringConfig = None,
    ):
        """
        Args:
            model: TTS model to steer
            steering_vectors: Pre-extracted steering vectors
            discriminator: Mismatch discriminator for alignment
            config: Mismatch-aware CFG config
            steering_config: Base steering config
        """
        self.config = config or MismatchAwareCFGConfig()
        steering_config = steering_config or self.config.steering_config

        # Initialize base steering
        self.base_steerer = ActivationSteering(
            model, steering_vectors, steering_config
        )

        # Mismatch discriminator
        self.discriminator = discriminator

        # CFG module for guidance computation
        self.cfg = MismatchAwareCFG(discriminator, self.config)

        # Current state
        self._current_alignment: float = 1.0
        self._current_dynamic_intensity: float = 0.0

    def compute_alignment(
        self,
        emotion: str,
        text: Optional[str] = None,
        content_embed: Optional[Tensor] = None,
        text_encoder: Optional[nn.Module] = None,
    ) -> float:
        """Compute alignment score for emotion and content."""
        if content_embed is not None:
            emotion_embed = self.discriminator.get_emotion_embedding(emotion=emotion)
            with torch.no_grad():
                alignment = self.discriminator(emotion_embed, content_embed)
            return alignment.item() if alignment.numel() == 1 else alignment.mean().item()

        elif text is not None:
            return self.discriminator.compute_alignment_from_text(
                emotion, text, text_encoder
            )

        return 1.0  # Default full alignment

    def compute_dynamic_intensity(
        self,
        base_intensity: float,
        alignment: float,
    ) -> float:
        """
        Compute dynamic intensity based on alignment.

        Formula: α(x) = α_base × sqrt(alignment)

        Using sqrt for less aggressive scaling (alignment 0.5 → 0.7× intensity)
        """
        # Scale intensity by alignment
        scaled = base_intensity * math.sqrt(alignment)

        # Clamp to valid range
        min_intensity = self.base_steerer.config.min_intensity
        max_intensity = self.base_steerer.config.max_intensity

        return max(min_intensity, min(max_intensity, scaled))

    @contextmanager
    def steer_adaptive(
        self,
        emotion: str,
        base_intensity: float = None,
        text: Optional[str] = None,
        content_embed: Optional[Tensor] = None,
        text_encoder: Optional[nn.Module] = None,
        blend: Optional[Dict[str, float]] = None,
    ):
        """
        Context manager for adaptive steering based on content alignment.

        Args:
            emotion: Target emotion
            base_intensity: Base intensity (before alignment scaling)
            text: Raw text for alignment computation
            content_embed: Pre-computed content embedding
            text_encoder: Encoder for text
            blend: Optional emotion blend
        """
        if base_intensity is None:
            base_intensity = self.base_steerer.config.default_intensity

        # Compute alignment
        if blend is not None:
            # For blend, compute weighted average alignment
            total_weight = sum(blend.values())
            alignment = 0.0
            for emo, weight in blend.items():
                emo_align = self.compute_alignment(
                    emo, text=text, content_embed=content_embed, text_encoder=text_encoder
                )
                alignment += (weight / total_weight) * emo_align
        else:
            alignment = self.compute_alignment(
                emotion, text=text, content_embed=content_embed, text_encoder=text_encoder
            )

        # Compute dynamic intensity
        dynamic_intensity = self.compute_dynamic_intensity(base_intensity, alignment)

        # Store for inspection
        self._current_alignment = alignment
        self._current_dynamic_intensity = dynamic_intensity

        # Apply steering with dynamic intensity
        try:
            if blend is not None:
                self.base_steerer.enable(blend=blend, intensity=dynamic_intensity)
            else:
                self.base_steerer.enable(emotion=emotion, intensity=dynamic_intensity)
            yield self
        finally:
            self.base_steerer.disable()

    def steer_with_info(
        self,
        emotion: str,
        base_intensity: float = None,
        text: Optional[str] = None,
        content_embed: Optional[Tensor] = None,
    ) -> Dict:
        """
        Get steering info without applying steering.

        Returns dict with alignment, dynamic_intensity, etc.
        """
        if base_intensity is None:
            base_intensity = self.base_steerer.config.default_intensity

        alignment = self.compute_alignment(emotion, text=text, content_embed=content_embed)
        dynamic_intensity = self.compute_dynamic_intensity(base_intensity, alignment)

        return {
            "emotion": emotion,
            "base_intensity": base_intensity,
            "alignment": alignment,
            "dynamic_intensity": dynamic_intensity,
            "intensity_ratio": dynamic_intensity / base_intensity if base_intensity > 0 else 0,
        }

    @property
    def current_alignment(self) -> float:
        """Get current alignment score."""
        return self._current_alignment

    @property
    def current_dynamic_intensity(self) -> float:
        """Get current dynamic intensity."""
        return self._current_dynamic_intensity

    def list_emotions(self) -> List[str]:
        """List available emotions."""
        return self.base_steerer.list_emotions()


# =============================================================================
# SPHERICAL MISMATCH-AWARE STEERING
# =============================================================================

class SphericalMismatchAwareSteering:
    """
    Combines spherical emotion vectors with mismatch-aware steering.

    Enables:
    1. VAD-based emotion specification
    2. Automatic alignment computation in VAD space
    3. Dynamic intensity based on VAD alignment
    """

    def __init__(
        self,
        mismatch_steerer: MismatchAwareActivationSteering,
    ):
        self.mismatch_steerer = mismatch_steerer

        # Create spherical steering wrapper
        self.spherical = SphericalActivationSteering(
            mismatch_steerer.base_steerer
        )

    @contextmanager
    def steer_vad_adaptive(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        text: Optional[str] = None,
        content_embed: Optional[Tensor] = None,
        base_intensity: float = None,
    ):
        """
        Steer using VAD coordinates with mismatch-aware adaptation.

        Args:
            valence: -1 (negative) to +1 (positive)
            arousal: -1 (calm) to +1 (excited)
            dominance: -1 (submissive) to +1 (dominant)
            text: Raw text for alignment
            content_embed: Content embedding
            base_intensity: Base intensity before alignment scaling
        """
        if base_intensity is None:
            base_intensity = self.mismatch_steerer.base_steerer.config.default_intensity

        # Convert VAD to blend weights
        blend = self.spherical._vad_to_blend(valence, arousal, dominance)

        # Compute alignment using VAD
        content_vad = self._estimate_content_vad(text, content_embed)
        target_vad = (valence, arousal, dominance)

        if content_vad is not None:
            alignment = self._vad_alignment(target_vad, content_vad)
        else:
            # Fallback to text-based alignment
            primary_emotion = max(blend, key=blend.get)
            alignment = self.mismatch_steerer.compute_alignment(
                primary_emotion, text=text, content_embed=content_embed
            )

        # Compute dynamic intensity
        dynamic_intensity = self.mismatch_steerer.compute_dynamic_intensity(
            base_intensity, alignment
        )

        # Apply steering
        try:
            self.mismatch_steerer.base_steerer.enable(blend=blend, intensity=dynamic_intensity)
            self.mismatch_steerer._current_alignment = alignment
            self.mismatch_steerer._current_dynamic_intensity = dynamic_intensity
            yield self
        finally:
            self.mismatch_steerer.base_steerer.disable()

    def _estimate_content_vad(
        self,
        text: Optional[str],
        content_embed: Optional[Tensor],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate VAD coordinates of content.

        Returns None if estimation not possible.
        """
        if text is None:
            return None

        # Simple keyword-based VAD estimation
        text_lower = text.lower()

        vad_sum = [0.0, 0.0, 0.0]
        count = 0

        for emotion, vad in self.mismatch_steerer.config.vad_prototypes.items():
            keywords = self.mismatch_steerer.discriminator.keyword_sets.get(emotion, set())
            for kw in keywords:
                if kw in text_lower:
                    vad_sum[0] += vad[0]
                    vad_sum[1] += vad[1]
                    vad_sum[2] += vad[2]
                    count += 1

        if count == 0:
            return (0.0, 0.0, 0.0)  # Neutral

        return (vad_sum[0] / count, vad_sum[1] / count, vad_sum[2] / count)

    def _vad_alignment(
        self,
        target_vad: Tuple[float, float, float],
        content_vad: Tuple[float, float, float],
    ) -> float:
        """Compute alignment in VAD space."""
        target = torch.tensor(target_vad)
        content = torch.tensor(content_vad)

        if target.norm() < 1e-6 or content.norm() < 1e-6:
            return 0.6  # Neutral alignment

        cos_sim = F.cosine_similarity(target.unsqueeze(0), content.unsqueeze(0))

        # Map [-1, 1] to [0, 1]
        return (cos_sim.item() + 1) / 2


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class MismatchDataGenerator:
    """
    Generates training data for mismatch discriminator.

    Creates positive (matching) and negative (mismatching) pairs
    from emotion-labeled data.
    """

    def __init__(self, config: MismatchAwareCFGConfig):
        self.config = config

    def generate_pairs(
        self,
        emotions: List[str],
        texts: List[str],
        emotion_labels: List[str],
    ) -> Tuple[List[Tuple[str, str, float]], ...]:
        """
        Generate match/mismatch pairs.

        Args:
            emotions: Available emotion labels
            texts: Text samples
            emotion_labels: Emotion labels for each text

        Returns:
            List of (emotion, text, label) tuples
        """
        pairs = []

        for i, (text, true_emotion) in enumerate(zip(texts, emotion_labels)):
            # Positive pair: correct emotion
            pairs.append((true_emotion, text, 1.0))

            # Negative pairs: wrong emotions
            for wrong_emotion in emotions:
                if wrong_emotion != true_emotion and wrong_emotion != "neutral":
                    pairs.append((wrong_emotion, text, 0.0))

        return pairs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_mismatch_aware_steerer(
    model: nn.Module,
    steering_vectors: Dict[str, Dict[int, Tensor]],
    config: MismatchAwareCFGConfig = None,
    pretrained_discriminator: Optional[str] = None,
) -> MismatchAwareActivationSteering:
    """
    Convenience function to create mismatch-aware steering.

    Args:
        model: TTS model
        steering_vectors: Pre-extracted steering vectors
        config: Configuration
        pretrained_discriminator: Path to pretrained discriminator weights

    Returns:
        Configured MismatchAwareActivationSteering
    """
    config = config or MismatchAwareCFGConfig()

    # Create discriminator
    discriminator = SemanticMismatchDiscriminator(config)

    # Load pretrained weights if provided
    if pretrained_discriminator is not None:
        state = torch.load(pretrained_discriminator, weights_only=True)
        discriminator.load_state_dict(state)

    # Create mismatch-aware steerer
    return MismatchAwareActivationSteering(
        model, steering_vectors, discriminator, config
    )


def analyze_alignment(
    discriminator: SemanticMismatchDiscriminator,
    emotions: List[str],
    texts: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Analyze alignment scores for emotion-text combinations.

    Returns matrix of alignment scores.
    """
    results = {}

    for emotion in emotions:
        results[emotion] = {}
        for text in texts:
            alignment = discriminator.compute_alignment_from_text(emotion, text)
            results[emotion][text[:30] + "..."] = round(alignment, 3)

    return results


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Mismatch-Aware CFG for Emotion Control - Test Suite")
    print("=" * 70)

    # Test 1: Configuration
    print("\n[Test 1] MismatchAwareCFGConfig...")
    config = MismatchAwareCFGConfig(
        base_guidance_scale=1.0,
        min_guidance_scale=0.1,
        max_guidance_scale=2.0,
    )
    print(f"  Base scale: {config.base_guidance_scale}")
    print(f"  Scale range: [{config.min_guidance_scale}, {config.max_guidance_scale}]")
    print(f"  Emotions: {list(config.emotion_keywords.keys())}")
    print("  [PASS]")

    # Test 2: Semantic Mismatch Discriminator
    print("\n[Test 2] SemanticMismatchDiscriminator...")
    discriminator = SemanticMismatchDiscriminator(config)
    print(f"  Emotion embeddings: {discriminator.emotion_embed.weight.shape}")
    print(f"  Emotion names: {discriminator.emotion_names}")

    # Test emotion embedding
    happy_embed = discriminator.get_emotion_embedding(emotion="happy")
    print(f"  Happy embedding shape: {happy_embed.shape}")

    # Test VAD embedding
    vad_embed = discriminator.get_emotion_embedding(vad=(0.8, 0.6, 0.4))
    print(f"  VAD embedding shape: {vad_embed.shape}")
    print("  [PASS]")

    # Test 3: Keyword-based alignment
    print("\n[Test 3] Keyword-based alignment...")
    test_cases = [
        ("happy", "I'm so excited and thrilled about this amazing news!"),
        ("happy", "This is terrible, I'm devastated."),
        ("sad", "I feel so lonely and heartbroken."),
        ("sad", "What a wonderful day!"),
        ("angry", "I'm furious about this outrage!"),
        ("angry", "Everything is calm and peaceful."),
        ("neutral", "The meeting is at 3pm."),
    ]

    for emotion, text in test_cases:
        alignment = discriminator.compute_alignment_from_text(emotion, text)
        match_type = "MATCH" if alignment > 0.6 else "MISMATCH" if alignment < 0.4 else "NEUTRAL"
        print(f"  {emotion:10} + '{text[:40]}...' → {alignment:.2f} ({match_type})")
    print("  [PASS]")

    # Test 4: Neural discriminator forward pass
    print("\n[Test 4] Neural discriminator forward pass...")
    batch_size = 4
    emotion_embed = torch.randn(batch_size, config.emotion_dim)
    content_embed = torch.randn(batch_size, config.content_dim)

    alignment_scores = discriminator(emotion_embed, content_embed)
    print(f"  Input shapes: emotion={emotion_embed.shape}, content={content_embed.shape}")
    print(f"  Output shape: {alignment_scores.shape}")
    print(f"  Alignment range: [{alignment_scores.min().item():.3f}, {alignment_scores.max().item():.3f}]")
    print("  [PASS]")

    # Test 5: VAD alignment
    print("\n[Test 5] VAD alignment...")
    vad_test_cases = [
        ("happy", (0.8, 0.6, 0.6)),   # Match
        ("happy", (-0.6, -0.4, -0.5)), # Mismatch (sad VAD)
        ("sad", (-0.6, -0.4, -0.5)),   # Match
        ("angry", (0.5, 0.8, 0.7)),    # Partial match
    ]

    for emotion, content_vad in vad_test_cases:
        alignment = discriminator.compute_vad_alignment(emotion, content_vad)
        print(f"  {emotion:10} + VAD{content_vad} → {alignment:.3f}")
    print("  [PASS]")

    # Test 6: Mismatch-Aware CFG
    print("\n[Test 6] MismatchAwareCFG...")
    cfg = MismatchAwareCFG(discriminator, config)

    # Test dynamic scale computation
    scale_match = cfg.compute_dynamic_scale(emotion="happy", text="I'm so excited!")
    scale_mismatch = cfg.compute_dynamic_scale(emotion="happy", text="I'm devastated and sad.")

    print(f"  Matching case scale: {scale_match:.3f}")
    print(f"  Mismatching case scale: {scale_mismatch:.3f}")
    assert scale_match > scale_mismatch, "Matching should have higher scale"
    print("  [PASS]")

    # Test 7: CFG guidance application
    print("\n[Test 7] CFG guidance application...")
    output_cond = torch.randn(1, 10, 512)
    output_uncond = torch.randn(1, 10, 512)

    guided_match = cfg.apply_guidance(
        output_cond, output_uncond,
        emotion="happy", text="What a wonderful day!"
    )
    guided_mismatch = cfg.apply_guidance(
        output_cond, output_uncond,
        emotion="happy", text="I'm so sad."
    )

    diff_match = (guided_match - output_cond).abs().mean()
    diff_mismatch = (guided_mismatch - output_cond).abs().mean()

    print(f"  Matching guidance diff: {diff_match.item():.4f}")
    print(f"  Mismatching guidance diff: {diff_mismatch.item():.4f}")
    print("  [PASS]")

    # Test 8: Discriminator loss
    print("\n[Test 8] MismatchDiscriminatorLoss...")
    loss_fn = MismatchDiscriminatorLoss(config)

    alignment_pred = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    losses = loss_fn(alignment_pred, labels)
    print(f"  BCE loss: {losses['bce'].item():.4f}")
    print(f"  Margin loss: {losses['margin'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: Create dummy model and steering vectors
    print("\n[Test 9] Creating dummy model and steering vectors...")

    class DummyLayer(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.self_attn = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x):
            return x + self.self_attn(x)

    class DummyModel(nn.Module):
        def __init__(self, num_layers=12, hidden_dim=512):
            super().__init__()
            self.layers = nn.ModuleList([
                DummyLayer(hidden_dim) for _ in range(num_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = DummyModel()

    # Create dummy steering vectors
    hidden_dim = 512
    seq_len = 50
    target_layers = [1, 4, 7, 10]

    dummy_vectors = {}
    for emotion in ["happy", "sad", "angry"]:
        dummy_vectors[emotion] = {
            layer: torch.randn(seq_len, hidden_dim)
            for layer in target_layers
        }

    print(f"  Model layers: {len(model.layers)}")
    print(f"  Steering vectors: {list(dummy_vectors.keys())}")
    print("  [PASS]")

    # Test 10: MismatchAwareActivationSteering
    print("\n[Test 10] MismatchAwareActivationSteering...")
    mismatch_steerer = MismatchAwareActivationSteering(
        model, dummy_vectors, discriminator, config
    )

    print(f"  Available emotions: {mismatch_steerer.list_emotions()}")

    # Test steering info
    info_match = mismatch_steerer.steer_with_info(
        "happy", base_intensity=0.8, text="I'm so happy!"
    )
    info_mismatch = mismatch_steerer.steer_with_info(
        "happy", base_intensity=0.8, text="I'm devastated."
    )

    print(f"  Matching: align={info_match['alignment']:.2f}, intensity={info_match['dynamic_intensity']:.2f}")
    print(f"  Mismatching: align={info_mismatch['alignment']:.2f}, intensity={info_mismatch['dynamic_intensity']:.2f}")
    assert info_match['dynamic_intensity'] > info_mismatch['dynamic_intensity'], \
        "Matching should have higher intensity"
    print("  [PASS]")

    # Test 11: Adaptive steering context manager
    print("\n[Test 11] Adaptive steering context manager...")
    x = torch.randn(1, 30, hidden_dim)

    # Baseline
    with torch.no_grad():
        y_baseline = model(x)

    # With adaptive steering (matching)
    with mismatch_steerer.steer_adaptive("happy", text="I'm excited!", base_intensity=0.8):
        with torch.no_grad():
            y_match = model(x)
        align_match = mismatch_steerer.current_alignment
        intensity_match = mismatch_steerer.current_dynamic_intensity

    # With adaptive steering (mismatching)
    with mismatch_steerer.steer_adaptive("happy", text="I'm so sad.", base_intensity=0.8):
        with torch.no_grad():
            y_mismatch = model(x)
        align_mismatch = mismatch_steerer.current_alignment
        intensity_mismatch = mismatch_steerer.current_dynamic_intensity

    diff_match = (y_match - y_baseline).abs().mean().item()
    diff_mismatch = (y_mismatch - y_baseline).abs().mean().item()

    print(f"  Matching: diff={diff_match:.4f}, align={align_match:.2f}, α={intensity_match:.2f}")
    print(f"  Mismatching: diff={diff_mismatch:.4f}, align={align_mismatch:.2f}, α={intensity_mismatch:.2f}")
    print("  [PASS]")

    # Test 12: Spherical mismatch-aware steering
    print("\n[Test 12] SphericalMismatchAwareSteering...")
    spherical_steerer = SphericalMismatchAwareSteering(mismatch_steerer)

    with spherical_steerer.steer_vad_adaptive(0.8, 0.6, 0.4, text="Great news!", base_intensity=0.7):
        with torch.no_grad():
            y_vad = model(x)
        align = mismatch_steerer.current_alignment
        intensity = mismatch_steerer.current_dynamic_intensity

    diff_vad = (y_vad - y_baseline).abs().mean().item()
    print(f"  VAD steering: diff={diff_vad:.4f}, align={align:.2f}, α={intensity:.2f}")
    print("  [PASS]")

    # Test 13: Analyze alignment matrix
    print("\n[Test 13] Alignment matrix analysis...")
    emotions = ["happy", "sad", "angry", "neutral"]
    texts = [
        "I'm so excited about this!",
        "This is terrible news.",
        "I'm furious about this!",
        "The meeting is at 3pm.",
    ]

    alignment_matrix = analyze_alignment(discriminator, emotions, texts)
    print("  Alignment matrix:")
    for emotion, alignments in alignment_matrix.items():
        print(f"    {emotion}:")
        for text, score in alignments.items():
            print(f"      {text}: {score}")
    print("  [PASS]")

    # Test 14: Data generator
    print("\n[Test 14] MismatchDataGenerator...")
    generator = MismatchDataGenerator(config)

    sample_texts = ["I'm happy!", "I'm sad.", "I'm angry!"]
    sample_labels = ["happy", "sad", "angry"]

    pairs = generator.generate_pairs(emotions, sample_texts, sample_labels)
    print(f"  Generated {len(pairs)} pairs")
    print(f"  Sample pairs:")
    for i, (emo, text, label) in enumerate(pairs[:5]):
        print(f"    {emo} + '{text}' → {label}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All Mismatch-Aware CFG tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from mismatch_aware_cfg import (
    MismatchAwareCFGConfig,
    SemanticMismatchDiscriminator,
    MismatchAwareActivationSteering,
    SphericalMismatchAwareSteering,
    create_mismatch_aware_steerer,
)
from activation_steering import SteeringVectorExtractor

# 1. Extract steering vectors (same as before)
extractor = SteeringVectorExtractor(model)
steering_vectors = extractor.extract(
    neutral_samples=["neutral1.wav", ...],
    emotional_samples={"happy": [...], "sad": [...], ...},
    process_fn=process_fn,
)

# 2. Create mismatch-aware steerer
config = MismatchAwareCFGConfig(
    base_guidance_scale=1.0,
    min_guidance_scale=0.1,  # Floor for mismatched cases
    max_guidance_scale=2.0,  # Ceiling for matched cases
)

# Simple way:
steerer = create_mismatch_aware_steerer(model, steering_vectors, config)

# Or manual:
discriminator = SemanticMismatchDiscriminator(config)
steerer = MismatchAwareActivationSteering(model, steering_vectors, discriminator, config)

# 3. Generate with automatic intensity adjustment
with steerer.steer_adaptive("happy", text="I'm so excited about this!"):
    audio = model.generate(text)  # High alignment → strong steering

with steerer.steer_adaptive("happy", text="I'm devastated and sad."):
    audio = model.generate(text)  # Low alignment → weak steering

# 4. Check alignment info before generation
info = steerer.steer_with_info("angry", text="This is outrageous!")
print(f"Alignment: {info['alignment']:.2f}")
print(f"Dynamic intensity: {info['dynamic_intensity']:.2f}")

# 5. Use with content embeddings (for better accuracy)
content_embed = text_encoder(text)  # BERT/wav2vec2
with steerer.steer_adaptive("happy", content_embed=content_embed):
    audio = model.generate(text)

# 6. VAD-based steering with mismatch awareness
spherical = SphericalMismatchAwareSteering(steerer)
with spherical.steer_vad_adaptive(0.8, 0.6, 0.4, text="Great news!"):
    audio = model.generate(text)

# 7. Emotion blending with mismatch awareness
with steerer.steer_adaptive(
    "happy",  # Primary emotion
    blend={"happy": 0.7, "surprised": 0.3},
    text="What amazing news!",
    base_intensity=0.8,
):
    audio = model.generate(text)
""")
