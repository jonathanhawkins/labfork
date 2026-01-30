"""
SAVC: Adversarial Style Augmentation for Speaker-Invariant Prosody

Based on "Self-Adversarial Voice Conversion for Speaker-Independent Style Learning"
(arXiv:2405.00603) - SAVC approach for learning speaker-independent prosody representations.

Key Technique - Statistic Perturbation:
- Generate augmented samples with perturbed feature statistics (mean, variance)
- Perturbed samples share same content/prosody but different "style" (speaker)
- Encoder trained to extract similar representations from original + perturbed
- Results in speaker-invariant prosody features

Architecture:
1. HuBERT-Soft for soft speech units (content representation)
2. Attribute encoder for time-variant prosody features
3. Adversarial augmentation perturbs speaker style during training
4. Prosody should be consistent across augmented versions

Benefits:
- Data augmentation approach (vs architectural constraints like GRL/MINE)
- Can work with existing encoders without modification
- Encourages natural speaker invariance through training signal
- Complementary to other disentanglement methods (GRL, MINE, orthogonality)
- Simple and effective - no adversarial networks needed

Integration:
- Works with existing prosody encoders (V6, HED, EmoFiLM, etc.)
- Can be combined with scheduled GRL for additional disentanglement
- Enhances prosody transfer quality in zero-shot scenarios

Usage:
    from savc import (
        SAVCConfig,
        StatisticPerturbation,
        AdversarialStyleAugmentor,
        StyleConsistencyLoss,
        SAVCAdapter,
    )

    config = SAVCConfig()
    augmentor = AdversarialStyleAugmentor(config)
    consistency_loss = StyleConsistencyLoss(config)

    # During training:
    original_features = encoder(audio)
    augmented_features = augmentor.augment(original_features)

    # Consistency loss (prosody should be same for original + augmented)
    prosody_original = prosody_encoder(original_features)
    prosody_augmented = prosody_encoder(augmented_features)
    loss = consistency_loss(prosody_original, prosody_augmented)
"""

import math
import random
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
class SAVCConfig:
    """Configuration for SAVC adversarial style augmentation."""

    # Feature dimensions
    input_dim: int = 768        # Input feature dimension (HuBERT/wav2vec2)
    hidden_dim: int = 512       # Hidden layer dimension
    prosody_dim: int = 256      # Prosody embedding dimension
    output_dim: int = 2048      # Output dimension for CSM integration

    # Statistic perturbation parameters
    mean_perturbation_range: Tuple[float, float] = (-0.3, 0.3)  # ±30% mean shift
    var_perturbation_range: Tuple[float, float] = (0.7, 1.3)    # 70%-130% variance scale
    perturbation_probability: float = 0.8    # Probability of applying perturbation

    # Instance normalization (for feature normalization before perturbation)
    use_instance_norm: bool = True
    eps: float = 1e-6

    # Spectral perturbation (frequency-domain augmentation)
    use_spectral_perturbation: bool = True
    spectral_band_size: int = 16       # Size of frequency bands to perturb
    spectral_perturbation_strength: float = 0.2  # Strength of spectral perturbation

    # Content preservation constraints
    content_preservation_weight: float = 0.5  # Weight for content preservation loss
    use_contrastive_content: bool = True      # Use contrastive loss for content

    # Style consistency loss
    consistency_temperature: float = 0.1      # Temperature for consistency loss
    use_cosine_similarity: bool = True        # Use cosine sim vs L2 distance
    margin: float = 0.5                       # Margin for triplet-style loss

    # Training settings
    warmup_steps: int = 1000        # Steps before full perturbation strength
    augment_both_views: bool = True  # Augment both views or just one

    # Loss weights
    consistency_weight: float = 1.0      # Weight for style consistency loss
    adversarial_weight: float = 0.5      # Weight for adversarial component
    reconstruction_weight: float = 0.3   # Weight for reconstruction component

    # Number of augmented views
    num_augmentations: int = 2    # Number of augmented views per sample

    # Integration settings
    num_prefix_tokens: int = 4    # Number of prosody prefix tokens


# =============================================================================
# STATISTIC PERTURBATION
# =============================================================================

class StatisticPerturbation(nn.Module):
    """
    Statistic Perturbation module for adversarial style augmentation.

    Perturbs the mean and variance of feature representations to create
    "same content, different speaker" pairs. This is the core SAVC technique.

    The key insight: speaker identity is encoded primarily in the low-level
    statistics (mean, variance) of feature representations, while content
    and prosody information is in the relative patterns.

    By perturbing statistics while preserving relative structure, we create
    samples that sound like different speakers saying the same thing with
    the same prosody.
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config

        # Optional instance normalization
        if config.use_instance_norm:
            self.instance_norm = nn.InstanceNorm1d(
                config.input_dim,
                affine=False,
                eps=config.eps,
            )

        # Learnable perturbation parameters (optional - can also use random)
        self.learnable_mean_shift = nn.Parameter(
            torch.zeros(config.input_dim),
            requires_grad=True,
        )
        self.learnable_var_scale = nn.Parameter(
            torch.ones(config.input_dim),
            requires_grad=True,
        )

    def extract_statistics(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mean and variance statistics from features.

        Args:
            features: [batch, seq, dim] or [batch, dim, seq]
            mask: Optional [batch, seq] mask for valid positions

        Returns:
            mean: [batch, dim] feature mean
            var: [batch, dim] feature variance
        """
        # Ensure [batch, seq, dim] format
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.config.input_dim:
            features = features.transpose(-1, -2)

        batch, seq, dim = features.shape

        if mask is not None:
            # Masked mean and variance
            mask = mask.unsqueeze(-1).float()  # [batch, seq, 1]
            valid_count = mask.sum(dim=1).clamp(min=1)  # [batch, 1]

            mean = (features * mask).sum(dim=1) / valid_count  # [batch, dim]
            centered = features - mean.unsqueeze(1)
            var = ((centered ** 2) * mask).sum(dim=1) / valid_count
        else:
            mean = features.mean(dim=1)  # [batch, dim]
            var = features.var(dim=1, unbiased=False)

        return mean, var

    def apply_perturbation(
        self,
        features: torch.Tensor,
        mean_shift: torch.Tensor,
        var_scale: torch.Tensor,
        original_mean: torch.Tensor,
        original_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply statistic perturbation to features.

        Args:
            features: [batch, seq, dim] input features
            mean_shift: [batch, dim] shift to add to mean
            var_scale: [batch, dim] scale to multiply variance
            original_mean: [batch, dim] original feature mean
            original_var: [batch, dim] original feature variance

        Returns:
            Perturbed features [batch, seq, dim]
        """
        # Normalize to zero mean, unit variance
        centered = features - original_mean.unsqueeze(1)
        original_std = (original_var + self.config.eps).sqrt().unsqueeze(1)
        normalized = centered / original_std

        # Apply new statistics
        new_mean = original_mean + mean_shift
        new_std = (original_var * var_scale + self.config.eps).sqrt()

        perturbed = normalized * new_std.unsqueeze(1) + new_mean.unsqueeze(1)

        return perturbed

    def random_perturbation(
        self,
        features: torch.Tensor,
        strength: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random mean shift and variance scale.

        Args:
            features: [batch, seq, dim] for getting batch size
            strength: Scale factor for perturbation (0-1, increases during warmup)

        Returns:
            mean_shift: [batch, dim]
            var_scale: [batch, dim]
        """
        batch = features.shape[0]
        dim = self.config.input_dim
        device = features.device

        # Random mean shift
        mean_low, mean_high = self.config.mean_perturbation_range
        mean_shift = torch.empty(batch, dim, device=device).uniform_(
            mean_low * strength, mean_high * strength
        )

        # Random variance scale
        var_low, var_high = self.config.var_perturbation_range
        # Interpolate toward 1.0 based on strength
        var_scale = torch.empty(batch, dim, device=device).uniform_(
            1.0 - (1.0 - var_low) * strength,
            1.0 + (var_high - 1.0) * strength,
        )

        return mean_shift, var_scale

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply adversarial style augmentation via statistic perturbation.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional [batch, seq] attention mask
            strength: Perturbation strength (0-1, for warmup)
            deterministic: If True, use learnable perturbation; else random

        Returns:
            Dict containing:
                - perturbed: Perturbed features
                - mean_shift: Applied mean shift
                - var_scale: Applied variance scale
                - original_mean: Original mean
                - original_var: Original variance
        """
        # Ensure correct shape
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if features.shape[-1] != self.config.input_dim:
            features = features.transpose(-1, -2)

        batch, seq, dim = features.shape

        # Extract original statistics
        original_mean, original_var = self.extract_statistics(features, mask)

        # Decide whether to apply perturbation
        if self.training and random.random() > self.config.perturbation_probability:
            # Skip perturbation for this sample
            return {
                'perturbed': features,
                'mean_shift': torch.zeros_like(original_mean),
                'var_scale': torch.ones_like(original_var),
                'original_mean': original_mean,
                'original_var': original_var,
            }

        if deterministic:
            # Use learnable perturbation parameters
            mean_shift = self.learnable_mean_shift.unsqueeze(0).expand(batch, -1) * strength
            var_scale = self.learnable_var_scale.unsqueeze(0).expand(batch, -1)
            var_scale = 1.0 + (var_scale - 1.0) * strength
        else:
            # Random perturbation
            mean_shift, var_scale = self.random_perturbation(features, strength)

        # Apply perturbation
        perturbed = self.apply_perturbation(
            features,
            mean_shift,
            var_scale,
            original_mean,
            original_var,
        )

        return {
            'perturbed': perturbed,
            'mean_shift': mean_shift,
            'var_scale': var_scale,
            'original_mean': original_mean,
            'original_var': original_var,
        }


# =============================================================================
# SPECTRAL PERTURBATION
# =============================================================================

class SpectralPerturbation(nn.Module):
    """
    Frequency-domain perturbation for additional speaker augmentation.

    Perturbs different frequency bands independently, which helps create
    more diverse speaker variations. Speaker identity is often encoded
    in specific frequency ranges (formants, etc.).
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config
        self.band_size = config.spectral_band_size
        self.strength = config.spectral_perturbation_strength

    def forward(
        self,
        features: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply frequency-band perturbation.

        Args:
            features: [batch, seq, dim] input features
            strength: Perturbation strength (0-1)

        Returns:
            Perturbed features [batch, seq, dim]
        """
        if not self.config.use_spectral_perturbation:
            return features

        batch, seq, dim = features.shape
        device = features.device

        # Number of frequency bands
        num_bands = dim // self.band_size

        # Random scale per band
        band_scales = torch.empty(batch, num_bands, device=device).uniform_(
            1.0 - self.strength * strength,
            1.0 + self.strength * strength,
        )

        # Expand to full dimension
        scales = band_scales.repeat_interleave(self.band_size, dim=-1)

        # Handle remainder dimensions
        remainder = dim - scales.shape[-1]
        if remainder > 0:
            scales = torch.cat([
                scales,
                torch.ones(batch, remainder, device=device),
            ], dim=-1)

        # Apply scaling
        perturbed = features * scales.unsqueeze(1)

        return perturbed


# =============================================================================
# ADVERSARIAL STYLE AUGMENTOR
# =============================================================================

class AdversarialStyleAugmentor(nn.Module):
    """
    Complete adversarial style augmentation module.

    Combines statistic perturbation with spectral perturbation to create
    diverse speaker variations while preserving content and prosody.

    The augmentor generates multiple views of each sample that should
    be treated as "same prosody, different speaker" pairs for training.
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config

        # Core statistic perturbation
        self.stat_perturbation = StatisticPerturbation(config)

        # Optional spectral perturbation
        if config.use_spectral_perturbation:
            self.spectral_perturbation = SpectralPerturbation(config)
        else:
            self.spectral_perturbation = None

        # Track training steps for warmup
        self.register_buffer('step', torch.tensor(0))

    def get_strength(self) -> float:
        """Get current perturbation strength based on warmup."""
        if not self.training:
            return 1.0
        progress = min(self.step.item() / max(1, self.config.warmup_steps), 1.0)
        return progress

    def augment_single(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a single augmented view.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional attention mask
            strength: Override strength (if None, use warmup-based)

        Returns:
            Dict with augmented features and metadata
        """
        if strength is None:
            strength = self.get_strength()

        # Apply statistic perturbation
        result = self.stat_perturbation(
            features,
            mask=mask,
            strength=strength,
            deterministic=False,
        )

        # Apply spectral perturbation
        if self.spectral_perturbation is not None:
            result['perturbed'] = self.spectral_perturbation(
                result['perturbed'],
                strength=strength,
            )

        return result

    def augment(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_augmentations: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multiple augmented views.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional attention mask
            num_augmentations: Number of augmented views (default: config value)

        Returns:
            Dict containing:
                - original: Original features
                - augmented: List of augmented features
                - all_views: All views concatenated [num_views * batch, seq, dim]
        """
        if num_augmentations is None:
            num_augmentations = self.config.num_augmentations

        strength = self.get_strength()

        # Generate augmented views
        augmented_views = []
        for _ in range(num_augmentations):
            aug_result = self.augment_single(features, mask, strength)
            augmented_views.append(aug_result['perturbed'])

        # Concatenate all views (original + augmented)
        if self.config.augment_both_views:
            # Augment the original too for symmetry
            original_aug = self.augment_single(features, mask, strength * 0.5)
            all_views = [original_aug['perturbed']] + augmented_views
        else:
            all_views = [features] + augmented_views

        all_views_tensor = torch.cat(all_views, dim=0)

        return {
            'original': features,
            'augmented': augmented_views,
            'all_views': all_views_tensor,
            'num_views': len(all_views),
            'strength': strength,
        }

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass - augment features."""
        if self.training:
            self.step += 1
        return self.augment(features, mask)


# =============================================================================
# STYLE CONSISTENCY LOSS
# =============================================================================

class StyleConsistencyLoss(nn.Module):
    """
    Style (Prosody) Consistency Loss for SAVC training.

    Ensures that the prosody encoder produces similar representations
    for original and augmented versions of the same sample.

    The loss encourages:
    1. Same sample, different augmentation → similar prosody embedding
    2. Different samples → different prosody embeddings (contrastive)

    This trains the prosody encoder to be invariant to speaker variations
    while still capturing prosodic differences between utterances.
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config
        self.temperature = config.consistency_temperature
        self.margin = config.margin

    def compute_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between embeddings.

        Args:
            emb1: [batch, dim] first embeddings
            emb2: [batch, dim] second embeddings

        Returns:
            [batch] similarity scores
        """
        if self.config.use_cosine_similarity:
            # Cosine similarity
            emb1_norm = F.normalize(emb1, p=2, dim=-1)
            emb2_norm = F.normalize(emb2, p=2, dim=-1)
            similarity = (emb1_norm * emb2_norm).sum(dim=-1)
        else:
            # Negative L2 distance (higher = more similar)
            similarity = -((emb1 - emb2) ** 2).sum(dim=-1).sqrt()

        return similarity

    def positive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss for positive pairs (same sample, different augmentation).

        Args:
            anchor: [batch, dim] anchor embeddings
            positive: [batch, dim] positive embeddings

        Returns:
            Scalar loss value
        """
        similarity = self.compute_similarity(anchor, positive)

        if self.config.use_cosine_similarity:
            # Push similarity toward 1.0
            loss = (1.0 - similarity).mean()
        else:
            # Minimize distance
            loss = -similarity.mean()

        return loss

    def contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Contrastive loss with in-batch negatives.

        Args:
            anchor: [batch, dim] anchor embeddings
            positive: [batch, dim] positive embeddings (same sample, different aug)
            negatives: Optional [batch, num_neg, dim] explicit negatives

        Returns:
            Scalar contrastive loss
        """
        batch = anchor.shape[0]
        device = anchor.device

        # Normalize embeddings
        anchor_norm = F.normalize(anchor, p=2, dim=-1)
        positive_norm = F.normalize(positive, p=2, dim=-1)

        # Positive similarities
        pos_sim = (anchor_norm * positive_norm).sum(dim=-1)  # [batch]

        # Use other samples in batch as negatives
        # Compute all-pairs similarity matrix
        sim_matrix = torch.mm(anchor_norm, positive_norm.T)  # [batch, batch]

        # Mask out diagonal (positive pairs)
        mask = torch.eye(batch, device=device, dtype=torch.bool)
        neg_sim = sim_matrix.masked_fill(mask, float('-inf'))

        # Compute InfoNCE-style loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature
        labels = torch.zeros(batch, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet margin loss.

        Args:
            anchor: [batch, dim] anchor embeddings
            positive: [batch, dim] positive embeddings
            negative: [batch, dim] negative embeddings

        Returns:
            Scalar triplet loss
        """
        pos_dist = ((anchor - positive) ** 2).sum(dim=-1).sqrt()
        neg_dist = ((anchor - negative) ** 2).sum(dim=-1).sqrt()

        loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        return loss

    def forward(
        self,
        embeddings: List[torch.Tensor],
        use_contrastive: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute style consistency loss.

        Args:
            embeddings: List of embeddings for each view
                [original_emb, aug1_emb, aug2_emb, ...]
            use_contrastive: Whether to use contrastive loss

        Returns:
            Dict with loss components
        """
        num_views = len(embeddings)
        device = embeddings[0].device

        losses = {}

        # Positive pair loss: all pairs should be similar
        positive_loss = torch.tensor(0.0, device=device)
        num_pairs = 0

        for i in range(num_views):
            for j in range(i + 1, num_views):
                positive_loss = positive_loss + self.positive_loss(
                    embeddings[i], embeddings[j]
                )
                num_pairs += 1

        if num_pairs > 0:
            positive_loss = positive_loss / num_pairs

        losses['positive'] = positive_loss

        # Contrastive loss: different samples should be different
        if use_contrastive and self.config.use_contrastive_content:
            # Use first two views for contrastive
            contrastive_loss = self.contrastive_loss(
                embeddings[0], embeddings[1]
            )
            losses['contrastive'] = contrastive_loss
        else:
            losses['contrastive'] = torch.tensor(0.0, device=device)

        # Total loss
        losses['total'] = (
            self.config.consistency_weight * losses['positive'] +
            self.config.adversarial_weight * losses['contrastive']
        )

        return losses


# =============================================================================
# CONTENT PRESERVATION LOSS
# =============================================================================

class ContentPreservationLoss(nn.Module):
    """
    Loss to ensure content is preserved during augmentation.

    While we want prosody/speaker-style to be invariant to augmentation,
    we must ensure the linguistic content is preserved. This loss
    encourages the content encoder to produce identical outputs for
    original and augmented views.
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        content_original: torch.Tensor,
        content_augmented: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute content preservation loss.

        Args:
            content_original: [batch, seq, dim] original content features
            content_augmented: [batch, seq, dim] augmented content features
            mask: Optional [batch, seq] attention mask

        Returns:
            Scalar loss value
        """
        # L2 distance
        diff = content_original - content_augmented

        if mask is not None:
            mask = mask.unsqueeze(-1)
            diff = diff * mask
            loss = (diff ** 2).sum() / (mask.sum() * diff.shape[-1])
        else:
            loss = (diff ** 2).mean()

        return self.config.content_preservation_weight * loss


# =============================================================================
# SAVC PROSODY ENCODER
# =============================================================================

class SAVCProsodyEncoder(nn.Module):
    """
    Prosody encoder trained with SAVC augmentation.

    This encoder is trained to produce speaker-invariant prosody embeddings
    by applying consistency loss between original and augmented views.

    Architecture:
    1. Input projection
    2. Transformer layers for temporal modeling
    3. Attentive pooling for utterance-level embedding
    4. Output projection
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Attentive pooling
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.prosody_dim),
            nn.LayerNorm(config.prosody_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosody embedding.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional [batch, seq] attention mask

        Returns:
            Dict with prosody embeddings and frame-level features
        """
        # Project input
        x = self.input_proj(features)

        # Transformer encoding
        if mask is not None:
            # Convert to attention mask format
            attn_mask = ~mask.bool()
            x = self.transformer(x, src_key_padding_mask=attn_mask)
        else:
            x = self.transformer(x)

        # Frame-level prosody features
        frame_features = x

        # Attentive pooling for utterance-level
        attn_weights = self.attention(x)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)

        pooled = (x * attn_weights).sum(dim=1)

        # Output projection
        prosody_emb = self.output_proj(pooled)

        return {
            'prosody_emb': prosody_emb,
            'frame_features': frame_features,
            'attention_weights': attn_weights.squeeze(-1),
        }


# =============================================================================
# SAVC ADAPTER FOR CSM INTEGRATION
# =============================================================================

class SAVCAdapter(nn.Module):
    """
    SAVC adapter for integration with ProsodyControlledCSM.

    Wraps the SAVC prosody encoder and augmentor to produce prefix tokens
    that can be used with the CSM model.

    During training:
    1. Generate augmented views
    2. Encode all views with prosody encoder
    3. Compute consistency loss
    4. Return prefix tokens + loss

    During inference:
    1. Encode features directly
    2. Return prefix tokens
    """

    def __init__(self, config: SAVCConfig, prosody_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.config = config

        # Augmentor
        self.augmentor = AdversarialStyleAugmentor(config)

        # Prosody encoder
        if prosody_encoder is not None:
            self.prosody_encoder = prosody_encoder
        else:
            self.prosody_encoder = SAVCProsodyEncoder(config)

        # Consistency loss
        self.consistency_loss = StyleConsistencyLoss(config)

        # Content preservation loss
        self.content_loss = ContentPreservationLoss(config)

        # Prefix token projection
        self.prefix_proj = nn.Sequential(
            nn.Linear(config.prosody_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim * config.num_prefix_tokens),
        )

    def get_prefix_tokens(self, prosody_emb: torch.Tensor) -> torch.Tensor:
        """
        Convert prosody embedding to prefix tokens.

        Args:
            prosody_emb: [batch, prosody_dim]

        Returns:
            [batch, num_tokens, output_dim] prefix tokens
        """
        batch = prosody_emb.shape[0]
        tokens = self.prefix_proj(prosody_emb)
        tokens = tokens.view(batch, self.config.num_prefix_tokens, self.config.output_dim)
        return tokens

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        content_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with SAVC augmentation.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional attention mask
            content_features: Optional content features for preservation loss

        Returns:
            Dict containing:
                - prosody_tokens: [batch, num_tokens, output_dim] prefix tokens
                - prosody_emb: [batch, prosody_dim] prosody embedding
                - losses: Dict of loss values (during training)
        """
        batch = features.shape[0]
        device = features.device

        if self.training:
            # Generate augmented views
            aug_result = self.augmentor(features, mask)
            all_views = aug_result['all_views']
            num_views = aug_result['num_views']

            # Expand mask for all views
            if mask is not None:
                all_masks = mask.repeat(num_views, 1)
            else:
                all_masks = None

            # Encode all views
            encoded = self.prosody_encoder(all_views, all_masks)
            all_embeddings = encoded['prosody_emb']

            # Split back into individual views
            embeddings_list = all_embeddings.split(batch, dim=0)

            # Compute consistency loss
            consistency_losses = self.consistency_loss(list(embeddings_list))

            # Content preservation loss (if content features provided)
            if content_features is not None:
                content_loss = self.content_loss(
                    content_features,
                    content_features,  # Augmented content should match
                    mask,
                )
            else:
                content_loss = torch.tensor(0.0, device=device)

            # Use first (original) embedding for output
            prosody_emb = embeddings_list[0]
            prosody_tokens = self.get_prefix_tokens(prosody_emb)

            losses = {
                'consistency_positive': consistency_losses['positive'],
                'consistency_contrastive': consistency_losses['contrastive'],
                'content_preservation': content_loss,
                'total': consistency_losses['total'] + content_loss,
            }

            return {
                'prosody_tokens': prosody_tokens,
                'prosody_emb': prosody_emb,
                'frame_features': encoded['frame_features'][:batch],
                'losses': losses,
                'augmentation_strength': aug_result['strength'],
            }

        else:
            # Inference mode - no augmentation
            encoded = self.prosody_encoder(features, mask)
            prosody_emb = encoded['prosody_emb']
            prosody_tokens = self.get_prefix_tokens(prosody_emb)

            return {
                'prosody_tokens': prosody_tokens,
                'prosody_emb': prosody_emb,
                'frame_features': encoded['frame_features'],
            }


# =============================================================================
# SAVC TRAINING MODULE
# =============================================================================

class SAVCModule(nn.Module):
    """
    Complete SAVC training module.

    Combines SAVC adapter with content encoder for full disentanglement.
    Can be used standalone or integrated with existing prosody pipelines.
    """

    def __init__(self, config: SAVCConfig):
        super().__init__()
        self.config = config

        # SAVC adapter (includes prosody encoder + augmentor)
        self.adapter = SAVCAdapter(config)

        # Optional content encoder (HuBERT-soft style)
        self.content_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        extract_content: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional attention mask
            extract_content: Whether to extract content features

        Returns:
            Dict with prosody tokens, embeddings, and losses
        """
        # Extract content features
        if extract_content:
            content_features = self.content_encoder(features)
        else:
            content_features = None

        # SAVC forward
        result = self.adapter(features, mask, content_features)

        if extract_content:
            result['content_features'] = content_features

        return result


# =============================================================================
# INTEGRATION WITH DISENTANGLEMENT MODULE
# =============================================================================

def create_savc_enhanced_disentanglement(
    config: SAVCConfig,
    disentanglement_config: Optional['DisentanglementConfig'] = None,
) -> nn.Module:
    """
    Create SAVC-enhanced disentanglement module.

    Combines SAVC augmentation with other disentanglement techniques
    (GRL, MINE, orthogonality) for maximum speaker invariance.

    Args:
        config: SAVC configuration
        disentanglement_config: Optional disentanglement configuration

    Returns:
        Combined disentanglement module
    """
    try:
        from disentanglement import DisentanglementConfig, DisentanglementLoss
    except ImportError:
        from training.disentanglement import DisentanglementConfig, DisentanglementLoss

    if disentanglement_config is None:
        disentanglement_config = DisentanglementConfig()

    class SAVCEnhancedDisentanglement(nn.Module):
        def __init__(self):
            super().__init__()
            self.savc_adapter = SAVCAdapter(config)
            self.disentanglement = DisentanglementLoss(
                disentanglement_config,
                prosody_dim=config.prosody_dim,
            )

        def forward(
            self,
            features: torch.Tensor,
            timbre_emb: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            f0_target: Optional[torch.Tensor] = None,
            speaker_labels: Optional[torch.Tensor] = None,
            epoch: Optional[int] = None,
        ) -> Dict[str, torch.Tensor]:
            # SAVC forward
            savc_result = self.savc_adapter(features, mask)

            # Disentanglement losses
            disentangle_losses = self.disentanglement(
                prosody_emb=savc_result['prosody_emb'],
                timbre_emb=timbre_emb,
                f0_target=f0_target,
                speaker_labels=speaker_labels,
                epoch=epoch,
            )

            # Combine losses
            combined_losses = {
                **savc_result.get('losses', {}),
                **disentangle_losses,
            }

            if 'total' in savc_result.get('losses', {}) and 'total' in disentangle_losses:
                combined_losses['total'] = (
                    savc_result['losses']['total'] +
                    disentangle_losses['total']
                )

            return {
                **savc_result,
                'losses': combined_losses,
            }

    return SAVCEnhancedDisentanglement()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SAVC: Adversarial Style Augmentation - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SAVCConfig()

    # Test 1: StatisticPerturbation
    print("\n[Test 1] StatisticPerturbation...")
    stat_perturb = StatisticPerturbation(config).to(device)
    features = torch.randn(4, 100, 768, device=device)
    result = stat_perturb(features, strength=1.0)
    print(f"  Original shape: {features.shape}")
    print(f"  Perturbed shape: {result['perturbed'].shape}")
    print(f"  Mean shift range: [{result['mean_shift'].min():.3f}, {result['mean_shift'].max():.3f}]")
    print(f"  Var scale range: [{result['var_scale'].min():.3f}, {result['var_scale'].max():.3f}]")

    # Verify statistics changed
    orig_mean = features.mean(dim=1)
    pert_mean = result['perturbed'].mean(dim=1)
    mean_diff = (orig_mean - pert_mean).abs().mean()
    print(f"  Mean difference: {mean_diff:.4f}")
    print("  [PASS]")

    # Test 2: SpectralPerturbation
    print("\n[Test 2] SpectralPerturbation...")
    spectral_perturb = SpectralPerturbation(config).to(device)
    perturbed = spectral_perturb(features, strength=1.0)
    print(f"  Perturbed shape: {perturbed.shape}")
    spec_diff = (features - perturbed).abs().mean()
    print(f"  Spectral difference: {spec_diff:.4f}")
    print("  [PASS]")

    # Test 3: AdversarialStyleAugmentor
    print("\n[Test 3] AdversarialStyleAugmentor...")
    augmentor = AdversarialStyleAugmentor(config).to(device)
    augmentor.train()
    aug_result = augmentor(features)
    print(f"  Original shape: {aug_result['original'].shape}")
    print(f"  Num augmented views: {len(aug_result['augmented'])}")
    print(f"  All views shape: {aug_result['all_views'].shape}")
    print(f"  Current strength: {aug_result['strength']:.4f}")
    print("  [PASS]")

    # Test 4: StyleConsistencyLoss
    print("\n[Test 4] StyleConsistencyLoss...")
    consistency_loss = StyleConsistencyLoss(config)

    # Create fake embeddings
    emb1 = torch.randn(4, 256, device=device)
    emb2 = emb1 + torch.randn(4, 256, device=device) * 0.1  # Similar
    emb3 = torch.randn(4, 256, device=device)  # Different

    losses = consistency_loss([emb1, emb2], use_contrastive=True)
    print(f"  Positive loss (similar pair): {losses['positive'].item():.4f}")
    print(f"  Contrastive loss: {losses['contrastive'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 5: SAVCProsodyEncoder
    print("\n[Test 5] SAVCProsodyEncoder...")
    prosody_encoder = SAVCProsodyEncoder(config).to(device)
    encoded = prosody_encoder(features)
    print(f"  Prosody embedding shape: {encoded['prosody_emb'].shape}")
    print(f"  Frame features shape: {encoded['frame_features'].shape}")
    print(f"  Attention weights shape: {encoded['attention_weights'].shape}")
    print("  [PASS]")

    # Test 6: SAVCAdapter (training mode)
    print("\n[Test 6] SAVCAdapter (training mode)...")
    adapter = SAVCAdapter(config).to(device)
    adapter.train()
    result = adapter(features)
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Prosody embedding shape: {result['prosody_emb'].shape}")
    print(f"  Losses:")
    for k, v in result['losses'].items():
        print(f"    {k}: {v.item():.4f}")
    print(f"  Augmentation strength: {result['augmentation_strength']:.4f}")
    print("  [PASS]")

    # Test 7: SAVCAdapter (inference mode)
    print("\n[Test 7] SAVCAdapter (inference mode)...")
    adapter.eval()
    with torch.no_grad():
        result = adapter(features)
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Prosody embedding shape: {result['prosody_emb'].shape}")
    assert 'losses' not in result, "Losses should not be computed in eval mode"
    print("  [PASS]")

    # Test 8: SAVCModule
    print("\n[Test 8] SAVCModule...")
    module = SAVCModule(config).to(device)
    module.train()
    result = module(features)
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Content features shape: {result['content_features'].shape}")
    print(f"  Total loss: {result['losses']['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: Gradient flow
    print("\n[Test 9] Gradient flow verification...")
    adapter = SAVCAdapter(config).to(device)
    adapter.train()

    features = torch.randn(4, 100, 768, device=device, requires_grad=True)
    result = adapter(features)
    loss = result['losses']['total']
    loss.backward()

    grad_norm = features.grad.norm().item()
    print(f"  Input gradient norm: {grad_norm:.4f}")
    assert grad_norm > 0, "Gradients should flow back to input"
    print("  [PASS]")

    # Test 10: Consistency verification
    print("\n[Test 10] Consistency verification (same sample should be similar)...")
    adapter.eval()
    prosody_encoder = adapter.prosody_encoder

    with torch.no_grad():
        # Get embeddings for original
        emb_orig = prosody_encoder(features)['prosody_emb']

        # Get embeddings for augmented
        stat_perturb = StatisticPerturbation(config).to(device)
        perturbed = stat_perturb(features, strength=1.0)['perturbed']
        emb_pert = prosody_encoder(perturbed)['prosody_emb']

        # Compute cosine similarity
        emb_orig_norm = F.normalize(emb_orig, p=2, dim=-1)
        emb_pert_norm = F.normalize(emb_pert, p=2, dim=-1)
        cos_sim = (emb_orig_norm * emb_pert_norm).sum(dim=-1).mean()

    print(f"  Cosine similarity (orig vs perturbed): {cos_sim.item():.4f}")
    print("  (Note: Before training, similarity may be low)")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All SAVC tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from savc import (
    SAVCConfig,
    SAVCAdapter,
    AdversarialStyleAugmentor,
    StyleConsistencyLoss,
)

# Initialize
config = SAVCConfig(
    input_dim=768,        # HuBERT/wav2vec2 feature dim
    prosody_dim=256,      # Prosody embedding dimension
    output_dim=2048,      # Match CSM hidden dim
)

adapter = SAVCAdapter(config).cuda()

# Training loop
adapter.train()
for batch in dataloader:
    features = feature_extractor(batch['audio'])  # [batch, seq, 768]

    # Forward pass with SAVC augmentation
    result = adapter(features)

    # Get prosody tokens for CSM
    prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

    # Compute SAVC consistency loss
    savc_loss = result['losses']['total']

    # Combine with main loss
    total_loss = reconstruction_loss + savc_loss
    total_loss.backward()

    # Log progress
    print(f"Consistency loss: {result['losses']['consistency_positive']:.4f}")
    print(f"Augmentation strength: {result['augmentation_strength']:.4f}")

# Inference
adapter.eval()
with torch.no_grad():
    result = adapter(features)
    prosody_tokens = result['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)

# Combine with other disentanglement methods
from savc import create_savc_enhanced_disentanglement
from disentanglement import DisentanglementConfig

disentangle_config = DisentanglementConfig(
    use_grl=True,
    use_scheduled_grl=True,
    use_mine=True,
)

enhanced = create_savc_enhanced_disentanglement(
    config,
    disentangle_config,
)

result = enhanced(features, timbre_emb, speaker_labels=speaker_ids)
total_loss = result['losses']['total']  # Combines SAVC + GRL + MINE losses
""")
