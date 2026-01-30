"""
EMORL-TTS: VAD-space Intensity with Local Emphasis Regulation

Based on EMORL-TTS (arXiv:2510.05758) - Emotion-Aware Reinforcement Learning for TTS

Key Innovation:
Unifies global emotion intensity control in VAD space with local word-level emphasis
regulation using reinforcement learning. This enables both:
- Sentence-level emotion strength control (via VAD coordinates)
- Word-level emphasis peaks for natural, expressive speech

Architecture:
1. Global Emotion Controller: VAD-based emotion intensity at utterance level
2. Local Emphasis Regulator: Word-level emphasis positions and strengths
3. Dual Reward System: Global VAD matching + Local emphasis clarity

Benefits:
- Independent control of global intensity vs local emphasis
- Improved emotional accuracy while preserving naturalness
- Measurable emphasis peaks at target word positions
- Smooth interpolation between emphasis levels

Usage:
    python train_emorl_tts.py --config config/emorl_tts.yaml \
        --checkpoint ../checkpoints/prosody_v7/best.pt \
        --manifest ../data/emotion_manifest.json
"""

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import base multi-reward RL components
from multi_reward_rl import (
    MultiRewardRLConfig,
    GRPOPolicy,
    GRPOLoss,
    CERRewardFunction,
    SIMRewardFunction,
    NaturalnessRewardFunction,
    collate_fn,
)

# =============================================================================
# EMORL-TTS CONFIGURATION
# =============================================================================

@dataclass
class EMORLConfig(MultiRewardRLConfig):
    """Extended configuration for EMORL-TTS with local emphasis regulation."""

    # =============================================================================
    # Global VAD Emotion Control
    # =============================================================================

    # VAD prediction hidden size
    vad_hidden_size: int = 256

    # Number of VAD prediction layers
    vad_num_layers: int = 2

    # VAD prototype smoothing (for soft emotion boundaries)
    vad_prototype_smoothing: float = 0.1

    # =============================================================================
    # Local Word-Level Emphasis Control
    # =============================================================================

    # Enable local emphasis regulation
    use_local_emphasis: bool = True

    # Maximum number of emphasis words per utterance
    max_emphasis_words: int = 4

    # Emphasis position tolerance (in word positions)
    emphasis_position_tolerance: float = 0.5

    # Emphasis strength bins (for discretized control)
    num_emphasis_levels: int = 5  # 0=none, 1=light, 2=medium, 3=strong, 4=very strong

    # Emphasis features dimension
    emphasis_feature_dim: int = 64

    # Temporal resolution for emphasis (frames per word)
    emphasis_temporal_resolution: int = 8

    # =============================================================================
    # Dual Reward System
    # =============================================================================

    # Global VAD matching reward weight
    global_vad_reward_weight: float = 1.0

    # Local emphasis clarity reward weight
    local_emphasis_reward_weight: float = 0.8

    # Emphasis position accuracy bonus
    emphasis_position_bonus: float = 0.3

    # Emphasis strength accuracy bonus
    emphasis_strength_bonus: float = 0.2

    # Naturalness preservation penalty for over-emphasis
    over_emphasis_penalty: float = 0.1

    # =============================================================================
    # Training Settings
    # =============================================================================

    # Use emphasis curriculum (start with strong emphasis, decay)
    use_emphasis_curriculum: bool = True
    emphasis_curriculum_epochs: int = 3

    # Emphasis data augmentation (random emphasis targets)
    emphasis_augmentation_prob: float = 0.2

    # Output
    output_dir: str = "checkpoints/emorl_tts"


# =============================================================================
# VAD PROTOTYPE BANK
# =============================================================================

class VADPrototypeBank:
    """
    VAD (Valence-Arousal-Dominance) prototypes for emotions.

    Based on Russell's Circumplex Model of Affect extended to 3D.
    Supports soft boundaries and emotion interpolation.
    """

    # Core emotion prototypes in VAD space
    PROTOTYPES = {
        'neutral': torch.tensor([0.0, 0.0, 0.0]),
        'happy': torch.tensor([0.8, 0.6, 0.6]),
        'joy': torch.tensor([0.9, 0.7, 0.5]),
        'excited': torch.tensor([0.7, 0.9, 0.6]),
        'sad': torch.tensor([-0.6, -0.4, -0.5]),
        'depressed': torch.tensor([-0.7, -0.5, -0.7]),
        'angry': torch.tensor([-0.5, 0.8, 0.7]),
        'rage': torch.tensor([-0.7, 0.95, 0.8]),
        'fearful': torch.tensor([-0.7, 0.7, -0.7]),
        'anxious': torch.tensor([-0.5, 0.6, -0.5]),
        'surprised': torch.tensor([0.3, 0.8, 0.2]),
        'shocked': torch.tensor([0.1, 0.95, 0.1]),
        'disgusted': torch.tensor([-0.6, 0.3, 0.4]),
        'contempt': torch.tensor([-0.4, 0.2, 0.6]),
        'calm': torch.tensor([0.4, -0.5, 0.3]),
        'relaxed': torch.tensor([0.5, -0.6, 0.4]),
        'tender': torch.tensor([0.7, -0.2, 0.1]),
        'bored': torch.tensor([-0.2, -0.6, -0.3]),
    }

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        self.prototype_tensor = self._build_prototype_tensor()
        self.emotion_names = list(self.PROTOTYPES.keys())

    def _build_prototype_tensor(self) -> torch.Tensor:
        """Build stacked tensor of all prototypes."""
        return torch.stack(list(self.PROTOTYPES.values()))

    def get_vad(self, emotion: str) -> torch.Tensor:
        """Get VAD coordinates for an emotion."""
        if emotion.lower() in self.PROTOTYPES:
            return self.PROTOTYPES[emotion.lower()].clone()
        return self.PROTOTYPES['neutral'].clone()

    def get_emotion_from_vad(self, vad: torch.Tensor) -> Tuple[str, float]:
        """Find closest emotion to VAD coordinates."""
        if vad.dim() == 1:
            vad = vad.unsqueeze(0)

        distances = torch.cdist(vad, self.prototype_tensor.to(vad.device))
        min_idx = distances.argmin(dim=-1)
        min_dist = distances.min(dim=-1).values

        emotion = self.emotion_names[min_idx.item()]
        confidence = 1.0 / (1.0 + min_dist.item())

        return emotion, confidence

    def interpolate_emotions(
        self,
        emotions: List[str],
        weights: List[float],
    ) -> torch.Tensor:
        """Interpolate between multiple emotions."""
        vads = torch.stack([self.get_vad(e) for e in emotions])
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()

        return (vads * weights.unsqueeze(-1)).sum(dim=0)

    def scale_intensity(
        self,
        vad: torch.Tensor,
        intensity: float,
        preserve_direction: bool = True,
    ) -> torch.Tensor:
        """Scale VAD intensity while optionally preserving direction."""
        if preserve_direction:
            norm = torch.norm(vad, dim=-1, keepdim=True) + 1e-8
            direction = vad / norm
            scaled_norm = norm * intensity
            return direction * scaled_norm
        else:
            return vad * intensity


# =============================================================================
# WORD-LEVEL EMPHASIS EXTRACTOR
# =============================================================================

class WordEmphasisExtractor(nn.Module):
    """
    Extracts word-level emphasis from prosody contours.

    Emphasis detection based on:
    - Energy peaks (louder = more emphasized)
    - F0 peaks (higher pitch = more emphasized)
    - Duration (longer = more emphasized)
    - Contrast with neighboring words
    """

    def __init__(self, config: EMORLConfig):
        super().__init__()
        self.config = config

        # Emphasis feature projection
        # Input: F0 + energy + duration features
        self.emphasis_projector = nn.Sequential(
            nn.Linear(32, config.emphasis_feature_dim),
            nn.GELU(),
            nn.Linear(config.emphasis_feature_dim, config.emphasis_feature_dim),
        )

        # Emphasis strength predictor (per word)
        self.strength_predictor = nn.Sequential(
            nn.Linear(config.emphasis_feature_dim, config.emphasis_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.emphasis_feature_dim // 2, config.num_emphasis_levels),
        )

        # Temporal attention for word boundaries
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.emphasis_feature_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

    def extract_from_contour(
        self,
        pitch_contour: torch.Tensor,  # [batch, time]
        energy_contour: torch.Tensor,  # [batch, time]
        word_boundaries: Optional[torch.Tensor] = None,  # [batch, num_words, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract emphasis features from prosody contours.

        Returns:
            emphasis_positions: [batch, max_emphasis_words] word indices
            emphasis_strengths: [batch, max_emphasis_words] strength levels 0-4
            emphasis_features: [batch, max_emphasis_words, feature_dim]
        """
        batch_size = pitch_contour.shape[0]
        device = pitch_contour.device
        time_len = pitch_contour.shape[1]

        # If no word boundaries, create uniform segments
        if word_boundaries is None:
            num_segments = 8  # Default 8 words
            segment_len = time_len // num_segments
            word_boundaries = torch.stack([
                torch.arange(num_segments, device=device) * segment_len,
                torch.arange(1, num_segments + 1, device=device) * segment_len,
            ], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)

        num_words = word_boundaries.shape[1]

        # Extract per-word features
        word_features = []

        for w in range(num_words):
            start = word_boundaries[:, w, 0].long()  # [batch]
            end = word_boundaries[:, w, 1].long()    # [batch]

            # Extract features for each sample in batch
            pitch_feats = []
            energy_feats = []

            for b in range(batch_size):
                s, e = start[b].item(), min(end[b].item(), time_len)
                if s >= e:
                    s, e = max(0, e - 1), e

                p_segment = pitch_contour[b, s:e]
                e_segment = energy_contour[b, s:e]

                # Compute word-level statistics
                pitch_stats = torch.stack([
                    p_segment.mean(),
                    p_segment.max() if len(p_segment) > 0 else torch.tensor(0.0, device=device),
                    p_segment.std() if len(p_segment) > 1 else torch.tensor(0.0, device=device),
                    (p_segment.max() - p_segment.min()) if len(p_segment) > 0 else torch.tensor(0.0, device=device),
                ])

                energy_stats = torch.stack([
                    e_segment.mean(),
                    e_segment.max() if len(e_segment) > 0 else torch.tensor(0.0, device=device),
                    e_segment.std() if len(e_segment) > 1 else torch.tensor(0.0, device=device),
                    (e_segment.max() - e_segment.min()) if len(e_segment) > 0 else torch.tensor(0.0, device=device),
                ])

                pitch_feats.append(pitch_stats)
                energy_feats.append(energy_stats)

            pitch_feats = torch.stack(pitch_feats)  # [batch, 4]
            energy_feats = torch.stack(energy_feats)  # [batch, 4]

            # Duration feature
            duration = (end - start).float() / time_len
            duration = duration.unsqueeze(-1)  # [batch, 1]

            # Combine features
            word_feat = torch.cat([
                pitch_feats,
                energy_feats,
                duration.expand(-1, 4),  # Pad to 4
            ], dim=-1)  # [batch, 12]

            # Pad to 32 (expected by projector)
            word_feat = F.pad(word_feat, (0, 32 - word_feat.shape[-1]))

            word_features.append(word_feat)

        word_features = torch.stack(word_features, dim=1)  # [batch, num_words, 32]

        # Project to emphasis features
        emphasis_features = self.emphasis_projector(word_features)  # [batch, num_words, feature_dim]

        # Apply temporal attention for context
        emphasis_features, _ = self.temporal_attention(
            emphasis_features, emphasis_features, emphasis_features
        )

        # Predict emphasis strength per word
        strength_logits = self.strength_predictor(emphasis_features)  # [batch, num_words, num_levels]
        strength_probs = F.softmax(strength_logits, dim=-1)
        emphasis_strengths = strength_probs.argmax(dim=-1)  # [batch, num_words]

        # Find top-k emphasized words
        max_emphasis = min(self.config.max_emphasis_words, num_words)
        strength_scores = emphasis_strengths.float()

        # Add small noise for tie-breaking
        strength_scores = strength_scores + torch.rand_like(strength_scores) * 0.01

        # Get top-k positions
        _, top_indices = torch.topk(strength_scores, max_emphasis, dim=-1)

        # Gather features and strengths for top words
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, max_emphasis)
        top_features = emphasis_features[batch_indices, top_indices]  # [batch, max_emphasis, feature_dim]
        top_strengths = emphasis_strengths[batch_indices, top_indices]  # [batch, max_emphasis]

        return {
            'emphasis_positions': top_indices,  # [batch, max_emphasis_words]
            'emphasis_strengths': top_strengths,  # [batch, max_emphasis_words]
            'emphasis_features': top_features,  # [batch, max_emphasis_words, feature_dim]
            'all_strengths': emphasis_strengths,  # [batch, num_words]
            'strength_logits': strength_logits,  # [batch, num_words, num_levels]
        }


# =============================================================================
# LOCAL EMPHASIS REGULATOR
# =============================================================================

class LocalEmphasisRegulator(nn.Module):
    """
    Regulates local word-level emphasis in prosody embeddings.

    Takes target emphasis positions and strengths, and modifies
    prosody embeddings to produce the desired emphasis pattern.
    """

    def __init__(self, config: EMORLConfig):
        super().__init__()
        self.config = config

        # Emphasis embedding (learnable per strength level)
        self.emphasis_embeddings = nn.Embedding(
            config.num_emphasis_levels,
            config.emphasis_feature_dim,
        )

        # Position encoding for emphasis positions
        self.position_encoding = nn.Embedding(
            config.max_emphasis_words * 4,  # Support up to 4x max words
            config.emphasis_feature_dim,
        )

        # Emphasis-to-prosody projection
        self.emphasis_to_prosody = nn.Sequential(
            nn.Linear(config.emphasis_feature_dim * 2, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        # Prosody modulation network
        self.modulation_network = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),  # Output in [-1, 1] for modulation
        )

        # Gating network (controls emphasis strength)
        self.emphasis_gate = nn.Sequential(
            nn.Linear(config.emphasis_feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        target_emphasis_positions: torch.Tensor,  # [batch, max_emphasis_words]
        target_emphasis_strengths: torch.Tensor,  # [batch, max_emphasis_words]
    ) -> torch.Tensor:
        """
        Apply emphasis regulation to prosody embeddings.

        Args:
            prosody_embedding: Base prosody embeddings from encoder
            target_emphasis_positions: Word indices to emphasize
            target_emphasis_strengths: Strength levels (0-4) for each position

        Returns:
            Modified prosody embeddings with emphasis
        """
        batch_size, num_tokens, hidden_size = prosody_embedding.shape
        device = prosody_embedding.device

        # Get emphasis embeddings
        emphasis_emb = self.emphasis_embeddings(target_emphasis_strengths)  # [batch, max_emp, feat]

        # Add position encoding
        positions_clamped = target_emphasis_positions.clamp(0, self.position_encoding.num_embeddings - 1)
        position_emb = self.position_encoding(positions_clamped)  # [batch, max_emp, feat]

        # Combine emphasis and position
        combined_emb = torch.cat([emphasis_emb, position_emb], dim=-1)  # [batch, max_emp, feat*2]

        # Project to prosody space
        emphasis_prosody = self.emphasis_to_prosody(combined_emb)  # [batch, max_emp, hidden]

        # Compute emphasis gate (how much to apply)
        gate = self.emphasis_gate(emphasis_emb)  # [batch, max_emp, 1]

        # Aggregate emphasis signal across all emphasis words
        emphasis_signal = (emphasis_prosody * gate).mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        emphasis_signal = emphasis_signal.expand(-1, num_tokens, -1)  # [batch, num_tokens, hidden]

        # Compute modulation
        combined = torch.cat([prosody_embedding, emphasis_signal], dim=-1)
        modulation = self.modulation_network(combined)  # [batch, num_tokens, hidden]

        # Apply modulation (residual style)
        output = prosody_embedding + modulation * 0.1  # Scale down modulation

        return output


# =============================================================================
# EMPHASIS REWARD FUNCTION
# =============================================================================

class EmphasisRewardFunction(nn.Module):
    """
    Reward function for local emphasis clarity.

    Measures:
    1. Position accuracy: Are emphasis peaks at target positions?
    2. Strength accuracy: Are emphasis strengths correct?
    3. Contrast: Is there clear contrast between emphasized and non-emphasized?
    4. Naturalness: Does emphasis sound natural (not over-exaggerated)?
    """

    def __init__(self, config: EMORLConfig):
        super().__init__()
        self.config = config

        # Emphasis strength predictor from prosody embeddings
        self.strength_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.emphasis_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.emphasis_feature_dim, config.num_emphasis_levels),
        )

        # Emphasis detection network
        self.emphasis_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.emphasis_feature_dim),
            nn.GELU(),
            nn.Linear(config.emphasis_feature_dim, 1),
            nn.Sigmoid(),  # Emphasis probability
        )

    def compute_reward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        target_positions: torch.Tensor,   # [batch, max_emphasis_words]
        target_strengths: torch.Tensor,   # [batch, max_emphasis_words]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute emphasis clarity reward.

        Returns dict with:
            position_reward: Accuracy of emphasis positions
            strength_reward: Accuracy of emphasis strengths
            contrast_reward: Clarity of emphasis contrast
            total_reward: Combined reward
        """
        batch_size, num_tokens, hidden_size = prosody_embedding.shape
        device = prosody_embedding.device

        # Predict emphasis probability per token
        emphasis_probs = self.emphasis_detector(prosody_embedding).squeeze(-1)  # [batch, num_tokens]

        # Predict emphasis strength per token
        strength_logits = self.strength_predictor(prosody_embedding)  # [batch, num_tokens, num_levels]
        strength_probs = F.softmax(strength_logits, dim=-1)

        # Position reward: Do predicted emphasis peaks align with targets?
        position_reward = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            # Get predicted peak positions
            top_k = min(self.config.max_emphasis_words, num_tokens)
            _, pred_positions = torch.topk(emphasis_probs[b], top_k)

            # Compare with target positions (with tolerance)
            target_pos = target_positions[b]

            for t_pos in target_pos:
                if t_pos < 0:
                    continue
                # Scale target position to token space
                t_pos_scaled = (t_pos.float() / 8 * num_tokens).long().clamp(0, num_tokens - 1)

                # Check if any predicted position is within tolerance
                distances = (pred_positions.float() - t_pos_scaled.float()).abs()
                min_dist = distances.min()
                tolerance = self.config.emphasis_position_tolerance * num_tokens / 8

                if min_dist <= tolerance:
                    position_reward[b] += 1.0 / len(target_pos[target_pos >= 0])

        # Strength reward: Do emphasis strengths match targets?
        strength_reward = torch.zeros(batch_size, device=device)

        for b in range(batch_size):
            for i, (pos, strength) in enumerate(zip(target_positions[b], target_strengths[b])):
                if pos < 0:
                    continue
                # Scale position to token space
                pos_scaled = (pos.float() / 8 * num_tokens).long().clamp(0, num_tokens - 1)

                # Get predicted strength at this position
                pred_strength_prob = strength_probs[b, pos_scaled]
                strength_val = strength.clamp(0, self.config.num_emphasis_levels - 1)

                # Reward based on probability of correct strength
                strength_reward[b] += pred_strength_prob[strength_val]

        # Normalize strength reward
        num_targets = (target_positions >= 0).float().sum(dim=-1).clamp(min=1)
        strength_reward = strength_reward / num_targets

        # Contrast reward: Is there clear distinction between emphasized and non-emphasized?
        # High variance in emphasis probabilities = good contrast
        emphasis_mean = emphasis_probs.mean(dim=-1, keepdim=True)
        emphasis_var = ((emphasis_probs - emphasis_mean) ** 2).mean(dim=-1)
        contrast_reward = torch.tanh(emphasis_var * 10)  # Scale to [0, 1]

        # Over-emphasis penalty: Penalize if too many tokens have high emphasis
        num_emphasized = (emphasis_probs > 0.5).float().sum(dim=-1)
        max_emphasized = self.config.max_emphasis_words * 2  # Allow some slack
        over_emphasis = F.relu(num_emphasized - max_emphasized) / num_tokens

        # Total reward
        total_reward = (
            self.config.emphasis_position_bonus * position_reward +
            self.config.emphasis_strength_bonus * strength_reward +
            self.config.local_emphasis_reward_weight * contrast_reward -
            self.config.over_emphasis_penalty * over_emphasis
        )

        return {
            'position_reward': position_reward,
            'strength_reward': strength_reward,
            'contrast_reward': contrast_reward,
            'over_emphasis_penalty': over_emphasis,
            'total_reward': total_reward,
        }


# =============================================================================
# GLOBAL VAD REWARD FUNCTION
# =============================================================================

class GlobalVADRewardFunction(nn.Module):
    """
    Reward function for global VAD-based emotion intensity.

    Measures how well the prosody embedding matches the target
    VAD coordinates at the utterance level.
    """

    def __init__(self, config: EMORLConfig):
        super().__init__()
        self.config = config
        self.vad_bank = VADPrototypeBank(smoothing=config.vad_prototype_smoothing)

        # VAD predictor from prosody embeddings
        self.vad_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.vad_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            *[
                nn.Sequential(
                    nn.Linear(config.vad_hidden_size, config.vad_hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(config.vad_num_layers - 1)
            ],
            nn.Linear(config.vad_hidden_size, 3),  # V, A, D
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Intensity predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.vad_hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.vad_hidden_size // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Emotion classifier for auxiliary supervision
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.vad_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.vad_hidden_size, 8),  # 8 basic emotions
        )

    def compute_reward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        target_vad: torch.Tensor,         # [batch, 3]
        target_intensity: torch.Tensor,   # [batch]
        target_emotion: Optional[torch.Tensor] = None,  # [batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute global VAD matching reward.

        Returns dict with:
            vad_reward: VAD coordinate matching reward
            intensity_reward: Intensity matching reward
            emotion_reward: Emotion classification accuracy
            total_reward: Combined reward
        """
        device = prosody_embedding.device
        batch_size = prosody_embedding.shape[0]

        # Pool over tokens for global representation
        pooled = prosody_embedding.mean(dim=1)  # [batch, hidden]

        # Predict VAD
        predicted_vad = self.vad_predictor(pooled)  # [batch, 3]

        # VAD reward: Cosine similarity
        vad_sim = F.cosine_similarity(predicted_vad, target_vad.to(device), dim=-1)
        vad_reward = (vad_sim + 1.0) / 2.0  # Normalize to [0, 1]

        # Predict intensity
        predicted_intensity = self.intensity_predictor(pooled).squeeze(-1)  # [batch]

        # Intensity reward: Negative L1 distance
        intensity_error = (predicted_intensity - target_intensity.to(device)).abs()
        intensity_reward = 1.0 - intensity_error

        # Emotion classification reward
        emotion_reward = torch.zeros(batch_size, device=device)
        if target_emotion is not None:
            emotion_logits = self.emotion_classifier(pooled)
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            target_emotion = target_emotion.to(device)
            valid_mask = target_emotion >= 0
            if valid_mask.any():
                emotion_reward[valid_mask] = emotion_probs[valid_mask].gather(
                    1, target_emotion[valid_mask].unsqueeze(1)
                ).squeeze(1)

        # Total reward
        total_reward = (
            self.config.global_vad_reward_weight * vad_reward +
            self.config.intensity_reward_weight * intensity_reward +
            0.3 * emotion_reward
        )

        return {
            'vad_reward': vad_reward,
            'intensity_reward': intensity_reward,
            'emotion_reward': emotion_reward,
            'predicted_vad': predicted_vad,
            'predicted_intensity': predicted_intensity,
            'total_reward': total_reward,
        }


# =============================================================================
# EMORL-TTS REWARD AGGREGATOR
# =============================================================================

class EMORLRewardAggregator(nn.Module):
    """
    Aggregates EMORL-TTS dual reward system:
    1. Global VAD-based emotion intensity
    2. Local word-level emphasis clarity
    3. Base rewards (CER, SIM, Naturalness)
    """

    def __init__(self, config: EMORLConfig):
        super().__init__()
        self.config = config

        # Base reward functions
        self.cer_reward = CERRewardFunction(config)
        self.sim_reward = SIMRewardFunction(config)
        self.naturalness_reward = NaturalnessRewardFunction(config)

        # EMORL-TTS specific rewards
        self.global_vad_reward = GlobalVADRewardFunction(config)
        self.emphasis_reward = EmphasisRewardFunction(config)

        # Adaptive weighting
        self.register_buffer('reward_weights', torch.ones(5))

    def compute_rewards(
        self,
        prosody_embedding: torch.Tensor,
        target_vad: torch.Tensor,
        target_intensity: torch.Tensor,
        target_emotion: Optional[torch.Tensor] = None,
        target_emphasis_positions: Optional[torch.Tensor] = None,
        target_emphasis_strengths: Optional[torch.Tensor] = None,
        target_text: Optional[List[str]] = None,
        reference_speaker_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all rewards."""
        device = prosody_embedding.device
        batch_size = prosody_embedding.shape[0]

        rewards = {}

        # Base rewards
        rewards['cer'] = self.cer_reward.compute_reward(prosody_embedding)
        rewards['sim'] = self.sim_reward.compute_reward(
            prosody_embedding, reference_speaker_emb=reference_speaker_emb
        )
        rewards['naturalness'] = self.naturalness_reward.compute_reward(prosody_embedding)

        # Global VAD reward
        global_vad_results = self.global_vad_reward.compute_reward(
            prosody_embedding,
            target_vad,
            target_intensity,
            target_emotion,
        )
        rewards['global_vad'] = global_vad_results['total_reward']
        rewards['vad_detail'] = global_vad_results

        # Local emphasis reward
        if self.config.use_local_emphasis and target_emphasis_positions is not None:
            emphasis_results = self.emphasis_reward.compute_reward(
                prosody_embedding,
                target_emphasis_positions,
                target_emphasis_strengths,
            )
            rewards['emphasis'] = emphasis_results['total_reward']
            rewards['emphasis_detail'] = emphasis_results
        else:
            rewards['emphasis'] = torch.zeros(batch_size, device=device)

        # Total reward (weighted sum)
        weights = F.softmax(self.reward_weights, dim=0)
        rewards['total'] = (
            weights[0] * rewards['cer'] +
            weights[1] * rewards['sim'] +
            weights[2] * rewards['naturalness'] +
            weights[3] * rewards['global_vad'] +
            weights[4] * rewards['emphasis']
        )

        return rewards


# =============================================================================
# EMORL-TTS POLICY
# =============================================================================

class EMORLPolicy(nn.Module):
    """
    EMORL-TTS Policy combining:
    1. Base GRPO policy for prosody generation
    2. Local emphasis regulator for word-level control
    """

    def __init__(self, prosody_encoder: nn.Module, config: EMORLConfig):
        super().__init__()
        self.config = config

        # Base GRPO policy
        self.grpo_policy = GRPOPolicy(prosody_encoder, config)

        # Local emphasis regulator
        self.emphasis_regulator = LocalEmphasisRegulator(config)

    def forward(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
        target_emphasis_positions: Optional[torch.Tensor] = None,
        target_emphasis_strengths: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate prosody with optional emphasis regulation.

        Returns:
            embedding: [batch, num_tokens, hidden] with emphasis applied
            log_prob: [batch]
            value: [batch]
        """
        # Get base prosody from GRPO policy
        embedding, log_prob, value = self.grpo_policy(
            semantic, acoustic, rhythm, contour, deterministic=deterministic
        )

        # Apply emphasis regulation if targets provided
        if target_emphasis_positions is not None and target_emphasis_strengths is not None:
            embedding = self.emphasis_regulator(
                embedding,
                target_emphasis_positions,
                target_emphasis_strengths,
            )

        return embedding, log_prob, value

    def sample_group(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
        target_emphasis_positions: Optional[torch.Tensor] = None,
        target_emphasis_strengths: Optional[torch.Tensor] = None,
        group_size: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample group of prosody embeddings with emphasis."""
        if group_size is None:
            group_size = self.config.group_size

        batch_size = semantic.shape[0]

        # Expand inputs for group sampling
        semantic_exp = semantic.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, semantic.shape[-1])
        acoustic_exp = acoustic.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, acoustic.shape[-1])
        rhythm_exp = rhythm.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, rhythm.shape[-1])
        contour_exp = contour.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, contour.shape[-1])

        # Expand emphasis targets if provided
        emp_pos_exp = None
        emp_str_exp = None
        if target_emphasis_positions is not None:
            emp_pos_exp = target_emphasis_positions.unsqueeze(1).expand(
                -1, group_size, -1
            ).reshape(-1, target_emphasis_positions.shape[-1])
            emp_str_exp = target_emphasis_strengths.unsqueeze(1).expand(
                -1, group_size, -1
            ).reshape(-1, target_emphasis_strengths.shape[-1])

        # Sample all at once
        samples, log_probs, values = self.forward(
            semantic_exp, acoustic_exp, rhythm_exp, contour_exp,
            target_emphasis_positions=emp_pos_exp,
            target_emphasis_strengths=emp_str_exp,
            deterministic=False,
        )

        # Reshape to [batch, group_size, ...]
        num_tokens = samples.shape[1]
        hidden = samples.shape[2]

        samples = samples.view(batch_size, group_size, num_tokens, hidden)
        log_probs = log_probs.view(batch_size, group_size)
        values = values.view(batch_size, group_size)

        return samples, log_probs, values


# =============================================================================
# EMORL-TTS TRAINER
# =============================================================================

class EMORLTrainer:
    """
    Trainer for EMORL-TTS with dual reward system.

    Training phases:
    1. Warm-up: SFT-heavy, learn base prosody
    2. Emphasis curriculum: Gradually introduce emphasis targets
    3. RL fine-tuning: GRPO with dual rewards
    """

    def __init__(
        self,
        config: EMORLConfig,
        prosody_encoder: nn.Module,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or self._setup_device()

        # Policy
        self.policy = EMORLPolicy(prosody_encoder, config).to(self.device)

        # Reference policy (frozen)
        if config.use_reference_model:
            import copy
            self.ref_policy = copy.deepcopy(self.policy)
            self.ref_policy.eval()
            for param in self.ref_policy.parameters():
                param.requires_grad = False
            print("Created frozen reference policy for EMORL-TTS")
        else:
            self.ref_policy = None

        # Reward aggregator
        self.reward_aggregator = EMORLRewardAggregator(config).to(self.device)

        # Word emphasis extractor
        self.emphasis_extractor = WordEmphasisExtractor(config).to(self.device)

        # GRPO loss
        self.grpo_loss = GRPOLoss(config)

        # SFT loss
        self.mse_loss = nn.MSELoss()

        # Optimizer
        trainable_params = list(self.policy.parameters())
        trainable_params += list(self.reward_aggregator.parameters())
        trainable_params += list(self.emphasis_extractor.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_reward = float('-inf')
        self.current_sft_weight = config.sft_weight

        # Emphasis curriculum state
        self.emphasis_curriculum_active = config.use_emphasis_curriculum
        self.emphasis_weight_current = 0.1 if config.use_emphasis_curriculum else 1.0

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _update_emphasis_curriculum(self):
        """Update emphasis weight based on curriculum."""
        if not self.emphasis_curriculum_active:
            return

        if self.current_epoch < self.config.emphasis_curriculum_epochs:
            # Linearly increase emphasis weight
            progress = (self.current_epoch + 1) / self.config.emphasis_curriculum_epochs
            self.emphasis_weight_current = 0.1 + 0.9 * progress
        else:
            self.emphasis_weight_current = 1.0
            self.emphasis_curriculum_active = False

    def _extract_or_augment_emphasis(
        self,
        pitch_contour: torch.Tensor,
        energy_contour: torch.Tensor,
        augment: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract emphasis from contours or generate augmented targets."""
        batch_size = pitch_contour.shape[0]
        device = pitch_contour.device

        # Extract emphasis from prosody contours
        emphasis_results = self.emphasis_extractor.extract_from_contour(
            pitch_contour, energy_contour
        )

        positions = emphasis_results['emphasis_positions']
        strengths = emphasis_results['emphasis_strengths']

        # Optional augmentation: randomly modify emphasis targets
        if augment and random.random() < self.config.emphasis_augmentation_prob:
            # Random position shift
            pos_shift = torch.randint(-1, 2, positions.shape, device=device)
            positions = (positions + pos_shift).clamp(0, 7)

            # Random strength variation
            str_shift = torch.randint(-1, 2, strengths.shape, device=device)
            strengths = (strengths + str_shift).clamp(0, self.config.num_emphasis_levels - 1)

        return positions, strengths

    def _compute_sft_loss(
        self,
        prosody_embedding: torch.Tensor,
        target_prosody: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute SFT reconstruction loss."""
        target = torch.cat([
            target_prosody['semantic'],
            target_prosody['acoustic'],
            target_prosody['rhythm'],
            target_prosody['contour'][:, :8] if target_prosody['contour'].shape[1] > 8
            else target_prosody['contour'],
        ], dim=-1)

        if target.shape[-1] < prosody_embedding.shape[-1]:
            target = F.pad(target, (0, prosody_embedding.shape[-1] - target.shape[-1]))

        target = target.unsqueeze(1).expand_as(prosody_embedding[:, :1, :])

        return self.mse_loss(
            prosody_embedding[:, 0, :target.shape[-1]],
            target[:, 0, :]
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single EMORL-TTS training step."""
        self.policy.train()
        self.reward_aggregator.train()
        self.emphasis_extractor.train()

        # Move to device
        prosody_dict = {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }

        target_vad = batch['target_vad'].to(self.device)
        target_intensity = batch['target_intensity'].to(self.device)
        target_emotion = batch.get('emotion_label')
        if target_emotion is not None:
            target_emotion = target_emotion.to(self.device)

        batch_size = prosody_dict['semantic'].shape[0]
        group_size = self.config.group_size

        # Extract or get emphasis targets
        if 'emphasis_positions' in batch:
            emphasis_positions = batch['emphasis_positions'].to(self.device)
            emphasis_strengths = batch['emphasis_strengths'].to(self.device)
        else:
            # Extract from prosody contour (use contour as proxy for pitch)
            pitch_contour = prosody_dict['contour']
            # Create energy proxy from acoustic features
            energy_contour = prosody_dict['acoustic'][:, 0:1].expand(-1, 32)

            emphasis_positions, emphasis_strengths = self._extract_or_augment_emphasis(
                pitch_contour, energy_contour,
                augment=self.training
            )

        # Sample group from current policy
        samples, log_probs, values = self.policy.sample_group(
            prosody_dict['semantic'],
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
            target_emphasis_positions=emphasis_positions,
            target_emphasis_strengths=emphasis_strengths,
            group_size=group_size,
        )

        # Get reference log probs
        with torch.no_grad():
            if self.ref_policy is not None:
                _, ref_log_probs, _ = self.ref_policy.sample_group(
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                    target_emphasis_positions=emphasis_positions,
                    target_emphasis_strengths=emphasis_strengths,
                    group_size=group_size,
                )
            else:
                ref_log_probs = log_probs.detach()

        # Compute rewards for each sample in group
        samples_flat = samples.view(-1, samples.shape[2], samples.shape[3])

        # Expand targets
        target_vad_exp = target_vad.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, 3)
        target_intensity_exp = target_intensity.unsqueeze(1).expand(-1, group_size).reshape(-1)
        target_emotion_exp = None
        if target_emotion is not None:
            target_emotion_exp = target_emotion.unsqueeze(1).expand(-1, group_size).reshape(-1)
        emphasis_positions_exp = emphasis_positions.unsqueeze(1).expand(
            -1, group_size, -1
        ).reshape(-1, emphasis_positions.shape[-1])
        emphasis_strengths_exp = emphasis_strengths.unsqueeze(1).expand(
            -1, group_size, -1
        ).reshape(-1, emphasis_strengths.shape[-1])

        # Compute rewards
        rewards_dict = self.reward_aggregator.compute_rewards(
            samples_flat,
            target_vad=target_vad_exp,
            target_intensity=target_intensity_exp,
            target_emotion=target_emotion_exp,
            target_emphasis_positions=emphasis_positions_exp,
            target_emphasis_strengths=emphasis_strengths_exp,
        )

        # Apply emphasis curriculum weight
        rewards_total = rewards_dict['total']
        emphasis_contribution = rewards_dict['emphasis'] * (1 - self.emphasis_weight_current)
        rewards_adjusted = rewards_total - emphasis_contribution + \
                          rewards_dict['emphasis'] * self.emphasis_weight_current

        rewards = rewards_adjusted.view(batch_size, group_size)

        # Compute GRPO loss
        grpo_losses = self.grpo_loss(log_probs, ref_log_probs, rewards, values)

        # SFT regularization
        with torch.no_grad():
            det_output, _, _ = self.policy(
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'],
                target_emphasis_positions=emphasis_positions,
                target_emphasis_strengths=emphasis_strengths,
                deterministic=True,
            )
        sft_loss = self._compute_sft_loss(det_output, prosody_dict)

        # Combined loss
        total_loss = grpo_losses['total'] + self.current_sft_weight * sft_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()
        self.global_step += 1

        return {
            'total_loss': total_loss.item(),
            'policy_loss': grpo_losses['policy_loss'].item(),
            'kl_loss': grpo_losses['kl_loss'].item(),
            'sft_loss': sft_loss.item(),
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'global_vad_reward': rewards_dict['global_vad'].mean().item(),
            'emphasis_reward': rewards_dict['emphasis'].mean().item(),
            'cer_reward': rewards_dict['cer'].mean().item(),
            'sim_reward': rewards_dict['sim'].mean().item(),
            'naturalness_reward': rewards_dict['naturalness'].mean().item(),
            'emphasis_weight': self.emphasis_weight_current,
            'kl': grpo_losses['kl'].item(),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.policy.eval()
        self.reward_aggregator.eval()

        total_metrics = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                prosody_dict = {
                    'semantic': batch['prosody_semantic'].to(self.device),
                    'acoustic': batch['prosody_acoustic'].to(self.device),
                    'rhythm': batch['prosody_rhythm'].to(self.device),
                    'contour': batch['prosody_contour'].to(self.device),
                }

                target_vad = batch['target_vad'].to(self.device)
                target_intensity = batch['target_intensity'].to(self.device)
                target_emotion = batch.get('emotion_label')
                if target_emotion is not None:
                    target_emotion = target_emotion.to(self.device)

                # Extract emphasis
                emphasis_positions, emphasis_strengths = self._extract_or_augment_emphasis(
                    prosody_dict['contour'],
                    prosody_dict['acoustic'][:, 0:1].expand(-1, 32),
                    augment=False,
                )

                # Deterministic output
                output, _, _ = self.policy(
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                    target_emphasis_positions=emphasis_positions,
                    target_emphasis_strengths=emphasis_strengths,
                    deterministic=True,
                )

                # Compute rewards
                rewards_dict = self.reward_aggregator.compute_rewards(
                    output,
                    target_vad=target_vad,
                    target_intensity=target_intensity,
                    target_emotion=target_emotion,
                    target_emphasis_positions=emphasis_positions,
                    target_emphasis_strengths=emphasis_strengths,
                )

                total_metrics['reward'] += rewards_dict['total'].mean().item()
                total_metrics['global_vad_reward'] += rewards_dict['global_vad'].mean().item()
                total_metrics['emphasis_reward'] += rewards_dict['emphasis'].mean().item()
                total_metrics['naturalness_reward'] += rewards_dict['naturalness'].mean().item()
                num_batches += 1

        for key in total_metrics:
            total_metrics[key] /= max(1, num_batches)

        return dict(total_metrics)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Main training loop."""
        print(f"\nStarting EMORL-TTS training for {self.config.num_epochs} epochs")
        print(f"  GRPO group size: {self.config.group_size}")
        print(f"  Global VAD weight: {self.config.global_vad_reward_weight}")
        print(f"  Local emphasis weight: {self.config.local_emphasis_reward_weight}")
        print(f"  Emphasis curriculum: {self.config.use_emphasis_curriculum}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self._update_emphasis_curriculum()

            epoch_metrics = defaultdict(float)
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1

                if self.global_step % self.config.log_every == 0:
                    print(f"  Step {self.global_step}: "
                          f"loss={metrics['total_loss']:.4f}, "
                          f"reward={metrics['reward_mean']:.4f}, "
                          f"vad={metrics['global_vad_reward']:.4f}, "
                          f"emph={metrics['emphasis_reward']:.4f}")

            # Epoch summary
            for key in epoch_metrics:
                epoch_metrics[key] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            print(f"  Train - loss: {epoch_metrics['total_loss']:.4f}, "
                  f"reward: {epoch_metrics['reward_mean']:.4f}")
            print(f"  Rewards - VAD: {epoch_metrics['global_vad_reward']:.4f}, "
                  f"Emphasis: {epoch_metrics['emphasis_reward']:.4f}, "
                  f"Natural: {epoch_metrics['naturalness_reward']:.4f}")
            print(f"  Emphasis curriculum weight: {self.emphasis_weight_current:.3f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"  Val   - reward: {val_metrics['reward']:.4f}, "
                      f"vad: {val_metrics['global_vad_reward']:.4f}, "
                      f"emph: {val_metrics['emphasis_reward']:.4f}")

                if val_metrics['reward'] > self.best_reward:
                    self.best_reward = val_metrics['reward']
                    self.save_checkpoint('best')

            # Decay SFT weight
            self.current_sft_weight *= self.config.sft_decay
            print(f"  SFT weight decayed to: {self.current_sft_weight:.4f}")

            if (epoch + 1) % self.config.save_every_epochs == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')

        self.save_checkpoint('final')
        print(f"\nTraining complete! Best reward: {self.best_reward:.4f}")

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_reward': self.best_reward,
            'current_sft_weight': self.current_sft_weight,
            'emphasis_weight_current': self.emphasis_weight_current,
            'policy': self.policy.state_dict(),
            'reward_aggregator': self.reward_aggregator.state_dict(),
            'emphasis_extractor': self.emphasis_extractor.state_dict(),
            'config': {
                'group_size': self.config.group_size,
                'global_vad_reward_weight': self.config.global_vad_reward_weight,
                'local_emphasis_reward_weight': self.config.local_emphasis_reward_weight,
                'use_local_emphasis': self.config.use_local_emphasis,
                'max_emphasis_words': self.config.max_emphasis_words,
                'num_emphasis_levels': self.config.num_emphasis_levels,
            },
        }

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")


# =============================================================================
# DATASET ADAPTER
# =============================================================================

class EMORLDataset(Dataset):
    """
    Dataset for EMORL-TTS training.

    Extends base dataset with word-level emphasis annotations.
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        config: EMORLConfig,
    ):
        self.config = config
        self.prosody_cache_dir = Path(prosody_cache_dir)

        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples for EMORL-TTS training")

        # VAD prototypes
        self.vad_bank = VADPrototypeBank()

    def __len__(self):
        return len(self.samples)

    def _get_emotion_label(self, sample: dict) -> int:
        """Extract emotion label."""
        emotion_to_idx = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
        }

        emotion = sample.get('emotion', '').lower()
        if not emotion:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion', '').lower()
        if not emotion:
            emotions = sample.get('prosody', {}).get('semantic', {}).get('emotions', {})
            if emotions:
                emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()

        return emotion_to_idx.get(emotion, -1)

    def _get_vad(self, sample: dict) -> torch.Tensor:
        """Get VAD coordinates."""
        emotion_label = self._get_emotion_label(sample)
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted', 'calm']

        if 0 <= emotion_label < len(emotion_names):
            vad = self.vad_bank.get_vad(emotion_names[emotion_label])
        else:
            vad = self.vad_bank.get_vad('neutral')

        return vad

    def _get_intensity(self, sample: dict) -> float:
        """Extract intensity."""
        acoustic = sample.get('prosody', {}).get('acoustic', {})
        if isinstance(acoustic, dict):
            return acoustic.get('intensity_mean', 0.5)
        return 0.5

    def _get_emphasis_annotations(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract or derive word-level emphasis annotations.

        Priority:
        1. Explicit annotations in sample
        2. Derived from prosody contour peaks
        3. Default (no emphasis)
        """
        # Check for explicit annotations
        if 'emphasis' in sample:
            emphasis = sample['emphasis']
            positions = torch.tensor(emphasis.get('positions', [0]), dtype=torch.long)
            strengths = torch.tensor(emphasis.get('strengths', [2]), dtype=torch.long)
        else:
            # Derive from prosody contour
            contour = sample.get('prosody', {}).get('contour', [])
            if isinstance(contour, list) and len(contour) > 0:
                contour_t = torch.tensor(contour, dtype=torch.float32)
                # Find peaks (positions with higher than average value)
                mean_val = contour_t.mean()
                std_val = contour_t.std() + 1e-8

                # Peak positions (1 std above mean)
                peaks = (contour_t > mean_val + std_val).nonzero(as_tuple=True)[0]

                if len(peaks) > 0:
                    # Map to word positions (8 words)
                    positions = (peaks.float() / len(contour) * 8).long().unique()[:self.config.max_emphasis_words]

                    # Strength based on how far above threshold
                    strengths = torch.ones_like(positions) * 2  # Medium by default
                    for i, p in enumerate(peaks[:len(positions)]):
                        strength_val = (contour_t[p] - mean_val) / std_val
                        strengths[i] = min(4, int(strength_val) + 1)
                else:
                    positions = torch.zeros(1, dtype=torch.long)
                    strengths = torch.zeros(1, dtype=torch.long)
            else:
                positions = torch.zeros(1, dtype=torch.long)
                strengths = torch.zeros(1, dtype=torch.long)

        # Pad to max_emphasis_words
        if len(positions) < self.config.max_emphasis_words:
            pad_len = self.config.max_emphasis_words - len(positions)
            positions = F.pad(positions, (0, pad_len), value=-1)  # -1 = no emphasis
            strengths = F.pad(strengths, (0, pad_len), value=0)
        else:
            positions = positions[:self.config.max_emphasis_words]
            strengths = strengths[:self.config.max_emphasis_words]

        return positions, strengths

    def _load_prosody(self, sample: dict) -> Dict[str, torch.Tensor]:
        """Load prosody features."""
        prosody = sample.get('prosody', {})

        def to_tensor(data, dim):
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, dict):
                values = list(data.values())[:dim]
                t = torch.tensor(values, dtype=torch.float32)
            elif isinstance(data, (list, np.ndarray)):
                t = torch.tensor(data, dtype=torch.float32)
            else:
                return torch.zeros(dim)

            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[-1] < dim:
                t = F.pad(t, (0, dim - t.shape[-1]))
            elif t.shape[-1] > dim:
                t = t[..., :dim]
            return t

        return {
            'semantic': to_tensor(prosody.get('semantic', {}), 8),
            'acoustic': to_tensor(prosody.get('acoustic', {}), 5),
            'rhythm': to_tensor(prosody.get('rhythm', {}), 4),
            'contour': to_tensor(prosody.get('contour', []), 32),
        }

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        prosody = self._load_prosody(sample)
        emotion_label = self._get_emotion_label(sample)
        vad = self._get_vad(sample)
        intensity = self._get_intensity(sample)
        emphasis_positions, emphasis_strengths = self._get_emphasis_annotations(sample)

        return {
            'text': sample.get('text', ''),
            'prosody_semantic': prosody['semantic'],
            'prosody_acoustic': prosody['acoustic'],
            'prosody_rhythm': prosody['rhythm'],
            'prosody_contour': prosody['contour'],
            'emotion_label': emotion_label,
            'target_vad': vad,
            'target_intensity': torch.tensor(intensity, dtype=torch.float32),
            'emphasis_positions': emphasis_positions,
            'emphasis_strengths': emphasis_strengths,
        }


def emorl_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with emphasis annotations."""
    return {
        'text': [item['text'] for item in batch],
        'prosody_semantic': torch.stack([item['prosody_semantic'] for item in batch]),
        'prosody_acoustic': torch.stack([item['prosody_acoustic'] for item in batch]),
        'prosody_rhythm': torch.stack([item['prosody_rhythm'] for item in batch]),
        'prosody_contour': torch.stack([item['prosody_contour'] for item in batch]),
        'emotion_label': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long),
        'target_vad': torch.stack([item['target_vad'] for item in batch]),
        'target_intensity': torch.stack([item['target_intensity'] for item in batch]),
        'emphasis_positions': torch.stack([item['emphasis_positions'] for item in batch]),
        'emphasis_strengths': torch.stack([item['emphasis_strengths'] for item in batch]),
    }


# =============================================================================
# TESTING
# =============================================================================

def test_emorl_tts(config: EMORLConfig):
    """Test EMORL-TTS components."""
    print("\n[Test] EMORL-TTS Components")
    print("-" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create mock prosody encoder
    class MockProsodyEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            input_dim = 8 + 5 + 4 + 32
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size * config.num_prosody_tokens),
            )
            self.config = config

        def forward(self, semantic, acoustic, rhythm, contour):
            x = torch.cat([semantic, acoustic, rhythm, contour], dim=-1)
            out = self.encoder(x)
            batch = x.shape[0]
            return out.view(batch, self.config.num_prosody_tokens, self.config.hidden_size)

    prosody_encoder = MockProsodyEncoder(config).to(device)

    # Test 1: VAD Prototype Bank
    print("\n[Test 1] VAD Prototype Bank...")
    vad_bank = VADPrototypeBank()

    happy_vad = vad_bank.get_vad('happy')
    print(f"  Happy VAD: {happy_vad.tolist()}")

    # Interpolate happy + sad
    mixed_vad = vad_bank.interpolate_emotions(['happy', 'sad'], [0.7, 0.3])
    print(f"  Mixed (70% happy, 30% sad): {mixed_vad.tolist()}")

    # Scale intensity
    scaled = vad_bank.scale_intensity(happy_vad, 0.5)
    print(f"  Happy at 50% intensity: {scaled.tolist()}")
    print("  [PASS]")

    # Test 2: Word Emphasis Extractor
    print("\n[Test 2] Word Emphasis Extractor...")
    emphasis_extractor = WordEmphasisExtractor(config).to(device)

    batch_size = 2
    time_len = 32
    pitch_contour = torch.randn(batch_size, time_len, device=device)
    energy_contour = torch.randn(batch_size, time_len, device=device)

    # Add peaks for emphasis
    pitch_contour[:, 8] += 2.0  # Peak at word 2
    pitch_contour[:, 24] += 1.5  # Peak at word 6

    results = emphasis_extractor.extract_from_contour(pitch_contour, energy_contour)

    print(f"  Emphasis positions: {results['emphasis_positions'].tolist()}")
    print(f"  Emphasis strengths: {results['emphasis_strengths'].tolist()}")
    print("  [PASS]")

    # Test 3: Local Emphasis Regulator
    print("\n[Test 3] Local Emphasis Regulator...")
    regulator = LocalEmphasisRegulator(config).to(device)

    prosody_embedding = torch.randn(batch_size, config.num_prosody_tokens, config.hidden_size, device=device)
    target_positions = torch.tensor([[1, 3, -1, -1], [2, 5, -1, -1]], device=device)
    target_strengths = torch.tensor([[3, 2, 0, 0], [4, 2, 0, 0]], device=device)

    regulated = regulator(prosody_embedding, target_positions, target_strengths)

    print(f"  Input shape: {prosody_embedding.shape}")
    print(f"  Output shape: {regulated.shape}")
    diff = (regulated - prosody_embedding).abs().mean().item()
    print(f"  Mean modulation: {diff:.6f}")
    print("  [PASS]")

    # Test 4: Global VAD Reward
    print("\n[Test 4] Global VAD Reward...")
    vad_reward_fn = GlobalVADRewardFunction(config).to(device)

    target_vad = torch.tensor([[0.8, 0.6, 0.6], [-0.6, -0.4, -0.5]], device=device)
    target_intensity = torch.tensor([0.8, 0.5], device=device)
    target_emotion = torch.tensor([1, 2], device=device)

    vad_results = vad_reward_fn.compute_reward(
        prosody_embedding, target_vad, target_intensity, target_emotion
    )

    print(f"  VAD reward: {vad_results['vad_reward'].tolist()}")
    print(f"  Intensity reward: {vad_results['intensity_reward'].tolist()}")
    print(f"  Total reward: {vad_results['total_reward'].tolist()}")
    print("  [PASS]")

    # Test 5: Emphasis Reward
    print("\n[Test 5] Emphasis Reward...")
    emphasis_reward_fn = EmphasisRewardFunction(config).to(device)

    emphasis_results = emphasis_reward_fn.compute_reward(
        prosody_embedding, target_positions, target_strengths
    )

    print(f"  Position reward: {emphasis_results['position_reward'].tolist()}")
    print(f"  Strength reward: {emphasis_results['strength_reward'].tolist()}")
    print(f"  Contrast reward: {emphasis_results['contrast_reward'].tolist()}")
    print(f"  Total reward: {emphasis_results['total_reward'].tolist()}")
    print("  [PASS]")

    # Test 6: EMORL Policy
    print("\n[Test 6] EMORL Policy...")
    policy = EMORLPolicy(prosody_encoder, config).to(device)

    semantic = torch.randn(batch_size, 8, device=device)
    acoustic = torch.randn(batch_size, 5, device=device)
    rhythm = torch.randn(batch_size, 4, device=device)
    contour = torch.randn(batch_size, 32, device=device)

    # With emphasis
    output, log_prob, value = policy(
        semantic, acoustic, rhythm, contour,
        target_emphasis_positions=target_positions,
        target_emphasis_strengths=target_strengths,
        deterministic=True,
    )

    print(f"  Output shape: {output.shape}")
    print(f"  Log prob: {log_prob.tolist()}")
    print("  [PASS]")

    # Test 7: Full Training Step
    print("\n[Test 7] Full Training Step...")
    trainer = EMORLTrainer(config, prosody_encoder)

    batch = {
        'prosody_semantic': torch.randn(batch_size, 8),
        'prosody_acoustic': torch.randn(batch_size, 5),
        'prosody_rhythm': torch.randn(batch_size, 4),
        'prosody_contour': torch.randn(batch_size, 32),
        'emotion_label': torch.tensor([1, 2]),
        'target_vad': target_vad.cpu(),
        'target_intensity': target_intensity.cpu(),
        'emphasis_positions': target_positions.cpu(),
        'emphasis_strengths': target_strengths.cpu(),
    }

    metrics = trainer.train_step(batch)

    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Reward mean: {metrics['reward_mean']:.4f}")
    print(f"  Global VAD reward: {metrics['global_vad_reward']:.4f}")
    print(f"  Emphasis reward: {metrics['emphasis_reward']:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All EMORL-TTS tests passed!")
    print("=" * 70)

    print("\nEMORL-TTS Features:")
    print("-" * 40)
    print("1. Global VAD-based emotion intensity control")
    print("2. Local word-level emphasis regulation")
    print("3. Dual reward system (VAD + Emphasis)")
    print("4. Emphasis curriculum learning")
    print("5. Word emphasis extraction from prosody contours")

    print("\nUsage:")
    print("-" * 40)
    print("""
# Train EMORL-TTS
python train_emorl_tts.py --config config/emorl_tts.yaml \\
  --checkpoint ../checkpoints/prosody_v7/best.pt \\
  --manifest ../data/emotion_manifest.json

# Test mode
python emorl_tts.py --test
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EMORL-TTS: VAD + Local Emphasis RL")
    parser.add_argument('--config', type=str, default='config/emorl_tts.yaml')
    parser.add_argument('--checkpoint', type=str, help='Pre-trained prosody checkpoint')
    parser.add_argument('--manifest', type=str, help='Training manifest')
    parser.add_argument('--val_manifest', type=str, help='Validation manifest')
    parser.add_argument('--output_dir', type=str, default='checkpoints/emorl_tts')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = EMORLConfig(**{k: v for k, v in config_dict.items() if hasattr(EMORLConfig, k)})
    else:
        config = EMORLConfig()

    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 70)
    print("EMORL-TTS: VAD-space Intensity + Local Emphasis Regulation")
    print("=" * 70)

    if args.test:
        test_emorl_tts(config)
        return

    # Load prosody encoder
    if args.checkpoint:
        from prosody_conditioning import ProsodyConfig, ProsodyEncoder

        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        prosody_config = ProsodyConfig(**checkpoint.get('prosody_config', {}))
        prosody_encoder = ProsodyEncoder(prosody_config)
        prosody_encoder.load_state_dict(checkpoint['prosody_encoder'])
        print(f"Loaded prosody encoder from {args.checkpoint}")
    else:
        from prosody_conditioning import ProsodyConfig, ProsodyEncoder

        prosody_config = ProsodyConfig(hidden_size=config.hidden_size)
        prosody_encoder = ProsodyEncoder(prosody_config)
        print("Created fresh prosody encoder")

    # Create trainer
    trainer = EMORLTrainer(config, prosody_encoder)

    # Create dataset
    if args.manifest:
        train_dataset = EMORLDataset(
            manifest_path=args.manifest,
            prosody_cache_dir='data/prosody_cache',
            config=config,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=emorl_collate_fn,
        )

        val_loader = None
        if args.val_manifest:
            val_dataset = EMORLDataset(
                manifest_path=args.val_manifest,
                prosody_cache_dir='data/prosody_cache',
                config=config,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=emorl_collate_fn,
            )

        trainer.train(train_loader, val_loader)
    else:
        print("\nNo manifest provided. Run with --manifest to train.")
        print("\nEMORL-TTS unifies:")
        print("  - Global emotion intensity in VAD space")
        print("  - Local word-level emphasis regulation")
        print("  - Dual reward: VAD matching + emphasis clarity")


if __name__ == "__main__":
    main()
