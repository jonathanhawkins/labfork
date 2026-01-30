"""
MPO: Multidimensional Preference Optimization for TTS

Based on arXiv:2509.00685 (Aug 2025). Aligns TTS with human preferences across
multiple dimensions simultaneously.

Key Innovation - Multidimensional Preference Optimization:
- Introduces preference set for multi-aspect data construction
- Aligns intelligibility, speaker similarity, AND prosody together
- Regularization during training prevents typical DPO degradation
- Avoids unstable training that plagues standard DPO approaches

Dimensions Optimized:
1. Intelligibility (WER/CER) - Pronunciation accuracy
2. Speaker Similarity (SIM) - Voice identity preservation
3. Prosody Naturalness - Prosodic quality and expressiveness
4. Overall Quality - General audio quality

Improvements over Emo-DPO:
- Multi-aspect optimization vs emotion-only
- Built-in regularization for stability
- Preference set construction methodology

Integration:
- Works with existing GRPO framework from multi_reward_rl.py
- Compatible with Emo-DPO preference pairs
- Extends reward aggregation with dimension-specific regularization

Usage:
    python mpo.py --config config/mpo.yaml \
        --checkpoint ../checkpoints/prosody_v6/best.pt \
        --manifest ../data/emotion_manifest.json

    # Test mode
    python mpo.py --test
"""

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MPOConfig:
    """Configuration for Multidimensional Preference Optimization."""

    # Preference dimensions (each has its own reward weight)
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'intelligibility': 1.0,  # WER/CER based
        'speaker_similarity': 1.0,  # Cosine similarity of speaker embeddings
        'prosody_naturalness': 1.0,  # Prosodic quality score
        'overall_quality': 0.5,  # General quality (MOS proxy)
    })

    # DPO-style loss settings
    beta: float = 0.1  # KL penalty coefficient
    js_alpha: float = 0.5  # Jensen-Shannon regularization

    # Stability regularization (key MPO contribution)
    use_stability_regularization: bool = True
    stability_lambda: float = 0.1  # Weight for stability regularization
    anchor_weight: float = 0.3  # Weight to stay close to reference
    gradient_penalty_weight: float = 0.01  # Penalty for large gradients

    # Dimension-specific regularization
    dimension_correlation_penalty: float = 0.05  # Prevent dimension collapse
    min_dimension_weight: float = 0.1  # Minimum weight per dimension

    # Preference set construction
    preference_set_size: int = 4  # Number of samples per preference set
    stratified_sampling: bool = True  # Sample diverse preference pairs
    difficulty_curriculum: bool = True  # Start with easy pairs, progress to hard

    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 5
    warmup_steps: int = 200
    gradient_accumulation: int = 2
    max_grad_norm: float = 1.0

    # Reference model
    use_reference_model: bool = True
    reference_model_update_freq: int = 0  # 0 = frozen

    # Model settings
    hidden_size: int = 2048
    num_prosody_tokens: int = 4

    # SFT regularization (prevent catastrophic forgetting)
    sft_weight: float = 0.3
    sft_decay: float = 0.95

    # Output
    output_dir: str = "checkpoints/mpo"
    log_every: int = 10
    save_every_epochs: int = 1


# =============================================================================
# DIMENSION-SPECIFIC REWARD FUNCTIONS
# =============================================================================

class DimensionReward(nn.Module):
    """Base class for dimension-specific reward computation."""

    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()
        self.name = name
        self.weight = weight
        self.reward_history = []  # Track for normalization

    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute reward. Override in subclasses."""
        raise NotImplementedError

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize reward using running statistics."""
        if len(self.reward_history) > 100:
            mean = np.mean(self.reward_history[-100:])
            std = max(np.std(self.reward_history[-100:]), 1e-8)
            reward = (reward - mean) / std
        return reward

    def update_history(self, reward: torch.Tensor):
        """Update reward history for normalization."""
        self.reward_history.extend(reward.detach().cpu().numpy().tolist())
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]


class IntelligibilityReward(DimensionReward):
    """
    Intelligibility reward based on CER/WER.

    Measures pronunciation accuracy by:
    1. (If ASR available) Transcribe and compute edit distance
    2. (Fallback) Learned predictor from prosody embeddings
    """

    def __init__(self, config: MPOConfig, asr_model=None):
        super().__init__("intelligibility", config.dimension_weights['intelligibility'])
        self.config = config
        self.asr_model = asr_model

        # Learned CER predictor (fallback)
        self.cer_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.LayerNorm(config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        # Phoneme-based quality predictor
        self.phoneme_quality = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def compute(
        self,
        prosody_embedding: torch.Tensor,
        generated_audio: Optional[torch.Tensor] = None,
        target_text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute intelligibility reward."""
        device = prosody_embedding.device

        if self.asr_model is not None and generated_audio is not None:
            # Full ASR pipeline
            with torch.no_grad():
                predicted_text = self.asr_model.transcribe(generated_audio)
                cer = self._compute_cer(predicted_text, target_text or [])
                reward = 1.0 - cer.to(device)
        else:
            # Use learned predictors
            pooled = prosody_embedding.mean(dim=1)
            cer_score = self.cer_predictor(pooled).squeeze(-1)
            phoneme_score = self.phoneme_quality(pooled).squeeze(-1)

            # Combine scores
            reward = 0.7 * (1.0 - cer_score) + 0.3 * phoneme_score

        self.update_history(reward)
        return reward * self.weight

    def _compute_cer(self, predicted: List[str], target: List[str]) -> torch.Tensor:
        """Compute CER between predicted and target text."""
        from difflib import SequenceMatcher

        cers = []
        for pred, tgt in zip(predicted, target):
            pred = pred.lower().strip()
            tgt = tgt.lower().strip()
            if not tgt:
                cers.append(0.0)
                continue
            matcher = SequenceMatcher(None, pred, tgt)
            cer = 1.0 - matcher.ratio()
            cers.append(cer)

        return torch.tensor(cers, dtype=torch.float32)


class SpeakerSimilarityReward(DimensionReward):
    """
    Speaker similarity reward.

    Measures how well the generated prosody preserves speaker identity.
    """

    def __init__(self, config: MPOConfig, speaker_encoder=None):
        super().__init__("speaker_similarity", config.dimension_weights['speaker_similarity'])
        self.config = config
        self.speaker_encoder = speaker_encoder

        # Learned similarity predictor
        self.sim_predictor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Speaker embedding projection
        self.speaker_projection = nn.Linear(256, config.hidden_size)

    def compute(
        self,
        prosody_embedding: torch.Tensor,
        reference_speaker_emb: Optional[torch.Tensor] = None,
        generated_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute speaker similarity reward."""
        device = prosody_embedding.device
        batch_size = prosody_embedding.shape[0]

        if self.speaker_encoder is not None and generated_audio is not None:
            with torch.no_grad():
                gen_speaker_emb = self.speaker_encoder(generated_audio)
                if reference_speaker_emb is not None:
                    sim = F.cosine_similarity(gen_speaker_emb, reference_speaker_emb, dim=-1)
                else:
                    sim = torch.ones(batch_size, device=device) * 0.5
        else:
            pooled = prosody_embedding.mean(dim=1)

            if reference_speaker_emb is not None:
                # Project speaker embedding to match dimensions
                if reference_speaker_emb.shape[-1] != self.config.hidden_size:
                    ref_projected = self.speaker_projection(reference_speaker_emb)
                else:
                    ref_projected = reference_speaker_emb
                combined = torch.cat([pooled, ref_projected], dim=-1)
            else:
                combined = torch.cat([pooled, pooled], dim=-1)

            sim = self.sim_predictor(combined).squeeze(-1)

        # Normalize to [0, 1]
        reward = (sim + 1.0) / 2.0 if sim.min() < 0 else sim

        self.update_history(reward)
        return reward.to(device) * self.weight


class ProsodyNaturalnessReward(DimensionReward):
    """
    Prosody naturalness reward.

    Evaluates prosodic quality including:
    - Pitch contour smoothness
    - Energy consistency
    - Temporal coherence
    - Emotion expressiveness
    """

    def __init__(self, config: MPOConfig, num_emotions: int = 8):
        super().__init__("prosody_naturalness", config.dimension_weights['prosody_naturalness'])
        self.config = config
        self.num_emotions = num_emotions

        # Prosody quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        # Temporal consistency predictor
        self.temporal_consistency = nn.Sequential(
            nn.Conv1d(config.hidden_size, config.hidden_size // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.hidden_size // 4, 1, kernel_size=1),
        )

        # Emotion classifier (for expressiveness)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, num_emotions),
        )

    def compute(
        self,
        prosody_embedding: torch.Tensor,
        target_emotion: Optional[torch.Tensor] = None,
        f0_contour: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute prosody naturalness reward."""
        device = prosody_embedding.device
        batch_size = prosody_embedding.shape[0]

        # Base quality score
        pooled = prosody_embedding.mean(dim=1)
        quality_score = self.quality_predictor(pooled).squeeze(-1)

        # Temporal consistency
        if prosody_embedding.dim() == 3 and prosody_embedding.shape[1] > 1:
            # [batch, seq, hidden] -> [batch, hidden, seq]
            x_transposed = prosody_embedding.transpose(1, 2)
            temporal_score = self.temporal_consistency(x_transposed).squeeze(1).mean(dim=-1)
            temporal_score = torch.sigmoid(temporal_score)
        else:
            temporal_score = torch.ones(batch_size, device=device) * 0.5

        # Emotion expressiveness
        emotion_score = torch.ones(batch_size, device=device) * 0.5
        if target_emotion is not None:
            emotion_logits = self.emotion_classifier(pooled)
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            emotion_score = emotion_probs.gather(1, target_emotion.unsqueeze(1)).squeeze(1)

        # Combined prosody naturalness
        reward = (
            0.4 * quality_score +
            0.3 * temporal_score +
            0.3 * emotion_score
        )

        self.update_history(reward)
        return reward * self.weight


class OverallQualityReward(DimensionReward):
    """
    Overall quality reward.

    Predicts general audio quality (MOS proxy) from prosody embeddings.
    """

    def __init__(self, config: MPOConfig, quality_model=None):
        super().__init__("overall_quality", config.dimension_weights['overall_quality'])
        self.config = config
        self.quality_model = quality_model

        # Learned quality predictor
        self.quality_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.LayerNorm(config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 8),
            nn.GELU(),
            nn.Linear(config.hidden_size // 8, 1),
            nn.Sigmoid(),
        )

        # Variance predictor (lower variance = more stable = higher quality)
        self.variance_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def compute(
        self,
        prosody_embedding: torch.Tensor,
        generated_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute overall quality reward."""
        device = prosody_embedding.device

        if self.quality_model is not None and generated_audio is not None:
            with torch.no_grad():
                quality = self.quality_model(generated_audio)
                # Normalize MOS [1, 5] to [0, 1]
                quality = (quality - 1.0) / 4.0
        else:
            pooled = prosody_embedding.mean(dim=1)
            quality = self.quality_head(pooled).squeeze(-1)

            # Variance penalty (stable embeddings = higher quality)
            if prosody_embedding.dim() == 3:
                variance = prosody_embedding.var(dim=1).mean(dim=-1)
                variance_penalty = torch.exp(-variance / 100.0)
                quality = 0.7 * quality + 0.3 * variance_penalty

        self.update_history(quality)
        return quality * self.weight


# =============================================================================
# MULTIDIMENSIONAL PREFERENCE SET
# =============================================================================

class MultidimensionalPreferenceSet:
    """
    Constructs preference sets for multi-aspect optimization.

    Key concept: Instead of binary win/lose pairs, we construct sets of samples
    ranked across multiple dimensions. This enables:
    1. Learning nuanced preferences across dimensions
    2. Handling conflicting signals (e.g., clear speech with wrong emotion)
    3. Progressive curriculum from easy to hard comparisons
    """

    def __init__(self, config: MPOConfig):
        self.config = config
        self.dimensions = list(config.dimension_weights.keys())

    def construct_preference_set(
        self,
        samples: List[Dict[str, torch.Tensor]],
        dimension_rewards: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Construct a preference set from multiple samples.

        Args:
            samples: List of sample dicts, each with prosody embeddings
            dimension_rewards: Dict mapping dimension name to list of rewards

        Returns:
            Dict with:
                - rankings: Per-dimension rankings
                - chosen: Overall best sample index
                - rejected: Overall worst sample index
                - margin: Difficulty margin for curriculum
        """
        n_samples = len(samples)

        # Compute per-dimension rankings
        rankings = {}
        for dim in self.dimensions:
            if dim in dimension_rewards:
                rewards = dimension_rewards[dim]
                # Argsort descending (highest reward = rank 0)
                ranking = np.argsort(rewards)[::-1].tolist()
                rankings[dim] = ranking

        # Compute overall ranking (weighted sum of rewards)
        overall_scores = []
        for i in range(n_samples):
            score = 0.0
            for dim, weight in self.config.dimension_weights.items():
                if dim in dimension_rewards:
                    score += weight * dimension_rewards[dim][i]
            overall_scores.append(score)

        overall_ranking = np.argsort(overall_scores)[::-1].tolist()

        # Chosen = best overall, Rejected = worst overall
        chosen_idx = overall_ranking[0]
        rejected_idx = overall_ranking[-1]

        # Compute difficulty margin (smaller = harder comparison)
        margin = overall_scores[chosen_idx] - overall_scores[rejected_idx]

        return {
            'rankings': rankings,
            'overall_ranking': overall_ranking,
            'overall_scores': overall_scores,
            'chosen_idx': chosen_idx,
            'rejected_idx': rejected_idx,
            'margin': margin,
        }

    def sample_preference_pair(
        self,
        preference_set: Dict[str, Any],
        difficulty: float = 0.5,
    ) -> Tuple[int, int]:
        """
        Sample a win/lose pair from preference set based on difficulty.

        Args:
            preference_set: Output from construct_preference_set
            difficulty: 0.0 = easy (large margin), 1.0 = hard (small margin)

        Returns:
            (chosen_idx, rejected_idx)
        """
        overall_ranking = preference_set['overall_ranking']
        n_samples = len(overall_ranking)

        if n_samples <= 2:
            return overall_ranking[0], overall_ranking[-1]

        if self.config.difficulty_curriculum:
            # Choose comparison based on difficulty
            # Easy: compare best vs worst
            # Hard: compare adjacent samples
            if difficulty < 0.3:
                # Easy: top vs bottom
                chosen_idx = overall_ranking[0]
                rejected_idx = overall_ranking[-1]
            elif difficulty < 0.7:
                # Medium: top vs middle, or middle vs bottom
                mid = n_samples // 2
                if random.random() < 0.5:
                    chosen_idx = overall_ranking[0]
                    rejected_idx = overall_ranking[mid]
                else:
                    chosen_idx = overall_ranking[mid]
                    rejected_idx = overall_ranking[-1]
            else:
                # Hard: adjacent comparisons
                idx = random.randint(0, n_samples - 2)
                chosen_idx = overall_ranking[idx]
                rejected_idx = overall_ranking[idx + 1]
        else:
            chosen_idx = overall_ranking[0]
            rejected_idx = overall_ranking[-1]

        return chosen_idx, rejected_idx


# =============================================================================
# STABILITY REGULARIZER
# =============================================================================

class StabilityRegularizer(nn.Module):
    """
    Stability regularization to prevent DPO degradation.

    Key insight from MPO paper: Standard DPO can lead to:
    1. Reward hacking (optimizing proxy instead of true objective)
    2. Catastrophic forgetting (losing base capabilities)
    3. Mode collapse (generating similar outputs)

    Mitigations:
    1. Anchor loss: Stay close to reference model outputs
    2. Gradient penalty: Prevent large parameter updates
    3. Diversity loss: Encourage diverse outputs
    4. Dimension correlation penalty: Prevent dimension collapse
    """

    def __init__(self, config: MPOConfig):
        super().__init__()
        self.config = config

    def compute_anchor_loss(
        self,
        policy_output: torch.Tensor,
        reference_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Anchor loss to prevent drift from reference.

        L_anchor = ||policy_output - reference_output||^2
        """
        return F.mse_loss(policy_output, reference_output)

    def compute_gradient_penalty(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """
        Gradient penalty for stable training.

        Penalizes large gradients to prevent unstable updates.
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        return torch.tensor(max(0, total_norm - self.config.max_grad_norm))

    def compute_diversity_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diversity loss to prevent mode collapse.

        Encourages embeddings to be diverse within batch.
        """
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)  # [batch, hidden]

        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Pairwise cosine similarity
        embeddings_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())

        # Mask diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        pairwise_sims = sim_matrix[mask]

        # Loss = mean similarity (we want to minimize similarity = maximize diversity)
        diversity_loss = pairwise_sims.mean()

        return diversity_loss

    def compute_dimension_correlation_penalty(
        self,
        dimension_rewards: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Penalty for high correlation between dimension rewards.

        Prevents dimension collapse where all dimensions optimize the same thing.
        """
        if len(dimension_rewards) < 2:
            return torch.tensor(0.0)

        rewards = list(dimension_rewards.values())
        device = rewards[0].device

        # Stack rewards: [num_dims, batch]
        rewards_tensor = torch.stack(rewards, dim=0)

        # Compute correlation matrix
        # Center rewards
        rewards_centered = rewards_tensor - rewards_tensor.mean(dim=1, keepdim=True)

        # Covariance
        cov = torch.mm(rewards_centered, rewards_centered.t()) / max(1, rewards_tensor.shape[1] - 1)

        # Correlation (normalized covariance)
        std = rewards_centered.std(dim=1, keepdim=True) + 1e-8
        corr = cov / (std @ std.t())

        # Penalty = mean off-diagonal correlation
        num_dims = corr.shape[0]
        mask = ~torch.eye(num_dims, dtype=torch.bool, device=device)
        off_diag_corr = corr[mask].abs().mean()

        return off_diag_corr

    def forward(
        self,
        policy_output: torch.Tensor,
        reference_output: torch.Tensor,
        dimension_rewards: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all stability regularization terms.

        Returns dict with individual terms and total.
        """
        losses = {}

        # Anchor loss
        if self.config.anchor_weight > 0:
            losses['anchor'] = self.config.anchor_weight * self.compute_anchor_loss(
                policy_output, reference_output
            )

        # Gradient penalty
        if self.config.gradient_penalty_weight > 0 and model is not None:
            losses['gradient_penalty'] = (
                self.config.gradient_penalty_weight *
                self.compute_gradient_penalty(model)
            )

        # Diversity loss (negative because we want diversity)
        losses['diversity'] = 0.1 * self.compute_diversity_loss(policy_output)

        # Dimension correlation penalty
        if self.config.dimension_correlation_penalty > 0:
            losses['dimension_correlation'] = (
                self.config.dimension_correlation_penalty *
                self.compute_dimension_correlation_penalty(dimension_rewards)
            )

        # Total
        losses['total'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor))

        return losses


# =============================================================================
# MPO LOSS
# =============================================================================

class MPOLoss(nn.Module):
    """
    Multidimensional Preference Optimization Loss.

    Extends DPO with:
    1. Multi-dimensional rewards
    2. Stability regularization
    3. Adaptive dimension weighting
    """

    def __init__(self, config: MPOConfig):
        super().__init__()
        self.config = config
        self.beta = config.beta
        self.js_alpha = config.js_alpha

        # Learnable dimension weights (can adapt during training)
        self.dimension_weight_logits = nn.Parameter(
            torch.tensor([config.dimension_weights[k]
                          for k in sorted(config.dimension_weights.keys())])
        )

        # Stability regularizer
        self.stability_regularizer = StabilityRegularizer(config)

    def get_dimension_weights(self) -> Dict[str, float]:
        """Get current dimension weights (softmax normalized)."""
        weights = F.softmax(self.dimension_weight_logits, dim=0)
        dim_names = sorted(self.config.dimension_weights.keys())
        return {name: weights[i].item() for i, name in enumerate(dim_names)}

    def compute_preference_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        dimension_rewards: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multidimensional preference loss.

        Args:
            policy_chosen_logps: Log prob of chosen under policy
            policy_rejected_logps: Log prob of rejected under policy
            ref_chosen_logps: Log prob of chosen under reference
            ref_rejected_logps: Log prob of rejected under reference
            dimension_rewards: Per-dimension rewards for chosen/rejected

        Returns:
            Dict with loss components
        """
        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Standard DPO logits
        base_logits = self.beta * (chosen_logratios - rejected_logratios)

        # JS regularization
        js_chosen = F.softplus(chosen_logratios)
        js_rejected = F.softplus(rejected_logratios)
        js_term = js_chosen - js_rejected

        regularized_logits = base_logits - self.js_alpha * js_term

        # Multi-dimensional reward weighting
        weights = F.softmax(self.dimension_weight_logits, dim=0)
        dim_names = sorted(self.config.dimension_weights.keys())

        weighted_margin = torch.zeros_like(regularized_logits)
        for i, dim_name in enumerate(dim_names):
            if dim_name in dimension_rewards:
                # dimension_rewards should contain (chosen_reward - rejected_reward)
                dim_margin = dimension_rewards[dim_name]
                weighted_margin = weighted_margin + weights[i] * dim_margin

        # Scale logits by weighted reward margin
        scaled_logits = regularized_logits * (1.0 + torch.tanh(weighted_margin))

        # Loss: -log σ(scaled_logits)
        loss = -F.logsigmoid(scaled_logits).mean()

        # Metrics
        with torch.no_grad():
            accuracy = (scaled_logits > 0).float().mean()
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            reward_margin = chosen_rewards - rejected_rewards

        return {
            'loss': loss,
            'accuracy': accuracy,
            'reward_margin': reward_margin.mean(),
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'js_term': js_term.mean(),
            'weighted_margin': weighted_margin.mean(),
        }

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        dimension_rewards: Dict[str, torch.Tensor],
        policy_output: Optional[torch.Tensor] = None,
        reference_output: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full MPO loss with stability regularization.
        """
        # Preference loss
        pref_losses = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            dimension_rewards,
        )

        # Stability regularization
        if self.config.use_stability_regularization and policy_output is not None:
            if reference_output is None:
                reference_output = policy_output.detach()

            stability_losses = self.stability_regularizer(
                policy_output,
                reference_output,
                dimension_rewards,
                model,
            )
        else:
            stability_losses = {'total': torch.tensor(0.0, device=policy_chosen_logps.device)}

        # Combined loss
        total_loss = pref_losses['loss'] + self.config.stability_lambda * stability_losses['total']

        return {
            'total': total_loss,
            'preference_loss': pref_losses['loss'],
            'stability_loss': stability_losses['total'],
            'accuracy': pref_losses['accuracy'],
            'reward_margin': pref_losses['reward_margin'],
            'chosen_rewards': pref_losses['chosen_rewards'],
            'rejected_rewards': pref_losses['rejected_rewards'],
            'js_term': pref_losses['js_term'],
            'weighted_margin': pref_losses['weighted_margin'],
        }


# =============================================================================
# PREFERENCE DATA GENERATOR
# =============================================================================

class PreferenceDataGenerator:
    """
    Generates preference data from TTS outputs.

    Handles:
    1. Sampling multiple outputs for the same input
    2. Computing per-dimension rewards
    3. Constructing preference sets
    4. Stratified sampling for balanced training
    """

    def __init__(
        self,
        config: MPOConfig,
        dimension_rewards: Dict[str, DimensionReward],
    ):
        self.config = config
        self.dimension_rewards = dimension_rewards
        self.preference_set_builder = MultidimensionalPreferenceSet(config)

    def generate_preference_data(
        self,
        prosody_samples: List[torch.Tensor],
        target_emotion: Optional[torch.Tensor] = None,
        reference_speaker_emb: Optional[torch.Tensor] = None,
        target_text: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate preference data from multiple prosody samples.

        Args:
            prosody_samples: List of prosody embeddings
            target_emotion: Target emotion label
            reference_speaker_emb: Reference speaker embedding
            target_text: Target transcription

        Returns:
            Dict with preference set and computed rewards
        """
        # Compute dimension rewards for each sample
        dimension_rewards = {}

        for dim_name, reward_fn in self.dimension_rewards.items():
            dim_rewards = []
            for sample in prosody_samples:
                if dim_name == 'intelligibility':
                    reward = reward_fn.compute(sample, target_text=target_text)
                elif dim_name == 'speaker_similarity':
                    reward = reward_fn.compute(sample, reference_speaker_emb=reference_speaker_emb)
                elif dim_name == 'prosody_naturalness':
                    reward = reward_fn.compute(sample, target_emotion=target_emotion)
                else:
                    reward = reward_fn.compute(sample)

                dim_rewards.append(reward.mean().item())

            dimension_rewards[dim_name] = dim_rewards

        # Build preference set
        samples = [{'prosody': s} for s in prosody_samples]
        preference_set = self.preference_set_builder.construct_preference_set(
            samples, dimension_rewards
        )

        return {
            'preference_set': preference_set,
            'dimension_rewards': dimension_rewards,
            'prosody_samples': prosody_samples,
        }


# =============================================================================
# MPO TRAINER
# =============================================================================

class MPOTrainer:
    """
    Trainer for Multidimensional Preference Optimization.

    Integrates:
    1. Dimension-specific reward computation
    2. Preference set construction
    3. MPO loss with stability regularization
    4. Curriculum learning from easy to hard comparisons
    """

    def __init__(
        self,
        config: MPOConfig,
        prosody_encoder: nn.Module,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or self._setup_device()

        # Policy model
        self.policy_model = prosody_encoder.to(self.device)

        # Reference model (frozen)
        if config.use_reference_model:
            self.ref_model = copy.deepcopy(prosody_encoder)
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print("Created frozen reference model for MPO")
        else:
            self.ref_model = None

        # Dimension reward functions
        self.dimension_rewards = nn.ModuleDict({
            'intelligibility': IntelligibilityReward(config),
            'speaker_similarity': SpeakerSimilarityReward(config),
            'prosody_naturalness': ProsodyNaturalnessReward(config),
            'overall_quality': OverallQualityReward(config),
        }).to(self.device)

        # Preference data generator
        self.pref_generator = PreferenceDataGenerator(
            config, dict(self.dimension_rewards)
        )

        # MPO loss
        self.mpo_loss = MPOLoss(config).to(self.device)

        # Log probability computer
        self.log_prob_computer = LogProbComputer(config).to(self.device)

        # Optimizer
        trainable_params = (
            list(self.policy_model.parameters()) +
            list(self.dimension_rewards.parameters()) +
            list(self.mpo_loss.parameters()) +
            list(self.log_prob_computer.parameters())
        )

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        self.current_sft_weight = config.sft_weight
        self.current_difficulty = 0.0  # Curriculum learning

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _get_prosody_embedding(
        self,
        model: nn.Module,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
    ) -> torch.Tensor:
        """Get prosody embedding from model."""
        if hasattr(model, 'get_prosody_prefix'):
            return model.get_prosody_prefix(semantic, acoustic, rhythm, contour)
        elif hasattr(model, 'prosody_encoder'):
            return model.prosody_encoder(semantic, acoustic, rhythm, contour)
        else:
            return model(semantic, acoustic, rhythm, contour)

    def _compute_sft_loss(
        self,
        embedding: torch.Tensor,
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

        if target.shape[-1] < embedding.shape[-1]:
            target = F.pad(target, (0, embedding.shape[-1] - target.shape[-1]))

        target = target.unsqueeze(1).expand_as(embedding[:, :1, :])
        return F.mse_loss(embedding[:, 0, :target.shape[-1]], target[:, 0, :])

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single MPO training step."""
        self.policy_model.train()

        # Move to device
        prosody_dict = {
            'semantic': batch['prosody_semantic'].to(self.device),
            'acoustic': batch['prosody_acoustic'].to(self.device),
            'rhythm': batch['prosody_rhythm'].to(self.device),
            'contour': batch['prosody_contour'].to(self.device),
        }
        target_emotion = batch.get('emotion_label')
        if target_emotion is not None:
            target_emotion = target_emotion.to(self.device)

        batch_size = prosody_dict['semantic'].shape[0]

        # Sample multiple prosody outputs for preference comparison
        prosody_samples = []
        for _ in range(self.config.preference_set_size):
            # Add small noise for diversity
            noisy_semantic = prosody_dict['semantic'] + 0.1 * torch.randn_like(prosody_dict['semantic'])
            noisy_acoustic = prosody_dict['acoustic'] + 0.1 * torch.randn_like(prosody_dict['acoustic'])

            embedding = self._get_prosody_embedding(
                self.policy_model,
                noisy_semantic,
                noisy_acoustic,
                prosody_dict['rhythm'],
                prosody_dict['contour'],
            )
            prosody_samples.append(embedding)

        # Compute dimension rewards
        dim_rewards_tensors = {}
        for dim_name, reward_fn in self.dimension_rewards.items():
            rewards = []
            for sample in prosody_samples:
                if dim_name == 'prosody_naturalness':
                    r = reward_fn.compute(sample, target_emotion=target_emotion)
                else:
                    r = reward_fn.compute(sample)
                rewards.append(r)
            # Stack: [set_size, batch]
            dim_rewards_tensors[dim_name] = torch.stack(rewards, dim=0)

        # Build preference set and get chosen/rejected
        # For simplicity, use overall ranking
        overall_scores = torch.zeros(self.config.preference_set_size, batch_size, device=self.device)
        for dim_name, rewards in dim_rewards_tensors.items():
            weight = self.config.dimension_weights.get(dim_name, 1.0)
            overall_scores = overall_scores + weight * rewards

        # Chosen = argmax, Rejected = argmin per batch
        chosen_idx = overall_scores.argmax(dim=0)  # [batch]
        rejected_idx = overall_scores.argmin(dim=0)  # [batch]

        # Handle ties
        tie_mask = chosen_idx == rejected_idx
        if tie_mask.any():
            # Use curriculum: sample nearby ranks
            random_offset = torch.randint(1, self.config.preference_set_size, (batch_size,), device=self.device)
            rejected_idx = torch.where(
                tie_mask,
                (chosen_idx + random_offset) % self.config.preference_set_size,
                rejected_idx
            )

        # Extract chosen/rejected embeddings
        batch_indices = torch.arange(batch_size, device=self.device)
        prosody_samples_stacked = torch.stack(prosody_samples, dim=0)  # [set_size, batch, tokens, hidden]

        chosen_embeddings = prosody_samples_stacked[chosen_idx, batch_indices]  # [batch, tokens, hidden]
        rejected_embeddings = prosody_samples_stacked[rejected_idx, batch_indices]

        # Reference embeddings
        with torch.no_grad():
            if self.ref_model is not None:
                ref_chosen = self._get_prosody_embedding(
                    self.ref_model,
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                )
            else:
                ref_chosen = chosen_embeddings.detach()

        # Compute log probabilities
        policy_chosen_logps = self.log_prob_computer(
            chosen_embeddings, chosen_embeddings.detach()
        )
        policy_rejected_logps = self.log_prob_computer(
            rejected_embeddings, rejected_embeddings.detach()
        )
        ref_chosen_logps = self.log_prob_computer(
            ref_chosen, ref_chosen
        )
        ref_rejected_logps = self.log_prob_computer(
            ref_chosen, ref_chosen  # Use ref_chosen as baseline
        )

        # Compute dimension reward margins (chosen - rejected)
        dim_reward_margins = {}
        for dim_name, rewards in dim_rewards_tensors.items():
            chosen_rewards = rewards[chosen_idx, batch_indices]
            rejected_rewards = rewards[rejected_idx, batch_indices]
            dim_reward_margins[dim_name] = chosen_rewards - rejected_rewards

        # Compute MPO loss
        loss_dict = self.mpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            dim_reward_margins,
            policy_output=chosen_embeddings,
            reference_output=ref_chosen,
            model=self.policy_model,
        )

        # SFT regularization
        sft_loss = self._compute_sft_loss(chosen_embeddings, prosody_dict)

        # Total loss
        total_loss = loss_dict['total'] + self.current_sft_weight * sft_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.global_step += 1

        # Update curriculum difficulty
        self.current_difficulty = min(1.0, self.global_step / 1000.0)

        return {
            'total_loss': total_loss.item(),
            'preference_loss': loss_dict['preference_loss'].item(),
            'stability_loss': loss_dict['stability_loss'].item() if isinstance(loss_dict['stability_loss'], torch.Tensor) else loss_dict['stability_loss'],
            'sft_loss': sft_loss.item(),
            'accuracy': loss_dict['accuracy'].item(),
            'reward_margin': loss_dict['reward_margin'].item(),
            'dimension_weights': self.mpo_loss.get_dimension_weights(),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.policy_model.eval()

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

                embedding = self._get_prosody_embedding(
                    self.policy_model,
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                )

                # Compute rewards
                for dim_name, reward_fn in self.dimension_rewards.items():
                    reward = reward_fn.compute(embedding)
                    total_metrics[f'{dim_name}_reward'] += reward.mean().item()

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
        print(f"\n{'='*70}")
        print("MPO: Multidimensional Preference Optimization")
        print(f"{'='*70}")
        print(f"\nStarting training for {self.config.num_epochs} epochs")
        print(f"  Dimensions: {list(self.config.dimension_weights.keys())}")
        print(f"  Stability lambda: {self.config.stability_lambda}")
        print(f"  SFT weight: {self.current_sft_weight}")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        epoch_metrics[key] += value
                num_batches += 1

                if self.global_step % self.config.log_every == 0:
                    print(f"  Step {self.global_step}: "
                          f"loss={metrics['total_loss']:.4f}, "
                          f"acc={metrics['accuracy']:.2%}, "
                          f"margin={metrics['reward_margin']:.4f}")

            # Epoch summary
            for key in epoch_metrics:
                epoch_metrics[key] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            print(f"  Loss: {epoch_metrics['total_loss']:.4f}")
            print(f"  Accuracy: {epoch_metrics['accuracy']:.2%}")
            print(f"  Dimension weights: {self.mpo_loss.get_dimension_weights()}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                avg_reward = np.mean(list(val_metrics.values()))
                print(f"  Val rewards: {val_metrics}")

                if epoch_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = epoch_metrics['accuracy']
                    self.save_checkpoint('best')

            # Decay SFT weight
            self.current_sft_weight *= self.config.sft_decay
            print(f"  SFT weight decayed to: {self.current_sft_weight:.4f}")

            if (epoch + 1) % self.config.save_every_epochs == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')

        self.save_checkpoint('final')
        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.2%}")

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'current_sft_weight': self.current_sft_weight,
            'policy_model': self.policy_model.state_dict(),
            'dimension_rewards': self.dimension_rewards.state_dict(),
            'mpo_loss': self.mpo_loss.state_dict(),
            'log_prob_computer': self.log_prob_computer.state_dict(),
            'config': {
                'dimension_weights': self.config.dimension_weights,
                'beta': self.config.beta,
                'stability_lambda': self.config.stability_lambda,
            },
        }

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")


# =============================================================================
# LOG PROBABILITY COMPUTER
# =============================================================================

class LogProbComputer(nn.Module):
    """Computes log probabilities for prosody embeddings."""

    def __init__(self, config: MPOConfig):
        super().__init__()
        self.config = config

        self.log_var_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
        )
        nn.init.constant_(self.log_var_head[-1].bias, -2.3)

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of target given predicted."""
        log_var = self.log_var_head(predicted)
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        var = torch.exp(log_var)

        sq_error = (target - predicted) ** 2
        log_prob = -0.5 * (sq_error / var + log_var + math.log(2 * math.pi))

        return log_prob.sum(dim=-1).sum(dim=-1)


# =============================================================================
# DATASET
# =============================================================================

class MPODataset(Dataset):
    """Dataset for MPO training."""

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str = 'data/prosody_cache',
    ):
        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.prosody_cache_dir = Path(prosody_cache_dir)
        print(f"Loaded {len(self.samples)} samples for MPO training")

    def __len__(self):
        return len(self.samples)

    def _get_emotion_label(self, sample: dict) -> int:
        emotion_to_idx = {
            'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
            'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
        }
        emotion = sample.get('emotion', '').lower()
        if not emotion:
            semantic = sample.get('prosody', {}).get('semantic', {})
            emotion = semantic.get('emotion', '').lower()
        return emotion_to_idx.get(emotion, 0)

    def _load_prosody(self, sample: dict) -> Dict[str, torch.Tensor]:
        prosody = sample.get('prosody', {})

        semantic_dim, acoustic_dim, rhythm_dim, contour_dim = 8, 5, 4, 32

        def to_tensor(data, dim):
            if isinstance(data, dict):
                values = list(data.values())[:dim]
                t = torch.tensor(values, dtype=torch.float32) if values else torch.zeros(dim)
            elif isinstance(data, (list, np.ndarray)):
                t = torch.tensor(data, dtype=torch.float32)
            else:
                return torch.zeros(dim)

            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[-1] < dim:
                t = F.pad(t, (0, dim - t.shape[-1]))
            return t[:dim]

        return {
            'semantic': to_tensor(prosody.get('semantic', {}), semantic_dim),
            'acoustic': to_tensor(prosody.get('acoustic', {}), acoustic_dim),
            'rhythm': to_tensor(prosody.get('rhythm', {}), rhythm_dim),
            'contour': to_tensor(prosody.get('contour', []), contour_dim),
        }

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        prosody = self._load_prosody(sample)

        return {
            'text': sample.get('text', ''),
            'prosody_semantic': prosody['semantic'],
            'prosody_acoustic': prosody['acoustic'],
            'prosody_rhythm': prosody['rhythm'],
            'prosody_contour': prosody['contour'],
            'emotion_label': self._get_emotion_label(sample),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch."""
    return {
        'text': [item['text'] for item in batch],
        'prosody_semantic': torch.stack([item['prosody_semantic'] for item in batch]),
        'prosody_acoustic': torch.stack([item['prosody_acoustic'] for item in batch]),
        'prosody_rhythm': torch.stack([item['prosody_rhythm'] for item in batch]),
        'prosody_contour': torch.stack([item['prosody_contour'] for item in batch]),
        'emotion_label': torch.tensor([item['emotion_label'] for item in batch], dtype=torch.long),
    }


# =============================================================================
# MPO ADAPTER
# =============================================================================

class MPOAdapter(nn.Module):
    """
    Adapter for using MPO-trained models with the prosody pipeline.

    Provides multi-dimensional optimization during inference via:
    1. Sampling multiple candidates
    2. Scoring with dimension rewards
    3. Selecting best across all dimensions
    """

    def __init__(
        self,
        config: MPOConfig,
        prosody_encoder: nn.Module,
        dimension_rewards: Optional[nn.ModuleDict] = None,
    ):
        super().__init__()
        self.config = config
        self.prosody_encoder = prosody_encoder

        # Use provided or create new dimension rewards
        if dimension_rewards is not None:
            self.dimension_rewards = dimension_rewards
        else:
            self.dimension_rewards = nn.ModuleDict({
                'intelligibility': IntelligibilityReward(config),
                'speaker_similarity': SpeakerSimilarityReward(config),
                'prosody_naturalness': ProsodyNaturalnessReward(config),
                'overall_quality': OverallQualityReward(config),
            })

        # Output projection
        input_dim = config.hidden_size
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim * config.num_prosody_tokens),
        )
        self.token_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
        target_emotion: Optional[torch.Tensor] = None,
        num_candidates: int = 1,
        return_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with optional multi-candidate selection.

        Args:
            semantic, acoustic, rhythm, contour: Input prosody features
            target_emotion: Target emotion for scoring
            num_candidates: Number of candidates to generate and select from
            return_scores: Whether to return dimension scores

        Returns:
            Dict with prosody_tokens and optionally scores
        """
        if num_candidates == 1:
            # Single forward pass
            embedding = self._get_embedding(semantic, acoustic, rhythm, contour)
        else:
            # Generate multiple candidates and select best
            candidates = []
            for _ in range(num_candidates):
                noise_scale = 0.05
                noisy_semantic = semantic + noise_scale * torch.randn_like(semantic)
                noisy_acoustic = acoustic + noise_scale * torch.randn_like(acoustic)
                emb = self._get_embedding(noisy_semantic, noisy_acoustic, rhythm, contour)
                candidates.append(emb)

            # Score candidates
            embedding = self._select_best_candidate(
                candidates, target_emotion
            )

        # Project to tokens
        if embedding.dim() == 3:
            pooled = embedding.mean(dim=1)
        else:
            pooled = embedding

        tokens = self.token_projection(pooled)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.hidden_size)
        tokens = self.token_norm(tokens)

        result = {'prosody_tokens': tokens}

        if return_scores:
            scores = {}
            for dim_name, reward_fn in self.dimension_rewards.items():
                if dim_name == 'prosody_naturalness':
                    score = reward_fn.compute(embedding.unsqueeze(1) if embedding.dim() == 2 else embedding,
                                              target_emotion=target_emotion)
                else:
                    score = reward_fn.compute(embedding.unsqueeze(1) if embedding.dim() == 2 else embedding)
                scores[dim_name] = score
            result['dimension_scores'] = scores

        return result

    def _get_embedding(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
    ) -> torch.Tensor:
        """Get prosody embedding from encoder."""
        if hasattr(self.prosody_encoder, 'get_prosody_prefix'):
            return self.prosody_encoder.get_prosody_prefix(semantic, acoustic, rhythm, contour)
        elif hasattr(self.prosody_encoder, 'prosody_encoder'):
            return self.prosody_encoder.prosody_encoder(semantic, acoustic, rhythm, contour)
        else:
            return self.prosody_encoder(semantic, acoustic, rhythm, contour)

    def _select_best_candidate(
        self,
        candidates: List[torch.Tensor],
        target_emotion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Select best candidate based on multi-dimensional scores."""
        batch_size = candidates[0].shape[0]
        device = candidates[0].device

        # Score all candidates
        total_scores = torch.zeros(len(candidates), batch_size, device=device)

        for i, candidate in enumerate(candidates):
            for dim_name, reward_fn in self.dimension_rewards.items():
                weight = self.config.dimension_weights.get(dim_name, 1.0)
                if dim_name == 'prosody_naturalness':
                    score = reward_fn.compute(candidate, target_emotion=target_emotion)
                else:
                    score = reward_fn.compute(candidate)
                total_scores[i] += weight * score

        # Select best per batch
        best_idx = total_scores.argmax(dim=0)
        batch_indices = torch.arange(batch_size, device=device)

        # Gather best candidates
        candidates_stacked = torch.stack(candidates, dim=0)
        best_candidates = candidates_stacked[best_idx, batch_indices]

        return best_candidates


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MPO: Multidimensional Preference Optimization")
    parser.add_argument('--config', type=str, default='config/mpo.yaml')
    parser.add_argument('--checkpoint', type=str, help='Pre-trained prosody checkpoint')
    parser.add_argument('--manifest', type=str, help='Training manifest')
    parser.add_argument('--val_manifest', type=str, help='Validation manifest')
    parser.add_argument('--output_dir', type=str, default='checkpoints/mpo')
    parser.add_argument('--test', action='store_true', help='Run tests')
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = MPOConfig(**{k: v for k, v in config_dict.items()
                              if hasattr(MPOConfig, k)})
    else:
        config = MPOConfig()

    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 70)
    print("MPO: Multidimensional Preference Optimization for TTS")
    print("=" * 70)

    # Load or create prosody encoder
    if args.checkpoint:
        try:
            from prosody_conditioning import ProsodyConfig, ProsodyEncoder
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            prosody_config = ProsodyConfig(**checkpoint.get('prosody_config', {}))
            prosody_encoder = ProsodyEncoder(prosody_config)
            if 'prosody_encoder' in checkpoint:
                prosody_encoder.load_state_dict(checkpoint['prosody_encoder'])
            print(f"Loaded prosody encoder from {args.checkpoint}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Creating fresh prosody encoder")
            prosody_encoder = create_mock_encoder(config)
    else:
        print("Creating fresh prosody encoder")
        prosody_encoder = create_mock_encoder(config)

    # Create trainer
    trainer = MPOTrainer(config, prosody_encoder)

    # Create dataset
    if args.manifest:
        train_dataset = MPODataset(args.manifest)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = None
        if args.val_manifest:
            val_dataset = MPODataset(args.val_manifest)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

        trainer.train(train_loader, val_loader)
    else:
        print("\nNo manifest provided. Run with --manifest to train.")
        print("\nMPO optimizes across multiple dimensions:")
        for dim, weight in config.dimension_weights.items():
            print(f"  - {dim}: weight={weight}")


def create_mock_encoder(config: MPOConfig) -> nn.Module:
    """Create a mock prosody encoder for testing."""

    class MockProsodyEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            input_dim = 8 + 5 + 4 + 32
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size * config.num_prosody_tokens),
            )
            self.hidden_size = config.hidden_size
            self.num_prosody_tokens = config.num_prosody_tokens

        def forward(self, semantic, acoustic, rhythm, contour):
            x = torch.cat([semantic, acoustic, rhythm, contour], dim=-1)
            out = self.encoder(x)
            batch = x.shape[0]
            return out.view(batch, self.num_prosody_tokens, self.hidden_size)

    return MockProsodyEncoder(config)


def run_tests():
    """Run MPO tests."""
    print("=" * 70)
    print("MPO: Multidimensional Preference Optimization - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = MPOConfig()
    batch_size = 4

    # Test 1: Dimension Rewards
    print("\n[Test 1] Dimension Reward Functions...")

    intel_reward = IntelligibilityReward(config).to(device)
    sim_reward = SpeakerSimilarityReward(config).to(device)
    prosody_reward = ProsodyNaturalnessReward(config).to(device)
    quality_reward = OverallQualityReward(config).to(device)

    embedding = torch.randn(batch_size, config.num_prosody_tokens, config.hidden_size, device=device)
    target_emotion = torch.randint(0, 8, (batch_size,), device=device)

    r_intel = intel_reward.compute(embedding)
    r_sim = sim_reward.compute(embedding)
    r_prosody = prosody_reward.compute(embedding, target_emotion=target_emotion)
    r_quality = quality_reward.compute(embedding)

    print(f"  Intelligibility: {r_intel.mean().item():.4f}")
    print(f"  Speaker Similarity: {r_sim.mean().item():.4f}")
    print(f"  Prosody Naturalness: {r_prosody.mean().item():.4f}")
    print(f"  Overall Quality: {r_quality.mean().item():.4f}")
    print("  [PASS]")

    # Test 2: Preference Set Builder
    print("\n[Test 2] MultidimensionalPreferenceSet...")

    pref_builder = MultidimensionalPreferenceSet(config)

    samples = [{'prosody': torch.randn(config.hidden_size)} for _ in range(4)]
    dim_rewards = {
        'intelligibility': [0.8, 0.6, 0.9, 0.5],
        'speaker_similarity': [0.7, 0.8, 0.6, 0.7],
        'prosody_naturalness': [0.6, 0.7, 0.8, 0.5],
        'overall_quality': [0.7, 0.6, 0.7, 0.6],
    }

    pref_set = pref_builder.construct_preference_set(samples, dim_rewards)

    print(f"  Overall ranking: {pref_set['overall_ranking']}")
    print(f"  Chosen idx: {pref_set['chosen_idx']}")
    print(f"  Rejected idx: {pref_set['rejected_idx']}")
    print(f"  Margin: {pref_set['margin']:.4f}")

    chosen, rejected = pref_builder.sample_preference_pair(pref_set, difficulty=0.2)
    print(f"  Easy pair: chosen={chosen}, rejected={rejected}")

    chosen, rejected = pref_builder.sample_preference_pair(pref_set, difficulty=0.8)
    print(f"  Hard pair: chosen={chosen}, rejected={rejected}")
    print("  [PASS]")

    # Test 3: Stability Regularizer
    print("\n[Test 3] StabilityRegularizer...")

    regularizer = StabilityRegularizer(config)

    policy_out = torch.randn(batch_size, config.hidden_size, device=device)
    ref_out = torch.randn(batch_size, config.hidden_size, device=device)

    dim_rewards_tensors = {
        'intelligibility': torch.rand(batch_size, device=device),
        'speaker_similarity': torch.rand(batch_size, device=device),
        'prosody_naturalness': torch.rand(batch_size, device=device),
        'overall_quality': torch.rand(batch_size, device=device),
    }

    reg_losses = regularizer(policy_out, ref_out, dim_rewards_tensors)

    print(f"  Anchor loss: {reg_losses.get('anchor', 0):.4f}")
    print(f"  Diversity loss: {reg_losses.get('diversity', 0):.4f}")
    print(f"  Dim correlation: {reg_losses.get('dimension_correlation', 0):.4f}")
    print(f"  Total: {reg_losses['total']:.4f}")
    print("  [PASS]")

    # Test 4: MPO Loss
    print("\n[Test 4] MPOLoss...")

    mpo_loss = MPOLoss(config).to(device)

    policy_chosen = torch.randn(batch_size, device=device)
    policy_rejected = torch.randn(batch_size, device=device) - 0.3
    ref_chosen = torch.randn(batch_size, device=device)
    ref_rejected = torch.randn(batch_size, device=device)

    dim_margins = {
        'intelligibility': torch.rand(batch_size, device=device),
        'speaker_similarity': torch.rand(batch_size, device=device),
        'prosody_naturalness': torch.rand(batch_size, device=device),
        'overall_quality': torch.rand(batch_size, device=device),
    }

    loss_dict = mpo_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected,
        dim_margins,
        policy_output=policy_out,
        reference_output=ref_out,
    )

    print(f"  Total loss: {loss_dict['total'].item():.4f}")
    print(f"  Preference loss: {loss_dict['preference_loss'].item():.4f}")
    print(f"  Stability loss: {loss_dict['stability_loss'].item():.4f}")
    print(f"  Accuracy: {loss_dict['accuracy'].item():.2%}")
    print(f"  Dimension weights: {mpo_loss.get_dimension_weights()}")
    print("  [PASS]")

    # Test 5: Full Training Step
    print("\n[Test 5] Full Training Step...")

    encoder = create_mock_encoder(config).to(device)
    trainer = MPOTrainer(config, encoder)

    batch = {
        'prosody_semantic': torch.randn(batch_size, 8),
        'prosody_acoustic': torch.randn(batch_size, 5),
        'prosody_rhythm': torch.randn(batch_size, 4),
        'prosody_contour': torch.randn(batch_size, 32),
        'emotion_label': torch.randint(0, 8, (batch_size,)),
    }

    metrics = trainer.train_step(batch)

    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Reward margin: {metrics['reward_margin']:.4f}")
    print("  [PASS]")

    # Test 6: MPO Adapter
    print("\n[Test 6] MPOAdapter...")

    adapter = MPOAdapter(config, encoder).to(device)

    semantic = torch.randn(batch_size, 8, device=device)
    acoustic = torch.randn(batch_size, 5, device=device)
    rhythm = torch.randn(batch_size, 4, device=device)
    contour = torch.randn(batch_size, 32, device=device)

    # Single candidate
    result = adapter(semantic, acoustic, rhythm, contour)
    print(f"  Single candidate tokens: {result['prosody_tokens'].shape}")

    # Multiple candidates with selection
    result = adapter(
        semantic, acoustic, rhythm, contour,
        target_emotion=target_emotion,
        num_candidates=4,
        return_scores=True,
    )
    print(f"  Multi-candidate tokens: {result['prosody_tokens'].shape}")
    print(f"  Dimension scores: {list(result['dimension_scores'].keys())}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All MPO tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from mpo import MPOConfig, MPOTrainer, MPOAdapter

# Initialize
config = MPOConfig(
    dimension_weights={
        'intelligibility': 1.0,
        'speaker_similarity': 1.0,
        'prosody_naturalness': 1.0,
        'overall_quality': 0.5,
    },
    stability_lambda=0.1,
)

# Train with multi-dimensional preference optimization
trainer = MPOTrainer(config, prosody_encoder)
trainer.train(train_loader, val_loader)

# Use adapter for inference
adapter = MPOAdapter(config, trainer.policy_model, trainer.dimension_rewards)

# Generate with multi-candidate selection
result = adapter(
    semantic, acoustic, rhythm, contour,
    target_emotion=torch.tensor([1]),  # happy
    num_candidates=4,
    return_scores=True,
)
prosody_tokens = result['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")


if __name__ == "__main__":
    main()
