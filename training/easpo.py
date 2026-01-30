"""
EASPO: Emotion-Aware Stepwise Preference Optimization for Diffusion TTS

Based on arXiv:2509.25416 (Sept 2025). Post-training framework for emotion alignment
with dense supervision at each denoising step.

Key Innovation - EASPO (Emotion-Aware Stepwise Preference Optimization):
- Reformulates preference optimization as local, time-conditioned task
- Aligns win/lose candidates at EACH denoising step from shared latent
- Dense emotion-aligned rewards vs sparse utterance-level feedback (Emo-DPO)
- EASPM (Emotion-Aware Stepwise Preference Model) scores noisy intermediate states

How it works:
1. At each denoising step, produce candidate samples (branching)
2. EASPM scores emotional expressiveness on noisy states
3. Select win-lose pair that differs subtly in prosody
4. Update model with stepwise preference signal
5. Random sample continues to next step

Advantages over Emo-DPO:
- Dense temporal supervision (every step) vs sparse (final output)
- More nuanced preference signal at each noise level
- Better emotion alignment through stepwise guidance
- Captures prosodic details that affect emotional perception

References:
- EASPO: https://arxiv.org/abs/2509.25416
- Emo-DPO: https://arxiv.org/abs/2409.10157
- DDPO: https://arxiv.org/abs/2305.13301 (diffusion DPO)
"""

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import from existing implementations
from prosody_flow import (
    ProsodyFlowConfig,
    ProsodyFlow,
    VectorFieldNetwork,
    GaussianConditionalPath,
    ODESolver,
    SinusoidalTimeEmbedding,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EASPOConfig:
    """Configuration for EASPO training."""

    # Base flow model config
    base_config: ProsodyFlowConfig = field(default_factory=ProsodyFlowConfig)

    # EASPM (Emotion-Aware Stepwise Preference Model) settings
    easpm_hidden_dim: int = 512
    easpm_num_layers: int = 4
    easpm_num_heads: int = 8
    easpm_dropout: float = 0.1

    # Emotion representation
    emotion_dim: int = 256
    num_emotions: int = 8
    use_vad: bool = True
    vad_dim: int = 3

    # Stepwise preference settings
    num_denoising_steps: int = 50  # Total diffusion steps
    num_candidates_per_step: int = 4  # Candidates to sample at each step
    preference_temperature: float = 1.0  # Temperature for preference scoring

    # DPO-style loss settings
    beta: float = 0.1  # KL penalty coefficient
    js_alpha: float = 0.5  # Jensen-Shannon regularization

    # Stepwise loss weighting
    step_weight_schedule: str = "linear"  # linear, cosine, uniform
    early_step_weight: float = 0.5  # Weight for early steps (t near 0)
    late_step_weight: float = 1.0  # Weight for late steps (t near 1)

    # Multi-scale preference (coarse + fine)
    use_multiscale: bool = True
    coarse_step_interval: int = 10  # Evaluate every 10 steps for coarse
    fine_step_interval: int = 1  # Evaluate every step for fine-grained

    # Training settings
    learning_rate: float = 1e-5
    easpm_learning_rate: float = 1e-4  # Separate LR for reward model
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation: int = 2
    max_grad_norm: float = 1.0

    # Reference model settings
    use_reference_model: bool = True
    freeze_reference: bool = True

    # EASPM pre-training
    pretrain_easpm_epochs: int = 1  # Pre-train EASPM before EASPO
    easpm_contrastive_margin: float = 0.5

    # Output settings
    num_prosody_tokens: int = 4


# Emotion mappings
EMOTION_TO_IDX = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
    'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}

# VAD prototypes (Valence-Arousal-Dominance)
VAD_PROTOTYPES = {
    0: (0.0, 0.0, 0.0),    # neutral
    1: (0.8, 0.6, 0.6),    # happy
    2: (-0.6, -0.4, -0.5), # sad
    3: (-0.5, 0.8, 0.7),   # angry
    4: (-0.7, 0.7, -0.7),  # fearful
    5: (0.3, 0.8, 0.2),    # surprised
    6: (-0.6, 0.3, 0.4),   # disgusted
    7: (0.4, -0.5, 0.3),   # calm
}


# =============================================================================
# EASPM: EMOTION-AWARE STEPWISE PREFERENCE MODEL
# =============================================================================

class TimeConditionedEmotionEncoder(nn.Module):
    """
    Encodes emotion conditioned on diffusion time step.

    The expected emotion characteristics change based on noise level:
    - Early steps (t ≈ 0): Coarse emotional structure
    - Late steps (t ≈ 1): Fine-grained prosodic details
    """

    def __init__(self, config: EASPOConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(config.easpm_hidden_dim)

        # Emotion embedding
        self.emotion_embed = nn.Embedding(config.num_emotions, config.emotion_dim)

        # VAD encoder
        if config.use_vad:
            self.vad_encoder = nn.Sequential(
                nn.Linear(config.vad_dim, config.emotion_dim),
                nn.LayerNorm(config.emotion_dim),
                nn.GELU(),
                nn.Linear(config.emotion_dim, config.emotion_dim),
            )

        # VAD prototypes buffer
        vad_values = torch.tensor([VAD_PROTOTYPES[i] for i in range(config.num_emotions)])
        self.register_buffer('vad_prototypes', vad_values)

        # Time-emotion fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.emotion_dim + config.easpm_hidden_dim, config.easpm_hidden_dim),
            nn.LayerNorm(config.easpm_hidden_dim),
            nn.GELU(),
            nn.Linear(config.easpm_hidden_dim, config.easpm_hidden_dim),
        )

    def forward(
        self,
        t: torch.Tensor,
        emotion_ids: Optional[torch.Tensor] = None,
        vad_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode emotion conditioned on time step.

        Args:
            t: Diffusion time [batch]
            emotion_ids: Discrete emotion indices [batch]
            vad_coords: Continuous VAD coordinates [batch, 3]

        Returns:
            Time-conditioned emotion embedding [batch, hidden_dim]
        """
        batch_size = t.shape[0]

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, hidden_dim]

        # Emotion embedding
        if vad_coords is not None and self.config.use_vad:
            emotion_emb = self.vad_encoder(vad_coords)
        elif emotion_ids is not None:
            emotion_emb = self.emotion_embed(emotion_ids)
            # Optionally add VAD info
            if self.config.use_vad:
                vad = self.vad_prototypes[emotion_ids]
                vad_emb = self.vad_encoder(vad)
                emotion_emb = emotion_emb + 0.5 * vad_emb
        else:
            # Default to neutral
            emotion_emb = self.emotion_embed(
                torch.zeros(batch_size, dtype=torch.long, device=t.device)
            )

        # Fuse time and emotion
        combined = torch.cat([emotion_emb, t_emb], dim=-1)
        return self.fusion(combined)


class NoisyStateEncoder(nn.Module):
    """
    Encodes noisy intermediate states for preference scoring.

    Key insight: The encoder must be able to extract emotional information
    from noisy latent states at various noise levels.
    """

    def __init__(self, config: EASPOConfig):
        super().__init__()
        self.config = config

        # Input projection (prosody_dim + time_emb → hidden)
        self.input_proj = nn.Sequential(
            nn.Linear(config.base_config.prosody_dim + config.easpm_hidden_dim,
                      config.easpm_hidden_dim),
            nn.LayerNorm(config.easpm_hidden_dim),
            nn.GELU(),
        )

        # Noise-aware attention layers
        self.layers = nn.ModuleList([
            NoisyStateAttentionBlock(config)
            for _ in range(config.easpm_num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.easpm_hidden_dim),
            nn.Linear(config.easpm_hidden_dim, config.easpm_hidden_dim),
            nn.GELU(),
            nn.Linear(config.easpm_hidden_dim, config.easpm_hidden_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode noisy state.

        Args:
            x_t: Noisy prosody latent [batch, prosody_dim]
            t: Diffusion time [batch]

        Returns:
            State encoding [batch, hidden_dim]
        """
        # Time embedding
        time_embed = SinusoidalTimeEmbedding(self.config.easpm_hidden_dim)
        t_emb = time_embed(t)  # [batch, hidden_dim]

        # Input projection
        x_input = torch.cat([x_t, t_emb], dim=-1)
        h = self.input_proj(x_input)  # [batch, hidden_dim]

        # Apply attention layers
        h = h.unsqueeze(1)  # [batch, 1, hidden_dim]
        for layer in self.layers:
            h = layer(h, t)
        h = h.squeeze(1)  # [batch, hidden_dim]

        return self.output_proj(h)


class NoisyStateAttentionBlock(nn.Module):
    """Attention block for noisy state encoding."""

    def __init__(self, config: EASPOConfig):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            config.easpm_hidden_dim,
            config.easpm_num_heads,
            dropout=config.easpm_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.easpm_hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(config.easpm_hidden_dim, config.easpm_hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.easpm_dropout),
            nn.Linear(config.easpm_hidden_dim * 4, config.easpm_hidden_dim),
            nn.Dropout(config.easpm_dropout),
        )
        self.norm2 = nn.LayerNorm(config.easpm_hidden_dim)

        # Time-adaptive layer norm
        self.time_scale = nn.Linear(1, config.easpm_hidden_dim)
        self.time_shift = nn.Linear(1, config.easpm_hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Time-adaptive modulation
        t_input = t.unsqueeze(-1)  # [batch, 1]
        scale = self.time_scale(t_input).unsqueeze(1)  # [batch, 1, hidden]
        shift = self.time_shift(t_input).unsqueeze(1)

        # FFN with time modulation
        x = x + self.ffn(self.norm2(x) * (1 + scale) + shift)

        return x


class EASPM(nn.Module):
    """
    Emotion-Aware Stepwise Preference Model.

    Scores noisy intermediate states for emotional expressiveness.
    Key capability: Can evaluate emotion quality at any noise level.

    The model learns to predict:
    - Emotional expressiveness score for a given (x_t, t, emotion) tuple
    - Relative preference between two samples at the same step
    """

    def __init__(self, config: EASPOConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.state_encoder = NoisyStateEncoder(config)
        self.emotion_encoder = TimeConditionedEmotionEncoder(config)

        # Preference head: computes compatibility between state and target emotion
        self.preference_head = nn.Sequential(
            nn.Linear(config.easpm_hidden_dim * 2, config.easpm_hidden_dim),
            nn.LayerNorm(config.easpm_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.easpm_dropout),
            nn.Linear(config.easpm_hidden_dim, config.easpm_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.easpm_hidden_dim // 2, 1),
        )

        # Auxiliary emotion classifier (for multi-task learning)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.easpm_hidden_dim, config.easpm_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.easpm_hidden_dim // 2, config.num_emotions),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target_emotion_ids: Optional[torch.Tensor] = None,
        target_vad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Score a noisy state for emotional expressiveness.

        Args:
            x_t: Noisy prosody latent [batch, prosody_dim]
            t: Diffusion time [batch]
            target_emotion_ids: Target emotion indices [batch]
            target_vad: Target VAD coordinates [batch, 3]

        Returns:
            Dict with 'score', 'state_emb', 'emotion_emb', 'emotion_logits'
        """
        # Encode state
        state_emb = self.state_encoder(x_t, t)  # [batch, hidden_dim]

        # Encode target emotion
        emotion_emb = self.emotion_encoder(t, target_emotion_ids, target_vad)  # [batch, hidden_dim]

        # Compute preference score
        combined = torch.cat([state_emb, emotion_emb], dim=-1)
        score = self.preference_head(combined).squeeze(-1)  # [batch]

        # Auxiliary emotion classification
        emotion_logits = self.emotion_classifier(state_emb)  # [batch, num_emotions]

        return {
            'score': score,
            'state_emb': state_emb,
            'emotion_emb': emotion_emb,
            'emotion_logits': emotion_logits,
        }

    def compute_pairwise_preference(
        self,
        x_t_1: torch.Tensor,
        x_t_2: torch.Tensor,
        t: torch.Tensor,
        target_emotion_ids: Optional[torch.Tensor] = None,
        target_vad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relative preference between two samples.

        Args:
            x_t_1: First sample [batch, prosody_dim]
            x_t_2: Second sample [batch, prosody_dim]
            t: Diffusion time [batch]
            target_emotion_ids: Target emotion [batch]
            target_vad: Target VAD [batch, 3]

        Returns:
            Dict with preference logits and probabilities
        """
        # Score both samples
        out1 = self.forward(x_t_1, t, target_emotion_ids, target_vad)
        out2 = self.forward(x_t_2, t, target_emotion_ids, target_vad)

        # Preference logits: which is better for target emotion?
        preference_logits = out1['score'] - out2['score']
        preference_prob = torch.sigmoid(preference_logits)

        return {
            'preference_logits': preference_logits,
            'preference_prob': preference_prob,
            'score_1': out1['score'],
            'score_2': out2['score'],
        }


# =============================================================================
# CANDIDATE SAMPLER
# =============================================================================

class StepwiseCandidateSampler:
    """
    Samples candidate prosody trajectories at each denoising step.

    Strategy:
    1. At step t, generate K candidate samples via different noise
    2. Score all candidates with EASPM
    3. Select best (win) and worst (lose) for preference learning
    4. Randomly select one to continue to next step
    """

    def __init__(
        self,
        flow_model: ProsodyFlow,
        easpm: EASPM,
        config: EASPOConfig,
    ):
        self.flow_model = flow_model
        self.easpm = easpm
        self.config = config
        self.path = GaussianConditionalPath(config.base_config.sigma_min)

    @torch.no_grad()
    def sample_candidates_at_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        target_emotion_ids: Optional[torch.Tensor] = None,
        num_candidates: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate and score candidate samples at a single step.

        Args:
            x_t: Current noisy state [batch, prosody_dim]
            t: Current time [batch]
            dt: Time step size
            text_cond: Text conditioning
            text_mask: Text mask
            target_emotion_ids: Target emotion for scoring
            num_candidates: Number of candidates to generate

        Returns:
            Dict with candidates, scores, and win/lose indices
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Generate candidates by adding different noise perturbations
        candidates = []
        for _ in range(num_candidates):
            # Predict velocity with small noise perturbation
            noise_scale = 0.1 * (1 - t.mean().item())  # Less noise as t → 1
            x_perturbed = x_t + noise_scale * torch.randn_like(x_t)

            # Take one step
            velocity = self.flow_model.vector_field(t, x_perturbed, text_cond, text_mask)
            x_next = x_perturbed + dt * velocity

            candidates.append(x_next)

        # Stack candidates: [num_candidates, batch, prosody_dim]
        candidates = torch.stack(candidates, dim=0)

        # Score all candidates
        scores = []
        t_next = t + dt
        for i in range(num_candidates):
            out = self.easpm(
                candidates[i],
                t_next.expand(batch_size),
                target_emotion_ids,
            )
            scores.append(out['score'])

        scores = torch.stack(scores, dim=0)  # [num_candidates, batch]

        # Find win (highest score) and lose (lowest score) per batch
        win_idx = scores.argmax(dim=0)  # [batch]
        lose_idx = scores.argmin(dim=0)  # [batch]

        # Handle case where win == lose (tie)
        tie_mask = win_idx == lose_idx
        if tie_mask.any():
            # Use random for ties
            random_idx = torch.randint(0, num_candidates, (batch_size,), device=device)
            lose_idx = torch.where(tie_mask, random_idx, lose_idx)

        # Extract win/lose samples
        batch_indices = torch.arange(batch_size, device=device)
        win_samples = candidates[win_idx, batch_indices]  # [batch, prosody_dim]
        lose_samples = candidates[lose_idx, batch_indices]  # [batch, prosody_dim]

        # Random continuation (for next step)
        random_idx = torch.randint(0, num_candidates, (batch_size,), device=device)
        continue_samples = candidates[random_idx, batch_indices]

        return {
            'candidates': candidates,  # [K, batch, dim]
            'scores': scores,  # [K, batch]
            'win_samples': win_samples,  # [batch, dim]
            'lose_samples': lose_samples,  # [batch, dim]
            'continue_samples': continue_samples,  # [batch, dim]
            'win_scores': scores[win_idx, batch_indices],
            'lose_scores': scores[lose_idx, batch_indices],
            'win_idx': win_idx,
            'lose_idx': lose_idx,
        }


# =============================================================================
# STEPWISE PREFERENCE LOSS
# =============================================================================

class StepwisePreferenceLoss(nn.Module):
    """
    Stepwise preference optimization loss.

    Extends DPO to operate at each denoising step:
    L_step(t) = -log σ(β * (r(x_t^win) - r(x_t^lose)))

    Total loss is weighted sum over all steps:
    L = Σ_t w(t) * L_step(t)
    """

    def __init__(self, config: EASPOConfig):
        super().__init__()
        self.config = config
        self.beta = config.beta
        self.js_alpha = config.js_alpha

    def compute_step_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute loss weight for step t.

        Early steps (t ≈ 0) get lower weight since they're more noisy.
        Late steps (t ≈ 1) get higher weight for fine-grained alignment.
        """
        if self.config.step_weight_schedule == "uniform":
            return torch.ones_like(t)

        elif self.config.step_weight_schedule == "linear":
            # Linear interpolation from early to late weight
            weight = (
                self.config.early_step_weight * (1 - t) +
                self.config.late_step_weight * t
            )
            return weight

        elif self.config.step_weight_schedule == "cosine":
            # Cosine schedule
            weight = (
                self.config.early_step_weight +
                (self.config.late_step_weight - self.config.early_step_weight) *
                (1 - torch.cos(t * math.pi)) / 2
            )
            return weight

        else:
            return torch.ones_like(t)

    def forward(
        self,
        policy_win_logps: torch.Tensor,    # [batch] log P(x_t^win)
        policy_lose_logps: torch.Tensor,   # [batch] log P(x_t^lose)
        ref_win_logps: torch.Tensor,       # [batch] log P_ref(x_t^win)
        ref_lose_logps: torch.Tensor,      # [batch] log P_ref(x_t^lose)
        t: torch.Tensor,                   # [batch] current time step
        win_scores: Optional[torch.Tensor] = None,   # EASPM scores
        lose_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stepwise preference loss.

        Args:
            policy_win_logps: Policy log prob for winning sample
            policy_lose_logps: Policy log prob for losing sample
            ref_win_logps: Reference log prob for winning sample
            ref_lose_logps: Reference log prob for losing sample
            t: Current diffusion time
            win_scores: Optional EASPM scores for additional reward signal
            lose_scores: Optional EASPM scores for additional reward signal

        Returns:
            Loss dict with components
        """
        # Log ratios
        win_logratios = policy_win_logps - ref_win_logps
        lose_logratios = policy_lose_logps - ref_lose_logps

        # DPO logits
        base_logits = self.beta * (win_logratios - lose_logratios)

        # Optional: incorporate EASPM scores as additional reward
        if win_scores is not None and lose_scores is not None:
            score_diff = (win_scores - lose_scores).detach()
            # Weight by score difference magnitude
            score_weight = torch.sigmoid(score_diff)
            base_logits = base_logits * score_weight

        # JS regularization
        js_chosen = F.softplus(win_logratios)
        js_rejected = F.softplus(lose_logratios)
        js_term = js_chosen - js_rejected

        regularized_logits = base_logits - self.js_alpha * js_term

        # Step-weighted loss
        step_weights = self.compute_step_weight(t)
        step_loss = -F.logsigmoid(regularized_logits)
        weighted_loss = (step_loss * step_weights).mean()

        # Metrics
        with torch.no_grad():
            accuracy = (regularized_logits > 0).float().mean()
            win_rewards = self.beta * win_logratios
            lose_rewards = self.beta * lose_logratios
            reward_margin = win_rewards - lose_rewards

        return {
            'loss': weighted_loss,
            'step_loss': step_loss.mean(),
            'accuracy': accuracy,
            'reward_margin': reward_margin.mean(),
            'win_rewards': win_rewards.mean(),
            'lose_rewards': lose_rewards.mean(),
            'js_term': js_term.mean(),
            'step_weights': step_weights.mean(),
        }


# =============================================================================
# EASPO TRAINER
# =============================================================================

class EASPOTrainer:
    """
    EASPO training framework.

    Training procedure:
    1. Initialize flow model and EASPM
    2. Pre-train EASPM on emotion classification
    3. For each training sample:
       a. Run full denoising trajectory
       b. At each step, sample candidates and compute preference
       c. Accumulate stepwise preference loss
       d. Update flow model with accumulated gradients
    """

    def __init__(
        self,
        config: EASPOConfig,
        flow_model: ProsodyFlow,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or self._setup_device()

        # Flow model (policy)
        self.flow_model = flow_model.to(self.device)

        # Reference model (frozen copy)
        if config.use_reference_model:
            self.ref_model = copy.deepcopy(flow_model)
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print("Created frozen reference model")
        else:
            self.ref_model = None

        # EASPM (reward model)
        self.easpm = EASPM(config).to(self.device)

        # Candidate sampler
        self.candidate_sampler = StepwiseCandidateSampler(
            flow_model=self.flow_model,
            easpm=self.easpm,
            config=config,
        )

        # Loss functions
        self.stepwise_loss = StepwisePreferenceLoss(config)
        self.easpm_ce_loss = nn.CrossEntropyLoss()

        # Log prob computer
        self.log_prob_computer = StepwiseLogProbComputer(config)

        # Optimizers
        self.flow_optimizer = torch.optim.AdamW(
            self.flow_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.easpm_optimizer = torch.optim.AdamW(
            self.easpm.parameters(),
            lr=config.easpm_learning_rate,
            weight_decay=0.01,
        )

        # Gaussian path
        self.path = GaussianConditionalPath(config.base_config.sigma_min)

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def pretrain_easpm(
        self,
        train_loader: DataLoader,
        num_epochs: int = 1,
    ):
        """
        Pre-train EASPM on emotion classification.

        This helps the model learn to extract emotion from noisy states
        before using it for preference learning.
        """
        print("\n[EASPM Pre-training]")

        self.easpm.train()

        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            num_batches = 0

            for batch in train_loader:
                # Get prosody targets
                prosody = batch['prosody'].to(self.device)
                emotion_ids = batch['emotion_ids'].to(self.device)

                # Sample random time
                t = torch.rand(prosody.shape[0], device=self.device)

                # Create noisy samples
                noise = torch.randn_like(prosody)
                x_t = self.path.sample_xt(t, noise, prosody)

                # Forward pass
                out = self.easpm(x_t, t, emotion_ids)

                # Emotion classification loss
                loss = self.easpm_ce_loss(out['emotion_logits'], emotion_ids)

                # Backward
                self.easpm_optimizer.zero_grad()
                loss.backward()
                self.easpm_optimizer.step()

                # Metrics
                pred = out['emotion_logits'].argmax(dim=-1)
                acc = (pred == emotion_ids).float().mean()

                total_loss += loss.item()
                total_acc += acc.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            avg_acc = total_acc / max(1, num_batches)
            print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

    def train_step(
        self,
        prosody_target: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single EASPO training step.

        Runs full denoising trajectory and collects stepwise preferences.
        """
        self.flow_model.train()
        self.easpm.eval()  # Freeze EASPM during policy training

        batch_size = prosody_target.shape[0]
        num_steps = self.config.num_denoising_steps
        dt = 1.0 / num_steps

        # Start from noise
        x = torch.randn_like(prosody_target)

        # Collect losses over trajectory
        total_step_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0
        num_preference_steps = 0

        # Track which steps to evaluate (multiscale)
        if self.config.use_multiscale:
            coarse_steps = set(range(0, num_steps, self.config.coarse_step_interval))
            fine_steps = set(range(0, num_steps, self.config.fine_step_interval))
            eval_steps = coarse_steps | fine_steps
        else:
            eval_steps = set(range(num_steps))

        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=self.device)
            t_next = t + dt

            if step in eval_steps:
                # Sample candidates and compute preference
                with torch.no_grad():
                    candidates = self.candidate_sampler.sample_candidates_at_step(
                        x, t, dt, text_cond, text_mask, emotion_ids,
                        num_candidates=self.config.num_candidates_per_step,
                    )

                win_samples = candidates['win_samples']
                lose_samples = candidates['lose_samples']

                # Compute log probabilities for policy
                policy_win_logps = self.log_prob_computer.compute_step_logprob(
                    self.flow_model, x, win_samples, t, dt, text_cond, text_mask
                )
                policy_lose_logps = self.log_prob_computer.compute_step_logprob(
                    self.flow_model, x, lose_samples, t, dt, text_cond, text_mask
                )

                # Reference log probs
                with torch.no_grad():
                    if self.ref_model is not None:
                        ref_win_logps = self.log_prob_computer.compute_step_logprob(
                            self.ref_model, x, win_samples, t, dt, text_cond, text_mask
                        )
                        ref_lose_logps = self.log_prob_computer.compute_step_logprob(
                            self.ref_model, x, lose_samples, t, dt, text_cond, text_mask
                        )
                    else:
                        ref_win_logps = policy_win_logps.detach()
                        ref_lose_logps = policy_lose_logps.detach()

                # Compute stepwise preference loss
                loss_out = self.stepwise_loss(
                    policy_win_logps,
                    policy_lose_logps,
                    ref_win_logps,
                    ref_lose_logps,
                    t_next,
                    candidates['win_scores'],
                    candidates['lose_scores'],
                )

                total_step_loss += loss_out['loss']
                total_accuracy += loss_out['accuracy'].item()
                total_reward_margin += loss_out['reward_margin'].item()
                num_preference_steps += 1

                # Use random continuation for next step
                x = candidates['continue_samples']
            else:
                # Just take a normal step (no preference)
                with torch.no_grad():
                    velocity = self.flow_model.vector_field(t, x, text_cond, text_mask)
                    x = x + dt * velocity

        # Average loss over steps
        if num_preference_steps > 0:
            avg_step_loss = total_step_loss / num_preference_steps

            # Backward pass
            self.flow_optimizer.zero_grad()
            avg_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.flow_model.parameters(),
                self.config.max_grad_norm
            )
            self.flow_optimizer.step()

            self.global_step += 1

            return {
                'total_loss': avg_step_loss.item(),
                'step_loss': total_step_loss.item() / num_preference_steps,
                'accuracy': total_accuracy / num_preference_steps,
                'reward_margin': total_reward_margin / num_preference_steps,
                'num_steps_evaluated': num_preference_steps,
            }

        return {'total_loss': 0.0, 'accuracy': 0.0}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("EASPO: Emotion-Aware Stepwise Preference Optimization")
        print(f"{'='*60}")

        # Pre-train EASPM
        if self.config.pretrain_easpm_epochs > 0:
            self.pretrain_easpm(train_loader, self.config.pretrain_easpm_epochs)

        print(f"\nStarting EASPO training for {self.config.num_epochs} epochs")
        print(f"  Denoising steps: {self.config.num_denoising_steps}")
        print(f"  Candidates per step: {self.config.num_candidates_per_step}")
        print(f"  Step weight schedule: {self.config.step_weight_schedule}")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Extract batch data
                prosody = batch['prosody'].to(self.device)
                text_cond = batch.get('text_cond')
                if text_cond is not None:
                    text_cond = text_cond.to(self.device)
                text_mask = batch.get('text_mask')
                if text_mask is not None:
                    text_mask = text_mask.to(self.device)
                emotion_ids = batch.get('emotion_ids')
                if emotion_ids is not None:
                    emotion_ids = emotion_ids.to(self.device)

                # Train step
                metrics = self.train_step(prosody, text_cond, text_mask, emotion_ids)

                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1

                # Log
                if self.global_step % 10 == 0:
                    print(f"  Step {self.global_step}: "
                          f"loss={metrics['total_loss']:.4f}, "
                          f"acc={metrics['accuracy']:.2%}")

            # Epoch summary
            for key in epoch_metrics:
                epoch_metrics[key] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            print(f"  Loss: {epoch_metrics['total_loss']:.4f}")
            print(f"  Accuracy: {epoch_metrics['accuracy']:.2%}")
            print(f"  Reward margin: {epoch_metrics['reward_margin']:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"  Val accuracy: {val_metrics['accuracy']:.2%}")

                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    self.save_checkpoint('best')

        self.save_checkpoint('final')
        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.2%}")

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.flow_model.eval()
        self.easpm.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                prosody = batch['prosody'].to(self.device)
                emotion_ids = batch.get('emotion_ids')
                if emotion_ids is not None:
                    emotion_ids = emotion_ids.to(self.device)

                # Sample and check if EASPM correctly scores
                t = torch.full((prosody.shape[0],), 0.8, device=self.device)
                noise = torch.randn_like(prosody)
                x_t = self.path.sample_xt(t, noise, prosody)

                out = self.easpm(x_t, t, emotion_ids)
                pred = out['emotion_logits'].argmax(dim=-1)

                if emotion_ids is not None:
                    total_correct += (pred == emotion_ids).sum().item()
                    total_samples += emotion_ids.shape[0]

        accuracy = total_correct / max(1, total_samples)
        return {'accuracy': accuracy}

    def save_checkpoint(self, name: str, output_dir: str = 'checkpoints/easpo'):
        """Save checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'config': {
                'beta': self.config.beta,
                'js_alpha': self.config.js_alpha,
                'num_denoising_steps': self.config.num_denoising_steps,
                'num_candidates_per_step': self.config.num_candidates_per_step,
            },
            'flow_model': self.flow_model.state_dict(),
            'easpm': self.easpm.state_dict(),
            'flow_optimizer': self.flow_optimizer.state_dict(),
            'easpm_optimizer': self.easpm_optimizer.state_dict(),
        }

        torch.save(checkpoint, output_path / f'{name}.pt')
        print(f"Saved checkpoint: {output_path / f'{name}.pt'}")


# =============================================================================
# STEPWISE LOG PROBABILITY
# =============================================================================

class StepwiseLogProbComputer(nn.Module):
    """
    Computes log probability for a single denoising step.

    For flow matching:
    log P(x_{t+dt} | x_t) ≈ -||v_θ(t, x_t) - (x_{t+dt} - x_t) / dt||²
    """

    def __init__(self, config: EASPOConfig):
        super().__init__()
        self.config = config

    def compute_step_logprob(
        self,
        model: ProsodyFlow,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log probability of x_next given x_t under the flow model.

        Args:
            model: Flow model
            x_t: Current state [batch, dim]
            x_next: Next state [batch, dim]
            t: Current time [batch]
            dt: Time step
            text_cond: Text conditioning
            text_mask: Text mask

        Returns:
            Log probability [batch]
        """
        # Predict velocity
        velocity = model.vector_field(t, x_t, text_cond, text_mask)

        # Empirical velocity
        empirical_velocity = (x_next - x_t) / dt

        # Negative squared error as log probability
        sq_error = (velocity - empirical_velocity).pow(2).sum(dim=-1)

        # Log probability (higher = better match)
        log_prob = -0.5 * sq_error

        return log_prob


# =============================================================================
# EASPO ADAPTER
# =============================================================================

class EASPOAdapter(nn.Module):
    """
    Adapter for using EASPO-trained flow model.

    Provides interface compatible with existing prosody pipeline.
    """

    def __init__(
        self,
        config: EASPOConfig,
        flow_model: ProsodyFlow,
        easpm: Optional[EASPM] = None,
    ):
        super().__init__()
        self.config = config
        self.flow_model = flow_model
        self.easpm = easpm

        # Output projection
        self.token_projection = nn.Sequential(
            nn.Linear(config.base_config.prosody_dim, config.base_config.prosody_dim),
            nn.LayerNorm(config.base_config.prosody_dim),
            nn.GELU(),
            nn.Linear(
                config.base_config.prosody_dim,
                config.base_config.prosody_dim * config.num_prosody_tokens
            ),
        )
        self.token_norm = nn.LayerNorm(config.base_config.prosody_dim)

    def forward(
        self,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        use_easpm_guidance: bool = False,
        guidance_scale: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with optional EASPM guidance.

        Args:
            text_cond: Text conditioning
            text_mask: Text mask
            emotion_ids: Target emotion for guided generation
            num_samples: Number of samples
            temperature: Sampling temperature
            use_easpm_guidance: Use EASPM for guided sampling
            guidance_scale: Scale for EASPM guidance

        Returns:
            Dict with prosody_tokens and scores
        """
        if use_easpm_guidance and self.easpm is not None and emotion_ids is not None:
            # Guided sampling with EASPM
            prosody = self._guided_sample(
                text_cond, text_mask, emotion_ids,
                num_samples, temperature, guidance_scale
            )
        else:
            # Standard sampling
            prosody = self.flow_model.sample(
                text_cond, text_mask,
                num_samples=num_samples,
                temperature=temperature,
            )

        # Project to tokens
        tokens = self.token_projection(prosody)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.base_config.prosody_dim)
        tokens = self.token_norm(tokens)

        result = {'prosody_tokens': tokens}

        # Compute EASPM scores if available
        if self.easpm is not None and emotion_ids is not None:
            with torch.no_grad():
                t = torch.ones(prosody.shape[0], device=prosody.device)  # Final step
                out = self.easpm(prosody, t, emotion_ids.expand(prosody.shape[0]))
                result['easpm_scores'] = out['score']

        return result

    def _guided_sample(
        self,
        text_cond: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        emotion_ids: torch.Tensor,
        num_samples: int,
        temperature: float,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Guided sampling with EASPM feedback."""
        device = next(self.flow_model.parameters()).device
        batch_size = emotion_ids.shape[0] * num_samples

        # Initialize from noise
        x = torch.randn(batch_size, self.config.base_config.prosody_dim, device=device)
        x = x * temperature

        num_steps = self.config.num_denoising_steps
        dt = 1.0 / num_steps

        # Expand conditions
        if text_cond is not None:
            text_cond = text_cond.repeat_interleave(num_samples, dim=0)
        if text_mask is not None:
            text_mask = text_mask.repeat_interleave(num_samples, dim=0)
        emotion_ids_exp = emotion_ids.repeat_interleave(num_samples, dim=0)

        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)

            # Get base velocity
            velocity = self.flow_model.vector_field(t, x, text_cond, text_mask)

            # Compute EASPM guidance
            x_for_grad = x.detach().requires_grad_(True)
            out = self.easpm(x_for_grad, t, emotion_ids_exp)
            score = out['score'].sum()
            guidance_grad = torch.autograd.grad(score, x_for_grad)[0]

            # Apply guidance
            velocity = velocity + guidance_scale * guidance_grad

            # Step
            x = x + dt * velocity

        return x


# =============================================================================
# DATASET
# =============================================================================

class EASPODataset(Dataset):
    """Dataset for EASPO training."""

    def __init__(
        self,
        manifest_path: str,
        prosody_dim: int = 2048,
        text_dim: int = 768,
        max_text_len: int = 128,
    ):
        with open(manifest_path) as f:
            self.samples = json.load(f)

        self.prosody_dim = prosody_dim
        self.text_dim = text_dim
        self.max_text_len = max_text_len

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get prosody (or create synthetic)
        prosody = self._get_prosody(sample)

        # Get emotion
        emotion = sample.get('emotion', 'neutral').lower()
        emotion_id = EMOTION_TO_IDX.get(emotion, 0)

        # Get text embedding (placeholder - would use real encoder)
        text_cond = torch.randn(self.max_text_len, self.text_dim)
        text_mask = torch.ones(self.max_text_len, dtype=torch.bool)

        return {
            'prosody': prosody,
            'emotion_ids': torch.tensor(emotion_id, dtype=torch.long),
            'text_cond': text_cond,
            'text_mask': text_mask,
        }

    def _get_prosody(self, sample: dict) -> torch.Tensor:
        """Extract or generate prosody vector."""
        prosody_dict = sample.get('prosody', {})

        if isinstance(prosody_dict, dict):
            # Build from components
            semantic = prosody_dict.get('semantic', {})
            acoustic = prosody_dict.get('acoustic', {})

            # Simple encoding
            values = []

            # Semantic features
            if isinstance(semantic, dict):
                emotions = semantic.get('emotions', {})
                for e in ['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral']:
                    values.append(emotions.get(e, 0.0))

            # Acoustic features
            if isinstance(acoustic, dict):
                values.extend([
                    acoustic.get('pitch_mean', 0.0),
                    acoustic.get('pitch_std', 0.0),
                    acoustic.get('intensity_mean', 0.0),
                ])

            if values:
                prosody = torch.tensor(values, dtype=torch.float32)
                # Pad to prosody_dim
                if prosody.shape[0] < self.prosody_dim:
                    prosody = F.pad(prosody, (0, self.prosody_dim - prosody.shape[0]))
                return prosody[:self.prosody_dim]

        # Generate random prosody as fallback
        return torch.randn(self.prosody_dim)


def collate_easpo(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for EASPO dataset."""
    return {
        'prosody': torch.stack([item['prosody'] for item in batch]),
        'emotion_ids': torch.stack([item['emotion_ids'] for item in batch]),
        'text_cond': torch.stack([item['text_cond'] for item in batch]),
        'text_mask': torch.stack([item['text_mask'] for item in batch]),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EASPO: Stepwise Preference Optimization")
    parser.add_argument('--config', type=str, default='config/easpo.yaml')
    parser.add_argument('--manifest', type=str, help='Training manifest')
    parser.add_argument('--val_manifest', type=str, help='Validation manifest')
    parser.add_argument('--checkpoint', type=str, help='Flow model checkpoint')
    parser.add_argument('--output_dir', type=str, default='checkpoints/easpo')
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
        config = EASPOConfig(**{k: v for k, v in config_dict.items()
                                if hasattr(EASPOConfig, k)})
    else:
        config = EASPOConfig()

    print("=" * 60)
    print("EASPO: Emotion-Aware Stepwise Preference Optimization")
    print("=" * 60)

    # Create flow model
    flow_model = ProsodyFlow(config.base_config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'flow_model' in checkpoint:
            flow_model.load_state_dict(checkpoint['flow_model'])
        elif 'model' in checkpoint:
            flow_model.load_state_dict(checkpoint['model'])
        print(f"Loaded flow model from {args.checkpoint}")

    # Create dataset
    if args.manifest:
        dataset = EASPODataset(args.manifest)
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_easpo,
        )
    else:
        print("No manifest provided. Run with --manifest to train.")
        return

    # Create trainer
    trainer = EASPOTrainer(config, flow_model)

    # Train
    val_loader = None
    if args.val_manifest:
        val_dataset = EASPODataset(args.val_manifest)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_easpo,
        )

    trainer.train(train_loader, val_loader)


def run_tests():
    """Run EASPO tests."""
    print("=" * 70)
    print("EASPO: Emotion-Aware Stepwise Preference Optimization - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = EASPOConfig()
    batch_size = 2
    prosody_dim = config.base_config.prosody_dim

    # Test 1: TimeConditionedEmotionEncoder
    print("\n[Test 1] TimeConditionedEmotionEncoder...")
    emotion_encoder = TimeConditionedEmotionEncoder(config).to(device)
    t = torch.rand(batch_size, device=device)
    emotion_ids = torch.randint(0, config.num_emotions, (batch_size,), device=device)

    emotion_emb = emotion_encoder(t, emotion_ids)
    print(f"  Input: t={t.shape}, emotion_ids={emotion_ids.shape}")
    print(f"  Output: {emotion_emb.shape}")
    assert emotion_emb.shape == (batch_size, config.easpm_hidden_dim)
    print("  [PASS]")

    # Test 2: NoisyStateEncoder
    print("\n[Test 2] NoisyStateEncoder...")
    state_encoder = NoisyStateEncoder(config).to(device)
    x_t = torch.randn(batch_size, prosody_dim, device=device)

    state_emb = state_encoder(x_t, t)
    print(f"  Input: x_t={x_t.shape}, t={t.shape}")
    print(f"  Output: {state_emb.shape}")
    assert state_emb.shape == (batch_size, config.easpm_hidden_dim)
    print("  [PASS]")

    # Test 3: EASPM
    print("\n[Test 3] EASPM (Emotion-Aware Stepwise Preference Model)...")
    easpm = EASPM(config).to(device)

    out = easpm(x_t, t, emotion_ids)
    print(f"  Score: {out['score'].shape}")
    print(f"  State embedding: {out['state_emb'].shape}")
    print(f"  Emotion logits: {out['emotion_logits'].shape}")

    # Test pairwise preference
    x_t_2 = torch.randn(batch_size, prosody_dim, device=device)
    pref_out = easpm.compute_pairwise_preference(x_t, x_t_2, t, emotion_ids)
    print(f"  Preference logits: {pref_out['preference_logits'].shape}")
    print(f"  Preference prob: {pref_out['preference_prob'].tolist()}")
    print("  [PASS]")

    # Test 4: StepwisePreferenceLoss
    print("\n[Test 4] StepwisePreferenceLoss...")
    loss_fn = StepwisePreferenceLoss(config)

    policy_win = torch.randn(batch_size, device=device)
    policy_lose = torch.randn(batch_size, device=device) - 0.5
    ref_win = torch.randn(batch_size, device=device)
    ref_lose = torch.randn(batch_size, device=device)

    loss_out = loss_fn(
        policy_win, policy_lose,
        ref_win, ref_lose,
        t
    )
    print(f"  Loss: {loss_out['loss'].item():.4f}")
    print(f"  Accuracy: {loss_out['accuracy'].item():.2%}")
    print(f"  Step weights: {loss_out['step_weights'].item():.4f}")
    print("  [PASS]")

    # Test 5: StepwiseCandidateSampler
    print("\n[Test 5] StepwiseCandidateSampler...")
    flow_model = ProsodyFlow(config.base_config).to(device)

    sampler = StepwiseCandidateSampler(flow_model, easpm, config)

    candidates = sampler.sample_candidates_at_step(
        x_t, t, dt=0.02,
        target_emotion_ids=emotion_ids,
        num_candidates=4,
    )
    print(f"  Candidates: {candidates['candidates'].shape}")
    print(f"  Scores: {candidates['scores'].shape}")
    print(f"  Win samples: {candidates['win_samples'].shape}")
    print(f"  Lose samples: {candidates['lose_samples'].shape}")
    print(f"  Win scores: {candidates['win_scores'].tolist()}")
    print(f"  Lose scores: {candidates['lose_scores'].tolist()}")
    print("  [PASS]")

    # Test 6: StepwiseLogProbComputer
    print("\n[Test 6] StepwiseLogProbComputer...")
    log_prob_computer = StepwiseLogProbComputer(config)

    x_next = x_t + 0.1 * torch.randn_like(x_t)
    log_prob = log_prob_computer.compute_step_logprob(
        flow_model, x_t, x_next, t, dt=0.02
    )
    print(f"  Log prob: {log_prob.shape}")
    print(f"  Values: {log_prob.tolist()}")
    print("  [PASS]")

    # Test 7: EASPO Adapter
    print("\n[Test 7] EASPOAdapter...")
    adapter = EASPOAdapter(config, flow_model, easpm).to(device)

    text_cond = torch.randn(batch_size, 10, config.base_config.text_dim, device=device)
    text_mask = torch.ones(batch_size, 10, dtype=torch.bool, device=device)

    out = adapter(
        text_cond, text_mask,
        emotion_ids=emotion_ids,
        num_samples=1,
        temperature=1.0,
    )
    print(f"  Prosody tokens: {out['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 8: Guided sampling
    print("\n[Test 8] EASPM-guided sampling...")
    out_guided = adapter(
        text_cond, text_mask,
        emotion_ids=emotion_ids,
        use_easpm_guidance=True,
        guidance_scale=2.0,
    )
    print(f"  Guided tokens: {out_guided['prosody_tokens'].shape}")
    if 'easpm_scores' in out_guided:
        print(f"  EASPM scores: {out_guided['easpm_scores'].tolist()}")
    print("  [PASS]")

    # Test 9: Step weight schedule
    print("\n[Test 9] Step weight schedule...")
    loss_fn = StepwisePreferenceLoss(config)

    for schedule in ['uniform', 'linear', 'cosine']:
        config_temp = EASPOConfig(step_weight_schedule=schedule)
        loss_fn_temp = StepwisePreferenceLoss(config_temp)

        t_vals = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = [loss_fn_temp.compute_step_weight(torch.tensor([t])).item()
                   for t in t_vals]
        print(f"  {schedule}: {[f'{w:.3f}' for w in weights]}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All EASPO tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from easpo import EASPOConfig, EASPOTrainer, EASPOAdapter, ProsodyFlow

# Initialize
config = EASPOConfig()
flow_model = ProsodyFlow(config.base_config).cuda()

# Load pre-trained flow model (e.g., from ProsodyFlow training)
flow_model.load_state_dict(torch.load('prosody_flow.pt'))

# Create EASPO trainer
trainer = EASPOTrainer(config, flow_model)

# Train with stepwise preference optimization
trainer.train(train_loader, val_loader)

# Use trained model with adapter
adapter = EASPOAdapter(config, trainer.flow_model, trainer.easpm)

# Generate emotion-aligned prosody
out = adapter(
    text_cond,
    emotion_ids=torch.tensor([1]),  # happy
    use_easpm_guidance=True,
    guidance_scale=2.0,
)
prosody_tokens = out['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")


if __name__ == "__main__":
    main()
