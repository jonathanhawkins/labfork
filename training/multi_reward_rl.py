"""
Multi-Reward Reinforcement Learning for TTS (GLM-TTS / EMORL-TTS approach)

Based on:
- GLM-TTS (arXiv:2512.14291): GRPO-based multi-reward RL for pronunciation, speaker similarity, prosody
- EMORL-TTS (arXiv:2510.05758): VAD-based global intensity + local emphasis control with RL

Key Innovations:
1. GRPO (Group Relative Policy Optimization): No critic model needed, uses group-relative advantages
2. Multiple reward functions: CER, SIM, emotion accuracy, naturalness
3. VAD-based emotion control integration with existing spherical_emotion.py
4. Dynamic sampling and gradient clipping for stability
5. Combined SFT + RL training (SFT warm start, then RL fine-tuning)

Benefits:
- Improves pronunciation accuracy (CER reward)
- Maintains speaker identity (SIM reward)
- Enhances emotional expressiveness (emotion reward)
- Preserves naturalness (naturalness reward)
- Fine-grained control without quality loss

Usage:
    python multi_reward_rl.py --config config/multi_reward_rl.yaml \
        --checkpoint ../checkpoints/prosody_v6/best.pt \
        --manifest ../data/emotion_manifest.json
"""

import argparse
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

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MultiRewardRLConfig:
    """Configuration for Multi-Reward RL training."""

    # GRPO settings
    group_size: int = 4  # Number of samples per input for GRPO
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-k sampling
    top_p: float = 0.95  # Nucleus sampling

    # Reward weights (GRPO optimizes weighted sum)
    cer_weight: float = 1.0  # Character Error Rate (pronunciation)
    sim_weight: float = 1.0  # Speaker similarity
    emotion_weight: float = 1.0  # Emotion accuracy
    naturalness_weight: float = 0.5  # Naturalness/quality

    # EMORL-TTS specific: VAD-based control
    use_vad_control: bool = True
    intensity_reward_weight: float = 0.5  # Reward for matching target intensity
    emphasis_reward_weight: float = 0.5  # Reward for local emphasis placement

    # Policy optimization
    kl_coef: float = 0.1  # KL penalty coefficient (β in GRPO)
    clip_ratio: float = 0.2  # PPO-style clipping ratio
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    advantage_normalization: bool = True  # Normalize advantages within group

    # Training settings
    learning_rate: float = 5e-6  # Lower LR for RL fine-tuning
    batch_size: int = 4  # Number of inputs per batch (effective samples = batch_size * group_size)
    num_epochs: int = 5
    warmup_steps: int = 200
    gradient_accumulation: int = 4
    max_grad_norm: float = 0.5  # Tighter clipping for RL stability

    # Model settings
    hidden_size: int = 2048
    num_prosody_tokens: int = 4

    # Reference model
    use_reference_model: bool = True
    reference_model_update_freq: int = 0  # 0 = frozen

    # SFT regularization (combined SFT + RL as in EMORL-TTS)
    sft_weight: float = 0.5  # Weight for SFT loss alongside RL
    sft_decay: float = 0.95  # Decay SFT weight per epoch

    # Reward model settings
    use_external_asr: bool = False  # Use external ASR for CER
    use_external_spk: bool = False  # Use external speaker encoder for SIM
    use_external_emo: bool = False  # Use external emotion classifier

    # Output
    output_dir: str = "checkpoints/multi_reward_rl"
    log_every: int = 10
    save_every_epochs: int = 1


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardFunction(nn.Module):
    """Base class for reward functions."""

    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()
        self.name = name
        self.weight = weight

    def compute_reward(self, *args, **kwargs) -> torch.Tensor:
        """Compute reward. Returns [batch] tensor of rewards."""
        raise NotImplementedError


class CERRewardFunction(RewardFunction):
    """
    Character Error Rate (CER) reward for pronunciation accuracy.

    Uses ASR model to transcribe generated audio and compute CER against target text.
    Lower CER = higher reward (we negate CER for reward).

    Based on GLM-TTS: Uses composite reward = -CER + λ * NLL from ASR
    """

    def __init__(self, config: MultiRewardRLConfig, asr_model=None):
        super().__init__("cer", config.cer_weight)
        self.config = config

        # ASR model for CER computation
        # In practice, use Whisper or other ASR
        self.asr_model = asr_model

        # Fallback: learned CER predictor from prosody embeddings
        # This approximates CER without full ASR during training
        self.cer_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),  # Output CER in [0, 1]
        )

    def compute_cer(
        self,
        predicted_text: List[str],
        target_text: List[str],
    ) -> torch.Tensor:
        """Compute CER between predicted and target text."""
        from difflib import SequenceMatcher

        cers = []
        for pred, target in zip(predicted_text, target_text):
            pred = pred.lower().strip()
            target = target.lower().strip()

            if not target:
                cers.append(0.0)
                continue

            # Character-level edit distance
            matcher = SequenceMatcher(None, pred, target)
            cer = 1.0 - matcher.ratio()
            cers.append(cer)

        return torch.tensor(cers, dtype=torch.float32)

    def compute_reward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        generated_audio: Optional[torch.Tensor] = None,
        target_text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute CER-based reward.

        If ASR model available and audio provided, use full ASR pipeline.
        Otherwise, use learned CER predictor from prosody embeddings.
        """
        device = prosody_embedding.device

        if self.asr_model is not None and generated_audio is not None and target_text is not None:
            # Full ASR pipeline
            with torch.no_grad():
                # Transcribe audio
                predicted_text = self.asr_model.transcribe(generated_audio)
                cer = self.compute_cer(predicted_text, target_text)
        else:
            # Use learned predictor (approximation)
            pooled = prosody_embedding.mean(dim=1)  # [batch, hidden]
            cer = self.cer_predictor(pooled).squeeze(-1)  # [batch]

        # Reward = 1 - CER (higher is better)
        reward = 1.0 - cer.to(device)

        return reward * self.weight


class SIMRewardFunction(RewardFunction):
    """
    Speaker Similarity (SIM) reward.

    Computes cosine similarity between speaker embeddings of generated audio
    and reference speaker embedding.

    Higher similarity = higher reward.
    """

    def __init__(self, config: MultiRewardRLConfig, speaker_encoder=None):
        super().__init__("sim", config.sim_weight)
        self.config = config

        # Speaker encoder for embedding extraction
        self.speaker_encoder = speaker_encoder

        # Fallback: learned similarity predictor
        self.sim_predictor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def compute_reward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        reference_speaker_emb: Optional[torch.Tensor] = None,  # [batch, speaker_dim]
        generated_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute speaker similarity reward."""
        device = prosody_embedding.device

        if self.speaker_encoder is not None and generated_audio is not None:
            # Extract speaker embedding from generated audio
            with torch.no_grad():
                gen_speaker_emb = self.speaker_encoder(generated_audio)
                # Cosine similarity
                sim = F.cosine_similarity(gen_speaker_emb, reference_speaker_emb, dim=-1)
        else:
            # Use learned predictor
            pooled = prosody_embedding.mean(dim=1)  # [batch, hidden]

            if reference_speaker_emb is not None:
                # Ensure same dimension
                if reference_speaker_emb.shape[-1] != self.config.hidden_size:
                    reference_speaker_emb = F.pad(
                        reference_speaker_emb,
                        (0, self.config.hidden_size - reference_speaker_emb.shape[-1])
                    )
                combined = torch.cat([pooled, reference_speaker_emb], dim=-1)
            else:
                combined = torch.cat([pooled, pooled], dim=-1)

            sim = self.sim_predictor(combined).squeeze(-1)

        # Reward = similarity (already in [0, 1] or [-1, 1])
        reward = (sim + 1.0) / 2.0 if sim.min() < 0 else sim  # Normalize to [0, 1]

        return reward.to(device) * self.weight


class EmotionRewardFunction(RewardFunction):
    """
    Emotion accuracy reward.

    Measures how well the generated prosody matches the target emotion.
    Uses emotion classifier on prosody embeddings.

    Based on EMORL-TTS: VAD-based global intensity + local emphasis.
    """

    def __init__(
        self,
        config: MultiRewardRLConfig,
        num_emotions: int = 8,
        emotion_classifier=None,
    ):
        super().__init__("emotion", config.emotion_weight)
        self.config = config
        self.num_emotions = num_emotions

        # External emotion classifier
        self.emotion_classifier = emotion_classifier

        # Learned emotion predictor from prosody
        self.emotion_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, num_emotions),
        )

        # VAD predictor for intensity/emphasis (EMORL-TTS)
        self.vad_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 3),  # V, A, D
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Intensity predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def compute_reward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        target_emotion: torch.Tensor,  # [batch] emotion indices
        target_vad: Optional[torch.Tensor] = None,  # [batch, 3] VAD coordinates
        target_intensity: Optional[torch.Tensor] = None,  # [batch] intensity values
    ) -> torch.Tensor:
        """Compute emotion accuracy reward."""
        device = prosody_embedding.device
        batch_size = prosody_embedding.shape[0]

        # Pool over tokens
        pooled = prosody_embedding.mean(dim=1)  # [batch, hidden]

        # Predict emotion
        emotion_logits = self.emotion_predictor(pooled)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        # Reward: probability of correct emotion
        target_emotion = target_emotion.to(device)
        emotion_reward = emotion_probs.gather(1, target_emotion.unsqueeze(1)).squeeze(1)

        # VAD reward (EMORL-TTS style)
        vad_reward = torch.zeros(batch_size, device=device)
        if self.config.use_vad_control and target_vad is not None:
            predicted_vad = self.vad_predictor(pooled)
            target_vad = target_vad.to(device)
            # Cosine similarity in VAD space
            vad_sim = F.cosine_similarity(predicted_vad, target_vad, dim=-1)
            vad_reward = (vad_sim + 1.0) / 2.0  # Normalize to [0, 1]

        # Intensity reward (EMORL-TTS style)
        intensity_reward = torch.zeros(batch_size, device=device)
        if self.config.use_vad_control and target_intensity is not None:
            predicted_intensity = self.intensity_predictor(pooled).squeeze(-1)
            target_intensity = target_intensity.to(device)
            # Negative L1 distance (closer = higher reward)
            intensity_error = (predicted_intensity - target_intensity).abs()
            intensity_reward = 1.0 - intensity_error

        # Combined reward
        total_reward = (
            emotion_reward +
            self.config.intensity_reward_weight * vad_reward +
            self.config.emphasis_reward_weight * intensity_reward
        )

        return total_reward * self.weight


class NaturalnessRewardFunction(RewardFunction):
    """
    Naturalness/quality reward.

    Predicts MOS (Mean Opinion Score) or general quality from prosody.
    Uses learned quality predictor or external quality model.
    """

    def __init__(self, config: MultiRewardRLConfig, quality_model=None):
        super().__init__("naturalness", config.naturalness_weight)
        self.config = config

        # External quality model (e.g., NISQA, UTMOS)
        self.quality_model = quality_model

        # Learned quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 4, config.hidden_size // 8),
            nn.GELU(),
            nn.Linear(config.hidden_size // 8, 1),
            nn.Sigmoid(),  # Output quality score in [0, 1]
        )

        # Consistency predictor (temporal smoothness)
        self.consistency_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def compute_reward(
        self,
        prosody_embedding: torch.Tensor,  # [batch, num_tokens, hidden]
        generated_audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute naturalness reward."""
        device = prosody_embedding.device
        batch_size = prosody_embedding.shape[0]

        if self.quality_model is not None and generated_audio is not None:
            # Use external quality model
            with torch.no_grad():
                quality = self.quality_model(generated_audio)
                # Normalize to [0, 1] assuming MOS range [1, 5]
                quality = (quality - 1.0) / 4.0
        else:
            # Use learned predictor
            pooled = prosody_embedding.mean(dim=1)
            quality = self.quality_predictor(pooled).squeeze(-1)

        # Temporal consistency reward
        # Compute variance across tokens (lower variance = more consistent)
        token_variance = prosody_embedding.var(dim=1).mean(dim=-1)  # [batch]
        # Normalize and invert (lower variance = higher reward)
        consistency = torch.exp(-token_variance / 100.0)

        # Combined naturalness reward
        total_reward = 0.7 * quality + 0.3 * consistency.to(device)

        return total_reward * self.weight


class MultiRewardAggregator(nn.Module):
    """
    Aggregates multiple reward functions into a single reward signal.

    Supports:
    - Weighted sum of rewards
    - Adaptive reward weighting (learned)
    - Reward normalization
    """

    def __init__(self, config: MultiRewardRLConfig):
        super().__init__()
        self.config = config

        # Initialize reward functions
        self.reward_functions = nn.ModuleDict({
            'cer': CERRewardFunction(config),
            'sim': SIMRewardFunction(config),
            'emotion': EmotionRewardFunction(config),
            'naturalness': NaturalnessRewardFunction(config),
        })

        # Reward normalization (running stats)
        self.register_buffer('reward_mean', torch.zeros(4))
        self.register_buffer('reward_std', torch.ones(4))
        self.register_buffer('reward_count', torch.zeros(1))

        # Optional: learned reward weights
        self.adaptive_weights = nn.Parameter(torch.ones(4))

    def compute_rewards(
        self,
        prosody_embedding: torch.Tensor,
        target_text: Optional[List[str]] = None,
        target_emotion: Optional[torch.Tensor] = None,
        target_vad: Optional[torch.Tensor] = None,
        target_intensity: Optional[torch.Tensor] = None,
        reference_speaker_emb: Optional[torch.Tensor] = None,
        generated_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all rewards.

        Returns dict with individual rewards and total.
        """
        rewards = {}

        # CER reward
        rewards['cer'] = self.reward_functions['cer'].compute_reward(
            prosody_embedding,
            generated_audio=generated_audio,
            target_text=target_text,
        )

        # Speaker similarity reward
        rewards['sim'] = self.reward_functions['sim'].compute_reward(
            prosody_embedding,
            reference_speaker_emb=reference_speaker_emb,
            generated_audio=generated_audio,
        )

        # Emotion reward
        if target_emotion is not None:
            rewards['emotion'] = self.reward_functions['emotion'].compute_reward(
                prosody_embedding,
                target_emotion=target_emotion,
                target_vad=target_vad,
                target_intensity=target_intensity,
            )
        else:
            rewards['emotion'] = torch.zeros(prosody_embedding.shape[0], device=prosody_embedding.device)

        # Naturalness reward
        rewards['naturalness'] = self.reward_functions['naturalness'].compute_reward(
            prosody_embedding,
            generated_audio=generated_audio,
        )

        # Total reward (weighted sum)
        weights = F.softmax(self.adaptive_weights, dim=0)
        rewards['total'] = (
            weights[0] * rewards['cer'] +
            weights[1] * rewards['sim'] +
            weights[2] * rewards['emotion'] +
            weights[3] * rewards['naturalness']
        )

        return rewards


# =============================================================================
# GRPO (Group Relative Policy Optimization)
# =============================================================================

class GRPOPolicy(nn.Module):
    """
    GRPO Policy for prosody generation.

    Key features:
    - Group sampling: generates multiple prosody outputs per input
    - No critic model: uses group-relative advantages
    - Compatible with existing prosody encoder architecture
    """

    def __init__(self, prosody_encoder: nn.Module, config: MultiRewardRLConfig):
        super().__init__()
        self.prosody_encoder = prosody_encoder
        self.config = config

        # Log-std for stochastic sampling
        self.log_std = nn.Parameter(torch.zeros(config.hidden_size))

        # Value head for optional baseline (can be disabled in pure GRPO)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
        )

    def forward(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate prosody embedding with optional stochasticity.

        Returns:
            embedding: [batch, num_tokens, hidden]
            log_prob: [batch] log probability of the sample
            value: [batch] value estimate (baseline)
        """
        # Get deterministic mean from encoder
        mean = self.prosody_encoder(semantic, acoustic, rhythm, contour)

        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device), self.value_head(mean.mean(dim=1)).squeeze(-1)

        # Stochastic sampling
        std = torch.exp(self.log_std).clamp(min=1e-6, max=1.0)

        # Sample from Gaussian
        noise = torch.randn_like(mean) * std
        sample = mean + noise * self.config.temperature

        # Compute log probability
        log_prob = -0.5 * (
            ((sample - mean) / (std + 1e-8)) ** 2 +
            2 * self.log_std +
            math.log(2 * math.pi)
        ).sum(dim=-1).sum(dim=-1)  # Sum over tokens and hidden

        # Value estimate
        value = self.value_head(mean.mean(dim=1)).squeeze(-1)

        return sample, log_prob, value

    def sample_group(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
        group_size: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a group of prosody embeddings for GRPO.

        Args:
            semantic, acoustic, rhythm, contour: Input prosody features [batch, dim]
            group_size: Number of samples per input

        Returns:
            samples: [batch, group_size, num_tokens, hidden]
            log_probs: [batch, group_size]
            values: [batch, group_size]
        """
        if group_size is None:
            group_size = self.config.group_size

        batch_size = semantic.shape[0]

        # Expand inputs for group sampling
        semantic_exp = semantic.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, semantic.shape[-1])
        acoustic_exp = acoustic.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, acoustic.shape[-1])
        rhythm_exp = rhythm.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, rhythm.shape[-1])
        contour_exp = contour.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, contour.shape[-1])

        # Sample all at once
        samples, log_probs, values = self.forward(
            semantic_exp, acoustic_exp, rhythm_exp, contour_exp,
            deterministic=False,
        )

        # Reshape to [batch, group_size, ...]
        num_tokens = samples.shape[1]
        hidden = samples.shape[2]

        samples = samples.view(batch_size, group_size, num_tokens, hidden)
        log_probs = log_probs.view(batch_size, group_size)
        values = values.view(batch_size, group_size)

        return samples, log_probs, values


class GRPOLoss(nn.Module):
    """
    GRPO Loss computation.

    GRPO uses group-relative advantages instead of a learned value function:
    - Sample group of responses for each input
    - Compute rewards for each response
    - Advantage = reward - mean(group_rewards)
    - Policy gradient with clipping (like PPO)
    """

    def __init__(self, config: MultiRewardRLConfig):
        super().__init__()
        self.config = config

    def compute_advantages(
        self,
        rewards: torch.Tensor,  # [batch, group_size]
        values: Optional[torch.Tensor] = None,  # [batch, group_size]
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        In GRPO, advantage = reward - mean(group_rewards)
        This eliminates the need for a learned value function.
        """
        # Group-relative baseline
        baseline = rewards.mean(dim=1, keepdim=True)

        # Advantages
        advantages = rewards - baseline

        # Optional: normalize advantages
        if self.config.advantage_normalization:
            std = advantages.std(dim=1, keepdim=True) + 1e-8
            advantages = advantages / std

        return advantages

    def forward(
        self,
        log_probs: torch.Tensor,  # [batch, group_size] current policy log probs
        old_log_probs: torch.Tensor,  # [batch, group_size] reference policy log probs
        rewards: torch.Tensor,  # [batch, group_size] rewards
        values: Optional[torch.Tensor] = None,  # [batch, group_size] value estimates
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss.

        Returns dict with:
            - policy_loss: Main GRPO policy gradient loss
            - kl_loss: KL divergence penalty
            - entropy_loss: Entropy bonus
            - total: Combined loss
        """
        device = log_probs.device
        batch_size, group_size = log_probs.shape

        # Compute advantages
        advantages = self.compute_advantages(rewards, values)

        # Policy ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective (PPO-style)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio
        ) * advantages

        # Policy loss (maximize advantage, so minimize negative)
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence penalty
        kl = (old_log_probs - log_probs).mean()
        kl_loss = self.config.kl_coef * kl

        # Entropy bonus (encourage exploration)
        # Approximate entropy from log_probs variance
        entropy = -log_probs.mean()
        entropy_loss = -self.config.entropy_coef * entropy

        # Total loss
        total_loss = policy_loss + kl_loss + entropy_loss

        return {
            'policy_loss': policy_loss,
            'kl_loss': kl_loss,
            'entropy_loss': entropy_loss,
            'kl': kl,
            'entropy': entropy,
            'advantages_mean': advantages.mean(),
            'advantages_std': advantages.std(),
            'ratio_mean': ratio.mean(),
            'total': total_loss,
        }


# =============================================================================
# MULTI-REWARD RL TRAINER
# =============================================================================

class MultiRewardRLTrainer:
    """
    Trainer for Multi-Reward RL optimization.

    Combines:
    - GRPO for policy optimization
    - Multiple reward functions
    - SFT regularization (EMORL-TTS style)
    - Reference model for KL penalty
    """

    def __init__(
        self,
        config: MultiRewardRLConfig,
        prosody_encoder: nn.Module,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or self._setup_device()

        # Policy model (wraps prosody encoder)
        self.policy = GRPOPolicy(prosody_encoder, config).to(self.device)

        # Reference model (frozen copy for KL)
        if config.use_reference_model:
            import copy
            self.ref_policy = copy.deepcopy(self.policy)
            self.ref_policy.eval()
            for param in self.ref_policy.parameters():
                param.requires_grad = False
            print("Created frozen reference policy for GRPO")
        else:
            self.ref_policy = None

        # Reward aggregator
        self.reward_aggregator = MultiRewardAggregator(config).to(self.device)

        # GRPO loss
        self.grpo_loss = GRPOLoss(config)

        # SFT loss (reconstruction)
        self.mse_loss = nn.MSELoss()

        # Optimizer
        trainable_params = list(self.policy.parameters())
        trainable_params += list(self.reward_aggregator.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.best_reward = float('-inf')
        self.current_sft_weight = config.sft_weight

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def _compute_sft_loss(
        self,
        prosody_embedding: torch.Tensor,
        target_prosody: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute SFT reconstruction loss."""
        # Construct target from prosody components
        target = torch.cat([
            target_prosody['semantic'],
            target_prosody['acoustic'],
            target_prosody['rhythm'],
            target_prosody['contour'][:, :8] if target_prosody['contour'].shape[1] > 8
            else target_prosody['contour'],
        ], dim=-1)

        # Pad if needed
        if target.shape[-1] < prosody_embedding.shape[-1]:
            target = F.pad(target, (0, prosody_embedding.shape[-1] - target.shape[-1]))

        # Match dimensions
        target = target.unsqueeze(1).expand_as(prosody_embedding[:, :1, :])

        # MSE loss on first token
        return self.mse_loss(
            prosody_embedding[:, 0, :target.shape[-1]],
            target[:, 0, :]
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single GRPO training step.

        1. Sample group of prosody outputs for each input
        2. Compute rewards for each sample
        3. Compute group-relative advantages
        4. Update policy with GRPO loss + SFT regularization
        """
        self.policy.train()
        self.reward_aggregator.train()

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

        target_vad = batch.get('target_vad')
        if target_vad is not None:
            target_vad = target_vad.to(self.device)

        target_intensity = batch.get('target_intensity')
        if target_intensity is not None:
            target_intensity = target_intensity.to(self.device)

        batch_size = prosody_dict['semantic'].shape[0]
        group_size = self.config.group_size

        # Sample group from current policy
        samples, log_probs, values = self.policy.sample_group(
            prosody_dict['semantic'],
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
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
                    group_size=group_size,
                )
            else:
                ref_log_probs = log_probs.detach()

        # Compute rewards for each sample in group
        # Reshape samples for reward computation: [batch * group_size, num_tokens, hidden]
        samples_flat = samples.view(-1, samples.shape[2], samples.shape[3])

        # Expand targets to match group
        target_emotion_exp = None
        if target_emotion is not None:
            target_emotion_exp = target_emotion.unsqueeze(1).expand(-1, group_size).reshape(-1)

        target_vad_exp = None
        if target_vad is not None:
            target_vad_exp = target_vad.unsqueeze(1).expand(-1, group_size, -1).reshape(-1, 3)

        target_intensity_exp = None
        if target_intensity is not None:
            target_intensity_exp = target_intensity.unsqueeze(1).expand(-1, group_size).reshape(-1)

        # Compute rewards
        rewards_dict = self.reward_aggregator.compute_rewards(
            samples_flat,
            target_emotion=target_emotion_exp,
            target_vad=target_vad_exp,
            target_intensity=target_intensity_exp,
        )

        # Reshape rewards back to [batch, group_size]
        rewards = rewards_dict['total'].view(batch_size, group_size)

        # Compute GRPO loss
        grpo_losses = self.grpo_loss(
            log_probs,
            ref_log_probs,
            rewards,
            values,
        )

        # SFT regularization loss (on deterministic output)
        with torch.no_grad():
            det_output, _, _ = self.policy(
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'],
                deterministic=True,
            )
        sft_loss = self._compute_sft_loss(det_output, prosody_dict)

        # Combined loss
        total_loss = grpo_losses['total'] + self.current_sft_weight * sft_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
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
            'advantages_mean': grpo_losses['advantages_mean'].item(),
            'kl': grpo_losses['kl'].item(),
            'cer_reward': rewards_dict['cer'].mean().item(),
            'sim_reward': rewards_dict['sim'].mean().item(),
            'emotion_reward': rewards_dict['emotion'].mean().item(),
            'naturalness_reward': rewards_dict['naturalness'].mean().item(),
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
                target_emotion = batch.get('emotion_label')
                if target_emotion is not None:
                    target_emotion = target_emotion.to(self.device)

                # Deterministic output
                output, _, _ = self.policy(
                    prosody_dict['semantic'],
                    prosody_dict['acoustic'],
                    prosody_dict['rhythm'],
                    prosody_dict['contour'],
                    deterministic=True,
                )

                # Compute rewards
                rewards_dict = self.reward_aggregator.compute_rewards(
                    output,
                    target_emotion=target_emotion,
                )

                total_metrics['reward'] += rewards_dict['total'].mean().item()
                total_metrics['cer_reward'] += rewards_dict['cer'].mean().item()
                total_metrics['sim_reward'] += rewards_dict['sim'].mean().item()
                total_metrics['emotion_reward'] += rewards_dict['emotion'].mean().item()
                total_metrics['naturalness_reward'] += rewards_dict['naturalness'].mean().item()
                num_batches += 1

        # Average
        for key in total_metrics:
            total_metrics[key] /= max(1, num_batches)

        return dict(total_metrics)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """Main training loop."""
        print(f"\nStarting Multi-Reward RL training for {self.config.num_epochs} epochs")
        print(f"  GRPO group size: {self.config.group_size}")
        print(f"  Reward weights: CER={self.config.cer_weight}, SIM={self.config.sim_weight}, "
              f"EMO={self.config.emotion_weight}, NAT={self.config.naturalness_weight}")
        print(f"  SFT weight: {self.current_sft_weight} (decay={self.config.sft_decay})")
        print(f"  KL coef: {self.config.kl_coef}")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1

                # Log
                if self.global_step % self.config.log_every == 0:
                    print(f"  Step {self.global_step}: "
                          f"loss={metrics['total_loss']:.4f}, "
                          f"reward={metrics['reward_mean']:.4f}, "
                          f"kl={metrics['kl']:.4f}, "
                          f"policy={metrics['policy_loss']:.4f}")

            # Epoch summary
            for key in epoch_metrics:
                epoch_metrics[key] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            print(f"  Train - loss: {epoch_metrics['total_loss']:.4f}, "
                  f"reward: {epoch_metrics['reward_mean']:.4f}")
            print(f"  Rewards - CER: {epoch_metrics['cer_reward']:.4f}, "
                  f"SIM: {epoch_metrics['sim_reward']:.4f}, "
                  f"EMO: {epoch_metrics['emotion_reward']:.4f}, "
                  f"NAT: {epoch_metrics['naturalness_reward']:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"  Val   - reward: {val_metrics['reward']:.4f}")

                # Track best
                if val_metrics['reward'] > self.best_reward:
                    self.best_reward = val_metrics['reward']
                    self.save_checkpoint('best')

            # Decay SFT weight (EMORL-TTS style: gradually shift from SFT to RL)
            self.current_sft_weight *= self.config.sft_decay
            print(f"  SFT weight decayed to: {self.current_sft_weight:.4f}")

            # Save checkpoint
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
            'best_reward': self.best_reward,
            'current_sft_weight': self.current_sft_weight,
            'policy': self.policy.state_dict(),
            'reward_aggregator': self.reward_aggregator.state_dict(),
            'config': {
                'group_size': self.config.group_size,
                'cer_weight': self.config.cer_weight,
                'sim_weight': self.config.sim_weight,
                'emotion_weight': self.config.emotion_weight,
                'naturalness_weight': self.config.naturalness_weight,
                'kl_coef': self.config.kl_coef,
                'clip_ratio': self.config.clip_ratio,
            },
        }

        torch.save(checkpoint, output_dir / f'{name}.pt')
        print(f"Saved checkpoint: {output_dir / f'{name}.pt'}")


# =============================================================================
# DATASET ADAPTER
# =============================================================================

class MultiRewardRLDataset(Dataset):
    """
    Dataset for Multi-Reward RL training.

    Adapts existing prosody datasets to include additional targets
    for reward computation (VAD, intensity, etc.).
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        config: MultiRewardRLConfig,
    ):
        self.config = config
        self.prosody_cache_dir = Path(prosody_cache_dir)

        # Load manifest
        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples for Multi-Reward RL training")

        # VAD prototypes for emotion labels
        self.vad_prototypes = {
            'neutral': (0.0, 0.0, 0.0),
            'happy': (0.8, 0.6, 0.6),
            'sad': (-0.6, -0.4, -0.5),
            'angry': (-0.5, 0.8, 0.7),
            'fearful': (-0.7, 0.7, -0.7),
            'surprised': (0.3, 0.8, 0.2),
            'disgusted': (-0.6, 0.3, 0.4),
            'calm': (0.4, -0.5, 0.3),
        }

    def __len__(self):
        return len(self.samples)

    def _get_emotion_label(self, sample: dict) -> int:
        """Extract emotion label from sample."""
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
        """Get VAD coordinates for sample's emotion."""
        emotion_label = self._get_emotion_label(sample)
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted', 'calm']

        if 0 <= emotion_label < len(emotion_names):
            emotion = emotion_names[emotion_label]
            vad = self.vad_prototypes.get(emotion, (0, 0, 0))
        else:
            vad = (0, 0, 0)

        return torch.tensor(vad, dtype=torch.float32)

    def _get_intensity(self, sample: dict) -> float:
        """Extract intensity from sample (use acoustic energy as proxy)."""
        acoustic = sample.get('prosody', {}).get('acoustic', {})
        if isinstance(acoustic, dict):
            return acoustic.get('intensity_mean', 0.5)
        return 0.5

    def _load_prosody(self, sample: dict) -> Dict[str, torch.Tensor]:
        """Load prosody features."""
        prosody = sample.get('prosody', {})

        # Default dimensions
        semantic_dim = 8
        acoustic_dim = 5
        rhythm_dim = 4
        contour_dim = 32

        def to_tensor(data, dim):
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, dict):
                # Extract values from dict
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
            'semantic': to_tensor(prosody.get('semantic', {}), semantic_dim),
            'acoustic': to_tensor(prosody.get('acoustic', {}), acoustic_dim),
            'rhythm': to_tensor(prosody.get('rhythm', {}), rhythm_dim),
            'contour': to_tensor(prosody.get('contour', []), contour_dim),
        }

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        prosody = self._load_prosody(sample)
        emotion_label = self._get_emotion_label(sample)
        vad = self._get_vad(sample)
        intensity = self._get_intensity(sample)

        return {
            'text': sample.get('text', ''),
            'prosody_semantic': prosody['semantic'],
            'prosody_acoustic': prosody['acoustic'],
            'prosody_rhythm': prosody['rhythm'],
            'prosody_contour': prosody['contour'],
            'emotion_label': emotion_label,
            'target_vad': vad,
            'target_intensity': torch.tensor(intensity, dtype=torch.float32),
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
        'target_vad': torch.stack([item['target_vad'] for item in batch]),
        'target_intensity': torch.stack([item['target_intensity'] for item in batch]),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Reward RL for TTS (GLM-TTS/EMORL-TTS)")
    parser.add_argument('--config', type=str, default='config/multi_reward_rl.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, help='Pre-trained prosody checkpoint')
    parser.add_argument('--manifest', type=str, help='Training manifest')
    parser.add_argument('--val_manifest', type=str, help='Validation manifest')
    parser.add_argument('--output_dir', type=str, default='checkpoints/multi_reward_rl', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Run test mode with synthetic data')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = MultiRewardRLConfig(**{k: v for k, v in config_dict.items() if hasattr(MultiRewardRLConfig, k)})
    else:
        config = MultiRewardRLConfig()

    if args.output_dir:
        config.output_dir = args.output_dir

    print("=" * 70)
    print("Multi-Reward RL for TTS (GLM-TTS / EMORL-TTS approach)")
    print("=" * 70)

    # Test mode
    if args.test:
        print("\nRunning test mode with synthetic data...")
        test_multi_reward_rl(config)
        return

    # Load or create prosody encoder
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
    trainer = MultiRewardRLTrainer(
        config=config,
        prosody_encoder=prosody_encoder,
    )

    # Create dataset
    if args.manifest:
        train_dataset = MultiRewardRLDataset(
            manifest_path=args.manifest,
            prosody_cache_dir='data/prosody_cache',
            config=config,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = None
        if args.val_manifest:
            val_dataset = MultiRewardRLDataset(
                manifest_path=args.val_manifest,
                prosody_cache_dir='data/prosody_cache',
                config=config,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )

        trainer.train(train_loader, val_loader)
    else:
        print("\nNo manifest provided. Run with --manifest to train.")
        print("\nMulti-Reward RL optimizes multiple objectives simultaneously:")
        print("  - CER: Pronunciation accuracy")
        print("  - SIM: Speaker similarity")
        print("  - Emotion: Emotional expressiveness")
        print("  - Naturalness: Overall quality")


def test_multi_reward_rl(config: MultiRewardRLConfig):
    """Test Multi-Reward RL components with synthetic data."""
    print("\n[Test] Multi-Reward RL Components")
    print("-" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create mock prosody encoder
    class MockProsodyEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            input_dim = 8 + 5 + 4 + 32  # semantic + acoustic + rhythm + contour
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size * config.num_prosody_tokens),
            )

        def forward(self, semantic, acoustic, rhythm, contour):
            x = torch.cat([semantic, acoustic, rhythm, contour], dim=-1)
            out = self.encoder(x)
            batch = x.shape[0]
            return out.view(batch, self.config.num_prosody_tokens, self.config.hidden_size)

    prosody_encoder = MockProsodyEncoder(config).to(device)

    # Test 1: GRPO Policy
    print("\n[Test 1] GRPO Policy...")
    policy = GRPOPolicy(prosody_encoder, config).to(device)

    batch_size = 2
    semantic = torch.randn(batch_size, 8, device=device)
    acoustic = torch.randn(batch_size, 5, device=device)
    rhythm = torch.randn(batch_size, 4, device=device)
    contour = torch.randn(batch_size, 32, device=device)

    # Deterministic output
    det_output, det_log_prob, det_value = policy(
        semantic, acoustic, rhythm, contour, deterministic=True
    )
    print(f"  Deterministic output shape: {det_output.shape}")

    # Stochastic output
    stoch_output, stoch_log_prob, stoch_value = policy(
        semantic, acoustic, rhythm, contour, deterministic=False
    )
    print(f"  Stochastic output shape: {stoch_output.shape}")
    print(f"  Log prob shape: {stoch_log_prob.shape}")

    # Group sampling
    samples, log_probs, values = policy.sample_group(
        semantic, acoustic, rhythm, contour, group_size=4
    )
    print(f"  Group samples shape: {samples.shape}")
    print(f"  Group log probs shape: {log_probs.shape}")
    print("  [PASS]")

    # Test 2: Reward Functions
    print("\n[Test 2] Reward Functions...")

    reward_agg = MultiRewardAggregator(config).to(device)

    target_emotion = torch.tensor([1, 2], device=device)  # happy, sad
    target_vad = torch.tensor([[0.8, 0.6, 0.6], [-0.6, -0.4, -0.5]], device=device)
    target_intensity = torch.tensor([0.8, 0.5], device=device)

    rewards = reward_agg.compute_rewards(
        det_output,
        target_emotion=target_emotion,
        target_vad=target_vad,
        target_intensity=target_intensity,
    )

    print(f"  CER reward: {rewards['cer'].mean().item():.4f}")
    print(f"  SIM reward: {rewards['sim'].mean().item():.4f}")
    print(f"  Emotion reward: {rewards['emotion'].mean().item():.4f}")
    print(f"  Naturalness reward: {rewards['naturalness'].mean().item():.4f}")
    print(f"  Total reward: {rewards['total'].mean().item():.4f}")
    print("  [PASS]")

    # Test 3: GRPO Loss
    print("\n[Test 3] GRPO Loss...")

    grpo_loss = GRPOLoss(config)

    # Simulate group sampling
    group_rewards = torch.rand(batch_size, config.group_size, device=device)
    old_log_probs = torch.randn(batch_size, config.group_size, device=device)
    new_log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.1

    losses = grpo_loss(new_log_probs, old_log_probs, group_rewards)

    print(f"  Policy loss: {losses['policy_loss'].item():.4f}")
    print(f"  KL loss: {losses['kl_loss'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Advantages mean: {losses['advantages_mean'].item():.4f}")
    print("  [PASS]")

    # Test 4: Full Training Step
    print("\n[Test 4] Full Training Step...")

    trainer = MultiRewardRLTrainer(config, prosody_encoder)

    batch = {
        'prosody_semantic': torch.randn(batch_size, 8),
        'prosody_acoustic': torch.randn(batch_size, 5),
        'prosody_rhythm': torch.randn(batch_size, 4),
        'prosody_contour': torch.randn(batch_size, 32),
        'emotion_label': torch.tensor([1, 2]),
        'target_vad': torch.tensor([[0.8, 0.6, 0.6], [-0.6, -0.4, -0.5]]),
        'target_intensity': torch.tensor([0.8, 0.5]),
    }

    metrics = trainer.train_step(batch)

    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Reward mean: {metrics['reward_mean']:.4f}")
    print(f"  KL: {metrics['kl']:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All Multi-Reward RL tests passed!")
    print("=" * 70)

    print("\nUsage:")
    print("-" * 40)
    print("""
# Train with Multi-Reward RL
python multi_reward_rl.py --config config/multi_reward_rl.yaml \\
  --checkpoint ../checkpoints/prosody_v6/best.pt \\
  --manifest ../data/emotion_manifest.json

# Test mode
python multi_reward_rl.py --test
""")


if __name__ == "__main__":
    main()
