"""
Emo-DPO: Direct Preference Optimization for Emotional Speech Synthesis

Based on "Emo-DPO: Controllable Emotional Speech Synthesis through Direct Preference
Optimization" (ICASSP 2024/2025):
https://arxiv.org/abs/2409.10157

Key Innovation: Uses DPO to capture nuanced prosodic differences between positive-negative
emotion pairs, enhancing emotional expressiveness over standard supervised learning.

Benefits:
- More efficient than RLHF (no separate reward model needed)
- Captures subtle emotional nuances that supervised learning misses
- Can be applied as fine-tuning stage after initial prosody training
- Works with existing emotion-labeled data by creating preference pairs

Implementation:
1. Create preference pairs from emotion dataset (chosen=target emotion, rejected=different emotion)
2. Implement JS-regularized DPO loss for stable training
3. Combined training: α·ℒ_DPO + γ·ℒ_KL + θ·ℒ_SFT
4. Fine-tune after initial prosody conditioning training

Usage:
    python emo_dpo.py --config config/emo_dpo.yaml --checkpoint path/to/prosody_checkpoint.pt
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

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmoDPOConfig:
    """Configuration for Emo-DPO training."""

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty coefficient (temperature)
    js_alpha: float = 0.5  # Jensen-Shannon regularization weight

    # Loss weights (following paper: α=γ=θ=1)
    dpo_weight: float = 1.0      # α - DPO loss weight
    kl_weight: float = 1.0       # γ - KL loss weight
    sft_weight: float = 1.0      # θ - SFT loss weight

    # Label smoothing for KL loss
    label_smoothing: float = 0.1

    # Preference pair settings
    same_speaker_pairs: bool = True  # Prefer pairs from same speaker
    same_text_pairs: bool = True     # Prefer pairs with same text (ideal)

    # Training settings
    learning_rate: float = 1e-5  # Lower LR for DPO fine-tuning
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation: int = 2
    max_grad_norm: float = 1.0

    # Model settings
    hidden_size: int = 2048
    num_prosody_tokens: int = 4

    # Reference model settings
    use_reference_model: bool = True  # Keep copy of SFT model for KL
    reference_model_update_freq: int = 0  # 0 = never update (frozen)

    # Emotions
    num_emotions: int = 8
    emotion_names: List[str] = field(default_factory=lambda: [
        'neutral', 'happy', 'sad', 'angry',
        'fearful', 'surprised', 'disgusted', 'calm'
    ])


# Emotion indices (matching train_prosody_conditioned.py)
EMOTION_TO_IDX = {
    'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,
    'fearful': 4, 'surprised': 5, 'disgusted': 6, 'calm': 7,
    'excited': 8, 'contempt': 9,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


# =============================================================================
# PREFERENCE PAIR DATASET
# =============================================================================

@dataclass
class EmotionPreferencePair:
    """A single preference pair for DPO training."""

    # Text/transcript (same for both)
    text: str

    # Chosen sample (target emotion)
    chosen_prosody: Dict[str, torch.Tensor]  # semantic, acoustic, rhythm, contour
    chosen_emotion: int

    # Rejected sample (different emotion)
    rejected_prosody: Dict[str, torch.Tensor]
    rejected_emotion: int

    # Optional metadata (must come after required fields)
    chosen_audio_path: Optional[str] = None
    rejected_audio_path: Optional[str] = None
    speaker_id: Optional[str] = None
    target_emotion_name: str = ""


class EmotionPreferencePairDataset(Dataset):
    """
    Dataset that creates preference pairs from emotion-labeled data.

    For each sample with target emotion E:
    - Chosen: The original sample with emotion E
    - Rejected: A sample with a different emotion but similar/same text

    This teaches the model to prefer correct emotional expressions over incorrect ones.
    """

    def __init__(
        self,
        manifest_path: str,
        prosody_cache_dir: str,
        config: EmoDPOConfig,
        max_pairs_per_sample: int = 3,  # Create up to 3 negative pairs per sample
    ):
        self.config = config
        self.prosody_cache_dir = Path(prosody_cache_dir)
        self.max_pairs_per_sample = max_pairs_per_sample

        # Load manifest
        with open(manifest_path) as f:
            self.samples = json.load(f)

        print(f"Loaded {len(self.samples)} samples from manifest")

        # Index samples by emotion for efficient pair creation
        self.samples_by_emotion = defaultdict(list)
        self.samples_by_text = defaultdict(list)
        self.samples_by_speaker = defaultdict(list)

        for idx, sample in enumerate(self.samples):
            emotion = self._get_emotion(sample)
            text = self._normalize_text(sample.get('text', ''))
            speaker = sample.get('speaker_id', 'default')

            if emotion is not None:
                self.samples_by_emotion[emotion].append(idx)
            if text:
                self.samples_by_text[text].append(idx)
            self.samples_by_speaker[speaker].append(idx)

        # Log emotion distribution
        print("Emotion distribution:")
        for emotion, indices in sorted(self.samples_by_emotion.items()):
            emotion_name = IDX_TO_EMOTION.get(emotion, f'unknown_{emotion}')
            print(f"  {emotion_name}: {len(indices)} samples")

        # Create preference pairs
        self.pairs = self._create_preference_pairs()
        print(f"Created {len(self.pairs)} preference pairs")

    def _get_emotion(self, sample: dict) -> Optional[int]:
        """Extract emotion label from sample."""
        # Try direct emotion field
        emotion = sample.get('emotion', '').lower()
        if emotion in EMOTION_TO_IDX:
            return EMOTION_TO_IDX[emotion]

        # Try prosody.semantic.emotion
        semantic = sample.get('prosody', {}).get('semantic', {})
        emotion = semantic.get('emotion', '').lower()
        if emotion in EMOTION_TO_IDX:
            return EMOTION_TO_IDX[emotion]

        # Try prosody.semantic.emotions dict (pick max)
        emotions = semantic.get('emotions', {})
        if emotions:
            top_emotion = max(emotions.items(), key=lambda kv: kv[1])[0].lower()
            if top_emotion in EMOTION_TO_IDX:
                return EMOTION_TO_IDX[top_emotion]

        return None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching similar utterances."""
        import re
        # Lowercase, remove punctuation, collapse whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _create_preference_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Create preference pairs (chosen_idx, rejected_idx, target_emotion).

        Strategy:
        1. First try to find pairs with same text (ideal for DPO)
        2. Fall back to same speaker with different emotion
        3. Fall back to any sample with different emotion
        """
        pairs = []

        for target_emotion, chosen_indices in self.samples_by_emotion.items():
            other_emotions = [e for e in self.samples_by_emotion.keys() if e != target_emotion]

            if not other_emotions:
                continue

            for chosen_idx in chosen_indices:
                chosen_sample = self.samples[chosen_idx]
                chosen_text = self._normalize_text(chosen_sample.get('text', ''))
                chosen_speaker = chosen_sample.get('speaker_id', 'default')

                # Find rejection candidates
                rejection_candidates = []

                # Priority 1: Same text, different emotion
                if self.config.same_text_pairs and chosen_text:
                    same_text_indices = self.samples_by_text.get(chosen_text, [])
                    for idx in same_text_indices:
                        if idx != chosen_idx:
                            sample_emotion = self._get_emotion(self.samples[idx])
                            if sample_emotion != target_emotion:
                                rejection_candidates.append((idx, 'same_text'))

                # Priority 2: Same speaker, different emotion
                if self.config.same_speaker_pairs and len(rejection_candidates) < self.max_pairs_per_sample:
                    same_speaker_indices = self.samples_by_speaker.get(chosen_speaker, [])
                    for idx in same_speaker_indices:
                        if idx != chosen_idx:
                            sample_emotion = self._get_emotion(self.samples[idx])
                            if sample_emotion != target_emotion:
                                if not any(r[0] == idx for r in rejection_candidates):
                                    rejection_candidates.append((idx, 'same_speaker'))

                # Priority 3: Any sample with different emotion
                if len(rejection_candidates) < self.max_pairs_per_sample:
                    for other_emotion in other_emotions:
                        for idx in self.samples_by_emotion[other_emotion]:
                            if idx != chosen_idx:
                                if not any(r[0] == idx for r in rejection_candidates):
                                    rejection_candidates.append((idx, 'different'))
                                    if len(rejection_candidates) >= self.max_pairs_per_sample * 2:
                                        break
                        if len(rejection_candidates) >= self.max_pairs_per_sample * 2:
                            break

                # Select up to max_pairs_per_sample
                # Prioritize same_text > same_speaker > different
                selected = []
                for priority in ['same_text', 'same_speaker', 'different']:
                    for idx, source in rejection_candidates:
                        if source == priority and len(selected) < self.max_pairs_per_sample:
                            selected.append(idx)

                for rejected_idx in selected:
                    pairs.append((chosen_idx, rejected_idx, target_emotion))

        # Shuffle pairs
        random.shuffle(pairs)

        return pairs

    def _load_prosody(self, sample: dict, idx: int) -> Dict[str, torch.Tensor]:
        """Load prosody features for a sample."""
        # Check cache first
        audio_path = sample.get('audio_path', sample.get('path', sample.get('audio', '')))
        if audio_path:
            import hashlib
            path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
            cache_path = self.prosody_cache_dir / f"prosody_{path_hash}.pt"

            if cache_path.exists():
                return torch.load(cache_path)

        # Extract from sample prosody dict
        prosody = sample.get('prosody', {})

        # Convert to tensors with default dimensions
        from prosody_conditioning import ProsodyConfig
        config = ProsodyConfig()

        def to_tensor(data, dim):
            if isinstance(data, torch.Tensor):
                return data
            if isinstance(data, (list, np.ndarray)):
                t = torch.tensor(data, dtype=torch.float32)
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                # Pad or truncate to expected dimension
                if t.shape[-1] < dim:
                    t = F.pad(t, (0, dim - t.shape[-1]))
                elif t.shape[-1] > dim:
                    t = t[..., :dim]
                return t
            return torch.zeros(dim)

        semantic = prosody.get('semantic', {})
        acoustic = prosody.get('acoustic', {})
        rhythm = prosody.get('rhythm', {})
        contour = prosody.get('contour', prosody.get('pitch_contour', []))

        # Build tensors
        if isinstance(semantic, dict):
            # Extract emotion scores as semantic vector
            emotions = semantic.get('emotions', {})
            semantic_vec = [emotions.get(e, 0.0) for e in ['happy', 'sad', 'angry', 'fearful',
                                                           'surprised', 'disgusted', 'neutral', 'calm']]
            semantic_tensor = torch.tensor(semantic_vec, dtype=torch.float32)
        else:
            semantic_tensor = to_tensor(semantic, config.semantic_dim)

        if isinstance(acoustic, dict):
            acoustic_vec = [
                acoustic.get('pitch_mean', 0.0),
                acoustic.get('pitch_std', 0.0),
                acoustic.get('intensity_mean', 0.0),
                acoustic.get('intensity_std', 0.0),
                acoustic.get('hnr_mean', 0.0),
            ]
            acoustic_tensor = torch.tensor(acoustic_vec, dtype=torch.float32)
            acoustic_tensor = F.pad(acoustic_tensor, (0, config.acoustic_dim - len(acoustic_vec)))
        else:
            acoustic_tensor = to_tensor(acoustic, config.acoustic_dim)

        if isinstance(rhythm, dict):
            rhythm_vec = [
                rhythm.get('speaking_rate', 0.0),
                rhythm.get('pause_ratio', 0.0),
                rhythm.get('syllable_rate', 0.0),
            ]
            rhythm_tensor = torch.tensor(rhythm_vec, dtype=torch.float32)
            rhythm_tensor = F.pad(rhythm_tensor, (0, config.rhythm_dim - len(rhythm_vec)))
        else:
            rhythm_tensor = to_tensor(rhythm, config.rhythm_dim)

        contour_tensor = to_tensor(contour, config.contour_dim)

        return {
            'semantic': semantic_tensor,
            'acoustic': acoustic_tensor,
            'rhythm': rhythm_tensor,
            'contour': contour_tensor,
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        chosen_idx, rejected_idx, target_emotion = self.pairs[idx]

        chosen_sample = self.samples[chosen_idx]
        rejected_sample = self.samples[rejected_idx]

        # Load prosody
        chosen_prosody = self._load_prosody(chosen_sample, chosen_idx)
        rejected_prosody = self._load_prosody(rejected_sample, rejected_idx)

        # Get emotions
        chosen_emotion = self._get_emotion(chosen_sample) or target_emotion
        rejected_emotion = self._get_emotion(rejected_sample) or 0

        return {
            'text': chosen_sample.get('text', ''),
            'target_emotion': target_emotion,

            # Chosen (preferred) sample
            'chosen_semantic': chosen_prosody['semantic'],
            'chosen_acoustic': chosen_prosody['acoustic'],
            'chosen_rhythm': chosen_prosody['rhythm'],
            'chosen_contour': chosen_prosody['contour'],
            'chosen_emotion': chosen_emotion,

            # Rejected sample
            'rejected_semantic': rejected_prosody['semantic'],
            'rejected_acoustic': rejected_prosody['acoustic'],
            'rejected_rhythm': rejected_prosody['rhythm'],
            'rejected_contour': rejected_prosody['contour'],
            'rejected_emotion': rejected_emotion,
        }


def collate_preference_pairs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate preference pairs into batched tensors."""
    return {
        'text': [item['text'] for item in batch],
        'target_emotion': torch.tensor([item['target_emotion'] for item in batch], dtype=torch.long),

        # Chosen
        'chosen_semantic': torch.stack([item['chosen_semantic'] for item in batch]),
        'chosen_acoustic': torch.stack([item['chosen_acoustic'] for item in batch]),
        'chosen_rhythm': torch.stack([item['chosen_rhythm'] for item in batch]),
        'chosen_contour': torch.stack([item['chosen_contour'] for item in batch]),
        'chosen_emotion': torch.tensor([item['chosen_emotion'] for item in batch], dtype=torch.long),

        # Rejected
        'rejected_semantic': torch.stack([item['rejected_semantic'] for item in batch]),
        'rejected_acoustic': torch.stack([item['rejected_acoustic'] for item in batch]),
        'rejected_rhythm': torch.stack([item['rejected_rhythm'] for item in batch]),
        'rejected_contour': torch.stack([item['rejected_contour'] for item in batch]),
        'rejected_emotion': torch.tensor([item['rejected_emotion'] for item in batch], dtype=torch.long),
    }


# =============================================================================
# DPO LOSS FUNCTIONS
# =============================================================================

class DPOLoss(nn.Module):
    """
    Standard Direct Preference Optimization loss.

    ℒ_DPO = -𝔼[log σ(β(log π(y+|x)/π_ref(y+|x) - log π(y-|x)/π_ref(y-|x)))]

    Where:
    - π is the policy (current model)
    - π_ref is the reference model (frozen SFT model)
    - y+ is the chosen/preferred output
    - y- is the rejected output
    - β controls the strength of the KL penalty
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # [batch] log P(y+|x) from policy
        policy_rejected_logps: torch.Tensor,  # [batch] log P(y-|x) from policy
        ref_chosen_logps: torch.Tensor,       # [batch] log P(y+|x) from reference
        ref_rejected_logps: torch.Tensor,     # [batch] log P(y-|x) from reference
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DPO loss.

        Returns:
            loss: Scalar loss value
            metrics: Dict with useful metrics for logging
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO logits: β * (log π(y+)/π_ref(y+) - log π(y-)/π_ref(y-))
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # Loss: -log σ(logits) = log(1 + exp(-logits))
        loss = -F.logsigmoid(logits).mean()

        # Compute metrics
        with torch.no_grad():
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            reward_margin = chosen_rewards - rejected_rewards
            accuracy = (logits > 0).float().mean()

        metrics = {
            'loss': loss.detach(),
            'chosen_rewards': chosen_rewards.mean().detach(),
            'rejected_rewards': rejected_rewards.mean().detach(),
            'reward_margin': reward_margin.mean().detach(),
            'accuracy': accuracy.detach(),
            'logits_mean': logits.mean().detach(),
            'logits_std': logits.std().detach(),
        }

        return loss, metrics


class JSRegularizedDPOLoss(nn.Module):
    """
    Jensen-Shannon Regularized DPO Loss.

    Adds JS divergence regularization for more stable training:
    - More balanced and interpretable preference learning
    - Bounded and symmetric divergence
    - Prevents extreme log-ratio differences

    From Emo-DPO paper: Uses JS divergence to smooth optimization.
    """

    def __init__(self, beta: float = 0.1, js_alpha: float = 0.5):
        super().__init__()
        self.beta = beta
        self.js_alpha = js_alpha  # Weight for JS regularization
        self.base_dpo = DPOLoss(beta)

    def _js_divergence(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon style regularization term.

        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)

        Approximated as: log(1 + exp(chosen_logratio)) - log(1 + exp(rejected_logratio))
        This provides bounded and symmetric regularization.
        """
        # Softplus approximation of JS divergence components
        js_chosen = F.softplus(chosen_logratios)  # log(1 + exp(x))
        js_rejected = F.softplus(rejected_logratios)

        # Symmetric JS-style regularization
        js_term = js_chosen - js_rejected

        return js_term

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute JS-regularized DPO loss.
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Standard DPO logits
        base_logits = self.beta * (chosen_logratios - rejected_logratios)

        # JS regularization
        js_term = self._js_divergence(chosen_logratios, rejected_logratios)

        # Regularized logits: subtract JS term to prevent extreme differences
        regularized_logits = base_logits - self.js_alpha * js_term

        # Loss: -log σ(regularized_logits)
        loss = -F.logsigmoid(regularized_logits).mean()

        # Compute metrics
        with torch.no_grad():
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            reward_margin = chosen_rewards - rejected_rewards
            accuracy = (regularized_logits > 0).float().mean()

        metrics = {
            'loss': loss.detach(),
            'js_term': js_term.mean().detach(),
            'base_logits': base_logits.mean().detach(),
            'regularized_logits': regularized_logits.mean().detach(),
            'chosen_rewards': chosen_rewards.mean().detach(),
            'rejected_rewards': rejected_rewards.mean().detach(),
            'reward_margin': reward_margin.mean().detach(),
            'accuracy': accuracy.detach(),
        }

        return loss, metrics


# =============================================================================
# PROSODY LOG PROBABILITY COMPUTATION
# =============================================================================

class ProsodyLogProbComputer(nn.Module):
    """
    Computes log probabilities for prosody embeddings.

    For DPO, we need log P(y|x) where:
    - x is the input (text + emotion prompt)
    - y is the output (prosody embedding)

    We model this as a Gaussian distribution over the prosody embedding space,
    where the model predicts the mean and we assume fixed or learned variance.
    """

    def __init__(self, hidden_size: int = 2048, num_tokens: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tokens = num_tokens

        # Log variance predictor (learnable uncertainty)
        self.log_var_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

        # Initialize to reasonable variance (σ² ≈ 0.1)
        nn.init.constant_(self.log_var_head[-1].bias, -2.3)

    def compute_log_prob(
        self,
        predicted_embedding: torch.Tensor,  # [batch, num_tokens, hidden] from model
        target_embedding: torch.Tensor,     # [batch, num_tokens, hidden] ground truth
        reduce: bool = True,
    ) -> torch.Tensor:
        """
        Compute log probability of target given predicted distribution.

        Uses Gaussian likelihood: log P(y|μ,σ) = -0.5 * ((y-μ)²/σ² + log σ² + log 2π)

        Args:
            predicted_embedding: Model's predicted prosody embedding (mean of distribution)
            target_embedding: Ground truth prosody embedding
            reduce: If True, return scalar; if False, return per-sample values

        Returns:
            Log probability (higher = better match)
        """
        # Get log variance from model
        log_var = self.log_var_head(predicted_embedding)  # [batch, num_tokens, 1]

        # Clamp for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=5.0)
        var = torch.exp(log_var)

        # Compute squared error
        sq_error = (target_embedding - predicted_embedding) ** 2

        # Gaussian log likelihood (per dimension)
        log_prob = -0.5 * (sq_error / var + log_var + math.log(2 * math.pi))

        # Sum over dimensions and tokens
        log_prob = log_prob.sum(dim=-1).sum(dim=-1)  # [batch]

        if reduce:
            return log_prob.mean()
        return log_prob

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper for compute_log_prob."""
        return self.compute_log_prob(predicted, target, reduce=False)


# =============================================================================
# EMO-DPO TRAINER
# =============================================================================

class EmoDPOTrainer:
    """
    Trainer for Emo-DPO preference optimization.

    Training procedure:
    1. Load pre-trained prosody encoder (from SFT stage)
    2. Create reference copy (frozen) for KL computation
    3. Train with combined loss: α·ℒ_DPO + γ·ℒ_KL + θ·ℒ_SFT
    4. Save fine-tuned model
    """

    def __init__(
        self,
        config: EmoDPOConfig,
        prosody_model: nn.Module,  # Pre-trained ProsodyControlledCSM or similar
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or self._setup_device()

        # Main policy model
        self.policy_model = prosody_model.to(self.device)

        # Reference model (frozen copy for DPO)
        if config.use_reference_model:
            import copy
            self.ref_model = copy.deepcopy(prosody_model)
            self.ref_model = self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            print("Created frozen reference model for DPO")
        else:
            self.ref_model = None

        # Log probability computer
        self.log_prob_computer = ProsodyLogProbComputer(
            hidden_size=config.hidden_size,
            num_tokens=config.num_prosody_tokens,
        ).to(self.device)

        # DPO loss (JS-regularized)
        self.dpo_loss = JSRegularizedDPOLoss(
            beta=config.beta,
            js_alpha=config.js_alpha,
        )

        # Label-smoothing KL loss for regularization
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

        # Optimizer
        trainable_params = list(self.policy_model.parameters())
        trainable_params += list(self.log_prob_computer.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
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
        """
        Get prosody embedding from model.

        Works with ProsodyControlledCSM or ProsodyEncoder.
        """
        if hasattr(model, 'get_prosody_prefix'):
            # ProsodyControlledCSM
            return model.get_prosody_prefix(semantic, acoustic, rhythm, contour)
        elif hasattr(model, 'prosody_encoder'):
            # Model with prosody_encoder attribute
            return model.prosody_encoder(semantic, acoustic, rhythm, contour)
        else:
            # Assume model itself is the encoder
            return model(semantic, acoustic, rhythm, contour)

    def _compute_sft_loss(
        self,
        embedding: torch.Tensor,
        target_prosody: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute supervised fine-tuning loss (reconstruction).

        This ensures the model doesn't drift too far from the original behavior.
        """
        # Construct target from prosody components
        target = torch.cat([
            target_prosody['semantic'],
            target_prosody['acoustic'],
            target_prosody['rhythm'],
            target_prosody['contour'][:, :8] if target_prosody['contour'].shape[1] > 8
            else target_prosody['contour'],
        ], dim=-1)

        # Pad if needed
        if target.shape[-1] < embedding.shape[-1]:
            target = F.pad(target, (0, embedding.shape[-1] - target.shape[-1]))

        # Match dimensions
        target = target.unsqueeze(1).expand_as(embedding[:, :1, :])

        # MSE loss on first token
        return F.mse_loss(embedding[:, 0, :target.shape[-1]], target[:, 0, :])

    def _compute_kl_loss(
        self,
        policy_embedding: torch.Tensor,
        ref_embedding: torch.Tensor,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute label-smoothed KL divergence loss.

        Keeps policy close to reference model distribution.
        """
        # Convert to log probabilities (softmax over hidden dim as pseudo-distribution)
        policy_log_probs = F.log_softmax(policy_embedding, dim=-1)
        ref_probs = F.softmax(ref_embedding, dim=-1)

        # Apply label smoothing
        smoothed_ref = (1 - label_smoothing) * ref_probs + \
                       label_smoothing / ref_probs.shape[-1]

        # KL divergence
        kl = F.kl_div(policy_log_probs, smoothed_ref, reduction='batchmean')

        return kl

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single DPO training step.

        Returns dict of metrics.
        """
        self.policy_model.train()
        self.log_prob_computer.train()

        # Move to device
        chosen_prosody = {
            'semantic': batch['chosen_semantic'].to(self.device),
            'acoustic': batch['chosen_acoustic'].to(self.device),
            'rhythm': batch['chosen_rhythm'].to(self.device),
            'contour': batch['chosen_contour'].to(self.device),
        }
        rejected_prosody = {
            'semantic': batch['rejected_semantic'].to(self.device),
            'acoustic': batch['rejected_acoustic'].to(self.device),
            'rhythm': batch['rejected_rhythm'].to(self.device),
            'contour': batch['rejected_contour'].to(self.device),
        }
        target_emotion = batch['target_emotion'].to(self.device)

        # Get policy embeddings
        policy_chosen_emb = self._get_prosody_embedding(
            self.policy_model,
            chosen_prosody['semantic'],
            chosen_prosody['acoustic'],
            chosen_prosody['rhythm'],
            chosen_prosody['contour'],
        )
        policy_rejected_emb = self._get_prosody_embedding(
            self.policy_model,
            rejected_prosody['semantic'],
            rejected_prosody['acoustic'],
            rejected_prosody['rhythm'],
            rejected_prosody['contour'],
        )

        # Get reference embeddings (frozen)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_chosen_emb = self._get_prosody_embedding(
                    self.ref_model,
                    chosen_prosody['semantic'],
                    chosen_prosody['acoustic'],
                    chosen_prosody['rhythm'],
                    chosen_prosody['contour'],
                )
                ref_rejected_emb = self._get_prosody_embedding(
                    self.ref_model,
                    rejected_prosody['semantic'],
                    rejected_prosody['acoustic'],
                    rejected_prosody['rhythm'],
                    rejected_prosody['contour'],
                )
            else:
                # No reference model - use detached policy
                ref_chosen_emb = policy_chosen_emb.detach()
                ref_rejected_emb = policy_rejected_emb.detach()

        # Compute log probabilities for DPO
        # Policy log probs
        policy_chosen_logps = self.log_prob_computer.compute_log_prob(
            policy_chosen_emb, policy_chosen_emb.detach(), reduce=False
        )
        policy_rejected_logps = self.log_prob_computer.compute_log_prob(
            policy_rejected_emb, policy_rejected_emb.detach(), reduce=False
        )

        # Reference log probs
        ref_chosen_logps = self.log_prob_computer.compute_log_prob(
            ref_chosen_emb, ref_chosen_emb, reduce=False
        )
        ref_rejected_logps = self.log_prob_computer.compute_log_prob(
            ref_rejected_emb, ref_rejected_emb, reduce=False
        )

        # Compute DPO loss
        dpo_loss, dpo_metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        # Compute SFT loss (reconstruction on chosen samples)
        sft_loss = self._compute_sft_loss(policy_chosen_emb, chosen_prosody)

        # Compute KL loss (keep policy close to reference)
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None:
            kl_loss = self._compute_kl_loss(
                policy_chosen_emb,
                ref_chosen_emb,
                self.config.label_smoothing,
            )

        # Combined loss: α·ℒ_DPO + γ·ℒ_KL + θ·ℒ_SFT
        total_loss = (
            self.config.dpo_weight * dpo_loss +
            self.config.kl_weight * kl_loss +
            self.config.sft_weight * sft_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()
        self.global_step += 1

        return {
            'total_loss': total_loss.item(),
            'dpo_loss': dpo_loss.item(),
            'kl_loss': kl_loss.item(),
            'sft_loss': sft_loss.item(),
            'accuracy': dpo_metrics['accuracy'].item(),
            'reward_margin': dpo_metrics['reward_margin'].item(),
            'js_term': dpo_metrics.get('js_term', torch.tensor(0.0)).item(),
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validation loop.

        Returns dict of validation metrics.
        """
        self.policy_model.eval()
        self.log_prob_computer.eval()

        total_metrics = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                chosen_prosody = {
                    'semantic': batch['chosen_semantic'].to(self.device),
                    'acoustic': batch['chosen_acoustic'].to(self.device),
                    'rhythm': batch['chosen_rhythm'].to(self.device),
                    'contour': batch['chosen_contour'].to(self.device),
                }
                rejected_prosody = {
                    'semantic': batch['rejected_semantic'].to(self.device),
                    'acoustic': batch['rejected_acoustic'].to(self.device),
                    'rhythm': batch['rejected_rhythm'].to(self.device),
                    'contour': batch['rejected_contour'].to(self.device),
                }

                # Get embeddings
                policy_chosen_emb = self._get_prosody_embedding(
                    self.policy_model,
                    chosen_prosody['semantic'],
                    chosen_prosody['acoustic'],
                    chosen_prosody['rhythm'],
                    chosen_prosody['contour'],
                )
                policy_rejected_emb = self._get_prosody_embedding(
                    self.policy_model,
                    rejected_prosody['semantic'],
                    rejected_prosody['acoustic'],
                    rejected_prosody['rhythm'],
                    rejected_prosody['contour'],
                )

                # Reference embeddings
                if self.ref_model is not None:
                    ref_chosen_emb = self._get_prosody_embedding(
                        self.ref_model,
                        chosen_prosody['semantic'],
                        chosen_prosody['acoustic'],
                        chosen_prosody['rhythm'],
                        chosen_prosody['contour'],
                    )
                    ref_rejected_emb = self._get_prosody_embedding(
                        self.ref_model,
                        rejected_prosody['semantic'],
                        rejected_prosody['acoustic'],
                        rejected_prosody['rhythm'],
                        rejected_prosody['contour'],
                    )
                else:
                    ref_chosen_emb = policy_chosen_emb
                    ref_rejected_emb = policy_rejected_emb

                # Compute log probs
                policy_chosen_logps = self.log_prob_computer(policy_chosen_emb, policy_chosen_emb)
                policy_rejected_logps = self.log_prob_computer(policy_rejected_emb, policy_rejected_emb)
                ref_chosen_logps = self.log_prob_computer(ref_chosen_emb, ref_chosen_emb)
                ref_rejected_logps = self.log_prob_computer(ref_rejected_emb, ref_rejected_emb)

                # DPO loss
                dpo_loss, dpo_metrics = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                )

                # SFT loss
                sft_loss = self._compute_sft_loss(policy_chosen_emb, chosen_prosody)

                # Accumulate metrics
                total_metrics['dpo_loss'] += dpo_loss.item()
                total_metrics['sft_loss'] += sft_loss.item()
                total_metrics['accuracy'] += dpo_metrics['accuracy'].item()
                total_metrics['reward_margin'] += dpo_metrics['reward_margin'].item()
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
        """
        Main training loop.
        """
        print(f"\nStarting Emo-DPO training for {self.config.num_epochs} epochs")
        print(f"  DPO weight: {self.config.dpo_weight}")
        print(f"  KL weight: {self.config.kl_weight}")
        print(f"  SFT weight: {self.config.sft_weight}")
        print(f"  Beta: {self.config.beta}")
        print(f"  JS alpha: {self.config.js_alpha}")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = defaultdict(float)
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)

                for key, value in metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1

                # Log
                if self.global_step % 10 == 0:
                    print(f"  Step {self.global_step}: "
                          f"loss={metrics['total_loss']:.4f}, "
                          f"dpo={metrics['dpo_loss']:.4f}, "
                          f"sft={metrics['sft_loss']:.4f}, "
                          f"acc={metrics['accuracy']:.2%}")

            # Epoch summary
            for key in epoch_metrics:
                epoch_metrics[key] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}:")
            print(f"  Train - loss: {epoch_metrics['total_loss']:.4f}, "
                  f"accuracy: {epoch_metrics['accuracy']:.2%}, "
                  f"reward_margin: {epoch_metrics['reward_margin']:.4f}")

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"  Val   - dpo_loss: {val_metrics['dpo_loss']:.4f}, "
                      f"accuracy: {val_metrics['accuracy']:.2%}")

                # Track best
                if val_metrics['accuracy'] > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    self.save_checkpoint('best')

        self.save_checkpoint('final')
        print(f"\nTraining complete! Best accuracy: {self.best_accuracy:.2%}")

    def save_checkpoint(self, name: str, output_dir: str = 'checkpoints/emo_dpo'):
        """Save checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'config': {
                'beta': self.config.beta,
                'js_alpha': self.config.js_alpha,
                'dpo_weight': self.config.dpo_weight,
                'kl_weight': self.config.kl_weight,
                'sft_weight': self.config.sft_weight,
            },
        }

        # Save policy model state
        if hasattr(self.policy_model, 'prosody_encoder'):
            checkpoint['prosody_encoder'] = self.policy_model.prosody_encoder.state_dict()
        else:
            checkpoint['model'] = self.policy_model.state_dict()

        # Save log prob computer
        checkpoint['log_prob_computer'] = self.log_prob_computer.state_dict()

        torch.save(checkpoint, output_path / f'{name}.pt')
        print(f"Saved checkpoint: {output_path / f'{name}.pt'}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_preference_pairs_from_manifest(
    manifest_path: str,
    output_path: str,
    config: EmoDPOConfig = None,
) -> str:
    """
    Create preference pairs JSON from emotion-labeled manifest.

    Returns path to the created pairs file.
    """
    if config is None:
        config = EmoDPOConfig()

    dataset = EmotionPreferencePairDataset(
        manifest_path=manifest_path,
        prosody_cache_dir='data/prosody_cache',
        config=config,
    )

    # Export pairs info
    pairs_info = []
    for chosen_idx, rejected_idx, target_emotion in dataset.pairs:
        chosen_sample = dataset.samples[chosen_idx]
        rejected_sample = dataset.samples[rejected_idx]

        pairs_info.append({
            'chosen_idx': chosen_idx,
            'rejected_idx': rejected_idx,
            'target_emotion': IDX_TO_EMOTION.get(target_emotion, 'unknown'),
            'chosen_text': chosen_sample.get('text', ''),
            'rejected_text': rejected_sample.get('text', ''),
            'chosen_emotion': IDX_TO_EMOTION.get(dataset._get_emotion(chosen_sample), 'unknown'),
            'rejected_emotion': IDX_TO_EMOTION.get(dataset._get_emotion(rejected_sample), 'unknown'),
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(pairs_info, f, indent=2)

    print(f"Created {len(pairs_info)} preference pairs at {output_path}")
    return str(output_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Emo-DPO: Preference Optimization for Emotional TTS")
    parser.add_argument('--config', type=str, default='config/emo_dpo.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, help='Pre-trained prosody checkpoint')
    parser.add_argument('--manifest', type=str, help='Training manifest with emotion labels')
    parser.add_argument('--val_manifest', type=str, help='Validation manifest')
    parser.add_argument('--output_dir', type=str, default='checkpoints/emo_dpo', help='Output directory')
    parser.add_argument('--create_pairs', action='store_true', help='Only create preference pairs')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = EmoDPOConfig(**{k: v for k, v in config_dict.items() if hasattr(EmoDPOConfig, k)})
    else:
        config = EmoDPOConfig()

    print("=" * 60)
    print("Emo-DPO: Direct Preference Optimization for Emotional Speech")
    print("=" * 60)

    # Just create pairs?
    if args.create_pairs:
        if not args.manifest:
            print("Error: --manifest required for --create_pairs")
            return
        create_preference_pairs_from_manifest(
            args.manifest,
            args.output_dir + '/preference_pairs.json',
            config,
        )
        return

    # Create dataset
    if args.manifest:
        train_dataset = EmotionPreferencePairDataset(
            manifest_path=args.manifest,
            prosody_cache_dir='data/prosody_cache',
            config=config,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_preference_pairs,
        )
    else:
        print("No manifest provided. Run with --manifest to train.")
        print("\nEmo-DPO creates preference pairs from emotion-labeled data:")
        print("  - Chosen: Sample with target emotion")
        print("  - Rejected: Sample with different emotion (same text preferred)")
        print("\nThe model learns to prefer correct emotional expressions.")
        return

    # Load pre-trained model
    if args.checkpoint:
        from prosody_conditioning import ProsodyControlledCSM, ProsodyConfig, ProsodyEncoder

        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        # Create prosody encoder
        prosody_config = ProsodyConfig(**checkpoint.get('prosody_config', {}))
        prosody_encoder = ProsodyEncoder(prosody_config)
        prosody_encoder.load_state_dict(checkpoint['prosody_encoder'])

        print(f"Loaded pre-trained prosody encoder from {args.checkpoint}")
    else:
        # Create fresh encoder for testing
        from prosody_conditioning import ProsodyConfig, ProsodyEncoder

        prosody_config = ProsodyConfig(hidden_size=config.hidden_size)
        prosody_encoder = ProsodyEncoder(prosody_config)
        print("Created fresh prosody encoder (no checkpoint provided)")

    # Create validation loader
    val_loader = None
    if args.val_manifest:
        val_dataset = EmotionPreferencePairDataset(
            manifest_path=args.val_manifest,
            prosody_cache_dir='data/prosody_cache',
            config=config,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_preference_pairs,
        )

    # Create trainer
    trainer = EmoDPOTrainer(
        config=config,
        prosody_model=prosody_encoder,
    )

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
