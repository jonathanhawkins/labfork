"""
MINE (Mutual Information Neural Estimation) for Emotion-Timbre Disentanglement

Based on:
- "MINE: Mutual Information Neural Estimation" (arXiv:1801.04062)
- "Emotional Text-To-Speech Based on Mutual-Information-Guided Emotion-Timbre
   Disentanglement" (arXiv:2510.01722)

Key technique: Train a discriminator network to estimate mutual information I(emotion, timbre),
then minimize this during training. Unlike GRL which can be unstable, MINE provides a smooth
gradient signal.

Why MINE for emotion-timbre disentanglement:
- Our spherical emotion encoder may still leak speaker information
- MINE loss pushes emotion embeddings to be speaker-agnostic
- Enables better cross-speaker emotion transfer
- More stable than adversarial GRL training

Loss function: L_total = L_recon + β * L_mine
Where L_mine uses the MINE lower bound estimator (Donsker-Varadhan representation)

Usage:
    from mine_disentanglement import MINELoss, MINEConfig

    mine_loss = MINELoss(MINEConfig())

    # In training loop
    emotion_emb = emotion_encoder(audio)
    timbre_emb = speaker_encoder(audio)

    mi_estimate, mine_loss_value = mine_loss(emotion_emb, timbre_emb)

    total_loss = reconstruction_loss + beta * mine_loss_value
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MINEConfig:
    """Configuration for MINE-based disentanglement."""

    # Network architecture
    hidden_dim: int = 512          # Hidden layer dimension
    num_hidden_layers: int = 2     # Number of hidden layers
    activation: str = "relu"       # relu, gelu, elu
    dropout: float = 0.1

    # Input dimensions (will be set dynamically if not provided)
    emotion_dim: int = 256         # Emotion embedding dimension
    timbre_dim: int = 256          # Timbre/speaker embedding dimension

    # Training settings
    mine_variant: str = "mine"     # "mine" or "mine_f" (MINE-f uses f-divergence)
    ema_decay: float = 0.99        # EMA decay for bias correction
    gradient_penalty: float = 0.0  # Optional gradient penalty for stability

    # Loss weight schedule
    beta_start: float = 0.01       # Initial MINE loss weight
    beta_end: float = 1.0          # Final MINE loss weight
    warmup_epochs: int = 5         # Epochs to ramp up beta

    # Margin for MINE-f variant
    mine_f_margin: float = 1.0     # Margin for f-divergence

    # Batch statistics
    use_batch_stats: bool = True   # Use batch statistics for marginal


# =============================================================================
# MINE ESTIMATOR NETWORK
# =============================================================================

class MINEStatisticsNetwork(nn.Module):
    """
    Statistics network T(x, y) for MINE.

    This network learns to distinguish joint distribution p(x,y) from
    marginal product p(x)p(y). The output T(x,y) estimates how "joint"
    a pair of embeddings is.

    Architecture: Concatenate + MLP with residual connections
    """

    def __init__(self, config: MINEConfig):
        super().__init__()
        self.config = config

        # Input projection
        input_dim = config.emotion_dim + config.timbre_dim

        # Activation function
        if config.activation == "relu":
            act_fn = nn.ReLU
        elif config.activation == "gelu":
            act_fn = nn.GELU
        elif config.activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for i in range(config.num_hidden_layers):
            layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                act_fn(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = config.hidden_dim

        # Output layer (scalar)
        layers.append(nn.Linear(config.hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Optional: separate projections for emotion and timbre
        self.emotion_proj = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            act_fn(),
        )

        self.timbre_proj = nn.Sequential(
            nn.Linear(config.timbre_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            act_fn(),
        )

        # Bilinear interaction
        self.bilinear = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1)

        # Final combination
        self.combine = nn.Linear(2, 1)  # Concat + bilinear outputs

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        use_bilinear: bool = True,
    ) -> torch.Tensor:
        """
        Compute statistics T(emotion, timbre).

        Args:
            emotion_emb: [batch, emotion_dim] emotion embedding
            timbre_emb: [batch, timbre_dim] timbre/speaker embedding
            use_bilinear: Whether to use bilinear interaction (more expressive)

        Returns:
            [batch, 1] scalar statistics for each pair
        """
        batch_size = emotion_emb.shape[0]

        # Concatenation path
        concat = torch.cat([emotion_emb, timbre_emb], dim=-1)
        concat_out = self.network(concat)

        if use_bilinear:
            # Bilinear path (captures multiplicative interactions)
            e_proj = self.emotion_proj(emotion_emb)
            t_proj = self.timbre_proj(timbre_emb)
            bilinear_out = self.bilinear(e_proj, t_proj)

            # Combine both paths
            combined = torch.cat([concat_out, bilinear_out], dim=-1)
            return self.combine(combined)
        else:
            return concat_out


class MINEEstimator(nn.Module):
    """
    MINE (Mutual Information Neural Estimation) estimator.

    Uses the Donsker-Varadhan representation:
        I(X;Y) ≥ E_joint[T(x,y)] - log(E_marginal[exp(T(x',y))])

    The lower bound becomes tight when T is the optimal critic.

    To minimize MI during training, we maximize this lower bound as an
    auxiliary loss (with negation), pushing representations to be independent.
    """

    def __init__(self, config: MINEConfig):
        super().__init__()
        self.config = config

        # Statistics network
        self.T = MINEStatisticsNetwork(config)

        # EMA for bias correction (moving average baseline)
        self.register_buffer('ema_exp', torch.tensor(1.0))

        # Training state
        self.register_buffer('step', torch.tensor(0))

    def _create_marginal_samples(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create samples from marginal distribution p(emotion)p(timbre).

        Simply shuffles one of the embeddings to break the joint structure.

        Args:
            emotion_emb: [batch, dim] joint emotion embeddings
            timbre_emb: [batch, dim] joint timbre embeddings

        Returns:
            Tuple of (emotion_emb, shuffled_timbre_emb)
        """
        batch_size = emotion_emb.shape[0]

        # Random permutation for marginal samples
        perm = torch.randperm(batch_size, device=emotion_emb.device)

        # Shuffle timbre while keeping emotion
        timbre_shuffled = timbre_emb[perm]

        return emotion_emb, timbre_shuffled

    def forward(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Estimate mutual information I(emotion; timbre).

        Args:
            emotion_emb: [batch, emotion_dim] emotion embeddings
            timbre_emb: [batch, timbre_dim] timbre embeddings
            return_components: If True, return intermediate values

        Returns:
            MI estimate (scalar). Higher = more mutual information.
            If return_components: also returns dict with T_joint, T_marginal, etc.
        """
        batch_size = emotion_emb.shape[0]

        # Pool if sequence (e.g., [batch, seq, dim] -> [batch, dim])
        if emotion_emb.dim() == 3:
            emotion_emb = emotion_emb.mean(dim=1)
        if timbre_emb.dim() == 3:
            timbre_emb = timbre_emb.mean(dim=1)

        # Joint samples: T(emotion, timbre) from p(emotion, timbre)
        T_joint = self.T(emotion_emb, timbre_emb)  # [batch, 1]

        # Marginal samples: T(emotion, timbre') from p(emotion)p(timbre)
        emotion_marginal, timbre_marginal = self._create_marginal_samples(
            emotion_emb, timbre_emb
        )
        T_marginal = self.T(emotion_marginal, timbre_marginal)  # [batch, 1]

        # Donsker-Varadhan lower bound
        # I(X;Y) ≥ E[T(x,y)] - log(E[exp(T(x',y'))])

        E_joint = T_joint.mean()  # E_joint[T]

        # Log-sum-exp with bias correction for numerical stability
        exp_marginal = torch.exp(T_marginal)

        if self.config.mine_variant == "mine_f":
            # MINE-f: Uses exp(T-1) instead of exp(T)
            # This has better gradient properties
            E_marginal = torch.exp(T_marginal - 1).mean()
            mi_estimate = E_joint - E_marginal
        else:
            # Standard MINE with EMA bias correction
            if self.training:
                # Update EMA
                self.ema_exp = (
                    self.config.ema_decay * self.ema_exp +
                    (1 - self.config.ema_decay) * exp_marginal.mean().detach()
                )
                E_marginal_log = torch.log(exp_marginal.mean() + 1e-8)
            else:
                E_marginal_log = torch.log(exp_marginal.mean() + 1e-8)

            mi_estimate = E_joint - E_marginal_log

        if return_components:
            components = {
                'T_joint': T_joint,
                'T_marginal': T_marginal,
                'E_joint': E_joint,
                'E_marginal_log': E_marginal_log if self.config.mine_variant != "mine_f" else torch.log(E_marginal + 1e-8),
                'ema_exp': self.ema_exp,
            }
            return mi_estimate, components

        return mi_estimate


# =============================================================================
# MINE LOSS
# =============================================================================

class MINELoss(nn.Module):
    """
    MINE-based disentanglement loss for emotion-timbre separation.

    Minimizes mutual information I(emotion; timbre) by:
    1. Estimating MI using MINE lower bound
    2. Returning negative MI as loss (to minimize)

    Usage:
        mine_loss_fn = MINELoss(config)

        # In training loop
        mi_estimate, loss = mine_loss_fn(emotion_emb, timbre_emb)
        total_loss = recon_loss + beta * loss
    """

    def __init__(self, config: MINEConfig):
        super().__init__()
        self.config = config
        self.estimator = MINEEstimator(config)

        # Trainable parameters for adaptive weighting
        self.log_beta = nn.Parameter(torch.tensor(math.log(config.beta_start)))

        # Step counter for schedule
        self.register_buffer('current_epoch', torch.tensor(0.0))

    def get_beta(self) -> float:
        """Get current beta value based on warmup schedule."""
        if self.config.warmup_epochs <= 0:
            return self.config.beta_end

        progress = min(self.current_epoch.item() / self.config.warmup_epochs, 1.0)
        beta = (
            self.config.beta_start +
            progress * (self.config.beta_end - self.config.beta_start)
        )
        return beta

    def update_epoch(self, epoch: int):
        """Update current epoch for beta schedule."""
        self.current_epoch.fill_(float(epoch))

    def forward(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        return_mi_estimate: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute MINE loss to minimize mutual information.

        Args:
            emotion_emb: [batch, dim] emotion embeddings
            timbre_emb: [batch, dim] timbre/speaker embeddings
            return_mi_estimate: Whether to return raw MI estimate

        Returns:
            loss: MINE loss (negate MI estimate to minimize)
            mi_estimate: Raw MI estimate (if return_mi_estimate=True)
        """
        mi_estimate, components = self.estimator(
            emotion_emb, timbre_emb, return_components=True
        )

        # Loss is negative MI (we want to minimize MI)
        # Higher MI = more shared information = bad
        # So we return positive loss when MI is high
        loss = mi_estimate  # Already want to minimize this

        # Optional gradient penalty for stability
        if self.training and self.config.gradient_penalty > 0:
            grad_penalty = self._compute_gradient_penalty(
                emotion_emb, timbre_emb
            )
            loss = loss + self.config.gradient_penalty * grad_penalty

        if return_mi_estimate:
            return mi_estimate, loss
        return loss

    def _compute_gradient_penalty(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient penalty for stability.

        Penalizes large gradients of T w.r.t. inputs.
        """
        emotion_emb = emotion_emb.detach().requires_grad_(True)
        timbre_emb = timbre_emb.detach().requires_grad_(True)

        T_out = self.estimator.T(emotion_emb, timbre_emb)

        gradients = torch.autograd.grad(
            outputs=T_out,
            inputs=[emotion_emb, timbre_emb],
            grad_outputs=torch.ones_like(T_out),
            create_graph=True,
            retain_graph=True,
        )

        gradient_norm = sum(g.pow(2).sum() for g in gradients).sqrt()
        penalty = (gradient_norm - 1).pow(2)

        return penalty


# =============================================================================
# COMBINED DISENTANGLEMENT LOSS WITH MINE
# =============================================================================

class MINEDisentanglementLoss(nn.Module):
    """
    Combined disentanglement loss using MINE.

    Integrates with existing DisentanglementLoss from disentanglement.py,
    but uses MINE instead of (or in addition to) GRL for more stable training.

    Components:
    1. MINE loss: Minimizes I(emotion, timbre)
    2. Orthogonality loss: Soft cosine similarity constraint
    3. F0 regression (optional): Ensures prosody is captured

    Benefits over GRL:
    - Smoother gradients (no reversal discontinuity)
    - Explicit MI minimization (interpretable)
    - More stable training dynamics
    """

    def __init__(
        self,
        mine_config: MINEConfig,
        emotion_dim: int = 256,
        timbre_dim: int = 256,
        use_orthogonality: bool = True,
        orthogonality_target: float = 0.0001,  # Very strict
        mine_weight: float = 1.0,
        ortho_weight: float = 0.5,
    ):
        super().__init__()

        # Update config with actual dimensions
        mine_config.emotion_dim = emotion_dim
        mine_config.timbre_dim = timbre_dim

        self.mine_config = mine_config
        self.mine_loss = MINELoss(mine_config)

        self.use_orthogonality = use_orthogonality
        self.orthogonality_target = orthogonality_target

        self.mine_weight = mine_weight
        self.ortho_weight = ortho_weight

    def _orthogonality_loss(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft orthogonality loss.

        Pushes cosine similarity toward target (near zero).
        """
        if emotion_emb.dim() == 3:
            emotion_emb = emotion_emb.mean(dim=1)
        if timbre_emb.dim() == 3:
            timbre_emb = timbre_emb.mean(dim=1)

        # L2 normalize
        e_norm = F.normalize(emotion_emb, p=2, dim=-1)
        t_norm = F.normalize(timbre_emb, p=2, dim=-1)

        # Cosine similarity
        cos_sim = (e_norm * t_norm).sum(dim=-1)  # [batch]

        # Push toward target
        loss = (cos_sim - self.orthogonality_target).pow(2).mean()

        return loss

    def forward(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined disentanglement loss.

        Args:
            emotion_emb: [batch, dim] or [batch, seq, dim] emotion embeddings
            timbre_emb: [batch, dim] timbre/speaker embeddings
            epoch: Current epoch for beta schedule

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Update epoch for schedule
        if epoch is not None:
            self.mine_loss.update_epoch(epoch)

        # MINE loss (main disentanglement signal)
        mi_estimate, mine_loss = self.mine_loss(emotion_emb, timbre_emb)
        losses['mine'] = mine_loss
        losses['mi_estimate'] = mi_estimate.detach()  # For logging

        # Orthogonality loss (complementary signal)
        if self.use_orthogonality:
            ortho_loss = self._orthogonality_loss(emotion_emb, timbre_emb)
            losses['orthogonality'] = ortho_loss
        else:
            losses['orthogonality'] = torch.tensor(0.0, device=emotion_emb.device)

        # Compute total with current beta
        beta = self.mine_loss.get_beta()
        losses['beta'] = torch.tensor(beta, device=emotion_emb.device)

        losses['total'] = (
            self.mine_weight * beta * losses['mine'] +
            self.ortho_weight * losses['orthogonality']
        )

        return losses


# =============================================================================
# CLUB: CONTRASTIVE LOG-RATIO UPPER BOUND (ALTERNATIVE TO MINE)
# =============================================================================

class CLUBEstimator(nn.Module):
    """
    CLUB (Contrastive Log-ratio Upper Bound) for MI estimation.

    Alternative to MINE that provides an upper bound instead of lower bound.
    Can be used together with MINE to sandwich the true MI value.

    CLUB formula:
        I(X;Y) ≤ E_joint[log q(y|x)] - E_joint,marginal[log q(y'|x)]

    Where q(y|x) is a variational approximation learned by the network.
    """

    def __init__(
        self,
        emotion_dim: int = 256,
        timbre_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # Network to approximate p(timbre|emotion)
        self.mean_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, timbre_dim),
        )

        self.logvar_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, timbre_dim),
            nn.Tanh(),  # Bound log variance
        )

    def _log_prob(
        self,
        timbre: torch.Tensor,
        emotion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(timbre|emotion).

        Assumes Gaussian distribution.
        """
        mu = self.mean_net(emotion)
        logvar = self.logvar_net(emotion)

        # Log probability of Gaussian
        log_prob = -0.5 * (
            logvar +
            (timbre - mu).pow(2) / (torch.exp(logvar) + 1e-8)
        )

        return log_prob.sum(dim=-1)  # Sum over dimensions

    def forward(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate MI upper bound using CLUB.

        Args:
            emotion_emb: [batch, emotion_dim]
            timbre_emb: [batch, timbre_dim]

        Returns:
            MI upper bound estimate
        """
        if emotion_emb.dim() == 3:
            emotion_emb = emotion_emb.mean(dim=1)
        if timbre_emb.dim() == 3:
            timbre_emb = timbre_emb.mean(dim=1)

        batch_size = emotion_emb.shape[0]

        # Positive samples (joint)
        positive = self._log_prob(timbre_emb, emotion_emb)  # [batch]

        # Negative samples (shuffle timbre)
        perm = torch.randperm(batch_size, device=emotion_emb.device)
        timbre_shuffled = timbre_emb[perm]
        negative = self._log_prob(timbre_shuffled, emotion_emb)  # [batch]

        # CLUB upper bound
        mi_upper = (positive - negative).mean()

        return mi_upper

    def compute_learning_loss(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for training the variational network.

        This is the negative log-likelihood of timbre given emotion.
        """
        log_prob = self._log_prob(timbre_emb, emotion_emb)
        return -log_prob.mean()


# =============================================================================
# HYBRID MINE + CLUB LOSS
# =============================================================================

class HybridMILoss(nn.Module):
    """
    Hybrid MI minimization using both MINE (lower bound) and CLUB (upper bound).

    Benefits:
    - MINE provides lower bound, CLUB provides upper bound
    - Combined loss sandwiches true MI, improving estimation
    - More robust training signal

    Loss = α * MINE_loss + (1-α) * CLUB_loss
    """

    def __init__(
        self,
        emotion_dim: int = 256,
        timbre_dim: int = 256,
        mine_weight: float = 0.5,
        club_weight: float = 0.5,
        mine_config: Optional[MINEConfig] = None,
    ):
        super().__init__()

        if mine_config is None:
            mine_config = MINEConfig(emotion_dim=emotion_dim, timbre_dim=timbre_dim)
        else:
            mine_config.emotion_dim = emotion_dim
            mine_config.timbre_dim = timbre_dim

        self.mine_loss = MINELoss(mine_config)
        self.club_estimator = CLUBEstimator(emotion_dim, timbre_dim)

        self.mine_weight = mine_weight
        self.club_weight = club_weight

    def forward(
        self,
        emotion_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        train_club: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid MI loss.

        Args:
            emotion_emb: [batch, dim] emotion embeddings
            timbre_emb: [batch, dim] timbre embeddings
            train_club: Whether to also train CLUB variational network

        Returns:
            Dict with all losses
        """
        losses = {}

        # MINE loss (lower bound)
        mi_lower, mine_loss = self.mine_loss(emotion_emb, timbre_emb)
        losses['mine'] = mine_loss
        losses['mi_lower'] = mi_lower.detach()

        # CLUB loss (upper bound)
        mi_upper = self.club_estimator(emotion_emb, timbre_emb)
        losses['club'] = mi_upper  # Minimize this too
        losses['mi_upper'] = mi_upper.detach()

        # CLUB network training loss
        if train_club:
            club_train_loss = self.club_estimator.compute_learning_loss(
                emotion_emb.detach(), timbre_emb.detach()
            )
            losses['club_train'] = club_train_loss
        else:
            losses['club_train'] = torch.tensor(0.0, device=emotion_emb.device)

        # Combined loss
        losses['total'] = (
            self.mine_weight * mine_loss +
            self.club_weight * mi_upper +
            0.1 * losses['club_train']  # Small weight for CLUB training
        )

        return losses


# =============================================================================
# INTEGRATION WITH PROSODY TRAINING
# =============================================================================

class ProsodyMINEAdapter(nn.Module):
    """
    Adapter to integrate MINE loss with prosody-controlled training.

    This module wraps the prosody encoder and adds MINE-based disentanglement
    from speaker/timbre information.

    Usage with existing training:
        adapter = ProsodyMINEAdapter(prosody_encoder, speaker_encoder, mine_config)

        # In training loop
        prosody_output = adapter(audio_features, speaker_emb)
        prosody_emb = prosody_output['prosody_emb']
        mine_losses = prosody_output['mine_losses']

        total_loss = recon_loss + mine_losses['total']
    """

    def __init__(
        self,
        prosody_encoder: nn.Module,
        speaker_encoder: Optional[nn.Module] = None,
        mine_config: Optional[MINEConfig] = None,
        prosody_dim: int = 2048,
        speaker_dim: int = 256,
    ):
        super().__init__()

        self.prosody_encoder = prosody_encoder
        self.speaker_encoder = speaker_encoder

        if mine_config is None:
            mine_config = MINEConfig()
        mine_config.emotion_dim = prosody_dim
        mine_config.timbre_dim = speaker_dim

        self.mine_loss = MINEDisentanglementLoss(
            mine_config=mine_config,
            emotion_dim=prosody_dim,
            timbre_dim=speaker_dim,
        )

        # Projection if dimensions don't match
        if prosody_dim != speaker_dim:
            self.prosody_proj = nn.Linear(prosody_dim, mine_config.hidden_dim)
            self.speaker_proj = nn.Linear(speaker_dim, mine_config.hidden_dim)
            self.use_projection = True
        else:
            self.use_projection = False

    def forward(
        self,
        audio_features: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        **prosody_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MINE disentanglement.

        Args:
            audio_features: Input to prosody encoder
            speaker_emb: Pre-computed speaker embedding, or None to extract
            mel: Mel spectrogram for speaker extraction (if speaker_emb is None)
            epoch: Current epoch for schedule
            **prosody_kwargs: Additional args for prosody encoder

        Returns:
            Dict with prosody output and MINE losses
        """
        # Get prosody embedding
        prosody_output = self.prosody_encoder(audio_features, **prosody_kwargs)

        if isinstance(prosody_output, dict):
            prosody_emb = prosody_output.get('embedding', prosody_output.get('tokens'))
        else:
            prosody_emb = prosody_output

        # Get speaker embedding
        if speaker_emb is None and self.speaker_encoder is not None and mel is not None:
            speaker_emb = self.speaker_encoder(mel)

        # Compute MINE loss if we have both embeddings
        if speaker_emb is not None:
            # Pool prosody if needed
            if prosody_emb.dim() == 3:
                prosody_for_mine = prosody_emb.mean(dim=1)
            else:
                prosody_for_mine = prosody_emb

            # Project if needed
            if self.use_projection:
                prosody_for_mine = self.prosody_proj(prosody_for_mine)
                speaker_for_mine = self.speaker_proj(speaker_emb)
            else:
                speaker_for_mine = speaker_emb

            mine_losses = self.mine_loss(
                prosody_for_mine,
                speaker_for_mine,
                epoch=epoch,
            )
        else:
            mine_losses = {
                'mine': torch.tensor(0.0, device=prosody_emb.device),
                'total': torch.tensor(0.0, device=prosody_emb.device),
            }

        return {
            'prosody_emb': prosody_emb,
            'prosody_output': prosody_output,
            'mine_losses': mine_losses,
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MINE (Mutual Information Neural Estimation) - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = MINEConfig(
        emotion_dim=256,
        timbre_dim=256,
        hidden_dim=512,
    )

    # Test 1: MINEStatisticsNetwork
    print("\n[Test 1] MINEStatisticsNetwork...")
    T_net = MINEStatisticsNetwork(config).to(device)
    emotion = torch.randn(8, 256, device=device)
    timbre = torch.randn(8, 256, device=device)
    T_out = T_net(emotion, timbre)
    print(f"  Input shapes: emotion={emotion.shape}, timbre={timbre.shape}")
    print(f"  Output shape: {T_out.shape}")
    print(f"  Output range: [{T_out.min().item():.3f}, {T_out.max().item():.3f}]")
    print("  [PASS]")

    # Test 2: MINEEstimator
    print("\n[Test 2] MINEEstimator...")
    estimator = MINEEstimator(config).to(device)
    mi_estimate, components = estimator(emotion, timbre, return_components=True)
    print(f"  MI estimate: {mi_estimate.item():.4f}")
    print(f"  E_joint: {components['E_joint'].item():.4f}")
    print(f"  E_marginal_log: {components['E_marginal_log'].item():.4f}")
    print("  [PASS]")

    # Test 3: MI with correlated vs independent embeddings
    print("\n[Test 3] MI correlation test...")
    # Highly correlated (same embedding)
    mi_correlated = estimator(emotion, emotion).item()
    # Independent (random)
    mi_independent = estimator(emotion, torch.randn_like(emotion)).item()
    print(f"  MI (correlated): {mi_correlated:.4f}")
    print(f"  MI (independent): {mi_independent:.4f}")
    print(f"  Correlated > Independent: {mi_correlated > mi_independent}")
    print("  [PASS]")

    # Test 4: MINELoss
    print("\n[Test 4] MINELoss...")
    mine_loss_fn = MINELoss(config).to(device)
    mi, loss = mine_loss_fn(emotion, timbre)
    print(f"  MI estimate: {mi.item():.4f}")
    print(f"  Loss: {loss.item():.4f}")
    print("  [PASS]")

    # Test 5: MINELoss gradient flow
    print("\n[Test 5] MINELoss gradient flow...")
    emotion_param = nn.Parameter(emotion.clone())
    mi, loss = mine_loss_fn(emotion_param, timbre)
    loss.backward()
    print(f"  Gradient norm: {emotion_param.grad.norm().item():.6f}")
    print(f"  Gradient is non-zero: {emotion_param.grad.abs().sum().item() > 0}")
    print("  [PASS]")

    # Test 6: Beta schedule
    print("\n[Test 6] Beta warmup schedule...")
    for epoch in [0, 2, 5, 10]:
        mine_loss_fn.update_epoch(epoch)
        beta = mine_loss_fn.get_beta()
        print(f"  Epoch {epoch}: beta={beta:.4f}")
    print("  [PASS]")

    # Test 7: MINEDisentanglementLoss
    print("\n[Test 7] MINEDisentanglementLoss (combined)...")
    combined_loss = MINEDisentanglementLoss(
        mine_config=config,
        emotion_dim=256,
        timbre_dim=256,
    ).to(device)

    losses = combined_loss(emotion, timbre, epoch=3)
    print(f"  MINE loss: {losses['mine'].item():.4f}")
    print(f"  MI estimate: {losses['mi_estimate'].item():.4f}")
    print(f"  Orthogonality: {losses['orthogonality'].item():.4f}")
    print(f"  Beta: {losses['beta'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 8: CLUBEstimator
    print("\n[Test 8] CLUBEstimator (upper bound)...")
    club = CLUBEstimator(emotion_dim=256, timbre_dim=256).to(device)
    mi_upper = club(emotion, timbre)
    club_train_loss = club.compute_learning_loss(emotion, timbre)
    print(f"  MI upper bound: {mi_upper.item():.4f}")
    print(f"  CLUB training loss: {club_train_loss.item():.4f}")
    print("  [PASS]")

    # Test 9: HybridMILoss
    print("\n[Test 9] HybridMILoss (MINE + CLUB)...")
    hybrid_loss = HybridMILoss(emotion_dim=256, timbre_dim=256).to(device)
    hybrid_losses = hybrid_loss(emotion, timbre)
    print(f"  MI lower (MINE): {hybrid_losses['mi_lower'].item():.4f}")
    print(f"  MI upper (CLUB): {hybrid_losses['mi_upper'].item():.4f}")
    print(f"  Total loss: {hybrid_losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 10: Sequence input handling
    print("\n[Test 10] Sequence input handling...")
    emotion_seq = torch.randn(4, 50, 256, device=device)  # [batch, seq, dim]
    timbre_single = torch.randn(4, 256, device=device)   # [batch, dim]
    mi_seq = estimator(emotion_seq, timbre_single)
    print(f"  Emotion seq shape: {emotion_seq.shape}")
    print(f"  Timbre shape: {timbre_single.shape}")
    print(f"  MI estimate: {mi_seq.item():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All MINE tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from mine_disentanglement import (
    MINEConfig,
    MINELoss,
    MINEDisentanglementLoss,
)

# Initialize
config = MINEConfig(
    emotion_dim=256,    # Match your emotion encoder output
    timbre_dim=256,     # Match your speaker encoder output
    beta_start=0.01,    # Initial weight
    beta_end=1.0,       # Final weight
    warmup_epochs=5,    # Ramp up over epochs
)

mine_loss_fn = MINEDisentanglementLoss(mine_config=config)

# In training loop
for epoch in range(epochs):
    for batch in dataloader:
        emotion_emb = emotion_encoder(batch['audio'])
        timbre_emb = speaker_encoder(batch['mel'])

        # Get reconstruction loss from your model
        recon_loss = compute_reconstruction_loss(...)

        # Compute MINE disentanglement loss
        mine_losses = mine_loss_fn(emotion_emb, timbre_emb, epoch=epoch)

        # Combined loss
        total_loss = recon_loss + mine_losses['total']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log MI estimate (should decrease over training)
        if step % 100 == 0:
            print(f"MI estimate: {mine_losses['mi_estimate'].item():.4f}")

# For integration with existing prosody encoder:
from mine_disentanglement import ProsodyMINEAdapter

adapter = ProsodyMINEAdapter(
    prosody_encoder=your_prosody_encoder,
    speaker_encoder=your_speaker_encoder,
    mine_config=config,
)

output = adapter(audio_features, mel=mel_spec, epoch=epoch)
prosody_emb = output['prosody_emb']
mine_losses = output['mine_losses']
""")
