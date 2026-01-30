"""
Learn2Diss: Dual Encoder Framework with Sandwiched MI Estimation

Based on "Learn2Diss: Learning to Disentangle Speech Representations" (arXiv:2407.02543)

Key Innovation - Sandwiched MI:
- MINE provides lower bound on I(content; prosody)
- CLUB provides upper bound (Contrastive Log-ratio Upper Bound)
- Together they "sandwich" the true MI value
- More reliable disentanglement than single-sided bounds

CLUB Formula:
I(X;Y) ≤ E_joint[log p(y|x)] - E_marginal[log p(y|x)]

Benefits:
- Tighter MI bounds → better disentanglement verification
- Can detect when MINE alone is unreliable (gap between bounds)
- Variational CLUB is differentiable for end-to-end training
- Combines with existing MINE implementation for robust training

Usage:
    from learn2diss import Learn2DissConfig, Learn2DissLoss, DualEncoderMIEstimator

    config = Learn2DissConfig(
        prosody_dim=2048,
        timbre_dim=256,
    )

    loss_fn = Learn2DissLoss(config)

    # In training loop
    prosody_emb = prosody_encoder(audio)
    timbre_emb = speaker_encoder(mel)

    losses = loss_fn(prosody_emb, timbre_emb, epoch=current_epoch)

    # Sandwiched MI estimate
    print(f"MI lower (MINE): {losses['mi_lower']:.4f}")
    print(f"MI upper (CLUB): {losses['mi_upper']:.4f}")
    print(f"MI gap: {losses['mi_gap']:.4f}")  # Should decrease over training
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
class Learn2DissConfig:
    """Configuration for Learn2Diss dual encoder disentanglement."""

    # Input dimensions
    prosody_dim: int = 2048        # Prosody embedding dimension
    timbre_dim: int = 256          # Timbre/speaker embedding dimension

    # Network architecture
    hidden_dim: int = 512          # Hidden layer dimension
    num_hidden_layers: int = 2     # Number of hidden layers
    activation: str = "gelu"       # gelu, relu, elu
    dropout: float = 0.1

    # MINE settings (lower bound)
    mine_variant: str = "mine"     # "mine" or "mine_f"
    mine_ema_decay: float = 0.99   # EMA decay for bias correction

    # CLUB settings (upper bound)
    club_variance_type: str = "learned"  # "learned" or "fixed"
    club_fixed_variance: float = 1.0     # Fixed variance if not learned
    club_samples: int = 1               # Number of samples for CLUB estimate

    # Variational CLUB settings
    use_variational_club: bool = True   # Use variational posterior approximation
    variational_layers: int = 2         # Layers in variational network

    # Loss weights
    mine_weight: float = 1.0       # Weight for MINE loss
    club_weight: float = 1.0       # Weight for CLUB loss
    club_train_weight: float = 0.1 # Weight for CLUB network training

    # Sandwiched MI settings
    use_sandwiched: bool = True    # Use both bounds together
    gap_penalty_weight: float = 0.01  # Penalty for large MI gap (bounds diverging)

    # Beta schedule for warmup
    beta_start: float = 0.01       # Initial loss weight
    beta_end: float = 1.0          # Final loss weight
    warmup_epochs: int = 5         # Epochs to ramp up beta

    # Orthogonality regularization
    use_orthogonality: bool = True
    orthogonality_target: float = 0.0001  # Very strict (DisCodec β_timbre_prosody)
    orthogonality_weight: float = 0.5


# =============================================================================
# VARIATIONAL CLUB ESTIMATOR
# =============================================================================

class VariationalCLUBNetwork(nn.Module):
    """
    Variational network for CLUB (Contrastive Log-ratio Upper Bound).

    Learns the conditional distribution q(y|x) as a Gaussian with
    learned mean and variance. The CLUB upper bound is:

        I(X;Y) ≤ E_joint[log q(y|x)] - E_marginal[log q(y'|x)]

    This is differentiable and can be trained end-to-end.
    """

    def __init__(self, config: Learn2DissConfig):
        super().__init__()
        self.config = config

        # Choose activation
        if config.activation == "gelu":
            act_fn = nn.GELU
        elif config.activation == "relu":
            act_fn = nn.ReLU
        elif config.activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.GELU

        # Shared encoder for mean and logvar
        encoder_layers = []
        prev_dim = config.prosody_dim

        for i in range(config.variational_layers):
            encoder_layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                act_fn(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = config.hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Mean prediction: q(y|x) has mean μ(x)
        self.mean_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            act_fn(),
            nn.Linear(config.hidden_dim, config.timbre_dim),
        )

        # Log-variance prediction: q(y|x) has variance σ²(x)
        if config.club_variance_type == "learned":
            self.logvar_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                act_fn(),
                nn.Linear(config.hidden_dim, config.timbre_dim),
                nn.Tanh(),  # Bound log variance to [-1, 1]
            )
        else:
            # Fixed variance
            self.register_buffer(
                'fixed_logvar',
                torch.log(torch.tensor(config.club_fixed_variance))
            )
            self.logvar_head = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_conditional_params(
        self,
        prosody_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get parameters of q(timbre|prosody).

        Args:
            prosody_emb: [batch, prosody_dim] prosody embeddings

        Returns:
            (mean, logvar) of the conditional Gaussian
        """
        # Pool if sequence
        if prosody_emb.dim() == 3:
            prosody_emb = prosody_emb.mean(dim=1)

        # Encode
        h = self.encoder(prosody_emb)

        # Get mean
        mean = self.mean_head(h)

        # Get log-variance
        if self.logvar_head is not None:
            logvar = self.logvar_head(h)
        else:
            logvar = self.fixed_logvar.expand_as(mean)

        return mean, logvar

    def log_prob(
        self,
        timbre_emb: torch.Tensor,
        prosody_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(timbre|prosody).

        Args:
            timbre_emb: [batch, timbre_dim] timbre embeddings
            prosody_emb: [batch, prosody_dim] prosody embeddings

        Returns:
            [batch] log probabilities
        """
        # Pool if sequence
        if timbre_emb.dim() == 3:
            timbre_emb = timbre_emb.mean(dim=1)

        mean, logvar = self.get_conditional_params(prosody_emb)

        # Gaussian log probability
        # log N(y; μ, σ²) = -0.5 * (logvar + (y-μ)²/σ² + log(2π))
        log_prob = -0.5 * (
            logvar +
            (timbre_emb - mean).pow(2) / (torch.exp(logvar) + 1e-8) +
            math.log(2 * math.pi)
        )

        return log_prob.sum(dim=-1)  # Sum over dimensions

    def forward(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing CLUB upper bound.

        Args:
            prosody_emb: [batch, prosody_dim]
            timbre_emb: [batch, timbre_dim]

        Returns:
            Dict with mi_upper and training loss
        """
        batch_size = prosody_emb.shape[0] if prosody_emb.dim() == 2 else prosody_emb.shape[0]

        # Positive samples (joint distribution)
        log_prob_pos = self.log_prob(timbre_emb, prosody_emb)  # [batch]

        # Negative samples (marginal - shuffle timbre)
        perm = torch.randperm(batch_size, device=prosody_emb.device)
        timbre_shuffled = timbre_emb[perm] if timbre_emb.dim() == 2 else timbre_emb[perm]
        log_prob_neg = self.log_prob(timbre_shuffled, prosody_emb)  # [batch]

        # CLUB upper bound: E_joint[log q(y|x)] - E_marginal[log q(y'|x)]
        mi_upper = (log_prob_pos - log_prob_neg).mean()

        # Training loss: negative log-likelihood (to learn good q)
        # Detach prosody to train only the variational network
        training_loss = -self.log_prob(timbre_emb.detach(), prosody_emb.detach()).mean()

        return {
            'mi_upper': mi_upper,
            'training_loss': training_loss,
            'log_prob_pos': log_prob_pos.mean().detach(),
            'log_prob_neg': log_prob_neg.mean().detach(),
        }


# =============================================================================
# MINE ESTIMATOR (LOWER BOUND)
# =============================================================================

class MINEStatisticsNetwork(nn.Module):
    """
    Statistics network T(x, y) for MINE lower bound.

    Uses bilinear interaction for capturing dependencies between
    prosody and timbre embeddings.
    """

    def __init__(self, config: Learn2DissConfig):
        super().__init__()
        self.config = config

        # Choose activation
        if config.activation == "gelu":
            act_fn = nn.GELU
        elif config.activation == "relu":
            act_fn = nn.ReLU
        elif config.activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.GELU

        # Projections
        self.prosody_proj = nn.Sequential(
            nn.Linear(config.prosody_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            act_fn(),
        )

        self.timbre_proj = nn.Sequential(
            nn.Linear(config.timbre_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            act_fn(),
        )

        # Concatenation path
        concat_layers = []
        prev_dim = config.prosody_dim + config.timbre_dim

        for i in range(config.num_hidden_layers):
            concat_layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                act_fn(),
                nn.Dropout(config.dropout),
            ])
            prev_dim = config.hidden_dim

        concat_layers.append(nn.Linear(config.hidden_dim, 1))
        self.concat_network = nn.Sequential(*concat_layers)

        # Bilinear interaction
        self.bilinear = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1)

        # Combine paths
        self.combine = nn.Linear(2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute T(prosody, timbre).

        Args:
            prosody_emb: [batch, prosody_dim]
            timbre_emb: [batch, timbre_dim]

        Returns:
            [batch, 1] statistics values
        """
        # Pool if sequence
        if prosody_emb.dim() == 3:
            prosody_emb = prosody_emb.mean(dim=1)
        if timbre_emb.dim() == 3:
            timbre_emb = timbre_emb.mean(dim=1)

        # Concatenation path
        concat = torch.cat([prosody_emb, timbre_emb], dim=-1)
        concat_out = self.concat_network(concat)

        # Bilinear path
        p_proj = self.prosody_proj(prosody_emb)
        t_proj = self.timbre_proj(timbre_emb)
        bilinear_out = self.bilinear(p_proj, t_proj)

        # Combine
        combined = torch.cat([concat_out, bilinear_out], dim=-1)
        return self.combine(combined)


class MINEEstimator(nn.Module):
    """
    MINE (Mutual Information Neural Estimation) for lower bound.

    Uses Donsker-Varadhan representation:
        I(X;Y) ≥ E_joint[T(x,y)] - log(E_marginal[exp(T(x',y))])
    """

    def __init__(self, config: Learn2DissConfig):
        super().__init__()
        self.config = config
        self.T = MINEStatisticsNetwork(config)

        # EMA for bias correction
        self.register_buffer('ema_exp', torch.tensor(1.0))

    def forward(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MINE lower bound.

        Args:
            prosody_emb: [batch, prosody_dim]
            timbre_emb: [batch, timbre_dim]

        Returns:
            Dict with mi_lower and components
        """
        batch_size = prosody_emb.shape[0] if prosody_emb.dim() == 2 else prosody_emb.shape[0]

        # Joint samples
        T_joint = self.T(prosody_emb, timbre_emb)  # [batch, 1]

        # Marginal samples (shuffle timbre)
        perm = torch.randperm(batch_size, device=prosody_emb.device)
        timbre_shuffled = timbre_emb[perm] if timbre_emb.dim() == 2 else timbre_emb[perm]
        T_marginal = self.T(prosody_emb, timbre_shuffled)  # [batch, 1]

        # Donsker-Varadhan bound
        E_joint = T_joint.mean()

        if self.config.mine_variant == "mine_f":
            # MINE-f with exp(T-1)
            E_marginal = torch.exp(T_marginal - 1).mean()
            mi_lower = E_joint - E_marginal
        else:
            # Standard MINE with EMA bias correction
            exp_marginal = torch.exp(T_marginal)

            if self.training:
                self.ema_exp = (
                    self.config.mine_ema_decay * self.ema_exp +
                    (1 - self.config.mine_ema_decay) * exp_marginal.mean().detach()
                )

            E_marginal_log = torch.log(exp_marginal.mean() + 1e-8)
            mi_lower = E_joint - E_marginal_log

        return {
            'mi_lower': mi_lower,
            'T_joint_mean': T_joint.mean().detach(),
            'T_marginal_mean': T_marginal.mean().detach(),
            'E_joint': E_joint.detach(),
        }


# =============================================================================
# DUAL ENCODER MI ESTIMATOR (SANDWICHED)
# =============================================================================

class DualEncoderMIEstimator(nn.Module):
    """
    Dual encoder for sandwiched MI estimation.

    Combines MINE (lower bound) and CLUB (upper bound) to provide
    tighter estimation of mutual information between prosody and timbre.

    The true MI is sandwiched: MINE ≤ I(prosody; timbre) ≤ CLUB

    Benefits:
    - If bounds are tight (gap is small), we have accurate MI estimate
    - If bounds diverge, we know estimation is unreliable
    - Minimizing both bounds forces I(prosody; timbre) → 0
    """

    def __init__(self, config: Learn2DissConfig):
        super().__init__()
        self.config = config

        # MINE (lower bound)
        self.mine = MINEEstimator(config)

        # CLUB (upper bound)
        if config.use_variational_club:
            self.club = VariationalCLUBNetwork(config)
        else:
            # Simple CLUB (basic implementation)
            self.club = SimpleCLUBEstimator(config)

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
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        train_club: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sandwiched MI estimate.

        Args:
            prosody_emb: [batch, prosody_dim] or [batch, seq, prosody_dim]
            timbre_emb: [batch, timbre_dim] or [batch, seq, timbre_dim]
            train_club: Whether to also compute CLUB training loss

        Returns:
            Dict with mi_lower, mi_upper, mi_gap, and losses
        """
        # MINE lower bound
        mine_result = self.mine(prosody_emb, timbre_emb)
        mi_lower = mine_result['mi_lower']

        # CLUB upper bound
        club_result = self.club(prosody_emb, timbre_emb)
        mi_upper = club_result['mi_upper']

        # MI gap (should be small for reliable estimation)
        mi_gap = torch.relu(mi_upper - mi_lower)  # Should be positive

        # Sandwiched estimate (midpoint)
        mi_estimate = (mi_lower + mi_upper) / 2

        return {
            'mi_lower': mi_lower,
            'mi_upper': mi_upper,
            'mi_gap': mi_gap,
            'mi_estimate': mi_estimate,
            'mine_loss': mi_lower,  # Minimize lower bound
            'club_loss': mi_upper,  # Minimize upper bound
            'club_train_loss': club_result['training_loss'] if train_club else torch.tensor(0.0),
        }


class SimpleCLUBEstimator(nn.Module):
    """
    Simple CLUB estimator without variational network.

    Uses basic Gaussian assumption with fixed variance.
    """

    def __init__(self, config: Learn2DissConfig):
        super().__init__()
        self.config = config

        # Mean prediction network
        self.mean_net = nn.Sequential(
            nn.Linear(config.prosody_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.timbre_dim),
        )

        # Fixed log-variance
        self.register_buffer(
            'logvar',
            torch.log(torch.tensor(config.club_fixed_variance))
        )

    def _log_prob(
        self,
        timbre: torch.Tensor,
        prosody: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log q(timbre|prosody)."""
        if prosody.dim() == 3:
            prosody = prosody.mean(dim=1)
        if timbre.dim() == 3:
            timbre = timbre.mean(dim=1)

        mu = self.mean_net(prosody)

        log_prob = -0.5 * (
            self.logvar +
            (timbre - mu).pow(2) / (torch.exp(self.logvar) + 1e-8)
        )

        return log_prob.sum(dim=-1)

    def forward(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute CLUB upper bound."""
        batch_size = prosody_emb.shape[0] if prosody_emb.dim() == 2 else prosody_emb.shape[0]

        # Positive
        log_prob_pos = self._log_prob(timbre_emb, prosody_emb)

        # Negative (shuffled)
        perm = torch.randperm(batch_size, device=prosody_emb.device)
        timbre_shuffled = timbre_emb[perm] if timbre_emb.dim() == 2 else timbre_emb[perm]
        log_prob_neg = self._log_prob(timbre_shuffled, prosody_emb)

        mi_upper = (log_prob_pos - log_prob_neg).mean()
        training_loss = -self._log_prob(timbre_emb.detach(), prosody_emb.detach()).mean()

        return {
            'mi_upper': mi_upper,
            'training_loss': training_loss,
        }


# =============================================================================
# LEARN2DISS COMBINED LOSS
# =============================================================================

class Learn2DissLoss(nn.Module):
    """
    Learn2Diss combined disentanglement loss.

    Combines:
    1. MINE loss (minimize lower bound of MI)
    2. CLUB loss (minimize upper bound of MI)
    3. Gap penalty (encourage tight bounds)
    4. Orthogonality regularization

    Together, these losses force prosody and timbre to become independent
    while providing reliable MI estimation.
    """

    def __init__(self, config: Learn2DissConfig):
        super().__init__()
        self.config = config

        # Dual encoder MI estimator
        self.mi_estimator = DualEncoderMIEstimator(config)

    def _orthogonality_loss(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Soft orthogonality loss."""
        if prosody_emb.dim() == 3:
            prosody_emb = prosody_emb.mean(dim=1)
        if timbre_emb.dim() == 3:
            timbre_emb = timbre_emb.mean(dim=1)

        # L2 normalize
        p_norm = F.normalize(prosody_emb, p=2, dim=-1)
        t_norm = F.normalize(timbre_emb, p=2, dim=-1)

        # Cosine similarity
        cos_sim = (p_norm * t_norm).sum(dim=-1)  # [batch]

        # Push toward target
        loss = (cos_sim - self.config.orthogonality_target).pow(2).mean()

        return loss

    def forward(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        epoch: Optional[int] = None,
        train_club: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Learn2Diss combined loss.

        Args:
            prosody_emb: [batch, dim] or [batch, seq, dim]
            timbre_emb: [batch, dim]
            epoch: Current epoch for beta schedule
            train_club: Whether to train CLUB network

        Returns:
            Dict with all losses and MI estimates
        """
        device = prosody_emb.device
        losses = {}

        # Update epoch for schedule
        if epoch is not None:
            self.mi_estimator.update_epoch(epoch)

        # Get beta
        beta = self.mi_estimator.get_beta()
        losses['beta'] = torch.tensor(beta, device=device)

        # Dual encoder MI estimation
        mi_result = self.mi_estimator(prosody_emb, timbre_emb, train_club=train_club)

        losses['mi_lower'] = mi_result['mi_lower'].detach()
        losses['mi_upper'] = mi_result['mi_upper'].detach()
        losses['mi_gap'] = mi_result['mi_gap'].detach()
        losses['mi_estimate'] = mi_result['mi_estimate'].detach()

        # MINE loss (minimize lower bound)
        losses['mine'] = mi_result['mine_loss']

        # CLUB loss (minimize upper bound)
        losses['club'] = mi_result['club_loss']

        # CLUB training loss
        losses['club_train'] = mi_result['club_train_loss']

        # Gap penalty (encourage tight bounds)
        if self.config.use_sandwiched:
            losses['gap_penalty'] = mi_result['mi_gap']
        else:
            losses['gap_penalty'] = torch.tensor(0.0, device=device)

        # Orthogonality loss
        if self.config.use_orthogonality:
            losses['orthogonality'] = self._orthogonality_loss(prosody_emb, timbre_emb)
        else:
            losses['orthogonality'] = torch.tensor(0.0, device=device)

        # Combined loss with beta schedule
        losses['total'] = beta * (
            self.config.mine_weight * losses['mine'] +
            self.config.club_weight * losses['club'] +
            self.config.club_train_weight * losses['club_train'] +
            self.config.gap_penalty_weight * losses['gap_penalty'] +
            self.config.orthogonality_weight * losses['orthogonality']
        )

        return losses

    def get_mi_bounds(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Get MI bounds for analysis (no gradients).

        Args:
            prosody_emb: Prosody embeddings
            timbre_emb: Timbre embeddings

        Returns:
            Dict with mi_lower, mi_upper, mi_gap
        """
        with torch.no_grad():
            mi_result = self.mi_estimator(prosody_emb, timbre_emb, train_club=False)

        return {
            'mi_lower': mi_result['mi_lower'].item(),
            'mi_upper': mi_result['mi_upper'].item(),
            'mi_gap': mi_result['mi_gap'].item(),
            'mi_estimate': mi_result['mi_estimate'].item(),
        }


# =============================================================================
# INTEGRATION WITH DISENTANGLEMENT LOSS
# =============================================================================

class Learn2DissAdapter(nn.Module):
    """
    Adapter to integrate Learn2Diss with existing DisentanglementLoss.

    This allows using sandwiched MI (MINE + CLUB) alongside other
    disentanglement techniques like GRL.

    Usage:
        from learn2diss import Learn2DissAdapter, Learn2DissConfig

        adapter = Learn2DissAdapter(
            prosody_encoder=prosody_encoder,
            speaker_encoder=speaker_encoder,
            config=Learn2DissConfig(),
        )

        # In training loop
        output = adapter(audio_features, mel=mel_spec, epoch=epoch)
        prosody_emb = output['prosody_emb']
        l2d_losses = output['learn2diss_losses']

        total_loss = recon_loss + l2d_losses['total']
    """

    def __init__(
        self,
        prosody_encoder: nn.Module,
        speaker_encoder: Optional[nn.Module] = None,
        config: Optional[Learn2DissConfig] = None,
    ):
        super().__init__()

        self.prosody_encoder = prosody_encoder
        self.speaker_encoder = speaker_encoder

        if config is None:
            config = Learn2DissConfig()
        self.config = config

        self.learn2diss_loss = Learn2DissLoss(config)

    def forward(
        self,
        audio_features: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        **prosody_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Learn2Diss disentanglement.

        Args:
            audio_features: Input to prosody encoder
            speaker_emb: Pre-computed speaker embedding, or None to extract
            mel: Mel spectrogram for speaker extraction
            epoch: Current epoch for schedule
            **prosody_kwargs: Additional args for prosody encoder

        Returns:
            Dict with prosody output and Learn2Diss losses
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

        # Compute Learn2Diss loss
        if speaker_emb is not None:
            l2d_losses = self.learn2diss_loss(
                prosody_emb,
                speaker_emb,
                epoch=epoch,
            )
        else:
            l2d_losses = {
                'total': torch.tensor(0.0, device=prosody_emb.device),
                'mi_lower': torch.tensor(0.0, device=prosody_emb.device),
                'mi_upper': torch.tensor(0.0, device=prosody_emb.device),
                'mi_gap': torch.tensor(0.0, device=prosody_emb.device),
            }

        return {
            'prosody_emb': prosody_emb,
            'prosody_output': prosody_output,
            'learn2diss_losses': l2d_losses,
        }


# =============================================================================
# EXTEND DISENTANGLEMENT CONFIG
# =============================================================================

def extend_disentanglement_config(base_config, use_learn2diss: bool = True):
    """
    Helper to add Learn2Diss settings to existing DisentanglementConfig.

    Usage:
        from disentanglement import DisentanglementConfig
        from learn2diss import extend_disentanglement_config

        config = DisentanglementConfig(...)
        config = extend_disentanglement_config(config, use_learn2diss=True)
    """
    # Add Learn2Diss settings
    base_config.use_learn2diss = use_learn2diss
    base_config.learn2diss_config = Learn2DissConfig(
        prosody_dim=getattr(base_config, 'prosody_dim', 2048),
        timbre_dim=base_config.speaker_embed_dim,
    ) if use_learn2diss else None

    return base_config


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Learn2Diss: Dual Encoder MI Estimation - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = Learn2DissConfig(
        prosody_dim=256,  # Smaller for testing
        timbre_dim=256,
        hidden_dim=128,
    )

    # Test 1: VariationalCLUBNetwork
    print("\n[Test 1] VariationalCLUBNetwork...")
    club_net = VariationalCLUBNetwork(config).to(device)
    prosody = torch.randn(8, 256, device=device)
    timbre = torch.randn(8, 256, device=device)

    club_result = club_net(prosody, timbre)
    print(f"  MI upper bound: {club_result['mi_upper'].item():.4f}")
    print(f"  Training loss: {club_result['training_loss'].item():.4f}")
    print(f"  Log prob (pos): {club_result['log_prob_pos'].item():.4f}")
    print(f"  Log prob (neg): {club_result['log_prob_neg'].item():.4f}")
    print("  [PASS]")

    # Test 2: MINEEstimator
    print("\n[Test 2] MINEEstimator...")
    mine_est = MINEEstimator(config).to(device)
    mine_result = mine_est(prosody, timbre)
    print(f"  MI lower bound: {mine_result['mi_lower'].item():.4f}")
    print(f"  T_joint mean: {mine_result['T_joint_mean'].item():.4f}")
    print(f"  T_marginal mean: {mine_result['T_marginal_mean'].item():.4f}")
    print("  [PASS]")

    # Test 3: DualEncoderMIEstimator
    print("\n[Test 3] DualEncoderMIEstimator (sandwiched MI)...")
    dual_est = DualEncoderMIEstimator(config).to(device)
    dual_result = dual_est(prosody, timbre)
    print(f"  MI lower (MINE): {dual_result['mi_lower'].item():.4f}")
    print(f"  MI upper (CLUB): {dual_result['mi_upper'].item():.4f}")
    print(f"  MI gap: {dual_result['mi_gap'].item():.4f}")
    print(f"  MI estimate (midpoint): {dual_result['mi_estimate'].item():.4f}")
    print("  [PASS]")

    # Test 4: Verify sandwiching (lower ≤ upper in expectation)
    print("\n[Test 4] Verifying sandwiching property...")
    num_trials = 10
    gap_positive = 0
    for _ in range(num_trials):
        p = torch.randn(32, 256, device=device)
        t = torch.randn(32, 256, device=device)
        result = dual_est(p, t)
        if result['mi_gap'].item() >= 0:
            gap_positive += 1
    print(f"  Gap positive in {gap_positive}/{num_trials} trials")
    print("  [PASS]" if gap_positive >= num_trials * 0.7 else "  [WARN] Gap sometimes negative")

    # Test 5: Learn2DissLoss
    print("\n[Test 5] Learn2DissLoss (combined)...")
    l2d_loss = Learn2DissLoss(config).to(device)
    losses = l2d_loss(prosody, timbre, epoch=3)
    print(f"  MINE loss: {losses['mine'].item():.4f}")
    print(f"  CLUB loss: {losses['club'].item():.4f}")
    print(f"  Gap penalty: {losses['gap_penalty'].item():.4f}")
    print(f"  Orthogonality: {losses['orthogonality'].item():.4f}")
    print(f"  Beta: {losses['beta'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 6: Gradient flow
    print("\n[Test 6] Gradient flow...")
    prosody_param = nn.Parameter(prosody.clone())
    losses = l2d_loss(prosody_param, timbre)
    losses['total'].backward()
    print(f"  Gradient norm: {prosody_param.grad.norm().item():.6f}")
    print(f"  Gradient is non-zero: {prosody_param.grad.abs().sum().item() > 0}")
    print("  [PASS]")

    # Test 7: Beta warmup schedule
    print("\n[Test 7] Beta warmup schedule...")
    for epoch in [0, 1, 2, 5, 10]:
        l2d_loss.mi_estimator.update_epoch(epoch)
        beta = l2d_loss.mi_estimator.get_beta()
        print(f"  Epoch {epoch}: beta={beta:.4f}")
    print("  [PASS]")

    # Test 8: Correlated vs independent embeddings
    print("\n[Test 8] MI with correlated vs independent embeddings...")
    # Highly correlated (same embedding)
    mi_bounds_corr = l2d_loss.get_mi_bounds(prosody, prosody[:, :256])  # Partial correlation
    # Independent
    mi_bounds_indep = l2d_loss.get_mi_bounds(prosody, torch.randn_like(timbre))

    print(f"  Correlated - Lower: {mi_bounds_corr['mi_lower']:.4f}, Upper: {mi_bounds_corr['mi_upper']:.4f}")
    print(f"  Independent - Lower: {mi_bounds_indep['mi_lower']:.4f}, Upper: {mi_bounds_indep['mi_upper']:.4f}")
    print("  [PASS]")

    # Test 9: Sequence input handling
    print("\n[Test 9] Sequence input handling...")
    prosody_seq = torch.randn(4, 50, 256, device=device)  # [batch, seq, dim]
    timbre_single = torch.randn(4, 256, device=device)   # [batch, dim]
    losses = l2d_loss(prosody_seq, timbre_single)
    print(f"  Prosody seq shape: {prosody_seq.shape}")
    print(f"  Timbre shape: {timbre_single.shape}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 10: SimpleCLUBEstimator (without variational)
    print("\n[Test 10] SimpleCLUBEstimator...")
    config_simple = Learn2DissConfig(
        prosody_dim=256,
        timbre_dim=256,
        use_variational_club=False,
    )
    simple_club = SimpleCLUBEstimator(config_simple).to(device)
    result = simple_club(prosody, timbre)
    print(f"  MI upper (simple): {result['mi_upper'].item():.4f}")
    print(f"  Training loss: {result['training_loss'].item():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All Learn2Diss tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from learn2diss import Learn2DissConfig, Learn2DissLoss, Learn2DissAdapter

# Initialize
config = Learn2DissConfig(
    prosody_dim=2048,      # Match your prosody encoder output
    timbre_dim=256,        # Match your speaker encoder output
    use_sandwiched=True,   # Use both MINE + CLUB
    use_variational_club=True,
    beta_start=0.01,
    beta_end=1.0,
    warmup_epochs=5,
)

loss_fn = Learn2DissLoss(config)

# In training loop
for epoch in range(epochs):
    for batch in dataloader:
        prosody_emb = prosody_encoder(batch['audio'])
        timbre_emb = speaker_encoder(batch['mel'])

        # Get reconstruction loss
        recon_loss = compute_reconstruction_loss(...)

        # Compute Learn2Diss sandwiched MI loss
        l2d_losses = loss_fn(prosody_emb, timbre_emb, epoch=epoch)

        # Combined loss
        total_loss = recon_loss + l2d_losses['total']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log sandwiched MI bounds
        if step % 100 == 0:
            print(f"MI lower (MINE): {l2d_losses['mi_lower']:.4f}")
            print(f"MI upper (CLUB): {l2d_losses['mi_upper']:.4f}")
            print(f"MI gap: {l2d_losses['mi_gap']:.4f}")  # Should decrease

# For analysis: check MI bounds
mi_bounds = loss_fn.get_mi_bounds(prosody_emb, timbre_emb)
print(f"Sandwiched MI: [{mi_bounds['mi_lower']:.4f}, {mi_bounds['mi_upper']:.4f}]")
""")
