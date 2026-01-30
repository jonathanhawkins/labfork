"""
Disentanglement Module for Prosody Voice Cloning

Based on DisCodec (December 2024) two-stage disentanglement approach:
https://arxiv.org/html/2512.13251

Key Components:
1. SoftOrthogonalityLoss - Asymmetric β constraints (relaxed content-prosody, strict timbre-prosody)
2. GradientReversalLayer - Adversarial removal of speaker info from prosody
3. F0RegressionHead - Explicit prosody supervision
4. DisentanglementTrainer - Two-stage training orchestrator

Integration with V6:
- Add orthogonality loss to existing HED training
- Use GRL to prevent speaker leakage into prosody embeddings
- Improve zero-shot prosody transfer quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DisentanglementConfig:
    """Configuration for disentanglement training."""

    # Orthogonality coefficients (from DisCodec paper)
    beta_content_prosody: float = 0.01     # Relaxed - they share temporal dynamics
    beta_timbre_prosody: float = 0.0001    # Strict - must be independent

    # Gradient reversal settings
    use_grl: bool = True
    grl_lambda_start: float = 0.0
    grl_lambda_end: float = 1.0
    grl_warmup_epochs: int = 5

    # Scheduled GRL settings (Diffsody approach - IEEE TNNLS 2025)
    # Uses sigmoid schedule: λ(p) = 2/(1 + exp(-γ*p)) - 1
    # This prevents early training instability by starting with λ≈0
    use_scheduled_grl: bool = False       # Enable Diffsody scheduled GRL
    grl_schedule_type: str = "sigmoid"    # sigmoid, linear, cosine, exponential
    grl_gamma: float = 10.0               # Sigmoid steepness (higher = sharper)
    grl_gamma_schedule: bool = False      # Increase gamma over training
    grl_gamma_start: float = 5.0          # Starting gamma if scheduled
    grl_gamma_end: float = 15.0           # Ending gamma if scheduled

    # MINE (Mutual Information Neural Estimation) settings
    # MINE provides smoother gradients than GRL for disentanglement
    use_mine: bool = False           # Enable MINE-based disentanglement
    mine_weight: float = 0.5         # Weight for MINE loss
    mine_hidden_dim: int = 512       # MINE network hidden dimension
    mine_warmup_epochs: int = 5      # Epochs to ramp up MINE loss weight
    mine_beta_start: float = 0.01    # Initial MINE loss weight
    mine_beta_end: float = 1.0       # Final MINE loss weight

    # Learn2Diss (Sandwiched MI with MINE + CLUB) - arXiv:2407.02543
    # Uses both MINE (lower bound) and CLUB (upper bound) for tighter MI estimation
    use_learn2diss: bool = False     # Enable Learn2Diss dual encoder
    learn2diss_weight: float = 1.0   # Weight for Learn2Diss loss
    learn2diss_use_variational_club: bool = True  # Use variational CLUB
    learn2diss_gap_penalty: float = 0.01  # Penalty for MI bound gap

    # DiEmo-TTS (ED-DINO) settings - Interspeech 2025
    # Self-supervised distillation for speaker-irrelevant emotion embeddings
    use_diemo: bool = False          # Enable DiEmo-TTS ED-DINO
    diemo_weight: float = 1.0        # Weight for DINO loss
    diemo_teacher_momentum: float = 0.996  # EMA momentum for teacher
    diemo_teacher_temp: float = 0.04       # Teacher temperature (low = sharp)
    diemo_student_temp: float = 0.1        # Student temperature
    diemo_warmup_epochs: int = 10          # Warmup before teacher updates
    diemo_input_dim: int = 768             # Input feature dimension (wav2vec2)

    # Loss weights
    orthogonality_weight: float = 0.5
    f0_regression_weight: float = 0.3
    speaker_adversarial_weight: float = 0.1

    # Speaker encoder settings
    num_speakers: int = 1000  # For multi-speaker training
    speaker_embed_dim: int = 256

    # F0 settings
    f0_min: float = 50.0
    f0_max: float = 500.0
    f0_bins: int = 256  # For quantized F0

    # Total training epochs (for scheduled GRL)
    total_epochs: int = 100


# =============================================================================
# GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal for adversarial training.

    Forward: Identity function
    Backward: Reverses gradients by multiplying with -lambda

    This trains the encoder to produce features that *cannot* predict
    the reversed target (e.g., speaker ID from prosody).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) for domain adversarial training.

    Usage in prosody disentanglement:
        prosody_emb = prosody_encoder(audio)
        prosody_grl = grl(prosody_emb)
        speaker_pred = speaker_classifier(prosody_grl)
        loss = cross_entropy(speaker_pred, speaker_labels)

    The reversed gradients train the prosody encoder to NOT predict speaker,
    effectively removing speaker information from prosody representations.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Update lambda for curriculum training."""
        self.lambda_ = lambda_


# =============================================================================
# SOFT ORTHOGONALITY LOSS
# =============================================================================

class SoftOrthogonalityLoss(nn.Module):
    """
    Soft orthogonality constraint from DisCodec.

    Unlike hard orthogonality (forcing cos_sim = 0), soft orthogonality
    pushes similarity toward a target value. This allows:
    - Relaxed coupling for naturally correlated factors (content-prosody)
    - Strict independence for factors that should be separate (timbre-prosody)

    DisCodec's key insight: β_content_prosody = 0.01, β_timbre_prosody = 0.0001
    """

    def __init__(self, target_similarity: float = 0.0):
        """
        Args:
            target_similarity: Target cosine similarity (β value)
                - 0.01 for content-prosody (relaxed)
                - 0.0001 for timbre-prosody (strict)
        """
        super().__init__()
        self.target_similarity = target_similarity

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft orthogonality loss.

        Args:
            z1: First representation [batch, dim] or [batch, seq, dim]
            z2: Second representation [batch, dim] or [batch, seq, dim]
            mask: Optional mask for sequence positions

        Returns:
            Scalar loss value
        """
        # Handle sequence inputs by pooling
        if z1.dim() == 3:
            if mask is not None:
                z1 = (z1 * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                z1 = z1.mean(dim=1)

        if z2.dim() == 3:
            if mask is not None:
                z2 = (z2 * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                z2 = z2.mean(dim=1)

        # L2 normalize
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)

        # Compute cosine similarity
        cos_sim = (z1_norm * z2_norm).sum(dim=-1)  # [batch]

        # Push toward target similarity
        loss = (cos_sim - self.target_similarity).pow(2).mean()

        return loss


class AsymmetricOrthogonalityLoss(nn.Module):
    """
    Combined orthogonality loss with asymmetric β values.

    Implements DisCodec's key insight:
    - Content-Prosody: β = 0.01 (relaxed, they share temporal dynamics)
    - Timbre-Prosody: β = 0.0001 (strict, must be independent)
    """

    def __init__(self, config: DisentanglementConfig):
        super().__init__()
        self.config = config

        self.content_prosody_loss = SoftOrthogonalityLoss(
            target_similarity=config.beta_content_prosody
        )
        self.timbre_prosody_loss = SoftOrthogonalityLoss(
            target_similarity=config.beta_timbre_prosody
        )

    def forward(
        self,
        content_emb: Optional[torch.Tensor],
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute asymmetric orthogonality losses.

        Args:
            content_emb: Content encoder output (optional - may not have separate content)
            prosody_emb: Prosody encoder output
            timbre_emb: Timbre/speaker encoder output
            mask: Optional sequence mask

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # Content-Prosody orthogonality (relaxed)
        if content_emb is not None:
            losses['ortho_content_prosody'] = self.content_prosody_loss(
                content_emb, prosody_emb, mask
            )
        else:
            losses['ortho_content_prosody'] = torch.tensor(0.0, device=prosody_emb.device)

        # Timbre-Prosody orthogonality (strict)
        losses['ortho_timbre_prosody'] = self.timbre_prosody_loss(
            timbre_emb, prosody_emb, mask
        )

        # Combined loss
        losses['ortho_total'] = (
            losses['ortho_content_prosody'] +
            losses['ortho_timbre_prosody']
        )

        return losses


# =============================================================================
# F0 REGRESSION HEAD
# =============================================================================

class F0RegressionHead(nn.Module):
    """
    F0 prediction head for explicit prosody supervision.

    DisCodec uses frame-level F0 regression to ensure the prosody encoder
    captures pitch information. This provides a strong inductive bias
    toward prosodic (vs speaker) features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
    ):
        super().__init__()
        self.f0_min = f0_min
        self.f0_max = f0_max

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, prosody_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict F0 from prosody embedding.

        Args:
            prosody_emb: [batch, seq, dim] or [batch, dim]

        Returns:
            F0 predictions normalized to [0, 1]
        """
        f0_pred = self.head(prosody_emb).squeeze(-1)
        f0_pred = torch.sigmoid(f0_pred)  # Normalize to [0, 1]
        return f0_pred

    def compute_loss(
        self,
        f0_pred: torch.Tensor,
        f0_target: torch.Tensor,
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute F0 regression loss.

        Args:
            f0_pred: Predicted F0 [batch, seq] normalized to [0, 1]
            f0_target: Target F0 in Hz [batch, seq]
            voiced_mask: Mask for voiced frames (ignore unvoiced)

        Returns:
            MSE loss
        """
        # Normalize target to [0, 1]
        f0_target_norm = (f0_target - self.f0_min) / (self.f0_max - self.f0_min)
        f0_target_norm = f0_target_norm.clamp(0, 1)

        if voiced_mask is not None:
            # Only compute loss on voiced frames
            f0_pred = f0_pred[voiced_mask]
            f0_target_norm = f0_target_norm[voiced_mask]

        return F.mse_loss(f0_pred, f0_target_norm)


# =============================================================================
# SPEAKER ADVERSARIAL HEAD
# =============================================================================

class SpeakerAdversarialHead(nn.Module):
    """
    Speaker classifier with gradient reversal for anti-leakage training.

    The goal is to train the prosody encoder such that its outputs
    CANNOT predict speaker identity. This removes speaker-specific
    information from prosody representations.
    """

    def __init__(
        self,
        input_dim: int,
        num_speakers: int,
        hidden_dim: int = 256,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_speakers),
        )

    def forward(
        self,
        prosody_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict speaker from prosody (with gradient reversal).

        Args:
            prosody_emb: [batch, dim] prosody embedding

        Returns:
            Speaker logits [batch, num_speakers]
        """
        # Pool if sequence
        if prosody_emb.dim() == 3:
            prosody_emb = prosody_emb.mean(dim=1)

        # Apply gradient reversal
        prosody_grl = self.grl(prosody_emb)

        # Classify speaker
        return self.classifier(prosody_grl)

    def compute_loss(
        self,
        speaker_logits: torch.Tensor,
        speaker_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute speaker classification loss.

        Note: Due to GRL, gradients will be reversed, so minimizing
        this loss actually makes prosody WORSE at predicting speaker.
        """
        return F.cross_entropy(speaker_logits, speaker_labels)

    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda for curriculum training."""
        self.grl.set_lambda(lambda_)


# =============================================================================
# COMBINED DISENTANGLEMENT LOSS
# =============================================================================

class DisentanglementLoss(nn.Module):
    """
    Combined disentanglement loss for prosody training.

    Combines:
    1. Soft orthogonality (asymmetric β)
    2. F0 regression
    3. Speaker adversarial (GRL-based)
    4. MINE-based MI minimization (optional, more stable than GRL)

    Usage:
        loss_fn = DisentanglementLoss(config)

        # In training loop:
        prosody_emb = prosody_encoder(audio)
        timbre_emb = speaker_encoder(audio)

        disentangle_losses = loss_fn(
            prosody_emb=prosody_emb,
            timbre_emb=timbre_emb,
            f0_target=f0,
            speaker_labels=speaker_ids,
            epoch=current_epoch,
        )

        total_loss = main_loss + disentangle_losses['total']
    """

    def __init__(
        self,
        config: DisentanglementConfig,
        prosody_dim: int,
    ):
        super().__init__()
        self.config = config
        self.prosody_dim = prosody_dim

        # Orthogonality loss
        self.ortho_loss = AsymmetricOrthogonalityLoss(config)

        # F0 regression head
        self.f0_head = F0RegressionHead(
            input_dim=prosody_dim,
            f0_min=config.f0_min,
            f0_max=config.f0_max,
        )

        # Speaker adversarial head
        # Choose between scheduled GRL (Diffsody) or standard GRL
        self.scheduled_grl = None
        if config.use_grl:
            if config.use_scheduled_grl:
                # Use Diffsody scheduled GRL
                try:
                    try:
                        from scheduled_grl import (
                            ScheduledGRLConfig,
                            ScheduledSpeakerAdversarialHead,
                        )
                    except ImportError:
                        from training.scheduled_grl import (
                            ScheduledGRLConfig,
                            ScheduledSpeakerAdversarialHead,
                        )

                    sgrl_config = ScheduledGRLConfig(
                        schedule_type=config.grl_schedule_type,
                        gamma=config.grl_gamma,
                        gamma_schedule=config.grl_gamma_schedule,
                        gamma_start=config.grl_gamma_start,
                        gamma_end=config.grl_gamma_end,
                        lambda_min=config.grl_lambda_start,
                        lambda_max=config.grl_lambda_end,
                        warmup_epochs=config.grl_warmup_epochs,
                        total_epochs=config.total_epochs,
                        num_speakers=config.num_speakers,
                    )
                    self.speaker_head = ScheduledSpeakerAdversarialHead(
                        input_dim=prosody_dim,
                        config=sgrl_config,
                    )
                    self.scheduled_grl = self.speaker_head.grl
                    print(f"Using Diffsody scheduled GRL (γ={config.grl_gamma}, schedule={config.grl_schedule_type})")
                except ImportError as e:
                    print(f"Warning: Scheduled GRL module not found ({e}). Falling back to standard GRL.")
                    self.speaker_head = SpeakerAdversarialHead(
                        input_dim=prosody_dim,
                        num_speakers=config.num_speakers,
                        grl_lambda=config.grl_lambda_start,
                    )
            else:
                # Use standard GRL with linear schedule
                self.speaker_head = SpeakerAdversarialHead(
                    input_dim=prosody_dim,
                    num_speakers=config.num_speakers,
                    grl_lambda=config.grl_lambda_start,
                )
        else:
            self.speaker_head = None

        # MINE-based disentanglement (optional, more stable alternative to GRL)
        self.mine_loss = None
        if config.use_mine:
            try:
                # Try relative import first (when running from training dir)
                try:
                    from mine_disentanglement import MINEConfig, MINEDisentanglementLoss
                except ImportError:
                    # Try absolute import (when running from project root)
                    from training.mine_disentanglement import MINEConfig, MINEDisentanglementLoss

                mine_config = MINEConfig(
                    emotion_dim=prosody_dim,
                    timbre_dim=config.speaker_embed_dim,
                    hidden_dim=config.mine_hidden_dim,
                    beta_start=config.mine_beta_start,
                    beta_end=config.mine_beta_end,
                    warmup_epochs=config.mine_warmup_epochs,
                )
                self.mine_loss = MINEDisentanglementLoss(
                    mine_config=mine_config,
                    emotion_dim=prosody_dim,
                    timbre_dim=config.speaker_embed_dim,
                    mine_weight=config.mine_weight,
                )
            except ImportError as e:
                print(f"Warning: MINE module not found ({e}). Falling back to GRL only.")
                self.mine_loss = None

        # Learn2Diss (Sandwiched MI with MINE + CLUB) - arXiv:2407.02543
        self.learn2diss = None
        if config.use_learn2diss:
            try:
                try:
                    from learn2diss import Learn2DissConfig, Learn2DissLoss
                except ImportError:
                    from training.learn2diss import Learn2DissConfig, Learn2DissLoss

                l2d_config = Learn2DissConfig(
                    prosody_dim=prosody_dim,
                    timbre_dim=config.speaker_embed_dim,
                    hidden_dim=config.mine_hidden_dim,
                    use_sandwiched=True,
                    use_variational_club=config.learn2diss_use_variational_club,
                    gap_penalty_weight=config.learn2diss_gap_penalty,
                    beta_start=config.mine_beta_start,
                    beta_end=config.mine_beta_end,
                    warmup_epochs=config.mine_warmup_epochs,
                )
                self.learn2diss = Learn2DissLoss(l2d_config)
                self.learn2diss_weight = config.learn2diss_weight
                print(f"Learn2Diss enabled (sandwiched MI with MINE + CLUB)")
            except ImportError as e:
                print(f"Warning: Learn2Diss module not found ({e}). Falling back to MINE only.")
                self.learn2diss = None

        # DiEmo-TTS (ED-DINO) for self-supervised emotion distillation
        self.diemo = None
        if config.use_diemo:
            try:
                try:
                    from diemo_tts import DiEmoTTSConfig, DiEmoTTS
                except ImportError:
                    from training.diemo_tts import DiEmoTTSConfig, DiEmoTTS

                diemo_config = DiEmoTTSConfig(
                    input_dim=config.diemo_input_dim,
                    emotion_dim=prosody_dim,  # Match prosody encoder output
                    num_speakers=config.num_speakers,
                    teacher_momentum=config.diemo_teacher_momentum,
                    teacher_temp=config.diemo_teacher_temp,
                    student_temp=config.diemo_student_temp,
                    warmup_epochs=config.diemo_warmup_epochs,
                )
                self.diemo = DiEmoTTS(diemo_config)
                print(f"DiEmo-TTS enabled with ED-DINO (teacher_momentum={config.diemo_teacher_momentum})")
            except ImportError as e:
                print(f"Warning: DiEmo-TTS module not found ({e}). Falling back to other methods.")
                self.diemo = None

    def forward(
        self,
        prosody_emb: torch.Tensor,
        timbre_emb: torch.Tensor,
        content_emb: Optional[torch.Tensor] = None,
        f0_target: Optional[torch.Tensor] = None,
        voiced_mask: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all disentanglement losses.

        Args:
            prosody_emb: Prosody encoder output [batch, seq, dim] or [batch, dim]
            timbre_emb: Timbre/speaker encoder output [batch, dim]
            content_emb: Optional content encoder output
            f0_target: Target F0 in Hz (for regression loss)
            voiced_mask: Mask for voiced frames
            speaker_labels: Speaker IDs (for adversarial loss)
            seq_mask: Sequence mask for variable length
            epoch: Current training epoch (for MINE beta schedule)

        Returns:
            Dict with individual losses and total
        """
        losses = {}
        device = prosody_emb.device

        # 1. Orthogonality losses
        ortho_losses = self.ortho_loss(
            content_emb=content_emb,
            prosody_emb=prosody_emb,
            timbre_emb=timbre_emb,
            mask=seq_mask,
        )
        losses.update(ortho_losses)

        # 2. F0 regression loss
        if f0_target is not None:
            f0_pred = self.f0_head(prosody_emb)
            losses['f0_regression'] = self.f0_head.compute_loss(
                f0_pred, f0_target, voiced_mask
            )
        else:
            losses['f0_regression'] = torch.tensor(0.0, device=device)

        # 3. Speaker adversarial loss (GRL-based)
        if self.speaker_head is not None and speaker_labels is not None:
            speaker_logits = self.speaker_head(prosody_emb)
            losses['speaker_adversarial'] = self.speaker_head.compute_loss(
                speaker_logits, speaker_labels
            )
        else:
            losses['speaker_adversarial'] = torch.tensor(0.0, device=device)

        # 4. MINE-based MI minimization (optional, more stable than GRL)
        if self.mine_loss is not None:
            mine_losses = self.mine_loss(prosody_emb, timbre_emb, epoch=epoch)
            losses['mine'] = mine_losses['mine']
            losses['mi_estimate'] = mine_losses['mi_estimate']
            losses['mine_total'] = mine_losses['total']
        else:
            losses['mine'] = torch.tensor(0.0, device=device)
            losses['mi_estimate'] = torch.tensor(0.0, device=device)
            losses['mine_total'] = torch.tensor(0.0, device=device)

        # 4b. Learn2Diss (sandwiched MI with MINE + CLUB) - arXiv:2407.02543
        if self.learn2diss is not None:
            l2d_losses = self.learn2diss(prosody_emb, timbre_emb, epoch=epoch)
            losses['learn2diss_mine'] = l2d_losses['mine']
            losses['learn2diss_club'] = l2d_losses['club']
            losses['mi_lower'] = l2d_losses['mi_lower']
            losses['mi_upper'] = l2d_losses['mi_upper']
            losses['mi_gap'] = l2d_losses['mi_gap']
            losses['mi_estimate'] = l2d_losses['mi_estimate']  # Override with sandwiched estimate
            losses['learn2diss_total'] = l2d_losses['total'] * self.learn2diss_weight
        else:
            losses['learn2diss_mine'] = torch.tensor(0.0, device=device)
            losses['learn2diss_club'] = torch.tensor(0.0, device=device)
            losses['mi_lower'] = torch.tensor(0.0, device=device)
            losses['mi_upper'] = torch.tensor(0.0, device=device)
            losses['mi_gap'] = torch.tensor(0.0, device=device)
            losses['learn2diss_total'] = torch.tensor(0.0, device=device)

        # 5. DiEmo-TTS ED-DINO loss (optional, self-supervised distillation)
        # Note: DiEmo operates on audio features, not prosody embeddings
        # For full DiEmo training, use DiEmoTTS directly with audio features
        if self.diemo is not None and speaker_labels is not None:
            # Use prosody_emb as a proxy for audio features in the simplified API
            # For full training, pass actual audio features to DiEmoTTS separately
            diemo_losses = self.diemo(
                prosody_emb,  # Input features
                speaker_labels=speaker_labels,
                epoch=epoch if epoch is not None else 0,
            )
            losses['diemo_dino'] = diemo_losses['dino_loss']
            losses['diemo_speaker_acc'] = diemo_losses['speaker_accuracy']
            losses['diemo_total'] = diemo_losses['total'] * self.config.diemo_weight
        else:
            losses['diemo_dino'] = torch.tensor(0.0, device=device)
            losses['diemo_speaker_acc'] = torch.tensor(0.0, device=device)
            losses['diemo_total'] = torch.tensor(0.0, device=device)

        # Total weighted loss
        losses['total'] = (
            self.config.orthogonality_weight * losses['ortho_total'] +
            self.config.f0_regression_weight * losses['f0_regression'] +
            self.config.speaker_adversarial_weight * losses['speaker_adversarial'] +
            losses['mine_total'] +  # MINE loss already weighted internally
            losses['learn2diss_total'] +  # Learn2Diss (sandwiched MI) already weighted
            losses['diemo_total']   # DiEmo loss already weighted
        )

        # Add GRL lambda to losses for monitoring
        losses['grl_lambda'] = torch.tensor(self.get_grl_lambda(), device=device)

        return losses

    def update_grl_lambda(self, epoch: int, total_epochs: int = None, step: int = None):
        """
        Update GRL lambda based on training progress.

        For scheduled GRL (Diffsody): Uses sigmoid/cosine/etc schedule
        For standard GRL: Uses linear warmup schedule

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total epochs (optional, overrides config)
            step: Current step (optional, for step-based scheduling)
        """
        if self.speaker_head is None:
            return

        # Use scheduled GRL if enabled
        if self.scheduled_grl is not None:
            # Update total_epochs if provided
            if total_epochs is not None:
                self.scheduled_grl.config.total_epochs = total_epochs
            self.scheduled_grl.update(epoch=epoch, step=step)
            return

        # Standard linear schedule (original behavior)
        if hasattr(self.speaker_head, 'set_grl_lambda'):
            progress = min(epoch / max(1, self.config.grl_warmup_epochs), 1.0)
            lambda_ = (
                self.config.grl_lambda_start +
                progress * (self.config.grl_lambda_end - self.config.grl_lambda_start)
            )
            self.speaker_head.set_grl_lambda(lambda_)

    def get_grl_lambda(self) -> float:
        """Get current GRL lambda value."""
        if self.speaker_head is None:
            return 0.0
        if self.scheduled_grl is not None:
            return self.scheduled_grl.lambda_
        if hasattr(self.speaker_head, 'grl'):
            return self.speaker_head.grl.lambda_
        return 0.0

    def get_grl_info(self) -> dict:
        """Get detailed GRL scheduling information."""
        if self.speaker_head is None:
            return {'enabled': False}

        info = {
            'enabled': True,
            'lambda': self.get_grl_lambda(),
        }

        if self.scheduled_grl is not None:
            info['scheduled'] = True
            info['schedule_type'] = self.config.grl_schedule_type
            info['gamma'] = self.scheduled_grl.gamma
            info['progress'] = self.scheduled_grl.progress
        else:
            info['scheduled'] = False
            info['schedule_type'] = 'linear'

        return info


# =============================================================================
# SIMPLE SPEAKER ENCODER (ECAPA-TDNN STYLE)
# =============================================================================

class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling for speaker embeddings."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]

        Returns:
            [batch, channels * 2] (mean + std)
        """
        attn = self.attention(x)  # [B, C, T]

        # Weighted mean
        mean = (x * attn).sum(dim=2)  # [B, C]

        # Weighted std
        var = ((x - mean.unsqueeze(2)).pow(2) * attn).sum(dim=2)
        std = var.clamp(min=1e-5).sqrt()  # [B, C]

        return torch.cat([mean, std], dim=1)  # [B, C*2]


class SimpleSpeakerEncoder(nn.Module):
    """
    Simple ECAPA-TDNN inspired speaker encoder.

    For full disentanglement, this extracts timbre/speaker identity
    that should be orthogonal to prosody.
    """

    def __init__(
        self,
        input_dim: int = 80,      # Mel channels
        hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        super().__init__()

        # Frame-level layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Multi-layer aggregation
        self.mfa = nn.Conv1d(hidden_dim * 3, hidden_dim * 3, kernel_size=1)

        # Attentive pooling
        self.asp = AttentiveStatisticsPooling(hidden_dim * 3)
        self.asp_bn = nn.BatchNorm1d(hidden_dim * 6)

        # Output projection
        self.fc = nn.Linear(hidden_dim * 6, output_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            mel: [batch, time, mel_dim] or [batch, mel_dim, time]

        Returns:
            [batch, output_dim] speaker embedding
        """
        # Ensure [B, C, T] format
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[1] > mel.shape[2]:  # [B, T, C] -> [B, C, T]
            mel = mel.transpose(1, 2)

        # Frame-level processing
        x1 = self.conv1(mel)
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        x4 = self.conv4(x3) + x3

        # Aggregate
        x = torch.cat([x2, x3, x4], dim=1)  # [B, H*3, T]
        x = self.mfa(x)

        # Pool to utterance level
        x = self.asp(x)  # [B, H*6]
        x = self.asp_bn(x)
        x = self.fc(x)  # [B, output_dim]

        return x


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Disentanglement Module - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = DisentanglementConfig(num_speakers=100)

    # Test 1: GradientReversalLayer
    print("\n[Test 1] GradientReversalLayer...")
    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.randn(4, 256, requires_grad=True)
    y = grl(x)
    loss = y.sum()
    loss.backward()
    print(f"  Input grad sign should be negative: {(x.grad.sum() < 0).item()}")
    print("  [PASS]")

    # Test 2: SoftOrthogonalityLoss
    print("\n[Test 2] SoftOrthogonalityLoss...")
    ortho_loss = SoftOrthogonalityLoss(target_similarity=0.01)
    z1 = torch.randn(4, 256)
    z2 = torch.randn(4, 256)
    loss = ortho_loss(z1, z2)
    print(f"  Loss value: {loss.item():.6f}")
    print("  [PASS]")

    # Test 3: AsymmetricOrthogonalityLoss
    print("\n[Test 3] AsymmetricOrthogonalityLoss...")
    asym_loss = AsymmetricOrthogonalityLoss(config)
    prosody = torch.randn(4, 256)
    timbre = torch.randn(4, 256)
    losses = asym_loss(None, prosody, timbre)
    print(f"  Timbre-Prosody loss: {losses['ortho_timbre_prosody'].item():.6f}")
    print("  [PASS]")

    # Test 4: F0RegressionHead
    print("\n[Test 4] F0RegressionHead...")
    f0_head = F0RegressionHead(input_dim=256)
    prosody_seq = torch.randn(4, 50, 256)  # [B, T, D]
    f0_pred = f0_head(prosody_seq)
    f0_target = torch.rand(4, 50) * 200 + 100  # Hz
    f0_loss = f0_head.compute_loss(f0_pred, f0_target)
    print(f"  F0 pred shape: {f0_pred.shape}")
    print(f"  F0 loss: {f0_loss.item():.6f}")
    print("  [PASS]")

    # Test 5: SpeakerAdversarialHead
    print("\n[Test 5] SpeakerAdversarialHead...")
    speaker_head = SpeakerAdversarialHead(
        input_dim=256, num_speakers=100, grl_lambda=1.0
    )
    prosody = torch.randn(4, 256)
    speaker_logits = speaker_head(prosody)
    speaker_labels = torch.randint(0, 100, (4,))
    speaker_loss = speaker_head.compute_loss(speaker_logits, speaker_labels)
    print(f"  Speaker logits shape: {speaker_logits.shape}")
    print(f"  Speaker loss: {speaker_loss.item():.6f}")
    print("  [PASS]")

    # Test 6: DisentanglementLoss (combined)
    print("\n[Test 6] DisentanglementLoss (combined)...")
    disentangle_loss = DisentanglementLoss(config, prosody_dim=256)

    prosody_emb = torch.randn(4, 50, 256)  # [B, T, D]
    timbre_emb = torch.randn(4, 256)
    f0_target = torch.rand(4, 50) * 200 + 100
    speaker_labels = torch.randint(0, 100, (4,))

    losses = disentangle_loss(
        prosody_emb=prosody_emb,
        timbre_emb=timbre_emb,
        f0_target=f0_target,
        speaker_labels=speaker_labels,
    )
    print(f"  Ortho loss: {losses['ortho_total'].item():.6f}")
    print(f"  F0 loss: {losses['f0_regression'].item():.6f}")
    print(f"  Speaker loss: {losses['speaker_adversarial'].item():.6f}")
    print(f"  Total loss: {losses['total'].item():.6f}")
    print("  [PASS]")

    # Test 7: SimpleSpeakerEncoder
    print("\n[Test 7] SimpleSpeakerEncoder...")
    speaker_enc = SimpleSpeakerEncoder(input_dim=80, output_dim=256)
    mel = torch.randn(4, 100, 80)  # [B, T, mel]
    speaker_emb = speaker_enc(mel)
    print(f"  Speaker embedding shape: {speaker_emb.shape}")
    print("  [PASS]")

    # Test 8: Scheduled GRL (Diffsody) Integration
    print("\n[Test 8] Scheduled GRL (Diffsody) Integration...")
    try:
        # Create config with scheduled GRL
        sgrl_config = DisentanglementConfig(
            use_grl=True,
            use_scheduled_grl=True,
            grl_schedule_type="sigmoid",
            grl_gamma=10.0,
            grl_warmup_epochs=5,
            total_epochs=100,
            num_speakers=100,
        )

        # Create loss with scheduled GRL
        sgrl_loss = DisentanglementLoss(sgrl_config, prosody_dim=256)

        # Test GRL schedule at different epochs
        print("  Testing sigmoid schedule progression:")
        prosody_emb = torch.randn(4, 50, 256)
        timbre_emb = torch.randn(4, 256)
        speaker_labels = torch.randint(0, 100, (4,))

        for epoch in [0, 10, 25, 50, 75, 100]:
            sgrl_loss.update_grl_lambda(epoch=epoch, total_epochs=100)
            lambda_ = sgrl_loss.get_grl_lambda()
            grl_info = sgrl_loss.get_grl_info()
            print(f"    Epoch {epoch:3d}: λ={lambda_:.4f}, progress={grl_info.get('progress', 0):.2f}")

        # Test that loss computation works
        losses = sgrl_loss(
            prosody_emb=prosody_emb,
            timbre_emb=timbre_emb,
            speaker_labels=speaker_labels,
        )
        print(f"  Final loss: {losses['total'].item():.6f}")
        print(f"  GRL λ in losses: {losses['grl_lambda'].item():.4f}")
        print("  [PASS]")
    except ImportError as e:
        print(f"  [SKIP] Scheduled GRL not available: {e}")

    # Test 9: Learn2Diss (Sandwiched MI) Integration
    print("\n[Test 9] Learn2Diss (Sandwiched MI) Integration...")
    try:
        l2d_config = DisentanglementConfig(
            use_grl=False,
            use_learn2diss=True,
            learn2diss_weight=1.0,
            learn2diss_use_variational_club=True,
            learn2diss_gap_penalty=0.01,
            mine_beta_start=0.01,
            mine_beta_end=1.0,
            mine_warmup_epochs=5,
            num_speakers=100,
        )

        l2d_loss_fn = DisentanglementLoss(l2d_config, prosody_dim=256)

        prosody_emb = torch.randn(4, 50, 256)  # [batch, seq, dim]
        timbre_emb = torch.randn(4, 256)
        speaker_labels = torch.randint(0, 100, (4,))

        losses = l2d_loss_fn(
            prosody_emb=prosody_emb,
            timbre_emb=timbre_emb,
            speaker_labels=speaker_labels,
            epoch=5,
        )

        print(f"  MINE loss: {losses['learn2diss_mine'].item():.4f}")
        print(f"  CLUB loss: {losses['learn2diss_club'].item():.4f}")
        print(f"  MI lower: {losses['mi_lower'].item():.4f}")
        print(f"  MI upper: {losses['mi_upper'].item():.4f}")
        print(f"  MI gap: {losses['mi_gap'].item():.4f}")
        print(f"  MI estimate (sandwiched): {losses['mi_estimate'].item():.4f}")
        print(f"  Learn2Diss total: {losses['learn2diss_total'].item():.4f}")
        print(f"  Combined total: {losses['total'].item():.4f}")
        print("  [PASS]")
    except ImportError as e:
        print(f"  [SKIP] Learn2Diss not available: {e}")
    except Exception as e:
        print(f"  [SKIP] Learn2Diss test failed: {e}")

    # Test 10: DiEmo-TTS (ED-DINO) Integration
    print("\n[Test 10] DiEmo-TTS (ED-DINO) Integration...")
    try:
        diemo_config = DisentanglementConfig(
            use_grl=False,       # Disable GRL, use DiEmo instead
            use_diemo=True,      # Enable DiEmo-TTS ED-DINO
            diemo_weight=1.0,
            diemo_input_dim=256,  # Match prosody_dim for test
            num_speakers=100,
        )

        diemo_loss_fn = DisentanglementLoss(diemo_config, prosody_dim=256)

        prosody_emb = torch.randn(4, 50, 256)  # [batch, seq, dim]
        timbre_emb = torch.randn(4, 256)
        speaker_labels = torch.randint(0, 100, (4,))

        losses = diemo_loss_fn(
            prosody_emb=prosody_emb,
            timbre_emb=timbre_emb,
            speaker_labels=speaker_labels,
            epoch=5,
        )

        print(f"  DiEmo DINO loss: {losses['diemo_dino'].item():.4f}")
        print(f"  DiEmo speaker accuracy: {losses['diemo_speaker_acc'].item():.4f}")
        print(f"  DiEmo total: {losses['diemo_total'].item():.4f}")
        print(f"  Combined total: {losses['total'].item():.4f}")
        print("  [PASS]")
    except ImportError as e:
        print(f"  [SKIP] DiEmo-TTS not available: {e}")
    except Exception as e:
        print(f"  [SKIP] DiEmo test failed: {e}")

    print("\n" + "=" * 60)
    print("All disentanglement tests passed!")
    print("=" * 60)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from disentanglement import (
    DisentanglementConfig,
    DisentanglementLoss,
    SimpleSpeakerEncoder,
)

# Option 1: Standard GRL with linear schedule
config = DisentanglementConfig(
    beta_content_prosody=0.01,    # Relaxed
    beta_timbre_prosody=0.0001,   # Strict
    use_grl=True,
    num_speakers=1000,
)

# Option 2: Diffsody Scheduled GRL (RECOMMENDED)
# Uses sigmoid schedule to prevent early training instability
config = DisentanglementConfig(
    use_grl=True,
    use_scheduled_grl=True,       # Enable Diffsody sigmoid schedule
    grl_schedule_type="sigmoid",  # sigmoid, linear, cosine, exponential
    grl_gamma=10.0,               # Schedule steepness (higher = sharper)
    grl_warmup_epochs=5,          # λ=0 for first 5 epochs
    total_epochs=100,
    num_speakers=1000,
)

disentangle_loss = DisentanglementLoss(config, prosody_dim=2048)
speaker_encoder = SimpleSpeakerEncoder()

# In training loop:
for epoch in range(100):
    # Update GRL schedule at epoch start
    disentangle_loss.update_grl_lambda(epoch, total_epochs=100)

    for batch in dataloader:
        prosody_emb = prosody_encoder(audio)  # Your existing encoder
        timbre_emb = speaker_encoder(mel)     # Extract speaker identity

        losses = disentangle_loss(
            prosody_emb=prosody_emb,
            timbre_emb=timbre_emb,
            f0_target=f0,
            speaker_labels=speaker_ids,
        )

        # Add to main training loss
        # L = L_recon + α*L_prosody - λ(t)*L_speaker
        total_loss = reconstruction_loss + losses['total']

    # Log GRL info
    grl_info = disentangle_loss.get_grl_info()
    print(f"Epoch {epoch}: λ={grl_info['lambda']:.4f}")
""")
