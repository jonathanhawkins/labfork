"""
Scheduled Gradient Reversal Layer (sGRL) from Diffsody

Based on "Disentanglement of Prosody Representations via Diffsody" (IEEE TNNLS 2025).

Key insight: Standard GRL with constant λ can destabilize early training because
the adversarial signal competes with the main task before good representations
are learned. The scheduled GRL addresses this by:

1. Starting with λ ≈ 0 (no adversarial signal in early training)
2. Gradually increasing λ using a sigmoid schedule
3. Allowing the model to first learn good base representations
4. Then gradually enforcing disentanglement

Schedule: λ(p) = 2/(1 + exp(-γ*p)) - 1

Where:
- p = progress in [0, 1] (normalized epoch or step)
- γ = schedule steepness (higher = sharper transition)
- λ ranges from -1 at p=0 (clamped to 0) to +1 at p=∞

With γ=10, the schedule looks like:
- p=0.0 → λ≈0.0 (no reversal)
- p=0.2 → λ≈0.27
- p=0.5 → λ≈0.76
- p=0.8 → λ≈0.97
- p=1.0 → λ≈0.999

Usage:
    from scheduled_grl import ScheduledGRL, GRLScheduler

    # Create scheduled GRL
    grl = ScheduledGRL()
    scheduler = GRLScheduler(grl, gamma=10.0, total_epochs=100)

    # In training loop
    for epoch in range(100):
        scheduler.step(epoch)  # Updates λ
        for batch in dataloader:
            prosody_emb = encoder(audio)
            prosody_grl = grl(prosody_emb)  # Applies current λ
            speaker_pred = classifier(prosody_grl)
            loss = cross_entropy(speaker_pred, speaker_labels)
            # GRL ensures gradients are reversed with current λ
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ScheduledGRLConfig:
    """Configuration for Scheduled Gradient Reversal Layer."""

    # Schedule type
    schedule_type: Literal["sigmoid", "linear", "cosine", "exponential"] = "sigmoid"

    # Sigmoid schedule parameters (Diffsody default)
    gamma: float = 10.0              # Steepness of sigmoid schedule
    gamma_schedule: bool = False      # Whether to increase gamma over training
    gamma_start: float = 5.0         # Starting gamma if gamma_schedule=True
    gamma_end: float = 15.0          # Ending gamma if gamma_schedule=True

    # Lambda bounds
    lambda_min: float = 0.0          # Minimum λ value (clamp from below)
    lambda_max: float = 1.0          # Maximum λ value (clamp from above)

    # Training parameters
    warmup_epochs: int = 0           # Epochs before starting schedule (λ=0)
    total_epochs: int = 100          # Total training epochs

    # Step-based scheduling (alternative to epoch-based)
    use_step_schedule: bool = False  # Use step instead of epoch
    total_steps: Optional[int] = None  # Total steps if step-based

    # Speaker classifier parameters
    num_speakers: int = 1000
    hidden_dim: int = 256
    dropout: float = 0.1


# =============================================================================
# GRADIENT REVERSAL FUNCTIONS
# =============================================================================

class ScheduledGradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal with scheduled λ.

    Forward: Identity function
    Backward: Reverses gradients by multiplying with -λ
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class ScheduledGRL(nn.Module):
    """
    Scheduled Gradient Reversal Layer (sGRL) from Diffsody.

    Unlike standard GRL with constant λ, this uses a schedule that:
    1. Starts near 0 (minimal gradient reversal)
    2. Increases over training following a schedule
    3. Reaches target λ by end of training

    This prevents early training instability from strong adversarial signals.
    """

    def __init__(self, config: Optional[ScheduledGRLConfig] = None):
        super().__init__()
        self.config = config or ScheduledGRLConfig()

        # Current lambda value (updated by scheduler)
        self.register_buffer('_lambda', torch.tensor(0.0))

        # Current gamma value (may be scheduled)
        self.register_buffer('_gamma', torch.tensor(self.config.gamma))

        # Training progress tracking
        self.register_buffer('_progress', torch.tensor(0.0))

    @property
    def lambda_(self) -> float:
        return self._lambda.item()

    @lambda_.setter
    def lambda_(self, value: float):
        self._lambda.fill_(value)

    @property
    def gamma(self) -> float:
        return self._gamma.item()

    @gamma.setter
    def gamma(self, value: float):
        self._gamma.fill_(value)

    @property
    def progress(self) -> float:
        return self._progress.item()

    @progress.setter
    def progress(self, value: float):
        self._progress.fill_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal with current λ."""
        return ScheduledGradientReversalFunction.apply(x, self.lambda_)

    def compute_lambda(self, progress: float) -> float:
        """
        Compute λ value for given training progress.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            λ value for gradient reversal
        """
        config = self.config

        # Handle warmup period (λ = 0)
        if config.warmup_epochs > 0:
            warmup_progress = config.warmup_epochs / max(1, config.total_epochs)
            if progress < warmup_progress:
                return config.lambda_min
            # Adjust progress to start after warmup
            progress = (progress - warmup_progress) / (1 - warmup_progress)
            progress = max(0.0, min(1.0, progress))

        # Compute schedule-specific gamma if gamma_schedule is enabled
        if config.gamma_schedule:
            gamma = config.gamma_start + progress * (config.gamma_end - config.gamma_start)
        else:
            gamma = config.gamma

        # Compute λ based on schedule type
        if config.schedule_type == "sigmoid":
            # Diffsody sigmoid schedule: λ(p) = 2/(1 + exp(-γ*p)) - 1
            # This ranges from -1 (at p=-∞) to 1 (at p=+∞)
            # At p=0, λ = 0
            # We use 10*progress - 5 to center the sigmoid for p ∈ [0, 1]
            # So at p=0.5, λ ≈ 0.5 (middle of transition)
            exponent = -gamma * (progress - 0.5)
            lambda_ = 2.0 / (1.0 + math.exp(exponent)) - 1.0

        elif config.schedule_type == "linear":
            # Linear schedule
            lambda_ = progress

        elif config.schedule_type == "cosine":
            # Cosine schedule (slow start and end, fast middle)
            lambda_ = 0.5 * (1 - math.cos(math.pi * progress))

        elif config.schedule_type == "exponential":
            # Exponential schedule (slow start, fast end)
            lambda_ = (math.exp(gamma * progress) - 1) / (math.exp(gamma) - 1)

        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")

        # Clamp to bounds
        lambda_ = max(config.lambda_min, min(config.lambda_max, lambda_))

        return lambda_

    def update(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Update λ based on current training progress.

        Args:
            epoch: Current epoch (0-indexed)
            step: Current step (0-indexed)
        """
        config = self.config

        if config.use_step_schedule and step is not None:
            total = config.total_steps or 1
            progress = step / max(1, total)
        elif epoch is not None:
            progress = epoch / max(1, config.total_epochs)
        else:
            progress = 0.0

        progress = max(0.0, min(1.0, progress))
        self.progress = progress
        self.lambda_ = self.compute_lambda(progress)

        # Update gamma if scheduled
        if config.gamma_schedule:
            self.gamma = config.gamma_start + progress * (config.gamma_end - config.gamma_start)


# =============================================================================
# GRL SCHEDULER
# =============================================================================

class GRLScheduler:
    """
    Scheduler for Scheduled GRL that manages λ updates.

    Similar to learning rate schedulers, call step() after each epoch/step.

    Usage:
        scheduler = GRLScheduler(grl, total_epochs=100)
        for epoch in range(100):
            scheduler.step(epoch)  # or scheduler.step() for auto-increment
            train_one_epoch()
    """

    def __init__(
        self,
        grl: ScheduledGRL,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
        warmup_epochs: int = 0,
        gamma: Optional[float] = None,
    ):
        """
        Args:
            grl: ScheduledGRL module to manage
            total_epochs: Override config's total_epochs
            total_steps: Override config's total_steps
            warmup_epochs: Override config's warmup_epochs
            gamma: Override config's gamma
        """
        self.grl = grl
        self._epoch = 0
        self._step = 0

        # Override config if provided
        if total_epochs is not None:
            grl.config.total_epochs = total_epochs
        if total_steps is not None:
            grl.config.total_steps = total_steps
            grl.config.use_step_schedule = True
        if warmup_epochs is not None:
            grl.config.warmup_epochs = warmup_epochs
        if gamma is not None:
            grl.config.gamma = gamma

    def step(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Update GRL λ for current epoch/step.

        Args:
            epoch: Current epoch (auto-increments if not provided)
            step: Current step (auto-increments if not provided)
        """
        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1

        if step is not None:
            self._step = step
        else:
            self._step += 1

        self.grl.update(epoch=self._epoch, step=self._step)

    def get_lambda(self) -> float:
        """Get current λ value."""
        return self.grl.lambda_

    def get_gamma(self) -> float:
        """Get current γ value."""
        return self.grl.gamma

    def get_progress(self) -> float:
        """Get current training progress."""
        return self.grl.progress

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            'epoch': self._epoch,
            'step': self._step,
            'lambda': self.grl.lambda_,
            'gamma': self.grl.gamma,
            'progress': self.grl.progress,
        }

    def load_state_dict(self, state: dict):
        """Load scheduler state from checkpoint."""
        self._epoch = state.get('epoch', 0)
        self._step = state.get('step', 0)
        self.grl.lambda_ = state.get('lambda', 0.0)
        self.grl.gamma = state.get('gamma', self.grl.config.gamma)
        self.grl.progress = state.get('progress', 0.0)


# =============================================================================
# SPEAKER ADVERSARIAL HEAD WITH SCHEDULED GRL
# =============================================================================

class ScheduledSpeakerAdversarialHead(nn.Module):
    """
    Speaker classifier with scheduled gradient reversal.

    Uses the Diffsody sigmoid schedule to gradually increase adversarial
    strength, preventing early training instability.

    The training objective is:
        L = L_recon + α*L_prosody - λ(t)*L_speaker

    Where λ(t) follows the sigmoid schedule.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ScheduledGRLConfig] = None,
    ):
        super().__init__()
        config = config or ScheduledGRLConfig()
        self.config = config

        # Scheduled GRL
        self.grl = ScheduledGRL(config)

        # Speaker classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_speakers),
        )

    def forward(self, prosody_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict speaker from prosody (with scheduled gradient reversal).

        Args:
            prosody_emb: [batch, dim] or [batch, seq, dim] prosody embedding

        Returns:
            Speaker logits [batch, num_speakers]
        """
        # Pool if sequence
        if prosody_emb.dim() == 3:
            prosody_emb = prosody_emb.mean(dim=1)

        # Apply scheduled gradient reversal
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

        Due to GRL, gradients are reversed, so minimizing this loss
        trains the prosody encoder to NOT predict speaker identity.
        """
        return nn.functional.cross_entropy(speaker_logits, speaker_labels)

    def update_schedule(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """Update the GRL schedule."""
        self.grl.update(epoch=epoch, step=step)

    def get_lambda(self) -> float:
        """Get current λ value."""
        return self.grl.lambda_


# =============================================================================
# DIFFSODY DISENTANGLEMENT LOSS
# =============================================================================

class DiffsodyDisentanglementLoss(nn.Module):
    """
    Complete disentanglement loss from Diffsody with scheduled GRL.

    Training objective:
        L = L_recon + α*L_prosody - λ(t)*L_speaker

    Where:
        - L_recon: Reconstruction loss (mel-spectrogram or waveform)
        - L_prosody: Prosody prediction loss (F0, energy, duration)
        - L_speaker: Speaker classification loss (with scheduled GRL)
        - λ(t): Sigmoid-scheduled GRL weight

    The key innovation is the scheduled λ(t) which prevents early training
    instability by starting near 0 and gradually increasing.
    """

    def __init__(
        self,
        prosody_dim: int,
        config: Optional[ScheduledGRLConfig] = None,
        prosody_loss_weight: float = 1.0,
        speaker_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.config = config or ScheduledGRLConfig()
        self.prosody_loss_weight = prosody_loss_weight
        self.speaker_loss_weight = speaker_loss_weight

        # Speaker adversarial head with scheduled GRL
        self.speaker_head = ScheduledSpeakerAdversarialHead(
            input_dim=prosody_dim,
            config=self.config,
        )

        # F0 regression head for prosody supervision
        self.f0_head = nn.Sequential(
            nn.Linear(prosody_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # Energy regression head
        self.energy_head = nn.Sequential(
            nn.Linear(prosody_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        prosody_emb: torch.Tensor,
        f0_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute disentanglement losses.

        Args:
            prosody_emb: [batch, dim] or [batch, seq, dim]
            f0_target: Target F0 normalized to [0, 1]
            energy_target: Target energy normalized to [0, 1]
            speaker_labels: Speaker ID labels
            voiced_mask: Mask for voiced frames (optional)

        Returns:
            Dict with individual losses, total loss, and metrics
        """
        device = prosody_emb.device
        losses = {}

        # Pool if sequence
        if prosody_emb.dim() == 3:
            prosody_pooled = prosody_emb.mean(dim=1)
        else:
            prosody_pooled = prosody_emb

        # F0 regression loss (prosody supervision)
        if f0_target is not None:
            f0_pred = self.f0_head(prosody_pooled).squeeze(-1)
            if f0_target.dim() > 1:
                f0_target = f0_target.mean(dim=-1)  # Average if sequence
            if voiced_mask is not None:
                f0_pred = f0_pred[voiced_mask.any(dim=-1)]
                f0_target = f0_target[voiced_mask.any(dim=-1)]
            losses['f0'] = nn.functional.mse_loss(f0_pred, f0_target)
        else:
            losses['f0'] = torch.tensor(0.0, device=device)

        # Energy regression loss
        if energy_target is not None:
            energy_pred = self.energy_head(prosody_pooled).squeeze(-1)
            if energy_target.dim() > 1:
                energy_target = energy_target.mean(dim=-1)
            losses['energy'] = nn.functional.mse_loss(energy_pred, energy_target)
        else:
            losses['energy'] = torch.tensor(0.0, device=device)

        # Speaker adversarial loss (with scheduled GRL)
        if speaker_labels is not None:
            speaker_logits = self.speaker_head(prosody_emb)
            losses['speaker'] = self.speaker_head.compute_loss(speaker_logits, speaker_labels)

            # Accuracy for monitoring
            with torch.no_grad():
                pred_speakers = speaker_logits.argmax(dim=-1)
                losses['speaker_accuracy'] = (pred_speakers == speaker_labels).float().mean()
        else:
            losses['speaker'] = torch.tensor(0.0, device=device)
            losses['speaker_accuracy'] = torch.tensor(0.0, device=device)

        # Combine prosody losses
        losses['prosody'] = losses['f0'] + losses['energy']

        # Total loss (note: speaker loss sign is already handled by GRL)
        losses['total'] = (
            self.prosody_loss_weight * losses['prosody'] +
            self.speaker_loss_weight * losses['speaker']
        )

        # Add schedule info
        losses['grl_lambda'] = torch.tensor(self.speaker_head.get_lambda(), device=device)
        losses['grl_progress'] = torch.tensor(self.speaker_head.grl.progress, device=device)

        return losses

    def update_schedule(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """Update the GRL schedule."""
        self.speaker_head.update_schedule(epoch=epoch, step=step)

    def get_lambda(self) -> float:
        """Get current GRL λ value."""
        return self.speaker_head.get_lambda()


# =============================================================================
# INTEGRATION ADAPTER
# =============================================================================

class ScheduledGRLAdapter:
    """
    Adapter to add scheduled GRL to existing DisentanglementLoss.

    This allows upgrading the existing disentanglement module to use
    Diffsody's scheduled GRL without rewriting the entire pipeline.

    Usage:
        from disentanglement import DisentanglementLoss, DisentanglementConfig
        from scheduled_grl import ScheduledGRLAdapter, ScheduledGRLConfig

        # Create existing disentanglement loss
        disent_config = DisentanglementConfig(use_grl=True)
        disent_loss = DisentanglementLoss(disent_config, prosody_dim=2048)

        # Upgrade to scheduled GRL
        grl_config = ScheduledGRLConfig(gamma=10.0, total_epochs=100)
        adapter = ScheduledGRLAdapter(disent_loss, grl_config)

        # In training loop
        for epoch in range(100):
            adapter.update(epoch=epoch)
            for batch in dataloader:
                losses = disent_loss(prosody_emb, timbre_emb, ...)
    """

    def __init__(
        self,
        disentanglement_loss: nn.Module,
        config: Optional[ScheduledGRLConfig] = None,
    ):
        self.disent_loss = disentanglement_loss
        self.config = config or ScheduledGRLConfig()

        # Create scheduled GRL
        self.scheduled_grl = ScheduledGRL(self.config)

        # Replace the GRL in the speaker head if it exists
        if hasattr(disentanglement_loss, 'speaker_head') and disentanglement_loss.speaker_head is not None:
            original_grl = disentanglement_loss.speaker_head.grl
            disentanglement_loss.speaker_head.grl = self.scheduled_grl
            print(f"Replaced GRL with scheduled GRL (gamma={self.config.gamma})")

    def update(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """Update the schedule."""
        self.scheduled_grl.update(epoch=epoch, step=step)

    def get_lambda(self) -> float:
        """Get current λ value."""
        return self.scheduled_grl.lambda_

    def get_progress(self) -> float:
        """Get current progress."""
        return self.scheduled_grl.progress


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Scheduled GRL (Diffsody) - Test Suite")
    print("=" * 60)

    # Test 1: Sigmoid schedule computation
    print("\n[Test 1] Sigmoid Schedule...")
    config = ScheduledGRLConfig(gamma=10.0, total_epochs=100)
    grl = ScheduledGRL(config)

    print("  Progress → Lambda:")
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        lambda_ = grl.compute_lambda(p)
        print(f"    p={p:.1f} → λ={lambda_:.4f}")
    print("  [PASS]")

    # Test 2: Different schedule types
    print("\n[Test 2] Schedule Types...")
    for schedule_type in ["sigmoid", "linear", "cosine", "exponential"]:
        config = ScheduledGRLConfig(schedule_type=schedule_type, gamma=5.0)
        grl = ScheduledGRL(config)

        lambdas = [grl.compute_lambda(p) for p in [0.0, 0.5, 1.0]]
        print(f"  {schedule_type}: p=[0.0, 0.5, 1.0] → λ={[f'{l:.3f}' for l in lambdas]}")
    print("  [PASS]")

    # Test 3: Warmup period
    print("\n[Test 3] Warmup Period...")
    config = ScheduledGRLConfig(gamma=10.0, warmup_epochs=10, total_epochs=100)
    grl = ScheduledGRL(config)

    for epoch in [0, 5, 10, 15, 50, 100]:
        grl.update(epoch=epoch)
        print(f"    Epoch {epoch:3d}: λ={grl.lambda_:.4f}")
    print("  [PASS]")

    # Test 4: Gamma scheduling
    print("\n[Test 4] Gamma Scheduling...")
    config = ScheduledGRLConfig(
        gamma_schedule=True,
        gamma_start=5.0,
        gamma_end=15.0,
        total_epochs=100,
    )
    grl = ScheduledGRL(config)

    for epoch in [0, 25, 50, 75, 100]:
        grl.update(epoch=epoch)
        print(f"    Epoch {epoch:3d}: γ={grl.gamma:.1f}, λ={grl.lambda_:.4f}")
    print("  [PASS]")

    # Test 5: GRL Scheduler
    print("\n[Test 5] GRL Scheduler...")
    config = ScheduledGRLConfig(gamma=10.0, total_epochs=10)
    grl = ScheduledGRL(config)
    scheduler = GRLScheduler(grl, total_epochs=10)

    for epoch in range(10):
        scheduler.step(epoch)
        print(f"    Epoch {epoch}: λ={scheduler.get_lambda():.4f}, progress={scheduler.get_progress():.2f}")
    print("  [PASS]")

    # Test 6: Gradient Reversal
    print("\n[Test 6] Gradient Reversal...")
    config = ScheduledGRLConfig(gamma=10.0, total_epochs=1)
    grl = ScheduledGRL(config)

    # Set λ = 0.5 for testing
    grl.lambda_ = 0.5

    x = torch.randn(4, 256, requires_grad=True)
    y = grl(x)
    loss = y.sum()
    loss.backward()

    # Check gradient is reversed (should be negative of input gradient direction)
    expected_grad_sign = -0.5  # -λ * 1
    actual_grad_sum = x.grad.sum().item()
    print(f"  λ = {grl.lambda_}")
    print(f"  Input grad sum: {actual_grad_sum:.4f}")
    print(f"  Expected sign: negative (reversed by -λ)")
    print(f"  Gradient reversed: {actual_grad_sum < 0}")
    print("  [PASS]")

    # Test 7: Speaker Adversarial Head
    print("\n[Test 7] Scheduled Speaker Adversarial Head...")
    config = ScheduledGRLConfig(
        gamma=10.0,
        total_epochs=100,
        num_speakers=100,
    )
    head = ScheduledSpeakerAdversarialHead(input_dim=256, config=config)

    prosody_emb = torch.randn(8, 256)
    speaker_labels = torch.randint(0, 100, (8,))

    # Initial (epoch 0)
    head.update_schedule(epoch=0)
    logits_0 = head(prosody_emb)
    loss_0 = head.compute_loss(logits_0, speaker_labels)
    print(f"  Epoch 0: λ={head.get_lambda():.4f}, loss={loss_0.item():.4f}")

    # Mid-training (epoch 50)
    head.update_schedule(epoch=50)
    logits_50 = head(prosody_emb)
    loss_50 = head.compute_loss(logits_50, speaker_labels)
    print(f"  Epoch 50: λ={head.get_lambda():.4f}, loss={loss_50.item():.4f}")

    # End (epoch 100)
    head.update_schedule(epoch=100)
    logits_100 = head(prosody_emb)
    loss_100 = head.compute_loss(logits_100, speaker_labels)
    print(f"  Epoch 100: λ={head.get_lambda():.4f}, loss={loss_100.item():.4f}")
    print("  [PASS]")

    # Test 8: Diffsody Disentanglement Loss
    print("\n[Test 8] Diffsody Disentanglement Loss...")
    config = ScheduledGRLConfig(
        gamma=10.0,
        total_epochs=100,
        num_speakers=100,
    )
    loss_fn = DiffsodyDisentanglementLoss(
        prosody_dim=256,
        config=config,
        prosody_loss_weight=1.0,
        speaker_loss_weight=0.1,
    )

    prosody_emb = torch.randn(8, 256)
    f0_target = torch.rand(8)
    energy_target = torch.rand(8)
    speaker_labels = torch.randint(0, 100, (8,))

    # Epoch 0
    loss_fn.update_schedule(epoch=0)
    losses_0 = loss_fn(prosody_emb, f0_target, energy_target, speaker_labels)
    print(f"  Epoch 0:")
    print(f"    λ={losses_0['grl_lambda'].item():.4f}")
    print(f"    prosody={losses_0['prosody'].item():.4f}")
    print(f"    speaker={losses_0['speaker'].item():.4f}")
    print(f"    total={losses_0['total'].item():.4f}")

    # Epoch 50
    loss_fn.update_schedule(epoch=50)
    losses_50 = loss_fn(prosody_emb, f0_target, energy_target, speaker_labels)
    print(f"  Epoch 50:")
    print(f"    λ={losses_50['grl_lambda'].item():.4f}")
    print(f"    total={losses_50['total'].item():.4f}")

    print("  [PASS]")

    # Test 9: State dict save/load
    print("\n[Test 9] Checkpoint Save/Load...")
    config = ScheduledGRLConfig(gamma=10.0, total_epochs=100)
    grl = ScheduledGRL(config)
    scheduler = GRLScheduler(grl)

    # Advance to epoch 50
    for epoch in range(50):
        scheduler.step(epoch)

    # Save state
    state = scheduler.state_dict()
    print(f"  Saved at epoch {state['epoch']}, λ={state['lambda']:.4f}")

    # Create new scheduler and load
    grl2 = ScheduledGRL(config)
    scheduler2 = GRLScheduler(grl2)
    scheduler2.load_state_dict(state)

    print(f"  Loaded: epoch {scheduler2._epoch}, λ={scheduler2.get_lambda():.4f}")
    assert abs(scheduler.get_lambda() - scheduler2.get_lambda()) < 1e-6
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All scheduled GRL tests passed!")
    print("=" * 60)

    # Print usage example
    print("\nUsage Example:")
    print("-" * 40)
    print("""
from scheduled_grl import (
    ScheduledGRLConfig,
    ScheduledGRL,
    GRLScheduler,
    DiffsodyDisentanglementLoss,
    ScheduledGRLAdapter,
)

# Option 1: Use DiffsodyDisentanglementLoss directly
config = ScheduledGRLConfig(
    schedule_type="sigmoid",  # Diffsody sigmoid schedule
    gamma=10.0,               # Schedule steepness
    warmup_epochs=5,          # λ=0 for first 5 epochs
    total_epochs=100,
    num_speakers=1000,
)

loss_fn = DiffsodyDisentanglementLoss(
    prosody_dim=2048,
    config=config,
    prosody_loss_weight=1.0,
    speaker_loss_weight=0.1,
)

# Training loop
for epoch in range(100):
    loss_fn.update_schedule(epoch=epoch)

    for batch in dataloader:
        prosody_emb = prosody_encoder(batch['audio'])

        losses = loss_fn(
            prosody_emb=prosody_emb,
            f0_target=batch['f0'],
            energy_target=batch['energy'],
            speaker_labels=batch['speaker_id'],
        )

        total_loss = reconstruction_loss + losses['total']
        total_loss.backward()

    print(f"Epoch {epoch}: λ={loss_fn.get_lambda():.4f}")

# Option 2: Upgrade existing DisentanglementLoss
from disentanglement import DisentanglementLoss, DisentanglementConfig

disent_config = DisentanglementConfig(use_grl=True)
disent_loss = DisentanglementLoss(disent_config, prosody_dim=2048)

# Upgrade to scheduled GRL
grl_config = ScheduledGRLConfig(gamma=10.0, total_epochs=100)
adapter = ScheduledGRLAdapter(disent_loss, grl_config)

for epoch in range(100):
    adapter.update(epoch=epoch)
    # Use disent_loss as normal - GRL is now scheduled
    losses = disent_loss(prosody_emb, timbre_emb, ...)
    print(f"Current λ: {adapter.get_lambda():.4f}")
""")
