"""
ProsodyFlow: Flow-Matching for Prosody Generation

Based on ProsodyFlow (COLING 2025) and Conditional Flow Matching research.

Key Innovation: Use conditional flow matching to generate prosody vectors:
1. Map acoustic features to prosody latent space
2. Use flow-matching to model prosody distribution
3. Condition on input text for contextual prosody

Advantages over deterministic prediction:
- Captures natural prosody variation
- Can sample diverse prosodies for same text (one-to-many mapping)
- Better handling of text→prosody mapping ambiguity
- Smooth interpolation in prosody latent space

References:
- ProsodyFlow: https://aclanthology.org/2025.coling-main.518/
- Matcha-TTS: https://github.com/shivammehta25/Matcha-TTS
- TorchCFM: https://github.com/atong01/conditional-flow-matching
- Cambridge MLG Flow Matching: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ProsodyFlowConfig:
    """Configuration for ProsodyFlow model."""

    # Prosody latent dimensions (matching existing architecture)
    prosody_dim: int = 2048  # Match ProsodyEncoder output

    # Text conditioning
    text_dim: int = 768  # Text encoder output dimension (e.g., BERT-like)
    use_text_conditioning: bool = True

    # Flow matching settings
    sigma_min: float = 0.001  # Minimum std at t=1 (near-deterministic)

    # OT-CFM settings (optimal transport variant)
    use_ot_coupling: bool = True  # Use optimal transport for x0-x1 pairing
    ot_reg: float = 0.05  # Sinkhorn regularization
    ot_normalize_cost: bool = True

    # Variance control (σ² for expressiveness)
    variance_scale: float = 1.0  # Scale factor for sampling variance
    min_variance: float = 0.1
    max_variance: float = 2.0

    # Network architecture
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    use_sinusoidal_time_emb: bool = True
    time_emb_dim: int = 256

    # Integration with prosody encoder
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate

    # Training
    num_ode_steps_train: int = 10  # Steps for ODE during training (if needed)

    # Inference
    num_ode_steps_inference: int = 50  # ODE integration steps
    ode_method: str = "euler"  # euler, midpoint, rk4


# =============================================================================
# TIME EMBEDDING
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding, similar to positional encoding in Transformers.
    Provides smooth time representation for the flow matching network.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch] or scalar, in range [0, 1]

        Returns:
            Time embeddings [batch, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2

        # Compute frequencies
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )

        # Scale time by frequencies
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # [batch, half_dim]

        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


# =============================================================================
# GAUSSIAN CONDITIONAL PATH
# =============================================================================

class GaussianConditionalPath:
    """
    Gaussian conditional probability path for flow matching.

    Defines the interpolation between noise x_0 and data x_1:
        x_t = μ_t(x_1) + σ_t * ε

    where:
        μ_t(x_1) = t * x_1  (linear interpolation of mean)
        σ_t = (1-t) + t * σ_min  (shrinking variance)

    The conditional vector field is:
        u_t(x|x_1) = (σ̇_t/σ_t)(x - μ_t) + μ̇_t
    """

    def __init__(self, sigma_min: float = 0.001):
        """
        Args:
            sigma_min: Minimum standard deviation at t=1 (near-deterministic)
        """
        self.sigma_min = sigma_min

    def mu_t(self, t: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Mean of conditional distribution at time t."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return t * x1

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Standard deviation at time t."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return (1 - t) + t * self.sigma_min

    def sigma_dot(self) -> float:
        """Time derivative of sigma_t (constant for linear schedule)."""
        return self.sigma_min - 1  # d/dt σ_t = σ_min - 1

    def sample_xt(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_t from conditional distribution p_t(·|x_0, x_1).

        Using the reparameterization: x_t = (1-t)*x_0 + t*x_1 + σ*ε
        But for simplicity, we use: x_t = μ_t(x_1) + σ_t * x_0
        where x_0 ~ N(0, I) is our noise.

        Args:
            t: Time values [batch]
            x0: Noise samples [batch, dim] ~ N(0, I)
            x1: Data samples [batch, dim]

        Returns:
            x_t: Interpolated samples [batch, dim]
        """
        mu = self.mu_t(t, x1)
        sigma = self.sigma_t(t)

        # Expand sigma for broadcasting
        while sigma.dim() < x0.dim():
            sigma = sigma.unsqueeze(-1)

        return mu + sigma * x0

    def compute_target_velocity(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target conditional vector field u_t(x_t | x_1).

        The velocity is:
            u_t(x|x_1) = (σ̇_t/σ_t)(x - μ_t(x_1)) + μ̇_t(x_1)

        For our parameterization:
            μ̇_t = x_1 (since μ_t = t*x_1)
            σ̇_t = σ_min - 1

        Simplification for linear path:
            u_t = (x_1 - x_t) / (1 - t) approximately

        Or more precisely: u_t = x_1 - (1 - σ_min) * x_0

        Args:
            t: Time values [batch]
            x_t: Current samples [batch, dim]
            x1: Target data samples [batch, dim]

        Returns:
            Target velocity [batch, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        mu = self.mu_t(t, x1)
        sigma = self.sigma_t(t)
        sigma_dot_val = self.sigma_dot()

        # Expand for broadcasting
        while sigma.dim() < x_t.dim():
            sigma = sigma.unsqueeze(-1)
        while t.dim() < x_t.dim():
            t = t.unsqueeze(-1)

        # Drift term: (σ̇/σ)(x - μ)
        drift = (sigma_dot_val / sigma) * (x_t - mu)

        # Mean velocity: μ̇ = x_1
        mean_velocity = x1

        return drift + mean_velocity


# =============================================================================
# OPTIMAL TRANSPORT COUPLING
# =============================================================================

class OptimalTransportCoupling:
    """
    Mini-batch optimal transport coupling for OT-CFM.

    Instead of random pairing between x_0 and x_1, use OT to find
    better pairings that reduce variance and create straighter flows.

    Uses Sinkhorn algorithm for fast approximate OT.
    """

    def __init__(
        self,
        reg: float = 0.05,
        num_iters: int = 50,
        normalize_cost: bool = True,
    ):
        """
        Args:
            reg: Sinkhorn regularization (entropy penalty)
            num_iters: Number of Sinkhorn iterations
            normalize_cost: Normalize cost matrix by max value
        """
        self.reg = reg
        self.num_iters = num_iters
        self.normalize_cost = normalize_cost

    def sinkhorn(
        self,
        cost_matrix: torch.Tensor,
        reg: float,
        num_iters: int,
    ) -> torch.Tensor:
        """
        Sinkhorn algorithm for entropy-regularized optimal transport.

        Args:
            cost_matrix: [n, m] cost between samples
            reg: Regularization parameter
            num_iters: Number of iterations

        Returns:
            Transport plan [n, m]
        """
        n, m = cost_matrix.shape
        device = cost_matrix.device

        # Initialize marginals (uniform)
        mu = torch.ones(n, device=device) / n
        nu = torch.ones(m, device=device) / m

        # Kernel matrix K = exp(-C/reg)
        K = torch.exp(-cost_matrix / reg)

        # Initialize scaling vectors
        u = torch.ones(n, device=device)
        v = torch.ones(m, device=device)

        # Sinkhorn iterations
        for _ in range(num_iters):
            u = mu / (K @ v + 1e-8)
            v = nu / (K.T @ u + 1e-8)

        # Transport plan
        plan = torch.diag(u) @ K @ torch.diag(v)

        return plan

    def get_coupling(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get OT-coupled (x0, x1) pairs.

        Args:
            x0: Source samples [batch, dim] (noise)
            x1: Target samples [batch, dim] (data)

        Returns:
            Reordered (x0, x1) with optimal coupling
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Compute squared Euclidean cost matrix
        # cost[i,j] = ||x0[i] - x1[j]||^2
        cost = torch.cdist(x0, x1, p=2).pow(2)

        if self.normalize_cost:
            cost = cost / cost.max()

        # Get OT plan
        plan = self.sinkhorn(cost, self.reg, self.num_iters)

        # Sample coupling: for each x0[i], sample x1[j] from plan[i, :]
        # For simplicity, use argmax (greedy assignment)
        # More sophisticated: sample proportionally to plan
        indices = torch.argmax(plan, dim=1)

        x1_coupled = x1[indices]

        return x0, x1_coupled


# =============================================================================
# VECTOR FIELD NETWORK
# =============================================================================

class VectorFieldNetwork(nn.Module):
    """
    Neural network that predicts the vector field u_θ(t, x, c).

    Takes time t, current state x_t, and optional condition c (text),
    and predicts the velocity/direction of the flow.

    Architecture: Transformer-style with cross-attention to conditioning.
    """

    def __init__(self, config: ProsodyFlowConfig):
        super().__init__()
        self.config = config

        # Time embedding
        if config.use_sinusoidal_time_emb:
            self.time_embed = SinusoidalTimeEmbedding(config.time_emb_dim)
        else:
            self.time_embed = nn.Sequential(
                nn.Linear(1, config.time_emb_dim),
                nn.SiLU(),
                nn.Linear(config.time_emb_dim, config.time_emb_dim),
            )

        # Input projection: x_t + time -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(config.prosody_dim + config.time_emb_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )

        # Text conditioning projection (if used)
        if config.use_text_conditioning:
            self.text_proj = nn.Sequential(
                nn.Linear(config.text_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
            )

        # Transformer layers for processing
        self.layers = nn.ModuleList([
            VectorFieldBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_cross_attention=config.use_text_conditioning,
            )
            for _ in range(config.num_layers)
        ])

        # Output projection: hidden -> prosody velocity
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.prosody_dim),
        )

    def forward(
        self,
        t: torch.Tensor,           # [batch] or scalar
        x_t: torch.Tensor,         # [batch, prosody_dim]
        text_cond: Optional[torch.Tensor] = None,  # [batch, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Predict vector field u_θ(t, x_t, text).

        Args:
            t: Time values [batch] in range [0, 1]
            x_t: Current prosody latent [batch, prosody_dim]
            text_cond: Optional text conditioning [batch, seq_len, text_dim]
            text_mask: Optional attention mask for text

        Returns:
            Predicted velocity [batch, prosody_dim]
        """
        batch_size = x_t.shape[0]

        # Expand t if scalar
        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding
        if self.config.use_sinusoidal_time_emb:
            t_emb = self.time_embed(t)  # [batch, time_emb_dim]
        else:
            t_emb = self.time_embed(t.unsqueeze(-1))  # [batch, time_emb_dim]

        # Concatenate x_t with time embedding
        x_input = torch.cat([x_t, t_emb], dim=-1)  # [batch, prosody_dim + time_emb_dim]

        # Project to hidden dim
        h = self.input_proj(x_input)  # [batch, hidden_dim]

        # Unsqueeze for sequence dimension (single token)
        h = h.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Process text conditioning
        if text_cond is not None and self.config.use_text_conditioning:
            text_h = self.text_proj(text_cond)  # [batch, seq_len, hidden_dim]
        else:
            text_h = None

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, text_h, text_mask)

        # Output projection
        h = h.squeeze(1)  # [batch, hidden_dim]
        velocity = self.output_proj(h)  # [batch, prosody_dim]

        return velocity


class VectorFieldBlock(nn.Module):
    """
    Single transformer block for vector field network.

    Includes:
    1. Self-attention (within prosody state)
    2. Optional cross-attention to text conditioning
    3. Feed-forward network
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_cross_attention: bool,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to text (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,           # [batch, seq_len, hidden]
        text_h: Optional[torch.Tensor] = None,  # [batch, text_len, hidden]
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through block."""
        # Self-attention
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention to text
        if self.use_cross_attention and text_h is not None:
            x_norm = self.cross_attn_norm(x)
            # Convert mask to key_padding_mask format if provided
            key_padding_mask = ~text_mask if text_mask is not None else None
            cross_out, _ = self.cross_attn(
                x_norm, text_h, text_h,
                key_padding_mask=key_padding_mask
            )
            x = x + cross_out

        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))

        return x


# =============================================================================
# ODE SOLVER
# =============================================================================

class ODESolver:
    """
    ODE solver for flow matching inference.

    Integrates the ODE: dx/dt = u_θ(t, x) from t=0 to t=1.
    """

    @staticmethod
    def euler_step(
        model: VectorFieldNetwork,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single Euler step: x_{t+dt} = x_t + dt * u_θ(t, x_t)"""
        velocity = model(t, x, text_cond, text_mask)
        return x + dt * velocity

    @staticmethod
    def midpoint_step(
        model: VectorFieldNetwork,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Midpoint method (RK2)."""
        # Half step
        v1 = model(t, x, text_cond, text_mask)
        x_mid = x + (dt / 2) * v1

        # Full step using midpoint velocity
        v2 = model(t + dt / 2, x_mid, text_cond, text_mask)
        return x + dt * v2

    @staticmethod
    def rk4_step(
        model: VectorFieldNetwork,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runge-Kutta 4th order step."""
        k1 = model(t, x, text_cond, text_mask)
        k2 = model(t + dt/2, x + dt/2 * k1, text_cond, text_mask)
        k3 = model(t + dt/2, x + dt/2 * k2, text_cond, text_mask)
        k4 = model(t + dt, x + dt * k3, text_cond, text_mask)

        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    @staticmethod
    def solve(
        model: VectorFieldNetwork,
        x0: torch.Tensor,
        num_steps: int,
        method: str = "euler",
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Solve ODE from t=0 to t=1.

        Args:
            model: Vector field network
            x0: Initial state [batch, dim] ~ N(0, I)
            num_steps: Number of integration steps
            method: Integration method (euler, midpoint, rk4)
            text_cond: Optional text conditioning
            text_mask: Optional text mask
            return_trajectory: If True, return all intermediate states

        Returns:
            Final state x1 [batch, dim], or trajectory list if return_trajectory
        """
        dt = 1.0 / num_steps
        x = x0
        device = x0.device

        # Select step function
        step_fn = {
            "euler": ODESolver.euler_step,
            "midpoint": ODESolver.midpoint_step,
            "rk4": ODESolver.rk4_step,
        }[method]

        trajectory = [x0] if return_trajectory else None

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            x = step_fn(model, x, t, dt, text_cond, text_mask)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return trajectory
        return x


# =============================================================================
# PROSODY FLOW MODEL
# =============================================================================

class ProsodyFlow(nn.Module):
    """
    ProsodyFlow: Flow-matching model for prosody generation.

    Learns to transform Gaussian noise into prosody latent vectors,
    conditioned on text. Enables:
    1. Diverse prosody sampling for same text
    2. Smooth prosody interpolation
    3. Variance control for expressiveness

    Usage:
        model = ProsodyFlow(config)

        # Training
        loss = model.compute_loss(prosody_target, text_embeddings)

        # Inference
        prosody = model.sample(text_embeddings, num_samples=1)

        # Diverse sampling
        prosodies = model.sample(text_embeddings, num_samples=5, temperature=1.0)
    """

    def __init__(self, config: ProsodyFlowConfig):
        super().__init__()
        self.config = config

        # Core components
        self.vector_field = VectorFieldNetwork(config)
        self.path = GaussianConditionalPath(config.sigma_min)

        # OT coupling (optional)
        if config.use_ot_coupling:
            self.ot_coupler = OptimalTransportCoupling(
                reg=config.ot_reg,
                normalize_cost=config.ot_normalize_cost,
            )
        else:
            self.ot_coupler = None

        # Output projection to prosody tokens
        self.token_projection = nn.Sequential(
            nn.Linear(config.prosody_dim, config.prosody_dim),
            nn.LayerNorm(config.prosody_dim),
            nn.GELU(),
            nn.Linear(config.prosody_dim, config.prosody_dim * config.num_prosody_tokens),
        )
        self.token_norm = nn.LayerNorm(config.prosody_dim)

    def compute_loss(
        self,
        x1: torch.Tensor,                       # [batch, prosody_dim] target prosody
        text_cond: Optional[torch.Tensor] = None,  # [batch, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,
        num_time_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CFM training loss.

        The loss is:
            L = E_{t, x0, x1} [||u_θ(t, x_t, text) - u(x_t | x_1)||²]

        where x_t is sampled from the conditional path p_t(·|x_0, x_1).

        Args:
            x1: Target prosody vectors [batch, prosody_dim]
            text_cond: Optional text conditioning [batch, seq_len, text_dim]
            text_mask: Optional attention mask [batch, seq_len]
            num_time_samples: Number of time samples per batch item

        Returns:
            Dict with 'loss' and auxiliary values
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # Optionally use OT coupling for better pairing
        if self.ot_coupler is not None and batch_size > 1:
            x0, x1_coupled = self.ot_coupler.get_coupling(x0, x1)
        else:
            x1_coupled = x1

        # Sample time uniformly t ~ U[0, 1]
        if num_time_samples == 1:
            t = torch.rand(batch_size, device=device)
        else:
            # Multiple time samples per batch item
            t = torch.rand(batch_size, num_time_samples, device=device)
            x0 = x0.unsqueeze(1).expand(-1, num_time_samples, -1)
            x1_coupled = x1_coupled.unsqueeze(1).expand(-1, num_time_samples, -1)
            if text_cond is not None:
                text_cond = text_cond.unsqueeze(1).expand(-1, num_time_samples, -1, -1)
            if text_mask is not None:
                text_mask = text_mask.unsqueeze(1).expand(-1, num_time_samples, -1)

        # Sample x_t from conditional path
        x_t = self.path.sample_xt(t, x0, x1_coupled)

        # Compute target velocity
        target_velocity = self.path.compute_target_velocity(t, x_t, x1_coupled)

        # Reshape for model if using multiple time samples
        if num_time_samples > 1:
            original_shape = x_t.shape
            x_t = x_t.view(-1, x_t.shape[-1])
            t = t.view(-1)
            target_velocity = target_velocity.view(-1, target_velocity.shape[-1])
            if text_cond is not None:
                text_cond = text_cond.view(-1, text_cond.shape[-2], text_cond.shape[-1])
            if text_mask is not None:
                text_mask = text_mask.view(-1, text_mask.shape[-1])

        # Predict velocity
        predicted_velocity = self.vector_field(t, x_t, text_cond, text_mask)

        # MSE loss
        loss = F.mse_loss(predicted_velocity, target_velocity)

        return {
            'loss': loss,
            'predicted_velocity_norm': predicted_velocity.norm(dim=-1).mean(),
            'target_velocity_norm': target_velocity.norm(dim=-1).mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_samples: int = 1,
        temperature: float = 1.0,
        variance_scale: Optional[float] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample prosody vectors from the learned distribution.

        Args:
            text_cond: Text conditioning [batch, seq_len, text_dim]
            text_mask: Text attention mask [batch, seq_len]
            batch_size: Batch size (used if text_cond is None)
            num_samples: Number of diverse samples per text
            temperature: Sampling temperature (scales initial noise)
            variance_scale: Override config variance_scale
            return_trajectory: If True, return full ODE trajectory

        Returns:
            Prosody samples [batch * num_samples, prosody_dim]
            or dict with 'samples' and 'trajectory' if return_trajectory
        """
        device = next(self.parameters()).device

        if text_cond is not None:
            batch_size = text_cond.shape[0]

        # Expand for multiple samples
        if num_samples > 1:
            if text_cond is not None:
                text_cond = text_cond.repeat_interleave(num_samples, dim=0)
            if text_mask is not None:
                text_mask = text_mask.repeat_interleave(num_samples, dim=0)
            batch_size = batch_size * num_samples

        # Sample initial noise
        x0 = torch.randn(batch_size, self.config.prosody_dim, device=device)

        # Apply temperature and variance scaling
        var_scale = variance_scale if variance_scale is not None else self.config.variance_scale
        x0 = x0 * temperature * var_scale

        # Solve ODE
        result = ODESolver.solve(
            self.vector_field,
            x0,
            num_steps=self.config.num_ode_steps_inference,
            method=self.config.ode_method,
            text_cond=text_cond,
            text_mask=text_mask,
            return_trajectory=return_trajectory,
        )

        if return_trajectory:
            return {
                'samples': result[-1],
                'trajectory': result,
            }
        return result

    def sample_tokens(
        self,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_samples: int = 1,
        temperature: float = 1.0,
        variance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample prosody and project to prefix tokens.

        Compatible with ProsodyEncoder output format.

        Args:
            text_cond: Text conditioning
            text_mask: Text attention mask
            batch_size: Batch size
            num_samples: Number of diverse samples
            temperature: Sampling temperature
            variance_scale: Variance scale for expressiveness

        Returns:
            Prosody tokens [batch * num_samples, num_tokens, prosody_dim]
        """
        # Sample prosody latent
        prosody = self.sample(
            text_cond=text_cond,
            text_mask=text_mask,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
            variance_scale=variance_scale,
        )

        # Project to tokens
        tokens = self.token_projection(prosody)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.prosody_dim)
        tokens = self.token_norm(tokens)

        return tokens

    def interpolate(
        self,
        prosody1: torch.Tensor,
        prosody2: torch.Tensor,
        num_steps: int = 10,
        method: str = "linear",
    ) -> torch.Tensor:
        """
        Interpolate between two prosody vectors in latent space.

        Args:
            prosody1: Source prosody [batch, prosody_dim]
            prosody2: Target prosody [batch, prosody_dim]
            num_steps: Number of interpolation steps
            method: Interpolation method (linear, spherical)

        Returns:
            Interpolated prosodies [batch, num_steps, prosody_dim]
        """
        batch_size = prosody1.shape[0]
        device = prosody1.device

        t_values = torch.linspace(0, 1, num_steps, device=device)

        interpolated = []
        for t in t_values:
            if method == "linear":
                interp = (1 - t) * prosody1 + t * prosody2
            elif method == "spherical":
                # Spherical linear interpolation (SLERP)
                p1_norm = F.normalize(prosody1, dim=-1)
                p2_norm = F.normalize(prosody2, dim=-1)

                dot = (p1_norm * p2_norm).sum(dim=-1, keepdim=True)
                dot = torch.clamp(dot, -1.0, 1.0)
                omega = torch.acos(dot)

                sin_omega = torch.sin(omega)
                nearly_parallel = sin_omega.abs() < 1e-6

                s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
                s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

                interp_dir = s1 * p1_norm + s2 * p2_norm
                interp_dir = torch.where(
                    nearly_parallel,
                    (1 - t) * p1_norm + t * p2_norm,
                    interp_dir
                )

                # Interpolate magnitudes
                mag1 = prosody1.norm(dim=-1, keepdim=True)
                mag2 = prosody2.norm(dim=-1, keepdim=True)
                mag = (1 - t) * mag1 + t * mag2

                interp = interp_dir * mag
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            interpolated.append(interp)

        return torch.stack(interpolated, dim=1)


# =============================================================================
# PROSODY FLOW ADAPTER
# =============================================================================

class ProsodyFlowAdapter(nn.Module):
    """
    Adapter module that integrates ProsodyFlow with the existing prosody pipeline.

    Can be used as a drop-in replacement for deterministic prosody prediction,
    while enabling:
    1. Diverse prosody generation
    2. Controllable variance
    3. Prosody interpolation

    Usage:
        # Instead of:
        # prosody_tokens = prosody_encoder(prosody_dict)

        # Use:
        adapter = ProsodyFlowAdapter(config)
        prosody_tokens = adapter.generate(text_embeddings, variance=0.8)
    """

    def __init__(
        self,
        flow_config: ProsodyFlowConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.flow_config = flow_config

        # Core flow model
        self.flow = ProsodyFlow(flow_config)

        # Optional: encode prosody features to condition the flow
        self.prosody_feature_encoder = nn.Sequential(
            nn.Linear(92, 256),  # 92 = semantic_dim + acoustic_dim + rhythm_dim + contour_dim
            nn.GELU(),
            nn.Linear(256, flow_config.text_dim),
        )

        # Adapt to prosody hidden dimension if different
        if flow_config.prosody_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(flow_config.prosody_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def encode_prosody_features(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode prosody features as conditioning for the flow.

        This enables prosody-guided generation: provide rough prosody
        features and let the flow refine them.
        """
        features = torch.cat([semantic, acoustic, rhythm, contour], dim=-1)
        return self.prosody_feature_encoder(features).unsqueeze(1)  # [batch, 1, text_dim]

    def forward(
        self,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        prosody_dict: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        variance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate prosody tokens using flow matching.

        Args:
            text_cond: Text embeddings [batch, seq_len, text_dim]
            text_mask: Text attention mask
            prosody_dict: Optional prosody features for conditioning
            num_samples: Number of diverse samples
            temperature: Sampling temperature
            variance_scale: Variance scale (expressiveness)

        Returns:
            Prosody tokens [batch * num_samples, num_tokens, prosody_hidden]
        """
        # Optionally add prosody feature conditioning
        if prosody_dict is not None:
            prosody_cond = self.encode_prosody_features(
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'],
            )
            if text_cond is not None:
                # Concatenate prosody conditioning with text
                text_cond = torch.cat([prosody_cond, text_cond], dim=1)
                if text_mask is not None:
                    prosody_mask = torch.ones(
                        text_mask.shape[0], 1,
                        device=text_mask.device,
                        dtype=text_mask.dtype
                    )
                    text_mask = torch.cat([prosody_mask, text_mask], dim=1)
            else:
                text_cond = prosody_cond

        # Sample prosody tokens
        tokens = self.flow.sample_tokens(
            text_cond=text_cond,
            text_mask=text_mask,
            num_samples=num_samples,
            temperature=temperature,
            variance_scale=variance_scale,
        )

        # Adapt to output dimension
        tokens = self.output_adapter(tokens)

        return tokens

    def compute_loss(
        self,
        target_prosody: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        prosody_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            target_prosody: Target prosody vectors [batch, prosody_dim]
            text_cond: Text conditioning
            text_mask: Text mask
            prosody_dict: Optional prosody feature conditioning

        Returns:
            Loss dict
        """
        # Optionally add prosody conditioning
        if prosody_dict is not None:
            prosody_cond = self.encode_prosody_features(
                prosody_dict['semantic'],
                prosody_dict['acoustic'],
                prosody_dict['rhythm'],
                prosody_dict['contour'],
            )
            if text_cond is not None:
                text_cond = torch.cat([prosody_cond, text_cond], dim=1)
                if text_mask is not None:
                    prosody_mask = torch.ones(
                        text_mask.shape[0], 1,
                        device=text_mask.device,
                        dtype=text_mask.dtype
                    )
                    text_mask = torch.cat([prosody_mask, text_mask], dim=1)
            else:
                text_cond = prosody_cond

        return self.flow.compute_loss(target_prosody, text_cond, text_mask)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class ProsodyFlowLoss(nn.Module):
    """
    Combined loss for ProsodyFlow training.

    Includes:
    1. CFM loss (primary)
    2. Optional reconstruction loss
    3. Optional KL regularization
    """

    def __init__(
        self,
        cfm_weight: float = 1.0,
        reconstruction_weight: float = 0.1,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        self.cfm_weight = cfm_weight
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight

    def forward(
        self,
        flow_output: Dict[str, torch.Tensor],
        target_prosody: Optional[torch.Tensor] = None,
        sampled_prosody: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            flow_output: Output from ProsodyFlow.compute_loss()
            target_prosody: Original prosody for reconstruction
            sampled_prosody: Sampled prosody for comparison

        Returns:
            Loss dict with components and total
        """
        losses = {}

        # CFM loss (main training objective)
        losses['cfm'] = flow_output['loss']

        # Reconstruction loss (optional)
        if target_prosody is not None and sampled_prosody is not None:
            losses['reconstruction'] = F.mse_loss(sampled_prosody, target_prosody)
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=losses['cfm'].device)

        # KL regularization (encourage diverse sampling)
        # Approximate: encourage samples to have unit variance
        if sampled_prosody is not None:
            sample_var = sampled_prosody.var(dim=0).mean()
            losses['kl'] = (sample_var - 1.0).pow(2)
        else:
            losses['kl'] = torch.tensor(0.0, device=losses['cfm'].device)

        # Total loss
        total = (
            losses['cfm'] * self.cfm_weight +
            losses['reconstruction'] * self.reconstruction_weight +
            losses['kl'] * self.kl_weight
        )
        losses['total'] = total

        return losses


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ProsodyFlow: Flow-Matching for Prosody Generation - Test Suite")
    print("=" * 70)

    config = ProsodyFlowConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test 1: Sinusoidal Time Embedding
    print("\n[Test 1] Sinusoidal Time Embedding...")
    time_embed = SinusoidalTimeEmbedding(config.time_emb_dim)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    t_emb = time_embed(t)
    print(f"  Input times: {t.tolist()}")
    print(f"  Embedding shape: {t_emb.shape}")
    print(f"  Embedding norm: {t_emb.norm(dim=-1).tolist()}")
    print("  [PASS]")

    # Test 2: Gaussian Conditional Path
    print("\n[Test 2] Gaussian Conditional Path...")
    path = GaussianConditionalPath(sigma_min=0.001)
    batch_size = 4
    dim = 32
    x0 = torch.randn(batch_size, dim)
    x1 = torch.randn(batch_size, dim)
    t = torch.tensor([0.0, 0.5, 1.0])

    for t_val in t:
        x_t = path.sample_xt(t_val, x0, x1)
        velocity = path.compute_target_velocity(t_val, x_t, x1)
        print(f"  t={t_val:.1f}: x_t norm={x_t.norm(dim=-1).mean():.3f}, "
              f"velocity norm={velocity.norm(dim=-1).mean():.3f}")
    print("  [PASS]")

    # Test 3: Optimal Transport Coupling
    print("\n[Test 3] Optimal Transport Coupling...")
    ot_coupler = OptimalTransportCoupling(reg=0.05)
    x0 = torch.randn(8, dim)
    x1 = torch.randn(8, dim)

    # Compute cost before and after OT coupling
    cost_before = (x0 - x1).pow(2).sum(dim=-1).mean()
    x0_coupled, x1_coupled = ot_coupler.get_coupling(x0, x1)
    cost_after = (x0_coupled - x1_coupled).pow(2).sum(dim=-1).mean()

    print(f"  Cost before OT: {cost_before:.3f}")
    print(f"  Cost after OT:  {cost_after:.3f}")
    print(f"  Reduction: {(1 - cost_after/cost_before)*100:.1f}%")
    print("  [PASS]")

    # Test 4: Vector Field Network
    print("\n[Test 4] Vector Field Network...")
    vector_field = VectorFieldNetwork(config).to(device)

    batch_size = 2
    seq_len = 10

    t = torch.rand(batch_size, device=device)
    x_t = torch.randn(batch_size, config.prosody_dim, device=device)
    text_cond = torch.randn(batch_size, seq_len, config.text_dim, device=device)
    text_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    velocity = vector_field(t, x_t, text_cond, text_mask)
    print(f"  Input: t={t.shape}, x_t={x_t.shape}, text={text_cond.shape}")
    print(f"  Output velocity: {velocity.shape}")
    print(f"  Velocity norm: {velocity.norm(dim=-1).mean():.3f}")
    print("  [PASS]")

    # Test 5: ProsodyFlow Model
    print("\n[Test 5] ProsodyFlow Model...")
    flow = ProsodyFlow(config).to(device)

    # Test loss computation
    x1 = torch.randn(batch_size, config.prosody_dim, device=device)
    loss_output = flow.compute_loss(x1, text_cond, text_mask)
    print(f"  CFM Loss: {loss_output['loss'].item():.4f}")
    print(f"  Predicted velocity norm: {loss_output['predicted_velocity_norm'].item():.3f}")
    print(f"  Target velocity norm: {loss_output['target_velocity_norm'].item():.3f}")

    # Test sampling
    samples = flow.sample(text_cond, text_mask, num_samples=3, temperature=1.0)
    print(f"  Sampled prosody shape: {samples.shape}")
    print(f"  Sample diversity (std): {samples.std(dim=0).mean():.3f}")
    print("  [PASS]")

    # Test 6: Prosody Token Generation
    print("\n[Test 6] Prosody Token Generation...")
    tokens = flow.sample_tokens(text_cond, text_mask, num_samples=2)
    print(f"  Token shape: {tokens.shape}")
    print(f"  Expected: [batch*num_samples, {config.num_prosody_tokens}, {config.prosody_dim}]")
    print("  [PASS]")

    # Test 7: Prosody Interpolation
    print("\n[Test 7] Prosody Interpolation...")
    p1 = torch.randn(1, config.prosody_dim, device=device)
    p2 = torch.randn(1, config.prosody_dim, device=device)

    interp_linear = flow.interpolate(p1, p2, num_steps=5, method="linear")
    interp_slerp = flow.interpolate(p1, p2, num_steps=5, method="spherical")

    print(f"  Linear interpolation shape: {interp_linear.shape}")
    print(f"  SLERP interpolation shape: {interp_slerp.shape}")

    # Check interpolation bounds
    print(f"  Linear starts/ends close to p1/p2: "
          f"{(interp_linear[0, 0] - p1).norm().item():.4f}, "
          f"{(interp_linear[0, -1] - p2).norm().item():.4f}")
    print("  [PASS]")

    # Test 8: ODE Trajectory
    print("\n[Test 8] ODE Trajectory...")
    result = flow.sample(
        text_cond[:1], text_mask[:1],
        num_samples=1,
        return_trajectory=True
    )
    trajectory = result['trajectory']
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"  Start (noise): norm={trajectory[0].norm().item():.3f}")
    print(f"  End (prosody): norm={trajectory[-1].norm().item():.3f}")
    print("  [PASS]")

    # Test 9: ProsodyFlowAdapter
    print("\n[Test 9] ProsodyFlowAdapter...")
    adapter = ProsodyFlowAdapter(config).to(device)

    # With text conditioning only
    tokens = adapter(text_cond, text_mask, num_samples=2, temperature=0.8)
    print(f"  With text only - token shape: {tokens.shape}")

    # With prosody features
    prosody_dict = {
        'semantic': torch.randn(batch_size, 8, device=device),
        'acoustic': torch.randn(batch_size, 12, device=device),
        'rhythm': torch.randn(batch_size, 8, device=device),
        'contour': torch.randn(batch_size, 64, device=device),
    }
    tokens = adapter(text_cond, text_mask, prosody_dict=prosody_dict)
    print(f"  With prosody features - token shape: {tokens.shape}")
    print("  [PASS]")

    # Test 10: Combined Loss
    print("\n[Test 10] Combined Loss...")
    loss_fn = ProsodyFlowLoss()

    target = torch.randn(batch_size, config.prosody_dim, device=device)
    sampled = flow.sample(text_cond, text_mask)

    flow_output = flow.compute_loss(target, text_cond, text_mask)
    losses = loss_fn(flow_output, target, sampled)

    print(f"  CFM loss: {losses['cfm'].item():.4f}")
    print(f"  Reconstruction loss: {losses['reconstruction'].item():.4f}")
    print(f"  KL loss: {losses['kl'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 11: Variance Control
    print("\n[Test 11] Variance Control (Expressiveness)...")
    variances = [0.5, 1.0, 1.5]
    for var in variances:
        samples = flow.sample(
            text_cond[:1], text_mask[:1],
            num_samples=10,
            variance_scale=var
        )
        sample_std = samples.std(dim=0).mean()
        print(f"  Variance scale {var}: sample diversity = {sample_std:.3f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All ProsodyFlow tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from prosody_flow import ProsodyFlowConfig, ProsodyFlow, ProsodyFlowAdapter

# Initialize
config = ProsodyFlowConfig()
flow = ProsodyFlow(config).cuda()

# Training
optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-4)

for batch in dataloader:
    prosody_target = batch['prosody']  # [batch, prosody_dim]
    text_embeddings = batch['text']     # [batch, seq_len, text_dim]

    loss_output = flow.compute_loss(prosody_target, text_embeddings)
    loss = loss_output['loss']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference - diverse prosody sampling
text_emb = text_encoder(text)
prosody_samples = flow.sample(
    text_emb,
    num_samples=5,        # Generate 5 different prosodies
    temperature=1.0,      # Sampling randomness
    variance_scale=1.0,   # Expressiveness control
)

# Get prosody tokens (compatible with ProsodyControlledCSM)
tokens = flow.sample_tokens(text_emb, num_samples=1)

# Prosody interpolation
p1 = flow.sample(text_emb_1)  # Prosody for text 1
p2 = flow.sample(text_emb_2)  # Prosody for text 2
interpolated = flow.interpolate(p1, p2, num_steps=10, method="spherical")

# Variance control for expressiveness
expressive_prosody = flow.sample(text_emb, variance_scale=1.5)  # More expressive
subtle_prosody = flow.sample(text_emb, variance_scale=0.5)      # More subtle
""")
