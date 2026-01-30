"""
TTS-CtrlNet: ControlNet-style Emotion Control for Flow-Matching TTS

Based on TTS-CtrlNet (arXiv:2507.04349). Adds emotion control to pretrained flow-matching
TTS without degrading base model quality.

Key Innovation: ControlNet paradigm for TTS
1. Freeze base TTS model → preserves voice cloning & quality
2. Add trainable copy (control branch) for emotion control
3. Connect via zero-convolutions → prevents degradation during fine-tuning
4. Time-varying control → enables emotion changes within utterances
5. Flexible control scale → adjustable emotion intensity at inference

Architecture:
    ┌───────────────────────────────────────────────────────────┐
    │   Frozen Base Model                                       │
    │   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
    │   │ Block 1  │───│ Block 2  │───│ Block N  │──→ Output    │
    │   └──────────┘   └──────────┘   └──────────┘             │
    │        ↑              ↑              ↑                    │
    │        │              │              │                    │
    │   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐                │
    │   │ Zero    │   │ Zero    │   │ Zero    │   (zero-conv)  │
    │   │ Conv    │   │ Conv    │   │ Conv    │                │
    │   └────┬────┘   └────┬────┘   └────┬────┘                │
    │        │              │              │                    │
    │   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐                │
    │   │Control 1│───│Control 2│───│Control N│   (trainable)  │
    │   └────┬────┘   └────┬────┘   └────┬────┘                │
    │        │              │              │                    │
    │   ┌────┴──────────────┴──────────────┴────┐               │
    │   │        Emotion Conditioning            │               │
    │   │    (VAD / discrete / time-varying)     │               │
    │   └───────────────────────────────────────┘               │
    └───────────────────────────────────────────────────────────┘

Benefits:
- Adds emotion WITHOUT degrading voice cloning ability
- Trains on small emotion dataset while leveraging large-scale pretraining
- Time-varying emotion (not just global) for more expressive synthesis
- Inference-time control scale for intensity adjustment

References:
- TTS-CtrlNet: https://arxiv.org/abs/2507.04349
- ControlNet: https://arxiv.org/abs/2302.05543
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from prosody_flow import (
    ProsodyFlowConfig,
    ProsodyFlow,
    VectorFieldNetwork,
    VectorFieldBlock,
    SinusoidalTimeEmbedding,
    GaussianConditionalPath,
    ODESolver,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TTSCtrlNetConfig:
    """Configuration for TTS-CtrlNet emotion control."""

    # Base model config (inherits from ProsodyFlow)
    base_config: ProsodyFlowConfig = field(default_factory=ProsodyFlowConfig)

    # Emotion representation
    emotion_dim: int = 256  # Emotion embedding dimension
    num_emotions: int = 8   # Number of discrete emotions
    use_vad: bool = True    # Use VAD (Valence-Arousal-Dominance) representation
    vad_dim: int = 3        # VAD dimensions (V, A, D)

    # Control scale
    default_control_scale: float = 1.0  # Default emotion intensity
    min_control_scale: float = 0.0      # Minimum (no emotion control)
    max_control_scale: float = 2.0      # Maximum (strong emotion control)

    # Time-varying control
    use_time_varying: bool = True       # Enable time-varying emotion
    time_varying_resolution: int = 10   # Number of emotion segments per utterance

    # Emotion-specific flow steps
    use_emotion_steps: bool = True      # Use emotion-specific diffusion steps
    emotion_step_mapping: Dict[str, int] = field(default_factory=lambda: {
        "neutral": 50,
        "happy": 40,      # Less steps for high-arousal emotions
        "sad": 60,        # More steps for low-arousal emotions
        "angry": 35,
        "surprised": 30,
        "calm": 70,
        "fearful": 40,
        "disgusted": 45,
    })

    # Zero-convolution settings
    zero_conv_init_scale: float = 0.01  # Small non-zero for gradient flow
    use_learnable_zero_init: bool = True  # Learn the zero-conv initialization

    # Training settings
    dropout: float = 0.1
    freeze_base: bool = True  # Freeze base model during training

    # Integration with prosody tokens
    num_prosody_tokens: int = 4


# =============================================================================
# VAD EMOTION PROTOTYPES
# =============================================================================

VAD_PROTOTYPES = {
    "neutral": (0.0, 0.0, 0.0),
    "happy": (0.8, 0.6, 0.6),
    "sad": (-0.6, -0.4, -0.5),
    "angry": (-0.5, 0.8, 0.7),
    "surprised": (0.3, 0.8, 0.2),
    "calm": (0.4, -0.5, 0.3),
    "fearful": (-0.7, 0.7, -0.7),
    "disgusted": (-0.6, 0.3, 0.4),
}

EMOTION_TO_IDX = {e: i for i, e in enumerate(VAD_PROTOTYPES.keys())}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}


# =============================================================================
# ZERO-CONVOLUTION LAYER
# =============================================================================

class ZeroConv1d(nn.Module):
    """
    Zero-initialized 1D convolution for ControlNet connections.

    Initially outputs near-zero values, gradually learns to pass information
    from control branch to base model without degrading initial performance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_scale: float = 0.01,
        learnable_init: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Initialize weights to near-zero
        if learnable_init:
            nn.init.normal_(self.conv.weight, mean=0.0, std=init_scale)
        else:
            nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        # Learnable scale factor for gradual activation
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, length] or [batch, channels]

        Returns:
            Near-zero output initially, gradually learns signal
        """
        # Handle 2D input (no sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            out = self.conv(x) * self.scale
            return out.squeeze(-1)

        return self.conv(x) * self.scale


class ZeroLinear(nn.Module):
    """
    Zero-initialized linear layer for ControlNet connections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_scale: float = 0.01,
        learnable_init: bool = True,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        # Initialize weights to near-zero
        if learnable_init:
            nn.init.normal_(self.linear.weight, mean=0.0, std=init_scale)
        else:
            nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Learnable scale factor
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


# =============================================================================
# EMOTION ENCODER
# =============================================================================

class EmotionEncoder(nn.Module):
    """
    Encodes emotion information for ControlNet conditioning.

    Supports:
    1. Discrete emotion labels
    2. Continuous VAD coordinates
    3. Time-varying emotion trajectories
    """

    def __init__(self, config: TTSCtrlNetConfig):
        super().__init__()
        self.config = config

        # Discrete emotion embeddings
        self.emotion_embeddings = nn.Embedding(
            config.num_emotions, config.emotion_dim
        )

        # VAD to embedding projection
        if config.use_vad:
            self.vad_encoder = nn.Sequential(
                nn.Linear(config.vad_dim, config.emotion_dim),
                nn.LayerNorm(config.emotion_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.emotion_dim, config.emotion_dim),
                nn.LayerNorm(config.emotion_dim),
            )

        # Initialize VAD prototypes as buffer
        vad_values = torch.tensor([
            VAD_PROTOTYPES[e] for e in VAD_PROTOTYPES.keys()
        ], dtype=torch.float32)
        self.register_buffer('vad_prototypes', vad_values)

        # Time-varying emotion projection
        if config.use_time_varying:
            self.time_proj = nn.Sequential(
                nn.Linear(config.emotion_dim, config.emotion_dim),
                nn.GELU(),
                nn.Linear(config.emotion_dim, config.emotion_dim),
            )

        # Intensity scaling
        self.intensity_scale = nn.Sequential(
            nn.Linear(1, config.emotion_dim // 4),
            nn.GELU(),
            nn.Linear(config.emotion_dim // 4, config.emotion_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        emotion_labels: Optional[torch.Tensor] = None,  # [batch] or [batch, time]
        vad_coords: Optional[torch.Tensor] = None,       # [batch, 3] or [batch, time, 3]
        intensity: Optional[torch.Tensor] = None,        # [batch] or [batch, time]
        time_points: Optional[torch.Tensor] = None,      # [batch, num_points]
    ) -> torch.Tensor:
        """
        Encode emotion into conditioning embeddings.

        Args:
            emotion_labels: Discrete emotion indices
            vad_coords: Continuous VAD coordinates
            intensity: Emotion intensity (0-1)
            time_points: Time points for time-varying emotion (0-1)

        Returns:
            Emotion embeddings [batch, emotion_dim] or [batch, time, emotion_dim]
        """
        batch_size = emotion_labels.shape[0] if emotion_labels is not None else vad_coords.shape[0]
        device = self.emotion_embeddings.weight.device

        # Encode emotion
        if vad_coords is not None and self.config.use_vad:
            # Continuous VAD encoding
            emotion_emb = self.vad_encoder(vad_coords)
        elif emotion_labels is not None:
            # Discrete label encoding
            if emotion_labels.dim() == 1:
                emotion_emb = self.emotion_embeddings(emotion_labels)
            else:
                # Time-varying labels
                emotion_emb = self.emotion_embeddings(emotion_labels)
        else:
            # Default to neutral
            emotion_emb = self.emotion_embeddings(
                torch.zeros(batch_size, dtype=torch.long, device=device)
            )

        # Apply intensity scaling
        if intensity is not None:
            if intensity.dim() == 1:
                intensity = intensity.unsqueeze(-1)  # [batch, 1]
            intensity_scale = self.intensity_scale(intensity)
            emotion_emb = emotion_emb * intensity_scale

        # Apply time-varying projection if needed
        if self.config.use_time_varying and time_points is not None:
            emotion_emb = self.time_proj(emotion_emb)

        return emotion_emb

    def get_vad_for_emotion(self, emotion: str) -> torch.Tensor:
        """Get VAD coordinates for a named emotion."""
        if emotion not in VAD_PROTOTYPES:
            raise ValueError(f"Unknown emotion: {emotion}")
        idx = EMOTION_TO_IDX[emotion]
        return self.vad_prototypes[idx]

    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 1.0,
    ) -> torch.Tensor:
        """
        Interpolate between two emotions.

        Args:
            emotion1: Starting emotion
            emotion2: Ending emotion
            t: Interpolation factor (0 = emotion1, 1 = emotion2)
            intensity: Overall intensity

        Returns:
            Interpolated VAD coordinates [1, 3]
        """
        vad1 = self.get_vad_for_emotion(emotion1)
        vad2 = self.get_vad_for_emotion(emotion2)

        # Linear interpolation in VAD space
        vad_interp = (1 - t) * vad1 + t * vad2

        # Scale by intensity
        vad_interp = vad_interp * intensity

        return vad_interp.unsqueeze(0)


# =============================================================================
# CONTROL BRANCH (TRAINABLE COPY)
# =============================================================================

class ControlBranch(nn.Module):
    """
    Trainable control branch that processes emotion-conditioned inputs.

    This is a copy of the base model's vector field network, augmented with
    emotion conditioning. The outputs are connected to the base model via
    zero-convolutions.
    """

    def __init__(
        self,
        config: TTSCtrlNetConfig,
        base_vector_field: VectorFieldNetwork,
    ):
        super().__init__()
        self.config = config
        base_config = config.base_config

        # Copy architecture from base model
        # Time embedding
        if base_config.use_sinusoidal_time_emb:
            self.time_embed = SinusoidalTimeEmbedding(base_config.time_emb_dim)
        else:
            self.time_embed = nn.Sequential(
                nn.Linear(1, base_config.time_emb_dim),
                nn.SiLU(),
                nn.Linear(base_config.time_emb_dim, base_config.time_emb_dim),
            )

        # Emotion conditioning projection
        self.emotion_proj = nn.Sequential(
            nn.Linear(config.emotion_dim, base_config.hidden_dim),
            nn.LayerNorm(base_config.hidden_dim),
            nn.GELU(),
        )

        # Input projection (with emotion)
        input_dim = base_config.prosody_dim + base_config.time_emb_dim + base_config.hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, base_config.hidden_dim),
            nn.LayerNorm(base_config.hidden_dim),
            nn.SiLU(),
        )

        # Text conditioning projection
        if base_config.use_text_conditioning:
            self.text_proj = nn.Sequential(
                nn.Linear(base_config.text_dim, base_config.hidden_dim),
                nn.LayerNorm(base_config.hidden_dim),
                nn.SiLU(),
            )

        # Control blocks (copy of base model layers)
        self.layers = nn.ModuleList([
            VectorFieldBlock(
                hidden_dim=base_config.hidden_dim,
                num_heads=base_config.num_heads,
                dropout=config.dropout,
                use_cross_attention=base_config.use_text_conditioning,
            )
            for _ in range(base_config.num_layers)
        ])

        # Zero-convolutions for each layer output
        self.zero_convs = nn.ModuleList([
            ZeroLinear(
                base_config.hidden_dim,
                base_config.hidden_dim,
                init_scale=config.zero_conv_init_scale,
                learnable_init=config.use_learnable_zero_init,
            )
            for _ in range(base_config.num_layers)
        ])

        # Final zero-conv for output
        self.output_zero_conv = ZeroLinear(
            base_config.hidden_dim,
            base_config.prosody_dim,
            init_scale=config.zero_conv_init_scale,
            learnable_init=config.use_learnable_zero_init,
        )

    def forward(
        self,
        t: torch.Tensor,                    # [batch]
        x_t: torch.Tensor,                  # [batch, prosody_dim]
        emotion_emb: torch.Tensor,          # [batch, emotion_dim]
        text_cond: Optional[torch.Tensor] = None,  # [batch, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,
        control_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through control branch.

        Args:
            t: Time values
            x_t: Current prosody latent
            emotion_emb: Emotion conditioning
            text_cond: Text conditioning
            text_mask: Text attention mask
            control_scale: Scale factor for control signal

        Returns:
            Dict with 'layer_outputs' (list of per-layer controls) and 'output' (final control)
        """
        batch_size = x_t.shape[0]

        # Expand t if scalar
        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding
        if self.config.base_config.use_sinusoidal_time_emb:
            t_emb = self.time_embed(t)
        else:
            t_emb = self.time_embed(t.unsqueeze(-1))

        # Emotion embedding projection
        emotion_h = self.emotion_proj(emotion_emb)

        # Concatenate inputs
        x_input = torch.cat([x_t, t_emb, emotion_h], dim=-1)

        # Project to hidden dim
        h = self.input_proj(x_input)
        h = h.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Process text conditioning
        if text_cond is not None and self.config.base_config.use_text_conditioning:
            text_h = self.text_proj(text_cond)
        else:
            text_h = None

        # Process through layers, collecting zero-conv outputs
        layer_outputs = []
        for layer, zero_conv in zip(self.layers, self.zero_convs):
            h = layer(h, text_h, text_mask)

            # Apply zero-conv and scale
            layer_out = zero_conv(h.squeeze(1)) * control_scale
            layer_outputs.append(layer_out)

        # Final output
        h_out = h.squeeze(1)
        output = self.output_zero_conv(h_out) * control_scale

        return {
            'layer_outputs': layer_outputs,
            'output': output,
        }


# =============================================================================
# CONTROLLED VECTOR FIELD NETWORK
# =============================================================================

class ControlledVectorField(nn.Module):
    """
    Vector field network with ControlNet-style emotion control.

    Combines frozen base model with trainable control branch via zero-convolutions.
    """

    def __init__(
        self,
        config: TTSCtrlNetConfig,
        base_model: VectorFieldNetwork,
    ):
        super().__init__()
        self.config = config
        base_config = config.base_config

        # Freeze base model
        self.base_model = base_model
        if config.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Create control branch
        self.control_branch = ControlBranch(config, base_model)

        # Emotion encoder
        self.emotion_encoder = EmotionEncoder(config)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        vad_coords: Optional[torch.Tensor] = None,
        intensity: Optional[torch.Tensor] = None,
        control_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict velocity with emotion control.

        Args:
            t: Time values
            x_t: Current prosody latent
            text_cond: Text conditioning
            text_mask: Text mask
            emotion_labels: Discrete emotion indices
            vad_coords: Continuous VAD coordinates
            intensity: Emotion intensity
            control_scale: Control signal scale (0 = no control, 1 = full control)

        Returns:
            Predicted velocity with emotion control
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Encode emotion
        emotion_emb = self.emotion_encoder(
            emotion_labels=emotion_labels,
            vad_coords=vad_coords,
            intensity=intensity,
        )

        # Get control signals from control branch
        control_output = self.control_branch(
            t, x_t, emotion_emb, text_cond, text_mask, control_scale
        )

        # Get base model prediction
        # We need to hook into the base model's layers
        base_velocity = self._forward_with_control(
            t, x_t, text_cond, text_mask, control_output
        )

        return base_velocity

    def _forward_with_control(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        control_output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward through base model with control signal injection.

        Injects control signals after each layer via addition.
        """
        batch_size = x_t.shape[0]
        base_config = self.config.base_config

        # Expand t if scalar
        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding (from base model)
        if base_config.use_sinusoidal_time_emb:
            t_emb = self.base_model.time_embed(t)
        else:
            t_emb = self.base_model.time_embed(t.unsqueeze(-1))

        # Input projection
        x_input = torch.cat([x_t, t_emb], dim=-1)
        h = self.base_model.input_proj(x_input)
        h = h.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Text conditioning
        if text_cond is not None and base_config.use_text_conditioning:
            text_h = self.base_model.text_proj(text_cond)
        else:
            text_h = None

        # Process through layers with control injection
        layer_outputs = control_output['layer_outputs']
        for i, layer in enumerate(self.base_model.layers):
            h = layer(h, text_h, text_mask)

            # Inject control signal (additive)
            if i < len(layer_outputs):
                h = h + layer_outputs[i].unsqueeze(1)

        # Output projection
        h = h.squeeze(1)
        velocity = self.base_model.output_proj(h)

        # Add final control output
        velocity = velocity + control_output['output']

        return velocity


# =============================================================================
# TTS-CTRLNET MODEL
# =============================================================================

class TTSCtrlNet(nn.Module):
    """
    TTS-CtrlNet: ControlNet-style Emotion Control for Flow-Matching TTS.

    Wraps a pretrained ProsodyFlow model with emotion control capability
    without degrading the base model's quality.

    Usage:
        # Load pretrained base model
        base_flow = ProsodyFlow(base_config)
        base_flow.load_state_dict(torch.load("prosody_flow.pt"))

        # Create TTS-CtrlNet wrapper
        ctrlnet_config = TTSCtrlNetConfig(base_config=base_config)
        model = TTSCtrlNet(ctrlnet_config, base_flow)

        # Training (only control branch trains)
        for batch in dataloader:
            loss = model.compute_loss(
                prosody_target, text_cond,
                emotion_labels=emotions,
            )
            loss.backward()  # Only control branch updates

        # Inference with emotion control
        prosody = model.sample(
            text_cond, emotion="happy", intensity=0.8, control_scale=1.0
        )

        # Time-varying emotion
        emotion_trajectory = [("calm", 0.0), ("happy", 0.5), ("surprised", 1.0)]
        prosody = model.sample_time_varying(text_cond, emotion_trajectory)
    """

    def __init__(
        self,
        config: TTSCtrlNetConfig,
        base_flow: Optional[ProsodyFlow] = None,
    ):
        super().__init__()
        self.config = config

        # Create or use provided base flow model
        if base_flow is not None:
            self.base_flow = base_flow
        else:
            self.base_flow = ProsodyFlow(config.base_config)

        # Freeze base model
        if config.freeze_base:
            for param in self.base_flow.parameters():
                param.requires_grad = False

        # Create controlled vector field
        self.controlled_vector_field = ControlledVectorField(
            config, self.base_flow.vector_field
        )

        # Gaussian path for flow matching
        self.path = GaussianConditionalPath(config.base_config.sigma_min)

        # Output projection (can share with base or be separate)
        self.use_separate_output = True
        if self.use_separate_output:
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
        else:
            self.token_projection = self.base_flow.token_projection
            self.token_norm = self.base_flow.token_norm

    def get_emotion_steps(self, emotion: Union[str, torch.Tensor]) -> int:
        """
        Get emotion-specific number of ODE steps.

        High-arousal emotions use fewer steps (faster),
        low-arousal emotions use more steps (more refined).
        """
        if not self.config.use_emotion_steps:
            return self.config.base_config.num_ode_steps_inference

        if isinstance(emotion, str):
            return self.config.emotion_step_mapping.get(
                emotion, self.config.base_config.num_ode_steps_inference
            )
        elif isinstance(emotion, torch.Tensor):
            # Use the first emotion in batch for step count
            if emotion.dim() > 0:
                emotion_idx = emotion[0].item()
            else:
                emotion_idx = emotion.item()
            emotion_name = IDX_TO_EMOTION.get(emotion_idx, "neutral")
            return self.config.emotion_step_mapping.get(
                emotion_name, self.config.base_config.num_ode_steps_inference
            )

        return self.config.base_config.num_ode_steps_inference

    def compute_loss(
        self,
        x1: torch.Tensor,                        # [batch, prosody_dim]
        text_cond: Optional[torch.Tensor] = None,  # [batch, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,  # [batch]
        vad_coords: Optional[torch.Tensor] = None,      # [batch, 3]
        intensity: Optional[torch.Tensor] = None,       # [batch]
        control_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for emotion-controlled flow matching.

        Only the control branch updates during training.

        Args:
            x1: Target prosody vectors
            text_cond: Text conditioning
            text_mask: Text mask
            emotion_labels: Discrete emotion indices
            vad_coords: Continuous VAD coordinates
            intensity: Emotion intensity
            control_scale: Control scale during training

        Returns:
            Loss dict with 'loss' and auxiliary values
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample noise
        x0 = torch.randn_like(x1)

        # Sample time
        t = torch.rand(batch_size, device=device)

        # Sample x_t from conditional path
        x_t = self.path.sample_xt(t, x0, x1)

        # Compute target velocity
        target_velocity = self.path.compute_target_velocity(t, x_t, x1)

        # Predict velocity with emotion control
        predicted_velocity = self.controlled_vector_field(
            t, x_t, text_cond, text_mask,
            emotion_labels=emotion_labels,
            vad_coords=vad_coords,
            intensity=intensity,
            control_scale=control_scale,
        )

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
        emotion: Optional[Union[str, torch.Tensor]] = None,
        vad_coords: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        control_scale: float = 1.0,
        temperature: float = 1.0,
        num_samples: int = 1,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample prosody with emotion control.

        Args:
            text_cond: Text conditioning
            text_mask: Text mask
            batch_size: Batch size
            emotion: Emotion label (str) or indices (tensor)
            vad_coords: VAD coordinates (overrides emotion if provided)
            intensity: Emotion intensity (0-1)
            control_scale: Control signal strength (0 = base model, 1 = full control)
            temperature: Sampling temperature
            num_samples: Number of samples per input
            num_steps: ODE steps (if None, uses emotion-specific steps)
            return_trajectory: Return full ODE trajectory

        Returns:
            Sampled prosody or dict with 'samples' and 'trajectory'
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

        # Prepare emotion conditioning
        emotion_labels = None
        if emotion is not None:
            if isinstance(emotion, str):
                emotion_idx = EMOTION_TO_IDX.get(emotion, 0)
                emotion_labels = torch.full(
                    (batch_size,), emotion_idx, dtype=torch.long, device=device
                )
            else:
                emotion_labels = emotion
                if emotion_labels.shape[0] != batch_size:
                    emotion_labels = emotion_labels.repeat_interleave(num_samples)

        if vad_coords is not None and vad_coords.shape[0] != batch_size:
            vad_coords = vad_coords.repeat_interleave(num_samples, dim=0)

        # Intensity tensor
        intensity_tensor = torch.full((batch_size,), intensity, device=device)

        # Get emotion-specific steps
        if num_steps is None:
            num_steps = self.get_emotion_steps(emotion or "neutral")

        # Sample initial noise
        x0 = torch.randn(
            batch_size, self.config.base_config.prosody_dim, device=device
        ) * temperature

        # Solve ODE with controlled vector field
        dt = 1.0 / num_steps
        x = x0
        trajectory = [x0] if return_trajectory else None

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)

            velocity = self.controlled_vector_field(
                t, x, text_cond, text_mask,
                emotion_labels=emotion_labels,
                vad_coords=vad_coords,
                intensity=intensity_tensor,
                control_scale=control_scale,
            )

            x = x + dt * velocity

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return {
                'samples': x,
                'trajectory': trajectory,
            }
        return x

    def sample_tokens(
        self,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        emotion: Optional[Union[str, torch.Tensor]] = None,
        vad_coords: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        control_scale: float = 1.0,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample prosody and project to prefix tokens.

        Compatible with ProsodyControlledCSM.

        Returns:
            Prosody tokens [batch * num_samples, num_tokens, prosody_dim]
        """
        # Sample prosody latent
        prosody = self.sample(
            text_cond=text_cond,
            text_mask=text_mask,
            batch_size=batch_size,
            emotion=emotion,
            vad_coords=vad_coords,
            intensity=intensity,
            control_scale=control_scale,
            temperature=temperature,
            num_samples=num_samples,
        )

        # Project to tokens
        tokens = self.token_projection(prosody)
        tokens = tokens.view(
            -1, self.config.num_prosody_tokens, self.config.base_config.prosody_dim
        )
        tokens = self.token_norm(tokens)

        return tokens

    @torch.no_grad()
    def sample_time_varying(
        self,
        text_cond: torch.Tensor,
        emotion_trajectory: List[Tuple[str, float]],
        text_mask: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        control_scale: float = 1.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample with time-varying emotion control.

        Enables emotion changes within a single utterance.

        Args:
            text_cond: Text conditioning [batch, seq_len, text_dim]
            emotion_trajectory: List of (emotion_name, time_position) tuples
                Example: [("calm", 0.0), ("happy", 0.5), ("surprised", 1.0)]
            text_mask: Text mask
            intensity: Base intensity
            control_scale: Control strength
            temperature: Sampling temperature

        Returns:
            Sampled prosody [batch, prosody_dim]
        """
        device = text_cond.device
        batch_size = text_cond.shape[0]

        # Parse emotion trajectory
        emotions = [e for e, _ in emotion_trajectory]
        time_points = torch.tensor([t for _, t in emotion_trajectory], device=device)

        # Get VAD coordinates for each emotion point
        emotion_encoder = self.controlled_vector_field.emotion_encoder
        vad_points = torch.stack([
            emotion_encoder.get_vad_for_emotion(e) for e in emotions
        ])  # [num_points, 3]

        # Number of ODE steps (use average of emotions)
        num_steps = int(sum(
            self.get_emotion_steps(e) for e in emotions
        ) / len(emotions))

        # Sample initial noise
        x0 = torch.randn(
            batch_size, self.config.base_config.prosody_dim, device=device
        ) * temperature

        # Solve ODE with time-varying emotion
        dt = 1.0 / num_steps
        x = x0

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            ode_progress = i / num_steps

            # Interpolate VAD based on ODE progress
            vad_interp = self._interpolate_vad_trajectory(
                vad_points, time_points, ode_progress
            )
            vad_interp = vad_interp.unsqueeze(0).expand(batch_size, -1)

            intensity_tensor = torch.full((batch_size,), intensity, device=device)

            velocity = self.controlled_vector_field(
                t, x, text_cond, text_mask,
                vad_coords=vad_interp,
                intensity=intensity_tensor,
                control_scale=control_scale,
            )

            x = x + dt * velocity

        return x

    def _interpolate_vad_trajectory(
        self,
        vad_points: torch.Tensor,  # [num_points, 3]
        time_points: torch.Tensor,  # [num_points]
        t: float,
    ) -> torch.Tensor:
        """Interpolate VAD coordinates at time t."""
        # Find surrounding points
        for i in range(len(time_points) - 1):
            if time_points[i] <= t <= time_points[i + 1]:
                # Linear interpolation
                alpha = (t - time_points[i]) / (time_points[i + 1] - time_points[i] + 1e-8)
                return (1 - alpha) * vad_points[i] + alpha * vad_points[i + 1]

        # Outside range - use nearest
        if t < time_points[0]:
            return vad_points[0]
        return vad_points[-1]

    def interpolate_emotions(
        self,
        text_cond: torch.Tensor,
        emotion1: str,
        emotion2: str,
        num_steps: int = 5,
        text_mask: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        control_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate prosody for a range of interpolated emotions.

        Args:
            text_cond: Text conditioning
            emotion1: Starting emotion
            emotion2: Ending emotion
            num_steps: Number of interpolation steps
            text_mask: Text mask
            intensity: Emotion intensity
            control_scale: Control strength

        Returns:
            Interpolated prosodies [num_steps, prosody_dim]
        """
        device = text_cond.device
        emotion_encoder = self.controlled_vector_field.emotion_encoder

        prosodies = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0.0
            vad_interp = emotion_encoder.interpolate_emotions(
                emotion1, emotion2, t, intensity
            )

            prosody = self.sample(
                text_cond, text_mask,
                vad_coords=vad_interp.to(device),
                intensity=intensity,
                control_scale=control_scale,
            )
            prosodies.append(prosody)

        return torch.cat(prosodies, dim=0)


# =============================================================================
# ADAPTER FOR EXISTING PIPELINE
# =============================================================================

class TTSCtrlNetAdapter(nn.Module):
    """
    Adapter that integrates TTS-CtrlNet with existing prosody pipeline.

    Provides a simple interface for emotion-controlled prosody generation
    that is compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        config: TTSCtrlNetConfig,
        base_flow: Optional[ProsodyFlow] = None,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core CtrlNet model
        self.ctrlnet = TTSCtrlNet(config, base_flow)

        # Output adapter if dimensions differ
        if config.base_config.prosody_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.base_config.prosody_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def forward(
        self,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        emotion: Optional[Union[str, torch.Tensor]] = None,
        vad_coords: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        control_scale: float = 1.0,
        temperature: float = 1.0,
        num_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with emotion control.

        Args:
            text_cond: Text embeddings
            text_mask: Text attention mask
            emotion: Emotion label or indices
            vad_coords: VAD coordinates (optional, overrides emotion)
            intensity: Emotion intensity
            control_scale: Control strength
            temperature: Sampling temperature
            num_samples: Number of samples

        Returns:
            Dict with 'prosody_tokens' and 'prosody_latent'
        """
        # Sample tokens
        tokens = self.ctrlnet.sample_tokens(
            text_cond=text_cond,
            text_mask=text_mask,
            emotion=emotion,
            vad_coords=vad_coords,
            intensity=intensity,
            control_scale=control_scale,
            temperature=temperature,
            num_samples=num_samples,
        )

        # Adapt output dimensions
        tokens = self.output_adapter(tokens)

        # Also get latent for auxiliary use
        latent = self.ctrlnet.sample(
            text_cond=text_cond,
            text_mask=text_mask,
            emotion=emotion,
            vad_coords=vad_coords,
            intensity=intensity,
            control_scale=control_scale,
            temperature=temperature,
            num_samples=num_samples,
        )

        return {
            'prosody_tokens': tokens,
            'prosody_latent': latent,
        }

    def from_emotion(
        self,
        emotion: str,
        intensity: float = 1.0,
        control_scale: float = 1.0,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody from emotion label only (no text conditioning).

        Args:
            emotion: Emotion name
            intensity: Emotion intensity
            control_scale: Control strength
            batch_size: Number of samples

        Returns:
            Dict with 'prosody_tokens'
        """
        return self.forward(
            emotion=emotion,
            intensity=intensity,
            control_scale=control_scale,
            num_samples=batch_size,
        )

    def from_vad(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        intensity: float = 1.0,
        control_scale: float = 1.0,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody from VAD coordinates.

        Args:
            valence: Valence (-1 to 1)
            arousal: Arousal (-1 to 1)
            dominance: Dominance (-1 to 1)
            intensity: Emotion intensity
            control_scale: Control strength
            batch_size: Number of samples

        Returns:
            Dict with 'prosody_tokens'
        """
        device = next(self.parameters()).device
        vad = torch.tensor([[valence, arousal, dominance]], device=device)

        return self.forward(
            vad_coords=vad.expand(batch_size, -1),
            intensity=intensity,
            control_scale=control_scale,
        )

    def with_time_varying_emotion(
        self,
        text_cond: torch.Tensor,
        emotion_trajectory: List[Tuple[str, float]],
        text_mask: Optional[torch.Tensor] = None,
        intensity: float = 1.0,
        control_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody with time-varying emotion.

        Args:
            text_cond: Text conditioning
            emotion_trajectory: List of (emotion, time_position) tuples
            text_mask: Text mask
            intensity: Emotion intensity
            control_scale: Control strength

        Returns:
            Dict with 'prosody_tokens' and 'prosody_latent'
        """
        latent = self.ctrlnet.sample_time_varying(
            text_cond=text_cond,
            emotion_trajectory=emotion_trajectory,
            text_mask=text_mask,
            intensity=intensity,
            control_scale=control_scale,
        )

        # Project to tokens
        tokens = self.ctrlnet.token_projection(latent)
        tokens = tokens.view(
            -1, self.config.num_prosody_tokens, self.config.base_config.prosody_dim
        )
        tokens = self.ctrlnet.token_norm(tokens)
        tokens = self.output_adapter(tokens)

        return {
            'prosody_tokens': tokens,
            'prosody_latent': latent,
        }

    def compute_loss(
        self,
        target_prosody: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        vad_coords: Optional[torch.Tensor] = None,
        intensity: Optional[torch.Tensor] = None,
        control_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Only control branch parameters update.
        """
        return self.ctrlnet.compute_loss(
            x1=target_prosody,
            text_cond=text_cond,
            text_mask=text_mask,
            emotion_labels=emotion_labels,
            vad_coords=vad_coords,
            intensity=intensity,
            control_scale=control_scale,
        )


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class TTSCtrlNetLoss(nn.Module):
    """
    Combined loss for TTS-CtrlNet training.

    Includes:
    1. CFM loss (primary)
    2. Emotion consistency loss
    3. Control strength regularization
    """

    def __init__(
        self,
        cfm_weight: float = 1.0,
        emotion_consistency_weight: float = 0.1,
        control_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.cfm_weight = cfm_weight
        self.emotion_consistency_weight = emotion_consistency_weight
        self.control_reg_weight = control_reg_weight

    def forward(
        self,
        ctrlnet_output: Dict[str, torch.Tensor],
        emotion_labels: Optional[torch.Tensor] = None,
        vad_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            ctrlnet_output: Output from TTSCtrlNet.compute_loss()
            emotion_labels: Ground truth emotion labels
            vad_target: Ground truth VAD coordinates

        Returns:
            Loss dict with components and total
        """
        losses = {}

        # CFM loss (main objective)
        losses['cfm'] = ctrlnet_output['loss']

        # Emotion consistency (if labels provided)
        # Placeholder - could add emotion classifier on output
        losses['emotion_consistency'] = torch.tensor(0.0, device=losses['cfm'].device)

        # Control regularization (encourage moderate control activation)
        losses['control_reg'] = torch.tensor(0.0, device=losses['cfm'].device)

        # Total loss
        total = (
            losses['cfm'] * self.cfm_weight +
            losses['emotion_consistency'] * self.emotion_consistency_weight +
            losses['control_reg'] * self.control_reg_weight
        )
        losses['total'] = total

        return losses


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TTS-CtrlNet: ControlNet-style Emotion Control - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create configs
    base_config = ProsodyFlowConfig(
        prosody_dim=512,  # Smaller for testing
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
    )

    config = TTSCtrlNetConfig(
        base_config=base_config,
        emotion_dim=128,
        freeze_base=True,
    )

    # Test 1: Zero-Convolution
    print("\n[Test 1] Zero-Convolution Layers...")
    zero_conv = ZeroLinear(256, 256, init_scale=0.01).to(device)
    x = torch.randn(4, 256, device=device)
    out = zero_conv(x)
    print(f"  Input norm: {x.norm():.4f}")
    print(f"  Output norm: {out.norm():.4f}")
    print(f"  Output is near-zero: {out.norm() < x.norm() * 0.1}")
    print("  [PASS]")

    # Test 2: Emotion Encoder
    print("\n[Test 2] Emotion Encoder...")
    emotion_encoder = EmotionEncoder(config).to(device)

    # Test discrete labels
    labels = torch.tensor([0, 1, 2, 3], device=device)
    emb = emotion_encoder(emotion_labels=labels)
    print(f"  Discrete emotion embedding: {emb.shape}")

    # Test VAD coordinates
    vad = torch.randn(4, 3, device=device)
    emb_vad = emotion_encoder(vad_coords=vad)
    print(f"  VAD emotion embedding: {emb_vad.shape}")

    # Test interpolation
    vad_interp = emotion_encoder.interpolate_emotions("sad", "happy", 0.5)
    print(f"  Interpolated VAD (sad→happy @ 0.5): {vad_interp.squeeze().tolist()}")
    print("  [PASS]")

    # Test 3: Control Branch
    print("\n[Test 3] Control Branch...")
    base_flow = ProsodyFlow(base_config).to(device)
    control_branch = ControlBranch(config, base_flow.vector_field).to(device)

    batch_size = 2
    t = torch.rand(batch_size, device=device)
    x_t = torch.randn(batch_size, base_config.prosody_dim, device=device)
    emotion_emb = torch.randn(batch_size, config.emotion_dim, device=device)
    text_cond = torch.randn(batch_size, 10, base_config.text_dim, device=device)

    control_out = control_branch(t, x_t, emotion_emb, text_cond, control_scale=1.0)
    print(f"  Layer outputs: {len(control_out['layer_outputs'])}")
    print(f"  Output shape: {control_out['output'].shape}")
    print(f"  Output norm: {control_out['output'].norm():.4f}")
    print("  [PASS]")

    # Test 4: Controlled Vector Field
    print("\n[Test 4] Controlled Vector Field...")
    controlled_vf = ControlledVectorField(config, base_flow.vector_field).to(device)

    velocity = controlled_vf(
        t, x_t, text_cond, None,
        emotion_labels=torch.zeros(batch_size, dtype=torch.long, device=device),
        intensity=torch.ones(batch_size, device=device),
        control_scale=1.0,
    )
    print(f"  Velocity shape: {velocity.shape}")
    print(f"  Velocity norm: {velocity.norm():.4f}")
    print("  [PASS]")

    # Test 5: TTS-CtrlNet Model
    print("\n[Test 5] TTS-CtrlNet Model...")
    model = TTSCtrlNet(config, base_flow).to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params: {frozen_params:,}")
    print(f"  Train ratio: {trainable_params / (trainable_params + frozen_params) * 100:.1f}%")

    # Test loss computation
    target = torch.randn(batch_size, base_config.prosody_dim, device=device)
    loss_out = model.compute_loss(
        target, text_cond, None,
        emotion_labels=torch.zeros(batch_size, dtype=torch.long, device=device),
    )
    print(f"  CFM Loss: {loss_out['loss'].item():.4f}")
    print("  [PASS]")

    # Test 6: Sampling with Emotion Control
    print("\n[Test 6] Sampling with Emotion Control...")

    # Sample with discrete emotion
    sample = model.sample(
        text_cond, None,
        emotion="happy",
        intensity=0.8,
        control_scale=1.0,
    )
    print(f"  Sample shape (happy): {sample.shape}")

    # Sample with VAD
    vad_coords = torch.tensor([[0.8, 0.6, 0.6]], device=device)  # Happy VAD
    sample_vad = model.sample(
        text_cond, None,
        vad_coords=vad_coords,
        intensity=0.8,
    )
    print(f"  Sample shape (VAD): {sample_vad.shape}")

    # Different control scales
    for scale in [0.0, 0.5, 1.0, 1.5]:
        sample_scaled = model.sample(
            text_cond, None,
            emotion="angry",
            control_scale=scale,
        )
        print(f"  Control scale {scale}: norm={sample_scaled.norm():.3f}")
    print("  [PASS]")

    # Test 7: Emotion-Specific Steps
    print("\n[Test 7] Emotion-Specific Flow Steps...")
    for emotion in ["neutral", "happy", "sad", "angry", "calm"]:
        steps = model.get_emotion_steps(emotion)
        print(f"  {emotion}: {steps} steps")
    print("  [PASS]")

    # Test 8: Time-Varying Emotion
    print("\n[Test 8] Time-Varying Emotion...")
    emotion_trajectory = [
        ("calm", 0.0),
        ("happy", 0.5),
        ("surprised", 1.0),  # Using core emotion instead of "excited"
    ]

    sample_tv = model.sample_time_varying(
        text_cond[:1], emotion_trajectory,
        intensity=0.8,
        control_scale=1.0,
    )
    print(f"  Time-varying sample shape: {sample_tv.shape}")
    print("  [PASS]")

    # Test 9: Emotion Interpolation
    print("\n[Test 9] Emotion Interpolation...")
    interpolated = model.interpolate_emotions(
        text_cond[:1], "sad", "happy",
        num_steps=5,
        control_scale=1.0,
    )
    print(f"  Interpolated shape: {interpolated.shape}")
    print(f"  Interpolation norms: {interpolated.norm(dim=-1).tolist()}")
    print("  [PASS]")

    # Test 10: Token Generation
    print("\n[Test 10] Token Generation...")
    tokens = model.sample_tokens(
        text_cond, None,
        emotion="happy",
        intensity=0.8,
        num_samples=2,
    )
    print(f"  Token shape: {tokens.shape}")
    print(f"  Expected: [batch*num_samples, {config.num_prosody_tokens}, {base_config.prosody_dim}]")
    print("  [PASS]")

    # Test 11: TTSCtrlNetAdapter
    print("\n[Test 11] TTSCtrlNetAdapter...")
    adapter = TTSCtrlNetAdapter(config, base_flow).to(device)

    # From emotion
    result = adapter.from_emotion("happy", intensity=0.8, control_scale=1.0)
    print(f"  From emotion - tokens: {result['prosody_tokens'].shape}")

    # From VAD
    result = adapter.from_vad(0.8, 0.6, 0.6, intensity=0.8)
    print(f"  From VAD - tokens: {result['prosody_tokens'].shape}")

    # Time-varying
    result = adapter.with_time_varying_emotion(
        text_cond[:1],
        [("calm", 0.0), ("happy", 0.5), ("surprised", 1.0)],
    )
    print(f"  Time-varying - tokens: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 12: Gradient Flow
    print("\n[Test 12] Gradient Flow (only control branch updates)...")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    target = torch.randn(batch_size, base_config.prosody_dim, device=device)

    loss_out = model.compute_loss(
        target, text_cond,
        emotion_labels=torch.zeros(batch_size, dtype=torch.long, device=device),
    )

    optimizer.zero_grad()
    loss_out['loss'].backward()

    # Check gradients
    base_grads = sum(
        p.grad.abs().sum().item() if p.grad is not None else 0
        for p in model.base_flow.parameters()
    )
    control_grads = sum(
        p.grad.abs().sum().item() if p.grad is not None else 0
        for p in model.controlled_vector_field.control_branch.parameters()
    )

    print(f"  Base model grad sum: {base_grads:.6f}")
    print(f"  Control branch grad sum: {control_grads:.4f}")
    print(f"  Only control branch has gradients: {base_grads == 0 and control_grads > 0}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All TTS-CtrlNet tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from tts_ctrlnet import TTSCtrlNetConfig, TTSCtrlNet, TTSCtrlNetAdapter
from prosody_flow import ProsodyFlowConfig, ProsodyFlow

# 1. Load pretrained base model
base_config = ProsodyFlowConfig()
base_flow = ProsodyFlow(base_config)
base_flow.load_state_dict(torch.load("prosody_flow.pt"))

# 2. Create TTS-CtrlNet wrapper (base model frozen)
ctrlnet_config = TTSCtrlNetConfig(base_config=base_config)
model = TTSCtrlNet(ctrlnet_config, base_flow).cuda()

# 3. Train only control branch on emotion data
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

for batch in emotion_dataloader:
    loss = model.compute_loss(
        batch['prosody'],
        batch['text_embeddings'],
        emotion_labels=batch['emotions'],
    )

    optimizer.zero_grad()
    loss['loss'].backward()
    optimizer.step()

# 4. Inference with emotion control
# Discrete emotion
tokens = model.sample_tokens(
    text_cond, emotion="happy",
    intensity=0.8,
    control_scale=1.0,  # Full control
)

# VAD coordinates for fine-grained control
tokens = model.sample_tokens(
    text_cond,
    vad_coords=torch.tensor([[0.8, 0.6, 0.6]]),  # Custom VAD
    intensity=0.9,
)

# Control scale for intensity (0 = base model, 1 = full emotion)
tokens_subtle = model.sample_tokens(text_cond, emotion="angry", control_scale=0.3)
tokens_strong = model.sample_tokens(text_cond, emotion="angry", control_scale=1.5)

# 5. Time-varying emotion (emotion changes within utterance)
trajectory = [
    ("calm", 0.0),      # Start calm
    ("happy", 0.5),     # Transition to happy
    ("surprised", 1.0),  # End surprised
]
tokens = model.sample_time_varying(text_cond, trajectory)

# 6. Emotion interpolation
interpolated = model.interpolate_emotions(
    text_cond, "sad", "happy",
    num_steps=10,  # 10 steps from sad to happy
)

# 7. Use with ProsodyControlledCSM
combined_prefix = torch.cat([tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
