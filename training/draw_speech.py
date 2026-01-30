"""
DrawSpeech: Sketch-Conditioned Prosody Control

Based on DrawSpeech (arxiv.org/html/2501.04256v1) - ICASSP 2025.

Key Innovation: Users draw pitch/energy curves that serve as coarse guides for
prosody generation. A sketch-conditioned diffusion model refines these curves
into natural, expressive prosody while respecting the user's intent.

Architecture:
    User Sketch (pitch/energy curves) → SketchEncoder → Sketch Embeddings
                                                              ↓
    Text → TextEncoder →→→→→→→→→→ SketchConditionedFlow → Prosody Tokens
                                                              ↓
                                              ProsodyControlledCSM → Audio

Benefits:
- Intuitive control: Draw what you want to hear
- Natural output: Diffusion refines sketches into realistic prosody
- No reference audio needed: Control prosody from scratch
- Compatible with existing pipeline: Drop-in integration with ProsodyFlow
- Perfect for frontend 3D visualizer: Users draw directly on curves

References:
- DrawSpeech: https://arxiv.org/html/2501.04256v1
- DiffSpeech: https://arxiv.org/abs/2104.09527
- ProsodyFlow: Our existing flow-matching implementation
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from prosody_flow import (
    ProsodyFlowConfig,
    ProsodyFlow,
    VectorFieldNetwork,
    GaussianConditionalPath,
    OptimalTransportCoupling,
    ODESolver,
    SinusoidalTimeEmbedding,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SketchConfig:
    """Configuration for sketch encoder."""

    # Sketch dimensions
    sketch_length: int = 100  # Number of points in sketch (normalized time)
    num_curves: int = 2  # pitch + energy curves

    # Encoder architecture
    hidden_dim: int = 256
    num_conv_layers: int = 4
    kernel_size: int = 5
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    # Output
    sketch_embed_dim: int = 512  # Final sketch embedding dimension

    # Sketch preprocessing
    normalize_sketches: bool = True
    use_delta_encoding: bool = True  # Encode changes, not absolute values

    # Smoothing (sketch refinement)
    smooth_sigma: float = 2.0  # Gaussian smoothing for sketch curves


@dataclass
class DrawSpeechConfig:
    """Configuration for DrawSpeech model."""

    # Sketch encoder config
    sketch: SketchConfig = field(default_factory=SketchConfig)

    # Prosody flow config (inherit from existing)
    flow: ProsodyFlowConfig = field(default_factory=ProsodyFlowConfig)

    # Conditioning mode
    sketch_conditioning: str = "cross_attention"  # "cross_attention" or "concat"
    sketch_dropout: float = 0.1  # Dropout sketch conditioning during training

    # Classifier-free guidance
    use_cfg: bool = True
    cfg_scale: float = 2.0  # Guidance strength
    cfg_dropout: float = 0.1  # Probability of dropping conditioning during training

    # Integration with text
    combine_text_sketch: bool = True  # Fuse text and sketch before conditioning


# =============================================================================
# SKETCH PREPROCESSING
# =============================================================================

class SketchPreprocessor:
    """
    Preprocess user-drawn sketches for model input.

    Handles:
    - Resampling to fixed length
    - Normalization
    - Smoothing
    - Delta encoding (rate of change)
    """

    def __init__(self, config: SketchConfig):
        self.config = config

    def resample(
        self,
        sketch: torch.Tensor,
        target_length: int = None,
    ) -> torch.Tensor:
        """
        Resample sketch to fixed length.

        Args:
            sketch: [batch, num_curves, variable_length] or [num_curves, variable_length]
            target_length: Target length (default: config.sketch_length)

        Returns:
            Resampled sketch [batch, num_curves, target_length]
        """
        if target_length is None:
            target_length = self.config.sketch_length

        if sketch.dim() == 2:
            sketch = sketch.unsqueeze(0)

        # Use linear interpolation
        resampled = F.interpolate(
            sketch,
            size=target_length,
            mode='linear',
            align_corners=True,
        )

        return resampled

    def normalize(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Normalize sketch to [0, 1] range per curve.

        Args:
            sketch: [batch, num_curves, length]

        Returns:
            Normalized sketch
        """
        if not self.config.normalize_sketches:
            return sketch

        # Per-curve normalization
        min_vals = sketch.min(dim=-1, keepdim=True)[0]
        max_vals = sketch.max(dim=-1, keepdim=True)[0]
        range_vals = max_vals - min_vals + 1e-8

        normalized = (sketch - min_vals) / range_vals

        return normalized

    def smooth(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smoothing to sketch curves.

        Args:
            sketch: [batch, num_curves, length]

        Returns:
            Smoothed sketch
        """
        if self.config.smooth_sigma <= 0:
            return sketch

        # Create Gaussian kernel
        sigma = self.config.smooth_sigma
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, device=sketch.device, dtype=sketch.dtype)
        x = x - kernel_size // 2
        kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        # Apply convolution per curve
        kernel = kernel.view(1, 1, -1)
        padding = kernel_size // 2

        smoothed = F.conv1d(
            sketch,
            kernel.expand(sketch.shape[1], -1, -1),
            padding=padding,
            groups=sketch.shape[1],
        )

        return smoothed

    def compute_delta(self, sketch: torch.Tensor) -> torch.Tensor:
        """
        Compute delta (rate of change) encoding.

        Args:
            sketch: [batch, num_curves, length]

        Returns:
            Delta encoding [batch, num_curves, length]
        """
        if not self.config.use_delta_encoding:
            return torch.zeros_like(sketch)

        # Compute first derivative (finite differences)
        delta = torch.zeros_like(sketch)
        delta[..., 1:] = sketch[..., 1:] - sketch[..., :-1]

        return delta

    def preprocess(
        self,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full preprocessing pipeline for sketches.

        Args:
            pitch_sketch: User-drawn pitch curve [batch, length] or [length]
            energy_sketch: User-drawn energy curve [batch, length] or [length]

        Returns:
            Dict with preprocessed tensors:
            - 'sketch': [batch, num_curves, sketch_length]
            - 'delta': [batch, num_curves, sketch_length]
            - 'sketch_mask': [batch, sketch_length] attention mask
        """
        # Handle unbatched input
        if pitch_sketch.dim() == 1:
            pitch_sketch = pitch_sketch.unsqueeze(0)
        if energy_sketch.dim() == 1:
            energy_sketch = energy_sketch.unsqueeze(0)

        batch_size = pitch_sketch.shape[0]

        # Stack curves
        sketch = torch.stack([pitch_sketch, energy_sketch], dim=1)  # [B, 2, L]

        # Resample to fixed length
        sketch = self.resample(sketch)

        # Smooth
        sketch = self.smooth(sketch)

        # Normalize
        sketch = self.normalize(sketch)

        # Compute delta
        delta = self.compute_delta(sketch)

        # Create attention mask (all ones for fully specified sketch)
        sketch_mask = torch.ones(
            batch_size, self.config.sketch_length,
            device=sketch.device, dtype=torch.bool
        )

        return {
            'sketch': sketch,
            'delta': delta,
            'sketch_mask': sketch_mask,
        }


# =============================================================================
# SKETCH ENCODER
# =============================================================================

class SketchConvBlock(nn.Module):
    """Convolutional block for sketch encoding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection if dimensions match
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, length]
        Returns:
            [batch, out_channels, length]
        """
        residual = self.residual(x)

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x + residual


class SketchEncoder(nn.Module):
    """
    Encodes user-drawn pitch/energy sketches into embeddings.

    Architecture:
    1. Convolutional layers extract local patterns
    2. Transformer layers capture global dependencies
    3. Projection to final embedding space

    The encoder learns to understand:
    - Pitch trends (rising, falling, flat)
    - Energy dynamics (crescendo, decrescendo)
    - Phrase boundaries (pauses, resets)
    - Style patterns (emphatic peaks, smooth curves)
    """

    def __init__(self, config: SketchConfig):
        super().__init__()
        self.config = config

        # Input: [batch, num_curves + num_curves, sketch_length]
        # (original curves + delta encoding)
        input_channels = config.num_curves * 2  # curves + deltas

        # Initial projection
        self.input_proj = nn.Conv1d(
            input_channels,
            config.hidden_dim,
            kernel_size=1,
        )

        # Convolutional layers for local feature extraction
        conv_channels = [config.hidden_dim] * config.num_conv_layers
        self.conv_layers = nn.ModuleList([
            SketchConvBlock(
                in_ch if i == 0 else config.hidden_dim,
                out_ch,
                config.kernel_size,
                config.dropout,
            )
            for i, (in_ch, out_ch) in enumerate(
                zip([config.hidden_dim] + conv_channels[:-1], conv_channels)
            )
        ])

        # Positional encoding for transformer
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.sketch_length, config.hidden_dim) * 0.02
        )

        # Transformer layers for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.sketch_embed_dim),
            nn.LayerNorm(config.sketch_embed_dim),
        )

        # Global sketch embedding (pooled)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(config.sketch_embed_dim, config.sketch_embed_dim),
            nn.LayerNorm(config.sketch_embed_dim),
        )

    def forward(
        self,
        sketch: torch.Tensor,
        delta: torch.Tensor,
        sketch_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode sketch curves into embeddings.

        Args:
            sketch: [batch, num_curves, sketch_length] - normalized curves
            delta: [batch, num_curves, sketch_length] - rate of change
            sketch_mask: [batch, sketch_length] - attention mask

        Returns:
            Dict with:
            - 'sequence': [batch, sketch_length, sketch_embed_dim] - per-position
            - 'global': [batch, sketch_embed_dim] - pooled global embedding
        """
        batch_size = sketch.shape[0]

        # Concatenate sketch and delta
        x = torch.cat([sketch, delta], dim=1)  # [B, num_curves*2, L]

        # Initial projection
        x = self.input_proj(x)  # [B, hidden_dim, L]

        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)

        # Prepare for transformer: [B, L, hidden_dim]
        x = x.transpose(1, 2)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.shape[1], :]

        # Transformer encoding
        if sketch_mask is not None:
            # Convert to transformer format (True = masked)
            src_key_padding_mask = ~sketch_mask
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Output projection
        sequence = self.output_proj(x)  # [B, L, sketch_embed_dim]

        # Global pooling
        global_emb = self.global_pool(
            sequence.transpose(1, 2)
        )  # [B, sketch_embed_dim]

        return {
            'sequence': sequence,
            'global': global_emb,
        }


# =============================================================================
# SKETCH-CONDITIONED VECTOR FIELD
# =============================================================================

class SketchConditionedVectorField(nn.Module):
    """
    Vector field network conditioned on sketch embeddings.

    Extends VectorFieldNetwork to accept sketch conditioning via:
    1. Cross-attention to sketch sequence
    2. Global sketch embedding addition

    This enables the flow to follow the user's drawn curves while
    generating natural, refined prosody.
    """

    def __init__(
        self,
        flow_config: ProsodyFlowConfig,
        sketch_config: SketchConfig,
        conditioning_mode: str = "cross_attention",
    ):
        super().__init__()
        self.flow_config = flow_config
        self.sketch_config = sketch_config
        self.conditioning_mode = conditioning_mode

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(flow_config.time_emb_dim)

        # Input projection
        input_dim = flow_config.prosody_dim + flow_config.time_emb_dim

        # Add sketch global embedding if concat mode
        if conditioning_mode == "concat":
            input_dim += sketch_config.sketch_embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, flow_config.hidden_dim),
            nn.LayerNorm(flow_config.hidden_dim),
            nn.SiLU(),
        )

        # Sketch projection for cross-attention
        self.sketch_proj = nn.Sequential(
            nn.Linear(sketch_config.sketch_embed_dim, flow_config.hidden_dim),
            nn.LayerNorm(flow_config.hidden_dim),
            nn.SiLU(),
        )

        # Text conditioning projection
        if flow_config.use_text_conditioning:
            self.text_proj = nn.Sequential(
                nn.Linear(flow_config.text_dim, flow_config.hidden_dim),
                nn.LayerNorm(flow_config.hidden_dim),
                nn.SiLU(),
            )

        # Transformer layers with dual cross-attention
        self.layers = nn.ModuleList([
            SketchConditionedBlock(
                hidden_dim=flow_config.hidden_dim,
                num_heads=flow_config.num_heads,
                dropout=flow_config.dropout,
                use_text_attention=flow_config.use_text_conditioning,
            )
            for _ in range(flow_config.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(flow_config.hidden_dim),
            nn.Linear(flow_config.hidden_dim, flow_config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(flow_config.dropout),
            nn.Linear(flow_config.hidden_dim, flow_config.prosody_dim),
        )

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        sketch_emb: Dict[str, torch.Tensor],
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        sketch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity conditioned on sketch and text.

        Args:
            t: Time values [batch]
            x_t: Current prosody [batch, prosody_dim]
            sketch_emb: Dict from SketchEncoder with 'sequence' and 'global'
            text_cond: Optional text embeddings [batch, seq_len, text_dim]
            text_mask: Optional text mask [batch, seq_len]
            sketch_mask: Optional sketch mask [batch, sketch_length]

        Returns:
            Predicted velocity [batch, prosody_dim]
        """
        batch_size = x_t.shape[0]

        # Handle scalar t
        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_emb_dim]

        # Build input
        if self.conditioning_mode == "concat":
            x_input = torch.cat([x_t, t_emb, sketch_emb['global']], dim=-1)
        else:
            x_input = torch.cat([x_t, t_emb], dim=-1)

        # Project input
        h = self.input_proj(x_input)  # [B, hidden_dim]
        h = h.unsqueeze(1)  # [B, 1, hidden_dim]

        # Project sketch for cross-attention
        sketch_h = self.sketch_proj(sketch_emb['sequence'])  # [B, L, hidden_dim]

        # Project text if available
        text_h = None
        if text_cond is not None and self.flow_config.use_text_conditioning:
            text_h = self.text_proj(text_cond)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(
                h,
                sketch_h=sketch_h,
                text_h=text_h,
                sketch_mask=sketch_mask,
                text_mask=text_mask,
            )

        # Output
        h = h.squeeze(1)
        velocity = self.output_proj(h)

        return velocity


class SketchConditionedBlock(nn.Module):
    """
    Transformer block with dual cross-attention to sketch and text.

    Order:
    1. Self-attention
    2. Cross-attention to sketch
    3. Cross-attention to text (optional)
    4. Feed-forward
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_text_attention: bool = True,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to sketch
        self.sketch_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.sketch_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to text
        self.use_text_attention = use_text_attention
        if use_text_attention:
            self.text_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.text_attn_norm = nn.LayerNorm(hidden_dim)

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
        x: torch.Tensor,
        sketch_h: torch.Tensor,
        text_h: Optional[torch.Tensor] = None,
        sketch_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, hidden_dim]
            sketch_h: [batch, sketch_len, hidden_dim]
            text_h: Optional [batch, text_len, hidden_dim]
            sketch_mask: [batch, sketch_len]
            text_mask: [batch, text_len]
        """
        # Self-attention
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention to sketch
        x_norm = self.sketch_attn_norm(x)
        key_padding_mask = ~sketch_mask if sketch_mask is not None else None
        sketch_out, _ = self.sketch_attn(
            x_norm, sketch_h, sketch_h,
            key_padding_mask=key_padding_mask
        )
        x = x + sketch_out

        # Cross-attention to text
        if self.use_text_attention and text_h is not None:
            x_norm = self.text_attn_norm(x)
            key_padding_mask = ~text_mask if text_mask is not None else None
            text_out, _ = self.text_attn(
                x_norm, text_h, text_h,
                key_padding_mask=key_padding_mask
            )
            x = x + text_out

        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))

        return x


# =============================================================================
# DRAWSPEECH MODEL
# =============================================================================

class DrawSpeech(nn.Module):
    """
    DrawSpeech: Sketch-Conditioned Prosody Generation.

    Main model that combines:
    1. SketchEncoder - Encodes user-drawn curves
    2. SketchConditionedVectorField - Flow network with sketch conditioning
    3. Classifier-free guidance for controllable generation

    Usage:
        model = DrawSpeech(config)

        # Training
        loss = model.compute_loss(prosody_target, pitch_sketch, energy_sketch, text)

        # Inference
        prosody = model.sample(pitch_sketch, energy_sketch, text, cfg_scale=2.0)
    """

    def __init__(self, config: DrawSpeechConfig):
        super().__init__()
        self.config = config

        # Sketch preprocessor
        self.preprocessor = SketchPreprocessor(config.sketch)

        # Sketch encoder
        self.sketch_encoder = SketchEncoder(config.sketch)

        # Sketch-conditioned vector field
        self.vector_field = SketchConditionedVectorField(
            config.flow,
            config.sketch,
            conditioning_mode=config.sketch_conditioning,
        )

        # Conditional path
        self.path = GaussianConditionalPath(config.flow.sigma_min)

        # OT coupling
        if config.flow.use_ot_coupling:
            self.ot_coupler = OptimalTransportCoupling(
                reg=config.flow.ot_reg,
                normalize_cost=config.flow.ot_normalize_cost,
            )
        else:
            self.ot_coupler = None

        # Token projection (for compatibility with prosody pipeline)
        self.token_projection = nn.Sequential(
            nn.Linear(config.flow.prosody_dim, config.flow.prosody_dim),
            nn.LayerNorm(config.flow.prosody_dim),
            nn.GELU(),
            nn.Linear(
                config.flow.prosody_dim,
                config.flow.prosody_dim * config.flow.num_prosody_tokens
            ),
        )
        self.token_norm = nn.LayerNorm(config.flow.prosody_dim)

        # Null sketch embedding for CFG
        self.register_buffer(
            'null_sketch_sequence',
            torch.zeros(1, config.sketch.sketch_length, config.sketch.sketch_embed_dim)
        )
        self.register_buffer(
            'null_sketch_global',
            torch.zeros(1, config.sketch.sketch_embed_dim)
        )

    def encode_sketch(
        self,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Preprocess and encode sketch curves.

        Args:
            pitch_sketch: [batch, length] or [length]
            energy_sketch: [batch, length] or [length]

        Returns:
            (sketch_emb, sketch_mask)
        """
        # Preprocess
        preprocessed = self.preprocessor.preprocess(pitch_sketch, energy_sketch)

        # Encode
        sketch_emb = self.sketch_encoder(
            preprocessed['sketch'],
            preprocessed['delta'],
            preprocessed['sketch_mask'],
        )

        return sketch_emb, preprocessed['sketch_mask']

    def compute_loss(
        self,
        prosody_target: torch.Tensor,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with classifier-free guidance dropout.

        Args:
            prosody_target: Target prosody [batch, prosody_dim]
            pitch_sketch: Pitch sketch [batch, length]
            energy_sketch: Energy sketch [batch, length]
            text_cond: Optional text embeddings [batch, seq_len, text_dim]
            text_mask: Optional text mask

        Returns:
            Loss dict
        """
        batch_size = prosody_target.shape[0]
        device = prosody_target.device

        # Encode sketch
        sketch_emb, sketch_mask = self.encode_sketch(pitch_sketch, energy_sketch)

        # CFG dropout: randomly drop conditioning
        if self.training and self.config.use_cfg:
            drop_mask = torch.rand(batch_size, device=device) < self.config.cfg_dropout

            # Replace with null embeddings where dropped
            null_seq = self.null_sketch_sequence.expand(batch_size, -1, -1)
            null_global = self.null_sketch_global.expand(batch_size, -1)

            sketch_emb['sequence'] = torch.where(
                drop_mask[:, None, None],
                null_seq,
                sketch_emb['sequence'],
            )
            sketch_emb['global'] = torch.where(
                drop_mask[:, None],
                null_global,
                sketch_emb['global'],
            )

        # Sample noise
        x0 = torch.randn_like(prosody_target)
        x1 = prosody_target

        # OT coupling
        if self.ot_coupler is not None and batch_size > 1:
            x0, x1 = self.ot_coupler.get_coupling(x0, x1)

        # Sample time
        t = torch.rand(batch_size, device=device)

        # Sample x_t
        x_t = self.path.sample_xt(t, x0, x1)

        # Target velocity
        target_velocity = self.path.compute_target_velocity(t, x_t, x1)

        # Predict velocity
        predicted_velocity = self.vector_field(
            t, x_t, sketch_emb,
            text_cond=text_cond,
            text_mask=text_mask,
            sketch_mask=sketch_mask,
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
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        cfg_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample prosody conditioned on sketches.

        Args:
            pitch_sketch: Pitch curve [batch, length] or [length]
            energy_sketch: Energy curve [batch, length] or [length]
            text_cond: Optional text embeddings
            text_mask: Optional text mask
            num_samples: Number of samples per input
            temperature: Sampling temperature
            cfg_scale: Classifier-free guidance scale (None = config default)
            num_steps: ODE steps (None = config default)

        Returns:
            Sampled prosody [batch * num_samples, prosody_dim]
        """
        # Handle unbatched input
        if pitch_sketch.dim() == 1:
            pitch_sketch = pitch_sketch.unsqueeze(0)
        if energy_sketch.dim() == 1:
            energy_sketch = energy_sketch.unsqueeze(0)

        batch_size = pitch_sketch.shape[0]
        device = pitch_sketch.device

        # Encode sketch
        sketch_emb, sketch_mask = self.encode_sketch(pitch_sketch, energy_sketch)

        # Expand for multiple samples
        if num_samples > 1:
            sketch_emb['sequence'] = sketch_emb['sequence'].repeat_interleave(
                num_samples, dim=0
            )
            sketch_emb['global'] = sketch_emb['global'].repeat_interleave(
                num_samples, dim=0
            )
            sketch_mask = sketch_mask.repeat_interleave(num_samples, dim=0)

            if text_cond is not None:
                text_cond = text_cond.repeat_interleave(num_samples, dim=0)
            if text_mask is not None:
                text_mask = text_mask.repeat_interleave(num_samples, dim=0)

            batch_size = batch_size * num_samples

        # Sample initial noise
        x0 = torch.randn(
            batch_size, self.config.flow.prosody_dim,
            device=device
        ) * temperature

        # ODE integration with CFG
        cfg = cfg_scale if cfg_scale is not None else self.config.cfg_scale
        steps = num_steps if num_steps is not None else self.config.flow.num_ode_steps_inference

        if self.config.use_cfg and cfg > 1.0:
            x = self._sample_with_cfg(
                x0, sketch_emb, sketch_mask,
                text_cond, text_mask,
                steps, cfg,
            )
        else:
            x = self._sample_unconditional(
                x0, sketch_emb, sketch_mask,
                text_cond, text_mask,
                steps,
            )

        return x

    def _sample_unconditional(
        self,
        x0: torch.Tensor,
        sketch_emb: Dict[str, torch.Tensor],
        sketch_mask: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        num_steps: int,
    ) -> torch.Tensor:
        """Sample without CFG."""
        x = x0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=x.device)
            velocity = self.vector_field(
                t, x, sketch_emb,
                text_cond=text_cond,
                text_mask=text_mask,
                sketch_mask=sketch_mask,
            )
            x = x + dt * velocity

        return x

    def _sample_with_cfg(
        self,
        x0: torch.Tensor,
        sketch_emb: Dict[str, torch.Tensor],
        sketch_mask: torch.Tensor,
        text_cond: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        num_steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Sample with classifier-free guidance."""
        batch_size = x0.shape[0]
        x = x0
        dt = 1.0 / num_steps

        # Null embeddings for unconditional
        null_emb = {
            'sequence': self.null_sketch_sequence.expand(batch_size, -1, -1),
            'global': self.null_sketch_global.expand(batch_size, -1),
        }

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=x.device)

            # Conditional velocity
            v_cond = self.vector_field(
                t, x, sketch_emb,
                text_cond=text_cond,
                text_mask=text_mask,
                sketch_mask=sketch_mask,
            )

            # Unconditional velocity
            v_uncond = self.vector_field(
                t, x, null_emb,
                text_cond=text_cond,
                text_mask=text_mask,
                sketch_mask=sketch_mask,
            )

            # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond)

            x = x + dt * velocity

        return x

    def sample_tokens(
        self,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        temperature: float = 1.0,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample prosody and project to prefix tokens.

        Returns:
            Prosody tokens [batch * num_samples, num_tokens, prosody_dim]
        """
        # Sample prosody
        prosody = self.sample(
            pitch_sketch, energy_sketch,
            text_cond=text_cond,
            text_mask=text_mask,
            num_samples=num_samples,
            temperature=temperature,
            cfg_scale=cfg_scale,
        )

        # Project to tokens
        tokens = self.token_projection(prosody)
        tokens = tokens.view(
            -1,
            self.config.flow.num_prosody_tokens,
            self.config.flow.prosody_dim
        )
        tokens = self.token_norm(tokens)

        return tokens


# =============================================================================
# DRAWSPEECH ADAPTER
# =============================================================================

class DrawSpeechAdapter(nn.Module):
    """
    Adapter for integrating DrawSpeech with existing prosody pipeline.

    Provides compatibility with:
    - ProsodyControlledCSM
    - ProsodyFlowAdapter
    - Existing training infrastructure

    Usage:
        adapter = DrawSpeechAdapter(config)

        # From sketches directly
        tokens = adapter.from_sketch(pitch_sketch, energy_sketch, text_emb)

        # From keyframes (converted to sketches)
        tokens = adapter.from_keyframes(keyframes, duration, text_emb)
    """

    def __init__(
        self,
        config: DrawSpeechConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core model
        self.draw_speech = DrawSpeech(config)

        # Adapt to prosody hidden dimension
        if config.flow.prosody_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.flow.prosody_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def from_sketch(
        self,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **sample_kwargs,
    ) -> torch.Tensor:
        """
        Generate prosody tokens from sketches.

        Args:
            pitch_sketch: Pitch curve [batch, length]
            energy_sketch: Energy curve [batch, length]
            text_cond: Text embeddings
            text_mask: Text mask
            **sample_kwargs: Passed to sample_tokens()

        Returns:
            Prosody tokens [batch, num_tokens, prosody_hidden]
        """
        tokens = self.draw_speech.sample_tokens(
            pitch_sketch, energy_sketch,
            text_cond=text_cond,
            text_mask=text_mask,
            **sample_kwargs,
        )

        return self.output_adapter(tokens)

    def from_keyframes(
        self,
        keyframes: List[Dict],
        duration_seconds: float,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        **sample_kwargs,
    ) -> torch.Tensor:
        """
        Generate prosody tokens from keyframes.

        Converts keyframes to pitch/energy sketches, then generates.

        Args:
            keyframes: List of keyframe dicts with:
                - 'time': Normalized time [0, 1]
                - 'pitch': Target pitch value (normalized)
                - 'energy': Target energy value (normalized)
            duration_seconds: Total duration
            text_cond: Text embeddings
            text_mask: Text mask

        Returns:
            Prosody tokens [batch, num_tokens, prosody_hidden]
        """
        device = text_cond.device if text_cond is not None else 'cpu'

        # Convert keyframes to sketch curves
        sketch_length = self.config.sketch.sketch_length
        pitch_sketch = torch.zeros(sketch_length, device=device)
        energy_sketch = torch.zeros(sketch_length, device=device)

        if not keyframes:
            # Default: flat curves at 0.5
            pitch_sketch.fill_(0.5)
            energy_sketch.fill_(0.5)
        else:
            # Sort by time
            keyframes = sorted(keyframes, key=lambda k: k.get('time', 0))

            # Interpolate between keyframes
            times = torch.tensor([k.get('time', 0) for k in keyframes], device=device)
            pitches = torch.tensor([k.get('pitch', 0.5) for k in keyframes], device=device)
            energies = torch.tensor([k.get('energy', 0.5) for k in keyframes], device=device)

            # Linear interpolation to sketch length
            x = torch.linspace(0, 1, sketch_length, device=device)

            for i, xi in enumerate(x):
                # Find surrounding keyframes
                idx = torch.searchsorted(times, xi)

                if idx == 0:
                    pitch_sketch[i] = pitches[0]
                    energy_sketch[i] = energies[0]
                elif idx >= len(times):
                    pitch_sketch[i] = pitches[-1]
                    energy_sketch[i] = energies[-1]
                else:
                    # Linear interpolation
                    t0, t1 = times[idx - 1], times[idx]
                    alpha = (xi - t0) / (t1 - t0 + 1e-8)

                    pitch_sketch[i] = (1 - alpha) * pitches[idx - 1] + alpha * pitches[idx]
                    energy_sketch[i] = (1 - alpha) * energies[idx - 1] + alpha * energies[idx]

        # Add batch dimension
        pitch_sketch = pitch_sketch.unsqueeze(0)
        energy_sketch = energy_sketch.unsqueeze(0)

        return self.from_sketch(
            pitch_sketch, energy_sketch,
            text_cond=text_cond,
            text_mask=text_mask,
            **sample_kwargs,
        )

    def compute_loss(
        self,
        prosody_target: torch.Tensor,
        pitch_sketch: torch.Tensor,
        energy_sketch: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss."""
        return self.draw_speech.compute_loss(
            prosody_target, pitch_sketch, energy_sketch,
            text_cond=text_cond,
            text_mask=text_mask,
        )


# =============================================================================
# SKETCH-TO-PROSODY CONVERSION UTILITIES
# =============================================================================

def prosody_to_sketch(
    prosody_dict: Dict[str, torch.Tensor],
    sketch_length: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract sketch curves from prosody features.

    Useful for creating training pairs: actual prosody → ground truth sketches.

    Args:
        prosody_dict: Prosody features dict with 'contour' and 'acoustic'
        sketch_length: Target sketch length

    Returns:
        (pitch_sketch, energy_sketch) tensors
    """
    device = prosody_dict.get('contour', torch.zeros(1)).device

    # Extract pitch from contour
    if 'contour' in prosody_dict:
        contour = prosody_dict['contour']
        if contour.dim() == 1:
            contour = contour.unsqueeze(0)

        # Resample to sketch length
        pitch_sketch = F.interpolate(
            contour.unsqueeze(1),
            size=sketch_length,
            mode='linear',
            align_corners=True,
        ).squeeze(1)
    else:
        pitch_sketch = torch.ones(1, sketch_length, device=device) * 0.5

    # Extract energy from acoustic
    if 'acoustic' in prosody_dict:
        acoustic = prosody_dict['acoustic']
        if acoustic.dim() == 1:
            acoustic = acoustic.unsqueeze(0)

        # Energy is typically index 2 in acoustic features
        energy_value = acoustic[:, 2:3] if acoustic.shape[-1] > 2 else acoustic[:, :1]

        # Create constant energy curve (or could modulate based on contour)
        energy_sketch = energy_value.expand(-1, sketch_length)
    else:
        energy_sketch = torch.ones(1, sketch_length, device=device) * 0.5

    return pitch_sketch, energy_sketch


def sketch_from_emotion_profile(
    emotion: str,
    intensity: float = 1.0,
    sketch_length: int = 100,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate canonical sketch curves for an emotion.

    Args:
        emotion: Emotion name (happy, sad, angry, etc.)
        intensity: Intensity scaling [0, 1]
        sketch_length: Sketch length
        device: Target device

    Returns:
        (pitch_sketch, energy_sketch) tensors
    """
    t = torch.linspace(0, 1, sketch_length, device=device)

    # Emotion-specific patterns
    patterns = {
        'happy': {
            'pitch_base': 0.6,
            'pitch_variation': 0.15,
            'pitch_pattern': lambda t: torch.sin(t * 4 * math.pi),
            'energy_base': 0.7,
            'energy_trend': 0.0,
        },
        'sad': {
            'pitch_base': 0.4,
            'pitch_variation': 0.1,
            'pitch_pattern': lambda t: -t,  # Falling
            'energy_base': 0.3,
            'energy_trend': -0.1,
        },
        'angry': {
            'pitch_base': 0.55,
            'pitch_variation': 0.2,
            'pitch_pattern': lambda t: torch.sin(t * 6 * math.pi),  # Rapid
            'energy_base': 0.85,
            'energy_trend': 0.05,
        },
        'surprised': {
            'pitch_base': 0.5,
            'pitch_variation': 0.25,
            'pitch_pattern': lambda t: t,  # Rising
            'energy_base': 0.6,
            'energy_trend': 0.1,
        },
        'calm': {
            'pitch_base': 0.5,
            'pitch_variation': 0.05,
            'pitch_pattern': lambda t: torch.zeros_like(t),  # Flat
            'energy_base': 0.4,
            'energy_trend': 0.0,
        },
        'fearful': {
            'pitch_base': 0.55,
            'pitch_variation': 0.15,
            'pitch_pattern': lambda t: torch.sin(t * 8 * math.pi) * (1 - t),  # Trembling
            'energy_base': 0.5,
            'energy_trend': -0.05,
        },
        'neutral': {
            'pitch_base': 0.5,
            'pitch_variation': 0.08,
            'pitch_pattern': lambda t: torch.sin(t * 2 * math.pi) * 0.5,
            'energy_base': 0.5,
            'energy_trend': 0.0,
        },
    }

    pattern = patterns.get(emotion.lower(), patterns['neutral'])

    # Generate pitch sketch
    pitch_sketch = (
        pattern['pitch_base'] +
        pattern['pitch_variation'] * intensity * pattern['pitch_pattern'](t)
    )

    # Generate energy sketch
    energy_sketch = (
        pattern['energy_base'] + pattern['energy_trend'] * t
    ) * torch.ones_like(t)

    # Scale by intensity
    pitch_sketch = 0.5 + (pitch_sketch - 0.5) * intensity
    energy_sketch = 0.5 + (energy_sketch - 0.5) * intensity

    # Clamp to valid range
    pitch_sketch = torch.clamp(pitch_sketch, 0, 1)
    energy_sketch = torch.clamp(energy_sketch, 0, 1)

    return pitch_sketch.unsqueeze(0), energy_sketch.unsqueeze(0)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DrawSpeech: Sketch-Conditioned Prosody Control - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create config
    config = DrawSpeechConfig()

    # Test 1: Sketch Preprocessor
    print("\n[Test 1] Sketch Preprocessor...")
    preprocessor = SketchPreprocessor(config.sketch)

    # Create random sketches of varying lengths
    pitch = torch.randn(50)  # Variable length
    energy = torch.randn(50)

    preprocessed = preprocessor.preprocess(pitch, energy)
    print(f"  Input length: 50")
    print(f"  Output sketch shape: {preprocessed['sketch'].shape}")
    print(f"  Output delta shape: {preprocessed['delta'].shape}")
    print(f"  Output mask shape: {preprocessed['sketch_mask'].shape}")
    assert preprocessed['sketch'].shape == (1, 2, config.sketch.sketch_length)
    print("  [PASS]")

    # Test 2: Sketch Encoder
    print("\n[Test 2] Sketch Encoder...")
    encoder = SketchEncoder(config.sketch).to(device)

    batch_size = 2
    sketch = torch.randn(batch_size, 2, config.sketch.sketch_length, device=device)
    delta = torch.randn(batch_size, 2, config.sketch.sketch_length, device=device)
    mask = torch.ones(batch_size, config.sketch.sketch_length, dtype=torch.bool, device=device)

    emb = encoder(sketch, delta, mask)
    print(f"  Input sketch shape: {sketch.shape}")
    print(f"  Sequence embedding shape: {emb['sequence'].shape}")
    print(f"  Global embedding shape: {emb['global'].shape}")
    assert emb['sequence'].shape == (batch_size, config.sketch.sketch_length, config.sketch.sketch_embed_dim)
    assert emb['global'].shape == (batch_size, config.sketch.sketch_embed_dim)
    print("  [PASS]")

    # Test 3: Sketch-Conditioned Vector Field
    print("\n[Test 3] Sketch-Conditioned Vector Field...")
    vector_field = SketchConditionedVectorField(
        config.flow, config.sketch, "cross_attention"
    ).to(device)

    t = torch.rand(batch_size, device=device)
    x_t = torch.randn(batch_size, config.flow.prosody_dim, device=device)
    text = torch.randn(batch_size, 10, config.flow.text_dim, device=device)
    text_mask = torch.ones(batch_size, 10, dtype=torch.bool, device=device)

    velocity = vector_field(t, x_t, emb, text, text_mask, mask)
    print(f"  Input x_t shape: {x_t.shape}")
    print(f"  Output velocity shape: {velocity.shape}")
    assert velocity.shape == x_t.shape
    print("  [PASS]")

    # Test 4: DrawSpeech Model
    print("\n[Test 4] DrawSpeech Model...")
    model = DrawSpeech(config).to(device)

    pitch_sketch = torch.randn(batch_size, 80, device=device)
    energy_sketch = torch.randn(batch_size, 80, device=device)
    prosody_target = torch.randn(batch_size, config.flow.prosody_dim, device=device)

    # Training loss
    loss_output = model.compute_loss(
        prosody_target, pitch_sketch, energy_sketch, text, text_mask
    )
    print(f"  Training loss: {loss_output['loss'].item():.4f}")

    # Sampling
    samples = model.sample(
        pitch_sketch, energy_sketch, text, text_mask,
        num_samples=1, cfg_scale=2.0
    )
    print(f"  Sampled prosody shape: {samples.shape}")
    assert samples.shape == (batch_size, config.flow.prosody_dim)
    print("  [PASS]")

    # Test 5: Sample Tokens
    print("\n[Test 5] Sample Tokens...")
    tokens = model.sample_tokens(
        pitch_sketch, energy_sketch, text, text_mask,
        num_samples=2
    )
    print(f"  Token shape: {tokens.shape}")
    expected = (batch_size * 2, config.flow.num_prosody_tokens, config.flow.prosody_dim)
    assert tokens.shape == expected
    print("  [PASS]")

    # Test 6: DrawSpeech Adapter
    print("\n[Test 6] DrawSpeech Adapter...")
    adapter = DrawSpeechAdapter(config).to(device)

    # From sketch
    tokens = adapter.from_sketch(pitch_sketch, energy_sketch, text, text_mask)
    print(f"  From sketch - token shape: {tokens.shape}")

    # From keyframes
    keyframes = [
        {'time': 0.0, 'pitch': 0.5, 'energy': 0.5},
        {'time': 0.3, 'pitch': 0.8, 'energy': 0.7},
        {'time': 0.7, 'pitch': 0.4, 'energy': 0.3},
        {'time': 1.0, 'pitch': 0.5, 'energy': 0.5},
    ]
    tokens = adapter.from_keyframes(keyframes, 3.0, text[:1], text_mask[:1])
    print(f"  From keyframes - token shape: {tokens.shape}")
    print("  [PASS]")

    # Test 7: Emotion Sketch Generation
    print("\n[Test 7] Emotion Sketch Generation...")
    emotions = ['happy', 'sad', 'angry', 'calm']
    for emotion in emotions:
        pitch, energy = sketch_from_emotion_profile(emotion, intensity=0.8, device=device)
        print(f"  {emotion}: pitch range [{pitch.min():.2f}, {pitch.max():.2f}], "
              f"energy range [{energy.min():.2f}, {energy.max():.2f}]")
    print("  [PASS]")

    # Test 8: Prosody to Sketch Conversion
    print("\n[Test 8] Prosody to Sketch Conversion...")
    prosody_dict = {
        'contour': torch.randn(64, device=device),
        'acoustic': torch.randn(12, device=device),
    }
    pitch, energy = prosody_to_sketch(prosody_dict)
    print(f"  Pitch sketch shape: {pitch.shape}")
    print(f"  Energy sketch shape: {energy.shape}")
    print("  [PASS]")

    # Test 9: CFG Sampling
    print("\n[Test 9] Classifier-Free Guidance...")
    cfg_scales = [1.0, 2.0, 4.0]
    for cfg in cfg_scales:
        samples = model.sample(
            pitch_sketch[:1], energy_sketch[:1],
            text[:1], text_mask[:1],
            cfg_scale=cfg,
        )
        print(f"  CFG scale {cfg}: sample norm = {samples.norm().item():.3f}")
    print("  [PASS]")

    # Test 10: Multiple Samples
    print("\n[Test 10] Multiple Samples...")
    samples = model.sample(
        pitch_sketch[:1], energy_sketch[:1],
        text[:1], text_mask[:1],
        num_samples=5,
    )
    print(f"  5 samples shape: {samples.shape}")
    print(f"  Sample diversity (std): {samples.std(dim=0).mean():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All DrawSpeech tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from draw_speech import DrawSpeechConfig, DrawSpeech, DrawSpeechAdapter

# Initialize
config = DrawSpeechConfig()
model = DrawSpeech(config).cuda()

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    prosody_target = batch['prosody']
    pitch_sketch = batch['pitch_curve']  # User-drawn or extracted
    energy_sketch = batch['energy_curve']
    text_emb = batch['text_embeddings']

    loss_output = model.compute_loss(
        prosody_target, pitch_sketch, energy_sketch, text_emb
    )

    optimizer.zero_grad()
    loss_output['loss'].backward()
    optimizer.step()

# Inference - from user sketches
pitch_curve = torch.tensor([0.5, 0.6, 0.8, 0.7, 0.5])  # Rising then falling
energy_curve = torch.tensor([0.4, 0.5, 0.7, 0.6, 0.4])  # Emphasis in middle

prosody_tokens = model.sample_tokens(
    pitch_curve, energy_curve,
    text_embeddings,
    cfg_scale=2.0,  # Stronger adherence to sketch
)

# Generate audio with sketch-conditioned prosody
audio = csm_model.generate_with_prosody(
    text_input, prosody_tokens
)

# Using the adapter for keyframe-based control
adapter = DrawSpeechAdapter(config)

keyframes = [
    {'time': 0.0, 'pitch': 0.5, 'energy': 0.5},   # Start neutral
    {'time': 0.4, 'pitch': 0.8, 'energy': 0.8},   # Build up
    {'time': 0.6, 'pitch': 0.3, 'energy': 0.3},   # Drop
    {'time': 1.0, 'pitch': 0.5, 'energy': 0.5},   # Return to neutral
]

tokens = adapter.from_keyframes(keyframes, duration=3.0, text_cond=text_emb)
""")
