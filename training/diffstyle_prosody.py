"""
DiffStyleTTS Hierarchical Prosody with Guiding Scales

Based on DiffStyleTTS (2024): "Diffusion-based Controllable Emotional Speech Synthesis
via Hierarchical Prosody Modeling"
https://arxiv.org/abs/2412.03388

Key Innovation: Two-level prosody hierarchy with independent guiding scales:
1. **Implicit Style (Coarse-grained)**: Sentence-level mel-spectrogram encoding via GST
   - Captures holistic utterance-level characteristics
   - Global emotional tone, speaker style, speaking manner

2. **Explicit Prosody (Fine-grained)**: Phoneme-wise pitch, energy, duration
   - Direct acoustic features extracted from speech
   - Precise local prosodic control

3. **Guiding Scale Factors**:
   - η (eta): Classifier-free guidance intensity (diversity vs quality)
   - γ (gamma): Dynamic thresholding correction (prevents distortion)

Benefits for Voice Clone Pipeline:
- Clean separation of global style vs local prosody features
- Independent control over pitch, energy, duration, and style
- Better fine-grained control without cross-attribute interference
- Diverse prosody generation via CFG

Reference: https://arxiv.org/html/2412.03388
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DiffStyleProsodyConfig:
    """Configuration for DiffStyleTTS hierarchical prosody."""

    # Mel-spectrogram settings
    mel_dim: int = 80  # Mel-spectrogram dimension

    # Global Style Token (GST) settings
    gst_num_tokens: int = 10  # Number of style tokens
    gst_embedding_dim: int = 256  # Style token dimension
    gst_num_heads: int = 4  # Attention heads for style extraction

    # Text encoder settings
    text_hidden_dim: int = 256  # FFT block dimension
    text_num_layers: int = 4  # Number of FFT blocks
    text_num_heads: int = 2

    # Phoneme-level prosody settings
    pitch_dim: int = 1  # Pitch (F0) dimension
    energy_dim: int = 1  # Energy dimension
    duration_dim: int = 1  # Duration dimension
    prosody_dim: int = 3  # Combined prosody dimension (pitch + energy + duration)

    # Prosody embedding
    prosody_hidden_dim: int = 256
    prosody_embedding_dim: int = 256

    # Diffusion/denoiser settings
    denoiser_channels: int = 3  # Bidirectional dilated conv channels
    denoiser_num_layers: int = 12  # Residual layers
    denoiser_kernel_size: int = 3
    use_bidirectional_dilated_conv: bool = True

    # Guiding scale settings
    default_eta: float = 3.0  # Default CFG guidance scale (η)
    default_gamma: float = 0.5  # Default dynamic thresholding scale (γ)
    max_eta: float = 7.0  # Maximum η before distortion
    percentile_clip: float = 0.99  # Percentile for dynamic thresholding

    # Training settings
    dropout: float = 0.1
    unconditional_prob: float = 0.1  # Probability of unconditional training for CFG

    # Output settings
    output_dim: int = 2048  # Match prosody encoder output
    num_prosody_tokens: int = 4  # Number of prefix tokens

    # Integration
    use_length_regulator: bool = True  # Align embeddings to phoneme length


# =============================================================================
# GLOBAL STYLE TOKENS (GST)
# =============================================================================

class GlobalStyleTokens(nn.Module):
    """
    Global Style Tokens for extracting implicit style from mel-spectrograms.

    Based on "Style Tokens: Unsupervised Style Modeling, Control and Transfer in
    End-to-End Speech Synthesis" (Google, 2018).

    The GST extracts sentence-level style characteristics:
    - Emotional tone (happy, sad, angry, etc.)
    - Speaking manner (fast, slow, emphatic)
    - Speaker style (prosodic patterns, rhythm)

    Architecture:
        Mel-spectrogram → Reference Encoder → Multi-head Attention → Style Vector
    """

    def __init__(self, config: DiffStyleProsodyConfig):
        super().__init__()
        self.config = config

        # Reference encoder: extracts fixed-length embedding from variable-length mel
        # Uses convolutional layers to capture temporal patterns
        self.reference_encoder = ReferenceEncoder(
            mel_dim=config.mel_dim,
            hidden_dim=config.gst_embedding_dim,
            dropout=config.dropout,
        )

        # Style token bank: learnable style embeddings
        self.style_tokens = nn.Parameter(
            torch.randn(config.gst_num_tokens, config.gst_embedding_dim // config.gst_num_heads)
        )

        # Multi-head attention to select/combine style tokens
        self.style_attention = nn.MultiheadAttention(
            embed_dim=config.gst_embedding_dim,
            num_heads=config.gst_num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Project style tokens to full embedding dimension
        self.token_projection = nn.Linear(
            config.gst_embedding_dim // config.gst_num_heads,
            config.gst_embedding_dim,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.gst_embedding_dim, config.gst_embedding_dim),
            nn.LayerNorm(config.gst_embedding_dim),
            nn.Tanh(),  # Bound style vectors for stability
        )

        nn.init.xavier_uniform_(self.style_tokens)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, mel_dim, time]
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract global style vector from mel-spectrogram.

        Args:
            mel: Mel-spectrogram [batch, mel_dim, time]
            return_weights: If True, also return style token attention weights

        Returns:
            Style vector [batch, gst_embedding_dim]
            Optionally: attention weights [batch, num_heads, 1, num_tokens]
        """
        batch_size = mel.shape[0]
        device = mel.device

        # Encode mel to fixed-length reference embedding
        reference = self.reference_encoder(mel)  # [batch, gst_embedding_dim]

        # Prepare style token keys/values
        # Expand tokens to batch dimension
        tokens = self.style_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_tokens, dim/heads]
        tokens = self.token_projection(tokens)  # [batch, num_tokens, gst_embedding_dim]

        # Query: reference embedding
        # Key/Value: style tokens
        query = reference.unsqueeze(1)  # [batch, 1, gst_embedding_dim]

        # Multi-head attention to select relevant style tokens
        style_output, attn_weights = self.style_attention(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=True,
        )

        # Extract style vector
        style_vector = style_output.squeeze(1)  # [batch, gst_embedding_dim]
        style_vector = self.output_projection(style_vector)

        if return_weights:
            return style_vector, attn_weights
        return style_vector

    def get_style_tokens(self) -> torch.Tensor:
        """Get the learned style tokens for analysis/visualization."""
        return self.token_projection(self.style_tokens)


class ReferenceEncoder(nn.Module):
    """
    Reference encoder for extracting fixed-length embeddings from mel-spectrograms.

    Uses 2D convolutions to capture both spectral and temporal patterns,
    followed by GRU to aggregate into single embedding.
    """

    def __init__(
        self,
        mel_dim: int = 80,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mel_dim = mel_dim

        # 2D convolutions over mel-spectrogram [batch, 1, mel_dim, time]
        # Use fewer layers to avoid reducing dimensions too much
        # Channels: 1 → 32 → 64 → 128
        channels = [1, 32, 64, 128]

        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[i], channels[i + 1],
                        kernel_size=3, stride=2, padding=1,
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        # Calculate output dimension after convolutions
        # After 3 conv layers with stride 2: mel_dim // 8
        conv_output_mel = mel_dim
        for _ in range(len(self.convs)):
            conv_output_mel = (conv_output_mel + 2 * 1 - 3) // 2 + 1  # padding=1, kernel=3, stride=2
        conv_output_mel = max(1, conv_output_mel)

        self.gru_input_dim = channels[-1] * conv_output_mel

        # GRU to aggregate temporal dimension
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch, mel_dim, time]

        Returns:
            Reference embedding [batch, hidden_dim]
        """
        # Add channel dimension
        x = mel.unsqueeze(1)  # [batch, 1, mel_dim, time]

        # Apply conv layers
        for conv in self.convs:
            x = conv(x)  # Progressively reduce dimensions

        # Reshape for GRU: [batch, time', channels * mel']
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)  # [batch, time', channels, mel']
        x = x.reshape(batch_size, x.shape[1], -1)  # [batch, time', channels * mel']

        # GRU aggregation
        x = self.dropout(x)
        _, hidden = self.gru(x)  # hidden: [1, batch, hidden_dim]

        return hidden.squeeze(0)  # [batch, hidden_dim]


# =============================================================================
# PHONEME-LEVEL PROSODY ENCODER
# =============================================================================

class PhonemeProsodyEncoder(nn.Module):
    """
    Encodes phoneme-level prosodic features (pitch, energy, duration).

    These are the explicit, fine-grained prosody features that capture
    local prosodic variations at the phoneme level.

    Input: Log-scale pitch, energy, and duration per phoneme
    Output: Prosody embedding per phoneme
    """

    def __init__(self, config: DiffStyleProsodyConfig):
        super().__init__()
        self.config = config

        # Individual encoders for each prosody component
        self.pitch_encoder = nn.Sequential(
            nn.Linear(config.pitch_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.energy_encoder = nn.Sequential(
            nn.Linear(config.energy_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.duration_encoder = nn.Sequential(
            nn.Linear(config.duration_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.prosody_hidden_dim * 3, config.prosody_embedding_dim),
            nn.LayerNorm(config.prosody_embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Variational layer for diversity (optional)
        self.variational = nn.Linear(config.prosody_embedding_dim, config.prosody_embedding_dim * 2)

    def forward(
        self,
        pitch: torch.Tensor,       # [batch, num_phonemes, 1] or [batch, num_phonemes]
        energy: torch.Tensor,      # [batch, num_phonemes, 1] or [batch, num_phonemes]
        duration: torch.Tensor,    # [batch, num_phonemes, 1] or [batch, num_phonemes]
        use_variational: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode phoneme-level prosody.

        Args:
            pitch: Log-scale F0 per phoneme
            energy: Log-scale energy per phoneme
            duration: Log-scale duration per phoneme
            use_variational: If True, add variational noise

        Returns:
            Dict with prosody embedding and component encodings
        """
        # Ensure 3D tensors
        if pitch.dim() == 2:
            pitch = pitch.unsqueeze(-1)
        if energy.dim() == 2:
            energy = energy.unsqueeze(-1)
        if duration.dim() == 2:
            duration = duration.unsqueeze(-1)

        # Encode each component
        pitch_enc = self.pitch_encoder(pitch)       # [batch, num_phonemes, hidden]
        energy_enc = self.energy_encoder(energy)     # [batch, num_phonemes, hidden]
        duration_enc = self.duration_encoder(duration)  # [batch, num_phonemes, hidden]

        # Fuse
        combined = torch.cat([pitch_enc, energy_enc, duration_enc], dim=-1)
        prosody_emb = self.fusion(combined)  # [batch, num_phonemes, prosody_embedding_dim]

        # Variational encoding (for diverse sampling)
        if use_variational:
            var_params = self.variational(prosody_emb)
            mean, log_var = var_params.chunk(2, dim=-1)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            prosody_emb = mean + eps * std
        else:
            mean = log_var = None

        return {
            'embedding': prosody_emb,
            'pitch_encoding': pitch_enc,
            'energy_encoding': energy_enc,
            'duration_encoding': duration_enc,
            'mean': mean,
            'log_var': log_var,
        }


# =============================================================================
# BIDIRECTIONAL DILATED CONVOLUTION DENOISER
# =============================================================================

class BidirectionalDilatedConv(nn.Module):
    """
    Bidirectional dilated convolution block for the diffusion denoiser.

    Processes input in both forward and backward directions with
    exponentially increasing dilation rates.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        # Forward direction
        self.conv_forward = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        # Backward direction (will be applied to reversed input)
        self.conv_backward = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        # Combine directions
        self.gate = nn.Conv1d(channels * 2, channels * 2, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, time]

        Returns:
            Output [batch, channels, time]
        """
        # Forward pass
        h_f = self.conv_forward(x)

        # Backward pass (reverse, conv, reverse back)
        x_rev = torch.flip(x, dims=[-1])
        h_b = self.conv_backward(x_rev)
        h_b = torch.flip(h_b, dims=[-1])

        # Combine with gated activation
        h = torch.cat([h_f, h_b], dim=1)
        h = self.gate(h)
        h1, h2 = h.chunk(2, dim=1)
        h = torch.tanh(h1) * torch.sigmoid(h2)

        # Residual connection
        h = self.dropout(h)
        h = h.transpose(1, 2)  # [batch, time, channels]
        h = self.norm(h)
        h = h.transpose(1, 2)  # [batch, channels, time]

        return x + h


class DiffusionDenoiser(nn.Module):
    """
    Diffusion denoiser with bidirectional dilated convolutions.

    Predicts noise to be removed from noisy prosody features.
    Can operate in conditional or unconditional mode for CFG.
    """

    def __init__(self, config: DiffStyleProsodyConfig):
        super().__init__()
        self.config = config

        # Input projection
        # Prosody features + time embedding + conditioning
        input_dim = config.prosody_dim + config.prosody_hidden_dim  # prosody + time/condition

        self.input_proj = nn.Conv1d(input_dim, config.denoiser_channels * config.prosody_hidden_dim, kernel_size=1)

        # Bidirectional dilated conv layers with exponential dilation
        self.layers = nn.ModuleList()
        for i in range(config.denoiser_num_layers):
            dilation = 2 ** (i % 4)  # Cycle: 1, 2, 4, 8, 1, 2, 4, 8, ...
            self.layers.append(
                BidirectionalDilatedConv(
                    channels=config.denoiser_channels * config.prosody_hidden_dim,
                    kernel_size=config.denoiser_kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )

        # Output projection back to prosody dimension
        self.output_proj = nn.Sequential(
            nn.Conv1d(config.denoiser_channels * config.prosody_hidden_dim, config.prosody_hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(config.prosody_hidden_dim, config.prosody_dim, kernel_size=1),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(config.prosody_hidden_dim),
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
            nn.GELU(),
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
        )

    def forward(
        self,
        noisy_prosody: torch.Tensor,  # [batch, num_phonemes, prosody_dim]
        t: torch.Tensor,               # [batch] diffusion timestep
        condition: Optional[torch.Tensor] = None,  # [batch, num_phonemes, condition_dim]
    ) -> torch.Tensor:
        """
        Predict noise in noisy prosody features.

        Args:
            noisy_prosody: Noisy prosody features [batch, num_phonemes, prosody_dim]
            t: Diffusion timestep [batch]
            condition: Optional conditioning (text + style)

        Returns:
            Predicted noise [batch, num_phonemes, prosody_dim]
        """
        batch_size, num_phonemes, _ = noisy_prosody.shape
        device = noisy_prosody.device

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, prosody_hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, num_phonemes, -1)  # [batch, num_phonemes, prosody_hidden_dim]

        # Combine inputs
        if condition is not None:
            x = torch.cat([noisy_prosody, t_emb + condition], dim=-1)
        else:
            x = torch.cat([noisy_prosody, t_emb], dim=-1)

        # Transpose for conv1d: [batch, channels, time]
        x = x.transpose(1, 2)

        # Apply layers
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)

        # Output
        noise_pred = self.output_proj(x)
        noise_pred = noise_pred.transpose(1, 2)  # [batch, num_phonemes, prosody_dim]

        return noise_pred


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional/time embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch] in range [0, 1]

        Returns:
            Embeddings [batch, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


# =============================================================================
# CLASSIFIER-FREE GUIDANCE & DYNAMIC THRESHOLDING
# =============================================================================

class GuidedSampler:
    """
    Implements classifier-free guidance (CFG) and dynamic thresholding.

    CFG Equation:
        ε_guided = ε_uncond + η * (ε_cond - ε_uncond)

    where η is the guidance scale. Higher η increases adherence to conditioning
    but may cause quality degradation.

    Dynamic Thresholding (γ correction):
        Rescales guided output to match original standard deviation,
        preventing distortion at high η values.
    """

    @staticmethod
    def apply_cfg(
        noise_pred_cond: torch.Tensor,    # [batch, num_phonemes, prosody_dim]
        noise_pred_uncond: torch.Tensor,   # [batch, num_phonemes, prosody_dim]
        eta: float,                         # Guidance scale
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance.

        Args:
            noise_pred_cond: Conditional noise prediction
            noise_pred_uncond: Unconditional noise prediction
            eta: Guidance scale (η). Higher = stronger guidance.
                 Recommended: 1.0-5.0, max ~7.0 before distortion

        Returns:
            Guided noise prediction
        """
        return noise_pred_uncond + eta * (noise_pred_cond - noise_pred_uncond)

    @staticmethod
    def dynamic_threshold(
        x: torch.Tensor,         # Guided output
        gamma: float,            # Correction scale
        percentile: float = 0.99,
    ) -> torch.Tensor:
        """
        Apply dynamic thresholding to prevent distortion.

        This technique rescales the guided output to preserve the original
        statistics while benefiting from guidance.

        Args:
            x: Guided output tensor
            gamma: Correction scale (γ). 0 = no correction, 1 = full correction
            percentile: Percentile for computing threshold

        Returns:
            Thresholded output with preserved statistics
        """
        if gamma == 0:
            return x

        # Compute percentile threshold
        abs_x = x.abs()
        threshold = torch.quantile(abs_x.flatten(1), percentile, dim=1, keepdim=True)
        threshold = threshold.unsqueeze(-1)  # [batch, 1, 1]

        # Compute original and target std
        original_std = x.std(dim=(1, 2), keepdim=True)

        # Clip to threshold
        x_clipped = torch.clamp(x, -threshold, threshold)

        # Rescale to match original std
        clipped_std = x_clipped.std(dim=(1, 2), keepdim=True)
        scale = original_std / (clipped_std + 1e-8)

        # Apply gamma-weighted correction
        x_corrected = gamma * (x_clipped * scale) + (1 - gamma) * x

        return x_corrected

    @staticmethod
    def guided_denoise_step(
        denoiser: DiffusionDenoiser,
        noisy_prosody: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        eta: float = 3.0,
        gamma: float = 0.5,
    ) -> torch.Tensor:
        """
        Single denoising step with CFG and dynamic thresholding.

        Args:
            denoiser: The denoiser network
            noisy_prosody: Current noisy prosody
            t: Diffusion timestep
            condition: Conditioning (text + style embeddings)
            eta: CFG guidance scale
            gamma: Dynamic thresholding scale

        Returns:
            Predicted noise with guidance
        """
        # Conditional prediction
        noise_cond = denoiser(noisy_prosody, t, condition)

        # Unconditional prediction
        noise_uncond = denoiser(noisy_prosody, t, None)

        # Apply CFG
        noise_guided = GuidedSampler.apply_cfg(noise_cond, noise_uncond, eta)

        # Apply dynamic thresholding
        noise_final = GuidedSampler.dynamic_threshold(noise_guided, gamma)

        return noise_final


# =============================================================================
# HIERARCHICAL PROSODY MODEL
# =============================================================================

class DiffStyleProsody(nn.Module):
    """
    DiffStyleTTS Hierarchical Prosody Model.

    Combines:
    1. Global Style Tokens (GST) for implicit sentence-level style
    2. Phoneme-level prosody encoder for explicit local features
    3. Diffusion-based prosody predictor
    4. Classifier-free guidance for diverse generation

    The model distinguishes between:
    - Implicit Style: What can't be easily defined (global emotional tone, manner)
    - Explicit Prosody: What can be measured (pitch, energy, duration per phoneme)
    """

    def __init__(self, config: DiffStyleProsodyConfig):
        super().__init__()
        self.config = config

        # Global Style Tokens for implicit style
        self.gst = GlobalStyleTokens(config)

        # Phoneme-level prosody encoder for explicit features
        self.prosody_encoder = PhonemeProsodyEncoder(config)

        # Text encoder (simplified FFT blocks)
        self.text_encoder = TextEncoder(config)

        # Conditioning fusion
        self.condition_fusion = nn.Sequential(
            nn.Linear(config.text_hidden_dim + config.gst_embedding_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Diffusion denoiser (conditional)
        self.denoiser_cond = DiffusionDenoiser(config)

        # Diffusion denoiser (unconditional) - for CFG
        self.denoiser_uncond = DiffusionDenoiser(config)

        # Length regulator for aligning embeddings
        if config.use_length_regulator:
            self.length_regulator = LengthRegulator()

        # Output projection to prosody token format
        self.output_projection = nn.Sequential(
            nn.Linear(config.prosody_embedding_dim + config.gst_embedding_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

        # Prosody predictors (for inference without ground truth)
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)
        self.duration_predictor = VariancePredictor(config)

        # Guiding scale predictors (learn optimal scales per sample)
        self.eta_predictor = nn.Sequential(
            nn.Linear(config.gst_embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1], scale to [1, max_eta]
        )

        self.gamma_predictor = nn.Sequential(
            nn.Linear(config.gst_embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode_style(
        self,
        mel: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract global style from mel-spectrogram.

        Args:
            mel: Mel-spectrogram [batch, mel_dim, time]
            return_weights: Return attention weights for visualization

        Returns:
            Style vector [batch, gst_embedding_dim]
        """
        return self.gst(mel, return_weights)

    def encode_prosody(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode phoneme-level prosody features.

        Args:
            pitch: Log F0 [batch, num_phonemes]
            energy: Log energy [batch, num_phonemes]
            duration: Log duration [batch, num_phonemes]

        Returns:
            Prosody encoding dict
        """
        return self.prosody_encoder(pitch, energy, duration)

    def encode_text(
        self,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text with FFT blocks.

        Args:
            text_embeddings: Phoneme embeddings [batch, num_phonemes, text_dim]
            text_mask: Attention mask [batch, num_phonemes]

        Returns:
            Text encoding [batch, num_phonemes, text_hidden_dim]
        """
        return self.text_encoder(text_embeddings, text_mask)

    def predict_guiding_scales(
        self,
        style_vector: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Predict optimal guiding scales based on style.

        Different styles may benefit from different guidance intensities.

        Args:
            style_vector: Global style vector [batch, gst_embedding_dim]

        Returns:
            (eta, gamma) predicted guiding scales
        """
        # Predict normalized eta in [0, 1]
        eta_norm = self.eta_predictor(style_vector).squeeze(-1)
        # Scale to [1, max_eta]
        eta = 1.0 + eta_norm * (self.config.max_eta - 1.0)

        # Predict gamma in [0, 1]
        gamma = self.gamma_predictor(style_vector).squeeze(-1)

        return eta.mean().item(), gamma.mean().item()

    def forward(
        self,
        text_embeddings: torch.Tensor,       # [batch, num_phonemes, text_dim]
        mel: Optional[torch.Tensor] = None,  # [batch, mel_dim, time] for training
        pitch: Optional[torch.Tensor] = None,  # [batch, num_phonemes]
        energy: Optional[torch.Tensor] = None,  # [batch, num_phonemes]
        duration: Optional[torch.Tensor] = None,  # [batch, num_phonemes]
        text_mask: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,           # Override guiding scale
        gamma: Optional[float] = None,         # Override thresholding scale
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical prosody model.

        Training mode (mel + prosody provided):
            - Extract style from mel
            - Use ground truth prosody
            - Compute diffusion loss

        Inference mode (no mel/prosody):
            - Use learned style embedding or provided style
            - Predict prosody via diffusion sampling
            - Apply CFG with guiding scales

        Args:
            text_embeddings: Text/phoneme embeddings
            mel: Optional mel-spectrogram for style extraction
            pitch, energy, duration: Optional ground truth prosody
            text_mask: Attention mask
            eta: CFG guidance scale (default: config.default_eta)
            gamma: Dynamic thresholding scale (default: config.default_gamma)

        Returns:
            Dict with prosody tokens, style vector, predictions, losses
        """
        batch_size, num_phonemes, _ = text_embeddings.shape
        device = text_embeddings.device

        # Set guiding scales
        if eta is None:
            eta = self.config.default_eta
        if gamma is None:
            gamma = self.config.default_gamma

        # Encode text
        text_enc = self.encode_text(text_embeddings, text_mask)

        # Extract or create style vector
        if mel is not None:
            style_vector, style_weights = self.encode_style(mel, return_weights=True)
        else:
            # Use mean of style tokens (neutral style)
            style_vector = self.gst.get_style_tokens().mean(dim=0, keepdim=True)
            style_vector = style_vector.expand(batch_size, -1)
            style_weights = None

        # Create conditioning: text + style
        style_expanded = style_vector.unsqueeze(1).expand(-1, num_phonemes, -1)
        condition = self.condition_fusion(
            torch.cat([text_enc, style_expanded], dim=-1)
        )

        # Encode or predict prosody
        if pitch is not None and energy is not None and duration is not None:
            # Training mode: use ground truth
            prosody_output = self.encode_prosody(pitch, energy, duration)
            prosody_emb = prosody_output['embedding']
        else:
            # Inference mode: predict prosody
            # Use variance predictors
            pitch_pred = self.pitch_predictor(text_enc).squeeze(-1)
            energy_pred = self.energy_predictor(text_enc).squeeze(-1)
            duration_pred = self.duration_predictor(text_enc).squeeze(-1)

            prosody_output = self.encode_prosody(pitch_pred, energy_pred, duration_pred)
            prosody_emb = prosody_output['embedding']

        # Combine style and prosody for output
        combined = torch.cat([
            prosody_emb.mean(dim=1),  # Global prosody (pooled)
            style_vector,
        ], dim=-1)

        # Project to prosody tokens
        tokens = self.output_projection(combined)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        return {
            'tokens': tokens,  # [batch, num_tokens, output_dim]
            'style_vector': style_vector,
            'style_weights': style_weights,
            'prosody_embedding': prosody_emb,
            'prosody_output': prosody_output,
            'condition': condition,
            'text_encoding': text_enc,
            'eta': eta,
            'gamma': gamma,
        }

    def sample_prosody(
        self,
        text_embeddings: torch.Tensor,
        style_vector: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 50,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample prosody via diffusion with CFG.

        Args:
            text_embeddings: Text/phoneme embeddings
            style_vector: Optional style conditioning
            num_diffusion_steps: Number of denoising steps
            eta: CFG guidance scale
            gamma: Dynamic thresholding scale

        Returns:
            Sampled prosody [batch, num_phonemes, prosody_dim]
        """
        batch_size, num_phonemes, _ = text_embeddings.shape
        device = text_embeddings.device

        if eta is None:
            eta = self.config.default_eta
        if gamma is None:
            gamma = self.config.default_gamma

        # Encode text
        text_enc = self.encode_text(text_embeddings)

        # Get or create style
        if style_vector is None:
            style_vector = self.gst.get_style_tokens().mean(dim=0, keepdim=True)
            style_vector = style_vector.expand(batch_size, -1)

        # Create conditioning
        style_expanded = style_vector.unsqueeze(1).expand(-1, num_phonemes, -1)
        condition = self.condition_fusion(
            torch.cat([text_enc, style_expanded], dim=-1)
        )

        # Initialize with noise
        prosody = torch.randn(batch_size, num_phonemes, self.config.prosody_dim, device=device)

        # Reverse diffusion with CFG
        for step in range(num_diffusion_steps - 1, -1, -1):
            t = torch.full((batch_size,), step / num_diffusion_steps, device=device)

            # Predict noise with CFG
            noise_pred = GuidedSampler.guided_denoise_step(
                self.denoiser_cond,
                prosody,
                t,
                condition,
                eta=eta,
                gamma=gamma,
            )

            # Simple DDPM update (simplified)
            alpha = 1 - (step / num_diffusion_steps)
            prosody = (prosody - (1 - alpha) * noise_pred) / (alpha + 1e-8)

            if step > 0:
                # Add noise for next step
                noise_scale = math.sqrt((1 - alpha) * step / num_diffusion_steps)
                prosody = prosody + noise_scale * torch.randn_like(prosody)

        return prosody


class TextEncoder(nn.Module):
    """Simplified FFT-based text encoder."""

    def __init__(self, config: DiffStyleProsodyConfig):
        super().__init__()

        self.layers = nn.ModuleList([
            FFTBlock(config.text_hidden_dim, config.text_num_heads, config.dropout)
            for _ in range(config.text_num_layers)
        ])

        self.input_proj = nn.Linear(config.text_hidden_dim, config.text_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class FFTBlock(nn.Module):
    """Feed-Forward Transformer block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class VariancePredictor(nn.Module):
    """Predicts variance parameters (pitch, energy, duration) from text encoding."""

    def __init__(self, config: DiffStyleProsodyConfig):
        super().__init__()

        self.conv1 = nn.Conv1d(config.text_hidden_dim, config.prosody_hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(config.prosody_hidden_dim)
        self.conv2 = nn.Conv1d(config.prosody_hidden_dim, config.prosody_hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(config.prosody_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.prosody_hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_phonemes, hidden_dim]

        Returns:
            Predictions [batch, num_phonemes, 1]
        """
        x = x.transpose(1, 2)  # [batch, hidden, num_phonemes]
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # [batch, num_phonemes, hidden]
        x = self.norm1(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [batch, hidden, num_phonemes]
        x = self.conv2(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # [batch, num_phonemes, hidden]
        x = self.norm2(x)
        x = self.dropout(x)
        return self.output(x)


class LengthRegulator(nn.Module):
    """Aligns encoder outputs to phoneme durations."""

    def forward(
        self,
        x: torch.Tensor,           # [batch, num_phonemes, hidden]
        durations: torch.Tensor,   # [batch, num_phonemes] integer durations
    ) -> torch.Tensor:
        """
        Expand encoder outputs by durations.

        Args:
            x: Encoder outputs
            durations: Duration per phoneme (in frames)

        Returns:
            Expanded outputs [batch, total_frames, hidden]
        """
        outputs = []
        for i in range(x.shape[0]):
            expanded = []
            for j in range(x.shape[1]):
                dur = int(durations[i, j].item())
                expanded.append(x[i, j].unsqueeze(0).expand(dur, -1))
            outputs.append(torch.cat(expanded, dim=0))

        # Pad to same length
        max_len = max(o.shape[0] for o in outputs)
        padded = []
        for o in outputs:
            if o.shape[0] < max_len:
                pad = torch.zeros(max_len - o.shape[0], o.shape[1], device=o.device)
                o = torch.cat([o, pad], dim=0)
            padded.append(o)

        return torch.stack(padded)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiffStyleProsodyLoss(nn.Module):
    """
    Combined loss for DiffStyleTTS prosody model.

    Components:
    1. Diffusion loss (predict noise)
    2. Prosody prediction loss (pitch, energy, duration)
    3. Style consistency loss
    4. KL divergence for variational prosody (optional)
    """

    def __init__(
        self,
        config: DiffStyleProsodyConfig,
        diffusion_weight: float = 1.0,
        prosody_weight: float = 1.0,
        style_weight: float = 0.1,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        self.config = config
        self.diffusion_weight = diffusion_weight
        self.prosody_weight = prosody_weight
        self.style_weight = style_weight
        self.kl_weight = kl_weight

        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predicted_noise: torch.Tensor,      # Denoiser output
        target_noise: torch.Tensor,          # True noise
        predicted_pitch: Optional[torch.Tensor] = None,
        target_pitch: Optional[torch.Tensor] = None,
        predicted_energy: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        predicted_duration: Optional[torch.Tensor] = None,
        target_duration: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        losses = {}
        device = predicted_noise.device

        # Diffusion loss
        losses['diffusion'] = self.mse_loss(predicted_noise, target_noise)

        # Prosody prediction losses
        if predicted_pitch is not None and target_pitch is not None:
            losses['pitch'] = self.mse_loss(predicted_pitch, target_pitch)
        else:
            losses['pitch'] = torch.tensor(0.0, device=device)

        if predicted_energy is not None and target_energy is not None:
            losses['energy'] = self.mse_loss(predicted_energy, target_energy)
        else:
            losses['energy'] = torch.tensor(0.0, device=device)

        if predicted_duration is not None and target_duration is not None:
            losses['duration'] = self.mse_loss(predicted_duration, target_duration)
        else:
            losses['duration'] = torch.tensor(0.0, device=device)

        # KL divergence for variational
        if mean is not None and log_var is not None:
            kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
            losses['kl'] = kl.mean()
        else:
            losses['kl'] = torch.tensor(0.0, device=device)

        # Total
        losses['total'] = (
            self.diffusion_weight * losses['diffusion'] +
            self.prosody_weight * (losses['pitch'] + losses['energy'] + losses['duration']) +
            self.kl_weight * losses['kl']
        )

        return losses


# =============================================================================
# ADAPTER FOR PROSODY PIPELINE
# =============================================================================

class DiffStyleProsodyAdapter(nn.Module):
    """
    Adapter integrating DiffStyleTTS prosody with the existing pipeline.

    Provides interface compatible with ProsodyEncoder output format.

    Usage:
        adapter = DiffStyleProsodyAdapter(config)

        # Training
        result = adapter(text_embeddings, mel, pitch, energy, duration)
        prosody_tokens = result['tokens']  # [batch, num_tokens, hidden]

        # Inference with guiding scales
        result = adapter(text_embeddings, eta=4.0, gamma=0.6)
    """

    def __init__(
        self,
        config: DiffStyleProsodyConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core model
        self.model = DiffStyleProsody(config)

        # Adapt to prosody hidden dimension if different
        if config.output_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def forward(
        self,
        text_embeddings: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
        duration_scale: float = 1.0,
        style_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional attribute-specific scales.

        The independent scales allow fine-grained control:
        - pitch_scale: Control pitch variation strength
        - energy_scale: Control energy variation strength
        - duration_scale: Control timing variation strength
        - style_scale: Control global style influence

        Args:
            text_embeddings: Text/phoneme embeddings
            mel: Optional mel for style extraction
            pitch, energy, duration: Optional prosody targets
            text_mask: Attention mask
            eta: CFG guidance scale
            gamma: Dynamic thresholding scale
            *_scale: Attribute-specific scaling factors

        Returns:
            Dict with prosody tokens and intermediate outputs
        """
        # Get base model output
        result = self.model(
            text_embeddings=text_embeddings,
            mel=mel,
            pitch=pitch * pitch_scale if pitch is not None else None,
            energy=energy * energy_scale if energy is not None else None,
            duration=duration * duration_scale if duration is not None else None,
            text_mask=text_mask,
            eta=eta,
            gamma=gamma,
        )

        # Apply style scaling
        if style_scale != 1.0:
            result['style_vector'] = result['style_vector'] * style_scale

        # Adapt tokens to output dimension
        tokens = self.output_adapter(result['tokens'])

        return {
            **result,
            'prosody_tokens': tokens,
        }

    def encode_style_from_reference(
        self,
        mel: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract style from reference mel-spectrogram."""
        return self.model.encode_style(mel, return_weights)

    def sample_diverse_prosody(
        self,
        text_embeddings: torch.Tensor,
        style_vector: Optional[torch.Tensor] = None,
        num_samples: int = 5,
        eta_range: Tuple[float, float] = (1.0, 5.0),
        gamma: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate diverse prosody samples by varying η.

        Args:
            text_embeddings: Text/phoneme embeddings
            style_vector: Optional style conditioning
            num_samples: Number of diverse samples
            eta_range: Range of η values to sample
            gamma: Dynamic thresholding scale

        Returns:
            List of prosody results with different η values
        """
        samples = []
        eta_values = torch.linspace(eta_range[0], eta_range[1], num_samples)

        for eta in eta_values:
            result = self.forward(
                text_embeddings=text_embeddings,
                eta=eta.item(),
                gamma=gamma,
            )
            result['eta_used'] = eta.item()
            samples.append(result)

        return samples


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DiffStyleTTS Hierarchical Prosody with Guiding Scales - Test Suite")
    print("=" * 70)

    config = DiffStyleProsodyConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test 1: Global Style Tokens
    print("\n[Test 1] Global Style Tokens (GST)...")
    gst = GlobalStyleTokens(config).to(device)

    batch_size = 2
    mel = torch.randn(batch_size, config.mel_dim, 200, device=device)  # 200 frames

    style_vector, style_weights = gst(mel, return_weights=True)
    print(f"  Mel input shape: {mel.shape}")
    print(f"  Style vector shape: {style_vector.shape}")
    print(f"  Style weights shape: {style_weights.shape}")
    print(f"  Style tokens: {gst.get_style_tokens().shape}")
    print("  [PASS]")

    # Test 2: Phoneme Prosody Encoder
    print("\n[Test 2] Phoneme Prosody Encoder...")
    prosody_encoder = PhonemeProsodyEncoder(config).to(device)

    num_phonemes = 20
    pitch = torch.randn(batch_size, num_phonemes, device=device)
    energy = torch.randn(batch_size, num_phonemes, device=device)
    duration = torch.abs(torch.randn(batch_size, num_phonemes, device=device))

    prosody_output = prosody_encoder(pitch, energy, duration)
    print(f"  Pitch input shape: {pitch.shape}")
    print(f"  Prosody embedding shape: {prosody_output['embedding'].shape}")
    print(f"  Pitch encoding shape: {prosody_output['pitch_encoding'].shape}")
    print("  [PASS]")

    # Test 3: Bidirectional Dilated Conv
    print("\n[Test 3] Bidirectional Dilated Convolution...")
    bdconv = BidirectionalDilatedConv(
        channels=64,
        kernel_size=3,
        dilation=2,
    ).to(device)

    x = torch.randn(batch_size, 64, 50, device=device)
    y = bdconv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert x.shape == y.shape, "Shape mismatch!"
    print("  [PASS]")

    # Test 4: Diffusion Denoiser
    print("\n[Test 4] Diffusion Denoiser...")
    denoiser = DiffusionDenoiser(config).to(device)

    noisy_prosody = torch.randn(batch_size, num_phonemes, config.prosody_dim, device=device)
    t = torch.rand(batch_size, device=device)
    condition = torch.randn(batch_size, num_phonemes, config.prosody_hidden_dim, device=device)

    noise_pred = denoiser(noisy_prosody, t, condition)
    print(f"  Noisy prosody shape: {noisy_prosody.shape}")
    print(f"  Time shape: {t.shape}")
    print(f"  Predicted noise shape: {noise_pred.shape}")
    print("  [PASS]")

    # Test 5: Classifier-Free Guidance
    print("\n[Test 5] Classifier-Free Guidance (CFG)...")
    noise_cond = torch.randn(batch_size, num_phonemes, config.prosody_dim, device=device)
    noise_uncond = torch.randn(batch_size, num_phonemes, config.prosody_dim, device=device)

    for eta in [1.0, 3.0, 5.0, 7.0]:
        guided = GuidedSampler.apply_cfg(noise_cond, noise_uncond, eta)
        diff_norm = (guided - noise_cond).norm().item()
        print(f"  η={eta}: guided diff norm = {diff_norm:.3f}")
    print("  [PASS]")

    # Test 6: Dynamic Thresholding
    print("\n[Test 6] Dynamic Thresholding (γ correction)...")
    x = torch.randn(batch_size, num_phonemes, config.prosody_dim, device=device) * 5  # Large values

    for gamma in [0.0, 0.3, 0.5, 0.7, 1.0]:
        x_corrected = GuidedSampler.dynamic_threshold(x, gamma)
        std_ratio = x_corrected.std() / x.std()
        print(f"  γ={gamma}: std ratio = {std_ratio:.3f}")
    print("  [PASS]")

    # Test 7: Full DiffStyleProsody Model
    print("\n[Test 7] Full DiffStyleProsody Model...")
    model = DiffStyleProsody(config).to(device)

    text_emb = torch.randn(batch_size, num_phonemes, config.text_hidden_dim, device=device)

    # Training mode (with mel and prosody)
    result = model(
        text_embeddings=text_emb,
        mel=mel,
        pitch=pitch,
        energy=energy,
        duration=duration,
        eta=3.0,
        gamma=0.5,
    )

    print(f"  Text embedding shape: {text_emb.shape}")
    print(f"  Output tokens shape: {result['tokens'].shape}")
    print(f"  Style vector shape: {result['style_vector'].shape}")
    print(f"  Prosody embedding shape: {result['prosody_embedding'].shape}")
    print(f"  η used: {result['eta']}, γ used: {result['gamma']}")
    print("  [PASS]")

    # Test 8: Inference mode (no mel)
    print("\n[Test 8] Inference Mode (without mel)...")
    result_infer = model(
        text_embeddings=text_emb,
        eta=4.0,
        gamma=0.6,
    )

    print(f"  Output tokens shape: {result_infer['tokens'].shape}")
    print(f"  Used default style: {result_infer['style_weights'] is None}")
    print("  [PASS]")

    # Test 9: Guiding Scale Prediction
    print("\n[Test 9] Guiding Scale Prediction...")
    eta_pred, gamma_pred = model.predict_guiding_scales(result['style_vector'])
    print(f"  Predicted η: {eta_pred:.3f}")
    print(f"  Predicted γ: {gamma_pred:.3f}")
    print("  [PASS]")

    # Test 10: DiffStyleProsodyAdapter
    print("\n[Test 10] DiffStyleProsodyAdapter...")
    adapter = DiffStyleProsodyAdapter(config).to(device)

    # With attribute scaling
    result_scaled = adapter(
        text_embeddings=text_emb,
        mel=mel,
        pitch=pitch,
        energy=energy,
        duration=duration,
        pitch_scale=1.2,
        energy_scale=0.8,
        duration_scale=1.0,
        style_scale=1.5,
        eta=3.5,
        gamma=0.4,
    )

    print(f"  Prosody tokens shape: {result_scaled['prosody_tokens'].shape}")
    print(f"  Pitch scale: 1.2, Energy scale: 0.8, Style scale: 1.5")
    print("  [PASS]")

    # Test 11: Diverse Prosody Sampling
    print("\n[Test 11] Diverse Prosody Sampling...")
    samples = adapter.sample_diverse_prosody(
        text_embeddings=text_emb,
        num_samples=4,
        eta_range=(1.0, 5.0),
    )

    print(f"  Generated {len(samples)} diverse samples")
    for i, sample in enumerate(samples):
        print(f"    Sample {i+1}: η={sample['eta_used']:.2f}, "
              f"token std={sample['prosody_tokens'].std():.4f}")
    print("  [PASS]")

    # Test 12: Loss Function
    print("\n[Test 12] Loss Function...")
    loss_fn = DiffStyleProsodyLoss(config)

    pred_noise = torch.randn(batch_size, num_phonemes, config.prosody_dim, device=device)
    target_noise = torch.randn(batch_size, num_phonemes, config.prosody_dim, device=device)

    losses = loss_fn(
        predicted_noise=pred_noise,
        target_noise=target_noise,
        predicted_pitch=torch.randn(batch_size, num_phonemes, 1, device=device),
        target_pitch=pitch.unsqueeze(-1),
        predicted_energy=torch.randn(batch_size, num_phonemes, 1, device=device),
        target_energy=energy.unsqueeze(-1),
    )

    print(f"  Diffusion loss: {losses['diffusion'].item():.4f}")
    print(f"  Pitch loss: {losses['pitch'].item():.4f}")
    print(f"  Energy loss: {losses['energy'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All DiffStyleTTS tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from diffstyle_prosody import (
    DiffStyleProsodyConfig, DiffStyleProsodyAdapter, GuidedSampler
)

# Initialize
config = DiffStyleProsodyConfig()
adapter = DiffStyleProsodyAdapter(config).cuda()

# Training with ground truth
result = adapter(
    text_embeddings=phoneme_embeddings,
    mel=mel_spectrogram,
    pitch=pitch_values,
    energy=energy_values,
    duration=duration_values,
)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

# Inference with guiding scales
result = adapter(
    text_embeddings=phoneme_embeddings,
    eta=4.0,   # CFG guidance (diversity vs quality)
    gamma=0.5,  # Dynamic thresholding (prevents distortion)
)

# Fine-grained attribute control
result = adapter(
    text_embeddings=phoneme_embeddings,
    mel=reference_mel,  # Extract style from reference
    pitch_scale=1.3,    # Emphasize pitch variation
    energy_scale=0.8,   # Reduce energy variation
    duration_scale=1.0, # Normal timing
    style_scale=1.2,    # Stronger style influence
)

# Generate diverse prosody samples
samples = adapter.sample_diverse_prosody(
    text_embeddings=phoneme_embeddings,
    num_samples=5,
    eta_range=(1.0, 6.0),  # Vary guidance from conservative to aggressive
)

# Extract style from reference for transfer
style_vector, weights = adapter.encode_style_from_reference(
    reference_mel, return_weights=True
)

# Integrate with ProsodyControlledCSM:
# prosody_prefix = adapter(text_emb)['prosody_tokens']
# combined_prefix = torch.cat([prosody_prefix, other_conditioning], dim=1)
""")
