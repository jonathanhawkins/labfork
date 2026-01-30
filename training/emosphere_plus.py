"""
EmoSphere++: Multi-Level Style Encoder for Zero-Shot Emotional TTS

Based on EmoSphere++ (arXiv:2411.02625, Nov 2024):
"EmoSphere++: Emotion-Controllable Zero-Shot Text-to-Speech via Emotion-Adaptive Spherical Vector"

Key Innovations over original EmoSphere-TTS:
1. Emotion-Adaptive Spherical Vector (EASV) - computed from prosodic patterns WITHOUT human annotation
2. Multi-level style encoder for SEEN AND UNSEEN speakers
3. Additional loss functions for zero-shot emotion transfer
4. CFM (Conditional Flow Matching) decoder for fast, high-quality synthesis

Architecture:
```
Audio Input
    ↓
┌───────────────────────────────────────────────────────────┐
│               Multi-Level Style Encoder                    │
│  ┌─────────────────┐    ┌──────────────────────┐          │
│  │  Global Branch  │    │    Local Branch       │          │
│  │  (Utterance)    │    │  (Frame-level)        │          │
│  │  - GST-like     │    │  - Temporal Conv      │          │
│  │  - Attention    │    │  - Position encoding  │          │
│  └────────┬────────┘    └──────────┬───────────┘          │
│           └───────────┬────────────┘                       │
│                       ↓                                    │
│              Style Fusion Layer                            │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│         Emotion-Adaptive Spherical Vector (EASV)           │
│  - VAD from prosodic patterns (no labels needed)           │
│  - Positive valence → higher pitch/energy                  │
│  - Intensity from variance patterns                        │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│              CFM Decoder (Fast Synthesis)                  │
│  - Optimal transport flow matching                         │
│  - Few-step sampling (4-8 steps)                          │
│  - High-quality mel-spectrogram output                    │
└───────────────────────────────────────────────────────────┘
```

Multi-level style encoder captures:
- Global: Overall utterance emotion characteristics (timbre-independent)
- Local: Frame-level prosodic variations (pitch, energy dynamics)
- Speaker: Timbre information (disentangled from emotion)

EASV Computation (Training-Free, Annotation-Free):
- Pitch statistics: mean, std, slope → valence/arousal
- Energy patterns: mean, variance, peaks → dominance/arousal
- Duration patterns: speaking rate, pauses → arousal
- Combined into spherical VAD without manual labels

Benefits:
- Zero-shot emotion control for UNSEEN speakers
- Interpretable intensity control via spherical magnitude
- Generalizes across speakers (style encoder sees many speakers)
- Fast synthesis via CFM (4-8 steps vs 50+ for diffusion)

References:
- EmoSphere-TTS: arXiv:2406.07803 (original spherical emotion)
- EmoSphere++: arXiv:2411.02625 (multi-level extension)
- Matcha-TTS: Flow matching for TTS
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import existing spherical emotion utilities
from spherical_emotion import (
    SphericalEmotionConfig,
    SphericalEmotionEncoder,
    VAD_PROTOTYPES,
    CORE_EMOTIONS,
    cartesian_to_spherical,
    spherical_to_cartesian,
    EmotionInterpolator,
)

# Import CFM components from prosody_flow
from prosody_flow import (
    ProsodyFlowConfig,
    GaussianConditionalPath,
    OptimalTransportCoupling,
    ODESolver,
    SinusoidalTimeEmbedding,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmoSpherePlusConfig:
    """Configuration for EmoSphere++ model."""

    # Input dimensions
    input_dim: int = 768  # Audio feature dimension (wav2vec2/HuBERT)
    mel_dim: int = 80  # Mel spectrogram dimension

    # Multi-level style encoder
    global_style_dim: int = 256  # Global (utterance-level) style dimension
    local_style_dim: int = 128  # Local (frame-level) style dimension
    combined_style_dim: int = 384  # global_style_dim + local_style_dim

    # Global style encoder (GST-like)
    gst_num_heads: int = 4  # Number of attention heads for style tokens
    gst_num_tokens: int = 10  # Number of learnable style tokens
    gst_conv_channels: List[int] = field(default_factory=lambda: [32, 32, 64, 64, 128, 128])
    gst_conv_kernel: int = 3

    # Local style encoder
    local_conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    local_conv_kernel: int = 3
    local_num_layers: int = 4
    local_num_heads: int = 8

    # Spherical emotion (EASV)
    vad_dim: int = 3  # Valence, Arousal, Dominance
    emotion_embedding_dim: int = 256
    num_emotions: int = 8  # Core emotion categories

    # CFM decoder settings
    cfm_hidden_dim: int = 512
    cfm_num_layers: int = 6
    cfm_num_heads: int = 8
    cfm_sigma_min: float = 0.001
    cfm_num_steps_train: int = 10
    cfm_num_steps_inference: int = 8  # Fast inference (4-8 steps)
    cfm_ode_method: str = "midpoint"

    # Output
    output_dim: int = 2048  # Match CSM prosody hidden dim
    num_prosody_tokens: int = 4  # Prefix tokens for conditioning

    # Training
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Loss weights
    easv_reconstruction_weight: float = 1.0
    style_consistency_weight: float = 0.5
    emotion_transfer_weight: float = 0.3
    cfm_loss_weight: float = 1.0
    speaker_disentangle_weight: float = 0.2


# =============================================================================
# GLOBAL STYLE ENCODER (GST-LIKE)
# =============================================================================

class ReferenceEncoder(nn.Module):
    """
    Reference encoder that extracts global style from mel-spectrogram.

    Architecture:
    - 2D convolutional layers for spectral feature extraction
    - GRU for temporal summarization
    - Outputs fixed-size style representation

    Based on Global Style Tokens (GST) from Tacotron 2.
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Convolutional layers for spectral feature extraction
        channels = config.gst_conv_channels
        kernel = config.gst_conv_kernel

        layers = []
        in_channels = 1  # Mel spectrogram as single channel
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=kernel, stride=2, padding=kernel//2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate output size after convolutions
        # Assuming mel_dim=80, after 6 layers with stride 2: 80 / 2^6 ≈ 1
        # Width (time): variable / 2^6
        # We'll use adaptive pooling to handle variable lengths

        # GRU for temporal summarization
        self.gru = nn.GRU(
            input_size=channels[-1],  # Last conv channel
            hidden_size=config.global_style_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Project bidirectional output to style dimension
        self.output_proj = nn.Linear(config.global_style_dim * 2, config.global_style_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Extract global style from mel-spectrogram.

        Args:
            mel: [batch, time, mel_dim] mel spectrogram

        Returns:
            Global style embedding [batch, global_style_dim]
        """
        batch_size = mel.shape[0]

        # Add channel dimension and transpose for conv2d
        # [batch, time, mel] -> [batch, 1, mel, time]
        x = mel.transpose(1, 2).unsqueeze(1)

        # Apply convolutions
        x = self.conv_layers(x)  # [batch, channels, mel', time']

        # Reshape for GRU: [batch, time', channels]
        x = x.squeeze(2) if x.shape[2] == 1 else x.mean(dim=2)  # Pool mel dimension
        x = x.transpose(1, 2)  # [batch, time', channels]

        # GRU processing
        _, hidden = self.gru(x)  # hidden: [2, batch, hidden]

        # Concatenate bidirectional hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch, hidden*2]

        # Project to style dimension
        style = self.output_proj(hidden)  # [batch, global_style_dim]

        return style


class StyleTokenAttention(nn.Module):
    """
    Attention over learnable style tokens (Global Style Tokens).

    The mel-encoded query attends over a bank of learnable style tokens,
    producing a weighted combination as the global style.
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Learnable style token bank
        self.style_tokens = nn.Parameter(
            torch.randn(config.gst_num_tokens, config.global_style_dim) * 0.1
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.global_style_dim,
            num_heads=config.gst_num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Attend over style tokens using reference encoder output as query.

        Args:
            query: [batch, global_style_dim] from reference encoder
            return_weights: Whether to return attention weights

        Returns:
            Style embedding [batch, global_style_dim]
            Optionally: attention weights [batch, num_tokens]
        """
        batch_size = query.shape[0]

        # Expand style tokens for batch
        tokens = self.style_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Query as single token sequence
        query = query.unsqueeze(1)  # [batch, 1, dim]

        # Attention: query attends over style tokens
        output, weights = self.attention(query, tokens, tokens)

        # Remove sequence dimension
        output = output.squeeze(1)  # [batch, global_style_dim]

        if return_weights:
            weights = weights.squeeze(1)  # [batch, num_tokens]
            return output, weights
        return output


class GlobalStyleEncoder(nn.Module):
    """
    Global style encoder combining reference encoder and style token attention.

    Captures utterance-level style characteristics independent of content.
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        self.reference_encoder = ReferenceEncoder(config)
        self.style_attention = StyleTokenAttention(config)

        # Layer norm for output stability
        self.output_norm = nn.LayerNorm(config.global_style_dim)

    def forward(
        self,
        mel: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract global style from mel spectrogram.

        Args:
            mel: [batch, time, mel_dim]
            return_details: Whether to return intermediate representations

        Returns:
            Global style [batch, global_style_dim]
        """
        # Reference encoding
        ref_output = self.reference_encoder(mel)

        # Style token attention
        style, weights = self.style_attention(ref_output, return_weights=True)

        # Normalize
        style = self.output_norm(style)

        if return_details:
            return {
                'style': style,
                'reference_encoding': ref_output,
                'attention_weights': weights,
            }
        return style


# =============================================================================
# LOCAL STYLE ENCODER (FRAME-LEVEL)
# =============================================================================

class LocalStyleEncoder(nn.Module):
    """
    Local style encoder for frame-level prosodic variations.

    Captures fine-grained temporal dynamics:
    - Pitch contours and transitions
    - Energy dynamics and emphasis
    - Rhythm and timing patterns

    Architecture:
    - 1D temporal convolutions
    - Transformer for temporal context
    - Outputs per-frame style features
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # 1D temporal convolutions
        channels = config.local_conv_channels
        kernel = config.local_conv_kernel

        conv_layers = []
        in_channels = config.input_dim
        for i, out_channels in enumerate(channels):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=kernel, stride=1, padding=kernel//2
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(channels[-1])

        # Transformer layers for temporal context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels[-1],
            nhead=config.local_num_heads,
            dim_feedforward=channels[-1] * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.local_num_layers)

        # Project to local style dimension
        self.output_proj = nn.Sequential(
            nn.Linear(channels[-1], config.local_style_dim),
            nn.LayerNorm(config.local_style_dim),
        )

        # Global pooling for utterance-level summary
        self.attention_pool = AttentivePooling(config.local_style_dim)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract local frame-level style features.

        Args:
            features: [batch, seq_len, input_dim] audio features
            mask: [batch, seq_len] attention mask
            return_sequence: Whether to return frame-level features

        Returns:
            Local style [batch, local_style_dim]
            Optionally: frame features [batch, seq_len, local_style_dim]
        """
        # Transpose for conv1d: [batch, dim, seq]
        x = features.transpose(1, 2)

        # Apply convolutions
        x = self.conv_layers(x)  # [batch, channels, seq]

        # Transpose back: [batch, seq, channels]
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer for temporal context
        if mask is not None:
            # Convert to attention mask format
            attn_mask = ~mask  # True = ignore
            x = self.transformer(x, src_key_padding_mask=attn_mask)
        else:
            x = self.transformer(x)

        # Project to local style dimension
        frame_features = self.output_proj(x)  # [batch, seq, local_style_dim]

        # Pool for utterance-level representation
        if mask is not None:
            local_style = self.attention_pool(frame_features, mask)
        else:
            local_style = self.attention_pool(frame_features)

        if return_sequence:
            return local_style, frame_features
        return local_style


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim

        # Create position encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len].unsqueeze(0)


class AttentivePooling(nn.Module):
    """Attention-based pooling over sequence dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool sequence using attention weights.

        Args:
            x: [batch, seq, dim]
            mask: [batch, seq] valid positions

        Returns:
            Pooled representation [batch, dim]
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # [batch, seq]

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax over sequence
        weights = F.softmax(scores, dim=-1)  # [batch, seq]

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch, dim]

        return pooled


# =============================================================================
# MULTI-LEVEL STYLE ENCODER
# =============================================================================

class MultiLevelStyleEncoder(nn.Module):
    """
    Multi-level style encoder for EmoSphere++.

    Combines global (utterance-level) and local (frame-level) style encoding
    to capture both overall emotional characteristics and fine-grained
    prosodic variations.

    Key innovation: Generalizes to unseen speakers by learning style
    representations that are independent of speaker identity.
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Global style encoder (from mel)
        self.global_encoder = GlobalStyleEncoder(config)

        # Local style encoder (from audio features)
        self.local_encoder = LocalStyleEncoder(config)

        # Fusion layer for combining global and local
        self.fusion = nn.Sequential(
            nn.Linear(config.global_style_dim + config.local_style_dim, config.combined_style_dim),
            nn.LayerNorm(config.combined_style_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.combined_style_dim, config.combined_style_dim),
            nn.LayerNorm(config.combined_style_dim),
        )

        # Gating mechanism to balance global vs local
        self.gate = nn.Sequential(
            nn.Linear(config.global_style_dim + config.local_style_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Optional: project back to component dimensions for disentanglement
        self.global_projection = nn.Linear(config.combined_style_dim, config.global_style_dim)
        self.local_projection = nn.Linear(config.combined_style_dim, config.local_style_dim)

    def forward(
        self,
        mel: torch.Tensor,
        features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract multi-level style representation.

        Args:
            mel: [batch, time, mel_dim] mel spectrogram
            features: [batch, seq, input_dim] audio features (wav2vec2/HuBERT)
            feature_mask: [batch, seq] attention mask
            return_components: Whether to return individual style components

        Returns:
            Combined style [batch, combined_style_dim]
            Or dict with style components if return_components=True
        """
        # Extract global style from mel
        global_style = self.global_encoder(mel)  # [batch, global_style_dim]

        # Extract local style from features
        local_style, local_frames = self.local_encoder(
            features, feature_mask, return_sequence=True
        )  # [batch, local_style_dim], [batch, seq, local_style_dim]

        # Concatenate for fusion
        concat_style = torch.cat([global_style, local_style], dim=-1)

        # Compute gating weights
        gate_weights = self.gate(concat_style)  # [batch, 2]

        # Weighted combination before fusion
        weighted_global = global_style * gate_weights[:, 0:1]
        weighted_local = local_style * gate_weights[:, 1:2]

        # Fuse with gated components
        gated_concat = torch.cat([weighted_global, weighted_local], dim=-1)
        combined_style = self.fusion(gated_concat)

        if return_components:
            return {
                'combined_style': combined_style,
                'global_style': global_style,
                'local_style': local_style,
                'local_frames': local_frames,
                'gate_weights': gate_weights,
            }
        return combined_style


# =============================================================================
# EMOTION-ADAPTIVE SPHERICAL VECTOR (EASV)
# =============================================================================

class ProsodyToVADPredictor(nn.Module):
    """
    Predicts VAD (Valence-Arousal-Dominance) from prosodic patterns.

    Key insight from EmoSphere++: VAD can be inferred from prosody without
    explicit emotion labels using learned correlations:
    - Positive valence → higher pitch mean, wider pitch range
    - High arousal → higher energy, faster speaking rate
    - High dominance → louder, more pitch variation

    Training uses weakly supervised or unsupervised objectives.
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Prosodic feature dimensions
        # pitch: 4 (mean, std, min, max)
        # energy: 4 (mean, std, min, max)
        # duration: 3 (speaking_rate, pause_ratio, syllable_rate)
        # spectral: 4 (spectral_centroid, spectral_flux, hnr, jitter)
        prosody_dim = 15

        # Prosody encoder
        self.prosody_encoder = nn.Sequential(
            nn.Linear(prosody_dim, config.emotion_embedding_dim),
            nn.LayerNorm(config.emotion_embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emotion_embedding_dim, config.emotion_embedding_dim),
            nn.GELU(),
        )

        # VAD prediction heads
        self.valence_head = nn.Sequential(
            nn.Linear(config.emotion_embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # [-1, 1]
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(config.emotion_embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.dominance_head = nn.Sequential(
            nn.Linear(config.emotion_embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Learnable correlations for weak supervision
        # Based on research: pitch↑ → valence↑, energy↑ → arousal↑
        self.register_buffer('prosody_vad_prior', torch.tensor([
            # pitch_mean, pitch_std, pitch_min, pitch_max
            [0.3, 0.2, 0.1, 0.2],    # Valence correlations
            [0.2, 0.3, 0.0, 0.2],    # Arousal correlations
            [0.1, 0.2, 0.0, 0.2],    # Dominance correlations
        ]).T)

    def extract_prosody_features(
        self,
        pitch: torch.Tensor,
        energy: torch.Tensor,
        duration: torch.Tensor,
        spectral: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract prosodic statistics from frame-level features.

        Args:
            pitch: [batch, time] F0 values
            energy: [batch, time] energy values
            duration: [batch] or [batch, 3] duration stats
            spectral: Optional [batch, 4] spectral features

        Returns:
            Prosody features [batch, 15]
        """
        batch_size = pitch.shape[0]
        device = pitch.device

        # Pitch statistics
        pitch_mean = pitch.mean(dim=-1)
        pitch_std = pitch.std(dim=-1)
        pitch_min = pitch.min(dim=-1)[0]
        pitch_max = pitch.max(dim=-1)[0]
        pitch_features = torch.stack([pitch_mean, pitch_std, pitch_min, pitch_max], dim=-1)

        # Energy statistics
        energy_mean = energy.mean(dim=-1)
        energy_std = energy.std(dim=-1)
        energy_min = energy.min(dim=-1)[0]
        energy_max = energy.max(dim=-1)[0]
        energy_features = torch.stack([energy_mean, energy_std, energy_min, energy_max], dim=-1)

        # Duration features
        if duration.dim() == 1:
            duration_features = torch.stack([duration, duration * 0.1, duration * 0.05], dim=-1)
        else:
            duration_features = duration

        # Spectral features (or zeros)
        if spectral is None:
            spectral_features = torch.zeros(batch_size, 4, device=device)
        else:
            spectral_features = spectral

        # Normalize features
        pitch_features = (pitch_features - pitch_features.mean()) / (pitch_features.std() + 1e-8)
        energy_features = (energy_features - energy_features.mean()) / (energy_features.std() + 1e-8)

        # Concatenate all
        prosody = torch.cat([
            pitch_features, energy_features, duration_features[:, :3], spectral_features[:, :3]
        ], dim=-1)

        return prosody

    def forward(
        self,
        prosody_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict VAD from prosodic features.

        Args:
            prosody_features: [batch, 15] prosodic statistics

        Returns:
            Dict with VAD predictions and embedding
        """
        # Encode prosody
        embedding = self.prosody_encoder(prosody_features)

        # Predict VAD
        valence = self.valence_head(embedding)
        arousal = self.arousal_head(embedding)
        dominance = self.dominance_head(embedding)

        vad = torch.cat([valence, arousal, dominance], dim=-1)

        return {
            'vad': vad,
            'embedding': embedding,
            'valence': valence.squeeze(-1),
            'arousal': arousal.squeeze(-1),
            'dominance': dominance.squeeze(-1),
        }


class EmotionAdaptiveSphericalVector(nn.Module):
    """
    Emotion-Adaptive Spherical Vector (EASV) module.

    Computes spherical emotion representation from style features
    without requiring emotion labels (annotation-free).

    Key innovation: Uses multi-level style + prosody patterns to
    infer emotion vectors that lie on a sphere, enabling:
    1. Intensity control via magnitude (α parameter)
    2. Emotion type via direction
    3. Smooth interpolation between emotions
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Project combined style to VAD-compatible space
        self.style_to_vad = nn.Sequential(
            nn.Linear(config.combined_style_dim, config.emotion_embedding_dim),
            nn.LayerNorm(config.emotion_embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emotion_embedding_dim, config.vad_dim),
            nn.Tanh(),  # Bound to [-1, 1]
        )

        # Prosody-based VAD predictor (optional, for training signal)
        self.prosody_vad = ProsodyToVADPredictor(config)

        # Intensity predictor from style
        self.intensity_predictor = nn.Sequential(
            nn.Linear(config.combined_style_dim, config.emotion_embedding_dim // 2),
            nn.GELU(),
            nn.Linear(config.emotion_embedding_dim // 2, 1),
            nn.Sigmoid(),
        )

        # VAD to emotion embedding (spherical encoding)
        self.spherical_encoder = nn.Sequential(
            nn.Linear(7, config.emotion_embedding_dim),  # r, θ, φ + sin/cos
            nn.LayerNorm(config.emotion_embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emotion_embedding_dim, config.emotion_embedding_dim),
        )

        # Emotion classifier for weak supervision
        self.emotion_classifier = nn.Linear(config.emotion_embedding_dim, config.num_emotions)

        # Learnable emotion prototypes in VAD space
        vad_prototypes = torch.tensor([
            VAD_PROTOTYPES[e] for e in CORE_EMOTIONS
        ], dtype=torch.float32)
        self.register_buffer('emotion_prototypes', vad_prototypes)

    def forward(
        self,
        style: torch.Tensor,
        prosody_features: Optional[torch.Tensor] = None,
        intensity: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute EASV from style features.

        Args:
            style: [batch, combined_style_dim] from MultiLevelStyleEncoder
            prosody_features: Optional [batch, 15] for prosody-guided VAD
            intensity: Optional intensity scaling (α parameter)

        Returns:
            Dict with spherical emotion representations
        """
        batch_size = style.shape[0]
        device = style.device

        # Predict VAD from style
        vad_from_style = self.style_to_vad(style)  # [batch, 3]

        # Optionally refine with prosody-based prediction
        if prosody_features is not None:
            prosody_output = self.prosody_vad(prosody_features)
            vad_from_prosody = prosody_output['vad']

            # Weighted combination
            vad = 0.7 * vad_from_style + 0.3 * vad_from_prosody
        else:
            vad = vad_from_style

        # Predict or use provided intensity
        if intensity is None:
            predicted_intensity = self.intensity_predictor(style).squeeze(-1)
            intensity_tensor = predicted_intensity
        else:
            intensity_tensor = torch.full((batch_size,), intensity, device=device)
            predicted_intensity = intensity_tensor

        # Scale VAD by intensity
        vad_scaled = vad * intensity_tensor.unsqueeze(-1)

        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_spherical(vad_scaled)

        # Create spherical features
        spherical_features = torch.stack([
            r, theta, phi,
            torch.sin(theta), torch.cos(theta),
            torch.sin(phi), torch.cos(phi),
        ], dim=-1)

        # Encode to emotion embedding
        emotion_embedding = self.spherical_encoder(spherical_features)

        # Classify emotion for supervision
        emotion_logits = self.emotion_classifier(emotion_embedding)

        # Find nearest prototype emotion
        distances = torch.cdist(vad, self.emotion_prototypes.unsqueeze(0).expand(batch_size, -1, -1))
        nearest_emotion = distances.squeeze(1).argmin(dim=-1)

        return {
            'vad': vad,
            'vad_scaled': vad_scaled,
            'intensity': intensity_tensor,
            'predicted_intensity': predicted_intensity,
            'spherical': (r, theta, phi),
            'spherical_features': spherical_features,
            'emotion_embedding': emotion_embedding,
            'emotion_logits': emotion_logits,
            'nearest_emotion': nearest_emotion,
        }

    def encode_from_label(
        self,
        emotion_idx: torch.Tensor,
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion from label (for inference/testing).

        Args:
            emotion_idx: [batch] emotion indices
            intensity: Emotion intensity

        Returns:
            Spherical emotion representation
        """
        # Get VAD from prototypes
        vad = self.emotion_prototypes[emotion_idx]  # [batch, 3]

        # Scale by intensity
        vad_scaled = vad * intensity

        # Convert to spherical
        r, theta, phi = cartesian_to_spherical(vad_scaled)

        spherical_features = torch.stack([
            r, theta, phi,
            torch.sin(theta), torch.cos(theta),
            torch.sin(phi), torch.cos(phi),
        ], dim=-1)

        emotion_embedding = self.spherical_encoder(spherical_features)

        return {
            'vad': vad,
            'vad_scaled': vad_scaled,
            'intensity': torch.full_like(r, intensity),
            'spherical': (r, theta, phi),
            'emotion_embedding': emotion_embedding,
        }


# =============================================================================
# CFM DECODER FOR FAST SYNTHESIS
# =============================================================================

class CFMVectorField(nn.Module):
    """
    Vector field network for CFM decoder.

    Predicts velocity for flow matching conditioned on:
    - Style (from multi-level encoder)
    - Emotion (EASV)
    - Text embeddings
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(128)

        # Condition fusion: style + emotion + time
        condition_dim = config.combined_style_dim + config.emotion_embedding_dim + 128

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.output_dim + condition_dim, config.cfm_hidden_dim),
            nn.LayerNorm(config.cfm_hidden_dim),
            nn.GELU(),
        )

        # Transformer for velocity prediction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.cfm_hidden_dim,
            nhead=config.cfm_num_heads,
            dim_feedforward=config.cfm_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.cfm_num_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.cfm_hidden_dim),
            nn.Linear(config.cfm_hidden_dim, config.cfm_hidden_dim),
            nn.GELU(),
            nn.Linear(config.cfm_hidden_dim, config.output_dim),
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        style: torch.Tensor,
        emotion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field.

        Args:
            t: [batch] time values in [0, 1]
            x: [batch, output_dim] current state
            style: [batch, combined_style_dim] style embedding
            emotion: [batch, emotion_embedding_dim] emotion embedding

        Returns:
            Velocity [batch, output_dim]
        """
        batch_size = x.shape[0]

        if t.dim() == 0:
            t = t.expand(batch_size)

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, 128]

        # Concatenate conditions
        condition = torch.cat([style, emotion, t_emb], dim=-1)

        # Concatenate with state
        x_cond = torch.cat([x, condition], dim=-1)

        # Project
        h = self.input_proj(x_cond)

        # Add sequence dimension for transformer
        h = h.unsqueeze(1)  # [batch, 1, hidden]

        # Transform
        h = self.transformer(h)

        # Remove sequence dimension
        h = h.squeeze(1)

        # Output velocity
        velocity = self.output_proj(h)

        return velocity


class CFMDecoder(nn.Module):
    """
    Conditional Flow Matching decoder for fast synthesis.

    Uses optimal transport CFM for few-step generation (4-8 steps).
    Generates prosody tokens conditioned on style and emotion.
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Vector field network
        self.vector_field = CFMVectorField(config)

        # Gaussian path
        self.path = GaussianConditionalPath(config.cfm_sigma_min)

        # Optional OT coupling
        self.ot_coupler = OptimalTransportCoupling(reg=0.05)

        # Token projection
        self.token_proj = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )
        self.token_norm = nn.LayerNorm(config.output_dim)

    def compute_loss(
        self,
        target: torch.Tensor,
        style: torch.Tensor,
        emotion: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CFM training loss.

        Args:
            target: [batch, output_dim] target prosody
            style: [batch, combined_style_dim] style embedding
            emotion: [batch, emotion_embedding_dim] emotion embedding

        Returns:
            Loss dict
        """
        batch_size = target.shape[0]
        device = target.device

        # Sample noise
        x0 = torch.randn_like(target)

        # OT coupling for better training
        if batch_size > 1:
            x0, target_coupled = self.ot_coupler.get_coupling(x0, target)
        else:
            target_coupled = target

        # Sample time
        t = torch.rand(batch_size, device=device)

        # Sample x_t
        x_t = self.path.sample_xt(t, x0, target_coupled)

        # Compute target velocity
        target_velocity = self.path.compute_target_velocity(t, x_t, target_coupled)

        # Predict velocity
        predicted_velocity = self.vector_field(t, x_t, style, emotion)

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
        style: torch.Tensor,
        emotion: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample prosody using CFM.

        Args:
            style: [batch, combined_style_dim]
            emotion: [batch, emotion_embedding_dim]
            num_steps: Number of ODE steps (default from config)
            temperature: Sampling temperature

        Returns:
            Sampled prosody [batch, output_dim]
        """
        batch_size = style.shape[0]
        device = style.device

        if num_steps is None:
            num_steps = self.config.cfm_num_steps_inference

        # Initial noise
        x = torch.randn(batch_size, self.config.output_dim, device=device) * temperature

        # Solve ODE
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.tensor(i * dt, device=device)
            velocity = self.vector_field(t, x, style, emotion)
            x = x + dt * velocity

        return x

    def sample_tokens(
        self,
        style: torch.Tensor,
        emotion: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample and project to prosody tokens.

        Args:
            style: [batch, combined_style_dim]
            emotion: [batch, emotion_embedding_dim]
            num_steps: ODE steps
            temperature: Sampling temperature

        Returns:
            Prosody tokens [batch, num_tokens, output_dim]
        """
        # Sample prosody
        prosody = self.sample(style, emotion, num_steps, temperature)

        # Project to tokens
        tokens = self.token_proj(prosody)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.token_norm(tokens)

        return tokens


# =============================================================================
# ZERO-SHOT EMOTION TRANSFER LOSS
# =============================================================================

class ZeroShotEmotionTransferLoss(nn.Module):
    """
    Loss functions for zero-shot emotion transfer.

    Components:
    1. Style consistency: Same speaker should have similar style
    2. Emotion transfer: Different speakers with same emotion should match
    3. Speaker disentanglement: Emotion should be speaker-independent
    4. Reconstruction: Style+emotion should reconstruct prosody
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.style_consistency_weight = config.style_consistency_weight
        self.emotion_transfer_weight = config.emotion_transfer_weight
        self.speaker_disentangle_weight = config.speaker_disentangle_weight

        # Speaker classifier (for adversarial disentanglement)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(config.emotion_embedding_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1000),  # Assume max 1000 speakers
        )

        # Emotion classifier
        self.emotion_classifier = nn.Linear(config.emotion_embedding_dim, config.num_emotions)

    def forward(
        self,
        style1: torch.Tensor,
        style2: torch.Tensor,
        emotion1: torch.Tensor,
        emotion2: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        same_speaker_mask: Optional[torch.Tensor] = None,
        same_emotion_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute zero-shot transfer losses.

        Args:
            style1, style2: Style embeddings from two samples
            emotion1, emotion2: Emotion embeddings from two samples
            speaker_ids: Speaker identity labels
            emotion_labels: Emotion labels
            same_speaker_mask: Binary mask for same-speaker pairs
            same_emotion_mask: Binary mask for same-emotion pairs

        Returns:
            Dict of losses
        """
        losses = {}
        device = style1.device

        # Style consistency loss: same speaker → similar style
        if same_speaker_mask is not None:
            style_sim = F.cosine_similarity(style1, style2, dim=-1)
            # High similarity for same speaker
            style_target = same_speaker_mask.float()
            losses['style_consistency'] = F.mse_loss(style_sim, style_target)
        else:
            losses['style_consistency'] = torch.tensor(0.0, device=device)

        # Emotion transfer loss: same emotion → similar emotion embedding
        if same_emotion_mask is not None:
            emotion_sim = F.cosine_similarity(emotion1, emotion2, dim=-1)
            emotion_target = same_emotion_mask.float()
            losses['emotion_transfer'] = F.mse_loss(emotion_sim, emotion_target)
        else:
            losses['emotion_transfer'] = torch.tensor(0.0, device=device)

        # Speaker disentanglement: emotion shouldn't predict speaker
        if speaker_ids is not None:
            speaker_logits = self.speaker_classifier(emotion1)
            # Use gradient reversal or negative gradient
            speaker_loss = F.cross_entropy(speaker_logits, speaker_ids)
            losses['speaker_disentangle'] = -speaker_loss  # Negative to maximize entropy
        else:
            losses['speaker_disentangle'] = torch.tensor(0.0, device=device)

        # Emotion classification (for emotion embedding quality)
        if emotion_labels is not None:
            emotion_logits = self.emotion_classifier(emotion1)
            losses['emotion_classification'] = F.cross_entropy(emotion_logits, emotion_labels)
        else:
            losses['emotion_classification'] = torch.tensor(0.0, device=device)

        # Total loss
        total = (
            losses['style_consistency'] * self.style_consistency_weight +
            losses['emotion_transfer'] * self.emotion_transfer_weight +
            losses['speaker_disentangle'] * self.speaker_disentangle_weight +
            losses.get('emotion_classification', 0.0) * 0.3
        )
        losses['total'] = total

        return losses


# =============================================================================
# EMOSPHERE++ COMPLETE MODEL
# =============================================================================

class EmoSpherePlus(nn.Module):
    """
    EmoSphere++: Complete model for zero-shot emotional TTS.

    Combines:
    1. Multi-level style encoder (global + local)
    2. Emotion-Adaptive Spherical Vector (EASV)
    3. CFM decoder for fast synthesis
    4. Zero-shot emotion transfer capabilities

    Usage:
        model = EmoSpherePlus(config)

        # Extract style and emotion from reference
        output = model(mel, features)
        prosody_tokens = output['prosody_tokens']

        # Zero-shot emotion transfer
        tokens = model.transfer_emotion(
            source_mel, source_features,  # Content/style source
            target_emotion="happy",        # Target emotion
            intensity=0.8
        )
    """

    def __init__(self, config: EmoSpherePlusConfig):
        super().__init__()
        self.config = config

        # Multi-level style encoder
        self.style_encoder = MultiLevelStyleEncoder(config)

        # EASV module
        self.easv = EmotionAdaptiveSphericalVector(config)

        # CFM decoder
        self.cfm_decoder = CFMDecoder(config)

        # Loss modules
        self.transfer_loss = ZeroShotEmotionTransferLoss(config)

    def forward(
        self,
        mel: torch.Tensor,
        features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        intensity: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass extracting style, emotion, and generating prosody.

        Args:
            mel: [batch, time, mel_dim] mel spectrogram
            features: [batch, seq, input_dim] audio features
            feature_mask: [batch, seq] attention mask
            prosody_features: Optional [batch, 15] prosodic statistics
            intensity: Optional emotion intensity override

        Returns:
            Dict with all outputs
        """
        # Extract multi-level style
        style_output = self.style_encoder(mel, features, feature_mask, return_components=True)
        combined_style = style_output['combined_style']

        # Compute EASV
        easv_output = self.easv(combined_style, prosody_features, intensity)
        emotion_embedding = easv_output['emotion_embedding']

        # Generate prosody tokens via CFM
        prosody_tokens = self.cfm_decoder.sample_tokens(combined_style, emotion_embedding)

        return {
            'prosody_tokens': prosody_tokens,
            'style': style_output,
            'easv': easv_output,
            'combined_style': combined_style,
            'emotion_embedding': emotion_embedding,
            'vad': easv_output['vad'],
            'intensity': easv_output['intensity'],
        }

    def compute_training_loss(
        self,
        mel: torch.Tensor,
        features: torch.Tensor,
        target_prosody: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses.

        Args:
            mel: Mel spectrogram
            features: Audio features
            target_prosody: Target prosody for reconstruction
            feature_mask: Attention mask
            prosody_features: Prosodic statistics
            emotion_labels: Ground truth emotions
            speaker_ids: Speaker identities

        Returns:
            Dict of all losses
        """
        # Forward pass
        output = self.forward(mel, features, feature_mask, prosody_features)

        losses = {}

        # CFM reconstruction loss
        cfm_losses = self.cfm_decoder.compute_loss(
            target_prosody,
            output['combined_style'],
            output['emotion_embedding'],
        )
        losses['cfm'] = cfm_losses['loss']

        # EASV reconstruction (VAD should match prosody patterns)
        if prosody_features is not None:
            prosody_vad_output = self.easv.prosody_vad(prosody_features)
            losses['easv_reconstruction'] = F.mse_loss(
                output['vad'], prosody_vad_output['vad']
            )
        else:
            losses['easv_reconstruction'] = torch.tensor(0.0, device=mel.device)

        # Emotion classification
        if emotion_labels is not None:
            losses['emotion_classification'] = F.cross_entropy(
                output['easv']['emotion_logits'], emotion_labels
            )
        else:
            losses['emotion_classification'] = torch.tensor(0.0, device=mel.device)

        # Total loss
        total = (
            losses['cfm'] * self.config.cfm_loss_weight +
            losses['easv_reconstruction'] * self.config.easv_reconstruction_weight +
            losses.get('emotion_classification', 0.0) * 0.3
        )
        losses['total'] = total

        return losses

    def transfer_emotion(
        self,
        source_mel: torch.Tensor,
        source_features: torch.Tensor,
        target_emotion: Union[str, int, torch.Tensor],
        intensity: float = 0.7,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Zero-shot emotion transfer.

        Keep style from source, apply target emotion.

        Args:
            source_mel: Source mel spectrogram
            source_features: Source audio features
            target_emotion: Target emotion (name, index, or VAD)
            intensity: Emotion intensity
            feature_mask: Attention mask

        Returns:
            Prosody tokens with transferred emotion
        """
        # Extract style from source
        style_output = self.style_encoder(source_mel, source_features, feature_mask, return_components=True)
        combined_style = style_output['combined_style']

        # Get target emotion embedding
        if isinstance(target_emotion, str):
            emotion_idx = CORE_EMOTIONS.index(target_emotion.lower())
            emotion_idx = torch.tensor([emotion_idx], device=combined_style.device)
            easv_output = self.easv.encode_from_label(emotion_idx, intensity)
        elif isinstance(target_emotion, int):
            emotion_idx = torch.tensor([target_emotion], device=combined_style.device)
            easv_output = self.easv.encode_from_label(emotion_idx, intensity)
        else:
            # Assume VAD tensor
            vad = target_emotion.to(combined_style.device)
            if vad.dim() == 1:
                vad = vad.unsqueeze(0)
            # Scale by intensity
            vad_scaled = vad * intensity
            r, theta, phi = cartesian_to_spherical(vad_scaled)
            spherical_features = torch.stack([
                r, theta, phi,
                torch.sin(theta), torch.cos(theta),
                torch.sin(phi), torch.cos(phi),
            ], dim=-1)
            emotion_embedding = self.easv.spherical_encoder(spherical_features)
            easv_output = {'emotion_embedding': emotion_embedding}

        emotion_embedding = easv_output['emotion_embedding']

        # Expand if needed
        if emotion_embedding.shape[0] == 1 and combined_style.shape[0] > 1:
            emotion_embedding = emotion_embedding.expand(combined_style.shape[0], -1)

        # Generate with transferred emotion
        prosody_tokens = self.cfm_decoder.sample_tokens(combined_style, emotion_embedding)

        return prosody_tokens

    def interpolate_emotions(
        self,
        mel: torch.Tensor,
        features: torch.Tensor,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 0.7,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate between two emotions.

        Args:
            mel: Source mel spectrogram
            features: Source audio features
            emotion1: Start emotion
            emotion2: End emotion
            t: Interpolation factor [0, 1]
            intensity: Emotion intensity
            feature_mask: Attention mask

        Returns:
            Prosody tokens with interpolated emotion
        """
        # Extract style
        style_output = self.style_encoder(mel, features, feature_mask, return_components=True)
        combined_style = style_output['combined_style']

        # Get emotion VADs
        idx1 = CORE_EMOTIONS.index(emotion1.lower())
        idx2 = CORE_EMOTIONS.index(emotion2.lower())

        vad1 = self.easv.emotion_prototypes[idx1]
        vad2 = self.easv.emotion_prototypes[idx2]

        # Spherical interpolation
        vad_interp = EmotionInterpolator.slerp(
            vad1.unsqueeze(0), vad2.unsqueeze(0), t
        )

        # Scale by intensity
        vad_scaled = vad_interp * intensity

        # Encode
        r, theta, phi = cartesian_to_spherical(vad_scaled)
        spherical_features = torch.stack([
            r, theta, phi,
            torch.sin(theta), torch.cos(theta),
            torch.sin(phi), torch.cos(phi),
        ], dim=-1)
        emotion_embedding = self.easv.spherical_encoder(spherical_features)

        # Expand if needed
        if emotion_embedding.shape[0] == 1 and combined_style.shape[0] > 1:
            emotion_embedding = emotion_embedding.expand(combined_style.shape[0], -1)

        # Generate
        prosody_tokens = self.cfm_decoder.sample_tokens(combined_style, emotion_embedding)

        return prosody_tokens


# =============================================================================
# ADAPTER FOR CSM INTEGRATION
# =============================================================================

class EmoSpherePlusAdapter(nn.Module):
    """
    Adapter for integrating EmoSphere++ with the CSM prosody pipeline.

    Provides convenient interface for:
    - Extracting prosody tokens from audio
    - Emotion control via labels or VAD
    - Zero-shot emotion transfer
    - Intensity control
    """

    def __init__(
        self,
        config: EmoSpherePlusConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core model
        self.model = EmoSpherePlus(config)

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
        mel: torch.Tensor,
        features: torch.Tensor,
        feature_mask: Optional[torch.Tensor] = None,
        prosody_features: Optional[torch.Tensor] = None,
        intensity: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning adapted prosody tokens.
        """
        output = self.model(mel, features, feature_mask, prosody_features, intensity)

        # Adapt tokens
        tokens = self.output_adapter(output['prosody_tokens'])

        return {
            'prosody_tokens': tokens,
            'vad': output['vad'],
            'intensity': output['intensity'],
            'emotion_embedding': output['emotion_embedding'],
            'combined_style': output['combined_style'],
        }

    def from_emotion(
        self,
        mel: torch.Tensor,
        features: torch.Tensor,
        emotion: Union[str, int],
        intensity: float = 0.7,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate with specified emotion.
        """
        tokens = self.model.transfer_emotion(
            mel, features, emotion, intensity, feature_mask
        )
        tokens = self.output_adapter(tokens)

        return {'prosody_tokens': tokens}

    def interpolate(
        self,
        mel: torch.Tensor,
        features: torch.Tensor,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 0.7,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between emotions.
        """
        tokens = self.model.interpolate_emotions(
            mel, features, emotion1, emotion2, t, intensity, feature_mask
        )
        tokens = self.output_adapter(tokens)

        return {'prosody_tokens': tokens}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_emosphere_plus_adapter(
    config: Optional[EmoSpherePlusConfig] = None,
    prosody_hidden: int = 2048,
) -> EmoSpherePlusAdapter:
    """Create EmoSphere++ adapter with default config."""
    if config is None:
        config = EmoSpherePlusConfig()
    return EmoSpherePlusAdapter(config, prosody_hidden)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EmoSphere++: Multi-Level Style Encoder - Test Suite")
    print("=" * 70)

    config = EmoSpherePlusConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Test 1: Reference Encoder
    print("\n[Test 1] Reference Encoder...")
    ref_encoder = ReferenceEncoder(config).to(device)

    batch_size = 2
    time_frames = 100
    mel = torch.randn(batch_size, time_frames, config.mel_dim, device=device)

    style = ref_encoder(mel)
    print(f"  Input mel: {mel.shape}")
    print(f"  Output style: {style.shape}")
    assert style.shape == (batch_size, config.global_style_dim)
    print("  [PASS]")

    # Test 2: Style Token Attention
    print("\n[Test 2] Style Token Attention...")
    style_attn = StyleTokenAttention(config).to(device)

    attended, weights = style_attn(style, return_weights=True)
    print(f"  Input style: {style.shape}")
    print(f"  Attended output: {attended.shape}")
    print(f"  Attention weights: {weights.shape}")
    print(f"  Weights sum: {weights.sum(dim=-1).mean():.4f} (should be 1.0)")
    print("  [PASS]")

    # Test 3: Global Style Encoder
    print("\n[Test 3] Global Style Encoder...")
    global_encoder = GlobalStyleEncoder(config).to(device)

    output = global_encoder(mel, return_details=True)
    print(f"  Global style shape: {output['style'].shape}")
    print(f"  Reference encoding: {output['reference_encoding'].shape}")
    print("  [PASS]")

    # Test 4: Local Style Encoder
    print("\n[Test 4] Local Style Encoder...")
    local_encoder = LocalStyleEncoder(config).to(device)

    seq_len = 50
    features = torch.randn(batch_size, seq_len, config.input_dim, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    local_style, local_frames = local_encoder(features, mask, return_sequence=True)
    print(f"  Input features: {features.shape}")
    print(f"  Local style: {local_style.shape}")
    print(f"  Local frames: {local_frames.shape}")
    print("  [PASS]")

    # Test 5: Multi-Level Style Encoder
    print("\n[Test 5] Multi-Level Style Encoder...")
    ml_encoder = MultiLevelStyleEncoder(config).to(device)

    output = ml_encoder(mel, features, mask, return_components=True)
    print(f"  Combined style: {output['combined_style'].shape}")
    print(f"  Global style: {output['global_style'].shape}")
    print(f"  Local style: {output['local_style'].shape}")
    print(f"  Gate weights: {output['gate_weights'].tolist()}")
    print("  [PASS]")

    # Test 6: Prosody to VAD Predictor
    print("\n[Test 6] Prosody to VAD Predictor...")
    prosody_vad = ProsodyToVADPredictor(config).to(device)

    prosody_features = torch.randn(batch_size, 15, device=device)
    vad_output = prosody_vad(prosody_features)
    print(f"  Input prosody: {prosody_features.shape}")
    print(f"  Output VAD: {vad_output['vad'].shape}")
    print(f"  VAD values: {vad_output['vad'][0].tolist()}")
    print("  [PASS]")

    # Test 7: EASV
    print("\n[Test 7] Emotion-Adaptive Spherical Vector (EASV)...")
    easv = EmotionAdaptiveSphericalVector(config).to(device)

    combined_style = output['combined_style']
    easv_output = easv(combined_style, prosody_features)
    print(f"  Input style: {combined_style.shape}")
    print(f"  Output VAD: {easv_output['vad'].shape}")
    print(f"  Intensity: {easv_output['intensity'].tolist()}")
    print(f"  Emotion embedding: {easv_output['emotion_embedding'].shape}")
    print(f"  Nearest emotion: {[CORE_EMOTIONS[i] for i in easv_output['nearest_emotion'].tolist()]}")
    print("  [PASS]")

    # Test 8: EASV from label
    print("\n[Test 8] EASV from emotion label...")
    emotion_idx = torch.tensor([1, 3], device=device)  # happy, angry
    easv_label = easv.encode_from_label(emotion_idx, intensity=0.8)
    print(f"  Emotions: happy, angry")
    print(f"  VAD happy: {easv_label['vad'][0].tolist()}")
    print(f"  VAD angry: {easv_label['vad'][1].tolist()}")
    print("  [PASS]")

    # Test 9: CFM Vector Field
    print("\n[Test 9] CFM Vector Field...")
    vector_field = CFMVectorField(config).to(device)

    t = torch.rand(batch_size, device=device)
    x = torch.randn(batch_size, config.output_dim, device=device)
    emotion_emb = easv_output['emotion_embedding']

    velocity = vector_field(t, x, combined_style, emotion_emb)
    print(f"  Time: {t.shape}")
    print(f"  State: {x.shape}")
    print(f"  Velocity: {velocity.shape}")
    print(f"  Velocity norm: {velocity.norm(dim=-1).mean():.4f}")
    print("  [PASS]")

    # Test 10: CFM Decoder
    print("\n[Test 10] CFM Decoder...")
    cfm = CFMDecoder(config).to(device)

    # Training loss
    target = torch.randn(batch_size, config.output_dim, device=device)
    cfm_loss = cfm.compute_loss(target, combined_style, emotion_emb)
    print(f"  CFM loss: {cfm_loss['loss'].item():.4f}")

    # Sampling
    sampled = cfm.sample(combined_style, emotion_emb)
    print(f"  Sampled prosody: {sampled.shape}")

    # Token generation
    tokens = cfm.sample_tokens(combined_style, emotion_emb)
    print(f"  Generated tokens: {tokens.shape}")
    print("  [PASS]")

    # Test 11: Complete EmoSphere++ Model
    print("\n[Test 11] Complete EmoSphere++ Model...")
    model = EmoSpherePlus(config).to(device)

    output = model(mel, features, mask, prosody_features)
    print(f"  Prosody tokens: {output['prosody_tokens'].shape}")
    print(f"  VAD: {output['vad'].shape}")
    print(f"  Intensity: {output['intensity'].shape}")
    print("  [PASS]")

    # Test 12: Training Loss
    print("\n[Test 12] Training Loss...")
    target_prosody = torch.randn(batch_size, config.output_dim, device=device)
    emotion_labels = torch.randint(0, config.num_emotions, (batch_size,), device=device)

    losses = model.compute_training_loss(
        mel, features, target_prosody, mask, prosody_features, emotion_labels
    )
    print(f"  CFM loss: {losses['cfm'].item():.4f}")
    print(f"  EASV recon: {losses['easv_reconstruction'].item():.4f}")
    print(f"  Emotion class: {losses['emotion_classification'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 13: Zero-Shot Emotion Transfer
    print("\n[Test 13] Zero-Shot Emotion Transfer...")
    transferred = model.transfer_emotion(mel, features, "happy", intensity=0.9, feature_mask=mask)
    print(f"  Transferred tokens: {transferred.shape}")

    transferred_angry = model.transfer_emotion(mel, features, "angry", intensity=0.8, feature_mask=mask)
    print(f"  Happy vs Angry token diff: {(transferred - transferred_angry).norm():.4f}")
    print("  [PASS]")

    # Test 14: Emotion Interpolation
    print("\n[Test 14] Emotion Interpolation...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interp = model.interpolate_emotions(mel, features, "sad", "happy", t, intensity=0.7, feature_mask=mask)
        print(f"  t={t:.2f}: token norm = {interp.norm():.4f}")
    print("  [PASS]")

    # Test 15: Adapter
    print("\n[Test 15] EmoSphere++ Adapter...")
    adapter = EmoSpherePlusAdapter(config).to(device)

    result = adapter(mel, features, mask, prosody_features)
    print(f"  Prosody tokens: {result['prosody_tokens'].shape}")

    result_emotion = adapter.from_emotion(mel, features, "surprised", intensity=0.8, feature_mask=mask)
    print(f"  From emotion: {result_emotion['prosody_tokens'].shape}")

    result_interp = adapter.interpolate(mel, features, "calm", "fearful", t=0.5, intensity=0.7, feature_mask=mask)
    print(f"  Interpolated: {result_interp['prosody_tokens'].shape}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All EmoSphere++ tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from emosphere_plus import (
    EmoSpherePlusConfig,
    EmoSpherePlus,
    EmoSpherePlusAdapter,
    create_emosphere_plus_adapter,
)

# Initialize
config = EmoSpherePlusConfig()
adapter = create_emosphere_plus_adapter(config).cuda()

# Extract style and emotion from reference audio
result = adapter(
    mel=mel_spectrogram,         # [batch, time, 80]
    features=audio_features,      # [batch, seq, 768] (wav2vec2/HuBERT)
    feature_mask=mask,
)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

# Zero-shot emotion transfer
tokens = adapter.from_emotion(
    mel, features,
    emotion="happy",
    intensity=0.9,
)

# Emotion interpolation
tokens = adapter.interpolate(
    mel, features,
    emotion1="sad",
    emotion2="happy",
    t=0.5,  # Halfway between sad and happy
    intensity=0.8,
)

# Access EASV components
print(f"VAD: {result['vad']}")
print(f"Intensity: {result['intensity']}")

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
