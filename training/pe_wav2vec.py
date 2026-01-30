"""
PE-wav2vec: Prosody-Enhanced wav2vec 2.0 for Self-Supervised Prosody Learning

Based on "PE-Wav2vec: A Prosody-Enhanced Speech Model for Self-Supervised Prosody Learning in TTS"
(IEEE/ACM TASLP 2024) - https://ieeexplore.ieee.org/document/10645206/

Key Innovation: Apply LPC residual signal supervision to initial Transformer blocks
of wav2vec 2.0 architecture to learn prosodic features.

Why LPC Residual?
- LPC (Linear Predictive Coding) separates speech into:
  - Filter component (LPC coefficients): Vocal tract shape, formants
  - Source component (residual): Excitation signal, contains pitch/prosody info
- Supervising early transformer layers with LPC residual encourages learning of
  prosodic features (pitch, rhythm, energy patterns) in a self-supervised manner

Architecture:
1. LPC Residual Extractor: Extract prosody-rich residual signals from speech
2. PE-wav2vec Encoder: wav2vec 2.0 with auxiliary LPC residual prediction loss
   on initial transformer blocks (layers 1-4)
3. Prosody Embedding Extractor: Extract embeddings from prosody-supervised layers
4. S4LPR Adapter: Integration with TTS/CSM pipeline (similar to FastSpeech 2)

Benefits over vanilla wav2vec:
- Richer prosody descriptions without text transcription
- Self-supervised learning from large-scale unlabeled data
- Frame-level prosodic representations
- Better for TTS prosody conditioning than content-focused representations

Usage:
    from pe_wav2vec import (
        PEWav2VecConfig,
        LPCResidualExtractor,
        PEWav2VecEncoder,
        PEWav2VecAdapter,
        S4LPRProsodyPredictor,
    )

    # Initialize
    config = PEWav2VecConfig()
    encoder = PEWav2VecEncoder(config).cuda()

    # Training: with LPC residual supervision
    output = encoder(audio, compute_lpc_loss=True)
    loss = output['total_loss']  # wav2vec loss + LPC auxiliary loss

    # Inference: extract prosody embeddings
    prosody_emb = encoder.get_prosody_embedding(audio)

    # CSM integration
    adapter = PEWav2VecAdapter(config, encoder)
    prefix_tokens = adapter(audio)  # [batch, 4, 2048]

References:
- Paper: https://ieeexplore.ieee.org/document/10645206/
- Demo: https://ttsbylzc.github.io/PE-wav2vec
- Original wav2vec 2.0: https://arxiv.org/abs/2006.11477
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PEWav2VecConfig:
    """Configuration for PE-wav2vec prosody-enhanced encoder."""

    # wav2vec 2.0 base model
    wav2vec_model: str = "facebook/wav2vec2-base-960h"
    freeze_feature_extractor: bool = True  # Freeze CNN feature extractor
    freeze_upper_layers: bool = False  # Optionally freeze upper transformer layers

    # Prosody supervision layers
    prosody_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    # Which transformer layers to apply LPC residual supervision
    # Paper suggests initial blocks (1-4) for prosody learning

    # LPC settings
    lpc_order: int = 16  # LPC filter order (typical: 10-20)
    lpc_frame_length_ms: float = 25.0  # Frame length in ms
    lpc_hop_length_ms: float = 10.0  # Hop length in ms

    # Model dimensions
    hidden_dim: int = 768  # wav2vec2-base hidden dimension
    prosody_dim: int = 256  # Dimension for prosody features
    lpc_residual_dim: int = 64  # Dimension for LPC residual prediction

    # Auxiliary loss settings
    lpc_loss_weight: float = 0.1  # Weight for LPC auxiliary loss
    contrastive_loss_weight: float = 1.0  # Weight for wav2vec contrastive loss

    # Prosody embedding extraction
    prosody_pooling: str = "attention"  # mean, attention, or first
    num_prosody_heads: int = 4  # Attention heads for prosody pooling

    # Training settings
    dropout: float = 0.1
    layer_drop: float = 0.05  # Transformer layer dropout

    # Audio settings
    sample_rate: int = 16000  # wav2vec expects 16kHz
    normalize_audio: bool = True

    # Output for CSM integration
    output_dim: int = 2048  # CSM hidden size
    num_prefix_tokens: int = 4  # Number of prosody prefix tokens


# =============================================================================
# LPC RESIDUAL EXTRACTOR
# =============================================================================

class LPCResidualExtractor(nn.Module):
    """
    Extract LPC residual signals from speech for prosody supervision.

    LPC (Linear Predictive Coding) decomposes speech into:
    - Filter: LPC coefficients representing vocal tract (formants)
    - Source: Residual signal representing excitation (pitch/prosody)

    The residual is prosody-rich because it captures:
    - Fundamental frequency (F0) patterns
    - Energy/intensity variations
    - Temporal rhythm patterns

    Uses the Levinson-Durbin algorithm for LPC analysis.
    """

    def __init__(self, config: PEWav2VecConfig):
        super().__init__()
        self.config = config

        self.order = config.lpc_order
        self.frame_length = int(config.lpc_frame_length_ms * config.sample_rate / 1000)
        self.hop_length = int(config.lpc_hop_length_ms * config.sample_rate / 1000)

        # Window function
        self.register_buffer(
            'window',
            torch.hamming_window(self.frame_length)
        )

        # Pre-emphasis filter (high-pass to emphasize higher frequencies)
        self.pre_emphasis = 0.97

    def _levinson_durbin(
        self,
        autocorr: torch.Tensor,  # [batch, order + 1]
    ) -> torch.Tensor:
        """
        Levinson-Durbin recursion for computing LPC coefficients.

        Args:
            autocorr: Autocorrelation values [batch, order + 1]

        Returns:
            LPC coefficients [batch, order]
        """
        batch_size = autocorr.shape[0]
        device = autocorr.device

        # Initialize
        a = torch.zeros(batch_size, self.order + 1, device=device)
        a[:, 0] = 1.0

        e = autocorr[:, 0].clone()  # Prediction error

        for i in range(1, self.order + 1):
            # Compute reflection coefficient
            lambda_sum = torch.zeros(batch_size, device=device)
            for j in range(1, i):
                lambda_sum += a[:, j] * autocorr[:, i - j]

            # Avoid division by zero
            k = -(autocorr[:, i] + lambda_sum) / (e + 1e-10)

            # Update coefficients
            a_new = a.clone()
            for j in range(1, i):
                a_new[:, j] = a[:, j] + k * a[:, i - j]
            a_new[:, i] = k

            a = a_new

            # Update error
            e = e * (1 - k * k)

        return a[:, 1:]  # Return coefficients (exclude a[0]=1)

    def _compute_autocorrelation(
        self,
        frames: torch.Tensor,  # [batch, num_frames, frame_length]
    ) -> torch.Tensor:
        """
        Compute autocorrelation for each frame.

        Returns:
            [batch, num_frames, order + 1]
        """
        batch_size, num_frames, frame_length = frames.shape
        device = frames.device

        # Apply window
        windowed = frames * self.window.unsqueeze(0).unsqueeze(0)

        # Compute autocorrelation via FFT
        n_fft = 2 ** (int(math.log2(frame_length)) + 1)  # Next power of 2
        fft = torch.fft.rfft(windowed, n=n_fft, dim=-1)
        power_spectrum = fft.real ** 2 + fft.imag ** 2
        autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=-1)

        # Take only first order+1 values
        autocorr = autocorr_full[..., :self.order + 1]

        return autocorr

    def _apply_inverse_filter(
        self,
        frames: torch.Tensor,  # [batch, num_frames, frame_length]
        lpc_coeffs: torch.Tensor,  # [batch, num_frames, order]
    ) -> torch.Tensor:
        """
        Apply inverse filter to get LPC residual.

        The residual e[n] = x[n] + sum(a[k] * x[n-k]) for k=1..order

        Returns:
            [batch, num_frames, frame_length]
        """
        batch_size, num_frames, frame_length = frames.shape
        device = frames.device

        # Pad for filtering
        padded = F.pad(frames, (self.order, 0), mode='constant', value=0)

        residual = frames.clone()

        for k in range(self.order):
            shifted = padded[:, :, self.order - k - 1:self.order - k - 1 + frame_length]
            residual = residual + lpc_coeffs[:, :, k:k+1] * shifted

        return residual

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract LPC residual from audio.

        Args:
            audio: Raw audio waveform at 16kHz

        Returns:
            Dict containing:
                - residual: [batch, num_frames, frame_length] LPC residual
                - lpc_coeffs: [batch, num_frames, order] LPC coefficients
                - energy: [batch, num_frames] frame energy
                - residual_features: [batch, num_frames, feature_dim] residual features
        """
        batch_size = audio.shape[0]
        device = audio.device

        # Pre-emphasis
        audio_emph = torch.cat([
            audio[:, :1],
            audio[:, 1:] - self.pre_emphasis * audio[:, :-1]
        ], dim=1)

        # Frame the signal
        num_frames = (audio.shape[1] - self.frame_length) // self.hop_length + 1
        frames = audio_emph.unfold(1, self.frame_length, self.hop_length)  # [batch, num_frames, frame_length]

        # Compute autocorrelation
        autocorr = self._compute_autocorrelation(frames)

        # Compute LPC coefficients for each frame
        # Reshape for batch processing
        autocorr_flat = autocorr.reshape(-1, self.order + 1)
        lpc_flat = self._levinson_durbin(autocorr_flat)
        lpc_coeffs = lpc_flat.reshape(batch_size, num_frames, self.order)

        # Apply inverse filter to get residual
        residual = self._apply_inverse_filter(frames, lpc_coeffs)

        # Compute frame energy (log energy of residual)
        energy = torch.log(residual.pow(2).mean(dim=-1) + 1e-8)

        # Extract features from residual (simplified - use FFT magnitude)
        residual_fft = torch.fft.rfft(residual, dim=-1)
        residual_features = torch.abs(residual_fft)[:, :, :self.config.lpc_residual_dim]

        return {
            'residual': residual,
            'lpc_coeffs': lpc_coeffs,
            'energy': energy,
            'residual_features': residual_features,
            'num_frames': num_frames,
        }

    def get_target_for_layer(
        self,
        residual_features: torch.Tensor,  # [batch, num_frames, feature_dim]
        target_length: int,
    ) -> torch.Tensor:
        """
        Resample residual features to match transformer layer output length.

        Args:
            residual_features: LPC residual features
            target_length: Target sequence length from wav2vec

        Returns:
            [batch, target_length, feature_dim] resampled features
        """
        # Interpolate to match wav2vec output length
        features = residual_features.transpose(1, 2)  # [batch, feature_dim, frames]
        resampled = F.interpolate(
            features,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        return resampled.transpose(1, 2)  # [batch, target_length, feature_dim]


# =============================================================================
# PE-WAV2VEC ENCODER
# =============================================================================

class LPCAuxiliaryHead(nn.Module):
    """
    Auxiliary head for predicting LPC residual features from transformer hidden states.

    Applied to initial transformer blocks to encourage prosody learning.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, target_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden_dim]
    ) -> torch.Tensor:
        """
        Predict LPC residual features from hidden states.

        Returns:
            [batch, seq, target_dim]
        """
        return self.proj(hidden_states)


class ProsodyAttentionPooling(nn.Module):
    """
    Attention-based pooling for prosody embeddings.

    Learns to weight frames based on their prosodic importance.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_heads = num_heads

        # Query vector (learnable)
        self.query = nn.Parameter(torch.randn(1, num_heads, hidden_dim // num_heads))

        # Key and value projections
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, self.output_dim)

        self.scale = (hidden_dim // num_heads) ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden_dim]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool hidden states to single prosody embedding.

        Returns:
            [batch, output_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project keys and values
        keys = self.key_proj(hidden_states)  # [batch, seq, hidden]
        values = self.value_proj(hidden_states)  # [batch, seq, hidden]

        # Reshape for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # [batch, heads, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(query.unsqueeze(2), keys.transpose(-2, -1)) * self.scale
        # [batch, heads, 1, seq]

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attn_weights, values)  # [batch, heads, 1, head_dim]
        attended = attended.squeeze(2).reshape(batch_size, -1)  # [batch, hidden]

        # Output projection
        output = self.out_proj(attended)

        return output


class PEWav2VecEncoder(nn.Module):
    """
    Prosody-Enhanced wav2vec 2.0 encoder.

    Extends wav2vec 2.0 with LPC residual supervision on initial transformer blocks
    to learn prosodic features in a self-supervised manner.

    Key modifications:
    1. Add auxiliary LPC prediction heads to prosody layers (1-4)
    2. Extract prosody embeddings from these early layers
    3. Maintain original contrastive learning for content
    """

    def __init__(self, config: PEWav2VecConfig):
        super().__init__()
        self.config = config

        # LPC residual extractor
        self.lpc_extractor = LPCResidualExtractor(config)

        # Load pretrained wav2vec 2.0
        self._wav2vec = None
        self._wav2vec_loaded = False

        # Auxiliary LPC prediction heads for prosody layers
        self.lpc_heads = nn.ModuleDict({
            f"layer_{i}": LPCAuxiliaryHead(
                hidden_dim=config.hidden_dim,
                target_dim=config.lpc_residual_dim,
                dropout=config.dropout,
            )
            for i in config.prosody_layers
        })

        # Prosody feature fusion
        num_prosody_layers = len(config.prosody_layers)
        self.prosody_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * num_prosody_layers, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.prosody_dim),
            nn.LayerNorm(config.prosody_dim),
        )

        # Prosody pooling
        if config.prosody_pooling == "attention":
            self.prosody_pooling = ProsodyAttentionPooling(
                hidden_dim=config.prosody_dim,
                num_heads=config.num_prosody_heads,
                output_dim=config.prosody_dim,
            )
        else:
            self.prosody_pooling = None

        # Output projection for CSM integration
        self.output_proj = nn.Linear(config.prosody_dim, config.output_dim)

    def _load_wav2vec(self):
        """Lazy load pretrained wav2vec 2.0 model."""
        if not self._wav2vec_loaded:
            if not HAS_TRANSFORMERS:
                raise RuntimeError(
                    "transformers package required. Install with: pip install transformers"
                )

            print(f"Loading wav2vec 2.0: {self.config.wav2vec_model}")
            self._wav2vec = Wav2Vec2Model.from_pretrained(
                self.config.wav2vec_model,
                output_hidden_states=True,
            )

            # Freeze feature extractor
            if self.config.freeze_feature_extractor:
                self._wav2vec.feature_extractor._freeze_parameters()
                print("Frozen wav2vec feature extractor")

            # Optionally freeze upper layers (keep prosody layers trainable)
            if self.config.freeze_upper_layers:
                max_prosody_layer = max(self.config.prosody_layers)
                for i, layer in enumerate(self._wav2vec.encoder.layers):
                    if i >= max_prosody_layer:
                        for param in layer.parameters():
                            param.requires_grad = False
                print(f"Frozen transformer layers >= {max_prosody_layer}")

            self._wav2vec_loaded = True
            print("wav2vec 2.0 loaded successfully")

    @property
    def wav2vec(self):
        """Get wav2vec model (lazy load)."""
        self._load_wav2vec()
        return self._wav2vec

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
        attention_mask: Optional[torch.Tensor] = None,
        compute_lpc_loss: bool = True,
        return_all_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional LPC residual supervision.

        Args:
            audio: Raw audio waveform at 16kHz
            attention_mask: Optional attention mask
            compute_lpc_loss: Whether to compute LPC auxiliary loss
            return_all_hidden_states: Return all layer hidden states

        Returns:
            Dict containing:
                - prosody_embedding: [batch, prosody_dim] global prosody embedding
                - prosody_features: [batch, seq, prosody_dim] frame-level prosody
                - lpc_loss: LPC auxiliary loss (if compute_lpc_loss=True)
                - hidden_states: Dict of layer hidden states (if requested)
        """
        # Ensure wav2vec is loaded and on correct device
        device = audio.device
        if self.wav2vec.device != device:
            self._wav2vec = self._wav2vec.to(device)

        # Normalize audio if configured
        if self.config.normalize_audio:
            audio = audio / (audio.abs().max(dim=-1, keepdim=True).values + 1e-8)

        # Get wav2vec outputs with all hidden states
        wav2vec_output = self.wav2vec(
            audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # wav2vec hidden states: tuple of (embedding, layer1, layer2, ..., layerN)
        all_hidden_states = wav2vec_output.hidden_states
        seq_len = all_hidden_states[1].shape[1]

        # Resample attention mask to match output sequence length
        output_mask = None
        if attention_mask is not None:
            # Downsample mask to match wav2vec output length
            # wav2vec uses ~320x downsampling (hop_length)
            mask_float = attention_mask.float().unsqueeze(1)  # [batch, 1, audio_len]
            output_mask = F.interpolate(
                mask_float,
                size=seq_len,
                mode='nearest',
            ).squeeze(1).bool()  # [batch, seq_len]

        # Extract prosody layer outputs
        prosody_hidden_states = []
        for layer_idx in self.config.prosody_layers:
            # +1 because index 0 is embedding output
            if layer_idx < len(all_hidden_states):
                prosody_hidden_states.append(all_hidden_states[layer_idx])

        # Concatenate and fuse prosody features
        prosody_concat = torch.cat(prosody_hidden_states, dim=-1)
        prosody_features = self.prosody_fusion(prosody_concat)  # [batch, seq, prosody_dim]

        # Pool to global embedding
        if self.prosody_pooling is not None:
            prosody_embedding = self.prosody_pooling(prosody_features, output_mask)
        else:
            # Simple mean pooling
            if output_mask is not None:
                mask = output_mask.unsqueeze(-1).float()
                prosody_embedding = (prosody_features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                prosody_embedding = prosody_features.mean(dim=1)

        output = {
            'prosody_embedding': prosody_embedding,  # [batch, prosody_dim]
            'prosody_features': prosody_features,  # [batch, seq, prosody_dim]
            'seq_len': seq_len,
            'attention_mask_out': output_mask,
        }

        # Compute LPC auxiliary loss
        if compute_lpc_loss:
            lpc_output = self.lpc_extractor(audio)
            lpc_target = self.lpc_extractor.get_target_for_layer(
                lpc_output['residual_features'],
                target_length=seq_len,
            )

            lpc_losses = []
            for layer_idx in self.config.prosody_layers:
                if layer_idx < len(all_hidden_states):
                    hidden = all_hidden_states[layer_idx]
                    head = self.lpc_heads[f"layer_{layer_idx}"]
                    pred = head(hidden)

                    # MSE loss between predicted and actual LPC residual features
                    loss = F.mse_loss(pred, lpc_target)
                    lpc_losses.append(loss)

            total_lpc_loss = sum(lpc_losses) / len(lpc_losses) if lpc_losses else torch.tensor(0.0)
            output['lpc_loss'] = total_lpc_loss * self.config.lpc_loss_weight
            output['lpc_target'] = lpc_target

        if return_all_hidden_states:
            output['hidden_states'] = {
                f"layer_{i}": all_hidden_states[i]
                for i in range(len(all_hidden_states))
            }

        return output

    def get_prosody_embedding(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract prosody embedding for inference.

        Args:
            audio: Raw audio at 16kHz
            attention_mask: Optional mask

        Returns:
            [batch, prosody_dim] prosody embedding
        """
        with torch.no_grad():
            output = self.forward(audio, attention_mask, compute_lpc_loss=False)
        return output['prosody_embedding']

    def get_prosody_features(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract frame-level prosody features.

        Args:
            audio: Raw audio at 16kHz
            attention_mask: Optional mask

        Returns:
            [batch, seq, prosody_dim] frame-level prosody
        """
        with torch.no_grad():
            output = self.forward(audio, attention_mask, compute_lpc_loss=False)
        return output['prosody_features']


# =============================================================================
# S4LPR PROSODY PREDICTOR
# =============================================================================

class S4LPRProsodyPredictor(nn.Module):
    """
    S4LPR-style prosody predictor for TTS.

    Based on "Speech Synthesis model conditioned on Self-Supervisedly Learned
    Prosodic Representations" from the PE-wav2vec paper.

    Instead of using raw acoustic features (F0, energy) as in FastSpeech 2,
    this module predicts PE-wav2vec prosody embeddings from text.
    """

    def __init__(
        self,
        config: PEWav2VecConfig,
        text_dim: int = 512,  # Text encoder output dimension
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.config = config

        # Project text to prosody dimension
        self.text_proj = nn.Linear(text_dim, config.prosody_dim)

        # Transformer for predicting prosody from text
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.prosody_dim,
            nhead=num_heads,
            dim_feedforward=config.prosody_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Duration predictor (similar to FastSpeech 2)
        # Uses Conv1d which expects [batch, channels, seq]
        self.duration_conv = nn.Sequential(
            nn.Conv1d(config.prosody_dim, config.prosody_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.prosody_dim, config.prosody_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.duration_proj = nn.Linear(config.prosody_dim, 1)

        # Final prosody prediction head
        self.prosody_head = nn.Sequential(
            nn.Linear(config.prosody_dim, config.prosody_dim),
            nn.GELU(),
            nn.LayerNorm(config.prosody_dim),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, seq, text_dim]
        text_mask: Optional[torch.Tensor] = None,
        target_prosody: Optional[torch.Tensor] = None,  # [batch, seq, prosody_dim]
        target_duration: Optional[torch.Tensor] = None,  # [batch, seq]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict prosody from text embeddings.

        Args:
            text_embeddings: Text encoder output
            text_mask: Optional attention mask
            target_prosody: Ground truth prosody for training
            target_duration: Ground truth duration for training

        Returns:
            Dict with predicted prosody and optional losses
        """
        # Project text to prosody dimension
        hidden = self.text_proj(text_embeddings)

        # Transform
        hidden = self.transformer(hidden, src_key_padding_mask=~text_mask if text_mask is not None else None)

        # Predict prosody
        prosody_pred = self.prosody_head(hidden)

        # Predict duration
        duration_input = hidden.transpose(1, 2)  # [batch, dim, seq]
        duration_conv_out = self.duration_conv(duration_input)  # [batch, dim, seq]
        duration_conv_out = duration_conv_out.transpose(1, 2)  # [batch, seq, dim]
        duration_pred = self.duration_proj(duration_conv_out).squeeze(-1)  # [batch, seq]
        duration_pred = F.softplus(duration_pred)  # Ensure positive

        output = {
            'prosody_pred': prosody_pred,
            'duration_pred': duration_pred,
        }

        # Compute losses if targets provided
        if target_prosody is not None:
            # Align lengths
            min_len = min(prosody_pred.shape[1], target_prosody.shape[1])
            prosody_loss = F.mse_loss(
                prosody_pred[:, :min_len],
                target_prosody[:, :min_len],
            )
            output['prosody_loss'] = prosody_loss

        if target_duration is not None:
            min_len = min(duration_pred.shape[1], target_duration.shape[1])
            duration_loss = F.mse_loss(
                duration_pred[:, :min_len],
                target_duration[:, :min_len],
            )
            output['duration_loss'] = duration_loss

        return output


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class PEWav2VecAdapter(nn.Module):
    """
    Adapter for integrating PE-wav2vec prosody with CSM pipeline.

    Converts PE-wav2vec prosody embeddings to prefix tokens compatible
    with ProsodyControlledCSM and the existing prosody conditioning system.
    """

    def __init__(
        self,
        config: PEWav2VecConfig,
        encoder: Optional[PEWav2VecEncoder] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided encoder or create new one
        self.encoder = encoder if encoder is not None else PEWav2VecEncoder(config)

        # Project prosody to output dimension
        self.proj = nn.Linear(config.prosody_dim, config.output_dim)

        # Generate prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

        # Optional: temporal attention for frame-level conditioning
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.output_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        # Learnable query tokens for attention
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_prefix_tokens, config.output_dim)
        )

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
        use_temporal: bool = True,
    ) -> torch.Tensor:
        """
        Extract prosody prefix tokens for CSM conditioning.

        Args:
            audio: Raw audio waveform at 16kHz
            use_temporal: Use temporal attention (True) or simple pooling (False)

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        batch_size = audio.shape[0]
        device = audio.device

        # Get prosody features from encoder
        with torch.no_grad():
            output = self.encoder(audio, compute_lpc_loss=False)

        prosody_features = output['prosody_features']  # [batch, seq, prosody_dim]
        prosody_embedding = output['prosody_embedding']  # [batch, prosody_dim]

        # Project to output dimension
        prosody_proj = self.proj(prosody_features)  # [batch, seq, output_dim]

        if use_temporal:
            # Use attention to generate prefix tokens
            queries = self.query_tokens.expand(batch_size, -1, -1)  # [batch, num_tokens, output_dim]

            tokens, _ = self.temporal_attention(
                query=queries,
                key=prosody_proj,
                value=prosody_proj,
            )
        else:
            # Simple projection from global embedding
            global_proj = self.proj(prosody_embedding)  # [batch, output_dim]
            tokens = self.token_proj(global_proj)  # [batch, output_dim * num_tokens]
            tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)

        # Normalize
        tokens = self.norm(tokens)

        return tokens

    def get_prosody_embedding(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get prosody embedding compatible with existing interfaces.

        Returns:
            [batch, output_dim] prosody embedding
        """
        prosody_emb = self.encoder.get_prosody_embedding(audio)
        return self.proj(prosody_emb)

    def get_frame_prosody(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get frame-level prosody features.

        Returns:
            [batch, seq, output_dim] frame-level prosody
        """
        prosody_features = self.encoder.get_prosody_features(audio)
        return self.proj(prosody_features)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class PEWav2VecLoss(nn.Module):
    """
    Combined loss for PE-wav2vec training.

    Components:
    1. LPC auxiliary loss: Encourage prosody learning in initial layers
    2. Contrastive loss: Preserve wav2vec representation quality
    3. Reconstruction loss: Optional decoder reconstruction
    """

    def __init__(self, config: PEWav2VecConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        encoder_output: Dict[str, torch.Tensor],
        target_prosody: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            encoder_output: Output from PEWav2VecEncoder
            target_prosody: Optional ground truth prosody (from teacher model)

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # LPC auxiliary loss (main prosody supervision)
        if 'lpc_loss' in encoder_output:
            losses['lpc_loss'] = encoder_output['lpc_loss']

        # Optional prosody regression loss (if using teacher prosody)
        if target_prosody is not None:
            pred_prosody = encoder_output['prosody_features']
            min_len = min(pred_prosody.shape[1], target_prosody.shape[1])
            prosody_loss = F.mse_loss(
                pred_prosody[:, :min_len],
                target_prosody[:, :min_len],
            )
            losses['prosody_regression'] = prosody_loss * 0.1

        # Total loss
        total = sum(losses.values())
        losses['total'] = total

        return losses


# =============================================================================
# COMPARISON: PE-WAV2VEC VS VANILLA WAV2VEC
# =============================================================================

class ProsodyComparison(nn.Module):
    """
    Compare prosody quality between PE-wav2vec and vanilla wav2vec.

    Measures:
    1. F0 correlation: How well do embeddings correlate with F0?
    2. Energy correlation: How well do embeddings correlate with energy?
    3. Prosody discriminability: Can we classify emotions from embeddings?
    """

    def __init__(self, config: PEWav2VecConfig):
        super().__init__()
        self.config = config

        # Simple classifiers for comparison
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.prosody_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 8),  # 8 basic emotions
        )

        self.f0_predictor = nn.Linear(config.prosody_dim, 1)
        self.energy_predictor = nn.Linear(config.prosody_dim, 1)

    def compute_f0_correlation(
        self,
        prosody_embedding: torch.Tensor,  # [batch, prosody_dim]
        f0_mean: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """Compute correlation between embedding and F0."""
        pred_f0 = self.f0_predictor(prosody_embedding).squeeze(-1)
        correlation = torch.corrcoef(torch.stack([pred_f0, f0_mean]))[0, 1]
        return correlation

    def compute_energy_correlation(
        self,
        prosody_embedding: torch.Tensor,  # [batch, prosody_dim]
        energy_mean: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """Compute correlation between embedding and energy."""
        pred_energy = self.energy_predictor(prosody_embedding).squeeze(-1)
        correlation = torch.corrcoef(torch.stack([pred_energy, energy_mean]))[0, 1]
        return correlation

    def classify_emotion(
        self,
        prosody_embedding: torch.Tensor,  # [batch, prosody_dim]
    ) -> torch.Tensor:
        """Classify emotion from prosody embedding."""
        logits = self.emotion_classifier(prosody_embedding)
        return F.softmax(logits, dim=-1)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PE-wav2vec: Prosody-Enhanced wav2vec 2.0 - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PEWav2VecConfig()

    # Test parameters
    batch_size = 2
    audio_samples = 16000 * 2  # 2 seconds at 16kHz

    print(f"\nDevice: {device}")
    print(f"Config: {config}")

    # Test 1: LPC Residual Extractor
    print("\n[Test 1] LPC Residual Extractor...")
    lpc_extractor = LPCResidualExtractor(config).to(device)

    audio = torch.randn(batch_size, audio_samples).to(device)
    lpc_output = lpc_extractor(audio)

    print(f"  Audio input: {audio.shape}")
    print(f"  Residual: {lpc_output['residual'].shape}")
    print(f"  LPC coeffs: {lpc_output['lpc_coeffs'].shape}")
    print(f"  Energy: {lpc_output['energy'].shape}")
    print(f"  Residual features: {lpc_output['residual_features'].shape}")
    print(f"  Num frames: {lpc_output['num_frames']}")
    print("  [PASS]")

    # Test 2: Auxiliary Head
    print("\n[Test 2] LPC Auxiliary Head...")
    head = LPCAuxiliaryHead(
        hidden_dim=config.hidden_dim,
        target_dim=config.lpc_residual_dim,
        dropout=config.dropout,
    ).to(device)

    hidden = torch.randn(batch_size, 100, config.hidden_dim).to(device)
    pred = head(hidden)
    print(f"  Hidden input: {hidden.shape}")
    print(f"  Prediction: {pred.shape}")
    print("  [PASS]")

    # Test 3: Prosody Attention Pooling
    print("\n[Test 3] Prosody Attention Pooling...")
    pooling = ProsodyAttentionPooling(
        hidden_dim=config.prosody_dim,
        num_heads=config.num_prosody_heads,
        output_dim=config.prosody_dim,
    ).to(device)

    prosody_features = torch.randn(batch_size, 100, config.prosody_dim).to(device)
    pooled = pooling(prosody_features)
    print(f"  Features input: {prosody_features.shape}")
    print(f"  Pooled output: {pooled.shape}")
    print("  [PASS]")

    # Test 4: Adapter (without loading wav2vec)
    print("\n[Test 4] PE-wav2vec Adapter (mock encoder)...")

    # Create mock encoder output
    class MockEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, audio, compute_lpc_loss=False):
            batch_size = audio.shape[0]
            seq_len = audio.shape[1] // config.sample_rate * 50  # ~50 frames/sec
            return {
                'prosody_embedding': torch.randn(batch_size, config.prosody_dim, device=audio.device),
                'prosody_features': torch.randn(batch_size, int(seq_len), config.prosody_dim, device=audio.device),
                'seq_len': int(seq_len),
            }

    mock_encoder = MockEncoder(config).to(device)
    adapter = PEWav2VecAdapter(config, encoder=mock_encoder).to(device)

    prefix_tokens = adapter(audio, use_temporal=True)
    print(f"  Audio input: {audio.shape}")
    print(f"  Prefix tokens: {prefix_tokens.shape}")
    assert prefix_tokens.shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 5: S4LPR Prosody Predictor
    print("\n[Test 5] S4LPR Prosody Predictor...")
    predictor = S4LPRProsodyPredictor(
        config,
        text_dim=512,
        num_layers=2,
    ).to(device)

    text_emb = torch.randn(batch_size, 50, 512).to(device)
    target_prosody = torch.randn(batch_size, 50, config.prosody_dim).to(device)
    target_duration = torch.rand(batch_size, 50).to(device) * 5 + 1

    pred_output = predictor(
        text_emb,
        target_prosody=target_prosody,
        target_duration=target_duration,
    )

    print(f"  Text input: {text_emb.shape}")
    print(f"  Prosody pred: {pred_output['prosody_pred'].shape}")
    print(f"  Duration pred: {pred_output['duration_pred'].shape}")
    print(f"  Prosody loss: {pred_output['prosody_loss'].item():.4f}")
    print(f"  Duration loss: {pred_output['duration_loss'].item():.4f}")
    print("  [PASS]")

    # Test 6: Loss computation
    print("\n[Test 6] Loss computation...")
    loss_fn = PEWav2VecLoss(config)

    mock_output = {
        'lpc_loss': torch.tensor(0.5),
        'prosody_features': torch.randn(batch_size, 100, config.prosody_dim).to(device),
    }
    target = torch.randn(batch_size, 100, config.prosody_dim).to(device)

    losses = loss_fn(mock_output, target_prosody=target)
    print(f"  LPC loss: {losses['lpc_loss'].item():.4f}")
    print(f"  Prosody regression: {losses['prosody_regression'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 7: Backward pass
    print("\n[Test 7] Backward pass...")
    adapter.zero_grad()
    tokens = adapter(audio, use_temporal=False)
    loss = tokens.mean()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All PE-wav2vec tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. LPC RESIDUAL SUPERVISION:
       - Extracts prosody-rich residual from speech
       - Supervises initial transformer blocks (layers 1-4)
       - Self-supervised prosody learning without text

    2. PROSODY LAYER EXTRACTION:
       - Uses early layer embeddings for prosody
       - Attention-based temporal pooling
       - Frame-level or global prosody features

    3. S4LPR INTEGRATION:
       - Text-to-prosody prediction module
       - Similar to FastSpeech 2 variance adaptor
       - Uses learned representations instead of raw F0/energy

    4. CSM ADAPTER:
       adapter = PEWav2VecAdapter(config)
       prefix_tokens = adapter(audio)  # [batch, 4, 2048]
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from pe_wav2vec import (
    PEWav2VecConfig,
    PEWav2VecEncoder,
    PEWav2VecAdapter,
    S4LPRProsodyPredictor,
)

# Initialize
config = PEWav2VecConfig()
encoder = PEWav2VecEncoder(config).cuda()

# Training: with LPC residual supervision
output = encoder(audio, compute_lpc_loss=True)
lpc_loss = output['lpc_loss']  # Auxiliary loss for prosody learning

# Inference: extract prosody embeddings
prosody_emb = encoder.get_prosody_embedding(audio)  # [batch, 256]
prosody_features = encoder.get_prosody_features(audio)  # [batch, seq, 256]

# CSM integration
adapter = PEWav2VecAdapter(config, encoder)
prefix_tokens = adapter(audio)  # [batch, 4, 2048]

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([other_prosody, prefix_tokens], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)

# S4LPR for TTS
predictor = S4LPRProsodyPredictor(config, text_dim=512)
pred = predictor(text_embeddings, target_prosody=prosody_features)
# pred['prosody_pred'] for generating prosody from text
""")
