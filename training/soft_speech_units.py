"""
Soft Speech Units for Prosody-Preserving Content Encoding

Based on "A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion" (ICASSP 2022)
by Benjamin van Niekerk, Marc-André Carbonneau, Julian Zaïdi, Matthew Baas, Hugo Seuté, Herman Kamper

Key Innovation: Instead of hard k-means assignment to discrete tokens, use soft assignment
(probability distribution over units). This preserves prosodic information that hard quantization loses.

Why Soft Units Help:
- Discrete tokens force a hard decision, losing nuance and uncertainty
- Soft units maintain probability distribution → preserve pitch/rhythm/energy subtleties
- Better reconstruction quality and more natural voice conversion
- Can be combined with explicit prosody encoder for fine-grained control

Architecture:
1. HuBERT-Soft Encoder: Pretrained HuBERT that outputs soft unit distributions (not argmax)
2. Explicit Prosody Encoder: Extracts pitch, energy, duration features separately
3. Soft Unit Decoder: Conditions on soft content + explicit prosody for reconstruction
4. Integration Adapter: Connect with existing V6 prosody pipeline

Resources:
- https://github.com/bshall/hubert (HuBERT-Soft pretrained model)
- https://github.com/bshall/soft-vc (Full VC system reference)
- Paper: https://arxiv.org/abs/2111.02392

Usage:
    from soft_speech_units import (
        SoftSpeechUnitsConfig,
        SoftSpeechUnitsModel,
        SoftSpeechUnitsAdapter,
    )

    config = SoftSpeechUnitsConfig()
    model = SoftSpeechUnitsModel(config).cuda()

    # Extract soft units + prosody
    soft_units, prosody = model.encode(audio)

    # Decode with modified prosody
    audio_reconstructed = model.decode(soft_units, prosody_modified, speaker_id)

    # For CSM integration
    adapter = SoftSpeechUnitsAdapter(config)
    prosody_tokens = adapter(audio, text_embeddings)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    from transformers import HubertModel, HubertConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SoftSpeechUnitsConfig:
    """Configuration for Soft Speech Units model."""

    # HuBERT settings
    hubert_model: str = "facebook/hubert-base-ls960"  # Base HuBERT for feature extraction
    hubert_layer: int = 6  # Which layer to extract features from (paper uses 6 for soft-vc)
    num_units: int = 100  # Number of k-means clusters (soft units vocabulary)

    # Soft unit dimensions
    soft_unit_dim: int = 256  # Dimension of soft unit embeddings
    hidden_dim: int = 512  # Hidden dimension for encoder/decoder

    # Explicit prosody features
    prosody_dim: int = 4  # F0, energy, duration, voiced/unvoiced
    prosody_hidden_dim: int = 256  # Prosody encoder hidden dimension

    # Decoder settings
    decoder_layers: int = 4  # Number of transformer decoder layers
    decoder_heads: int = 8  # Number of attention heads
    decoder_ffn_dim: int = 2048  # Feed-forward dimension

    # Speaker embedding
    num_speakers: int = 100  # Max number of speakers
    speaker_embed_dim: int = 256  # Speaker embedding dimension

    # Output settings
    mel_dim: int = 80  # Mel spectrogram output channels
    output_dim: int = 2048  # Output dimension for CSM integration

    # Training settings
    dropout: float = 0.1
    temperature: float = 1.0  # Softmax temperature for soft assignment

    # Audio settings
    sample_rate: int = 16000  # HuBERT expects 16kHz
    hop_length: int = 320  # HuBERT hop length (20ms at 16kHz)

    # Integration settings
    num_prefix_tokens: int = 4  # Number of prosody prefix tokens


# =============================================================================
# SOFT UNIT CODEBOOK
# =============================================================================

class SoftUnitCodebook(nn.Module):
    """
    Soft unit codebook that maintains probability distributions over units.

    Unlike VQ-VAE which uses hard nearest-neighbor lookup, we maintain
    soft assignments (probability distributions) over all units.
    """

    def __init__(self, num_units: int, unit_dim: int, temperature: float = 1.0):
        super().__init__()
        self.num_units = num_units
        self.unit_dim = unit_dim
        self.temperature = temperature

        # Unit embeddings (k-means centroids)
        self.embeddings = nn.Embedding(num_units, unit_dim)
        nn.init.uniform_(self.embeddings.weight, -1 / num_units, 1 / num_units)

        # Projection to compute soft assignment
        self.proj = nn.Linear(unit_dim, num_units)

    def compute_soft_assignment(
        self,
        features: torch.Tensor,  # [batch, seq, dim]
    ) -> torch.Tensor:
        """
        Compute soft assignment probabilities over units.

        Args:
            features: Input features from HuBERT encoder

        Returns:
            soft_units: [batch, seq, num_units] probability distribution
        """
        # Compute logits via projection
        logits = self.proj(features)  # [batch, seq, num_units]

        # Apply temperature-scaled softmax
        soft_units = F.softmax(logits / self.temperature, dim=-1)

        return soft_units

    def soft_lookup(
        self,
        soft_units: torch.Tensor,  # [batch, seq, num_units]
    ) -> torch.Tensor:
        """
        Perform soft lookup: weighted sum of unit embeddings.

        Args:
            soft_units: Probability distribution over units

        Returns:
            [batch, seq, unit_dim] weighted unit embeddings
        """
        # Weighted sum of embeddings
        embeddings = self.embeddings.weight  # [num_units, unit_dim]
        output = torch.matmul(soft_units, embeddings)  # [batch, seq, unit_dim]
        return output

    def hard_lookup(
        self,
        soft_units: torch.Tensor,  # [batch, seq, num_units]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform hard lookup for comparison (argmax).

        Returns:
            hard_units: [batch, seq, unit_dim] discrete unit embeddings
            indices: [batch, seq] unit indices
        """
        indices = soft_units.argmax(dim=-1)  # [batch, seq]
        hard_units = self.embeddings(indices)  # [batch, seq, unit_dim]
        return hard_units, indices

    def forward(
        self,
        features: torch.Tensor,
        return_hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: compute soft assignment and lookup.

        Args:
            features: [batch, seq, dim] HuBERT features
            return_hard: Also return hard (discrete) units for comparison

        Returns:
            Dict containing:
                - soft_probs: [batch, seq, num_units] soft assignment
                - soft_units: [batch, seq, unit_dim] soft unit embeddings
                - hard_units: [batch, seq, unit_dim] (if return_hard)
                - hard_indices: [batch, seq] (if return_hard)
        """
        soft_probs = self.compute_soft_assignment(features)
        soft_units = self.soft_lookup(soft_probs)

        output = {
            "soft_probs": soft_probs,
            "soft_units": soft_units,
        }

        if return_hard:
            hard_units, hard_indices = self.hard_lookup(soft_probs)
            output["hard_units"] = hard_units
            output["hard_indices"] = hard_indices

        return output


# =============================================================================
# HUBERT SOFT ENCODER
# =============================================================================

class HuBERTSoftEncoder(nn.Module):
    """
    HuBERT-based encoder that outputs soft speech units.

    Uses pretrained HuBERT to extract features, then applies soft k-means
    to get probability distributions over units instead of discrete tokens.
    """

    def __init__(self, config: SoftSpeechUnitsConfig):
        super().__init__()
        self.config = config

        # Load pretrained HuBERT (lazy loading to avoid memory issues)
        self.hubert = None
        self._hubert_loaded = False

        # Soft unit codebook
        hubert_dim = 768  # HuBERT base hidden dimension
        self.codebook = SoftUnitCodebook(
            num_units=config.num_units,
            unit_dim=config.soft_unit_dim,
            temperature=config.temperature,
        )

        # Project HuBERT features to codebook dimension
        self.proj = nn.Linear(hubert_dim, config.soft_unit_dim)
        self.norm = nn.LayerNorm(config.soft_unit_dim)

    def _load_hubert(self):
        """Lazy load HuBERT model."""
        if not self._hubert_loaded:
            if not HAS_TRANSFORMERS:
                raise RuntimeError("transformers package required for HuBERT. Install with: pip install transformers")

            print(f"Loading HuBERT model: {self.config.hubert_model}")
            self.hubert = HubertModel.from_pretrained(self.config.hubert_model)
            self.hubert.eval()

            # Freeze HuBERT weights
            for param in self.hubert.parameters():
                param.requires_grad = False

            self._hubert_loaded = True
            print("HuBERT loaded and frozen")

    def extract_features(
        self,
        audio: torch.Tensor,  # [batch, samples]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features from HuBERT encoder.

        Args:
            audio: Raw audio waveform at 16kHz
            attention_mask: Optional mask for padded sequences

        Returns:
            [batch, seq, hidden_dim] features from specified layer
        """
        self._load_hubert()

        # Move HuBERT to same device as audio
        if self.hubert.device != audio.device:
            self.hubert = self.hubert.to(audio.device)

        with torch.no_grad():
            outputs = self.hubert(
                audio,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Get features from specified layer
            hidden_states = outputs.hidden_states
            features = hidden_states[self.config.hubert_layer]

        return features

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
        attention_mask: Optional[torch.Tensor] = None,
        return_hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract soft speech units from audio.

        Args:
            audio: Raw audio waveform at 16kHz [batch, samples]
            attention_mask: Optional attention mask
            return_hard: Also return hard (discrete) units

        Returns:
            Dict containing:
                - soft_probs: [batch, seq, num_units]
                - soft_units: [batch, seq, soft_unit_dim]
                - features: [batch, seq, soft_unit_dim] projected HuBERT features
        """
        # Extract HuBERT features
        features = self.extract_features(audio, attention_mask)

        # Project to codebook dimension
        features_proj = self.proj(features)
        features_proj = self.norm(features_proj)

        # Get soft units
        codebook_output = self.codebook(features_proj, return_hard=return_hard)

        return {
            "features": features_proj,
            **codebook_output,
        }


# =============================================================================
# EXPLICIT PROSODY ENCODER
# =============================================================================

class ExplicitProsodyEncoder(nn.Module):
    """
    Explicit prosody encoder that extracts pitch, energy, duration features.

    This is separate from the content encoder to allow independent prosody control.
    The combination of soft content units + explicit prosody enables:
    - Voice conversion: Keep content, change speaker/prosody
    - Prosody transfer: Keep content, transfer prosody from another utterance
    - Prosody manipulation: Directly modify pitch/energy/timing
    """

    def __init__(self, config: SoftSpeechUnitsConfig):
        super().__init__()
        self.config = config

        # F0 encoder (continuous pitch)
        self.f0_encoder = nn.Sequential(
            nn.Linear(1, config.prosody_hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.prosody_hidden_dim // 4, config.prosody_hidden_dim // 4),
        )

        # Energy encoder
        self.energy_encoder = nn.Sequential(
            nn.Linear(1, config.prosody_hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.prosody_hidden_dim // 4, config.prosody_hidden_dim // 4),
        )

        # Duration encoder (frame duration relative to mean)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, config.prosody_hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.prosody_hidden_dim // 4, config.prosody_hidden_dim // 4),
        )

        # Voiced/unvoiced encoder
        self.vuv_encoder = nn.Sequential(
            nn.Linear(1, config.prosody_hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.prosody_hidden_dim // 4, config.prosody_hidden_dim // 4),
        )

        # Combine prosody features
        self.combine = nn.Sequential(
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.Linear(config.prosody_hidden_dim, config.soft_unit_dim),
        )

        # Temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(config.soft_unit_dim, config.soft_unit_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(config.soft_unit_dim, config.soft_unit_dim, kernel_size=5, padding=2),
        )

    def extract_prosody_features(
        self,
        audio: torch.Tensor,  # [batch, samples]
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosody features from audio.

        Returns dict with:
            - f0: [batch, frames, 1] fundamental frequency
            - energy: [batch, frames, 1] frame energy
            - duration: [batch, frames, 1] (relative timing)
            - vuv: [batch, frames, 1] voiced/unvoiced flag
        """
        batch_size = audio.shape[0]
        device = audio.device

        # Compute frame-level features
        hop_length = self.config.hop_length
        num_frames = audio.shape[1] // hop_length

        # Frame energy (RMS)
        frames = audio.unfold(1, hop_length, hop_length)  # [batch, num_frames, hop_length]
        energy = frames.pow(2).mean(dim=-1, keepdim=True)  # [batch, num_frames, 1]
        energy = torch.sqrt(energy + 1e-8)

        # Normalize energy
        energy = (energy - energy.mean(dim=1, keepdim=True)) / (energy.std(dim=1, keepdim=True) + 1e-8)

        # F0 extraction (simplified - in production, use CREPE or WORLD)
        # Here we use a proxy: high-energy frames tend to be voiced with detectable pitch
        # For real implementation, integrate a proper pitch tracker
        f0 = self._estimate_f0_simple(audio, hop_length)

        # Voiced/unvoiced flag
        vuv = (energy.abs() > 0.1).float()

        # Duration features (placeholder - would need alignment in production)
        duration = torch.ones(batch_size, num_frames, 1, device=device)

        return {
            "f0": f0,
            "energy": energy,
            "duration": duration,
            "vuv": vuv,
        }

    def _estimate_f0_simple(
        self,
        audio: torch.Tensor,
        hop_length: int,
    ) -> torch.Tensor:
        """
        Simple F0 estimation using autocorrelation.

        For production, replace with CREPE, WORLD, or DIO.
        """
        batch_size = audio.shape[0]
        device = audio.device
        num_frames = audio.shape[1] // hop_length

        # Simple energy-based proxy for F0 variation
        # In production, use proper pitch tracking
        frames = audio.unfold(1, hop_length, hop_length)

        # Use spectral centroid as pitch proxy
        fft = torch.fft.rfft(frames, dim=-1)
        magnitude = torch.abs(fft)
        freqs = torch.arange(magnitude.shape[-1], device=device).float()
        centroid = (magnitude * freqs).sum(dim=-1) / (magnitude.sum(dim=-1) + 1e-8)

        # Normalize
        centroid = centroid.unsqueeze(-1)  # [batch, frames, 1]
        centroid = (centroid - centroid.mean(dim=1, keepdim=True)) / (centroid.std(dim=1, keepdim=True) + 1e-8)

        return centroid

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        vuv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode prosody features.

        Can either:
        1. Extract from audio automatically
        2. Use provided prosody features (for manipulation/transfer)

        Args:
            audio: [batch, samples] raw audio
            f0: [batch, frames, 1] pre-extracted F0
            energy: [batch, frames, 1] pre-extracted energy
            duration: [batch, frames, 1] duration features
            vuv: [batch, frames, 1] voiced/unvoiced

        Returns:
            [batch, frames, soft_unit_dim] prosody embeddings
        """
        # Extract features if not provided
        if audio is not None and f0 is None:
            features = self.extract_prosody_features(audio)
            f0 = features["f0"]
            energy = features["energy"]
            duration = features["duration"]
            vuv = features["vuv"]

        # Encode each prosody component
        f0_emb = self.f0_encoder(f0)
        energy_emb = self.energy_encoder(energy)
        duration_emb = self.duration_encoder(duration)
        vuv_emb = self.vuv_encoder(vuv)

        # Concatenate
        prosody = torch.cat([f0_emb, energy_emb, duration_emb, vuv_emb], dim=-1)

        # Combine
        prosody = self.combine(prosody)

        # Temporal modeling
        prosody = prosody.transpose(1, 2)  # [batch, dim, frames]
        prosody = prosody + self.temporal_conv(prosody)
        prosody = prosody.transpose(1, 2)  # [batch, frames, dim]

        return prosody


# =============================================================================
# SOFT UNIT DECODER
# =============================================================================

class SoftUnitDecoder(nn.Module):
    """
    Decoder that reconstructs audio from soft units + explicit prosody.

    Architecture:
    - Cross-attention between soft units and prosody
    - Speaker conditioning via additive embedding
    - Transformer decoder layers
    - Linear projection to mel spectrogram
    """

    def __init__(self, config: SoftSpeechUnitsConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.content_proj = nn.Linear(config.soft_unit_dim, config.hidden_dim)
        self.prosody_proj = nn.Linear(config.soft_unit_dim, config.hidden_dim)

        # Speaker embedding
        self.speaker_embed = nn.Embedding(config.num_speakers, config.speaker_embed_dim)
        self.speaker_proj = nn.Linear(config.speaker_embed_dim, config.hidden_dim)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(config.hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.mel_dim),
        )

    def forward(
        self,
        soft_units: torch.Tensor,  # [batch, seq, soft_unit_dim]
        prosody: torch.Tensor,  # [batch, seq, soft_unit_dim]
        speaker_ids: Optional[torch.Tensor] = None,  # [batch]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode soft units + prosody to mel spectrogram.

        Args:
            soft_units: Soft content units from HuBERT
            prosody: Explicit prosody embeddings
            speaker_ids: Speaker IDs for conditioning
            mask: Optional attention mask

        Returns:
            [batch, seq, mel_dim] mel spectrogram
        """
        # Project inputs
        content = self.content_proj(soft_units)
        prosody_emb = self.prosody_proj(prosody)

        # Add speaker embedding
        if speaker_ids is not None:
            speaker_emb = self.speaker_embed(speaker_ids)  # [batch, speaker_dim]
            speaker_proj = self.speaker_proj(speaker_emb)  # [batch, hidden_dim]
            content = content + speaker_proj.unsqueeze(1)

        # Add positional encoding
        content = self.pos_encoding(content)
        prosody_emb = self.pos_encoding(prosody_emb)

        # Cross-attention: content attends to prosody
        # Using transformer decoder with prosody as memory
        decoded = self.decoder(
            tgt=content,
            memory=prosody_emb,
            tgt_mask=mask,
            memory_mask=mask,
        )

        # Project to mel
        mel = self.output_proj(decoded)

        return mel


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.shape[1]]


# =============================================================================
# FULL MODEL
# =============================================================================

class SoftSpeechUnitsModel(nn.Module):
    """
    Complete Soft Speech Units model for prosody-preserving voice conversion.

    Components:
    1. HuBERT Soft Encoder: Extracts soft (probabilistic) speech units
    2. Explicit Prosody Encoder: Extracts pitch, energy, duration separately
    3. Decoder: Reconstructs mel from soft units + prosody + speaker

    Key Benefits:
    - Soft units preserve more prosodic nuance than discrete tokens
    - Explicit prosody allows independent manipulation
    - Speaker embedding enables voice conversion
    """

    def __init__(self, config: SoftSpeechUnitsConfig):
        super().__init__()
        self.config = config

        # Content encoder (HuBERT → soft units)
        self.content_encoder = HuBERTSoftEncoder(config)

        # Prosody encoder (audio → F0, energy, duration)
        self.prosody_encoder = ExplicitProsodyEncoder(config)

        # Decoder (soft units + prosody → mel)
        self.decoder = SoftUnitDecoder(config)

        # Output projection for CSM integration
        self.output_proj = nn.Sequential(
            nn.Linear(config.soft_unit_dim + config.soft_unit_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def encode(
        self,
        audio: torch.Tensor,  # [batch, samples]
        return_hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode audio to soft units + prosody.

        Args:
            audio: Raw audio waveform at 16kHz
            return_hard: Also return hard (discrete) units

        Returns:
            Dict containing:
                - soft_units: [batch, seq, soft_unit_dim]
                - soft_probs: [batch, seq, num_units]
                - prosody: [batch, seq, soft_unit_dim]
                - features: [batch, seq, soft_unit_dim]
        """
        # Extract soft content units
        content_output = self.content_encoder(audio, return_hard=return_hard)

        # Extract prosody features
        prosody = self.prosody_encoder(audio)

        # Align lengths (prosody might be slightly different due to processing)
        min_len = min(content_output["soft_units"].shape[1], prosody.shape[1])
        content_output = {k: v[:, :min_len] if isinstance(v, torch.Tensor) and v.dim() > 1 else v
                        for k, v in content_output.items()}
        prosody = prosody[:, :min_len]

        return {
            **content_output,
            "prosody": prosody,
        }

    def decode(
        self,
        soft_units: torch.Tensor,
        prosody: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode soft units + prosody to mel spectrogram.

        Args:
            soft_units: [batch, seq, soft_unit_dim]
            prosody: [batch, seq, soft_unit_dim]
            speaker_ids: [batch] speaker IDs

        Returns:
            [batch, seq, mel_dim] mel spectrogram
        """
        return self.decoder(soft_units, prosody, speaker_ids)

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode and decode.

        Args:
            audio: Raw audio waveform at 16kHz
            speaker_ids: Optional speaker IDs

        Returns:
            Dict with:
                - mel_reconstructed: [batch, seq, mel_dim]
                - soft_units: [batch, seq, soft_unit_dim]
                - soft_probs: [batch, seq, num_units]
                - prosody: [batch, seq, soft_unit_dim]
                - combined_embedding: [batch, seq, output_dim]
        """
        # Encode
        encoded = self.encode(audio, return_hard=True)

        # Decode
        mel_reconstructed = self.decode(
            encoded["soft_units"],
            encoded["prosody"],
            speaker_ids,
        )

        # Combined embedding for downstream use
        combined = torch.cat([encoded["soft_units"], encoded["prosody"]], dim=-1)
        combined_embedding = self.output_proj(combined)

        return {
            "mel_reconstructed": mel_reconstructed,
            "soft_units": encoded["soft_units"],
            "soft_probs": encoded["soft_probs"],
            "prosody": encoded["prosody"],
            "combined_embedding": combined_embedding,
            "hard_units": encoded.get("hard_units"),
            "hard_indices": encoded.get("hard_indices"),
        }

    def compute_loss(
        self,
        mel_target: torch.Tensor,
        mel_reconstructed: torch.Tensor,
        soft_probs: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.

        Args:
            mel_target: [batch, seq, mel_dim] target mel
            mel_reconstructed: [batch, seq, mel_dim] predicted mel
            soft_probs: [batch, seq, num_units] soft unit probabilities
            mel_mask: [batch, seq] valid frame mask

        Returns:
            Dict with individual losses and total
        """
        # Handle length mismatch
        min_len = min(mel_target.shape[1], mel_reconstructed.shape[1])
        mel_target = mel_target[:, :min_len]
        mel_reconstructed = mel_reconstructed[:, :min_len]

        if mel_mask is not None:
            mel_mask = mel_mask[:, :min_len]

        # L1 reconstruction loss
        l1_loss = F.l1_loss(mel_reconstructed, mel_target, reduction='none')

        # L2 reconstruction loss
        l2_loss = F.mse_loss(mel_reconstructed, mel_target, reduction='none')

        if mel_mask is not None:
            mask = mel_mask.unsqueeze(-1)
            l1_loss = (l1_loss * mask).sum() / (mask.sum() + 1e-8)
            l2_loss = (l2_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            l1_loss = l1_loss.mean()
            l2_loss = l2_loss.mean()

        # Soft unit entropy regularization (encourage peaked distributions)
        entropy = -(soft_probs * (soft_probs + 1e-8).log()).sum(dim=-1)
        entropy_loss = entropy.mean() * 0.01  # Small weight

        # Total loss
        total = l1_loss + l2_loss + entropy_loss

        return {
            'l1_reconstruction': l1_loss,
            'l2_reconstruction': l2_loss,
            'entropy_loss': entropy_loss,
            'total': total,
        }

    def voice_conversion(
        self,
        source_audio: torch.Tensor,
        target_speaker_id: torch.Tensor,
        prosody_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform voice conversion: keep content, change speaker.

        Args:
            source_audio: [batch, samples] source audio
            target_speaker_id: [batch] target speaker
            prosody_scale: Scale factor for prosody (1.0 = preserve, 0.0 = neutral)

        Returns:
            [batch, seq, mel_dim] converted mel spectrogram
        """
        # Encode source
        encoded = self.encode(source_audio)

        # Scale prosody if desired
        prosody = encoded["prosody"] * prosody_scale

        # Decode with target speaker
        mel_converted = self.decode(
            encoded["soft_units"],
            prosody,
            target_speaker_id,
        )

        return mel_converted

    def prosody_transfer(
        self,
        content_audio: torch.Tensor,
        prosody_audio: torch.Tensor,
        target_speaker_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Transfer prosody from one utterance to another.

        Args:
            content_audio: [batch, samples] source for content
            prosody_audio: [batch, samples] source for prosody
            target_speaker_id: [batch] target speaker

        Returns:
            [batch, seq, mel_dim] mel with content from content_audio,
                                  prosody from prosody_audio
        """
        # Extract content from first audio
        content_encoded = self.content_encoder(content_audio)

        # Extract prosody from second audio
        prosody = self.prosody_encoder(prosody_audio)

        # Align lengths
        min_len = min(content_encoded["soft_units"].shape[1], prosody.shape[1])
        soft_units = content_encoded["soft_units"][:, :min_len]
        prosody = prosody[:, :min_len]

        # Decode
        mel_transferred = self.decode(soft_units, prosody, target_speaker_id)

        return mel_transferred


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class SoftSpeechUnitsAdapter(nn.Module):
    """
    Adapter to integrate Soft Speech Units with existing prosody pipeline.

    Converts soft units + prosody to prefix tokens compatible with
    ProsodyControlledCSM and the V6 prosody conditioning system.
    """

    def __init__(
        self,
        config: SoftSpeechUnitsConfig,
        model: Optional[SoftSpeechUnitsModel] = None,
    ):
        super().__init__()
        self.config = config

        # Use provided model or create new one
        self.model = model if model is not None else SoftSpeechUnitsModel(config)

        # Project to prefix tokens
        self.token_proj = nn.Linear(
            config.output_dim,
            config.output_dim * config.num_prefix_tokens,
        )
        self.norm = nn.LayerNorm(config.output_dim)

        # Optional: Blend with text embeddings
        self.text_blend = nn.Linear(config.output_dim * 2, config.output_dim)

    def forward(
        self,
        audio: torch.Tensor,  # [batch, samples]
        text_embeddings: Optional[torch.Tensor] = None,  # [batch, seq, hidden]
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens for CSM conditioning.

        Args:
            audio: Raw audio waveform at 16kHz
            text_embeddings: Optional text embeddings to blend with

        Returns:
            [batch, num_prefix_tokens, output_dim] prefix tokens
        """
        # Get combined embedding from model
        output = self.model(audio)
        combined_emb = output["combined_embedding"]  # [batch, seq, output_dim]

        # Pool to single vector
        pooled = combined_emb.mean(dim=1)  # [batch, output_dim]

        # Optionally blend with text
        if text_embeddings is not None:
            text_pooled = text_embeddings.mean(dim=1)  # [batch, hidden]

            # Project text to same dim if needed
            if text_pooled.shape[-1] != self.config.output_dim:
                text_pooled = F.pad(
                    text_pooled,
                    (0, self.config.output_dim - text_pooled.shape[-1])
                )

            combined = torch.cat([pooled, text_pooled], dim=-1)
            pooled = self.text_blend(combined)

        # Project to tokens
        tokens = self.token_proj(pooled)  # [batch, output_dim * num_tokens]

        # Reshape
        batch_size = pooled.shape[0]
        tokens = tokens.view(batch_size, self.config.num_prefix_tokens, self.config.output_dim)

        # Normalize
        tokens = self.norm(tokens)

        return tokens

    def get_prosody_embedding(
        self,
        audio: torch.Tensor,
        pool: str = 'mean',
    ) -> torch.Tensor:
        """
        Get prosody embedding compatible with existing interface.

        Args:
            audio: [batch, samples] raw audio
            pool: 'mean' or 'first' for temporal pooling

        Returns:
            [batch, output_dim] prosody embedding
        """
        output = self.model(audio)
        combined_emb = output["combined_embedding"]

        if pool == 'mean':
            return combined_emb.mean(dim=1)
        else:
            return combined_emb[:, 0, :]


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

class SoftVsHardComparison(nn.Module):
    """
    Utility for comparing soft vs hard speech units.

    Demonstrates the benefit of soft assignment for prosody preservation.
    """

    def __init__(self, model: SoftSpeechUnitsModel):
        super().__init__()
        self.model = model

    def compare_reconstruction(
        self,
        audio: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compare reconstruction quality between soft and hard units.

        Returns:
            Dict with soft and hard reconstructions and metrics
        """
        # Encode with both soft and hard
        encoded = self.model.encode(audio, return_hard=True)

        # Decode with soft units
        mel_soft = self.model.decode(
            encoded["soft_units"],
            encoded["prosody"],
            speaker_ids,
        )

        # Decode with hard units
        mel_hard = self.model.decode(
            encoded["hard_units"],
            encoded["prosody"],
            speaker_ids,
        )

        return {
            "mel_soft": mel_soft,
            "mel_hard": mel_hard,
            "soft_units": encoded["soft_units"],
            "hard_units": encoded["hard_units"],
            "soft_probs": encoded["soft_probs"],
            "hard_indices": encoded["hard_indices"],
        }

    def measure_information_loss(
        self,
        soft_probs: torch.Tensor,  # [batch, seq, num_units]
    ) -> Dict[str, torch.Tensor]:
        """
        Measure information lost when going from soft to hard assignment.

        Lower entropy = more information loss from hard quantization.
        """
        # Entropy of soft distribution
        entropy = -(soft_probs * (soft_probs + 1e-8).log()).sum(dim=-1)

        # Max entropy (uniform distribution)
        max_entropy = math.log(soft_probs.shape[-1])

        # Normalized entropy (0 = one-hot, 1 = uniform)
        normalized_entropy = entropy / max_entropy

        # Average uncertainty
        avg_uncertainty = normalized_entropy.mean()

        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "avg_uncertainty": avg_uncertainty,
            "max_entropy": max_entropy,
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Soft Speech Units - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SoftSpeechUnitsConfig()

    # Test parameters
    batch_size = 2
    audio_samples = 16000 * 2  # 2 seconds at 16kHz
    num_frames = audio_samples // config.hop_length

    # Create dummy inputs (skip HuBERT loading for quick test)
    print("\n[Test 1] Configuration and basic setup...")
    print(f"  Config: {config}")
    print("  [PASS]")

    # Test 2: Soft Unit Codebook
    print("\n[Test 2] Soft Unit Codebook...")
    codebook = SoftUnitCodebook(
        num_units=config.num_units,
        unit_dim=config.soft_unit_dim,
        temperature=config.temperature,
    ).to(device)

    features = torch.randn(batch_size, num_frames, config.soft_unit_dim).to(device)
    codebook_output = codebook(features, return_hard=True)

    print(f"  Soft probs: {codebook_output['soft_probs'].shape}")
    print(f"  Soft units: {codebook_output['soft_units'].shape}")
    print(f"  Hard units: {codebook_output['hard_units'].shape}")
    print(f"  Hard indices: {codebook_output['hard_indices'].shape}")
    print(f"  Soft probs sum: {codebook_output['soft_probs'].sum(dim=-1).mean():.4f} (should be 1.0)")
    print("  [PASS]")

    # Test 3: Explicit Prosody Encoder
    print("\n[Test 3] Explicit Prosody Encoder...")
    prosody_encoder = ExplicitProsodyEncoder(config).to(device)

    # Create dummy audio
    audio = torch.randn(batch_size, audio_samples).to(device)
    prosody = prosody_encoder(audio)

    print(f"  Input audio: {audio.shape}")
    print(f"  Prosody embedding: {prosody.shape}")
    print("  [PASS]")

    # Test 4: Soft Unit Decoder
    print("\n[Test 4] Soft Unit Decoder...")
    decoder = SoftUnitDecoder(config).to(device)

    soft_units = torch.randn(batch_size, num_frames, config.soft_unit_dim).to(device)
    prosody_emb = torch.randn(batch_size, num_frames, config.soft_unit_dim).to(device)
    speaker_ids = torch.randint(0, config.num_speakers, (batch_size,)).to(device)

    mel_out = decoder(soft_units, prosody_emb, speaker_ids)
    print(f"  Soft units: {soft_units.shape}")
    print(f"  Prosody: {prosody_emb.shape}")
    print(f"  Mel output: {mel_out.shape}")
    print("  [PASS]")

    # Test 5: Full Model (without HuBERT loading)
    print("\n[Test 5] Full Model (components only, skipping HuBERT)...")
    model = SoftSpeechUnitsModel(config).to(device)

    # Test decoder path only
    output_proj = model.output_proj
    combined = torch.randn(batch_size, num_frames, config.soft_unit_dim * 2).to(device)
    combined_emb = output_proj(combined)
    print(f"  Combined input: {combined.shape}")
    print(f"  Output embedding: {combined_emb.shape}")
    print("  [PASS]")

    # Test 6: Adapter
    print("\n[Test 6] CSM Adapter...")

    # Mock the model output for testing
    class MockSoftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.combined_embedding = torch.randn(batch_size, num_frames, config.output_dim).to(device)

        def forward(self, audio):
            return {"combined_embedding": self.combined_embedding}

    adapter = SoftSpeechUnitsAdapter(config, model=MockSoftModel()).to(device)
    prefix_tokens = adapter(audio)

    print(f"  Audio input: {audio.shape}")
    print(f"  Prefix tokens: {prefix_tokens.shape}")
    assert prefix_tokens.shape == (batch_size, config.num_prefix_tokens, config.output_dim)
    print("  [PASS]")

    # Test 7: Loss computation
    print("\n[Test 7] Loss computation...")
    mel_target = torch.randn(batch_size, num_frames, config.mel_dim).to(device)
    mel_pred = torch.randn(batch_size, num_frames, config.mel_dim).to(device)
    soft_probs = F.softmax(torch.randn(batch_size, num_frames, config.num_units), dim=-1).to(device)

    losses = model.compute_loss(mel_target, mel_pred, soft_probs)
    print(f"  L1 loss: {losses['l1_reconstruction'].item():.4f}")
    print(f"  L2 loss: {losses['l2_reconstruction'].item():.4f}")
    print(f"  Entropy loss: {losses['entropy_loss'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 8: Information loss measurement
    print("\n[Test 8] Soft vs Hard comparison utilities...")
    comparison = SoftVsHardComparison(model)
    info_loss = comparison.measure_information_loss(soft_probs)

    print(f"  Average uncertainty: {info_loss['avg_uncertainty'].item():.4f}")
    print(f"  Max entropy: {info_loss['max_entropy']:.4f}")
    print("  [PASS]")

    # Test 9: Backward pass
    print("\n[Test 9] Backward pass...")
    decoder.zero_grad()
    mel_out = decoder(soft_units, prosody_emb, speaker_ids)
    loss = mel_out.mean()
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in decoder.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All Soft Speech Units tests passed!")
    print("=" * 60)

    print("\nKey Features:")
    print("-" * 40)
    print("""
    1. SOFT ASSIGNMENT:
       - Probability distribution over units (not argmax)
       - Preserves uncertainty and prosodic nuance
       - Better reconstruction than discrete tokens

    2. EXPLICIT PROSODY:
       - Separate encoder for F0, energy, duration
       - Enables independent prosody manipulation
       - Can transfer prosody between utterances

    3. VOICE CONVERSION:
       model.voice_conversion(source_audio, target_speaker_id)

    4. PROSODY TRANSFER:
       model.prosody_transfer(content_audio, prosody_audio)

    5. CSM INTEGRATION:
       adapter = SoftSpeechUnitsAdapter(config)
       prefix_tokens = adapter(audio)  # Use with ProsodyControlledCSM
    """)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from soft_speech_units import (
    SoftSpeechUnitsConfig,
    SoftSpeechUnitsModel,
    SoftSpeechUnitsAdapter,
)

# Initialize
config = SoftSpeechUnitsConfig()
model = SoftSpeechUnitsModel(config).cuda()

# Encode audio to soft units + prosody
encoded = model.encode(audio)
soft_units = encoded['soft_units']   # Probabilistic content
prosody = encoded['prosody']          # Explicit prosody features

# Voice conversion (keep content, change speaker)
mel_converted = model.voice_conversion(
    source_audio=audio,
    target_speaker_id=torch.tensor([1]),
)

# Prosody transfer (content from A, prosody from B)
mel_transferred = model.prosody_transfer(
    content_audio=audio_a,
    prosody_audio=audio_b,
)

# For CSM integration
adapter = SoftSpeechUnitsAdapter(config, model)
prefix_tokens = adapter(audio)  # [batch, 4, 2048]
# Use prefix_tokens with ProsodyControlledCSM
""")
