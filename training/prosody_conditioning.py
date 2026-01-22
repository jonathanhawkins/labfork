"""
Prosody Conditioning Module for CSM Voice Cloning

This module enables controllable speech synthesis by conditioning the CSM model
on extracted prosody features. The key innovation is bridging the gap between
prosody analysis (what we extract) and generation (what the model produces).

Architecture:
    Audio → ProsodyAnalyzer → 4 Prosody Vectors → ProsodyEncoder → Prefix Embeddings
                                                                          ↓
    Text → TextEmbedding → [Prosody Prefix | Text Embeddings] → CSM Backbone → Audio

This enables:
1. Training: Learn prosody → embedding mapping from paired data
2. Inference: Control generation by specifying prosody ("say this angrily")
3. Style Transfer: Extract prosody from reference audio, apply to new text

Reference: Your PRD.md "Cube" concept - the 4 prosody faces become conditioning vectors.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProsodyConfig:
    """Configuration for prosody conditioning."""
    # Prosody vector dimensions (from your prosody_analyzer.py)
    semantic_dim: int = 8       # Emotion scores (happy, sad, angry, etc.)
    acoustic_dim: int = 12      # Pitch stats, formants, HNR, jitter, shimmer
    rhythm_dim: int = 8         # Speaking rate, pause stats, syllable rate
    contour_dim: int = 64       # Pitch trajectory (downsampled time series)

    # Encoder settings
    hidden_size: int = 2048     # CSM backbone hidden size
    num_prosody_tokens: int = 4 # Number of prefix tokens to generate
    dropout: float = 0.1

    # Cross-attention settings (optional advanced mode)
    use_cross_attention: bool = False
    num_cross_attn_heads: int = 8


class ProsodyEncoder(nn.Module):
    """
    Encodes 4-layer prosody analysis into prefix embeddings for CSM.

    The "Cube" becomes reality:
    - Face 1 (Semantic): What emotion is being expressed
    - Face 2 (Acoustic): Physical voice properties (pitch, formants)
    - Face 3 (Rhythm): Timing patterns (pauses, rate)
    - Face 4 (Contour): Pitch trajectory over time

    All 4 faces are fused into a single prosody representation that conditions generation.
    """

    def __init__(self, config: ProsodyConfig):
        super().__init__()
        self.config = config

        # Total prosody input dimension
        total_prosody_dim = (
            config.semantic_dim +
            config.acoustic_dim +
            config.rhythm_dim +
            config.contour_dim
        )

        # Individual encoders for each prosody type
        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.acoustic_encoder = nn.Sequential(
            nn.Linear(config.acoustic_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.rhythm_encoder = nn.Sequential(
            nn.Linear(config.rhythm_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Contour encoder with temporal modeling (pitch trajectory is time-series)
        self.contour_encoder = nn.Sequential(
            nn.Linear(config.contour_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size * config.num_prosody_tokens),
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        semantic: torch.Tensor,      # [batch, semantic_dim]
        acoustic: torch.Tensor,      # [batch, acoustic_dim]
        rhythm: torch.Tensor,        # [batch, rhythm_dim]
        contour: torch.Tensor,       # [batch, contour_dim]
    ) -> torch.Tensor:
        """
        Encode prosody vectors into prefix embeddings.

        Returns:
            Tensor of shape [batch, num_prosody_tokens, hidden_size]
        """
        # Encode each prosody type
        sem_enc = self.semantic_encoder(semantic)     # [B, H/4]
        aco_enc = self.acoustic_encoder(acoustic)     # [B, H/4]
        rhy_enc = self.rhythm_encoder(rhythm)         # [B, H/4]
        con_enc = self.contour_encoder(contour)       # [B, H/4]

        # Concatenate all prosody encodings
        fused = torch.cat([sem_enc, aco_enc, rhy_enc, con_enc], dim=-1)  # [B, H]

        # Generate prefix tokens
        prefix = self.fusion(fused)  # [B, H * num_tokens]

        # Reshape to sequence of tokens
        batch_size = fused.shape[0]
        prefix = prefix.view(batch_size, self.config.num_prosody_tokens, self.config.hidden_size)

        # Normalize
        prefix = self.norm(prefix)

        return prefix


class TemporalProsodyEncoder(nn.Module):
    """
    Encodes temporal/per-segment prosody for keyframe-based control.

    Unlike ProsodyEncoder which averages prosody into global tokens,
    this encoder preserves temporal variation by encoding each time
    segment separately. This enables per-keyframe emotion control.

    Input shape: [batch, num_segments, dim] for each prosody type
    Output shape: [batch, num_segments, hidden_size]

    The output can be used as:
    1. Per-segment prefix tokens (one prosody token per time segment)
    2. Cross-attention targets for temporal conditioning
    """

    def __init__(self, config: ProsodyConfig):
        super().__init__()
        self.config = config

        # Per-segment encoders (shared across segments)
        self.semantic_encoder = nn.Sequential(
            nn.Linear(config.semantic_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.acoustic_encoder = nn.Sequential(
            nn.Linear(config.acoustic_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.rhythm_encoder = nn.Sequential(
            nn.Linear(config.rhythm_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.contour_encoder = nn.Sequential(
            nn.Linear(config.contour_dim, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Fusion to single hidden vector per segment
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Positional encoding for temporal segments
        self.segment_position_embedding = nn.Embedding(
            config.num_prosody_tokens * 4,  # Support up to 4x the default segments
            config.hidden_size
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size)

    def init_from_global_encoder(self, global_encoder: "ProsodyEncoder") -> None:
        """
        Initialize temporal encoder weights from a trained global ProsodyEncoder.

        This provides a sensible starting point for temporal conditioning without
        requiring a separate temporal training run.
        """
        self.semantic_encoder.load_state_dict(global_encoder.semantic_encoder.state_dict())
        self.acoustic_encoder.load_state_dict(global_encoder.acoustic_encoder.state_dict())
        self.rhythm_encoder.load_state_dict(global_encoder.rhythm_encoder.state_dict())
        self.contour_encoder.load_state_dict(global_encoder.contour_encoder.state_dict())

        # Copy the first fusion layer (shared shape). The global encoder's final
        # projection differs, so we intentionally skip it.
        self.fusion[0].load_state_dict(global_encoder.fusion[0].state_dict())

        # Match layer norm parameters
        self.norm.load_state_dict(global_encoder.norm.state_dict())

    def forward(
        self,
        semantic: torch.Tensor,      # [batch, num_segments, semantic_dim]
        acoustic: torch.Tensor,      # [batch, num_segments, acoustic_dim]
        rhythm: torch.Tensor,        # [batch, num_segments, rhythm_dim]
        contour: torch.Tensor,       # [batch, num_segments, contour_dim]
    ) -> torch.Tensor:
        """
        Encode temporal prosody into per-segment embeddings.

        Args:
            semantic: [batch, num_segments, semantic_dim]
            acoustic: [batch, num_segments, acoustic_dim]
            rhythm: [batch, num_segments, rhythm_dim]
            contour: [batch, num_segments, contour_dim]

        Returns:
            Tensor of shape [batch, num_segments, hidden_size]
            Each segment has its own prosody embedding, preserving temporal variation.
        """
        batch_size, num_segments = semantic.shape[:2]

        # Reshape to [batch * num_segments, dim] for encoding
        sem_flat = semantic.reshape(-1, self.config.semantic_dim)
        aco_flat = acoustic.reshape(-1, self.config.acoustic_dim)
        rhy_flat = rhythm.reshape(-1, self.config.rhythm_dim)
        con_flat = contour.reshape(-1, self.config.contour_dim)

        # Encode each prosody type
        sem_enc = self.semantic_encoder(sem_flat)   # [B*S, H/4]
        aco_enc = self.acoustic_encoder(aco_flat)   # [B*S, H/4]
        rhy_enc = self.rhythm_encoder(rhy_flat)     # [B*S, H/4]
        con_enc = self.contour_encoder(con_flat)    # [B*S, H/4]

        # Concatenate and fuse
        fused = torch.cat([sem_enc, aco_enc, rhy_enc, con_enc], dim=-1)  # [B*S, H]
        fused = self.fusion(fused)  # [B*S, H]

        # Reshape back to [batch, num_segments, hidden]
        output = fused.view(batch_size, num_segments, self.config.hidden_size)

        # Add positional encoding for segment order
        positions = torch.arange(num_segments, device=output.device)
        pos_embed = self.segment_position_embedding(positions)  # [S, H]
        output = output + pos_embed.unsqueeze(0)  # [B, S, H]

        # Normalize
        output = self.norm(output)

        return output


class ProsodyControlledCSM(nn.Module):
    """
    Wrapper that adds prosody conditioning to CSM.

    Supports two conditioning modes:
    1. Global: Single prosody vector averaged over time (original behavior)
    2. Temporal: Per-segment prosody tokens for keyframe control (new)

    Usage:
        model = ProsodyControlledCSM(csm_model, prosody_config)

        # Global mode (original): extract prosody from audio, learn to reconstruct
        prosody = extract_prosody(audio)
        loss = model.forward_with_prosody(text, audio, prosody)

        # Inference with global prosody
        prosody = {"emotion": "happy", "energy": 0.8}  # Or extract from reference
        audio = model.generate_with_prosody(text, prosody)

        # Temporal mode (new): per-segment keyframe control
        # Prosody dict should have shape [batch, num_segments, dim]
        temporal_prosody = get_temporal_prosody_tokens(keyframe_prosody)
        audio = model.generate_with_temporal_prosody(text, temporal_prosody)
    """

    def __init__(
        self,
        csm_model: nn.Module,
        prosody_config: ProsodyConfig,
        freeze_csm: bool = True,
        enable_temporal: bool = True,
    ):
        super().__init__()
        self.csm = csm_model
        self.prosody_encoder = ProsodyEncoder(prosody_config)
        self.config = prosody_config

        # Temporal prosody encoder for keyframe support
        self.enable_temporal = enable_temporal
        if enable_temporal:
            self.temporal_encoder = TemporalProsodyEncoder(prosody_config)

        # Freeze CSM if specified (train only prosody encoder)
        if freeze_csm:
            for param in self.csm.parameters():
                param.requires_grad = False

    def get_prosody_prefix(
        self,
        semantic: torch.Tensor,
        acoustic: torch.Tensor,
        rhythm: torch.Tensor,
        contour: torch.Tensor,
    ) -> torch.Tensor:
        """Get prosody prefix embeddings."""
        return self.prosody_encoder(semantic, acoustic, rhythm, contour)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prosody_dict: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with prosody conditioning.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            prosody_dict: Dict with 'semantic', 'acoustic', 'rhythm', 'contour' tensors
            labels: Optional labels for training

        Returns:
            Model outputs with loss if labels provided
        """
        # Get prosody prefix
        prosody_prefix = self.get_prosody_prefix(
            prosody_dict['semantic'],
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
        )

        # Get text embeddings from CSM
        text_embeds = self.csm.embed_text_tokens(input_ids)

        # Concatenate prosody prefix with text embeddings
        # [prosody_tokens | text_tokens]
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend attention mask for prosody tokens
        batch_size = input_ids.shape[0]
        prosody_mask = torch.ones(
            batch_size, self.config.num_prosody_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prosody_mask, attention_mask], dim=1)

        # Forward through CSM with combined embeddings
        outputs = self.csm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def generate_with_prosody(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prosody_dict: Dict[str, torch.Tensor],
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate audio with prosody control.

        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            prosody_dict: Prosody conditioning vectors
            **generate_kwargs: Generation parameters (temperature, etc.)

        Returns:
            Generated audio tensor
        """
        # Get combined embeddings
        prosody_prefix = self.get_prosody_prefix(
            prosody_dict['semantic'],
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
        )

        text_embeds = self.csm.embed_text_tokens(input_ids)
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend attention mask
        batch_size = input_ids.shape[0]
        prosody_mask = torch.ones(
            batch_size, self.config.num_prosody_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prosody_mask, attention_mask], dim=1)

        # Generate
        return self.csm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            output_audio=True,
            **generate_kwargs
        )

    def get_temporal_prosody_prefix(
        self,
        semantic: torch.Tensor,      # [batch, num_segments, semantic_dim]
        acoustic: torch.Tensor,      # [batch, num_segments, acoustic_dim]
        rhythm: torch.Tensor,        # [batch, num_segments, rhythm_dim]
        contour: torch.Tensor,       # [batch, num_segments, contour_dim]
    ) -> torch.Tensor:
        """
        Get per-segment prosody embeddings for temporal conditioning.

        Unlike get_prosody_prefix which produces global prefix tokens,
        this method produces one embedding per time segment, preserving
        the temporal variation from keyframe edits.

        Args:
            semantic: [batch, num_segments, semantic_dim]
            acoustic: [batch, num_segments, acoustic_dim]
            rhythm: [batch, num_segments, rhythm_dim]
            contour: [batch, num_segments, contour_dim]

        Returns:
            Tensor of shape [batch, num_segments, hidden_size]
        """
        if not self.enable_temporal:
            raise RuntimeError(
                "Temporal encoding not enabled. Initialize with enable_temporal=True"
            )
        return self.temporal_encoder(semantic, acoustic, rhythm, contour)

    def forward_temporal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prosody_dict: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with temporal prosody conditioning.

        This preserves per-keyframe edits by using one prosody token
        per time segment instead of averaging into global tokens.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            prosody_dict: Dict with temporal tensors of shape [batch, num_segments, dim]
                         Keys: 'semantic', 'acoustic', 'rhythm', 'contour'
            labels: Optional labels for training

        Returns:
            Model outputs with loss if labels provided
        """
        # Get temporal prosody embeddings
        prosody_prefix = self.get_temporal_prosody_prefix(
            prosody_dict['semantic'],
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
        )

        num_segments = prosody_prefix.shape[1]

        # Get text embeddings from CSM
        text_embeds = self.csm.embed_text_tokens(input_ids)

        # Concatenate temporal prosody prefix with text embeddings
        # [segment_0_prosody | segment_1_prosody | ... | text_tokens]
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend attention mask for prosody tokens
        batch_size = input_ids.shape[0]
        prosody_mask = torch.ones(
            batch_size, num_segments,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prosody_mask, attention_mask], dim=1)

        # Forward through CSM with combined embeddings
        outputs = self.csm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def generate_with_temporal_prosody(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prosody_dict: Dict[str, torch.Tensor],
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate audio with temporal/keyframe prosody control.

        This method preserves per-keyframe emotion edits instead of
        averaging them into a global style. Each time segment gets
        its own prosody conditioning token.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            prosody_dict: Dict with temporal tensors of shape [batch, num_segments, dim]
                         Keys: 'semantic', 'acoustic', 'rhythm', 'contour'
                         Use get_temporal_prosody_tokens() from keyframe_prosody.py
                         to convert keyframe output to this format.
            **generate_kwargs: Generation parameters (temperature, etc.)

        Returns:
            Generated audio tensor with prosody following the keyframe trajectory

        Example:
            >>> from keyframe_prosody import (
            ...     ProsodyKeyframe, keyframes_to_prosody, get_temporal_prosody_tokens
            ... )
            >>>
            >>> # Define emotion trajectory
            >>> keyframes = [
            ...     ProsodyKeyframe(time=0.0, emotion="neutral", intensity=0.5),
            ...     ProsodyKeyframe(time=0.5, emotion="happy", intensity=0.9),
            ...     ProsodyKeyframe(time=1.0, emotion="calm", intensity=0.4),
            ... ]
            >>>
            >>> # Convert to temporal prosody
            >>> prosody = keyframes_to_prosody(keyframes, duration_seconds=3.0)
            >>> temporal = get_temporal_prosody_tokens(prosody, num_segments=4)
            >>>
            >>> # Add batch dimension
            >>> temporal = {k: v.unsqueeze(0) for k, v in temporal.items()}
            >>>
            >>> # Generate with temporal control
            >>> audio = model.generate_with_temporal_prosody(
            ...     input_ids, attention_mask, temporal
            ... )
        """
        # Get temporal prosody embeddings
        prosody_prefix = self.get_temporal_prosody_prefix(
            prosody_dict['semantic'],
            prosody_dict['acoustic'],
            prosody_dict['rhythm'],
            prosody_dict['contour'],
        )

        num_segments = prosody_prefix.shape[1]

        # Get text embeddings
        text_embeds = self.csm.embed_text_tokens(input_ids)
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend attention mask
        batch_size = input_ids.shape[0]
        prosody_mask = torch.ones(
            batch_size, num_segments,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prosody_mask, attention_mask], dim=1)

        # Generate
        return self.csm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            output_audio=True,
            **generate_kwargs
        )


class EmotionToProody:
    """
    Helper to convert high-level emotion labels to prosody vectors.

    This enables natural-language-like control:
        prosody = EmotionToProsody.get_prosody("happy")
        audio = model.generate_with_prosody(text, prosody)

    Based on acoustic correlates of emotion from speech science literature.
    """

    # Typical prosody patterns for emotions (normalized 0-1)
    EMOTION_PROFILES = {
        "neutral": {
            "pitch_mean": 0.5, "pitch_std": 0.3,
            "energy": 0.5, "speaking_rate": 0.5,
            "pause_ratio": 0.3,
        },
        "happy": {
            "pitch_mean": 0.7, "pitch_std": 0.5,
            "energy": 0.7, "speaking_rate": 0.7,
            "pause_ratio": 0.2,
        },
        "sad": {
            "pitch_mean": 0.3, "pitch_std": 0.2,
            "energy": 0.3, "speaking_rate": 0.3,
            "pause_ratio": 0.5,
        },
        "angry": {
            "pitch_mean": 0.6, "pitch_std": 0.6,
            "energy": 0.9, "speaking_rate": 0.6,
            "pause_ratio": 0.1,
        },
        "surprised": {
            "pitch_mean": 0.8, "pitch_std": 0.7,
            "energy": 0.6, "speaking_rate": 0.8,
            "pause_ratio": 0.2,
        },
        "calm": {
            "pitch_mean": 0.4, "pitch_std": 0.2,
            "energy": 0.4, "speaking_rate": 0.4,
            "pause_ratio": 0.4,
        },
        "excited": {
            "pitch_mean": 0.8, "pitch_std": 0.6,
            "energy": 0.9, "speaking_rate": 0.9,
            "pause_ratio": 0.1,
        },
        "fearful": {
            "pitch_mean": 0.6, "pitch_std": 0.5,
            "energy": 0.5, "speaking_rate": 0.7,
            "pause_ratio": 0.3,
        },
    }

    @classmethod
    def get_prosody(
        cls,
        emotion: str,
        intensity: float = 1.0,
        config: ProsodyConfig = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert emotion label to prosody vectors.

        Args:
            emotion: Emotion name (happy, sad, angry, etc.)
            intensity: Intensity of the emotion (0-1)
            config: ProsodyConfig for dimensions

        Returns:
            Dict with prosody tensors ready for the model
        """
        if config is None:
            config = ProsodyConfig()

        emotion = emotion.lower()
        if emotion not in cls.EMOTION_PROFILES:
            emotion = "neutral"

        profile = cls.EMOTION_PROFILES[emotion]

        # Create semantic vector (one-hot-ish for emotion)
        emotions = list(cls.EMOTION_PROFILES.keys())
        semantic = torch.zeros(config.semantic_dim)
        if emotion in emotions:
            idx = emotions.index(emotion)
            if idx < config.semantic_dim:
                semantic[idx] = intensity

        # Create acoustic vector from profile
        acoustic = torch.zeros(config.acoustic_dim)
        acoustic[0] = profile["pitch_mean"] * intensity
        acoustic[1] = profile["pitch_std"] * intensity
        acoustic[2] = profile["energy"] * intensity
        # Fill rest with reasonable defaults
        acoustic[3:] = 0.5

        # Create rhythm vector
        rhythm = torch.zeros(config.rhythm_dim)
        rhythm[0] = profile["speaking_rate"] * intensity
        rhythm[1] = profile["pause_ratio"] * intensity
        rhythm[2:] = 0.5

        # Create contour (simplified - real would be time series)
        contour = torch.zeros(config.contour_dim)
        # Generate a simple contour based on emotion
        t = torch.linspace(0, 1, config.contour_dim)
        if emotion == "happy":
            contour = 0.5 + 0.2 * torch.sin(t * 4 * math.pi) * intensity
        elif emotion == "sad":
            contour = 0.5 - 0.2 * t * intensity  # Falling contour
        elif emotion == "angry":
            contour = 0.5 + 0.1 * torch.randn(config.contour_dim) * intensity
        elif emotion == "surprised":
            contour = 0.3 + 0.4 * t * intensity  # Rising contour
        else:
            contour = torch.ones(config.contour_dim) * 0.5

        return {
            "semantic": semantic.unsqueeze(0),
            "acoustic": acoustic.unsqueeze(0),
            "rhythm": rhythm.unsqueeze(0),
            "contour": contour.unsqueeze(0),
        }


def extract_prosody_for_conditioning(
    prosody_dict: dict,
    config: ProsodyConfig = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert prosody_analyzer output to conditioning tensors.

    This bridges your existing prosody_analyzer.py with the conditioning module.

    Args:
        prosody_dict: Output from ProsodyAnalyzer.analyze()
        config: ProsodyConfig for dimensions

    Returns:
        Dict with tensors ready for ProsodyControlledCSM
    """
    if config is None:
        config = ProsodyConfig()

    # Extract semantic (emotion from Qwen2-Audio)
    # prosody_analyzer outputs: semantic.emotion (str), semantic.emotion_confidence (float)
    semantic = torch.zeros(config.semantic_dim)
    if 'semantic' in prosody_dict and prosody_dict['semantic'] is not None:
        sem_data = prosody_dict['semantic']
        # Handle the actual format: emotion (string) + emotion_confidence (float)
        if 'emotion' in sem_data:
            emotion_name = sem_data['emotion'].lower()
            confidence = sem_data.get('emotion_confidence', 0.5)
            # Map emotion name to index using EmotionToProsody's list
            emotion_list = ["neutral", "happy", "sad", "angry", "surprised", "calm", "excited", "fearful"]
            if emotion_name in emotion_list:
                idx = emotion_list.index(emotion_name)
                if idx < config.semantic_dim:
                    semantic[idx] = confidence
            else:
                # Unknown emotion - set neutral with low confidence
                semantic[0] = 0.3
        # Also support legacy format: emotions (dict of scores)
        elif 'emotions' in sem_data:
            emotions = sem_data['emotions']
            for i, (emotion, score) in enumerate(emotions.items()):
                if i < config.semantic_dim:
                    semantic[i] = score

    # Extract acoustic features
    # prosody_analyzer outputs: pitch_mean, pitch_std, hnr, jitter, shimmer, f1_mean, f2_mean, f3_mean
    acoustic = torch.zeros(config.acoustic_dim)
    if 'acoustic' in prosody_dict:
        aco = prosody_dict['acoustic']
        acoustic[0] = aco.get('pitch_mean', 0) / 300  # Normalize
        acoustic[1] = aco.get('pitch_std', 0) / 50
        acoustic[2] = aco.get('hnr', 0) / 30
        acoustic[3] = aco.get('jitter', 0) * 100
        acoustic[4] = aco.get('shimmer', 0) * 10
        # Formants - prosody_analyzer uses f1_mean, f2_mean, f3_mean (not nested)
        acoustic[5] = aco.get('f1_mean', aco.get('formants', {}).get('f1', 500)) / 1000
        acoustic[6] = aco.get('f2_mean', aco.get('formants', {}).get('f2', 1500)) / 3000
        acoustic[7] = aco.get('f3_mean', aco.get('formants', {}).get('f3', 2500)) / 4000

    # Extract rhythm features
    # prosody_analyzer outputs: speaking_rate, articulation_rate, speech_to_pause_ratio, syllable_count, etc.
    rhythm = torch.zeros(config.rhythm_dim)
    if 'rhythm' in prosody_dict:
        rhy = prosody_dict['rhythm']
        rhythm[0] = rhy.get('speaking_rate', 4) / 8  # ~4 syllables/sec normal
        # Use speech_to_pause_ratio (actual field) or fall back to pause_ratio
        pause_ratio = rhy.get('pause_ratio', None)
        if pause_ratio is None:
            # Convert speech_to_pause_ratio to pause_ratio: if ratio is 4:1, pause is 0.2
            stp_ratio = rhy.get('speech_to_pause_ratio', 4.0)
            pause_ratio = 1.0 / (1.0 + stp_ratio) if stp_ratio < 100 else 0.0
        rhythm[1] = pause_ratio
        # Use syllable_count normalized by duration, or syllable_rate if available
        syllable_rate = rhy.get('syllable_rate', None)
        if syllable_rate is None:
            syllable_count = rhy.get('syllable_count', 0)
            duration = rhy.get('duration_seconds', 1.0)
            syllable_rate = syllable_count / duration if duration > 0 else 4
        rhythm[2] = syllable_rate / 8
        rhythm[3] = rhy.get('articulation_rate', 5) / 10

    # Extract contour (pitch trajectory)
    # prosody_analyzer outputs: contour.times, contour.values, contour.smoothed (all lists)
    contour = torch.zeros(config.contour_dim)
    if 'contour' in prosody_dict:
        contour_data = prosody_dict['contour']
        # Use smoothed values if available (preferred), otherwise fall back to values
        traj = None
        if 'smoothed' in contour_data and contour_data['smoothed']:
            traj = contour_data['smoothed']
        elif 'values' in contour_data and contour_data['values']:
            traj = contour_data['values']
        # Also support legacy format: pitch_trajectory
        elif 'pitch_trajectory' in contour_data:
            traj = contour_data['pitch_trajectory']

        if traj is not None:
            if isinstance(traj, list):
                traj = torch.tensor(traj, dtype=torch.float32)
            # Resample to fixed length
            if len(traj) > 0:
                traj = F.interpolate(
                    traj.view(1, 1, -1),
                    size=config.contour_dim,
                    mode='linear',
                    align_corners=False
                ).squeeze()
                # Normalize
                traj = (traj - traj.mean()) / (traj.std() + 1e-8)
                contour = torch.clamp(traj * 0.2 + 0.5, 0, 1)

    return {
        "semantic": semantic.unsqueeze(0),
        "acoustic": acoustic.unsqueeze(0),
        "rhythm": rhythm.unsqueeze(0),
        "contour": contour.unsqueeze(0),
    }


# Test the module
if __name__ == "__main__":
    print("Testing Prosody Conditioning Module")
    print("=" * 50)

    config = ProsodyConfig()
    encoder = ProsodyEncoder(config)

    # Create dummy prosody inputs
    batch_size = 2
    semantic = torch.randn(batch_size, config.semantic_dim)
    acoustic = torch.randn(batch_size, config.acoustic_dim)
    rhythm = torch.randn(batch_size, config.rhythm_dim)
    contour = torch.randn(batch_size, config.contour_dim)

    # Test encoder
    prefix = encoder(semantic, acoustic, rhythm, contour)
    print(f"Prosody prefix shape: {prefix.shape}")
    print(f"Expected: [{batch_size}, {config.num_prosody_tokens}, {config.hidden_size}]")

    # Test emotion conversion
    print("\nTesting emotion to prosody conversion:")
    for emotion in ["happy", "sad", "angry", "neutral"]:
        prosody = EmotionToProody.get_prosody(emotion)
        print(f"  {emotion}: semantic shape {prosody['semantic'].shape}")

    print("\nProsody conditioning module ready!")
    print("\nNext steps:")
    print("1. Integrate with CSM model")
    print("2. Create training loop for prosody encoder")
    print("3. Add controllable inference")
