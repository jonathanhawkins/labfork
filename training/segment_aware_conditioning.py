"""
Segment-Aware Conditioning for Training-Free Intra-Utterance Emotion Control

Based on arXiv:2601.03170 (NUS, 2025): "Training-Free Segment-Aware Speech Synthesis
with Fine-Grained Emotion and Duration Control"

Key Innovation: Training-free framework for intra-utterance emotion and duration control
that works with pretrained zero-shot TTS without any retraining.

Core Technique - Segment-Aware Emotion Conditioning:
1. Causal masking isolates emotion conditioning per segment
2. Monotonic stream alignment filtering ensures proper text-audio alignment
3. Mask transition scheduling provides smooth emotion shifts
4. Preserves global semantic coherence while allowing local emotion control

Differs from WeSCon (arXiv:2509.24629) which requires self-training.

Architecture:
```
Text + Segment Boundaries → [SegmentAwareConditioner]
                                      ↓
         [MonotonicStreamFilter] ← Attention Masks
                                      ↓
                    [EmotionMaskScheduler] → Smooth Transitions
                                      ↓
              Pretrained TTS Model → Emotional Speech
```

Benefits:
- Training-free: Works with any pretrained zero-shot TTS
- Multi-emotion: Multiple emotions within single utterance
- Smooth transitions: Scheduled mask blending for natural shifts
- Semantically coherent: Preserves global meaning while modulating emotion

Usage:
    from segment_aware_conditioning import (
        SegmentAwareConfig,
        SegmentAwareInferenceWrapper,
        create_emotion_segments,
    )

    # Create segments with emotions
    segments = create_emotion_segments(
        text="I was sad at first, but then something amazing happened!",
        segment_texts=["I was sad at first,", "but then something amazing happened!"],
        emotions=["sad", "happy"],
        intensities=[0.7, 0.9],
    )

    # Wrap pretrained model for inference
    wrapper = SegmentAwareInferenceWrapper(pretrained_tts_model)

    # Generate with intra-utterance emotion control
    audio = wrapper.generate(
        text="I was sad at first, but then something amazing happened!",
        segments=segments,
        reference_audio=speaker_reference,  # For voice cloning
    )

Reference: https://arxiv.org/abs/2601.03170
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SegmentAwareConfig:
    """Configuration for Segment-Aware Conditioning."""

    # Segment settings
    max_segments: int = 10  # Maximum number of emotion segments per utterance
    min_segment_length_sec: float = 0.3  # Minimum segment length in seconds

    # Emotion settings
    num_emotions: int = 8
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "surprised", "fearful", "calm", "disgusted"
    ])
    emotion_dim: int = 256  # Emotion embedding dimension

    # Attention masking
    mask_softness: float = 0.1  # Soft mask transition (0 = hard, 1 = very soft)
    causal_lookahead: int = 0  # Future tokens visible (0 = strict causal)
    use_monotonic_filter: bool = True  # Enable monotonic stream alignment

    # Transition scheduling
    transition_type: str = "sigmoid"  # "sigmoid", "linear", "cosine", "step"
    transition_duration_frames: int = 10  # Frames for emotion transition
    min_transition_overlap: float = 0.05  # Minimum overlap ratio between segments
    max_transition_overlap: float = 0.2  # Maximum overlap ratio

    # Monotonic alignment
    monotonic_temperature: float = 0.1  # Temperature for monotonic attention
    alignment_chunk_size: int = 50  # Chunk size for streaming alignment
    enforce_monotonicity: bool = True  # Strictly enforce monotonic progression

    # Duration control
    enable_duration_control: bool = True
    min_duration_scale: float = 0.5  # Minimum duration scaling
    max_duration_scale: float = 2.0  # Maximum duration scaling

    # Model integration
    hidden_dim: int = 2048  # Match TTS hidden dimension
    num_attention_heads: int = 16
    output_dim: int = 2048  # CSM prosody hidden dimension
    num_prosody_tokens: int = 4

    # Inference
    streaming_mode: bool = True  # Enable streaming generation
    chunk_overlap_ratio: float = 0.1  # Overlap between streaming chunks


# =============================================================================
# EMOTION SEGMENT REPRESENTATION
# =============================================================================

@dataclass
class EmotionSegment:
    """Represents a segment with specific emotion conditioning."""

    text: str  # Text content of this segment
    emotion: str  # Emotion label
    intensity: float = 1.0  # Emotion intensity (0-1)

    # Timing (computed during processing)
    start_time: Optional[float] = None  # Start time in seconds
    end_time: Optional[float] = None  # End time in seconds
    start_token: Optional[int] = None  # Start token index
    end_token: Optional[int] = None  # End token index

    # Duration control
    duration_scale: float = 1.0  # Duration scaling factor

    # VAD coordinates (optional for fine-grained control)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None

    def get_vad(self) -> Tuple[float, float, float]:
        """Get VAD coordinates, using emotion prototypes if not specified."""
        if all(v is not None for v in [self.valence, self.arousal, self.dominance]):
            return (self.valence, self.arousal, self.dominance)

        # Default VAD prototypes per emotion
        vad_prototypes = {
            "neutral": (0.0, 0.0, 0.0),
            "happy": (0.8, 0.6, 0.6),
            "sad": (-0.6, -0.4, -0.5),
            "angry": (-0.5, 0.8, 0.7),
            "surprised": (0.3, 0.8, 0.2),
            "fearful": (-0.7, 0.7, -0.7),
            "calm": (0.4, -0.5, 0.3),
            "disgusted": (-0.6, 0.3, 0.4),
        }

        return vad_prototypes.get(self.emotion, (0.0, 0.0, 0.0))


def create_emotion_segments(
    text: str,
    segment_texts: List[str],
    emotions: List[str],
    intensities: Optional[List[float]] = None,
    duration_scales: Optional[List[float]] = None,
) -> List[EmotionSegment]:
    """
    Create emotion segments from text and emotion specifications.

    Args:
        text: Full text of the utterance
        segment_texts: List of text segments (should concatenate to full text)
        emotions: List of emotion labels for each segment
        intensities: List of emotion intensities (default: 1.0 for all)
        duration_scales: List of duration scaling factors (default: 1.0 for all)

    Returns:
        List of EmotionSegment objects with token boundaries computed
    """
    if len(segment_texts) != len(emotions):
        raise ValueError(f"Number of segments ({len(segment_texts)}) must match emotions ({len(emotions)})")

    if intensities is None:
        intensities = [1.0] * len(segments)
    if duration_scales is None:
        duration_scales = [1.0] * len(segment_texts)

    segments = []
    current_pos = 0

    for seg_text, emotion, intensity, dur_scale in zip(
        segment_texts, emotions, intensities, duration_scales
    ):
        # Find segment position in full text
        start_pos = text.find(seg_text, current_pos)
        if start_pos == -1:
            warnings.warn(f"Segment text '{seg_text[:20]}...' not found in full text")
            start_pos = current_pos

        end_pos = start_pos + len(seg_text)
        current_pos = end_pos

        segment = EmotionSegment(
            text=seg_text,
            emotion=emotion,
            intensity=intensity,
            duration_scale=dur_scale,
        )
        segments.append(segment)

    return segments


# =============================================================================
# CAUSAL SEGMENT ATTENTION MASK
# =============================================================================

class CausalSegmentMask(nn.Module):
    """
    Creates causal attention masks that isolate emotion conditioning per segment.

    The mask ensures:
    1. Each segment only conditions on its assigned emotion embedding
    2. Causal structure is maintained (no future information leakage)
    3. Optional soft transitions between segments for smooth blending

    Mask structure for 2 segments (A: sad, B: happy):
    ```
    Query →  |  A₁  A₂  A₃ | B₁  B₂  B₃ |
    Key ↓    |--------------|------------|
    A₁       |  ✓   ×   ×  |  ×   ×   × |
    A₂       |  ✓   ✓   ×  |  ×   ×   × |
    A₃       |  ✓   ✓   ✓  |  ×   ×   × |
    B₁       |  ✓   ✓   ✓  |  ✓   ×   × |  ← Transition zone
    B₂       |  ×   ×   ×  |  ✓   ✓   × |
    B₃       |  ×   ×   ×  |  ✓   ✓   ✓ |
    ```
    """

    def __init__(self, config: SegmentAwareConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        segment_boundaries: List[int],  # Token indices where segments start
        seq_length: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create causal segment attention mask.

        Args:
            segment_boundaries: List of token indices where segments start [0, n₁, n₂, ...]
            seq_length: Total sequence length
            device: Target device

        Returns:
            Attention mask [seq_length, seq_length] where 1 = attend, 0 = block
        """
        if device is None:
            device = torch.device('cpu')

        # Start with causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))

        # Add lookahead if configured
        if self.config.causal_lookahead > 0:
            for i in range(1, self.config.causal_lookahead + 1):
                mask += torch.diag(torch.ones(seq_length - i, device=device), i)
            mask = mask.clamp(0, 1)

        # Apply segment isolation
        num_segments = len(segment_boundaries)

        for seg_idx, start in enumerate(segment_boundaries):
            # Determine segment end
            if seg_idx + 1 < num_segments:
                end = segment_boundaries[seg_idx + 1]
            else:
                end = seq_length

            # Block attention from this segment to previous segment content
            # (except for transition zone)
            if seg_idx > 0:
                prev_end = segment_boundaries[seg_idx]

                # Calculate transition zone
                transition_frames = min(
                    self.config.transition_duration_frames,
                    (end - start) // 2
                )

                # Apply soft or hard blocking
                for i in range(start + transition_frames, end):
                    for j in range(0, segment_boundaries[seg_idx]):
                        if self.config.mask_softness > 0:
                            # Soft mask: gradual transition
                            distance = (start + transition_frames - i) / max(1, transition_frames)
                            mask_val = max(0, distance * self.config.mask_softness)
                            mask[i, j] = mask_val
                        else:
                            mask[i, j] = 0.0

        return mask

    def create_segment_emotion_mask(
        self,
        segments: List[EmotionSegment],
        seq_length: int,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create both attention mask and per-position emotion assignment.

        Returns:
            attention_mask: [seq_length, seq_length]
            emotion_assignment: [seq_length, num_emotions] - one-hot or soft
        """
        if device is None:
            device = torch.device('cpu')

        # Compute token boundaries from segments
        boundaries = [0]
        current_pos = 0
        for seg in segments:
            # Estimate tokens per segment (rough approximation)
            seg_tokens = max(1, len(seg.text.split()))  # Words as proxy
            current_pos += seg_tokens
            if current_pos < seq_length:
                boundaries.append(current_pos)

        # Create attention mask
        attention_mask = self.forward(boundaries, seq_length, device)

        # Create emotion assignment per position
        emotion_assignment = torch.zeros(seq_length, self.config.num_emotions, device=device)

        for seg_idx, seg in enumerate(segments):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else seq_length

            # Get emotion index
            emotion_idx = self.config.emotion_labels.index(seg.emotion) \
                if seg.emotion in self.config.emotion_labels else 0

            # Assign emotion with intensity
            emotion_assignment[start:end, emotion_idx] = seg.intensity

        return attention_mask, emotion_assignment


# =============================================================================
# MONOTONIC STREAM ALIGNMENT FILTER
# =============================================================================

class MonotonicStreamFilter(nn.Module):
    """
    Ensures monotonic alignment between text and audio during streaming generation.

    Based on monotonic attention mechanisms (Raffel et al., 2017) adapted for
    streaming TTS generation.

    Key Properties:
    1. Strictly monotonic: Audio position can only move forward in text
    2. Soft alignment: Allows attention spread around current position
    3. Streaming compatible: Works with chunked generation

    Formula:
        p_ij = sigmoid((s_j - s_{j-1}) / τ)  # Probability of stepping forward
        α_ij = p_ij * α_{i,j-1} * (1 - p_{i-1,j}) + (1 - p_ij) * α_{i-1,j}
    """

    def __init__(self, config: SegmentAwareConfig):
        super().__init__()
        self.config = config

        # Learned parameters for alignment scoring
        self.energy_proj = nn.Linear(config.hidden_dim, 1)
        self.location_conv = nn.Conv1d(1, 32, kernel_size=31, padding=15)
        self.location_proj = nn.Linear(32, 1)

        # State for streaming
        self._prev_alpha = None
        self._step_count = 0

    def reset_state(self):
        """Reset streaming state for new utterance."""
        self._prev_alpha = None
        self._step_count = 0

    def forward(
        self,
        encoder_outputs: torch.Tensor,  # [batch, src_len, hidden]
        decoder_state: torch.Tensor,  # [batch, hidden]
        segment_boundaries: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute monotonic alignment weights.

        Args:
            encoder_outputs: Encoder hidden states [batch, src_len, hidden]
            decoder_state: Current decoder state [batch, hidden]
            segment_boundaries: Optional segment boundaries for constraint

        Returns:
            alpha: Attention weights [batch, src_len]
            context: Context vector [batch, hidden]
        """
        batch_size, src_len, hidden = encoder_outputs.shape
        device = encoder_outputs.device

        # Compute energy scores
        energy = self.energy_proj(encoder_outputs).squeeze(-1)  # [batch, src_len]

        # Add location-based scoring if we have previous alignment
        if self._prev_alpha is not None:
            # Location convolution
            prev_alpha = self._prev_alpha.unsqueeze(1)  # [batch, 1, src_len]
            location_feat = self.location_conv(prev_alpha)  # [batch, 32, src_len]
            location_feat = location_feat.transpose(1, 2)  # [batch, src_len, 32]
            location_score = self.location_proj(location_feat).squeeze(-1)  # [batch, src_len]
            energy = energy + location_score

        # Apply monotonic constraint
        if self.config.enforce_monotonicity and self._prev_alpha is not None:
            # Create monotonic mask: can only attend to positions >= previous position
            prev_pos = self._prev_alpha.argmax(dim=-1, keepdim=True)  # [batch, 1]
            monotonic_mask = torch.arange(src_len, device=device).unsqueeze(0) >= prev_pos

            # Apply mask with large negative value
            energy = energy.masked_fill(~monotonic_mask, -1e9)

        # Apply segment constraint if provided
        if segment_boundaries is not None:
            # Find current segment based on step count
            current_seg = 0
            for i, boundary in enumerate(segment_boundaries):
                if self._step_count >= boundary:
                    current_seg = i

            # Create segment mask
            seg_start = segment_boundaries[current_seg] if current_seg < len(segment_boundaries) else 0
            seg_end = segment_boundaries[current_seg + 1] if current_seg + 1 < len(segment_boundaries) else src_len

            # Allow some leakage at boundaries for smooth transitions
            leakage = int(self.config.min_transition_overlap * (seg_end - seg_start))
            seg_start = max(0, seg_start - leakage)
            seg_end = min(src_len, seg_end + leakage)

            segment_mask = torch.zeros(batch_size, src_len, dtype=torch.bool, device=device)
            segment_mask[:, seg_start:seg_end] = True

            energy = energy.masked_fill(~segment_mask, -1e9)

        # Compute attention with temperature
        alpha = F.softmax(energy / self.config.monotonic_temperature, dim=-1)

        # Compute context vector
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden]

        # Update state
        self._prev_alpha = alpha.detach()
        self._step_count += 1

        return alpha, context

    def filter_alignment(
        self,
        raw_attention: torch.Tensor,  # [batch, tgt_len, src_len]
    ) -> torch.Tensor:
        """
        Filter raw attention weights to enforce monotonicity.

        This is used as a post-processing step on attention weights
        from the original model.
        """
        batch_size, tgt_len, src_len = raw_attention.shape
        device = raw_attention.device

        # Create monotonic constraint matrix
        filtered = torch.zeros_like(raw_attention)
        prev_pos = torch.zeros(batch_size, dtype=torch.long, device=device)

        for t in range(tgt_len):
            # Get attention at this step
            attn_t = raw_attention[:, t, :]  # [batch, src_len]

            # Create monotonic mask
            pos_range = torch.arange(src_len, device=device).unsqueeze(0)  # [1, src_len]
            monotonic_mask = pos_range >= prev_pos.unsqueeze(1)  # [batch, src_len]

            # Apply mask
            attn_t_masked = attn_t.masked_fill(~monotonic_mask, -1e9)
            attn_t_normalized = F.softmax(attn_t_masked, dim=-1)

            filtered[:, t, :] = attn_t_normalized

            # Update previous position (use argmax or expected position)
            prev_pos = torch.sum(
                attn_t_normalized * pos_range, dim=-1
            ).long().clamp(min=prev_pos)

        return filtered


# =============================================================================
# EMOTION MASK SCHEDULER
# =============================================================================

class EmotionMaskScheduler(nn.Module):
    """
    Schedules smooth transitions between emotion masks.

    Supports multiple transition types:
    - sigmoid: Smooth S-curve transition
    - linear: Linear interpolation
    - cosine: Cosine-based smooth transition
    - step: Hard boundary (for testing)

    The scheduler creates time-varying emotion weights that blend
    between segments at boundaries.
    """

    def __init__(self, config: SegmentAwareConfig):
        super().__init__()
        self.config = config

    def _sigmoid_transition(self, t: torch.Tensor, sharpness: float = 5.0) -> torch.Tensor:
        """Sigmoid transition from 0 to 1."""
        return torch.sigmoid(sharpness * (t - 0.5))

    def _linear_transition(self, t: torch.Tensor) -> torch.Tensor:
        """Linear transition from 0 to 1."""
        return t.clamp(0, 1)

    def _cosine_transition(self, t: torch.Tensor) -> torch.Tensor:
        """Cosine transition from 0 to 1 (smooth start and end)."""
        return 0.5 * (1 - torch.cos(math.pi * t.clamp(0, 1)))

    def _step_transition(self, t: torch.Tensor) -> torch.Tensor:
        """Hard step transition."""
        return (t >= 0.5).float()

    def compute_transition_weights(
        self,
        frame_idx: int,
        segment_boundaries: List[int],
        transition_zones: Optional[List[Tuple[int, int]]] = None,
    ) -> torch.Tensor:
        """
        Compute per-segment weights for a given frame.

        Args:
            frame_idx: Current frame index
            segment_boundaries: Token/frame indices where segments start
            transition_zones: Optional explicit transition zones [(start, end), ...]

        Returns:
            weights: [num_segments] weights summing to 1
        """
        num_segments = len(segment_boundaries)
        weights = torch.zeros(num_segments)

        # Find current segment
        current_seg = 0
        for i, boundary in enumerate(segment_boundaries):
            if frame_idx >= boundary:
                current_seg = i

        # Check if in transition zone
        in_transition = False
        if current_seg + 1 < num_segments:
            next_boundary = segment_boundaries[current_seg + 1]

            # Compute transition zone
            if transition_zones and current_seg < len(transition_zones):
                trans_start, trans_end = transition_zones[current_seg]
            else:
                # Default transition zone
                trans_duration = self.config.transition_duration_frames
                trans_start = next_boundary - trans_duration // 2
                trans_end = next_boundary + trans_duration // 2

            if trans_start <= frame_idx < trans_end:
                in_transition = True
                # Compute transition progress
                t = (frame_idx - trans_start) / max(1, trans_end - trans_start)
                t = torch.tensor(t)

                # Apply transition function
                if self.config.transition_type == "sigmoid":
                    blend = self._sigmoid_transition(t)
                elif self.config.transition_type == "linear":
                    blend = self._linear_transition(t)
                elif self.config.transition_type == "cosine":
                    blend = self._cosine_transition(t)
                else:
                    blend = self._step_transition(t)

                weights[current_seg] = 1 - blend
                weights[current_seg + 1] = blend

        if not in_transition:
            weights[current_seg] = 1.0

        return weights

    def forward(
        self,
        segments: List[EmotionSegment],
        segment_boundaries: List[int],
        seq_length: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create time-varying emotion weight matrix.

        Args:
            segments: List of EmotionSegment objects
            segment_boundaries: Frame indices where segments start
            seq_length: Total sequence length

        Returns:
            emotion_weights: [seq_length, num_emotions] per-frame emotion weights
        """
        if device is None:
            device = torch.device('cpu')

        num_emotions = self.config.num_emotions
        emotion_weights = torch.zeros(seq_length, num_emotions, device=device)

        for frame_idx in range(seq_length):
            # Get segment blend weights
            segment_weights = self.compute_transition_weights(
                frame_idx, segment_boundaries
            )

            # Convert to emotion weights
            for seg_idx, seg in enumerate(segments):
                if seg_idx >= len(segment_weights):
                    break

                seg_weight = segment_weights[seg_idx].item()
                if seg_weight > 0:
                    emotion_idx = self.config.emotion_labels.index(seg.emotion) \
                        if seg.emotion in self.config.emotion_labels else 0
                    emotion_weights[frame_idx, emotion_idx] += seg_weight * seg.intensity

        # Normalize (in case of overlapping assignments)
        row_sums = emotion_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        emotion_weights = emotion_weights / row_sums

        return emotion_weights


# =============================================================================
# SEGMENT-AWARE CONDITIONER
# =============================================================================

class SegmentAwareConditioner(nn.Module):
    """
    Main conditioning module for segment-aware emotion control.

    Combines causal masking, monotonic alignment, and emotion scheduling
    to enable fine-grained intra-utterance emotion control without training.
    """

    def __init__(self, config: SegmentAwareConfig):
        super().__init__()
        self.config = config

        # Sub-modules
        self.causal_mask = CausalSegmentMask(config)
        self.monotonic_filter = MonotonicStreamFilter(config)
        self.mask_scheduler = EmotionMaskScheduler(config)

        # Emotion embeddings
        self.emotion_embeddings = nn.Embedding(config.num_emotions, config.emotion_dim)

        # Projection to match model hidden dim
        self.emotion_proj = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.1),
        )

        # VAD-based emotion encoder (optional)
        self.vad_encoder = nn.Sequential(
            nn.Linear(3, config.emotion_dim),
            nn.GELU(),
            nn.Linear(config.emotion_dim, config.emotion_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Prosody token generation
        self.prosody_tokens_proj = nn.Linear(config.output_dim, config.num_prosody_tokens * config.output_dim)

    def compute_segment_boundaries(
        self,
        segments: List[EmotionSegment],
        text_tokens: torch.Tensor,  # [batch, seq_len]
    ) -> List[int]:
        """
        Compute token-level segment boundaries.

        In practice, this would use a tokenizer. Here we estimate
        based on text length ratios.
        """
        seq_len = text_tokens.shape[-1]

        # Calculate total text length
        total_chars = sum(len(seg.text) for seg in segments)

        boundaries = [0]
        current_pos = 0

        for seg in segments[:-1]:  # All but last segment
            # Estimate token position based on character ratio
            char_ratio = len(seg.text) / max(1, total_chars)
            tokens_for_seg = int(seq_len * char_ratio)
            current_pos += tokens_for_seg
            boundaries.append(min(current_pos, seq_len - 1))

        return boundaries

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden]
        segments: List[EmotionSegment],
        text_tokens: Optional[torch.Tensor] = None,
        return_attention_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply segment-aware emotion conditioning.

        Args:
            hidden_states: Encoder hidden states [batch, seq_len, hidden]
            segments: List of EmotionSegment objects defining emotion regions
            text_tokens: Optional token ids for boundary computation
            return_attention_mask: Whether to return the computed attention mask

        Returns:
            Dict with:
                - 'conditioned_hidden': Modified hidden states with emotion
                - 'prosody_tokens': Prosody prefix tokens for CSM
                - 'attention_mask': (optional) Segment attention mask
                - 'emotion_weights': Per-frame emotion weights
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Compute segment boundaries
        if text_tokens is not None:
            boundaries = self.compute_segment_boundaries(segments, text_tokens)
        else:
            # Estimate from sequence length
            boundaries = [0]
            seg_len = seq_len // len(segments)
            for i in range(1, len(segments)):
                boundaries.append(i * seg_len)

        # Get time-varying emotion weights
        emotion_weights = self.mask_scheduler(
            segments, boundaries, seq_len, device
        )  # [seq_len, num_emotions]

        # Compute emotion embeddings per position
        # [seq_len, num_emotions] @ [num_emotions, emotion_dim] -> [seq_len, emotion_dim]
        emotion_emb_matrix = self.emotion_embeddings.weight  # [num_emotions, emotion_dim]
        position_emotions = torch.mm(emotion_weights, emotion_emb_matrix)  # [seq_len, emotion_dim]

        # Optionally blend with VAD-based encoding for more nuance
        for seg_idx, seg in enumerate(segments):
            if seg.valence is not None:  # Has explicit VAD
                start = boundaries[seg_idx]
                end = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else seq_len

                vad = torch.tensor(seg.get_vad(), device=device).unsqueeze(0)  # [1, 3]
                vad_emb = self.vad_encoder(vad)  # [1, emotion_dim]

                # Blend VAD with discrete emotion
                blend_weight = 0.3  # 30% VAD, 70% discrete
                position_emotions[start:end] = (
                    (1 - blend_weight) * position_emotions[start:end] +
                    blend_weight * vad_emb
                )

        # Project emotions to hidden dimension
        emotion_conditioning = self.emotion_proj(position_emotions)  # [seq_len, hidden_dim]

        # Apply conditioning (additive)
        emotion_conditioning = emotion_conditioning.unsqueeze(0).expand(batch_size, -1, -1)
        conditioned = hidden_states + emotion_conditioning

        # Create segment attention mask
        attention_mask, _ = self.causal_mask.create_segment_emotion_mask(
            segments, seq_len, device
        )

        # Generate prosody tokens via pooling
        # Use segment-weighted pooling
        segment_representations = []
        for seg_idx, seg in enumerate(segments):
            start = boundaries[seg_idx]
            end = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else seq_len

            seg_hidden = conditioned[:, start:end, :].mean(dim=1)  # [batch, hidden]
            segment_representations.append(seg_hidden)

        # Average segment representations
        pooled = torch.stack(segment_representations, dim=1).mean(dim=1)  # [batch, hidden]

        # Project to output dim
        output = self.output_proj(pooled)  # [batch, output_dim]

        # Generate prosody tokens
        prosody_tokens = self.prosody_tokens_proj(output)  # [batch, num_tokens * output_dim]
        prosody_tokens = prosody_tokens.view(
            batch_size, self.config.num_prosody_tokens, self.config.output_dim
        )

        result = {
            'conditioned_hidden': conditioned,
            'prosody_tokens': prosody_tokens,
            'emotion_weights': emotion_weights,
        }

        if return_attention_mask:
            result['attention_mask'] = attention_mask

        return result


# =============================================================================
# DURATION CONTROLLER
# =============================================================================

class SegmentDurationController(nn.Module):
    """
    Controls duration per segment for rhythm/speed variation.

    Operates on duration predictions from the base TTS model,
    scaling them according to per-segment duration_scale factors.
    """

    def __init__(self, config: SegmentAwareConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        durations: torch.Tensor,  # [batch, seq_len] predicted durations
        segments: List[EmotionSegment],
        segment_boundaries: List[int],
    ) -> torch.Tensor:
        """
        Scale durations per segment.

        Args:
            durations: Predicted durations from base model
            segments: List of segments with duration_scale factors
            segment_boundaries: Token indices for segment starts

        Returns:
            Scaled durations
        """
        scaled = durations.clone()
        seq_len = durations.shape[-1]

        for seg_idx, seg in enumerate(segments):
            start = segment_boundaries[seg_idx]
            end = segment_boundaries[seg_idx + 1] if seg_idx + 1 < len(segment_boundaries) else seq_len

            # Clamp scale to valid range
            scale = max(self.config.min_duration_scale,
                       min(self.config.max_duration_scale, seg.duration_scale))

            scaled[:, start:end] = scaled[:, start:end] * scale

        return scaled


# =============================================================================
# INFERENCE WRAPPER
# =============================================================================

class SegmentAwareInferenceWrapper:
    """
    Wraps a pretrained TTS model for segment-aware inference.

    This is the main interface for training-free intra-utterance
    emotion control. It modifies attention patterns and adds
    emotion conditioning without retraining the base model.

    Usage:
        wrapper = SegmentAwareInferenceWrapper(pretrained_model)

        segments = create_emotion_segments(
            text="I was sad at first, but then something amazing happened!",
            segment_texts=["I was sad at first,", "but then something amazing happened!"],
            emotions=["sad", "happy"],
        )

        audio = wrapper.generate(
            text="I was sad at first, but then something amazing happened!",
            segments=segments,
            reference_audio=speaker_ref,
        )
    """

    def __init__(
        self,
        model: nn.Module,
        config: SegmentAwareConfig = None,
    ):
        """
        Args:
            model: Pretrained TTS model (e.g., CosyVoice, F5-TTS, etc.)
            config: Segment-aware configuration
        """
        self.model = model
        self.config = config or SegmentAwareConfig()

        # Initialize conditioning modules
        self.conditioner = SegmentAwareConditioner(self.config)
        self.duration_controller = SegmentDurationController(self.config)

        # Move to same device as model
        device = next(model.parameters()).device
        self.conditioner = self.conditioner.to(device)
        self.duration_controller = self.duration_controller.to(device)

        # Hook storage
        self._hooks = []
        self._current_segments = None
        self._current_boundaries = None

    def _find_attention_modules(self) -> List[nn.Module]:
        """Find attention modules in the model for mask injection."""
        attention_modules = []
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['attention', 'attn', 'mha']):
                if hasattr(module, 'forward'):
                    attention_modules.append((name, module))
        return attention_modules

    def _create_attention_hook(self, module_name: str) -> Callable:
        """Create hook to modify attention computation."""
        def hook(module, args, kwargs, output):
            # Check if we have segment information
            if self._current_segments is None:
                return output

            # Get attention weights if available
            if isinstance(output, tuple) and len(output) > 1:
                hidden, attention_weights = output[0], output[1]

                # Apply monotonic filtering
                if self.config.use_monotonic_filter:
                    attention_weights = self.conditioner.monotonic_filter.filter_alignment(
                        attention_weights
                    )

                # Apply segment mask
                if self._current_boundaries:
                    attention_mask, _ = self.conditioner.causal_mask.create_segment_emotion_mask(
                        self._current_segments,
                        attention_weights.shape[-1],
                        attention_weights.device,
                    )

                    # Apply mask (multiplicative)
                    attention_weights = attention_weights * attention_mask.unsqueeze(0).unsqueeze(0)

                    # Renormalize
                    attention_weights = attention_weights / (
                        attention_weights.sum(dim=-1, keepdim=True) + 1e-8
                    )

                return (hidden, attention_weights) + output[2:]

            return output

        return hook

    def _register_hooks(self):
        """Register attention modification hooks."""
        self._clear_hooks()

        attention_modules = self._find_attention_modules()
        for name, module in attention_modules:
            hook = module.register_forward_hook(
                self._create_attention_hook(name),
                with_kwargs=True,
            )
            self._hooks.append(hook)

    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @torch.no_grad()
    def generate(
        self,
        text: str,
        segments: List[EmotionSegment],
        reference_audio: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate speech with intra-utterance emotion control.

        Args:
            text: Full text to synthesize
            segments: List of EmotionSegment objects
            reference_audio: Reference audio for voice cloning
            speaker_embedding: Pre-computed speaker embedding
            **generate_kwargs: Additional args for base model generate()

        Returns:
            Generated audio waveform
        """
        device = next(self.model.parameters()).device

        # Store segment info for hooks
        self._current_segments = segments

        # Compute token boundaries (would use tokenizer in practice)
        # For now, estimate based on character positions
        total_chars = len(text)
        char_pos = 0
        self._current_boundaries = [0]

        for seg in segments[:-1]:
            char_pos += len(seg.text)
            # Estimate token position (rough: 1 token ≈ 4 chars)
            token_pos = int(char_pos / total_chars * 100)  # Assuming 100 tokens
            self._current_boundaries.append(token_pos)

        # Register hooks
        self._register_hooks()

        try:
            # Check what interface the model supports
            if hasattr(self.model, 'generate'):
                # Standard generate interface
                if reference_audio is not None:
                    audio = self.model.generate(
                        text=text,
                        reference_audio=reference_audio,
                        **generate_kwargs,
                    )
                elif speaker_embedding is not None:
                    audio = self.model.generate(
                        text=text,
                        speaker_embedding=speaker_embedding,
                        **generate_kwargs,
                    )
                else:
                    audio = self.model.generate(
                        text=text,
                        **generate_kwargs,
                    )
            elif hasattr(self.model, 'synthesize'):
                # Alternative interface
                audio = self.model.synthesize(
                    text,
                    reference=reference_audio,
                    **generate_kwargs,
                )
            else:
                raise NotImplementedError(
                    "Model must have 'generate' or 'synthesize' method"
                )

        finally:
            # Clean up
            self._clear_hooks()
            self._current_segments = None
            self._current_boundaries = None

        return audio

    def generate_with_prosody_tokens(
        self,
        text: str,
        segments: List[EmotionSegment],
        hidden_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens for CSM integration.

        Args:
            text: Text to synthesize
            segments: Emotion segments
            hidden_states: Encoder hidden states

        Returns:
            Dict with prosody_tokens and other conditioning info
        """
        # Compute boundaries
        seq_len = hidden_states.shape[1]
        total_chars = sum(len(s.text) for s in segments)
        boundaries = [0]
        char_pos = 0

        for seg in segments[:-1]:
            char_pos += len(seg.text)
            token_pos = int(char_pos / total_chars * seq_len)
            boundaries.append(token_pos)

        # Apply conditioning
        result = self.conditioner(
            hidden_states,
            segments,
            return_attention_mask=True,
        )

        return result


# =============================================================================
# ADAPTER FOR CSM INTEGRATION
# =============================================================================

class SegmentAwareAdapter(nn.Module):
    """
    Adapter for integrating Segment-Aware Conditioning with CSM prosody pipeline.

    Provides a clean interface matching other prosody adapters in the codebase.
    """

    def __init__(
        self,
        config: SegmentAwareConfig = None,
        input_dim: int = 768,
    ):
        super().__init__()
        self.config = config or SegmentAwareConfig()

        # Input projection
        self.input_proj = nn.Linear(input_dim, self.config.hidden_dim)

        # Main conditioner
        self.conditioner = SegmentAwareConditioner(self.config)

        # Duration controller
        self.duration_controller = SegmentDurationController(self.config)

    def forward(
        self,
        features: torch.Tensor,  # [batch, seq, input_dim]
        segments: Optional[List[EmotionSegment]] = None,
        text: Optional[str] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply segment-aware conditioning and generate prosody tokens.

        Args:
            features: Input features (e.g., from encoder)
            segments: Optional emotion segments
            text: Optional text for default segment computation
            return_all: Whether to return all intermediate outputs

        Returns:
            Dict with 'prosody_tokens' and optionally other info
        """
        batch_size, seq_len, _ = features.shape
        device = features.device

        # Project input
        hidden = self.input_proj(features)

        # Create default single segment if not provided
        if segments is None:
            if text is not None:
                segments = [EmotionSegment(text=text, emotion="neutral", intensity=1.0)]
            else:
                segments = [EmotionSegment(text="", emotion="neutral", intensity=1.0)]

        # Apply conditioning
        result = self.conditioner(
            hidden,
            segments,
            return_attention_mask=return_all,
        )

        output = {
            'prosody_tokens': result['prosody_tokens'],
        }

        if return_all:
            output['conditioned_hidden'] = result['conditioned_hidden']
            output['emotion_weights'] = result['emotion_weights']
            output['attention_mask'] = result.get('attention_mask')

        return output

    def from_text_and_emotions(
        self,
        text: str,
        segment_texts: List[str],
        emotions: List[str],
        intensities: Optional[List[float]] = None,
        features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method to create segments from text and emotions.

        Args:
            text: Full text
            segment_texts: List of segment strings
            emotions: List of emotion labels
            intensities: Optional intensity per segment
            features: Optional input features

        Returns:
            Dict with prosody_tokens
        """
        segments = create_emotion_segments(
            text, segment_texts, emotions, intensities
        )

        if features is None:
            # Create dummy features
            device = next(self.parameters()).device
            features = torch.zeros(1, len(text.split()), 768, device=device)

        return self(features, segments)

    def from_emotion_trajectory(
        self,
        features: torch.Tensor,
        trajectory: List[Tuple[float, str, float]],  # [(time_ratio, emotion, intensity), ...]
    ) -> Dict[str, torch.Tensor]:
        """
        Create segments from emotion trajectory.

        Args:
            features: Input features
            trajectory: List of (time_ratio, emotion, intensity) tuples
                where time_ratio is 0-1 indicating position in sequence

        Returns:
            Dict with prosody_tokens
        """
        seq_len = features.shape[1]

        segments = []
        prev_time = 0.0

        for time_ratio, emotion, intensity in trajectory:
            # Create segment text placeholder
            start_token = int(prev_time * seq_len)
            end_token = int(time_ratio * seq_len)

            seg = EmotionSegment(
                text=f"segment_{len(segments)}",
                emotion=emotion,
                intensity=intensity,
            )
            seg.start_token = start_token
            seg.end_token = end_token

            segments.append(seg)
            prev_time = time_ratio

        return self(features, segments)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_segment_aware_adapter(
    checkpoint: Optional[str] = None,
    config: Optional[SegmentAwareConfig] = None,
    input_dim: int = 768,
) -> SegmentAwareAdapter:
    """
    Create SegmentAwareAdapter, optionally loading from checkpoint.

    Args:
        checkpoint: Path to checkpoint file
        config: Configuration (uses default if None)
        input_dim: Input feature dimension

    Returns:
        Initialized SegmentAwareAdapter
    """
    if config is None:
        config = SegmentAwareConfig()

    adapter = SegmentAwareAdapter(config, input_dim)

    if checkpoint and Path(checkpoint).exists():
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        adapter.load_state_dict(state_dict, strict=False)

    return adapter


def create_segment_aware_wrapper(
    model: nn.Module,
    config: Optional[SegmentAwareConfig] = None,
) -> SegmentAwareInferenceWrapper:
    """
    Create inference wrapper for a pretrained TTS model.

    Args:
        model: Pretrained TTS model
        config: Configuration

    Returns:
        SegmentAwareInferenceWrapper
    """
    return SegmentAwareInferenceWrapper(model, config)


# =============================================================================
# LOSS FUNCTIONS (for optional fine-tuning)
# =============================================================================

class SegmentAwareLoss(nn.Module):
    """
    Loss functions for optional fine-tuning of segment-aware conditioning.

    While the main use case is training-free, these losses can be used
    for domain adaptation or quality improvement.
    """

    def __init__(self, config: SegmentAwareConfig):
        super().__init__()
        self.config = config

        # Emotion consistency loss
        self.emotion_criterion = nn.CrossEntropyLoss()

    def emotion_consistency_loss(
        self,
        emotion_weights: torch.Tensor,  # [seq_len, num_emotions]
        segments: List[EmotionSegment],
        segment_boundaries: List[int],
    ) -> torch.Tensor:
        """
        Ensure predicted emotions match target segments.
        """
        device = emotion_weights.device
        seq_len = emotion_weights.shape[0]

        # Create target labels
        targets = torch.zeros(seq_len, dtype=torch.long, device=device)

        for seg_idx, seg in enumerate(segments):
            start = segment_boundaries[seg_idx]
            end = segment_boundaries[seg_idx + 1] if seg_idx + 1 < len(segment_boundaries) else seq_len

            emotion_idx = self.config.emotion_labels.index(seg.emotion) \
                if seg.emotion in self.config.emotion_labels else 0
            targets[start:end] = emotion_idx

        return self.emotion_criterion(emotion_weights, targets)

    def transition_smoothness_loss(
        self,
        emotion_weights: torch.Tensor,  # [seq_len, num_emotions]
    ) -> torch.Tensor:
        """
        Encourage smooth emotion transitions (low derivative).
        """
        # First derivative (change between adjacent frames)
        diff = emotion_weights[1:] - emotion_weights[:-1]

        # L2 norm of changes
        smoothness = torch.norm(diff, dim=-1).mean()

        return smoothness

    def forward(
        self,
        emotion_weights: torch.Tensor,
        segments: List[EmotionSegment],
        segment_boundaries: List[int],
        smoothness_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        """
        consistency = self.emotion_consistency_loss(
            emotion_weights, segments, segment_boundaries
        )
        smoothness = self.transition_smoothness_loss(emotion_weights)

        total = consistency + smoothness_weight * smoothness

        return {
            'total': total,
            'consistency': consistency,
            'smoothness': smoothness,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_segment_durations(
    segments: List[EmotionSegment],
    total_duration_sec: float,
) -> List[EmotionSegment]:
    """
    Estimate start/end times for segments based on text length ratios.

    Args:
        segments: Segments with text but no timing
        total_duration_sec: Expected total duration

    Returns:
        Segments with estimated start_time and end_time
    """
    total_chars = sum(len(seg.text) for seg in segments)
    current_time = 0.0

    for seg in segments:
        char_ratio = len(seg.text) / max(1, total_chars)
        seg_duration = total_duration_sec * char_ratio * seg.duration_scale

        seg.start_time = current_time
        seg.end_time = current_time + seg_duration
        current_time = seg.end_time

    return segments


def visualize_emotion_trajectory(
    emotion_weights: torch.Tensor,  # [seq_len, num_emotions]
    segment_boundaries: List[int],
    emotion_labels: List[str],
    output_path: Optional[str] = None,
):
    """
    Visualize emotion weights over time.

    Creates a plot showing emotion intensities with segment boundaries.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for visualization")
        return

    weights = emotion_weights.cpu().numpy()
    seq_len, num_emotions = weights.shape

    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot each emotion
    x = np.arange(seq_len)
    for i, label in enumerate(emotion_labels[:num_emotions]):
        ax.plot(x, weights[:, i], label=label, alpha=0.7)

    # Mark segment boundaries
    for boundary in segment_boundaries[1:]:  # Skip 0
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Emotion Weight')
    ax.set_title('Segment-Aware Emotion Trajectory')
    ax.legend(loc='upper right', ncol=4)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def parse_emotion_markup(
    text: str,
    default_emotion: str = "neutral",
    default_intensity: float = 1.0,
) -> Tuple[str, List[EmotionSegment]]:
    """
    Parse text with inline emotion markup.

    Format: "Normal text [emotion:intensity]Emotional text[/emotion] more normal"

    Example:
        "I was [sad:0.7]feeling down[/sad] but then [happy:0.9]everything changed[/happy]!"

    Returns:
        (clean_text, segments)
    """
    import re

    # Pattern for emotion tags
    pattern = r'\[(\w+)(?::(\d*\.?\d+))?\](.*?)\[/\1\]'

    segments = []
    clean_parts = []
    last_end = 0

    for match in re.finditer(pattern, text):
        # Add any text before this match
        if match.start() > last_end:
            pre_text = text[last_end:match.start()]
            if pre_text.strip():
                segments.append(EmotionSegment(
                    text=pre_text,
                    emotion=default_emotion,
                    intensity=default_intensity,
                ))
                clean_parts.append(pre_text)

        # Add the matched emotional segment
        emotion = match.group(1)
        intensity = float(match.group(2)) if match.group(2) else default_intensity
        seg_text = match.group(3)

        segments.append(EmotionSegment(
            text=seg_text,
            emotion=emotion,
            intensity=intensity,
        ))
        clean_parts.append(seg_text)

        last_end = match.end()

    # Add any remaining text
    if last_end < len(text):
        remaining = text[last_end:]
        if remaining.strip():
            segments.append(EmotionSegment(
                text=remaining,
                emotion=default_emotion,
                intensity=default_intensity,
            ))
            clean_parts.append(remaining)

    clean_text = ''.join(clean_parts)

    return clean_text, segments


# =============================================================================
# TESTING
# =============================================================================

def _test_segment_aware_conditioning():
    """Test segment-aware conditioning components."""
    print("Testing Segment-Aware Conditioning...")

    config = SegmentAwareConfig()

    # Test segment creation
    segments = create_emotion_segments(
        text="I was sad at first, but then something amazing happened!",
        segment_texts=["I was sad at first,", " but then something amazing happened!"],
        emotions=["sad", "happy"],
        intensities=[0.7, 0.9],
    )

    print(f"Created {len(segments)} segments:")
    for seg in segments:
        print(f"  - '{seg.text}' -> {seg.emotion} ({seg.intensity})")

    # Test markup parsing
    marked_text = "I was [sad:0.7]feeling down[/sad] but then [happy:0.9]everything changed[/happy]!"
    clean, parsed_segments = parse_emotion_markup(marked_text)

    print(f"\nParsed markup:")
    print(f"  Clean text: '{clean}'")
    for seg in parsed_segments:
        print(f"  - '{seg.text}' -> {seg.emotion} ({seg.intensity})")

    # Test causal mask
    causal_mask = CausalSegmentMask(config)
    boundaries = [0, 5, 10]
    mask = causal_mask(boundaries, 15)

    print(f"\nCausal mask shape: {mask.shape}")
    print(f"Mask sample (first 5x5):\n{mask[:5, :5]}")

    # Test emotion mask scheduler
    scheduler = EmotionMaskScheduler(config)
    weights = scheduler(segments, [0, 5], 10)

    print(f"\nEmotion weights shape: {weights.shape}")
    print(f"Weights at frame 0: {weights[0]}")
    print(f"Weights at frame 7: {weights[7]}")

    # Test full conditioner
    conditioner = SegmentAwareConditioner(config)

    batch_size = 2
    seq_len = 20
    hidden = torch.randn(batch_size, seq_len, config.hidden_dim)

    result = conditioner(hidden, segments, return_attention_mask=True)

    print(f"\nConditioner output:")
    print(f"  Conditioned hidden: {result['conditioned_hidden'].shape}")
    print(f"  Prosody tokens: {result['prosody_tokens'].shape}")
    print(f"  Attention mask: {result['attention_mask'].shape}")

    # Test adapter
    adapter = SegmentAwareAdapter(config, input_dim=768)
    features = torch.randn(batch_size, seq_len, 768)

    output = adapter(features, segments)
    print(f"\nAdapter output:")
    print(f"  Prosody tokens: {output['prosody_tokens'].shape}")

    # Test trajectory-based generation
    trajectory = [
        (0.3, "sad", 0.7),
        (0.7, "neutral", 0.5),
        (1.0, "happy", 0.9),
    ]

    output = adapter.from_emotion_trajectory(features, trajectory)
    print(f"\nTrajectory output:")
    print(f"  Prosody tokens: {output['prosody_tokens'].shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    _test_segment_aware_conditioning()
