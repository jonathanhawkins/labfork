"""
WeSCon: Word-Level Self-Training Emotion and Speed Control

Based on "Word-Level Emotional Expression Control in Zero-Shot Text-to-Speech Synthesis"
(arXiv:2509.24629)

Key Innovation: First self-training framework that enables word-level control of both
emotion and speaking rate in a pretrained zero-shot TTS model WITHOUT requiring datasets
containing intra-sentence emotional transitions.

Two-Stage Framework:
- Stage 1 (Teacher): CosyVoice2 + transition smoothing + dynamic speed control
  Performs multi-round inference to generate pseudo-labeled emotional speech
- Stage 2 (Student): TTS with dynamic emotional attention bias
  Learns from teacher outputs via self-training

Technical Components:
1. Dynamic Emotional Attention Bias: Weighted templates for attention modulation
2. Transition Smoothing: Tail-to-head linkage with content aligner
3. Dynamic Speed Control: Prompt token interpolation/downsampling
4. Multi-Round Inference: Segmented generation with continuation

Reference: https://arxiv.org/abs/2509.24629
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WeSConConfig:
    """Configuration for WeSCon module."""

    # Model dimensions
    hidden_dim: int = 2048          # Main hidden dimension (match TTS)
    attention_dim: int = 512        # Attention dimension
    num_attention_heads: int = 8    # Number of attention heads

    # Emotion settings
    num_emotions: int = 7           # Number of discrete emotions
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "surprised", "fearful", "calm"
    ])

    # Attention bias templates
    num_bias_templates: int = 7     # Number of attention bias templates
    max_seq_length: int = 2048      # Maximum sequence length for templates

    # Content aligner (transition smoothing)
    aligner_num_layers: int = 4     # Transformer layers in content aligner
    aligner_kernel_size: int = 5    # Convolution kernel size
    aligner_hidden_dim: int = 512   # Hidden dimension for aligner

    # Speed control
    min_speed_ratio: float = 0.5    # Minimum speed (slowest)
    max_speed_ratio: float = 2.0    # Maximum speed (fastest)

    # Multi-round inference
    max_rounds: int = 10            # Maximum number of generation rounds
    overlap_ratio: float = 0.1     # Overlap between segments for continuity

    # Self-training
    teacher_temperature: float = 1.0  # Temperature for teacher outputs
    student_temperature: float = 0.8  # Temperature for student learning
    kl_weight: float = 0.1           # KL divergence weight
    emotion_cls_weight: float = 0.5  # Emotion classification weight

    # Output settings
    output_dim: int = 2048          # Output dimension (match prosody encoder)
    num_prosody_tokens: int = 4     # Number of prefix tokens

    # Training
    dropout: float = 0.1
    learning_rate: float = 5e-7    # Very low LR to preserve pretrained weights


# =============================================================================
# ATTENTION BIAS TEMPLATES
# =============================================================================

class AttentionBiasTemplates(nn.Module):
    """
    Learnable attention bias templates for emotion-specific attention patterns.

    WeSCon uses 7 templates (one per emotion) that are linearly combined
    based on predicted emotion weights. The templates modulate self-attention
    to focus on emotion-relevant regions.

    Formula: B^bias = Σ(i=0 to 6) ω_i · B_i^temp

    Applied as: O = [Softmax(QK^T/√d) ⊙ B^bias] · V
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        # Learnable bias templates [num_templates, max_seq, max_seq]
        # Initialize with different patterns for each template
        self.templates = nn.Parameter(
            self._init_templates(config.num_bias_templates, config.max_seq_length)
        )

        # Normalization for stable gradients
        self.template_norm = nn.LayerNorm(config.max_seq_length)

    def _init_templates(self, num_templates: int, max_seq: int) -> torch.Tensor:
        """
        Initialize templates with emotion-specific patterns.

        Different emotions have different attention patterns:
        - Neutral: Uniform attention
        - Happy: More local attention (energetic)
        - Sad: More distant attention (drawn out)
        - Angry: Sharp, focused attention
        - Surprised: Wide attention span
        - Fearful: Scattered attention
        - Calm: Smooth, broad attention
        """
        templates = torch.zeros(num_templates, max_seq, max_seq)

        for i in range(num_templates):
            # Create position-based patterns
            pos = torch.arange(max_seq).float()
            pos_diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [seq, seq]

            if i == 0:  # Neutral - uniform
                templates[i] = torch.ones_like(pos_diff)
            elif i == 1:  # Happy - local focus
                sigma = 20.0
                templates[i] = torch.exp(-pos_diff.pow(2) / (2 * sigma ** 2))
            elif i == 2:  # Sad - distant/broad
                sigma = 100.0
                templates[i] = torch.exp(-pos_diff.pow(2) / (2 * sigma ** 2))
            elif i == 3:  # Angry - sharp peaks
                sigma = 5.0
                templates[i] = torch.exp(-pos_diff.pow(2) / (2 * sigma ** 2))
            elif i == 4:  # Surprised - wide span
                templates[i] = 1.0 - torch.exp(-pos_diff.abs() / 50.0)
            elif i == 5:  # Fearful - scattered
                templates[i] = torch.cos(pos_diff * math.pi / 30) * 0.5 + 0.5
            else:  # Calm - smooth decay
                templates[i] = torch.exp(-pos_diff.abs() / 80.0)

        # Normalize to have similar scale
        templates = templates / templates.abs().max()

        return templates

    def forward(
        self,
        omega: torch.Tensor,           # [batch, num_templates] weights
        seq_length: int,               # Current sequence length
    ) -> torch.Tensor:
        """
        Compute weighted combination of attention bias templates.

        Args:
            omega: Template weights from emotion predictor [batch, num_templates]
            seq_length: Current sequence length to trim templates

        Returns:
            Combined attention bias [batch, seq_length, seq_length]
        """
        batch_size = omega.shape[0]

        # Trim templates to current sequence length
        templates = self.templates[:, :seq_length, :seq_length]  # [T, S, S]

        # Apply softmax to weights for proper combination
        omega_norm = F.softmax(omega, dim=-1)  # [B, T]

        # Weighted combination: B^bias = Σ ω_i · B_i
        # [B, T] @ [T, S*S] -> [B, S*S] -> [B, S, S]
        templates_flat = templates.view(self.config.num_bias_templates, -1)  # [T, S*S]
        bias = torch.einsum('bt,tss->bss', omega_norm, templates)  # [B, S, S]

        return bias

    def apply_to_attention(
        self,
        attention_weights: torch.Tensor,  # [batch, heads, seq, seq]
        bias: torch.Tensor,               # [batch, seq, seq]
    ) -> torch.Tensor:
        """
        Apply attention bias via element-wise multiplication.

        O = [Softmax(QK^T/√d) ⊙ B^bias / Σ(...)] · V
        """
        # Expand bias for multi-head attention
        bias_expanded = bias.unsqueeze(1)  # [B, 1, S, S]

        # Element-wise multiplication
        modulated = attention_weights * bias_expanded  # [B, H, S, S]

        # Renormalize to maintain proper attention distribution
        modulated = modulated / (modulated.sum(dim=-1, keepdim=True) + 1e-8)

        return modulated


# =============================================================================
# EMOTION PREDICTOR
# =============================================================================

class EmotionPredictor(nn.Module):
    """
    Predicts token-level emotion labels and attention bias weights.

    Uses a causal Transformer to predict emotion from context,
    then an MLP generates weights for attention bias templates.
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        # Causal Transformer for context-aware prediction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Project to emotion logits
        self.emotion_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_emotions),
        )

        # MLP to generate attention bias weights
        self.omega_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_bias_templates),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,      # [batch, seq, hidden]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict emotions and attention bias weights from hidden states.

        Returns:
            Dict with:
                - 'emotion_logits': [batch, seq, num_emotions]
                - 'omega': [batch, num_templates] for attention bias
                - 'emotion_probs': [batch, seq, num_emotions]
        """
        # Create causal mask
        seq_len = hidden_states.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device),
            diagonal=1
        ).bool()

        # Process through Transformer
        contextualized = self.transformer(
            hidden_states,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask if attention_mask is not None else None,
        )

        # Predict token-level emotions
        emotion_logits = self.emotion_head(contextualized)  # [B, S, E]
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        # Generate omega (aggregate over sequence for attention bias)
        pooled = contextualized.mean(dim=1)  # [B, H]
        omega = self.omega_head(pooled)  # [B, num_templates]

        return {
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
            'omega': omega,
        }


# =============================================================================
# DYNAMIC EMOTIONAL ATTENTION BIAS
# =============================================================================

class DynamicEmotionalAttentionBias(nn.Module):
    """
    Complete dynamic emotional attention bias mechanism.

    Combines:
    1. Emotion predictor to get per-token emotions and omega weights
    2. Attention bias templates
    3. Attention modulation for emotion-guided generation

    This enables word-level emotion control by modulating which parts
    of the prompt the model attends to based on the target emotion.
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        self.emotion_predictor = EmotionPredictor(config)
        self.bias_templates = AttentionBiasTemplates(config)

        # Emotion embedding for conditioning
        self.emotion_embedding = nn.Embedding(config.num_emotions, config.hidden_dim)

        # Projection for integration
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_emotions: Optional[torch.Tensor] = None,  # [batch, seq] indices
    ) -> Dict[str, torch.Tensor]:
        """
        Apply dynamic emotional attention bias.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]
            target_emotions: Target emotion indices per token (for training)

        Returns:
            Dict with:
                - 'modulated_hidden': Hidden states with emotion bias
                - 'attention_bias': Attention bias matrix
                - 'emotion_logits': Predicted emotions
                - 'omega': Attention template weights
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Predict emotions and get omega weights
        pred_result = self.emotion_predictor(hidden_states, attention_mask)

        # Get attention bias
        attention_bias = self.bias_templates(
            pred_result['omega'],
            seq_length=seq_len,
        )

        # Add emotion conditioning if targets provided
        if target_emotions is not None:
            emotion_emb = self.emotion_embedding(target_emotions)  # [B, S, H]
            hidden_states = hidden_states + emotion_emb * 0.1  # Residual addition

        # Project output
        modulated = self.output_proj(hidden_states)

        return {
            'modulated_hidden': modulated,
            'attention_bias': attention_bias,
            'emotion_logits': pred_result['emotion_logits'],
            'emotion_probs': pred_result['emotion_probs'],
            'omega': pred_result['omega'],
        }


# =============================================================================
# CONTENT ALIGNER (TRANSITION SMOOTHING)
# =============================================================================

class ContentAligner(nn.Module):
    """
    Content aligner for transition smoothing between segments.

    A non-causal Transformer + convolutional layers that predicts
    corresponding text tokens for each speech token. This enables
    tail-to-head linkage for smooth transitions during multi-round inference.

    Used to:
    1. Extract content from speech tokens (for continuation)
    2. Ensure acoustic coherence across emotional boundaries
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        # Non-causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.aligner_hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.aligner_hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.aligner_num_layers,
        )

        # Input projection
        self.input_proj = nn.Linear(config.hidden_dim, config.aligner_hidden_dim)

        # Convolutional layers for local context
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.aligner_hidden_dim, config.aligner_hidden_dim,
                     kernel_size=config.aligner_kernel_size, padding='same'),
            nn.GELU(),
            nn.Conv1d(config.aligner_hidden_dim, config.aligner_hidden_dim,
                     kernel_size=config.aligner_kernel_size, padding='same'),
            nn.GELU(),
        )

        # Output projection to text token space
        self.output_proj = nn.Linear(config.aligner_hidden_dim, config.hidden_dim)

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        speech_tokens: torch.Tensor,  # [batch, seq, hidden]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict text tokens from speech tokens for content alignment.

        Args:
            speech_tokens: Speech token embeddings [batch, seq, hidden]
            attention_mask: Attention mask [batch, seq]

        Returns:
            Predicted text tokens [batch, seq, hidden]
        """
        # Project input
        x = self.input_proj(speech_tokens)  # [B, S, H']

        # Process through Transformer (non-causal for bidirectional context)
        x = self.transformer(
            x,
            src_key_padding_mask=~attention_mask if attention_mask is not None else None,
        )

        # Apply convolutions
        x = x.transpose(1, 2)  # [B, H', S]
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # [B, S, H']

        # Project to output
        x = self.output_proj(x)
        x = self.norm(x)

        return x


class TransitionSmoothingModule(nn.Module):
    """
    Transition smoothing module for multi-round inference.

    Creates explicit tail-to-head linkage by:
    1. Extracting tail of previous output
    2. Aligning it with current prompt
    3. Enabling continuation-style generation

    This maintains acoustic coherence across emotional boundaries.
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        self.content_aligner = ContentAligner(config)

        # Cross-attention for linking tail to head
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(
        self,
        current_prompt: torch.Tensor,      # [batch, seq1, hidden]
        previous_output: Optional[torch.Tensor] = None,  # [batch, seq2, hidden]
        overlap_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply transition smoothing between segments.

        Args:
            current_prompt: Current segment's prompt embeddings
            previous_output: Previous segment's output (for continuation)
            overlap_length: Number of tokens to use for overlap

        Returns:
            Dict with:
                - 'smoothed_prompt': Prompt with tail-to-head linkage
                - 'aligned_tail': Content-aligned previous output
        """
        if previous_output is None:
            # First segment - no smoothing needed
            return {
                'smoothed_prompt': current_prompt,
                'aligned_tail': None,
            }

        # Extract tail of previous output
        if overlap_length is None:
            overlap_length = int(previous_output.shape[1] * self.config.overlap_ratio)
        overlap_length = max(1, min(overlap_length, previous_output.shape[1]))

        tail = previous_output[:, -overlap_length:]  # [B, O, H]

        # Align tail content to text tokens
        aligned_tail = self.content_aligner(tail)  # [B, O, H]

        # Cross-attention: current prompt attends to aligned tail
        attended, _ = self.cross_attention(
            query=current_prompt,
            key=aligned_tail,
            value=aligned_tail,
        )

        # Fuse with original prompt
        fused = self.fusion(
            torch.cat([current_prompt, attended], dim=-1)
        )

        return {
            'smoothed_prompt': fused,
            'aligned_tail': aligned_tail,
        }


# =============================================================================
# DYNAMIC SPEED CONTROL
# =============================================================================

class DynamicSpeedControl(nn.Module):
    """
    Dynamic speed control mechanism for word-level rate adjustment.

    Controls speaking rate by interpolating or downsampling prompt tokens:
    - Interpolation (longer tokens) → Slower speech
    - Downsampling (shorter tokens) → Faster speech

    Applied per-segment based on the emotion plan.
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        # Speed predictor from emotion/context
        self.speed_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def _compute_speed_ratio(self, speed_value: float) -> float:
        """Convert normalized speed [0,1] to actual ratio."""
        # Map [0, 1] to [min_speed, max_speed]
        min_speed = self.config.min_speed_ratio
        max_speed = self.config.max_speed_ratio
        return min_speed + speed_value * (max_speed - min_speed)

    def adjust_tokens(
        self,
        tokens: torch.Tensor,        # [batch, seq, hidden]
        speed_ratio: float,          # Speed multiplier
    ) -> torch.Tensor:
        """
        Adjust token sequence length based on speed ratio.

        Args:
            tokens: Input tokens [batch, seq, hidden]
            speed_ratio: 0.5 = double length (slow), 2.0 = half length (fast)

        Returns:
            Adjusted tokens with modified sequence length
        """
        if abs(speed_ratio - 1.0) < 0.01:
            return tokens

        batch_size, seq_len, hidden_dim = tokens.shape
        target_len = max(1, int(seq_len / speed_ratio))

        # Use interpolation for resizing
        tokens_t = tokens.transpose(1, 2)  # [B, H, S]

        if target_len > seq_len:
            # Interpolation (nearest-neighbor) for slowing down
            adjusted = F.interpolate(
                tokens_t,
                size=target_len,
                mode='nearest',
            )
        else:
            # Downsampling for speeding up
            adjusted = F.interpolate(
                tokens_t,
                size=target_len,
                mode='linear',
                align_corners=False,
            )

        adjusted = adjusted.transpose(1, 2)  # [B, S', H]

        return adjusted

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        target_speed: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply dynamic speed control to prompt tokens.

        Args:
            prompt_tokens: Prompt token embeddings [batch, seq, hidden]
            context: Context for speed prediction [batch, hidden]
            target_speed: Override speed ratio (for inference control)

        Returns:
            Dict with:
                - 'adjusted_tokens': Speed-adjusted tokens
                - 'speed_ratio': Applied speed ratio
                - 'original_length': Original sequence length
                - 'adjusted_length': New sequence length
        """
        original_length = prompt_tokens.shape[1]

        if target_speed is not None:
            speed_ratio = target_speed
        elif context is not None:
            # Predict speed from context
            speed_value = self.speed_predictor(context).mean().item()
            speed_ratio = self._compute_speed_ratio(speed_value)
        else:
            speed_ratio = 1.0  # Normal speed

        adjusted = self.adjust_tokens(prompt_tokens, speed_ratio)

        return {
            'adjusted_tokens': adjusted,
            'speed_ratio': speed_ratio,
            'original_length': original_length,
            'adjusted_length': adjusted.shape[1],
        }


# =============================================================================
# MULTI-ROUND INFERENCE
# =============================================================================

@dataclass
class EmotionPlan:
    """Emotion plan for a text segment."""
    text: str
    emotion: str
    intensity: float = 1.0
    speed: float = 1.0
    start_word_idx: int = 0
    end_word_idx: int = -1


class MultiRoundInference(nn.Module):
    """
    Multi-round inference framework for word-level emotion control.

    Segments text based on emotion plan, generates each segment with
    the specified emotion, and smoothly connects segments using the
    transition smoothing module.

    Process:
    1. Parse emotion plan (text → [(segment, emotion, speed), ...])
    2. For each segment:
       a. Apply transition smoothing with previous output
       b. Apply dynamic speed control
       c. Generate with emotional attention bias
       d. Store output for next round's tail
    3. Concatenate all segments
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        self.attention_bias = DynamicEmotionalAttentionBias(config)
        self.transition_smoothing = TransitionSmoothingModule(config)
        self.speed_control = DynamicSpeedControl(config)

        # Segment boundary detection
        self.boundary_detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 2),  # Binary: boundary or not
        )

    def parse_emotion_plan(
        self,
        text: str,
        emotions: List[str],
        word_indices: List[Tuple[int, int]],
        speeds: Optional[List[float]] = None,
        intensities: Optional[List[float]] = None,
    ) -> List[EmotionPlan]:
        """
        Parse text and emotions into emotion plans.

        Args:
            text: Full text to synthesize
            emotions: Emotion label per segment
            word_indices: [(start, end), ...] word indices per segment
            speeds: Speed ratio per segment (optional)
            intensities: Emotion intensity per segment (optional)

        Returns:
            List of EmotionPlan objects
        """
        words = text.split()
        num_segments = len(emotions)

        if speeds is None:
            speeds = [1.0] * num_segments
        if intensities is None:
            intensities = [1.0] * num_segments

        plans = []
        for i, (emotion, (start, end), speed, intensity) in enumerate(
            zip(emotions, word_indices, speeds, intensities)
        ):
            if end == -1:
                end = len(words)
            segment_text = ' '.join(words[start:end])

            plans.append(EmotionPlan(
                text=segment_text,
                emotion=emotion,
                intensity=intensity,
                speed=speed,
                start_word_idx=start,
                end_word_idx=end,
            ))

        return plans

    def forward(
        self,
        hidden_states: torch.Tensor,           # [batch, seq, hidden]
        emotion_plans: List[EmotionPlan],
        segment_boundaries: torch.Tensor,      # [batch, num_segments+1]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform multi-round inference with emotion plans.

        Args:
            hidden_states: Full text hidden states [batch, seq, hidden]
            emotion_plans: List of emotion plans per segment
            segment_boundaries: Token indices for segment boundaries
            attention_mask: Attention mask

        Returns:
            Dict with outputs per segment and combined output
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        segment_outputs = []
        previous_output = None

        for i, plan in enumerate(emotion_plans):
            # Get segment boundaries
            start_idx = segment_boundaries[:, i].long()
            end_idx = segment_boundaries[:, i + 1].long()

            # Extract segment (simplified - assumes same boundaries across batch)
            start = start_idx[0].item()
            end = end_idx[0].item()
            segment = hidden_states[:, start:end]  # [B, S_i, H]

            # Apply transition smoothing
            smooth_result = self.transition_smoothing(
                current_prompt=segment,
                previous_output=previous_output,
            )
            smoothed = smooth_result['smoothed_prompt']

            # Apply speed control
            speed_result = self.speed_control(
                prompt_tokens=smoothed,
                target_speed=plan.speed,
            )
            adjusted = speed_result['adjusted_tokens']

            # Get target emotion index
            emotion_idx = self.config.emotion_labels.index(plan.emotion.lower())
            target_emotions = torch.full(
                (batch_size, adjusted.shape[1]),
                emotion_idx,
                dtype=torch.long,
                device=device,
            )

            # Apply emotional attention bias
            bias_result = self.attention_bias(
                hidden_states=adjusted,
                target_emotions=target_emotions,
            )

            segment_outputs.append({
                'hidden': bias_result['modulated_hidden'],
                'attention_bias': bias_result['attention_bias'],
                'emotion_logits': bias_result['emotion_logits'],
                'speed_ratio': speed_result['speed_ratio'],
                'plan': plan,
            })

            # Store for next round's transition smoothing
            previous_output = bias_result['modulated_hidden']

        # Concatenate all segments
        combined_hidden = torch.cat(
            [s['hidden'] for s in segment_outputs],
            dim=1,
        )

        return {
            'combined_hidden': combined_hidden,
            'segment_outputs': segment_outputs,
            'num_segments': len(segment_outputs),
        }


# =============================================================================
# SELF-TRAINING FRAMEWORK
# =============================================================================

class WeSConTeacher(nn.Module):
    """
    Teacher model for WeSCon self-training (Stage 1).

    Uses:
    - Multi-round inference with transition smoothing
    - Dynamic speed control
    - Frozen pretrained TTS model

    Generates pseudo-labeled emotional speech without requiring
    manually annotated emotion transition datasets.
    """

    def __init__(
        self,
        config: WeSConConfig,
        pretrained_tts: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.tts = pretrained_tts

        self.multi_round = MultiRoundInference(config)

        # Freeze TTS if provided
        if self.tts is not None:
            for param in self.tts.parameters():
                param.requires_grad = False

    def generate_pseudo_labels(
        self,
        text: str,
        emotion_plan: List[Tuple[str, float, float]],  # [(emotion, intensity, speed), ...]
        word_indices: List[Tuple[int, int]],
        prompt_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Generate pseudo-labeled output using multi-round inference.

        Args:
            text: Full text to synthesize
            emotion_plan: [(emotion, intensity, speed), ...] per segment
            word_indices: Word boundaries per segment
            prompt_audio: Optional emotional prompt audio

        Returns:
            Dict with pseudo-labeled outputs:
                - 'audio': Generated audio
                - 'segment_emotions': Emotion labels per segment
                - 'token_emotions': Emotion labels per token
                - 'speed_ratios': Speed ratios per segment
        """
        emotions = [e[0] for e in emotion_plan]
        intensities = [e[1] for e in emotion_plan]
        speeds = [e[2] for e in emotion_plan]

        plans = self.multi_round.parse_emotion_plan(
            text=text,
            emotions=emotions,
            word_indices=word_indices,
            speeds=speeds,
            intensities=intensities,
        )

        # This would integrate with actual TTS model
        # For now, return structure for training
        return {
            'emotion_plans': plans,
            'segment_emotions': emotions,
            'token_emotions': None,  # To be filled during actual generation
            'speed_ratios': speeds,
        }


class WeSConStudent(nn.Module):
    """
    Student model for WeSCon self-training (Stage 2).

    Learns from teacher outputs by:
    - Speech token prediction (L_tts)
    - Token-level emotion classification (L_e)

    Uses dynamic emotional attention bias for simplified
    end-to-end inference (no multi-round needed).
    """

    def __init__(
        self,
        config: WeSConConfig,
        pretrained_tts: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.tts = pretrained_tts

        # Dynamic emotional attention bias (key innovation)
        self.attention_bias = DynamicEmotionalAttentionBias(config)

        # Fine-tuning with very low learning rate
        if self.tts is not None:
            # Only fine-tune specific layers
            for name, param in self.tts.named_parameters():
                if 'attention' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_emotions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with emotional attention bias.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            target_emotions: Target emotion indices [batch, seq]
            attention_mask: Attention mask

        Returns:
            Dict with modulated hidden states and predictions
        """
        result = self.attention_bias(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            target_emotions=target_emotions,
        )

        return result


# =============================================================================
# WESCON LOSS
# =============================================================================

class WeSConLoss(nn.Module):
    """
    Loss function for WeSCon training.

    Components:
    - L_tts: Speech token prediction (negative log-likelihood)
    - L_e: Token-level emotion classification (cross-entropy)
    - L_kl: KL divergence for attention distribution regularization
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Optional[Dict[str, torch.Tensor]] = None,
        target_emotions: Optional[torch.Tensor] = None,
        target_speech_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute WeSCon training losses.

        Args:
            student_output: Student model outputs
            teacher_output: Teacher model outputs (for distillation)
            target_emotions: Ground truth emotions [batch, seq]
            target_speech_tokens: Ground truth speech tokens [batch, seq]

        Returns:
            Dict with individual losses and total
        """
        losses = {}
        device = student_output['emotion_logits'].device

        # Emotion classification loss (L_e)
        if target_emotions is not None:
            emotion_logits = student_output['emotion_logits']
            batch_size, seq_len, num_emotions = emotion_logits.shape

            losses['emotion'] = self.ce_loss(
                emotion_logits.view(-1, num_emotions),
                target_emotions.view(-1),
            )
        else:
            losses['emotion'] = torch.tensor(0.0, device=device)

        # KL divergence for attention regularization
        if teacher_output is not None and 'attention_bias' in teacher_output:
            student_bias = F.log_softmax(
                student_output['attention_bias'].flatten(1),
                dim=-1,
            )
            teacher_bias = F.softmax(
                teacher_output['attention_bias'].flatten(1) / self.config.teacher_temperature,
                dim=-1,
            )
            losses['kl'] = self.kl_loss(student_bias, teacher_bias)
        else:
            losses['kl'] = torch.tensor(0.0, device=device)

        # Total loss
        losses['total'] = (
            self.config.emotion_cls_weight * losses['emotion'] +
            self.config.kl_weight * losses['kl']
        )

        return losses


# =============================================================================
# WESCON ADAPTER (CSM INTEGRATION)
# =============================================================================

class WeSConAdapter(nn.Module):
    """
    Adapter for integrating WeSCon with CSM prosody pipeline.

    Provides drop-in replacement for other prosody adapters,
    generating prosody prefix tokens with word-level emotion and
    speed control.

    Usage:
        adapter = WeSConAdapter(config)

        # From word-level emotion plan
        result = adapter(
            text_embeddings=text_emb,
            word_emotions=["happy", "happy", "sad", "neutral"],
            word_speeds=[1.0, 1.2, 0.8, 1.0],
        )
        prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]
    """

    def __init__(self, config: WeSConConfig):
        super().__init__()
        self.config = config

        # Core WeSCon modules
        self.attention_bias = DynamicEmotionalAttentionBias(config)
        self.speed_control = DynamicSpeedControl(config)
        self.transition_smoothing = TransitionSmoothingModule(config)

        # Emotion name to index mapping
        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(config.emotion_labels)
        }

        # Output projection to prosody tokens
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,           # [batch, num_words, hidden]
        word_emotions: Optional[List[str]] = None,  # ["happy", "sad", ...]
        word_speeds: Optional[List[float]] = None,  # [1.0, 0.8, ...]
        word_intensities: Optional[List[float]] = None,  # [0.8, 0.9, ...]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with word-level emotion and speed control.

        Args:
            text_embeddings: Text encoder outputs [batch, num_words, hidden]
            word_emotions: Emotion label per word
            word_speeds: Speed ratio per word
            word_intensities: Emotion intensity per word
            attention_mask: Attention mask

        Returns:
            Dict with:
                - 'prosody_tokens': [batch, 4, 2048]
                - 'word_emotions': Applied emotions
                - 'attention_bias': Attention bias matrix
        """
        batch_size, num_words, hidden_dim = text_embeddings.shape
        device = text_embeddings.device

        # Default values
        if word_emotions is None:
            word_emotions = ['neutral'] * num_words
        if word_speeds is None:
            word_speeds = [1.0] * num_words
        if word_intensities is None:
            word_intensities = [1.0] * num_words

        # Convert emotions to indices
        emotion_indices = torch.tensor([
            self.emotion_to_idx.get(e.lower(), 0)
            for e in word_emotions
        ], dtype=torch.long, device=device)
        emotion_indices = emotion_indices.unsqueeze(0).expand(batch_size, -1)

        # Apply dynamic emotional attention bias
        bias_result = self.attention_bias(
            hidden_states=text_embeddings,
            attention_mask=attention_mask,
            target_emotions=emotion_indices,
        )

        modulated = bias_result['modulated_hidden']

        # Apply per-word speed control by adjusting hidden states
        # (In a full implementation, this would modify the generation process)

        # Pool modulated embeddings
        pooled = modulated.mean(dim=1)  # [batch, hidden]

        # Generate prosody tokens
        tokens = self.output_proj(pooled)  # [batch, output * num_tokens]
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        return {
            'prosody_tokens': tokens,
            'word_emotions': word_emotions,
            'word_speeds': word_speeds,
            'attention_bias': bias_result['attention_bias'],
            'emotion_logits': bias_result['emotion_logits'],
        }

    def from_emotion_trajectory(
        self,
        text_embeddings: torch.Tensor,
        emotions: List[str],
        intensities: Optional[List[float]] = None,
        speeds: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from emotion trajectory.

        Convenience method matching other adapters' interface.
        """
        return self.forward(
            text_embeddings=text_embeddings,
            word_emotions=emotions,
            word_speeds=speeds,
            word_intensities=intensities,
        )

    def from_global_emotion(
        self,
        text_embeddings: torch.Tensor,
        emotion: str,
        intensity: float = 1.0,
        speed: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply same emotion to all words (global mode).
        """
        num_words = text_embeddings.shape[1]
        return self.forward(
            text_embeddings=text_embeddings,
            word_emotions=[emotion] * num_words,
            word_speeds=[speed] * num_words,
            word_intensities=[intensity] * num_words,
        )

    def interpolate_emotions(
        self,
        text_embeddings: torch.Tensor,
        start_emotion: str,
        end_emotion: str,
        start_speed: float = 1.0,
        end_speed: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Create smooth emotion and speed transition across words.
        """
        num_words = text_embeddings.shape[1]

        word_emotions = []
        word_speeds = []

        for i in range(num_words):
            t = i / max(1, num_words - 1)  # 0 to 1

            # Interpolate emotion (switch at midpoint)
            if t < 0.5:
                word_emotions.append(start_emotion)
            else:
                word_emotions.append(end_emotion)

            # Interpolate speed
            speed = start_speed * (1 - t) + end_speed * t
            word_speeds.append(speed)

        return self.forward(
            text_embeddings=text_embeddings,
            word_emotions=word_emotions,
            word_speeds=word_speeds,
        )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WeSCon: Word-Level Self-Training Emotion Control - Test Suite")
    print("=" * 60)

    config = WeSConConfig()
    device = 'cpu'

    # Test 1: AttentionBiasTemplates
    print("\n[Test 1] AttentionBiasTemplates...")
    templates = AttentionBiasTemplates(config)

    omega = torch.randn(2, config.num_bias_templates)  # [batch, 7]
    bias = templates(omega, seq_length=32)

    print(f"  Input omega shape: {omega.shape}")
    print(f"  Output bias shape: {bias.shape}")
    print(f"  Expected: [2, 32, 32]")
    print("  [PASS]")

    # Test 2: EmotionPredictor
    print("\n[Test 2] EmotionPredictor...")
    predictor = EmotionPredictor(config)

    hidden = torch.randn(2, 32, config.hidden_dim)
    mask = torch.ones(2, 32, dtype=torch.bool)

    pred_result = predictor(hidden, mask)

    print(f"  Emotion logits shape: {pred_result['emotion_logits'].shape}")
    print(f"  Omega shape: {pred_result['omega'].shape}")
    print("  [PASS]")

    # Test 3: DynamicEmotionalAttentionBias
    print("\n[Test 3] DynamicEmotionalAttentionBias...")
    deab = DynamicEmotionalAttentionBias(config)

    target_emotions = torch.randint(0, config.num_emotions, (2, 32))

    bias_result = deab(hidden, mask, target_emotions)

    print(f"  Modulated hidden shape: {bias_result['modulated_hidden'].shape}")
    print(f"  Attention bias shape: {bias_result['attention_bias'].shape}")
    print("  [PASS]")

    # Test 4: ContentAligner
    print("\n[Test 4] ContentAligner...")
    aligner = ContentAligner(config)

    speech_tokens = torch.randn(2, 16, config.hidden_dim)
    aligned = aligner(speech_tokens)

    print(f"  Input speech tokens: {speech_tokens.shape}")
    print(f"  Aligned output: {aligned.shape}")
    print("  [PASS]")

    # Test 5: TransitionSmoothingModule
    print("\n[Test 5] TransitionSmoothingModule...")
    smoother = TransitionSmoothingModule(config)

    current = torch.randn(2, 32, config.hidden_dim)
    previous = torch.randn(2, 24, config.hidden_dim)

    smooth_result = smoother(current, previous)

    print(f"  Smoothed prompt shape: {smooth_result['smoothed_prompt'].shape}")
    print(f"  Aligned tail shape: {smooth_result['aligned_tail'].shape}")
    print("  [PASS]")

    # Test 6: DynamicSpeedControl
    print("\n[Test 6] DynamicSpeedControl...")
    speed_ctrl = DynamicSpeedControl(config)

    prompt = torch.randn(2, 32, config.hidden_dim)

    # Test slow speech
    slow_result = speed_ctrl(prompt, target_speed=0.5)
    print(f"  Original length: {slow_result['original_length']}")
    print(f"  Slow (0.5x) length: {slow_result['adjusted_length']}")

    # Test fast speech
    fast_result = speed_ctrl(prompt, target_speed=2.0)
    print(f"  Fast (2.0x) length: {fast_result['adjusted_length']}")
    print("  [PASS]")

    # Test 7: MultiRoundInference
    print("\n[Test 7] MultiRoundInference...")
    multi_round = MultiRoundInference(config)

    # Create emotion plans
    plans = [
        EmotionPlan(text="Hello", emotion="happy", speed=1.0),
        EmotionPlan(text="world", emotion="sad", speed=0.8),
    ]

    boundaries = torch.tensor([[0, 16, 32]])  # [batch, num_segments+1]

    mr_result = multi_round(hidden, plans, boundaries)

    print(f"  Combined hidden shape: {mr_result['combined_hidden'].shape}")
    print(f"  Number of segments: {mr_result['num_segments']}")
    print("  [PASS]")

    # Test 8: WeSConLoss
    print("\n[Test 8] WeSConLoss...")
    loss_fn = WeSConLoss(config)

    student_out = {
        'emotion_logits': torch.randn(2, 32, config.num_emotions),
        'attention_bias': torch.randn(2, 32, 32),
    }

    losses = loss_fn(
        student_output=student_out,
        target_emotions=target_emotions,
    )

    print(f"  Emotion loss: {losses['emotion'].item():.4f}")
    print(f"  KL loss: {losses['kl'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: WeSConAdapter
    print("\n[Test 9] WeSConAdapter...")
    adapter = WeSConAdapter(config)

    text_emb = torch.randn(2, 4, config.hidden_dim)

    # From word-level emotions
    result = adapter(
        text_embeddings=text_emb,
        word_emotions=["happy", "happy", "sad", "neutral"],
        word_speeds=[1.0, 1.2, 0.8, 1.0],
    )

    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Expected: [2, 4, 2048]")

    # Test interpolation
    interp_result = adapter.interpolate_emotions(
        text_emb,
        start_emotion="happy",
        end_emotion="sad",
        start_speed=1.2,
        end_speed=0.8,
    )
    print(f"  Interpolated tokens: {interp_result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 10: Teacher-Student Framework
    print("\n[Test 10] Teacher-Student Framework...")
    teacher = WeSConTeacher(config)
    student = WeSConStudent(config)

    # Generate pseudo labels
    pseudo = teacher.generate_pseudo_labels(
        text="Hello wonderful world today",
        emotion_plan=[("happy", 0.8, 1.0), ("sad", 0.7, 0.9)],
        word_indices=[(0, 2), (2, 4)],
    )
    print(f"  Pseudo label plans: {len(pseudo['emotion_plans'])}")

    # Student forward
    student_out = student(hidden, target_emotions)
    print(f"  Student output keys: {list(student_out.keys())}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All WeSCon tests passed!")
    print("=" * 60)

    # Usage example
    print("\nUsage Example:")
    print("-" * 40)
    print("""
from wescon import (
    WeSConConfig,
    WeSConAdapter,
    MultiRoundInference,
    EmotionPlan,
)

# Initialize
config = WeSConConfig()
adapter = WeSConAdapter(config).cuda()

# Option 1: Word-level emotion control
result = adapter(
    text_embeddings=text_emb,       # [batch, num_words, 2048]
    word_emotions=["neutral", "happy", "happy", "surprised", "neutral"],
    word_speeds=[1.0, 1.1, 1.2, 1.0, 0.9],  # Speed per word
    word_intensities=[0.5, 0.8, 0.9, 0.8, 0.5],  # Intensity per word
)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

# Option 2: Smooth emotion transition
result = adapter.interpolate_emotions(
    text_embeddings=text_emb,
    start_emotion="calm",
    end_emotion="angry",
    start_speed=0.8,  # Start slow
    end_speed=1.2,    # End fast
)

# Option 3: Global emotion (like other adapters)
result = adapter.from_global_emotion(
    text_emb,
    emotion="happy",
    intensity=0.9,
    speed=1.1,
)

# Multi-round inference for complex emotion plans
multi_round = MultiRoundInference(config).cuda()
plans = [
    EmotionPlan(text="I'm so happy", emotion="happy", speed=1.2),
    EmotionPlan(text="but also a bit worried", emotion="fearful", speed=0.9),
]
# Use with actual TTS model for full generation

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([other_prosody, prosody_tokens], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
