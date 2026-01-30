"""
Chatterbox-Style Emotion Exaggeration with Paralinguistic Tags

Based on Chatterbox by Resemble AI (December 2025):
"State-of-the-Art Open-Source TTS with Emotion Exaggeration and Paralinguistic Prompting"

Key Innovations:
1. Single-parameter emotion exaggeration (0.0=monotone → 2.0=dramatic)
   - Default 0.5 works well for most prompts
   - Higher exaggeration speeds up speech; lower cfg_weight compensates

2. Native paralinguistic tags in text prompts:
   - [laugh], [chuckle], [sigh], [cough], [gasp], [hmm], [uh], [um]
   - Model performs these reactions naturally in cloned voice
   - Same emotional tone as surrounding speech
   - No post-processing, no splicing, no manual editing

3. Streamlined architecture (Chatterbox Turbo):
   - 350M parameters (vs 500M original)
   - Speech-token-to-mel decoder distilled to single step
   - Up to 6x faster than real-time on GPU

4. CFG weight control:
   - Lower cfg_weight (0.3) with higher exaggeration for expressive speech
   - Higher cfg_weight for more controlled, deliberate pacing

GitHub: https://github.com/resemble-ai/chatterbox
HuggingFace: ResembleAI/chatterbox-turbo
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChatterboxConfig:
    """Configuration for Chatterbox-style emotion exaggeration."""

    # Model dimensions
    input_dim: int = 768  # Input feature dimension (wav2vec2/HuBERT)
    hidden_dim: int = 512  # Hidden layer dimension
    emotion_dim: int = 256  # Emotion embedding dimension
    output_dim: int = 2048  # Output to match prosody encoder (CSM)

    # Emotion exaggeration
    default_exaggeration: float = 0.5  # Default emotion exaggeration level
    min_exaggeration: float = 0.0  # Monotone
    max_exaggeration: float = 2.0  # Dramatic

    # CFG (Classifier-Free Guidance) weight
    default_cfg_weight: float = 0.5
    min_cfg_weight: float = 0.0
    max_cfg_weight: float = 1.0

    # Paralinguistic tag settings
    tag_embedding_dim: int = 128  # Dimension for tag embeddings
    max_tags_per_utterance: int = 10  # Maximum tags allowed

    # Single-step mel decoder (distilled)
    use_single_step_decoder: bool = True
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8

    # Speaker encoder
    speaker_dim: int = 256

    # Training settings
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate


# =============================================================================
# PARALINGUISTIC TAGS
# =============================================================================

# Supported paralinguistic tags and their semantic categories
PARALINGUISTIC_TAGS = {
    # Laughter family
    "laugh": {"category": "laughter", "intensity": 1.0, "duration_frames": 20},
    "chuckle": {"category": "laughter", "intensity": 0.5, "duration_frames": 10},
    "giggle": {"category": "laughter", "intensity": 0.3, "duration_frames": 8},
    "snicker": {"category": "laughter", "intensity": 0.4, "duration_frames": 8},

    # Breathing family
    "sigh": {"category": "breathing", "intensity": 0.6, "duration_frames": 15},
    "gasp": {"category": "breathing", "intensity": 0.8, "duration_frames": 8},
    "exhale": {"category": "breathing", "intensity": 0.4, "duration_frames": 10},
    "inhale": {"category": "breathing", "intensity": 0.3, "duration_frames": 8},
    "yawn": {"category": "breathing", "intensity": 0.5, "duration_frames": 25},

    # Throat sounds
    "cough": {"category": "throat", "intensity": 0.7, "duration_frames": 12},
    "clear_throat": {"category": "throat", "intensity": 0.5, "duration_frames": 8},
    "sniff": {"category": "throat", "intensity": 0.3, "duration_frames": 5},

    # Hesitation/filler sounds
    "hmm": {"category": "filler", "intensity": 0.3, "duration_frames": 10},
    "uh": {"category": "filler", "intensity": 0.2, "duration_frames": 5},
    "um": {"category": "filler", "intensity": 0.2, "duration_frames": 6},
    "er": {"category": "filler", "intensity": 0.2, "duration_frames": 4},
    "ah": {"category": "filler", "intensity": 0.3, "duration_frames": 5},

    # Emotional reactions
    "sob": {"category": "emotion", "intensity": 0.9, "duration_frames": 15},
    "cry": {"category": "emotion", "intensity": 1.0, "duration_frames": 20},
    "whimper": {"category": "emotion", "intensity": 0.6, "duration_frames": 10},
    "groan": {"category": "emotion", "intensity": 0.7, "duration_frames": 12},
    "moan": {"category": "emotion", "intensity": 0.6, "duration_frames": 10},

    # Surprise/attention
    "oh": {"category": "surprise", "intensity": 0.5, "duration_frames": 5},
    "wow": {"category": "surprise", "intensity": 0.7, "duration_frames": 8},
    "huh": {"category": "surprise", "intensity": 0.4, "duration_frames": 5},

    # Affirmation/negation
    "mmhmm": {"category": "affirm", "intensity": 0.3, "duration_frames": 8},
    "uh_huh": {"category": "affirm", "intensity": 0.3, "duration_frames": 6},
    "nuh_uh": {"category": "negate", "intensity": 0.4, "duration_frames": 8},
    "tsk": {"category": "negate", "intensity": 0.5, "duration_frames": 4},
}

TAG_CATEGORIES = list(set(tag["category"] for tag in PARALINGUISTIC_TAGS.values()))
TAG_TO_IDX = {tag: i for i, tag in enumerate(PARALINGUISTIC_TAGS.keys())}
IDX_TO_TAG = {i: tag for tag, i in TAG_TO_IDX.items()}

# Regex pattern for extracting tags from text
TAG_PATTERN = re.compile(r'\[([a-zA-Z_]+)\]')


# =============================================================================
# TAG PARSER
# =============================================================================

@dataclass
class ParsedTag:
    """Represents a parsed paralinguistic tag from text."""
    tag_name: str
    position: int  # Character position in original text
    word_index: int  # Word index (for alignment)
    category: str
    intensity: float
    duration_frames: int


def parse_paralinguistic_tags(text: str) -> Tuple[str, List[ParsedTag]]:
    """
    Parse paralinguistic tags from text.

    Args:
        text: Input text with tags like "[laugh]", "[sigh]", etc.

    Returns:
        Tuple of (clean_text, list_of_parsed_tags)

    Example:
        >>> text = "Hi there, Sarah here [chuckle], have you got one minute?"
        >>> clean, tags = parse_paralinguistic_tags(text)
        >>> clean
        'Hi there, Sarah here , have you got one minute?'
        >>> tags[0].tag_name
        'chuckle'
    """
    tags = []
    matches = list(TAG_PATTERN.finditer(text))

    # Compute word index for each tag
    for match in matches:
        tag_name = match.group(1).lower()

        if tag_name not in PARALINGUISTIC_TAGS:
            continue

        tag_info = PARALINGUISTIC_TAGS[tag_name]

        # Count words before this position
        text_before = text[:match.start()]
        word_index = len(text_before.split())

        tags.append(ParsedTag(
            tag_name=tag_name,
            position=match.start(),
            word_index=word_index,
            category=tag_info["category"],
            intensity=tag_info["intensity"],
            duration_frames=tag_info["duration_frames"],
        ))

    # Remove tags from text
    clean_text = TAG_PATTERN.sub('', text)
    # Clean up extra spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text, tags


def get_supported_tags() -> List[str]:
    """Get list of all supported paralinguistic tags."""
    return list(PARALINGUISTIC_TAGS.keys())


def describe_tag(tag_name: str) -> str:
    """Get description of a paralinguistic tag."""
    if tag_name not in PARALINGUISTIC_TAGS:
        return f"Unknown tag: {tag_name}"

    info = PARALINGUISTIC_TAGS[tag_name]
    return (
        f"[{tag_name}]: {info['category']} sound, "
        f"intensity={info['intensity']:.1f}, "
        f"~{info['duration_frames'] * 10}ms duration"
    )


# =============================================================================
# PARALINGUISTIC TAG ENCODER
# =============================================================================

class ParalinguisticTagEncoder(nn.Module):
    """
    Encodes paralinguistic tags into embeddings.

    Each tag is encoded as:
    1. Tag embedding (learned per tag type)
    2. Category embedding (learned per category)
    3. Intensity modulation
    4. Position encoding (relative to word boundary)
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        num_tags = len(PARALINGUISTIC_TAGS)
        num_categories = len(TAG_CATEGORIES)

        # Tag embeddings
        self.tag_embedding = nn.Embedding(num_tags, config.tag_embedding_dim)

        # Category embeddings
        self.category_embedding = nn.Embedding(num_categories, config.tag_embedding_dim // 2)
        self.category_to_idx = {cat: i for i, cat in enumerate(TAG_CATEGORIES)}

        # Intensity projection
        self.intensity_proj = nn.Sequential(
            nn.Linear(1, config.tag_embedding_dim // 4),
            nn.GELU(),
        )

        # Duration encoding
        self.duration_proj = nn.Sequential(
            nn.Linear(1, config.tag_embedding_dim // 4),
            nn.GELU(),
        )

        # Combine all features
        combined_dim = (
            config.tag_embedding_dim +  # tag
            config.tag_embedding_dim // 2 +  # category
            config.tag_embedding_dim // 4 +  # intensity
            config.tag_embedding_dim // 4  # duration
        )

        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, config.tag_embedding_dim),
            nn.LayerNorm(config.tag_embedding_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        parsed_tags: List[ParsedTag],
        batch_size: int = 1,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode parsed tags into embeddings.

        Args:
            parsed_tags: List of ParsedTag objects
            batch_size: Batch size for output

        Returns:
            tag_embeddings: [batch, max_tags, tag_embedding_dim]
            tag_mask: [batch, max_tags] boolean mask
        """
        device = self.tag_embedding.weight.device
        max_tags = self.config.max_tags_per_utterance

        # Initialize outputs
        embeddings = torch.zeros(batch_size, max_tags, self.config.tag_embedding_dim, device=device)
        mask = torch.zeros(batch_size, max_tags, dtype=torch.bool, device=device)

        for i, tag in enumerate(parsed_tags[:max_tags]):
            # Tag embedding
            tag_idx = TAG_TO_IDX[tag.tag_name]
            tag_emb = self.tag_embedding(torch.tensor(tag_idx, device=device))

            # Category embedding
            cat_idx = self.category_to_idx[tag.category]
            cat_emb = self.category_embedding(torch.tensor(cat_idx, device=device))

            # Intensity
            intensity = torch.tensor([[tag.intensity]], device=device)
            int_emb = self.intensity_proj(intensity).squeeze(0)

            # Duration
            duration = torch.tensor([[tag.duration_frames / 30.0]], device=device)  # Normalize
            dur_emb = self.duration_proj(duration).squeeze(0)

            # Combine
            combined = torch.cat([tag_emb, cat_emb, int_emb, dur_emb], dim=-1)
            out_emb = self.output_proj(combined)

            # Store (replicate across batch)
            embeddings[:, i] = out_emb.unsqueeze(0).expand(batch_size, -1)
            mask[:, i] = True

        return embeddings, mask

    def encode_tag_name(self, tag_name: str) -> Tensor:
        """Encode a single tag by name."""
        if tag_name not in PARALINGUISTIC_TAGS:
            raise ValueError(f"Unknown tag: {tag_name}")

        tag_info = PARALINGUISTIC_TAGS[tag_name]
        parsed = ParsedTag(
            tag_name=tag_name,
            position=0,
            word_index=0,
            category=tag_info["category"],
            intensity=tag_info["intensity"],
            duration_frames=tag_info["duration_frames"],
        )

        embeddings, _ = self.forward([parsed], batch_size=1)
        return embeddings[:, 0]  # [1, tag_embedding_dim]


# =============================================================================
# EMOTION EXAGGERATION MODULE
# =============================================================================

class EmotionExaggerationModule(nn.Module):
    """
    Controls emotion exaggeration level from monotone to dramatic.

    Exaggeration parameter:
        - 0.0: Monotone/flat (no emotional expression)
        - 0.5: Default/balanced (natural expression)
        - 1.0: Enhanced (clear emotional expression)
        - 1.5-2.0: Dramatic (highly expressive, may speed up speech)

    Higher exaggeration tends to speed up speech. To compensate,
    reduce cfg_weight for slower, more deliberate pacing.
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # Exaggeration scale network
        self.scale_net = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.emotion_dim),
        )

        # Learnable emotion prototypes (neutral baseline)
        self.neutral_prototype = nn.Parameter(torch.zeros(config.emotion_dim))

        # Emotion variance controller
        self.variance_net = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emotion_dim),
            nn.Sigmoid(),  # Output scale factors [0, 1]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

    def forward(
        self,
        emotion_features: Tensor,  # [batch, emotion_dim]
        exaggeration: float = None,
    ) -> Dict[str, Tensor]:
        """
        Apply emotion exaggeration.

        Args:
            emotion_features: Input emotion features
            exaggeration: Exaggeration level (0.0 to 2.0)

        Returns:
            Dictionary with exaggerated features and info
        """
        if exaggeration is None:
            exaggeration = self.config.default_exaggeration

        # Clamp exaggeration
        exaggeration = max(self.config.min_exaggeration,
                         min(self.config.max_exaggeration, exaggeration))

        batch_size = emotion_features.size(0)
        device = emotion_features.device

        # Compute exaggeration scaling
        exag_input = torch.tensor([[exaggeration]], device=device)
        scale_factors = self.scale_net(exag_input)  # [1, emotion_dim]
        scale_factors = scale_factors.expand(batch_size, -1)

        # Compute variance modulation
        variance_scale = self.variance_net(emotion_features)  # [batch, emotion_dim]

        # Distance from neutral
        neutral = self.neutral_prototype.unsqueeze(0).expand(batch_size, -1)
        deviation = emotion_features - neutral

        # Apply exaggeration: scale the deviation from neutral
        # exaggeration=0 -> stay at neutral
        # exaggeration=0.5 -> normal emotion
        # exaggeration>1 -> amplify emotion
        exaggeration_factor = exaggeration * 2.0  # Map 0.5 -> 1.0 (neutral)

        exaggerated = neutral + deviation * exaggeration_factor * variance_scale

        # Apply scale factors for fine-tuning
        exaggerated = exaggerated + scale_factors * (exaggeration - 0.5)

        # Final projection
        output = self.output_proj(exaggerated)

        return {
            "exaggerated_features": output,
            "exaggeration_level": exaggeration,
            "scale_factors": scale_factors,
            "variance_scale": variance_scale,
        }

    def interpolate_exaggeration(
        self,
        emotion_features: Tensor,
        start_exag: float,
        end_exag: float,
        num_steps: int = 5,
    ) -> List[Tensor]:
        """Generate features at interpolated exaggeration levels."""
        results = []
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0
            exag = start_exag + t * (end_exag - start_exag)
            result = self.forward(emotion_features, exaggeration=exag)
            results.append(result["exaggerated_features"])
        return results


# =============================================================================
# CFG (CLASSIFIER-FREE GUIDANCE) MODULE
# =============================================================================

class CFGModule(nn.Module):
    """
    Classifier-Free Guidance for controlling generation.

    CFG weight controls the balance between:
    - Unconditional generation (more diverse, less controlled)
    - Conditional generation (more controlled, follows prompt)

    Interaction with exaggeration:
    - Higher exaggeration + lower cfg_weight = expressive but deliberate
    - Lower exaggeration + higher cfg_weight = controlled and natural
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # CFG scale projection
        self.cfg_proj = nn.Sequential(
            nn.Linear(config.emotion_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

    def forward(
        self,
        conditional: Tensor,  # [batch, emotion_dim]
        unconditional: Tensor,  # [batch, emotion_dim]
        cfg_weight: float = None,
    ) -> Tensor:
        """
        Apply classifier-free guidance.

        Formula: output = unconditional + cfg_weight * (conditional - unconditional)

        Args:
            conditional: Conditioned features
            unconditional: Unconditioned features
            cfg_weight: CFG weight (0.0 to 1.0)

        Returns:
            CFG-weighted output features
        """
        if cfg_weight is None:
            cfg_weight = self.config.default_cfg_weight

        cfg_weight = max(self.config.min_cfg_weight,
                        min(self.config.max_cfg_weight, cfg_weight))

        # Standard CFG formula
        output = unconditional + cfg_weight * (conditional - unconditional)

        return output

    def adaptive_cfg(
        self,
        conditional: Tensor,
        unconditional: Tensor,
        exaggeration: float,
    ) -> Tensor:
        """
        Adaptive CFG that adjusts weight based on exaggeration level.

        Higher exaggeration -> lower cfg_weight for natural pacing.
        """
        # Adaptive formula: cfg = 0.7 - 0.3 * (exaggeration - 0.5)
        # exaggeration=0.5 -> cfg=0.7
        # exaggeration=1.0 -> cfg=0.55
        # exaggeration=1.5 -> cfg=0.4
        adaptive_weight = 0.7 - 0.3 * (exaggeration - 0.5)
        adaptive_weight = max(0.3, min(0.9, adaptive_weight))

        return self.forward(conditional, unconditional, adaptive_weight)


# =============================================================================
# SINGLE-STEP MEL DECODER (DISTILLED)
# =============================================================================

class SingleStepMelDecoder(nn.Module):
    """
    Distilled single-step speech-token-to-mel decoder.

    Chatterbox Turbo distills the decoder from 10 steps to 1 step
    while retaining high-fidelity audio output. This is conceptually
    similar to consistency models / progressive distillation.

    Architecture:
    - Transformer decoder with cross-attention to input features
    - Single forward pass produces mel spectrogram
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # Input projection
        input_total = config.emotion_dim + config.tag_embedding_dim + config.speaker_dim
        self.input_proj = nn.Linear(input_total, config.decoder_hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_hidden_dim,
            nhead=config.decoder_num_heads,
            dim_feedforward=config.decoder_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_num_layers,
        )

        # Output projection to prosody tokens
        self.output_proj = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.num_prosody_tokens * config.output_dim),
        )

        # Learnable queries for single-step decoding
        self.queries = nn.Parameter(
            torch.randn(1, config.num_prosody_tokens, config.decoder_hidden_dim)
        )

    def forward(
        self,
        emotion_features: Tensor,  # [batch, emotion_dim]
        tag_features: Tensor,  # [batch, num_tags, tag_dim] or [batch, tag_dim]
        speaker_features: Tensor,  # [batch, speaker_dim]
    ) -> Dict[str, Tensor]:
        """
        Single-step decode to prosody tokens.

        Args:
            emotion_features: Emotion features (with exaggeration applied)
            tag_features: Paralinguistic tag features
            speaker_features: Speaker identity features

        Returns:
            Dictionary with prosody tokens and auxiliary outputs
        """
        batch_size = emotion_features.size(0)

        # Aggregate tag features if sequence
        if tag_features.dim() == 3:
            tag_features = tag_features.mean(dim=1)  # [batch, tag_dim]

        # Pad tag features to match expected dimension
        if tag_features.size(-1) != self.config.tag_embedding_dim:
            tag_features = F.pad(
                tag_features,
                (0, self.config.tag_embedding_dim - tag_features.size(-1))
            )

        # Concatenate conditioning
        combined = torch.cat([emotion_features, tag_features, speaker_features], dim=-1)

        # Project and expand for cross-attention memory
        memory = self.input_proj(combined)  # [batch, decoder_hidden]
        memory = memory.unsqueeze(1)  # [batch, 1, decoder_hidden]

        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)  # [batch, num_tokens, decoder_hidden]

        # Single-step decode
        decoded = self.decoder(queries, memory)  # [batch, num_tokens, decoder_hidden]

        # Project to output
        output_flat = self.output_proj(decoded.mean(dim=1))  # [batch, num_tokens * output_dim]
        prosody_tokens = output_flat.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)

        return {
            "prosody_tokens": prosody_tokens,
            "decoded_features": decoded,
        }


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class ChatterboxSpeakerEncoder(nn.Module):
    """
    Encodes speaker identity from audio features.

    Supports zero-shot voice cloning from as little as 5 seconds
    of reference audio.
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Attentive pooling
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 4, 1),
        )

        # Speaker projection
        self.speaker_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.speaker_dim),
        )

    def forward(
        self,
        features: Tensor,  # [batch, seq, input_dim]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode speaker identity.

        Args:
            features: Audio features
            mask: Optional attention mask

        Returns:
            Speaker embedding [batch, speaker_dim]
        """
        x = self.feature_proj(features)

        attn_weights = self.attention(x)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)

        pooled = torch.sum(x * attn_weights, dim=1)
        speaker_emb = self.speaker_proj(pooled)

        return speaker_emb


# =============================================================================
# EMOTION ENCODER
# =============================================================================

class ChatterboxEmotionEncoder(nn.Module):
    """
    Encodes emotion from audio or produces neutral baseline.

    The emotion encoder extracts emotional characteristics from
    reference audio, which are then modulated by the exaggeration parameter.
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Temporal modeling
        self.temporal = nn.GRU(
            config.hidden_dim,
            config.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout,
        )

        # Emotion projection
        self.emotion_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

        # Neutral baseline
        self.neutral = nn.Parameter(torch.zeros(config.emotion_dim))

    def forward(
        self,
        features: Tensor,  # [batch, seq, input_dim]
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode emotion features.

        Args:
            features: Audio features
            mask: Optional attention mask

        Returns:
            Emotion embedding [batch, emotion_dim]
        """
        x = self.feature_proj(features)
        x, _ = self.temporal(x)

        # Global average pooling
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        emotion = self.emotion_proj(pooled)
        return emotion

    def get_neutral(self, batch_size: int, device: torch.device) -> Tensor:
        """Get neutral emotion baseline."""
        return self.neutral.unsqueeze(0).expand(batch_size, -1).to(device)


# =============================================================================
# CHATTERBOX MAIN MODULE
# =============================================================================

class Chatterbox(nn.Module):
    """
    Main Chatterbox module for emotion exaggeration and paralinguistic control.

    Features:
    1. Single-parameter emotion exaggeration (0.0 to 2.0)
    2. Native paralinguistic tags ([laugh], [sigh], etc.)
    3. CFG weight control for pacing
    4. Zero-shot voice cloning
    """

    def __init__(self, config: ChatterboxConfig):
        super().__init__()
        self.config = config

        # Core components
        self.speaker_encoder = ChatterboxSpeakerEncoder(config)
        self.emotion_encoder = ChatterboxEmotionEncoder(config)
        self.tag_encoder = ParalinguisticTagEncoder(config)
        self.exaggeration_module = EmotionExaggerationModule(config)
        self.cfg_module = CFGModule(config)

        if config.use_single_step_decoder:
            self.decoder = SingleStepMelDecoder(config)
        else:
            self.decoder = None

        # Direct output projection (when not using decoder)
        self.output_proj = nn.Sequential(
            nn.Linear(config.emotion_dim + config.speaker_dim + config.tag_embedding_dim,
                     config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_prosody_tokens * config.output_dim),
        )

    def forward(
        self,
        features: Tensor,  # [batch, seq, input_dim]
        text: Optional[str] = None,
        exaggeration: float = None,
        cfg_weight: float = None,
        use_adaptive_cfg: bool = True,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with emotion exaggeration and paralinguistic control.

        Args:
            features: Audio features from reference
            text: Optional text with paralinguistic tags
            exaggeration: Emotion exaggeration level (0.0-2.0)
            cfg_weight: CFG weight (0.0-1.0)
            use_adaptive_cfg: Auto-adjust CFG based on exaggeration
            mask: Optional attention mask

        Returns:
            Dictionary with prosody_tokens and auxiliary info
        """
        if exaggeration is None:
            exaggeration = self.config.default_exaggeration

        batch_size = features.size(0)

        # Encode speaker
        speaker_emb = self.speaker_encoder(features, mask)

        # Encode emotion
        emotion_emb = self.emotion_encoder(features, mask)

        # Apply exaggeration
        exag_result = self.exaggeration_module(emotion_emb, exaggeration)
        exaggerated_emotion = exag_result["exaggerated_features"]

        # Parse paralinguistic tags from text
        if text is not None:
            clean_text, parsed_tags = parse_paralinguistic_tags(text)
            tag_emb, tag_mask = self.tag_encoder(parsed_tags, batch_size)
            # Aggregate tags
            if tag_mask.any():
                tag_features = tag_emb[tag_mask.unsqueeze(-1).expand_as(tag_emb)].view(
                    batch_size, -1, self.config.tag_embedding_dim
                ).mean(dim=1)
            else:
                tag_features = torch.zeros(batch_size, self.config.tag_embedding_dim,
                                          device=features.device)
        else:
            clean_text = None
            parsed_tags = []
            tag_features = torch.zeros(batch_size, self.config.tag_embedding_dim,
                                       device=features.device)

        # Apply CFG
        neutral_emotion = self.emotion_encoder.get_neutral(batch_size, features.device)
        if use_adaptive_cfg:
            cfg_emotion = self.cfg_module.adaptive_cfg(
                exaggerated_emotion, neutral_emotion, exaggeration
            )
        else:
            cfg_emotion = self.cfg_module(
                exaggerated_emotion, neutral_emotion, cfg_weight
            )

        # Generate prosody tokens
        if self.decoder is not None:
            decode_result = self.decoder(cfg_emotion, tag_features, speaker_emb)
            prosody_tokens = decode_result["prosody_tokens"]
        else:
            combined = torch.cat([cfg_emotion, speaker_emb, tag_features], dim=-1)
            tokens_flat = self.output_proj(combined)
            prosody_tokens = tokens_flat.view(
                batch_size,
                self.config.num_prosody_tokens,
                self.config.output_dim
            )

        return {
            "prosody_tokens": prosody_tokens,
            "speaker_emb": speaker_emb,
            "emotion_emb": emotion_emb,
            "exaggerated_emotion": exaggerated_emotion,
            "cfg_emotion": cfg_emotion,
            "tag_features": tag_features,
            "exaggeration": exaggeration,
            "clean_text": clean_text,
            "parsed_tags": parsed_tags,
        }

    def from_exaggeration(
        self,
        features: Tensor,
        exaggeration: float,
        text: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Generate with specified exaggeration level."""
        return self.forward(features, text=text, exaggeration=exaggeration)

    def sweep_exaggeration(
        self,
        features: Tensor,
        levels: List[float] = None,
        text: Optional[str] = None,
    ) -> Dict[str, List[Tensor]]:
        """Generate at multiple exaggeration levels for comparison."""
        if levels is None:
            levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

        results = {"levels": levels, "tokens": [], "emotions": []}

        for level in levels:
            result = self.forward(features, text=text, exaggeration=level)
            results["tokens"].append(result["prosody_tokens"])
            results["emotions"].append(result["exaggerated_emotion"])

        return results


# =============================================================================
# CHATTERBOX ADAPTER FOR CSM PIPELINE
# =============================================================================

class ChatterboxAdapter(nn.Module):
    """
    Adapter for integrating Chatterbox with existing prosody pipeline.

    Provides drop-in replacement for other prosody encoders with
    emotion exaggeration and paralinguistic tag support.
    """

    def __init__(
        self,
        config: ChatterboxConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Main Chatterbox module
        self.chatterbox = Chatterbox(config)

        # Align output dimension if needed
        if config.output_dim != prosody_hidden:
            self.output_align = nn.Linear(config.output_dim, prosody_hidden)
        else:
            self.output_align = nn.Identity()

    def forward(
        self,
        features: Tensor,
        text: Optional[str] = None,
        exaggeration: float = None,
        cfg_weight: float = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for pipeline integration.

        Args:
            features: Audio features [batch, seq, dim]
            text: Optional text with paralinguistic tags
            exaggeration: Emotion exaggeration level
            cfg_weight: CFG weight

        Returns:
            Dictionary with prosody_tokens and auxiliary info
        """
        result = self.chatterbox(
            features,
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        # Align output
        result["prosody_tokens"] = self.output_align(result["prosody_tokens"])

        return result

    def from_text_with_tags(
        self,
        features: Tensor,
        text: str,
        exaggeration: float = 0.5,
    ) -> Dict[str, Tensor]:
        """
        Generate prosody tokens from text containing paralinguistic tags.

        Example:
            text = "Hi there [laugh], how are you doing today?"
        """
        return self.forward(features, text=text, exaggeration=exaggeration)

    def sweep_exaggeration(
        self,
        features: Tensor,
        levels: List[float] = None,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate at multiple exaggeration levels."""
        return self.chatterbox.sweep_exaggeration(features, levels, text)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class ChatterboxLoss(nn.Module):
    """
    Loss functions for training Chatterbox.

    Components:
    1. Exaggeration consistency: Same content at different exaggeration levels
       should produce proportionally different outputs
    2. Tag reconstruction: Paralinguistic tags should be recoverable
    3. Speaker preservation: Speaker identity should be maintained
    4. Prosody quality: Generated prosody should match reference
    """

    def __init__(
        self,
        config: ChatterboxConfig,
        exag_consistency_weight: float = 1.0,
        tag_weight: float = 0.5,
        speaker_weight: float = 0.5,
        prosody_weight: float = 1.0,
    ):
        super().__init__()
        self.config = config
        self.exag_consistency_weight = exag_consistency_weight
        self.tag_weight = tag_weight
        self.speaker_weight = speaker_weight
        self.prosody_weight = prosody_weight

        # Tag classifier (for reconstruction loss)
        self.tag_classifier = nn.Linear(config.tag_embedding_dim, len(PARALINGUISTIC_TAGS))

        # Speaker classifier (for preservation loss)
        # (Placeholder - would need num_speakers in config)
        self.speaker_classifier = None

    def forward(
        self,
        result: Dict[str, Tensor],
        target_prosody: Optional[Tensor] = None,
        tag_labels: Optional[Tensor] = None,
        speaker_labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute training losses.

        Args:
            result: Output from Chatterbox forward pass
            target_prosody: Ground truth prosody features
            tag_labels: Ground truth tag indices
            speaker_labels: Ground truth speaker indices

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Exaggeration consistency loss
        # Higher exaggeration should increase deviation from neutral
        emotion = result["emotion_emb"]
        exag_emotion = result["exaggerated_emotion"]
        exaggeration = result["exaggeration"]

        # Expected deviation should scale with exaggeration
        deviation = torch.norm(exag_emotion - emotion, p=2, dim=-1)
        expected_scale = exaggeration * 2.0  # exag=0.5 -> scale=1.0

        # Loss: deviation should match expected scale
        losses["exag_consistency"] = F.mse_loss(
            deviation,
            torch.full_like(deviation, expected_scale)
        )

        # Tag reconstruction loss
        if tag_labels is not None:
            tag_features = result["tag_features"]
            tag_logits = self.tag_classifier(tag_features)
            losses["tag"] = F.cross_entropy(tag_logits, tag_labels)
        else:
            losses["tag"] = torch.tensor(0.0, device=emotion.device)

        # Speaker preservation loss
        if speaker_labels is not None and self.speaker_classifier is not None:
            speaker_emb = result["speaker_emb"]
            speaker_logits = self.speaker_classifier(speaker_emb)
            losses["speaker"] = F.cross_entropy(speaker_logits, speaker_labels)
        else:
            losses["speaker"] = torch.tensor(0.0, device=emotion.device)

        # Prosody quality loss
        if target_prosody is not None:
            prosody_tokens = result["prosody_tokens"]
            losses["prosody"] = F.mse_loss(prosody_tokens, target_prosody)
        else:
            losses["prosody"] = torch.tensor(0.0, device=emotion.device)

        # Total loss
        losses["total"] = (
            self.exag_consistency_weight * losses["exag_consistency"] +
            self.tag_weight * losses["tag"] +
            self.speaker_weight * losses["speaker"] +
            self.prosody_weight * losses["prosody"]
        )

        return losses


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_chatterbox_adapter(
    config: Optional[ChatterboxConfig] = None,
    prosody_hidden: int = 2048,
) -> ChatterboxAdapter:
    """
    Factory function to create Chatterbox adapter.

    Args:
        config: Chatterbox configuration
        prosody_hidden: Output dimension for prosody tokens

    Returns:
        Configured ChatterboxAdapter
    """
    if config is None:
        config = ChatterboxConfig()

    return ChatterboxAdapter(config, prosody_hidden=prosody_hidden)


def exaggeration_to_description(exaggeration: float) -> str:
    """Convert exaggeration value to human-readable description."""
    if exaggeration <= 0.1:
        return "monotone"
    elif exaggeration <= 0.3:
        return "subdued"
    elif exaggeration <= 0.5:
        return "natural"
    elif exaggeration <= 0.75:
        return "expressive"
    elif exaggeration <= 1.0:
        return "enhanced"
    elif exaggeration <= 1.5:
        return "dramatic"
    else:
        return "theatrical"


def suggest_cfg_weight(exaggeration: float) -> float:
    """
    Suggest CFG weight based on exaggeration level.

    Higher exaggeration tends to speed up speech, so we reduce
    cfg_weight for slower, more deliberate pacing.
    """
    # Linear mapping: exag=0 -> cfg=0.7, exag=2 -> cfg=0.3
    cfg = 0.7 - 0.2 * exaggeration
    return max(0.3, min(0.9, cfg))


def format_text_with_tags(
    text: str,
    tag_positions: Dict[int, str],
) -> str:
    """
    Insert paralinguistic tags at specified word positions.

    Args:
        text: Original text
        tag_positions: Dict mapping word index to tag name

    Returns:
        Text with tags inserted

    Example:
        >>> format_text_with_tags("Hello how are you", {1: "laugh"})
        'Hello [laugh] how are you'
    """
    words = text.split()
    result_words = []

    for i, word in enumerate(words):
        if i in tag_positions:
            tag = tag_positions[i]
            result_words.append(f"[{tag}]")
        result_words.append(word)

    return " ".join(result_words)


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Config
    "ChatterboxConfig",
    # Main modules
    "Chatterbox",
    "ChatterboxAdapter",
    "ChatterboxLoss",
    # Components
    "ParalinguisticTagEncoder",
    "EmotionExaggerationModule",
    "CFGModule",
    "SingleStepMelDecoder",
    "ChatterboxSpeakerEncoder",
    "ChatterboxEmotionEncoder",
    # Tag utilities
    "parse_paralinguistic_tags",
    "ParsedTag",
    "get_supported_tags",
    "describe_tag",
    "format_text_with_tags",
    "PARALINGUISTIC_TAGS",
    "TAG_CATEGORIES",
    # Factory functions
    "create_chatterbox_adapter",
    "exaggeration_to_description",
    "suggest_cfg_weight",
]
