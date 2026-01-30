"""
IndexTTS2: 8-Dimensional Emotion Vector Control for TTS

Based on IndexTTS2 (arXiv:2506.21619):
"IndexTTS 2: Controllable Emotional Text-to-Speech"
https://github.com/index-tts/index-tts

Key Innovations:
1. 8-Dimensional Emotion Vector: Independent intensity control (0-1) for each emotion:
   - happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
   - Enables fine-grained emotion blending and intensity control

2. Duration Control: Explicitly specify number of output tokens
   - Free autoregressive mode: Generate until EOS
   - Constrained mode: Generate exactly N tokens (useful for video dubbing)

3. Emotion-Timbre Disentanglement: Independent control of voice identity and emotion
   - Emotion encoder extracts speaker-independent emotion features
   - Timbre encoder captures speaker identity
   - Cross-modal disentanglement via adversarial training

4. Three-Stage Training: Progressive curriculum for stable emotion learning
   - Stage 1: Pre-training on neutral TTS (establish synthesis capability)
   - Stage 2: Emotion classification training (learn emotion representations)
   - Stage 3: Full emotional TTS training with disentanglement

5. Soft Instruction Mechanism: Text descriptions (via Qwen3) guide generation
   - Process freestyle emotion descriptions
   - Map to 8-dim emotion vector

Integration with CSM pipeline:
- IndexTTS2Adapter generates prosody prefix tokens from 8-dim emotion vectors
- Compatible with existing ProsodyControlledCSM
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IndexTTS2Config:
    """Configuration for IndexTTS2 8-dimensional emotion control."""

    # 8 emotion dimensions (IndexTTS2 specification)
    emotion_labels: Tuple[str, ...] = (
        "happy", "angry", "sad", "afraid",
        "disgusted", "melancholic", "surprised", "calm"
    )
    num_emotions: int = 8  # Fixed for 8-dim vector

    # Model dimensions
    input_dim: int = 768  # Input feature dimension (wav2vec2/HuBERT)
    emotion_dim: int = 256  # Emotion embedding dimension
    timbre_dim: int = 256  # Timbre/speaker embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 2048  # Output to match prosody encoder (CSM)

    # GPT/Transformer settings for autoregressive generation
    gpt_hidden_dim: int = 1024  # GPT latent dimension
    gpt_num_layers: int = 12  # GPT transformer layers
    gpt_num_heads: int = 16  # Attention heads
    gpt_max_tokens: int = 2048  # Maximum output tokens

    # Duration control
    enable_duration_control: bool = True
    duration_predictor_hidden: int = 256
    min_output_tokens: int = 10
    max_output_tokens: int = 1500

    # Soft instruction (Qwen-based text encoder)
    soft_instruction_model: str = "Qwen/Qwen2.5-0.5B"  # Or Qwen3
    soft_instruction_dim: int = 896  # Qwen hidden dim
    use_soft_instruction: bool = True

    # Three-stage training settings
    stage1_epochs: int = 50  # Neutral TTS pre-training
    stage2_epochs: int = 30  # Emotion classification
    stage3_epochs: int = 100  # Full emotional TTS

    # Disentanglement
    use_adversarial_disentanglement: bool = True
    adversarial_weight: float = 0.1
    emotion_classification_weight: float = 1.0

    # Training settings
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens


# =============================================================================
# 8-DIMENSIONAL EMOTION VECTOR UTILITIES
# =============================================================================

# Emotion labels in order (matches IndexTTS2 paper)
EMOTION_LABELS = [
    "happy", "angry", "sad", "afraid",
    "disgusted", "melancholic", "surprised", "calm"
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}

# Default emotion profiles (normalized intensities)
EMOTION_PROFILES = {
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Pure calm
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3],  # Happy with slight surprise
    "angry": [0.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],  # Angry with slight disgust
    "sad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0],  # Sad with melancholy
    "afraid": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.0],  # Afraid with surprise
    "disgusted": [0.0, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Disgusted with anger
    "melancholic": [0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.2],  # Melancholic
    "surprised": [0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.0],  # Surprised with happiness/fear
    "calm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Pure calm
    # Mixed emotions
    "bittersweet": [0.5, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0],  # Happy + Sad
    "anxious": [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0],  # Low-level fear
    "excited": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],  # Happy + surprised
    "contemptuous": [0.0, 0.5, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0],  # Angry + disgusted
}


def create_emotion_vector(
    emotions: Dict[str, float] = None,
    profile: str = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create an 8-dimensional emotion vector.

    Args:
        emotions: Dict mapping emotion names to intensities (0-1)
        profile: Name of predefined profile (e.g., "happy", "sad")
        device: Target device

    Returns:
        8-dim emotion vector [8]
    """
    if device is None:
        device = torch.device("cpu")

    # Start with zeros
    vector = torch.zeros(8, device=device)

    if profile is not None:
        # Use predefined profile
        if profile.lower() in EMOTION_PROFILES:
            vector = torch.tensor(EMOTION_PROFILES[profile.lower()], device=device)

    if emotions is not None:
        # Override with provided emotions
        for emotion, intensity in emotions.items():
            emotion_lower = emotion.lower()
            if emotion_lower in EMOTION_TO_IDX:
                idx = EMOTION_TO_IDX[emotion_lower]
                vector[idx] = float(intensity)

    return vector


def normalize_emotion_vector(vector: torch.Tensor, method: str = "none") -> torch.Tensor:
    """
    Normalize emotion vector.

    Args:
        vector: 8-dim emotion vector
        method: "none" (keep as-is), "l1" (sum to 1), "l2" (unit norm), "softmax"

    Returns:
        Normalized vector
    """
    if method == "none":
        return vector.clamp(0, 1)
    elif method == "l1":
        return vector / (vector.sum() + 1e-8)
    elif method == "l2":
        return F.normalize(vector, p=2, dim=-1)
    elif method == "softmax":
        return F.softmax(vector, dim=-1)
    else:
        return vector.clamp(0, 1)


def emotion_vector_to_description(vector: torch.Tensor, threshold: float = 0.2) -> str:
    """
    Convert emotion vector to natural language description.

    Args:
        vector: 8-dim emotion vector
        threshold: Minimum intensity to include in description

    Returns:
        Natural language description
    """
    active_emotions = []

    for i, intensity in enumerate(vector):
        if intensity > threshold:
            emotion = EMOTION_LABELS[i]
            if intensity > 0.8:
                modifier = "intensely"
            elif intensity > 0.5:
                modifier = "moderately"
            else:
                modifier = "slightly"
            active_emotions.append(f"{modifier} {emotion}")

    if not active_emotions:
        return "speaking in a neutral tone"

    if len(active_emotions) == 1:
        return f"expressing {active_emotions[0]} emotion"
    else:
        emotions_str = ", ".join(active_emotions[:-1]) + f" and {active_emotions[-1]}"
        return f"expressing {emotions_str}"


# =============================================================================
# 8-DIMENSIONAL EMOTION ENCODER
# =============================================================================

class EmotionVectorEncoder(nn.Module):
    """
    Encodes 8-dimensional emotion vectors to embeddings.

    Takes [batch, 8] emotion intensity vectors and produces
    rich embeddings suitable for conditioning TTS generation.
    """

    def __init__(self, config: IndexTTS2Config):
        super().__init__()
        self.config = config

        # Per-emotion learnable embeddings
        self.emotion_embeddings = nn.Embedding(config.num_emotions, config.emotion_dim)

        # Intensity modulation network
        self.intensity_modulator = nn.Sequential(
            nn.Linear(config.num_emotions, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Fusion network (combines weighted emotion embeddings)
        self.fusion_network = nn.Sequential(
            nn.Linear(config.emotion_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(config.emotion_dim)

    def forward(
        self,
        emotion_vector: torch.Tensor,  # [batch, 8]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode 8-dim emotion vector to embedding.

        Args:
            emotion_vector: Emotion intensities [batch, 8], values in [0, 1]

        Returns:
            Dict with emotion embedding and analysis
        """
        batch_size = emotion_vector.shape[0]
        device = emotion_vector.device

        # Clamp to valid range
        emotion_vector = emotion_vector.clamp(0, 1)

        # Get all emotion embeddings [8, emotion_dim]
        all_embeddings = self.emotion_embeddings.weight

        # Weight by intensity: [batch, 8, 1] * [1, 8, emotion_dim]
        weighted_embeddings = emotion_vector.unsqueeze(-1) * all_embeddings.unsqueeze(0)

        # Sum to get combined embedding [batch, emotion_dim]
        combined_embedding = weighted_embeddings.sum(dim=1)

        # Process intensity pattern
        intensity_features = self.intensity_modulator(emotion_vector)

        # Fuse embedding with intensity features
        fused = torch.cat([combined_embedding, intensity_features], dim=-1)
        emotion_embedding = self.fusion_network(fused)
        emotion_embedding = self.output_norm(emotion_embedding)

        # Compute dominant emotion
        dominant_idx = emotion_vector.argmax(dim=-1)

        return {
            "embedding": emotion_embedding,
            "weighted_embeddings": weighted_embeddings,
            "intensity_features": intensity_features,
            "dominant_emotion": dominant_idx,
            "emotion_vector": emotion_vector,
        }


# =============================================================================
# TIMBRE (SPEAKER) ENCODER
# =============================================================================

class TimbreEncoder(nn.Module):
    """
    Encodes speaker identity/timbre from audio features.

    Designed to capture speaker-specific characteristics while being
    independent of emotional content.
    """

    def __init__(self, config: IndexTTS2Config):
        super().__init__()
        self.config = config

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Attention pooling for sequence aggregation
        self.attention_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Timbre projection
        self.timbre_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.timbre_dim),
            nn.LayerNorm(config.timbre_dim),
        )

    def forward(
        self,
        audio_features: torch.Tensor,  # [batch, seq_len, input_dim]
        mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract timbre embedding from audio features.

        Args:
            audio_features: Audio encoder features [batch, seq_len, input_dim]
            mask: Attention mask [batch, seq_len]

        Returns:
            Dict with timbre embedding
        """
        batch_size = audio_features.shape[0]

        # Encode features
        encoded = self.feature_encoder(audio_features)

        # Attention pooling
        query = self.attention_query.expand(batch_size, -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)

        pooled, attention_weights = self.attention(
            query, encoded, encoded,
            key_padding_mask=key_padding_mask,
        )
        pooled = pooled.squeeze(1)  # [batch, hidden_dim]

        # Project to timbre space
        timbre_embedding = self.timbre_projection(pooled)

        return {
            "embedding": timbre_embedding,
            "attention_weights": attention_weights,
            "encoded_features": encoded,
        }


# =============================================================================
# AUDIO EMOTION EXTRACTOR
# =============================================================================

class AudioEmotionExtractor(nn.Module):
    """
    Extracts 8-dimensional emotion vector from audio features.

    Processes audio (via wav2vec2/HuBERT features) and predicts
    the intensity of each of the 8 emotions.
    """

    def __init__(self, config: IndexTTS2Config):
        super().__init__()
        self.config = config

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Temporal modeling with transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 4, 1),
        )

        # 8-dim emotion prediction head
        self.emotion_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_emotions),
            nn.Sigmoid(),  # Output in [0, 1] for each emotion
        )

    def forward(
        self,
        audio_features: torch.Tensor,  # [batch, seq_len, input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract 8-dim emotion vector from audio.

        Args:
            audio_features: Audio encoder features [batch, seq_len, input_dim]
            mask: Attention mask [batch, seq_len]

        Returns:
            Dict with 8-dim emotion vector and features
        """
        # Encode features
        encoded = self.feature_encoder(audio_features)

        # Temporal modeling
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)

        temporal = self.temporal_encoder(
            encoded,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Attention pooling
        attention_scores = self.attention_pool(temporal)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(-1) == 0, float('-inf')
            )
        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = (temporal * attention_weights).sum(dim=1)  # [batch, hidden_dim]

        # Predict 8-dim emotion vector
        emotion_vector = self.emotion_head(pooled)  # [batch, 8]

        return {
            "emotion_vector": emotion_vector,
            "pooled_features": pooled,
            "temporal_features": temporal,
            "attention_weights": attention_weights.squeeze(-1),
        }


# =============================================================================
# SOFT INSTRUCTION ENCODER (TEXT-TO-EMOTION)
# =============================================================================

class SoftInstructionEncoder(nn.Module):
    """
    Encodes natural language emotion descriptions to 8-dim vectors.

    Uses a language model (Qwen2.5/Qwen3) to understand freestyle
    emotion descriptions and map them to the 8-dimensional space.
    """

    def __init__(self, config: IndexTTS2Config):
        super().__init__()
        self.config = config

        # Simple text encoder (placeholder - in practice use Qwen)
        self.text_embedding = nn.Embedding(50000, config.soft_instruction_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            self._sinusoidal_pos_encoding(512, config.soft_instruction_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.soft_instruction_dim,
            nhead=8,
            dim_feedforward=config.soft_instruction_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.soft_instruction_dim))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=config.soft_instruction_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Project to 8-dim emotion vector
        self.emotion_projection = nn.Sequential(
            nn.Linear(config.soft_instruction_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_emotions),
            nn.Sigmoid(),
        )

        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(config.soft_instruction_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def _sinusoidal_pos_encoding(self, length: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        text_ids: torch.Tensor,  # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text description to 8-dim emotion vector.

        Args:
            text_ids: Tokenized text [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dict with emotion vector and confidence
        """
        batch_size, seq_len = text_ids.shape
        device = text_ids.device

        # Embed text
        embedded = self.text_embedding(text_ids)

        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0).to(device)

        # Encode
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        encoded = self.encoder(embedded, src_key_padding_mask=key_padding_mask)

        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, attn_weights = self.pool_attention(
            query, encoded, encoded,
            key_padding_mask=key_padding_mask,
        )
        pooled = pooled.squeeze(1)  # [batch, soft_instruction_dim]

        # Predict emotion vector
        emotion_vector = self.emotion_projection(pooled)

        # Predict confidence
        confidence = self.confidence_head(pooled)

        return {
            "emotion_vector": emotion_vector,
            "confidence": confidence.squeeze(-1),
            "encoded_text": encoded,
            "attention_weights": attn_weights,
        }


# =============================================================================
# DURATION PREDICTOR
# =============================================================================

class DurationPredictor(nn.Module):
    """
    Predicts output duration (number of tokens) from input features.

    Can operate in:
    - Free mode: Predicts expected duration
    - Constrained mode: Enforces target duration during generation
    """

    def __init__(self, config: IndexTTS2Config):
        super().__init__()
        self.config = config

        # Duration prediction from combined features
        self.predictor = nn.Sequential(
            nn.Linear(config.emotion_dim + config.timbre_dim + config.hidden_dim,
                     config.duration_predictor_hidden),
            nn.LayerNorm(config.duration_predictor_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.duration_predictor_hidden, config.duration_predictor_hidden),
            nn.GELU(),
            nn.Linear(config.duration_predictor_hidden, 1),
        )

        # Text length encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        emotion_embedding: torch.Tensor,  # [batch, emotion_dim]
        timbre_embedding: torch.Tensor,  # [batch, timbre_dim]
        text_features: torch.Tensor,  # [batch, hidden_dim] or [batch, seq, hidden]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict output duration.

        Args:
            emotion_embedding: Emotion embedding
            timbre_embedding: Timbre embedding
            text_features: Text features (pooled or sequence)

        Returns:
            Dict with predicted duration
        """
        # Pool text features if sequence
        if text_features.dim() == 3:
            text_pooled = text_features.mean(dim=1)
        else:
            text_pooled = text_features

        text_encoded = self.text_encoder(text_pooled)

        # Combine features
        combined = torch.cat([emotion_embedding, timbre_embedding, text_encoded], dim=-1)

        # Predict duration (log scale for numerical stability)
        log_duration = self.predictor(combined)
        duration = torch.exp(log_duration).clamp(
            self.config.min_output_tokens,
            self.config.max_output_tokens
        )

        return {
            "duration": duration.squeeze(-1),
            "log_duration": log_duration.squeeze(-1),
        }


# =============================================================================
# EMOTION-TIMBRE DISENTANGLEMENT
# =============================================================================

class EmotionTimbreDisentanglement(nn.Module):
    """
    Enforces disentanglement between emotion and timbre representations.

    Uses adversarial training:
    - Emotion encoder should NOT be able to predict speaker identity
    - Timbre encoder should NOT be able to predict emotion
    """

    def __init__(self, config: IndexTTS2Config, num_speakers: int = 1000):
        super().__init__()
        self.config = config
        self.num_speakers = num_speakers

        # Adversarial speaker classifier from emotion features
        self.speaker_from_emotion = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_speakers),
        )

        # Adversarial emotion classifier from timbre features
        self.emotion_from_timbre = nn.Sequential(
            nn.Linear(config.timbre_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_emotions),
        )

        # Gradient reversal scale (increases during training)
        self.grl_scale = 0.0

    def set_grl_scale(self, scale: float):
        """Set gradient reversal scale."""
        self.grl_scale = scale

    def forward(
        self,
        emotion_embedding: torch.Tensor,
        timbre_embedding: torch.Tensor,
        speaker_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement losses.

        Args:
            emotion_embedding: Emotion embedding [batch, emotion_dim]
            timbre_embedding: Timbre embedding [batch, timbre_dim]
            speaker_labels: Ground truth speaker IDs [batch]
            emotion_labels: Ground truth emotion indices [batch]

        Returns:
            Dict with adversarial losses
        """
        losses = {}
        device = emotion_embedding.device

        # Reverse gradients for adversarial training
        if self.training:
            emotion_reversed = emotion_embedding - self.grl_scale * emotion_embedding.detach() + \
                              self.grl_scale * emotion_embedding
            timbre_reversed = timbre_embedding - self.grl_scale * timbre_embedding.detach() + \
                             self.grl_scale * timbre_embedding
        else:
            emotion_reversed = emotion_embedding
            timbre_reversed = timbre_embedding

        # Speaker prediction from emotion (should fail = uniform)
        speaker_logits = self.speaker_from_emotion(emotion_reversed)
        if speaker_labels is not None:
            losses["speaker_adv"] = F.cross_entropy(speaker_logits, speaker_labels)

            # Compute accuracy (should be near chance = 1/num_speakers)
            speaker_acc = (speaker_logits.argmax(dim=-1) == speaker_labels).float().mean()
            losses["speaker_adv_acc"] = speaker_acc
        else:
            losses["speaker_adv"] = torch.tensor(0.0, device=device)
            losses["speaker_adv_acc"] = torch.tensor(0.0, device=device)

        # Emotion prediction from timbre (should fail = uniform)
        emotion_logits = self.emotion_from_timbre(timbre_reversed)
        if emotion_labels is not None:
            losses["emotion_adv"] = F.cross_entropy(emotion_logits, emotion_labels)

            emotion_acc = (emotion_logits.argmax(dim=-1) == emotion_labels).float().mean()
            losses["emotion_adv_acc"] = emotion_acc
        else:
            losses["emotion_adv"] = torch.tensor(0.0, device=device)
            losses["emotion_adv_acc"] = torch.tensor(0.0, device=device)

        # Total adversarial loss (negate because we want classifiers to fail)
        losses["total"] = -(losses["speaker_adv"] + losses["emotion_adv"])

        return losses


# =============================================================================
# GPT LATENT DECODER
# =============================================================================

class GPTLatentDecoder(nn.Module):
    """
    GPT-style autoregressive decoder for audio token generation.

    Conditioned on:
    - 8-dim emotion vector (encoded)
    - Timbre embedding
    - Text embeddings
    """

    def __init__(self, config: IndexTTS2Config, vocab_size: int = 4096):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size + 1, config.gpt_hidden_dim)  # +1 for BOS

        # Positional embedding
        self.pos_embedding = nn.Embedding(config.gpt_max_tokens, config.gpt_hidden_dim)

        # Condition projection
        self.condition_projection = nn.Linear(
            config.emotion_dim + config.timbre_dim,
            config.gpt_hidden_dim,
        )

        # Memory projection (text encoder hidden_dim -> gpt_hidden_dim)
        self.memory_projection = nn.Linear(config.hidden_dim, config.gpt_hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.gpt_hidden_dim,
            nhead=config.gpt_num_heads,
            dim_feedforward=config.gpt_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.gpt_num_layers)

        # Output head
        self.output_norm = nn.LayerNorm(config.gpt_hidden_dim)
        self.output_head = nn.Linear(config.gpt_hidden_dim, vocab_size)

        # Duration embedding for constrained generation
        self.duration_embedding = nn.Linear(1, config.gpt_hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,  # [batch, seq_len]
        emotion_embedding: torch.Tensor,  # [batch, emotion_dim]
        timbre_embedding: torch.Tensor,  # [batch, timbre_dim]
        text_memory: torch.Tensor,  # [batch, text_len, hidden_dim]
        target_duration: Optional[torch.Tensor] = None,  # [batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GPT decoder.

        Args:
            tokens: Input token IDs [batch, seq_len]
            emotion_embedding: Emotion embedding
            timbre_embedding: Timbre embedding
            text_memory: Text encoder outputs for cross-attention
            target_duration: Optional target duration for constrained generation

        Returns:
            Dict with logits and hidden states
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(tokens) + self.pos_embedding(positions)

        # Add condition embedding to first position
        condition = self.condition_projection(
            torch.cat([emotion_embedding, timbre_embedding], dim=-1)
        )
        hidden[:, 0] = hidden[:, 0] + condition

        # Add duration embedding if provided
        if target_duration is not None:
            duration_emb = self.duration_embedding(target_duration.unsqueeze(-1))
            hidden[:, 0] = hidden[:, 0] + duration_emb

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )

        # Project memory to gpt_hidden_dim
        memory = self.memory_projection(text_memory)

        # Decode
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal_mask,
        )

        # Output
        decoded = self.output_norm(decoded)
        logits = self.output_head(decoded)

        return {
            "logits": logits,
            "hidden_states": decoded,
        }

    def generate(
        self,
        emotion_embedding: torch.Tensor,
        timbre_embedding: torch.Tensor,
        text_memory: torch.Tensor,
        target_duration: Optional[int] = None,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation.

        Args:
            emotion_embedding: Emotion embedding
            timbre_embedding: Timbre embedding
            text_memory: Text encoder outputs
            target_duration: Target number of tokens (constrained mode)
            max_tokens: Maximum tokens (free mode)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Dict with generated tokens
        """
        batch_size = emotion_embedding.shape[0]
        device = emotion_embedding.device

        # Start with BOS token
        tokens = torch.full((batch_size, 1), self.vocab_size, device=device)  # BOS

        target_len = target_duration if target_duration is not None else max_tokens
        duration_tensor = torch.tensor([target_len], device=device).expand(batch_size)

        for step in range(target_len - 1):
            outputs = self.forward(
                tokens,
                emotion_embedding,
                timbre_embedding,
                text_memory,
                target_duration=duration_tensor if target_duration is not None else None,
            )

            logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                for b in range(batch_size):
                    logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return {
            "tokens": tokens[:, 1:],  # Remove BOS
            "num_tokens": tokens.shape[1] - 1,
        }


# =============================================================================
# INDEXTTS2 MAIN MODEL
# =============================================================================

class IndexTTS2(nn.Module):
    """
    IndexTTS2: 8-Dimensional Emotion Vector Control for TTS.

    Combines:
    1. EmotionVectorEncoder: Encodes 8-dim emotion vectors
    2. TimbreEncoder: Captures speaker identity
    3. AudioEmotionExtractor: Extracts emotion from audio
    4. SoftInstructionEncoder: Text-to-emotion mapping
    5. DurationPredictor: Output duration prediction
    6. EmotionTimbreDisentanglement: Adversarial disentanglement
    7. GPTLatentDecoder: Autoregressive token generation

    Usage:
        config = IndexTTS2Config()
        model = IndexTTS2(config)

        # From 8-dim emotion vector
        tokens = model.generate(
            text_ids,
            emotion_vector=[0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3],  # happy
            reference_audio=speaker_audio,
        )
    """

    def __init__(self, config: IndexTTS2Config, num_speakers: int = 1000):
        super().__init__()
        self.config = config

        # Core encoders
        self.emotion_encoder = EmotionVectorEncoder(config)
        self.timbre_encoder = TimbreEncoder(config)
        self.audio_emotion_extractor = AudioEmotionExtractor(config)

        # Soft instruction encoder
        if config.use_soft_instruction:
            self.soft_instruction_encoder = SoftInstructionEncoder(config)
        else:
            self.soft_instruction_encoder = None

        # Duration predictor
        if config.enable_duration_control:
            self.duration_predictor = DurationPredictor(config)
        else:
            self.duration_predictor = None

        # Disentanglement
        if config.use_adversarial_disentanglement:
            self.disentanglement = EmotionTimbreDisentanglement(config, num_speakers)
        else:
            self.disentanglement = None

        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Embedding(50000, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        # GPT decoder
        self.gpt_decoder = GPTLatentDecoder(config)

        # Prosody token projection (for CSM integration)
        self.prosody_projection = nn.Sequential(
            nn.Linear(config.emotion_dim + config.timbre_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * config.num_prosody_tokens),
        )
        self.prosody_output_norm = nn.LayerNorm(config.output_dim)

        # Training stage tracking
        self.current_stage = 1

    def set_training_stage(self, stage: int):
        """Set training stage (1, 2, or 3)."""
        assert stage in [1, 2, 3], "Stage must be 1, 2, or 3"
        self.current_stage = stage

        # Adjust what's trainable
        if stage == 1:
            # Pre-training: freeze emotion components
            self._freeze_emotion_components()
        elif stage == 2:
            # Emotion training: unfreeze emotion, freeze generation
            self._unfreeze_emotion_components()
            self._freeze_generation_components()
        else:
            # Full training: everything trainable
            self._unfreeze_all()

    def _freeze_emotion_components(self):
        """Freeze emotion-related components."""
        for param in self.emotion_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_emotion_extractor.parameters():
            param.requires_grad = False
        if self.soft_instruction_encoder is not None:
            for param in self.soft_instruction_encoder.parameters():
                param.requires_grad = False

    def _unfreeze_emotion_components(self):
        """Unfreeze emotion-related components."""
        for param in self.emotion_encoder.parameters():
            param.requires_grad = True
        for param in self.audio_emotion_extractor.parameters():
            param.requires_grad = True
        if self.soft_instruction_encoder is not None:
            for param in self.soft_instruction_encoder.parameters():
                param.requires_grad = True

    def _freeze_generation_components(self):
        """Freeze generation components."""
        for param in self.gpt_decoder.parameters():
            param.requires_grad = False

    def _unfreeze_all(self):
        """Unfreeze all components."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        text_ids: torch.Tensor,  # [batch, text_len]
        audio_features: torch.Tensor,  # [batch, seq_len, input_dim]
        emotion_vector: Optional[torch.Tensor] = None,  # [batch, 8]
        speaker_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,  # [batch, token_len]
        description_ids: Optional[torch.Tensor] = None,  # [batch, desc_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text_ids: Input text tokens
            audio_features: Audio features for timbre extraction
            emotion_vector: 8-dim emotion vector (if provided)
            speaker_labels: Speaker IDs for disentanglement
            emotion_labels: Emotion indices for disentanglement
            target_tokens: Target audio tokens for training
            description_ids: Text description for soft instruction

        Returns:
            Dict with all outputs
        """
        batch_size = text_ids.shape[0]
        device = text_ids.device

        # Extract timbre from audio
        timbre_output = self.timbre_encoder(audio_features)
        timbre_embedding = timbre_output["embedding"]

        # Get emotion vector
        if emotion_vector is None:
            # Extract from audio
            emotion_output = self.audio_emotion_extractor(audio_features)
            emotion_vector = emotion_output["emotion_vector"]

        # Or from text description
        if description_ids is not None and self.soft_instruction_encoder is not None:
            soft_output = self.soft_instruction_encoder(description_ids)
            # Blend with audio-extracted (if available)
            soft_vector = soft_output["emotion_vector"]
            confidence = soft_output["confidence"].unsqueeze(-1)
            emotion_vector = confidence * soft_vector + (1 - confidence) * emotion_vector

        # Encode emotion vector to embedding
        emotion_enc_output = self.emotion_encoder(emotion_vector)
        emotion_embedding = emotion_enc_output["embedding"]

        # Encode text
        text_encoded = self.text_encoder(text_ids)

        # Duration prediction
        duration_output = None
        if self.duration_predictor is not None:
            duration_output = self.duration_predictor(
                emotion_embedding,
                timbre_embedding,
                text_encoded,
            )

        # Disentanglement losses
        disentangle_output = None
        if self.disentanglement is not None:
            disentangle_output = self.disentanglement(
                emotion_embedding,
                timbre_embedding,
                speaker_labels,
                emotion_labels,
            )

        # GPT decoding (if training with target tokens)
        gpt_output = None
        if target_tokens is not None:
            gpt_output = self.gpt_decoder(
                target_tokens,
                emotion_embedding,
                timbre_embedding,
                text_encoded,
                target_duration=duration_output["duration"] if duration_output else None,
            )

        # Generate prosody tokens for CSM
        combined = torch.cat([emotion_embedding, timbre_embedding], dim=-1)
        prosody_tokens = self.prosody_projection(combined)
        prosody_tokens = prosody_tokens.view(
            batch_size, self.config.num_prosody_tokens, self.config.output_dim
        )
        prosody_tokens = self.prosody_output_norm(prosody_tokens)

        return {
            "emotion_vector": emotion_vector,
            "emotion_embedding": emotion_embedding,
            "timbre_embedding": timbre_embedding,
            "prosody_tokens": prosody_tokens,
            "duration_output": duration_output,
            "disentangle_output": disentangle_output,
            "gpt_output": gpt_output,
            "text_encoded": text_encoded,
        }

    def generate(
        self,
        text_ids: torch.Tensor,
        emotion_vector: Union[torch.Tensor, List[float], Dict[str, float]],
        reference_audio: torch.Tensor,
        target_duration: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate audio tokens with 8-dim emotion control.

        Args:
            text_ids: Input text tokens [batch, text_len]
            emotion_vector: 8-dim emotion vector, list, or dict
            reference_audio: Reference audio features for timbre [batch, seq, input_dim]
            target_duration: Target number of output tokens (optional)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            Dict with generated tokens and prosody tokens
        """
        batch_size = text_ids.shape[0]
        device = text_ids.device

        # Convert emotion vector if needed
        if isinstance(emotion_vector, dict):
            emotion_vector = create_emotion_vector(emotions=emotion_vector, device=device)
        elif isinstance(emotion_vector, list):
            emotion_vector = torch.tensor(emotion_vector, device=device)

        if emotion_vector.dim() == 1:
            emotion_vector = emotion_vector.unsqueeze(0).expand(batch_size, -1)

        # Extract timbre
        timbre_output = self.timbre_encoder(reference_audio)
        timbre_embedding = timbre_output["embedding"]

        # Encode emotion
        emotion_enc = self.emotion_encoder(emotion_vector)
        emotion_embedding = emotion_enc["embedding"]

        # Encode text
        text_encoded = self.text_encoder(text_ids)

        # Predict duration if not provided
        if target_duration is None and self.duration_predictor is not None:
            dur_output = self.duration_predictor(
                emotion_embedding, timbre_embedding, text_encoded
            )
            target_duration = int(dur_output["duration"].mean().item())

        # Generate tokens
        gen_output = self.gpt_decoder.generate(
            emotion_embedding,
            timbre_embedding,
            text_encoded,
            target_duration=target_duration,
            temperature=temperature,
            top_p=top_p,
        )

        # Generate prosody tokens for CSM
        combined = torch.cat([emotion_embedding, timbre_embedding], dim=-1)
        prosody_tokens = self.prosody_projection(combined)
        prosody_tokens = prosody_tokens.view(
            batch_size, self.config.num_prosody_tokens, self.config.output_dim
        )
        prosody_tokens = self.prosody_output_norm(prosody_tokens)

        return {
            "tokens": gen_output["tokens"],
            "prosody_tokens": prosody_tokens,
            "emotion_vector": emotion_vector,
            "predicted_duration": target_duration,
        }


# =============================================================================
# INDEXTTS2 ADAPTER FOR CSM INTEGRATION
# =============================================================================

class IndexTTS2Adapter(nn.Module):
    """
    Adapter for integrating IndexTTS2 with the existing prosody pipeline.

    Provides a simple interface for 8-dimensional emotion control.

    Usage:
        adapter = IndexTTS2Adapter(config)

        # From 8-dim vector
        tokens = adapter.from_emotion_vector([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3])

        # From profile name
        tokens = adapter.from_profile("happy", intensity=0.8)

        # From individual emotions
        tokens = adapter.from_emotions(happy=0.8, surprised=0.2)
    """

    def __init__(
        self,
        config: IndexTTS2Config,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core emotion encoder
        self.emotion_encoder = EmotionVectorEncoder(config)

        # Prosody adapter
        if config.emotion_dim != prosody_hidden:
            self.prosody_adapter = nn.Sequential(
                nn.Linear(config.emotion_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.prosody_adapter = nn.Identity()

        # Multi-token projection
        self.token_projection = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * config.num_prosody_tokens),
        )
        self.output_norm = nn.LayerNorm(config.output_dim)

    def from_emotion_vector(
        self,
        emotion_vector: Union[torch.Tensor, List[float]],
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from 8-dim emotion vector.

        Args:
            emotion_vector: 8-dim emotion intensities [0-1]
            batch_size: Batch size

        Returns:
            Dict with prosody tokens and emotion info
        """
        device = next(self.parameters()).device

        if isinstance(emotion_vector, list):
            emotion_vector = torch.tensor(emotion_vector, device=device)
        else:
            emotion_vector = emotion_vector.to(device)

        if emotion_vector.dim() == 1:
            emotion_vector = emotion_vector.unsqueeze(0).expand(batch_size, -1)

        # Encode
        enc_output = self.emotion_encoder(emotion_vector)
        embedding = enc_output["embedding"]

        # Generate tokens
        tokens = self.token_projection(embedding)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        # Adapt to prosody hidden
        adapted = self.prosody_adapter(embedding)

        return {
            "prosody_tokens": tokens,
            "emotion_embedding": adapted,
            "emotion_vector": emotion_vector,
            "dominant_emotion": enc_output["dominant_emotion"],
        }

    def from_profile(
        self,
        profile_name: str,
        intensity: float = 1.0,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from emotion profile.

        Args:
            profile_name: Profile name (happy, sad, angry, etc.)
            intensity: Overall intensity scaling
            batch_size: Batch size

        Returns:
            Dict with prosody tokens
        """
        device = next(self.parameters()).device

        vector = create_emotion_vector(profile=profile_name, device=device)
        vector = vector * intensity

        return self.from_emotion_vector(vector, batch_size)

    def from_emotions(
        self,
        batch_size: int = 1,
        **emotions: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from individual emotion intensities.

        Args:
            batch_size: Batch size
            **emotions: Keyword arguments for each emotion (e.g., happy=0.8)

        Returns:
            Dict with prosody tokens
        """
        device = next(self.parameters()).device

        vector = create_emotion_vector(emotions=emotions, device=device)

        return self.from_emotion_vector(vector, batch_size)

    def interpolate(
        self,
        vector1: Union[torch.Tensor, List[float]],
        vector2: Union[torch.Tensor, List[float]],
        t: float,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotion vectors.

        Args:
            vector1: Source emotion vector
            vector2: Target emotion vector
            t: Interpolation factor [0, 1]
            batch_size: Batch size

        Returns:
            Dict with interpolated prosody tokens
        """
        device = next(self.parameters()).device

        if isinstance(vector1, list):
            vector1 = torch.tensor(vector1, device=device)
        if isinstance(vector2, list):
            vector2 = torch.tensor(vector2, device=device)

        interpolated = vector1 * (1 - t) + vector2 * t

        return self.from_emotion_vector(interpolated, batch_size)

    def forward(
        self,
        emotion_vector: Optional[torch.Tensor] = None,
        profile: Optional[str] = None,
        intensity: float = 1.0,
        **emotions: float,
    ) -> torch.Tensor:
        """
        Simple forward returning prosody tokens.

        Args:
            emotion_vector: 8-dim emotion vector
            profile: Profile name
            intensity: Intensity scaling
            **emotions: Individual emotion intensities

        Returns:
            Prosody tokens [batch, num_tokens, hidden_dim]
        """
        if emotion_vector is not None:
            result = self.from_emotion_vector(emotion_vector)
        elif profile is not None:
            result = self.from_profile(profile, intensity)
        elif emotions:
            result = self.from_emotions(**emotions)
        else:
            result = self.from_profile("neutral")

        return result["prosody_tokens"]


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class IndexTTS2Loss(nn.Module):
    """
    Combined loss function for IndexTTS2 training.

    Supports three-stage training curriculum:
    - Stage 1: TTS loss only (neutral pre-training)
    - Stage 2: Emotion classification loss
    - Stage 3: Full losses including disentanglement
    """

    def __init__(
        self,
        config: IndexTTS2Config,
        tts_weight: float = 1.0,
        emotion_class_weight: float = 0.5,
        disentangle_weight: float = 0.1,
        duration_weight: float = 0.3,
    ):
        super().__init__()
        self.config = config
        self.tts_weight = tts_weight
        self.emotion_class_weight = emotion_class_weight
        self.disentangle_weight = disentangle_weight
        self.duration_weight = duration_weight

        # Emotion classification loss
        self.emotion_ce = nn.CrossEntropyLoss()

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_tokens: torch.Tensor,
        target_emotion_idx: Optional[torch.Tensor] = None,
        target_duration: Optional[torch.Tensor] = None,
        stage: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for training.

        Args:
            model_output: Output from IndexTTS2 forward
            target_tokens: Ground truth audio tokens
            target_emotion_idx: Ground truth dominant emotion index
            target_duration: Ground truth duration
            stage: Training stage (1, 2, or 3)

        Returns:
            Dict with individual and total losses
        """
        losses = {}
        device = target_tokens.device

        # TTS loss (all stages)
        if model_output.get("gpt_output") is not None:
            gpt_logits = model_output["gpt_output"]["logits"]

            # Flatten for loss
            vocab_size = gpt_logits.shape[-1]
            logits_flat = gpt_logits[:, :-1, :].reshape(-1, vocab_size)
            targets_flat = target_tokens[:, 1:].reshape(-1)

            min_len = min(logits_flat.shape[0], targets_flat.shape[0])
            losses["tts"] = F.cross_entropy(
                logits_flat[:min_len], targets_flat[:min_len]
            )
        else:
            losses["tts"] = torch.tensor(0.0, device=device)

        # Emotion classification loss (stages 2, 3)
        if stage >= 2 and target_emotion_idx is not None:
            emotion_vector = model_output["emotion_vector"]
            emotion_logits = emotion_vector  # Treat as logits
            losses["emotion_class"] = self.emotion_ce(emotion_logits, target_emotion_idx)
        else:
            losses["emotion_class"] = torch.tensor(0.0, device=device)

        # Disentanglement loss (stage 3)
        if stage >= 3 and model_output.get("disentangle_output") is not None:
            dis_output = model_output["disentangle_output"]
            losses["disentangle"] = -dis_output["total"]  # Negate (we want classifiers to fail)
        else:
            losses["disentangle"] = torch.tensor(0.0, device=device)

        # Duration loss (all stages if enabled)
        if model_output.get("duration_output") is not None and target_duration is not None:
            pred_log_dur = model_output["duration_output"]["log_duration"]
            target_log_dur = torch.log(target_duration.float() + 1)
            losses["duration"] = F.mse_loss(pred_log_dur, target_log_dur)
        else:
            losses["duration"] = torch.tensor(0.0, device=device)

        # Total loss with stage-appropriate weights
        if stage == 1:
            total = losses["tts"] * self.tts_weight
        elif stage == 2:
            total = (
                losses["tts"] * self.tts_weight +
                losses["emotion_class"] * self.emotion_class_weight
            )
        else:
            total = (
                losses["tts"] * self.tts_weight +
                losses["emotion_class"] * self.emotion_class_weight +
                losses["disentangle"] * self.disentangle_weight +
                losses["duration"] * self.duration_weight
            )

        losses["total"] = total

        return losses


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IndexTTS2 - 8-Dimensional Emotion Vector Control")
    print("=" * 70)

    config = IndexTTS2Config()

    # Test 1: Emotion Vector Creation
    print("\n[Test 1] 8-Dimensional Emotion Vector...")
    print(f"  Emotions: {EMOTION_LABELS}")

    vector = create_emotion_vector(profile="happy")
    print(f"  Happy profile: {vector.tolist()}")

    vector = create_emotion_vector(emotions={"happy": 0.8, "surprised": 0.3})
    print(f"  Custom (happy=0.8, surprised=0.3): {vector.tolist()}")

    desc = emotion_vector_to_description(vector)
    print(f"  Description: \"{desc}\"")
    print("  [PASS]")

    # Test 2: EmotionVectorEncoder
    print("\n[Test 2] EmotionVectorEncoder...")
    encoder = EmotionVectorEncoder(config)

    batch_size = 2
    vectors = torch.rand(batch_size, 8)
    output = encoder(vectors)

    print(f"  Input shape: {vectors.shape}")
    print(f"  Embedding shape: {output['embedding'].shape}")
    print(f"  Dominant emotions: {output['dominant_emotion'].tolist()}")
    print("  [PASS]")

    # Test 3: TimbreEncoder
    print("\n[Test 3] TimbreEncoder...")
    timbre_enc = TimbreEncoder(config)

    audio_features = torch.randn(batch_size, 100, config.input_dim)
    timbre_output = timbre_enc(audio_features)

    print(f"  Audio features shape: {audio_features.shape}")
    print(f"  Timbre embedding shape: {timbre_output['embedding'].shape}")
    print("  [PASS]")

    # Test 4: AudioEmotionExtractor
    print("\n[Test 4] AudioEmotionExtractor...")
    audio_extractor = AudioEmotionExtractor(config)

    audio_output = audio_extractor(audio_features)
    print(f"  Extracted emotion vector shape: {audio_output['emotion_vector'].shape}")
    print(f"  Example vector: {audio_output['emotion_vector'][0].tolist()}")
    print("  [PASS]")

    # Test 5: SoftInstructionEncoder
    print("\n[Test 5] SoftInstructionEncoder...")
    soft_enc = SoftInstructionEncoder(config)

    text_ids = torch.randint(0, 1000, (batch_size, 20))
    soft_output = soft_enc(text_ids)

    print(f"  Text input shape: {text_ids.shape}")
    print(f"  Predicted emotion vector shape: {soft_output['emotion_vector'].shape}")
    print(f"  Confidence: {soft_output['confidence'].tolist()}")
    print("  [PASS]")

    # Test 6: DurationPredictor
    print("\n[Test 6] DurationPredictor...")
    dur_pred = DurationPredictor(config)

    text_feat = torch.randn(batch_size, config.hidden_dim)
    dur_output = dur_pred(
        output["embedding"],
        timbre_output["embedding"],
        text_feat,
    )

    print(f"  Predicted durations: {dur_output['duration'].tolist()}")
    print("  [PASS]")

    # Test 7: EmotionTimbreDisentanglement
    print("\n[Test 7] EmotionTimbreDisentanglement...")
    disentangle = EmotionTimbreDisentanglement(config, num_speakers=100)
    disentangle.set_grl_scale(0.5)

    speaker_labels = torch.randint(0, 100, (batch_size,))
    emotion_labels = torch.randint(0, 8, (batch_size,))

    dis_output = disentangle(
        output["embedding"],
        timbre_output["embedding"],
        speaker_labels,
        emotion_labels,
    )

    print(f"  Speaker adversarial loss: {dis_output['speaker_adv'].item():.4f}")
    print(f"  Emotion adversarial loss: {dis_output['emotion_adv'].item():.4f}")
    print(f"  Speaker accuracy: {dis_output['speaker_adv_acc'].item():.4f}")
    print("  [PASS]")

    # Test 8: GPTLatentDecoder
    print("\n[Test 8] GPTLatentDecoder...")
    gpt = GPTLatentDecoder(config)

    tokens = torch.randint(0, 4096, (batch_size, 50))
    text_memory = torch.randn(batch_size, 30, config.hidden_dim)

    gpt_output = gpt(
        tokens,
        output["embedding"],
        timbre_output["embedding"],
        text_memory,
    )

    print(f"  Input tokens shape: {tokens.shape}")
    print(f"  Output logits shape: {gpt_output['logits'].shape}")
    print("  [PASS]")

    # Test 9: Full IndexTTS2 Model
    print("\n[Test 9] Full IndexTTS2 Model...")
    model = IndexTTS2(config, num_speakers=100)

    text_ids = torch.randint(0, 1000, (batch_size, 30))

    model_output = model(
        text_ids,
        audio_features,
        emotion_vector=vectors,
        speaker_labels=speaker_labels,
        emotion_labels=emotion_labels,
        target_tokens=tokens,
    )

    print(f"  Prosody tokens shape: {model_output['prosody_tokens'].shape}")
    print(f"  Emotion vector shape: {model_output['emotion_vector'].shape}")
    print("  [PASS]")

    # Test 10: IndexTTS2Adapter
    print("\n[Test 10] IndexTTS2Adapter (CSM Integration)...")
    adapter = IndexTTS2Adapter(config)

    # From vector
    result1 = adapter.from_emotion_vector([0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0])
    print(f"  From vector - tokens shape: {result1['prosody_tokens'].shape}")

    # From profile
    result2 = adapter.from_profile("sad", intensity=0.7)
    print(f"  From profile - tokens shape: {result2['prosody_tokens'].shape}")

    # From individual emotions
    result3 = adapter.from_emotions(angry=0.6, disgusted=0.3)
    print(f"  From emotions - tokens shape: {result3['prosody_tokens'].shape}")

    # Interpolation
    result4 = adapter.interpolate([1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], t=0.5)
    print(f"  Interpolated - dominant: {EMOTION_LABELS[result4['dominant_emotion'].item()]}")
    print("  [PASS]")

    # Test 11: Loss Function
    print("\n[Test 11] Loss Function (Three-Stage)...")
    loss_fn = IndexTTS2Loss(config)

    for stage in [1, 2, 3]:
        losses = loss_fn(
            model_output,
            target_tokens=tokens,
            target_emotion_idx=emotion_labels,
            target_duration=torch.randint(100, 500, (batch_size,)),
            stage=stage,
        )
        print(f"  Stage {stage} - Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 12: Three-Stage Training
    print("\n[Test 12] Three-Stage Training Setup...")
    for stage in [1, 2, 3]:
        model.set_training_stage(stage)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"  Stage {stage}: {trainable:,} / {total:,} params trainable "
              f"({100*trainable/total:.1f}%)")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All IndexTTS2 tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from indextts2 import (
    IndexTTS2Config,
    IndexTTS2Adapter,
    create_emotion_vector,
    emotion_vector_to_description,
    EMOTION_LABELS,
)

# Initialize
config = IndexTTS2Config()
adapter = IndexTTS2Adapter(config)

# Option 1: From 8-dimensional vector
vector = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3]  # happy + surprised + calm
result = adapter.from_emotion_vector(vector)
prosody_tokens = result['prosody_tokens']  # [1, 4, 2048]

# Option 2: From profile name
result = adapter.from_profile("happy", intensity=0.9)

# Option 3: From individual emotions
result = adapter.from_emotions(happy=0.7, surprised=0.3, calm=0.2)

# Option 4: Create custom vector
vector = create_emotion_vector(
    emotions={"angry": 0.6, "disgusted": 0.3},
    device=torch.device("cuda")
)

# Convert vector to description
desc = emotion_vector_to_description(vector)
# -> "expressing moderately angry and slightly disgusted emotion"

# Interpolate between emotions
result = adapter.interpolate(
    [1, 0, 0, 0, 0, 0, 0, 0],  # pure happy
    [0, 0, 1, 0, 0, 0, 0, 0],  # pure sad
    t=0.4  # 40% toward sad
)

# Use with ProsodyControlledCSM:
# combined_prefix = torch.cat([prosody_prefix, result['prosody_tokens']], dim=1)
# output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
