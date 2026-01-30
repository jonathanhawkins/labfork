"""
EmoKnob: Direction Vector Emotion Control

Based on EmoKnob (EMNLP 2024): "EmoKnob: Enhance Voice Cloning with Fine-Grained
Emotion Control" - arXiv:2410.00316

Key Innovation: Extract emotion direction vectors in speaker embedding space from
pairs of neutral/emotional samples, then manipulate embeddings with scalar intensity.

Core Formula:
    v_e = (1/N) * Σ (u_e^i - u_n^i) / ||u_e^i - u_n^i||  (normalized direction)
    u_s,e = u_s + α * v_e  (emotion-controlled embedding)

Where:
    - u_e^i: embedding of emotional sample i
    - u_n^i: embedding of corresponding neutral sample i
    - v_e: normalized emotion direction vector
    - u_s: target speaker embedding
    - α: scalar intensity knob (0 = neutral, higher = more emotional)

Benefits:
1. Few-shot emotion extraction (only 2 samples needed per emotion)
2. Text-based emotion control via LLM-generated TTS or retrieval
3. Fine-grained intensity control with scalar knob
4. 83% preference over commercial TTS for emotion conveyance

Text-Based Emotion Control Methods:
1. Synthetic Data: LLM generates emotional text → TTS synthesizes → extract direction
2. Retrieval: Find similar emotional transcripts in corpus → use corresponding audio

GitHub: github.com/tonychenxyz/emoknob
Paper: https://emoknob.cs.columbia.edu/
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EmoKnobConfig:
    """Configuration for EmoKnob direction vector emotion control."""

    # Embedding dimensions
    input_dim: int = 768  # Input feature dimension (e.g., from wav2vec2/HuBERT)
    speaker_dim: int = 256  # Speaker embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 2048  # Output to match prosody encoder

    # Direction vector settings
    num_pairs_for_extraction: int = 5  # Number of (emotional, neutral) pairs
    normalize_direction: bool = True  # Normalize direction vectors

    # Intensity control
    default_intensity: float = 0.7  # Default α when not specified
    min_intensity: float = 0.0  # Minimum intensity (neutral)
    max_intensity: float = 2.0  # Maximum intensity (strong emotion)

    # Text-based emotion extraction
    use_text_emotion: bool = True  # Enable text-based emotion control
    text_embedding_dim: int = 384  # Sentence transformer dimension
    retrieval_topk: int = 5  # Top-k retrieval results

    # Training settings
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Pre-computed direction vectors
    direction_cache_path: str = "./emoknob_directions"

    # Number of emotions
    num_emotions: int = 8  # neutral, happy, sad, angry, surprised, calm, fearful, disgusted

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate


# Standard emotion categories
EMOKNOB_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "calm",
    "fearful",
    "disgusted",
    "excited",  # Extended emotions
    "tender",
    "anxious",
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOKNOB_EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class SpeakerEncoder(nn.Module):
    """
    Encodes audio features to speaker embedding space.

    Uses attentive pooling for variable-length inputs.
    Compatible with wav2vec2, HuBERT, and mel spectrogram inputs.
    """

    def __init__(self, config: EmoKnobConfig):
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
        mask: Optional[Tensor] = None,  # [batch, seq]
    ) -> Tensor:
        """
        Encode audio features to speaker embedding.

        Args:
            features: Audio features [batch, seq, input_dim]
            mask: Optional attention mask [batch, seq]

        Returns:
            Speaker embedding [batch, speaker_dim]
        """
        # Project features
        x = self.feature_proj(features)  # [batch, seq, hidden]

        # Compute attention weights
        attn_weights = self.attention(x)  # [batch, seq, 1]

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted pooling
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch, hidden]

        # Project to speaker space
        speaker_emb = self.speaker_proj(pooled)  # [batch, speaker_dim]

        return speaker_emb


# =============================================================================
# DIRECTION VECTOR EXTRACTOR
# =============================================================================

class DirectionVectorExtractor:
    """
    Extracts emotion direction vectors from paired samples.

    Given pairs of (emotional, neutral) samples from the same speaker,
    computes the normalized direction vector that represents the emotion.

    Algorithm:
        1. Encode both samples with speaker encoder
        2. Compute difference: d = u_e - u_n
        3. Normalize: v = d / ||d||
        4. Average across pairs for robustness
    """

    def __init__(
        self,
        speaker_encoder: SpeakerEncoder,
        config: EmoKnobConfig,
    ):
        self.speaker_encoder = speaker_encoder
        self.config = config
        self.device = next(speaker_encoder.parameters()).device

    @torch.no_grad()
    def extract_direction(
        self,
        emotional_features: List[Tensor],  # List of [1, seq, dim]
        neutral_features: List[Tensor],  # List of [1, seq, dim]
    ) -> Tensor:
        """
        Extract emotion direction vector from paired samples.

        Args:
            emotional_features: List of emotional sample features
            neutral_features: List of corresponding neutral sample features

        Returns:
            Normalized direction vector [speaker_dim]
        """
        assert len(emotional_features) == len(neutral_features), \
            "Must have equal number of emotional and neutral samples"

        self.speaker_encoder.eval()

        direction_vectors = []

        for emo_feat, neu_feat in zip(emotional_features, neutral_features):
            # Move to device
            emo_feat = emo_feat.to(self.device)
            neu_feat = neu_feat.to(self.device)

            # Encode both samples
            emo_emb = self.speaker_encoder(emo_feat)  # [1, speaker_dim]
            neu_emb = self.speaker_encoder(neu_feat)  # [1, speaker_dim]

            # Compute difference
            diff = emo_emb - neu_emb  # [1, speaker_dim]

            # Normalize if configured
            if self.config.normalize_direction:
                diff = F.normalize(diff, p=2, dim=-1)

            direction_vectors.append(diff)

        # Stack and average
        directions = torch.cat(direction_vectors, dim=0)  # [num_pairs, speaker_dim]
        avg_direction = directions.mean(dim=0)  # [speaker_dim]

        # Final normalization
        if self.config.normalize_direction:
            avg_direction = F.normalize(avg_direction, p=2, dim=-1)

        return avg_direction

    def extract_all_emotions(
        self,
        samples: Dict[str, Dict[str, List[Tensor]]],
        # Format: {"happy": {"emotional": [feat1, feat2, ...], "neutral": [feat1, feat2, ...]}}
    ) -> Dict[str, Tensor]:
        """
        Extract direction vectors for all emotions.

        Args:
            samples: Dictionary mapping emotion to paired samples

        Returns:
            Dictionary mapping emotion to direction vector
        """
        directions = {}

        for emotion, data in samples.items():
            if emotion == "neutral":
                continue

            emo_features = data["emotional"]
            neu_features = data["neutral"]

            direction = self.extract_direction(emo_features, neu_features)
            directions[emotion] = direction

        return directions


# =============================================================================
# EMOTION DIRECTION MANIPULATOR
# =============================================================================

class EmotionDirectionManipulator(nn.Module):
    """
    Manipulates speaker embeddings using emotion direction vectors.

    Core formula: u_s,e = u_s + α * v_e

    Where:
        - u_s: original speaker embedding
        - v_e: emotion direction vector
        - α: intensity scalar
        - u_s,e: emotion-controlled speaker embedding
    """

    def __init__(self, config: EmoKnobConfig):
        super().__init__()
        self.config = config

        # Learnable per-emotion scaling factors (optional refinement)
        self.emotion_scales = nn.ParameterDict()

        # Direction vector cache (loaded from disk or computed)
        self.direction_cache: Dict[str, Tensor] = {}

    def register_direction(
        self,
        emotion: str,
        direction: Tensor,
        learnable_scale: bool = True,
    ):
        """Register a pre-computed direction vector for an emotion."""
        # Store direction
        self.register_buffer(f"dir_{emotion}", direction)
        self.direction_cache[emotion] = direction

        # Optional learnable scale
        if learnable_scale:
            self.emotion_scales[emotion] = nn.Parameter(torch.ones(1))

    def load_directions(self, path: str):
        """Load pre-computed direction vectors from disk."""
        path = Path(path)

        if not path.exists():
            warnings.warn(f"Direction cache not found at {path}")
            return

        for file in path.glob("*.pt"):
            emotion = file.stem
            direction = torch.load(file, map_location='cpu')
            self.register_direction(emotion, direction)

    def save_directions(self, path: str):
        """Save computed direction vectors to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for emotion, direction in self.direction_cache.items():
            torch.save(direction.cpu(), path / f"{emotion}.pt")

    def forward(
        self,
        speaker_emb: Tensor,  # [batch, speaker_dim]
        emotion: str,
        intensity: float = None,
    ) -> Tensor:
        """
        Apply emotion direction to speaker embedding.

        Args:
            speaker_emb: Original speaker embedding
            emotion: Target emotion name
            intensity: Emotion intensity (α parameter)

        Returns:
            Emotion-controlled speaker embedding
        """
        if intensity is None:
            intensity = self.config.default_intensity

        # Clamp intensity
        intensity = max(self.config.min_intensity,
                       min(self.config.max_intensity, intensity))

        # Get direction vector
        if emotion not in self.direction_cache:
            warnings.warn(f"Emotion '{emotion}' not in cache, returning unchanged")
            return speaker_emb

        direction = self.direction_cache[emotion].to(speaker_emb.device)

        # Apply learnable scale if available
        if emotion in self.emotion_scales:
            scale = self.emotion_scales[emotion]
            direction = direction * scale

        # Apply direction: u_s,e = u_s + α * v_e
        return speaker_emb + intensity * direction

    def interpolate_emotions(
        self,
        speaker_emb: Tensor,
        emotions: List[str],
        weights: List[float],
        intensity: float = None,
    ) -> Tensor:
        """
        Blend multiple emotions with weighted interpolation.

        Args:
            speaker_emb: Original speaker embedding
            emotions: List of emotion names
            weights: Corresponding weights (should sum to 1)
            intensity: Overall intensity

        Returns:
            Blended emotion speaker embedding
        """
        if intensity is None:
            intensity = self.config.default_intensity

        # Normalize weights
        weights = torch.tensor(weights, device=speaker_emb.device)
        weights = weights / weights.sum()

        # Compute blended direction
        blended_direction = torch.zeros_like(speaker_emb[0])

        for emotion, weight in zip(emotions, weights):
            if emotion in self.direction_cache:
                direction = self.direction_cache[emotion].to(speaker_emb.device)
                blended_direction = blended_direction + weight * direction

        # Normalize blended direction
        if self.config.normalize_direction:
            blended_direction = F.normalize(blended_direction, p=2, dim=-1)

        return speaker_emb + intensity * blended_direction


# =============================================================================
# TEXT-BASED EMOTION EXTRACTION
# =============================================================================

class TextEmotionExtractor(nn.Module):
    """
    Extracts emotion from text descriptions.

    Two methods:
    1. Synthetic Data: Generate emotional text via LLM → TTS → extract direction
    2. Retrieval: Find similar transcripts in corpus → use corresponding audio
    """

    def __init__(self, config: EmoKnobConfig):
        super().__init__()
        self.config = config

        # Text encoder (sentence transformer)
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.speaker_dim),
        )

        # Direction predictor from text
        self.direction_predictor = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.speaker_dim),
        )

        # Retrieval index (transcript embeddings)
        self.retrieval_index: Optional[Tensor] = None
        self.retrieval_emotions: List[str] = []
        self.retrieval_directions: Dict[str, Tensor] = {}

    def build_retrieval_index(
        self,
        transcript_embeddings: Tensor,  # [num_transcripts, text_dim]
        emotions: List[str],
        directions: Dict[str, Tensor],
    ):
        """Build retrieval index from transcript-emotion pairs."""
        self.register_buffer("retrieval_index", transcript_embeddings)
        self.retrieval_emotions = emotions
        self.retrieval_directions = directions

    def retrieve_direction(
        self,
        text_embedding: Tensor,  # [batch, text_dim]
    ) -> Tuple[Tensor, List[str]]:
        """
        Retrieve emotion direction via text similarity.

        Args:
            text_embedding: Query text embedding

        Returns:
            direction: Retrieved direction vector
            emotions: Retrieved emotion labels
        """
        if self.retrieval_index is None:
            raise ValueError("Retrieval index not built. Call build_retrieval_index first.")

        # Compute cosine similarity
        text_norm = F.normalize(text_embedding, p=2, dim=-1)
        index_norm = F.normalize(self.retrieval_index, p=2, dim=-1)

        similarity = torch.matmul(text_norm, index_norm.T)  # [batch, num_transcripts]

        # Get top-k
        topk_vals, topk_idx = torch.topk(similarity, k=self.config.retrieval_topk, dim=-1)

        # Aggregate directions
        batch_size = text_embedding.size(0)
        directions = []
        emotions = []

        for b in range(batch_size):
            batch_dirs = []
            batch_emos = []

            for idx in topk_idx[b]:
                emotion = self.retrieval_emotions[idx.item()]
                batch_emos.append(emotion)

                if emotion in self.retrieval_directions:
                    batch_dirs.append(self.retrieval_directions[emotion])

            if batch_dirs:
                avg_dir = torch.stack(batch_dirs).mean(dim=0)
                directions.append(avg_dir)
            else:
                directions.append(torch.zeros(self.config.speaker_dim))

            emotions.append(batch_emos)

        return torch.stack(directions), emotions

    def predict_direction(
        self,
        text_embedding: Tensor,  # [batch, text_dim]
    ) -> Tensor:
        """
        Directly predict emotion direction from text embedding.

        Args:
            text_embedding: Text embedding [batch, text_dim]

        Returns:
            Predicted direction vector [batch, speaker_dim]
        """
        direction = self.direction_predictor(text_embedding)

        if self.config.normalize_direction:
            direction = F.normalize(direction, p=2, dim=-1)

        return direction


# =============================================================================
# EMOKNOB MAIN MODULE
# =============================================================================

class EmoKnob(nn.Module):
    """
    Main EmoKnob module combining all components.

    Provides:
    1. Direction vector extraction from paired samples
    2. Speaker embedding manipulation with intensity control
    3. Text-based emotion extraction
    4. Integration with TTS pipeline
    """

    def __init__(self, config: EmoKnobConfig):
        super().__init__()
        self.config = config

        # Components
        self.speaker_encoder = SpeakerEncoder(config)
        self.manipulator = EmotionDirectionManipulator(config)

        if config.use_text_emotion:
            self.text_extractor = TextEmotionExtractor(config)
        else:
            self.text_extractor = None

        # Output projection to prosody tokens
        self.output_proj = nn.Sequential(
            nn.Linear(config.speaker_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_prosody_tokens * config.output_dim),
        )

    def extract_directions(
        self,
        samples: Dict[str, Dict[str, List[Tensor]]],
    ) -> Dict[str, Tensor]:
        """
        Extract direction vectors from training samples.

        Args:
            samples: Dictionary mapping emotion to paired samples

        Returns:
            Dictionary mapping emotion to direction vector
        """
        extractor = DirectionVectorExtractor(self.speaker_encoder, self.config)
        directions = extractor.extract_all_emotions(samples)

        # Register with manipulator
        for emotion, direction in directions.items():
            self.manipulator.register_direction(emotion, direction)

        return directions

    def forward(
        self,
        features: Tensor,  # [batch, seq, input_dim]
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
        text_embedding: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with emotion control.

        Args:
            features: Audio features [batch, seq, input_dim]
            emotion: Target emotion (if specified directly)
            intensity: Emotion intensity (α parameter)
            text_embedding: Optional text embedding for text-based control

        Returns:
            Dictionary with:
                - speaker_emb: Original speaker embedding
                - controlled_emb: Emotion-controlled embedding
                - prosody_tokens: Prosody prefix tokens for TTS
        """
        batch_size = features.size(0)

        # Encode speaker embedding
        speaker_emb = self.speaker_encoder(features)  # [batch, speaker_dim]

        # Apply emotion control
        if emotion is not None:
            # Direct emotion specification
            controlled_emb = self.manipulator(speaker_emb, emotion, intensity)
        elif text_embedding is not None and self.text_extractor is not None:
            # Text-based emotion control
            direction = self.text_extractor.predict_direction(text_embedding)
            if intensity is None:
                intensity = self.config.default_intensity
            controlled_emb = speaker_emb + intensity * direction
        else:
            # No emotion control (neutral)
            controlled_emb = speaker_emb

        # Project to prosody tokens
        tokens_flat = self.output_proj(controlled_emb)  # [batch, num_tokens * output_dim]
        prosody_tokens = tokens_flat.view(
            batch_size,
            self.config.num_prosody_tokens,
            self.config.output_dim
        )

        return {
            "speaker_emb": speaker_emb,
            "controlled_emb": controlled_emb,
            "prosody_tokens": prosody_tokens,
        }

    def from_emotion(
        self,
        features: Tensor,
        emotion: str,
        intensity: float = None,
    ) -> Dict[str, Tensor]:
        """Convenience method for emotion-based generation."""
        return self.forward(features, emotion=emotion, intensity=intensity)

    def from_text(
        self,
        features: Tensor,
        text_embedding: Tensor,
        intensity: float = None,
    ) -> Dict[str, Tensor]:
        """Convenience method for text-based generation."""
        return self.forward(features, text_embedding=text_embedding, intensity=intensity)

    def interpolate(
        self,
        features: Tensor,
        emotions: List[str],
        weights: List[float],
        intensity: float = None,
    ) -> Dict[str, Tensor]:
        """Generate with blended emotions."""
        batch_size = features.size(0)

        speaker_emb = self.speaker_encoder(features)
        controlled_emb = self.manipulator.interpolate_emotions(
            speaker_emb, emotions, weights, intensity
        )

        tokens_flat = self.output_proj(controlled_emb)
        prosody_tokens = tokens_flat.view(
            batch_size,
            self.config.num_prosody_tokens,
            self.config.output_dim
        )

        return {
            "speaker_emb": speaker_emb,
            "controlled_emb": controlled_emb,
            "prosody_tokens": prosody_tokens,
        }


# =============================================================================
# EMOKNOB ADAPTER FOR CSM PIPELINE
# =============================================================================

class EmoKnobAdapter(nn.Module):
    """
    Adapter for integrating EmoKnob with existing prosody pipeline.

    Provides drop-in replacement for other prosody encoders.
    """

    def __init__(
        self,
        config: EmoKnobConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Main EmoKnob module
        self.emoknob = EmoKnob(config)

        # Align output dimension if needed
        if config.output_dim != prosody_hidden:
            self.output_align = nn.Linear(config.output_dim, prosody_hidden)
        else:
            self.output_align = nn.Identity()

    def forward(
        self,
        features: Tensor,
        emotion: Optional[str] = None,
        intensity: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for pipeline integration.

        Args:
            features: Audio features [batch, seq, dim]
            emotion: Target emotion
            intensity: Emotion intensity

        Returns:
            Dictionary with prosody_tokens and other info
        """
        result = self.emoknob(features, emotion=emotion, intensity=intensity)

        # Align output
        result["prosody_tokens"] = self.output_align(result["prosody_tokens"])

        return result

    def from_emotion(
        self,
        features: Tensor,
        emotion: str,
        intensity: float = None,
    ) -> Dict[str, Tensor]:
        """Generate prosody tokens from emotion specification."""
        return self.forward(features, emotion=emotion, intensity=intensity)

    def sweep_intensities(
        self,
        features: Tensor,
        emotion: str,
        intensities: List[float] = None,
    ) -> Dict[str, List[Tensor]]:
        """
        Generate samples at different intensities for comparison.

        Args:
            features: Audio features
            emotion: Target emotion
            intensities: List of intensities to try

        Returns:
            Dictionary with lists of results per intensity
        """
        if intensities is None:
            intensities = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]

        results = {"intensities": intensities, "tokens": []}

        for intensity in intensities:
            result = self.forward(features, emotion=emotion, intensity=intensity)
            results["tokens"].append(result["prosody_tokens"])

        return results

    def load_directions(self, path: str):
        """Load pre-computed direction vectors."""
        self.emoknob.manipulator.load_directions(path)

    def save_directions(self, path: str):
        """Save computed direction vectors."""
        self.emoknob.manipulator.save_directions(path)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class EmoKnobLoss(nn.Module):
    """
    Loss functions for training EmoKnob.

    Components:
    1. Direction alignment loss: Ensure extracted directions match emotions
    2. Reconstruction loss: Speaker embeddings reconstruct audio features
    3. Contrastive loss: Different emotions have different directions
    4. Intensity consistency: Higher intensity → more change
    """

    def __init__(
        self,
        config: EmoKnobConfig,
        direction_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        consistency_weight: float = 0.3,
    ):
        super().__init__()
        self.config = config
        self.direction_weight = direction_weight
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight

        # Emotion classifier for direction alignment
        self.emotion_classifier = nn.Linear(config.speaker_dim, config.num_emotions)

    def forward(
        self,
        speaker_emb: Tensor,
        controlled_emb: Tensor,
        emotion_labels: Tensor,
        direction_vectors: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute training losses.

        Args:
            speaker_emb: Original speaker embeddings
            controlled_emb: Emotion-controlled embeddings
            emotion_labels: Ground truth emotion indices
            direction_vectors: Pre-computed direction vectors

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Direction alignment: classify emotion from direction
        direction = controlled_emb - speaker_emb
        emotion_logits = self.emotion_classifier(direction)
        losses["direction"] = F.cross_entropy(emotion_logits, emotion_labels)

        # Contrastive loss: different emotions should have different directions
        batch_size = direction.size(0)
        if batch_size > 1:
            direction_norm = F.normalize(direction, p=2, dim=-1)
            similarity = torch.matmul(direction_norm, direction_norm.T)  # [B, B]

            # Same emotion pairs should be similar
            labels = emotion_labels.unsqueeze(0) == emotion_labels.unsqueeze(1)
            labels = labels.float()

            # Push apart different emotions
            contrastive = -torch.log(
                torch.exp(similarity / 0.07).diag() /
                torch.exp(similarity / 0.07).sum(dim=-1)
            )
            losses["contrastive"] = contrastive.mean()
        else:
            losses["contrastive"] = torch.tensor(0.0, device=speaker_emb.device)

        # Intensity consistency: direction magnitude should scale with intensity
        direction_norm_val = torch.norm(direction, p=2, dim=-1)
        losses["consistency"] = direction_norm_val.var()  # Low variance = consistent

        # Total loss
        losses["total"] = (
            self.direction_weight * losses["direction"] +
            self.contrastive_weight * losses["contrastive"] +
            self.consistency_weight * losses["consistency"]
        )

        return losses


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_emoknob_adapter(
    config: Optional[EmoKnobConfig] = None,
    direction_cache_path: Optional[str] = None,
    prosody_hidden: int = 2048,
) -> EmoKnobAdapter:
    """
    Factory function to create EmoKnob adapter.

    Args:
        config: EmoKnob configuration
        direction_cache_path: Path to pre-computed directions
        prosody_hidden: Output dimension for prosody tokens

    Returns:
        Configured EmoKnobAdapter
    """
    if config is None:
        config = EmoKnobConfig()

    adapter = EmoKnobAdapter(config, prosody_hidden=prosody_hidden)

    if direction_cache_path is not None:
        adapter.load_directions(direction_cache_path)

    return adapter


def compute_emotion_statistics(
    samples: Dict[str, Dict[str, List[Tensor]]],
    speaker_encoder: SpeakerEncoder,
) -> Dict[str, Dict[str, Tensor]]:
    """
    Compute statistics for emotion direction vectors.

    Useful for analyzing the quality of extracted directions.
    """
    device = next(speaker_encoder.parameters()).device
    speaker_encoder.eval()

    stats = {}

    with torch.no_grad():
        for emotion, data in samples.items():
            if emotion == "neutral":
                continue

            emo_features = data["emotional"]
            neu_features = data["neutral"]

            directions = []
            for emo_feat, neu_feat in zip(emo_features, neu_features):
                emo_emb = speaker_encoder(emo_feat.to(device))
                neu_emb = speaker_encoder(neu_feat.to(device))
                diff = emo_emb - neu_emb
                directions.append(diff.squeeze(0))

            directions = torch.stack(directions)

            stats[emotion] = {
                "mean": directions.mean(dim=0),
                "std": directions.std(dim=0),
                "norm_mean": torch.norm(directions, p=2, dim=-1).mean(),
                "norm_std": torch.norm(directions, p=2, dim=-1).std(),
                "num_samples": len(directions),
            }

    return stats


def intensity_to_description(intensity: float) -> str:
    """Convert intensity value to human-readable description."""
    if intensity <= 0.1:
        return "neutral"
    elif intensity <= 0.3:
        return "slightly"
    elif intensity <= 0.5:
        return "mildly"
    elif intensity <= 0.7:
        return "moderately"
    elif intensity <= 1.0:
        return "strongly"
    elif intensity <= 1.3:
        return "very strongly"
    else:
        return "intensely"


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "EmoKnobConfig",
    "EmoKnob",
    "EmoKnobAdapter",
    "EmoKnobLoss",
    "SpeakerEncoder",
    "DirectionVectorExtractor",
    "EmotionDirectionManipulator",
    "TextEmotionExtractor",
    "create_emoknob_adapter",
    "compute_emotion_statistics",
    "intensity_to_description",
    "EMOKNOB_EMOTIONS",
    "EMOTION_TO_IDX",
    "IDX_TO_EMOTION",
]
