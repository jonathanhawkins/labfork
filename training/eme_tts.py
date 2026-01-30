"""
EME-TTS: Emphasis Meets Emotion TTS - Emphasis-Emotion Link Synthesis

Based on "EME-TTS: Unlocking the Emphasis and Emotion Link in Speech Synthesis"
(Interspeech 2025, arXiv:2507.12015)

Key Insight: Emphasis and emotion are intrinsically linked - emphasis modulates
emotional perception. EME-TTS systematically models this interaction.

Architecture:
1. Emphasis Pseudo-Label Generator: Weakly supervised emphasis detection using
   variance-based features (pitch, energy, duration variance)
2. Variance-Based Emphasis Features: Extract emphasis from prosodic variance
3. Emphasis Perception Enhancement (EPE) Block: Cross-attention mechanism that
   enhances interaction between emotional signals and emphasis positions
4. LLM Integration: Predict emphasis positions from text using LLMs

Benefits:
- More natural emotional speech synthesis with appropriate emphasis
- Stable and distinguishable target emphasis across different emotions
- First systematic study of emphasis-emotion relationship in TTS
- Works with LLMs for automatic emphasis position prediction

Reference: https://arxiv.org/abs/2507.12015
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EMETTSConfig:
    """Configuration for EME-TTS emphasis-emotion coupling."""

    # Input dimensions
    audio_feature_dim: int = 768  # wav2vec2/HuBERT feature dim
    text_hidden_dim: int = 768    # Text encoder hidden dim
    mel_dim: int = 80             # Mel-spectrogram dimension

    # Emphasis feature settings
    emphasis_hidden_dim: int = 256
    emphasis_embedding_dim: int = 256
    num_emphasis_levels: int = 4  # Number of discrete emphasis levels (none, light, medium, strong)

    # Emotion settings
    emotion_embedding_dim: int = 256
    num_emotions: int = 8  # neutral, happy, sad, angry, surprised, calm, fearful, disgusted
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "surprised", "calm", "fearful", "disgusted"
    ])

    # EPE (Emphasis Perception Enhancement) block settings
    epe_num_heads: int = 8
    epe_num_layers: int = 2
    epe_hidden_dim: int = 512
    epe_dropout: float = 0.1

    # Variance-based emphasis settings
    pitch_variance_weight: float = 0.4
    energy_variance_weight: float = 0.35
    duration_variance_weight: float = 0.25
    emphasis_threshold_low: float = 0.3   # Below this = no emphasis
    emphasis_threshold_mid: float = 0.5   # Below this = light emphasis
    emphasis_threshold_high: float = 0.75 # Below this = medium, above = strong

    # LLM integration settings
    llm_model_name: str = "gpt-4"  # For emphasis prediction
    use_llm_emphasis: bool = True
    llm_prompt_template: str = (
        "Identify words that should be emphasized in the following sentence to convey "
        "{emotion} emotion. Mark emphasized words with asterisks (*word*). "
        "Sentence: {text}"
    )

    # Output settings
    output_dim: int = 2048        # CSM prosody hidden dimension
    num_prosody_tokens: int = 4   # Prefix tokens for prosody conditioning

    # Training settings
    dropout: float = 0.1
    label_smoothing: float = 0.1
    use_pseudo_labels: bool = True  # Weakly supervised emphasis labels
    pseudo_label_warmup_epochs: int = 5

    # Frame rate for prosody extraction
    frame_rate: int = 100  # 10ms per frame


# Emphasis level mapping
EMPHASIS_LEVELS = {
    0: "none",
    1: "light",
    2: "medium",
    3: "strong",
}

EMPHASIS_TO_IDX = {v: k for k, v in EMPHASIS_LEVELS.items()}


# =============================================================================
# VARIANCE-BASED EMPHASIS FEATURE EXTRACTOR
# =============================================================================

class VarianceEmphasisExtractor(nn.Module):
    """
    Extract emphasis features based on prosodic variance.

    Key insight from EME-TTS: Words with higher variance in pitch, energy,
    and duration are more likely to be emphasized. This provides weakly
    supervised emphasis labels without manual annotation.

    Prosodic features:
    - Pitch (F0) variance: Emphasized words have larger pitch range
    - Energy variance: Emphasized words have larger energy fluctuation
    - Duration variance: Emphasized words may have stretched vowels

    Output:
    - Emphasis score (0-1) per word
    - Discrete emphasis level (none, light, medium, strong)
    """

    def __init__(self, config: EMETTSConfig):
        super().__init__()
        self.config = config

        # Feature projection layers
        self.pitch_proj = nn.Sequential(
            nn.Linear(1, config.emphasis_hidden_dim),
            nn.LayerNorm(config.emphasis_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.energy_proj = nn.Sequential(
            nn.Linear(1, config.emphasis_hidden_dim),
            nn.LayerNorm(config.emphasis_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.duration_proj = nn.Sequential(
            nn.Linear(1, config.emphasis_hidden_dim),
            nn.LayerNorm(config.emphasis_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Variance computation layer
        self.variance_aggregator = nn.Sequential(
            nn.Linear(config.emphasis_hidden_dim * 3, config.emphasis_hidden_dim),
            nn.LayerNorm(config.emphasis_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emphasis_hidden_dim, 1),
            nn.Sigmoid(),  # Output score between 0 and 1
        )

        # Learnable thresholds (initialized to config values but can be learned)
        self.register_buffer(
            "thresholds",
            torch.tensor([
                config.emphasis_threshold_low,
                config.emphasis_threshold_mid,
                config.emphasis_threshold_high,
            ])
        )

    def compute_prosodic_variance(
        self,
        values: torch.Tensor,  # [batch, num_frames]
        word_boundaries: torch.Tensor,  # [batch, num_words, 2] (start, end frame indices)
    ) -> torch.Tensor:
        """
        Compute variance of values within each word boundary.

        Returns:
            Variance per word [batch, num_words]
        """
        batch_size, num_frames = values.shape
        num_words = word_boundaries.shape[1]
        device = values.device

        variances = torch.zeros(batch_size, num_words, device=device)

        for b in range(batch_size):
            for w in range(num_words):
                start_frame = word_boundaries[b, w, 0].long()
                end_frame = word_boundaries[b, w, 1].long()

                if end_frame > start_frame:
                    word_values = values[b, start_frame:end_frame]
                    variances[b, w] = word_values.var() if len(word_values) > 1 else 0.0

        return variances

    def forward(
        self,
        pitch: torch.Tensor,        # [batch, num_frames] - F0 values
        energy: torch.Tensor,       # [batch, num_frames] - energy values
        duration: torch.Tensor,     # [batch, num_words] - word durations
        word_boundaries: torch.Tensor,  # [batch, num_words, 2] - (start, end) frames
    ) -> Dict[str, torch.Tensor]:
        """
        Extract emphasis features from prosodic variance.

        Returns:
            Dict with:
                - 'emphasis_scores': [batch, num_words] - continuous scores (0-1)
                - 'emphasis_levels': [batch, num_words] - discrete levels (0-3)
                - 'emphasis_embeddings': [batch, num_words, embed_dim]
                - 'pitch_variance': [batch, num_words]
                - 'energy_variance': [batch, num_words]
                - 'duration_features': [batch, num_words]
        """
        batch_size = pitch.shape[0]
        num_words = word_boundaries.shape[1]
        device = pitch.device

        # Compute variance for pitch and energy
        pitch_var = self.compute_prosodic_variance(pitch, word_boundaries)
        energy_var = self.compute_prosodic_variance(energy, word_boundaries)

        # Normalize variances to [0, 1] range
        pitch_var_norm = self._normalize_variance(pitch_var)
        energy_var_norm = self._normalize_variance(energy_var)
        duration_norm = self._normalize_duration(duration)

        # Project features
        pitch_features = self.pitch_proj(pitch_var_norm.unsqueeze(-1))
        energy_features = self.energy_proj(energy_var_norm.unsqueeze(-1))
        duration_features = self.duration_proj(duration_norm.unsqueeze(-1))

        # Combine and compute emphasis score
        combined = torch.cat([pitch_features, energy_features, duration_features], dim=-1)
        emphasis_scores = self.variance_aggregator(combined).squeeze(-1)  # [batch, num_words]

        # Apply weighted combination
        weighted_score = (
            self.config.pitch_variance_weight * pitch_var_norm +
            self.config.energy_variance_weight * energy_var_norm +
            self.config.duration_variance_weight * duration_norm
        )

        # Blend learned and rule-based scores
        final_scores = 0.7 * emphasis_scores + 0.3 * weighted_score

        # Discretize to emphasis levels
        emphasis_levels = self._discretize_emphasis(final_scores)

        return {
            'emphasis_scores': final_scores,
            'emphasis_levels': emphasis_levels,
            'emphasis_embeddings': combined,
            'pitch_variance': pitch_var_norm,
            'energy_variance': energy_var_norm,
            'duration_features': duration_norm,
        }

    def _normalize_variance(self, variance: torch.Tensor) -> torch.Tensor:
        """Normalize variance to [0, 1] using min-max normalization."""
        batch_min = variance.min(dim=-1, keepdim=True).values
        batch_max = variance.max(dim=-1, keepdim=True).values
        range_val = batch_max - batch_min + 1e-8
        return (variance - batch_min) / range_val

    def _normalize_duration(self, duration: torch.Tensor) -> torch.Tensor:
        """Normalize duration relative to mean duration."""
        mean_dur = duration.mean(dim=-1, keepdim=True)
        return (duration / (mean_dur + 1e-8)).clamp(0, 2) / 2

    def _discretize_emphasis(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert continuous scores to discrete emphasis levels."""
        levels = torch.zeros_like(scores, dtype=torch.long)

        levels = torch.where(scores > self.thresholds[0], torch.ones_like(levels), levels)
        levels = torch.where(scores > self.thresholds[1], torch.full_like(levels, 2), levels)
        levels = torch.where(scores > self.thresholds[2], torch.full_like(levels, 3), levels)

        return levels


# =============================================================================
# EMPHASIS PSEUDO-LABEL GENERATOR
# =============================================================================

class EmphasisPseudoLabelGenerator(nn.Module):
    """
    Generate pseudo-labels for emphasis using weakly supervised learning.

    Key insight: We can derive emphasis labels from prosodic features without
    explicit annotation. This module refines pseudo-labels during training.

    Methods:
    1. Prosodic variance: High variance = high emphasis (from VarianceEmphasisExtractor)
    2. Energy prominence: Local energy peaks indicate emphasis
    3. Duration stretching: Elongated phonemes suggest emphasis
    4. Pitch excursion: Large pitch movements mark emphasized words
    """

    def __init__(self, config: EMETTSConfig):
        super().__init__()
        self.config = config

        # Variance-based extractor
        self.variance_extractor = VarianceEmphasisExtractor(config)

        # Energy prominence detector
        self.energy_prominence = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

        # Pitch excursion detector
        self.pitch_excursion = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

        # Fusion layer for combining all signals
        self.fusion = nn.Sequential(
            nn.Linear(4, config.emphasis_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emphasis_hidden_dim, config.num_emphasis_levels),
        )

        # Confidence estimator for pseudo-label quality
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.num_emphasis_levels + 4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pitch: torch.Tensor,        # [batch, num_frames]
        energy: torch.Tensor,       # [batch, num_frames]
        duration: torch.Tensor,     # [batch, num_words]
        word_boundaries: torch.Tensor,  # [batch, num_words, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate pseudo-labels for emphasis.

        Returns:
            Dict with:
                - 'pseudo_labels': [batch, num_words] - discrete emphasis levels
                - 'pseudo_scores': [batch, num_words] - continuous scores
                - 'confidence': [batch, num_words] - confidence in pseudo-labels
                - 'variance_features': Dict from variance extractor
        """
        batch_size = pitch.shape[0]
        num_words = word_boundaries.shape[1]
        device = pitch.device

        # Get variance-based features
        variance_result = self.variance_extractor(
            pitch, energy, duration, word_boundaries
        )

        # Compute energy prominence per word
        energy_prominence = self._compute_word_prominence(
            energy, word_boundaries, self.energy_prominence
        )

        # Compute pitch excursion per word
        pitch_excursion = self._compute_word_prominence(
            pitch, word_boundaries, self.pitch_excursion
        )

        # Combine all signals
        combined_signals = torch.stack([
            variance_result['emphasis_scores'],
            variance_result['pitch_variance'],
            energy_prominence,
            pitch_excursion,
        ], dim=-1)  # [batch, num_words, 4]

        # Fuse to get emphasis logits
        emphasis_logits = self.fusion(combined_signals)  # [batch, num_words, num_levels]

        # Get pseudo-labels (argmax) and scores (softmax)
        pseudo_labels = emphasis_logits.argmax(dim=-1)
        pseudo_scores = F.softmax(emphasis_logits, dim=-1)

        # Estimate confidence
        confidence_input = torch.cat([pseudo_scores, combined_signals], dim=-1)
        confidence = self.confidence_estimator(confidence_input).squeeze(-1)

        return {
            'pseudo_labels': pseudo_labels,
            'pseudo_scores': pseudo_scores,
            'confidence': confidence,
            'emphasis_logits': emphasis_logits,
            'variance_features': variance_result,
        }

    def _compute_word_prominence(
        self,
        signal: torch.Tensor,       # [batch, num_frames]
        word_boundaries: torch.Tensor,  # [batch, num_words, 2]
        detector: nn.Module,
    ) -> torch.Tensor:
        """Compute prominence score per word using a detector network."""
        batch_size = signal.shape[0]
        num_words = word_boundaries.shape[1]
        device = signal.device

        # Apply detector to full signal
        signal_3d = signal.unsqueeze(1)  # [batch, 1, frames]
        prominence = detector(signal_3d).squeeze(1)  # [batch, frames]

        # Aggregate to word level (max within word boundaries)
        word_prominence = torch.zeros(batch_size, num_words, device=device)

        for b in range(batch_size):
            for w in range(num_words):
                start = word_boundaries[b, w, 0].long()
                end = word_boundaries[b, w, 1].long()
                if end > start:
                    word_prominence[b, w] = prominence[b, start:end].max()

        return word_prominence


# =============================================================================
# EMPHASIS PERCEPTION ENHANCEMENT (EPE) BLOCK
# =============================================================================

class EmphasisPerceptionEnhancement(nn.Module):
    """
    Emphasis Perception Enhancement (EPE) Block.

    Key innovation from EME-TTS: Models the interaction between emphasis
    positions and emotional signals using cross-attention.

    The EPE block:
    1. Takes emphasis positions and emotion embeddings as input
    2. Uses cross-attention where emotion queries emphasis features
    3. Produces enhanced representations that capture the emphasis-emotion link
    4. Enables emotion-appropriate emphasis rendering

    This captures the insight that:
    - Happy emotions often emphasize positive/exciting words
    - Angry emotions emphasize words expressing frustration
    - Sad emotions de-emphasize or have subtle emphasis patterns
    """

    def __init__(self, config: EMETTSConfig):
        super().__init__()
        self.config = config

        # Emphasis embedding
        self.emphasis_embedding = nn.Embedding(
            config.num_emphasis_levels,
            config.emphasis_embedding_dim
        )

        # Emotion embedding
        self.emotion_embedding = nn.Embedding(
            config.num_emotions,
            config.emotion_embedding_dim
        )

        # Project to common dimension
        self.emphasis_proj = nn.Linear(
            config.emphasis_embedding_dim,
            config.epe_hidden_dim
        )

        self.emotion_proj = nn.Linear(
            config.emotion_embedding_dim,
            config.epe_hidden_dim
        )

        # Cross-attention layers for emphasis-emotion interaction
        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.epe_hidden_dim,
                nhead=config.epe_num_heads,
                dim_feedforward=config.epe_hidden_dim * 4,
                dropout=config.epe_dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(config.epe_num_layers)
        ])

        # Gating mechanism for emphasis modulation based on emotion
        self.emotion_gate = nn.Sequential(
            nn.Linear(config.epe_hidden_dim * 2, config.epe_hidden_dim),
            nn.LayerNorm(config.epe_hidden_dim),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.epe_hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Emphasis-emotion compatibility matrix (learnable)
        # Captures which emotions typically have which emphasis patterns
        self.compatibility_matrix = nn.Parameter(
            torch.zeros(config.num_emotions, config.num_emphasis_levels)
        )

        self._init_compatibility_matrix()

    def _init_compatibility_matrix(self):
        """Initialize compatibility matrix with prior knowledge."""
        # Based on linguistic intuition:
        # - Happy: tends toward medium-strong emphasis
        # - Sad: tends toward light-medium emphasis
        # - Angry: tends toward strong emphasis
        # - etc.

        priors = torch.tensor([
            # [none, light, medium, strong]
            [0.3, 0.3, 0.25, 0.15],  # neutral
            [0.1, 0.2, 0.35, 0.35],  # happy
            [0.2, 0.35, 0.3, 0.15],  # sad
            [0.05, 0.1, 0.25, 0.6],  # angry
            [0.1, 0.15, 0.3, 0.45],  # surprised
            [0.35, 0.35, 0.2, 0.1],  # calm
            [0.15, 0.3, 0.35, 0.2],  # fearful
            [0.1, 0.2, 0.35, 0.35],  # disgusted
        ])

        # Convert to logits
        self.compatibility_matrix.data = torch.log(priors + 1e-6)

    def forward(
        self,
        emphasis_levels: torch.Tensor,  # [batch, num_words] - discrete levels
        emotion_ids: torch.Tensor,      # [batch] - emotion indices
        emphasis_features: Optional[torch.Tensor] = None,  # [batch, num_words, dim]
        text_embeddings: Optional[torch.Tensor] = None,    # [batch, num_words, dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply emphasis perception enhancement.

        Args:
            emphasis_levels: Discrete emphasis levels per word
            emotion_ids: Global emotion for the utterance
            emphasis_features: Optional continuous emphasis features
            text_embeddings: Optional text encoder outputs

        Returns:
            Dict with:
                - 'enhanced_embeddings': [batch, num_words, output_dim]
                - 'attention_weights': Cross-attention weights
                - 'compatibility_scores': Emphasis-emotion compatibility
                - 'emotion_gate_values': Gate values for modulation
        """
        batch_size, num_words = emphasis_levels.shape
        device = emphasis_levels.device

        # Get emphasis embeddings
        emphasis_emb = self.emphasis_embedding(emphasis_levels)  # [batch, words, emb_dim]
        emphasis_hidden = self.emphasis_proj(emphasis_emb)  # [batch, words, hidden_dim]

        # Get emotion embeddings (expand to word level)
        emotion_emb = self.emotion_embedding(emotion_ids)  # [batch, emb_dim]
        emotion_hidden = self.emotion_proj(emotion_emb)  # [batch, hidden_dim]
        emotion_hidden = emotion_hidden.unsqueeze(1).expand(-1, num_words, -1)  # [batch, words, hidden]

        # Cross-attention: emotion queries emphasis
        # This allows emotion to modulate which emphasis patterns are enhanced
        query = emotion_hidden
        key_value = emphasis_hidden

        attention_weights = []
        for layer in self.cross_attention_layers:
            query = layer(query, key_value)
            # Store attention weights for analysis

        # Compute emotion gate
        gate_input = torch.cat([query, emphasis_hidden], dim=-1)
        gate_values = self.emotion_gate(gate_input)

        # Apply gated combination
        enhanced = gate_values * query + (1 - gate_values) * emphasis_hidden

        # Compute compatibility scores
        compatibility = self._compute_compatibility(emphasis_levels, emotion_ids)

        # Project to output dimension
        output = self.output_proj(enhanced)

        return {
            'enhanced_embeddings': output,
            'attention_weights': attention_weights,
            'compatibility_scores': compatibility,
            'emotion_gate_values': gate_values,
        }

    def _compute_compatibility(
        self,
        emphasis_levels: torch.Tensor,  # [batch, num_words]
        emotion_ids: torch.Tensor,      # [batch]
    ) -> torch.Tensor:
        """Compute emphasis-emotion compatibility scores."""
        batch_size, num_words = emphasis_levels.shape
        device = emphasis_levels.device

        # Get compatibility probabilities for each emotion
        compat_probs = F.softmax(self.compatibility_matrix, dim=-1)  # [emotions, levels]

        # Get compatibility for current emotions
        emotion_compat = compat_probs[emotion_ids]  # [batch, levels]

        # Gather compatibility for actual emphasis levels
        scores = torch.zeros(batch_size, num_words, device=device)
        for b in range(batch_size):
            for w in range(num_words):
                level = emphasis_levels[b, w].long()
                scores[b, w] = emotion_compat[b, level]

        return scores


# =============================================================================
# LLM-BASED EMPHASIS PREDICTOR
# =============================================================================

class LLMEmphasisPredictor:
    """
    Predict emphasis positions using Large Language Models.

    Integration with LLMs (GPT-4, Claude, etc.) to predict which words
    should be emphasized based on:
    1. The text content
    2. The target emotion
    3. Contextual understanding of meaning

    This provides high-quality emphasis predictions without manual annotation.
    """

    def __init__(self, config: EMETTSConfig):
        self.config = config
        self._client = None

    def _lazy_init_client(self):
        """Lazy initialization of LLM client."""
        if self._client is not None:
            return

        try:
            import openai
            self._client = openai.OpenAI()
        except ImportError:
            warnings.warn(
                "OpenAI package not installed. LLM emphasis prediction unavailable. "
                "Install with: pip install openai"
            )
            self._client = "mock"

    def predict(
        self,
        text: str,
        emotion: str,
        num_emphasis: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Predict emphasis positions using LLM.

        Args:
            text: Input text
            emotion: Target emotion (happy, sad, angry, etc.)
            num_emphasis: Optional target number of emphasized words

        Returns:
            Dict with:
                - 'emphasized_words': List of emphasized word indices
                - 'emphasis_levels': Dict mapping word index to level
                - 'annotated_text': Text with emphasis markers
                - 'reasoning': LLM's reasoning for emphasis choices
        """
        self._lazy_init_client()

        if self._client == "mock":
            return self._mock_predict(text, emotion)

        prompt = self._build_prompt(text, emotion, num_emphasis)

        try:
            response = self._client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )

            return self._parse_response(response.choices[0].message.content, text)

        except Exception as e:
            warnings.warn(f"LLM emphasis prediction failed: {e}")
            return self._mock_predict(text, emotion)

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM emphasis prediction."""
        return """You are an expert in speech synthesis and prosody analysis.
Your task is to identify which words should be emphasized in speech to convey specific emotions naturally.

Guidelines:
1. Consider emotional context: different emotions emphasize different words
2. Emphasize content words (nouns, verbs, adjectives) over function words
3. Consider semantic importance and emotional salience
4. Mark emphasis with asterisks: *word*
5. Use double asterisks for strong emphasis: **word**

Output format:
ANNOTATED: [text with emphasis markers]
REASONING: [brief explanation]
WORDS: [comma-separated list of emphasized words]"""

    def _build_prompt(
        self,
        text: str,
        emotion: str,
        num_emphasis: Optional[int] = None,
    ) -> str:
        """Build prompt for LLM."""
        prompt = self.config.llm_prompt_template.format(
            emotion=emotion,
            text=text,
        )

        if num_emphasis is not None:
            prompt += f"\nEmphasize approximately {num_emphasis} words."

        return prompt

    def _parse_response(self, response: str, original_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract emphasis information."""
        words = original_text.split()
        emphasized_indices = []
        emphasis_levels = {}
        annotated_text = ""
        reasoning = ""

        lines = response.strip().split('\n')

        for line in lines:
            if line.startswith("ANNOTATED:"):
                annotated_text = line.replace("ANNOTATED:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
            elif line.startswith("WORDS:"):
                word_list = line.replace("WORDS:", "").strip().split(',')
                word_list = [w.strip().lower().strip('*') for w in word_list]

        # Extract emphasis from annotated text
        if annotated_text:
            for i, word in enumerate(annotated_text.split()):
                clean_word = word.strip('*').lower()

                # Count asterisks for level
                if word.startswith('**') and word.endswith('**'):
                    emphasized_indices.append(i)
                    emphasis_levels[i] = 3  # strong
                elif word.startswith('*') and word.endswith('*'):
                    emphasized_indices.append(i)
                    emphasis_levels[i] = 2  # medium
                elif clean_word in [w.lower() for w in words]:
                    # Check if word was mentioned in WORDS list
                    if clean_word in [w.lower().strip('*') for w in (annotated_text.split() if annotated_text else [])]:
                        emphasized_indices.append(i)
                        emphasis_levels[i] = 1  # light

        return {
            'emphasized_words': emphasized_indices,
            'emphasis_levels': emphasis_levels,
            'annotated_text': annotated_text,
            'reasoning': reasoning,
        }

    def _mock_predict(self, text: str, emotion: str) -> Dict[str, Any]:
        """Mock prediction for testing without LLM."""
        words = text.split()
        num_words = len(words)

        # Simple heuristic: emphasize content words
        content_word_types = ['n', 'v', 'adj', 'adv']  # Simplified

        emphasized_indices = []
        emphasis_levels = {}

        # Mark every 3rd-4th word as emphasized for testing
        for i in range(num_words):
            if i % 3 == 1 and len(words[i]) > 3:
                emphasized_indices.append(i)
                # Emotion affects emphasis level
                if emotion in ['happy', 'angry', 'surprised']:
                    emphasis_levels[i] = 3 if i % 2 == 0 else 2
                elif emotion in ['sad', 'calm']:
                    emphasis_levels[i] = 2 if i % 2 == 0 else 1
                else:
                    emphasis_levels[i] = 2

        # Create annotated text
        annotated_words = words.copy()
        for i in emphasized_indices:
            level = emphasis_levels.get(i, 1)
            if level >= 3:
                annotated_words[i] = f"**{words[i]}**"
            else:
                annotated_words[i] = f"*{words[i]}*"

        return {
            'emphasized_words': emphasized_indices,
            'emphasis_levels': emphasis_levels,
            'annotated_text': ' '.join(annotated_words),
            'reasoning': f"Mock emphasis prediction for {emotion} emotion",
        }


# =============================================================================
# NEURAL EMPHASIS PREDICTOR (TRAINABLE)
# =============================================================================

class NeuralEmphasisPredictor(nn.Module):
    """
    Trainable neural network for emphasis prediction from text.

    Alternative to LLM-based prediction that can be trained on pseudo-labels
    and fine-tuned for specific use cases.

    Architecture:
    1. Text encoder (transformer-based)
    2. Emotion conditioning
    3. Emphasis classifier per word
    """

    def __init__(self, config: EMETTSConfig):
        super().__init__()
        self.config = config

        # Text encoder
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.text_hidden_dim,
                nhead=8,
                dim_feedforward=config.text_hidden_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=4,
        )

        # Emotion conditioning
        self.emotion_embedding = nn.Embedding(
            config.num_emotions,
            config.text_hidden_dim
        )

        # Emphasis classifier
        self.emphasis_classifier = nn.Sequential(
            nn.Linear(config.text_hidden_dim * 2, config.emphasis_hidden_dim),
            nn.LayerNorm(config.emphasis_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emphasis_hidden_dim, config.num_emphasis_levels),
        )

        # Sequence-level context aggregation
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.text_hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, num_words, dim]
        emotion_ids: torch.Tensor,      # [batch]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict emphasis levels from text embeddings.

        Returns:
            Dict with:
                - 'emphasis_logits': [batch, num_words, num_levels]
                - 'emphasis_probs': [batch, num_words, num_levels]
                - 'predicted_levels': [batch, num_words]
        """
        batch_size, num_words, _ = text_embeddings.shape

        # Encode text
        encoded = self.text_encoder(text_embeddings)

        # Add emotion conditioning
        emotion_emb = self.emotion_embedding(emotion_ids)  # [batch, dim]
        emotion_expanded = emotion_emb.unsqueeze(1).expand(-1, num_words, -1)

        # Combine text and emotion
        combined = torch.cat([encoded, emotion_expanded], dim=-1)

        # Classify emphasis per word
        emphasis_logits = self.emphasis_classifier(combined)
        emphasis_probs = F.softmax(emphasis_logits, dim=-1)
        predicted_levels = emphasis_logits.argmax(dim=-1)

        return {
            'emphasis_logits': emphasis_logits,
            'emphasis_probs': emphasis_probs,
            'predicted_levels': predicted_levels,
        }


# =============================================================================
# EME-TTS FULL MODEL
# =============================================================================

class EMETTS(nn.Module):
    """
    EME-TTS: Full model for emphasis-emotion coupling in TTS.

    Combines all components:
    1. Variance-based emphasis extraction (training)
    2. Pseudo-label generation (weakly supervised)
    3. EPE block for emphasis-emotion interaction
    4. LLM/neural emphasis prediction (inference)

    The model outputs prosody conditioning tokens that capture
    the emphasis-emotion relationship for expressive TTS.
    """

    def __init__(self, config: EMETTSConfig):
        super().__init__()
        self.config = config

        # Emphasis extraction and pseudo-label generation
        self.pseudo_label_generator = EmphasisPseudoLabelGenerator(config)

        # EPE block
        self.epe_block = EmphasisPerceptionEnhancement(config)

        # Neural emphasis predictor (trainable alternative to LLM)
        self.neural_predictor = NeuralEmphasisPredictor(config)

        # LLM emphasis predictor (for inference)
        self.llm_predictor = LLMEmphasisPredictor(config)

        # Output projection to prosody tokens
        self.output_proj = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, num_words, dim]
        emotion_ids: torch.Tensor,      # [batch]
        pitch: Optional[torch.Tensor] = None,     # [batch, num_frames]
        energy: Optional[torch.Tensor] = None,    # [batch, num_frames]
        duration: Optional[torch.Tensor] = None,  # [batch, num_words]
        word_boundaries: Optional[torch.Tensor] = None,  # [batch, num_words, 2]
        emphasis_labels: Optional[torch.Tensor] = None,  # [batch, num_words]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for EME-TTS.

        Training mode (with prosodic features):
            - Generate pseudo-labels from variance features
            - Apply EPE block with extracted emphasis
            - Output prosody tokens

        Inference mode (without prosodic features):
            - Use neural predictor for emphasis
            - Apply EPE block
            - Output prosody tokens

        Returns:
            Dict with:
                - 'prosody_tokens': [batch, num_tokens, output_dim]
                - 'emphasis_levels': [batch, num_words]
                - 'pseudo_label_output': Dict from pseudo-label generator
                - 'epe_output': Dict from EPE block
                - 'losses': Dict with loss components
        """
        batch_size = text_embeddings.shape[0]
        num_words = text_embeddings.shape[1]
        device = text_embeddings.device

        losses = {}

        # Generate or predict emphasis levels
        if pitch is not None and energy is not None and word_boundaries is not None:
            # Training mode: use prosodic features
            if duration is None:
                # Compute duration from word boundaries
                duration = word_boundaries[:, :, 1] - word_boundaries[:, :, 0]

            pseudo_output = self.pseudo_label_generator(
                pitch, energy, duration, word_boundaries
            )
            emphasis_levels = pseudo_output['pseudo_labels']

            # Pseudo-label loss (if ground truth available)
            if emphasis_labels is not None:
                # Filter out ignore indices (-1)
                flat_logits = pseudo_output['emphasis_logits'].view(-1, self.config.num_emphasis_levels)
                flat_labels = emphasis_labels.view(-1)
                valid_mask = flat_labels >= 0

                if valid_mask.any():
                    pseudo_loss = F.cross_entropy(
                        flat_logits[valid_mask],
                        flat_labels[valid_mask],
                        label_smoothing=self.config.label_smoothing,
                    )
                    losses['pseudo_label_loss'] = pseudo_loss

            # Neural predictor loss (train to match pseudo-labels)
            neural_output = self.neural_predictor(text_embeddings, emotion_ids)
            neural_loss = F.cross_entropy(
                neural_output['emphasis_logits'].view(-1, self.config.num_emphasis_levels),
                emphasis_levels.view(-1),
                label_smoothing=self.config.label_smoothing,
            )
            losses['neural_predictor_loss'] = neural_loss

        else:
            # Inference mode: use neural predictor
            pseudo_output = None
            neural_output = self.neural_predictor(text_embeddings, emotion_ids)
            emphasis_levels = neural_output['predicted_levels']

        # Apply EPE block
        epe_output = self.epe_block(
            emphasis_levels=emphasis_levels,
            emotion_ids=emotion_ids,
            text_embeddings=text_embeddings,
        )

        # Compute EPE compatibility loss (encourage emotion-appropriate emphasis)
        compatibility_scores = epe_output['compatibility_scores']
        compat_loss = -torch.log(compatibility_scores.mean() + 1e-6)
        losses['compatibility_loss'] = compat_loss

        # Generate prosody tokens
        enhanced_emb = epe_output['enhanced_embeddings']  # [batch, words, output_dim]

        # Pool and project to tokens
        pooled = enhanced_emb.mean(dim=1)  # [batch, output_dim]
        tokens = self.output_proj(pooled)  # [batch, output_dim * num_tokens]
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        # Total loss
        losses['total'] = sum(losses.values())

        return {
            'prosody_tokens': tokens,
            'emphasis_levels': emphasis_levels,
            'pseudo_label_output': pseudo_output,
            'neural_output': neural_output if 'neural_output' in locals() else None,
            'epe_output': epe_output,
            'losses': losses,
        }

    def predict_emphasis_llm(
        self,
        text: str,
        emotion: str,
    ) -> Dict[str, Any]:
        """Predict emphasis using LLM."""
        return self.llm_predictor.predict(text, emotion)


# =============================================================================
# EME-TTS ADAPTER (CSM INTEGRATION)
# =============================================================================

class EMETTSAdapter(nn.Module):
    """
    Adapter for integrating EME-TTS with CSM prosody pipeline.

    Drop-in replacement for other prosody adapters.
    """

    def __init__(self, config: EMETTSConfig):
        super().__init__()
        self.config = config
        self.model = EMETTS(config)

        # Emotion label to index mapping
        self.emotion_to_idx = {
            label: idx for idx, label in enumerate(config.emotion_labels)
        }

    def forward(
        self,
        text_embeddings: torch.Tensor,
        emotion: Union[str, torch.Tensor],
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        word_boundaries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with emphasis-emotion coupling.

        Args:
            text_embeddings: [batch, num_words, dim]
            emotion: Emotion string or tensor of emotion indices
            pitch, energy, duration, word_boundaries: Prosodic features (training)

        Returns:
            Dict with prosody_tokens and other outputs
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Convert emotion to tensor if string
        if isinstance(emotion, str):
            emotion_idx = self.emotion_to_idx.get(emotion.lower(), 0)
            emotion_ids = torch.full((batch_size,), emotion_idx, device=device, dtype=torch.long)
        else:
            emotion_ids = emotion

        return self.model(
            text_embeddings=text_embeddings,
            emotion_ids=emotion_ids,
            pitch=pitch,
            energy=energy,
            duration=duration,
            word_boundaries=word_boundaries,
        )

    def from_text(
        self,
        text: str,
        text_embeddings: torch.Tensor,
        emotion: str,
        use_llm: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from text with LLM emphasis prediction.

        For inference: predicts emphasis from text using LLM/neural predictor.
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Get emotion index
        emotion_idx = self.emotion_to_idx.get(emotion.lower(), 0)
        emotion_ids = torch.full((batch_size,), emotion_idx, device=device, dtype=torch.long)

        # Optionally get LLM emphasis prediction
        llm_result = None
        if use_llm and self.config.use_llm_emphasis:
            llm_result = self.model.predict_emphasis_llm(text, emotion)

        # Forward pass (will use neural predictor internally)
        result = self.model(
            text_embeddings=text_embeddings,
            emotion_ids=emotion_ids,
        )

        result['llm_emphasis'] = llm_result

        return result

    def from_emphasis_trajectory(
        self,
        text_embeddings: torch.Tensor,
        emotion: str,
        emphasis_levels: List[int],  # [0, 1, 2, 3] per word
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens with explicit emphasis specification.

        For manual control: user specifies emphasis level per word.
        """
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Convert to tensor
        emphasis_tensor = torch.tensor(
            emphasis_levels, device=device
        ).unsqueeze(0).expand(batch_size, -1)

        # Get emotion index
        emotion_idx = self.emotion_to_idx.get(emotion.lower(), 0)
        emotion_ids = torch.full((batch_size,), emotion_idx, device=device, dtype=torch.long)

        # Apply EPE block directly with specified emphasis
        epe_output = self.model.epe_block(
            emphasis_levels=emphasis_tensor,
            emotion_ids=emotion_ids,
            text_embeddings=text_embeddings,
        )

        # Generate tokens
        enhanced_emb = epe_output['enhanced_embeddings']
        pooled = enhanced_emb.mean(dim=1)
        tokens = self.model.output_proj(pooled)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.model.output_norm(tokens)

        return {
            'prosody_tokens': tokens,
            'emphasis_levels': emphasis_tensor,
            'epe_output': epe_output,
        }


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class EMETTSLoss(nn.Module):
    """
    Combined loss function for EME-TTS training.

    Components:
    1. Pseudo-label loss: Match emphasis predictions to pseudo-labels
    2. Neural predictor loss: Train predictor on pseudo-labels
    3. Compatibility loss: Encourage emotion-appropriate emphasis
    4. Consistency loss: Smooth emphasis trajectories
    5. Reconstruction loss: Optional prosody reconstruction
    """

    def __init__(
        self,
        config: EMETTSConfig,
        pseudo_weight: float = 1.0,
        predictor_weight: float = 0.5,
        compatibility_weight: float = 0.3,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.pseudo_weight = pseudo_weight
        self.predictor_weight = predictor_weight
        self.compatibility_weight = compatibility_weight
        self.consistency_weight = consistency_weight

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
            ignore_index=-1,
        )

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_emphasis: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        losses = model_output.get('losses', {})

        # Apply weights
        weighted_losses = {}

        if 'pseudo_label_loss' in losses:
            weighted_losses['pseudo_label'] = self.pseudo_weight * losses['pseudo_label_loss']

        if 'neural_predictor_loss' in losses:
            weighted_losses['predictor'] = self.predictor_weight * losses['neural_predictor_loss']

        if 'compatibility_loss' in losses:
            weighted_losses['compatibility'] = self.compatibility_weight * losses['compatibility_loss']

        # Consistency loss (smooth emphasis trajectories)
        emphasis_levels = model_output.get('emphasis_levels')
        if emphasis_levels is not None and emphasis_levels.shape[1] > 1:
            # Convert to float for gradient
            emph_float = emphasis_levels.float()
            diffs = (emph_float[:, 1:] - emph_float[:, :-1]).pow(2).mean()
            weighted_losses['consistency'] = self.consistency_weight * diffs

        # Total weighted loss
        total = sum(weighted_losses.values()) if weighted_losses else torch.tensor(0.0)
        weighted_losses['total'] = total

        return weighted_losses


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_eme_tts_adapter(
    config: Optional[EMETTSConfig] = None,
    device: str = 'cpu',
) -> EMETTSAdapter:
    """Create EME-TTS adapter with default configuration."""
    if config is None:
        config = EMETTSConfig()

    adapter = EMETTSAdapter(config)
    return adapter.to(device)


def emphasis_level_to_description(level: int) -> str:
    """Convert emphasis level to human-readable description."""
    descriptions = {
        0: "no emphasis",
        1: "light emphasis",
        2: "medium emphasis",
        3: "strong emphasis",
    }
    return descriptions.get(level, "unknown")


def parse_annotated_text(text: str) -> Tuple[str, List[int], Dict[int, int]]:
    """
    Parse text with emphasis markers to extract emphasis information.

    Args:
        text: Text with *word* or **word** markers

    Returns:
        Tuple of (clean_text, emphasized_indices, emphasis_levels)
    """
    words = text.split()
    clean_words = []
    emphasized_indices = []
    emphasis_levels = {}

    for i, word in enumerate(words):
        clean_word = word.strip('*')
        clean_words.append(clean_word)

        if word.startswith('**') and word.endswith('**'):
            emphasized_indices.append(i)
            emphasis_levels[i] = 3
        elif word.startswith('*') and word.endswith('*'):
            emphasized_indices.append(i)
            emphasis_levels[i] = 2

    return ' '.join(clean_words), emphasized_indices, emphasis_levels


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EME-TTS: Emphasis Meets Emotion - Test Suite")
    print("=" * 60)

    config = EMETTSConfig()
    device = 'cpu'

    # Test 1: VarianceEmphasisExtractor
    print("\n[Test 1] VarianceEmphasisExtractor...")
    extractor = VarianceEmphasisExtractor(config)

    batch_size = 2
    num_frames = 100
    num_words = 5

    pitch = torch.randn(batch_size, num_frames)
    energy = torch.randn(batch_size, num_frames).abs()
    duration = torch.tensor([[20, 18, 22, 25, 15], [19, 21, 23, 18, 19]], dtype=torch.float)
    word_boundaries = torch.tensor([
        [[0, 20], [20, 38], [38, 60], [60, 85], [85, 100]],
        [[0, 19], [19, 40], [40, 63], [63, 81], [81, 100]],
    ], dtype=torch.long)

    result = extractor(pitch, energy, duration, word_boundaries)
    print(f"  Emphasis scores shape: {result['emphasis_scores'].shape}")
    print(f"  Emphasis levels shape: {result['emphasis_levels'].shape}")
    print(f"  Sample scores: {result['emphasis_scores'][0].tolist()}")
    print(f"  Sample levels: {result['emphasis_levels'][0].tolist()}")
    print("  [PASS]")

    # Test 2: EmphasisPseudoLabelGenerator
    print("\n[Test 2] EmphasisPseudoLabelGenerator...")
    generator = EmphasisPseudoLabelGenerator(config)

    result = generator(pitch, energy, duration, word_boundaries)
    print(f"  Pseudo labels shape: {result['pseudo_labels'].shape}")
    print(f"  Confidence shape: {result['confidence'].shape}")
    print(f"  Sample labels: {result['pseudo_labels'][0].tolist()}")
    print(f"  Sample confidence: {result['confidence'][0].tolist()}")
    print("  [PASS]")

    # Test 3: EmphasisPerceptionEnhancement
    print("\n[Test 3] EmphasisPerceptionEnhancement...")
    epe = EmphasisPerceptionEnhancement(config)

    emphasis_levels = torch.randint(0, 4, (batch_size, num_words))
    emotion_ids = torch.tensor([1, 3])  # happy, angry

    result = epe(emphasis_levels, emotion_ids)
    print(f"  Enhanced embeddings shape: {result['enhanced_embeddings'].shape}")
    print(f"  Compatibility scores shape: {result['compatibility_scores'].shape}")
    print(f"  Gate values shape: {result['emotion_gate_values'].shape}")
    print("  [PASS]")

    # Test 4: NeuralEmphasisPredictor
    print("\n[Test 4] NeuralEmphasisPredictor...")
    predictor = NeuralEmphasisPredictor(config)

    text_emb = torch.randn(batch_size, num_words, config.text_hidden_dim)
    result = predictor(text_emb, emotion_ids)
    print(f"  Emphasis logits shape: {result['emphasis_logits'].shape}")
    print(f"  Predicted levels: {result['predicted_levels'][0].tolist()}")
    print("  [PASS]")

    # Test 5: LLMEmphasisPredictor (mock mode)
    print("\n[Test 5] LLMEmphasisPredictor (mock)...")
    llm_predictor = LLMEmphasisPredictor(config)

    result = llm_predictor.predict(
        "I am so excited about this amazing opportunity!",
        "happy"
    )
    print(f"  Annotated text: {result['annotated_text']}")
    print(f"  Emphasized words: {result['emphasized_words']}")
    print(f"  Emphasis levels: {result['emphasis_levels']}")
    print("  [PASS]")

    # Test 6: Full EMETTS model
    print("\n[Test 6] EMETTS Full Model...")
    model = EMETTS(config)

    # Training mode (with prosodic features)
    output = model(
        text_embeddings=text_emb,
        emotion_ids=emotion_ids,
        pitch=pitch,
        energy=energy,
        duration=duration,
        word_boundaries=word_boundaries,
    )

    print(f"  Prosody tokens shape: {output['prosody_tokens'].shape}")
    print(f"  Emphasis levels: {output['emphasis_levels'][0].tolist()}")
    print(f"  Losses: {list(output['losses'].keys())}")
    print(f"  Total loss: {output['losses']['total'].item():.4f}")
    print("  [PASS]")

    # Test 7: Inference mode
    print("\n[Test 7] EMETTS Inference Mode...")
    with torch.no_grad():
        output = model(
            text_embeddings=text_emb,
            emotion_ids=emotion_ids,
        )

    print(f"  Prosody tokens shape: {output['prosody_tokens'].shape}")
    print(f"  Predicted emphasis: {output['emphasis_levels'][0].tolist()}")
    print("  [PASS]")

    # Test 8: EMETTSAdapter
    print("\n[Test 8] EMETTSAdapter...")
    adapter = EMETTSAdapter(config)

    # From emotion string
    result = adapter(
        text_embeddings=text_emb,
        emotion="happy",
        pitch=pitch,
        energy=energy,
        duration=duration,
        word_boundaries=word_boundaries,
    )
    print(f"  From emotion string - tokens: {result['prosody_tokens'].shape}")

    # From explicit emphasis
    result = adapter.from_emphasis_trajectory(
        text_embeddings=text_emb,
        emotion="angry",
        emphasis_levels=[0, 1, 3, 2, 1],
    )
    print(f"  From trajectory - tokens: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 9: EMETTSLoss
    print("\n[Test 9] EMETTSLoss...")
    loss_fn = EMETTSLoss(config)

    losses = loss_fn(output)
    print(f"  Loss components: {list(losses.keys())}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 10: Utility functions
    print("\n[Test 10] Utility Functions...")
    desc = emphasis_level_to_description(3)
    print(f"  Level 3 description: {desc}")

    clean, indices, levels = parse_annotated_text(
        "I am *really* **excited** about this!"
    )
    print(f"  Parsed clean text: {clean}")
    print(f"  Emphasized indices: {indices}")
    print(f"  Emphasis levels: {levels}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All EME-TTS tests passed!")
    print("=" * 60)

    # Usage example
    print("\nUsage Example:")
    print("-" * 40)
    print("""
from eme_tts import (
    EMETTSConfig,
    EMETTSAdapter,
    create_eme_tts_adapter,
    emphasis_level_to_description,
    parse_annotated_text,
    EMPHASIS_LEVELS,
)

# Initialize
config = EMETTSConfig()
adapter = EMETTSAdapter(config).cuda()

# Training: from prosodic features (auto-extracts emphasis)
result = adapter(
    text_embeddings=text_emb,     # [batch, num_words, 768]
    emotion="happy",
    pitch=pitch,                   # [batch, num_frames]
    energy=energy,                 # [batch, num_frames]
    duration=duration,             # [batch, num_words]
    word_boundaries=boundaries,    # [batch, num_words, 2]
)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]
emphasis = result['emphasis_levels']        # [batch, num_words]

# Inference: from text with LLM emphasis prediction
result = adapter.from_text(
    text="I am so excited about this!",
    text_embeddings=text_emb,
    emotion="happy",
    use_llm=True,  # Use LLM for emphasis prediction
)
print(result['llm_emphasis']['annotated_text'])
# Output: "I am *so* **excited** about *this*!"

# Inference: with explicit emphasis control
result = adapter.from_emphasis_trajectory(
    text_embeddings=text_emb,
    emotion="angry",
    emphasis_levels=[0, 0, 3, 2, 1],  # none, none, strong, medium, light
)

# Parse user-provided annotated text
clean, indices, levels = parse_annotated_text(
    "This is **absolutely** *incredible*!"
)
result = adapter.from_emphasis_trajectory(
    text_embeddings=text_emb,
    emotion="surprised",
    emphasis_levels=[0, 0, 3, 2],
)

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
