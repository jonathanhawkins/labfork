"""
EmoVoice: LLM-based Emotional Text-To-Speech with Freestyle Natural Language Control

Based on "EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting"
ACM MM '25 - arXiv:2504.12867
https://github.com/yanghaha0908/EmoVoice

Key Innovation: Fine-grained emotion control via natural language descriptions instead of
categorical emotion labels. Uses LLM backbone (Qwen2.5) to understand freestyle emotion
prompts like "expressing supportive joy and pride" or "speaking with nervous anticipation".

Architecture:
1. LLM Backbone (Qwen2.5-0.5B/1.5B): Pre-trained language model for understanding prompts
2. Vocabulary Extension: Text tokens + Audio tokens (CosyVoice semantic codebook)
3. Semantic Group Modeling: Predicts G tokens per step for faster training (G=3)
4. Phoneme Boost (EmoVoice-PP): Parallel phoneme + audio token output for content consistency
5. Flow Matching + HiFi-GAN: Converts 50Hz semantic tokens to waveforms

Training:
- Phase 1: Pre-training on standard TTS data (neutral emotion)
- Phase 2: Fine-tuning on emotion-labeled data with natural language descriptions

Benefits:
- Fine-grained emotion nuances beyond categorical labels
- Natural language interface for users
- State-of-the-art emotion accuracy on EmoVoice-DB
- Better content consistency via phoneme boosting (CoT/CoM inspired)

Integration with CSM pipeline:
- EmoVoiceAdapter generates prosody prefix tokens from emotion descriptions
- Compatible with existing ProsodyControlledCSM
"""

import math
import re
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
class EmoVoiceConfig:
    """Configuration for EmoVoice model."""

    # LLM Backbone
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"  # or "Qwen/Qwen2.5-1.5B"
    llm_hidden_dim: int = 896  # Qwen2.5-0.5B hidden dim
    llm_num_layers: int = 24  # Qwen2.5-0.5B layers
    use_pretrained_llm: bool = True
    freeze_llm_layers: int = 0  # Number of layers to freeze (0 = train all)

    # Audio Token Vocabulary
    num_audio_tokens: int = 4096  # CosyVoice semantic codebook size
    audio_token_rate: int = 50  # 50Hz semantic tokens

    # Phoneme Boost (EmoVoice-PP)
    use_phoneme_boost: bool = True
    num_phoneme_tokens: int = 256  # Phoneme vocabulary size
    phoneme_rate: int = 11  # ~11Hz phoneme tokens

    # Semantic Group Modeling
    semantic_group_size: int = 3  # Predict G tokens per step

    # Emotion Prompt Processing
    max_description_length: int = 64  # Max tokens for emotion description
    system_prompt: str = "Say this sentence with emotion of {description}."
    neutral_prompt: str = "Say this sentence."

    # Architecture
    embedding_dim: int = 512
    hidden_dim: int = 1024
    output_dim: int = 2048  # CSM prosody hidden dimension
    num_prosody_tokens: int = 4  # Prefix tokens for prosody conditioning
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    finetune_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    repetition_penalty: float = 1.2

    # Flow matching (placeholder for integration)
    use_flow_matching: bool = True
    flow_matching_steps: int = 32


# =============================================================================
# EMOTION PROMPT TEMPLATES
# =============================================================================

# Example freestyle emotion descriptions from EmoVoice-DB
EMOTION_DESCRIPTION_EXAMPLES = {
    "happy": [
        "expressing genuine happiness and warmth",
        "speaking with joyful enthusiasm",
        "conveying delighted satisfaction",
        "expressing supportive joy and pride",
        "speaking with bright, cheerful energy",
    ],
    "sad": [
        "expressing deep sadness and melancholy",
        "speaking with sorrowful resignation",
        "conveying heartfelt disappointment",
        "expressing mournful contemplation",
        "speaking with quiet grief",
    ],
    "angry": [
        "expressing intense frustration and anger",
        "speaking with fierce indignation",
        "conveying barely contained rage",
        "expressing sharp irritation",
        "speaking with heated resentment",
    ],
    "fearful": [
        "expressing anxious apprehension",
        "speaking with nervous anticipation",
        "conveying worried uncertainty",
        "expressing trembling fear",
        "speaking with panicked urgency",
    ],
    "surprised": [
        "expressing sudden astonishment",
        "speaking with shocked disbelief",
        "conveying startled amazement",
        "expressing unexpected wonder",
        "speaking with stunned realization",
    ],
    "disgusted": [
        "expressing strong disgust and revulsion",
        "speaking with disdainful contempt",
        "conveying visceral repulsion",
        "expressing deep disapproval",
        "speaking with scornful distaste",
    ],
    "neutral": [
        "speaking in a calm, measured tone",
        "expressing matter-of-fact clarity",
        "conveying composed neutrality",
        "speaking with professional detachment",
        "expressing balanced objectivity",
    ],
    "excited": [
        "expressing bubbling excitement",
        "speaking with eager anticipation",
        "conveying thrilled enthusiasm",
        "expressing animated exhilaration",
        "speaking with infectious energy",
    ],
    "tender": [
        "expressing gentle affection",
        "speaking with warm tenderness",
        "conveying loving care",
        "expressing soft compassion",
        "speaking with nurturing kindness",
    ],
    "contempt": [
        "expressing cold contempt",
        "speaking with dismissive superiority",
        "conveying patronizing disdain",
        "expressing aloof mockery",
        "speaking with sarcastic derision",
    ],
}

# Emotion keywords for basic parsing
EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "cheerful", "delighted", "pleased", "glad", "content"],
    "sad": ["sad", "sorrow", "melancholy", "grief", "mournful", "unhappy", "depressed"],
    "angry": ["angry", "furious", "rage", "mad", "irritated", "frustrated", "annoyed"],
    "fearful": ["fear", "anxious", "scared", "nervous", "worried", "terrified", "panic"],
    "surprised": ["surprised", "shocked", "astonished", "amazed", "startled", "stunned"],
    "disgusted": ["disgust", "repulsed", "revolted", "nauseated", "disdain", "contempt"],
    "excited": ["excited", "thrilled", "eager", "enthusiastic", "exhilarated", "animated"],
    "tender": ["tender", "gentle", "affectionate", "loving", "caring", "compassionate"],
    "neutral": ["neutral", "calm", "composed", "matter-of-fact", "objective", "balanced"],
}


# =============================================================================
# EMOTION PROMPT PROCESSOR
# =============================================================================

class EmotionPromptProcessor(nn.Module):
    """
    Processes natural language emotion descriptions into embeddings.

    The processor:
    1. Validates and normalizes emotion descriptions
    2. Encodes descriptions using a text encoder (BERT-like or LLM embeddings)
    3. Extracts emotion-relevant features
    4. Generates conditioning vectors for the TTS model

    Supports both:
    - Categorical emotions: "happy", "sad", etc.
    - Freestyle descriptions: "expressing supportive joy and pride"
    """

    def __init__(self, config: EmoVoiceConfig):
        super().__init__()
        self.config = config

        # Simple description encoder (in practice, use LLM's internal embeddings)
        # Here we use a lightweight transformer encoder
        self.description_embed = nn.Embedding(
            50000,  # Vocabulary size (placeholder)
            config.embedding_dim,
        )

        # Positional encoding for description
        self.pos_encoding = nn.Parameter(
            self._sinusoidal_pos_encoding(config.max_description_length, config.embedding_dim)
        )

        # Transformer encoder for description understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.description_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, config.embedding_dim))

        # Project to emotion embedding
        self.emotion_projection = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        # Emotion classifier (for validation/analysis)
        self.emotion_classifier = nn.Linear(config.embedding_dim, len(EMOTION_KEYWORDS))

    def _sinusoidal_pos_encoding(self, length: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def parse_emotion_description(self, description: str) -> Dict[str, Any]:
        """
        Parse and validate an emotion description.

        Args:
            description: Natural language emotion description

        Returns:
            Dict with parsed emotion info and validation results
        """
        description = description.strip().lower()

        # Check if it's a simple categorical emotion
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if description in keywords or description == emotion:
                return {
                    "type": "categorical",
                    "emotion": emotion,
                    "description": description,
                    "valid": True,
                }

        # It's a freestyle description - validate format
        # Should be present participle verb phrase
        valid = True
        warnings = []

        # Check for present participle (basic heuristic)
        participle_patterns = [
            r"\bexpressing\b", r"\bspeaking\b", r"\bconveying\b",
            r"\bshowing\b", r"\bdemonstrating\b", r"\bportraying\b",
        ]
        has_participle = any(re.search(p, description) for p in participle_patterns)
        if not has_participle:
            warnings.append("Description should use present participle verb phrases")

        # Check length (not too short)
        word_count = len(description.split())
        if word_count < 3:
            warnings.append("Description may be too brief for fine-grained control")

        # Detect primary emotion from keywords
        detected_emotions = []
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if any(kw in description for kw in keywords):
                detected_emotions.append(emotion)

        return {
            "type": "freestyle",
            "emotion": detected_emotions[0] if detected_emotions else "neutral",
            "detected_emotions": detected_emotions,
            "description": description,
            "valid": valid,
            "warnings": warnings,
            "word_count": word_count,
        }

    def forward(
        self,
        description_ids: torch.Tensor,  # [batch, seq_len] tokenized descriptions
        description_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion descriptions.

        Args:
            description_ids: Tokenized description IDs
            description_mask: Attention mask for descriptions

        Returns:
            Dict with emotion embeddings and analysis
        """
        batch_size, seq_len = description_ids.shape
        device = description_ids.device

        # Embed descriptions
        description_emb = self.description_embed(description_ids)

        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        description_emb = description_emb + pos_enc.to(device)

        # Create attention mask if not provided
        if description_mask is None:
            description_mask = torch.ones(batch_size, seq_len, device=device)

        # Encode with transformer
        # Convert mask to boolean for transformer
        src_key_padding_mask = (description_mask == 0)
        encoded = self.description_encoder(
            description_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Attention pooling
        query = self.query.expand(batch_size, -1, -1)
        pooled, attention_weights = self.attention_pool(
            query, encoded, encoded,
            key_padding_mask=src_key_padding_mask,
        )
        pooled = pooled.squeeze(1)  # [batch, embedding_dim]

        # Project to emotion embedding
        emotion_emb = self.emotion_projection(pooled)

        # Classify emotion (for analysis)
        emotion_logits = self.emotion_classifier(emotion_emb)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        return {
            "emotion_embedding": emotion_emb,
            "emotion_logits": emotion_logits,
            "emotion_probs": emotion_probs,
            "attention_weights": attention_weights,
            "encoded_description": encoded,
        }


# =============================================================================
# SEMANTIC GROUP MODELING
# =============================================================================

class SemanticGroupLayer(nn.Module):
    """
    Semantic Group Modeling layer for faster training.

    Instead of predicting one token at a time, predicts G tokens per step.
    This enables 2.64x faster training (G=3) without quality loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        group_size: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.group_size = group_size

        # Project hidden to group logits
        self.group_projection = nn.Linear(hidden_dim, vocab_size * group_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to group token logits.

        Args:
            hidden: [batch, seq_len, hidden_dim]

        Returns:
            logits: [batch, seq_len * group_size, vocab_size]
        """
        batch_size, seq_len, _ = hidden.shape

        # Project to group logits
        group_logits = self.group_projection(hidden)  # [batch, seq_len, vocab * group]

        # Reshape to separate tokens
        group_logits = group_logits.view(batch_size, seq_len, self.group_size, self.vocab_size)

        # Flatten to [batch, seq_len * group_size, vocab_size]
        output = group_logits.view(batch_size, seq_len * self.group_size, self.vocab_size)

        return output

    def decode(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Decode group logits to tokens.

        Args:
            logits: [batch, seq_len * group_size, vocab_size]
            temperature: Sampling temperature

        Returns:
            tokens: [batch, seq_len * group_size]
        """
        if temperature <= 0:
            return logits.argmax(dim=-1)

        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])


# =============================================================================
# PHONEME BOOST MODULE
# =============================================================================

class PhonemeBoostModule(nn.Module):
    """
    Phoneme Boost for content consistency (EmoVoice-PP).

    Outputs phoneme tokens and audio tokens in parallel, inspired by
    Chain-of-Thought (CoT) and Chain-of-Modality (CoM) techniques.

    The phoneme sequence (~11Hz) acts as intermediate supervision,
    guiding the generation of audio tokens (~17Hz effective after grouping).
    """

    def __init__(self, config: EmoVoiceConfig):
        super().__init__()
        self.config = config

        # Phoneme embedding
        self.phoneme_embed = nn.Embedding(
            config.num_phoneme_tokens,
            config.embedding_dim,
        )

        # Phoneme prediction head
        self.phoneme_head = nn.Sequential(
            nn.Linear(config.llm_hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_phoneme_tokens),
        )

        # Audio-phoneme alignment layer
        # Projects phoneme features to assist audio generation
        self.alignment_projection = nn.Linear(
            config.embedding_dim,
            config.llm_hidden_dim,
        )

        # Gating mechanism for phoneme influence
        self.phoneme_gate = nn.Sequential(
            nn.Linear(config.llm_hidden_dim * 2, config.llm_hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        target_phonemes: Optional[torch.Tensor] = None,  # [batch, phoneme_len] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Predict phonemes and generate phoneme-guided features.

        Args:
            hidden_states: LLM hidden states
            target_phonemes: Ground truth phonemes for training

        Returns:
            Dict with phoneme predictions and guided features
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Predict phoneme logits
        # Phonemes are predicted at lower rate, so we downsample
        phoneme_rate_factor = self.config.audio_token_rate // self.config.phoneme_rate

        # Pool hidden states for phoneme prediction
        if seq_len > phoneme_rate_factor:
            phoneme_len = seq_len // phoneme_rate_factor
            hidden_pooled = hidden_states[:, :phoneme_len * phoneme_rate_factor, :]
            hidden_pooled = hidden_pooled.view(
                batch_size, phoneme_len, phoneme_rate_factor, hidden_dim
            ).mean(dim=2)
        else:
            hidden_pooled = hidden_states.mean(dim=1, keepdim=True)
            phoneme_len = 1

        # Predict phonemes
        phoneme_logits = self.phoneme_head(hidden_pooled)  # [batch, phoneme_len, num_phonemes]

        # Get phoneme embeddings (from predictions or targets)
        if target_phonemes is not None:
            # Training: use ground truth
            phoneme_emb = self.phoneme_embed(target_phonemes)
        else:
            # Inference: use predictions
            phoneme_tokens = phoneme_logits.argmax(dim=-1)
            phoneme_emb = self.phoneme_embed(phoneme_tokens)

        # Upsample phoneme embeddings to match audio rate
        if phoneme_emb.size(1) < seq_len:
            phoneme_emb = F.interpolate(
                phoneme_emb.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        # Project phoneme features
        phoneme_features = self.alignment_projection(phoneme_emb)

        # Gate phoneme influence on audio generation
        combined = torch.cat([hidden_states, phoneme_features], dim=-1)
        gate = self.phoneme_gate(combined)

        # Guided hidden states
        guided_hidden = hidden_states + gate * phoneme_features

        return {
            "phoneme_logits": phoneme_logits,
            "phoneme_features": phoneme_features,
            "guided_hidden": guided_hidden,
            "gate_values": gate,
        }


# =============================================================================
# EMOVOICE LLM BACKBONE
# =============================================================================

class EmoVoiceLLMBackbone(nn.Module):
    """
    LLM Backbone for EmoVoice.

    Wraps an LLM (Qwen2.5) to:
    1. Process text input with emotion prompts
    2. Predict audio semantic tokens autoregressively
    3. Support vocabulary extension for audio tokens

    In practice, this would load actual Qwen2.5 weights.
    Here we provide a placeholder implementation.
    """

    def __init__(self, config: EmoVoiceConfig):
        super().__init__()
        self.config = config

        # Extended vocabulary embedding
        # Original text vocab + audio tokens + phoneme tokens (if used)
        total_vocab = 50000 + config.num_audio_tokens
        if config.use_phoneme_boost:
            total_vocab += config.num_phoneme_tokens

        self.embedding = nn.Embedding(total_vocab, config.llm_hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 8192, config.llm_hidden_dim) * 0.02
        )

        # Transformer layers (simplified)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.llm_hidden_dim,
            nhead=16,
            dim_feedforward=config.llm_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN like Qwen2.5
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=config.llm_num_layers)

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.llm_hidden_dim)

        # Audio token prediction head (with semantic grouping)
        self.audio_head = SemanticGroupLayer(
            hidden_dim=config.llm_hidden_dim,
            vocab_size=config.num_audio_tokens,
            group_size=config.semantic_group_size,
        )

        # Phoneme boost module
        if config.use_phoneme_boost:
            self.phoneme_boost = PhonemeBoostModule(config)
        else:
            self.phoneme_boost = None

        # Audio token offset in vocabulary
        self.audio_token_offset = 50000

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,
        target_audio_tokens: Optional[torch.Tensor] = None,  # [batch, audio_len]
        target_phonemes: Optional[torch.Tensor] = None,  # [batch, phoneme_len]
        emotion_embedding: Optional[torch.Tensor] = None,  # [batch, embedding_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the LLM backbone.

        Args:
            input_ids: Text input token IDs
            attention_mask: Attention mask
            target_audio_tokens: Ground truth audio tokens for training
            target_phonemes: Ground truth phonemes for phoneme boost
            emotion_embedding: Emotion embedding from prompt processor

        Returns:
            Dict with predictions and hidden states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed inputs
        hidden = self.embedding(input_ids)

        # Add positional encoding
        hidden = hidden + self.pos_encoding[:, :seq_len, :]

        # Add emotion embedding if provided
        if emotion_embedding is not None:
            # Broadcast emotion to all positions
            emotion_expanded = emotion_embedding.unsqueeze(1)
            if emotion_expanded.size(-1) != hidden.size(-1):
                # Project emotion to hidden dim
                emotion_expanded = F.linear(
                    emotion_expanded,
                    torch.randn(hidden.size(-1), emotion_expanded.size(-1), device=device) * 0.02
                )
            hidden = hidden + emotion_expanded * 0.1  # Scale emotion influence

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )

        # Process through transformer layers
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden = self.layers(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        hidden = self.final_norm(hidden)

        # Apply phoneme boost if enabled
        phoneme_outputs = None
        if self.phoneme_boost is not None:
            phoneme_outputs = self.phoneme_boost(hidden, target_phonemes)
            hidden = phoneme_outputs["guided_hidden"]

        # Predict audio tokens
        audio_logits = self.audio_head(hidden)

        return {
            "hidden_states": hidden,
            "audio_logits": audio_logits,
            "phoneme_outputs": phoneme_outputs,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        emotion_embedding: Optional[torch.Tensor] = None,
        max_audio_tokens: int = 500,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation of audio tokens.

        Args:
            input_ids: Text prompt token IDs
            emotion_embedding: Emotion embedding
            max_audio_tokens: Maximum number of audio tokens to generate
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeated tokens

        Returns:
            Dict with generated audio tokens
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated_tokens = []
        current_input = input_ids

        # Track generated tokens for repetition penalty
        all_generated = torch.zeros(batch_size, self.config.num_audio_tokens, device=device)

        for step in range(max_audio_tokens // self.config.semantic_group_size):
            # Forward pass
            outputs = self.forward(
                current_input,
                emotion_embedding=emotion_embedding,
            )

            # Get logits for next token group
            audio_logits = outputs["audio_logits"][:, -self.config.semantic_group_size:, :]

            # Apply repetition penalty
            for i in range(batch_size):
                audio_logits[i] = audio_logits[i] / (
                    1.0 + all_generated[i].unsqueeze(0) * (repetition_penalty - 1.0)
                )

            # Sample tokens
            if temperature <= 0:
                next_tokens = audio_logits.argmax(dim=-1)
            else:
                probs = F.softmax(audio_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1
                ).view(batch_size, self.config.semantic_group_size)

            generated_tokens.append(next_tokens)

            # Update tracking
            for i in range(batch_size):
                for t in next_tokens[i]:
                    all_generated[i, t] += 1

            # Append to input (with audio token offset)
            next_input_ids = next_tokens + self.audio_token_offset
            current_input = torch.cat([current_input, next_input_ids], dim=1)

            # Check for EOS (simplified)
            if (next_tokens == 0).all():  # Assume 0 is EOS
                break

        audio_tokens = torch.cat(generated_tokens, dim=1)

        return {
            "audio_tokens": audio_tokens,
            "num_tokens": audio_tokens.shape[1],
        }


# =============================================================================
# EMOVOICE MAIN MODEL
# =============================================================================

class EmoVoice(nn.Module):
    """
    EmoVoice: LLM-based Emotional TTS with Freestyle Natural Language Control.

    The model combines:
    1. EmotionPromptProcessor: Understands natural language emotion descriptions
    2. EmoVoiceLLMBackbone: Generates audio tokens autoregressively
    3. PhonemeBoostModule: Ensures content consistency (optional)
    4. SemanticGroupLayer: Fast training via grouped prediction

    Usage:
        config = EmoVoiceConfig()
        model = EmoVoice(config)

        # Training
        loss = model.compute_loss(
            text_ids, audio_targets,
            emotion_description="expressing joyful enthusiasm"
        )

        # Inference
        audio_tokens = model.generate(
            text="Hello, how are you?",
            emotion="expressing warm friendliness",
        )
    """

    def __init__(self, config: EmoVoiceConfig):
        super().__init__()
        self.config = config

        # Components
        self.prompt_processor = EmotionPromptProcessor(config)
        self.backbone = EmoVoiceLLMBackbone(config)

        # Output projection for CSM integration
        self.prosody_projection = nn.Sequential(
            nn.Linear(config.llm_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * config.num_prosody_tokens),
        )
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        description_ids: Optional[torch.Tensor] = None,
        description_mask: Optional[torch.Tensor] = None,
        target_audio_tokens: Optional[torch.Tensor] = None,
        target_phonemes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            text_ids: Text input token IDs [batch, text_len]
            text_mask: Text attention mask
            description_ids: Emotion description token IDs [batch, desc_len]
            description_mask: Description attention mask
            target_audio_tokens: Ground truth audio tokens [batch, audio_len]
            target_phonemes: Ground truth phonemes [batch, phoneme_len]

        Returns:
            Dict with all model outputs
        """
        # Process emotion description
        emotion_output = None
        emotion_embedding = None
        if description_ids is not None:
            emotion_output = self.prompt_processor(description_ids, description_mask)
            emotion_embedding = emotion_output["emotion_embedding"]

        # Forward through backbone
        backbone_output = self.backbone(
            text_ids,
            attention_mask=text_mask,
            target_audio_tokens=target_audio_tokens,
            target_phonemes=target_phonemes,
            emotion_embedding=emotion_embedding,
        )

        # Generate prosody tokens for CSM integration
        hidden_pooled = backbone_output["hidden_states"].mean(dim=1)  # [batch, hidden_dim]
        prosody_tokens = self.prosody_projection(hidden_pooled)
        prosody_tokens = prosody_tokens.view(
            -1, self.config.num_prosody_tokens, self.config.output_dim
        )
        prosody_tokens = self.output_norm(prosody_tokens)

        return {
            "audio_logits": backbone_output["audio_logits"],
            "hidden_states": backbone_output["hidden_states"],
            "phoneme_outputs": backbone_output["phoneme_outputs"],
            "emotion_output": emotion_output,
            "prosody_tokens": prosody_tokens,
        }

    def compute_loss(
        self,
        text_ids: torch.Tensor,
        target_audio_tokens: torch.Tensor,
        description_ids: Optional[torch.Tensor] = None,
        target_phonemes: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        description_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            text_ids: Text input tokens
            target_audio_tokens: Ground truth audio tokens
            description_ids: Emotion description tokens
            target_phonemes: Ground truth phonemes
            text_mask: Text attention mask
            description_mask: Description attention mask

        Returns:
            Dict with loss values
        """
        # Forward pass
        outputs = self.forward(
            text_ids=text_ids,
            text_mask=text_mask,
            description_ids=description_ids,
            description_mask=description_mask,
            target_audio_tokens=target_audio_tokens,
            target_phonemes=target_phonemes,
        )

        losses = {}

        # Audio token prediction loss (cross-entropy)
        audio_logits = outputs["audio_logits"]

        # Flatten for loss computation
        batch_size, seq_len, vocab_size = audio_logits.shape
        target_len = target_audio_tokens.shape[1]

        # Ensure logits and targets have same length
        min_len = min(seq_len, target_len)
        audio_logits_flat = audio_logits[:, :min_len, :].reshape(-1, vocab_size)
        targets_flat = target_audio_tokens[:, :min_len].reshape(-1)

        losses["audio_ce"] = F.cross_entropy(audio_logits_flat, targets_flat)

        # Phoneme prediction loss (if using phoneme boost)
        if outputs["phoneme_outputs"] is not None:
            phoneme_logits = outputs["phoneme_outputs"]["phoneme_logits"]
            if target_phonemes is not None:
                phoneme_len = min(phoneme_logits.size(1), target_phonemes.size(1))
                phoneme_logits_flat = phoneme_logits[:, :phoneme_len, :].reshape(
                    -1, self.config.num_phoneme_tokens
                )
                phoneme_targets_flat = target_phonemes[:, :phoneme_len].reshape(-1)
                losses["phoneme_ce"] = F.cross_entropy(phoneme_logits_flat, phoneme_targets_flat)
            else:
                losses["phoneme_ce"] = torch.tensor(0.0, device=audio_logits.device)

        # Total loss
        total_loss = losses["audio_ce"]
        if "phoneme_ce" in losses:
            total_loss = total_loss + 0.3 * losses["phoneme_ce"]  # Phoneme loss weight

        losses["total"] = total_loss

        return losses

    def generate(
        self,
        text_ids: torch.Tensor,
        description_ids: Optional[torch.Tensor] = None,
        description_mask: Optional[torch.Tensor] = None,
        max_audio_tokens: int = 500,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate audio tokens from text with emotion control.

        Args:
            text_ids: Text input token IDs
            description_ids: Emotion description token IDs
            description_mask: Description attention mask
            max_audio_tokens: Maximum audio tokens to generate
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty

        Returns:
            Dict with generated audio tokens and prosody tokens
        """
        # Get emotion embedding
        emotion_embedding = None
        if description_ids is not None:
            emotion_output = self.prompt_processor(description_ids, description_mask)
            emotion_embedding = emotion_output["emotion_embedding"]

        # Generate audio tokens
        gen_output = self.backbone.generate(
            text_ids,
            emotion_embedding=emotion_embedding,
            max_audio_tokens=max_audio_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        # Get prosody tokens for CSM integration
        with torch.no_grad():
            outputs = self.forward(
                text_ids,
                description_ids=description_ids,
                description_mask=description_mask,
            )

        return {
            "audio_tokens": gen_output["audio_tokens"],
            "prosody_tokens": outputs["prosody_tokens"],
            "num_audio_tokens": gen_output["num_tokens"],
        }


# =============================================================================
# EMOVOICE ADAPTER FOR CSM INTEGRATION
# =============================================================================

class EmoVoiceAdapter(nn.Module):
    """
    Adapter for integrating EmoVoice with the existing prosody pipeline.

    Provides a simple interface to:
    1. Process natural language emotion descriptions
    2. Generate prosody prefix tokens
    3. Compatible with ProsodyControlledCSM

    Usage:
        adapter = EmoVoiceAdapter(config)

        # From natural language description
        tokens = adapter.encode_description("expressing warm happiness")

        # From categorical emotion + intensity
        tokens = adapter.encode_emotion("happy", intensity=0.8)

        # Use with CSM
        prefix = torch.cat([prosody_prefix, tokens['prosody_tokens']], dim=1)
    """

    def __init__(
        self,
        config: EmoVoiceConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core EmoVoice model
        self.emovoice = EmoVoice(config)

        # Prosody adapter (if dimensions differ)
        if config.output_dim != prosody_hidden:
            self.prosody_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.prosody_adapter = nn.Identity()

        # Simple tokenizer (placeholder - in practice use real tokenizer)
        self.vocab = self._build_simple_vocab()

        # Emotion description templates
        self.description_templates = EMOTION_DESCRIPTION_EXAMPLES

    def _build_simple_vocab(self) -> Dict[str, int]:
        """Build a simple word-to-id vocabulary."""
        words = set()
        for descriptions in EMOTION_DESCRIPTION_EXAMPLES.values():
            for desc in descriptions:
                words.update(desc.lower().split())

        # Add common words
        common = ["the", "a", "an", "with", "and", "of", "to", "in", "is", "are"]
        words.update(common)

        vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        for i, word in enumerate(sorted(words)):
            vocab[word] = i + 3

        return vocab

    def tokenize(self, text: str) -> torch.Tensor:
        """Simple tokenization."""
        words = text.lower().split()
        ids = [self.vocab.get(w, 1) for w in words]  # 1 = <unk>
        return torch.tensor(ids)

    def encode_description(
        self,
        description: str,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a natural language emotion description.

        Args:
            description: Natural language emotion description
            batch_size: Batch size for output

        Returns:
            Dict with prosody tokens and analysis
        """
        device = next(self.parameters()).device

        # Parse description
        parsed = self.emovoice.prompt_processor.parse_emotion_description(description)

        # Tokenize
        desc_ids = self.tokenize(description).unsqueeze(0).to(device)
        desc_ids = desc_ids.expand(batch_size, -1)

        # Create placeholder text (the actual text would come from the TTS input)
        text_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Forward through EmoVoice
        outputs = self.emovoice(
            text_ids=text_ids,
            description_ids=desc_ids,
        )

        # Adapt to prosody dimension
        prosody_tokens = self.prosody_adapter(outputs["prosody_tokens"])

        return {
            "prosody_tokens": prosody_tokens,
            "parsed_description": parsed,
            "emotion_embedding": outputs["emotion_output"]["emotion_embedding"],
            "emotion_probs": outputs["emotion_output"]["emotion_probs"],
        }

    def encode_emotion(
        self,
        emotion: str,
        intensity: float = 0.7,
        variant: int = 0,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a categorical emotion with optional intensity.

        Args:
            emotion: Emotion name (happy, sad, angry, etc.)
            intensity: Emotion intensity (0-1)
            variant: Which description variant to use (0-4)
            batch_size: Batch size

        Returns:
            Dict with prosody tokens
        """
        emotion = emotion.lower()

        # Get description template
        if emotion in self.description_templates:
            descriptions = self.description_templates[emotion]
            variant_idx = variant % len(descriptions)
            description = descriptions[variant_idx]
        else:
            # Fallback to simple description
            description = f"expressing {emotion} emotion"

        # Modify description based on intensity
        if intensity < 0.3:
            description = f"slightly {description}"
        elif intensity > 0.8:
            description = f"intensely {description}"

        return self.encode_description(description, batch_size)

    def interpolate_descriptions(
        self,
        description1: str,
        description2: str,
        t: float,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotion descriptions.

        Args:
            description1: Source description
            description2: Target description
            t: Interpolation factor [0, 1]
            batch_size: Batch size

        Returns:
            Dict with interpolated prosody tokens
        """
        device = next(self.parameters()).device

        # Encode both descriptions
        result1 = self.encode_description(description1, batch_size)
        result2 = self.encode_description(description2, batch_size)

        # Interpolate embeddings
        emb1 = result1["emotion_embedding"]
        emb2 = result2["emotion_embedding"]
        interpolated_emb = emb1 * (1 - t) + emb2 * t

        # Interpolate prosody tokens
        tokens1 = result1["prosody_tokens"]
        tokens2 = result2["prosody_tokens"]
        interpolated_tokens = tokens1 * (1 - t) + tokens2 * t

        return {
            "prosody_tokens": interpolated_tokens,
            "emotion_embedding": interpolated_emb,
            "source_description": description1,
            "target_description": description2,
            "interpolation": t,
        }

    def forward(
        self,
        description: str = None,
        emotion: str = None,
        intensity: float = 0.7,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Simple forward pass returning prosody tokens.

        Args:
            description: Natural language description (priority)
            emotion: Categorical emotion (fallback)
            intensity: Emotion intensity
            batch_size: Batch size

        Returns:
            Prosody tokens [batch, num_tokens, hidden_dim]
        """
        if description is not None:
            result = self.encode_description(description, batch_size)
        elif emotion is not None:
            result = self.encode_emotion(emotion, intensity, batch_size=batch_size)
        else:
            # Default to neutral
            result = self.encode_emotion("neutral", batch_size=batch_size)

        return result["prosody_tokens"]


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class EmoVoiceLoss(nn.Module):
    """
    Combined loss function for EmoVoice training.

    Components:
    1. Audio CE loss: Main semantic token prediction
    2. Phoneme CE loss: Content consistency via phoneme boost
    3. Emotion classification loss: Ensure emotion understanding
    4. Contrastive loss: Better emotion discrimination
    """

    def __init__(
        self,
        config: EmoVoiceConfig,
        audio_weight: float = 1.0,
        phoneme_weight: float = 0.3,
        emotion_class_weight: float = 0.2,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.audio_weight = audio_weight
        self.phoneme_weight = phoneme_weight
        self.emotion_class_weight = emotion_class_weight
        self.contrastive_weight = contrastive_weight

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_audio_tokens: torch.Tensor,
        target_phonemes: Optional[torch.Tensor] = None,
        target_emotion_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            model_output: Output from EmoVoice forward pass
            target_audio_tokens: Ground truth audio tokens
            target_phonemes: Ground truth phonemes
            target_emotion_idx: Ground truth emotion indices

        Returns:
            Dict with individual and total losses
        """
        losses = {}
        device = target_audio_tokens.device

        # Audio prediction loss
        audio_logits = model_output["audio_logits"]
        batch_size, seq_len, vocab_size = audio_logits.shape
        target_len = target_audio_tokens.shape[1]

        min_len = min(seq_len, target_len)
        audio_logits_flat = audio_logits[:, :min_len, :].reshape(-1, vocab_size)
        targets_flat = target_audio_tokens[:, :min_len].reshape(-1)

        losses["audio"] = F.cross_entropy(audio_logits_flat, targets_flat)

        # Phoneme loss
        if (model_output["phoneme_outputs"] is not None and
            target_phonemes is not None):
            phoneme_logits = model_output["phoneme_outputs"]["phoneme_logits"]
            phoneme_len = min(phoneme_logits.size(1), target_phonemes.size(1))

            phoneme_logits_flat = phoneme_logits[:, :phoneme_len, :].reshape(
                -1, self.config.num_phoneme_tokens
            )
            phoneme_targets_flat = target_phonemes[:, :phoneme_len].reshape(-1)

            losses["phoneme"] = F.cross_entropy(phoneme_logits_flat, phoneme_targets_flat)
        else:
            losses["phoneme"] = torch.tensor(0.0, device=device)

        # Emotion classification loss
        if (model_output["emotion_output"] is not None and
            target_emotion_idx is not None):
            emotion_logits = model_output["emotion_output"]["emotion_logits"]
            losses["emotion_class"] = F.cross_entropy(emotion_logits, target_emotion_idx)
        else:
            losses["emotion_class"] = torch.tensor(0.0, device=device)

        # Contrastive loss (simplified)
        if model_output["emotion_output"] is not None:
            emotion_emb = model_output["emotion_output"]["emotion_embedding"]
            # Normalize embeddings
            emotion_emb_norm = F.normalize(emotion_emb, dim=-1)
            # Compute similarity matrix
            similarity = torch.matmul(emotion_emb_norm, emotion_emb_norm.T)
            # Create target (same emotion = similar, different = dissimilar)
            if target_emotion_idx is not None:
                target_sim = (target_emotion_idx.unsqueeze(0) == target_emotion_idx.unsqueeze(1)).float()
                losses["contrastive"] = F.mse_loss(similarity, target_sim)
            else:
                losses["contrastive"] = torch.tensor(0.0, device=device)
        else:
            losses["contrastive"] = torch.tensor(0.0, device=device)

        # Total loss
        total = (
            losses["audio"] * self.audio_weight +
            losses["phoneme"] * self.phoneme_weight +
            losses["emotion_class"] * self.emotion_class_weight +
            losses["contrastive"] * self.contrastive_weight
        )
        losses["total"] = total

        return losses


# =============================================================================
# NATURAL LANGUAGE EMOTION INTERFACE
# =============================================================================

def generate_emotion_description(
    emotion: str,
    intensity: float = 0.7,
    add_modifiers: bool = True,
) -> str:
    """
    Generate a natural language emotion description.

    Args:
        emotion: Base emotion name
        intensity: Emotion intensity (0-1)
        add_modifiers: Add intensity modifiers

    Returns:
        Natural language description
    """
    emotion = emotion.lower()

    # Get base descriptions
    if emotion in EMOTION_DESCRIPTION_EXAMPLES:
        import random
        base = random.choice(EMOTION_DESCRIPTION_EXAMPLES[emotion])
    else:
        base = f"expressing {emotion} emotion"

    # Add intensity modifiers
    if add_modifiers:
        if intensity < 0.2:
            base = f"very subtly {base}"
        elif intensity < 0.4:
            base = f"gently {base}"
        elif intensity > 0.9:
            base = f"intensely and powerfully {base}"
        elif intensity > 0.7:
            base = f"strongly {base}"

    return base


def parse_freestyle_prompt(prompt: str) -> Dict[str, Any]:
    """
    Parse a freestyle emotion prompt into structured information.

    Args:
        prompt: Natural language emotion prompt

    Returns:
        Dict with detected emotions, intensity hints, and analysis
    """
    prompt_lower = prompt.lower()

    # Detect emotions
    detected_emotions = []
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            detected_emotions.append(emotion)

    # Detect intensity hints
    intensity_hints = []
    high_intensity = ["intensely", "strongly", "very", "extremely", "powerfully"]
    low_intensity = ["slightly", "subtly", "gently", "mildly", "softly"]

    for hint in high_intensity:
        if hint in prompt_lower:
            intensity_hints.append(("high", hint))
    for hint in low_intensity:
        if hint in prompt_lower:
            intensity_hints.append(("low", hint))

    # Estimate intensity
    if any(level == "high" for level, _ in intensity_hints):
        estimated_intensity = 0.85
    elif any(level == "low" for level, _ in intensity_hints):
        estimated_intensity = 0.35
    else:
        estimated_intensity = 0.6

    return {
        "original_prompt": prompt,
        "detected_emotions": detected_emotions,
        "primary_emotion": detected_emotions[0] if detected_emotions else "neutral",
        "intensity_hints": intensity_hints,
        "estimated_intensity": estimated_intensity,
        "word_count": len(prompt.split()),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EmoVoice - Freestyle Natural Language Emotion Control")
    print("=" * 70)

    config = EmoVoiceConfig()

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    print(f"  LLM Model: {config.llm_model_name}")
    print(f"  Audio Token Vocab: {config.num_audio_tokens}")
    print(f"  Semantic Group Size: {config.semantic_group_size}")
    print(f"  Phoneme Boost: {config.use_phoneme_boost}")
    print("  [PASS]")

    # Test 2: Emotion Description Examples
    print("\n[Test 2] Emotion Description Examples...")
    for emotion in ["happy", "sad", "angry"]:
        examples = EMOTION_DESCRIPTION_EXAMPLES.get(emotion, [])[:2]
        print(f"  {emotion}:")
        for ex in examples:
            print(f"    - \"{ex}\"")
    print("  [PASS]")

    # Test 3: Prompt Parsing
    print("\n[Test 3] Freestyle Prompt Parsing...")
    test_prompts = [
        "expressing warm happiness and contentment",
        "speaking with intense frustration and barely contained anger",
        "conveying gentle sadness and quiet longing",
        "speaking matter-of-factly with professional detachment",
    ]

    for prompt in test_prompts:
        parsed = parse_freestyle_prompt(prompt)
        print(f"  Input: \"{prompt}\"")
        print(f"    Primary: {parsed['primary_emotion']}, "
              f"Intensity: {parsed['estimated_intensity']:.2f}")
    print("  [PASS]")

    # Test 4: Emotion Prompt Processor
    print("\n[Test 4] Emotion Prompt Processor...")
    processor = EmotionPromptProcessor(config)

    batch_size = 2
    seq_len = 10
    desc_ids = torch.randint(0, 1000, (batch_size, seq_len))

    output = processor(desc_ids)
    print(f"  Input shape: {desc_ids.shape}")
    print(f"  Emotion embedding shape: {output['emotion_embedding'].shape}")
    print(f"  Emotion probs shape: {output['emotion_probs'].shape}")
    print("  [PASS]")

    # Test 5: Semantic Group Layer
    print("\n[Test 5] Semantic Group Modeling...")
    group_layer = SemanticGroupLayer(
        hidden_dim=config.llm_hidden_dim,
        vocab_size=config.num_audio_tokens,
        group_size=config.semantic_group_size,
    )

    hidden = torch.randn(batch_size, 20, config.llm_hidden_dim)
    group_logits = group_layer(hidden)
    print(f"  Input hidden: {hidden.shape}")
    print(f"  Group logits: {group_logits.shape}")
    print(f"  Expected: [2, 60, 4096] (20 * 3 = 60)")
    print("  [PASS]")

    # Test 6: Phoneme Boost Module
    print("\n[Test 6] Phoneme Boost Module...")
    phoneme_boost = PhonemeBoostModule(config)

    phoneme_output = phoneme_boost(hidden)
    print(f"  Phoneme logits shape: {phoneme_output['phoneme_logits'].shape}")
    print(f"  Guided hidden shape: {phoneme_output['guided_hidden'].shape}")
    print(f"  Gate values shape: {phoneme_output['gate_values'].shape}")
    print("  [PASS]")

    # Test 7: EmoVoice LLM Backbone
    print("\n[Test 7] EmoVoice LLM Backbone...")
    backbone = EmoVoiceLLMBackbone(config)

    text_ids = torch.randint(0, 50000, (batch_size, 32))
    emotion_emb = torch.randn(batch_size, config.embedding_dim)

    backbone_output = backbone(text_ids, emotion_embedding=emotion_emb)
    print(f"  Text input: {text_ids.shape}")
    print(f"  Hidden states: {backbone_output['hidden_states'].shape}")
    print(f"  Audio logits: {backbone_output['audio_logits'].shape}")
    print("  [PASS]")

    # Test 8: Full EmoVoice Model
    print("\n[Test 8] Full EmoVoice Model...")
    model = EmoVoice(config)

    desc_ids = torch.randint(0, 1000, (batch_size, 8))

    output = model(
        text_ids=text_ids,
        description_ids=desc_ids,
    )
    print(f"  Audio logits: {output['audio_logits'].shape}")
    print(f"  Prosody tokens: {output['prosody_tokens'].shape}")
    print(f"  Emotion embedding: {output['emotion_output']['emotion_embedding'].shape}")
    print("  [PASS]")

    # Test 9: Loss Computation
    print("\n[Test 9] Loss Computation...")
    target_audio = torch.randint(0, config.num_audio_tokens, (batch_size, 96))

    losses = model.compute_loss(
        text_ids=text_ids,
        target_audio_tokens=target_audio,
        description_ids=desc_ids,
    )
    print(f"  Audio CE loss: {losses['audio_ce'].item():.4f}")
    if 'phoneme_ce' in losses:
        print(f"  Phoneme CE loss: {losses['phoneme_ce'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 10: EmoVoice Adapter
    print("\n[Test 10] EmoVoice Adapter (CSM Integration)...")
    adapter = EmoVoiceAdapter(config)

    # Test with description
    result1 = adapter.encode_description("expressing joyful enthusiasm")
    print(f"  Description encoding prosody tokens: {result1['prosody_tokens'].shape}")
    print(f"  Parsed: {result1['parsed_description']['type']}")

    # Test with emotion
    result2 = adapter.encode_emotion("sad", intensity=0.6)
    print(f"  Emotion encoding prosody tokens: {result2['prosody_tokens'].shape}")

    # Test interpolation
    result3 = adapter.interpolate_descriptions(
        "expressing happiness",
        "expressing sadness",
        t=0.5
    )
    print(f"  Interpolated prosody tokens: {result3['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 11: Generation Description Helper
    print("\n[Test 11] Description Generation...")
    for emotion in ["happy", "angry", "neutral"]:
        for intensity in [0.3, 0.7, 0.95]:
            desc = generate_emotion_description(emotion, intensity)
            print(f"  {emotion} @ {intensity:.1f}: \"{desc}\"")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All EmoVoice tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from emovoice import (
    EmoVoiceConfig,
    EmoVoice,
    EmoVoiceAdapter,
    generate_emotion_description,
    parse_freestyle_prompt,
)

# Initialize
config = EmoVoiceConfig()
adapter = EmoVoiceAdapter(config)

# Option 1: Natural language description
tokens = adapter.encode_description("expressing warm happiness and genuine delight")
prosody_prefix = tokens['prosody_tokens']  # [1, 4, 2048]

# Option 2: Categorical emotion with intensity
tokens = adapter.encode_emotion("sad", intensity=0.8)
prosody_prefix = tokens['prosody_tokens']

# Option 3: Generate description from parameters
desc = generate_emotion_description("angry", intensity=0.9)
# -> "intensely and powerfully expressing fierce indignation"
tokens = adapter.encode_description(desc)

# Option 4: Interpolate between emotions
tokens = adapter.interpolate_descriptions(
    "expressing calm composure",
    "expressing nervous anxiety",
    t=0.4  # 40% toward nervous
)

# Parse freestyle prompt for analysis
info = parse_freestyle_prompt("speaking with quiet grief and deep longing")
print(info['primary_emotion'])  # "sad"
print(info['estimated_intensity'])  # 0.6

# Use with ProsodyControlledCSM:
# combined_prefix = torch.cat([other_prosody, prosody_prefix], dim=1)
# output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
