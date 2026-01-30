"""
MPE-TTS: Multi-Modal Prompt Emotion Encoder for Expressive TTS

Based on "MPE-TTS: Multi-Modal Prompt Emotion Encoder for Expressive Text-To-Speech"
Interspeech 2025 - arXiv:2505.18453
Demo: https://mpetts-demo.github.io/

Key Innovation: Accept emotion prompts as text, image, OR speech. Uses Emotion2Vec for
speech, CLIP+adapter for text/image, MSE loss to unify modalities into shared emotion
latent space.

Architecture:
1. Multi-Modal Prompt Emotion Encoder (MPEE)
   - Speech branch: Emotion2Vec+ Large for robust speech emotion extraction
   - Text branch: CLIP text encoder + learnable adapter
   - Image branch: CLIP image encoder + learnable adapter
   - MSE loss aligns text/image to speech emotion space

2. Disentanglement: Content (Conformer), Timbre (ECAPA-TDNN), Emotion (MPEE), Prosody (VQ)

3. Prosody Predictor: 8-layer Transformer for autoregressive prosody prediction

4. Acoustic Model: Diffusion-based mel-spectrogram generation

Three-Stage Training:
1. Emotion Stage: Train MPEE to align modalities (100 epochs on MEAD-TTS)
2. Acoustic Stage: Train diffusion decoder (500k steps on LibriTTS, 50 epochs on MEAD)
3. Prosody Stage: Train predictor with ECL (50 epochs)

Benefits:
- Multi-modal emotion control (text, image, speech)
- Novel interfaces like emotion from facial expression images
- Unified emotion latent space across modalities
- State-of-the-art emotional expressiveness

Reference: https://arxiv.org/abs/2505.18453
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
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MPETTSConfig:
    """Configuration for MPE-TTS multi-modal emotion encoder."""

    # ========== Multi-Modal Encoder Settings ==========

    # Speech branch (Emotion2Vec)
    emotion2vec_model: str = "iic/emotion2vec_plus_large"  # Emotion2Vec+ Large
    speech_emotion_dim: int = 768  # Emotion2Vec output dimension
    use_pretrained_emotion2vec: bool = True

    # Text/Image branch (CLIP)
    clip_model: str = "openai/clip-vit-base-patch32"  # CLIP model
    clip_dim: int = 512  # CLIP embedding dimension
    freeze_clip: bool = True  # Freeze CLIP during training

    # Adapter settings (for text/image to emotion alignment)
    adapter_hidden_dim: int = 512
    adapter_num_layers: int = 2
    adapter_dropout: float = 0.1
    adapter_activation: str = "gelu"

    # Shared emotion latent space
    emotion_latent_dim: int = 256  # Unified emotion embedding dimension

    # Emotion categories
    num_emotions: int = 8
    emotion_labels: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted", "contempt"
    ])

    # ========== Content Encoder ==========
    content_encoder_layers: int = 5
    content_encoder_dim: int = 512

    # ========== Timbre Encoder (ECAPA-TDNN) ==========
    timbre_dim: int = 192

    # ========== Prosody Encoder (VQ) ==========
    prosody_codebook_size: int = 512
    prosody_code_dim: int = 64
    prosody_num_quantizers: int = 2

    # ========== Prosody Predictor ==========
    predictor_num_layers: int = 8  # Paper: 8 Transformer layers
    predictor_num_heads: int = 8   # Paper: 8 attention heads
    predictor_dim: int = 768       # Paper: 768 embedding dimensions
    predictor_ff_dim: int = 3072   # 4x hidden dim
    predictor_dropout: float = 0.1

    # ========== Output Settings ==========
    output_dim: int = 2048  # CSM prosody hidden dimension
    num_prosody_tokens: int = 4  # Prefix tokens for prosody conditioning

    # ========== Training Settings ==========
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Loss weights
    mse_weight: float = 1.0  # MSE for modality alignment
    ecl_weight: float = 0.5  # Emotion Consistency Loss
    prosody_weight: float = 1.0  # Prosody prediction loss
    reconstruction_weight: float = 0.5  # Mel reconstruction

    # Three-stage training
    stage1_epochs: int = 100  # MPEE training
    stage2_steps: int = 500000  # Acoustic model pre-training
    stage2_finetune_epochs: int = 50  # Acoustic fine-tuning
    stage3_epochs: int = 50  # Prosody predictor training

    # Warmup
    warmup_steps: int = 40000  # Codebook warmup

    # ========== Inference Settings ==========
    default_intensity: float = 0.8
    emotion_scale: float = 1.0


# Emotion to VAD mapping for consistency
EMOTION_VAD = {
    "neutral": (0.0, 0.0, 0.0),
    "happy": (0.8, 0.6, 0.6),
    "sad": (-0.6, -0.4, -0.5),
    "angry": (-0.5, 0.8, 0.7),
    "surprised": (0.3, 0.8, 0.2),
    "fearful": (-0.7, 0.7, -0.7),
    "disgusted": (-0.6, 0.3, 0.4),
    "contempt": (-0.4, 0.2, 0.5),
}

EMOTION_TO_IDX = {e: i for i, e in enumerate(MPETTSConfig().emotion_labels)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}


# =============================================================================
# CLIP ADAPTER
# =============================================================================

class CLIPAdapter(nn.Module):
    """
    Learnable adapter to transform CLIP embeddings to emotion latent space.

    CLIP is frozen, only adapter layers are trained to align with speech emotion space.
    Uses multiple linear layers with non-linearities for flexible mapping.
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config

        # Build adapter layers
        layers = []
        input_dim = config.clip_dim

        for i in range(config.adapter_num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, config.adapter_hidden_dim),
                nn.LayerNorm(config.adapter_hidden_dim) if config.use_layer_norm else nn.Identity(),
                nn.GELU() if config.adapter_activation == "gelu" else nn.ReLU(),
                nn.Dropout(config.adapter_dropout),
            ])
            input_dim = config.adapter_hidden_dim

        # Final projection to emotion latent space
        layers.append(nn.Linear(input_dim, config.emotion_latent_dim))

        self.adapter = nn.Sequential(*layers)

    def forward(self, clip_embedding: Tensor) -> Tensor:
        """
        Transform CLIP embedding to emotion latent space.

        Args:
            clip_embedding: [batch, clip_dim] - CLIP text or image embedding

        Returns:
            emotion_embedding: [batch, emotion_latent_dim]
        """
        return self.adapter(clip_embedding)


# =============================================================================
# SPEECH EMOTION ENCODER (Emotion2Vec)
# =============================================================================

class SpeechEmotionEncoder(nn.Module):
    """
    Speech emotion encoder using Emotion2Vec+ Large.

    Emotion2Vec is a self-supervised model pre-trained on speech emotion data.
    We use it to extract emotion-rich representations from speech.

    Output is projected to the shared emotion latent space.
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._processor = None

        # Projection to shared emotion space
        self.projection = nn.Sequential(
            nn.Linear(config.speech_emotion_dim, config.adapter_hidden_dim),
            nn.LayerNorm(config.adapter_hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_hidden_dim, config.emotion_latent_dim),
        )

        # Attentive pooling for sequence to vector
        self.attention = nn.Sequential(
            nn.Linear(config.speech_emotion_dim, config.speech_emotion_dim // 4),
            nn.Tanh(),
            nn.Linear(config.speech_emotion_dim // 4, 1),
        )

    def _lazy_load(self, device: torch.device):
        """Lazy load Emotion2Vec model on first use."""
        if self._model is not None:
            return

        try:
            from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification

            self._processor = AutoProcessor.from_pretrained(
                self.config.emotion2vec_model,
                trust_remote_code=True
            )

            self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.config.emotion2vec_model,
                trust_remote_code=True,
                output_hidden_states=True
            ).to(device)

            self._model.eval()

        except Exception as e:
            warnings.warn(
                f"Failed to load Emotion2Vec from {self.config.emotion2vec_model}: {e}\n"
                "Using fallback mock encoder."
            )
            self._model = "mock"

    def extract_features(
        self,
        audio: Tensor,  # [batch, samples]
        sample_rate: int = 16000,
    ) -> Tensor:
        """
        Extract emotion features from audio.

        Args:
            audio: [batch, samples] at 16kHz
            sample_rate: Audio sample rate

        Returns:
            features: [batch, seq, speech_emotion_dim]
        """
        device = audio.device
        self._lazy_load(device)

        if self._model == "mock":
            # Mock features for testing
            batch_size = audio.shape[0]
            seq_len = audio.shape[1] // 320  # ~50Hz
            return torch.randn(batch_size, seq_len, self.config.speech_emotion_dim, device=device)

        with torch.no_grad():
            # Process audio
            inputs = self._processor(
                audio.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            ).to(device)

            # Get hidden states
            outputs = self._model(**inputs)

            # Use last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                features = outputs.hidden_states[-1]  # [batch, seq, hidden]
            else:
                # Fallback to pooled output
                features = outputs.logits.unsqueeze(1)

        return features

    def forward(
        self,
        audio: Optional[Tensor] = None,  # [batch, samples]
        features: Optional[Tensor] = None,  # [batch, seq, speech_emotion_dim]
        sample_rate: int = 16000,
    ) -> Tensor:
        """
        Encode speech to emotion embedding.

        Args:
            audio: [batch, samples] raw audio at 16kHz
            features: [batch, seq, speech_emotion_dim] pre-extracted features
            sample_rate: Audio sample rate

        Returns:
            emotion_embedding: [batch, emotion_latent_dim]
        """
        if features is None:
            if audio is None:
                raise ValueError("Either audio or features must be provided")
            features = self.extract_features(audio, sample_rate)

        # Attentive pooling
        attn_weights = self.attention(features)  # [batch, seq, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        pooled = torch.sum(features * attn_weights, dim=1)  # [batch, speech_emotion_dim]

        # Project to shared emotion space
        emotion_emb = self.projection(pooled)  # [batch, emotion_latent_dim]

        return emotion_emb


# =============================================================================
# TEXT EMOTION ENCODER (CLIP Text)
# =============================================================================

class TextEmotionEncoder(nn.Module):
    """
    Text emotion encoder using CLIP text encoder + learnable adapter.

    CLIP encoder is frozen, only the adapter is trained to align
    text emotion descriptions with the speech emotion space.
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config
        self._clip_model = None
        self._tokenizer = None

        # Adapter for CLIP → emotion space
        self.adapter = CLIPAdapter(config)

    def _lazy_load(self, device: torch.device):
        """Lazy load CLIP model on first use."""
        if self._clip_model is not None:
            return

        try:
            from transformers import CLIPModel, CLIPTokenizer

            self._tokenizer = CLIPTokenizer.from_pretrained(self.config.clip_model)
            self._clip_model = CLIPModel.from_pretrained(self.config.clip_model).to(device)

            if self.config.freeze_clip:
                self._clip_model.eval()
                for param in self._clip_model.parameters():
                    param.requires_grad = False

        except Exception as e:
            warnings.warn(
                f"Failed to load CLIP from {self.config.clip_model}: {e}\n"
                "Using fallback mock encoder."
            )
            self._clip_model = "mock"

    def forward(
        self,
        text: Optional[Union[str, List[str]]] = None,
        text_embeddings: Optional[Tensor] = None,  # [batch, clip_dim]
    ) -> Tensor:
        """
        Encode text emotion description to emotion embedding.

        Args:
            text: Emotion description string(s) like "expressing genuine happiness"
            text_embeddings: Pre-computed CLIP text embeddings [batch, clip_dim]

        Returns:
            emotion_embedding: [batch, emotion_latent_dim]
        """
        if text_embeddings is None:
            if text is None:
                raise ValueError("Either text or text_embeddings must be provided")

            if isinstance(text, str):
                text = [text]

            device = next(self.adapter.parameters()).device
            self._lazy_load(device)

            if self._clip_model == "mock":
                # Mock embeddings for testing
                return torch.randn(len(text), self.config.emotion_latent_dim, device=device)

            # Tokenize and encode
            with torch.no_grad():
                inputs = self._tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to(device)

                text_embeddings = self._clip_model.get_text_features(**inputs)

        # Apply adapter
        emotion_emb = self.adapter(text_embeddings)

        return emotion_emb


# =============================================================================
# IMAGE EMOTION ENCODER (CLIP Vision)
# =============================================================================

class ImageEmotionEncoder(nn.Module):
    """
    Image emotion encoder using CLIP vision encoder + learnable adapter.

    Enables novel interfaces like emotion from facial expression images.
    CLIP encoder is frozen, only the adapter is trained.
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config
        self._clip_model = None
        self._processor = None

        # Adapter for CLIP → emotion space
        self.adapter = CLIPAdapter(config)

    def _lazy_load(self, device: torch.device):
        """Lazy load CLIP model on first use."""
        if self._clip_model is not None:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.config.clip_model)
            self._clip_model = CLIPModel.from_pretrained(self.config.clip_model).to(device)

            if self.config.freeze_clip:
                self._clip_model.eval()
                for param in self._clip_model.parameters():
                    param.requires_grad = False

        except Exception as e:
            warnings.warn(
                f"Failed to load CLIP from {self.config.clip_model}: {e}\n"
                "Using fallback mock encoder."
            )
            self._clip_model = "mock"

    def forward(
        self,
        images: Optional[Any] = None,  # PIL images or tensor
        image_embeddings: Optional[Tensor] = None,  # [batch, clip_dim]
    ) -> Tensor:
        """
        Encode image to emotion embedding.

        Args:
            images: PIL Image(s) or preprocessed tensor
            image_embeddings: Pre-computed CLIP image embeddings [batch, clip_dim]

        Returns:
            emotion_embedding: [batch, emotion_latent_dim]
        """
        if image_embeddings is None:
            if images is None:
                raise ValueError("Either images or image_embeddings must be provided")

            device = next(self.adapter.parameters()).device
            self._lazy_load(device)

            if self._clip_model == "mock":
                # Mock embeddings for testing
                batch_size = 1 if not isinstance(images, list) else len(images)
                return torch.randn(batch_size, self.config.emotion_latent_dim, device=device)

            # Process and encode images
            with torch.no_grad():
                inputs = self._processor(images=images, return_tensors="pt").to(device)
                image_embeddings = self._clip_model.get_image_features(**inputs)

        # Apply adapter
        emotion_emb = self.adapter(image_embeddings)

        return emotion_emb


# =============================================================================
# MULTI-MODAL PROMPT EMOTION ENCODER (MPEE)
# =============================================================================

class MultiModalPromptEmotionEncoder(nn.Module):
    """
    Multi-Modal Prompt Emotion Encoder (MPEE).

    Core contribution of MPE-TTS: Accepts emotion prompts as text, image, OR speech.
    All modalities are aligned to a shared emotion latent space via MSE loss.

    Training:
        - Speech encoder provides ground truth emotion embeddings
        - Text/Image adapters are trained to match speech embeddings
        - Loss: MSE(E_text, E_speech) + MSE(E_image, E_speech)

    Inference:
        - Can use ANY modality for emotion specification
        - Enables novel interfaces like emotion from facial expression photos
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config

        # Modality-specific encoders
        self.speech_encoder = SpeechEmotionEncoder(config)
        self.text_encoder = TextEmotionEncoder(config)
        self.image_encoder = ImageEmotionEncoder(config)

        # Emotion classifier (for consistency loss)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.emotion_latent_dim, config.adapter_hidden_dim),
            nn.LayerNorm(config.adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_hidden_dim, config.num_emotions),
        )

    def forward(
        self,
        speech_audio: Optional[Tensor] = None,
        speech_features: Optional[Tensor] = None,
        text: Optional[Union[str, List[str]]] = None,
        text_embeddings: Optional[Tensor] = None,
        images: Optional[Any] = None,
        image_embeddings: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Encode multi-modal emotion prompts.

        Args:
            speech_audio: [batch, samples] raw audio at 16kHz
            speech_features: [batch, seq, dim] pre-extracted speech features
            text: Emotion description string(s)
            text_embeddings: Pre-computed CLIP text embeddings
            images: PIL images or tensor
            image_embeddings: Pre-computed CLIP image embeddings
            return_all: Whether to return all modality embeddings

        Returns:
            Dictionary with:
                - emotion_embedding: Unified emotion embedding (uses available modality)
                - speech_emotion: Speech emotion embedding (if available)
                - text_emotion: Text emotion embedding (if available)
                - image_emotion: Image emotion embedding (if available)
                - emotion_logits: Emotion classification logits
        """
        result = {}

        # Encode each available modality
        if speech_audio is not None or speech_features is not None:
            result['speech_emotion'] = self.speech_encoder(
                audio=speech_audio,
                features=speech_features
            )

        if text is not None or text_embeddings is not None:
            result['text_emotion'] = self.text_encoder(
                text=text,
                text_embeddings=text_embeddings
            )

        if images is not None or image_embeddings is not None:
            result['image_emotion'] = self.image_encoder(
                images=images,
                image_embeddings=image_embeddings
            )

        # Select unified emotion embedding (priority: speech > text > image)
        if 'speech_emotion' in result:
            emotion_emb = result['speech_emotion']
        elif 'text_emotion' in result:
            emotion_emb = result['text_emotion']
        elif 'image_emotion' in result:
            emotion_emb = result['image_emotion']
        else:
            raise ValueError("At least one modality input must be provided")

        result['emotion_embedding'] = emotion_emb

        # Emotion classification
        result['emotion_logits'] = self.emotion_classifier(emotion_emb)

        return result

    def compute_alignment_loss(
        self,
        speech_audio: Tensor,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Any] = None,
        text_embeddings: Optional[Tensor] = None,
        image_embeddings: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute modality alignment loss (MSE between text/image and speech).

        The speech encoder is the "anchor" and provides ground truth emotions.
        Text and image adapters are trained to match speech emotion embeddings.

        Loss: L_MPEE = MSE(E_text, E_speech) + MSE(E_image, E_speech)
        """
        losses = {}

        # Get speech emotion (ground truth)
        speech_emotion = self.speech_encoder(audio=speech_audio)

        # Text alignment loss
        if text is not None or text_embeddings is not None:
            text_emotion = self.text_encoder(text=text, text_embeddings=text_embeddings)
            losses['text_mse'] = F.mse_loss(text_emotion, speech_emotion.detach())

        # Image alignment loss
        if images is not None or image_embeddings is not None:
            image_emotion = self.image_encoder(images=images, image_embeddings=image_embeddings)
            losses['image_mse'] = F.mse_loss(image_emotion, speech_emotion.detach())

        # Total MPEE loss
        losses['mpee_total'] = sum(losses.values())

        return losses


# =============================================================================
# PROSODY PREDICTOR (LLM-like Transformer)
# =============================================================================

class ProsodyPredictor(nn.Module):
    """
    Prosody Predictor: LLM-like autoregressive model for prosody prediction.

    Architecture: 8 Transformer layers with 8 attention heads and 768 embedding dims.
    Accepts content, timbre, and emotion as conditioning inputs.

    Output: Prosody codes (VQ indices) or continuous prosody vectors.
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.content_proj = nn.Linear(config.content_encoder_dim, config.predictor_dim)
        self.timbre_proj = nn.Linear(config.timbre_dim, config.predictor_dim)
        self.emotion_proj = nn.Linear(config.emotion_latent_dim, config.predictor_dim)

        # Prosody embedding for autoregressive decoding
        self.prosody_embedding = nn.Embedding(
            config.prosody_codebook_size + 1,  # +1 for start token
            config.predictor_dim
        )

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(config.predictor_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.predictor_dim,
            nhead=config.predictor_num_heads,
            dim_feedforward=config.predictor_ff_dim,
            dropout=config.predictor_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.predictor_num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.predictor_dim, config.prosody_codebook_size)

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.predictor_dim)

    def forward(
        self,
        content: Tensor,  # [batch, seq, content_dim]
        timbre: Tensor,   # [batch, timbre_dim]
        emotion: Tensor,  # [batch, emotion_latent_dim]
        prosody_target: Optional[Tensor] = None,  # [batch, seq] - prosody code indices
        teacher_forcing: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Predict prosody codes from content, timbre, and emotion.

        Args:
            content: Content features [batch, seq, content_dim]
            timbre: Timbre embedding [batch, timbre_dim]
            emotion: Emotion embedding [batch, emotion_latent_dim]
            prosody_target: Target prosody codes for teacher forcing
            teacher_forcing: Whether to use teacher forcing

        Returns:
            Dictionary with prosody_logits, prosody_codes
        """
        batch_size, seq_len, _ = content.shape
        device = content.device

        # Project inputs
        content_emb = self.content_proj(content)  # [batch, seq, predictor_dim]
        timbre_emb = self.timbre_proj(timbre).unsqueeze(1)  # [batch, 1, predictor_dim]
        emotion_emb = self.emotion_proj(emotion).unsqueeze(1)  # [batch, 1, predictor_dim]

        # Create memory (content + global conditioning)
        memory = content_emb + timbre_emb + emotion_emb
        memory = self.pos_encoder(memory)

        if teacher_forcing and prosody_target is not None:
            # Shift target for autoregressive training
            start_token = torch.full((batch_size, 1), self.config.prosody_codebook_size, device=device)
            tgt_input = torch.cat([start_token, prosody_target[:, :-1]], dim=1)
            tgt_emb = self.prosody_embedding(tgt_input)
            tgt_emb = self.pos_encoder(tgt_emb)

            # Causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

            # Decode
            output = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
            output = self.layer_norm(output)
            logits = self.output_proj(output)

            return {
                'prosody_logits': logits,
                'prosody_codes': logits.argmax(dim=-1),
            }
        else:
            # Autoregressive generation
            return self._generate(memory, seq_len, device)

    def _generate(
        self,
        memory: Tensor,
        max_len: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Autoregressive prosody generation."""
        batch_size = memory.shape[0]

        # Start with start token
        generated = torch.full((batch_size, 1), self.config.prosody_codebook_size, device=device)
        all_logits = []

        for _ in range(max_len):
            tgt_emb = self.prosody_embedding(generated)
            tgt_emb = self.pos_encoder(tgt_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.shape[1], device=device)

            output = self.transformer(tgt_emb, memory, tgt_mask=tgt_mask)
            output = self.layer_norm(output[:, -1:])
            logits = self.output_proj(output)

            all_logits.append(logits)
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

        return {
            'prosody_logits': torch.cat(all_logits, dim=1),
            'prosody_codes': generated[:, 1:],  # Remove start token
        }


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# EMOTION CONSISTENCY LOSS (ECL)
# =============================================================================

class EmotionConsistencyLoss(nn.Module):
    """
    Emotion Consistency Loss (ECL).

    Ensures predicted prosody codes preserve emotional information by
    training a classifier to recognize emotions from predicted prosody.

    The classifier is trained on predicted prosody → emotion cross-entropy.
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config

        # Prosody → Emotion classifier
        self.prosody_classifier = nn.Sequential(
            nn.Embedding(config.prosody_codebook_size, config.predictor_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.predictor_dim,
                    nhead=4,
                    dim_feedforward=config.predictor_dim * 2,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                num_layers=2,
            ),
            AttentivePooling(config.predictor_dim),
            nn.Linear(config.predictor_dim, config.num_emotions),
        )

    def forward(
        self,
        prosody_codes: Tensor,  # [batch, seq] - predicted prosody codes
        emotion_labels: Tensor,  # [batch] - target emotion indices
    ) -> Tensor:
        """
        Compute emotion consistency loss.

        Args:
            prosody_codes: Predicted prosody code indices
            emotion_labels: Ground truth emotion labels

        Returns:
            Cross-entropy loss
        """
        logits = self.prosody_classifier(prosody_codes)
        return F.cross_entropy(logits, emotion_labels)


class AttentivePooling(nn.Module):
    """Attentive pooling for sequence to vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Handle both raw codes and transformer output
        if x.dim() == 2:  # [batch, seq] - codes
            return x.float().mean(dim=1)

        attn_weights = F.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)


# =============================================================================
# MPE-TTS COMPLETE LOSS
# =============================================================================

class MPETTSLoss(nn.Module):
    """
    Complete loss function for MPE-TTS training.

    Combines:
    1. MPEE Loss: MSE for modality alignment (text/image → speech)
    2. Prosody Loss: Cross-entropy for prosody code prediction
    3. ECL: Emotion consistency loss for predicted prosody
    4. Reconstruction: Mel spectrogram reconstruction (optional)
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config

        # ECL module
        self.ecl = EmotionConsistencyLoss(config)

    def forward(
        self,
        mpee_outputs: Dict[str, Tensor],
        prosody_outputs: Optional[Dict[str, Tensor]] = None,
        prosody_targets: Optional[Tensor] = None,
        emotion_labels: Optional[Tensor] = None,
        mel_predicted: Optional[Tensor] = None,
        mel_target: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute all MPE-TTS losses.

        Args:
            mpee_outputs: Output from MPEE forward pass
            prosody_outputs: Output from prosody predictor
            prosody_targets: Ground truth prosody codes
            emotion_labels: Ground truth emotion labels
            mel_predicted: Predicted mel spectrogram
            mel_target: Target mel spectrogram

        Returns:
            Dictionary with all loss components
        """
        losses = {}

        # MPEE alignment losses (already computed if available)
        if 'text_mse' in mpee_outputs:
            losses['text_mse'] = mpee_outputs['text_mse']
        if 'image_mse' in mpee_outputs:
            losses['image_mse'] = mpee_outputs['image_mse']

        # Prosody prediction loss
        if prosody_outputs is not None and prosody_targets is not None:
            logits = prosody_outputs['prosody_logits']
            losses['prosody_ce'] = F.cross_entropy(
                logits.view(-1, self.config.prosody_codebook_size),
                prosody_targets.view(-1),
            )

        # Emotion consistency loss
        if prosody_outputs is not None and emotion_labels is not None:
            losses['ecl'] = self.ecl(prosody_outputs['prosody_codes'], emotion_labels)

        # Mel reconstruction loss
        if mel_predicted is not None and mel_target is not None:
            losses['mel_l1'] = F.l1_loss(mel_predicted, mel_target)
            losses['mel_l2'] = F.mse_loss(mel_predicted, mel_target)
            losses['reconstruction'] = losses['mel_l1'] + losses['mel_l2']

        # Emotion classification loss (from MPEE)
        if 'emotion_logits' in mpee_outputs and emotion_labels is not None:
            logits = mpee_outputs['emotion_logits']
            # Handle batch size mismatch
            if logits.shape[0] != emotion_labels.shape[0]:
                # Truncate to smaller batch
                min_batch = min(logits.shape[0], emotion_labels.shape[0])
                logits = logits[:min_batch]
                emotion_labels = emotion_labels[:min_batch]
            losses['emotion_ce'] = F.cross_entropy(logits, emotion_labels)

        # Compute weighted total
        total = 0.0
        if 'text_mse' in losses:
            total += self.config.mse_weight * losses['text_mse']
        if 'image_mse' in losses:
            total += self.config.mse_weight * losses['image_mse']
        if 'prosody_ce' in losses:
            total += self.config.prosody_weight * losses['prosody_ce']
        if 'ecl' in losses:
            total += self.config.ecl_weight * losses['ecl']
        if 'reconstruction' in losses:
            total += self.config.reconstruction_weight * losses['reconstruction']
        if 'emotion_ce' in losses:
            total += 0.5 * losses['emotion_ce']

        losses['total'] = total if isinstance(total, Tensor) else torch.tensor(total)

        return losses


# =============================================================================
# MPE-TTS ADAPTER FOR CSM INTEGRATION
# =============================================================================

class MPETTSAdapter(nn.Module):
    """
    Adapter for integrating MPE-TTS with ProsodyControlledCSM.

    Converts multi-modal emotion prompts to prosody prefix tokens.

    Usage:
        adapter = MPETTSAdapter(config).cuda()

        # From speech emotion
        tokens = adapter.from_speech(audio)

        # From text emotion
        tokens = adapter.from_text("expressing genuine happiness")

        # From image emotion
        tokens = adapter.from_image(facial_expression_image)

        # Combined with CSM
        output = csm_model(input_ids, prosody_prefix=tokens['prosody_tokens'])
    """

    def __init__(self, config: MPETTSConfig):
        super().__init__()
        self.config = config

        # Multi-modal encoder
        self.mpee = MultiModalPromptEmotionEncoder(config)

        # Projection to prosody prefix tokens
        self.token_proj = nn.Sequential(
            nn.Linear(config.emotion_latent_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Learnable positional embeddings for prefix tokens
        self.prefix_positions = nn.Parameter(
            torch.randn(1, config.num_prosody_tokens, config.output_dim) * 0.02
        )

    def forward(
        self,
        speech_audio: Optional[Tensor] = None,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Any] = None,
        intensity: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Generate prosody prefix tokens from multi-modal emotion prompts.

        Args:
            speech_audio: [batch, samples] raw audio at 16kHz
            text: Emotion description string(s)
            images: PIL images or tensor
            intensity: Emotion intensity scaling factor

        Returns:
            Dictionary with:
                - prosody_tokens: [batch, num_tokens, output_dim]
                - emotion_embedding: [batch, emotion_latent_dim]
                - emotion_logits: [batch, num_emotions]
        """
        # Get emotion embedding from available modality
        mpee_out = self.mpee(
            speech_audio=speech_audio,
            text=text,
            images=images,
        )

        emotion_emb = mpee_out['emotion_embedding']  # [batch, emotion_latent_dim]

        # Apply intensity scaling
        emotion_emb = emotion_emb * intensity

        # Project to prefix token dimension
        token_emb = self.token_proj(emotion_emb)  # [batch, output_dim]

        # Expand to multiple prefix tokens with positional embeddings
        batch_size = emotion_emb.shape[0]
        tokens = token_emb.unsqueeze(1).expand(-1, self.config.num_prosody_tokens, -1)
        tokens = tokens + self.prefix_positions

        return {
            'prosody_tokens': tokens,
            'emotion_embedding': emotion_emb,
            'emotion_logits': mpee_out['emotion_logits'],
        }

    def from_speech(
        self,
        audio: Tensor,
        intensity: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Generate prosody tokens from speech audio."""
        return self(speech_audio=audio, intensity=intensity)

    def from_text(
        self,
        text: Union[str, List[str]],
        intensity: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Generate prosody tokens from text description."""
        return self(text=text, intensity=intensity)

    def from_image(
        self,
        images: Any,
        intensity: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Generate prosody tokens from image(s)."""
        return self(images=images, intensity=intensity)

    def from_emotion_label(
        self,
        emotion: str,
        intensity: float = 0.8,
    ) -> Dict[str, Tensor]:
        """
        Generate prosody tokens from emotion label.

        Uses text-based pathway with generated emotion description.
        """
        descriptions = {
            "neutral": "speaking in a calm, neutral tone",
            "happy": "expressing genuine happiness and joy",
            "sad": "speaking with deep sadness and melancholy",
            "angry": "expressing intense anger and frustration",
            "surprised": "speaking with sudden astonishment",
            "fearful": "expressing anxious fear and worry",
            "disgusted": "speaking with disgust and revulsion",
            "contempt": "expressing cold contempt and disdain",
        }

        text = descriptions.get(emotion, descriptions["neutral"])
        return self.from_text(text, intensity=intensity)

    def interpolate_emotions(
        self,
        text1: str,
        text2: str,
        t: float = 0.5,
        method: str = "linear",
    ) -> Dict[str, Tensor]:
        """
        Interpolate between two text emotion descriptions.

        Args:
            text1: First emotion description
            text2: Second emotion description
            t: Interpolation factor (0.0 = text1, 1.0 = text2)
            method: "linear" or "spherical"

        Returns:
            Interpolated prosody tokens
        """
        # Get embeddings for both
        out1 = self.mpee(text=text1)
        out2 = self.mpee(text=text2)

        emb1 = out1['emotion_embedding']
        emb2 = out2['emotion_embedding']

        # Interpolate
        if method == "spherical":
            # Spherical linear interpolation (SLERP)
            emb1_norm = F.normalize(emb1, dim=-1)
            emb2_norm = F.normalize(emb2, dim=-1)

            omega = torch.acos(torch.clamp((emb1_norm * emb2_norm).sum(dim=-1, keepdim=True), -1.0, 1.0))

            sin_omega = torch.sin(omega)
            s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
            s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

            # Preserve magnitudes
            mag1 = emb1.norm(dim=-1, keepdim=True)
            mag2 = emb2.norm(dim=-1, keepdim=True)
            interpolated_mag = (1 - t) * mag1 + t * mag2

            interpolated = (s1 * emb1_norm + s2 * emb2_norm) * interpolated_mag
        else:
            # Linear interpolation
            interpolated = (1 - t) * emb1 + t * emb2

        # Project to tokens
        token_emb = self.token_proj(interpolated)
        batch_size = interpolated.shape[0]
        tokens = token_emb.unsqueeze(1).expand(-1, self.config.num_prosody_tokens, -1)
        tokens = tokens + self.prefix_positions

        return {
            'prosody_tokens': tokens,
            'emotion_embedding': interpolated,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_mpetts_adapter(
    config: Optional[MPETTSConfig] = None,
    checkpoint_path: Optional[str] = None,
) -> MPETTSAdapter:
    """
    Create MPE-TTS adapter with optional checkpoint loading.

    Args:
        config: MPETTSConfig or None for defaults
        checkpoint_path: Path to trained checkpoint

    Returns:
        MPETTSAdapter instance
    """
    if config is None:
        config = MPETTSConfig()

    adapter = MPETTSAdapter(config)

    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            adapter.load_state_dict(state_dict['model_state_dict'])
        else:
            adapter.load_state_dict(state_dict)

    return adapter


def emotion_description_from_image(
    image_path: str,
    emotion_model: Optional[str] = None,
) -> str:
    """
    Generate emotion description from facial expression image.

    This is a placeholder for integration with facial expression recognition.
    In practice, could use:
    - Pre-trained FER models (e.g., AffectNet)
    - CLIP zero-shot classification
    - VLM (e.g., LLaVA, GPT-4V)

    Args:
        image_path: Path to image file
        emotion_model: Optional model name for FER

    Returns:
        Text description of detected emotion
    """
    # Placeholder - in practice, use FER or VLM
    return "expressing a neutral expression"


def estimate_modality_alignment_quality(
    adapter: MPETTSAdapter,
    test_audio: Tensor,
    test_texts: List[str],
) -> Dict[str, float]:
    """
    Estimate how well text/image modalities align with speech emotions.

    Computes cosine similarity between speech and text emotion embeddings
    for the same emotional content.

    Args:
        adapter: Trained MPE-TTS adapter
        test_audio: [batch, samples] test audio with known emotions
        test_texts: Corresponding emotion descriptions

    Returns:
        Dictionary with alignment metrics
    """
    adapter.eval()
    with torch.no_grad():
        speech_out = adapter.from_speech(test_audio)
        text_out = adapter.from_text(test_texts)

        speech_emb = F.normalize(speech_out['emotion_embedding'], dim=-1)
        text_emb = F.normalize(text_out['emotion_embedding'], dim=-1)

        # Compute cosine similarity
        similarity = (speech_emb * text_emb).sum(dim=-1)

        return {
            'mean_similarity': similarity.mean().item(),
            'min_similarity': similarity.min().item(),
            'max_similarity': similarity.max().item(),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Test MPE-TTS components
    print("Testing MPE-TTS multi-modal emotion encoder...")

    config = MPETTSConfig()
    adapter = MPETTSAdapter(config)

    # Test text emotion
    result = adapter.from_text("expressing genuine happiness and warmth")
    print(f"Text emotion - tokens shape: {result['prosody_tokens'].shape}")
    print(f"Text emotion - embedding shape: {result['emotion_embedding'].shape}")

    # Test emotion label
    result = adapter.from_emotion_label("angry", intensity=0.9)
    print(f"Label emotion - tokens shape: {result['prosody_tokens'].shape}")

    # Test speech emotion (with mock features)
    dummy_audio = torch.randn(2, 16000)
    result = adapter.from_speech(dummy_audio)
    print(f"Speech emotion - tokens shape: {result['prosody_tokens'].shape}")

    # Test emotion interpolation
    result = adapter.interpolate_emotions(
        "expressing sadness",
        "expressing happiness",
        t=0.5,
        method="spherical",
    )
    print(f"Interpolated emotion - tokens shape: {result['prosody_tokens'].shape}")

    # Test MPEE
    mpee = MultiModalPromptEmotionEncoder(config)
    mpee_out = mpee(text=["happy voice", "sad voice"], speech_audio=dummy_audio)
    print(f"MPEE output keys: {mpee_out.keys()}")

    # Test alignment loss
    alignment_losses = mpee.compute_alignment_loss(
        speech_audio=dummy_audio,
        text=["happy voice", "sad voice"],
    )
    print(f"Alignment losses: {alignment_losses}")

    # Test prosody predictor
    predictor = ProsodyPredictor(config)
    content = torch.randn(2, 50, config.content_encoder_dim)
    timbre = torch.randn(2, config.timbre_dim)
    emotion = torch.randn(2, config.emotion_latent_dim)
    prosody_target = torch.randint(0, config.prosody_codebook_size, (2, 50))

    pred_out = predictor(content, timbre, emotion, prosody_target)
    print(f"Prosody predictor - logits shape: {pred_out['prosody_logits'].shape}")
    print(f"Prosody predictor - codes shape: {pred_out['prosody_codes'].shape}")

    # Test loss function
    loss_fn = MPETTSLoss(config)
    losses = loss_fn(
        mpee_outputs=mpee_out,
        prosody_outputs=pred_out,
        prosody_targets=prosody_target,
        emotion_labels=torch.randint(0, config.num_emotions, (2,)),
    )
    print(f"Loss components: {list(losses.keys())}")
    print(f"Total loss: {losses['total'].item():.4f}")

    print("\nMPE-TTS test completed successfully!")
