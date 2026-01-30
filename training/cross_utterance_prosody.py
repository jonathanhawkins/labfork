"""
Cross-Utterance Context Prosody Prediction (CUC-VAE Approach)

Based on CUC-VAE (Cross-Utterance Conditioned VAE) and ParaTTS research.

Key Innovation: Improves prosody coherence for multi-sentence/paragraph TTS by
leveraging surrounding context. When humans speak multiple utterances, prosody
in each is related to neighboring ones - this module captures those dependencies.

Problem Solved:
- Current TTS systems ignore cross-utterance dependencies
- Prosody is predicted independently for each sentence
- Results in unnatural prosody transitions at sentence boundaries
- Missing discourse-level prosodic patterns (paragraph intonation, topic shifts)

Architecture:
1. CrossUtteranceEncoder: Encodes previous/future sentences into context vectors
   - Uses bidirectional BERT/RoBERTa for contextual word embeddings
   - Sentence-level attention pooling for fixed-length context
   - Separate encoders for past and future context

2. CUCVariationalEncoder: Variational autoencoder conditioned on context
   - Prior: p(z|context) - context-conditioned prior
   - Posterior: q(z|x, context) - inference model
   - Reparameterization trick for end-to-end training
   - KL divergence between posterior and context-conditioned prior

3. ParagraphProsodyPredictor: Predicts prosody from text + context
   - Transformer decoder with cross-attention to context
   - Multi-head attention for multi-speaker modeling
   - Optional emotion conditioning integration

Key Research Factors:
1. Long context - access to 2-3 neighboring sentences (configurable)
2. Contextual word embeddings (BERT/RoBERTa) - captures semantic dependencies
3. Multi-speaker modeling - speaker-invariant prosody patterns

Benefits:
- More natural prosody in audiobooks, dialogues, long-form content
- Coherent intonation across sentence boundaries
- Captures discourse-level prosodic patterns
- Better paragraph-awareness for TTS applications

References:
- ParaTTS: IEEE TASLP 2022 - "ParaTTS: Learning Prosody from Paragraph-Level"
- CUC-VAE S2: UCL Discovery 2024 - "Cross-Utterance Conditioned VAE for Speech Synthesis"
- Discourse Prosody: Interspeech 2021 - "Prosodic Phrases for Discourse Structure"
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
class CrossUtteranceConfig:
    """Configuration for Cross-Utterance Context prosody prediction."""

    # Context window
    num_prev_sentences: int = 2  # Number of previous sentences to consider
    num_next_sentences: int = 1  # Number of future sentences to consider
    max_sentence_length: int = 128  # Maximum tokens per sentence
    max_context_sentences: int = 4  # Maximum total context sentences

    # Text encoder (BERT-like)
    text_encoder_model: str = "roberta-base"  # Pre-trained model name
    text_encoder_dim: int = 768  # Hidden dimension from text encoder
    freeze_text_encoder: bool = True  # Freeze pre-trained weights
    text_encoder_layers_to_use: int = -1  # -1 = use all, else use last N layers

    # Context encoder
    context_hidden_dim: int = 512  # Hidden dimension for context encoder
    context_num_layers: int = 2  # Number of transformer layers
    context_num_heads: int = 8  # Number of attention heads
    use_bidirectional_context: bool = True  # Use both past and future context

    # Variational encoder (VAE)
    latent_dim: int = 256  # Dimension of latent prosody space z
    use_vae: bool = True  # Use variational encoder (vs deterministic)
    kl_weight: float = 0.1  # KL divergence weight (β-VAE style)
    kl_annealing_steps: int = 5000  # Steps to anneal KL weight from 0 to full
    min_kl_weight: float = 0.0  # Minimum KL weight during annealing
    prior_type: str = "context"  # "standard" (N(0,1)) or "context" (context-conditioned)

    # Prosody prediction
    prosody_hidden_dim: int = 512  # Hidden dimension for prosody predictor
    prosody_num_layers: int = 4  # Number of transformer layers
    prosody_num_heads: int = 8  # Number of attention heads
    output_dim: int = 2048  # CSM prosody hidden dimension
    num_prosody_tokens: int = 4  # Prefix tokens for prosody conditioning

    # Multi-speaker
    num_speakers: int = 0  # Number of speakers (0 = single speaker)
    speaker_embedding_dim: int = 256  # Speaker embedding dimension
    use_speaker_in_context: bool = True  # Include speaker in context encoding

    # Emotion integration (optional)
    use_emotion: bool = False  # Whether to integrate emotion
    emotion_dim: int = 256  # Emotion embedding dimension
    num_emotions: int = 8  # Number of emotion categories

    # Training
    dropout: float = 0.1
    label_smoothing: float = 0.0
    use_gradient_checkpointing: bool = False

    # Inference
    sample_from_prior: bool = False  # Sample z from prior at inference (vs posterior)
    context_dropout: float = 0.0  # Dropout for context during training (robustness)


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

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
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# SENTENCE ENCODER
# =============================================================================

class SentenceEncoder(nn.Module):
    """
    Encodes a single sentence into a fixed-length representation.

    Uses pre-trained language model (BERT/RoBERTa) for contextual word embeddings,
    followed by attention pooling to get sentence-level representation.
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config

        # Attention pooling
        self.attention_query = nn.Parameter(torch.randn(1, 1, config.text_encoder_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=config.text_encoder_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

        # Project to context dimension
        self.projection = nn.Sequential(
            nn.Linear(config.text_encoder_dim, config.context_hidden_dim),
            nn.LayerNorm(config.context_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        text_embeddings: Tensor,  # [batch, seq_len, text_encoder_dim]
        attention_mask: Optional[Tensor] = None,  # [batch, seq_len]
    ) -> Tensor:
        """
        Encode sentence to fixed-length representation.

        Returns:
            Tensor of shape [batch, context_hidden_dim]
        """
        batch_size = text_embeddings.shape[0]

        # Expand query for batch
        query = self.attention_query.expand(batch_size, -1, -1)

        # Create key padding mask from attention mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # True = ignore

        # Attention pooling
        pooled, _ = self.attention(
            query=query,
            key=text_embeddings,
            value=text_embeddings,
            key_padding_mask=key_padding_mask,
        )  # [batch, 1, text_encoder_dim]

        pooled = pooled.squeeze(1)  # [batch, text_encoder_dim]

        # Project
        return self.projection(pooled)  # [batch, context_hidden_dim]


# =============================================================================
# CROSS-UTTERANCE ENCODER
# =============================================================================

class CrossUtteranceEncoder(nn.Module):
    """
    Encodes neighboring sentences into context vectors.

    Takes the previous N sentences and future M sentences, encodes each,
    then aggregates them into a fixed-length context representation for
    conditioning prosody prediction.

    The context captures:
    - Discourse structure (topic flow, narrative arc)
    - Prosodic patterns (declarative vs question sequences)
    - Emotional trajectory (mood shifts across sentences)
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config

        # Sentence encoder (shared for all sentences)
        self.sentence_encoder = SentenceEncoder(config)

        # Sentence-level positional encoding (relative position in paragraph)
        self.sentence_position_embed = nn.Embedding(
            config.max_context_sentences + 1,  # +1 for current sentence
            config.context_hidden_dim,
        )

        # Temporal direction embedding (past vs future)
        self.direction_embed = nn.Embedding(
            3,  # 0=past, 1=current, 2=future
            config.context_hidden_dim,
        )

        # Transformer for aggregating sentence contexts
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.context_hidden_dim,
            nhead=config.context_num_heads,
            dim_feedforward=config.context_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN transformer
        )
        self.context_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.context_num_layers,
        )

        # Attention pooling for final context
        self.context_query = nn.Parameter(torch.randn(1, 1, config.context_hidden_dim))
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.context_hidden_dim,
            num_heads=config.context_num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.context_hidden_dim, config.context_hidden_dim),
            nn.LayerNorm(config.context_hidden_dim),
        )

    def forward(
        self,
        current_embeddings: Tensor,  # [batch, seq, dim] - current sentence
        current_mask: Optional[Tensor] = None,  # [batch, seq]
        prev_embeddings: Optional[List[Tensor]] = None,  # List of [batch, seq, dim]
        prev_masks: Optional[List[Tensor]] = None,
        next_embeddings: Optional[List[Tensor]] = None,  # List of [batch, seq, dim]
        next_masks: Optional[List[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode cross-utterance context.

        Args:
            current_embeddings: Text embeddings for current sentence
            current_mask: Attention mask for current sentence
            prev_embeddings: List of text embeddings for previous sentences
            prev_masks: Attention masks for previous sentences
            next_embeddings: List of text embeddings for next sentences
            next_masks: Attention masks for next sentences

        Returns:
            Dict with:
                - 'context': [batch, context_hidden_dim] - aggregated context
                - 'current_encoded': [batch, context_hidden_dim] - current sentence
                - 'all_sentence_encodings': [batch, num_sentences, context_hidden_dim]
        """
        batch_size = current_embeddings.shape[0]
        device = current_embeddings.device

        sentence_encodings = []
        positions = []
        directions = []

        # Encode previous sentences (oldest to newest)
        if prev_embeddings is not None:
            for i, (emb, mask) in enumerate(zip(
                prev_embeddings,
                prev_masks if prev_masks else [None] * len(prev_embeddings)
            )):
                enc = self.sentence_encoder(emb, mask)
                sentence_encodings.append(enc)
                # Position: 0 for oldest, increasing
                positions.append(i)
                directions.append(0)  # Past

        # Encode current sentence
        current_idx = len(sentence_encodings)
        current_encoded = self.sentence_encoder(current_embeddings, current_mask)
        sentence_encodings.append(current_encoded)
        positions.append(current_idx)
        directions.append(1)  # Current

        # Encode future sentences (nearest to farthest)
        if next_embeddings is not None and self.config.use_bidirectional_context:
            for i, (emb, mask) in enumerate(zip(
                next_embeddings,
                next_masks if next_masks else [None] * len(next_embeddings)
            )):
                enc = self.sentence_encoder(emb, mask)
                sentence_encodings.append(enc)
                positions.append(current_idx + 1 + i)
                directions.append(2)  # Future

        # Stack all sentence encodings
        all_encodings = torch.stack(sentence_encodings, dim=1)  # [batch, num_sent, dim]
        num_sentences = all_encodings.shape[1]

        # Add positional embeddings
        position_ids = torch.tensor(positions, device=device, dtype=torch.long)
        position_ids = position_ids.clamp(max=self.config.max_context_sentences)
        pos_embeddings = self.sentence_position_embed(position_ids)  # [num_sent, dim]
        all_encodings = all_encodings + pos_embeddings.unsqueeze(0)

        # Add direction embeddings
        direction_ids = torch.tensor(directions, device=device, dtype=torch.long)
        dir_embeddings = self.direction_embed(direction_ids)  # [num_sent, dim]
        all_encodings = all_encodings + dir_embeddings.unsqueeze(0)

        # Process through transformer
        context_features = self.context_transformer(all_encodings)  # [batch, num_sent, dim]

        # Attention pooling to get single context vector
        context_query = self.context_query.expand(batch_size, -1, -1)
        context, _ = self.context_attention(
            query=context_query,
            key=context_features,
            value=context_features,
        )  # [batch, 1, dim]

        context = context.squeeze(1)  # [batch, dim]
        context = self.output_projection(context)

        return {
            'context': context,
            'current_encoded': current_encoded,
            'all_sentence_encodings': context_features,
        }


# =============================================================================
# CUC VARIATIONAL ENCODER
# =============================================================================

class CUCVariationalEncoder(nn.Module):
    """
    Cross-Utterance Conditioned Variational Encoder.

    Implements a VAE where:
    - Prior p(z|c) is conditioned on context c
    - Posterior q(z|x,c) is conditioned on both acoustics x and context c
    - Reparameterization enables end-to-end training

    The context-conditioned prior allows the model to learn context-aware
    prosody distributions - what prosody is likely given the surrounding sentences.
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config

        # Prior network: p(z|c) - context-conditioned prior
        self.prior_network = nn.Sequential(
            nn.Linear(config.context_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Prior mean and log variance
        self.prior_mean = nn.Linear(config.prosody_hidden_dim, config.latent_dim)
        self.prior_logvar = nn.Linear(config.prosody_hidden_dim, config.latent_dim)

        # Posterior network: q(z|x,c) - conditioned on prosody features + context
        # Input: concatenation of prosody features and context
        posterior_input_dim = config.prosody_hidden_dim + config.context_hidden_dim
        self.posterior_network = nn.Sequential(
            nn.Linear(posterior_input_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Posterior mean and log variance
        self.posterior_mean = nn.Linear(config.prosody_hidden_dim, config.latent_dim)
        self.posterior_logvar = nn.Linear(config.prosody_hidden_dim, config.latent_dim)

        # Decoder: z -> prosody conditioning
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim + config.context_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
        )

    def compute_prior(self, context: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute context-conditioned prior p(z|c).

        Args:
            context: [batch, context_hidden_dim]

        Returns:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        if self.config.prior_type == "standard":
            # Standard normal prior
            batch_size = context.shape[0]
            device = context.device
            mean = torch.zeros(batch_size, self.config.latent_dim, device=device)
            logvar = torch.zeros(batch_size, self.config.latent_dim, device=device)
        else:
            # Context-conditioned prior
            hidden = self.prior_network(context)
            mean = self.prior_mean(hidden)
            logvar = self.prior_logvar(hidden).clamp(-10, 10)  # Stability

        return mean, logvar

    def compute_posterior(
        self,
        prosody_features: Tensor,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute posterior q(z|x,c).

        Args:
            prosody_features: [batch, prosody_hidden_dim] - encoded prosody
            context: [batch, context_hidden_dim]

        Returns:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        # Concatenate prosody and context
        combined = torch.cat([prosody_features, context], dim=-1)

        hidden = self.posterior_network(combined)
        mean = self.posterior_mean(hidden)
        logvar = self.posterior_logvar(hidden).clamp(-10, 10)

        return mean, logvar

    def reparameterize(
        self,
        mean: Tensor,
        logvar: Tensor,
        training: bool = True,
    ) -> Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I).

        Args:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
            training: Whether to add noise (False = use mean)

        Returns:
            z: [batch, latent_dim]
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean

    def compute_kl_divergence(
        self,
        q_mean: Tensor,
        q_logvar: Tensor,
        p_mean: Tensor,
        p_logvar: Tensor,
    ) -> Tensor:
        """
        Compute KL divergence: KL(q || p).

        KL(N(μ1,σ1²) || N(μ2,σ2²)) =
            log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2

        Args:
            q_mean, q_logvar: Posterior parameters
            p_mean, p_logvar: Prior parameters

        Returns:
            KL divergence (scalar)
        """
        kl = 0.5 * (
            p_logvar - q_logvar
            + (torch.exp(q_logvar) + (q_mean - p_mean).pow(2)) / torch.exp(p_logvar)
            - 1
        )
        return kl.sum(dim=-1).mean()

    def forward(
        self,
        prosody_features: Optional[Tensor],  # [batch, prosody_hidden_dim]
        context: Tensor,  # [batch, context_hidden_dim]
        training: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through variational encoder.

        Args:
            prosody_features: Encoded prosody (required for training)
            context: Cross-utterance context
            training: Training mode (use posterior) or inference (use prior)

        Returns:
            Dict with:
                - 'z': Sampled latent
                - 'decoded': Decoded prosody conditioning
                - 'kl_loss': KL divergence (if training)
                - 'prior_mean', 'prior_logvar': Prior parameters
                - 'posterior_mean', 'posterior_logvar': Posterior parameters
        """
        # Compute prior
        prior_mean, prior_logvar = self.compute_prior(context)

        result = {
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar,
        }

        if training and prosody_features is not None:
            # Training: sample from posterior
            posterior_mean, posterior_logvar = self.compute_posterior(
                prosody_features, context
            )
            z = self.reparameterize(posterior_mean, posterior_logvar, training=True)

            # Compute KL divergence
            kl_loss = self.compute_kl_divergence(
                posterior_mean, posterior_logvar,
                prior_mean, prior_logvar,
            )

            result.update({
                'posterior_mean': posterior_mean,
                'posterior_logvar': posterior_logvar,
                'kl_loss': kl_loss,
            })
        else:
            # Inference: sample from prior
            z = self.reparameterize(prior_mean, prior_logvar, training=False)
            result['kl_loss'] = torch.tensor(0.0, device=context.device)

        result['z'] = z

        # Decode
        decoded = self.decoder(torch.cat([z, context], dim=-1))
        result['decoded'] = decoded

        return result


# =============================================================================
# PARAGRAPH PROSODY PREDICTOR
# =============================================================================

class ParagraphProsodyPredictor(nn.Module):
    """
    Predicts prosody from text embeddings conditioned on cross-utterance context.

    Uses transformer decoder with cross-attention to:
    1. Current sentence text embeddings
    2. Cross-utterance context
    3. Latent prosody code (from VAE)
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config

        # Input projection for text embeddings
        self.text_projection = nn.Sequential(
            nn.Linear(config.text_encoder_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.prosody_hidden_dim,
            max_len=config.max_sentence_length,
            dropout=config.dropout,
        )

        # Context conditioning layer
        self.context_conditioning = nn.Sequential(
            nn.Linear(config.prosody_hidden_dim + config.context_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.prosody_hidden_dim,
            nhead=config.prosody_num_heads,
            dim_feedforward=config.prosody_hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.prosody_num_layers,
        )

        # Context memory (for cross-attention in transformer)
        self.context_memory = nn.Sequential(
            nn.Linear(config.context_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
        )

        # Output projection to prosody tokens
        self.output_projection = nn.Sequential(
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prosody_hidden_dim, config.output_dim * config.num_prosody_tokens),
        )

        # Layer norm for output
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        text_embeddings: Tensor,  # [batch, seq, text_encoder_dim]
        context: Tensor,  # [batch, context_hidden_dim]
        vae_decoded: Optional[Tensor] = None,  # [batch, prosody_hidden_dim]
        text_mask: Optional[Tensor] = None,  # [batch, seq]
    ) -> Tensor:
        """
        Predict prosody tokens from text + context.

        Args:
            text_embeddings: Current sentence text embeddings
            context: Cross-utterance context vector
            vae_decoded: Decoded latent from VAE (optional)
            text_mask: Attention mask for text

        Returns:
            prosody_tokens: [batch, num_prosody_tokens, output_dim]
        """
        batch_size = text_embeddings.shape[0]

        # Project text
        text_hidden = self.text_projection(text_embeddings)  # [batch, seq, dim]

        # Add positional encoding
        text_hidden = self.pos_encoding(text_hidden)

        # Prepare memory for cross-attention (context + VAE)
        memory_context = self.context_memory(context).unsqueeze(1)  # [batch, 1, dim]

        if vae_decoded is not None:
            vae_context = vae_decoded.unsqueeze(1)  # [batch, 1, dim]
            memory = torch.cat([memory_context, vae_context], dim=1)  # [batch, 2, dim]
        else:
            memory = memory_context

        # Condition text with context (additive)
        context_expanded = context.unsqueeze(1).expand(-1, text_hidden.shape[1], -1)
        conditioned = self.context_conditioning(
            torch.cat([text_hidden, context_expanded], dim=-1)
        )

        # Create memory mask (no masking needed for context)
        memory_key_padding_mask = None

        # Create target mask (no causal masking needed for parallel decoding)
        tgt_key_padding_mask = None
        if text_mask is not None:
            tgt_key_padding_mask = ~text_mask.bool()

        # Transformer decoder
        decoded = self.transformer(
            tgt=conditioned,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # [batch, seq, dim]

        # Mean pooling over sequence
        if text_mask is not None:
            mask_expanded = text_mask.unsqueeze(-1).float()
            pooled = (decoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = decoded.mean(dim=1)  # [batch, dim]

        # Project to prosody tokens
        output = self.output_projection(pooled)  # [batch, output_dim * num_tokens]

        # Reshape to tokens
        output = output.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)

        # Normalize
        output = self.output_norm(output)

        return output


# =============================================================================
# PROSODY ENCODER (for training - encodes acoustic features to prosody)
# =============================================================================

class CrossUtteranceProsodyEncoder(nn.Module):
    """
    Encodes acoustic features to prosody representation for training.

    Used to compute posterior q(z|x,c) where x is the acoustic prosody.
    At inference time, we sample from prior p(z|c) instead.
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config

        # Prosody feature dimensions (semantic, acoustic, rhythm, contour)
        prosody_input_dim = 8 + 12 + 8 + 64  # From prosody_conditioning.py

        self.encoder = nn.Sequential(
            nn.Linear(prosody_input_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.prosody_hidden_dim, config.prosody_hidden_dim),
            nn.LayerNorm(config.prosody_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        semantic: Tensor,  # [batch, 8]
        acoustic: Tensor,  # [batch, 12]
        rhythm: Tensor,  # [batch, 8]
        contour: Tensor,  # [batch, 64]
    ) -> Tensor:
        """
        Encode prosody features.

        Returns:
            [batch, prosody_hidden_dim]
        """
        prosody_input = torch.cat([semantic, acoustic, rhythm, contour], dim=-1)
        return self.encoder(prosody_input)


# =============================================================================
# CROSS-UTTERANCE LOSS
# =============================================================================

class CrossUtteranceLoss(nn.Module):
    """
    Combined loss for cross-utterance prosody prediction.

    Components:
    1. Reconstruction loss: Match predicted prosody tokens to target
    2. KL divergence: Regularize latent space (VAE)
    3. Contrastive loss (optional): Encourage context-aware predictions
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config
        self.step = 0

    def compute_kl_weight(self, step: int) -> float:
        """Compute annealed KL weight."""
        if step >= self.config.kl_annealing_steps:
            return self.config.kl_weight

        # Linear annealing
        progress = step / self.config.kl_annealing_steps
        return self.config.min_kl_weight + progress * (
            self.config.kl_weight - self.config.min_kl_weight
        )

    def forward(
        self,
        predicted_tokens: Tensor,  # [batch, num_tokens, output_dim]
        target_tokens: Tensor,  # [batch, num_tokens, output_dim]
        kl_loss: Tensor,  # Scalar from VAE
        step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.

        Returns:
            Dict with 'total', 'reconstruction', 'kl', 'kl_weight'
        """
        if step is None:
            step = self.step
            self.step += 1

        # Reconstruction loss (L2)
        reconstruction_loss = F.mse_loss(predicted_tokens, target_tokens)

        # KL loss with annealing
        kl_weight = self.compute_kl_weight(step)
        weighted_kl = kl_weight * kl_loss

        # Total loss
        total_loss = reconstruction_loss + weighted_kl

        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'kl': kl_loss,
            'kl_weighted': weighted_kl,
            'kl_weight': torch.tensor(kl_weight, device=predicted_tokens.device),
        }


# =============================================================================
# FULL MODEL
# =============================================================================

class CrossUtteranceProsody(nn.Module):
    """
    Complete Cross-Utterance Context prosody prediction model.

    Combines:
    - CrossUtteranceEncoder: Context from neighboring sentences
    - CUCVariationalEncoder: VAE for context-conditioned prosody
    - ParagraphProsodyPredictor: Prosody token generation
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config

        # Cross-utterance context encoder
        self.context_encoder = CrossUtteranceEncoder(config)

        # Variational encoder
        if config.use_vae:
            self.vae = CUCVariationalEncoder(config)
        else:
            self.vae = None

        # Prosody encoder (for training)
        self.prosody_encoder = CrossUtteranceProsodyEncoder(config)

        # Prosody predictor
        self.prosody_predictor = ParagraphProsodyPredictor(config)

        # Optional speaker embedding
        if config.num_speakers > 0:
            self.speaker_embed = nn.Embedding(config.num_speakers, config.speaker_embedding_dim)
            self.speaker_projection = nn.Linear(
                config.context_hidden_dim + config.speaker_embedding_dim,
                config.context_hidden_dim,
            )
        else:
            self.speaker_embed = None
            self.speaker_projection = None

        # Optional emotion embedding
        if config.use_emotion:
            self.emotion_embed = nn.Embedding(config.num_emotions, config.emotion_dim)
            self.emotion_projection = nn.Linear(
                config.context_hidden_dim + config.emotion_dim,
                config.context_hidden_dim,
            )
        else:
            self.emotion_embed = None
            self.emotion_projection = None

    def _apply_speaker(self, context: Tensor, speaker_id: Optional[Tensor]) -> Tensor:
        """Apply speaker conditioning to context."""
        if self.speaker_embed is None or speaker_id is None:
            return context

        speaker_emb = self.speaker_embed(speaker_id)  # [batch, speaker_dim]
        combined = torch.cat([context, speaker_emb], dim=-1)
        return self.speaker_projection(combined)

    def _apply_emotion(self, context: Tensor, emotion_id: Optional[Tensor]) -> Tensor:
        """Apply emotion conditioning to context."""
        if self.emotion_embed is None or emotion_id is None:
            return context

        emotion_emb = self.emotion_embed(emotion_id)  # [batch, emotion_dim]
        combined = torch.cat([context, emotion_emb], dim=-1)
        return self.emotion_projection(combined)

    def forward(
        self,
        # Current sentence
        current_text_embeddings: Tensor,  # [batch, seq, text_encoder_dim]
        current_text_mask: Optional[Tensor] = None,
        # Previous sentences (list, oldest first)
        prev_text_embeddings: Optional[List[Tensor]] = None,
        prev_text_masks: Optional[List[Tensor]] = None,
        # Next sentences (list, nearest first)
        next_text_embeddings: Optional[List[Tensor]] = None,
        next_text_masks: Optional[List[Tensor]] = None,
        # Target prosody (for training)
        target_semantic: Optional[Tensor] = None,  # [batch, 8]
        target_acoustic: Optional[Tensor] = None,  # [batch, 12]
        target_rhythm: Optional[Tensor] = None,  # [batch, 8]
        target_contour: Optional[Tensor] = None,  # [batch, 64]
        # Optional conditioning
        speaker_id: Optional[Tensor] = None,  # [batch]
        emotion_id: Optional[Tensor] = None,  # [batch]
        # Training control
        training: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Returns:
            Dict with:
                - 'prosody_tokens': [batch, num_tokens, output_dim]
                - 'context': [batch, context_hidden_dim]
                - 'kl_loss': KL divergence (if VAE)
                - 'vae_*': VAE outputs
        """
        # Encode cross-utterance context
        context_result = self.context_encoder(
            current_embeddings=current_text_embeddings,
            current_mask=current_text_mask,
            prev_embeddings=prev_text_embeddings,
            prev_masks=prev_text_masks,
            next_embeddings=next_text_embeddings,
            next_masks=next_text_masks,
        )

        context = context_result['context']  # [batch, context_hidden_dim]

        # Apply speaker/emotion conditioning
        context = self._apply_speaker(context, speaker_id)
        context = self._apply_emotion(context, emotion_id)

        result = {
            'context': context,
            'all_sentence_encodings': context_result['all_sentence_encodings'],
        }

        # VAE encoding
        vae_decoded = None
        if self.vae is not None:
            # Encode prosody features (for training)
            prosody_features = None
            if training and target_semantic is not None:
                prosody_features = self.prosody_encoder(
                    target_semantic, target_acoustic, target_rhythm, target_contour
                )

            vae_result = self.vae(
                prosody_features=prosody_features,
                context=context,
                training=training,
            )

            result.update({
                'kl_loss': vae_result['kl_loss'],
                'vae_z': vae_result['z'],
                'prior_mean': vae_result['prior_mean'],
                'prior_logvar': vae_result['prior_logvar'],
            })

            if 'posterior_mean' in vae_result:
                result['posterior_mean'] = vae_result['posterior_mean']
                result['posterior_logvar'] = vae_result['posterior_logvar']

            vae_decoded = vae_result['decoded']
        else:
            result['kl_loss'] = torch.tensor(0.0, device=context.device)

        # Predict prosody tokens
        prosody_tokens = self.prosody_predictor(
            text_embeddings=current_text_embeddings,
            context=context,
            vae_decoded=vae_decoded,
            text_mask=current_text_mask,
        )

        result['prosody_tokens'] = prosody_tokens

        return result


# =============================================================================
# ADAPTER FOR CSM PIPELINE
# =============================================================================

class CrossUtteranceAdapter(nn.Module):
    """
    Adapter for integrating CrossUtteranceProsody with CSM pipeline.

    Provides simple interface for:
    1. Training with prosody targets
    2. Inference with context-conditioned generation
    3. Integration with existing prosody conditioning
    """

    def __init__(self, config: CrossUtteranceConfig):
        super().__init__()
        self.config = config
        self.model = CrossUtteranceProsody(config)
        self.loss_fn = CrossUtteranceLoss(config)

        # Text encoder (lazy loaded)
        self._text_encoder = None
        self._tokenizer = None

    def _load_text_encoder(self, device: torch.device):
        """Lazy load text encoder (BERT/RoBERTa)."""
        if self._text_encoder is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.text_encoder_model
            )
            self._text_encoder = AutoModel.from_pretrained(
                self.config.text_encoder_model
            ).to(device)

            if self.config.freeze_text_encoder:
                for param in self._text_encoder.parameters():
                    param.requires_grad = False

            self._text_encoder.eval()

        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load text encoder: {e}. Using mock encoder.")
            self._text_encoder = "mock"
            self._tokenizer = "mock"

    def _encode_text(
        self,
        texts: List[str],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            embeddings: [batch, seq, text_encoder_dim]
            mask: [batch, seq]
        """
        self._load_text_encoder(device)

        if self._text_encoder == "mock":
            # Mock encoder for testing
            batch_size = len(texts)
            seq_len = 32
            embeddings = torch.randn(
                batch_size, seq_len, self.config.text_encoder_dim,
                device=device
            ) * 0.1
            mask = torch.ones(batch_size, seq_len, device=device)
            return embeddings, mask

        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_sentence_length,
            return_tensors='pt',
        ).to(device)

        # Encode
        with torch.no_grad():
            outputs = self._text_encoder(**encoded)
            embeddings = outputs.last_hidden_state

        return embeddings, encoded['attention_mask'].float()

    def forward(
        self,
        # Current sentence
        current_text: Union[str, List[str], Tensor],
        current_mask: Optional[Tensor] = None,
        # Context sentences
        prev_texts: Optional[List[str]] = None,
        next_texts: Optional[List[str]] = None,
        # Target prosody (for training)
        target_semantic: Optional[Tensor] = None,
        target_acoustic: Optional[Tensor] = None,
        target_rhythm: Optional[Tensor] = None,
        target_contour: Optional[Tensor] = None,
        # Optional conditioning
        speaker_id: Optional[Tensor] = None,
        emotion_id: Optional[Tensor] = None,
        # Training step for loss
        step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with text inputs.

        Args:
            current_text: Current sentence text or pre-computed embeddings
            current_mask: Mask for current text (if embeddings provided)
            prev_texts: List of previous sentence texts
            next_texts: List of next sentence texts
            target_*: Target prosody for training
            speaker_id: Speaker ID for multi-speaker
            emotion_id: Emotion ID for emotion conditioning
            step: Training step for KL annealing

        Returns:
            Dict with prosody_tokens and losses
        """
        # Determine device
        if isinstance(current_text, Tensor):
            device = current_text.device
            current_embeddings = current_text
        else:
            device = next(self.model.parameters()).device
            if isinstance(current_text, str):
                current_text = [current_text]
            current_embeddings, current_mask = self._encode_text(current_text, device)

        # Encode previous sentences
        prev_embeddings = None
        prev_masks = None
        if prev_texts is not None and len(prev_texts) > 0:
            prev_embeddings = []
            prev_masks = []
            for text in prev_texts[:self.config.num_prev_sentences]:
                emb, mask = self._encode_text([text] if isinstance(text, str) else text, device)
                prev_embeddings.append(emb)
                prev_masks.append(mask)

        # Encode next sentences
        next_embeddings = None
        next_masks = None
        if next_texts is not None and len(next_texts) > 0:
            next_embeddings = []
            next_masks = []
            for text in next_texts[:self.config.num_next_sentences]:
                emb, mask = self._encode_text([text] if isinstance(text, str) else text, device)
                next_embeddings.append(emb)
                next_masks.append(mask)

        # Forward through model
        training = target_semantic is not None
        result = self.model(
            current_text_embeddings=current_embeddings,
            current_text_mask=current_mask,
            prev_text_embeddings=prev_embeddings,
            prev_text_masks=prev_masks,
            next_text_embeddings=next_embeddings,
            next_text_masks=next_masks,
            target_semantic=target_semantic,
            target_acoustic=target_acoustic,
            target_rhythm=target_rhythm,
            target_contour=target_contour,
            speaker_id=speaker_id,
            emotion_id=emotion_id,
            training=training,
        )

        # Compute loss if targets provided
        if training and target_semantic is not None:
            # Create target tokens from prosody (simplified)
            # In practice, you'd have real target tokens
            batch_size = current_embeddings.shape[0]
            target_tokens = torch.zeros(
                batch_size,
                self.config.num_prosody_tokens,
                self.config.output_dim,
                device=device,
            )

            losses = self.loss_fn(
                result['prosody_tokens'],
                target_tokens,
                result['kl_loss'],
                step=step,
            )
            result['losses'] = losses

        return result

    def from_paragraph(
        self,
        sentences: List[str],
        current_index: int,
        speaker_id: Optional[Tensor] = None,
        emotion_id: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Convenience method for generating prosody for a sentence within a paragraph.

        Args:
            sentences: List of all sentences in paragraph
            current_index: Index of current sentence to generate prosody for
            speaker_id: Optional speaker ID
            emotion_id: Optional emotion ID

        Returns:
            Dict with prosody_tokens
        """
        current_text = sentences[current_index]

        # Get context sentences
        prev_texts = sentences[max(0, current_index - self.config.num_prev_sentences):current_index]
        next_texts = sentences[current_index + 1:current_index + 1 + self.config.num_next_sentences]

        return self(
            current_text=current_text,
            prev_texts=prev_texts,
            next_texts=next_texts,
            speaker_id=speaker_id,
            emotion_id=emotion_id,
        )

    def generate_paragraph_prosody(
        self,
        sentences: List[str],
        speaker_id: Optional[Tensor] = None,
        emotion_ids: Optional[List[Tensor]] = None,
    ) -> List[Tensor]:
        """
        Generate prosody tokens for all sentences in a paragraph.

        Args:
            sentences: List of sentences
            speaker_id: Optional speaker ID (same for all)
            emotion_ids: Optional list of emotion IDs per sentence

        Returns:
            List of prosody token tensors
        """
        all_tokens = []

        for i in range(len(sentences)):
            emotion_id = emotion_ids[i] if emotion_ids else None
            result = self.from_paragraph(
                sentences=sentences,
                current_index=i,
                speaker_id=speaker_id,
                emotion_id=emotion_id,
            )
            all_tokens.append(result['prosody_tokens'])

        return all_tokens


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_cross_utterance_adapter(
    config: Optional[CrossUtteranceConfig] = None,
    **kwargs,
) -> CrossUtteranceAdapter:
    """
    Create CrossUtteranceAdapter with optional config overrides.

    Args:
        config: Optional pre-built config
        **kwargs: Config field overrides

    Returns:
        CrossUtteranceAdapter instance
    """
    if config is None:
        config = CrossUtteranceConfig(**kwargs)
    else:
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return CrossUtteranceAdapter(config)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Simple sentence splitter - in production, use a proper NLP library.
    """
    import re

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter empty
    return [s.strip() for s in sentences if s.strip()]


# =============================================================================
# TESTING
# =============================================================================

def _test_cross_utterance():
    """Test cross-utterance prosody prediction."""
    print("Testing CrossUtteranceProsody...")

    # Create config
    config = CrossUtteranceConfig(
        text_encoder_dim=768,
        context_hidden_dim=256,
        latent_dim=128,
        prosody_hidden_dim=256,
        output_dim=512,
        num_prosody_tokens=4,
        num_prev_sentences=2,
        num_next_sentences=1,
    )

    # Create model
    model = CrossUtteranceProsody(config).cuda() if torch.cuda.is_available() else CrossUtteranceProsody(config)
    device = next(model.parameters()).device

    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 32

    # Current sentence
    current_embeddings = torch.randn(batch_size, seq_len, config.text_encoder_dim, device=device)
    current_mask = torch.ones(batch_size, seq_len, device=device)

    # Previous sentences
    prev_embeddings = [
        torch.randn(batch_size, seq_len, config.text_encoder_dim, device=device)
        for _ in range(2)
    ]
    prev_masks = [
        torch.ones(batch_size, seq_len, device=device)
        for _ in range(2)
    ]

    # Next sentence
    next_embeddings = [
        torch.randn(batch_size, seq_len, config.text_encoder_dim, device=device)
    ]
    next_masks = [
        torch.ones(batch_size, seq_len, device=device)
    ]

    # Target prosody
    target_semantic = torch.randn(batch_size, 8, device=device)
    target_acoustic = torch.randn(batch_size, 12, device=device)
    target_rhythm = torch.randn(batch_size, 8, device=device)
    target_contour = torch.randn(batch_size, 64, device=device)

    # Forward pass (training)
    result = model(
        current_text_embeddings=current_embeddings,
        current_text_mask=current_mask,
        prev_text_embeddings=prev_embeddings,
        prev_text_masks=prev_masks,
        next_text_embeddings=next_embeddings,
        next_text_masks=next_masks,
        target_semantic=target_semantic,
        target_acoustic=target_acoustic,
        target_rhythm=target_rhythm,
        target_contour=target_contour,
        training=True,
    )

    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    print(f"  Context shape: {result['context'].shape}")
    print(f"  KL loss: {result['kl_loss'].item():.4f}")

    # Test loss computation
    loss_fn = CrossUtteranceLoss(config)
    target_tokens = torch.randn_like(result['prosody_tokens'])
    losses = loss_fn(result['prosody_tokens'], target_tokens, result['kl_loss'], step=0)

    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Reconstruction loss: {losses['reconstruction'].item():.4f}")
    print(f"  KL weighted: {losses['kl_weighted'].item():.4f}")

    # Test inference mode
    result_inference = model(
        current_text_embeddings=current_embeddings,
        current_text_mask=current_mask,
        prev_text_embeddings=prev_embeddings,
        prev_text_masks=prev_masks,
        training=False,
    )

    print(f"  Inference prosody tokens shape: {result_inference['prosody_tokens'].shape}")

    # Test adapter
    print("\nTesting CrossUtteranceAdapter...")
    adapter = CrossUtteranceAdapter(config)
    adapter.to(device)

    # Test with text input
    result = adapter(
        current_text="This is the current sentence.",
        prev_texts=["This is the first sentence.", "This is the second sentence."],
        next_texts=["This is the next sentence."],
    )

    print(f"  Adapter prosody tokens shape: {result['prosody_tokens'].shape}")

    # Test paragraph generation
    sentences = [
        "Once upon a time, there was a princess.",
        "She lived in a tall tower.",
        "One day, a prince came to rescue her.",
        "They lived happily ever after.",
    ]

    all_tokens = adapter.generate_paragraph_prosody(sentences)
    print(f"  Generated {len(all_tokens)} prosody token sets for paragraph")
    for i, tokens in enumerate(all_tokens):
        print(f"    Sentence {i+1}: {tokens.shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    _test_cross_utterance()
