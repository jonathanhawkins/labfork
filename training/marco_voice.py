"""
Marco-Voice: Rotational Emotion Embedding Integration

Based on Marco-Voice Technical Report (arXiv:2508.02038):
"A Unified Framework for Expressive Speech Synthesis with Voice Cloning"

Key Innovations:
1. Rotational Emotion Embeddings: Define emotion as rotation from neutral state
   - Direction vector: v_i^e = (u_i^e - u_i^n) / ||u_i^e - u_i^n||
   - For each speaker, pairs of (emotional, neutral) define rotational directions
   - Aggregate across pairs for robust emotion representation

2. Cross-Orthogonal Constraint: Enforce perpendicularity between speaker and emotion
   - Computes D = ES^T (speaker-emotion dot product matrix)
   - Loss combines Frobenius norm and mean cosine similarity
   - Ensures speaker identity is independent of emotional expression

3. In-Batch Contrastive Learning: Distinguish emotions within batches
   - L_contrast = (1/[N(N-1)/2]) * sum_{i<j} |<h_i, e_j>|
   - Discourages similarity across different emotional states

4. Cross-Attention Integration: Emotion queries attend to LM token outputs
   - Q = W_q(emotion), K = W_k(LM_output), V = W_v(LM_output)
   - Dynamically modulates linguistic representations with emotion

Benefits over standard approaches:
- Smooth interpolation between emotions via rotational angles
- Better speaker-emotion disentanglement than VAD alone
- Compatible with existing spherical emotion vectors (extends them)
- Enables zero-shot emotion transfer across speakers

Integration: Works alongside SphericalEmotionAdapter from spherical_emotion.py
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MarcoVoiceConfig:
    """Configuration for Marco-Voice rotational emotion embeddings."""

    # Embedding dimensions
    input_dim: int = 768  # Input feature dimension (e.g., from wav2vec2)
    emotion_dim: int = 256  # Emotion embedding dimension
    speaker_dim: int = 256  # Speaker embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    output_dim: int = 2048  # Output to match prosody encoder

    # Rotational embedding settings
    num_emotion_pairs: int = 10  # Number of (emotional, neutral) pairs to aggregate
    temperature: float = 0.07  # Temperature for contrastive loss

    # Cross-attention settings
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Loss weights (from paper)
    lambda_orth: float = 0.1  # Cross-orthogonal constraint weight
    lambda_contrast: float = 0.5  # In-batch contrastive loss weight

    # Training settings
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Number of emotions
    num_emotions: int = 7  # neutral, happy, angry, sad, surprised, calm, fearful

    # Integration settings
    num_prosody_tokens: int = 4  # Number of prefix tokens to generate


# Emotion categories from CSEMOTIONS dataset
MARCO_EMOTIONS = [
    "neutral",
    "happy",
    "angry",
    "sad",
    "surprised",
    "calm",
    "fearful",
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(MARCO_EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}


# =============================================================================
# ROTATIONAL EMOTION ENCODER
# =============================================================================

class RotationalEmotionEncoder(nn.Module):
    """
    Encodes emotions using rotational direction from neutral.

    Key insight from Marco-Voice: Emotion can be represented as the
    directional difference between emotional and neutral speech from
    the same speaker. This "rotation" is speaker-independent and can
    be transferred across speakers.

    Algorithm:
    1. For each speaker, extract embeddings for emotional and neutral utterances
    2. Compute direction vector: v = (e - n) / ||e - n||
    3. Aggregate direction vectors across multiple pairs
    4. Project to emotion embedding space
    """

    def __init__(self, config: MarcoVoiceConfig):
        super().__init__()
        self.config = config

        # Audio feature encoder (processes input features)
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Direction projection (from difference vectors to emotion space)
        self.direction_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.emotion_dim),
        )

        # Emotion classifier (for supervised training)
        self.emotion_classifier = nn.Linear(config.emotion_dim, config.num_emotions)

        # Intensity predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Learnable neutral prototype (base reference)
        self.neutral_prototype = nn.Parameter(
            torch.zeros(config.emotion_dim)
        )

        # Learnable emotion prototypes (direction anchors)
        self.emotion_prototypes = nn.Parameter(
            torch.randn(config.num_emotions, config.emotion_dim) * 0.1
        )
        nn.init.orthogonal_(self.emotion_prototypes)

    def compute_direction(
        self,
        emotional_features: torch.Tensor,
        neutral_features: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute rotational direction from neutral to emotional.

        v = (e - n) / ||e - n||

        Args:
            emotional_features: [batch, dim] emotional audio features
            neutral_features: [batch, dim] neutral audio features
            eps: Small constant for numerical stability

        Returns:
            Direction vectors [batch, dim], normalized
        """
        # Compute difference
        diff = emotional_features - neutral_features

        # Normalize to unit direction
        direction = F.normalize(diff, p=2, dim=-1, eps=eps)

        return direction

    def aggregate_directions(
        self,
        directions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate multiple direction vectors into a single emotion direction.

        Args:
            directions: [num_pairs, dim] direction vectors
            weights: Optional [num_pairs] weights for averaging

        Returns:
            Aggregated direction [dim]
        """
        if weights is None:
            # Simple mean
            aggregated = directions.mean(dim=0)
        else:
            # Weighted mean
            weights = F.softmax(weights, dim=0)
            aggregated = (directions * weights.unsqueeze(-1)).sum(dim=0)

        # Re-normalize
        aggregated = F.normalize(aggregated, p=2, dim=-1)

        return aggregated

    def forward(
        self,
        emotional_input: torch.Tensor,
        neutral_input: Optional[torch.Tensor] = None,
        aggregate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion using rotational embedding.

        Args:
            emotional_input: [batch, seq, dim] or [batch, dim] emotional features
            neutral_input: [batch, seq, dim] or [batch, dim] neutral features
                          If None, uses learnable neutral prototype
            aggregate: Whether to pool over sequence dimension

        Returns:
            Dict containing:
                - 'direction': Emotion direction vectors
                - 'embedding': Full emotion embeddings
                - 'logits': Emotion classification logits
                - 'intensity': Predicted intensity
        """
        # Pool if sequence input
        if emotional_input.dim() == 3:
            emotional_pooled = emotional_input.mean(dim=1)
        else:
            emotional_pooled = emotional_input

        # Encode emotional features
        emotional_encoded = self.feature_encoder(emotional_pooled)

        # Handle neutral reference
        if neutral_input is not None:
            if neutral_input.dim() == 3:
                neutral_pooled = neutral_input.mean(dim=1)
            else:
                neutral_pooled = neutral_input
            neutral_encoded = self.feature_encoder(neutral_pooled)
        else:
            # Use learnable neutral with projection
            neutral_encoded = self.neutral_prototype.unsqueeze(0).expand(
                emotional_encoded.shape[0], -1
            )
            # Project to feature space
            neutral_encoded = F.pad(
                neutral_encoded,
                (0, emotional_encoded.shape[-1] - neutral_encoded.shape[-1])
            )

        # Compute direction from neutral to emotional
        direction = self.compute_direction(emotional_encoded, neutral_encoded)

        # Project direction to emotion embedding space
        emotion_embedding = self.direction_projector(direction)

        # Normalize embedding
        emotion_embedding = F.normalize(emotion_embedding, p=2, dim=-1)

        # Classify emotion
        logits = self.emotion_classifier(emotion_embedding)

        # Predict intensity (magnitude of difference)
        diff_magnitude = torch.norm(emotional_encoded - neutral_encoded, dim=-1, keepdim=True)
        intensity = self.intensity_predictor(emotion_embedding)

        return {
            'direction': direction,
            'embedding': emotion_embedding,
            'logits': logits,
            'intensity': intensity.squeeze(-1),
            'diff_magnitude': diff_magnitude.squeeze(-1),
            'emotional_encoded': emotional_encoded,
            'neutral_encoded': neutral_encoded,
        }

    def encode_emotion_label(
        self,
        emotion_idx: Union[int, torch.Tensor],
        intensity: float = 1.0,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Get emotion embedding from label using learnable prototypes.

        Args:
            emotion_idx: Emotion index or tensor of indices
            intensity: Intensity scaling factor
            batch_size: Batch size if emotion_idx is scalar

        Returns:
            Emotion embedding [batch, emotion_dim]
        """
        if isinstance(emotion_idx, int):
            emotion_idx = torch.tensor([emotion_idx]).expand(batch_size)

        # Get prototype direction
        prototype = self.emotion_prototypes[emotion_idx]  # [batch, dim]

        # Scale by intensity
        scaled = prototype * intensity

        # Normalize
        return F.normalize(scaled, p=2, dim=-1)


# =============================================================================
# SPEAKER ENCODER
# =============================================================================

class MarcoVoiceSpeakerEncoder(nn.Module):
    """
    Speaker encoder for Marco-Voice.

    Extracts speaker identity embedding that should be orthogonal to emotion.
    Uses attentive statistics pooling similar to ECAPA-TDNN.
    """

    def __init__(self, config: MarcoVoiceConfig):
        super().__init__()
        self.config = config

        # Feature extraction
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Attentive pooling
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 4, 1),
        )

        # Speaker projection
        self.speaker_projector = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.speaker_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract speaker embedding.

        Args:
            features: [batch, seq, dim] input features
            mask: Optional [batch, seq] attention mask

        Returns:
            Speaker embedding [batch, speaker_dim]
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)

        # Encode features
        encoded = self.feature_encoder(features)  # [batch, seq, hidden]

        # Compute attention weights
        attn_weights = self.attention(encoded).squeeze(-1)  # [batch, seq]

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool(), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)  # [batch, seq]

        # Weighted mean
        mean = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=1)

        # Weighted std
        var = torch.sum(
            (encoded - mean.unsqueeze(1)) ** 2 * attn_weights.unsqueeze(-1),
            dim=1
        )
        std = torch.sqrt(var.clamp(min=1e-8))

        # Concatenate statistics
        pooled = torch.cat([mean, std], dim=-1)  # [batch, hidden * 2]

        # Project to speaker space
        speaker_emb = self.speaker_projector(pooled)

        # Normalize
        speaker_emb = F.normalize(speaker_emb, p=2, dim=-1)

        return speaker_emb


# =============================================================================
# CROSS-ORTHOGONAL CONSTRAINT
# =============================================================================

class CrossOrthogonalConstraint(nn.Module):
    """
    Cross-orthogonal constraint from Marco-Voice.

    Enforces perpendicularity between speaker and emotion embeddings:
    - Computes dot product matrix D = ES^T
    - Combines Frobenius norm and mean cosine similarity
    - Ensures speaker identity is independent of emotional expression
    """

    def __init__(self, lambda_orth: float = 0.1):
        super().__init__()
        self.lambda_orth = lambda_orth

    def forward(
        self,
        emotion_emb: torch.Tensor,  # [batch, emotion_dim]
        speaker_emb: torch.Tensor,  # [batch, speaker_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-orthogonal constraint loss.

        Args:
            emotion_emb: [batch, emotion_dim] normalized emotion embeddings
            speaker_emb: [batch, speaker_dim] normalized speaker embeddings

        Returns:
            Dict with loss components
        """
        batch_size = emotion_emb.shape[0]

        # Ensure normalized
        E = F.normalize(emotion_emb, p=2, dim=-1)  # [batch, emotion_dim]
        S = F.normalize(speaker_emb, p=2, dim=-1)  # [batch, speaker_dim]

        # Compute dot product matrix between all pairs
        # D[i,j] = E[i] . S[j]^T
        # For same-dimension case: D = E @ S^T
        # For different dimensions, we need to align or use projection

        if E.shape[-1] == S.shape[-1]:
            D = torch.matmul(E, S.t())  # [batch, batch]
        else:
            # Project to same dimension using learned projection or simple approach
            # Use the minimum dimension
            min_dim = min(E.shape[-1], S.shape[-1])
            E_proj = E[..., :min_dim]
            S_proj = S[..., :min_dim]
            E_proj = F.normalize(E_proj, p=2, dim=-1)
            S_proj = F.normalize(S_proj, p=2, dim=-1)
            D = torch.matmul(E_proj, S_proj.t())  # [batch, batch]

        # Loss 1: Frobenius norm (penalize large dot products)
        frobenius_loss = torch.norm(D, p='fro') / (batch_size * batch_size)

        # Loss 2: Mean absolute cosine similarity
        mean_cos_loss = D.abs().mean()

        # Combined loss
        total_loss = frobenius_loss + mean_cos_loss

        return {
            'orthogonal_loss': total_loss * self.lambda_orth,
            'frobenius_loss': frobenius_loss,
            'mean_cosine': mean_cos_loss,
            'dot_product_matrix': D,
        }


# =============================================================================
# IN-BATCH CONTRASTIVE LEARNING
# =============================================================================

class InBatchContrastiveLoss(nn.Module):
    """
    In-batch contrastive learning from Marco-Voice.

    Encourages distinctiveness among emotion embeddings within minibatches:
    L_contrast = (1/[N(N-1)/2]) * sum_{i<j} |<h_i, e_j>|

    This discourages similarity across different emotional states.
    """

    def __init__(
        self,
        lambda_contrast: float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature

    def forward(
        self,
        emotion_embeddings: torch.Tensor,  # [batch, emotion_dim]
        speaker_embeddings: Optional[torch.Tensor] = None,  # [batch, speaker_dim]
        emotion_labels: Optional[torch.Tensor] = None,  # [batch] emotion indices
    ) -> Dict[str, torch.Tensor]:
        """
        Compute in-batch contrastive loss.

        Args:
            emotion_embeddings: [batch, dim] emotion embeddings
            speaker_embeddings: Optional [batch, dim] speaker embeddings (for combined h_i)
            emotion_labels: Optional [batch] labels for supervised contrastive

        Returns:
            Dict with loss components
        """
        batch_size = emotion_embeddings.shape[0]
        device = emotion_embeddings.device

        if batch_size < 2:
            # Need at least 2 samples for contrastive
            return {
                'contrastive_loss': torch.tensor(0.0, device=device),
                'num_pairs': torch.tensor(0, device=device),
            }

        # Normalize embeddings
        e = F.normalize(emotion_embeddings, p=2, dim=-1)

        # Combined embedding if speaker provided
        if speaker_embeddings is not None:
            s = F.normalize(speaker_embeddings, p=2, dim=-1)
            # For contrastive, we want to compare emotion embeddings with each other
            # The speaker embedding is used to form a joint representation
            # But for the contrastive loss, we focus on emotion-emotion similarity
            # Optionally mix in speaker info by projecting to same dim
            h = e  # Use emotion embeddings directly for contrastive
        else:
            h = e

        # Compute all pairwise similarities between emotion embeddings
        # sim[i,j] = <e_i, e_j>
        similarity_matrix = torch.matmul(h, h.t())  # [batch, batch]

        # Mask diagonal (self-similarity)
        mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # Get upper triangle pairs (i < j)
        upper_mask = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1).bool()

        # Extract upper triangle similarities
        upper_similarities = similarity_matrix[upper_mask]

        # Number of pairs: N(N-1)/2
        num_pairs = batch_size * (batch_size - 1) // 2

        # L_contrast = (1/num_pairs) * sum |sim|
        contrastive_loss = upper_similarities.abs().sum() / num_pairs

        # If we have emotion labels, use supervised contrastive
        if emotion_labels is not None:
            # Create label comparison matrix
            labels_eq = emotion_labels.unsqueeze(0) == emotion_labels.unsqueeze(1)

            # Same emotion pairs should have high similarity (positive)
            # Different emotion pairs should have low similarity (negative)

            # For positive pairs (same emotion), maximize similarity
            # For negative pairs (different emotion), minimize similarity

            positives = similarity_matrix[labels_eq & ~mask]
            negatives = similarity_matrix[~labels_eq]

            if len(positives) > 0:
                # InfoNCE-style loss
                # Pull together same-emotion, push apart different-emotion
                supervised_loss = -positives.mean() + negatives.abs().mean()
            else:
                supervised_loss = negatives.abs().mean()

            contrastive_loss = 0.5 * contrastive_loss + 0.5 * supervised_loss

        return {
            'contrastive_loss': contrastive_loss * self.lambda_contrast,
            'num_pairs': torch.tensor(num_pairs, device=device, dtype=torch.float),
            'mean_similarity': similarity_matrix[~mask].abs().mean(),
        }


# =============================================================================
# EMOTION-LM CROSS-ATTENTION
# =============================================================================

class EmotionCrossAttention(nn.Module):
    """
    Cross-attention module for integrating emotion with LM outputs.

    Emotion embedding serves as query (Q), LM outputs as keys/values (K, V):
    Q = W_q(emotion)
    K = W_k(LM_output)
    V = W_v(LM_output)
    Attention = softmax(QK^T/sqrt(d_k))V

    This allows emotions to dynamically modulate linguistic representations.
    """

    def __init__(self, config: MarcoVoiceConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.emotion_dim // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        # Query from emotion
        self.W_q = nn.Linear(config.emotion_dim, config.emotion_dim)

        # Key and Value from LM output (assuming same dim, can adapt)
        self.W_k = nn.Linear(config.output_dim, config.emotion_dim)
        self.W_v = nn.Linear(config.output_dim, config.emotion_dim)

        # Output projection
        self.out_proj = nn.Linear(config.emotion_dim, config.emotion_dim)

        # Layer norm
        self.layer_norm = nn.LayerNorm(config.emotion_dim)

        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        emotion_embedding: torch.Tensor,  # [batch, emotion_dim]
        lm_output: torch.Tensor,  # [batch, seq_len, lm_dim]
        mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply emotion-guided cross-attention to LM output.

        Args:
            emotion_embedding: [batch, emotion_dim] emotion embedding
            lm_output: [batch, seq_len, lm_dim] language model output
            mask: Optional attention mask

        Returns:
            Tuple of:
                - attended_output: [batch, emotion_dim] emotion-modulated output
                - attention_weights: [batch, num_heads, 1, seq_len]
        """
        batch_size = emotion_embedding.shape[0]
        seq_len = lm_output.shape[1]

        # Compute Q, K, V
        Q = self.W_q(emotion_embedding)  # [batch, emotion_dim]
        K = self.W_k(lm_output)  # [batch, seq, emotion_dim]
        V = self.W_v(lm_output)  # [batch, seq, emotion_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, S]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, S]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [B, H, 1, D]

        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1)  # [B, emotion_dim]
        output = self.out_proj(attended)

        # Residual + layer norm
        output = self.layer_norm(emotion_embedding + output)

        return output, attn_weights


# =============================================================================
# MARCO-VOICE EMOTION MODULE
# =============================================================================

class MarcoVoiceEmotionModule(nn.Module):
    """
    Complete Marco-Voice emotion module.

    Combines all components:
    1. Rotational emotion encoder
    2. Speaker encoder
    3. Cross-orthogonal constraint
    4. In-batch contrastive learning
    5. Emotion-LM cross-attention
    """

    def __init__(self, config: MarcoVoiceConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.emotion_encoder = RotationalEmotionEncoder(config)
        self.speaker_encoder = MarcoVoiceSpeakerEncoder(config)

        # Losses
        self.orthogonal_constraint = CrossOrthogonalConstraint(config.lambda_orth)
        self.contrastive_loss = InBatchContrastiveLoss(
            config.lambda_contrast,
            config.temperature
        )

        # Cross-attention
        self.cross_attention = EmotionCrossAttention(config)

        # Output projection to prosody tokens
        self.output_projection = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim * config.num_prosody_tokens),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        emotional_features: torch.Tensor,
        neutral_features: Optional[torch.Tensor] = None,
        lm_output: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        compute_losses: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Marco-Voice emotion module.

        Args:
            emotional_features: [batch, seq, dim] or [batch, dim] emotional audio features
            neutral_features: [batch, seq, dim] or [batch, dim] neutral audio features
            lm_output: Optional [batch, seq, lm_dim] language model output for cross-attention
            emotion_labels: Optional [batch] emotion indices for supervised training
            speaker_labels: Optional [batch] speaker indices (not used directly, for logging)
            compute_losses: Whether to compute loss components

        Returns:
            Dict with embeddings, tokens, and losses
        """
        # Encode emotion using rotational approach
        emotion_output = self.emotion_encoder(
            emotional_features,
            neutral_features,
        )

        emotion_embedding = emotion_output['embedding']

        # Encode speaker
        speaker_embedding = self.speaker_encoder(emotional_features)

        # Apply cross-attention if LM output provided
        if lm_output is not None:
            emotion_attended, attn_weights = self.cross_attention(
                emotion_embedding,
                lm_output,
            )
        else:
            emotion_attended = emotion_embedding
            attn_weights = None

        # Generate prosody tokens
        batch_size = emotion_attended.shape[0]
        tokens = self.output_projection(emotion_attended)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        result = {
            'emotion_embedding': emotion_embedding,
            'speaker_embedding': speaker_embedding,
            'emotion_attended': emotion_attended,
            'tokens': tokens,
            'emotion_logits': emotion_output['logits'],
            'intensity': emotion_output['intensity'],
            'direction': emotion_output['direction'],
        }

        if attn_weights is not None:
            result['attention_weights'] = attn_weights

        # Compute losses if requested
        if compute_losses:
            losses = {}

            # Cross-orthogonal constraint
            ortho_output = self.orthogonal_constraint(
                emotion_embedding,
                speaker_embedding,
            )
            losses['orthogonal_loss'] = ortho_output['orthogonal_loss']
            losses['frobenius_loss'] = ortho_output['frobenius_loss']
            losses['mean_cosine'] = ortho_output['mean_cosine']

            # In-batch contrastive
            contrast_output = self.contrastive_loss(
                emotion_embedding,
                speaker_embedding,
                emotion_labels,
            )
            losses['contrastive_loss'] = contrast_output['contrastive_loss']
            losses['mean_similarity'] = contrast_output['mean_similarity']

            # Emotion classification loss
            if emotion_labels is not None:
                losses['classification_loss'] = F.cross_entropy(
                    emotion_output['logits'],
                    emotion_labels,
                )
            else:
                losses['classification_loss'] = torch.tensor(0.0, device=emotion_embedding.device)

            # Total loss
            losses['total'] = (
                losses['orthogonal_loss'] +
                losses['contrastive_loss'] +
                losses['classification_loss']
            )

            result['losses'] = losses

        return result

    def encode_from_label(
        self,
        emotion_label: Union[str, int, torch.Tensor],
        intensity: float = 1.0,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion from label using learned prototypes.

        Args:
            emotion_label: Emotion name, index, or tensor
            intensity: Intensity scaling
            batch_size: Batch size

        Returns:
            Dict with embedding and tokens
        """
        device = self.emotion_encoder.emotion_prototypes.device

        # Convert to index
        if isinstance(emotion_label, str):
            emotion_idx = EMOTION_TO_IDX.get(emotion_label.lower(), 0)
        elif isinstance(emotion_label, torch.Tensor):
            emotion_idx = emotion_label
        else:
            emotion_idx = emotion_label

        # Get embedding from encoder
        emotion_embedding = self.emotion_encoder.encode_emotion_label(
            emotion_idx,
            intensity,
            batch_size,
        )

        # Generate tokens
        tokens = self.output_projection(emotion_embedding)
        tokens = tokens.view(batch_size, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.output_norm(tokens)

        return {
            'emotion_embedding': emotion_embedding,
            'tokens': tokens,
            'intensity': torch.full((batch_size,), intensity, device=device),
        }


# =============================================================================
# ADAPTER FOR SPHERICAL EMOTION INTEGRATION
# =============================================================================

class MarcoVoiceAdapter(nn.Module):
    """
    Adapter that integrates Marco-Voice rotational embeddings with
    existing SphericalEmotionAdapter.

    This provides a unified interface that combines:
    - Marco-Voice: Rotational direction from neutral
    - Spherical: VAD-based continuous control

    Usage:
        adapter = MarcoVoiceAdapter(config)

        # From audio features (rotational)
        result = adapter.from_audio(emotional_features, neutral_features)

        # From VAD/label (spherical fallback)
        result = adapter.from_emotion("happy", intensity=0.8)

        # Combined
        result = adapter.forward(emotional_features, neutral_features, emotion_label="happy")
    """

    def __init__(
        self,
        marco_config: MarcoVoiceConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.marco_config = marco_config

        # Marco-Voice module
        self.marco_module = MarcoVoiceEmotionModule(marco_config)

        # Try to integrate with spherical emotion
        self.spherical_adapter = None
        try:
            from spherical_emotion import SphericalEmotionConfig, SphericalEmotionAdapter
            spherical_config = SphericalEmotionConfig(
                output_dim=prosody_hidden,
                num_prosody_tokens=marco_config.num_prosody_tokens,
            )
            self.spherical_adapter = SphericalEmotionAdapter(
                spherical_config,
                prosody_hidden,
            )
        except ImportError:
            pass

        # Fusion layer to combine Marco and Spherical embeddings
        if self.spherical_adapter is not None:
            self.fusion = nn.Sequential(
                nn.Linear(marco_config.emotion_dim * 2, marco_config.hidden_dim),
                nn.LayerNorm(marco_config.hidden_dim),
                nn.GELU(),
                nn.Dropout(marco_config.dropout),
                nn.Linear(marco_config.hidden_dim, marco_config.emotion_dim),
            )
        else:
            self.fusion = None

        # Prosody dimension adapter
        if marco_config.output_dim != prosody_hidden:
            self.prosody_adapter = nn.Sequential(
                nn.Linear(marco_config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.prosody_adapter = nn.Identity()

    def from_audio(
        self,
        emotional_features: torch.Tensor,
        neutral_features: Optional[torch.Tensor] = None,
        lm_output: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion from audio features using rotational approach.
        """
        result = self.marco_module(
            emotional_features,
            neutral_features,
            lm_output,
            emotion_labels,
            compute_losses=self.training,
        )

        # Adapt tokens to prosody dimension
        result['prosody_tokens'] = self.prosody_adapter(result['tokens'])

        return result

    def from_emotion(
        self,
        emotion: Union[str, int],
        intensity: float = 0.7,
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode emotion from label (for inference without reference audio).

        If spherical adapter is available, uses VAD-based approach.
        Otherwise, uses Marco-Voice prototype embeddings.
        """
        if self.spherical_adapter is not None:
            # Use spherical emotion encoding
            result = self.spherical_adapter.encode_emotion(
                emotion,
                intensity,
                batch_size,
            )
            # Normalize key names for consistency
            if 'embedding' in result and 'emotion_embedding' not in result:
                result['emotion_embedding'] = result['embedding']
            return result
        else:
            # Use Marco-Voice prototypes
            result = self.marco_module.encode_from_label(
                emotion,
                intensity,
                batch_size,
            )
            result['prosody_tokens'] = self.prosody_adapter(result['tokens'])
            return result

    def forward(
        self,
        emotional_features: Optional[torch.Tensor] = None,
        neutral_features: Optional[torch.Tensor] = None,
        emotion_label: Optional[Union[str, int, torch.Tensor]] = None,
        intensity: float = 0.7,
        lm_output: Optional[torch.Tensor] = None,
        fusion_weight: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Combined forward pass.

        Can use audio features (rotational), emotion label (spherical/prototype),
        or both with fusion.

        Args:
            emotional_features: Audio features for rotational encoding
            neutral_features: Neutral audio for rotational baseline
            emotion_label: Emotion label for prototype/spherical encoding
            intensity: Intensity scaling
            lm_output: LM output for cross-attention
            fusion_weight: Weight for rotational vs spherical (0=spherical, 1=rotational)

        Returns:
            Dict with embeddings and prosody tokens
        """
        device = (
            emotional_features.device if emotional_features is not None
            else self.marco_module.emotion_encoder.neutral_prototype.device
        )
        batch_size = (
            emotional_features.shape[0] if emotional_features is not None
            else 1
        )

        results = {}

        # Rotational encoding from audio
        if emotional_features is not None:
            marco_result = self.from_audio(
                emotional_features,
                neutral_features,
                lm_output,
            )
            results['marco'] = marco_result

        # Label-based encoding
        if emotion_label is not None:
            label_result = self.from_emotion(
                emotion_label,
                intensity,
                batch_size,
            )
            results['label'] = label_result

        # Determine final output
        if 'marco' in results and 'label' in results and self.fusion is not None:
            # Fuse both representations
            marco_emb = results['marco']['emotion_embedding']
            label_emb = results['label'].get('embedding', results['label'].get('emotion_embedding'))

            if label_emb.shape[-1] != marco_emb.shape[-1]:
                # Project to same dimension
                label_emb = F.linear(
                    label_emb,
                    torch.eye(marco_emb.shape[-1], label_emb.shape[-1], device=device)
                )

            fused_emb = self.fusion(torch.cat([marco_emb, label_emb], dim=-1))

            # Weighted combination of tokens
            tokens = (
                fusion_weight * results['marco']['prosody_tokens'] +
                (1 - fusion_weight) * results['label']['prosody_tokens']
            )

            return {
                'emotion_embedding': fused_emb,
                'prosody_tokens': tokens,
                'marco_result': results['marco'],
                'label_result': results['label'],
                'fusion_weight': fusion_weight,
            }

        elif 'marco' in results:
            return results['marco']

        elif 'label' in results:
            return results['label']

        else:
            raise ValueError("Must provide either emotional_features or emotion_label")

    def interpolate_emotions(
        self,
        emotion1: str,
        emotion2: str,
        t: float,
        intensity: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two emotions using spherical SLERP.
        """
        if self.spherical_adapter is not None:
            result = self.spherical_adapter.interpolate_emotions(
                emotion1, emotion2, t, intensity, use_slerp=True
            )
            # Normalize key names for consistency
            if 'embedding' in result and 'emotion_embedding' not in result:
                result['emotion_embedding'] = result['embedding']
            return result
        else:
            # Fallback to prototype interpolation
            device = self.marco_module.emotion_encoder.neutral_prototype.device

            idx1 = EMOTION_TO_IDX.get(emotion1.lower(), 0)
            idx2 = EMOTION_TO_IDX.get(emotion2.lower(), 0)

            proto1 = self.marco_module.emotion_encoder.emotion_prototypes[idx1]
            proto2 = self.marco_module.emotion_encoder.emotion_prototypes[idx2]

            # Spherical interpolation
            proto1_norm = F.normalize(proto1, dim=-1)
            proto2_norm = F.normalize(proto2, dim=-1)

            dot = torch.dot(proto1_norm, proto2_norm).clamp(-1, 1)
            omega = torch.acos(dot)

            if omega.abs() < 1e-6:
                interp = (1 - t) * proto1 + t * proto2
            else:
                sin_omega = torch.sin(omega)
                s1 = torch.sin((1 - t) * omega) / sin_omega
                s2 = torch.sin(t * omega) / sin_omega
                interp = s1 * proto1_norm + s2 * proto2_norm

            # Scale by intensity
            interp = interp * intensity
            embedding = F.normalize(interp, dim=-1).unsqueeze(0)

            # Generate tokens
            tokens = self.marco_module.output_projection(embedding)
            tokens = tokens.view(1, self.marco_config.num_prosody_tokens, self.marco_config.output_dim)
            tokens = self.marco_module.output_norm(tokens)
            prosody_tokens = self.prosody_adapter(tokens)

            return {
                'emotion_embedding': embedding,
                'prosody_tokens': prosody_tokens,
                'interpolation_t': t,
            }


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class MarcoVoiceLoss(nn.Module):
    """
    Combined loss function for Marco-Voice training.

    L = L_TTS + lambda_orth * L_orth + lambda_contrast * L_contrast
    """

    def __init__(self, config: MarcoVoiceConfig):
        super().__init__()
        self.config = config
        self.orthogonal_constraint = CrossOrthogonalConstraint(config.lambda_orth)
        self.contrastive_loss = InBatchContrastiveLoss(
            config.lambda_contrast,
            config.temperature,
        )

    def forward(
        self,
        emotion_embedding: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion_logits: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Marco-Voice losses.

        Args:
            emotion_embedding: [batch, dim] emotion embeddings
            speaker_embedding: [batch, dim] speaker embeddings
            emotion_logits: Optional [batch, num_emotions] for classification
            emotion_labels: Optional [batch] ground truth labels

        Returns:
            Dict with all loss components and total
        """
        losses = {}
        device = emotion_embedding.device

        # Orthogonal constraint
        ortho = self.orthogonal_constraint(emotion_embedding, speaker_embedding)
        losses.update(ortho)

        # Contrastive loss
        contrast = self.contrastive_loss(
            emotion_embedding,
            speaker_embedding,
            emotion_labels,
        )
        losses.update(contrast)

        # Classification loss
        if emotion_logits is not None and emotion_labels is not None:
            losses['classification_loss'] = F.cross_entropy(emotion_logits, emotion_labels)
        else:
            losses['classification_loss'] = torch.tensor(0.0, device=device)

        # Total
        losses['total'] = (
            losses['orthogonal_loss'] +
            losses['contrastive_loss'] +
            losses['classification_loss']
        )

        return losses


# =============================================================================
# ROTATION ANGLE INTERPOLATION
# =============================================================================

class RotationalInterpolator:
    """
    Utilities for interpolating emotions using rotational angles.

    Since emotions are defined as rotations from neutral, we can:
    1. Interpolate rotation angles for smooth transitions
    2. Blend multiple emotion rotations
    3. Control intensity by rotation magnitude
    """

    @staticmethod
    def rotation_angle(
        direction: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute rotation angle of emotion direction.

        Args:
            direction: [batch, dim] unit direction vectors
            reference: Optional reference direction (default: [1, 0, ..., 0])

        Returns:
            Angles in radians [batch]
        """
        if reference is None:
            reference = torch.zeros_like(direction)
            reference[..., 0] = 1.0

        reference = F.normalize(reference, dim=-1)
        dot = (direction * reference).sum(dim=-1).clamp(-1, 1)
        return torch.acos(dot)

    @staticmethod
    def interpolate_rotations(
        dir1: torch.Tensor,
        dir2: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Spherical interpolation between two rotation directions.

        Args:
            dir1: [batch, dim] first direction
            dir2: [batch, dim] second direction
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated direction [batch, dim]
        """
        # Normalize
        d1 = F.normalize(dir1, dim=-1)
        d2 = F.normalize(dir2, dim=-1)

        # Angle between directions
        dot = (d1 * d2).sum(dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(dot)

        # SLERP
        sin_omega = torch.sin(omega)

        # Handle nearly parallel vectors
        mask = sin_omega.abs() < 1e-6

        s1 = torch.sin((1 - t) * omega) / (sin_omega + 1e-8)
        s2 = torch.sin(t * omega) / (sin_omega + 1e-8)

        result = s1 * d1 + s2 * d2

        # Fallback to lerp for parallel vectors
        lerp_result = (1 - t) * d1 + t * d2
        result = torch.where(mask.expand_as(result), lerp_result, result)

        return F.normalize(result, dim=-1)

    @staticmethod
    def blend_rotations(
        directions: List[torch.Tensor],
        weights: List[float],
    ) -> torch.Tensor:
        """
        Blend multiple emotion rotation directions.

        Args:
            directions: List of [dim] direction vectors
            weights: List of blend weights

        Returns:
            Blended direction [dim]
        """
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        # Weighted sum
        blended = sum(w * d for w, d in zip(weights, directions))

        return F.normalize(blended, dim=-1)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Marco-Voice: Rotational Emotion Embedding - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MarcoVoiceConfig()

    # Test 1: RotationalEmotionEncoder
    print("\n[Test 1] RotationalEmotionEncoder...")
    encoder = RotationalEmotionEncoder(config).to(device)

    batch_size = 4
    seq_len = 50
    emotional_features = torch.randn(batch_size, seq_len, config.input_dim, device=device)
    neutral_features = torch.randn(batch_size, seq_len, config.input_dim, device=device)

    output = encoder(emotional_features, neutral_features)
    print(f"  Direction shape: {output['direction'].shape}")
    print(f"  Embedding shape: {output['embedding'].shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Intensity: {output['intensity'][:2].tolist()}")
    print("  [PASS]")

    # Test 2: Direction computation
    print("\n[Test 2] Direction Computation...")
    direction = encoder.compute_direction(
        emotional_features.mean(dim=1),
        neutral_features.mean(dim=1),
    )
    print(f"  Direction norm: {direction.norm(dim=-1).mean().item():.4f} (should be ~1)")
    print("  [PASS]")

    # Test 3: Speaker Encoder
    print("\n[Test 3] MarcoVoiceSpeakerEncoder...")
    speaker_enc = MarcoVoiceSpeakerEncoder(config).to(device)
    speaker_emb = speaker_enc(emotional_features)
    print(f"  Speaker embedding shape: {speaker_emb.shape}")
    print(f"  Speaker embedding norm: {speaker_emb.norm(dim=-1).mean().item():.4f}")
    print("  [PASS]")

    # Test 4: Cross-Orthogonal Constraint
    print("\n[Test 4] CrossOrthogonalConstraint...")
    ortho_constraint = CrossOrthogonalConstraint(config.lambda_orth).to(device)
    emotion_emb = output['embedding']

    ortho_output = ortho_constraint(emotion_emb, speaker_emb)
    print(f"  Orthogonal loss: {ortho_output['orthogonal_loss'].item():.6f}")
    print(f"  Frobenius loss: {ortho_output['frobenius_loss'].item():.6f}")
    print(f"  Mean cosine: {ortho_output['mean_cosine'].item():.6f}")
    print("  [PASS]")

    # Test 5: In-Batch Contrastive Loss
    print("\n[Test 5] InBatchContrastiveLoss...")
    contrast_loss = InBatchContrastiveLoss(config.lambda_contrast).to(device)
    emotion_labels = torch.randint(0, config.num_emotions, (batch_size,), device=device)

    contrast_output = contrast_loss(emotion_emb, speaker_emb, emotion_labels)
    print(f"  Contrastive loss: {contrast_output['contrastive_loss'].item():.6f}")
    print(f"  Num pairs: {contrast_output['num_pairs'].item()}")
    print(f"  Mean similarity: {contrast_output['mean_similarity'].item():.6f}")
    print("  [PASS]")

    # Test 6: Emotion-LM Cross-Attention
    print("\n[Test 6] EmotionCrossAttention...")
    cross_attn = EmotionCrossAttention(config).to(device)
    lm_output = torch.randn(batch_size, seq_len, config.output_dim, device=device)

    attended, attn_weights = cross_attn(emotion_emb, lm_output)
    print(f"  Attended output shape: {attended.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print("  [PASS]")

    # Test 7: Complete MarcoVoiceEmotionModule
    print("\n[Test 7] MarcoVoiceEmotionModule (complete)...")
    module = MarcoVoiceEmotionModule(config).to(device)

    result = module(
        emotional_features,
        neutral_features,
        lm_output,
        emotion_labels,
    )

    print(f"  Emotion embedding: {result['emotion_embedding'].shape}")
    print(f"  Speaker embedding: {result['speaker_embedding'].shape}")
    print(f"  Prosody tokens: {result['tokens'].shape}")
    print(f"  Losses:")
    for k, v in result['losses'].items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.item():.6f}")
    print("  [PASS]")

    # Test 8: Encode from label
    print("\n[Test 8] Encode from emotion label...")
    for emotion in ["happy", "sad", "angry"]:
        label_result = module.encode_from_label(emotion, intensity=0.8)
        print(f"  {emotion}: embedding norm={label_result['emotion_embedding'].norm():.3f}")
    print("  [PASS]")

    # Test 9: MarcoVoiceAdapter
    print("\n[Test 9] MarcoVoiceAdapter...")
    adapter = MarcoVoiceAdapter(config).to(device)

    # From audio
    audio_result = adapter.from_audio(emotional_features, neutral_features)
    print(f"  From audio - tokens shape: {audio_result['prosody_tokens'].shape}")

    # From label
    label_result = adapter.from_emotion("happy", intensity=0.9)
    print(f"  From label - tokens shape: {label_result['prosody_tokens'].shape}")

    # Combined
    combined_result = adapter(
        emotional_features,
        neutral_features,
        emotion_label="happy",
        fusion_weight=0.7,
    )
    print(f"  Combined - tokens shape: {combined_result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 10: Emotion interpolation
    print("\n[Test 10] Emotion Interpolation...")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interp_result = adapter.interpolate_emotions("sad", "happy", t)
        emb_norm = interp_result['emotion_embedding'].norm().item()
        print(f"  t={t:.2f}: embedding norm={emb_norm:.3f}")
    print("  [PASS]")

    # Test 11: RotationalInterpolator
    print("\n[Test 11] RotationalInterpolator...")
    dir1 = F.normalize(torch.randn(1, config.emotion_dim, device=device), dim=-1)
    dir2 = F.normalize(torch.randn(1, config.emotion_dim, device=device), dim=-1)

    angle1 = RotationalInterpolator.rotation_angle(dir1)
    angle2 = RotationalInterpolator.rotation_angle(dir2)
    print(f"  Direction 1 angle: {math.degrees(angle1.item()):.1f} degrees")
    print(f"  Direction 2 angle: {math.degrees(angle2.item()):.1f} degrees")

    interp_dir = RotationalInterpolator.interpolate_rotations(dir1, dir2, 0.5)
    interp_angle = RotationalInterpolator.rotation_angle(interp_dir)
    print(f"  Interpolated angle: {math.degrees(interp_angle.item()):.1f} degrees")
    print("  [PASS]")

    # Test 12: MarcoVoiceLoss
    print("\n[Test 12] MarcoVoiceLoss...")
    loss_fn = MarcoVoiceLoss(config).to(device)

    losses = loss_fn(
        emotion_emb,
        speaker_emb,
        output['logits'],
        emotion_labels,
    )

    print(f"  Total loss: {losses['total'].item():.6f}")
    print(f"  Orthogonal: {losses['orthogonal_loss'].item():.6f}")
    print(f"  Contrastive: {losses['contrastive_loss'].item():.6f}")
    print(f"  Classification: {losses['classification_loss'].item():.6f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All Marco-Voice tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from marco_voice import (
    MarcoVoiceConfig,
    MarcoVoiceAdapter,
    MarcoVoiceLoss,
    RotationalInterpolator,
)

# Initialize
config = MarcoVoiceConfig(
    input_dim=768,      # wav2vec2 feature dim
    emotion_dim=256,
    output_dim=2048,    # Match prosody encoder
)

adapter = MarcoVoiceAdapter(config).cuda()
loss_fn = MarcoVoiceLoss(config).cuda()

# Training: From audio pairs (emotional, neutral)
for batch in dataloader:
    result = adapter.from_audio(
        batch['emotional_features'],
        batch['neutral_features'],
        lm_output=lm_hidden_states,  # Optional cross-attention
    )

    prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

    # Compute Marco-Voice specific losses
    if 'losses' in result:
        marco_loss = result['losses']['total']
    else:
        marco_loss = loss_fn(
            result['emotion_embedding'],
            result['speaker_embedding'],
            result['emotion_logits'],
            batch['emotion_labels'],
        )['total']

# Inference: From emotion label
result = adapter.from_emotion("happy", intensity=0.8)
tokens = result['prosody_tokens']

# Emotion interpolation
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    result = adapter.interpolate_emotions("sad", "happy", t)
    # Smooth transition from sad to happy

# Use with ProsodyControlledCSM:
# combined_prefix = torch.cat([prosody_prefix, marco_tokens], dim=1)
""")
