"""
ParaMETA: Disentangled Paralinguistic Representation Learning

Based on "ParaMETA: Disentangled Paralinguistic Learning"
(arXiv:2601.12289, January 2025)

Key Techniques:
1. META Embedding Space: Speech samples with shared labels grouped together
2. Task-Specific Subspaces: Independent projections for emotion, gender, age, accent
   - Optimized independently to reduce inter-task interference
   - Same-class samples cluster regardless of other attributes
3. Prototype-Based Text-Speech Alignment: Inspired by LLaVA for efficient cross-modal
4. Direct Projection: No expensive joint embedding computation

Benefits:
- Single model handles multiple paralinguistic tasks simultaneously
- Reduced negative transfer between tasks via subspace isolation
- Supports both speech-based and text-based style prompts
- More efficient than ParaStyleTTS for multi-attribute control
- Unified framework for learning and controlling speaking styles

Architecture:
- Shared encoder produces META embeddings
- Multiple projection heads for task-specific spaces
- Prototype-based text-speech alignment
- Direct projection without expensive joint embedding

Integration:
- ParaMETAEncoder with shared backbone
- TaskSpecificProjector for each attribute
- PrototypeAligner for text-speech correspondence
- ParaMETAAdapter for CSM integration

Usage:
    from parameta import (
        ParaMETAConfig,
        ParaMETA,
        ParaMETAAdapter,
        PARALINGUISTIC_TASKS,
    )

    # Initialize
    config = ParaMETAConfig()
    model = ParaMETA(config).cuda()

    # Training: Multi-task learning
    losses = model(
        audio_features,
        emotion_labels=emotion_labels,
        gender_labels=gender_labels,
        age_labels=age_labels,
    )

    # Inference: Get task-specific embeddings
    embeddings = model.get_embeddings(audio_features)
    emotion_emb = embeddings['emotion']
    gender_emb = embeddings['gender']

    # CSM integration
    adapter = ParaMETAAdapter(config)
    prosody_tokens = adapter(audio_features)['prosody_tokens']
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
class ParaMETAConfig:
    """Configuration for ParaMETA disentangled paralinguistic learning."""

    # Input dimensions
    input_dim: int = 768  # Input feature dimension (wav2vec2/HuBERT)
    mel_dim: int = 80  # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 320  # ~20ms at 16kHz

    # META embedding space
    meta_dim: int = 512  # Shared META embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    num_heads: int = 8  # Transformer attention heads
    num_layers: int = 6  # Transformer layers for shared encoder
    dropout: float = 0.1

    # Task-specific subspaces
    task_subspace_dim: int = 128  # Dimension per task subspace
    num_projection_layers: int = 2  # Layers in task projectors

    # Paralinguistic tasks
    # Each task has: (num_classes, whether it's discrete classification)
    tasks: Dict[str, Tuple[int, bool]] = field(default_factory=lambda: {
        "emotion": (8, True),  # 8 emotions (categorical)
        "gender": (2, True),  # male/female (categorical)
        "age": (5, True),  # age groups: child, teen, young_adult, adult, senior
        "accent": (10, True),  # 10 common accents
        "speaking_rate": (3, True),  # slow, normal, fast
        "energy": (3, True),  # low, medium, high
    })

    # Prototype-based alignment
    num_prototypes: int = 128  # Number of learned prototypes
    prototype_dim: int = 256  # Prototype embedding dimension
    alignment_temperature: float = 0.07  # Temperature for contrastive loss

    # Text encoder for text-speech alignment
    text_encoder_dim: int = 768  # Text encoder output dimension
    use_text_alignment: bool = True

    # Loss weights
    classification_weight: float = 1.0  # Weight for classification losses
    contrastive_weight: float = 0.5  # Weight for contrastive clustering
    prototype_weight: float = 0.3  # Weight for prototype alignment
    orthogonality_weight: float = 0.1  # Weight for subspace orthogonality
    intra_class_weight: float = 0.2  # Weight for intra-class compactness

    # Training settings
    use_layer_norm: bool = True
    gradient_accumulation: int = 1

    # Output for CSM integration
    output_dim: int = 2048  # Match CSM hidden dimension
    num_prosody_tokens: int = 4  # Number of prefix tokens


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

# Predefined labels for each task
PARALINGUISTIC_TASKS = {
    "emotion": [
        "neutral", "happy", "sad", "angry",
        "fearful", "surprised", "disgusted", "calm"
    ],
    "gender": ["male", "female"],
    "age": ["child", "teen", "young_adult", "adult", "senior"],
    "accent": [
        "american", "british", "australian", "indian",
        "chinese", "spanish", "french", "german",
        "japanese", "korean"
    ],
    "speaking_rate": ["slow", "normal", "fast"],
    "energy": ["low", "medium", "high"],
}

# Task name to index mapping
TASK_TO_IDX = {task: i for i, task in enumerate(PARALINGUISTIC_TASKS.keys())}
IDX_TO_TASK = {i: task for task, i in TASK_TO_IDX.items()}


def get_task_labels(task: str) -> List[str]:
    """Get label names for a task."""
    return PARALINGUISTIC_TASKS.get(task, [])


def get_num_classes(task: str) -> int:
    """Get number of classes for a task."""
    return len(PARALINGUISTIC_TASKS.get(task, []))


def label_to_idx(task: str, label: str) -> int:
    """Convert label string to index."""
    labels = get_task_labels(task)
    if label.lower() in [l.lower() for l in labels]:
        return [l.lower() for l in labels].index(label.lower())
    return 0


def idx_to_label(task: str, idx: int) -> str:
    """Convert index to label string."""
    labels = get_task_labels(task)
    if 0 <= idx < len(labels):
        return labels[idx]
    return "unknown"


# =============================================================================
# SHARED ENCODER (META EMBEDDING SPACE)
# =============================================================================

class ParaMETAEncoder(nn.Module):
    """
    Shared encoder that produces META embeddings.

    The META embedding space groups speech samples with shared labels
    together regardless of other attributes. This forms the foundation
    for task-specific projections.

    Architecture:
    - Input projection with LayerNorm
    - Positional encoding
    - Transformer encoder layers
    - Global pooling (attention-weighted)
    - Output to META dimension
    """

    def __init__(self, config: ParaMETAConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # [CLS] token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
        )

        # Output projection to META embedding
        self.meta_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.meta_dim),
            nn.LayerNorm(config.meta_dim) if config.use_layer_norm else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract META embedding from audio features.

        Args:
            x: [batch, seq, input_dim] audio features
            mask: [batch, seq] attention mask (True = valid)
            return_sequence: Return sequence-level features

        Returns:
            META embedding [batch, meta_dim] or dict with sequence
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        if seq_len <= self.pos_encoding.shape[1]:
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Interpolate positional encoding if needed
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Update mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer
        # PyTorch expects mask where True = ignore
        attn_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract CLS token embedding
        cls_embedding = x[:, 0, :]
        sequence_features = x[:, 1:, :]

        # Attention pooling
        if mask is not None:
            seq_mask = mask[:, 1:]
        else:
            seq_mask = None

        attn_weights = self.attention_pool(sequence_features).squeeze(-1)
        if seq_mask is not None:
            attn_weights = attn_weights.masked_fill(~seq_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled_embedding = (sequence_features * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Combine CLS and attention-pooled
        combined = cls_embedding + pooled_embedding

        # Project to META embedding
        meta_emb = self.meta_proj(combined)

        if return_sequence:
            sequence_meta = self.meta_proj(sequence_features)
            return {
                'meta_embedding': meta_emb,
                'sequence': sequence_meta,
                'attention_weights': attn_weights,
                'cls_features': cls_embedding,
            }

        return meta_emb


# =============================================================================
# TASK-SPECIFIC PROJECTOR
# =============================================================================

class TaskSpecificProjector(nn.Module):
    """
    Projects META embedding to task-specific subspace.

    Each task has its own projector to create independent subspaces.
    This reduces inter-task interference and enables disentangled control.

    The projector includes:
    - MLP layers for non-linear projection
    - Optional classification head for supervised learning
    - Subspace normalization
    """

    def __init__(
        self,
        config: ParaMETAConfig,
        task_name: str,
        num_classes: int,
        is_discrete: bool = True,
    ):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.num_classes = num_classes
        self.is_discrete = is_discrete

        # MLP projector
        layers = []
        in_dim = config.meta_dim
        for i in range(config.num_projection_layers):
            out_dim = config.task_subspace_dim if i == config.num_projection_layers - 1 else config.hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if config.use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = out_dim

        self.projector = nn.Sequential(*layers)

        # Classification head (for discrete tasks)
        if is_discrete and num_classes > 0:
            self.classifier = nn.Linear(config.task_subspace_dim, num_classes)
        else:
            self.classifier = None

        # Learnable class embeddings (for contrastive learning)
        if is_discrete and num_classes > 0:
            self.class_embeddings = nn.Embedding(num_classes, config.task_subspace_dim)
        else:
            self.class_embeddings = None

    def forward(
        self,
        meta_embedding: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Project META embedding to task-specific subspace.

        Args:
            meta_embedding: [batch, meta_dim] META embeddings
            labels: [batch] ground truth labels (for training)

        Returns:
            Dict with subspace embedding, logits, and losses
        """
        # Project to task subspace
        subspace_emb = self.projector(meta_embedding)

        result = {
            'embedding': subspace_emb,
            'task_name': self.task_name,
        }

        # Classification (if discrete task)
        if self.classifier is not None:
            logits = self.classifier(subspace_emb)
            result['logits'] = logits

            if labels is not None:
                # Classification loss
                cls_loss = F.cross_entropy(logits, labels)
                result['classification_loss'] = cls_loss

                # Accuracy
                preds = logits.argmax(dim=-1)
                accuracy = (preds == labels).float().mean()
                result['accuracy'] = accuracy

                # Contrastive intra-class loss
                if self.class_embeddings is not None:
                    class_embs = self.class_embeddings(labels)  # [batch, subspace_dim]
                    # Pull samples toward their class embedding
                    intra_class_loss = 1 - F.cosine_similarity(subspace_emb, class_embs).mean()
                    result['intra_class_loss'] = intra_class_loss

        return result


# =============================================================================
# PROTOTYPE ALIGNER (TEXT-SPEECH ALIGNMENT)
# =============================================================================

class PrototypeAligner(nn.Module):
    """
    Prototype-based text-speech alignment.

    Inspired by LLaVA, uses learnable prototypes to bridge
    text and speech modalities without expensive joint embedding.

    Key idea:
    - Learn a set of prototypes that capture paralinguistic patterns
    - Align both text and speech to these prototypes
    - Enables text-based style prompting at inference

    Benefits:
    - Efficient: O(N*K) instead of O(N^2) for alignment
    - Interpretable: Prototypes capture distinct style patterns
    - Flexible: Can be used with any text encoder
    """

    def __init__(self, config: ParaMETAConfig):
        super().__init__()
        self.config = config

        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.zeros(config.num_prototypes, config.prototype_dim)
        )
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        # Speech to prototype projection
        self.speech_proj = nn.Sequential(
            nn.Linear(config.meta_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.prototype_dim),
        )

        # Text to prototype projection
        if config.use_text_alignment:
            self.text_proj = nn.Sequential(
                nn.Linear(config.text_encoder_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.prototype_dim),
            )
        else:
            self.text_proj = None

        # Temperature for contrastive loss
        self.temperature = config.alignment_temperature

        # Prototype to task mapping
        # Each prototype is associated with task-specific information
        self.prototype_task_heads = nn.ModuleDict()
        for task_name, (num_classes, is_discrete) in config.tasks.items():
            if is_discrete:
                self.prototype_task_heads[task_name] = nn.Linear(
                    config.prototype_dim, num_classes
                )

    def compute_prototype_similarity(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between embeddings and prototypes.

        Args:
            embeddings: [batch, prototype_dim] projected embeddings

        Returns:
            Similarity scores [batch, num_prototypes]
        """
        # L2 normalize
        emb_norm = F.normalize(embeddings, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        # Cosine similarity
        similarity = torch.matmul(emb_norm, proto_norm.T)

        return similarity / self.temperature

    def forward(
        self,
        speech_embedding: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Align speech (and optionally text) to prototypes.

        Args:
            speech_embedding: [batch, meta_dim] speech META embedding
            text_embedding: [batch, text_dim] text embedding (optional)

        Returns:
            Dict with prototype assignments and alignments
        """
        batch_size = speech_embedding.shape[0]
        device = speech_embedding.device

        # Project speech to prototype space
        speech_proj = self.speech_proj(speech_embedding)
        speech_similarity = self.compute_prototype_similarity(speech_proj)

        result = {
            'speech_projection': speech_proj,
            'speech_similarity': speech_similarity,
            'prototype_assignment': F.softmax(speech_similarity, dim=-1),
        }

        # Get weighted prototype representation
        proto_weights = F.softmax(speech_similarity, dim=-1)  # [batch, num_prototypes]
        weighted_prototypes = torch.matmul(proto_weights, self.prototypes)  # [batch, prototype_dim]
        result['weighted_prototype'] = weighted_prototypes

        # Text alignment (if available)
        if text_embedding is not None and self.text_proj is not None:
            text_proj = self.text_proj(text_embedding)
            text_similarity = self.compute_prototype_similarity(text_proj)

            result['text_projection'] = text_proj
            result['text_similarity'] = text_similarity

            # Cross-modal alignment loss
            # Encourage speech and text to have similar prototype assignments
            speech_dist = F.softmax(speech_similarity, dim=-1)
            text_dist = F.softmax(text_similarity, dim=-1)

            # KL divergence (symmetrized)
            kl_st = F.kl_div(
                speech_dist.log(),
                text_dist,
                reduction='batchmean'
            )
            kl_ts = F.kl_div(
                text_dist.log(),
                speech_dist,
                reduction='batchmean'
            )
            alignment_loss = 0.5 * (kl_st + kl_ts)
            result['alignment_loss'] = alignment_loss

        # Prototype diversity regularization
        # Encourage prototypes to be different from each other
        proto_norm = F.normalize(self.prototypes, dim=-1)
        proto_similarity = torch.matmul(proto_norm, proto_norm.T)
        # Minimize off-diagonal elements
        eye = torch.eye(self.config.num_prototypes, device=device)
        diversity_loss = (proto_similarity * (1 - eye)).pow(2).mean()
        result['diversity_loss'] = diversity_loss

        # Task predictions from prototypes
        for task_name, head in self.prototype_task_heads.items():
            task_logits = head(weighted_prototypes)
            result[f'{task_name}_proto_logits'] = task_logits

        return result

    def get_text_style_embedding(
        self,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get style embedding from text description.

        This enables text-based style prompting at inference.

        Args:
            text_embedding: [batch, text_dim] text encoder output

        Returns:
            Style embedding [batch, prototype_dim]
        """
        if self.text_proj is None:
            raise ValueError("Text projection not available")

        text_proj = self.text_proj(text_embedding)
        similarity = self.compute_prototype_similarity(text_proj)
        weights = F.softmax(similarity, dim=-1)
        style_emb = torch.matmul(weights, self.prototypes)

        return style_emb


# =============================================================================
# SUBSPACE ORTHOGONALITY LOSS
# =============================================================================

class SubspaceOrthogonalityLoss(nn.Module):
    """
    Encourages task-specific subspaces to be orthogonal.

    This reduces inter-task interference and ensures each task
    captures unique information.

    Loss = sum_{i != j} |<v_i, v_j>|^2

    where v_i is the mean embedding for task i.
    """

    def __init__(self, config: ParaMETAConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        task_embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute orthogonality loss between task subspaces.

        Args:
            task_embeddings: Dict mapping task name to [batch, subspace_dim] embeddings

        Returns:
            Orthogonality loss
        """
        if len(task_embeddings) < 2:
            return torch.tensor(0.0, device=next(iter(task_embeddings.values())).device)

        # Get mean embedding for each task
        task_means = []
        for task_name, emb in task_embeddings.items():
            task_means.append(F.normalize(emb.mean(dim=0), dim=-1))

        # Stack: [num_tasks, subspace_dim]
        task_means = torch.stack(task_means, dim=0)

        # Compute pairwise similarities
        similarity = torch.matmul(task_means, task_means.T)

        # Minimize off-diagonal elements
        num_tasks = len(task_embeddings)
        eye = torch.eye(num_tasks, device=similarity.device)
        off_diag = similarity * (1 - eye)

        # L2 loss on off-diagonal
        loss = off_diag.pow(2).sum() / (num_tasks * (num_tasks - 1))

        return loss


# =============================================================================
# INTRA-CLASS COMPACTNESS LOSS
# =============================================================================

class IntraClassCompactnessLoss(nn.Module):
    """
    Encourages samples of the same class to cluster together.

    For each task, samples with the same label should have
    similar embeddings in the task-specific subspace.

    Loss = E[1 - cos_sim(x_i, centroid(class_i))]
    """

    def __init__(self, config: ParaMETAConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """
        Compute intra-class compactness loss.

        Args:
            embeddings: [batch, subspace_dim] task embeddings
            labels: [batch] class labels
            num_classes: Number of classes

        Returns:
            Compactness loss
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        emb_norm = F.normalize(embeddings, dim=-1)

        # Compute class centroids
        centroids = torch.zeros(num_classes, embeddings.shape[1], device=device)
        counts = torch.zeros(num_classes, device=device)

        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = emb_norm[mask].mean(dim=0)
                counts[c] = mask.sum()

        # Normalize centroids
        centroids = F.normalize(centroids, dim=-1)

        # Compute loss: distance from each sample to its centroid
        sample_centroids = centroids[labels]  # [batch, subspace_dim]
        similarity = (emb_norm * sample_centroids).sum(dim=-1)

        # Loss = 1 - cosine similarity
        loss = (1 - similarity).mean()

        return loss


# =============================================================================
# COMPLETE ParaMETA MODULE
# =============================================================================

class ParaMETA(nn.Module):
    """
    Complete ParaMETA module for disentangled paralinguistic learning.

    Combines:
    1. ParaMETAEncoder: Shared encoder for META embeddings
    2. TaskSpecificProjectors: Independent subspaces per task
    3. PrototypeAligner: Text-speech alignment via prototypes
    4. Orthogonality/Compactness losses: Disentanglement regularization

    Training:
    - Multi-task learning with task-specific classification losses
    - Contrastive learning for intra-class clustering
    - Subspace orthogonality for inter-task independence
    - Prototype alignment for text-based control

    Inference:
    - Extract task-specific embeddings from speech
    - Use text prompts for style specification
    - Generate prosody tokens for CSM integration
    """

    def __init__(self, config: ParaMETAConfig):
        super().__init__()
        self.config = config

        # Shared META encoder
        self.encoder = ParaMETAEncoder(config)

        # Task-specific projectors
        self.task_projectors = nn.ModuleDict()
        for task_name, (num_classes, is_discrete) in config.tasks.items():
            self.task_projectors[task_name] = TaskSpecificProjector(
                config, task_name, num_classes, is_discrete
            )

        # Prototype aligner
        self.prototype_aligner = PrototypeAligner(config)

        # Losses
        self.orthogonality_loss = SubspaceOrthogonalityLoss(config)
        self.compactness_loss = IntraClassCompactnessLoss(config)

        # Combined embedding fusion
        total_task_dim = len(config.tasks) * config.task_subspace_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_task_dim + config.prototype_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.meta_dim),
        )

        # Task names for iteration
        self.task_names = list(config.tasks.keys())

    def forward(
        self,
        audio_features: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **task_labels,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task learning.

        Args:
            audio_features: [batch, seq, input_dim] audio features
            text_embedding: [batch, text_dim] optional text embedding
            mask: [batch, seq] attention mask
            **task_labels: Task-specific labels (e.g., emotion_labels, gender_labels)

        Returns:
            Dict with losses and embeddings
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device

        # Get META embedding
        meta_emb = self.encoder(audio_features, mask)

        result = {
            'meta_embedding': meta_emb,
        }

        # Process each task
        task_embeddings = {}
        total_cls_loss = torch.tensor(0.0, device=device)
        total_intra_loss = torch.tensor(0.0, device=device)
        num_tasks_with_labels = 0

        for task_name in self.task_names:
            projector = self.task_projectors[task_name]
            label_key = f"{task_name}_labels"
            labels = task_labels.get(label_key)

            # Project to task subspace
            task_result = projector(meta_emb, labels)
            task_embeddings[task_name] = task_result['embedding']

            result[f'{task_name}_embedding'] = task_result['embedding']

            if 'logits' in task_result:
                result[f'{task_name}_logits'] = task_result['logits']

            if labels is not None:
                num_tasks_with_labels += 1

                if 'classification_loss' in task_result:
                    total_cls_loss = total_cls_loss + task_result['classification_loss']
                    result[f'{task_name}_cls_loss'] = task_result['classification_loss']

                if 'accuracy' in task_result:
                    result[f'{task_name}_accuracy'] = task_result['accuracy']

                if 'intra_class_loss' in task_result:
                    total_intra_loss = total_intra_loss + task_result['intra_class_loss']

                # Compactness loss
                num_classes = self.config.tasks[task_name][0]
                compact_loss = self.compactness_loss(
                    task_result['embedding'],
                    labels,
                    num_classes,
                )
                total_intra_loss = total_intra_loss + compact_loss

        # Average classification loss
        if num_tasks_with_labels > 0:
            result['classification_loss'] = total_cls_loss / num_tasks_with_labels
            result['intra_class_loss'] = total_intra_loss / num_tasks_with_labels
        else:
            result['classification_loss'] = torch.tensor(0.0, device=device)
            result['intra_class_loss'] = torch.tensor(0.0, device=device)

        # Orthogonality loss between task subspaces
        ortho_loss = self.orthogonality_loss(task_embeddings)
        result['orthogonality_loss'] = ortho_loss

        # Prototype alignment
        proto_result = self.prototype_aligner(meta_emb, text_embedding)
        result['prototype_assignment'] = proto_result['prototype_assignment']
        result['weighted_prototype'] = proto_result['weighted_prototype']
        result['diversity_loss'] = proto_result['diversity_loss']

        if 'alignment_loss' in proto_result:
            result['alignment_loss'] = proto_result['alignment_loss']
        else:
            result['alignment_loss'] = torch.tensor(0.0, device=device)

        # Total loss
        result['total_loss'] = (
            self.config.classification_weight * result['classification_loss'] +
            self.config.intra_class_weight * result['intra_class_loss'] +
            self.config.orthogonality_weight * ortho_loss +
            self.config.prototype_weight * (
                result['diversity_loss'] + result['alignment_loss']
            )
        )

        # Fused embedding (all tasks + prototype)
        all_task_embs = [task_embeddings[t] for t in self.task_names]
        all_task_embs.append(proto_result['weighted_prototype'])
        concat_emb = torch.cat(all_task_embs, dim=-1)
        fused_emb = self.fusion(concat_emb)
        result['fused_embedding'] = fused_emb

        return result

    def get_embeddings(
        self,
        audio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract task-specific embeddings for inference.

        Args:
            audio_features: [batch, seq, input_dim] audio features
            mask: [batch, seq] attention mask
            tasks: List of tasks to extract (all if None)

        Returns:
            Dict mapping task name to embeddings
        """
        # Get META embedding
        meta_emb = self.encoder(audio_features, mask)

        if tasks is None:
            tasks = self.task_names

        embeddings = {
            'meta': meta_emb,
        }

        for task_name in tasks:
            if task_name in self.task_projectors:
                projector = self.task_projectors[task_name]
                task_result = projector(meta_emb)
                embeddings[task_name] = task_result['embedding']

                if 'logits' in task_result:
                    preds = task_result['logits'].argmax(dim=-1)
                    embeddings[f'{task_name}_pred'] = preds
                    # Convert to labels
                    pred_labels = [
                        idx_to_label(task_name, p.item())
                        for p in preds
                    ]
                    embeddings[f'{task_name}_labels'] = pred_labels

        # Prototype embedding
        proto_result = self.prototype_aligner(meta_emb)
        embeddings['prototype'] = proto_result['weighted_prototype']

        return embeddings

    def get_text_style_embedding(
        self,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get style embedding from text description.

        Args:
            text_embedding: [batch, text_dim] text encoder output

        Returns:
            Style embedding for conditioning
        """
        return self.prototype_aligner.get_text_style_embedding(text_embedding)

    def predict(
        self,
        audio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Predict all paralinguistic attributes.

        Args:
            audio_features: [batch, seq, input_dim] audio features
            mask: [batch, seq] attention mask

        Returns:
            Dict with predictions for all tasks
        """
        embeddings = self.get_embeddings(audio_features, mask)

        predictions = {}
        for task_name in self.task_names:
            if f'{task_name}_labels' in embeddings:
                predictions[task_name] = embeddings[f'{task_name}_labels']
            elif f'{task_name}_pred' in embeddings:
                predictions[task_name] = embeddings[f'{task_name}_pred']

        return predictions


# =============================================================================
# CSM INTEGRATION ADAPTER
# =============================================================================

class ParaMETAAdapter(nn.Module):
    """
    Adapter to integrate ParaMETA with CSM prosody pipeline.

    Converts paralinguistic embeddings to prosody prefix tokens
    compatible with ProsodyControlledCSM.

    Features:
    - Multi-task embedding fusion
    - Text-based style prompting
    - Controllable attribute mixing
    """

    def __init__(
        self,
        config: ParaMETAConfig,
        prosody_hidden: int = 2048,
        num_prosody_tokens: int = 4,
    ):
        super().__init__()
        self.config = config
        self.prosody_hidden = prosody_hidden
        self.num_tokens = num_prosody_tokens

        # ParaMETA model
        self.parameta = ParaMETA(config)

        # Token generator from fused embedding
        self.token_generator = nn.Sequential(
            nn.Linear(config.meta_dim, prosody_hidden),
            nn.LayerNorm(prosody_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(prosody_hidden, prosody_hidden * num_prosody_tokens),
        )

        # Per-task token generators (for fine-grained control)
        self.task_token_generators = nn.ModuleDict()
        for task_name in config.tasks.keys():
            self.task_token_generators[task_name] = nn.Sequential(
                nn.Linear(config.task_subspace_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
                nn.GELU(),
            )

        # Task weight modulator
        self.task_weights = nn.Parameter(torch.ones(len(config.tasks)))

    def forward(
        self,
        audio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        task_weights: Optional[Dict[str, float]] = None,
        **task_labels,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from audio features.

        Args:
            audio_features: [batch, seq, input_dim] audio features
            mask: [batch, seq] attention mask
            text_embedding: [batch, text_dim] optional text for style prompting
            task_weights: Optional weights for each task
            **task_labels: Labels for training

        Returns:
            Dict with prosody tokens and losses
        """
        batch_size = audio_features.shape[0]

        # Forward through ParaMETA
        parameta_out = self.parameta(
            audio_features, text_embedding, mask, **task_labels
        )

        result = {
            'meta_embedding': parameta_out['meta_embedding'],
            'fused_embedding': parameta_out['fused_embedding'],
        }

        # Copy losses
        for key in ['total_loss', 'classification_loss', 'orthogonality_loss',
                    'intra_class_loss', 'alignment_loss']:
            if key in parameta_out:
                result[key] = parameta_out[key]

        # Copy task-specific outputs
        for task_name in self.config.tasks.keys():
            for suffix in ['_embedding', '_logits', '_accuracy', '_cls_loss']:
                key = f'{task_name}{suffix}'
                if key in parameta_out:
                    result[key] = parameta_out[key]

        # Generate prosody tokens from fused embedding
        fused_emb = parameta_out['fused_embedding']
        tokens = self.token_generator(fused_emb)
        tokens = tokens.view(batch_size, self.num_tokens, self.prosody_hidden)
        result['prosody_tokens'] = tokens

        # Generate task-weighted tokens
        if task_weights is not None:
            weighted_tokens = torch.zeros(
                batch_size, self.prosody_hidden, device=tokens.device
            )
            for i, task_name in enumerate(self.config.tasks.keys()):
                weight = task_weights.get(task_name, 1.0)
                task_emb = parameta_out[f'{task_name}_embedding']
                task_tokens = self.task_token_generators[task_name](task_emb)
                weighted_tokens = weighted_tokens + weight * task_tokens

            # Average and expand to num_tokens
            weighted_tokens = weighted_tokens / len(self.config.tasks)
            result['task_weighted_tokens'] = weighted_tokens.unsqueeze(1).expand(
                -1, self.num_tokens, -1
            )

        return result

    def from_text(
        self,
        text_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from text description.

        Enables text-based style prompting without reference audio.

        Args:
            text_embedding: [batch, text_dim] text encoder output

        Returns:
            Dict with prosody tokens
        """
        # Get style embedding from prototypes
        style_emb = self.parameta.get_text_style_embedding(text_embedding)

        # Generate tokens
        # First project to meta dim
        batch_size = text_embedding.shape[0]

        # Use prototype-based style
        tokens = self.token_generator(
            # Project style_emb to meta_dim (it's currently prototype_dim)
            F.linear(
                style_emb,
                torch.eye(
                    self.config.meta_dim,
                    self.config.prototype_dim,
                    device=style_emb.device
                )[:, :style_emb.shape[-1]]
            )
        )
        tokens = tokens.view(batch_size, self.num_tokens, self.prosody_hidden)

        return {'prosody_tokens': tokens}

    def from_attributes(
        self,
        emotion: Optional[str] = None,
        gender: Optional[str] = None,
        age: Optional[str] = None,
        accent: Optional[str] = None,
        speaking_rate: Optional[str] = None,
        energy: Optional[str] = None,
        device: torch.device = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from explicit attributes.

        Args:
            emotion: Emotion label
            gender: Gender label
            age: Age group label
            accent: Accent label
            speaking_rate: Speaking rate label
            energy: Energy level label
            device: Target device

        Returns:
            Dict with prosody tokens
        """
        if device is None:
            device = next(self.parameters()).device

        # Create one-hot encodings and use class embeddings
        embeddings = []

        for task_name, label in [
            ('emotion', emotion),
            ('gender', gender),
            ('age', age),
            ('accent', accent),
            ('speaking_rate', speaking_rate),
            ('energy', energy),
        ]:
            if task_name not in self.config.tasks:
                continue

            projector = self.parameta.task_projectors[task_name]

            if label is not None and projector.class_embeddings is not None:
                idx = label_to_idx(task_name, label)
                emb = projector.class_embeddings(
                    torch.tensor([idx], device=device)
                )
            else:
                # Use zero embedding if label not specified
                emb = torch.zeros(
                    1, self.config.task_subspace_dim, device=device
                )

            embeddings.append(emb)

        # Concatenate and generate tokens
        concat_emb = torch.cat(embeddings, dim=-1)

        # Project to prosody tokens
        # Need to match expected input size for fusion
        proto_emb = torch.zeros(1, self.config.prototype_dim, device=device)
        full_emb = torch.cat([concat_emb, proto_emb], dim=-1)

        fused = self.parameta.fusion(full_emb)
        tokens = self.token_generator(fused)
        tokens = tokens.view(1, self.num_tokens, self.prosody_hidden)

        return {'prosody_tokens': tokens}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_parameta_adapter(
    checkpoint: Optional[str] = None,
    config: Optional[ParaMETAConfig] = None,
    device: Optional[torch.device] = None,
) -> ParaMETAAdapter:
    """
    Create ParaMETAAdapter with optional checkpoint loading.

    Args:
        checkpoint: Path to checkpoint file
        config: Configuration (uses default if None)
        device: Target device

    Returns:
        ParaMETAAdapter instance
    """
    if config is None:
        config = ParaMETAConfig()

    adapter = ParaMETAAdapter(config)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        adapter.load_state_dict(state_dict)

    if device is not None:
        adapter = adapter.to(device)

    return adapter


def describe_predictions(predictions: Dict[str, Any]) -> str:
    """
    Create natural language description of predictions.

    Args:
        predictions: Dict from model.predict()

    Returns:
        Human-readable description
    """
    parts = []

    if 'emotion' in predictions:
        emotion = predictions['emotion']
        if isinstance(emotion, list):
            emotion = emotion[0]
        parts.append(f"expressing {emotion} emotion")

    if 'gender' in predictions:
        gender = predictions['gender']
        if isinstance(gender, list):
            gender = gender[0]
        parts.append(f"{gender} voice")

    if 'age' in predictions:
        age = predictions['age']
        if isinstance(age, list):
            age = age[0]
        parts.append(f"{age.replace('_', ' ')} speaker")

    if 'accent' in predictions:
        accent = predictions['accent']
        if isinstance(accent, list):
            accent = accent[0]
        parts.append(f"with {accent} accent")

    if 'speaking_rate' in predictions:
        rate = predictions['speaking_rate']
        if isinstance(rate, list):
            rate = rate[0]
        parts.append(f"at {rate} pace")

    if 'energy' in predictions:
        energy = predictions['energy']
        if isinstance(energy, list):
            energy = energy[0]
        parts.append(f"with {energy} energy")

    if not parts:
        return "neutral speaking style"

    return ", ".join(parts)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ParaMETA: Disentangled Paralinguistic Learning - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = ParaMETAConfig(
        input_dim=768,
        meta_dim=512,
        task_subspace_dim=128,
        num_prototypes=64,
    )

    # Test 1: ParaMETAEncoder
    print("\n[Test 1] ParaMETAEncoder...")
    encoder = ParaMETAEncoder(config).to(device)
    features = torch.randn(2, 50, config.input_dim, device=device)
    mask = torch.ones(2, 50, device=device, dtype=torch.bool)

    meta_emb = encoder(features, mask)
    print(f"  Input shape: {features.shape}")
    print(f"  META embedding shape: {meta_emb.shape}")
    assert meta_emb.shape == (2, config.meta_dim)

    meta_dict = encoder(features, mask, return_sequence=True)
    print(f"  Sequence shape: {meta_dict['sequence'].shape}")
    print("  [PASS]")

    # Test 2: TaskSpecificProjector
    print("\n[Test 2] TaskSpecificProjector...")
    projector = TaskSpecificProjector(
        config, "emotion", 8, is_discrete=True
    ).to(device)

    labels = torch.randint(0, 8, (2,), device=device)
    proj_result = projector(meta_emb, labels)

    print(f"  Task embedding shape: {proj_result['embedding'].shape}")
    print(f"  Logits shape: {proj_result['logits'].shape}")
    print(f"  Classification loss: {proj_result['classification_loss'].item():.4f}")
    print(f"  Accuracy: {proj_result['accuracy'].item():.4f}")
    print("  [PASS]")

    # Test 3: PrototypeAligner
    print("\n[Test 3] PrototypeAligner...")
    aligner = PrototypeAligner(config).to(device)
    text_emb = torch.randn(2, config.text_encoder_dim, device=device)

    proto_result = aligner(meta_emb, text_emb)
    print(f"  Speech similarity shape: {proto_result['speech_similarity'].shape}")
    print(f"  Prototype assignment shape: {proto_result['prototype_assignment'].shape}")
    print(f"  Weighted prototype shape: {proto_result['weighted_prototype'].shape}")
    print(f"  Alignment loss: {proto_result['alignment_loss'].item():.4f}")
    print(f"  Diversity loss: {proto_result['diversity_loss'].item():.4f}")
    print("  [PASS]")

    # Test 4: SubspaceOrthogonalityLoss
    print("\n[Test 4] SubspaceOrthogonalityLoss...")
    ortho_loss_fn = SubspaceOrthogonalityLoss(config)
    task_embs = {
        'emotion': torch.randn(2, config.task_subspace_dim, device=device),
        'gender': torch.randn(2, config.task_subspace_dim, device=device),
        'age': torch.randn(2, config.task_subspace_dim, device=device),
    }
    ortho_loss = ortho_loss_fn(task_embs)
    print(f"  Orthogonality loss: {ortho_loss.item():.4f}")
    print("  [PASS]")

    # Test 5: IntraClassCompactnessLoss
    print("\n[Test 5] IntraClassCompactnessLoss...")
    compact_loss_fn = IntraClassCompactnessLoss(config)
    emb = torch.randn(8, config.task_subspace_dim, device=device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    compact_loss = compact_loss_fn(emb, labels, num_classes=4)
    print(f"  Compactness loss: {compact_loss.item():.4f}")
    print("  [PASS]")

    # Test 6: Complete ParaMETA
    print("\n[Test 6] Complete ParaMETA module...")
    model = ParaMETA(config).to(device)

    # Create batch with labels
    batch_size = 4
    audio_features = torch.randn(batch_size, 50, config.input_dim, device=device)
    emotion_labels = torch.randint(0, 8, (batch_size,), device=device)
    gender_labels = torch.randint(0, 2, (batch_size,), device=device)
    age_labels = torch.randint(0, 5, (batch_size,), device=device)

    output = model(
        audio_features,
        emotion_labels=emotion_labels,
        gender_labels=gender_labels,
        age_labels=age_labels,
    )

    print(f"  META embedding shape: {output['meta_embedding'].shape}")
    print(f"  Fused embedding shape: {output['fused_embedding'].shape}")
    print(f"  Total loss: {output['total_loss'].item():.4f}")
    print(f"  Classification loss: {output['classification_loss'].item():.4f}")
    print(f"  Orthogonality loss: {output['orthogonality_loss'].item():.4f}")

    for task in ['emotion', 'gender', 'age']:
        print(f"  {task} accuracy: {output[f'{task}_accuracy'].item():.4f}")
    print("  [PASS]")

    # Test 7: Get embeddings (inference)
    print("\n[Test 7] Get embeddings (inference)...")
    embeddings = model.get_embeddings(audio_features)
    print(f"  Available embeddings: {list(embeddings.keys())}")
    print(f"  Emotion embedding: {embeddings['emotion'].shape}")
    print(f"  Emotion predictions: {embeddings['emotion_labels']}")
    print("  [PASS]")

    # Test 8: Predict all attributes
    print("\n[Test 8] Predict all attributes...")
    predictions = model.predict(audio_features)
    print(f"  Predictions: {predictions}")
    description = describe_predictions(predictions)
    print(f"  Description: {description}")
    print("  [PASS]")

    # Test 9: ParaMETAAdapter
    print("\n[Test 9] ParaMETAAdapter...")
    adapter = ParaMETAAdapter(config).to(device)

    adapter_out = adapter(
        audio_features,
        emotion_labels=emotion_labels,
        gender_labels=gender_labels,
    )
    print(f"  Prosody tokens shape: {adapter_out['prosody_tokens'].shape}")
    print(f"  Total loss: {adapter_out['total_loss'].item():.4f}")
    print("  [PASS]")

    # Test 10: From attributes
    print("\n[Test 10] From explicit attributes...")
    attr_out = adapter.from_attributes(
        emotion="happy",
        gender="female",
        age="young_adult",
        speaking_rate="fast",
        device=device,
    )
    print(f"  Prosody tokens shape: {attr_out['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 11: Gradient flow
    print("\n[Test 11] Gradient flow...")
    audio_features.requires_grad_(True)
    output = model(audio_features, emotion_labels=emotion_labels)
    output['total_loss'].backward()
    print(f"  Input gradient norm: {audio_features.grad.norm().item():.6f}")
    print(f"  Gradients flow correctly: {audio_features.grad.abs().sum().item() > 0}")
    print("  [PASS]")

    # Test 12: Task label utilities
    print("\n[Test 12] Task label utilities...")
    print(f"  Emotion labels: {get_task_labels('emotion')}")
    print(f"  Number of emotions: {get_num_classes('emotion')}")
    print(f"  'happy' -> index: {label_to_idx('emotion', 'happy')}")
    print(f"  index 2 -> label: {idx_to_label('emotion', 2)}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All ParaMETA tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from parameta import (
    ParaMETAConfig,
    ParaMETA,
    ParaMETAAdapter,
    PARALINGUISTIC_TASKS,
    describe_predictions,
)

# Initialize
config = ParaMETAConfig()
model = ParaMETA(config).cuda()

# Training: Multi-task learning
for batch in dataloader:
    audio_features = feature_extractor(batch['audio'])

    losses = model(
        audio_features,
        emotion_labels=batch['emotion'],
        gender_labels=batch['gender'],
        age_labels=batch['age'],
        accent_labels=batch['accent'],
    )

    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()

    # Monitor per-task performance
    for task in PARALINGUISTIC_TASKS.keys():
        if f'{task}_accuracy' in losses:
            print(f"{task} accuracy: {losses[f'{task}_accuracy'].item():.4f}")

# Inference: Get all paralinguistic attributes
model.eval()
with torch.no_grad():
    predictions = model.predict(audio_features)
    description = describe_predictions(predictions)
    print(description)  # "expressing happy emotion, female voice, young adult speaker"

# Get task-specific embeddings
embeddings = model.get_embeddings(audio_features)
emotion_emb = embeddings['emotion']  # [batch, 128]
gender_emb = embeddings['gender']    # [batch, 128]

# CSM integration
adapter = ParaMETAAdapter(config)
result = adapter(audio_features)
prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

# Text-based style prompting
text_emb = text_encoder("speaking with excitement and energy")
style_tokens = adapter.from_text(text_emb)['prosody_tokens']

# Explicit attribute control
tokens = adapter.from_attributes(
    emotion="excited",
    gender="male",
    age="adult",
    speaking_rate="fast",
    energy="high",
)['prosody_tokens']

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
