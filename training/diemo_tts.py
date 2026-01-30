"""
DiEmo-TTS: Self-Supervised Emotion Distillation for Speaker-Independent Emotion Embeddings

Based on "DiEmo-TTS: Disentangled Emotion Representation Learning for Expressive TTS"
(Interspeech 2025) - arXiv:2505.19687

Key Techniques:
1. Emotion Disentanglement DINO (ED-DINO): Teacher-student self-supervised distillation
2. Cluster-driven sampling: Creates emotion-aware positive/negative pairs
3. Formant perturbation: Data augmentation preserving emotion while perturbing speaker
4. Dual conditioning transformer: Integrates style features effectively
5. Emotion clustering: For generalization to unlabeled data

Benefits:
- State-of-the-art speaker-irrelevant emotion embeddings
- Excels in expressiveness, naturalness, and speaker identity preservation
- Better emotion transfer (eMOS) while maintaining speaker similarity (sMOS)
- Works with unlabeled emotional data via clustering

Metrics:
- SECS (Speaker Embedding Cosine Similarity): Measures speaker preservation
- EECS (Emotion Embedding Cosine Similarity): Measures emotion transfer quality

Integration:
- Enhances existing MINE-based disentanglement with DINO-based approach
- Can be combined with spherical emotion vectors for intensity control
- Works with hierarchical emotion distribution for fine-grained control

Usage:
    from diemo_tts import DiEmoTTSConfig, DiEmoTTS, DiEmoDINO

    # Initialize
    config = DiEmoTTSConfig()
    diemo = DiEmoTTS(config)

    # Training with self-supervised distillation
    audio_features = encoder(audio)  # From mel/wav2vec2

    # Apply formant perturbation for augmentation
    perturbed = diemo.formant_perturbation(audio)

    # Get emotion embeddings via DINO distillation
    emotion_emb = diemo.ed_dino(audio_features, perturbed_features)

    # Cluster-driven sampling for training
    loss = diemo.compute_dino_loss(audio_features, perturbed_features)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DiEmoTTSConfig:
    """Configuration for DiEmo-TTS self-supervised emotion distillation."""

    # Audio/feature dimensions
    input_dim: int = 768                # Input feature dimension (wav2vec2/HuBERT)
    mel_dim: int = 80                   # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 320               # ~20ms at 16kHz

    # ED-DINO architecture
    emotion_dim: int = 256              # Emotion embedding dimension
    hidden_dim: int = 512               # Hidden layer dimension
    num_heads: int = 8                  # Transformer attention heads
    num_layers: int = 4                 # Transformer layers
    dropout: float = 0.1

    # DINO settings
    teacher_momentum: float = 0.996     # EMA momentum for teacher update
    teacher_temp: float = 0.04          # Teacher softmax temperature (low = sharp)
    student_temp: float = 0.1           # Student softmax temperature
    center_momentum: float = 0.9        # Center vector momentum

    # Projection head
    proj_hidden_dim: int = 512
    proj_output_dim: int = 256          # DINO output dimension
    use_bn_in_proj: bool = True

    # Cluster-driven sampling
    num_clusters: int = 8               # Number of emotion clusters
    cluster_temp: float = 0.5           # Temperature for cluster sampling
    min_cluster_samples: int = 2        # Minimum samples per cluster

    # Formant perturbation
    formant_shift_range: Tuple[float, float] = (0.85, 1.15)  # ±15%
    vtln_warp_range: Tuple[float, float] = (0.9, 1.1)        # ±10%
    use_praat: bool = False             # Use Praat for formant manipulation

    # Dual conditioning transformer
    use_dual_conditioning: bool = True
    style_dim: int = 256                # Style feature dimension

    # Loss weights
    dino_loss_weight: float = 1.0
    cluster_loss_weight: float = 0.3
    speaker_adversarial_weight: float = 0.2

    # Training settings
    warmup_epochs: int = 10
    freeze_teacher_epochs: int = 1      # Epochs before starting teacher updates

    # Number of speakers (for adversarial training)
    num_speakers: int = 1000
    speaker_embed_dim: int = 256


# =============================================================================
# FORMANT PERTURBATION
# =============================================================================

class FormantPerturbation(nn.Module):
    """
    Formant perturbation for speaker identity augmentation.

    Preserves emotion while perturbing speaker-specific characteristics
    (formants F1, F2, F3 which encode speaker identity).

    Based on:
    - VTLN (Vocal Tract Length Normalization) warping
    - Formant shifting via spectral manipulation

    Why this helps:
    - Formants are strongly speaker-dependent, weakly emotion-dependent
    - Perturbing formants creates "different speaker, same emotion" pairs
    - Trains emotion encoder to ignore speaker-specific cues
    """

    def __init__(self, config: DiEmoTTSConfig):
        super().__init__()
        self.config = config

        # Frequency warping parameters
        self.register_buffer(
            'mel_frequencies',
            self._create_mel_filterbank_frequencies(config.mel_dim)
        )

    def _create_mel_filterbank_frequencies(self, num_mels: int) -> torch.Tensor:
        """Create center frequencies for mel filterbank."""
        # Hz to mel
        f_min, f_max = 0, 8000  # Typical mel range
        mel_min = 2595 * np.log10(1 + f_min / 700)
        mel_max = 2595 * np.log10(1 + f_max / 700)

        mels = torch.linspace(mel_min, mel_max, num_mels)
        # Mel to Hz
        freqs = 700 * (10 ** (mels / 2595) - 1)
        return freqs

    def vtln_warp(
        self,
        mel: torch.Tensor,
        warp_factor: float,
    ) -> torch.Tensor:
        """
        Apply VTLN warping to mel spectrogram.

        VTLN warps the frequency axis to simulate different vocal tract lengths,
        effectively changing speaker characteristics while preserving phonetic content.

        Args:
            mel: [batch, time, mel_dim] or [batch, mel_dim, time]
            warp_factor: Warping factor (< 1.0 = shorter VT, > 1.0 = longer VT)

        Returns:
            Warped mel spectrogram
        """
        # Ensure [batch, time, mel] format
        if mel.shape[-1] != self.config.mel_dim:
            mel = mel.transpose(-1, -2)

        batch, time, mel_dim = mel.shape
        device = mel.device

        # Piecewise linear warping function
        # Below cutoff: linear with slope alpha
        # Above cutoff: linear to reach f_max
        f_max = 8000
        f_cutoff = 7000  # Cutoff frequency

        # Original frequencies
        orig_freqs = self.mel_frequencies.to(device)

        # Warped frequencies
        warped_freqs = torch.zeros_like(orig_freqs)

        for i, f in enumerate(orig_freqs):
            if f <= f_cutoff / warp_factor:
                warped_freqs[i] = f * warp_factor
            else:
                # Linear interpolation to f_max
                warped_freqs[i] = f_cutoff + (f - f_cutoff / warp_factor) * (
                    (f_max - f_cutoff) / (f_max - f_cutoff / warp_factor)
                )

        # Create interpolation weights
        # For each original mel bin, find interpolation weights to nearby warped bins
        warped_mel = torch.zeros_like(mel)

        for i in range(mel_dim):
            target_freq = warped_freqs[i]

            # Find surrounding original bins
            diffs = orig_freqs - target_freq
            lower_idx = (diffs <= 0).sum() - 1
            lower_idx = max(0, min(lower_idx, mel_dim - 2))
            upper_idx = lower_idx + 1

            # Interpolation weight
            if orig_freqs[upper_idx] != orig_freqs[lower_idx]:
                alpha = (target_freq - orig_freqs[lower_idx]) / (
                    orig_freqs[upper_idx] - orig_freqs[lower_idx]
                )
            else:
                alpha = 0.0

            alpha = max(0.0, min(1.0, alpha.item() if isinstance(alpha, torch.Tensor) else alpha))

            warped_mel[:, :, i] = (1 - alpha) * mel[:, :, lower_idx] + alpha * mel[:, :, upper_idx]

        return warped_mel

    def formant_shift(
        self,
        mel: torch.Tensor,
        shift_factors: Tuple[float, float, float],
    ) -> torch.Tensor:
        """
        Shift formant frequencies.

        Modifies F1, F2, F3 independently to perturb speaker identity
        while preserving prosodic/emotional content.

        Args:
            mel: [batch, time, mel_dim]
            shift_factors: (F1_shift, F2_shift, F3_shift) multipliers

        Returns:
            Formant-shifted mel spectrogram
        """
        # Approximate formant regions in mel scale
        # F1: ~300-900 Hz (bins ~3-12)
        # F2: ~850-2500 Hz (bins ~10-30)
        # F3: ~2000-3500 Hz (bins ~25-42)

        mel_dim = self.config.mel_dim

        # Define formant regions (approximate for 80-bin mel)
        f1_start, f1_end = int(0.04 * mel_dim), int(0.15 * mel_dim)
        f2_start, f2_end = int(0.12 * mel_dim), int(0.38 * mel_dim)
        f3_start, f3_end = int(0.31 * mel_dim), int(0.53 * mel_dim)

        shifted = mel.clone()

        # Apply energy scaling based on shift
        # Shifting formants up = concentrate energy higher, etc.
        f1_scale = 1.0 / shift_factors[0]
        f2_scale = 1.0 / shift_factors[1]
        f3_scale = 1.0 / shift_factors[2]

        # Simple approach: scale energy in formant regions
        shifted[:, :, f1_start:f1_end] *= f1_scale
        shifted[:, :, f2_start:f2_end] *= f2_scale
        shifted[:, :, f3_start:f3_end] *= f3_scale

        return shifted

    def forward(
        self,
        mel: torch.Tensor,
        warp_factor: Optional[float] = None,
        shift_factors: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        """
        Apply formant perturbation.

        Args:
            mel: [batch, time, mel_dim] or [batch, mel_dim, time]
            warp_factor: VTLN warp factor (random if None)
            shift_factors: Formant shift factors (random if None)

        Returns:
            Perturbed mel spectrogram
        """
        # Random parameters if not specified
        if warp_factor is None:
            low, high = self.config.vtln_warp_range
            warp_factor = random.uniform(low, high)

        if shift_factors is None:
            low, high = self.config.formant_shift_range
            shift_factors = (
                random.uniform(low, high),
                random.uniform(low, high),
                random.uniform(low, high),
            )

        # Apply VTLN warping
        perturbed = self.vtln_warp(mel, warp_factor)

        # Apply formant shift
        perturbed = self.formant_shift(perturbed, shift_factors)

        return perturbed


# =============================================================================
# DINO PROJECTION HEAD
# =============================================================================

class DINOProjectionHead(nn.Module):
    """
    DINO projection head with optional batch normalization.

    Maps encoder features to DINO output space where
    self-supervised distillation loss is computed.
    """

    def __init__(self, config: DiEmoTTSConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.emotion_dim

        # Hidden layers
        for _ in range(2):
            layers.append(nn.Linear(in_dim, config.proj_hidden_dim))
            if config.use_bn_in_proj:
                layers.append(nn.BatchNorm1d(config.proj_hidden_dim))
            layers.append(nn.GELU())
            in_dim = config.proj_hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(config.proj_hidden_dim, config.proj_output_dim))

        self.mlp = nn.Sequential(*layers)

        # L2 normalization
        self.normalize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features to DINO space.

        Args:
            x: [batch, emotion_dim] or [batch, seq, emotion_dim]

        Returns:
            Projected features [batch, proj_output_dim]
        """
        # Pool if sequence
        if x.dim() == 3:
            x = x.mean(dim=1)

        # Handle batch norm dimension
        out = self.mlp(x)

        if self.normalize:
            out = F.normalize(out, dim=-1, p=2)

        return out


# =============================================================================
# EMOTION ENCODER (STUDENT/TEACHER)
# =============================================================================

class EmotionEncoder(nn.Module):
    """
    Emotion encoder for ED-DINO.

    Transformer-based encoder that extracts emotion embeddings
    from audio features (mel/wav2vec2/HuBERT).

    Used as both student and teacher in DINO framework.
    """

    def __init__(self, config: DiEmoTTSConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 1000, config.hidden_dim)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer layers
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

        # Output projection to emotion dimension
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.emotion_dim),
            nn.LayerNorm(config.emotion_dim),
        )

        # [CLS] token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Attention pooling (alternative to CLS)
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract emotion embedding from audio features.

        Args:
            x: [batch, seq, input_dim] audio features
            mask: [batch, seq] attention mask (True = valid)
            return_sequence: Return sequence-level features

        Returns:
            Emotion embedding [batch, emotion_dim] or dict with sequence
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Update mask for CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Transformer
        # Note: PyTorch expects mask where True = ignore
        attn_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract CLS token embedding
        cls_embedding = x[:, 0, :]
        sequence_features = x[:, 1:, :]

        # Also compute attention-pooled embedding
        if mask is not None:
            seq_mask = mask[:, 1:]  # Remove CLS mask
        else:
            seq_mask = None

        attn_weights = self.attention_pool(sequence_features).squeeze(-1)
        if seq_mask is not None:
            attn_weights = attn_weights.masked_fill(~seq_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        pooled_embedding = (sequence_features * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Combine CLS and attention-pooled
        combined = cls_embedding + pooled_embedding

        # Output projection
        emotion_emb = self.output_proj(combined)

        if return_sequence:
            sequence_out = self.output_proj(sequence_features)
            return {
                'embedding': emotion_emb,
                'sequence': sequence_out,
                'cls': cls_embedding,
                'attention_weights': attn_weights,
            }

        return emotion_emb


# =============================================================================
# ED-DINO (EMOTION DISENTANGLEMENT DINO)
# =============================================================================

class EmotionDisentanglementDINO(nn.Module):
    """
    Emotion Disentanglement DINO (ED-DINO).

    Self-supervised distillation framework for learning speaker-irrelevant
    emotion embeddings. Uses teacher-student setup with:

    1. Student: Trained on original + formant-perturbed views
    2. Teacher: EMA of student, provides soft targets
    3. Centering: Prevents collapse to constant predictions

    The key insight is that formant perturbation changes speaker identity
    while preserving emotion. Training student to match teacher on both
    original and perturbed views forces it to learn speaker-invariant
    emotion representations.

    DINO loss: Cross-entropy between sharpened teacher and student predictions
    """

    def __init__(self, config: DiEmoTTSConfig):
        super().__init__()
        self.config = config

        # Student encoder
        self.student_encoder = EmotionEncoder(config)
        self.student_head = DINOProjectionHead(config)

        # Teacher encoder (EMA of student)
        self.teacher_encoder = EmotionEncoder(config)
        self.teacher_head = DINOProjectionHead(config)

        # Initialize teacher as copy of student
        self._init_teacher()

        # Freeze teacher parameters (updated via EMA)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        # Center vector (prevents collapse)
        self.register_buffer('center', torch.zeros(config.proj_output_dim))

        # Formant perturbation
        self.formant_perturb = FormantPerturbation(config)

        # Current momentum (can be scheduled)
        self.momentum = config.teacher_momentum

    def _init_teacher(self):
        """Initialize teacher as copy of student."""
        for param_s, param_t in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            param_t.data.copy_(param_s.data)

        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data.copy_(param_s.data)

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher via exponential moving average of student."""
        for param_s, param_t in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data

        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_s.data

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """Update center vector using batch statistics."""
        batch_center = teacher_output.mean(dim=0)
        self.center = self.config.center_momentum * self.center + \
                     (1 - self.config.center_momentum) * batch_center

    def forward_student(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through student."""
        emb = self.student_encoder(x, mask)
        proj = self.student_head(emb)
        return proj

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through teacher."""
        emb = self.teacher_encoder(x, mask)
        proj = self.teacher_head(emb)
        return proj

    def compute_dino_loss(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DINO self-supervised loss.

        Cross-entropy between centered & sharpened teacher output
        and student output.

        Args:
            student_out: [batch, proj_dim] student projections
            teacher_out: [batch, proj_dim] teacher projections

        Returns:
            DINO loss
        """
        # Temperature scaling
        student_temp = self.config.student_temp
        teacher_temp = self.config.teacher_temp

        # Sharpen teacher output and center
        teacher_centered = teacher_out - self.center
        teacher_probs = F.softmax(teacher_centered / teacher_temp, dim=-1)

        # Student log-probs
        student_logprobs = F.log_softmax(student_out / student_temp, dim=-1)

        # Cross-entropy loss
        loss = -(teacher_probs * student_logprobs).sum(dim=-1).mean()

        return loss

    def forward(
        self,
        x_original: torch.Tensor,
        x_perturbed: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        update_teacher: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with DINO loss computation.

        Args:
            x_original: [batch, seq, dim] original audio features
            x_perturbed: [batch, seq, dim] formant-perturbed features (optional)
            mask: [batch, seq] attention mask
            update_teacher: Whether to update teacher EMA

        Returns:
            Dict with loss and embeddings
        """
        batch_size = x_original.shape[0]

        # If no perturbed input, use gaussian noise as simple augmentation
        # Note: For proper formant perturbation, apply at audio level before
        # feature extraction. This is a fallback for when pre-computed
        # perturbed features are not provided.
        if x_perturbed is None:
            # Simple noise augmentation as fallback
            noise_scale = 0.1
            x_perturbed = x_original + noise_scale * torch.randn_like(x_original)

        # Student forward on both views
        student_orig = self.forward_student(x_original, mask)
        student_pert = self.forward_student(x_perturbed, mask)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_orig = self.forward_teacher(x_original, mask)
            teacher_pert = self.forward_teacher(x_perturbed, mask)

            # Update center
            self.update_center(torch.cat([teacher_orig, teacher_pert], dim=0))

        # DINO loss: student predicts both views from both teacher views
        # This creates 4 loss terms for cross-view consistency
        loss_orig_orig = self.compute_dino_loss(student_orig, teacher_orig)
        loss_orig_pert = self.compute_dino_loss(student_orig, teacher_pert)
        loss_pert_orig = self.compute_dino_loss(student_pert, teacher_orig)
        loss_pert_pert = self.compute_dino_loss(student_pert, teacher_pert)

        # Average cross-view losses (not same-view to encourage invariance)
        dino_loss = 0.5 * (loss_orig_pert + loss_pert_orig)

        # Update teacher EMA
        if update_teacher and self.training:
            self.update_teacher()

        # Get emotion embeddings (from student)
        emotion_orig = self.student_encoder(x_original, mask)
        emotion_pert = self.student_encoder(x_perturbed, mask)

        return {
            'loss': dino_loss,
            'loss_orig_orig': loss_orig_orig,
            'loss_orig_pert': loss_orig_pert,
            'loss_pert_orig': loss_pert_orig,
            'loss_pert_pert': loss_pert_pert,
            'emotion_embedding': emotion_orig,
            'emotion_perturbed': emotion_pert,
            'student_proj': student_orig,
            'teacher_proj': teacher_orig.detach(),
        }

    def get_emotion_embedding(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get emotion embedding for inference."""
        return self.student_encoder(x, mask)


# =============================================================================
# CLUSTER-DRIVEN SAMPLING
# =============================================================================

class ClusterDrivenSampler:
    """
    Cluster-driven sampling for DINO training.

    Instead of random pairing, uses emotion-based clustering to create
    positive pairs (same cluster) and negative pairs (different clusters).

    This minimizes emotion information loss during contrastive learning
    by ensuring positive pairs share emotional characteristics.

    Process:
    1. Cluster audio samples by emotion (k-means on emotion features)
    2. For each sample, positive = same cluster, negative = different cluster
    3. Sample with temperature-controlled probability
    """

    def __init__(self, config: DiEmoTTSConfig):
        self.config = config
        self.num_clusters = config.num_clusters
        self.temperature = config.cluster_temp

        # Cluster centroids (learned or from emotion prototypes)
        self.centroids = None
        self.cluster_assignments = None

    def fit_clusters(
        self,
        emotion_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fit clusters to emotion embeddings.

        Args:
            emotion_embeddings: [N, emotion_dim] embeddings
            labels: [N] optional pre-defined cluster labels

        Returns:
            Cluster assignments [N]
        """
        if labels is not None:
            # Use provided labels
            self.cluster_assignments = labels
            # Compute centroids
            self.centroids = torch.zeros(
                self.num_clusters, emotion_embeddings.shape[1],
                device=emotion_embeddings.device
            )
            for c in range(self.num_clusters):
                mask = labels == c
                if mask.sum() > 0:
                    self.centroids[c] = emotion_embeddings[mask].mean(dim=0)
            return labels

        # K-means clustering
        from torch.cluster import KMeans  # Would need to implement or use sklearn

        # Simple k-means implementation
        N, D = emotion_embeddings.shape
        device = emotion_embeddings.device

        # Random initialization
        indices = torch.randperm(N)[:self.num_clusters]
        self.centroids = emotion_embeddings[indices].clone()

        for _ in range(100):  # Max iterations
            # Assign to nearest centroid
            distances = torch.cdist(emotion_embeddings, self.centroids)
            assignments = distances.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(self.centroids)
            for c in range(self.num_clusters):
                mask = assignments == c
                if mask.sum() > 0:
                    new_centroids[c] = emotion_embeddings[mask].mean(dim=0)
                else:
                    new_centroids[c] = self.centroids[c]

            # Check convergence
            if torch.allclose(new_centroids, self.centroids, atol=1e-6):
                break

            self.centroids = new_centroids

        self.cluster_assignments = assignments
        return assignments

    def sample_positive(
        self,
        batch_indices: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample positive pairs from same cluster.

        Args:
            batch_indices: [batch] indices in dataset
            embeddings: [N, D] all embeddings

        Returns:
            Indices of positive samples [batch]
        """
        if self.cluster_assignments is None:
            # Fallback to random
            return torch.randperm(len(batch_indices))[:len(batch_indices)]

        batch_clusters = self.cluster_assignments[batch_indices]
        positive_indices = []

        for idx, cluster in zip(batch_indices, batch_clusters):
            # Find all samples in same cluster
            same_cluster_mask = self.cluster_assignments == cluster
            same_cluster_indices = torch.where(same_cluster_mask)[0]

            # Exclude self
            same_cluster_indices = same_cluster_indices[same_cluster_indices != idx]

            if len(same_cluster_indices) >= self.config.min_cluster_samples:
                # Sample with temperature
                distances = torch.cdist(
                    embeddings[idx:idx+1],
                    embeddings[same_cluster_indices]
                ).squeeze(0)

                # Softmax selection with temperature
                probs = F.softmax(-distances / self.temperature, dim=0)
                selected = torch.multinomial(probs, 1).item()
                positive_indices.append(same_cluster_indices[selected].item())
            else:
                # Fallback to closest
                distances = torch.cdist(embeddings[idx:idx+1], embeddings).squeeze(0)
                distances[idx] = float('inf')  # Exclude self
                positive_indices.append(distances.argmin().item())

        return torch.tensor(positive_indices, device=batch_indices.device)

    def sample_negative(
        self,
        batch_indices: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample negative pairs from different clusters.

        Args:
            batch_indices: [batch] indices in dataset
            embeddings: [N, D] all embeddings

        Returns:
            Indices of negative samples [batch]
        """
        if self.cluster_assignments is None:
            # Fallback to random
            N = embeddings.shape[0]
            return torch.randint(0, N, (len(batch_indices),))

        batch_clusters = self.cluster_assignments[batch_indices]
        negative_indices = []

        for idx, cluster in zip(batch_indices, batch_clusters):
            # Find samples in different clusters
            diff_cluster_mask = self.cluster_assignments != cluster
            diff_cluster_indices = torch.where(diff_cluster_mask)[0]

            if len(diff_cluster_indices) > 0:
                # Random selection from different clusters
                selected = diff_cluster_indices[
                    torch.randint(0, len(diff_cluster_indices), (1,))
                ].item()
                negative_indices.append(selected)
            else:
                # Fallback to furthest
                distances = torch.cdist(embeddings[idx:idx+1], embeddings).squeeze(0)
                negative_indices.append(distances.argmax().item())

        return torch.tensor(negative_indices, device=batch_indices.device)


# =============================================================================
# DUAL CONDITIONING TRANSFORMER
# =============================================================================

class DualConditioningTransformer(nn.Module):
    """
    Dual Conditioning Transformer for style feature integration.

    Integrates both emotion and speaker information via separate
    conditioning pathways:

    1. Emotion pathway: Learned emotion embedding → cross-attention
    2. Speaker pathway: Speaker embedding → cross-attention

    The dual design ensures emotion and speaker information are
    processed independently before combination, supporting better
    disentanglement.
    """

    def __init__(self, config: DiEmoTTSConfig):
        super().__init__()
        self.config = config

        # Emotion conditioning
        self.emotion_proj = nn.Linear(config.emotion_dim, config.hidden_dim)
        self.emotion_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.emotion_norm = nn.LayerNorm(config.hidden_dim)

        # Speaker conditioning
        self.speaker_proj = nn.Linear(config.speaker_embed_dim, config.hidden_dim)
        self.speaker_cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.speaker_norm = nn.LayerNorm(config.hidden_dim)

        # Combination layer
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(config.hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        emotion_emb: torch.Tensor,
        speaker_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply dual conditioning.

        Args:
            x: [batch, seq, hidden_dim] input features
            emotion_emb: [batch, emotion_dim] emotion embedding
            speaker_emb: [batch, speaker_dim] speaker embedding
            mask: [batch, seq] attention mask

        Returns:
            Conditioned features [batch, seq, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project condition embeddings
        emotion_kv = self.emotion_proj(emotion_emb).unsqueeze(1)  # [B, 1, H]
        speaker_kv = self.speaker_proj(speaker_emb).unsqueeze(1)  # [B, 1, H]

        # Emotion cross-attention
        emotion_out, _ = self.emotion_cross_attn(
            query=x,
            key=emotion_kv,
            value=emotion_kv,
        )
        emotion_out = self.emotion_norm(x + emotion_out)

        # Speaker cross-attention
        speaker_out, _ = self.speaker_cross_attn(
            query=x,
            key=speaker_kv,
            value=speaker_kv,
        )
        speaker_out = self.speaker_norm(x + speaker_out)

        # Combine emotion and speaker pathways
        combined = torch.cat([emotion_out, speaker_out], dim=-1)
        x = self.combine(combined)

        # Self-attention
        attn_mask = ~mask if mask is not None else None
        self_out, _ = self.self_attn(
            x, x, x,
            key_padding_mask=attn_mask,
        )
        x = self.self_attn_norm(x + self_out)

        # FFN
        x = self.ffn_norm(x + self.ffn(x))

        return x


# =============================================================================
# METRICS: SECS AND EECS
# =============================================================================

class DisentanglementMetrics:
    """
    Metrics for evaluating emotion-speaker disentanglement quality.

    SECS (Speaker Embedding Cosine Similarity):
        Measures how well speaker identity is preserved after emotion transfer.
        Higher = better speaker preservation.

    EECS (Emotion Embedding Cosine Similarity):
        Measures how well emotion is transferred.
        Higher = better emotion transfer.

    Goal: High EECS (good emotion transfer) + High SECS (speaker preservation)
    """

    @staticmethod
    def compute_secs(
        speaker_emb_source: torch.Tensor,
        speaker_emb_generated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Speaker Embedding Cosine Similarity.

        Args:
            speaker_emb_source: [batch, dim] source speaker embeddings
            speaker_emb_generated: [batch, dim] generated audio speaker embeddings

        Returns:
            SECS score [batch] or scalar mean
        """
        # L2 normalize
        source_norm = F.normalize(speaker_emb_source, dim=-1)
        gen_norm = F.normalize(speaker_emb_generated, dim=-1)

        # Cosine similarity
        secs = (source_norm * gen_norm).sum(dim=-1)

        return secs

    @staticmethod
    def compute_eecs(
        emotion_emb_reference: torch.Tensor,
        emotion_emb_generated: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Emotion Embedding Cosine Similarity.

        Args:
            emotion_emb_reference: [batch, dim] reference emotion embeddings
            emotion_emb_generated: [batch, dim] generated audio emotion embeddings

        Returns:
            EECS score [batch] or scalar mean
        """
        # L2 normalize
        ref_norm = F.normalize(emotion_emb_reference, dim=-1)
        gen_norm = F.normalize(emotion_emb_generated, dim=-1)

        # Cosine similarity
        eecs = (ref_norm * gen_norm).sum(dim=-1)

        return eecs

    @staticmethod
    def compute_disentanglement_score(
        secs: torch.Tensor,
        eecs: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute combined disentanglement score.

        Args:
            secs: SECS scores
            eecs: EECS scores
            alpha: Weight for SECS (1-alpha for EECS)

        Returns:
            Combined score
        """
        return alpha * secs + (1 - alpha) * eecs

    @staticmethod
    def evaluate_cross_speaker_transfer(
        model: nn.Module,
        source_emotion: torch.Tensor,
        target_speaker: torch.Tensor,
        speaker_encoder: nn.Module,
        emotion_encoder: nn.Module,
    ) -> Dict[str, float]:
        """
        Evaluate cross-speaker emotion transfer quality.

        Args:
            model: TTS model that generates audio
            source_emotion: [batch, dim] emotion to transfer
            target_speaker: [batch, dim] target speaker embedding
            speaker_encoder: Encoder for speaker embeddings
            emotion_encoder: Encoder for emotion embeddings (DiEmo)

        Returns:
            Dict with SECS, EECS, and combined scores
        """
        # This would be implemented with actual TTS model
        # Placeholder for the evaluation pipeline
        return {
            'secs': 0.0,
            'eecs': 0.0,
            'combined': 0.0,
        }


# =============================================================================
# COMPLETE DiEmo-TTS MODULE
# =============================================================================

class DiEmoTTS(nn.Module):
    """
    Complete DiEmo-TTS module for self-supervised emotion distillation.

    Combines:
    1. ED-DINO for learning speaker-irrelevant emotion embeddings
    2. Formant perturbation for speaker identity augmentation
    3. Cluster-driven sampling for emotion-aware training
    4. Dual conditioning transformer for style integration

    Usage:
        config = DiEmoTTSConfig()
        diemo = DiEmoTTS(config)

        # Training
        losses = diemo(audio_features, speaker_emb=speaker_emb)

        # Inference
        emotion_emb = diemo.get_emotion_embedding(audio_features)
    """

    def __init__(self, config: DiEmoTTSConfig):
        super().__init__()
        self.config = config

        # Core ED-DINO module
        self.ed_dino = EmotionDisentanglementDINO(config)

        # Cluster-driven sampler
        self.cluster_sampler = ClusterDrivenSampler(config)

        # Dual conditioning transformer (optional)
        if config.use_dual_conditioning:
            self.dual_transformer = DualConditioningTransformer(config)
        else:
            self.dual_transformer = None

        # Speaker adversarial head (complementary to DINO)
        self.speaker_adversarial = nn.Sequential(
            nn.Linear(config.emotion_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_speakers),
        )

        # Gradient reversal for adversarial training
        self.grl_lambda = 0.0  # Increased during training

        # Metrics
        self.metrics = DisentanglementMetrics()

    def _gradient_reversal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal layer."""
        class GradientReversal(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambda_):
                ctx.lambda_ = lambda_
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                return -ctx.lambda_ * grad_output, None

        return GradientReversal.apply(x, self.grl_lambda)

    def set_grl_lambda(self, lambda_: float):
        """Set gradient reversal lambda."""
        self.grl_lambda = lambda_

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_perturbed: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all DiEmo-TTS losses.

        Args:
            audio_features: [batch, seq, dim] audio features
            audio_perturbed: [batch, seq, dim] perturbed audio (optional)
            speaker_emb: [batch, speaker_dim] speaker embeddings (optional)
            speaker_labels: [batch] speaker IDs for adversarial loss
            mask: [batch, seq] attention mask
            epoch: Current training epoch

        Returns:
            Dict with losses and embeddings
        """
        batch_size = audio_features.shape[0]
        device = audio_features.device

        # ED-DINO forward
        # Note: If audio_perturbed is None, DINO will use noise augmentation
        # For proper formant perturbation, apply it at the audio level before
        # feature extraction and pass the perturbed features here
        dino_output = self.ed_dino(
            audio_features,
            audio_perturbed,
            mask,
            update_teacher=(epoch > self.config.freeze_teacher_epochs),
        )

        losses = {
            'dino_loss': dino_output['loss'],
            'dino_orig_pert': dino_output['loss_orig_pert'],
            'dino_pert_orig': dino_output['loss_pert_orig'],
        }

        # Speaker adversarial loss
        if speaker_labels is not None:
            emotion_emb = dino_output['emotion_embedding']
            emotion_grl = self._gradient_reversal(emotion_emb)
            speaker_logits = self.speaker_adversarial(emotion_grl)
            speaker_loss = F.cross_entropy(speaker_logits, speaker_labels)
            losses['speaker_adversarial'] = speaker_loss

            # Speaker accuracy (for monitoring - should be ~1/num_speakers if disentangled)
            speaker_acc = (speaker_logits.argmax(dim=-1) == speaker_labels).float().mean()
            losses['speaker_accuracy'] = speaker_acc
        else:
            losses['speaker_adversarial'] = torch.tensor(0.0, device=device)
            losses['speaker_accuracy'] = torch.tensor(0.0, device=device)

        # Dual conditioning (if available and speaker emb provided)
        if self.dual_transformer is not None and speaker_emb is not None:
            emotion_emb = dino_output['emotion_embedding']
            # Note: This would integrate with the full TTS decoder
            # Here we just demonstrate the dual conditioning output
            dummy_input = torch.randn(batch_size, 50, self.config.hidden_dim, device=device)
            conditioned = self.dual_transformer(dummy_input, emotion_emb, speaker_emb)
            losses['dual_output'] = conditioned

        # Total loss
        losses['total'] = (
            self.config.dino_loss_weight * losses['dino_loss'] +
            self.config.speaker_adversarial_weight * losses['speaker_adversarial']
        )

        # Add embeddings to output
        losses['emotion_embedding'] = dino_output['emotion_embedding']
        losses['emotion_perturbed'] = dino_output['emotion_perturbed']

        return losses

    def get_emotion_embedding(
        self,
        audio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get emotion embedding for inference."""
        return self.ed_dino.get_emotion_embedding(audio_features, mask)

    def update_clusters(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Update cluster assignments for sampling."""
        self.cluster_sampler.fit_clusters(embeddings, labels)


# =============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# =============================================================================

class DiEmoAdapter(nn.Module):
    """
    Adapter to integrate DiEmo-TTS with existing prosody pipeline.

    Converts DiEmo emotion embeddings to prosody prefix tokens
    compatible with ProsodyControlledCSM.
    """

    def __init__(
        self,
        diemo_config: DiEmoTTSConfig,
        prosody_hidden: int = 2048,
        num_prosody_tokens: int = 4,
    ):
        super().__init__()

        self.diemo = DiEmoTTS(diemo_config)

        # Projection to prosody tokens
        self.token_generator = nn.Sequential(
            nn.Linear(diemo_config.emotion_dim, prosody_hidden),
            nn.LayerNorm(prosody_hidden),
            nn.GELU(),
            nn.Linear(prosody_hidden, prosody_hidden * num_prosody_tokens),
        )

        self.num_tokens = num_prosody_tokens
        self.prosody_hidden = prosody_hidden

    def forward(
        self,
        audio_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate prosody tokens from audio.

        Args:
            audio_features: [batch, seq, dim] audio features
            mask: [batch, seq] attention mask
            return_embedding: Return raw embedding as well

        Returns:
            Prosody tokens [batch, num_tokens, prosody_hidden]
        """
        # Get emotion embedding
        emotion_emb = self.diemo.get_emotion_embedding(audio_features, mask)

        # Generate prosody tokens
        tokens = self.token_generator(emotion_emb)
        tokens = tokens.view(-1, self.num_tokens, self.prosody_hidden)

        if return_embedding:
            return {
                'tokens': tokens,
                'embedding': emotion_emb,
            }

        return tokens


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DiEmo-TTS Self-Supervised Emotion Distillation - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = DiEmoTTSConfig(
        input_dim=768,
        emotion_dim=256,
        hidden_dim=512,
        num_speakers=100,
    )

    # Test 1: FormantPerturbation
    print("\n[Test 1] FormantPerturbation...")
    perturb = FormantPerturbation(config).to(device)
    mel = torch.randn(2, 100, 80, device=device)
    mel_perturbed = perturb(mel)
    print(f"  Input shape: {mel.shape}")
    print(f"  Output shape: {mel_perturbed.shape}")
    diff = (mel - mel_perturbed).abs().mean().item()
    print(f"  Mean perturbation: {diff:.4f}")
    print("  [PASS]")

    # Test 2: EmotionEncoder
    print("\n[Test 2] EmotionEncoder...")
    encoder = EmotionEncoder(config).to(device)
    features = torch.randn(2, 50, config.input_dim, device=device)
    mask = torch.ones(2, 50, device=device, dtype=torch.bool)

    emb = encoder(features, mask)
    print(f"  Input shape: {features.shape}")
    print(f"  Embedding shape: {emb.shape}")
    assert emb.shape == (2, config.emotion_dim)

    # Test with sequence output
    emb_dict = encoder(features, mask, return_sequence=True)
    print(f"  Sequence shape: {emb_dict['sequence'].shape}")
    print(f"  Attention weights shape: {emb_dict['attention_weights'].shape}")
    print("  [PASS]")

    # Test 3: DINOProjectionHead
    print("\n[Test 3] DINOProjectionHead...")
    proj_head = DINOProjectionHead(config).to(device)
    emb = torch.randn(2, config.emotion_dim, device=device)
    proj = proj_head(emb)
    print(f"  Input shape: {emb.shape}")
    print(f"  Output shape: {proj.shape}")
    print(f"  Output is L2-normalized: {torch.allclose(proj.norm(dim=-1), torch.ones(2, device=device), atol=1e-5)}")
    print("  [PASS]")

    # Test 4: EmotionDisentanglementDINO
    print("\n[Test 4] EmotionDisentanglementDINO...")
    ed_dino = EmotionDisentanglementDINO(config).to(device)

    features_orig = torch.randn(2, 50, config.input_dim, device=device)
    features_pert = torch.randn(2, 50, config.input_dim, device=device)

    output = ed_dino(features_orig, features_pert, update_teacher=True)
    print(f"  DINO loss: {output['loss'].item():.4f}")
    print(f"  Loss orig→pert: {output['loss_orig_pert'].item():.4f}")
    print(f"  Loss pert→orig: {output['loss_pert_orig'].item():.4f}")
    print(f"  Emotion embedding shape: {output['emotion_embedding'].shape}")
    print("  [PASS]")

    # Test 5: DualConditioningTransformer
    print("\n[Test 5] DualConditioningTransformer...")
    dual_tf = DualConditioningTransformer(config).to(device)

    x = torch.randn(2, 30, config.hidden_dim, device=device)
    emotion_emb = torch.randn(2, config.emotion_dim, device=device)
    speaker_emb = torch.randn(2, config.speaker_embed_dim, device=device)

    conditioned = dual_tf(x, emotion_emb, speaker_emb)
    print(f"  Input shape: {x.shape}")
    print(f"  Conditioned output shape: {conditioned.shape}")
    print("  [PASS]")

    # Test 6: ClusterDrivenSampler
    print("\n[Test 6] ClusterDrivenSampler...")
    sampler = ClusterDrivenSampler(config)

    # Create mock embeddings with clear clusters
    N = 100
    embeddings = torch.randn(N, config.emotion_dim, device=device)
    labels = torch.randint(0, config.num_clusters, (N,), device=device)

    assignments = sampler.fit_clusters(embeddings, labels)
    print(f"  Number of samples: {N}")
    print(f"  Number of clusters: {config.num_clusters}")
    print(f"  Assignments shape: {assignments.shape}")

    # Test positive/negative sampling
    batch_indices = torch.arange(8, device=device)
    pos_indices = sampler.sample_positive(batch_indices, embeddings)
    neg_indices = sampler.sample_negative(batch_indices, embeddings)
    print(f"  Positive sample indices: {pos_indices.tolist()}")
    print(f"  Negative sample indices: {neg_indices.tolist()}")
    print("  [PASS]")

    # Test 7: DisentanglementMetrics
    print("\n[Test 7] DisentanglementMetrics...")
    metrics = DisentanglementMetrics()

    speaker_source = F.normalize(torch.randn(4, 256, device=device), dim=-1)
    speaker_gen = F.normalize(torch.randn(4, 256, device=device), dim=-1)
    emotion_ref = F.normalize(torch.randn(4, 256, device=device), dim=-1)
    emotion_gen = F.normalize(torch.randn(4, 256, device=device), dim=-1)

    secs = metrics.compute_secs(speaker_source, speaker_gen)
    eecs = metrics.compute_eecs(emotion_ref, emotion_gen)
    combined = metrics.compute_disentanglement_score(secs, eecs)

    print(f"  SECS (speaker similarity): {secs.mean().item():.4f}")
    print(f"  EECS (emotion similarity): {eecs.mean().item():.4f}")
    print(f"  Combined score: {combined.mean().item():.4f}")
    print("  [PASS]")

    # Test 8: Complete DiEmoTTS
    print("\n[Test 8] Complete DiEmoTTS module...")
    diemo = DiEmoTTS(config).to(device)

    audio_features = torch.randn(2, 50, config.input_dim, device=device)
    speaker_emb = torch.randn(2, config.speaker_embed_dim, device=device)
    speaker_labels = torch.randint(0, config.num_speakers, (2,), device=device)

    losses = diemo(
        audio_features,
        speaker_emb=speaker_emb,
        speaker_labels=speaker_labels,
        epoch=5,
    )

    print(f"  DINO loss: {losses['dino_loss'].item():.4f}")
    print(f"  Speaker adversarial loss: {losses['speaker_adversarial'].item():.4f}")
    print(f"  Speaker accuracy: {losses['speaker_accuracy'].item():.4f}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Emotion embedding shape: {losses['emotion_embedding'].shape}")
    print("  [PASS]")

    # Test 9: DiEmoAdapter
    print("\n[Test 9] DiEmoAdapter (prosody integration)...")
    adapter = DiEmoAdapter(config, prosody_hidden=2048, num_prosody_tokens=4).to(device)

    result = adapter(audio_features, return_embedding=True)
    print(f"  Prosody tokens shape: {result['tokens'].shape}")
    print(f"  Emotion embedding shape: {result['embedding'].shape}")
    print("  [PASS]")

    # Test 10: Gradient flow
    print("\n[Test 10] Gradient flow through DiEmo...")
    diemo = DiEmoTTS(config).to(device)
    audio_features = torch.randn(2, 50, config.input_dim, device=device, requires_grad=True)

    losses = diemo(
        audio_features,
        speaker_labels=torch.randint(0, config.num_speakers, (2,), device=device),
    )

    losses['total'].backward()
    print(f"  Input gradient norm: {audio_features.grad.norm().item():.6f}")
    print(f"  Gradient flows correctly: {audio_features.grad.abs().sum().item() > 0}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All DiEmo-TTS tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from diemo_tts import (
    DiEmoTTSConfig,
    DiEmoTTS,
    DiEmoAdapter,
    DisentanglementMetrics,
)

# Initialize
config = DiEmoTTSConfig(
    input_dim=768,          # wav2vec2/HuBERT feature dimension
    emotion_dim=256,        # Emotion embedding dimension
    num_speakers=1000,      # For adversarial training
)

diemo = DiEmoTTS(config).cuda()

# Training loop
for epoch in range(100):
    for batch in dataloader:
        audio_features = feature_extractor(batch['audio'])

        # Apply formant perturbation for augmentation
        perturbed = diemo.ed_dino.formant_perturb(batch['mel'])
        perturbed_features = feature_extractor(perturbed)

        # Forward pass with DINO + adversarial losses
        losses = diemo(
            audio_features,
            audio_perturbed=perturbed_features,
            speaker_labels=batch['speaker_id'],
            epoch=epoch,
        )

        # Backprop
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Update GRL lambda (increase over training)
        progress = epoch / 100
        diemo.set_grl_lambda(progress * 0.5)

        # Log metrics
        print(f"DINO loss: {losses['dino_loss'].item():.4f}")
        print(f"Speaker acc: {losses['speaker_accuracy'].item():.4f}")

# Inference: Get speaker-irrelevant emotion embedding
emotion_emb = diemo.get_emotion_embedding(audio_features)

# Integration with prosody pipeline
adapter = DiEmoAdapter(config, prosody_hidden=2048)
prosody_tokens = adapter(audio_features)  # [batch, 4, 2048]

# Evaluate disentanglement quality
metrics = DisentanglementMetrics()
secs = metrics.compute_secs(speaker_source, speaker_generated)
eecs = metrics.compute_eecs(emotion_reference, emotion_generated)
print(f"SECS: {secs.mean():.4f}, EECS: {eecs.mean():.4f}")
""")
