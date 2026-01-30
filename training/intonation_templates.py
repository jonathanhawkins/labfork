"""
Intonation Template Clustering (Into-TTS Approach)

Based on "Into-TTS: Intonation Template Based Prosody Control System" (Samsung Research, 2022)
https://arxiv.org/abs/2204.01271

Key Innovation: Discover discrete intonation patterns through unsupervised clustering of F0 contours.
Users can then explicitly select intonation templates at inference time for controllable prosody.

How it works:
1. **F0 Extraction**: Extract sentence-final F0 contours from training data
2. **Template Discovery**: Cluster F0 contours using k-means or GMM to find natural patterns
3. **Intonation Encoder**: Maps template index → embedding for conditioning the TTS model
4. **Intonation Predictor**: Optionally predict suitable template from text (automatic mode)

Benefits:
- No additional annotation needed - templates are discovered from data
- Explicit control: select template index at inference for desired intonation
- Interpretable: discovered templates often correspond to questions, statements, emphasis
- Compatible with existing prosody pipeline and DrawSpeech sketch interface

Example discovered templates:
- Template 0: Falling (statement)
- Template 1: Rising (question)
- Template 2: Rise-fall (emphasis)
- Template 3: Flat (neutral)
- Template 4: High-flat (exclamation)

References:
- Into-TTS: https://arxiv.org/abs/2204.01271
- ProsodyFM: https://arxiv.org/html/2412.11795v1
"""

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class IntonationTemplateConfig:
    """Configuration for intonation template clustering."""

    # F0 extraction settings
    f0_min: float = 50.0  # Minimum F0 (Hz)
    f0_max: float = 800.0  # Maximum F0 (Hz)
    hop_length_ms: float = 10.0  # Frame hop length in ms
    sample_rate: int = 24000  # Audio sample rate

    # Contour settings
    contour_length: int = 50  # Fixed length for normalized contours
    use_sentence_final: bool = True  # Use only final portion of utterance
    final_portion: float = 0.3  # Use last 30% for sentence-final contour
    log_f0: bool = True  # Use log-scale F0
    normalize_speaker: bool = True  # Speaker-normalize F0

    # Clustering settings
    num_templates: int = 8  # Number of intonation templates
    clustering_method: str = "kmeans"  # "kmeans", "gmm", "minibatch_kmeans"
    kmeans_init: str = "k-means++"  # k-means initialization
    gmm_covariance: str = "full"  # GMM covariance type
    min_samples_per_template: int = 10  # Minimum samples to keep template

    # Encoder settings
    template_embed_dim: int = 256  # Template embedding dimension
    hidden_dim: int = 512  # Hidden layer dimension
    num_encoder_layers: int = 2  # Number of encoder layers
    dropout: float = 0.1

    # Predictor settings
    use_predictor: bool = True  # Whether to use text-to-template predictor
    predictor_hidden_dim: int = 256
    predictor_num_layers: int = 2
    text_embed_dim: int = 256  # Input text embedding dimension

    # Output settings
    output_dim: int = 2048  # Output dimension for prosody pipeline
    num_prosody_tokens: int = 4  # Number of prosody prefix tokens

    # Training
    reconstruction_weight: float = 1.0  # Weight for contour reconstruction
    classification_weight: float = 1.0  # Weight for template classification
    kl_weight: float = 0.1  # Weight for KL divergence (soft assignment)

    # Paths
    template_cache_path: str = "checkpoints/intonation_templates"


# =============================================================================
# F0 EXTRACTION AND PREPROCESSING
# =============================================================================

class F0Extractor:
    """
    Extract and preprocess F0 contours for template clustering.

    Supports multiple F0 extraction backends:
    - parselmouth (Praat)
    - librosa (pyin)
    - torchaudio (if available)
    """

    def __init__(self, config: IntonationTemplateConfig):
        self.config = config

        # Try to import parselmouth
        try:
            import parselmouth
            self.backend = "parselmouth"
        except ImportError:
            try:
                import librosa
                self.backend = "librosa"
            except ImportError:
                self.backend = "basic"
                print("Warning: Neither parselmouth nor librosa available. Using basic F0 extraction.")

    def extract_f0(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 contour from audio.

        Args:
            audio: Audio waveform [samples] or [1, samples]
            sample_rate: Sample rate (default: config.sample_rate)

        Returns:
            (times, f0_values) - Time points and F0 values in Hz
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate

        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        if self.backend == "parselmouth":
            return self._extract_parselmouth(audio, sample_rate)
        elif self.backend == "librosa":
            return self._extract_librosa(audio, sample_rate)
        else:
            return self._extract_basic(audio, sample_rate)

    def _extract_parselmouth(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using Parselmouth (Praat)."""
        import parselmouth

        sound = parselmouth.Sound(audio, sample_rate)
        pitch = sound.to_pitch(
            time_step=self.config.hop_length_ms / 1000.0,
            pitch_floor=self.config.f0_min,
            pitch_ceiling=self.config.f0_max,
        )

        times = pitch.xs()
        f0_values = np.array([
            pitch.get_value_at_time(t) if pitch.get_value_at_time(t) else 0.0
            for t in times
        ])

        return times, f0_values

    def _extract_librosa(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using librosa pyin."""
        import librosa

        hop_length = int(sample_rate * self.config.hop_length_ms / 1000.0)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.config.f0_min,
            fmax=self.config.f0_max,
            sr=sample_rate,
            hop_length=hop_length,
        )

        times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)

        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)

        return times, f0

    def _extract_basic(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Basic F0 extraction using autocorrelation."""
        hop_length = int(sample_rate * self.config.hop_length_ms / 1000.0)
        frame_length = hop_length * 4

        num_frames = (len(audio) - frame_length) // hop_length + 1

        f0_values = []
        times = []

        for i in range(num_frames):
            start = i * hop_length
            frame = audio[start:start + frame_length]

            # Simple autocorrelation-based F0
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            # Find peaks
            min_lag = int(sample_rate / self.config.f0_max)
            max_lag = int(sample_rate / self.config.f0_min)

            if max_lag < len(corr):
                search_region = corr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    if corr[peak_idx] > 0.3 * corr[0]:  # Voicing threshold
                        f0 = sample_rate / peak_idx
                    else:
                        f0 = 0.0
                else:
                    f0 = 0.0
            else:
                f0 = 0.0

            f0_values.append(f0)
            times.append(start / sample_rate)

        return np.array(times), np.array(f0_values)

    def normalize_contour(
        self,
        f0: np.ndarray,
        times: Optional[np.ndarray] = None,
        speaker_mean: Optional[float] = None,
        speaker_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalize F0 contour for clustering.

        Steps:
        1. Extract sentence-final portion (if configured)
        2. Interpolate unvoiced regions
        3. Log transform
        4. Speaker normalization
        5. Resample to fixed length

        Args:
            f0: Raw F0 values [frames]
            times: Time values [frames]
            speaker_mean: Speaker F0 mean (for normalization)
            speaker_std: Speaker F0 std (for normalization)

        Returns:
            Normalized contour [contour_length]
        """
        f0 = f0.copy()

        # Extract sentence-final portion
        if self.config.use_sentence_final:
            start_idx = int(len(f0) * (1 - self.config.final_portion))
            f0 = f0[start_idx:]

        # Handle empty or all-unvoiced
        voiced_mask = f0 > 0
        if not voiced_mask.any():
            return np.zeros(self.config.contour_length)

        # Interpolate unvoiced regions
        voiced_indices = np.where(voiced_mask)[0]
        voiced_values = f0[voiced_indices]

        all_indices = np.arange(len(f0))
        f0_interp = np.interp(all_indices, voiced_indices, voiced_values)

        # Log transform
        if self.config.log_f0:
            f0_interp = np.log(f0_interp + 1e-8)

        # Speaker normalization
        if self.config.normalize_speaker and speaker_mean is not None:
            f0_interp = (f0_interp - speaker_mean) / (speaker_std + 1e-8)
        else:
            # Self-normalize
            f0_mean = f0_interp.mean()
            f0_std = f0_interp.std() + 1e-8
            f0_interp = (f0_interp - f0_mean) / f0_std

        # Resample to fixed length
        if len(f0_interp) != self.config.contour_length:
            x_old = np.linspace(0, 1, len(f0_interp))
            x_new = np.linspace(0, 1, self.config.contour_length)
            f0_interp = np.interp(x_new, x_old, f0_interp)

        return f0_interp

    def process_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        sample_rate: Optional[int] = None,
        speaker_mean: Optional[float] = None,
        speaker_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Full pipeline: audio → normalized contour.

        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            speaker_mean: Speaker F0 mean
            speaker_std: Speaker F0 std

        Returns:
            Normalized F0 contour [contour_length]
        """
        times, f0 = self.extract_f0(audio, sample_rate)
        return self.normalize_contour(f0, times, speaker_mean, speaker_std)


# =============================================================================
# TEMPLATE CLUSTERING
# =============================================================================

class IntonationTemplateClustering:
    """
    Discover intonation templates through unsupervised clustering.

    Supports:
    - K-means clustering (default)
    - Gaussian Mixture Models (soft assignment)
    - Mini-batch K-means (for large datasets)
    """

    def __init__(self, config: IntonationTemplateConfig):
        self.config = config
        self.clusterer = None
        self.scaler = None
        self.templates = None  # [num_templates, contour_length]
        self.template_counts = None

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is required for clustering. Install with: pip install scikit-learn"
            )

    def fit(
        self,
        contours: np.ndarray,  # [num_samples, contour_length]
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Fit clustering model to discover templates.

        Args:
            contours: Normalized F0 contours [num_samples, contour_length]
            verbose: Print progress

        Returns:
            Dict with templates, assignments, and metrics
        """
        if verbose:
            print(f"Clustering {len(contours)} contours into {self.config.num_templates} templates...")

        # Standardize features
        self.scaler = StandardScaler()
        contours_scaled = self.scaler.fit_transform(contours)

        # Fit clustering model
        if self.config.clustering_method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.config.num_templates,
                init=self.config.kmeans_init,
                n_init=10,
                random_state=42,
            )
            labels = self.clusterer.fit_predict(contours_scaled)

        elif self.config.clustering_method == "minibatch_kmeans":
            self.clusterer = MiniBatchKMeans(
                n_clusters=self.config.num_templates,
                init=self.config.kmeans_init,
                batch_size=256,
                n_init=10,
                random_state=42,
            )
            labels = self.clusterer.fit_predict(contours_scaled)

        elif self.config.clustering_method == "gmm":
            self.clusterer = GaussianMixture(
                n_components=self.config.num_templates,
                covariance_type=self.config.gmm_covariance,
                n_init=5,
                random_state=42,
            )
            self.clusterer.fit(contours_scaled)
            labels = self.clusterer.predict(contours_scaled)

        else:
            raise ValueError(f"Unknown clustering method: {self.config.clustering_method}")

        # Compute template centroids (in original space)
        self.templates = np.zeros((self.config.num_templates, self.config.contour_length))
        self.template_counts = np.zeros(self.config.num_templates, dtype=int)

        for i in range(self.config.num_templates):
            mask = labels == i
            if mask.any():
                self.templates[i] = contours[mask].mean(axis=0)
                self.template_counts[i] = mask.sum()

        # Compute metrics
        inertia = getattr(self.clusterer, 'inertia_', None)
        if inertia is None and hasattr(self.clusterer, 'score'):
            inertia = -self.clusterer.score(contours_scaled)

        result = {
            'templates': self.templates,
            'labels': labels,
            'counts': self.template_counts,
            'inertia': inertia,
        }

        if verbose:
            print(f"  Templates discovered. Sample counts: {self.template_counts.tolist()}")
            print(f"  Inertia: {inertia:.4f}" if inertia else "")

        return result

    def predict(
        self,
        contours: np.ndarray,  # [num_samples, contour_length]
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict template assignments for new contours.

        Args:
            contours: Normalized contours [num_samples, contour_length]
            return_probs: Return soft assignment probabilities

        Returns:
            Template indices [num_samples] or (indices, probs)
        """
        if self.clusterer is None:
            raise RuntimeError("Clustering model not fitted. Call fit() first.")

        contours_scaled = self.scaler.transform(contours)
        labels = self.clusterer.predict(contours_scaled)

        if return_probs:
            if hasattr(self.clusterer, 'predict_proba'):
                probs = self.clusterer.predict_proba(contours_scaled)
            else:
                # For k-means, compute soft assignment based on distances
                distances = self.clusterer.transform(contours_scaled)
                probs = F.softmax(torch.from_numpy(-distances), dim=-1).numpy()
            return labels, probs

        return labels

    def get_template(self, index: int) -> np.ndarray:
        """Get template centroid by index."""
        if self.templates is None:
            raise RuntimeError("No templates available. Call fit() first.")
        return self.templates[index]

    def save(self, path: Union[str, Path]) -> None:
        """Save clustering model and templates."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'config': self.config,
            'clusterer': self.clusterer,
            'scaler': self.scaler,
            'templates': self.templates,
            'template_counts': self.template_counts,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: Union[str, Path]) -> None:
        """Load clustering model and templates."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.clusterer = state['clusterer']
        self.scaler = state['scaler']
        self.templates = state['templates']
        self.template_counts = state['template_counts']

    def describe_templates(self) -> List[str]:
        """
        Generate human-readable descriptions of discovered templates.

        Analyzes template shape to assign descriptive labels.
        """
        if self.templates is None:
            return []

        descriptions = []

        for i, template in enumerate(self.templates):
            # Analyze contour shape
            start_val = template[:5].mean()
            end_val = template[-5:].mean()
            mid_val = template[len(template)//2 - 2:len(template)//2 + 2].mean()

            trend = end_val - start_val
            mid_peak = mid_val - max(start_val, end_val)

            # Classify pattern
            if abs(trend) < 0.3:
                if mid_peak > 0.3:
                    pattern = "rise-fall (emphasis)"
                elif mid_peak < -0.3:
                    pattern = "fall-rise (uncertainty)"
                else:
                    if abs(start_val) < 0.3:
                        pattern = "flat (neutral)"
                    else:
                        pattern = f"flat-{'high' if start_val > 0 else 'low'}"
            elif trend > 0:
                if trend > 0.8:
                    pattern = "strong-rising (question)"
                else:
                    pattern = "rising (continuation)"
            else:
                if trend < -0.8:
                    pattern = "strong-falling (statement)"
                else:
                    pattern = "falling (finality)"

            count = self.template_counts[i] if self.template_counts is not None else 0
            descriptions.append(f"Template {i}: {pattern} (n={count})")

        return descriptions


# =============================================================================
# INTONATION ENCODER
# =============================================================================

class IntonationEncoder(nn.Module):
    """
    Encode intonation template index into embedding.

    Can operate in two modes:
    1. Hard assignment: Lookup embedding by template index
    2. Soft assignment: Weighted combination of embeddings by probabilities

    The embedding is then used to condition the TTS model.
    """

    def __init__(self, config: IntonationTemplateConfig):
        super().__init__()
        self.config = config

        # Template embeddings (learned)
        self.template_embeddings = nn.Embedding(
            config.num_templates,
            config.template_embed_dim,
        )

        # Initialize embeddings with small variance
        nn.init.normal_(self.template_embeddings.weight, std=0.02)

        # Encoder network (processes embedding for output)
        layers = []
        in_dim = config.template_embed_dim

        for i in range(config.num_encoder_layers):
            out_dim = config.hidden_dim if i < config.num_encoder_layers - 1 else config.output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Project to prosody tokens
        self.token_projection = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
            nn.LayerNorm(config.output_dim * config.num_prosody_tokens),
        )

    def forward(
        self,
        template_indices: Optional[torch.Tensor] = None,  # [batch]
        template_probs: Optional[torch.Tensor] = None,    # [batch, num_templates]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode template(s) to embeddings.

        Args:
            template_indices: Hard template assignments [batch]
            template_probs: Soft template assignments [batch, num_templates]

        Returns:
            Dict with:
            - embedding: [batch, template_embed_dim]
            - output: [batch, output_dim]
            - prosody_tokens: [batch, num_tokens, output_dim]
        """
        if template_indices is not None:
            # Hard assignment
            embedding = self.template_embeddings(template_indices)
        elif template_probs is not None:
            # Soft assignment: weighted combination
            all_embeddings = self.template_embeddings.weight  # [num_templates, embed_dim]
            embedding = torch.matmul(template_probs, all_embeddings)  # [batch, embed_dim]
        else:
            raise ValueError("Either template_indices or template_probs must be provided")

        # Encode
        output = self.encoder(embedding)  # [batch, output_dim]

        # Project to tokens
        tokens = self.token_projection(output)  # [batch, output_dim * num_tokens]
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.output_dim)

        return {
            'embedding': embedding,
            'output': output,
            'prosody_tokens': tokens,
        }

    def get_all_embeddings(self) -> torch.Tensor:
        """Get all template embeddings."""
        return self.template_embeddings.weight


# =============================================================================
# INTONATION PREDICTOR
# =============================================================================

class IntonationPredictor(nn.Module):
    """
    Predict suitable intonation template from text input.

    This enables automatic template selection when the user doesn't specify one.
    The predictor learns to map text features to template distributions.

    Input: Text embeddings (from phoneme encoder or text encoder)
    Output: Template probabilities (soft) or template index (hard)
    """

    def __init__(self, config: IntonationTemplateConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.text_embed_dim, config.predictor_hidden_dim)

        # Temporal processing (for sequence input)
        self.temporal_conv = nn.Conv1d(
            config.predictor_hidden_dim,
            config.predictor_hidden_dim,
            kernel_size=5,
            padding=2,
        )

        # Self-attention for global context
        self.self_attn = nn.MultiheadAttention(
            config.predictor_hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(config.predictor_hidden_dim)

        # Pooling query (learnable)
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.predictor_hidden_dim))

        # MLP layers
        layers = []
        in_dim = config.predictor_hidden_dim

        for i in range(config.predictor_num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, config.predictor_hidden_dim),
                nn.LayerNorm(config.predictor_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = config.predictor_hidden_dim

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # Output head
        self.output_head = nn.Linear(config.predictor_hidden_dim, config.num_templates)

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [batch, seq_len, text_dim]
        text_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict template distribution from text.

        Args:
            text_embeddings: Text/phoneme embeddings [batch, seq_len, text_dim]
            text_mask: Attention mask [batch, seq_len]
            temperature: Softmax temperature (lower = more peaked)

        Returns:
            Dict with:
            - logits: Raw logits [batch, num_templates]
            - probs: Template probabilities [batch, num_templates]
            - predicted: Predicted template index [batch]
        """
        batch_size = text_embeddings.shape[0]

        # Project input
        x = self.input_proj(text_embeddings)  # [batch, seq, hidden]

        # Temporal convolution
        x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        # Self-attention with pooling
        x_norm = self.attn_norm(x)
        query = self.pool_query.expand(batch_size, -1, -1)

        key_padding_mask = ~text_mask if text_mask is not None else None

        pooled, _ = self.self_attn(
            query, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
        )
        pooled = pooled.squeeze(1)  # [batch, hidden]

        # MLP
        pooled = self.mlp(pooled)

        # Output
        logits = self.output_head(pooled)  # [batch, num_templates]
        probs = F.softmax(logits / temperature, dim=-1)
        predicted = probs.argmax(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predicted': predicted,
        }


# =============================================================================
# CONTOUR DECODER (For Training)
# =============================================================================

class ContourDecoder(nn.Module):
    """
    Decode template embedding back to F0 contour.

    Used during training to ensure templates are learning meaningful patterns.
    The reconstruction loss encourages the encoder to preserve contour information.
    """

    def __init__(self, config: IntonationTemplateConfig):
        super().__init__()
        self.config = config

        # Decode from template embedding to contour
        self.decoder = nn.Sequential(
            nn.Linear(config.template_embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.contour_length),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding to contour.

        Args:
            embedding: Template embedding [batch, template_embed_dim]

        Returns:
            Reconstructed contour [batch, contour_length]
        """
        return self.decoder(embedding)


# =============================================================================
# COMPLETE INTONATION TEMPLATE MODULE
# =============================================================================

class IntonationTemplateModule(nn.Module):
    """
    Complete intonation template system combining:
    - Intonation encoder (template → embedding → prosody tokens)
    - Intonation predictor (text → template distribution)
    - Contour decoder (for training)

    Usage:
        module = IntonationTemplateModule(config)

        # Training: from known template
        result = module(template_indices=template_ids)

        # Inference (manual): specify template
        result = module(template_indices=torch.tensor([2]))  # Use template 2

        # Inference (auto): predict from text
        result = module(text_embeddings=text_emb)
    """

    def __init__(self, config: IntonationTemplateConfig):
        super().__init__()
        self.config = config

        # Components
        self.encoder = IntonationEncoder(config)
        self.decoder = ContourDecoder(config)

        if config.use_predictor:
            self.predictor = IntonationPredictor(config)
        else:
            self.predictor = None

        # Template centroids (loaded from clustering)
        self.register_buffer(
            'template_centroids',
            torch.zeros(config.num_templates, config.contour_length)
        )

    def load_templates(self, clustering: IntonationTemplateClustering) -> None:
        """Load template centroids from clustering model."""
        self.template_centroids = torch.from_numpy(
            clustering.templates
        ).float().to(self.template_centroids.device)

    def forward(
        self,
        template_indices: Optional[torch.Tensor] = None,  # [batch]
        template_probs: Optional[torch.Tensor] = None,    # [batch, num_templates]
        text_embeddings: Optional[torch.Tensor] = None,   # [batch, seq, dim]
        text_mask: Optional[torch.Tensor] = None,
        return_contour: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the module.

        Modes:
        1. Manual template selection: provide template_indices
        2. Soft template selection: provide template_probs
        3. Automatic prediction: provide text_embeddings

        Args:
            template_indices: Hard template indices [batch]
            template_probs: Soft template probabilities [batch, num_templates]
            text_embeddings: Text embeddings for prediction [batch, seq, dim]
            text_mask: Text attention mask [batch, seq]
            return_contour: Whether to decode contour (for training)

        Returns:
            Dict with embeddings, tokens, and optionally reconstructed contour
        """
        # Determine template assignment
        if template_indices is not None or template_probs is not None:
            # Use provided templates
            pass
        elif text_embeddings is not None and self.predictor is not None:
            # Predict from text
            pred_result = self.predictor(text_embeddings, text_mask)
            template_probs = pred_result['probs']
        else:
            raise ValueError(
                "Must provide template_indices, template_probs, or text_embeddings"
            )

        # Encode template(s)
        enc_result = self.encoder(template_indices, template_probs)

        result = {
            'embedding': enc_result['embedding'],
            'output': enc_result['output'],
            'prosody_tokens': enc_result['prosody_tokens'],
        }

        # Add prediction info if used
        if text_embeddings is not None and self.predictor is not None:
            result['predicted_probs'] = template_probs
            result['predicted_template'] = template_probs.argmax(dim=-1)

        # Decode contour for training
        if return_contour:
            result['reconstructed_contour'] = self.decoder(enc_result['embedding'])

        return result

    def get_template_tokens(
        self,
        template_index: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Get prosody tokens for a specific template.

        Convenience method for inference.

        Args:
            template_index: Template index
            batch_size: Batch size

        Returns:
            Prosody tokens [batch, num_tokens, output_dim]
        """
        device = self.template_centroids.device
        indices = torch.full((batch_size,), template_index, device=device, dtype=torch.long)
        result = self(template_indices=indices)
        return result['prosody_tokens']


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class IntonationTemplateLoss(nn.Module):
    """
    Loss function for training the intonation template system.

    Components:
    1. Classification loss: Template prediction accuracy
    2. Reconstruction loss: Contour reconstruction quality
    3. KL divergence: Encourage soft assignments (optional)
    """

    def __init__(self, config: IntonationTemplateConfig):
        super().__init__()
        self.config = config

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        module_output: Dict[str, torch.Tensor],
        target_indices: Optional[torch.Tensor] = None,  # [batch]
        target_contours: Optional[torch.Tensor] = None,  # [batch, contour_length]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            module_output: Output from IntonationTemplateModule
            target_indices: Ground truth template indices
            target_contours: Ground truth F0 contours

        Returns:
            Dict with loss values
        """
        losses = {}
        total_loss = 0.0

        # Classification loss (if predictor output available)
        if 'predicted_probs' in module_output and target_indices is not None:
            # We need logits for CE loss; recompute from probs
            probs = module_output['predicted_probs']
            # Approximate logits (add small epsilon to avoid log(0))
            logits = torch.log(probs + 1e-10)

            cls_loss = self.ce_loss(logits, target_indices)
            losses['classification_loss'] = cls_loss
            total_loss = total_loss + self.config.classification_weight * cls_loss

            # Accuracy
            with torch.no_grad():
                predicted = probs.argmax(dim=-1)
                accuracy = (predicted == target_indices).float().mean()
                losses['accuracy'] = accuracy

        # Reconstruction loss
        if 'reconstructed_contour' in module_output and target_contours is not None:
            recon_loss = self.mse_loss(
                module_output['reconstructed_contour'],
                target_contours
            )
            losses['reconstruction_loss'] = recon_loss
            total_loss = total_loss + self.config.reconstruction_weight * recon_loss

        # KL divergence for soft assignment (encourage peaky distributions)
        if 'predicted_probs' in module_output:
            probs = module_output['predicted_probs']
            # Target: uniform distribution (encourage exploration)
            # Or: encourage entropy (avoid collapse)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            kl_loss = -entropy  # Minimize negative entropy = maximize entropy
            losses['kl_loss'] = kl_loss
            total_loss = total_loss + self.config.kl_weight * kl_loss

        losses['total'] = total_loss
        return losses


# =============================================================================
# ADAPTER FOR PROSODY PIPELINE
# =============================================================================

class IntonationTemplateAdapter(nn.Module):
    """
    Adapter for integrating intonation templates with existing prosody pipeline.

    Provides compatibility with:
    - ProsodyControlledCSM
    - DrawSpeech sketch interface
    - Other prosody conditioning modules

    Usage:
        adapter = IntonationTemplateAdapter(config)

        # Load pre-trained clustering and encoder
        adapter.load_pretrained("checkpoints/intonation_templates/model.pt")

        # Manual template selection
        tokens = adapter.from_template(template_index=2)

        # Automatic selection from text
        tokens = adapter.from_text(text_embeddings)

        # Integration with sketch interface
        sketch_info = adapter.template_to_sketch(template_index=2)
    """

    def __init__(
        self,
        config: IntonationTemplateConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core module
        self.module = IntonationTemplateModule(config)

        # Clustering model (loaded separately)
        self.clustering: Optional[IntonationTemplateClustering] = None

        # Output adapter
        if config.output_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def load_clustering(self, path: Union[str, Path]) -> None:
        """Load clustering model."""
        self.clustering = IntonationTemplateClustering(self.config)
        self.clustering.load(path)
        self.module.load_templates(self.clustering)

    def load_pretrained(self, path: Union[str, Path]) -> None:
        """Load complete pretrained adapter (module + clustering)."""
        path = Path(path)

        # Load module weights
        module_path = path / "module.pt" if path.is_dir() else path
        if module_path.exists():
            state_dict = torch.load(module_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

        # Load clustering
        cluster_path = path / "clustering.pkl" if path.is_dir() else path.with_suffix('.pkl')
        if cluster_path.exists():
            self.load_clustering(cluster_path)

    def from_template(
        self,
        template_index: Union[int, torch.Tensor],
        batch_size: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate prosody tokens from template index.

        Args:
            template_index: Template index (int or tensor)
            batch_size: Batch size if template_index is int

        Returns:
            Dict with prosody_tokens and template info
        """
        device = next(self.parameters()).device

        if isinstance(template_index, int):
            indices = torch.full((batch_size,), template_index, device=device, dtype=torch.long)
        else:
            indices = template_index.to(device)

        result = self.module(template_indices=indices, return_contour=True)

        # Adapt output
        tokens = self.output_adapter(result['prosody_tokens'])

        return {
            'prosody_tokens': tokens,
            'template_index': indices,
            'template_contour': result.get('reconstructed_contour'),
        }

    def from_text(
        self,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict template and generate prosody tokens from text.

        Args:
            text_embeddings: Text embeddings [batch, seq, dim]
            text_mask: Attention mask
            temperature: Prediction temperature

        Returns:
            Dict with prosody_tokens, predicted template, and probabilities
        """
        if self.module.predictor is None:
            raise RuntimeError("Predictor not available. Initialize with use_predictor=True")

        # Predict template
        pred_result = self.module.predictor(text_embeddings, text_mask, temperature)

        # Encode predicted template
        result = self.module(
            template_probs=pred_result['probs'],
            return_contour=True,
        )

        # Adapt output
        tokens = self.output_adapter(result['prosody_tokens'])

        return {
            'prosody_tokens': tokens,
            'predicted_template': pred_result['predicted'],
            'template_probs': pred_result['probs'],
            'template_contour': result.get('reconstructed_contour'),
        }

    def from_contour(
        self,
        f0_contour: torch.Tensor,  # [batch, contour_length]
    ) -> Dict[str, torch.Tensor]:
        """
        Find best matching template for given contour.

        Args:
            f0_contour: Normalized F0 contour [batch, contour_length]

        Returns:
            Dict with prosody_tokens and matched template
        """
        if self.clustering is None:
            raise RuntimeError("Clustering not loaded. Call load_clustering() first.")

        # Find matching template
        contour_np = f0_contour.cpu().numpy()
        labels, probs = self.clustering.predict(contour_np, return_probs=True)

        # Convert to tensor
        probs = torch.from_numpy(probs).float().to(f0_contour.device)

        # Encode
        result = self.module(template_probs=probs, return_contour=True)

        # Adapt output
        tokens = self.output_adapter(result['prosody_tokens'])

        return {
            'prosody_tokens': tokens,
            'matched_template': torch.from_numpy(labels).to(f0_contour.device),
            'template_probs': probs,
        }

    def template_to_sketch(
        self,
        template_index: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert template to DrawSpeech-compatible sketch curves.

        This enables integration with the sketch-based prosody interface.

        Args:
            template_index: Template index

        Returns:
            Dict with pitch_sketch, energy_sketch for DrawSpeech
        """
        if self.clustering is None:
            raise RuntimeError("Clustering not loaded. Call load_clustering() first.")

        # Get template contour
        template_contour = self.clustering.get_template(template_index)

        # Normalize to [0, 1] for sketch
        template_min = template_contour.min()
        template_max = template_contour.max()
        template_range = template_max - template_min + 1e-8

        pitch_sketch = (template_contour - template_min) / template_range

        # Energy sketch: derive from pitch (simplified)
        # In practice, could use separate energy templates
        energy_sketch = 0.5 + 0.2 * pitch_sketch  # Slight correlation with pitch

        return {
            'pitch_sketch': torch.from_numpy(pitch_sketch).float(),
            'energy_sketch': torch.from_numpy(energy_sketch).float(),
            'template_index': template_index,
        }

    def get_template_descriptions(self) -> List[str]:
        """Get human-readable descriptions of templates."""
        if self.clustering is None:
            return [f"Template {i}" for i in range(self.config.num_templates)]
        return self.clustering.describe_templates()

    def compute_loss(
        self,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        target_indices: torch.Tensor,
        target_contours: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            text_embeddings: Text embeddings [batch, seq, dim]
            text_mask: Attention mask
            target_indices: Ground truth template indices [batch]
            target_contours: Ground truth contours [batch, contour_length]

        Returns:
            Dict with loss values
        """
        result = self.module(
            text_embeddings=text_embeddings,
            text_mask=text_mask,
            return_contour=True,
        )

        loss_fn = IntonationTemplateLoss(self.config)
        return loss_fn(result, target_indices, target_contours)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_templates_from_dataset(
    audio_paths: List[str],
    config: IntonationTemplateConfig,
    speaker_info: Optional[Dict[str, Dict]] = None,
    verbose: bool = True,
) -> IntonationTemplateClustering:
    """
    Extract F0 contours from dataset and cluster into templates.

    Args:
        audio_paths: List of audio file paths
        config: Configuration
        speaker_info: Optional speaker normalization info
        verbose: Print progress

    Returns:
        Fitted IntonationTemplateClustering
    """
    import torchaudio

    extractor = F0Extractor(config)
    contours = []

    if verbose:
        print(f"Extracting F0 contours from {len(audio_paths)} files...")

    for i, path in enumerate(audio_paths):
        try:
            audio, sr = torchaudio.load(path)

            # Resample if needed
            if sr != config.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, config.sample_rate)

            # Get speaker info if available
            speaker_mean = speaker_std = None
            if speaker_info:
                # Extract speaker ID from path (customize as needed)
                speaker_id = Path(path).parent.name
                if speaker_id in speaker_info:
                    speaker_mean = speaker_info[speaker_id].get('f0_mean')
                    speaker_std = speaker_info[speaker_id].get('f0_std')

            # Process
            contour = extractor.process_audio(
                audio, config.sample_rate,
                speaker_mean, speaker_std
            )

            # Skip if invalid
            if np.abs(contour).sum() > 0:
                contours.append(contour)

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to process {path}: {e}")

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(audio_paths)} files...")

    contours = np.array(contours)

    if verbose:
        print(f"  Extracted {len(contours)} valid contours")

    # Cluster
    clustering = IntonationTemplateClustering(config)
    clustering.fit(contours, verbose=verbose)

    return clustering


def visualize_templates(
    clustering: IntonationTemplateClustering,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize discovered intonation templates.

    Args:
        clustering: Fitted clustering model
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    templates = clustering.templates
    descriptions = clustering.describe_templates()

    n_templates = len(templates)
    cols = 4
    rows = (n_templates + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten() if n_templates > 1 else [axes]

    for i, (template, desc) in enumerate(zip(templates, descriptions)):
        ax = axes[i]
        ax.plot(template, linewidth=2)
        ax.set_title(desc.split('(')[0].strip(), fontsize=10)
        ax.set_xlabel('Time (normalized)')
        ax.set_ylabel('F0 (normalized)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Hide empty subplots
    for i in range(n_templates, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved template visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Intonation Template Clustering (Into-TTS Approach) - Test Suite")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    config = IntonationTemplateConfig()

    # Test 1: F0 Extractor
    print("\n[Test 1] F0 Extractor...")
    extractor = F0Extractor(config)

    # Generate synthetic audio
    sr = config.sample_rate
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    f0_target = 150 + 50 * np.sin(2 * np.pi * t)  # Varying F0
    audio = np.sin(2 * np.pi * np.cumsum(f0_target) / sr)  # Simple FM synthesis

    times, f0 = extractor.extract_f0(audio, sr)
    print(f"  F0 extracted: {len(f0)} frames")

    contour = extractor.normalize_contour(f0, times)
    print(f"  Normalized contour shape: {contour.shape}")
    print("  [PASS]")

    # Test 2: Template Clustering
    print("\n[Test 2] Template Clustering...")

    # Generate synthetic contours for different patterns
    np.random.seed(42)
    n_samples = 200
    contour_length = config.contour_length

    patterns = []
    # Rising (questions)
    for _ in range(n_samples // 4):
        t = np.linspace(0, 1, contour_length)
        patterns.append(t + np.random.randn(contour_length) * 0.1)
    # Falling (statements)
    for _ in range(n_samples // 4):
        t = np.linspace(0, 1, contour_length)
        patterns.append(-t + np.random.randn(contour_length) * 0.1)
    # Flat
    for _ in range(n_samples // 4):
        patterns.append(np.random.randn(contour_length) * 0.1)
    # Rise-fall (emphasis)
    for _ in range(n_samples // 4):
        t = np.linspace(0, 1, contour_length)
        patterns.append(np.sin(np.pi * t) + np.random.randn(contour_length) * 0.1)

    contours = np.array(patterns)

    if SKLEARN_AVAILABLE:
        clustering = IntonationTemplateClustering(config)
        result = clustering.fit(contours, verbose=True)

        print(f"  Templates discovered: {len(clustering.templates)}")
        print("  Descriptions:")
        for desc in clustering.describe_templates():
            print(f"    {desc}")

        # Test prediction
        test_contour = contours[:5]
        labels, probs = clustering.predict(test_contour, return_probs=True)
        print(f"  Predicted templates: {labels.tolist()}")
        print("  [PASS]")
    else:
        print("  [SKIP] sklearn not available")
        clustering = None

    # Test 3: Intonation Encoder
    print("\n[Test 3] Intonation Encoder...")
    encoder = IntonationEncoder(config).to(device)

    batch_size = 4
    template_indices = torch.randint(0, config.num_templates, (batch_size,), device=device)

    result = encoder(template_indices=template_indices)
    print(f"  Embedding shape: {result['embedding'].shape}")
    print(f"  Output shape: {result['output'].shape}")
    print(f"  Prosody tokens shape: {result['prosody_tokens'].shape}")
    assert result['prosody_tokens'].shape == (batch_size, config.num_prosody_tokens, config.output_dim)
    print("  [PASS]")

    # Test 4: Soft Assignment
    print("\n[Test 4] Soft Assignment...")
    template_probs = F.softmax(torch.randn(batch_size, config.num_templates, device=device), dim=-1)

    result = encoder(template_probs=template_probs)
    print(f"  Soft assignment prosody tokens: {result['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 5: Intonation Predictor
    print("\n[Test 5] Intonation Predictor...")
    predictor = IntonationPredictor(config).to(device)

    seq_len = 20
    text_emb = torch.randn(batch_size, seq_len, config.text_embed_dim, device=device)
    text_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    result = predictor(text_emb, text_mask)
    print(f"  Logits shape: {result['logits'].shape}")
    print(f"  Probs shape: {result['probs'].shape}")
    print(f"  Predicted templates: {result['predicted'].tolist()}")
    print("  [PASS]")

    # Test 6: Complete Module
    print("\n[Test 6] Complete Intonation Template Module...")
    module = IntonationTemplateModule(config).to(device)

    # Manual template
    result = module(template_indices=template_indices, return_contour=True)
    print(f"  Manual selection - tokens: {result['prosody_tokens'].shape}")

    # Auto prediction
    result = module(text_embeddings=text_emb, text_mask=text_mask, return_contour=True)
    print(f"  Auto prediction - tokens: {result['prosody_tokens'].shape}")
    print(f"  Predicted template: {result['predicted_template'].tolist()}")
    print("  [PASS]")

    # Test 7: Loss Function
    print("\n[Test 7] Loss Computation...")
    loss_fn = IntonationTemplateLoss(config)

    target_contours = torch.randn(batch_size, config.contour_length, device=device)

    result = module(text_embeddings=text_emb, text_mask=text_mask, return_contour=True)
    losses = loss_fn(result, template_indices, target_contours)

    print(f"  Classification loss: {losses.get('classification_loss', 'N/A')}")
    print(f"  Reconstruction loss: {losses.get('reconstruction_loss', 'N/A')}")
    print(f"  Total loss: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 8: Adapter
    print("\n[Test 8] Intonation Template Adapter...")
    adapter = IntonationTemplateAdapter(config).to(device)

    # Manual template
    result = adapter.from_template(template_index=2, batch_size=2)
    print(f"  From template - tokens: {result['prosody_tokens'].shape}")

    # From text
    result = adapter.from_text(text_emb, text_mask)
    print(f"  From text - tokens: {result['prosody_tokens'].shape}")
    print(f"  Predicted template: {result['predicted_template'].tolist()}")
    print("  [PASS]")

    # Test 9: Gradient Flow
    print("\n[Test 9] Gradient Flow...")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)

    result = adapter.from_text(text_emb, text_mask)
    loss = adapter.compute_loss(text_emb, text_mask, template_indices, target_contours)['total']
    loss.backward()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in adapter.parameters() if p.grad is not None) ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")
    print("  [PASS]")

    # Test 10: Template to Sketch
    print("\n[Test 10] Template to Sketch Conversion...")
    if clustering is not None:
        adapter.clustering = clustering
        adapter.module.load_templates(clustering)

        sketch = adapter.template_to_sketch(template_index=0)
        print(f"  Pitch sketch shape: {sketch['pitch_sketch'].shape}")
        print(f"  Energy sketch shape: {sketch['energy_sketch'].shape}")
        print("  [PASS]")
    else:
        print("  [SKIP] No clustering model")

    print("\n" + "=" * 70)
    print("All Intonation Template tests passed!")
    print("=" * 70)

    # Model stats
    print("\nModel Statistics:")
    print("-" * 40)
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from intonation_templates import (
    IntonationTemplateConfig,
    IntonationTemplateAdapter,
    IntonationTemplateClustering,
    extract_templates_from_dataset,
)

# 1. Discover templates from training data
config = IntonationTemplateConfig(num_templates=8)
audio_files = glob.glob("data/train/**/*.wav")
clustering = extract_templates_from_dataset(audio_files, config)

# Save clustering model
clustering.save("checkpoints/intonation_templates/clustering.pkl")

# 2. Initialize adapter
adapter = IntonationTemplateAdapter(config).cuda()
adapter.load_clustering("checkpoints/intonation_templates/clustering.pkl")

# 3. Training
for batch in dataloader:
    text_emb = text_encoder(batch['text'])
    target_indices = batch['template_indices']  # From clustering
    target_contours = batch['f0_contours']

    loss = adapter.compute_loss(text_emb, text_mask, target_indices, target_contours)
    loss['total'].backward()
    optimizer.step()

# 4. Inference - Manual template selection
# Use template descriptions to choose
descriptions = adapter.get_template_descriptions()
for desc in descriptions:
    print(desc)

# Generate with specific template (e.g., rising for question)
tokens = adapter.from_template(template_index=1)  # Rising pattern
audio = csm_model.generate_with_prosody(text, tokens['prosody_tokens'])

# 5. Inference - Automatic selection
tokens = adapter.from_text(text_embeddings)
print(f"Predicted template: {tokens['predicted_template']}")
audio = csm_model.generate_with_prosody(text, tokens['prosody_tokens'])

# 6. Integration with DrawSpeech
sketch = adapter.template_to_sketch(template_index=2)
draw_speech_tokens = draw_speech_adapter.from_sketch(
    sketch['pitch_sketch'],
    sketch['energy_sketch'],
    text_embeddings
)
""")

    print("\nSources:")
    print("-" * 40)
    print("- Into-TTS paper: https://arxiv.org/abs/2204.01271")
    print("- Into-TTS demo: https://srtts.github.io/IntoTTS/")
    print("- ProsodyFM: https://arxiv.org/html/2412.11795v1")
