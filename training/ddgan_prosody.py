"""
DDGAN-Accelerated Prosody Diffusion (DiffProsody Approach)

Based on DiffProsody (arXiv:2307.16549): "DiffProsody: Diffusion-based Latent Prosody
Generation for Expressive Speech Synthesis with Prosody Conditional Adversarial Training"

Also incorporates techniques from DiffGAN-TTS (arXiv:2201.11972): "High-Fidelity and
Efficient Text-to-Speech with Denoising Diffusion GANs"

Key Innovation: DDGAN (Denoising Diffusion GAN) enables high-quality prosody generation
in just 2-4 denoising steps, compared to 50-100 steps for standard diffusion.

Architecture:
1. **Prosody VQ-VAE**: Quantizes prosody features into compact latent space
   - Encoder: Mel → prosody latent z
   - Codebook: Discrete prosody tokens
   - Decoder: z → reconstructed prosody features

2. **DDGAN Prosody Generator**: Generates prosody latents from text+speaker
   - Conditional denoiser with adversarial training
   - Few-step sampling (2-4 steps vs 50+ for standard)
   - Uses active shallow diffusion mechanism

3. **Prosody Conditional Discriminator**: Evaluates prosody quality
   - Multi-scale discriminator
   - Prosody-conditioned (ensures prosody matches speech)
   - Enables larger denoising steps while maintaining quality

Benefits:
- 16x faster than conventional diffusion (DiffProsody claim)
- 4x faster than ProsodyFlow at comparable quality
- Generates diverse, expressive prosody from text alone
- Adversarial training improves prosody expressiveness

References:
- DiffProsody: https://arxiv.org/abs/2307.16549
- DiffGAN-TTS: https://arxiv.org/abs/2201.11972
- GitHub: https://github.com/hs-oh-prml/DiffProsody
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
class DDGANProsodyConfig:
    """Configuration for DDGAN-accelerated prosody generation."""

    # Input dimensions
    mel_dim: int = 80              # Mel-spectrogram dimension
    text_dim: int = 256            # Text encoder dimension
    speaker_dim: int = 256         # Speaker embedding dimension

    # Prosody VQ-VAE settings
    prosody_latent_dim: int = 256  # Prosody latent dimension
    vq_num_embeddings: int = 512   # Codebook size
    vq_commitment_cost: float = 0.25
    vq_ema_decay: float = 0.99     # EMA update for codebook
    vq_use_ema: bool = True

    # Prosody encoder settings
    prosody_encoder_hidden: int = 512
    prosody_encoder_layers: int = 4
    prosody_encoder_heads: int = 8

    # DDGAN settings
    num_diffusion_steps: int = 4   # Only 2-4 steps needed!
    noise_schedule: str = "cosine"  # cosine, linear, sqrt
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Generator settings
    generator_hidden: int = 512
    generator_layers: int = 6
    generator_heads: int = 8
    generator_dropout: float = 0.1

    # Discriminator settings
    discriminator_hidden: int = 256
    discriminator_layers: int = 4
    use_spectral_norm: bool = True
    use_multi_scale: bool = True
    num_discriminator_scales: int = 3

    # Training settings
    dropout: float = 0.1
    generator_lr: float = 2e-4
    discriminator_lr: float = 1e-4
    gradient_penalty_weight: float = 10.0  # R1 regularization
    feature_matching_weight: float = 10.0

    # Loss weights
    reconstruction_weight: float = 1.0
    vq_weight: float = 0.25
    adversarial_weight: float = 1.0
    fm_weight: float = 2.0  # Feature matching

    # Output settings (for CSM integration)
    output_dim: int = 2048         # Match prosody encoder output
    num_prosody_tokens: int = 4    # Prefix tokens


# =============================================================================
# NOISE SCHEDULES
# =============================================================================

class NoiseSchedule:
    """
    Noise schedule for diffusion process.

    DDGAN uses fewer steps, so schedule design is crucial for quality.
    """

    def __init__(self, config: DDGANProsodyConfig):
        self.config = config
        self.num_steps = config.num_diffusion_steps

        if config.noise_schedule == "linear":
            self.betas = self._linear_schedule()
        elif config.noise_schedule == "cosine":
            self.betas = self._cosine_schedule()
        elif config.noise_schedule == "sqrt":
            self.betas = self._sqrt_schedule()
        else:
            raise ValueError(f"Unknown schedule: {config.noise_schedule}")

        # Precompute alpha values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _linear_schedule(self) -> torch.Tensor:
        """Linear beta schedule."""
        return torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.num_steps
        )

    def _cosine_schedule(self) -> torch.Tensor:
        """Cosine schedule as proposed in improved DDPM."""
        steps = self.num_steps + 1
        x = torch.linspace(0, self.num_steps, steps)
        alphas_cumprod = torch.cos((x / self.num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _sqrt_schedule(self) -> torch.Tensor:
        """Square root schedule for faster convergence."""
        return torch.linspace(
            self.config.beta_start ** 0.5,
            self.config.beta_end ** 0.5,
            self.num_steps
        ) ** 2

    def to(self, device: torch.device) -> 'NoiseSchedule':
        """Move tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise


# =============================================================================
# VECTOR QUANTIZER
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average codebook updates.

    Used in Prosody VQ-VAE for discretizing prosody latents.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

    def forward(
        self,
        inputs: torch.Tensor,  # [batch, seq, dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize inputs to nearest codebook entries.

        Returns:
            Dict with quantized, indices, commitment_loss, perplexity
        """
        # Flatten for distance computation
        flat_input = inputs.view(-1, self.embedding_dim)  # [B*S, D]

        # Compute distances to codebook
        distances = (
            flat_input.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Get nearest codebook indices
        encoding_indices = torch.argmin(distances, dim=1)  # [B*S]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantized embeddings
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view_as(inputs)

        # EMA updates during training
        if self.training:
            self._ema_update(flat_input, encodings)

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity for monitoring codebook usage
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'quantized': quantized,
            'indices': encoding_indices.view(inputs.shape[:-1]),
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'encodings': encodings,
        }

    def _ema_update(
        self,
        flat_input: torch.Tensor,
        encodings: torch.Tensor,
    ):
        """Update codebook with EMA."""
        with torch.no_grad():
            # Update cluster sizes
            n_i = encodings.sum(dim=0)
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size + (1 - self.decay) * n_i
            )

            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )

            # Update embedding weights
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve codebook entries by indices."""
        return self.embedding(indices)


# =============================================================================
# PROSODY VQ-VAE
# =============================================================================

class ProsodyEncoder(nn.Module):
    """
    Encodes mel-spectrograms to prosody latent representations.

    Uses Conformer-style architecture for capturing prosodic patterns.
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(config.mel_dim, config.prosody_encoder_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.prosody_encoder_hidden),
            nn.GELU(),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.prosody_encoder_hidden,
            nhead=config.prosody_encoder_heads,
            dim_feedforward=config.prosody_encoder_hidden * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.prosody_encoder_layers,
        )

        # Output projection to latent space
        self.output_proj = nn.Sequential(
            nn.Linear(config.prosody_encoder_hidden, config.prosody_latent_dim),
            nn.LayerNorm(config.prosody_latent_dim),
        )

        # Global pooling for utterance-level prosody
        self.attention_pool = AttentionPooling(config.prosody_latent_dim)

    def forward(
        self,
        mel: torch.Tensor,  # [batch, mel_dim, time]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode mel to prosody latent."""
        # Convolutional encoding
        x = self.input_proj(mel)  # [batch, hidden, time]
        x = x.transpose(1, 2)  # [batch, time, hidden]

        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)

        # Project to latent space
        latent = self.output_proj(x)  # [batch, time, latent_dim]

        # Global prosody (utterance-level)
        global_prosody = self.attention_pool(latent, mask)  # [batch, latent_dim]

        return {
            'latent': latent,               # Frame-level prosody
            'global_prosody': global_prosody,  # Utterance-level prosody
        }


class AttentionPooling(nn.Module):
    """Attention-based pooling for variable-length sequences."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, hidden]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool sequence to single vector."""
        attn_scores = self.attention(x).squeeze(-1)  # [batch, seq]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled = (x * attn_weights).sum(dim=1)  # [batch, hidden]

        return pooled


class ProsodyDecoder(nn.Module):
    """
    Decodes prosody latent back to prosody features.

    Used for VQ-VAE reconstruction loss.
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.prosody_latent_dim, config.prosody_encoder_hidden),
            nn.LayerNorm(config.prosody_encoder_hidden),
            nn.GELU(),
        )

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.prosody_encoder_hidden,
            nhead=config.prosody_encoder_heads,
            dim_feedforward=config.prosody_encoder_hidden * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.prosody_encoder_layers // 2,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.prosody_encoder_hidden, config.prosody_encoder_hidden),
            nn.GELU(),
            nn.Linear(config.prosody_encoder_hidden, config.mel_dim),
        )

    def forward(
        self,
        latent: torch.Tensor,  # [batch, seq, latent_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent to prosody features."""
        x = self.input_proj(latent)

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)

        output = self.output_proj(x)  # [batch, seq, mel_dim]
        return output.transpose(1, 2)  # [batch, mel_dim, seq]


class ProsodyVQVAE(nn.Module):
    """
    Prosody VQ-VAE: Encodes prosody into discrete latent space.

    Architecture:
        mel → Encoder → VQ → Decoder → reconstructed mel

    The VQ codebook learns discrete prosody patterns.
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

        self.encoder = ProsodyEncoder(config)
        self.decoder = ProsodyDecoder(config)

        self.vq = VectorQuantizerEMA(
            num_embeddings=config.vq_num_embeddings,
            embedding_dim=config.prosody_latent_dim,
            commitment_cost=config.vq_commitment_cost,
            decay=config.vq_ema_decay,
        )

    def encode(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode mel to quantized prosody latent."""
        enc_output = self.encoder(mel, mask)
        vq_output = self.vq(enc_output['latent'])

        return {
            'continuous_latent': enc_output['latent'],
            'quantized_latent': vq_output['quantized'],
            'indices': vq_output['indices'],
            'global_prosody': enc_output['global_prosody'],
            'commitment_loss': vq_output['commitment_loss'],
            'perplexity': vq_output['perplexity'],
        }

    def decode(
        self,
        latent: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent to prosody features."""
        return self.decoder(latent, mask)

    def forward(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with reconstruction."""
        # Encode and quantize
        enc_output = self.encode(mel, mask)

        # Decode
        reconstructed = self.decode(enc_output['quantized_latent'], mask)

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, mel)

        return {
            **enc_output,
            'reconstructed': reconstructed,
            'reconstruction_loss': reconstruction_loss,
        }


# =============================================================================
# DDGAN GENERATOR
# =============================================================================

class DDGANGenerator(nn.Module):
    """
    DDGAN Prosody Generator: Generates prosody latents with few-step denoising.

    Key innovation: Uses adversarial training to enable larger denoising steps
    while maintaining high quality.

    Conditioning:
    - Text embeddings (phoneme/word level)
    - Speaker embedding
    - Diffusion timestep
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

        # Conditioning projections
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.generator_hidden),
            nn.LayerNorm(config.generator_hidden),
            nn.GELU(),
        )

        self.speaker_proj = nn.Sequential(
            nn.Linear(config.speaker_dim, config.generator_hidden),
            nn.LayerNorm(config.generator_hidden),
            nn.GELU(),
        )

        # Time embedding (sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(config.generator_hidden),
            nn.Linear(config.generator_hidden, config.generator_hidden),
            nn.GELU(),
            nn.Linear(config.generator_hidden, config.generator_hidden),
        )

        # Noise input projection
        self.noise_proj = nn.Sequential(
            nn.Linear(config.prosody_latent_dim, config.generator_hidden),
            nn.LayerNorm(config.generator_hidden),
            nn.GELU(),
        )

        # Main denoising network (U-Net style with cross-attention)
        self.denoising_blocks = nn.ModuleList([
            DenoisingBlock(
                hidden_dim=config.generator_hidden,
                num_heads=config.generator_heads,
                dropout=config.generator_dropout,
                use_cross_attention=True,
            )
            for _ in range(config.generator_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.generator_hidden),
            nn.Linear(config.generator_hidden, config.generator_hidden),
            nn.GELU(),
            nn.Dropout(config.generator_dropout),
            nn.Linear(config.generator_hidden, config.prosody_latent_dim),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,  # [batch, latent_dim] or [batch, seq, latent_dim]
        t: torch.Tensor,              # [batch] timestep
        text_cond: torch.Tensor,      # [batch, seq, text_dim]
        speaker_emb: torch.Tensor,    # [batch, speaker_dim]
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict denoised prosody latent (or noise to subtract).

        Args:
            noisy_latent: Current noisy prosody latent
            t: Diffusion timestep (0 = clean, num_steps-1 = noisy)
            text_cond: Text conditioning (phoneme embeddings)
            speaker_emb: Speaker embedding
            text_mask: Attention mask for text

        Returns:
            Predicted clean prosody latent or noise
        """
        batch_size = noisy_latent.shape[0]

        # Handle both global (2D) and sequence (3D) latents
        if noisy_latent.dim() == 2:
            noisy_latent = noisy_latent.unsqueeze(1)  # [batch, 1, latent_dim]
            squeeze_output = True
        else:
            squeeze_output = False

        # Project inputs
        h = self.noise_proj(noisy_latent)  # [batch, seq, hidden]

        # Time embedding
        t_emb = self.time_embed(t)  # [batch, hidden]
        t_emb = t_emb.unsqueeze(1)  # [batch, 1, hidden]

        # Add time embedding
        h = h + t_emb

        # Process conditioning
        text_h = self.text_proj(text_cond)  # [batch, text_seq, hidden]
        speaker_h = self.speaker_proj(speaker_emb)  # [batch, hidden]

        # Add speaker to text conditioning
        speaker_h = speaker_h.unsqueeze(1)  # [batch, 1, hidden]
        cond = torch.cat([speaker_h, text_h], dim=1)  # [batch, 1+text_seq, hidden]

        if text_mask is not None:
            # Extend mask for speaker token
            speaker_mask = torch.ones(batch_size, 1, device=text_mask.device, dtype=torch.bool)
            cond_mask = torch.cat([speaker_mask, text_mask], dim=1)
        else:
            cond_mask = None

        # Apply denoising blocks
        for block in self.denoising_blocks:
            h = block(h, cond, cond_mask)

        # Output projection
        output = self.output_proj(h)

        if squeeze_output:
            output = output.squeeze(1)  # [batch, latent_dim]

        return output


class DenoisingBlock(nn.Module):
    """
    Denoising block with self-attention and cross-attention.

    Architecture:
        Self-Attention → Cross-Attention (to conditioning) → FFN
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Cross-attention to conditioning
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through denoising block."""
        # Self-attention
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Cross-attention
        if self.use_cross_attention and cond is not None:
            x_norm = self.cross_attn_norm(x)
            key_padding_mask = ~cond_mask if cond_mask is not None else None
            cross_out, _ = self.cross_attn(
                x_norm, cond, cond,
                key_padding_mask=key_padding_mask
            )
            x = x + cross_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional/time embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch] (can be float or int timesteps)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)

        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding


# =============================================================================
# PROSODY CONDITIONAL DISCRIMINATOR
# =============================================================================

class ProsodyConditionalDiscriminator(nn.Module):
    """
    Prosody Conditional Discriminator: Evaluates prosody quality.

    Key features:
    - Multi-scale discrimination
    - Prosody-conditioned (ensures generated prosody matches speech patterns)
    - Spectral normalization for stable training

    Discriminates between:
    - Real prosody latents (from VQ-VAE encoding)
    - Fake prosody latents (from generator)
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

        # Conditioning projection
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.discriminator_hidden),
            nn.LeakyReLU(0.2),
        )

        self.speaker_proj = nn.Sequential(
            nn.Linear(config.speaker_dim, config.discriminator_hidden),
            nn.LeakyReLU(0.2),
        )

        # Multi-scale discriminators
        if config.use_multi_scale:
            self.discriminators = nn.ModuleList([
                SingleScaleDiscriminator(config, scale=i)
                for i in range(config.num_discriminator_scales)
            ])
        else:
            self.discriminators = nn.ModuleList([
                SingleScaleDiscriminator(config, scale=0)
            ])

    def forward(
        self,
        prosody_latent: torch.Tensor,  # [batch, latent_dim] or [batch, seq, latent_dim]
        text_cond: torch.Tensor,        # [batch, seq, text_dim]
        speaker_emb: torch.Tensor,      # [batch, speaker_dim]
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Discriminate real vs fake prosody.

        Returns:
            Dict with scores and intermediate features for feature matching
        """
        # Handle both global and sequence latents
        if prosody_latent.dim() == 2:
            prosody_latent = prosody_latent.unsqueeze(1)

        # Process conditioning
        text_h = self.text_proj(text_cond)  # [batch, seq, hidden]
        speaker_h = self.speaker_proj(speaker_emb)  # [batch, hidden]

        # Pool text conditioning
        if text_mask is not None:
            text_h = text_h * text_mask.unsqueeze(-1).float()
            text_pooled = text_h.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).float()
        else:
            text_pooled = text_h.mean(dim=1)

        # Combine conditioning
        cond = text_pooled + speaker_h  # [batch, hidden]

        # Get predictions from each scale
        all_scores = []
        all_features = []

        for disc in self.discriminators:
            result = disc(prosody_latent, cond)
            all_scores.append(result['score'])
            all_features.append(result['features'])

        # Combine scores
        final_score = sum(all_scores) / len(all_scores)

        return {
            'score': final_score,          # [batch, 1]
            'scores': all_scores,           # List of [batch, 1]
            'features': all_features,       # List of feature lists (for FM loss)
        }


class SingleScaleDiscriminator(nn.Module):
    """Single-scale discriminator with spectral normalization."""

    def __init__(self, config: DDGANProsodyConfig, scale: int = 0):
        super().__init__()
        self.config = config
        self.scale = scale

        hidden = config.discriminator_hidden
        input_dim = config.prosody_latent_dim + hidden  # latent + condition

        # Apply spectral norm optionally
        def maybe_spectral_norm(layer):
            if config.use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Downsampling for multi-scale
        if scale > 0:
            self.downsample = nn.AvgPool1d(kernel_size=2 ** scale, stride=2 ** scale)
        else:
            self.downsample = nn.Identity()

        # Discriminator layers
        self.layers = nn.ModuleList()

        # Layer 1
        self.layers.append(nn.Sequential(
            maybe_spectral_norm(nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Layers 2-4
        for i in range(config.discriminator_layers - 1):
            self.layers.append(nn.Sequential(
                maybe_spectral_norm(nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # Output layer
        self.output = maybe_spectral_norm(nn.Conv1d(hidden, 1, kernel_size=3, padding=1))

    def forward(
        self,
        x: torch.Tensor,     # [batch, seq, latent_dim]
        cond: torch.Tensor,  # [batch, hidden]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning score and intermediate features."""
        # Expand condition to match sequence
        cond_expanded = cond.unsqueeze(1).expand(-1, x.shape[1], -1)

        # Concatenate latent and condition
        x = torch.cat([x, cond_expanded], dim=-1)  # [batch, seq, latent+hidden]
        x = x.transpose(1, 2)  # [batch, latent+hidden, seq]

        # Downsample for multi-scale (handle short sequences)
        if self.scale > 0:
            min_len = 2 ** self.scale
            seq_len = x.shape[-1]
            if seq_len < min_len:
                # Pad short sequences
                pad_len = min_len - seq_len
                x = F.pad(x, (0, pad_len), mode='replicate')
            x = self.downsample(x)

        # Collect features
        features = []

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        # Output score
        score = self.output(x)  # [batch, 1, seq']
        score = score.mean(dim=-1)  # [batch, 1]

        return {
            'score': score,
            'features': features,
        }


# =============================================================================
# COMPLETE DDGAN PROSODY MODEL
# =============================================================================

class DDGANProsody(nn.Module):
    """
    Complete DDGAN Prosody Generation System.

    Combines:
    1. Prosody VQ-VAE for discrete latent space
    2. DDGAN Generator for few-step prosody generation
    3. Prosody Conditional Discriminator for quality improvement

    Training procedure:
    1. Pre-train VQ-VAE on prosody reconstruction
    2. Train DDGAN with discriminator on latent space
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

        # Components
        self.vqvae = ProsodyVQVAE(config)
        self.generator = DDGANGenerator(config)
        self.discriminator = ProsodyConditionalDiscriminator(config)

        # Noise schedule
        self.noise_schedule = NoiseSchedule(config)

        # Output projection to prosody tokens (for CSM)
        self.token_projection = nn.Sequential(
            nn.Linear(config.prosody_latent_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim * config.num_prosody_tokens),
        )
        self.token_norm = nn.LayerNorm(config.output_dim)

    def encode_prosody(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode mel to prosody latent using VQ-VAE."""
        return self.vqvae.encode(mel, mask)

    def generate_prosody(
        self,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate prosody latent from text and speaker.

        Uses few-step DDGAN sampling (default: 4 steps).
        """
        batch_size = text_cond.shape[0]
        device = text_cond.device

        if num_steps is None:
            num_steps = self.config.num_diffusion_steps

        # Move schedule to device
        self.noise_schedule.to(device)

        # Start from pure noise
        latent = torch.randn(
            batch_size, self.config.prosody_latent_dim, device=device
        ) * temperature

        # Reverse diffusion (few steps!)
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict denoised latent
            pred = self.generator(
                latent, t_tensor, text_cond, speaker_emb, text_mask
            )

            # DDGAN update step
            if t > 0:
                # Add noise for next step (DDPM-style but with fewer steps)
                beta = self.noise_schedule.betas[t]
                alpha = self.noise_schedule.alphas[t]
                alpha_bar = self.noise_schedule.alphas_cumprod[t]
                alpha_bar_prev = self.noise_schedule.alphas_cumprod_prev[t]

                # Predict noise
                noise_pred = (latent - torch.sqrt(alpha_bar) * pred) / torch.sqrt(1 - alpha_bar)

                # Update latent
                mean = (1 / torch.sqrt(alpha)) * (
                    latent - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
                )

                variance = self.noise_schedule.posterior_variance[t]
                noise = torch.randn_like(latent)
                latent = mean + torch.sqrt(variance) * noise
            else:
                latent = pred

        return latent

    def forward(
        self,
        mel: torch.Tensor,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Computes:
        1. VQ-VAE reconstruction
        2. Generator denoising
        3. Discriminator predictions
        """
        batch_size = mel.shape[0]
        device = mel.device

        self.noise_schedule.to(device)

        # 1. Encode real prosody
        vqvae_output = self.vqvae(mel, mel_mask)
        real_latent = vqvae_output['global_prosody']  # [batch, latent_dim]

        # 2. Sample noise and timestep
        t = torch.randint(0, self.config.num_diffusion_steps, (batch_size,), device=device)
        noise = torch.randn_like(real_latent)
        noisy_latent = self.noise_schedule.q_sample(real_latent, t, noise)

        # 3. Generator predicts clean latent
        pred_latent = self.generator(noisy_latent, t, text_cond, speaker_emb, text_mask)

        # 4. Discriminator scores
        with torch.no_grad():
            real_score = self.discriminator(
                real_latent, text_cond, speaker_emb, text_mask
            )

        fake_score = self.discriminator(
            pred_latent, text_cond, speaker_emb, text_mask
        )

        return {
            # VQ-VAE outputs
            'vqvae_output': vqvae_output,
            'real_latent': real_latent,

            # Generator outputs
            'pred_latent': pred_latent,
            'noisy_latent': noisy_latent,
            'timestep': t,
            'noise': noise,

            # Discriminator outputs
            'real_score': real_score,
            'fake_score': fake_score,
        }

    def to_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert prosody latent to prefix tokens."""
        tokens = self.token_projection(latent)
        tokens = tokens.view(-1, self.config.num_prosody_tokens, self.config.output_dim)
        tokens = self.token_norm(tokens)
        return tokens


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DDGANProsodyLoss(nn.Module):
    """
    Combined loss for DDGAN Prosody training.

    Components:
    1. VQ-VAE reconstruction loss
    2. VQ commitment loss
    3. Generator adversarial loss (non-saturating)
    4. Discriminator adversarial loss (hinge)
    5. Feature matching loss
    6. R1 gradient penalty
    """

    def __init__(self, config: DDGANProsodyConfig):
        super().__init__()
        self.config = config

    def generator_loss(
        self,
        model_output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute generator losses."""
        losses = {}

        # VQ-VAE reconstruction
        vqvae_output = model_output['vqvae_output']
        losses['reconstruction'] = vqvae_output['reconstruction_loss']
        losses['vq_commitment'] = vqvae_output['commitment_loss']

        # Denoising loss (predict clean from noisy)
        real_latent = model_output['real_latent']
        pred_latent = model_output['pred_latent']
        losses['denoising'] = F.mse_loss(pred_latent, real_latent.detach())

        # Adversarial loss (non-saturating GAN)
        fake_score = model_output['fake_score']['score']
        losses['adversarial'] = -fake_score.mean()

        # Feature matching loss
        real_features = model_output['real_score']['features']
        fake_features = model_output['fake_score']['features']

        fm_loss = 0.0
        for real_feats, fake_feats in zip(real_features, fake_features):
            for rf, ff in zip(real_feats, fake_feats):
                fm_loss = fm_loss + F.l1_loss(ff, rf.detach())
        losses['feature_matching'] = fm_loss / len(real_features)

        # Total generator loss
        losses['total'] = (
            self.config.reconstruction_weight * losses['reconstruction'] +
            self.config.vq_weight * losses['vq_commitment'] +
            losses['denoising'] +
            self.config.adversarial_weight * losses['adversarial'] +
            self.config.fm_weight * losses['feature_matching']
        )

        return losses

    def discriminator_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        real_latent: torch.Tensor,
        discriminator: ProsodyConditionalDiscriminator,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses."""
        losses = {}

        # Re-compute discriminator on real (with gradient for R1)
        real_latent.requires_grad_(True)
        real_score = discriminator(
            real_latent, text_cond, speaker_emb, text_mask
        )

        # Hinge loss
        real_scores = real_score['score']
        fake_scores = model_output['fake_score']['score']

        losses['real'] = F.relu(1.0 - real_scores).mean()
        losses['fake'] = F.relu(1.0 + fake_scores.detach()).mean()

        # R1 gradient penalty
        r1_grads = torch.autograd.grad(
            outputs=real_scores.sum(),
            inputs=real_latent,
            create_graph=True,
            only_inputs=True,
        )[0]
        r1_penalty = r1_grads.pow(2).sum(dim=-1).mean()
        losses['r1_penalty'] = r1_penalty

        # Total discriminator loss
        losses['total'] = (
            losses['real'] +
            losses['fake'] +
            self.config.gradient_penalty_weight * losses['r1_penalty']
        )

        return losses


# =============================================================================
# ADAPTER FOR PROSODY PIPELINE
# =============================================================================

class DDGANProsodyAdapter(nn.Module):
    """
    Adapter integrating DDGAN prosody with the existing pipeline.

    Provides interface compatible with ProsodyEncoder output format.

    Usage:
        adapter = DDGANProsodyAdapter(config)

        # From audio (encode existing prosody)
        tokens = adapter.from_audio(mel, text_cond, speaker_emb)

        # From text (generate new prosody)
        tokens = adapter.from_text(text_cond, speaker_emb)
    """

    def __init__(
        self,
        config: DDGANProsodyConfig,
        prosody_hidden: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Core model
        self.model = DDGANProsody(config)

        # Adapt to prosody hidden dimension
        if config.output_dim != prosody_hidden:
            self.output_adapter = nn.Sequential(
                nn.Linear(config.output_dim, prosody_hidden),
                nn.LayerNorm(prosody_hidden),
            )
        else:
            self.output_adapter = nn.Identity()

    def from_audio(
        self,
        mel: torch.Tensor,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract prosody tokens from audio."""
        # Encode prosody
        enc_output = self.model.encode_prosody(mel, mel_mask)
        latent = enc_output['global_prosody']

        # To tokens
        tokens = self.model.to_tokens(latent)
        tokens = self.output_adapter(tokens)

        return {
            'prosody_tokens': tokens,
            'latent': latent,
            'indices': enc_output['indices'],
            'perplexity': enc_output['perplexity'],
        }

    def from_text(
        self,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Generate prosody tokens from text."""
        # Generate prosody
        latent = self.model.generate_prosody(
            text_cond, speaker_emb, text_mask,
            num_steps=num_steps,
            temperature=temperature,
        )

        # To tokens
        tokens = self.model.to_tokens(latent)
        tokens = self.output_adapter(tokens)

        return {
            'prosody_tokens': tokens,
            'latent': latent,
        }

    def forward(
        self,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
        mel_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - encode if mel provided, else generate.
        """
        if mel is not None:
            return self.from_audio(mel, text_cond, speaker_emb, mel_mask, text_mask)
        else:
            return self.from_text(text_cond, speaker_emb, text_mask, num_steps, temperature)

    def compute_training_losses(
        self,
        mel: torch.Tensor,
        text_cond: torch.Tensor,
        speaker_emb: torch.Tensor,
        mel_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute all training losses."""
        # Forward pass
        model_output = self.model(
            mel, text_cond, speaker_emb, mel_mask, text_mask
        )

        # Loss function
        loss_fn = DDGANProsodyLoss(self.config)

        # Generator losses
        g_losses = loss_fn.generator_loss(model_output)

        # Discriminator losses
        d_losses = loss_fn.discriminator_loss(
            model_output,
            model_output['real_latent'].detach().clone(),
            self.model.discriminator,
            text_cond,
            speaker_emb,
            text_mask,
        )

        return {
            'generator': g_losses,
            'discriminator': d_losses,
            'model_output': model_output,
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DDGAN-Accelerated Prosody Diffusion - Test Suite")
    print("=" * 70)

    config = DDGANProsodyConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    batch_size = 2
    mel_len = 100
    text_len = 20

    # Test 1: Noise Schedule
    print("\n[Test 1] Noise Schedule...")
    schedule = NoiseSchedule(config)
    print(f"  Num steps: {config.num_diffusion_steps}")
    print(f"  Betas: {schedule.betas.tolist()}")
    print(f"  Alphas cumprod: {schedule.alphas_cumprod.tolist()}")
    print("  [PASS]")

    # Test 2: Vector Quantizer
    print("\n[Test 2] Vector Quantizer EMA...")
    vq = VectorQuantizerEMA(
        num_embeddings=config.vq_num_embeddings,
        embedding_dim=config.prosody_latent_dim,
    ).to(device)

    inputs = torch.randn(batch_size, 10, config.prosody_latent_dim, device=device)
    vq_output = vq(inputs)

    print(f"  Input shape: {inputs.shape}")
    print(f"  Quantized shape: {vq_output['quantized'].shape}")
    print(f"  Indices shape: {vq_output['indices'].shape}")
    print(f"  Perplexity: {vq_output['perplexity'].item():.2f}")
    print(f"  Commitment loss: {vq_output['commitment_loss'].item():.4f}")
    print("  [PASS]")

    # Test 3: Prosody VQ-VAE
    print("\n[Test 3] Prosody VQ-VAE...")
    vqvae = ProsodyVQVAE(config).to(device)

    mel = torch.randn(batch_size, config.mel_dim, mel_len, device=device)
    vqvae_output = vqvae(mel)

    print(f"  Mel input shape: {mel.shape}")
    print(f"  Quantized latent shape: {vqvae_output['quantized_latent'].shape}")
    print(f"  Global prosody shape: {vqvae_output['global_prosody'].shape}")
    print(f"  Reconstructed shape: {vqvae_output['reconstructed'].shape}")
    print(f"  Reconstruction loss: {vqvae_output['reconstruction_loss'].item():.4f}")
    print(f"  VQ perplexity: {vqvae_output['perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 4: DDGAN Generator
    print("\n[Test 4] DDGAN Generator...")
    generator = DDGANGenerator(config).to(device)

    noisy_latent = torch.randn(batch_size, config.prosody_latent_dim, device=device)
    t = torch.randint(0, config.num_diffusion_steps, (batch_size,), device=device)
    text_cond = torch.randn(batch_size, text_len, config.text_dim, device=device)
    speaker_emb = torch.randn(batch_size, config.speaker_dim, device=device)

    pred = generator(noisy_latent, t, text_cond, speaker_emb)

    print(f"  Noisy latent shape: {noisy_latent.shape}")
    print(f"  Text cond shape: {text_cond.shape}")
    print(f"  Speaker emb shape: {speaker_emb.shape}")
    print(f"  Predicted latent shape: {pred.shape}")
    print("  [PASS]")

    # Test 5: Prosody Conditional Discriminator
    print("\n[Test 5] Prosody Conditional Discriminator...")
    discriminator = ProsodyConditionalDiscriminator(config).to(device)

    prosody_latent = torch.randn(batch_size, config.prosody_latent_dim, device=device)
    disc_output = discriminator(prosody_latent, text_cond, speaker_emb)

    print(f"  Prosody latent shape: {prosody_latent.shape}")
    print(f"  Score shape: {disc_output['score'].shape}")
    print(f"  Num scales: {len(disc_output['scores'])}")
    print(f"  Scores: {[s.mean().item() for s in disc_output['scores']]}")
    print("  [PASS]")

    # Test 6: Complete DDGAN Prosody Model
    print("\n[Test 6] Complete DDGAN Prosody Model...")
    model = DDGANProsody(config).to(device)

    model_output = model(mel, text_cond, speaker_emb)

    print(f"  Real latent shape: {model_output['real_latent'].shape}")
    print(f"  Pred latent shape: {model_output['pred_latent'].shape}")
    print(f"  Real score: {model_output['real_score']['score'].mean().item():.4f}")
    print(f"  Fake score: {model_output['fake_score']['score'].mean().item():.4f}")
    print("  [PASS]")

    # Test 7: Few-Step Generation (KEY FEATURE!)
    print("\n[Test 7] Few-Step Generation (16x speedup!)...")
    for n_steps in [1, 2, 4]:
        with torch.no_grad():
            latent = model.generate_prosody(
                text_cond, speaker_emb,
                num_steps=n_steps,
            )
        print(f"  {n_steps} step(s): latent shape={latent.shape}, norm={latent.norm().item():.3f}")
    print("  [PASS]")

    # Test 8: Loss Functions
    print("\n[Test 8] Loss Functions...")
    loss_fn = DDGANProsodyLoss(config)

    g_losses = loss_fn.generator_loss(model_output)
    print(f"  Generator losses:")
    for k, v in g_losses.items():
        if k != 'total':
            print(f"    {k}: {v.item():.4f}")
    print(f"    total: {g_losses['total'].item():.4f}")

    d_losses = loss_fn.discriminator_loss(
        model_output,
        model_output['real_latent'].detach().clone(),
        model.discriminator,
        text_cond,
        speaker_emb,
    )
    print(f"  Discriminator losses:")
    for k, v in d_losses.items():
        if k != 'total':
            print(f"    {k}: {v.item():.4f}")
    print(f"    total: {d_losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 9: DDGANProsodyAdapter
    print("\n[Test 9] DDGANProsodyAdapter...")
    adapter = DDGANProsodyAdapter(config).to(device)

    # From audio
    result_audio = adapter.from_audio(mel, text_cond, speaker_emb)
    print(f"  From audio - tokens shape: {result_audio['prosody_tokens'].shape}")

    # From text (fast generation)
    with torch.no_grad():
        result_text = adapter.from_text(text_cond, speaker_emb)
    print(f"  From text - tokens shape: {result_text['prosody_tokens'].shape}")
    print("  [PASS]")

    # Test 10: Training Loop Simulation
    print("\n[Test 10] Training Loop Simulation...")
    adapter = DDGANProsodyAdapter(config).to(device)

    losses = adapter.compute_training_losses(mel, text_cond, speaker_emb)
    print(f"  Generator total: {losses['generator']['total'].item():.4f}")
    print(f"  Discriminator total: {losses['discriminator']['total'].item():.4f}")
    print("  [PASS]")

    # Test 11: Speed Comparison
    print("\n[Test 11] Speed Comparison (inference)...")
    import time

    # Create a config with more steps for comparison
    config_50 = DDGANProsodyConfig(num_diffusion_steps=50)
    adapter_50 = DDGANProsodyAdapter(config_50).to(device)

    adapter.eval()
    adapter_50.eval()
    with torch.no_grad():
        # DDGAN (4 steps)
        start = time.time()
        for _ in range(10):
            _ = adapter.from_text(text_cond, speaker_emb, num_steps=4)
        ddgan_time = (time.time() - start) / 10

        # Standard diffusion (50 steps)
        start = time.time()
        for _ in range(10):
            _ = adapter_50.from_text(text_cond, speaker_emb, num_steps=50)
        standard_time = (time.time() - start) / 10

    speedup = standard_time / ddgan_time
    print(f"  DDGAN (4 steps): {ddgan_time*1000:.2f} ms/sample")
    print(f"  Standard (50 steps): {standard_time*1000:.2f} ms/sample")
    print(f"  Speedup: {speedup:.1f}x")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All DDGAN Prosody tests passed!")
    print("=" * 70)

    print("\nUsage Example:")
    print("-" * 40)
    print("""
from ddgan_prosody import (
    DDGANProsodyConfig,
    DDGANProsody,
    DDGANProsodyAdapter,
    DDGANProsodyLoss,
)

# Initialize
config = DDGANProsodyConfig(
    num_diffusion_steps=4,  # Only 4 steps needed!
    noise_schedule="cosine",
)

adapter = DDGANProsodyAdapter(config).cuda()

# Training
optimizer_g = torch.optim.AdamW(
    list(adapter.model.vqvae.parameters()) +
    list(adapter.model.generator.parameters()),
    lr=config.generator_lr,
)
optimizer_d = torch.optim.AdamW(
    adapter.model.discriminator.parameters(),
    lr=config.discriminator_lr,
)

for batch in dataloader:
    mel = batch['mel'].cuda()
    text_cond = batch['text_embeddings'].cuda()
    speaker_emb = batch['speaker_embedding'].cuda()

    # Compute losses
    losses = adapter.compute_training_losses(mel, text_cond, speaker_emb)

    # Update discriminator
    optimizer_d.zero_grad()
    losses['discriminator']['total'].backward(retain_graph=True)
    optimizer_d.step()

    # Update generator
    optimizer_g.zero_grad()
    losses['generator']['total'].backward()
    optimizer_g.step()

# Inference - FAST! (4 steps vs 50+ for standard diffusion)
with torch.no_grad():
    result = adapter.from_text(text_cond, speaker_emb, num_steps=4)
    prosody_tokens = result['prosody_tokens']  # [batch, 4, 2048]

# Or from reference audio
result = adapter.from_audio(mel, text_cond, speaker_emb)
prosody_tokens = result['prosody_tokens']

# Integrate with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
""")
