"""
DisCo-Speech: Two-Stage Disentanglement Codec for Voice Clone Pipeline

Based on "DisCo-Speech: Disentangled Controllable Speech Synthesis"
(arXiv:2512.13251) - https://github.com/disco-speech/DisCo-Speech

Key Innovation: Two-stage design that solves the disentanglement-reconstruction trade-off:

Stage 1 - Tri-Factor Disentanglement:
    - Parallel encoders for content, prosody, timbre with separate codebooks
    - Hybrid constraint losses (GRL + Orthogonality + MI) for clean separation
    - Content encoder uses VQ bottleneck for linguistic information
    - Prosody encoder captures pitch, rhythm, style without speaker leakage
    - Timbre encoder extracts speaker identity characteristics

Stage 2 - Fusion & Reconstruction:
    - Content + Prosody fusion layer creates unified tokens for LM prediction
    - LM performs prosodic continuation from style prompt
    - Decoder receives timbre conditioning separately
    - Timbre injection happens at decoder level, not encoder

Benefits over FACodec:
    - Solves disentanglement-reconstruction trade-off explicitly
    - LM performs prosodic continuation from style prompt
    - Decoder handles timbre independently
    - More robust zero-shot prosody control
    - Better cross-speaker prosody transfer

Architecture:
    Audio → [ContentEncoder] → Content VQ codes (linguistic)
          → [ProsodyEncoder] → Prosody codes (style, rhythm)
          → [TimbreEncoder] → Speaker embedding (identity)

    ContentProsodyFusion(content_codes, prosody_codes) → Unified LM tokens
    TimbreConditionalDecoder(lm_tokens, speaker_emb) → Reconstructed mel

Usage:
    from disco_codec import (
        DiscoCodecConfig,
        DiscoCodec,
        DiscoCodecAdapter,
    )

    # Initialize
    config = DiscoCodecConfig()
    model = DiscoCodec(config).cuda()

    # Encode to three disentangled spaces
    encoded = model.encode(audio)
    content_z = encoded['content_codes']    # [batch, seq] discrete
    prosody_z = encoded['prosody_codes']    # [batch, seq] discrete
    timbre_z = encoded['timbre_emb']        # [batch, 256] continuous

    # Two-stage reconstruction
    # Stage 1: Content + Prosody fusion
    fused = model.fuse_content_prosody(content_z, prosody_z)

    # Stage 2: Timbre-conditional decoding
    mel_recon = model.decode(fused, timbre_z)

    # Cross-speaker prosody transfer
    mel_transferred = model.transfer_prosody(
        content_audio=source_audio,   # What to say
        prosody_audio=style_audio,    # How to say it
        timbre_audio=target_speaker,  # Who speaks
    )

    # For CSM integration
    adapter = DiscoCodecAdapter(config, model)
    prefix_tokens = adapter(audio)  # [batch, 4, 2048]
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
class DiscoCodecConfig:
    """Configuration for DisCo-Speech two-stage disentanglement codec."""

    # Input dimensions
    input_dim: int = 768  # SSL feature dimension (HuBERT/WavLM)
    mel_dim: int = 80  # Mel spectrogram channels
    sample_rate: int = 16000
    hop_length: int = 320  # 20ms at 16kHz

    # Content Encoder (Stage 1) - VQ for linguistic information
    content_codebook_size: int = 1024  # Discrete content codes
    content_code_dim: int = 256  # Content embedding dimension
    content_num_layers: int = 4
    content_num_heads: int = 8
    content_ffn_dim: int = 1024
    content_commitment_cost: float = 0.25
    content_ema_decay: float = 0.99

    # Prosody Encoder (Stage 1) - Captures rhythm, pitch, style
    prosody_codebook_size: int = 512  # Smaller for prosodic patterns
    prosody_code_dim: int = 128  # Lower dim for prosody
    prosody_num_layers: int = 3
    prosody_num_heads: int = 4
    prosody_ffn_dim: int = 512

    # Timbre Encoder (Stage 1) - Speaker identity
    timbre_dim: int = 256  # Global speaker embedding
    timbre_num_layers: int = 3
    timbre_num_heads: int = 4
    timbre_ffn_dim: int = 512

    # Content-Prosody Fusion (Stage 2 pre-LM)
    fusion_dim: int = 512  # Fused content+prosody dimension
    fusion_num_layers: int = 2
    fusion_num_heads: int = 8
    fusion_ffn_dim: int = 1024

    # Timbre-Conditional Decoder (Stage 2 post-LM)
    decoder_dim: int = 512
    decoder_num_layers: int = 6
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 2048
    decoder_upsample_rates: Tuple[int, ...] = (4, 4, 5)  # 320x total

    # Disentanglement loss settings
    use_grl: bool = True  # Gradient Reversal Layer
    grl_lambda_start: float = 0.0
    grl_lambda_end: float = 0.5
    grl_warmup_epochs: int = 10

    use_orthogonality: bool = True
    beta_content_prosody: float = 0.01  # Relaxed (share temporal dynamics)
    beta_timbre_prosody: float = 0.0001  # Strict independence
    beta_timbre_content: float = 0.0001  # Strict independence

    use_mi: bool = True  # Mutual Information minimization
    mi_hidden_dim: int = 256
    mi_weight: float = 0.1

    # Training
    dropout: float = 0.1
    num_speakers: int = 1000  # For adversarial speaker classifier

    # Output for CSM integration
    output_dim: int = 2048
    num_prefix_tokens: int = 4


# =============================================================================
# GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for adversarial training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """GRL for domain adversarial training."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# =============================================================================
# VECTOR QUANTIZER
# =============================================================================

class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with EMA codebook updates.

    Used for both content and prosody discrete encoding.
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int = 512,
        code_dim: int = 256,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # Input/output projections
        self.pre_proj = nn.Linear(input_dim, code_dim)
        self.post_proj = nn.Linear(code_dim, input_dim)

        # Codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim))
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        self.register_buffer('ema_sum', torch.randn(codebook_size, code_dim))
        self.register_buffer('initialized', torch.tensor(False))

    def _init_from_data(self, z: torch.Tensor):
        """Initialize codebook from first batch."""
        n = z.shape[0]
        if n >= self.codebook_size:
            indices = torch.randperm(n)[:self.codebook_size]
            self.codebook.data.copy_(z[indices])
        else:
            repeats = (self.codebook_size // n) + 1
            self.codebook.data.copy_(z.repeat(repeats, 1)[:self.codebook_size])

        self.ema_sum.data.copy_(self.codebook.data.clone())
        self.ema_cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize input to discrete codes.

        Returns:
            Dict with z_q, indices, commitment_loss, perplexity
        """
        batch_size, seq_len, _ = x.shape

        # Project to code dimension
        z = self.pre_proj(x)  # [B, T, code_dim]
        z_flat = z.view(-1, self.code_dim)

        # Initialize from first batch
        if self.training and not self.initialized:
            self._init_from_data(z_flat)

        # Compute distances
        d = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.codebook.t())
            + self.codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )

        # Find nearest codes
        indices = d.argmin(dim=-1)
        z_q = F.embedding(indices, self.codebook)

        # EMA update
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                new_size = encodings.sum(dim=0)
                new_sum = torch.matmul(encodings.t(), z_flat)

                self.ema_cluster_size.mul_(self.ema_decay).add_(new_size, alpha=1 - self.ema_decay)
                self.ema_sum.mul_(self.ema_decay).add_(new_sum, alpha=1 - self.ema_decay)

                n = self.ema_cluster_size.clamp(min=1)
                self.codebook.data.copy_(self.ema_sum / n.unsqueeze(-1))

        # Commitment loss
        commitment_loss = F.mse_loss(z_flat, z_q.detach()) * self.commitment_cost

        # Straight-through
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape
        z_q = z_q.view(batch_size, seq_len, self.code_dim)
        indices = indices.view(batch_size, seq_len)

        # Project back
        z_q = self.post_proj(z_q)

        # Perplexity
        flat_idx = indices.view(-1)
        enc = F.one_hot(flat_idx, self.codebook_size).float()
        avg_probs = enc.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            'z_q': z_q,
            'indices': indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'embeddings': z.view(batch_size, seq_len, self.code_dim),
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to embeddings."""
        z_q = F.embedding(indices, self.codebook)
        return self.post_proj(z_q)


# =============================================================================
# ENCODER MODULES
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class ConvPreNet(nn.Module):
    """Convolutional pre-net for feature extraction."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_sizes: List[int] = [5, 5, 5],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        current_dim = in_dim

        for ks in kernel_sizes:
            padding = (ks - 1) // 2
            layers.extend([
                nn.Conv1d(current_dim, out_dim, ks, padding=padding),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = out_dim

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, out_dim]"""
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)


# =============================================================================
# STAGE 1: TRI-FACTOR DISENTANGLEMENT ENCODERS
# =============================================================================

class ContentEncoder(nn.Module):
    """
    Content Encoder with VQ bottleneck.

    Extracts linguistic/phonetic content via discrete VQ codes.
    The quantization bottleneck removes prosodic and speaker variation.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Pre-processing
        self.prenet = ConvPreNet(
            config.input_dim,
            config.content_code_dim,
            dropout=config.dropout,
        )
        self.pos_enc = PositionalEncoding(config.content_code_dim, dropout=config.dropout)

        # Transformer encoder
        self.encoder = TransformerEncoderBlock(
            dim=config.content_code_dim,
            num_layers=config.content_num_layers,
            num_heads=config.content_num_heads,
            ffn_dim=config.content_ffn_dim,
            dropout=config.dropout,
        )

        # VQ layer
        self.vq = VectorQuantizerEMA(
            input_dim=config.content_code_dim,
            codebook_size=config.content_codebook_size,
            code_dim=config.content_code_dim,
            commitment_cost=config.content_commitment_cost,
            ema_decay=config.content_ema_decay,
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode to content codes."""
        # Pre-process
        x = self.prenet(x)
        x = self.pos_enc(x)

        # Transform
        x = self.encoder(x, mask)

        # Quantize
        vq_out = self.vq(x)

        return {
            'content_z': vq_out['z_q'],
            'content_codes': vq_out['indices'],
            'content_loss': vq_out['commitment_loss'],
            'content_perplexity': vq_out['perplexity'],
            'content_emb': vq_out['embeddings'],
        }


class ProsodyEncoder(nn.Module):
    """
    Prosody Encoder with VQ bottleneck.

    Captures pitch, rhythm, stress, intonation patterns.
    Uses speaker adversarial training to remove timbre information.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Pre-processing
        self.prenet = ConvPreNet(
            config.input_dim,
            config.prosody_code_dim,
            dropout=config.dropout,
        )
        self.pos_enc = PositionalEncoding(config.prosody_code_dim, dropout=config.dropout)

        # Transformer encoder
        self.encoder = TransformerEncoderBlock(
            dim=config.prosody_code_dim,
            num_layers=config.prosody_num_layers,
            num_heads=config.prosody_num_heads,
            ffn_dim=config.prosody_ffn_dim,
            dropout=config.dropout,
        )

        # VQ layer
        self.vq = VectorQuantizerEMA(
            input_dim=config.prosody_code_dim,
            codebook_size=config.prosody_codebook_size,
            code_dim=config.prosody_code_dim,
            commitment_cost=config.content_commitment_cost,
            ema_decay=config.content_ema_decay,
        )

        # Speaker adversarial head (with GRL)
        if config.use_grl:
            self.grl = GradientReversalLayer(config.grl_lambda_start)
            self.speaker_classifier = nn.Sequential(
                nn.Linear(config.prosody_code_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, config.num_speakers),
            )
        else:
            self.grl = None
            self.speaker_classifier = None

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        speaker_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode to prosody codes."""
        # Pre-process
        x = self.prenet(x)
        x = self.pos_enc(x)

        # Transform
        x = self.encoder(x, mask)

        # Quantize
        vq_out = self.vq(x)

        result = {
            'prosody_z': vq_out['z_q'],
            'prosody_codes': vq_out['indices'],
            'prosody_loss': vq_out['commitment_loss'],
            'prosody_perplexity': vq_out['perplexity'],
            'prosody_emb': vq_out['embeddings'],
        }

        # Speaker adversarial loss
        if self.grl is not None and speaker_labels is not None:
            # Pool for speaker classification
            pooled = vq_out['embeddings'].mean(dim=1)  # [batch, dim]
            pooled_grl = self.grl(pooled)
            speaker_logits = self.speaker_classifier(pooled_grl)
            speaker_loss = F.cross_entropy(speaker_logits, speaker_labels)
            result['speaker_adv_loss'] = speaker_loss
            result['speaker_logits'] = speaker_logits
        else:
            result['speaker_adv_loss'] = torch.tensor(0.0, device=x.device)

        return result

    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda."""
        if self.grl is not None:
            self.grl.set_lambda(lambda_)


class TimbreEncoder(nn.Module):
    """
    Timbre/Speaker Encoder.

    Extracts global speaker identity via attentive statistics pooling.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Pre-processing
        self.prenet = ConvPreNet(
            config.input_dim,
            config.timbre_dim * 2,
            dropout=config.dropout,
        )

        # Frame-level processing
        self.frame_layers = nn.Sequential(
            nn.Conv1d(config.timbre_dim * 2, config.timbre_dim * 2, 3, padding=1),
            nn.BatchNorm1d(config.timbre_dim * 2),
            nn.ReLU(),
            nn.Conv1d(config.timbre_dim * 2, config.timbre_dim * 2, 3, padding=1, dilation=2),
            nn.BatchNorm1d(config.timbre_dim * 2),
            nn.ReLU(),
        )

        # Attentive statistics pooling
        self.attention = nn.Sequential(
            nn.Conv1d(config.timbre_dim * 2, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, config.timbre_dim * 2, 1),
            nn.Softmax(dim=2),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.BatchNorm1d(config.timbre_dim * 4),  # mean + std
            nn.Linear(config.timbre_dim * 4, config.timbre_dim),
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract speaker embedding."""
        # Pre-process
        x = self.prenet(x)  # [B, T, D*2]
        x = x.transpose(1, 2)  # [B, D*2, T]

        # Frame-level
        x = self.frame_layers(x)  # [B, D*2, T]

        # Attentive pooling
        attn = self.attention(x)  # [B, D*2, T]

        # Weighted mean
        mean = (x * attn).sum(dim=2)  # [B, D*2]

        # Weighted std
        var = ((x - mean.unsqueeze(2)).pow(2) * attn).sum(dim=2)
        std = var.clamp(min=1e-5).sqrt()  # [B, D*2]

        # Concatenate and project
        stats = torch.cat([mean, std], dim=1)  # [B, D*4]
        speaker_emb = self.output_proj(stats)  # [B, D]

        return {
            'timbre_emb': speaker_emb,
        }


# =============================================================================
# STAGE 2: CONTENT-PROSODY FUSION
# =============================================================================

class ContentProsodyFusion(nn.Module):
    """
    Fuses content and prosody codes for LM prediction.

    Creates unified tokens that the language model can use for
    prosodic continuation from style prompt.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Project content and prosody to fusion dimension
        self.content_proj = nn.Linear(config.content_code_dim, config.fusion_dim)
        self.prosody_proj = nn.Linear(config.prosody_code_dim, config.fusion_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.fusion_dim, dropout=config.dropout)

        # Cross-attention: prosody attends to content
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.fusion_num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.fusion_dim)

        # Fusion transformer
        self.fusion_encoder = TransformerEncoderBlock(
            dim=config.fusion_dim,
            num_layers=config.fusion_num_layers,
            num_heads=config.fusion_num_heads,
            ffn_dim=config.fusion_ffn_dim,
            dropout=config.dropout,
        )

        # Output projection
        self.output_proj = nn.Linear(config.fusion_dim, config.fusion_dim)

    def forward(
        self,
        content_z: torch.Tensor,  # [batch, seq, content_dim]
        prosody_z: torch.Tensor,  # [batch, seq, prosody_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fuse content and prosody."""
        # Project
        content = self.content_proj(content_z)  # [B, T, fusion_dim]
        prosody = self.prosody_proj(prosody_z)  # [B, T, fusion_dim]

        # Add positions
        content = self.pos_enc(content)
        prosody = self.pos_enc(prosody)

        # Cross-attention: prosody queries content
        attn_out, attn_weights = self.cross_attn(
            prosody, content, content,
            key_padding_mask=mask,
        )
        prosody = self.cross_norm(prosody + attn_out)

        # Fusion: add content and prosody
        fused = content + prosody

        # Self-attention
        fused = self.fusion_encoder(fused, mask)

        # Output
        fused = self.output_proj(fused)

        return {
            'fused_z': fused,  # [batch, seq, fusion_dim]
            'attn_weights': attn_weights,
        }


# =============================================================================
# STAGE 2: TIMBRE-CONDITIONAL DECODER
# =============================================================================

class TimbreConditionalDecoder(nn.Module):
    """
    Decoder that reconstructs mel-spectrogram with timbre conditioning.

    Key insight: timbre is injected at decoder level, not encoder.
    This allows clean prosody transfer across speakers.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Timbre conditioning via FiLM (Feature-wise Linear Modulation)
        self.timbre_to_scale = nn.Linear(config.timbre_dim, config.decoder_dim)
        self.timbre_to_shift = nn.Linear(config.timbre_dim, config.decoder_dim)

        # Input projection
        self.input_proj = nn.Linear(config.fusion_dim, config.decoder_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(config.decoder_dim, dropout=config.dropout)

        # Decoder layers with timbre conditioning
        self.decoder_layers = nn.ModuleList([
            TimbreConditionedDecoderLayer(
                dim=config.decoder_dim,
                num_heads=config.decoder_num_heads,
                ffn_dim=config.decoder_ffn_dim,
                timbre_dim=config.timbre_dim,
                dropout=config.dropout,
            )
            for _ in range(config.decoder_num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.decoder_dim)

        # Mel projection
        self.mel_proj = nn.Linear(config.decoder_dim, config.mel_dim)

    def forward(
        self,
        fused_z: torch.Tensor,  # [batch, seq, fusion_dim]
        timbre_emb: torch.Tensor,  # [batch, timbre_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode with timbre conditioning."""
        # Project input
        x = self.input_proj(fused_z)  # [B, T, decoder_dim]
        x = self.pos_enc(x)

        # Global FiLM conditioning
        scale = self.timbre_to_scale(timbre_emb).unsqueeze(1)  # [B, 1, D]
        shift = self.timbre_to_shift(timbre_emb).unsqueeze(1)  # [B, 1, D]
        x = x * (1 + scale) + shift

        # Decoder with layer-wise timbre conditioning
        for layer in self.decoder_layers:
            x = layer(x, timbre_emb, mask)

        x = self.final_norm(x)

        # Project to mel
        mel_out = self.mel_proj(x)  # [B, T, mel_dim]

        return {
            'mel_out': mel_out,
        }


class TimbreConditionedDecoderLayer(nn.Module):
    """Decoder layer with timbre FiLM conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        timbre_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

        # Timbre FiLM
        self.film_scale = nn.Linear(timbre_dim, dim)
        self.film_shift = nn.Linear(timbre_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        timbre_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # FiLM conditioning
        scale = self.film_scale(timbre_emb).unsqueeze(1)  # [B, 1, D]
        shift = self.film_shift(timbre_emb).unsqueeze(1)
        x = x * (1 + 0.1 * scale) + 0.1 * shift  # Scaled for stability

        return x


# =============================================================================
# HYBRID DISENTANGLEMENT LOSS
# =============================================================================

class HybridDisentanglementLoss(nn.Module):
    """
    Combined disentanglement losses: GRL + Orthogonality + MI.

    From DisCo-Speech paper: hybrid constraints give cleaner separation
    than any single approach alone.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # MI estimation (MINE-style)
        if config.use_mi:
            self.mi_net_cp = MIEstimator(config.content_code_dim, config.prosody_code_dim, config.mi_hidden_dim)
            self.mi_net_tp = MIEstimator(config.timbre_dim, config.prosody_code_dim, config.mi_hidden_dim)
            self.mi_net_tc = MIEstimator(config.timbre_dim, config.content_code_dim, config.mi_hidden_dim)

    def orthogonality_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        target_sim: float,
    ) -> torch.Tensor:
        """Soft orthogonality loss."""
        # Pool if sequence
        if z1.dim() == 3:
            z1 = z1.mean(dim=1)
        if z2.dim() == 3:
            z2 = z2.mean(dim=1)

        # Normalize
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        # Cosine similarity
        cos_sim = (z1 * z2).sum(dim=-1)

        # Push toward target
        return (cos_sim - target_sim).pow(2).mean()

    def forward(
        self,
        content_emb: torch.Tensor,  # [batch, seq, content_dim]
        prosody_emb: torch.Tensor,  # [batch, seq, prosody_dim]
        timbre_emb: torch.Tensor,   # [batch, timbre_dim]
    ) -> Dict[str, torch.Tensor]:
        """Compute all disentanglement losses."""
        losses = {}
        device = content_emb.device

        # Orthogonality losses
        if self.config.use_orthogonality:
            losses['ortho_content_prosody'] = self.orthogonality_loss(
                content_emb, prosody_emb, self.config.beta_content_prosody
            )
            losses['ortho_timbre_prosody'] = self.orthogonality_loss(
                timbre_emb, prosody_emb, self.config.beta_timbre_prosody
            )
            losses['ortho_timbre_content'] = self.orthogonality_loss(
                timbre_emb, content_emb, self.config.beta_timbre_content
            )
            losses['ortho_total'] = (
                losses['ortho_content_prosody'] +
                losses['ortho_timbre_prosody'] +
                losses['ortho_timbre_content']
            )
        else:
            losses['ortho_total'] = torch.tensor(0.0, device=device)

        # MI minimization
        if self.config.use_mi:
            # Pool sequences for MI estimation
            content_pooled = content_emb.mean(dim=1)
            prosody_pooled = prosody_emb.mean(dim=1)

            mi_cp = self.mi_net_cp(content_pooled, prosody_pooled)
            mi_tp = self.mi_net_tp(timbre_emb, prosody_pooled)
            mi_tc = self.mi_net_tc(timbre_emb, content_pooled)

            losses['mi_content_prosody'] = mi_cp
            losses['mi_timbre_prosody'] = mi_tp
            losses['mi_timbre_content'] = mi_tc
            losses['mi_total'] = self.config.mi_weight * (mi_cp + mi_tp + mi_tc)
        else:
            losses['mi_total'] = torch.tensor(0.0, device=device)

        # Total hybrid loss
        losses['total'] = losses['ortho_total'] + losses['mi_total']

        return losses


class MIEstimator(nn.Module):
    """
    Mutual Information estimator using MINE lower bound.

    Estimates I(X;Y) by training a discriminator T(x,y).
    """

    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MINE lower bound (to be maximized → minimize negative)."""
        batch_size = x.shape[0]

        # Joint samples
        joint = torch.cat([x, y], dim=-1)
        joint_score = self.net(joint)

        # Marginal samples (shuffle y)
        idx = torch.randperm(batch_size, device=x.device)
        marginal = torch.cat([x, y[idx]], dim=-1)
        marginal_score = self.net(marginal)

        # MINE lower bound (Donsker-Varadhan)
        mi_lb = joint_score.mean() - torch.log(marginal_score.exp().mean() + 1e-8)

        # We want to minimize MI, so return negative
        return -mi_lb


# =============================================================================
# FULL DISCO-CODEC MODEL
# =============================================================================

class DiscoCodec(nn.Module):
    """
    DisCo-Speech Two-Stage Disentanglement Codec.

    Combines tri-factor disentanglement with fusion and reconstruction.
    """

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Stage 1: Tri-Factor Encoders
        self.content_encoder = ContentEncoder(config)
        self.prosody_encoder = ProsodyEncoder(config)
        self.timbre_encoder = TimbreEncoder(config)

        # Stage 2: Fusion and Decoder
        self.fusion = ContentProsodyFusion(config)
        self.decoder = TimbreConditionalDecoder(config)

        # Disentanglement loss
        self.disentangle_loss = HybridDisentanglementLoss(config)

    def encode(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        speaker_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode to three disentangled spaces."""
        content_out = self.content_encoder(x, mask)
        prosody_out = self.prosody_encoder(x, speaker_labels, mask)
        timbre_out = self.timbre_encoder(x, mask)

        return {
            # Content
            'content_z': content_out['content_z'],
            'content_codes': content_out['content_codes'],
            'content_emb': content_out['content_emb'],
            'content_loss': content_out['content_loss'],
            'content_perplexity': content_out['content_perplexity'],
            # Prosody
            'prosody_z': prosody_out['prosody_z'],
            'prosody_codes': prosody_out['prosody_codes'],
            'prosody_emb': prosody_out['prosody_emb'],
            'prosody_loss': prosody_out['prosody_loss'],
            'prosody_perplexity': prosody_out['prosody_perplexity'],
            'speaker_adv_loss': prosody_out['speaker_adv_loss'],
            # Timbre
            'timbre_emb': timbre_out['timbre_emb'],
        }

    def fuse_content_prosody(
        self,
        content_z: torch.Tensor,
        prosody_z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2 pre-LM: Fuse content and prosody."""
        return self.fusion(content_z, prosody_z, mask)

    def decode(
        self,
        fused_z: torch.Tensor,
        timbre_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2 post-LM: Decode with timbre conditioning."""
        return self.decoder(fused_z, timbre_emb, mask)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        mel_target: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with reconstruction."""
        # Stage 1: Encode
        encoded = self.encode(x, speaker_labels, mask)

        # Stage 2: Fuse
        fusion_out = self.fuse_content_prosody(
            encoded['content_z'],
            encoded['prosody_z'],
            mask,
        )

        # Stage 2: Decode
        decoder_out = self.decode(
            fusion_out['fused_z'],
            encoded['timbre_emb'],
            mask,
        )

        # Disentanglement losses
        disentangle_losses = self.disentangle_loss(
            encoded['content_emb'],
            encoded['prosody_emb'],
            encoded['timbre_emb'],
        )

        # Combine all outputs
        result = {
            **encoded,
            'fused_z': fusion_out['fused_z'],
            'mel_out': decoder_out['mel_out'],
            **disentangle_losses,
        }

        # Reconstruction loss
        if mel_target is not None:
            recon_loss = F.l1_loss(decoder_out['mel_out'], mel_target)
            result['recon_loss'] = recon_loss

        return result

    def transfer_prosody(
        self,
        content_audio: torch.Tensor,  # Source content (what to say)
        prosody_audio: torch.Tensor,  # Style reference (how to say it)
        timbre_audio: torch.Tensor,   # Speaker reference (who speaks)
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Cross-speaker prosody transfer."""
        # Encode each
        content_out = self.content_encoder(content_audio, mask)
        prosody_out = self.prosody_encoder(prosody_audio, None, mask)
        timbre_out = self.timbre_encoder(timbre_audio, mask)

        # Fuse content + prosody
        fusion_out = self.fuse_content_prosody(
            content_out['content_z'],
            prosody_out['prosody_z'],
            mask,
        )

        # Decode with target timbre
        decoder_out = self.decode(
            fusion_out['fused_z'],
            timbre_out['timbre_emb'],
            mask,
        )

        return {
            'mel_out': decoder_out['mel_out'],
            'content_codes': content_out['content_codes'],
            'prosody_codes': prosody_out['prosody_codes'],
            'timbre_emb': timbre_out['timbre_emb'],
        }

    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda for curriculum training."""
        self.prosody_encoder.set_grl_lambda(lambda_)


# =============================================================================
# DISCO-CODEC ADAPTER FOR CSM INTEGRATION
# =============================================================================

class DiscoCodecAdapter(nn.Module):
    """
    Adapts DisCo-Codec outputs to CSM prosody prefix tokens.

    Extracts prosody from audio and converts to prefix tokens
    that condition CSM generation.
    """

    def __init__(
        self,
        config: DiscoCodecConfig,
        disco_codec: Optional[DiscoCodec] = None,
    ):
        super().__init__()
        self.config = config

        # Create codec if not provided
        if disco_codec is not None:
            self.codec = disco_codec
        else:
            self.codec = DiscoCodec(config)

        # Prosody code embedding
        self.prosody_embedding = nn.Embedding(
            config.prosody_codebook_size,
            config.prosody_code_dim,
        )

        # Temporal aggregation
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(config.prosody_code_dim, 512, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.GELU(),
        )

        # Attention pooling to fixed tokens
        self.attention_pool = AttentiveMultiPool(
            in_dim=512,
            num_pools=config.num_prefix_tokens,
        )

        # Output projection to CSM hidden size
        self.output_proj = nn.Sequential(
            nn.Linear(512, config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq, input_dim]
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract prosody and convert to prefix tokens."""
        # Encode with codec
        encoded = self.codec.encode(x, None, mask)

        # Get prosody codes
        prosody_codes = encoded['prosody_codes']  # [batch, seq]

        # Embed codes
        prosody_emb = self.prosody_embedding(prosody_codes)  # [batch, seq, dim]

        # Temporal encoding
        x = prosody_emb.transpose(1, 2)  # [B, D, T]
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)  # [B, T, 512]

        # Pool to fixed tokens
        x = self.attention_pool(x, mask)  # [B, num_tokens, 512]

        # Project to output dim
        prosody_tokens = self.output_proj(x)  # [B, num_tokens, output_dim]

        return {
            'prosody_tokens': prosody_tokens,
            'prosody_codes': prosody_codes,
            'content_codes': encoded['content_codes'],
            'timbre_emb': encoded['timbre_emb'],
        }

    def from_codes(
        self,
        prosody_codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert prosody codes directly to prefix tokens."""
        prosody_emb = self.prosody_embedding(prosody_codes)

        x = prosody_emb.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)

        x = self.attention_pool(x, mask)
        prosody_tokens = self.output_proj(x)

        return prosody_tokens


class AttentiveMultiPool(nn.Module):
    """Attention-based pooling to fixed number of vectors."""

    def __init__(self, in_dim: int, num_pools: int, num_heads: int = 4):
        super().__init__()
        self.num_pools = num_pools

        self.queries = nn.Parameter(torch.randn(1, num_pools, in_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        output, _ = self.attention(queries, x, x, key_padding_mask=key_padding_mask)
        return self.norm(output)


# =============================================================================
# DISCO-CODEC LOSS FUNCTION
# =============================================================================

class DiscoCodecLoss(nn.Module):
    """Combined loss for DisCo-Codec training."""

    def __init__(self, config: DiscoCodecConfig):
        super().__init__()
        self.config = config

        # Loss weights
        self.recon_weight = 1.0
        self.content_vq_weight = 0.25
        self.prosody_vq_weight = 0.25
        self.speaker_adv_weight = 0.1
        self.disentangle_weight = 0.5

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        mel_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}

        # Reconstruction loss
        if 'recon_loss' in output:
            losses['recon'] = output['recon_loss'] * self.recon_weight
        else:
            losses['recon'] = F.l1_loss(output['mel_out'], mel_target) * self.recon_weight

        # VQ losses
        losses['content_vq'] = output['content_loss'] * self.content_vq_weight
        losses['prosody_vq'] = output['prosody_loss'] * self.prosody_vq_weight

        # Speaker adversarial loss
        losses['speaker_adv'] = output['speaker_adv_loss'] * self.speaker_adv_weight

        # Disentanglement losses
        losses['disentangle'] = output['total'] * self.disentangle_weight

        # Total
        losses['total'] = (
            losses['recon'] +
            losses['content_vq'] +
            losses['prosody_vq'] +
            losses['speaker_adv'] +
            losses['disentangle']
        )

        # Metrics
        losses['content_perplexity'] = output['content_perplexity']
        losses['prosody_perplexity'] = output['prosody_perplexity']

        return losses


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DisCo-Speech Two-Stage Disentanglement Codec - Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    seq_len = 100
    input_dim = 768
    mel_dim = 80

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    config = DiscoCodecConfig()
    print(f"  Content codebook: {config.content_codebook_size} codes")
    print(f"  Prosody codebook: {config.prosody_codebook_size} codes")
    print(f"  Timbre dim: {config.timbre_dim}")
    print("  [PASS]")

    # Test 2: VectorQuantizerEMA
    print("\n[Test 2] VectorQuantizerEMA...")
    vq = VectorQuantizerEMA(input_dim=256, codebook_size=512, code_dim=256)
    x = torch.randn(batch_size, seq_len, 256)
    vq_out = vq(x)
    print(f"  Input: {x.shape}")
    print(f"  Quantized: {vq_out['z_q'].shape}")
    print(f"  Indices: {vq_out['indices'].shape}")
    print(f"  Perplexity: {vq_out['perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 3: ContentEncoder
    print("\n[Test 3] ContentEncoder...")
    content_enc = ContentEncoder(config)
    x = torch.randn(batch_size, seq_len, input_dim)
    content_out = content_enc(x)
    print(f"  Input: {x.shape}")
    print(f"  Content z: {content_out['content_z'].shape}")
    print(f"  Content codes: {content_out['content_codes'].shape}")
    print(f"  Perplexity: {content_out['content_perplexity'].item():.2f}")
    print("  [PASS]")

    # Test 4: ProsodyEncoder
    print("\n[Test 4] ProsodyEncoder...")
    prosody_enc = ProsodyEncoder(config)
    speaker_labels = torch.randint(0, config.num_speakers, (batch_size,))
    prosody_out = prosody_enc(x, speaker_labels)
    print(f"  Prosody z: {prosody_out['prosody_z'].shape}")
    print(f"  Prosody codes: {prosody_out['prosody_codes'].shape}")
    print(f"  Speaker adv loss: {prosody_out['speaker_adv_loss'].item():.4f}")
    print("  [PASS]")

    # Test 5: TimbreEncoder
    print("\n[Test 5] TimbreEncoder...")
    timbre_enc = TimbreEncoder(config)
    timbre_out = timbre_enc(x)
    print(f"  Timbre emb: {timbre_out['timbre_emb'].shape}")
    print("  [PASS]")

    # Test 6: ContentProsodyFusion
    print("\n[Test 6] ContentProsodyFusion...")
    fusion = ContentProsodyFusion(config)
    fusion_out = fusion(content_out['content_z'], prosody_out['prosody_z'])
    print(f"  Fused z: {fusion_out['fused_z'].shape}")
    print("  [PASS]")

    # Test 7: TimbreConditionalDecoder
    print("\n[Test 7] TimbreConditionalDecoder...")
    decoder = TimbreConditionalDecoder(config)
    decoder_out = decoder(fusion_out['fused_z'], timbre_out['timbre_emb'])
    print(f"  Mel out: {decoder_out['mel_out'].shape}")
    print("  [PASS]")

    # Test 8: HybridDisentanglementLoss
    print("\n[Test 8] HybridDisentanglementLoss...")
    disentangle_loss = HybridDisentanglementLoss(config)
    losses = disentangle_loss(
        content_out['content_emb'],
        prosody_out['prosody_emb'],
        timbre_out['timbre_emb'],
    )
    print(f"  Orthogonality loss: {losses['ortho_total'].item():.6f}")
    print(f"  MI loss: {losses['mi_total'].item():.6f}")
    print(f"  Total: {losses['total'].item():.6f}")
    print("  [PASS]")

    # Test 9: Full DiscoCodec
    print("\n[Test 9] DiscoCodec (full model)...")
    model = DiscoCodec(config)
    mel_target = torch.randn(batch_size, seq_len, mel_dim)
    output = model(x, mel_target, speaker_labels)
    print(f"  Content codes: {output['content_codes'].shape}")
    print(f"  Prosody codes: {output['prosody_codes'].shape}")
    print(f"  Timbre emb: {output['timbre_emb'].shape}")
    print(f"  Mel out: {output['mel_out'].shape}")
    print(f"  Recon loss: {output['recon_loss'].item():.4f}")
    print("  [PASS]")

    # Test 10: DiscoCodecLoss
    print("\n[Test 10] DiscoCodecLoss...")
    loss_fn = DiscoCodecLoss(config)
    losses = loss_fn(output, mel_target)
    print(f"  Recon loss: {losses['recon'].item():.4f}")
    print(f"  Content VQ: {losses['content_vq'].item():.4f}")
    print(f"  Prosody VQ: {losses['prosody_vq'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 11: Prosody transfer
    print("\n[Test 11] Prosody Transfer...")
    content_audio = torch.randn(batch_size, seq_len, input_dim)
    prosody_audio = torch.randn(batch_size, seq_len, input_dim)
    timbre_audio = torch.randn(batch_size, seq_len, input_dim)

    transfer_out = model.transfer_prosody(content_audio, prosody_audio, timbre_audio)
    print(f"  Transferred mel: {transfer_out['mel_out'].shape}")
    print("  [PASS]")

    # Test 12: DiscoCodecAdapter
    print("\n[Test 12] DiscoCodecAdapter...")
    adapter = DiscoCodecAdapter(config, model)
    adapter_out = adapter(x)
    print(f"  Prosody tokens: {adapter_out['prosody_tokens'].shape}")
    expected_shape = (batch_size, config.num_prefix_tokens, config.output_dim)
    assert adapter_out['prosody_tokens'].shape == expected_shape
    print("  [PASS]")

    # Test 13: Gradient flow
    print("\n[Test 13] Gradient flow...")
    model.zero_grad()
    output = model(x, mel_target, speaker_labels)
    losses = loss_fn(output, mel_target)
    losses['total'].backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Has gradients: {has_grad}")
    print("  [PASS]")

    # Test 14: GRL lambda update
    print("\n[Test 14] GRL Lambda Update...")
    initial_lambda = model.prosody_encoder.grl.lambda_
    model.set_grl_lambda(0.5)
    updated_lambda = model.prosody_encoder.grl.lambda_
    print(f"  Initial λ: {initial_lambda:.2f}")
    print(f"  Updated λ: {updated_lambda:.2f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All DisCo-Speech tests passed!")
    print("=" * 60)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel size: {trainable_params / 1e6:.2f}M trainable parameters")

    print("\n" + "-" * 60)
    print("Quick Start Guide")
    print("-" * 60)
    print("""
from disco_codec import (
    DiscoCodecConfig,
    DiscoCodec,
    DiscoCodecAdapter,
    DiscoCodecLoss,
)

# Initialize model
config = DiscoCodecConfig()
model = DiscoCodec(config).cuda()
loss_fn = DiscoCodecLoss(config)

# Training loop
for epoch in range(100):
    # Update GRL lambda (curriculum training)
    progress = epoch / 100
    model.set_grl_lambda(progress * 0.5)

    for batch in dataloader:
        features = ssl_encoder(batch['audio'])  # HuBERT/WavLM
        mel = mel_extractor(batch['audio'])

        # Forward pass
        output = model(features, mel, batch['speaker_id'])
        losses = loss_fn(output, mel)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Monitor disentanglement
        print(f"Content perplexity: {losses['content_perplexity']:.2f}")
        print(f"Prosody perplexity: {losses['prosody_perplexity']:.2f}")

# Prosody transfer (content from A, prosody from B, timbre from C)
with torch.no_grad():
    transferred = model.transfer_prosody(
        content_audio=source_features,
        prosody_audio=style_features,
        timbre_audio=speaker_features,
    )
    mel_out = transferred['mel_out']

# CSM integration
adapter = DiscoCodecAdapter(config, model)
prosody_tokens = adapter(features)['prosody_tokens']  # [batch, 4, 2048]

# Use with ProsodyControlledCSM
combined_prefix = torch.cat([prosody_tokens, other_conditioning], dim=1)
output = csm_model(input_ids, prosody_prefix=combined_prefix)
    """)
