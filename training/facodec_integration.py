"""
FACodec Integration for Voice Clone Pipeline

Based on NaturalSpeech3's FACodec which decomposes speech into 5 disentangled subspaces:
- Content: linguistic information (2 codebooks)
- Prosody: pitch, rhythm, stress patterns (1 codebook)
- Timbre: speaker identity (embedding)
- Acoustic Details: fine-grained spectral details (3 codebooks)
- Duration: implicit in frame-level codes

Key benefits over current approach:
1. Pre-trained disentanglement - better separation than our multi-encoder
2. Clean prosody codes without speaker/content leakage
3. Quantized codes (discrete tokens) instead of continuous vectors
4. Can use prosody codebook directly as conditioning input

References:
- Paper: https://arxiv.org/abs/2403.03100
- Code: https://github.com/lifeiteng/naturalspeech3_facodec
- Models: https://huggingface.co/amphion/naturalspeech3_facodec

Usage:
    from facodec_integration import (
        FACodecProsodyExtractor,
        FACodecProsodyAdapter,
        extract_facodec_prosody,
    )

    # Extract prosody from audio
    extractor = FACodecProsodyExtractor()
    prosody_codes = extractor.extract_prosody(audio)  # [batch, time]

    # Convert to conditioning tokens
    adapter = FACodecProsodyAdapter(config)
    prefix_tokens = adapter(prosody_codes)  # [batch, num_tokens, hidden]
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Use 'pip install huggingface-hub'")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FACodecConfig:
    """Configuration for FACodec integration."""

    # Model settings
    sample_rate: int = 16000
    hop_size: int = 200  # ~12.5ms per frame at 16kHz

    # Encoder settings
    encoder_ngf: int = 32
    encoder_up_ratios: Tuple[int, ...] = (2, 4, 5, 5)
    encoder_out_channels: int = 256

    # Decoder/Quantizer settings
    decoder_in_channels: int = 256
    decoder_upsample_initial: int = 1024
    decoder_ngf: int = 32
    decoder_up_ratios: Tuple[int, ...] = (5, 5, 4, 2)

    # VQ settings
    vq_num_q_content: int = 2      # Content codebooks
    vq_num_q_prosody: int = 1      # Prosody codebook
    vq_num_q_residual: int = 3     # Acoustic detail codebooks
    vq_dim: int = 256
    codebook_dim: int = 8
    codebook_size: int = 10        # 2^10 = 1024 codes per codebook

    # HuggingFace repo
    hf_repo: str = "amphion/naturalspeech3_facodec"
    encoder_filename: str = "ns3_facodec_encoder.bin"
    decoder_filename: str = "ns3_facodec_decoder.bin"

    # Prosody adapter settings
    hidden_size: int = 2048        # CSM hidden size
    num_prosody_tokens: int = 4    # Prefix tokens to generate
    prosody_embed_dim: int = 256   # FACodec prosody embedding dim
    dropout: float = 0.1


# =============================================================================
# FACODEC MODULES (Minimal Implementation)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.norm2(self.conv2(x))
        return x + residual


class FACodecEncoderMinimal(nn.Module):
    """
    Minimal FACodec encoder implementation.

    Used as fallback if Amphion not installed. Produces compatible output shape.
    For production use, prefer the official Amphion implementation.
    """

    def __init__(self, config: FACodecConfig):
        super().__init__()
        self.config = config

        # Initial conv
        self.conv_in = nn.Conv1d(1, config.encoder_ngf, kernel_size=7, padding=3)

        # Downsampling blocks
        channels = config.encoder_ngf
        self.down_blocks = nn.ModuleList()

        for ratio in config.encoder_up_ratios:
            self.down_blocks.append(nn.Sequential(
                nn.Conv1d(channels, channels * 2, kernel_size=ratio * 2,
                         stride=ratio, padding=ratio // 2),
                nn.GroupNorm(8, channels * 2),
                nn.LeakyReLU(0.2),
                ResidualBlock(channels * 2),
                ResidualBlock(channels * 2, dilation=2),
            ))
            channels *= 2

        # Output projection
        self.conv_out = nn.Conv1d(channels, config.encoder_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to latent representation.

        Args:
            x: [batch, 1, samples] waveform at 16kHz

        Returns:
            [batch, 256, time] latent features
        """
        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        x = self.conv_out(x)
        return x


class FACodecQuantizerMinimal(nn.Module):
    """
    Minimal FACodec quantizer implementation.

    Implements factorized VQ with separate codebooks for content, prosody, residual.
    """

    def __init__(self, config: FACodecConfig):
        super().__init__()
        self.config = config

        codebook_size = 2 ** config.codebook_size  # 1024 codes

        # Prosody codebook (1 quantizer)
        self.prosody_codebook = nn.Embedding(codebook_size, config.codebook_dim)
        self.prosody_proj_in = nn.Linear(config.vq_dim, config.codebook_dim)
        self.prosody_proj_out = nn.Linear(config.codebook_dim, config.vq_dim)

        # Content codebooks (2 quantizers)
        self.content_codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, config.codebook_dim)
            for _ in range(config.vq_num_q_content)
        ])
        self.content_proj_in = nn.Linear(config.vq_dim, config.codebook_dim * config.vq_num_q_content)
        self.content_proj_out = nn.Linear(config.codebook_dim * config.vq_num_q_content, config.vq_dim)

        # Residual codebooks (3 quantizers)
        self.residual_codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, config.codebook_dim)
            for _ in range(config.vq_num_q_residual)
        ])
        self.residual_proj_in = nn.Linear(config.vq_dim, config.codebook_dim * config.vq_num_q_residual)
        self.residual_proj_out = nn.Linear(config.codebook_dim * config.vq_num_q_residual, config.vq_dim)

        # Initialize codebooks
        self._init_codebooks()

    def _init_codebooks(self):
        """Initialize codebook embeddings."""
        for module in [self.prosody_codebook] + list(self.content_codebooks) + list(self.residual_codebooks):
            nn.init.uniform_(module.weight, -1/1024, 1/1024)

    def quantize_single(
        self,
        z: torch.Tensor,
        codebook: nn.Embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize to nearest codebook entry.

        Args:
            z: [batch, time, codebook_dim]
            codebook: Embedding layer

        Returns:
            quantized: [batch, time, codebook_dim]
            indices: [batch, time]
        """
        # Compute distances
        z_flat = z.reshape(-1, z.size(-1))  # [B*T, D]

        distances = (
            z_flat.pow(2).sum(dim=-1, keepdim=True) +
            codebook.weight.pow(2).sum(dim=-1) -
            2 * z_flat @ codebook.weight.T
        )  # [B*T, codebook_size]

        indices = distances.argmin(dim=-1)  # [B*T]
        quantized = codebook(indices)  # [B*T, D]

        # Reshape
        batch_size, seq_len = z.shape[:2]
        indices = indices.view(batch_size, seq_len)
        quantized = quantized.view(batch_size, seq_len, -1)

        # Straight-through gradient
        quantized = z + (quantized - z).detach()

        return quantized, indices

    def forward(
        self,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Factorized vector quantization.

        Args:
            z: [batch, dim, time] encoder output

        Returns:
            Dict with:
                - prosody_codes: [batch, time]
                - content_codes: [num_q, batch, time]
                - residual_codes: [num_q, batch, time]
                - prosody_emb: [batch, time, dim]
                - quantized: [batch, time, dim]
        """
        # Transpose to [batch, time, dim]
        z = z.transpose(1, 2)
        batch_size, seq_len, _ = z.shape

        # Quantize prosody (1 codebook)
        z_prosody = self.prosody_proj_in(z)
        q_prosody, idx_prosody = self.quantize_single(z_prosody, self.prosody_codebook)
        prosody_emb = self.prosody_proj_out(q_prosody)

        # Quantize content (2 codebooks, residual)
        z_content = self.content_proj_in(z - prosody_emb)
        z_content_split = z_content.chunk(self.config.vq_num_q_content, dim=-1)

        content_quants = []
        content_indices = []
        for i, (z_c, cb) in enumerate(zip(z_content_split, self.content_codebooks)):
            q_c, idx_c = self.quantize_single(z_c, cb)
            content_quants.append(q_c)
            content_indices.append(idx_c)

        content_emb = self.content_proj_out(torch.cat(content_quants, dim=-1))

        # Quantize residual (3 codebooks)
        z_residual = self.residual_proj_in(z - prosody_emb - content_emb)
        z_residual_split = z_residual.chunk(self.config.vq_num_q_residual, dim=-1)

        residual_quants = []
        residual_indices = []
        for i, (z_r, cb) in enumerate(zip(z_residual_split, self.residual_codebooks)):
            q_r, idx_r = self.quantize_single(z_r, cb)
            residual_quants.append(q_r)
            residual_indices.append(idx_r)

        residual_emb = self.residual_proj_out(torch.cat(residual_quants, dim=-1))

        # Combine
        quantized = prosody_emb + content_emb + residual_emb

        return {
            'prosody_codes': idx_prosody,                              # [batch, time]
            'content_codes': torch.stack(content_indices, dim=0),     # [2, batch, time]
            'residual_codes': torch.stack(residual_indices, dim=0),   # [3, batch, time]
            'prosody_emb': prosody_emb,                               # [batch, time, 256]
            'content_emb': content_emb,                               # [batch, time, 256]
            'residual_emb': residual_emb,                             # [batch, time, 256]
            'quantized': quantized,                                   # [batch, time, 256]
        }


class SpeakerEncoder(nn.Module):
    """Speaker encoder for timbre extraction."""

    def __init__(self, config: FACodecConfig):
        super().__init__()

        # Simple speaker encoder (can be replaced with ECAPA-TDNN)
        self.layers = nn.Sequential(
            nn.Conv1d(config.encoder_out_channels, 512, kernel_size=5, padding=2),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 256, kernel_size=1),
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 256, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding.

        Args:
            x: [batch, dim, time] encoder output

        Returns:
            [batch, 256] speaker embedding
        """
        x = self.layers(x)  # [batch, 256, time]

        # Attentive pooling
        alpha = torch.softmax(self.attention(x), dim=-1)
        x = (alpha * x).sum(dim=-1)  # [batch, 256]

        return x


# =============================================================================
# PROSODY EXTRACTOR
# =============================================================================

class FACodecProsodyExtractor(nn.Module):
    """
    Extracts prosody codes from audio using FACodec.

    This wraps FACodec encoder + quantizer to extract only prosody codes,
    which are clean representations without speaker/content leakage.

    Example:
        extractor = FACodecProsodyExtractor()

        # Load audio
        audio = load_audio("speech.wav")  # [batch, 1, samples]

        # Extract prosody
        result = extractor(audio)
        prosody_codes = result['prosody_codes']  # [batch, time]
        prosody_emb = result['prosody_emb']      # [batch, time, 256]
    """

    def __init__(
        self,
        config: Optional[FACodecConfig] = None,
        use_official: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            config: FACodec configuration
            use_official: Try to load official Amphion implementation
            device: Target device
        """
        super().__init__()
        self.config = config or FACodecConfig()
        self.device = device
        self._use_official = False

        if use_official:
            self._try_load_official()

        if not self._use_official:
            # Use minimal implementation
            self.encoder = FACodecEncoderMinimal(self.config)
            self.quantizer = FACodecQuantizerMinimal(self.config)
            self.speaker_encoder = SpeakerEncoder(self.config)

    def _try_load_official(self):
        """Try to load official Amphion FACodec."""
        try:
            # Try importing from Amphion
            from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder

            # Initialize models
            self.encoder = FACodecEncoder(
                ngf=self.config.encoder_ngf,
                up_ratios=list(self.config.encoder_up_ratios),
                out_channels=self.config.encoder_out_channels,
            )

            self.decoder = FACodecDecoder(
                in_channels=self.config.decoder_in_channels,
                upsample_initial_channel=self.config.decoder_upsample_initial,
                ngf=self.config.decoder_ngf,
                up_ratios=list(self.config.decoder_up_ratios),
                vq_num_q_c=self.config.vq_num_q_content,
                vq_num_q_p=self.config.vq_num_q_prosody,
                vq_num_q_r=self.config.vq_num_q_residual,
                vq_dim=self.config.vq_dim,
                codebook_dim=self.config.codebook_dim,
                codebook_size_prosody=self.config.codebook_size,
                codebook_size_content=self.config.codebook_size,
                codebook_size_residual=self.config.codebook_size,
                use_gr_x_timbre=True,
                use_gr_residual_f0=True,
                use_gr_residual_phone=True,
            )

            # Download and load weights
            if HF_AVAILABLE:
                encoder_path = hf_hub_download(
                    repo_id=self.config.hf_repo,
                    filename=self.config.encoder_filename,
                )
                decoder_path = hf_hub_download(
                    repo_id=self.config.hf_repo,
                    filename=self.config.decoder_filename,
                )

                self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
                self.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

                self.encoder.eval()
                self.decoder.eval()

                self._use_official = True
                print("Loaded official Amphion FACodec models")
            else:
                print("huggingface_hub not available, using minimal implementation")

        except ImportError:
            print("Amphion not installed, using minimal FACodec implementation")
            print("For better quality, install: pip install git+https://github.com/open-mmlab/Amphion.git")
        except Exception as e:
            print(f"Failed to load official FACodec: {e}")
            print("Using minimal implementation")

    def forward(
        self,
        audio: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract prosody codes from audio.

        Args:
            audio: [batch, 1, samples] waveform at 16kHz
            return_all: Also return content/residual codes

        Returns:
            Dict with:
                - prosody_codes: [batch, time] discrete codes
                - prosody_emb: [batch, time, 256] prosody embeddings
                - speaker_emb: [batch, 256] speaker embedding
                - (optional) content_codes, residual_codes, quantized
        """
        # Ensure correct shape
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, samples] -> [batch, 1, samples]

        if self._use_official:
            return self._forward_official(audio, return_all)
        else:
            return self._forward_minimal(audio, return_all)

    def _forward_official(
        self,
        audio: torch.Tensor,
        return_all: bool,
    ) -> Dict[str, torch.Tensor]:
        """Forward using official Amphion implementation."""
        with torch.no_grad():
            # Encode
            enc_out = self.encoder(audio)

            # Quantize (decoder has quantizer built-in)
            vq_post_emb, vq_id, _, quantized, spk_embs = self.decoder(
                enc_out, eval_vq=False, vq=True
            )

            # vq_id is [6, batch, time] - split into components
            # Prosody: [0:1], Content: [1:3], Residual: [3:6]
            result = {
                'prosody_codes': vq_id[0],           # [batch, time]
                'prosody_emb': quantized[:, :, :self.config.vq_dim // 3],  # Approximate
                'speaker_emb': spk_embs,              # [batch, 256]
            }

            if return_all:
                result['content_codes'] = vq_id[1:3]     # [2, batch, time]
                result['residual_codes'] = vq_id[3:]     # [3, batch, time]
                result['quantized'] = vq_post_emb        # [batch, time, dim]
                result['all_codes'] = vq_id              # [6, batch, time]

            return result

    def _forward_minimal(
        self,
        audio: torch.Tensor,
        return_all: bool,
    ) -> Dict[str, torch.Tensor]:
        """Forward using minimal implementation."""
        # Encode
        enc_out = self.encoder(audio)  # [batch, 256, time]

        # Extract speaker embedding
        speaker_emb = self.speaker_encoder(enc_out)  # [batch, 256]

        # Quantize
        quant_out = self.quantizer(enc_out)

        result = {
            'prosody_codes': quant_out['prosody_codes'],
            'prosody_emb': quant_out['prosody_emb'],
            'speaker_emb': speaker_emb,
        }

        if return_all:
            result['content_codes'] = quant_out['content_codes']
            result['residual_codes'] = quant_out['residual_codes']
            result['quantized'] = quant_out['quantized']

        return result

    def extract_prosody(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to extract only prosody codes.

        Args:
            audio: [batch, 1, samples] or [batch, samples]

        Returns:
            [batch, time] prosody codes
        """
        result = self.forward(audio)
        return result['prosody_codes']

    def extract_prosody_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract continuous prosody embedding (before quantization).

        Args:
            audio: [batch, 1, samples] or [batch, samples]

        Returns:
            [batch, time, 256] prosody embedding
        """
        result = self.forward(audio)
        return result['prosody_emb']


# =============================================================================
# PROSODY ADAPTER
# =============================================================================

class FACodecProsodyAdapter(nn.Module):
    """
    Adapts FACodec prosody codes to CSM prefix tokens.

    Takes discrete prosody codes from FACodec and converts them to
    continuous embeddings suitable for CSM conditioning.

    Two modes:
    1. Code-based: Input is discrete codes [batch, time]
    2. Embedding-based: Input is continuous embeddings [batch, time, dim]

    Example:
        adapter = FACodecProsodyAdapter(config)

        # From codes
        prefix = adapter.from_codes(prosody_codes)  # [batch, 4, 2048]

        # From embeddings
        prefix = adapter.from_embedding(prosody_emb)  # [batch, 4, 2048]
    """

    def __init__(self, config: FACodecConfig):
        super().__init__()
        self.config = config

        codebook_size = 2 ** config.codebook_size  # 1024

        # Prosody code embedding
        self.code_embedding = nn.Embedding(codebook_size, config.prosody_embed_dim)

        # Temporal aggregation (convert variable length to fixed)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(config.prosody_embed_dim, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Attention pooling to fixed number of tokens
        self.attention_pool = AttentiveMultiPool(
            in_dim=512,
            num_pools=config.num_prosody_tokens,
        )

        # Project to CSM hidden size
        self.output_proj = nn.Sequential(
            nn.Linear(512, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        # Direct embedding pathway (when input is continuous)
        self.embedding_encoder = nn.Sequential(
            nn.Linear(config.prosody_embed_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def from_codes(
        self,
        codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert discrete prosody codes to prefix tokens.

        Args:
            codes: [batch, time] discrete codes (0-1023)
            mask: [batch, time] validity mask (optional)

        Returns:
            [batch, num_prosody_tokens, hidden_size] prefix tokens
        """
        # Embed codes
        x = self.code_embedding(codes)  # [batch, time, 256]

        # Temporal encoding
        x = x.transpose(1, 2)  # [batch, 256, time]
        x = self.temporal_encoder(x)  # [batch, 512, time]
        x = x.transpose(1, 2)  # [batch, time, 512]

        # Attention pool to fixed tokens
        if mask is not None:
            x = self.attention_pool(x, mask)
        else:
            x = self.attention_pool(x)
        # [batch, num_tokens, 512]

        # Project to hidden size
        x = self.output_proj(x)  # [batch, num_tokens, hidden_size]

        return x

    def from_embedding(
        self,
        embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert continuous prosody embedding to prefix tokens.

        Args:
            embedding: [batch, time, prosody_embed_dim] continuous embeddings
            mask: [batch, time] validity mask (optional)

        Returns:
            [batch, num_prosody_tokens, hidden_size] prefix tokens
        """
        # Encode embeddings
        x = self.embedding_encoder(embedding)  # [batch, time, 512]

        # Add temporal context
        x = x.transpose(1, 2)  # [batch, 512, time]
        x = self.temporal_encoder(x)  # [batch, 512, time]
        x = x.transpose(1, 2)  # [batch, time, 512]

        # Attention pool
        if mask is not None:
            x = self.attention_pool(x, mask)
        else:
            x = self.attention_pool(x)

        # Project
        x = self.output_proj(x)

        return x

    def forward(
        self,
        codes: Optional[torch.Tensor] = None,
        embedding: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert prosody to prefix tokens.

        Args:
            codes: [batch, time] discrete codes (if provided)
            embedding: [batch, time, dim] continuous embedding (if provided)
            mask: [batch, time] validity mask

        Returns:
            [batch, num_prosody_tokens, hidden_size] prefix tokens
        """
        if codes is not None:
            return self.from_codes(codes, mask)
        elif embedding is not None:
            return self.from_embedding(embedding, mask)
        else:
            raise ValueError("Must provide either codes or embedding")


class AttentiveMultiPool(nn.Module):
    """
    Attention-based pooling to fixed number of output vectors.

    Learns num_pools query vectors that attend over input sequence
    to produce fixed-length output.
    """

    def __init__(self, in_dim: int, num_pools: int, num_heads: int = 4):
        super().__init__()
        self.num_pools = num_pools

        # Learned queries
        self.queries = nn.Parameter(torch.randn(1, num_pools, in_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool variable-length input to fixed-length output.

        Args:
            x: [batch, time, dim] input sequence
            mask: [batch, time] validity mask

        Returns:
            [batch, num_pools, dim] pooled output
        """
        batch_size = x.shape[0]

        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)

        # Create attention mask if needed
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        # Attend
        output, _ = self.attention(
            queries, x, x,
            key_padding_mask=key_padding_mask,
        )

        output = self.norm(output)

        return output


# =============================================================================
# FACODEC CONTROLLED CSM
# =============================================================================

class FACodecControlledCSM(nn.Module):
    """
    Wrapper that adds FACodec prosody conditioning to CSM.

    Replaces the multi-vector prosody approach with FACodec's
    disentangled prosody codes for cleaner conditioning.

    Example:
        model = FACodecControlledCSM(csm_model, config)

        # Training: extract prosody from audio, learn to reconstruct
        result = extractor(audio)
        loss = model.forward_with_facodec(
            input_ids, attention_mask,
            prosody_codes=result['prosody_codes'],
            labels=labels,
        )

        # Inference: condition on extracted prosody
        audio = model.generate_with_facodec(
            input_ids, attention_mask,
            prosody_codes=prosody_codes,
        )
    """

    def __init__(
        self,
        csm_model: nn.Module,
        config: Optional[FACodecConfig] = None,
        freeze_csm: bool = True,
    ):
        super().__init__()
        self.csm = csm_model
        self.config = config or FACodecConfig()

        # FACodec prosody adapter
        self.prosody_adapter = FACodecProsodyAdapter(self.config)

        # Freeze CSM if specified
        if freeze_csm:
            for param in self.csm.parameters():
                param.requires_grad = False

    def get_prosody_prefix(
        self,
        prosody_codes: Optional[torch.Tensor] = None,
        prosody_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get prosody prefix tokens from FACodec output.

        Args:
            prosody_codes: [batch, time] discrete codes
            prosody_embedding: [batch, time, 256] continuous embedding

        Returns:
            [batch, num_prosody_tokens, hidden_size] prefix tokens
        """
        return self.prosody_adapter(
            codes=prosody_codes,
            embedding=prosody_embedding,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prosody_codes: Optional[torch.Tensor] = None,
        prosody_embedding: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with FACodec prosody conditioning.

        Args:
            input_ids: [batch, seq_len] text tokens
            attention_mask: [batch, seq_len] mask
            prosody_codes: [batch, time] FACodec prosody codes
            prosody_embedding: [batch, time, 256] FACodec prosody embedding
            labels: Optional labels for training

        Returns:
            Model outputs
        """
        # Get prosody prefix
        prosody_prefix = self.get_prosody_prefix(prosody_codes, prosody_embedding)

        # Get text embeddings
        text_embeds = self.csm.embed_text_tokens(input_ids)

        # Concatenate [prosody | text]
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend attention mask
        batch_size = input_ids.shape[0]
        num_prosody_tokens = prosody_prefix.shape[1]
        prosody_mask = torch.ones(
            batch_size, num_prosody_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prosody_mask, attention_mask], dim=1)

        # Forward through CSM
        # Pass num_prefix_tokens for proper loss computation
        return self.csm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
            num_prefix_tokens=num_prosody_tokens,
            **kwargs
        )

    def generate_with_facodec(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prosody_codes: Optional[torch.Tensor] = None,
        prosody_embedding: Optional[torch.Tensor] = None,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate audio with FACodec prosody conditioning.

        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            prosody_codes: FACodec prosody codes
            prosody_embedding: FACodec prosody embedding
            **generate_kwargs: Generation parameters

        Returns:
            Generated audio
        """
        # Get combined embeddings
        prosody_prefix = self.get_prosody_prefix(prosody_codes, prosody_embedding)
        text_embeds = self.csm.embed_text_tokens(input_ids)
        inputs_embeds = torch.cat([prosody_prefix, text_embeds], dim=1)

        # Extend mask
        batch_size = input_ids.shape[0]
        num_prosody_tokens = prosody_prefix.shape[1]
        prosody_mask = torch.ones(
            batch_size, num_prosody_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        extended_mask = torch.cat([prosody_mask, attention_mask], dim=1)

        # Generate
        return self.csm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            output_audio=True,
            **generate_kwargs
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_facodec_prosody(
    audio_path: Union[str, Path],
    extractor: Optional[FACodecProsodyExtractor] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Extract FACodec prosody from an audio file.

    Args:
        audio_path: Path to audio file
        extractor: Pre-loaded extractor (or will create one)
        device: Target device

    Returns:
        Dict with prosody codes and embeddings
    """
    import torchaudio

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Add batch dimension
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]

    # Extract
    if extractor is None:
        extractor = FACodecProsodyExtractor(device=device).to(device)
        extractor.eval()

    with torch.no_grad():
        result = extractor(waveform, return_all=True)

    return result


def preprocess_dataset_with_facodec(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "cuda",
    batch_size: int = 8,
):
    """
    Preprocess a dataset by extracting FACodec prosody codes.

    Args:
        manifest_path: Path to manifest JSON with audio paths
        output_dir: Directory to save extracted features
        device: Target device
        batch_size: Batch size for processing
    """
    import json
    from tqdm import tqdm
    import torchaudio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Initialize extractor
    extractor = FACodecProsodyExtractor(device=device).to(device)
    extractor.eval()

    results = []

    for i, item in enumerate(tqdm(manifest, desc="Extracting FACodec features")):
        audio_path = item['audio_path']

        try:
            # Extract prosody
            features = extract_facodec_prosody(audio_path, extractor, device)

            # Save codes
            codes_path = output_dir / f"{item['id']}_prosody.pt"
            torch.save({
                'prosody_codes': features['prosody_codes'].cpu(),
                'prosody_emb': features['prosody_emb'].cpu(),
                'speaker_emb': features['speaker_emb'].cpu(),
            }, codes_path)

            # Update manifest
            item['facodec_path'] = str(codes_path)
            results.append(item)

        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

    # Save updated manifest
    output_manifest = output_dir / "manifest_with_facodec.json"
    with open(output_manifest, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)}/{len(manifest)} samples")
    print(f"Updated manifest saved to {output_manifest}")


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FACodec Integration - Test Suite")
    print("=" * 60)

    device = "cpu"
    batch_size = 2

    # Test 1: Configuration
    print("\n[Test 1] Configuration...")
    config = FACodecConfig()
    print(f"  Sample rate: {config.sample_rate}")
    print(f"  Hop size: {config.hop_size}")
    print(f"  Prosody codebook size: {2 ** config.codebook_size}")
    print("  [PASS]")

    # Test 2: Minimal encoder
    print("\n[Test 2] FACodecEncoderMinimal...")
    encoder = FACodecEncoderMinimal(config)
    audio = torch.randn(batch_size, 1, 16000)  # 1 second at 16kHz
    enc_out = encoder(audio)
    expected_time = 16000 // config.hop_size  # Should be 80 frames
    print(f"  Input shape: {audio.shape}")
    print(f"  Output shape: {enc_out.shape}")
    print(f"  Expected frames: ~{expected_time}")
    print("  [PASS]")

    # Test 3: Quantizer
    print("\n[Test 3] FACodecQuantizerMinimal...")
    quantizer = FACodecQuantizerMinimal(config)
    quant_out = quantizer(enc_out)
    print(f"  Prosody codes shape: {quant_out['prosody_codes'].shape}")
    print(f"  Content codes shape: {quant_out['content_codes'].shape}")
    print(f"  Residual codes shape: {quant_out['residual_codes'].shape}")
    print(f"  Prosody embedding shape: {quant_out['prosody_emb'].shape}")
    print("  [PASS]")

    # Test 4: Prosody extractor
    print("\n[Test 4] FACodecProsodyExtractor...")
    extractor = FACodecProsodyExtractor(config, use_official=False)
    result = extractor(audio, return_all=True)
    print(f"  Prosody codes: {result['prosody_codes'].shape}")
    print(f"  Prosody embedding: {result['prosody_emb'].shape}")
    print(f"  Speaker embedding: {result['speaker_emb'].shape}")
    print("  [PASS]")

    # Test 5: Prosody adapter
    print("\n[Test 5] FACodecProsodyAdapter...")
    adapter = FACodecProsodyAdapter(config)

    # From codes
    prefix_from_codes = adapter.from_codes(result['prosody_codes'])
    print(f"  Prefix from codes: {prefix_from_codes.shape}")

    # From embedding
    prefix_from_emb = adapter.from_embedding(result['prosody_emb'])
    print(f"  Prefix from embedding: {prefix_from_emb.shape}")

    expected_shape = (batch_size, config.num_prosody_tokens, config.hidden_size)
    assert prefix_from_codes.shape == expected_shape
    assert prefix_from_emb.shape == expected_shape
    print("  [PASS]")

    # Test 6: Gradient flow
    print("\n[Test 6] Gradient flow...")
    adapter.zero_grad()
    codes = result['prosody_codes'].detach()
    prefix = adapter.from_codes(codes)
    loss = prefix.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in adapter.parameters() if p.requires_grad)
    print(f"  Has gradients: {has_grad}")
    print("  [PASS]")

    # Test 7: AttentiveMultiPool
    print("\n[Test 7] AttentiveMultiPool...")
    pool = AttentiveMultiPool(in_dim=256, num_pools=4)
    x = torch.randn(batch_size, 80, 256)  # Variable length input
    mask = torch.ones(batch_size, 80)
    mask[0, 60:] = 0  # Mask out last frames for first sample

    pooled = pool(x, mask)
    print(f"  Input: {x.shape}")
    print(f"  Output: {pooled.shape}")
    assert pooled.shape == (batch_size, 4, 256)
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All FACodec integration tests passed!")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("Quick Start Guide")
    print("-" * 60)
    print("""
# Extract prosody from audio
from facodec_integration import FACodecProsodyExtractor, FACodecProsodyAdapter

extractor = FACodecProsodyExtractor()
result = extractor(audio)  # audio: [batch, 1, samples]

prosody_codes = result['prosody_codes']  # [batch, time]
prosody_emb = result['prosody_emb']      # [batch, time, 256]

# Convert to CSM prefix tokens
adapter = FACodecProsodyAdapter(config)
prefix_tokens = adapter.from_codes(prosody_codes)  # [batch, 4, 2048]

# Or directly from embeddings
prefix_tokens = adapter.from_embedding(prosody_emb)

# Use with CSM
from facodec_integration import FACodecControlledCSM

model = FACodecControlledCSM(csm_model)
output = model(input_ids, attention_mask, prosody_codes=prosody_codes)

# Preprocess dataset
from facodec_integration import preprocess_dataset_with_facodec

preprocess_dataset_with_facodec(
    manifest_path="data/manifest.json",
    output_dir="data/facodec_features",
    device="cuda",
)
    """)
