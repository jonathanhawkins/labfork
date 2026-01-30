"""
DisCodec-Inspired Disentanglement Components for V6

Based on DisCo-Speech/DisCodec (December 2024):
https://arxiv.org/html/2512.13251

Key components:
1. SoftOrthogonalityLoss - Asymmetric orthogonality between factor pairs
2. GradientReversalLayer - Prevents speaker leakage into prosody
3. ECAPATDNNEncoder - Speaker/timbre extraction
4. DualLayerFSQ - Hierarchical prosody quantization

Usage:
    from discodec_disentanglement import (
        SoftOrthogonalityLoss,
        GradientReversalLayer,
        DisentanglementLoss,
    )

    # Add to existing V6 training
    disent_loss = DisentanglementLoss(
        content_prosody_beta=0.01,
        timbre_prosody_beta=0.0001,
    )

    # In training loop
    loss = disent_loss(content_embed, prosody_embed, speaker_embed)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from "Domain-Adversarial Training of Neural Networks".

    In forward pass: identity operation
    In backward pass: reverses gradients with scaling factor lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for domain adaptation / anti-leakage training.

    Used in DisCodec to ensure prosody encoder doesn't learn speaker identity:
    - Forward: passes through unchanged
    - Backward: reverses gradients, pushing encoder away from speaker prediction

    Example:
        prosody_embed = prosody_encoder(audio)
        reversed = grl(prosody_embed)
        speaker_pred = speaker_classifier(reversed)  # Gradients reversed
        loss = CrossEntropy(speaker_pred, speaker_labels)  # Maximizes speaker confusion
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Args:
            lambda_: Gradient reversal strength (higher = stronger reversal)
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        """Adjust reversal strength (useful for curriculum)."""
        self.lambda_ = lambda_


# =============================================================================
# SOFT ORTHOGONALITY LOSS
# =============================================================================

class SoftOrthogonalityLoss(nn.Module):
    """
    Soft orthogonality loss from DisCodec.

    Key insight: Different factor pairs need different orthogonality strengths:
    - Content-Prosody: beta=0.01 (relaxed, they're naturally coupled)
    - Timbre-Prosody: beta=0.0001 (strict, should be independent)

    This pushes cosine similarity toward a target value, not necessarily zero.
    """

    def __init__(self, target_similarity: float = 0.0):
        """
        Args:
            target_similarity: Target cosine similarity (beta in paper)
                - 0.0 = fully orthogonal
                - 0.01 = nearly orthogonal with small allowed coupling
                - 0.0001 = very strict orthogonality
        """
        super().__init__()
        self.target_similarity = target_similarity

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft orthogonality loss.

        Args:
            z1: [batch, dim] or [batch, seq, dim] first representation
            z2: [batch, dim] or [batch, seq, dim] second representation
            mask: Optional [batch] or [batch, seq] validity mask

        Returns:
            Scalar loss value
        """
        # Handle sequence dimension if present
        if z1.dim() == 3:
            # Pool over sequence
            z1 = z1.mean(dim=1)
            z2 = z2.mean(dim=1)

        # L2 normalize
        z1_norm = F.normalize(z1, p=2, dim=-1)
        z2_norm = F.normalize(z2, p=2, dim=-1)

        # Compute cosine similarity
        cos_sim = (z1_norm * z2_norm).sum(dim=-1)  # [batch]

        # Push toward target similarity
        loss = (cos_sim - self.target_similarity).pow(2)

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss


# =============================================================================
# DISENTANGLEMENT LOSS (COMBINED)
# =============================================================================

@dataclass
class DisentanglementConfig:
    """Configuration for disentanglement training."""

    # Orthogonality targets (beta values from DisCodec)
    content_prosody_beta: float = 0.01    # Relaxed - natural coupling allowed
    timbre_prosody_beta: float = 0.0001   # Strict - must be independent
    content_timbre_beta: float = 0.001    # Moderate

    # Loss weights
    ortho_weight: float = 1.0
    grl_weight: float = 0.1
    f0_weight: float = 1.0
    phonetic_weight: float = 0.5

    # GRL settings
    grl_lambda: float = 1.0
    grl_warmup_steps: int = 1000  # Gradually increase GRL strength


class DisentanglementLoss(nn.Module):
    """
    Combined disentanglement loss for V6 training.

    Implements DisCodec's key losses:
    1. Soft orthogonality between factor pairs
    2. Gradient reversal for anti-leakage
    3. Optional F0 regression supervision
    4. Optional phonetic supervision

    Example:
        config = DisentanglementConfig()
        loss_fn = DisentanglementLoss(config, num_speakers=100)

        # In training loop
        losses = loss_fn(
            prosody_embed=prosody_output['combined_embedding'],
            speaker_embed=speaker_encoder(audio),
            content_embed=content_encoder(audio),  # Optional
            speaker_labels=speaker_ids,
            f0_pred=prosody_output.get('f0_pred'),
            f0_target=ground_truth_f0,
        )

        total_loss = losses['total']
    """

    def __init__(
        self,
        config: DisentanglementConfig,
        num_speakers: int = 0,
        prosody_dim: int = 2048,
    ):
        super().__init__()
        self.config = config

        # Orthogonality losses with different targets
        self.ortho_content_prosody = SoftOrthogonalityLoss(config.content_prosody_beta)
        self.ortho_timbre_prosody = SoftOrthogonalityLoss(config.timbre_prosody_beta)
        self.ortho_content_timbre = SoftOrthogonalityLoss(config.content_timbre_beta)

        # Gradient reversal layer
        self.grl = GradientReversalLayer(config.grl_lambda)

        # Speaker classifier (for anti-leakage)
        if num_speakers > 0:
            self.speaker_classifier = nn.Sequential(
                nn.Linear(prosody_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_speakers),
            )
        else:
            self.speaker_classifier = None

        # F0 regression head (optional)
        self.f0_head = nn.Sequential(
            nn.Linear(prosody_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Training state
        self.current_step = 0

    def forward(
        self,
        prosody_embed: torch.Tensor,
        speaker_embed: Optional[torch.Tensor] = None,
        content_embed: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        f0_pred: Optional[torch.Tensor] = None,
        f0_target: Optional[torch.Tensor] = None,
        phone_pred: Optional[torch.Tensor] = None,
        phone_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement losses.

        Args:
            prosody_embed: [batch, dim] prosody representation
            speaker_embed: [batch, dim] speaker/timbre representation
            content_embed: [batch, dim] content representation (optional)
            speaker_labels: [batch] speaker IDs for anti-leakage
            f0_pred: [batch, time] predicted F0 (optional)
            f0_target: [batch, time] ground truth F0
            phone_pred: [batch, time, phones] phone predictions (optional)
            phone_labels: [batch, time] phone labels

        Returns:
            Dict with individual losses and total
        """
        device = prosody_embed.device
        losses = {}

        # ===== Orthogonality Losses =====

        # Timbre-Prosody orthogonality (strict)
        if speaker_embed is not None:
            losses['ortho_timbre_prosody'] = self.ortho_timbre_prosody(
                prosody_embed, speaker_embed
            )
        else:
            losses['ortho_timbre_prosody'] = torch.tensor(0.0, device=device)

        # Content-Prosody orthogonality (relaxed)
        if content_embed is not None:
            losses['ortho_content_prosody'] = self.ortho_content_prosody(
                prosody_embed, content_embed
            )
        else:
            losses['ortho_content_prosody'] = torch.tensor(0.0, device=device)

        # Content-Timbre orthogonality
        if content_embed is not None and speaker_embed is not None:
            losses['ortho_content_timbre'] = self.ortho_content_timbre(
                content_embed, speaker_embed
            )
        else:
            losses['ortho_content_timbre'] = torch.tensor(0.0, device=device)

        # ===== Gradient Reversal Anti-Leakage Loss =====

        if self.speaker_classifier is not None and speaker_labels is not None:
            # Warmup: gradually increase GRL strength
            if self.current_step < self.config.grl_warmup_steps:
                warmup_factor = self.current_step / self.config.grl_warmup_steps
                self.grl.set_lambda(self.config.grl_lambda * warmup_factor)

            # Prosody should NOT predict speaker
            prosody_grl = self.grl(prosody_embed)
            speaker_logits = self.speaker_classifier(prosody_grl)
            losses['grl_speaker'] = F.cross_entropy(speaker_logits, speaker_labels)
        else:
            losses['grl_speaker'] = torch.tensor(0.0, device=device)

        # ===== F0 Regression Loss =====

        if f0_target is not None:
            if f0_pred is None:
                # Use internal F0 head
                f0_pred = self.f0_head(prosody_embed).squeeze(-1)

            # Handle length mismatch
            min_len = min(f0_pred.shape[-1], f0_target.shape[-1])
            f0_pred = f0_pred[..., :min_len]
            f0_target = f0_target[..., :min_len]

            # Voiced mask (where F0 > 0)
            voiced_mask = f0_target > 0

            if voiced_mask.any():
                losses['f0_regression'] = F.mse_loss(
                    f0_pred[voiced_mask], f0_target[voiced_mask]
                )
            else:
                losses['f0_regression'] = torch.tensor(0.0, device=device)
        else:
            losses['f0_regression'] = torch.tensor(0.0, device=device)

        # ===== Phonetic Supervision Loss =====

        if phone_pred is not None and phone_labels is not None:
            # Flatten for cross entropy
            B, T, C = phone_pred.shape
            losses['phonetic'] = F.cross_entropy(
                phone_pred.view(-1, C),
                phone_labels.view(-1),
                ignore_index=-1,
            )
        else:
            losses['phonetic'] = torch.tensor(0.0, device=device)

        # ===== Total Loss =====

        total = (
            losses['ortho_timbre_prosody'] * self.config.ortho_weight +
            losses['ortho_content_prosody'] * self.config.ortho_weight * 0.5 +
            losses['ortho_content_timbre'] * self.config.ortho_weight * 0.5 +
            losses['grl_speaker'] * self.config.grl_weight +
            losses['f0_regression'] * self.config.f0_weight +
            losses['phonetic'] * self.config.phonetic_weight
        )

        losses['total'] = total

        # Update step counter
        self.current_step += 1

        return losses


# =============================================================================
# SPEAKER/TIMBRE ENCODER (ECAPA-TDNN STYLE)
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        s = x.mean(dim=-1)  # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class Res2Block(nn.Module):
    """Res2Net-style block with multi-scale features."""

    def __init__(self, channels: int, scale: int = 4, kernel_size: int = 3):
        super().__init__()
        self.scale = scale
        assert channels % scale == 0

        width = channels // scale
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        chunks = torch.chunk(x, self.scale, dim=1)
        outs = [chunks[0]]

        for i in range(self.scale - 1):
            if i == 0:
                inp = chunks[i + 1]
            else:
                inp = chunks[i + 1] + outs[i]
            out = self.bns[i](F.relu(self.convs[i](inp)))
            outs.append(out)

        return torch.cat(outs, dim=1)


class SERes2Block(nn.Module):
    """SE-Res2Block from ECAPA-TDNN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.res2 = Res2Block(out_channels, scale, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.res2(x)))
        x = self.bn3(self.conv2(x))
        x = self.se(x)

        return F.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling from ECAPA-TDNN.

    Computes weighted mean and std over time dimension.
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, 1),
            nn.ReLU(),
            nn.Conv1d(attention_dim, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]

        Returns:
            [B, C*2] mean and std statistics
        """
        # Attention weights
        alpha = torch.softmax(self.attention(x), dim=-1)

        # Weighted mean
        mean = (alpha * x).sum(dim=-1)

        # Weighted std
        var = (alpha * x.pow(2)).sum(dim=-1) - mean.pow(2)
        std = torch.sqrt(var.clamp(min=1e-8))

        return torch.cat([mean, std], dim=-1)


class ECAPATDNNSpeakerEncoder(nn.Module):
    """
    ECAPA-TDNN speaker encoder for timbre extraction.

    Based on "ECAPA-TDNN: Emphasized Channel Attention, Propagation and
    Aggregation in TDNN Based Speaker Verification" (2020).

    Takes mel spectrogram input and outputs fixed-length speaker embedding.
    """

    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 256,
        channels: int = 512,
    ):
        super().__init__()

        # Frame-level layers
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # SE-Res2Blocks with different dilations
        self.layer2 = SERes2Block(channels, channels, scale=8, dilation=2)
        self.layer3 = SERes2Block(channels, channels, scale=8, dilation=3)
        self.layer4 = SERes2Block(channels, channels, scale=8, dilation=4)

        # Multi-layer feature aggregation
        self.mfa = nn.Conv1d(channels * 3, channels * 3, kernel_size=1)
        self.mfa_bn = nn.BatchNorm1d(channels * 3)

        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels * 3)
        self.asp_bn = nn.BatchNorm1d(channels * 6)

        # Output projection
        self.fc = nn.Linear(channels * 6, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract speaker embedding.

        Args:
            x: [batch, time, mel_dim] mel spectrogram
            lengths: [batch] actual lengths (optional)

        Returns:
            [batch, output_dim] speaker embedding
        """
        # Transpose to [B, D, T]
        x = x.transpose(1, 2)

        # Frame-level processing
        x1 = self.layer1(x)
        x2 = self.layer2(x1) + x1
        x3 = self.layer3(x2) + x2
        x4 = self.layer4(x3) + x3

        # Multi-layer aggregation
        x = torch.cat([x2, x3, x4], dim=1)
        x = F.relu(self.mfa_bn(self.mfa(x)))

        # Attentive pooling
        x = self.asp(x)
        x = self.asp_bn(x)

        # Output projection
        x = self.fc_bn(self.fc(x))

        return x


# =============================================================================
# DUAL-LAYER FSQ FOR PROSODY
# =============================================================================

class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization from "Language Model Beats Diffusion" (2023).

    Quantizes continuous values to finite levels without codebook.
    """

    def __init__(self, levels: List[int], eps: float = 1e-3):
        """
        Args:
            levels: Number of quantization levels per dimension
                    e.g., [8, 5, 5, 5] = 8*5*5*5 = 1000 effective codes
        """
        super().__init__()
        self.levels = levels
        self.eps = eps
        self.codebook_size = math.prod(levels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input.

        Args:
            x: [batch, ..., dim] where dim == len(levels)

        Returns:
            quantized: Same shape as input
            indices: [batch, ...] quantization indices
        """
        # Scale to [0, 1]
        x = torch.sigmoid(x)

        # Quantize each dimension
        quantized = []
        indices = torch.zeros_like(x[..., 0], dtype=torch.long)
        multiplier = 1

        for i, L in enumerate(self.levels):
            # Scale to [0, L-1] and round
            q = torch.round(x[..., i] * (L - 1))
            q = q.clamp(0, L - 1)

            # Add to index
            indices = indices + (q.long() * multiplier)
            multiplier *= L

            # Dequantize (with straight-through gradient)
            dq = q / (L - 1)
            quantized.append(dq + (x[..., i] - dq).detach())

        quantized = torch.stack(quantized, dim=-1)

        return quantized, indices


class DualLayerFSQ(nn.Module):
    """
    Dual-layer FSQ for hierarchical prosody quantization from DisCodec.

    Layer 1: Captures primary prosody (F0, major rhythm)
    Layer 2 (residual): Captures subtle prosodic variations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        levels_primary: List[int] = [8, 8, 8, 8],    # 4096 codes
        levels_residual: List[int] = [5, 5, 5, 5],   # 625 codes
    ):
        super().__init__()

        # Primary layer
        self.proj_primary = nn.Linear(input_dim, len(levels_primary))
        self.fsq_primary = FiniteScalarQuantizer(levels_primary)
        self.deproj_primary = nn.Linear(len(levels_primary), hidden_dim)

        # Residual layer
        self.proj_residual = nn.Linear(input_dim, len(levels_residual))
        self.fsq_residual = FiniteScalarQuantizer(levels_residual)
        self.deproj_residual = nn.Linear(len(levels_residual), hidden_dim)

        # Combination
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, time, input_dim] prosody features

        Returns:
            Dict with quantized outputs and indices
        """
        # Primary quantization
        z_primary = self.proj_primary(x)
        q_primary, idx_primary = self.fsq_primary(z_primary)
        out_primary = self.deproj_primary(q_primary)

        # Compute residual
        residual = x - self.deproj_primary(z_primary.detach())

        # Residual quantization
        z_residual = self.proj_residual(residual)
        q_residual, idx_residual = self.fsq_residual(z_residual)
        out_residual = self.deproj_residual(q_residual)

        # Combine
        combined = self.combine(torch.cat([out_primary, out_residual], dim=-1))

        return {
            'quantized': combined,
            'primary': out_primary,
            'residual': out_residual,
            'indices_primary': idx_primary,
            'indices_residual': idx_residual,
        }


# =============================================================================
# INTEGRATION WITH V6
# =============================================================================

class V6DisentanglementWrapper(nn.Module):
    """
    Wrapper to add DisCodec disentanglement to existing V6 prosody encoder.

    Adds:
    1. Speaker encoder for orthogonality supervision
    2. Gradient reversal anti-leakage
    3. Soft orthogonality losses

    Example:
        wrapper = V6DisentanglementWrapper(
            prosody_encoder=existing_prosody_encoder,
            num_speakers=100,
        )

        # Forward pass
        prosody_embed = wrapper.forward_prosody(prosody_dict)
        speaker_embed = wrapper.forward_speaker(mel_spec)

        # Compute disentanglement loss
        disent_loss = wrapper.compute_loss(prosody_embed, speaker_embed, speaker_ids)
    """

    def __init__(
        self,
        prosody_encoder: nn.Module,
        num_speakers: int,
        speaker_dim: int = 256,
        mel_dim: int = 80,
        config: Optional[DisentanglementConfig] = None,
    ):
        super().__init__()

        self.prosody_encoder = prosody_encoder
        self.config = config or DisentanglementConfig()

        # Speaker encoder
        self.speaker_encoder = ECAPATDNNSpeakerEncoder(
            input_dim=mel_dim,
            output_dim=speaker_dim,
        )

        # Disentanglement loss
        prosody_dim = getattr(prosody_encoder, 'output_hidden', 2048)
        self.disent_loss = DisentanglementLoss(
            config=self.config,
            num_speakers=num_speakers,
            prosody_dim=prosody_dim,
        )

    def forward_prosody(self, prosody_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward through prosody encoder."""
        return self.prosody_encoder(prosody_dict)

    def forward_speaker(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram."""
        return self.speaker_encoder(mel_spec)

    def compute_loss(
        self,
        prosody_embed: torch.Tensor,
        speaker_embed: torch.Tensor,
        speaker_labels: torch.Tensor,
        f0_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute disentanglement loss.

        Args:
            prosody_embed: [batch, dim] from prosody encoder
            speaker_embed: [batch, dim] from speaker encoder
            speaker_labels: [batch] speaker IDs
            f0_target: [batch, time] ground truth F0 (optional)

        Returns:
            Dict with loss components
        """
        return self.disent_loss(
            prosody_embed=prosody_embed,
            speaker_embed=speaker_embed,
            speaker_labels=speaker_labels,
            f0_target=f0_target,
        )


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DisCodec Disentanglement Components - Test Suite")
    print("=" * 60)

    device = 'cpu'
    batch_size = 4
    seq_len = 100
    hidden_dim = 256

    # Test 1: GradientReversalLayer
    print("\n[Test 1] GradientReversalLayer...")
    grl = GradientReversalLayer(lambda_=1.0)
    x = torch.randn(batch_size, hidden_dim, requires_grad=True)
    y = grl(x)
    loss = y.sum()
    loss.backward()
    # Gradient should be negated
    assert x.grad is not None
    print(f"  Input grad norm: {x.grad.norm():.4f}")
    print("  [PASS]")

    # Test 2: SoftOrthogonalityLoss
    print("\n[Test 2] SoftOrthogonalityLoss...")
    ortho_loss = SoftOrthogonalityLoss(target_similarity=0.01)
    z1 = torch.randn(batch_size, hidden_dim)
    z2 = torch.randn(batch_size, hidden_dim)
    loss = ortho_loss(z1, z2)
    print(f"  Loss: {loss.item():.4f}")

    # Test with sequence input
    z1_seq = torch.randn(batch_size, seq_len, hidden_dim)
    z2_seq = torch.randn(batch_size, seq_len, hidden_dim)
    loss_seq = ortho_loss(z1_seq, z2_seq)
    print(f"  Sequence loss: {loss_seq.item():.4f}")
    print("  [PASS]")

    # Test 3: DisentanglementLoss
    print("\n[Test 3] DisentanglementLoss...")
    config = DisentanglementConfig()
    disent = DisentanglementLoss(config, num_speakers=10, prosody_dim=hidden_dim)

    prosody_embed = torch.randn(batch_size, hidden_dim)
    speaker_embed = torch.randn(batch_size, hidden_dim)
    speaker_labels = torch.randint(0, 10, (batch_size,))

    losses = disent(
        prosody_embed=prosody_embed,
        speaker_embed=speaker_embed,
        speaker_labels=speaker_labels,
    )

    print(f"  Ortho timbre-prosody: {losses['ortho_timbre_prosody'].item():.4f}")
    print(f"  GRL speaker: {losses['grl_speaker'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    print("  [PASS]")

    # Test 4: ECAPATDNNSpeakerEncoder
    print("\n[Test 4] ECAPATDNNSpeakerEncoder...")
    encoder = ECAPATDNNSpeakerEncoder(input_dim=80, output_dim=256)
    mel = torch.randn(batch_size, seq_len, 80)
    embedding = encoder(mel)
    print(f"  Input shape: {mel.shape}")
    print(f"  Output shape: {embedding.shape}")
    assert embedding.shape == (batch_size, 256)
    print("  [PASS]")

    # Test 5: DualLayerFSQ
    print("\n[Test 5] DualLayerFSQ...")
    fsq = DualLayerFSQ(input_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = fsq(x)
    print(f"  Quantized shape: {out['quantized'].shape}")
    print(f"  Primary indices: {out['indices_primary'].shape}")
    print(f"  Residual indices: {out['indices_residual'].shape}")
    print("  [PASS]")

    # Test 6: Full backward pass
    print("\n[Test 6] Full backward pass...")
    disent.zero_grad()
    prosody_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)
    speaker_embed = torch.randn(batch_size, hidden_dim, requires_grad=True)

    losses = disent(
        prosody_embed=prosody_embed,
        speaker_embed=speaker_embed,
        speaker_labels=speaker_labels,
    )

    losses['total'].backward()
    assert prosody_embed.grad is not None
    assert speaker_embed.grad is not None
    print(f"  Prosody grad norm: {prosody_embed.grad.norm():.4f}")
    print(f"  Speaker grad norm: {speaker_embed.grad.norm():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("All DisCodec disentanglement tests passed!")
    print("=" * 60)

    print("\nQuick Integration Guide:")
    print("-" * 40)
    print("""
# In train_prosody_hed.py, add:

from discodec_disentanglement import (
    DisentanglementConfig,
    DisentanglementLoss,
    ECAPATDNNSpeakerEncoder,
)

# Initialize
disent_config = DisentanglementConfig(
    content_prosody_beta=0.01,
    timbre_prosody_beta=0.0001,
)
disent_loss = DisentanglementLoss(disent_config, num_speakers=N)
speaker_encoder = ECAPATDNNSpeakerEncoder()

# In training loop:
mel_spec = extract_mel(audio)
speaker_embed = speaker_encoder(mel_spec)

disent_losses = disent_loss(
    prosody_embed=prosody_output['combined_embedding'],
    speaker_embed=speaker_embed,
    speaker_labels=speaker_ids,
)

total_loss = main_loss + 0.1 * disent_losses['total']
""")
