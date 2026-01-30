# DisCodec Two-Stage Disentanglement Approach

## Research Summary

**Paper**: DisCo-Speech/DisCodec (December 2024)
**Source**: https://arxiv.org/html/2512.13251
**Task**: #12 - Explore DisCodec two-stage disentanglement approach

---

## 1. Core Innovation: Tri-Factor Disentanglement

DisCodec explicitly disentangles speech into three independent factors:

| Factor | What it Captures | Encoder Architecture |
|--------|------------------|---------------------|
| **Content** | Linguistic/phonetic information | DAC-style conv blocks + FSQ quantization |
| **Prosody** | F0, rhythm, emphasis | Dilated causal convs + dual-layer residual FSQ |
| **Timbre** | Speaker identity | ECAPA-TDNN + cross-attention pooling |

### Key Insight: Asymmetric Disentanglement

The critical finding is that different factor pairs require different orthogonality strengths:

```
Content ↔ Prosody:  β = 0.01    (relaxed - they're naturally coupled)
Timbre ↔ Prosody:   β = 0.0001 (strict - must be independent)
```

**Why?** Content and prosody are both temporal-dynamic and intrinsically coupled (emphasis on certain words is content-driven). But timbre and prosody should be theoretically independent - the same prosodic pattern should work for any speaker.

---

## 2. Two-Stage Training Architecture

### Stage 1: Tri-Factor Disentanglement

```
Audio Waveform
      │
      ├──→ Content Encoder (Ec)  ──→ FSQ Quantize ──→ Zc
      │        ↓
      │    Phonetic Loss (CE vs phone labels)
      │
      ├──→ Prosody Encoder (Ep) ──→ Dual-FSQ ──→ Zp
      │        ↓                      │
      │    F0 Regression Loss    Correlation Loss
      │
      └──→ Timbre Encoder (Et)  ──→ ECAPA-TDNN ──→ Zt
               ↓
           Speaker CE Loss + GRL (anti-leakage)

Soft Orthogonality Constraints:
  cos(Zc, Zp) → βc = 0.01
  cos(Zt, Zp) → βt = 0.0001
```

### Stage 2: Fusion and Reconstruction

```
Zc + Zp (content + prosody)
    │
    ↓
Re-Quantize (unified token sequence)
    │
    ↓
Transformer LM Decoder
    │
    + Zt (timbre injection)
    │
    ↓
BigVGAN-v2 Generator
    │
    ↓
Reconstructed Audio

Losses: Multi-scale reconstruction + Adversarial
```

---

## 3. Loss Functions in Detail

### 3.1 Phonetic Supervision Loss (Content Purity)

Uses a finetuned Wav2Vec-based phone recognition model:

```python
L_phonetic = CrossEntropy(phone_classifier(Zc), phone_labels)
```

This ensures Zc captures *only* linguistic content, not prosodic variation.

### 3.2 F0 Regression Loss (Prosody Core)

Dual-layer FSQ for prosody:
- **Layer 1**: Primary prosody (pitch information)
- **Layer 2**: Residual prosody (subtle variations)

```python
# Primary F0 loss
L_f0 = L2(predict_f0(Zp_layer1), ground_truth_f0)

# Correlation loss for residual layer
L_corr = 1 - correlation(Zp_layer2, prosody_residual)
```

### 3.3 Soft Orthogonality Loss

```python
def soft_orthogonality_loss(Z1, Z2, target_similarity):
    """
    Push cosine similarity toward target (not necessarily zero).

    Args:
        Z1, Z2: Encoded representations [B, D]
        target_similarity: β value (0.01 for content-prosody, 0.0001 for timbre-prosody)
    """
    cos_sim = F.cosine_similarity(Z1, Z2, dim=-1)
    return (cos_sim - target_similarity).pow(2).mean()

# Usage
L_ortho_cp = soft_orthogonality_loss(Zc, Zp, beta_c=0.01)
L_ortho_tp = soft_orthogonality_loss(Zt, Zp, beta_t=0.0001)
```

### 3.4 Speaker Classification + Gradient Reversal

Prevents timbre leakage into prosody:

```python
class GradientReversalLayer(nn.Module):
    def forward(self, x):
        return x

    def backward(self, grad):
        return -grad  # Reverse gradients

# Anti-leakage: prosody should NOT predict speaker
L_grl = CrossEntropy(speaker_classifier(GRL(Zp)), speaker_labels)
```

---

## 4. Application to V6 Architecture

### 4.1 Current V6 Limitations

1. **No explicit disentanglement** - Prosody encoder may capture speaker characteristics
2. **Single-stage training** - Content/prosody/timbre mixed during optimization
3. **No orthogonality constraints** - Representations may overlap

### 4.2 Proposed Integration

#### Option A: Add Orthogonality to Existing Encoders

Minimal change - add soft orthogonality losses to current training:

```python
class V6WithOrthogonality(nn.Module):
    def __init__(self):
        # Existing V6 components
        self.prosody_encoder = ProsodyEncoder(config)
        self.hed_encoder = HierarchicalEmotionEncoder(hed_config)

        # NEW: Add speaker encoder for orthogonality
        self.speaker_encoder = ECAPATDNNEncoder(speaker_dim=256)

        # NEW: Gradient reversal for anti-leakage
        self.grl = GradientReversalLayer()
        self.speaker_classifier = nn.Linear(prosody_dim, num_speakers)

    def compute_disentanglement_loss(self, audio, prosody_embed, speaker_embed):
        # Strict orthogonality between prosody and speaker
        ortho_loss = soft_orthogonality_loss(
            prosody_embed, speaker_embed,
            target_similarity=0.0001
        )

        # Anti-leakage: prosody shouldn't predict speaker
        prosody_grl = self.grl(prosody_embed)
        speaker_pred = self.speaker_classifier(prosody_grl)
        grl_loss = F.cross_entropy(speaker_pred, speaker_labels)

        return ortho_loss + 0.1 * grl_loss
```

#### Option B: Full Two-Stage Training

More significant rewrite following DisCodec exactly:

**Stage 1 Config** (`config/prosody_discodec_stage1.yaml`):
```yaml
# Stage 1: Train disentangled encoders
training:
  stage: 1
  freeze_decoder: true

encoders:
  content:
    type: "dac_style"
    quantizer: "fsq"
    supervision: "phonetic"  # Wav2Vec phone classifier

  prosody:
    type: "dilated_causal"
    quantizer: "dual_fsq"
    supervision: "f0_regression"

  timbre:
    type: "ecapa_tdnn"
    supervision: "speaker_ce"

orthogonality:
  content_prosody_beta: 0.01
  timbre_prosody_beta: 0.0001
  use_grl: true
  grl_lambda: 0.1
```

**Stage 2 Config** (`config/prosody_discodec_stage2.yaml`):
```yaml
# Stage 2: Fusion with frozen encoders
training:
  stage: 2
  freeze_encoders: true

fusion:
  content_prosody: "sum_requantize"
  timbre_injection: "cross_attention"

decoder:
  type: "transformer_bigvgan"
  adversarial: true
```

---

## 5. Implementation Roadmap

### Phase 1: Soft Orthogonality (Low Effort, High Impact)

Add orthogonality constraints to existing `train_prosody_hed.py`:

```python
# In training loop, add after standard losses:
if config.use_orthogonality:
    # Extract speaker embedding from audio
    with torch.no_grad():
        speaker_embed = speaker_encoder(audio)

    # Prosody-speaker orthogonality
    ortho_loss = soft_orthogonality_loss(
        prosody_output['combined_embedding'],
        speaker_embed,
        target_similarity=0.0001
    )

    total_loss = total_loss + config.ortho_weight * ortho_loss
```

**Estimated improvement**: 10-20% better prosody transfer quality

### Phase 2: Gradient Reversal Layer (Medium Effort)

Add anti-leakage training:

```python
class ProsodyEncoderWithGRL(ProsodyEncoder):
    def __init__(self, config, num_speakers):
        super().__init__(config)
        self.grl = GradientReversalLayer()
        self.speaker_head = nn.Linear(config.output_hidden, num_speakers)

    def forward(self, prosody_dict):
        embedding = super().forward(prosody_dict)

        # During training, also predict speaker (with reversed gradients)
        speaker_logits = self.speaker_head(self.grl(embedding))

        return embedding, speaker_logits
```

**Estimated improvement**: 20-30% reduction in speaker leakage

### Phase 3: Full Two-Stage Training (High Effort)

Requires:
1. New content encoder with phonetic supervision
2. Dual-layer FSQ for prosody
3. ECAPA-TDNN speaker encoder
4. Stage 1/Stage 2 training scripts
5. Fusion module with re-quantization

**Estimated improvement**: Best disentanglement quality

---

## 6. Quantitative Targets

Based on DisCodec paper results:

| Metric | Current V6 (est.) | With Orthogonality | Full DisCodec |
|--------|-------------------|-------------------|---------------|
| Speaker Similarity (SSIM) | 0.45-0.50 | 0.55-0.58 | 0.61 |
| F0 Correlation | 0.40-0.45 | 0.50-0.55 | 0.59 |
| UTMOS (naturalness) | 3.5-3.8 | 3.7-3.9 | 3.98 |
| Prosody Transfer Quality | Medium | Good | Excellent |

---

## 7. Code Artifacts

### 7.1 SoftOrthogonalityLoss

```python
class SoftOrthogonalityLoss(nn.Module):
    """
    Soft orthogonality loss from DisCodec.

    Pushes cosine similarity toward a target value (not necessarily zero).
    Allows controlled coupling between factors.
    """

    def __init__(self, target_similarity: float = 0.0):
        super().__init__()
        self.target_similarity = target_similarity

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: [batch, dim] encoded representations

        Returns:
            Scalar loss
        """
        # Normalize
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)

        # Cosine similarity
        cos_sim = (z1_norm * z2_norm).sum(dim=-1)  # [batch]

        # Push toward target
        loss = (cos_sim - self.target_similarity).pow(2).mean()

        return loss
```

### 7.2 GradientReversalLayer

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
```

### 7.3 ECAPATDNNSpeakerEncoder

```python
class ECAPATDNNSpeakerEncoder(nn.Module):
    """
    ECAPA-TDNN style speaker encoder for timbre extraction.

    Uses attentive statistics pooling for variable-length inputs.
    """

    def __init__(self, input_dim: int = 80, output_dim: int = 256):
        super().__init__()

        # Frame-level layers
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # SE-Res2Block layers
        self.layer2 = SERes2Block(512, 512, scale=8)
        self.layer3 = SERes2Block(512, 512, scale=8)
        self.layer4 = SERes2Block(512, 512, scale=8)

        # Aggregation
        self.mfa = nn.Conv1d(512 * 3, 1536, kernel_size=1)
        self.asp = AttentiveStatisticsPooling(1536)
        self.asp_bn = nn.BatchNorm1d(3072)

        # Output projection
        self.fc = nn.Linear(3072, output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, time, input_dim] or mel spectrogram

        Returns:
            [batch, output_dim] speaker embedding
        """
        x = x.transpose(1, 2)  # [B, D, T]

        x1 = self.layer1(x)
        x2 = self.layer2(x1) + x1
        x3 = self.layer3(x2) + x2
        x4 = self.layer4(x3) + x3

        x = torch.cat([x2, x3, x4], dim=1)
        x = self.mfa(x)

        x = self.asp(x)
        x = self.asp_bn(x)
        x = self.fc(x)

        return x
```

---

## 8. Next Steps

1. **Immediate**: Add `SoftOrthogonalityLoss` to V6 training
2. **Short-term**: Implement `GradientReversalLayer` for anti-leakage
3. **Medium-term**: Train/integrate ECAPA-TDNN speaker encoder
4. **Long-term**: Full two-stage DisCodec pipeline

---

## 9. References

- [DisCodec Paper](https://arxiv.org/html/2512.13251)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) - Speaker verification
- [Gradient Reversal](https://arxiv.org/abs/1409.7495) - Domain adaptation
- [FSQ](https://arxiv.org/abs/2309.15505) - Finite Scalar Quantization
