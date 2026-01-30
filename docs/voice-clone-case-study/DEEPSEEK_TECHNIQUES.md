# DeepSeek Techniques Applied to Voice Cloning

This document explains the DeepSeek innovations integrated into our training pipeline and how they optimize training on M4 Pro Mac (64GB).

---

## Table of Contents

1. [Multi-head Latent Attention (MLA)](#1-multi-head-latent-attention-mla)
2. [Multi-Token Prediction (MTP)](#2-multi-token-prediction-mtp)
3. [Auxiliary-Loss-Free Load Balancing](#3-auxiliary-loss-free-load-balancing)
4. [DeepSeek Learning Rate Schedule](#4-deepseek-learning-rate-schedule)
5. [M4 Pro Optimizations](#5-m4-pro-optimizations)
6. [Expected Benefits](#6-expected-benefits)

---

## 1. Multi-head Latent Attention (MLA)

**Source:** DeepSeek-V2, DeepSeek-V3 Technical Reports

### The Problem
Standard transformer attention caches full K, V tensors:
- Memory: `O(batch × seq_len × num_heads × head_dim × 2)`
- For long audio sequences, this dominates memory

### DeepSeek's Solution
Compress KV into a low-rank latent space:

```
Standard:        x → Q, K, V → Attention → Output
                     ↓   ↓
                   Cache K, V (large!)

MLA:             x → Q           → Attention → Output
                 x → c_KV (small) → K, V (on-demand)
                     ↓
                   Cache c_KV (small!)
```

### Implementation

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, kv_lora_rank=512):
        # Compress KV to lower dimension
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank)  # Down
        self.k_up_proj = nn.Linear(kv_lora_rank, num_heads * head_dim)  # Up
        self.v_up_proj = nn.Linear(kv_lora_rank, num_heads * head_dim)  # Up
    
    def forward(self, x, past_kv_compressed=None):
        # Compress to latent
        kv_compressed = self.kv_down_proj(x)  # [B, L, kv_lora_rank]
        
        # Cache the compressed version (much smaller!)
        if past_kv_compressed is not None:
            kv_compressed = torch.cat([past_kv_compressed, kv_compressed], dim=1)
        
        # Decompress on-demand for attention
        K = self.k_up_proj(kv_compressed)
        V = self.v_up_proj(kv_compressed)
        
        # ... standard attention ...
        
        return output, kv_compressed  # Return small cache
```

### Memory Savings

| Config | Standard KV Cache | MLA KV Cache | Savings |
|--------|-------------------|--------------|---------|
| 16 heads, 64 dim, 4096 seq | 64 MB | 8 MB | **8×** |
| 32 heads, 128 dim, 8192 seq | 512 MB | 32 MB | **16×** |

### Relevance to Voice Cloning
- Audio sequences are long (30s = 3750 frames at 12.5Hz)
- MLA allows longer context without OOM
- Better prosody from more conversation history

---

## 2. Multi-Token Prediction (MTP)

**Source:** DeepSeek-V3 Technical Report, Meta's "Better & Faster LLMs" Paper

### The Problem
Standard next-token prediction:
- Model sees "The cat sat on the" → predicts "mat"
- Only 1 supervision signal per position
- Slow to learn long-range patterns

### DeepSeek's Solution
Predict multiple future tokens simultaneously:

```
Standard:    [Input] → Predict token t+1

MTP:         [Input] → Predict token t+1
                    → Predict token t+2  
                    → Predict token t+3
                    → Predict token t+4
```

### Implementation

```python
class MultiTokenPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_predict=4):
        # Separate heads for each future position
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for _ in range(num_predict)
        ])
        
        # Transform between predictions
        self.transforms = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_predict - 1)
        ])
    
    def forward(self, hidden, labels=None):
        logits = []
        h = hidden
        
        for i, head in enumerate(self.heads):
            logits.append(head(h))
            if i < len(self.transforms):
                h = self.transforms[i](h)
        
        # Weight future predictions (exponential decay)
        if labels is not None:
            loss = sum(
                (0.5 ** i) * cross_entropy(logits[i], labels[:, i:])
                for i in range(len(logits))
            )
        
        return logits, loss
```

### Benefits for Voice Cloning

| Benefit | Explanation |
|---------|-------------|
| **Denser signal** | 4× more gradients per training sample |
| **Faster convergence** | ~20-30% fewer steps to same loss |
| **Better prosody** | Forces model to plan ahead (prosody is predictive) |
| **Reduced overfitting** | More supervision regularizes |

### Why It Works for Audio
Speech has strong sequential dependencies:
- Pitch contours span multiple frames
- Emphasis patterns are predictable
- Rhythm follows prosodic units

MTP forces the model to learn these patterns explicitly.

---

## 3. Auxiliary-Loss-Free Load Balancing

**Source:** DeepSeek-V3 Technical Report

### The Problem (for MoE models)
Mixture-of-Experts models need balanced expert usage:
- Standard: Add auxiliary loss to penalize imbalance
- But: Auxiliary loss can hurt main task performance

### DeepSeek's Solution
Balance without changing the loss:

```python
class AuxLossFreeBalancer:
    def __init__(self, num_experts, balance_factor=0.01):
        self.expert_usage = torch.ones(num_experts) / num_experts  # EMA
        self.bias = torch.zeros(num_experts)
    
    def update(self, routing_weights):
        # Track usage with EMA
        batch_usage = routing_weights.mean(dim=[0, 1])
        self.expert_usage = 0.99 * self.expert_usage + 0.01 * batch_usage
        
        # Compute bias correction
        imbalance = self.expert_usage - (1 / num_experts)
        self.bias = -self.balance_factor * imbalance
        
        return self.bias  # Add to routing logits
```

### How It Works
1. Track which experts are overused (EMA)
2. Add negative bias to overused experts' routing scores
3. Underused experts get positive bias
4. No auxiliary loss term needed!

### Relevance to Voice Cloning
- CSM isn't MoE, but the principle applies to attention heads
- Some heads may specialize (prosody, content, speaker)
- Balanced utilization = better features

---

## 4. DeepSeek Learning Rate Schedule

**Source:** DeepSeek-V2, V3 Training Configurations

### The Schedule

```
       │     ┌──────────────────────┐
       │    /│                      │╲
LR     │   / │      Stable          │ ╲
       │  /  │      Phase           │  ╲
       │ /   │                      │   ╲___________
       │/    │                      │
       └─────┴──────────────────────┴──────────────────
         Warmup    Stable Phase     Cosine Decay

Standard:  Warmup → Cosine Decay (no stable phase)
DeepSeek:  Warmup → Stable → Cosine Decay
```

### Implementation

```python
def deepseek_lr_schedule(step, warmup, stable, total, min_lr_ratio=0.1):
    if step < warmup:
        return step / warmup
    elif step < warmup + stable:
        return 1.0
    else:
        decay_step = step - warmup - stable
        decay_total = total - warmup - stable
        progress = decay_step / decay_total
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * progress))
```

### Benefits

| Aspect | Standard | DeepSeek |
|--------|----------|----------|
| Early training | Rapid LR changes | Stable learning |
| Peak phase | Brief | Extended |
| Convergence | Good | Better |
| Loss smoothness | Bumpy | Smooth |

### For M4 Pro
- Longer stable phase compensates for slower training
- Smoother gradients = better MPS utilization
- Less risk of divergence with larger batches

---

## 5. M4 Pro Optimizations

### Unified Memory Architecture

M4 Pro's 64GB is shared between CPU and GPU:

```
┌─────────────────────────────────────────────┐
│                  64GB LPDDR5X                │
│     ┌─────────────────────────────────┐     │
│     │         Unified Memory          │     │
│     │                                 │     │
│     │   CPU ◄──────────────────► GPU  │     │
│     │         273 GB/s bandwidth      │     │
│     │                                 │     │
│     └─────────────────────────────────┘     │
└─────────────────────────────────────────────┘
```

### Optimizations Applied

| Optimization | Why | How |
|--------------|-----|-----|
| **Larger batches** | More memory available | batch_size=12 vs 4 |
| **FP32 training** | MPS FP16 slower | precision="fp32" |
| **torch.compile** | MPS backend optimization | use_compile=true |
| **No multiprocessing** | MPS data loading issues | num_workers=0 |

### Memory Budget

```
Model weights:     ~4 GB
Gradients:         ~4 GB  
Optimizer states:  ~8 GB
Activations:       ~12 GB (with checkpointing)
MTP head:          ~0.5 GB
Batch (12×30s):    ~2 GB
─────────────────────────
Total:             ~30 GB

Headroom:          ~34 GB (for peaks, system)
```

### torch.compile for MPS

```python
# Applied automatically in optimize_for_m4_pro()
model = torch.compile(model, mode='reduce-overhead')
```

Benefits:
- Kernel fusion (fewer memory ops)
- Optimized attention patterns
- ~10-20% speedup on M4 Pro

---

## 6. Expected Benefits

### Convergence Speed

| Technique | Speedup | Notes |
|-----------|---------|-------|
| MTP | ~25% fewer steps | Denser supervision |
| DeepSeek LR | ~10% fewer steps | Smoother training |
| Larger batch (M4) | ~15% faster | More parallelism |
| **Combined** | **~40-50% faster** | |

### Quality Improvements

| Technique | Benefit |
|-----------|---------|
| MTP | Better prosody planning |
| MLA | Longer context = better coherence |
| Stable LR | Less overfitting |

### Memory Efficiency

| Technique | Savings |
|-----------|---------|
| MLA (if applicable) | 8-16× KV cache |
| Gradient checkpointing | ~40% activations |
| FP32 on M4 (vs FP16 copies) | ~10% overhead |

---

## Running Training with DeepSeek Techniques

```bash
cd training

# Start with dashboard
python train_deepseek.py \
    --config config/m4_pro_deepseek.yaml \
    --dashboard

# Open http://localhost:8001 for metrics
# Open http://localhost:3000/training for dashboard UI
```

### Config Options

```yaml
# Enable/disable techniques
use_mtp: true           # Multi-Token Prediction
mtp_tokens: 4           # How many tokens to predict
use_compile: true       # torch.compile optimization

# DeepSeek LR schedule
warmup_steps: 500
stable_steps: 500       # NEW: stable phase
min_lr_ratio: 0.1       # Decay to 10% of peak
```

---

## References

1. **DeepSeek-V3 Technical Report** (2024)
   - Multi-head Latent Attention
   - Auxiliary-loss-free load balancing
   - Training configurations

2. **DeepSeek-V2 Technical Report** (2024)
   - MLA introduction
   - Memory efficiency analysis

3. **"Better & Faster Large Language Models via Multi-Token Prediction"** - Meta (2024)
   - MTP theory and experiments
   - Convergence analysis

4. **Apple Silicon Optimization Guide**
   - MPS backend best practices
   - Unified memory utilization

---

*Document Version: 1.0*
*Last Updated: January 2026*
