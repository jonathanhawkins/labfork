# Speculative Decoding - Usage Guide

This guide demonstrates how to use LabFork's speculative decoding system for distributed inference.

## Overview

Speculative decoding achieves ~3x speedup by:
1. **Phones/browsers** (crowd tier) generate draft tokens using small models (Qwen-0.5B, Phi-2)
2. **GPUs** (power tier) verify drafts in parallel using large models (Mistral-7B, Phi-3-mini)
3. **Accepted tokens** are kept, **rejected tokens** trigger recompute
4. **Batching** groups multiple drafts for efficient GPU verification

## Quick Start

### 1. Browser-Side Draft Generation

```typescript
import { getDraftGenerator } from "@/lib/compute/draft-generator";

// Initialize draft generator
const generator = getDraftGenerator();
await generator.initialize("Qwen2-0.5B");

// Generate draft tokens
const result = await generator.generateDraft({
  context: "The quick brown fox",
  draftCount: 8,
  modelId: "Qwen2-0.5B",
  temperature: 0.8,
  deviceId: "my-device-id",
});

console.log("Draft:", result.draft);
console.log("Tokens/sec:", result.tokensPerSecond);
console.log("Avg confidence:", result.draft.avgConfidence);
```

### 2. GPU-Side Draft Verification

```typescript
import { getDraftVerifier } from "@/lib/compute/draft-verifier";

// Initialize verifier
const verifier = getDraftVerifier();
await verifier.initialize("Phi-3-mini");

// Verify draft
const result = await verifier.verifyDraft({
  draft: myDraft,
  modelId: "Phi-3-mini",
  acceptanceThreshold: 0.8,
  deviceId: "gpu-device-id",
  temperature: 0.8,
});

console.log("Final text:", result.finalText);
console.log("Acceptance rate:", result.acceptanceRate);
console.log("Speedup:", result.speedupFactor + "x");
console.log("Time saved:", result.timeSaved + "ms");
```

### 3. End-to-End Distributed Flow

```typescript
import { getDeviceAgent } from "@/lib/compute/device-agent";

// On crowd-tier device (phone/browser)
const crowdAgent = getDeviceAgent({
  modelId: "Qwen2-0.5B",
  deviceName: "My Phone",
});

await crowdAgent.start();
// Agent automatically receives draft_generation tasks

// On power-tier device (GPU)
const gpuAgent = getDeviceAgent({
  modelId: "Phi-3-mini",
  deviceName: "My GPU",
});

await gpuAgent.start();
// Agent automatically receives draft_verification tasks
```

### 4. API Usage

```typescript
// Submit draft for verification
const response = await fetch("/api/compute/speculative", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    draft: myDraftSequence,
    verifyModelId: "Phi-3-mini",
    acceptanceThreshold: 0.8,
    deviceId: "my-device",
    temperature: 0.8,
  }),
});

const { result, speedupFactor, creditsAwarded } = await response.json();
console.log("Verification complete!");
console.log("Speedup:", speedupFactor);
console.log("Credits earned:", creditsAwarded);
```

## Algorithm Details

### Acceptance/Rejection

Token is accepted if:
```
P_verify(token) / P_draft(token) >= threshold
```

Typical threshold: 0.8-1.0

### Speedup Calculation

```
Speedup = (K + 1) / (1 + R)
```

Where:
- K = number of draft tokens
- R = number of rejections

Example:
- 8 draft tokens, 6 accepted → Speedup = 9 / 3 = 3x
- 8 draft tokens, 7 accepted → Speedup = 9 / 2 = 4.5x

### Batching

```typescript
import { batchDrafts } from "@/lib/compute/speculative-decoding";

const drafts = [draft1, draft2, draft3, draft4, draft5];
const batches = batchDrafts(drafts, 4); // Max 4 per batch

// batches = [[draft1, draft2, draft3, draft4], [draft5]]
```

## Performance Tips

### Model Selection

**Draft models** (small, fast):
- `Qwen2-0.5B`: Ultra-small, perfect for mobile (~0.5GB VRAM)
- `TinyLlama-1.1B`: Fast and capable (~1GB VRAM)
- `Phi-2`: Higher quality drafts (~2GB VRAM)

**Verification models** (large, accurate):
- `Phi-3-mini`: Excellent quality/speed (~3GB VRAM)
- `Llama-3.2-3B`: Better quality (~3GB VRAM)
- `Mistral-7B`: Best quality (~5GB VRAM)

### Optimal Configuration

```typescript
const config = {
  draftCount: 8, // Sweet spot for 3x speedup
  acceptanceThreshold: 0.8, // Balance quality vs acceptance rate
  batchSize: 4, // Verify 4 drafts in parallel
  temperature: 0.8, // Slightly higher for diversity
};
```

### Device Tier Matching

| Device Tier | Role | Model |
|-------------|------|-------|
| Crowd | Draft generation | Qwen2-0.5B, TinyLlama-1.1B |
| Standard | Draft generation | Phi-2, Llama-3.2-1B |
| Power | Verification | Phi-3-mini, Mistral-7B |

## Monitoring & Stats

```typescript
import { getDraftGenerator, getDraftVerifier } from "@/lib/compute";

// Get verification stats
const verifier = getDraftVerifier();
const results = await verifier.verifyBatch(options);
const stats = verifier.getBatchStats(results);

console.log({
  avgAcceptanceRate: stats.avgAcceptanceRate,
  avgSpeedupFactor: stats.avgSpeedupFactor,
  totalTimeSaved: stats.totalTimeSaved,
  totalTokensVerified: stats.totalTokensVerified,
  totalTokensAccepted: stats.totalTokensAccepted,
});

// Get network-wide stats
const response = await fetch("/api/compute/speculative");
const networkStats = await response.json();

console.log({
  totalDrafts: networkStats.totalDrafts,
  avgAcceptanceRate: networkStats.avgAcceptanceRate,
  avgSpeedupFactor: networkStats.avgSpeedupFactor,
  activeDevices: networkStats.activeDevices,
});
```

## Utilities

### Validate Draft

```typescript
import { validateDraft } from "@/lib/compute/speculative-decoding";

const validation = validateDraft(myDraft);
if (!validation.valid) {
  console.error("Invalid draft:", validation.errors);
}
```

### Calculate Speedup

```typescript
import { calculateSpeedupFactor } from "@/lib/compute/speculative-decoding";

const speedup = calculateSpeedupFactor(8, 6); // 8 drafts, 6 accepted
console.log("Speedup:", speedup + "x"); // 3x
```

### Estimate Time Saved

```typescript
import { estimateTimeSaved } from "@/lib/compute/speculative-decoding";

const timeSaved = estimateTimeSaved(
  8, // draft count
  6, // accepted count
  50, // sequential time per token (ms)
  100, // draft time (ms)
  200 // verify time (ms)
);

console.log("Time saved:", timeSaved + "ms");
```

## Credit System

Credits are awarded based on:
- **Draft generation**: 0.8 credits base
- **Draft verification**: 3 credits base
- **Token bonus**: +0.1 per accepted token
- **Speedup bonus**: +(speedup - 1) * 2

Example:
- Verify 8 tokens, 6 accepted, 3x speedup
- Credits = 3 + (6 * 0.1) + (3 - 1) * 2 = 7.6 credits

## Best Practices

1. **Match models to device capability**
   - Don't run large models on crowd tier
   - Don't waste power tier on draft generation

2. **Batch verification tasks**
   - Verify 4+ drafts together on GPUs
   - Reduces overhead, increases throughput

3. **Monitor acceptance rates**
   - Target 70-80% acceptance
   - Lower = need better draft model or lower threshold
   - Higher = can increase threshold for quality

4. **Tune temperature**
   - Higher temp (0.8-1.0) = more diversity, lower acceptance
   - Lower temp (0.5-0.7) = higher acceptance, less diversity

5. **Handle errors gracefully**
   - Validate drafts before verification
   - Implement retry logic for failures
   - Fall back to sequential generation if needed

## Troubleshooting

### Low acceptance rate (<50%)

- Draft model quality too low
- Threshold too high
- Context mismatch between draft and verify

**Solution**: Use better draft model or lower threshold

### Poor speedup (<2x)

- Too many rejections
- Batch size too small
- Network overhead

**Solution**: Increase batch size, improve draft quality

### High latency

- Model not preloaded
- Network slow
- GPU saturated

**Solution**: Preload models, batch requests, scale GPUs

## Examples

See working examples:
- `/lib/compute/__tests__/speculative-decoding.test.ts` - Unit tests
- `/components/compute/SpeculativeDecodingDemo.tsx` - React component (to be created)
- `/lib/compute/SPECULATIVE_DECODING_EXAMPLE.md` - This file

## References

- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [WebLLM Documentation](https://github.com/mlc-ai/web-llm)
- [LabFork Compute Documentation](./CONTRIBUTOR_QUICK_START.md)
