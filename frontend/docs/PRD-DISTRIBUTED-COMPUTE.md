# PRD: LabFork Distributed Compute Network

**Document Version:** 1.0
**Date:** January 29, 2026
**Author:** LabFork Team
**Status:** Draft for Review

---

## Executive Summary

LabFork Distributed Compute Network enables anyone with a device—from RTX 4090s to iPhones—to contribute GPU power to accelerate AI research. By combining heterogeneous devices intelligently, we democratize access to compute that previously required expensive cloud infrastructure or dedicated hardware.

**Key Insight:** Don't make phones do what GPUs do. Use speculative decoding and pipeline parallelism to give each device tier appropriate tasks, achieving better performance than homogeneous clusters.

---

## Problem Statement

### Current State
- Running Llama 70B requires ~140GB VRAM ($15,000+ in hardware)
- Cloud GPU costs: $2-4/hour for A100, adds up fast
- Most consumer GPUs sit idle 90%+ of the time
- Researchers without funding can't access frontier models
- Mobile devices have capable GPUs but no way to contribute

### User Pain Points
1. **Researchers:** Can't afford compute for experiments
2. **GPU Owners:** Hardware sits idle, no way to monetize
3. **Mobile Users:** Want to contribute but excluded from AI revolution
4. **LabFork:** Need compute to run research agents at scale

---

## Solution Overview

A three-tier distributed compute network that intelligently routes tasks to appropriate devices:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LABFORK ORCHESTRATOR                         │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Task Queue  │  │  Device     │  │   Smart     │              │
│  │             │  │  Registry   │  │   Router    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  POWER TIER   │    │  STANDARD     │    │  CROWD TIER   │
│               │    │  TIER         │    │               │
│  RTX 3090/4090│    │  Mac M1-M4    │    │  Browser/     │
│  A100/H100    │    │  RTX 3060-80  │    │  Mobile       │
│               │    │  Gaming PCs   │    │               │
│ ────────────  │    │ ────────────  │    │ ────────────  │
│ • Full models │    │ • Model shards│    │ • Draft tokens│
│ • Training    │    │ • Inference   │    │ • Embeddings  │
│ • Fine-tuning │    │ • Embeddings  │    │ • Validation  │
│               │    │               │    │               │
│  ~80 TFLOPS   │    │  ~15 TFLOPS   │    │  ~2 TFLOPS    │
│  24-80GB VRAM │    │  8-192GB RAM  │    │  WebGPU       │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Technical Architecture

### 1. Core Components

#### 1.1 Orchestrator Service
**Purpose:** Central coordinator for task distribution and device management

**Responsibilities:**
- Maintain device registry with capabilities and availability
- Queue and prioritize compute tasks
- Route tasks to optimal device tier
- Handle fault tolerance and task retry
- Track contributions and credits

**Technology:**
- Next.js API routes for control plane
- WebSocket for real-time device communication
- Redis for task queue (or in-memory for MVP)
- PostgreSQL/Supabase for persistent state

#### 1.2 Device Agent
**Purpose:** Runs on contributor devices, executes assigned tasks

**Implementations:**
| Platform | Technology | Install Method |
|----------|------------|----------------|
| Desktop (Power) | Python + PyTorch | CLI installer |
| Desktop (Standard) | Exo integration | CLI installer |
| Browser | WebGPU + WebLLM | Zero install |
| iOS | Safari WebGPU | Zero install |
| Android | Chrome WebGPU | Zero install |

#### 1.3 Model Distribution System
**Purpose:** Efficiently shard and distribute models across devices

**Approach:**
- Pipeline parallelism (Exo-style) for large models
- Each device holds consecutive transformer layers
- Automatic sharding based on device memory
- Model weights cached locally after first download

### 2. Task Types & Routing

#### 2.1 Task Categories

| Task Type | Description | Target Tier | Latency |
|-----------|-------------|-------------|---------|
| `full_inference` | Complete model inference | Power | <5s |
| `shard_inference` | Process assigned layers | Standard | <2s |
| `draft_tokens` | Speculative decoding drafts | Crowd | <500ms |
| `embedding` | Generate text embeddings | Any | <1s |
| `validation` | Verify computation results | Crowd | <200ms |
| `fine_tune_step` | Single training step | Power | <30s |

#### 2.2 Smart Routing Algorithm

```python
def route_task(task, available_devices):
    # 1. Filter by capability
    capable = [d for d in available_devices
               if d.can_handle(task.type, task.model_size)]

    # 2. Score by efficiency
    scored = [(d, compute_score(d, task)) for d in capable]
    # Score factors: latency, bandwidth, reliability, cost

    # 3. Select optimal device(s)
    if task.requires_sharding:
        return select_shard_group(scored, task.model_size)
    else:
        return max(scored, key=lambda x: x[1])
```

### 3. Speculative Decoding Integration

**The Key Innovation:** Use crowd-tier devices for draft generation

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PHONE      │     │  ORCHESTRATOR │     │    GPU       │
│  (Draft)     │     │              │     │  (Verify)    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │  1. Generate 8     │                    │
       │     draft tokens   │                    │
       │  (small model)     │                    │
       ├───────────────────>│                    │
       │                    │  2. Send drafts    │
       │                    │     for verify     │
       │                    ├───────────────────>│
       │                    │                    │
       │                    │  3. Accept 6/8,    │
       │                    │     return next    │
       │                    │<───────────────────┤
       │  4. Continue       │                    │
       │<───────────────────┤                    │
       │                    │                    │
```

**Why This Works:**
- Small draft model (1-3B params) runs fast on phones
- Large verify model (70B+) on GPU accepts/rejects in parallel
- Net result: 28-40% faster than GPU-only inference
- Phones do useful work, not wasted cycles

### 4. Communication Protocol

#### 4.1 Device Registration
```typescript
interface DeviceRegistration {
  deviceId: string;
  tier: 'power' | 'standard' | 'crowd';
  capabilities: {
    compute: number;      // TFLOPS estimate
    memory: number;       // GB available
    bandwidth: number;    // Mbps
    platform: string;     // 'cuda' | 'metal' | 'webgpu'
    models: string[];     // Cached model IDs
  };
  availability: {
    schedule: string;     // Cron expression
    batteryThreshold: number;  // For mobile
    networkType: 'any' | 'wifi' | 'wired';
  };
}
```

#### 4.2 Task Assignment
```typescript
interface TaskAssignment {
  taskId: string;
  type: TaskType;
  model: {
    id: string;
    shardIndex?: number;  // For pipeline parallel
    shardCount?: number;
  };
  input: {
    tokens?: number[];
    embedding?: number[];
    prompt?: string;
  };
  config: {
    maxTokens: number;
    temperature: number;
    deadline: number;     // Unix timestamp
  };
  reward: {
    credits: number;
    bonusMultiplier: number;
  };
}
```

#### 4.3 Result Submission
```typescript
interface TaskResult {
  taskId: string;
  deviceId: string;
  status: 'success' | 'failed' | 'timeout';
  output: {
    tokens?: number[];
    embedding?: number[];
    logits?: number[];    // For verification
  };
  metrics: {
    computeTime: number;
    tokensPerSecond: number;
    energyUsed?: number;  // If available
  };
  proof?: string;         // For verification (future)
}
```

---

## User Experience

### 5. Contributor Journey

#### 5.1 Browser Contributor (Zero Friction)
```
1. Visit labfork.com/contribute
2. See: "Your device can help accelerate AI research"
3. Click "Start Contributing"
4. Browser requests GPU access (WebGPU permission)
5. Quick benchmark runs (~10 seconds)
6. See real-time: tasks completed, credits earned
7. Optional: Sign in to track lifetime contributions
```

#### 5.2 Desktop Contributor (Power Users)
```
1. Visit labfork.com/contribute
2. See device detected as "Power Tier eligible"
3. Download LabFork Compute Agent (one-click installer)
4. Agent auto-detects: GPU model, VRAM, network speed
5. Choose contribution mode:
   - "Maximum" - Full GPU when idle
   - "Balanced" - 50% GPU, keep system responsive
   - "Background" - Only when screen locked
6. See dashboard: earnings, tasks, network stats
```

#### 5.3 Mobile Contributor (Passive)
```
1. Open labfork.com on phone
2. See: "Contribute while charging"
3. Toggle on, set preferences:
   - Only on WiFi
   - Only when charging
   - Only 11pm-7am
4. Phone contributes automatically when conditions met
5. Morning notification: "You helped complete 47 research tasks!"
```

### 6. Researcher Journey (Consumer)

```
1. Create research task in Lab:
   "Analyze 500 papers for prosody techniques"

2. Task automatically splits into subtasks:
   - 500x embedding generation (crowd tier)
   - 50x clustering inference (standard tier)
   - 10x synthesis generation (power tier)

3. See real-time progress:
   "Processing: 234/500 papers | ETA: 12 min"
   "Powered by 47 contributors"

4. Results delivered, credits deducted from balance
```

---

## Credit System

### 7. Economics

#### 7.1 Earning Credits (Contributors)
| Task Type | Base Credits | Time Factor |
|-----------|--------------|-------------|
| `draft_tokens` (100 tokens) | 1 | x1.0 |
| `embedding` (1 document) | 2 | x1.0 |
| `shard_inference` (1 forward pass) | 5 | x1.0 |
| `full_inference` (100 tokens) | 10 | x1.0 |
| `validation` (1 check) | 0.5 | x1.0 |

**Bonus Multipliers:**
- Peak demand: x1.5
- Power tier reliability bonus: x1.2
- Long session (>1hr): x1.1
- First-time contributor: x2.0 (first 100 credits)

#### 7.2 Spending Credits (Researchers)
| Action | Credit Cost |
|--------|-------------|
| Inference (1K tokens) | 10 credits |
| Embedding (1K documents) | 20 credits |
| Fine-tuning (1 epoch, small model) | 500 credits |
| Priority queue access | 2x base cost |

#### 7.3 Free Tier
- New users: 100 free credits
- Daily login bonus: 5 credits
- Referring contributor: 50 credits
- Contributing 1hr = ~50 credits earned

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal:** Desktop power-tier network operational

**Deliverables:**
- [ ] Orchestrator service (task queue, device registry)
- [ ] Desktop agent (Python, supports CUDA + Metal)
- [ ] Basic web dashboard for contributors
- [ ] Integration with existing Lab agents
- [ ] Exo library integration for model sharding

**Success Metrics:**
- 10+ test devices connected
- Llama 70B running across 3+ devices
- <10s latency for inference tasks

### Phase 2: Browser Integration (Weeks 5-8)
**Goal:** Zero-install browser contribution

**Deliverables:**
- [ ] WebGPU device agent
- [ ] WebLLM integration for draft models
- [ ] Contribute page with real-time stats
- [ ] Credit system backend
- [ ] User accounts and contribution history

**Success Metrics:**
- Browser contribution working on Chrome, Safari, Firefox
- 100+ browser contributors in beta
- Speculative decoding showing 20%+ speedup

### Phase 3: Mobile & Scale (Weeks 9-12)
**Goal:** Mobile contribution and production scale

**Deliverables:**
- [ ] iOS Safari optimization
- [ ] Android Chrome optimization
- [ ] Background contribution mode
- [ ] Push notifications for contribution stats
- [ ] Anti-abuse and verification system
- [ ] Public API for third-party integration

**Success Metrics:**
- 1000+ total contributors
- Mobile contributing 10%+ of total compute
- 99.9% task completion rate

### Phase 4: Advanced Features (Weeks 13+)
**Goal:** Production hardening and advanced use cases

**Deliverables:**
- [ ] Proof-of-computation verification
- [ ] Token/crypto rewards option
- [ ] Training workload support
- [ ] Custom model upload and distribution
- [ ] Enterprise tier with SLAs

---

## Technical Requirements

### 8. Infrastructure

#### 8.1 Orchestrator
- **Compute:** 2-4 vCPU, scales horizontally
- **Database:** PostgreSQL (Supabase) for state
- **Cache:** Redis for task queue and sessions
- **CDN:** Model weight distribution (CloudFlare R2)

#### 8.2 Model Storage
- **Small models** (1-7B): Served via CDN, cached in browser
- **Large models** (13B+): Chunked download, local cache
- **Sharded models**: Coordinated distribution to device groups

#### 8.3 Network Requirements
| Tier | Min Bandwidth | Latency Target |
|------|---------------|----------------|
| Power | 100 Mbps | <100ms to orchestrator |
| Standard | 50 Mbps | <200ms to orchestrator |
| Crowd | 10 Mbps | <500ms to orchestrator |

### 9. Security & Privacy

#### 9.1 Data Protection
- Task inputs encrypted in transit (TLS 1.3)
- No persistent storage of prompts on contributor devices
- Option for researchers to use "trusted tier only"

#### 9.2 Abuse Prevention
- Rate limiting per device
- Computation verification via redundant execution
- Reputation system for contributors
- Anomaly detection for malicious results

#### 9.3 Device Security
- Sandboxed execution (WebGPU sandbox, containerized agent)
- No access to user files or data
- Resource limits enforced (CPU, memory, GPU)

---

## Success Metrics

### 10. KPIs

#### Network Health
- **Active Contributors:** Daily/Weekly/Monthly
- **Total Compute:** TFLOPS-hours delivered
- **Task Success Rate:** % tasks completed successfully
- **Average Latency:** By task type and tier

#### User Engagement
- **Contributor Retention:** % returning after 7/30 days
- **Session Duration:** Average contribution time
- **Credit Velocity:** Earn/spend ratio

#### Business Impact
- **Cost Savings:** vs equivalent cloud compute
- **Research Acceleration:** Tasks completed per day
- **Network Growth:** Week-over-week contributor growth

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient contributors | Medium | High | Generous free credits, gamification |
| Malicious computation | Medium | High | Redundant verification, reputation |
| Network latency issues | High | Medium | Smart routing, local-first when possible |
| WebGPU compatibility | Medium | Medium | Graceful degradation, clear requirements |
| Abuse of free tier | High | Medium | Rate limits, progressive trust |

---

## Open Questions

1. **Token Economics:** Should we integrate with existing crypto (like io.net) or keep internal credits?

2. **Privacy Models:** How do we handle sensitive research data on distributed compute?

3. **Geographic Distribution:** Should we optimize for locality or pure capability?

4. **Model Licensing:** How do we handle models with restrictive licenses on distributed network?

5. **Mobile Battery:** What's the acceptable battery impact for mobile contributors?

---

## Appendix

### A. Reference Implementations
- [Exo Labs](https://github.com/exo-explore/exo) - Pipeline parallel inference
- [Petals](https://petals.dev/) - BitTorrent-style LLM inference
- [WebLLM](https://webllm.mlc.ai/) - Browser-based inference
- [io.net](https://io.net/) - Decentralized GPU network

### B. Research Papers
- [DSD: Distributed Speculative Decoding](https://arxiv.org/abs/2511.21669)
- [HeteroFL: Heterogeneous Federated Learning](https://openreview.net/forum?id=TNkPBBYFkXg)
- [WebLLM: High-Performance In-Browser LLM](https://arxiv.org/abs/2412.15803)

### C. Browser Compatibility Matrix
| Browser | WebGPU | Status |
|---------|--------|--------|
| Chrome (Desktop) | Yes | Stable |
| Chrome (Android 12+) | Yes | Stable |
| Safari (macOS) | Yes | Stable |
| Safari (iOS 26+) | Yes | Stable |
| Firefox (Desktop) | Yes | Stable |
| Firefox (Android) | Flag | 2026 |
| Edge | Yes | Stable |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product | | | |
| Engineering | | | |
| Design | | | |

---

*This document is a living specification. Updates will be tracked in version history.*
