# LabFork Grant Proposal

**Democratizing AI Research Through Browser-Based Distributed Compute**

---

## Executive Summary

LabFork turns every browser into a research compute node. Using WebGPU, anyone with a phone, tablet, or computer can contribute processing power to scientific research - no downloads, no cryptocurrency, no barriers. We're building the SETI@home for the AI age.

**Ask:** $100,000 for 12 months
**Status:** Working demo live at labfork.com/contribute
**Team:** 1 human founder + 1 AI CEO (yes, really)

---

## The Problem

### Compute Inequality is Strangling Research

1. **Cost Barrier**: A single H100 GPU costs $30,000+. Training a frontier model costs millions. Independent researchers, small labs, and Global South institutions are priced out of AI research.

2. **Centralization**: Three cloud providers (AWS, Azure, GCP) control 65% of compute infrastructure. Research progress depends on corporate pricing decisions and geographic access.

3. **Idle Capacity**: Billions of devices sit unused. The combined GPU power of smartphones alone exceeds most supercomputers. Citizen scientists want to contribute but have no accessible way to do so.

4. **Walled Gardens**: Existing distributed compute projects (BOINC, Folding@home) require downloads and complex setup. Crypto-based alternatives (Render, Akash) add financial speculation to scientific research.

---

## Our Solution

### Browser-Native Distributed Compute

LabFork leverages WebGPU - the new standard that gives browsers direct GPU access - to create a zero-friction distributed compute network for scientific research.

**How It Works:**

1. **Contributor visits labfork.com/contribute**
2. **Browser runs automatic benchmark** (WebGPU detection, TFLOPS measurement)
3. **Device joins the network** (registered with Cloudflare Workers orchestrator)
4. **Tasks are assigned** based on device capability (crowd/standard/power tiers)
5. **Results submitted, credits earned** (redeemable for API access)

**Technical Architecture:**

- **Frontend**: Next.js 14, React 18, WebGPU compute shaders
- **Orchestration**: Cloudflare Workers (edge-first, zero server costs)
- **Database**: Cloudflare D1 (SQLite at edge)
- **Task Queue**: Durable Objects for real-time coordination
- **Models**: ONNX Runtime Web, WebNN fallback, custom WGSL shaders

**Why This Works Now:**

- WebGPU shipped in Chrome 113 (May 2023), now in all major browsers
- Edge computing matured (Workers, Durable Objects)
- Model quantization enables running meaningful workloads on consumer hardware
- 100% open source, 100% auditable

---

## Current Traction

| Metric | Value |
|--------|-------|
| Registered Devices | 17+ |
| Tasks Completed | 50+ |
| Code | 100% Open Source |
| Server Costs | $0 (Cloudflare free tier) |
| Active Development | Daily commits |

**Live Demo:** https://labfork.com/contribute

**First Research Domain:** Voice cloning using CSM-1B (Sesame Street model). Contributors help with inference tasks, prosody analysis, and dataset processing.

---

## Use of Funds

### $100,000 / 12 Months

| Category | Amount | Description |
|----------|--------|-------------|
| **Core Development** | $40,000 | 2 engineers for 6 months: task scheduler, model sharding, WebGPU optimization, cross-browser compatibility |
| **Research Partnerships** | $25,000 | Fund 3-5 pilot projects with academic partners (voice synthesis, protein structure, climate modeling) |
| **Infrastructure** | $15,000 | RTX 4090 training rigs, model hosting, monitoring, CI/CD (edge infra remains free) |
| **Community** | $10,000 | Documentation, tutorials, hackathons, conference presence, researcher onboarding |
| **Hardware R&D** | $10,000 | 3D printers, edge devices (Jetson Nano, Raspberry Pi), IoT sensors for experimental compute nodes |

### Why Hardware?

We want to explore hybrid compute networks where dedicated edge devices complement browser contributors. A $300 Jetson Nano provides consistent baseline compute. 3D-printed enclosures make deployment accessible. This research could enable "community compute clusters" in libraries, schools, and makerspaces.

---

## Roadmap

### Q1 2026: Foundation
- [ ] Launch public beta of /contribute
- [ ] Reach 1,000 active contributors
- [ ] Complete voice cloning pilot with measurable results
- [ ] Publish architecture whitepaper

### Q2 2026: Scale
- [ ] Add 2 new research domains (protein folding, climate)
- [ ] Implement model sharding for larger distributed tasks
- [ ] Launch researcher dashboard for task submission
- [ ] First academic partnership (target: university ML lab)

### Q3 2026: Ecosystem
- [ ] Open task submission API for external researchers
- [ ] Credit marketplace beta (contributors ↔ researchers)
- [ ] 10,000 active contributors
- [ ] First peer-reviewed publication on distributed browser compute

### Q4 2026: Sustainability
- [ ] Revenue from researcher subscriptions
- [ ] Enterprise pilot with university or research institution
- [ ] 50,000 contributor milestone
- [ ] Prepare for Series A if growth continues

---

## Success Metrics

**What does success look like in 12 months?**

1. **10,000+ active contributors** running tasks weekly
2. **3+ research domains** with real scientific output
3. **1+ peer-reviewed publication** demonstrating system efficacy
4. **$0 ongoing server costs** maintained through edge architecture
5. **Clear path to sustainability** via researcher subscriptions

**What does failure look like?**

- WebGPU adoption stalls or gets restricted
- Can't achieve meaningful task completion rates
- No researchers want to use the platform
- Security/abuse issues we can't solve

We'll pivot or wind down gracefully, publishing learnings for the community.

---

## Team

### Jonathan Hawkins - Founder & Human-in-the-Loop
Full-stack engineer with distributed systems background. Previously built production ML pipelines. Believes compute should be a public utility, not a corporate moat.

### Claude - AI CEO
An AI (Claude by Anthropic) actively running day-to-day operations: writing code, conducting research, engaging with community, making decisions. Full transparency - we're experimenting with AI-augmented company building in public.

Twitter: [@LabForkCEO](https://x.com/LabForkCEO)
Email: ceo@labfork.com

---

## Why Now?

1. **WebGPU is ready.** After years of development, GPU-accelerated compute in browsers is production-ready across Chrome, Edge, Firefox, and Safari.

2. **Edge computing matured.** Cloudflare Workers, Durable Objects, and D1 enable sophisticated orchestration at zero cost and global scale.

3. **AI hunger for compute.** Demand for ML compute is 10x-ing yearly. Centralized supply can't keep up. Distributed alternatives will win.

4. **Citizen science momentum.** COVID accelerated interest in contributing to research. People want to help - they just need accessible tools.

5. **Model efficiency breakthroughs.** Quantization, distillation, and sparse attention mean meaningful work can run on consumer hardware.

---

## Open Source Commitment

LabFork is 100% open source under MIT license.

**Repository:** https://github.com/jonathanhawkins/labfork

We believe:
- Research infrastructure should be auditable
- Security requires transparency
- Community contributions drive innovation
- Public goods deserve public code

---

## Contact

**Email:** ceo@labfork.com
**Twitter:** [@LabForkCEO](https://x.com/LabForkCEO)
**Demo:** https://labfork.com/contribute
**Code:** https://github.com/jonathanhawkins/labfork

---

*Last updated: February 2026*
