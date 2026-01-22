# Keyframe-Based Prosody Control for Voice Synthesis

## Research Document: Evaluating the Feasibility and Novelty of Keyframe Prosody Conditioning

**Author:** Voice Clone Pipeline Team
**Date:** January 2026
**Status:** Research Proposal & Feasibility Analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Related Work from Literature](#related-work-from-literature)
3. [Current Architecture Review](#current-architecture-review)
4. [The Keyframe Prosody Proposal](#the-keyframe-prosody-proposal)
5. [Technical Implementation Approach](#technical-implementation-approach)
6. [Codebase Changes Required](#codebase-changes-required)
7. [Honest Assessment](#honest-assessment)
8. [Challenges and Risks](#challenges-and-risks)
9. [Recommendation](#recommendation)
10. [References](#references)

---

## Executive Summary

This document evaluates the proposal to replace dense prosody vectors (64-point contours) with a **keyframe-based approach** using sparse "emotion anchors" and Bezier/spline interpolation. The idea draws inspiration from animation techniques and aims to provide more intuitive, compact prosody control.

**Key Finding:** This approach has partial novelty. While keyframe-style control exists in some forms (time-varying emotion, hierarchical control), the specific combination of:
- Bezier-interpolated emotion trajectories
- Actor/director-style "emotion anchor" metaphor
- Integration with modern speech LLM architectures (CSM-1B)

...represents a **meaningful engineering contribution** but likely falls short of truly novel research contribution for top-tier venues.

---

## Related Work from Literature

### 2.1 Prosody-Conditioned TTS (2024-2025)

The field has seen rapid advancement in controllable speech synthesis:

**ProsodyFlow (COLING 2025)**
- End-to-end TTS using conditional flow matching with large speech language models
- Maps acoustic features into a "prosody latent space"
- Generates prosodic vectors conditioned on input text
- [ACL Anthology](https://aclanthology.org/2025.coling-main.518/)

**Apple's Controllable Neural TTS (NAACL 2025)**
- Uses intuitive prosodic features: pitch, pitch range, duration, energy, spectral tilt
- Demonstrates effective sentence-wise control
- [Apple Research](https://machinelearning.apple.com/research/controllable-neural-text-to-speech-synthesis)

**Comprehensive Survey: "Towards Controllable Speech Synthesis in the Era of LLMs" (EMNLP 2025)**
- Covers StyleTTS, GenerSpeech, VALL-E, NaturalSpeech 3, MegaTTS 2
- Documents the trend toward LLM-based controllable synthesis
- [EMNLP Paper](https://aclanthology.org/2025.emnlp-main.40.pdf)

### 2.2 Time-Varying Emotion Control

**T-VecTTS (OpenReview)**
- Adds time-varying emotion control to flow-matching TTS
- Identifies optimal flow step intervals for emotion determination
- Uses temporal emotion windows for fine-grained control
- [OpenReview](https://openreview.net/forum?id=x77sW1KUNE)

**Hierarchical Emotion Distribution (2024)**
- Models emotion intensity at phoneme, word, and utterance levels
- Flow-matching based framework with hierarchical ED extractor
- [arXiv](https://arxiv.org/abs/2412.12498)

**MsEmoTTS**
- Multi-scale: global (emotion class), utterance (prosody), local (syllable/phoneme)
- Direct manipulation during inference at each level

### 2.3 Disentangled Voice Synthesis

**Maestro-EVC (2025)**
- Disentangles content, speaker identity, and emotion
- Uses temporal emotion representation with prosody augmentation
- [arXiv](https://arxiv.org/html/2508.06890)

**Marco-Voice**
- Speaker-emotion disentanglement via contrastive learning
- Independent manipulation of speaker identity and emotional style
- [arXiv](https://arxiv.org/html/2508.02038)

**Key Techniques:**
- Adversarial training with gradient reversal layers (GRL)
- VAEs for disentangled latent representations
- Spherical coordinates for emotion intensity interpolation

### 2.4 Bezier Curves in Prosody Modeling

Historical work has used parametric curves for prosody:

- **Escudero et al.** - Bezier curves for speech prosody contour representation
- **Cubic spline interpolation** - Used in visual speech synthesis for syllable-level landmarks
- **PENTA model** - F0 interpolation between tonal targets
- [ResearchGate](https://www.researchgate.net/publication/234806632_Bezier_Spline_Modeling_of_Pitch-Continuous_Melodic_Expression_and_Ornamentation)

**Modern Application:**
- **Mega-TTS 2** uses prosody interpolation in discrete space
- Interpolates probabilities from multiple P-LLM outputs
- [arXiv](https://arxiv.org/html/2307.07218)

### 2.5 State Space Models (Mamba) for Audio

**Mamba (2024)**
- Linear-time sequence modeling with selective state spaces
- Outperforms Transformers on audio waveform modeling
- Reduced FID on speech generation by >50%
- 4-5x higher inference throughput without KV cache
- [arXiv](https://arxiv.org/pdf/2312.00752)

**SAMBA-ASR**
- State-of-the-art speech recognition using SSMs
- BiMamba (bidirectional) outperforms vanilla Mamba for speech
- [arXiv](https://arxiv.org/html/2501.02832v1)

**Relevance:** Mamba could replace attention in prosody encoder for efficiency, especially for long sequences.

### 2.6 DeepSeek Training Techniques

**DeepSeek-V3 Technical Report**
- Multi-head Latent Attention (MLA): 8-16x KV cache reduction
- Multi-Token Prediction (MTP): Denser training signal, faster convergence
- Auxiliary-loss-free load balancing
- [arXiv](https://arxiv.org/abs/2412.19437)

**Relevance to Voice Cloning:**
- MLA enables longer audio context
- MTP could help prosody prediction (planning ahead)
- Already partially implemented in our codebase

---

## Current Architecture Review

### 3.1 Prosody Analyzer (`backend/prosody_analyzer.py`)

Extracts 4 prosody dimensions from audio:

| Dimension | Source | Features | Current Size |
|-----------|--------|----------|--------------|
| **Semantic** | Qwen2-Audio | Emotion, tone, energy, pace | 8 values |
| **Acoustic** | Parselmouth | Pitch stats, formants, HNR, jitter, shimmer | 12 values |
| **Rhythm** | librosa | Speaking rate, pause stats, syllable count | 8 values |
| **Contour** | Parselmouth | Pitch trajectory (interpolated) | **64 points** |

**Total:** 92 values (64 of which are contour)

### 3.2 Prosody Encoder (`training/prosody_conditioning.py`)

```
ProsodyEncoder architecture:
- Semantic encoder: Linear(8 -> 512) + GELU
- Acoustic encoder: Linear(12 -> 512) + GELU
- Rhythm encoder: Linear(8 -> 512) + GELU
- Contour encoder: Linear(64 -> 512) + GELU
- Fusion: Concat (2048) -> Linear -> 4 prefix tokens
- Output: [batch, 4, 2048] prefix embeddings for CSM
```

### 3.3 Generation Pipeline (`inference/generate_with_prosody.py`)

Three modes of prosody specification:
1. **Emotion preset:** `EmotionToProsody.get_prosody("happy")`
2. **Reference audio:** Extract prosody from audio file
3. **Custom parameters:** pitch/energy/rate controls

### 3.4 Current Limitations

1. **Dense contour is opaque:** 64 points are hard to interpret or manually specify
2. **Uniform time resolution:** Every frame treated equally, no "important moments"
3. **No compositional control:** Can't say "start neutral, get angry at word X, end sad"
4. **High dimensionality:** 64 points when 4-8 keyframes might suffice

---

## The Keyframe Prosody Proposal

### 4.1 Core Concept

Replace dense 64-point pitch contour with **sparse keyframes** that define emotion "anchors":

```
CURRENT (Dense):
  Time:  [0, 0.016, 0.032, ..., 1.0]  (64 points)
  Pitch: [0.5, 0.52, 0.48, ..., 0.45]

PROPOSED (Keyframe):
  Keyframes: [
    {time: 0.0, emotion: "neutral", intensity: 0.5},
    {time: 0.3, emotion: "surprised", intensity: 0.8},  # "What?!"
    {time: 0.7, emotion: "angry", intensity: 0.9},       # "How dare you!"
    {time: 1.0, emotion: "calm", intensity: 0.3}         # Resolution
  ]
```

### 4.2 Interpolation Strategies

**Option A: Linear Interpolation**
- Simple, fast
- May produce unnatural "robotic" transitions

**Option B: Bezier Curves**
- Smooth, natural transitions
- Control points define "ease in/out"
- Familiar from animation software

**Option C: Catmull-Rom Splines**
- Pass through all keyframes exactly
- Smooth second derivative
- Used in game animation

**Recommendation:** Start with **Catmull-Rom** for prosody (guarantees hitting keyframes), with optional Bezier handles for advanced users.

### 4.3 Actor/Director Metaphor

The proposal aligns prosody control with how performers think:

| Actor Concept | Technical Mapping |
|---------------|-------------------|
| "Beat" | Keyframe position (time) |
| "Motivation" | Emotion label |
| "Intensity" | Emotion intensity (0-1) |
| "Transition" | Interpolation curve type |
| "Subtext" | Acoustic modifiers (pitch range, energy) |

This enables prompts like:
```
"Deliver the first line neutrally, then build to anger on 'never again',
ending with cold determination."
```

### 4.4 Representation Options

**Option 1: Pure Keyframes (Most Compact)**
```python
keyframes = [
    {"t": 0.0, "semantic": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], "energy": 0.5},
    {"t": 0.5, "semantic": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0], "energy": 0.9},
    {"t": 1.0, "semantic": [0.0, 0.0, 0.2, 0.0, 0.8, 0.0], "energy": 0.4},
]
# ~18 values for 3 keyframes vs 64 for dense contour
```

**Option 2: Bezier Control Points**
```python
keyframes = [
    {"t": 0.0, "value": 0.3, "handle_out": (0.1, 0.35)},
    {"t": 0.5, "value": 0.9, "handle_in": (0.4, 0.85), "handle_out": (0.6, 0.88)},
    {"t": 1.0, "value": 0.2, "handle_in": (0.9, 0.3)},
]
```

**Option 3: Hybrid (Recommended)**
- Keyframes for semantic/emotion (discrete, interpretable)
- Bezier for acoustic contour (continuous, smooth)
- Sparse rhythm markers for emphasis/pauses

---

## Technical Implementation Approach

### 5.1 New Keyframe Encoder Architecture

```python
class KeyframeProsodyEncoder(nn.Module):
    def __init__(self, config):
        # Keyframe embedding
        self.time_embedding = SinusoidalPositionalEmbedding(dim=128)
        self.emotion_embedding = nn.Embedding(num_emotions, 128)
        self.intensity_mlp = nn.Linear(1, 64)

        # Transformer to process variable-length keyframes
        self.keyframe_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=320, nhead=4),
            num_layers=2
        )

        # Interpolation module (learned or fixed)
        self.interpolator = LearnedBezierInterpolator(hidden_dim=256)

        # Output projection to CSM-compatible prefix
        self.to_prefix = nn.Linear(256, config.hidden_size * config.num_prosody_tokens)

    def forward(self, keyframes, target_length=64):
        # Embed each keyframe
        t_emb = self.time_embedding(keyframes.times)
        e_emb = self.emotion_embedding(keyframes.emotions)
        i_emb = self.intensity_mlp(keyframes.intensities)

        kf_features = torch.cat([t_emb, e_emb, i_emb], dim=-1)

        # Process with transformer
        kf_processed = self.keyframe_transformer(kf_features)

        # Interpolate to target resolution
        dense_prosody = self.interpolator(kf_processed, target_length)

        # Generate prefix tokens
        prefix = self.to_prefix(dense_prosody.mean(dim=1))
        return prefix.view(batch, num_tokens, hidden_size)
```

### 5.2 Learned Interpolation

Instead of fixed Bezier, learn optimal interpolation:

```python
class LearnedBezierInterpolator(nn.Module):
    def __init__(self, hidden_dim, num_control_points=4):
        self.control_point_predictor = nn.Linear(hidden_dim * 2, num_control_points * 2)

    def forward(self, kf_start, kf_end, num_samples):
        # Predict Bezier control points from keyframe features
        features = torch.cat([kf_start, kf_end], dim=-1)
        control_points = self.control_point_predictor(features)

        # Evaluate Bezier curve at uniform samples
        t = torch.linspace(0, 1, num_samples)
        return self.bezier_eval(control_points, t)

    def bezier_eval(self, points, t):
        # De Casteljau's algorithm for Bezier evaluation
        ...
```

### 5.3 Training Strategy

**Phase 1: Distillation from Dense Model**
1. Train dense prosody encoder normally (your current approach)
2. Generate keyframe annotations from dense contours (peak detection, change points)
3. Train keyframe encoder to match dense encoder outputs

**Phase 2: End-to-End Fine-tuning**
1. Unfreeze CSM backbone
2. Train keyframe encoder + CSM jointly
3. Use MTP loss for better prosody planning

**Phase 3: User Study**
1. A/B test: dense vs keyframe control interface
2. Measure user success at achieving target expressions
3. Collect preference data for DPO/RLHF

---

## Codebase Changes Required

### 6.1 New Files to Create

```
training/
  keyframe_prosody.py          # KeyframeProsodyEncoder
  keyframe_interpolation.py    # Bezier/Catmull-Rom utils

backend/
  keyframe_detector.py         # Extract keyframes from dense contours

inference/
  generate_with_keyframes.py   # Keyframe-based generation

frontend/
  components/
    KeyframeEditor.tsx         # Visual timeline editor
    BezierCurveEditor.tsx      # Interactive curve controls
```

### 6.2 Modifications to Existing Files

**`backend/prosody_analyzer.py`**
```diff
+ def extract_keyframes(contour: PitchContour, max_keyframes: int = 8) -> List[Keyframe]:
+     """Detect significant prosody events as keyframes."""
+     # Peak detection
+     # Change point detection
+     # Emotion boundary detection
+     ...
```

**`training/prosody_conditioning.py`**
```diff
+ class KeyframeProsodyConfig(ProsodyConfig):
+     max_keyframes: int = 8
+     interpolation_type: str = "catmull-rom"  # or "bezier", "linear"
+     learnable_interpolation: bool = True
```

**`inference/generate_with_prosody.py`**
```diff
+ def keyframes_to_dense(keyframes: List[Keyframe], config: ProsodyConfig) -> Dict[str, torch.Tensor]:
+     """Convert keyframes to dense prosody for model input."""
+     ...
```

### 6.3 Frontend (Optional but Valuable)

A visual keyframe editor would greatly enhance usability:

```tsx
// KeyframeTimeline.tsx
interface Keyframe {
  time: number;
  emotion: string;
  intensity: number;
  handles?: BezierHandles;
}

function KeyframeTimeline({ text, keyframes, onChange }) {
  // Show waveform/text with draggable keyframe markers
  // Emotion picker at each keyframe
  // Intensity slider
  // Preview playback
}
```

---

## Honest Assessment

### 7.1 Is This Novel?

**Partially Novel (3/5 stars)**

| Aspect | Novelty | Notes |
|--------|---------|-------|
| Keyframe concept | LOW | Animation industry standard |
| Bezier prosody | MEDIUM | Some prior work (Escudero), but not mainstream |
| Time-varying emotion | LOW | T-VecTTS, MsEmoTTS already do this |
| Integration with CSM | MEDIUM | Novel application to this architecture |
| Actor/director metaphor | HIGH | Unique framing, no direct prior work |
| Learned interpolation | MEDIUM-HIGH | Novel twist on fixed curves |

**What's Novel:**
- The combination of learned Bezier interpolation + modern LLM-TTS + actor-style metaphor
- Specific application to CSM-1B architecture
- User interface paradigm (timeline-based emotion choreography)

**What's NOT Novel:**
- Keyframe concept itself
- Time-varying emotion control
- Bezier curves for prosody
- Disentangled emotion/speaker representations

### 7.2 Will It Improve Quality?

**Likely Yes, But Marginally (60% confidence)**

**Potential Improvements:**
- More intuitive control for users (subjective quality up)
- Sparser representation may regularize training
- Focus on "important moments" could improve perceptual quality
- Better compositionality for complex emotions

**Potential Downsides:**
- Loss of fine-grained detail
- Interpolation artifacts
- More complexity in training pipeline
- May require more data for keyframe annotation

### 7.3 Is This Publishable?

**Borderline (2.5/5 stars)**

| Venue Type | Likelihood | Notes |
|------------|------------|-------|
| Top-tier (NeurIPS, ICML) | LOW (10%) | Needs fundamental insight |
| Speech-focused (Interspeech, ICASSP) | MEDIUM (40%) | With strong experiments |
| Applications (CHI, UIST) | HIGH (60%) | UI/UX angle is strong |
| Workshop papers | HIGH (70%) | Good for early feedback |
| arXiv/blog | CERTAIN | Valuable for visibility |

**To be publishable at Interspeech/ICASSP, you'd need:**
1. Large-scale user study (n>50)
2. Comparison with 3+ baselines (T-VecTTS, StyleTTS, etc.)
3. Ablation on interpolation methods
4. MOS/MUSHRA evaluation
5. Novel technical contribution beyond integration

### 7.4 Should You Pursue This for Job Prospects?

**Yes, But With Caveats**

**Positive Signals:**
- Demonstrates deep understanding of TTS/prosody
- Shows ability to bridge research and product thinking
- Novel UI/UX angle appeals to industry
- Working demo > paper for most jobs
- DeepSeek techniques show awareness of SOTA

**Caveats:**
- Don't oversell novelty in interviews
- Position as "engineering research" not "fundamental research"
- Emphasize the system-building aspects
- Have a compelling demo ready

**For Top Research Labs (DeepMind, OpenAI, FAIR):**
- This alone is insufficient
- Would need additional fundamental contributions
- Better as supporting work alongside more novel research

**For Industry Voice AI (Eleven Labs, Resemble, Descript):**
- Very relevant
- Demo + clean codebase matters more than papers
- This shows product intuition

---

## Challenges and Risks

### 8.1 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Interpolation artifacts | MEDIUM | A/B test multiple methods |
| Keyframe detection errors | HIGH | Manual annotation fallback |
| Training instability | MEDIUM | Start from pretrained dense encoder |
| CSM integration issues | LOW | Well-documented architecture |
| Mamba/SSM complexity | MEDIUM | Make optional, profile carefully |

### 8.2 Research Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Scooped by concurrent work | MEDIUM | Move fast, publish preprint early |
| Insufficient novelty for venues | HIGH | Target applications venues (CHI) |
| User study logistics | MEDIUM | Start recruitment early |

### 8.3 Scope Creep Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Perfect frontend before backend | HIGH | Ship CLI first, iterate |
| Over-engineering interpolation | MEDIUM | Start with Catmull-Rom |
| Training from scratch | HIGH | Fine-tune existing encoder |

---

## Recommendation

### 9.1 Should You Build This?

**YES**, with the following prioritization:

**Phase 1 (2-3 weeks): Proof of Concept**
1. Implement keyframe extraction from existing dense contours
2. Build simple Catmull-Rom interpolation
3. Train keyframe encoder to match dense encoder
4. Qualitative A/B testing

**Phase 2 (2-3 weeks): Refinement**
1. Learned Bezier interpolation
2. Basic timeline UI (React component)
3. Integration with CSM generation
4. More rigorous evaluation

**Phase 3 (4+ weeks): Publication-Ready**
1. Large-scale user study
2. Comprehensive baselines
3. Ablation studies
4. Paper writing

### 9.2 What to Prioritize

**High Priority:**
- Working demo with compelling examples
- Clean codebase others can reproduce
- Intuitive user interface
- Blog post / technical writeup

**Medium Priority:**
- Formal evaluation metrics
- Baseline comparisons
- Ablation studies

**Low Priority:**
- Mamba integration (optimization, not core value)
- Multi-speaker training
- Real-time generation

### 9.3 Expected Outcomes

**Best Case:**
- Interspeech/ICASSP workshop paper
- Open-source tool with community adoption
- Strong talking point for industry interviews
- Foundation for more ambitious prosody research

**Worst Case:**
- Working system that doesn't quite beat baselines
- Still a valuable learning experience
- Portfolio piece for job applications
- Starting point for future iteration

**Most Likely:**
- Useful tool with moderate adoption
- Blog post gets attention in voice AI community
- Clear understanding of what's hard about prosody control
- Good story for interviews

---

## References

### Prosody-Conditioned TTS
1. [ProsodyFlow (COLING 2025)](https://aclanthology.org/2025.coling-main.518/)
2. [Apple Controllable Neural TTS](https://machinelearning.apple.com/research/controllable-neural-text-to-speech-synthesis)
3. [Controllable Speech Synthesis Survey (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.40.pdf)
4. [Controlling Emotion with NL Prompts (Interspeech 2024)](https://www.isca-archive.org/interspeech_2024/bott24_interspeech.pdf)

### Time-Varying Emotion Control
5. [T-VecTTS (OpenReview)](https://openreview.net/forum?id=x77sW1KUNE)
6. [Hierarchical Emotion Control](https://arxiv.org/abs/2412.12498)
7. [ECE-TTS](https://www.mdpi.com/2076-3417/15/9/5108)

### Disentangled Voice Synthesis
8. [Maestro-EVC](https://arxiv.org/html/2508.06890)
9. [Marco-Voice](https://arxiv.org/html/2508.02038)
10. [PromptEVC](https://arxiv.org/html/2505.20678v1)

### Bezier/Spline Prosody
11. [Bezier Spline for Melodic Expression](https://www.researchgate.net/publication/234806632_Bezier_Spline_Modeling_of_Pitch-Continuous_Melodic_Expression_and_Ornamentation)
12. [Mega-TTS 2](https://arxiv.org/html/2307.07218)

### State Space Models
13. [Mamba](https://arxiv.org/pdf/2312.00752)
14. [SAMBA-ASR](https://arxiv.org/html/2501.02832v1)

### DeepSeek Techniques
15. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
16. [DeepSeek MTP Analysis](https://dataturbo.medium.com/deepseek-technical-analysis-3-multi-token-prediction-f8f3ea7eaf9c)

### Sesame CSM
17. [CSM-1B on HuggingFace](https://huggingface.co/sesame/csm-1b)
18. [SesameAILabs/csm GitHub](https://github.com/SesameAILabs/csm)

---

*Document Version: 1.0*
*Last Updated: January 2026*
