# PRD: Live Voice Transformation

**Product Requirements Document**
**Version:** 1.0
**Date:** January 2026
**Author:** Voice Clone Pipeline Team

---

## Executive Summary

Live Voice Transformation enables real-time emotion modification of spoken voice. Users speak into a microphone, and the system outputs their voice with a different emotional tone (happy, sad, angry, etc.) while preserving their voice identity.

**Core Value Proposition:** Speak normally, sound emotional - without acting skills.

---

## Problem Statement

Current voice synthesis requires either:
1. **Training** - Hours of recording specific emotions (CSM fine-tuning approach)
2. **Acting** - Users must perform emotions convincingly when recording
3. **Static samples** - Pocket TTS clones whatever emotion is in the reference audio

**User Pain Points:**
- "I can't act sad/angry/excited convincingly"
- "I want my voice to sound more emotional without re-recording"
- "I want real-time emotion control during live streaming/calls"

---

## Solution Overview

A real-time voice conversion pipeline that:
1. Captures microphone input
2. Processes through an emotion transfer model
3. Outputs voice with target emotion characteristics
4. Maintains <500ms end-to-end latency

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Mic Input  │────▶│   Chunker   │────▶│  Voice Conv │────▶│   Output    │
│  (Browser)  │     │  (WebSocket)│     │  (Seed-VC)  │     │   (Audio)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────┐
                    │              Emotion Reference Library               │
                    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
                    │  │Happy│ │ Sad │ │Angry│ │Calm │ │Fear │ │Excit│  │
                    │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  │
                    │         (User's recorded voice samples)            │
                    └─────────────────────────────────────────────────────┘
```

---

## Research Summary

### Voice Conversion Technologies (2025-2026)

| Technology | Latency | Quality | Hardware | Best For |
|------------|---------|---------|----------|----------|
| [Seed-VC](https://github.com/Plachtaa/seed-vc) | ~400ms | High | GPU (RTX 3060+) | Quality-focused streaming |
| [LLVC](https://github.com/KoeAI/LLVC) | ~20ms | Medium | CPU | Ultra-low latency |
| [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 90-170ms | High | GPU | Pre-trained voice models |
| [RT-VC](https://arxiv.org/html/2506.10289v1) | ~61ms | High | CPU | Newest research (2025) |
| [StreamVC](https://arxiv.org/abs/2401.03078) | ~100ms | High | Mobile | Mobile deployment |

### Recommended: Seed-VC

**Why Seed-VC:**
- Zero-shot voice conversion (no training per user)
- Supports `--convert-style true` for accent/emotion transfer
- 400ms latency is acceptable for streaming/gaming
- Active development, good community
- Works with RTX 4090 (available hardware)

**Seed-VC Performance on RTX 4090:**
- Algorithm latency: ~150ms per chunk
- Device latency: ~100ms
- Total: ~300-400ms end-to-end
- Can run multiple streams simultaneously

### Alternative: LLVC for CPU

For M4 Pro Mac (no GPU):
- 20ms latency at 16kHz
- 2.8x faster than real-time
- Lower quality than Seed-VC but usable
- Good for development/testing

---

## Integration with Existing Codebase

### Reusable Components

| Component | Path | Reuse Strategy |
|-----------|------|----------------|
| **KeyframeTimeline** | `/frontend/components/KeyframeTimeline.tsx` | Adapt for real-time emotion slider |
| **ProsodyPredictor** | `/backend/prosody_predictor.py` | Auto-detect emotion from text being spoken |
| **Keyframe Prosody** | `/backend/keyframe_prosody.py` | Catmull-Rom for smooth emotion transitions |
| **Voice Samples** | `/data/voice_samples/` | Emotion reference library (43 samples ready) |
| **Emotion Colors** | `EMOTION_COLORS` in KeyframeTimeline | Consistent UI for emotion visualization |

### Existing Emotion Reference Library

The user has already recorded 43 voice samples with emotions:
- Happy (5), Sad (5), Angry (5), Calm (5)
- Fearful (7), Excited (5), Surprised (5), Neutral (5), Test (1)

These can serve as emotion reference audio for Seed-VC's style transfer.

### Prosody System Integration

The existing `keyframe_prosody.py` provides:
```python
@dataclass
class ProsodyKeyframe:
    time: float          # Can use for scheduled emotion changes
    emotion: str         # Maps to reference audio selection
    intensity: float     # Maps to Seed-VC similarity_cfg_rate
    energy: Optional[float]
    pitch_tendency: Optional[str]
```

This can control:
1. **Which emotion reference** to use (happy.wav, sad.wav, etc.)
2. **Blend intensity** between neutral and emotional
3. **Smooth transitions** via Catmull-Rom interpolation

---

## Functional Requirements

### FR1: Real-Time Voice Capture
- Capture microphone audio via Web Audio API
- Chunk audio into 100-300ms segments
- Stream chunks to backend via WebSocket
- Support sample rates: 16kHz, 24kHz, 48kHz

### FR2: Emotion Selection
- UI for selecting target emotion (dropdown/buttons)
- Real-time emotion switching (<100ms UI response)
- Emotion intensity slider (0-100%)
- Auto-detect mode using ProsodyPredictor

### FR3: Voice Conversion Processing
- Process audio chunks through Seed-VC
- Use user's emotion-specific voice samples as reference
- Maintain voice identity while changing emotion
- Target latency: <500ms end-to-end

### FR4: Audio Output
- Play converted audio with minimal additional latency
- Option to output to virtual audio device (for calls)
- Download/save converted audio
- Monitor levels and quality

### FR5: Emotion Blending
- Smooth transition between emotions
- Support gradual changes (fade over 2-5 seconds)
- Keyframe-based scheduled emotion changes
- Use existing Catmull-Rom interpolation

---

## Technical Specifications

### Backend WebSocket API

```python
# New endpoints for live transformation
@app.websocket("/ws/voice-transform")
async def voice_transform_websocket(websocket: WebSocket):
    """
    WebSocket for real-time voice transformation.

    Client sends:
        {
            "type": "audio_chunk",
            "data": "<base64 audio>",
            "format": "pcm_16bit",
            "sample_rate": 24000
        }
        {
            "type": "set_emotion",
            "emotion": "happy",
            "intensity": 0.8
        }

    Server sends:
        {
            "type": "audio_output",
            "data": "<base64 audio>",
            "latency_ms": 350
        }
    """
```

### Frontend Components

```typescript
// New component for live voice transformation
interface LiveTransformProps {
    availableEmotions: string[];
    voiceSamples: Record<string, string>; // emotion -> sample path
    onLatencyChange?: (ms: number) => void;
}

// Reuse existing emotion visualization
import { EMOTION_COLORS } from '@/components/KeyframeTimeline';
```

### Processing Pipeline

```python
class LiveVoiceTransformer:
    def __init__(self, voice_samples_dir: Path):
        self.seed_vc = load_seed_vc_model()
        self.emotion_refs = self._load_emotion_references(voice_samples_dir)
        self.current_emotion = "neutral"
        self.current_intensity = 0.5

    def _load_emotion_references(self, samples_dir):
        """Load user's voice samples as emotion references."""
        refs = {}
        for emotion in ["happy", "sad", "angry", "calm", "fearful", "excited"]:
            sample_path = samples_dir / f"{emotion}_1.wav"
            if sample_path.exists():
                refs[emotion] = self.seed_vc.encode_reference(sample_path)
        return refs

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk with current emotion."""
        if self.current_emotion == "neutral":
            return audio_chunk

        ref = self.emotion_refs.get(self.current_emotion)
        if ref is None:
            return audio_chunk

        return self.seed_vc.convert(
            audio_chunk,
            reference=ref,
            intensity=self.current_intensity,
        )
```

---

## UI/UX Design

### Live Transform Page (`/live`)

```
┌────────────────────────────────────────────────────────────────┐
│  Voice Clone Pipeline    [Studio] [Perform] [Generate] [Live] │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │                  🎙️ Live Voice Transform              │    │
│   │                                                       │    │
│   │   ┌─────────────────────────────────────────────┐    │    │
│   │   │  ◉ ════════════════════════════════════ 🔊  │    │    │
│   │   │     Input Level        Output Level         │    │    │
│   │   └─────────────────────────────────────────────┘    │    │
│   │                                                       │    │
│   │   [🎤 Start Listening]        Latency: 340ms        │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │  Target Emotion                                       │    │
│   │                                                       │    │
│   │  [Neutral] [😊 Happy] [😢 Sad] [😠 Angry] [😌 Calm]  │    │
│   │  [😨 Fearful] [🤩 Excited] [😲 Surprised]            │    │
│   │                                                       │    │
│   │  Intensity: ═══════════●══════════ 70%               │    │
│   │                                                       │    │
│   │  ☑️ Auto-detect from speech                          │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐    │
│   │  Voice Reference: session_20260122/happy_2.wav  ▼    │    │
│   │  Using 43 recorded samples for emotion transfer       │    │
│   └──────────────────────────────────────────────────────┘    │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Emotion Transition Visualization

Reuse the KeyframeTimeline component for scheduled emotion changes:

```
┌────────────────────────────────────────────────────────────────┐
│  Scheduled Emotions (for pre-planned speeches)                 │
│                                                                │
│  [0%]─────◆───────────◆─────────────◆────────────[100%]       │
│          neutral      happy          calm                      │
│                                                                │
│  "Start normal, get excited in the middle, end calm"          │
└────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Backend Voice Conversion (Week 1)

**Tasks:**
1. Set up Seed-VC on RTX 4090 machine
2. Create `LiveVoiceTransformer` class
3. Implement WebSocket endpoint `/ws/voice-transform`
4. Test latency and quality with recorded samples

**Deliverables:**
- Working voice conversion pipeline
- Latency benchmarks
- Quality comparison samples

### Phase 2: Frontend Real-Time UI (Week 2)

**Tasks:**
1. Create `/live` page with audio visualization
2. Implement Web Audio API capture
3. Add emotion selection UI (reuse EMOTION_COLORS)
4. WebSocket client for streaming

**Deliverables:**
- Functional live transform page
- Real-time audio visualization
- Emotion switching UI

### Phase 3: Prosody Integration (Week 3)

**Tasks:**
1. Integrate ProsodyPredictor for auto-detect mode
2. Adapt KeyframeTimeline for scheduled emotions
3. Implement Catmull-Rom blending for smooth transitions
4. Add intensity control

**Deliverables:**
- Auto-detect emotion from speech
- Scheduled emotion keyframes
- Smooth emotion transitions

### Phase 4: Polish & Optimization (Week 4)

**Tasks:**
1. Optimize latency (target <400ms)
2. Add virtual audio device output
3. CPU fallback with LLVC for M4 Pro
4. Error handling and reconnection

**Deliverables:**
- Production-ready feature
- Documentation
- Performance metrics

---

## Hardware Requirements

### Recommended (RTX 4090)
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Latency: ~300-400ms
- Can handle real-time streaming

### Minimum (CPU)
- CPU: Apple M4 Pro or Intel i9
- Latency: ~500-800ms with LLVC
- Usable for testing, not ideal for streaming

### Network
- WebSocket connection to backend
- <50ms network latency to server
- ~100kbps upload for audio streaming

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end latency | <500ms | Time from mic input to speaker output |
| Voice similarity | >80% | MOS score comparing input/output identity |
| Emotion accuracy | >70% | Human evaluation of target emotion |
| UI responsiveness | <100ms | Time from emotion click to processing change |
| Session stability | >30min | Continuous streaming without errors |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Latency too high | Poor UX | Use LLVC fallback; optimize chunk size |
| Voice quality degraded | Unusable output | Tune Seed-VC params; use better reference samples |
| GPU not available | Can't process | Implement CPU fallback with LLVC |
| WebSocket disconnects | Interrupted sessions | Auto-reconnect with state recovery |
| Reference samples don't match | Poor emotion transfer | Let users select best sample per emotion |

---

## Future Enhancements

### v2.0: Advanced Features
- **Multi-emotion blending** - Mix happy + excited for nuanced emotions
- **Pitch/speed separate controls** - Independent of emotion
- **Voice morphing** - Transform between different speakers
- **Singing voice conversion** - Seed-VC supports this

### v3.0: Platform Integration
- **Virtual audio device** - Use in Zoom/Discord/OBS
- **Mobile app** - Real-time processing on device
- **API access** - For third-party integrations
- **Streaming platforms** - Twitch/YouTube integration

---

## Appendix

### A: Seed-VC Installation

```bash
# On RTX 4090 machine (WSL)
ssh doc@100.83.78.111
cd ~/dev/voice-clone-pipeline
git clone https://github.com/Plachtaa/seed-vc
cd seed-vc
pip install -r requirements.txt

# Test real-time conversion
python real_time_gui.py --diffusion-steps 4 --block-time 0.18
```

### B: WebSocket Protocol

```javascript
// Client-side audio streaming
const ws = new WebSocket('ws://localhost:8003/ws/voice-transform');

// Send audio chunk
ws.send(JSON.stringify({
    type: 'audio_chunk',
    data: base64AudioData,
    format: 'pcm_16bit',
    sample_rate: 24000
}));

// Change emotion
ws.send(JSON.stringify({
    type: 'set_emotion',
    emotion: 'happy',
    intensity: 0.8
}));

// Receive converted audio
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'audio_output') {
        playAudio(base64Decode(msg.data));
    }
};
```

### C: Latency Breakdown

```
Component              | Time (ms)
-----------------------|----------
Audio capture          | 10-20
WebSocket send         | 5-10
Network latency        | 10-50
Server receive/decode  | 5-10
Seed-VC inference      | 150-250
Server encode/send     | 5-10
Network return         | 10-50
Client decode/play     | 10-20
-----------------------|----------
TOTAL                  | 205-420ms
```

### D: References

- [Seed-VC GitHub](https://github.com/Plachtaa/seed-vc) - Zero-shot voice conversion
- [LLVC Paper](https://arxiv.org/abs/2311.00873) - Low-latency CPU voice conversion
- [RT-VC Paper](https://arxiv.org/html/2506.10289v1) - Real-time voice conversion (2025)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - Emotion exaggeration control
- [WebRTC Audio](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) - Browser audio streaming
- [StreamVC Paper](https://arxiv.org/abs/2401.03078) - Mobile voice conversion

---

*End of PRD*
