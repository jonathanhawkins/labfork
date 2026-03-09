# LabFork

**Fork research labs. Watch AI agents implement papers. Discover synergies across domains.**

LabFork is an open platform for collaborative AI research. Create a "lab" for any research domain, and AI agents will automatically find relevant papers, implement techniques, and discover connections between different fields.

![LabFork Landing Page](docs/screenshots/landing.png)

## What is LabFork?

Think of it as GitHub for AI research, but instead of just hosting code, you're hosting *active research programs* that AI agents can work on.

**The Core Loop:**
1. **Fork** a research lab or create your own
2. **Watch** AI agents implement papers and techniques in real-time
3. **Discover** synergies across domains (e.g., a climate modeling technique that helps drug discovery)
4. **Collaborate** with others working on similar problems

## Features

### Labs
- Create research labs for any domain (voice synthesis, quant trading, robotics, etc.)
- AI agents automatically find and implement relevant papers
- Track progress with real-time dashboards

### Domains
Built-in support for 9 research domains:
- Voice Clone (TTS, prosody control, emotion synthesis)
- Quant Trading
- Game AI
- Robotics ML
- Drug Discovery
- Climate Modeling
- NLP Research
- Computer Vision
- Biotech NLP

### Meta-Agents
AI agents that look *across* labs to find synergies between different research domains.

### Live View
Watch AI agents work in real-time as they implement techniques from papers.

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+ (for backend/training)

### Setup

```bash
# Clone the repo
git clone https://github.com/jonathanhawkins/labfork
cd labfork

# Install frontend
cd frontend && npm install

# Install backend
cd ../backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
# Terminal 1: Backend API
cd backend && source venv/bin/activate && python main.py --port 8003

# Terminal 2: Frontend
cd frontend && npm run dev -- -p 3003
```

Open http://localhost:3003

## Architecture

```
labfork/
├── frontend/                # Next.js web UI
│   ├── app/
│   │   ├── page.tsx        # Landing page
│   │   ├── explore/        # Browse public labs
│   │   ├── watch/          # Live agent view
│   │   ├── lab/            # Lab management
│   │   ├── domains/        # Domain browser
│   │   └── demos/          # Research technique demos
│   └── components/
│       ├── landing/        # Landing page components
│       ├── labs/           # Lab UI components
│       └── domain/         # Domain components
│
├── backend/                 # FastAPI server
│   ├── main.py             # API endpoints
│   └── prosody_analyzer.py # Example: voice domain analysis
│
├── .domains/               # Domain configurations
│   └── voice-clone/        # Voice clone domain (example)
│
└── .skills/                # Agent skills
    └── research-manager/   # Research orchestration
```

## Tech Stack

- **Frontend:** Next.js 14, React 18, Three.js, Tailwind CSS, shadcn/ui
- **Backend:** FastAPI, PyTorch, Whisper, Transformers
- **AI:** Claude Code agents, Ollama for local inference

## Case Study: Voice Clone Lab

LabFork was originally built to explore voice cloning with prosody control. The Voice Clone domain demonstrates:

- **Multi-layer prosody analysis** (semantic, acoustic, rhythm, contour)
- **Training pipelines** with DeepSeek techniques (MTP, LoRA)
- **Real-time audio processing** and 3D visualization

See [docs/voice-clone-case-study/](docs/voice-clone-case-study/) for the full voice clone documentation.

## Contributing

We welcome contributions! Please see:
- Issues for current tasks
- PRs welcome for bug fixes and new domains

## Support This Project

If LabFork is useful to you, consider [sponsoring the project on GitHub](https://github.com/sponsors/jonathanhawkins). Your support helps keep this open source and actively maintained.

## License

MIT License

---

*Built with Next.js, Three.js, and Claude*
