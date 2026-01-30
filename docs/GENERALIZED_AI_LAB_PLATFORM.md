# Generalized AI Research Lab Platform

## Vision

Transform the current Voice Clone Pipeline research lab into a **platform** where anyone can:

1. **Choose a research domain** (ML, game testing, quant trading, genetic algorithms, etc.)
2. **Import arXiv papers** and have agents implement/test the techniques
3. **Use Claude Code** to help non-experts set up their own research
4. **Watch live agents** work in beautiful 3D visualization
5. **Share their lab** publicly to inspire others

---

## Part 1: Current System Analysis

### Architecture Overview

The current system consists of four major components:

```
                    +-------------------+
                    |   3D Visualization |  (Lab3D.tsx)
                    |   - Cute robot agents
                    |   - Equipment props (GPU, Mic, Speaker)
                    |   - Real-time status updates
                    +-------------------+
                            |
                    +-------------------+
                    |   Frontend APIs    |  (Next.js API routes)
                    |   /api/tasks       - Task CRUD
                    |   /api/lab/*       - Agent status, GPU stats
                    |   /api/public/*    - Sanitized public data
                    +-------------------+
                            |
                    +-------------------+
                    |   Orchestrator     |  (orchestrator.js)
                    |   - Task prioritization
                    |   - Agent spawn/kill
                    |   - Health monitoring
                    |   - Cost tracking
                    +-------------------+
                            |
                    +-------------------+
                    |   Agent Runners    |  (tmux sessions)
                    |   - Codex (paid, deep analysis)
                    |   - Ollama (free, local)
                    |   - Claude Code tools
                    +-------------------+
```

### What's Hardcoded to Voice/TTS (Must Generalize)

| Component | Current Implementation | Generalization Needed |
|-----------|----------------------|----------------------|
| **3D Props** | Microphone, Speaker, Waveform | Domain-specific prop templates |
| **Activity Types** | `training`, `recording`, `generation` | Plugin-defined activity registry |
| **Task Templates** | TTS-specific research prompts | Domain research prompt templates |
| **Research Topics** | Prosody, emotion, TTS papers | Domain arXiv categories/keywords |
| **Evaluation** | Audio quality metrics | Domain-specific eval frameworks |
| **Demo Props** | EmotionVerify3D (F0 visualization) | Domain result visualizers |
| **Backend Scripts** | `generate_with_*.py`, `train_*.py` | Plugin script conventions |

### What's Already Generalizable

| Component | Why It's Ready |
|-----------|---------------|
| **Lab3D Core** | Agent rendering, animations, camera controls domain-agnostic |
| **Orchestrator** | Task prioritization, agent lifecycle, cost tracking work for any domain |
| **Task System** | Claude Code TaskCreate/TaskUpdate/TaskList are generic |
| **Agent Types** | Codex/Ollama/Claude Code work regardless of research domain |
| **Public Viewer** | Data sanitization, viewer count, suggestions system generic |
| **3D Agent Characters** | Cute robots work universally (though could be themed) |

---

## Part 2: Generalization Architecture

### Domain Plugin System

Each research domain is defined as a **plugin** with the following structure:

```
.domains/
  voice-clone/                    # Current domain (becomes a plugin)
    domain.yaml                   # Domain configuration
    props/                        # Custom 3D props
      microphone.tsx
      speaker.tsx
      waveform.tsx
    activities/                   # Activity type definitions
      training.yaml
      recording.yaml
      generation.yaml
    prompts/                      # Agent prompt templates
      research.md
      implementation.md
      evaluation.md
    evaluation/                   # Eval scripts and metrics
      quick_eval.py
      metrics.yaml
    visualization/                # Result visualizers
      emotion-verify.tsx
      f0-chart.tsx
    branding/                     # Lab theming
      colors.ts
      logo.svg

  quant-trading/                  # Example: Quant trading domain
    domain.yaml
    props/
      chart.tsx
      orderbook.tsx
      portfolio.tsx
    activities/
      backtesting.yaml
      optimization.yaml
      live-trading.yaml
    prompts/
      strategy-research.md
      backtest-analysis.md
    evaluation/
      sharpe_ratio.py
      drawdown.py
    visualization/
      equity-curve.tsx
      trade-log.tsx
```

### domain.yaml Schema

```yaml
# .domains/voice-clone/domain.yaml
name: "Voice Clone Pipeline"
slug: "voice-clone"
description: "AI-powered voice synthesis with prosody control"
version: "1.0.0"

# Lab branding
branding:
  primaryColor: "#4ecdc4"
  accentColor: "#ff6b6b"
  logo: "branding/logo.svg"
  backgroundStyle: "sky"  # sky, space, underwater, grid

# 3D scene configuration
scene:
  groundColor: "#b8e6c1"
  ambientLight: 0.6
  props:
    - type: "supercomputer"
      position: [-6, 0, -5]
      scale: 1.3
    - type: "microphone"         # Domain-specific prop
      position: [-6, 0, 5]
      scale: 2.0
    - type: "speaker"            # Domain-specific prop
      position: [6, 0, 5]
      scale: 2.0
    - type: "server"
      position: [6, 0, -5]
      scale: 2.0

# Research configuration
research:
  arxivCategories:
    - "cs.SD"     # Sound
    - "cs.CL"     # Computation and Language
    - "eess.AS"   # Audio and Speech Processing
  keywords:
    - "prosody conditioning TTS"
    - "emotion transfer voice cloning"
    - "neural speech synthesis"
  paperSources:
    - arxiv
    - semanticScholar
    - github

# Agent prompts
prompts:
  researchPreamble: |
    You are researching voice synthesis techniques.
    Focus on prosody control, emotion conditioning, and naturalness.
  implementationPreamble: |
    Implement training scripts in training/ and inference in inference/.
    Follow the existing code patterns.

# Evaluation
evaluation:
  metrics:
    - name: "MOS"
      description: "Mean Opinion Score"
      higherIsBetter: true
    - name: "F0_RMSE"
      description: "Fundamental frequency error"
      higherIsBetter: false
  quickEval: "inference/quick_eval.py"

# Hardware requirements
hardware:
  recommendedGPU: "24GB"
  minimumRAM: "32GB"
```

### Activity Type Definition

```yaml
# .domains/voice-clone/activities/training.yaml
id: "training"
name: "Model Training"
description: "Fine-tuning or training a voice model"

# 3D visualization
visualization:
  prop: "supercomputer"        # Which prop to use (or "none" for default)
  agentAnimation: "focused"    # How agent animates
  particleEffect: "sparks"     # Particle system type
  color: 0xff6b6b             # Accent color

# Agent behavior
agent:
  preferredType: "ollama"      # Default agent type
  timeout: 120                 # Minutes before considered stuck
  priority: 100                # Higher = more important

# Progress detection
progress:
  logPatterns:
    - regex: "Epoch (\\d+)"
      extract: "epoch"
    - regex: "loss: ([\\d.]+)"
      extract: "loss"
  completion:
    - "Training complete"
    - "Best model saved"
```

### Generalized Frontend Architecture

```
frontend/
  app/
    [domain]/                    # Dynamic route for domains
      page.tsx                   # Domain-specific lab view
      watch/
        page.tsx                 # Public viewer for this domain
    create/
      page.tsx                   # "Create Your Lab" wizard
    explore/
      page.tsx                   # Discover public labs

  components/
    lab/
      Lab3D.tsx                  # Core 3D (stays mostly same)
      Agent3D.tsx                # Agent rendering (stays same)
      DomainScene.tsx            # NEW: Loads domain-specific props
      props/
        core/                    # Universal props
          Supercomputer3D.tsx
          ServerRack3D.tsx
        domains/                 # Domain-specific props loaded dynamically
          voice/
          trading/
      activities/
        ActivityRegistry.ts      # NEW: Loads domain activity configs

    domain/
      DomainProvider.tsx         # Context for current domain config
      DomainSelector.tsx         # Domain picker component
      DomainBranding.tsx         # Applies domain colors/logo
```

---

## Part 3: ArXiv Paper Ingestion System

### Flow: Paper to Task to Implementation

```
                 +----------------+
                 |  ArXiv/Papers  |
                 +----------------+
                        |
                        v
              +-------------------+
              |  Paper Analyzer   |  (Claude analyzes abstract)
              |  - Extract technique
              |  - Assess relevance
              |  - Estimate complexity
              +-------------------+
                        |
                        v
              +-------------------+
              |  Task Generator   |  (Creates structured tasks)
              |  - Research task
              |  - Implementation task
              |  - Evaluation task
              +-------------------+
                        |
                        v
              +-------------------+
              |   Orchestrator    |  (Assigns to agents)
              +-------------------+
```

### Paper Ingestion API

```typescript
// POST /api/papers/ingest
interface PaperIngestionRequest {
  source: "arxiv" | "url" | "doi" | "manual";
  identifier: string;  // arXiv ID, URL, DOI, or title
  domain: string;      // Which domain to add to
  priority?: "high" | "normal" | "low";
}

interface PaperIngestionResponse {
  paper: {
    id: string;
    title: string;
    authors: string[];
    abstract: string;
    url: string;
    relevanceScore: number;  // 0-100
  };
  tasks: {
    research: TaskPreview;
    implementation?: TaskPreview;
    evaluation?: TaskPreview;
  };
  estimatedEffort: "hours" | "days" | "weeks";
}
```

### Paper Analysis Prompt

```markdown
## Paper Analysis Agent

Analyze this research paper for implementation in the {{domain.name}} lab.

**Paper**: {{paper.title}}
**Abstract**: {{paper.abstract}}

**Domain Context**: {{domain.description}}
**Existing Techniques**: {{domain.existingTechniques}}

Please provide:

1. **Relevance Score (0-100)**: How applicable is this to our domain?
2. **Key Technique**: What's the main innovation?
3. **Implementation Complexity**: Easy/Medium/Hard
4. **Required Resources**: GPU time, datasets, dependencies
5. **Integration Points**: How does this connect to existing code?

6. **Proposed Tasks**:
   - Research task (understand the technique)
   - Implementation task (build it)
   - Evaluation task (test it)

Output as structured JSON.
```

---

## Part 4: "Create Your Lab" Onboarding Flow

### User Journey

```
+-------------------+     +-------------------+     +-------------------+
|   Landing Page    | --> |  Domain Selector  | --> |  Lab Customizer   |
|   "Create Lab"    |     |  - Pre-built      |     |  - Name your lab  |
|                   |     |  - Custom         |     |  - Choose props   |
|                   |     |  - Import config  |     |  - Set branding   |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
|   Watch Agents    | <-- |  First Research   | <-- |  Add First Paper  |
|   Work Live!      |     |  Task Running     |     |  or Define Goal   |
+-------------------+     +-------------------+     +-------------------+
```

### Step 1: Domain Selection

```tsx
// /create - Domain selector page
const PRESET_DOMAINS = [
  {
    id: "voice-clone",
    name: "Voice Synthesis Lab",
    icon: "microphone",
    description: "Clone voices with prosody control",
    arxivCategories: ["cs.SD", "cs.CL"],
    difficulty: "intermediate",
  },
  {
    id: "game-ai",
    name: "Game AI Lab",
    icon: "gamepad",
    description: "Train agents to play games",
    arxivCategories: ["cs.AI", "cs.LG"],
    difficulty: "beginner",
  },
  {
    id: "quant-trading",
    name: "Quant Trading Lab",
    icon: "chart",
    description: "Genetic algorithms for trading strategies",
    arxivCategories: ["q-fin.PM", "cs.NE"],
    difficulty: "advanced",
  },
  {
    id: "computer-vision",
    name: "Computer Vision Lab",
    icon: "eye",
    description: "Implement and compare CV papers",
    arxivCategories: ["cs.CV"],
    difficulty: "intermediate",
  },
  {
    id: "nlp",
    name: "NLP Lab",
    icon: "text",
    description: "Language model experiments",
    arxivCategories: ["cs.CL", "cs.LG"],
    difficulty: "intermediate",
  },
  {
    id: "robotics-sim",
    name: "Robotics Simulation Lab",
    icon: "robot",
    description: "Robot control algorithms in simulation",
    arxivCategories: ["cs.RO", "cs.AI"],
    difficulty: "advanced",
  },
  {
    id: "custom",
    name: "Custom Lab",
    icon: "wrench",
    description: "Define your own research domain",
    difficulty: "expert",
  },
];
```

### Step 2: Lab Configuration Wizard

```tsx
// Lab configuration form
interface LabConfig {
  // Basic info
  name: string;
  description: string;
  visibility: "public" | "private" | "unlisted";

  // Research focus
  researchGoal: string;  // Natural language description
  arxivCategories: string[];
  keywords: string[];

  // Hardware
  hasGPU: boolean;
  gpuModel?: string;
  remoteGPU?: {
    host: string;
    user: string;
    // Claude helps set up SSH
  };

  // Branding
  primaryColor: string;
  labIcon: string;  // From predefined set
  backgroundTheme: "sky" | "space" | "underwater" | "grid" | "forest";

  // Initial content
  initialPapers?: string[];  // arXiv IDs to start with
  initialGoal?: string;      // Or natural language goal
}
```

### Step 3: Claude-Assisted Setup

For users who don't know where to start, Claude Code helps:

```markdown
## Lab Setup Assistant

**User Goal**: "I want to explore reinforcement learning for robot locomotion"

**Claude Analysis**:
Based on your goal, I recommend:

1. **Domain**: Robotics Simulation Lab
2. **arXiv Categories**: cs.RO, cs.LG
3. **Key Papers to Start**:
   - "Legged Robots that Keep on Learning" (2023)
   - "Learning Agile Locomotion via Adversarial Training" (2024)
4. **Suggested Tasks**:
   - Set up MuJoCo simulation environment
   - Implement PPO baseline
   - Reproduce paper #1's key result

Should I create these initial tasks for your lab?
```

---

## Part 5: Example Domain - Quant Trading Lab

### domain.yaml

```yaml
name: "Quant Trading Lab"
slug: "quant-trading"
description: "Genetic algorithms combining trading strategies, backtesting, and optimization"
version: "1.0.0"

branding:
  primaryColor: "#22c55e"  # Green for money
  accentColor: "#ef4444"   # Red for losses
  backgroundStyle: "grid"  # Financial grid aesthetic

scene:
  groundColor: "#1a1a2e"   # Dark trading floor
  ambientLight: 0.4
  props:
    - type: "trading-terminal"
      position: [-6, 0, -5]
      scale: 1.5
    - type: "chart-wall"
      position: [0, 2, -6]
      scale: 2.0
    - type: "server-rack"
      position: [6, 0, -5]
    - type: "orderbook"
      position: [-6, 0, 5]

research:
  arxivCategories:
    - "q-fin.PM"     # Portfolio Management
    - "q-fin.TR"     # Trading
    - "cs.NE"        # Neural and Evolutionary Computing
  keywords:
    - "genetic algorithms trading"
    - "reinforcement learning market making"
    - "portfolio optimization"
    - "pairs trading"
  dataSources:
    - yfinance
    - alpaca
    - binance

activities:
  - id: "backtesting"
    name: "Strategy Backtest"
    prop: "trading-terminal"
    color: 0x22c55e
  - id: "optimization"
    name: "Parameter Optimization"
    prop: "server-rack"
    color: 0x3b82f6
  - id: "live-paper"
    name: "Paper Trading"
    prop: "orderbook"
    color: 0xf59e0b

evaluation:
  metrics:
    - name: "Sharpe Ratio"
      higherIsBetter: true
    - name: "Max Drawdown"
      higherIsBetter: false
    - name: "Win Rate"
      higherIsBetter: true
    - name: "Profit Factor"
      higherIsBetter: true
```

### Quant-Specific 3D Props

```typescript
// ChartWall3D.tsx - Real-time candlestick charts on a wall display
export function createChartWall3D(options: ChartWallOptions): ChartWall3DRefs {
  const group = new THREE.Group();

  // Large display screen
  const screen = createLargeScreen(options.position);

  // Candlestick visualization (updated in real-time)
  const candlesticks = createCandlestickMesh();

  // Scrolling ticker tape at bottom
  const ticker = createTickerTape();

  return { group, screen, candlesticks, ticker };
}

// OrderBook3D.tsx - Visualizes bid/ask depth
export function createOrderBook3D(options: OrderBookOptions): OrderBook3DRefs {
  // Vertical bars showing order depth
  // Green on left (bids), Red on right (asks)
  // Animates as orders change
}

// TradingTerminal3D.tsx - Multi-monitor trading desk
export function createTradingTerminal3D(options: TerminalOptions): Terminal3DRefs {
  // Multiple screens showing different charts
  // Keyboard, mouse for the trading agent
  // Real-time P&L display
}
```

### Quant-Specific Activities

```yaml
# backtesting.yaml
id: "backtesting"
name: "Strategy Backtest"
description: "Running historical simulation of trading strategy"

visualization:
  prop: "trading-terminal"
  agentAnimation: "focused"
  particleEffect: "data-flow"
  color: 0x22c55e

progress:
  logPatterns:
    - regex: "Simulating (\\d{4}-\\d{2}-\\d{2})"
      extract: "date"
    - regex: "Portfolio value: \\$([\\d,]+)"
      extract: "portfolio_value"
    - regex: "Trades executed: (\\d+)"
      extract: "trade_count"
```

---

## Part 6: Implementation Roadmap

### Phase 1: Core Generalization (2-3 weeks)

1. **Extract Domain Plugin System**
   - Create `.domains/` directory structure
   - Move voice-clone specific code to plugin
   - Create domain.yaml schema

2. **Generalize Activity System**
   - Make ActivityRegistry load from domain config
   - Abstract prop creation to use domain definitions
   - Update Lab3D to load domain scene config

3. **Generalize Orchestrator**
   - Make research topics come from domain config
   - Abstract task templates to use domain prompts
   - Keep core scheduling/lifecycle logic unchanged

### Phase 2: Domain Selection UI (1-2 weeks)

4. **Domain Selector Component**
   - Grid of preset domains
   - Custom domain option
   - Import existing domain.yaml

5. **Dynamic Routing**
   - `/[domain]` for domain-specific lab views
   - `/[domain]/watch` for public viewers
   - Domain context provider

### Phase 3: Paper Ingestion (2-3 weeks)

6. **Paper Fetching Service**
   - arXiv API integration
   - Semantic Scholar API
   - GitHub repo discovery

7. **Paper Analysis Agent**
   - Claude-powered relevance scoring
   - Task generation from papers
   - Duplicate detection

8. **Paper Management UI**
   - Add paper by URL/arXiv ID
   - View paper queue
   - Track implementation status

### Phase 4: Create Your Lab Wizard (2-3 weeks)

9. **Wizard Flow**
   - Multi-step configuration
   - Claude-assisted goal definition
   - Hardware setup helper

10. **Lab Scaffolding**
    - Generate domain.yaml from wizard
    - Create initial directory structure
    - Set up initial tasks

### Phase 5: Multi-Lab & Sharing (2-3 weeks)

11. **Lab Registry**
    - Database of public labs
    - Search/filter by domain
    - Lab metrics (stars, activity)

12. **Lab Sharing**
    - Generate shareable URLs
    - Embed widget for external sites
    - Lab showcase page

### Phase 6: Social Layer & Sharing (2-3 weeks)

13. **Lab Portal System**
    - User lab URLs (`/labs/username/lab-slug`)
    - Lab profile pages with stats
    - Live 3D viewer for public labs

14. **Research Sharing Features**
    - Result cards for findings
    - Model sharing integration
    - Interactive demos

15. **Discovery & Community**
    - Explore page with filters
    - Trending labs
    - Domain directories
    - Star/fork system

16. **Social Engagement**
    - Comments on results
    - Research suggestions
    - Lab activity feeds
    - Following system

### Phase 7: Multi-Source Input (1-2 weeks)

17. **Research Source Integrations**
    - arXiv API (already planned)
    - GitHub repository analysis
    - Direct paper upload
    - Custom research goals

18. **Source Analysis**
    - Multi-source relevance scoring
    - Task generation from any source
    - Source-specific metadata extraction

### Phase 8: Easy Deployment (2-3 weeks)

19. **One-Click Deploy**
    - Vercel template with wizard
    - Docker Compose for local
    - Cloud deploy scripts (AWS, GCP, RunPod)

20. **Setup Wizard**
    - Hardware detection and config
    - SSH connection helper (for remote GPU)
    - Automated dependency installation
    - Domain selection and customization

21. **Deployment Modes**
    - Local (CPU or GPU)
    - Remote (SSH to existing machine)
    - Cloud (provision new instance)

### Phase 9: Example Domains (Ongoing)

22. **Build 5-7 Example Domains**
    - Voice Clone Lab (existing)
    - Quant Trading Lab
    - Game AI Lab
    - Computer Vision Lab
    - NLP Experiments Lab
    - Robotics Simulation Lab
    - Biology/Chemistry ML Lab

---

## Part 7: Technical Considerations

### Database Needs

For multi-lab support, we need persistent storage:

```typescript
// Lab schema
interface Lab {
  id: string;
  slug: string;
  name: string;
  description: string;
  domain: string;
  config: DomainConfig;
  ownerId: string;
  visibility: "public" | "private" | "unlisted";
  createdAt: Date;
  stats: {
    tasksCompleted: number;
    papersProcessed: number;
    agentHours: number;
    stars: number;
    viewers: number;
  };
}

// Could use:
// - SQLite (simple, file-based)
// - Postgres (scalable)
// - Supabase (managed, real-time)
```

### Authentication

For multi-user labs:

```typescript
// Options:
// - Clerk (already in project for voice-clone)
// - NextAuth.js (flexible)
// - Supabase Auth (if using Supabase DB)
```

### 3D Prop Loading

Dynamic prop loading for domain-specific visualizations:

```typescript
// Lazy load domain-specific props
const DomainProps = {
  'voice-clone': () => import('.domains/voice-clone/props'),
  'quant-trading': () => import('.domains/quant-trading/props'),
  // ...
};

async function loadDomainProps(domain: string) {
  const loader = DomainProps[domain];
  if (!loader) return null;
  return await loader();
}
```

### Cost Management

Per-lab cost tracking:

```typescript
interface LabCosts {
  labId: string;
  daily: number;
  weekly: number;
  monthly: number;
  limits: {
    dailyCap: number;
    weeklyCap: number;
    monthlyCap: number;
  };
  // Auto-pause lab when limits hit
}
```

---

## Part 8: Social Layer - Lab Portals & Research Sharing

### Vision: GitHub for AI Research Labs

Instead of just sharing code, users share **live research labs** where others can:
- Watch agents work in real-time
- See research results as they're produced
- Fork labs to build upon existing work
- Contribute suggestions and ideas

### Lab Portal System

```
Main Platform (airesearch.ai or similar)
├── /explore                    # Discover public labs
├── /trending                   # Most active labs this week
├── /domains/voice-clone        # Domain-specific directory
├── /domains/quant-trading      # Another domain
│
User Labs (personalized URLs)
├── /labs/username/my-quant-lab        # User's lab portal
├── /labs/username/voice-experiments   # Another lab
└── /labs/username/game-ai-testing     # Yet another lab
```

### Lab Portal Features

Each user's lab gets its own public portal:

```typescript
// Lab Portal Page (/labs/username/lab-slug)
interface LabPortal {
  // Live view
  live3D: Lab3DViewer;           // Watch agents work
  activityFeed: ActivityLog;     // Real-time updates

  // Research showcase
  publications: {
    papers: PaperCard[];         // Papers implemented
    results: ResultCard[];       // Key findings
    models: ModelCard[];         // Trained models
    demos: DemoCard[];           // Interactive demos
  };

  // Community features
  stats: {
    totalTasks: number;
    papersImplemented: number;
    agentHours: number;
    stars: number;               // GitHub-style stars
    forks: number;               // Other labs forked from this
    contributors: number;
  };

  // Engagement
  comments: Comment[];           // Discussion
  suggestions: Suggestion[];     // Community ideas
  related: Lab[];                // Similar labs
}
```

### Research Sharing Features

#### 1. Result Cards

```typescript
// Shareable result cards for showcasing findings
interface ResultCard {
  id: string;
  labId: string;
  title: string;                 // "Achieved 4.2 MOS with EmoTTS"
  description: string;
  type: "model" | "demo" | "finding" | "comparison";

  // Media
  thumbnail?: string;
  visualizations?: ChartData[];
  audioSamples?: AudioFile[];    // For TTS labs
  videos?: VideoFile[];          // For game AI, robotics

  // Metrics
  metrics: {
    name: string;
    value: number;
    unit: string;
    higherIsBetter: boolean;
  }[];

  // Context
  paper?: {
    arxivId: string;
    title: string;
  };
  reproducible: boolean;         // Can others run this?

  // Social
  likes: number;
  shares: number;
  comments: Comment[];
}
```

#### 2. Model Sharing

```typescript
// Share trained models with community
interface SharedModel {
  id: string;
  labId: string;
  name: string;
  description: string;

  // Model details
  architecture: string;
  parameters: string;            // "1.5B params"
  framework: "pytorch" | "tensorflow" | "jax";

  // Performance
  benchmark: {
    dataset: string;
    metrics: Metric[];
  }[];

  // Download
  downloadUrl?: string;          // If user wants to share weights
  huggingfaceId?: string;        // Link to HF model card

  // Usage
  quickStart: string;            // Code snippet to use model
  colab?: string;                // Google Colab notebook
}
```

#### 3. Interactive Demos

```typescript
// Allow users to try the research results
interface Demo {
  id: string;
  labId: string;
  type: "inference" | "visualization" | "comparison";

  // For voice labs
  tts?: {
    model: string;
    inputText: string;
    outputAudio: string;
  };

  // For trading labs
  backtest?: {
    strategy: string;
    historicalData: TimeSeriesData;
    results: BacktestResults;
  };

  // For game AI
  gameplay?: {
    game: string;
    agent: string;
    replayUrl: string;           // Video of agent playing
  };
}
```

### Main Platform Discovery Features

#### 1. Explore Page (`/explore`)

```typescript
interface ExploreView {
  // Filters
  filters: {
    domain: string[];
    activity: "active" | "completed" | "all";
    difficulty: "beginner" | "intermediate" | "advanced";
    hasGPU: boolean;
  };

  // Display modes
  viewMode: "grid" | "list" | "map";  // Map shows 3D mini-previews

  // Lab cards
  labs: LabCard[];
}

interface LabCard {
  id: string;
  name: string;
  owner: User;
  domain: string;
  description: string;
  thumbnail: string;             // 3D snapshot of lab

  // Activity indicators
  liveNow: boolean;              // Agents currently working
  lastActive: Date;

  // Stats
  stars: number;
  papersImplemented: number;
  agentHours: number;

  // Quick preview
  recentActivity: Activity[];    // Last 5 activities
  topResults: ResultCard[];      // Best findings
}
```

#### 2. Trending Labs (`/trending`)

```typescript
interface TrendingLabs {
  // Time-based trending
  today: Lab[];
  thisWeek: Lab[];
  thisMonth: Lab[];

  // Category trending
  byDomain: {
    [domain: string]: Lab[];
  };

  // Special categories
  mostStarred: Lab[];
  mostForked: Lab[];
  mostActiveAgents: Lab[];       // Highest agent hours
  bestResults: Lab[];            // Based on metrics
}
```

#### 3. Domain Directories (`/domains/voice-clone`)

```typescript
// Domain-specific discovery
interface DomainDirectory {
  domain: DomainConfig;

  // Featured labs
  featured: Lab[];               // Hand-picked by moderators

  // Community labs
  popular: Lab[];
  recent: Lab[];

  // Domain resources
  papers: {
    trending: ArxivPaper[];      // Most implemented papers
    recent: ArxivPaper[];
    classic: ArxivPaper[];       // Foundational papers
  };

  // Domain leaderboards
  leaderboards: {
    name: string;                // e.g., "Best MOS Score"
    metric: string;
    entries: LeaderboardEntry[];
  }[];
}
```

### Lab Profile & Analytics

```typescript
// User profile showing all their labs
interface UserProfile {
  username: string;
  bio: string;
  avatar: string;

  // Labs
  labs: Lab[];

  // Achievements
  stats: {
    totalLabs: number;
    totalPapers: number;
    totalAgentHours: number;
    totalStars: number;
    contributions: number;        // Contributions to others' labs
  };

  // Activity graph (GitHub-style)
  activityHeatmap: {
    date: Date;
    count: number;                // Tasks completed that day
  }[];

  // Recognition
  badges: Badge[];                // "First Paper", "100 Tasks", etc.
}
```

### One-Click Deployment

Make it trivial for anyone to run their own lab:

#### Option 1: Vercel Template

```bash
# Click "Deploy to Vercel" button on main site
# User gets prompted:
1. Name your lab
2. Choose domain (voice-clone, quant-trading, etc.)
3. Connect hardware (optional - can use free CPU-only mode)
   - SSH to your machine with GPU
   - Or use cloud GPU (RunPod, Lambda Labs)
4. Deploy

# Result: Live lab at username.airesearch.ai
```

#### Option 2: Docker Compose (Local)

```bash
# One command to run locally
curl -fsSL https://airesearch.ai/install.sh | sh

# Prompts:
# 1. Domain to start with?
# 2. Use local GPU or cloud?
# 3. Public or private lab?

# Starts:
# - Frontend on localhost:3000
# - Backend API on localhost:8003
# - Orchestrator in tmux
# - Auto-opens browser to lab
```

#### Option 3: Cloud Deploy Script

```bash
# Deploy to user's cloud account
./deploy.sh --cloud=aws --gpu=g4dn.xlarge
./deploy.sh --cloud=gcp --gpu=t4
./deploy.sh --cloud=runpod --gpu=rtx4090

# Script handles:
# - VM provisioning
# - GPU driver setup
# - Ollama installation
# - Lab stack deployment
# - Domain/SSL setup
```

### Deployment Configuration

```yaml
# .lab/deploy.yaml
name: "My Quant Trading Lab"
domain: "quant-trading"
visibility: "public"

# Hardware
hardware:
  mode: "remote"  # local, remote, or cloud
  remote:
    host: "your-gpu-host"
    user: "doc"
    gpu: "RTX 4090"
  # OR cloud:
  #   provider: "runpod"
  #   instance: "rtx4090"
  #   region: "us-west"

# Agents
agents:
  budget:
    daily: 10.00
    monthly: 200.00
  models:
    free: "qwen3-coder-32k"    # Ollama model
    paid: "claude-sonnet-4"    # For complex tasks

# Public portal
portal:
  enabled: true
  customDomain: "quant.mysite.com"  # Optional
  sharing:
    allowForks: true
    allowSuggestions: true
    showMetrics: true
```

### Social Features Implementation

#### Star/Fork System

```typescript
// GitHub-style stars and forks
interface LabActions {
  star(labId: string): Promise<void>;
  unstar(labId: string): Promise<void>;
  fork(labId: string, newName: string): Promise<Lab>;  // Clone config

  // Forking copies:
  // - domain.yaml
  // - Initial tasks
  // - Prompt templates
  // Does NOT copy:
  // - Trained models (unless explicitly shared)
  // - User data
}
```

#### Comments & Suggestions

```typescript
// Community engagement
interface LabDiscussion {
  // Comments on results
  comments: {
    targetType: "lab" | "result" | "model" | "demo";
    targetId: string;
    author: User;
    text: string;
    createdAt: Date;
  }[];

  // Research suggestions
  suggestions: {
    author: User;
    text: string;                // "Have you tried X paper?"
    type: "paper" | "technique" | "improvement";
    paperRef?: string;           // arXiv ID if suggesting paper
    votes: number;               // Community upvotes
    status: "open" | "accepted" | "implemented" | "rejected";
  }[];
}
```

#### Lab Feed

```typescript
// Activity feed for following labs
interface LabFeed {
  userId: string;
  following: string[];           // Lab IDs user follows

  feed: FeedItem[];
}

interface FeedItem {
  labId: string;
  lab: LabCard;
  type: "result" | "paper_implemented" | "model_trained" | "milestone";

  // Content
  title: string;                 // "Achieved 95% win rate on CartPole"
  description: string;
  media?: MediaFile[];

  // Social
  likes: number;
  comments: number;
  createdAt: Date;
}
```

### Multi-Source Research Input

Expand beyond arXiv to support diverse research sources:

```typescript
interface ResearchSource {
  type: "arxiv" | "github" | "paper_url" | "custom_goal";

  // arXiv paper
  arxiv?: {
    id: string;                  // "2401.12345"
  };

  // GitHub repository
  github?: {
    url: string;
    readme: string;
    papers?: string[];           // Linked papers
  };

  // Direct paper upload
  paper?: {
    title: string;
    abstract: string;
    pdfUrl?: string;
    authors: string[];
  };

  // Custom research goal
  customGoal?: {
    description: string;         // Natural language goal
    domain: string;
    expectedOutcome: string;
  };
}

// Examples:
// 1. arXiv: "Implement techniques from arXiv:2401.12345"
// 2. GitHub: "Reproduce results from github.com/openai/gpt-3"
// 3. Paper URL: "Implement paper at neurips.cc/paper/12345"
// 4. Custom: "Build a trading bot that combines momentum + mean reversion"
```

### Main Platform Landing Page

```typescript
// Homepage showcasing the platform
interface LandingPage {
  hero: {
    title: "AI Research Labs - Powered by Claude Code";
    subtitle: "Watch AI agents implement research papers in real-time";
    cta: "Create Your Lab";
    demo: Lab3DPreview;          // Animated 3D preview
  };

  // Live activity ticker
  liveLabs: {
    count: number;               // "127 labs running right now"
    recent: Activity[];          // Scrolling feed of live activity
  };

  // Featured labs
  featured: LabCard[];

  // Domains showcase
  domains: {
    name: string;
    icon: string;
    labCount: number;
    trending: Lab[];
  }[];

  // Success stories
  showcase: {
    title: string;
    lab: Lab;
    achievement: string;         // "Reproduced SOTA results"
    testimonial?: string;
  }[];

  // Quick start
  getStarted: {
    steps: string[];
    estimatedTime: string;       // "5 minutes to first lab"
  };
}
```

---

## Part 9: Why This Will Inspire People

### Accessibility

- **Non-experts can participate**: Claude helps them understand what to research
- **Visual feedback**: Seeing cute robots work makes AI research tangible
- **Low barrier**: Can start with pre-built domains and zero configuration
- **One-click deployment**: Launch your own lab in minutes

### Educational Value

- **Watch AI solve problems**: Learn by observing agent behavior
- **Paper-to-implementation pipeline**: Demystifies research reproduction
- **Shared learning**: Public labs let others learn from your experiments
- **Interactive demos**: Try research results yourself

### Community Aspect

- **Lab showcase**: Discover what others are researching
- **Fork labs**: Start from someone else's setup
- **Suggestions**: Public can suggest ideas for labs to pursue
- **Social proof**: Stars, comments, and engagement
- **Research sharing**: Publish results without writing papers
- **Collaborative discovery**: Community-driven research exploration

### The "Wow Factor"

- **Beautiful 3D visualization** makes research feel like a game
- **Real-time agent activity** is mesmerizing to watch
- **Domain theming** lets each lab feel unique (trading floor vs. voice studio)
- **Live research feed** - watch the community's discoveries unfold
- **Your own research portal** - professional showcase with zero effort

### Network Effects

- **More labs → more discoveries**: Each lab contributes to collective knowledge
- **More forks → faster progress**: Build on others' work instead of starting from scratch
- **More domains → broader impact**: Research democratization across all fields
- **More users → better plugins**: Community contributes domain templates

---

## Summary

This platform transforms isolated research projects into a **social, visual, accessible ecosystem** where:

1. **Domains are plugins** - Easy to add new research areas
2. **Papers flow to implementations** - Automated research reproduction
3. **Anyone can participate** - Claude assists non-experts
4. **Labs are shareable** - Inspire others with your research
5. **Visualization is universal** - Cute robots work in any domain
6. **Community drives discovery** - Research becomes social and collaborative
7. **One-click deployment** - Launch your lab in minutes
8. **Multiple input sources** - arXiv, GitHub, custom goals all supported

The voice clone lab becomes the **first example domain** in a much larger platform for AI research democratization - **a GitHub for live AI research labs**.
