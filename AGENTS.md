# Repository Guidelines

## Project Structure & Module Organization
This repo is organized by pipeline stage. Key locations:
- `backend/`: FastAPI API + prosody analysis (`main.py`, `prosody_analyzer.py`).
- `frontend/`: Next.js UI (`app/`, `components/`, Tailwind config).
- `training/`: dataset prep and training scripts (`prepare_dataset.py`, `train_deepseek.py`, `config/`).
- `inference/`: speech generation (`generate.py`).
- `data/`: audio datasets (raw/processed/labeled/splits).
- `models/`: checkpoints and downloaded models.
- `scripts/`: setup and model download utilities.
- `docs/`: design/tech notes; `PRD.md` at repo root for architecture.

## Build, Test, and Development Commands
Common workflows:
```bash
# setup
./scripts/setup_mac.sh
./scripts/setup_linux.sh

# download models
python scripts/download_models.py

# backend API
cd backend && source venv/bin/activate && python main.py

# frontend UI
cd frontend && npm run dev

# build/lint frontend
cd frontend && npm run build
cd frontend && npm run lint

# training
cd training
python prepare_dataset.py --input ../data/labeled --output ../data/splits
python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard

# inference
cd inference
python generate.py --model ../models/checkpoints/voice_v1/best.pt --text "Hello" --output out.wav
```

## Coding Style & Naming Conventions
- Python: 4-space indentation, snake_case for functions/vars, keep modules focused.
- TypeScript/TSX: 2-space indentation, double quotes, PascalCase React components.
- Follow existing patterns in `backend/` and `frontend/app/` before introducing new abstractions.

## Testing Guidelines
- No formal test suite is configured today; smoke test by running the backend + frontend flow.
- If adding tests, use pytest conventions (`backend/tests/test_*.py`) and keep fixtures minimal.

## Commit & Pull Request Guidelines
- Commit messages follow short, imperative summaries (e.g., "Fix frontend null safety...").
- PRs should include: clear description, linked issue (if any), test or smoke steps, and UI screenshots for frontend changes.
- Note any training config changes and expected hardware impact.

## Remote Training Setup (RTX 4090)

A Windows 10 machine with RTX 4090 (24GB VRAM) is available for training via Tailscale VPN.

### Connection Details
- **Host**: `doc@$REMOTE_GPU_HOST` (Tailscale IP)
- **Project Path**: `~/dev/labfork`
- **Conda Environment**: `voice` (REQUIRED - always activate before running Python)
- **GPU Command**: `/usr/lib/wsl/lib/nvidia-smi` (WSL2 path)

### SSH Access
```bash
# Basic connection
ssh doc@$REMOTE_GPU_HOST

# Run a command
ssh doc@$REMOTE_GPU_HOST "source ~/miniconda3/bin/activate && conda activate voice && <command>"

# Check GPU status
ssh doc@$REMOTE_GPU_HOST "/usr/lib/wsl/lib/nvidia-smi"

# Attach to training tmux session
ssh doc@$REMOTE_GPU_HOST -t "tmux attach -t training"
```

### File Syncing
```bash
# Sync from Mac to 4090 (run on Mac)
rsync -avz --progress \
  --exclude 'node_modules' --exclude '.next' --exclude 'venv' \
  /Users/light/dev/web-apps/labfork/ \
  doc@$REMOTE_GPU_HOST:~/dev/labfork/
```

## Vercel Deployment (Public /watch Page)

The `/watch` page is designed for public deployment to show the AI lab in action.

### Setup
1. Install Vercel CLI: `npm i -g vercel`
2. Link project: `cd frontend && vercel link`
3. Set environment variable for backend URL (optional):
   ```bash
   vercel env add BACKEND_URL
   # Enter: https://your-ngrok-url.ngrok-free.dev
   ```

### Deploy
```bash
cd frontend

# Preview deployment (test first)
vercel

# Production deployment
vercel --prod
```

### Environment Detection
The frontend automatically detects Vercel deployment:
- **On Vercel without BACKEND_URL**: Returns demo/simulated data
- **On Vercel with BACKEND_URL**: Fetches real GPU stats from ngrok tunnel
- **Local development**: Uses localhost:8003

### Public Features
- Live viewer count (`/api/viewers`)
- GPU stats (sanitized, no IPs) (`/api/public/gpu-stats`)
- 3D lab visualization (`PublicLabView`)
- Community suggestions (`/api/suggestions`)
- No admin controls, read-only view

## Hybrid AI System (FREE + PAID)

The 4090 runs a single hybrid system using FREE local Ollama for execution and PAID Codex CLI for complex analysis + reviews.

### Architecture
- **Lab Manager** (`~/bin/lab-manager`): FREE local Claude Code using qwen3-coder-32k (Ollama)
- **Research Manager Orchestrator** (`.skills/research-manager/rm orchestrator start`): auto-spawns agents + enforces limits
- **Codex** (`codex` CLI): PAID Codex for deep analysis and ALL reviews (codex-only for review tasks)

### Available Scripts on 4090
```bash
# Start the hybrid lab system
~/bin/start-hybrid-lab        # Starts lab-manager session

# Individual components
~/bin/lab-manager             # FREE Ollama-based Claude Code
codex                         # PAID Codex CLI for analysis
~/bin/execute-plan            # Execute JSON plans from Codex
~/bin/gpu-status              # Check GPU + Ollama status

# Monitoring
tmux attach -t lab-manager    # Watch the worker
~/bin/gpu-status              # Check system status
```

### Hybrid Workflow

**For SIMPLE tasks** (use FREE Ollama):
```bash
# Lab manager handles routine work automatically
TaskList                      # See pending tasks
TaskGet #36                   # Read task details
TaskUpdate #36 status=in_progress
# ... do the work ...
TaskUpdate #36 status=completed
```

**For COMPLEX tasks** (use PAID Codex CLI):
```bash
# 1. Run Codex in a tmux session and capture output
codex -c model=codex-mini-latest "Analyze task X and produce a JSON plan"

# 2. Execute the returned plan
~/dev/labfork/scripts/execute-plan \
  ~/dev/labfork/.codex-plans/plan_*.json

# 3. Mark task complete
TaskUpdate #36 status=completed
```

### When to Use Codex vs Ollama

**Use FREE Ollama** (lab-manager):
- Reading files, searching code
- Simple edits, formatting fixes
- Running tests, checking status
- File operations, git commands
- Following existing patterns

**Use PAID Codex** (codex CLI):
- Deep code analysis
- Multi-file refactoring
- Complex architectural decisions
- Creating comprehensive tests
- Understanding novel implementations
- **All review/audit/verification tasks**

**No Anthropic API usage**:
- Do not set `ANTHROPIC_API_KEY`
- If Codex is required and unavailable, fix Codex (do NOT fall back to Ollama for reviews)

### Cost Management
- Codex plan generation: paid usage (keep to hard problems only)
- Lab manager (Ollama): FREE, runs on local GPU
- Reserve Codex for tasks where 3-min Ollama response isn't adequate

## Research Run Registry (REQUIRED)

All experimental runs must write a `run.json` to the canonical registry:
`outputs/research/runs/<run_id>/run.json`

Use the registry CLI:
```bash
python scripts/research/run_registry.py new --title "V7 verification" --task-id 6 --technique prosody_v7
python inference/quick_eval.py --checkpoint <ckpt> --run-dir <run_dir>
python scripts/research/run_registry.py update --run-dir <run_dir> --metrics-file <run_dir>/metrics.json
python scripts/research/run_registry.py review --run-dir <run_dir> --status approved --reviewer "lead"
```

Managers must review and record a decision before any technique is considered “done.”

### Active Sessions
```bash
# Check what's running
ssh doc@$REMOTE_GPU_HOST "tmux list-sessions"

# Send commands to agents
ssh doc@$REMOTE_GPU_HOST "tmux send-keys -t lab-manager 'TaskList' C-m"

# View recent activity
ssh doc@$REMOTE_GPU_HOST "tmux capture-pane -t lab-manager -p | tail -30"
```

## Automation & Safety Notes
- Prefer `npm` for frontend tooling.
- Run `npm run lint` before commits that touch `frontend/`.
- Ask before adding new production dependencies or deleting files.
- Create a git checkpoint before significant changes.
- Configure local endpoints via `NEXT_PUBLIC_API_URL` (frontend) and `DATA_DIR` (backend).
- Always use conda `voice` environment on the 4090 machine.
- Use `--dangerously-skip-permissions` flag for autonomous agent operation.
