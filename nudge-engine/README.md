# Nudge Engine

**The Jiminy Cricket for AI coding agents.** A persistent little conscience that keeps your agents productive — without you having to manage them.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Cloudflare Workers](https://img.shields.io/badge/runs%20on-Cloudflare%20Workers-F38020.svg)](https://workers.cloudflare.com)
[![Cost](https://img.shields.io/badge/cost-%240%2Fmonth-brightgreen.svg)](#cost)

```
Every 15 minutes:
  1. Observe  — What's happening in the project?
  2. Decide   — What tasks should exist?
  3. Nudge    — Is anything stuck?
  4. Cleanup  — Prune the dead weight.
```

---

## Why

AI coding agents (Claude Code, Codex, Ollama) are powerful but **reactive**. They sit there waiting for you to tell them what to do. You become the bottleneck — the project manager feeding tasks to robots.

Nudge Engine is the missing cron loop. It watches your project, creates small tasks when things need attention, dispatches them to whatever agents are online, handles failures, and cleans up after itself.

**You stop being the project manager. The loop does it for you.**

---

## How It Works

Three HTTP endpoints. That's the whole protocol.

```
POST /register  →  "I exist, give me a token"
POST /poll      →  "Got anything for me?"
POST /report    →  "Here's what happened"
```

Any tool that can `curl` is a valid worker. No SDK. No framework. No lock-in.

A cron job runs every 15 minutes on Cloudflare Workers (free tier). It creates tasks using simple `if/then` rules, nudges stuck work, and cleans up dead workers. **No LLM in the loop.** Fast, free, deterministic.

---

## Deploy in 2 Minutes

### One-line setup

```bash
git clone https://github.com/labfork/nudge-engine.git
cd nudge-engine
./setup.sh
```

The script installs deps, creates the D1 database, applies the schema, and deploys.

### Manual setup

```bash
npm install

# Create database — copy the database_id into wrangler.toml
npx wrangler d1 create nudge-engine-db

# Apply schema
npx wrangler d1 execute nudge-engine-db --remote --file=schema.sql

# Set admin key (optional)
npx wrangler secret put ADMIN_API_KEY

# Deploy
npx wrangler deploy
```

### Run locally

```bash
npm run dev
# http://localhost:8787
```

---

## Quick Start

### 1. Register a worker

```bash
ENGINE="https://nudge-engine.YOUR-SUBDOMAIN.workers.dev"

curl -s -X POST $ENGINE/register \
  -H "Content-Type: application/json" \
  -d '{"name": "my-agent", "type": "claude-code"}' | jq .
```

```json
{ "id": "w_abc123", "token": "a1b2c3d4...", "registered": true }
```

### 2. Poll for work

```bash
curl -s -X POST $ENGINE/poll \
  -H "Authorization: Bearer a1b2c3d4..." | jq .
```

### 3. Report results

```bash
curl -s -X POST $ENGINE/report \
  -H "Authorization: Bearer a1b2c3d4..." \
  -H "Content-Type: application/json" \
  -d '{
    "taskId": "t_xyz789",
    "success": true,
    "result": {"summary": "Fixed the flaky test"}
  }'
```

### 4. Create a task manually

```bash
curl -s -X POST $ENGINE/tasks \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "action": "fix-tests",
    "description": "The auth tests are failing. Fix them.",
    "priority": 8
  }'
```

### 5. Check stats

```bash
curl -s $ENGINE/stats | jq .
```

---

## Works With Any Agent

Nudge Engine doesn't care what AI tool you use. If it can run in a shell, it can be a worker.

| Agent | Worker Script | How It Runs Tasks |
|-------|--------------|-------------------|
| **Claude Code** | `examples/claude-code-worker.sh` | `claude -p "task prompt"` |
| **OpenAI Codex** | `examples/codex-worker.sh` | `codex exec --full-auto "task prompt"` |
| **Ollama** | `examples/ollama-worker.sh` | Local LLM via `/api/generate` |
| **Any/Auto** | `examples/universal-worker.sh` | Auto-detects which CLI is installed |
| **Custom** | `examples/bash-worker.sh` | Your logic here |

### Quick start with any agent

```bash
# Register
ENGINE="https://nudge-engine.YOUR-SUBDOMAIN.workers.dev"
TOKEN=$(curl -s -X POST $ENGINE/register \
  -H "Content-Type: application/json" \
  -d '{"name": "my-worker", "type": "claude-code"}' | jq -r .token)

# Option A: Auto-detect (uses whatever CLI is installed)
ENGINE=$ENGINE TOKEN=$TOKEN ./examples/universal-worker.sh

# Option B: Specific agent
ENGINE=$ENGINE TOKEN=$TOKEN ./examples/claude-code-worker.sh
ENGINE=$ENGINE TOKEN=$TOKEN ./examples/codex-worker.sh
ENGINE=$ENGINE TOKEN=$TOKEN MODEL=qwen3:8b ./examples/ollama-worker.sh
```

### Mix agents on the same engine

Different agents can work on the same task queue. Register each one separately:

```bash
# Claude Code for complex refactoring
curl -s -X POST $ENGINE/register \
  -d '{"name": "claude-heavy", "type": "claude-code", "capabilities": ["code","refactor"]}' \
  -H "Content-Type: application/json"

# Codex for code review
curl -s -X POST $ENGINE/register \
  -d '{"name": "codex-reviewer", "type": "codex", "capabilities": ["review"]}' \
  -H "Content-Type: application/json"

# Local Ollama for triage (free, fast)
curl -s -X POST $ENGINE/register \
  -d '{"name": "ollama-triage", "type": "ollama", "capabilities": ["triage"]}' \
  -H "Content-Type: application/json"
```

### Observers

Observers feed information into the engine so the rules can create tasks automatically.

| Script | What it does |
|--------|-------------|
| `examples/git-observer.sh` | Reports CI status, stale PRs, new issues via `gh` CLI |
| `examples/github-action-observer.yml` | GitHub Action — posts observations on every push |
| `examples/tmux-session.sh` | One-command tmux setup: worker + observer + dashboard |

```bash
# Run the git observer
ENGINE=$ENGINE TOKEN=$TOKEN REPO=myorg/myrepo ./examples/git-observer.sh
```

### AGENTS.md and CLAUDE.md

The repo includes `AGENTS.md` (for Codex) and can coexist with `CLAUDE.md` (for Claude Code). Both files describe the project to the respective AI tool. If you're adding nudge-engine to an existing project, your agents will already read your project's instruction file and understand the codebase.

---

## tmux Guide

tmux is the easiest way to run workers in the background.

### One-command setup

```bash
ENGINE=$ENGINE TOKEN=$TOKEN REPO=myorg/myrepo ./examples/tmux-session.sh
tmux attach -t nudge
```

Creates a session with a Claude Code worker, stats dashboard, and git observer.

### Manual tmux setup

```bash
# Create a named session
tmux new-session -d -s nudge

# Start a worker in the current pane
tmux send-keys 'ENGINE=https://my-engine.workers.dev TOKEN=abc123 ./examples/claude-code-worker.sh' Enter

# Split and add a second worker
tmux split-window -h
tmux send-keys 'ENGINE=https://my-engine.workers.dev TOKEN=def456 ./examples/bash-worker.sh' Enter

# Split and add a stats dashboard
tmux split-window -v
tmux send-keys 'watch -n 30 "curl -s https://my-engine.workers.dev/stats | jq ."' Enter

# Attach
tmux attach -t nudge
```

### Essential tmux commands

```bash
# Sessions
tmux new-session -d -s nudge       # Create session in background
tmux attach -t nudge               # Attach to session
tmux detach                        # Detach (Ctrl+B, then D)
tmux kill-session -t nudge         # Kill session
tmux ls                            # List sessions

# Panes (inside tmux: Ctrl+B first, then the key)
Ctrl+B %                           # Split vertically (left/right)
Ctrl+B "                           # Split horizontally (top/bottom)
Ctrl+B arrow                       # Move between panes
Ctrl+B z                           # Zoom current pane (toggle fullscreen)
Ctrl+B x                           # Kill current pane

# Send commands to a pane from outside tmux
tmux send-keys -t nudge:0.0 'echo hello' Enter
tmux send-keys -t nudge:0.1 C-c    # Ctrl+C to stop a process
```

### Run workers across machines

```bash
# On your laptop
tmux new-session -d -s nudge

# Local Claude Code worker
tmux send-keys 'ENGINE=$ENGINE TOKEN=$TOKEN1 ./examples/claude-code-worker.sh' Enter

# SSH to GPU box, run Ollama worker there
tmux split-window -h
tmux send-keys 'ssh gpu-box "ENGINE=$ENGINE TOKEN=$TOKEN2 ~/nudge-engine/examples/bash-worker.sh"' Enter

tmux attach -t nudge
```

### Keep workers running after logout

```bash
# Option 1: tmux (recommended — you can reattach later)
tmux new-session -d -s nudge
tmux send-keys 'ENGINE=$ENGINE TOKEN=$TOKEN ./examples/claude-code-worker.sh' Enter
# Logout. The worker keeps running. Reattach with: tmux attach -t nudge

# Option 2: nohup (simpler, no reattach)
nohup env ENGINE=$ENGINE TOKEN=$TOKEN ./examples/claude-code-worker.sh > worker.log 2>&1 &

# Option 3: systemd (Linux, survives reboot)
# Create a unit file: see https://wiki.archlinux.org/title/Systemd#Writing_unit_files
```

### Monitor everything at once

```bash
# Create a 4-pane monitoring layout
tmux new-session -d -s monitor
tmux send-keys 'watch -n 30 "curl -s $ENGINE/stats | jq ."' Enter    # Stats
tmux split-window -h
tmux send-keys 'watch -n 10 "curl -s $ENGINE/workers | jq .workers[] | jq {name,status,current_task_id}"' Enter  # Workers
tmux split-window -v
tmux send-keys 'watch -n 15 "curl -s $ENGINE/tasks?status=assigned | jq .tasks"' Enter  # Active tasks
tmux select-pane -t 0
tmux split-window -v
tmux send-keys 'watch -n 20 "curl -s \"$ENGINE/log?limit=5\" | jq .events"' Enter  # Recent events
tmux select-layout tiled
tmux attach -t monitor
```

---

## GitHub Actions Integration

Automatically feed CI results into your nudge engine. Copy `examples/github-action-observer.yml` to `.github/workflows/` in your repo and add these secrets:

- `NUDGE_ENGINE_URL` — your engine URL
- `NUDGE_WORKER_TOKEN` — a registered worker token

Every push reports CI status and test failures. The cron picks them up and creates `fix-ci` or `fix-tests` tasks for your workers.

---

## Built-in Rules

The cron evaluates these rules every cycle. No LLM — just `if/then`:

| Rule | Trigger | Task Created | Priority |
|------|---------|-------------|----------|
| Failing CI | `git` observation with `failingCI: true` | `fix-ci` | 9 |
| Failing Tests | `tests` observation with failures | `fix-tests` | 8 |
| New Issues | `issues` observation with untriaged items | `triage-issues` | 7 |
| Recent Failures | 3+ task failures in 24h | `investigate-failures` | 7 |
| Stale PRs | `git` observation with `stalePRs` | `review-prs` | 6 |
| Empty Queue | No tasks + idle workers | `check-health` | 3 |

### Add your own rules

Edit `src/rules.ts`. Each rule is a pure function:

```typescript
const myRule: Rule = (ctx) => {
  const deploy = ctx.external.find((o) => o.type === 'deploy');
  if (!deploy?.data.pending) return null;

  return {
    action: 'run-deploy',
    description: 'A deployment is pending. Run the deploy pipeline.',
    priority: 8,
    source: 'cron',
  };
};

// Add to the RULES array at the bottom of the file
const RULES: Rule[] = [
  failingCI,
  failingTests,
  myRule,       // your rule
  newIssues,
  recentFailures,
  stalePRs,
  emptyQueue,
];
```

Then redeploy: `npx wrangler deploy`

---

## API Reference

### Core Protocol (worker token auth)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/register` | Register a new worker. Body: `{name, type, capabilities?}` |
| `POST` | `/poll` | Heartbeat + get next task. Returns `{task}` or `{task: null}` |
| `POST` | `/report` | Report task result. Body: `{taskId, success, result?, error?}` |
| `POST` | `/observe` | Submit an observation. Body: `{type, data}` |

### Admin (`ADMIN_API_KEY` auth)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Create a task. Body: `{action, description, priority?, context?}` |
| `GET` | `/tasks` | List tasks. Query: `status`, `limit`, `offset` |
| `GET` | `/tasks/:id` | Get task detail |
| `DELETE` | `/tasks/:id` | Delete a task |

### Public (no auth)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/workers` | List workers |
| `GET` | `/stats` | Network stats |
| `GET` | `/log` | Work log. Query: `limit`, `taskId`, `workerId` |

### Worker Types

| Type | For |
|------|-----|
| `claude-code` | Anthropic Claude Code CLI |
| `codex` | OpenAI Codex CLI |
| `ollama` | Local Ollama instance |
| `custom` | Anything else |

---

## Architecture

```
┌──────────────┐
│   Cron Job   │  Cloudflare Workers, every 15 min
│              │
│  1. Observe  │  Read task queue + worker fleet + observations
│  2. Decide   │  Run heuristic rules, create 0-3 tasks
│  3. Nudge    │  Reset stuck tasks, mark dead workers
│  4. Cleanup  │  Prune old data
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  D1 (SQLite) │  4 tables: workers, tasks, observations, work_log
└──────┬───────┘
       │
       ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Claude Code │  │    Codex     │  │    Ollama    │  Workers poll via HTTP
│   (worker)   │  │   (worker)   │  │   (worker)   │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## Cost

**$0/month** on Cloudflare's free tier:

- Workers: 100,000 requests/day
- D1: 5M rows read, 100K rows written/day
- Cron: unlimited triggers

That's enough for 50+ AI agents polling constantly.

---

## Development

```bash
npm install          # Install dependencies
npm run dev          # Local dev server at :8787
npm test             # Run tests (12 tests)
npm run typecheck    # TypeScript check
```

### Project Structure

```
nudge-engine/
  src/
    index.ts          # Entry point: Hono app + cron handler
    api.ts            # HTTP endpoints
    cron.ts           # The nudge loop (observe, decide, nudge, cleanup)
    observers.ts      # State observers (read-only queries)
    rules.ts          # Heuristic task creation rules (pure functions)
    utils.ts          # Shared utilities
  test/
    api.test.ts       # API tests (12 tests, full lifecycle coverage)
    setup.ts          # Test DB schema setup
    env.d.ts          # Test type declarations
  examples/
    universal-worker.sh     # Auto-detects Claude/Codex/Ollama
    claude-code-worker.sh   # Claude Code integration
    codex-worker.sh         # OpenAI Codex integration
    ollama-worker.sh        # Local Ollama integration
    bash-worker.sh          # Minimal starting point
    git-observer.sh         # GitHub observer (uses gh CLI)
    tmux-session.sh         # One-command tmux setup
    github-action-observer.yml  # CI/CD integration
  schema.sql          # D1 database schema (4 tables)
  wrangler.toml       # Cloudflare Workers config
  setup.sh            # One-line deploy script
  AGENTS.md           # For Codex users
```

## License

MIT
