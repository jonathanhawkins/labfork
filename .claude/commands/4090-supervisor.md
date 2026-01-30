---
description: Start/manage 4090 FREE supervisor system
allowed-tools: Bash
---

# 4090 Supervisor System

Manages FREE AI workers on RTX 4090 using local Ollama (qwen3-coder-32k).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              RTX 4090 ($REMOTE_GPU_HOST)               │
│                                                     │
│  ┌─────────────┐       tmux        ┌─────────────┐ │
│  │ SUPERVISOR  │ ───send-keys────▶ │ LAB-MANAGER │ │
│  │ (qwen3-32k) │ ◀──capture-pane── │ (qwen3-32k) │ │
│  └─────────────┘                   └─────────────┘ │
│         │                                          │
│         └── supervisor-watchdog (cron every 5min)  │
│                                                    │
│  Cost: $0.00 (all local Ollama)                   │
└─────────────────────────────────────────────────────┘
```

## Quick Commands

```bash
# Check status
ssh doc@$REMOTE_GPU_HOST "tmux list-sessions"

# Start supervisor system
ssh doc@$REMOTE_GPU_HOST "~/bin/start-supervisor"

# Attach to supervisor
ssh doc@$REMOTE_GPU_HOST -t "tmux attach -t supervisor"

# Attach to worker
ssh doc@$REMOTE_GPU_HOST -t "tmux attach -t lab-manager"

# Check watchdog log
ssh doc@$REMOTE_GPU_HOST "tail -20 ~/supervisor-watchdog.log"

# GPU status
ssh doc@$REMOTE_GPU_HOST "~/bin/gpu-status"
```

## Scripts on 4090 (~/bin/)

| Script | Purpose |
|--------|---------|
| `start-supervisor` | Starts both supervisor + lab-manager sessions |
| `supervisor` | Claude Code with Ollama for task management |
| `lab-manager` | Claude Code with Ollama for task execution |
| `supervisor-watchdog` | Cron job that monitors/restarts if needed |
| `gpu-status` | Shows GPU and Ollama status |

## Pending Tasks (for supervisor)

- #36: DDGAN-accelerated prosody diffusion
- #37: Word-level latent prosody vectors
- #38: Intonation template clustering
- #39: TTScore-pro FACodec evaluation
- #42: WeSCon word-level emotion control
- #43: DisCo-Speech disentanglement codec
- #73: ParaStyleTTS efficient paralinguistic

## How Supervisor Works

1. Supervisor reads pending tasks list
2. Assigns task to lab-manager via `tmux send-keys`
3. Monitors progress via `tmux capture-pane`
4. Provides guidance if worker gets stuck
5. Marks task complete and moves to next

## Watchdog (cron)

Runs every 5 minutes:
- Checks if supervisor/lab-manager sessions exist
- Restarts if down
- Checks if Ollama is running
- Auto-submits if stuck at prompt
- Logs to `~/supervisor-watchdog.log`
