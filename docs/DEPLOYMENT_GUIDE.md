# AI Research Lab - Deployment Guide

## Quick Start (5 Minutes)

Choose your deployment method based on your hardware:

### Option 1: Cloud Deploy (Recommended for Beginners)

```bash
# One-command cloud deployment
curl -fsSL https://airesearch.ai/deploy.sh | bash

# Follow prompts:
# 1. Choose domain (voice-clone, quant-trading, game-ai, etc.)
# 2. Select cloud provider (RunPod, AWS, GCP)
# 3. Pick GPU tier (RTX 4090, A100, or CPU-only free tier)
# 4. Set budget limits

# Result: Live lab running in ~5 minutes
# URL: https://your-username.airesearch.ai
```

**Cost**: $0.50-2/hour depending on GPU tier (free CPU tier available)

---

### Option 2: Local Machine (Free, Full Control)

```bash
# Clone and run locally
git clone https://github.com/airesearch-labs/platform
cd platform

# Interactive setup
./setup.sh

# Prompts:
# 1. Domain to start with?
# 2. Use GPU or CPU?
# 3. Public or private lab?
# 4. Agent budget? (optional)

# Starts:
# - Frontend on http://localhost:3000
# - Backend API on http://localhost:8003
# - Orchestrator in background
# - Auto-opens browser
```

**Requirements**:
- **CPU-only**: 16GB RAM, 50GB disk
- **With GPU**: NVIDIA GPU (8GB+ VRAM), CUDA 12+

---

### Option 3: Remote GPU (Best for Power Users)

Use your existing machine with GPU via SSH:

```bash
# Setup with remote GPU
./setup.sh --remote

# Prompts:
# 1. SSH details (host, user, key)
# 2. Test connection
# 3. Install dependencies on remote
# 4. Choose domain
# 5. Configure agent budget

# Frontend runs locally, agents run on remote GPU
```

**Perfect for**:
- Windows machine with RTX 4090
- Linux server in closet
- University cluster access

---

## Detailed Setup Guides

### Cloud Deployment (RunPod Example)

#### Step 1: Get RunPod API Key

1. Sign up at runpod.io
2. Go to Settings → API Keys
3. Create new API key
4. Copy key

#### Step 2: Deploy

```bash
export RUNPOD_API_KEY="your-key-here"

./deploy.sh \
  --cloud=runpod \
  --gpu=rtx4090 \
  --domain=quant-trading \
  --budget-daily=10
```

#### Step 3: Wait for Provisioning (~3-5 min)

```
✓ Creating RunPod instance...
✓ Installing CUDA drivers...
✓ Installing Ollama...
✓ Deploying lab stack...
✓ Setting up Tailscale...
✓ Configuring domain...

Your lab is ready!
URL: https://your-username.airesearch.ai
SSH: ssh runpod@100.123.45.67
```

#### Step 4: Access Your Lab

Visit the URL to see your 3D lab interface with agents ready to work!

---

### Local Docker Compose Deployment

#### Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# For GPU support, install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
```

#### Deploy

```bash
# Clone repository
git clone https://github.com/airesearch-labs/platform
cd platform

# Configure your lab
cp .env.example .env
nano .env  # Edit settings

# Start with GPU
docker-compose -f docker-compose.gpu.yml up -d

# OR start CPU-only (free tier)
docker-compose up -d
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_BACKEND_URL=http://backend:8003
      - NEXT_PUBLIC_LAB_NAME=${LAB_NAME}
      - NEXT_PUBLIC_DOMAIN=${DOMAIN}
    depends_on:
      - backend

  # Backend API
  backend:
    build: ./backend
    ports:
      - "8003:8003"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - OLLAMA_HOST=${OLLAMA_HOST:-http://ollama:11434}

  # Ollama (local LLM)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Orchestrator
  orchestrator:
    build: ./orchestrator
    volumes:
      - ./data:/app/data
      - ./.claude:/root/.claude
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OLLAMA_HOST=http://ollama:11434
      - DOMAIN=${DOMAIN}
    depends_on:
      - ollama
      - backend

volumes:
  ollama_data:
```

#### .env Configuration

```bash
# Lab settings
LAB_NAME="My AI Research Lab"
DOMAIN="quant-trading"              # or voice-clone, game-ai, etc.
VISIBILITY="public"                 # or private

# Hardware
USE_GPU=true                        # false for CPU-only
OLLAMA_HOST="http://ollama:11434"

# Agents
ANTHROPIC_API_KEY=""                # Optional - for paid Codex agent
AGENT_BUDGET_DAILY=5.00             # Optional spending limit

# Public portal (optional)
ENABLE_PUBLIC_PORTAL=true
CUSTOM_DOMAIN=""                    # Leave empty to use *.airesearch.ai
```

---

### Remote GPU Setup (Tailscale + SSH)

#### Scenario

You have:
- Mac/Linux laptop (for development)
- Windows PC with RTX 4090 (for training)
- Both connected via Tailscale VPN

#### Step 1: Setup Remote Machine

On Windows machine:

```powershell
# Install Ollama
winget install Ollama.Ollama

# Install Docker Desktop
winget install Docker.DockerDesktop

# Install Tailscale
winget install tailscale.tailscale

# Enable SSH (Windows 11)
Add-WindowsCapability -Online -Name OpenSSH.Server
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
```

#### Step 2: Deploy from Local Machine

On your laptop:

```bash
./deploy.sh --remote

# Interactive prompts:
Remote GPU Setup
================
SSH Host: $REMOTE_GPU_HOST
SSH User: doc
SSH Key: ~/.ssh/id_rsa
Test connection? [Y/n]: y

✓ SSH connection successful
✓ Detected: Windows 11, RTX 4090 (24GB)
✓ CUDA 12.3, Ollama installed

Domain to deploy: [voice-clone]
> quant-trading

Agent budget (daily): [$5.00]
> 10.00

Deploying...
✓ Synced project files
✓ Started Ollama on remote
✓ Started orchestrator
✓ Configured Tailscale funnel

Your lab is ready!
Frontend: http://localhost:3000 (runs locally)
Backend: https://your-pc.tail-abc123.ts.net (runs on GPU)
Agents: Running on RTX 4090
```

Frontend runs on your laptop, heavy compute happens on remote GPU!

---

## Configuration: domain.yaml

When you run setup, it creates a `domain.yaml` for your lab:

```yaml
# Auto-generated from setup wizard
name: "My Quant Trading Lab"
slug: "quant-trading"
description: "Genetic algorithms for trading strategies"
version: "1.0.0"

# Your branding
branding:
  primaryColor: "#22c55e"
  accentColor: "#ef4444"
  backgroundStyle: "grid"

# 3D scene (preset for quant-trading domain)
scene:
  groundColor: "#1a1a2e"
  props:
    - type: "trading-terminal"
      position: [-6, 0, -5]
    - type: "chart-wall"
      position: [0, 2, -6]
    # ... more props

# Research sources
research:
  arxivCategories: ["q-fin.PM", "cs.NE"]
  keywords:
    - "genetic algorithms trading"
    - "portfolio optimization"

# Agent settings
agents:
  budget:
    daily: 10.00
    monthly: 200.00
  models:
    free: "qwen3-coder-32k"
    paid: "claude-sonnet-4"

# Public portal
portal:
  enabled: true
  visibility: "public"
  allowForks: true
  allowSuggestions: true
```

You can edit this file anytime to customize your lab!

---

## Adding Your First Research

### From arXiv Paper

```bash
# CLI
./lab add-paper 2401.12345

# Or via web UI
http://localhost:3000/lab/add-paper
```

### From GitHub Repo

```bash
./lab add-github https://github.com/openai/baselines
```

### Custom Research Goal

```bash
./lab add-goal "Build a trading bot that uses momentum indicators"
```

The orchestrator will:
1. Analyze the paper/repo/goal
2. Create research tasks
3. Spawn agents to implement
4. Show progress in 3D visualization

---

## Accessing Your Lab

### Local Access

```
Frontend:  http://localhost:3000
Backend:   http://localhost:8003
Ollama:    http://localhost:11434
```

### Public Portal (if enabled)

```
Your Portal: https://your-username.airesearch.ai/labs/quant-trading
Live View:   https://your-username.airesearch.ai/labs/quant-trading/watch
```

Others can:
- Watch your agents work in real-time
- See your research results
- Star/fork your lab
- Suggest papers to implement

---

## Monitoring & Management

### Check Agent Status

```bash
# CLI
./lab status

# Output:
Lab: My Quant Trading Lab
Domain: quant-trading
Status: Running

Agents (3 active):
  - rm:codex-1    [RUNNING]  Implementing momentum strategy
  - rm:ollama-2   [RUNNING]  Backtesting on SPY data
  - rm:manager    [RUNNING]  Orchestrating tasks

Tasks: 12 pending, 3 in progress, 47 completed
Budget: $2.34 / $10.00 daily
GPU: RTX 4090 (47% util, 8.2GB / 24GB)
```

### View Logs

```bash
# Agent logs
./lab logs rm:codex-1

# Orchestrator logs
./lab logs orchestrator

# All logs
./lab logs --all
```

### Control Agents

```bash
# Pause all agents
./lab pause

# Resume
./lab resume

# Stop specific agent
./lab kill rm:codex-1

# Restart orchestrator
./lab restart orchestrator
```

---

## Cost Management

### Set Budget Limits

```yaml
# In domain.yaml
agents:
  budget:
    daily: 10.00      # Max $10/day
    weekly: 50.00     # Max $50/week
    monthly: 150.00   # Max $150/month

  # What happens when limit is hit
  onLimitReached: "pause"  # or "stop" or "switch-to-free"
```

### Track Spending

```bash
./lab costs

# Output:
Cost Report (last 7 days)
=========================
Codex (paid):     $23.45
Ollama (free):    $0.00
GPU compute:      $12.34
Total:            $35.79

Daily average:    $5.11
Projected month:  $153.30
```

### Cost-Saving Tips

1. **Use Ollama for most tasks** (free local LLM)
2. **Reserve Codex for complex analysis** (paid but powerful)
3. **CPU-only mode** works great for many domains (completely free!)
4. **Spot instances** on cloud providers (70% cheaper)

---

## Sharing Your Lab

### Make Lab Public

```bash
# Enable public portal
./lab publish

# Output:
✓ Lab published!

Your public portal: https://airesearch.ai/labs/johndoe/quant-trading

Share this link to let others:
- Watch your agents work live
- See your research results
- Fork your lab setup
- Suggest papers to implement
```

### Share Specific Results

```bash
# Share a trained model
./lab share model ./models/momentum-v2.pt \
  --title "Momentum Strategy (70% win rate)" \
  --description "Trained on SPY 2020-2024"

# Share a demo
./lab share demo --type backtest \
  --data ./results/backtest_2024.json

# Share a finding
./lab share result \
  --title "Achieved 2.3 Sharpe Ratio" \
  --metrics sharpe:2.3 max_dd:12.5 \
  --chart ./charts/equity_curve.png
```

### Embed Live View

```html
<!-- Embed your live lab on your website -->
<iframe
  src="https://airesearch.ai/labs/johndoe/quant-trading/embed"
  width="800"
  height="600"
  frameborder="0">
</iframe>
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Restart Ollama with GPU
docker-compose restart ollama
```

### Agents Not Starting

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check orchestrator logs
./lab logs orchestrator

# Restart orchestrator
./lab restart orchestrator
```

### Can't Access Public Portal

```bash
# Check Tailscale funnel
tailscale funnel status

# Restart funnel
tailscale funnel --bg 3000

# Check backend URL
echo $BACKEND_URL
```

### High Costs

```bash
# Check which agents are expensive
./lab costs --breakdown

# Switch to free tier
./lab config --free-only

# Pause paid agents
./lab pause --paid-only
```

---

## Updating

```bash
# Update to latest version
./lab update

# Or with Docker
docker-compose pull
docker-compose up -d
```

---

## Next Steps

1. **Add your first paper**: `./lab add-paper <arxiv-id>`
2. **Watch agents work**: Open http://localhost:3000
3. **Share your lab**: `./lab publish`
4. **Explore other labs**: https://airesearch.ai/explore
5. **Fork interesting labs**: Click "Fork" on any public lab
6. **Join the community**: https://discord.gg/airesearch

Welcome to AI Research Labs! 🤖✨
