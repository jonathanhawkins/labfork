#!/bin/bash
# Setup script for RTX 4090 Hybrid Lab System
# Run this ON the 4090 machine (via SSH)
#
# Creates:
# - lab-manager: FREE Ollama-based Claude Code
# - call-codex: Calls paid Codex CLI for complex analysis
# - execute-plan: Runs returned JSON plans
# - hybrid-workflow: Combined workflow demo

set -e

echo "=================================="
echo "  RTX 4090 Hybrid Lab System"
echo "=================================="
echo ""

PROJECT_DIR="$HOME/dev/voice-clone-pipeline"

# Create bin directory
mkdir -p ~/bin

# ============================================================
# 1. OLLAMA SETUP
# ============================================================
echo "[1/5] Setting up Ollama..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
fi

# Download qwen3-coder if not present
if ! ollama list | grep -q "qwen3-coder:30b"; then
    echo "Downloading qwen3-coder:30b (18GB)..."
    ollama pull qwen3-coder:30b
fi

# Create 32k context model
echo "Creating qwen3-coder-32k model..."
cat > /tmp/Modelfile.qwen3-coder-32k << 'EOF'
FROM qwen3-coder:30b
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
EOF
ollama create qwen3-coder-32k -f /tmp/Modelfile.qwen3-coder-32k

# ============================================================
# 2. LAB MANAGER SCRIPT
# ============================================================
echo "[2/5] Creating lab-manager script..."

cat > ~/bin/lab-manager << 'SCRIPT'
#!/bin/bash
# Lab Manager - FREE Claude Code with Ollama on RTX 4090
# Part of the Hybrid Lab System

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
fi

# Check VRAM - if training running, wait
VRAM_USED=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_USED" ] && [ "$VRAM_USED" -gt 12000 ]; then
    echo "Training detected! VRAM: ${VRAM_USED}MB"
    echo "Lab manager will use CPU mode or wait..."
fi

echo ""
echo "============================================"
echo "  LAB MANAGER - HYBRID SYSTEM"
echo "  Model: qwen3-coder-32k (FREE)"
echo "  For complex tasks: ~/bin/call-codex"
echo "============================================"
echo ""
echo "WORKFLOW:"
echo "  1. Read task with TaskList/TaskGet"
echo "  2. For complex analysis: call-codex --files <files> <task>"
echo "  3. Execute returned plan: execute-plan plan.json"
echo "  4. Mark task complete with TaskUpdate"
echo ""

# Run Claude Code with Ollama
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY=""

cd ~/dev/voice-clone-pipeline
claude --model qwen3-coder-32k "$@"
SCRIPT
chmod +x ~/bin/lab-manager

# ============================================================
# 3. CALL-CODEX SCRIPT (copies from project)
# ============================================================
echo "[3/5] Setting up call-codex..."

cp "$PROJECT_DIR/scripts/call-codex" ~/bin/call-codex
chmod +x ~/bin/call-codex

# Also copy execute-plan
cp "$PROJECT_DIR/scripts/execute-plan" ~/bin/execute-plan
chmod +x ~/bin/execute-plan

# ============================================================
# 4. GPU STATUS SCRIPT
# ============================================================
echo "[4/5] Creating gpu-status script..."

cat > ~/bin/gpu-status << 'SCRIPT'
#!/bin/bash
echo "=== RTX 4090 ==="
/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU info unavailable"
echo ""
echo "=== Ollama ==="
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Status: RUNNING"
    ollama ps 2>/dev/null || echo "No models loaded"
else
    echo "Status: STOPPED"
fi
echo ""
echo "=== Codex Plans ==="
PLAN_DIR="$HOME/dev/voice-clone-pipeline/.codex-plans"
if [ -d "$PLAN_DIR" ]; then
    ls -lh "$PLAN_DIR"/*.json 2>/dev/null | tail -5 || echo "No plans yet"
else
    echo "No plans directory"
fi
SCRIPT
chmod +x ~/bin/gpu-status

# ============================================================
# 5. HYBRID SUPERVISOR SCRIPT
# ============================================================
echo "[5/5] Creating lab startup script..."

cat > ~/bin/start-hybrid-lab << 'SCRIPT'
#!/bin/bash
# Start the hybrid lab system (single worker session)

# Kill existing sessions
tmux kill-session -t lab-manager 2>/dev/null || true

# Create lab-manager session
tmux new-session -d -s lab-manager -n main

# Main pane: Lab manager
tmux send-keys -t lab-manager 'source ~/miniconda3/bin/activate && conda activate voice' C-m
tmux send-keys -t lab-manager '~/bin/lab-manager' C-m

# Side pane: GPU monitor
tmux split-window -h -t lab-manager
tmux send-keys -t lab-manager 'watch -n 10 ~/bin/gpu-status' C-m

# Focus on lab manager main pane
tmux select-pane -t lab-manager:0.0

echo ""
echo "============================================"
echo "  HYBRID LAB SYSTEM STARTED"
echo "============================================"
echo ""
echo "Session:"
echo "  lab-manager - FREE Ollama worker"
echo ""
echo "Attach:"
echo "  tmux attach -t lab-manager  # Worker view"
echo ""
echo "Send commands to worker:"
echo "  tmux send-keys -t lab-manager '<command>' C-m"
echo ""
SCRIPT
chmod +x ~/bin/start-hybrid-lab

# Create .codex-plans directory
mkdir -p "$PROJECT_DIR/.codex-plans"

# Add to PATH if needed
if ! grep -q 'export PATH="$HOME/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
fi

# Create API key reminder file
cat > ~/.codex-api-key-reminder << 'EOF'
# Codex CLI Setup
#
# Ensure Codex CLI is installed and authenticated.
# If codex isn't on PATH, set:
#
#   export CODEX_PATH="/home/doc/.nvm/versions/node/<version>/bin/codex"
#
# Then run:
#   codex --version
EOF

echo ""
echo "=================================="
echo "  Setup Complete!"
echo "=================================="
echo ""
echo "HYBRID LAB COMMANDS:"
echo ""
echo "  start-hybrid-lab  - Start lab-manager session"
echo "  lab-manager       - Start FREE Claude Code (Ollama)"
echo "  call-codex        - Call paid Codex CLI for analysis"
echo "  execute-plan      - Execute returned JSON plans"
echo "  gpu-status        - Check system status"
echo ""
echo "WORKFLOW EXAMPLE:"
echo ""
echo "  1. Start hybrid lab:  ~/bin/start-hybrid-lab"
echo "  2. Attach to worker:  tmux attach -t lab-manager"
echo "  3. In lab-manager:"
echo "     - Read task: TaskGet #36"
echo "     - Call Codex: ~/bin/call-codex --files 'training/ddgan_prosody.py' 'Create training script'"
echo "     - Execute: ~/bin/execute-plan ~/.codex-plans/plan_*.json"
echo "     - Complete: TaskUpdate #36 status=completed"
echo ""
echo "IMPORTANT: Ensure Codex CLI is installed/authenticated"
echo "  See: ~/.codex-api-key-reminder"
echo ""
