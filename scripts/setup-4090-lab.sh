#!/bin/bash
# Setup script for RTX 4090 Lab Manager
# Run this ON the 4090 machine (via SSH)

set -e

echo "=================================="
echo "  RTX 4090 Lab Manager Setup"
echo "=================================="
echo ""

# Create bin directory
mkdir -p ~/bin

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
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

# Create lab-manager script
cat > ~/bin/lab-manager << 'SCRIPT'
#!/bin/bash
# Lab Manager - FREE Claude Code with Ollama on RTX 4090

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5
fi

# Check VRAM - if training running, wait
VRAM_USED=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$VRAM_USED" ] && [ "$VRAM_USED" -gt 12000 ]; then
    echo "Training detected! VRAM: ${VRAM_USED}MB"
    echo "Waiting for training to complete..."
    while [ "$VRAM_USED" -gt 12000 ]; do
        sleep 30
        VRAM_USED=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "  VRAM: ${VRAM_USED}MB"
    done
fi

# Pre-warm model
echo "Pre-warming model..."
echo "test" | timeout 60 ollama run qwen3-coder-32k > /dev/null 2>&1 || true

echo ""
echo "============================================"
echo "  LAB MANAGER - RTX 4090"
echo "  Model: qwen3-coder-32k (FREE)"
echo "  Speed: 40-50 tok/sec"
echo "============================================"
echo ""

# Run Claude Code with Ollama
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY=""

cd ~/dev/voice-clone-pipeline
claude --model qwen3-coder-32k "$@"
SCRIPT
chmod +x ~/bin/lab-manager

# Create GPU status script
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
SCRIPT
chmod +x ~/bin/gpu-status

# Create session starter
cat > ~/bin/start-lab-session << 'SCRIPT'
#!/bin/bash
tmux kill-session -t lab-manager 2>/dev/null || true

tmux new-session -d -s lab-manager -n main

# Main pane: Lab manager
tmux send-keys -t lab-manager 'source ~/miniconda3/bin/activate && conda activate voice && ~/bin/lab-manager' C-m

# Side pane: GPU monitor
tmux split-window -h -t lab-manager
tmux send-keys -t lab-manager 'watch -n 5 ~/bin/gpu-status' C-m

# Focus on lab manager
tmux select-pane -t lab-manager:0.0

echo "Lab session started!"
echo "Attach: tmux attach -t lab-manager"
SCRIPT
chmod +x ~/bin/start-lab-session

# Add to PATH if needed
if ! grep -q 'export PATH="$HOME/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "=================================="
echo "  Setup Complete!"
echo "=================================="
echo ""
echo "Commands available:"
echo "  lab-manager       - Start FREE Claude Code with Ollama"
echo "  start-lab-session - Start tmux session with GPU monitor"
echo "  gpu-status        - Check GPU and Ollama status"
echo ""
echo "From Mac, use:"
echo "  ssh \$REMOTE_GPU_USER@\$REMOTE_GPU_HOST '~/bin/start-lab-session'"
echo "  ssh \$REMOTE_GPU_USER@\$REMOTE_GPU_HOST -t 'tmux attach -t lab-manager'"
echo ""
