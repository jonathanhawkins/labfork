#!/bin/bash
# Setup script for Hybrid Agent Daemon on RTX 4090
# This script configures the 4090 to run both:
# 1. Workers-coordinated tasks (high priority)
# 2. Independent local research (background)

set -e

echo "=================================="
echo "  Hybrid Agent Daemon Setup"
echo "=================================="
echo ""

# Configuration
PROJECT_ROOT="$HOME/dev/voice-clone-pipeline"
SKILLS_DIR="$PROJECT_ROOT/.skills/research-manager"
STATE_DIR="$SKILLS_DIR/state"
LABS_DIR="$SKILLS_DIR/labs"

# Workers API URL (update this for your deployment)
WORKERS_URL="${WORKERS_API_URL:-https://labfork-workers.your-subdomain.workers.dev}"

# Create required directories
mkdir -p "$STATE_DIR"
mkdir -p "$STATE_DIR/outputs"
mkdir -p "$LABS_DIR/voice-clone/state"
mkdir -p "$LABS_DIR/voice-clone/state/research-results"

# Create default research queue if it doesn't exist
RESEARCH_QUEUE="$LABS_DIR/voice-clone/state/research-queue.json"
if [ ! -f "$RESEARCH_QUEUE" ]; then
    echo "Creating default research queue..."
    cat > "$RESEARCH_QUEUE" << 'EOF'
{
  "objectives": [
    {
      "id": "prosody-conditioning-review",
      "title": "Review prosody conditioning approaches",
      "description": "Research and document different approaches to prosody conditioning in TTS models. Focus on: 1) Reference encoder methods 2) Style tokens 3) Variational approaches 4) Recent transformer-based methods. Summarize strengths/weaknesses of each.",
      "priority": 8,
      "status": "pending",
      "tags": ["research", "prosody", "tts"]
    },
    {
      "id": "csm-1b-finetuning-guide",
      "title": "Document CSM-1B fine-tuning process",
      "description": "Create a comprehensive guide for fine-tuning CSM-1B with emotional prosody. Include: 1) Data requirements 2) Preprocessing pipeline 3) Training hyperparameters 4) Evaluation metrics 5) Common pitfalls.",
      "priority": 7,
      "status": "pending",
      "tags": ["documentation", "csm-1b", "training"]
    },
    {
      "id": "emotion-detection-benchmark",
      "title": "Benchmark emotion detection models",
      "description": "Evaluate available emotion detection models for prosody labeling: 1) wav2vec2-emotion 2) HuBERT-emotion 3) Qwen2-Audio semantic analysis. Compare accuracy, speed, and suitability for our pipeline.",
      "priority": 6,
      "status": "pending",
      "tags": ["benchmark", "emotion", "evaluation"]
    },
    {
      "id": "data-augmentation-strategies",
      "title": "Research data augmentation for voice cloning",
      "description": "Investigate data augmentation strategies that preserve emotional prosody: 1) Speed perturbation effects 2) Pitch shifting considerations 3) Room impulse response 4) Background noise. Document what helps vs hurts prosody learning.",
      "priority": 5,
      "status": "pending",
      "tags": ["research", "augmentation", "data"]
    }
  ],
  "lastUpdated": "2026-02-01T00:00:00Z"
}
EOF
    echo "Created: $RESEARCH_QUEUE"
fi

# Install Python dependencies if needed
echo "Checking Python dependencies..."
pip install aiohttp --quiet 2>/dev/null || true

# Copy the daemon script to multiple locations
echo "Installing hybrid-agent-daemon.py..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_SCRIPT="$SCRIPT_DIR/hybrid-agent-daemon.py"

# Also check workers/scripts if running from project root
if [ ! -f "$DAEMON_SCRIPT" ]; then
    DAEMON_SCRIPT="$PROJECT_ROOT/workers/scripts/hybrid-agent-daemon.py"
fi

if [ -f "$DAEMON_SCRIPT" ]; then
    # Install to ~/bin for easy CLI access
    cp "$DAEMON_SCRIPT" "$HOME/bin/hybrid-agent-daemon"
    chmod +x "$HOME/bin/hybrid-agent-daemon"
    echo "Installed: $HOME/bin/hybrid-agent-daemon"

    # Also install to skills dir for systemd service
    cp "$DAEMON_SCRIPT" "$SKILLS_DIR/hybrid-agent-daemon.py"
    chmod +x "$SKILLS_DIR/hybrid-agent-daemon.py"
    echo "Installed: $SKILLS_DIR/hybrid-agent-daemon.py"
else
    echo "Warning: hybrid-agent-daemon.py not found"
    echo "Checked: $SCRIPT_DIR and $PROJECT_ROOT/workers/scripts/"
    echo "You may need to sync the project first: scripts/4090-lab sync"
fi

# Create systemd user service
SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

cat > "$SYSTEMD_DIR/hybrid-agent.service" << EOF
[Unit]
Description=Hybrid Agent Daemon - Workers + Local Research
Documentation=https://github.com/jonathanhawkins/labfork
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
Environment="WORKERS_API_URL=$WORKERS_URL"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/usr/bin/python3 $SKILLS_DIR/hybrid-agent-daemon.py --poll-interval 30
Restart=always
RestartSec=30
WorkingDirectory=$PROJECT_ROOT

# Resource limits
MemoryMax=4G
CPUQuota=200%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=hybrid-agent

[Install]
WantedBy=default.target
EOF

echo "Created systemd service: $SYSTEMD_DIR/hybrid-agent.service"

# Create convenience scripts
cat > "$HOME/bin/hybrid-agent" << 'SCRIPT'
#!/bin/bash
# Hybrid Agent control script

case "$1" in
    start)
        echo "Starting hybrid agent..."
        systemctl --user start hybrid-agent
        echo "Started. Check status: hybrid-agent status"
        ;;
    stop)
        echo "Stopping hybrid agent..."
        systemctl --user stop hybrid-agent
        echo "Stopped."
        ;;
    restart)
        echo "Restarting hybrid agent..."
        systemctl --user restart hybrid-agent
        echo "Restarted."
        ;;
    status)
        systemctl --user status hybrid-agent
        ;;
    logs)
        journalctl --user -u hybrid-agent -f
        ;;
    run)
        # Run directly (not as service)
        shift
        python3 ~/bin/hybrid-agent-daemon "$@"
        ;;
    *)
        echo "Usage: hybrid-agent {start|stop|restart|status|logs|run}"
        echo ""
        echo "Commands:"
        echo "  start   - Start as systemd service"
        echo "  stop    - Stop the service"
        echo "  restart - Restart the service"
        echo "  status  - Show service status"
        echo "  logs    - Follow service logs"
        echo "  run     - Run directly (foreground)"
        exit 1
        ;;
esac
SCRIPT
chmod +x "$HOME/bin/hybrid-agent"
echo "Created: $HOME/bin/hybrid-agent"

# Reload systemd
systemctl --user daemon-reload 2>/dev/null || true

echo ""
echo "=================================="
echo "  Setup Complete!"
echo "=================================="
echo ""
echo "Commands available:"
echo "  hybrid-agent start    - Start daemon as service"
echo "  hybrid-agent stop     - Stop daemon"
echo "  hybrid-agent status   - Check status"
echo "  hybrid-agent logs     - View logs"
echo "  hybrid-agent run      - Run in foreground (for testing)"
echo ""
echo "The daemon will:"
echo "  1. Poll Workers API every 30s for high-priority tasks"
echo "  2. Execute Workers tasks when available"
echo "  3. Do independent research when Workers queue is empty"
echo "  4. Update agents.json for /watch page visibility"
echo ""
echo "To enable auto-start on boot:"
echo "  systemctl --user enable hybrid-agent"
echo ""
