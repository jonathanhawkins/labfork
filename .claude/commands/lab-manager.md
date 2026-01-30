---
description: Start FREE lab manager with context (local Ollama)
allowed-tools: Bash
---

# Starting Lab Manager

Launching FREE lab manager with initial context...

```bash
# Kill existing session if any
tmux kill-session -t lab-manager 2>/dev/null || true

# Start fresh lab-manager session
tmux new-session -d -s lab-manager -x 140 -y 40 "./scripts/claude-free"

# Wait for Claude Code to initialize
sleep 12

# Send the lab manager initialization prompt
tmux send-keys -t lab-manager "You are the LAB MANAGER for the labfork project. Your responsibilities:

1. MONITOR: Check task status, pending work, git changes
2. TRIAGE: Prioritize research implementations
3. COORDINATE: Track what needs to be done next
4. DOCUMENT: Keep notes on progress

IMPORTANT REMINDERS (local model limitations):
- Re-read CLAUDE.md before major decisions
- Check task list frequently with TaskList tool
- Keep responses focused - you have 32k context limit
- If unsure, read the relevant files first

Start by reading CLAUDE.md and listing pending tasks." Enter

# Wait a moment then send extra Enter to submit
sleep 2
tmux send-keys -t lab-manager Enter

echo ""
echo "✅ Lab manager started in tmux session 'lab-manager'"
echo ""
echo "To connect: tmux attach -t lab-manager"
echo "To detach:  Ctrl+B then D"
echo ""
```

**Cost: FREE** - Uses local qwen3-coder-32k via Ollama
