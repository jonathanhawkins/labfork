---
name: lab-manager
description: Start the FREE lab manager using local Ollama (qwen3-coder-32k)
metadata:
  tags: ollama, lab-manager, free, local-ai, management
---

# Lab Manager Skill

**Start a FREE lab manager session using local Ollama.**

This skill launches Claude Code with a local qwen3-coder-32k model for managing the voice-clone-pipeline project without API costs.

## Quick Start

Run `/lab-manager` in Claude Code or:

```bash
# Start lab manager in tmux
tmux new-session -s lab-manager "./scripts/claude-free"

# Or attach to existing session
tmux attach -t claude-free
```

## What the Lab Manager Can Do

- ✅ Read and analyze codebase
- ✅ Review training configs
- ✅ Check task status
- ✅ Explore research implementations
- ✅ Generate documentation
- ✅ Run bash commands
- ✅ Edit files

## Cost

**$0** - Runs entirely on local GPU via Ollama

## Prerequisites

```bash
# Ensure model exists with 32k context
ollama ps  # Should show qwen3-coder-32k with CONTEXT: 32768

# If not, create it:
cat > /tmp/Modelfile << 'EOF'
FROM qwen3-coder:30b
PARAMETER num_ctx 32768
EOF
ollama create qwen3-coder-32k -f /tmp/Modelfile
```

## Example Tasks

Once running, you can ask the lab manager:

- "Review the pending tasks and prioritize them"
- "Check what training configs are available"
- "Summarize the recent git commits"
- "Find all files related to emotion conditioning"
- "Run the frontend build and check for errors"
