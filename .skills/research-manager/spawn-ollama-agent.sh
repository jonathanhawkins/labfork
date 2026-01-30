#!/bin/bash
# Spawn Claude Code agent with Ollama backend
# Usage: spawn-ollama-agent.sh <task_file> <log_file>

set -e

# Save arguments first
TASK_FILE="$1"
LOG_FILE="$2"

# Activate conda environment (required for Python tools on 4090)
# Use eval to properly initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate voice 2>/dev/null || echo "Warning: could not activate conda voice env"
fi

# Set Ollama environment variables for Anthropic API compatibility (Ollama 0.14+)
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline"

# Use codex (which we have in ~/bin) with qwen3-coder:30b model
CODEX_BIN="$HOME/bin/codex"

if [ ! -x "$CODEX_BIN" ]; then
    echo "Error: codex not found at $CODEX_BIN" >&2
    exit 1
fi

# Verify task file exists
if [ ! -f "$TASK_FILE" ]; then
    echo "Error: task file not found: $TASK_FILE" >&2
    exit 1
fi

# Run codex exec with the task prompt (reads from stdin with -)
# Use --dangerously-bypass-approvals-and-sandbox for autonomous operation
cd "$HOME/dev/voice-clone-pipeline"

echo "Starting codex exec at $(date)" >> "$LOG_FILE"
echo "Task file: $TASK_FILE" >> "$LOG_FILE"

"$CODEX_BIN" exec \
  --oss \
  --local-provider ollama \
  --model qwen3-coder:30b \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  - < "$TASK_FILE" 2>&1 | tee -a "$LOG_FILE"

echo "Codex finished at $(date)" >> "$LOG_FILE"
