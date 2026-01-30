#!/bin/bash
# Spawn Codex agent (uses OpenAI API credits)
# Usage: spawn-codex-agent.sh <task_file> <log_file>

set -e

TASK_FILE="$1"
LOG_FILE="$2"

# Activate conda environment (required for Python tools)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate voice 2>/dev/null || echo "Warning: could not activate conda voice env"
fi

export CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline"

# Find codex binary
CODEX_BIN="$HOME/bin/codex"
if [ ! -x "$CODEX_BIN" ]; then
    CODEX_BIN=$(which codex 2>/dev/null || true)
fi

if [ ! -x "$CODEX_BIN" ]; then
    echo "Error: codex not found" >&2
    exit 1
fi

# Verify task file exists
if [ ! -f "$TASK_FILE" ]; then
    echo "Error: task file not found: $TASK_FILE" >&2
    exit 1
fi

cd "$HOME/dev/voice-clone-pipeline"

echo "Starting Codex agent at $(date)" >> "$LOG_FILE"
echo "Task file: $TASK_FILE" >> "$LOG_FILE"

# Run codex exec with stdin from task file
# Uses default model (gpt-5.2-codex with ChatGPT account), 30 min timeout
"$CODEX_BIN" exec \
  -c timeout=1800000 \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  - < "$TASK_FILE" 2>&1 | tee -a "$LOG_FILE"

echo "Codex finished at $(date)" >> "$LOG_FILE"
