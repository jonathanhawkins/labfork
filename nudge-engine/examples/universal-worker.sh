#!/bin/bash
#
# Universal Nudge Engine worker.
#
# Auto-detects which AI CLI is available and uses it:
#   1. Claude Code (claude)
#   2. Codex (codex)
#   3. Ollama (ollama)
#   4. Falls back to echo (dry-run mode)
#
# Usage:
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 ./universal-worker.sh
#
# Override auto-detection:
#   ENGINE=... TOKEN=... AGENT=codex ./universal-worker.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Auto-detect agent
detect_agent() {
  if [ -n "${AGENT:-}" ]; then
    echo "$AGENT"
  elif command -v claude &> /dev/null; then
    echo "claude"
  elif command -v codex &> /dev/null; then
    echo "codex"
  elif command -v ollama &> /dev/null && curl -sf "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "ollama"
  else
    echo "dry-run"
  fi
}

AGENT=$(detect_agent)
log "Agent: $AGENT"
log "Engine: $ENGINE"
log "Repo: $REPO_DIR"

# Run a task with the detected agent
run_task() {
  local action="$1"
  local description="$2"
  local context="$3"

  local prompt="Task: $action

$description

Context: $context

When finished, output a single line starting with SUMMARY: describing what you did."

  case "$AGENT" in
    claude)
      cd "$REPO_DIR"
      claude -p "$prompt" --output-format text 2>&1 || true
      ;;

    codex)
      cd "$REPO_DIR"
      codex exec --full-auto --sandbox workspace-write "$prompt" 2>/dev/null || true
      ;;

    ollama)
      local model="${MODEL:-qwen3:8b}"
      local timeout="${OLLAMA_TIMEOUT:-180}"
      curl -sf --max-time "$timeout" -X POST "http://localhost:11434/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"$model\",
          \"prompt\": $(echo "$prompt" | jq -Rs .),
          \"stream\": false
        }" | jq -r '.response // empty' || true
      ;;

    dry-run)
      echo "DRY RUN: Would execute task '$action'"
      echo "$description"
      echo "SUMMARY: Dry run of $action"
      ;;
  esac
}

log "Starting poll loop (every ${POLL_INTERVAL}s)"

# Disable set -e for the main loop — a daemon should never die from a stray exit code.
set +e

while true; do
  RESPONSE=$(curl -sf -X POST "$ENGINE/poll" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" 2>/dev/null || echo '{"task":null}')

  TASK_ID=$(echo "$RESPONSE" | jq -r '.task.id // empty')

  if [ -z "$TASK_ID" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  ACTION=$(echo "$RESPONSE" | jq -r '.task.action')
  DESC=$(echo "$RESPONSE" | jq -r '.task.description')
  CONTEXT=$(echo "$RESPONSE" | jq -c '.task.context // {}')

  log "Task: $ACTION ($TASK_ID) via $AGENT"

  OUTPUT=$(run_task "$ACTION" "$DESC" "$CONTEXT")

  # Extract summary
  SUMMARY=$(echo "$OUTPUT" | grep "^SUMMARY:" | tail -1 | sed 's/^SUMMARY: *//')
  if [ -z "$SUMMARY" ]; then
    SUMMARY="Completed $ACTION via $AGENT"
  fi

  # Determine success
  if echo "$OUTPUT" | tail -5 | grep -qi "error\|failed\|unable\|exception"; then
    ERROR_MSG=$(echo "$OUTPUT" | tail -3 | tr '\n' ' ')
    log "Failed: $TASK_ID"

    curl -sf -X POST "$ENGINE/report" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{
        \"taskId\": \"$TASK_ID\",
        \"success\": false,
        \"error\": $(echo "$ERROR_MSG" | jq -Rs .)
      }" > /dev/null 2>&1 || log "Warning: failed to report $TASK_ID"
  else
    log "Done: $TASK_ID — $SUMMARY"

    curl -sf -X POST "$ENGINE/report" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{
        \"taskId\": \"$TASK_ID\",
        \"success\": true,
        \"result\": {\"summary\": $(echo "$SUMMARY" | jq -Rs .)}
      }" > /dev/null 2>&1 || log "Warning: failed to report $TASK_ID"
  fi

  sleep 5
done
