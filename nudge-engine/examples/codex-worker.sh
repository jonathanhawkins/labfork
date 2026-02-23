#!/bin/bash
#
# Nudge Engine worker that dispatches tasks to OpenAI Codex CLI.
#
# Each task becomes a `codex exec` invocation that runs, completes, and reports back.
# Uses --full-auto for unattended execution.
#
# Prerequisites:
#   - Codex CLI installed: npm install -g @openai/codex
#   - Authenticated: CODEX_API_KEY set or `codex auth`
#
# Usage:
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 ./codex-worker.sh
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 REPO_DIR=/path/to/repo ./codex-worker.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
SANDBOX="${SANDBOX:-workspace-write}"  # read-only | workspace-write | danger-full-access

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Codex worker starting"
log "  Engine:   $ENGINE"
log "  Repo:     $REPO_DIR"
log "  Sandbox:  $SANDBOX"
log "  Polling:  every ${POLL_INTERVAL}s"

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

  log "Task: $ACTION ($TASK_ID)"

  # Build the prompt
  PROMPT="Task: $ACTION

$DESC

Context: $CONTEXT

When finished, output a single line starting with SUMMARY: describing what you did."

  # Run Codex in non-interactive mode
  # --full-auto: no approval prompts
  # --sandbox: control file access level
  cd "$REPO_DIR"
  OUTPUT=$(codex exec \
    --full-auto \
    --sandbox "$SANDBOX" \
    "$PROMPT" 2>/dev/null) || true

  # Extract summary
  SUMMARY=$(echo "$OUTPUT" | grep "^SUMMARY:" | tail -1 | sed 's/^SUMMARY: *//')
  if [ -z "$SUMMARY" ]; then
    SUMMARY="Completed $ACTION"
  fi

  # Determine success/failure
  if echo "$OUTPUT" | tail -5 | grep -qi "error\|failed\|unable\|exception"; then
    log "Failed: $TASK_ID"
    ERROR_MSG=$(echo "$OUTPUT" | tail -3 | tr '\n' ' ')

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
