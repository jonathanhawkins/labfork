#!/bin/bash
#
# Nudge Engine worker that dispatches tasks to Claude Code.
#
# Each task becomes a Claude Code session that runs, completes, and reports back.
# This is the "glue" between nudge-engine and the Claude Code CLI.
#
# Prerequisites:
#   - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
#   - Authenticated: claude login
#
# Usage:
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 ./claude-code-worker.sh
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 REPO_DIR=/path/to/repo ./claude-code-worker.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Claude Code worker starting"
log "  Engine:   $ENGINE"
log "  Repo:     $REPO_DIR"
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

  # Build the prompt for Claude Code
  PROMPT="You are working on task: $ACTION

$DESC

Context: $CONTEXT

Instructions:
- Work in the repository at $REPO_DIR
- Make the minimal changes needed
- Run tests if applicable
- When done, output a one-line summary starting with SUMMARY:"

  # Run Claude Code in non-interactive mode
  cd "$REPO_DIR"
  OUTPUT=$(claude -p "$PROMPT" --output-format text 2>&1) || true

  # Extract summary from output (last line starting with SUMMARY:)
  SUMMARY=$(echo "$OUTPUT" | grep "^SUMMARY:" | tail -1 | sed 's/^SUMMARY: *//')
  if [ -z "$SUMMARY" ]; then
    SUMMARY="Completed $ACTION (no explicit summary)"
  fi

  # Check if Claude Code succeeded (basic heuristic: no error in last lines)
  if echo "$OUTPUT" | tail -5 | grep -qi "error\|failed\|unable"; then
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
