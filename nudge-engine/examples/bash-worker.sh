#!/bin/bash
#
# Minimal Nudge Engine worker in bash.
# Polls for tasks, runs them, reports results.
#
# Usage:
#   # First register:
#   curl -s -X POST https://your-engine.workers.dev/register \
#     -H "Content-Type: application/json" \
#     -d '{"name": "my-worker", "type": "custom"}' | jq .
#
#   # Then run with the token:
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 ./bash-worker.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Worker starting. Polling $ENGINE every ${POLL_INTERVAL}s"

# Disable set -e for the main loop — a daemon should never die from a stray exit code.
set +e

while true; do
  # Poll for work
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
  log "Got task: $ACTION ($TASK_ID)"
  log "  $DESC"

  # ---- Do the work here ----
  # Replace this with your actual logic.
  # For example, run tests, lint code, deploy, etc.
  #
  # SUCCESS=true/false and SUMMARY="what happened"
  SUCCESS=true
  SUMMARY="Completed $ACTION"
  # --------------------------

  # Report result
  curl -sf -X POST "$ENGINE/report" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"taskId\": \"$TASK_ID\",
      \"success\": $SUCCESS,
      \"result\": {\"summary\": $(echo "$SUMMARY" | jq -Rs .)}
    }" > /dev/null 2>&1 || log "Warning: failed to report $TASK_ID"

  log "Reported: $TASK_ID ($( [ "$SUCCESS" = true ] && echo 'ok' || echo 'fail' ))"
  sleep 5
done
