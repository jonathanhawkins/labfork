#!/bin/bash
# Send a research goal to the Nudge Engine.
# Usage: ./send-goal.sh <goal-id>     — send a specific goal from research-goals.json
#        ./send-goal.sh --list        — list available goals
#        ./send-goal.sh --custom "..." — send a custom goal string

set -euo pipefail

ENGINE="${ENGINE:-https://nudge-engine.jonathan-hawkins.workers.dev}"
API_KEY="${API_KEY:-19a2468f6508323409c1120e7ab64d3cbfff27aa27a9dc41b48988ca9b803078}"
GOALS_FILE="$(dirname "$0")/research-goals.json"

if [ "${1:-}" = "--list" ]; then
  echo "Available research goals:"
  jq -r '.goals[] | "  \(.id) [\(.difficulty)] — \(.title)"' "$GOALS_FILE"
  exit 0
fi

if [ "${1:-}" = "--custom" ]; then
  shift
  OBJECTIVE="$*"
  echo "Sending custom goal: ${OBJECTIVE:0:80}..."
  curl -sf -X POST "$ENGINE/observe" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"type\": \"goals\",
      \"data\": { \"objectives\": [$(printf '%s' "$OBJECTIVE" | jq -Rs .)] }
    }" | jq .
  exit 0
fi

GOAL_ID="${1:?Usage: send-goal.sh <goal-id|--list|--custom '...'>}"

OBJECTIVE=$(jq -r --arg id "$GOAL_ID" '.goals[] | select(.id == $id) | .objective' "$GOALS_FILE")
if [ -z "$OBJECTIVE" ] || [ "$OBJECTIVE" = "null" ]; then
  echo "Error: goal '$GOAL_ID' not found. Run with --list to see available goals."
  exit 1
fi

TITLE=$(jq -r --arg id "$GOAL_ID" '.goals[] | select(.id == $id) | .title' "$GOALS_FILE")
echo "Sending goal: $TITLE"
echo "  $GOAL_ID"

curl -sf -X POST "$ENGINE/observe" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"goals\",
    \"data\": { \"objectives\": [$(printf '%s' "$OBJECTIVE" | jq -Rs .)] }
  }" | jq .

echo ""
echo "Goal submitted. Trigger cron to create tasks:"
echo "  curl -X POST $ENGINE/cron -H 'Authorization: Bearer $API_KEY'"
