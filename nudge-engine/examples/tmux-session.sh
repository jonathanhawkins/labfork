#!/bin/bash
#
# tmux multi-worker session.
#
# Creates a tmux session with one pane per worker:
#   - Pane 0: Claude Code worker
#   - Pane 1: Git observer (runs once, then exits)
#   - Pane 2: Dashboard (watches stats)
#
# Usage:
#   ENGINE=https://your-engine.workers.dev \
#   TOKEN=abc123 \
#   REPO=owner/repo \
#   ./tmux-session.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
REPO="${REPO:-}"
SESSION="nudge"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with the Claude Code worker
tmux new-session -d -s "$SESSION" -n "workers"
tmux send-keys -t "$SESSION:0" \
  "ENGINE=$ENGINE TOKEN=$TOKEN $SCRIPT_DIR/claude-code-worker.sh" Enter

# Split horizontally — stats dashboard
tmux split-window -h -t "$SESSION:0"
tmux send-keys -t "$SESSION:0.1" \
  "watch -n 30 'curl -s $ENGINE/stats | jq .'" Enter

# Split the right pane vertically — git observer on a loop
if [ -n "$REPO" ]; then
  tmux split-window -v -t "$SESSION:0.1"
  tmux send-keys -t "$SESSION:0.2" \
    "while true; do ENGINE=$ENGINE TOKEN=$TOKEN REPO=$REPO $SCRIPT_DIR/git-observer.sh; sleep 900; done" Enter
fi

# Nice layout
tmux select-layout -t "$SESSION:0" main-vertical

echo "tmux session '$SESSION' created. Attach with:"
echo "  tmux attach -t $SESSION"
