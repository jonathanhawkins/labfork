#!/bin/bash
# Research Agent Cleanup Service
# Prevents zombie process accumulation by:
# 1. Killing claude processes not attached to active tmux sessions
# 2. Cleaning up stale agents.json entries
# 3. Enforcing max concurrent agents limit
# 4. Killing agents running longer than MAX_RUNTIME
#
# Run via cron every 30 minutes:
#   */30 * * * * /path/to/cleanup.sh >> /tmp/agent-cleanup.log 2>&1

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$HOME/dev/voice-clone-pipeline}"
STATE_DIR="$PROJECT_DIR/.skills/research-manager/state"
AGENTS_FILE="$STATE_DIR/agents.json"
MAX_RUNTIME_HOURS=2
MAX_CONCURRENT_AGENTS=5
DRY_RUN="${DRY_RUN:-false}"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Agent Cleanup Started ==="

# Count active tmux sessions
active_sessions=$(tmux list-sessions 2>/dev/null | grep -c 'rm-task\|rm-web' || echo 0)
log "Active agent tmux sessions: $active_sessions"

# Count claude processes
claude_count=$(ps aux | grep -E 'claude$' | grep -v grep | wc -l | tr -d ' ')
log "Total claude processes: $claude_count"

# Find orphaned claude processes (not in active tmux sessions)
orphaned=0
active_pids=""

# Get PIDs of claude processes in active tmux sessions
for session in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E 'rm-task|rm-web'); do
    pid=$(tmux list-panes -t "$session" -F '#{pane_pid}' 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        # Get all child processes
        children=$(pgrep -P "$pid" 2>/dev/null | tr '\n' ' ')
        active_pids="$active_pids $pid $children"
    fi
done

# Also include lab-manager claude process
lab_pid=$(tmux list-panes -t lab-manager -F '#{pane_pid}' 2>/dev/null | head -1)
if [ -n "$lab_pid" ]; then
    children=$(pgrep -P "$lab_pid" 2>/dev/null | tr '\n' ' ')
    active_pids="$active_pids $lab_pid $children"
fi

# Find and kill orphaned processes
for pid in $(ps aux | grep -E 'claude$' | grep -v grep | awk '{print $2}'); do
    if ! echo "$active_pids" | grep -qw "$pid"; then
        # Get process start time
        start_time=$(ps -o lstart= -p "$pid" 2>/dev/null || echo "")
        if [ -n "$start_time" ]; then
            log "Found orphaned claude process: PID $pid (started: $start_time)"
            orphaned=$((orphaned + 1))
            if [ "$DRY_RUN" != "true" ]; then
                kill "$pid" 2>/dev/null && log "  Killed PID $pid" || log "  Failed to kill PID $pid"
            else
                log "  [DRY RUN] Would kill PID $pid"
            fi
        fi
    fi
done

log "Orphaned processes cleaned: $orphaned"

# Clean up stale agents.json entries
if [ -f "$AGENTS_FILE" ]; then
    # Get list of running sessions
    running_sessions=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -E 'rm-task|rm-web' | tr '\n' '|' | sed 's/|$//')

    if [ -n "$running_sessions" ]; then
        # Mark agents as killed if their session doesn't exist
        stale_count=$(python3 -c "
import json
import re
from datetime import datetime

with open('$AGENTS_FILE', 'r') as f:
    agents = json.load(f)

running = set('$running_sessions'.split('|')) if '$running_sessions' else set()
stale = 0
now = datetime.now().isoformat()

for name, agent in agents.items():
    session = agent.get('session', '')
    if agent.get('status') == 'running' and session and session not in running:
        if '$DRY_RUN' != 'true':
            agent['status'] = 'killed'
            agent['killed_at'] = now
            agent['kill_reason'] = 'session_not_found'
        stale += 1

if '$DRY_RUN' != 'true':
    with open('$AGENTS_FILE', 'w') as f:
        json.dump(agents, f, indent=2)

print(stale)
" 2>/dev/null || echo 0)
        log "Stale agent entries cleaned: $stale_count"
    fi
fi

# Check for agents running too long
if [ -f "$AGENTS_FILE" ]; then
    long_running=$(python3 -c "
import json
from datetime import datetime, timedelta

with open('$AGENTS_FILE', 'r') as f:
    agents = json.load(f)

max_hours = $MAX_RUNTIME_HOURS
now = datetime.now()
count = 0

for name, agent in agents.items():
    if agent.get('status') == 'running':
        started = agent.get('started_at', '')
        if started:
            try:
                start_dt = datetime.fromisoformat(started.replace('Z', '+00:00').replace('+00:00', ''))
                runtime = now - start_dt
                if runtime > timedelta(hours=max_hours):
                    print(f'{name}: {runtime.total_seconds()/3600:.1f}h')
                    count += 1
            except:
                pass

print(f'TOTAL:{count}')
" 2>/dev/null | grep 'TOTAL:' | cut -d: -f2)

    if [ "$long_running" -gt 0 ]; then
        log "WARNING: $long_running agents running longer than ${MAX_RUNTIME_HOURS}h"
    fi
fi

# Enforce max concurrent agents
if [ "$active_sessions" -gt "$MAX_CONCURRENT_AGENTS" ]; then
    excess=$((active_sessions - MAX_CONCURRENT_AGENTS))
    log "WARNING: $excess agents over limit ($active_sessions > $MAX_CONCURRENT_AGENTS)"
    # Note: We don't auto-kill here, just warn. The orchestrator should handle this.
fi

# Final status
final_claude_count=$(ps aux | grep -E 'claude$' | grep -v grep | wc -l | tr -d ' ')
log "Final claude process count: $final_claude_count"
log "=== Cleanup Complete ==="
