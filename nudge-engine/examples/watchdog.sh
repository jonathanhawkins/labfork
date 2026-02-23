#!/bin/bash
#
# Watchdog — restarts the Ollama worker if it's not running.
#
# Add to crontab on the worker machine:
#   crontab -e
#   */5 * * * * ~/bin/watchdog.sh >> ~/watchdog.log 2>&1
#
# That's it. If the worker dies, it's back within 5 minutes.

# Cron has a minimal environment — set PATH so jq and other tools are found.
export PATH="$HOME/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

WORKER_SCRIPT="${WORKER_SCRIPT:-$HOME/bin/run-nudge-worker.sh}"
WORKER_LOG="${WORKER_LOG:-$HOME/nudge-worker.log}"
PIDFILE="${PIDFILE:-$HOME/.nudge-worker.pid}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Primary check: is the worker process actually running?
# pgrep is the source of truth — PID file can go stale.
RUNNING_PID=$(pgrep -f "[o]llama-worker.sh" | head -1)

if [ -n "$RUNNING_PID" ]; then
  # Worker is alive. Make sure PID file matches.
  if [ -f "$PIDFILE" ]; then
    RECORDED_PID=$(cat "$PIDFILE")
    if [ "$RECORDED_PID" != "$RUNNING_PID" ]; then
      echo "$RUNNING_PID" > "$PIDFILE"
    fi
  else
    echo "$RUNNING_PID" > "$PIDFILE"
  fi
  exit 0
fi

# Worker is dead. Clean up stale PID file if present.
if [ -f "$PIDFILE" ]; then
  rm -f "$PIDFILE"
fi

# Restart
log "Worker not running. Restarting..."
nohup "$WORKER_SCRIPT" >> "$WORKER_LOG" 2>&1 &
echo $! > "$PIDFILE"
log "Started with PID $!"
