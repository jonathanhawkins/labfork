#!/bin/bash
# Diagnose why agents are failing
# Run this on the 4090 once SSH is back up

echo "=== Agent Diagnostics ==="
echo

echo "1. Codex Installation:"
which codex && echo "✓ Codex found" || echo "✗ Codex NOT found (this is the problem!)"
codex --version 2>/dev/null || echo "  Cannot check version"
echo

echo "2. Ollama Models:"
ollama list
echo

echo "3. Current Model:"
grep -a "qwen3-coder\|glm-claude\|MODEL" ~/.skills/research-manager/state/orchestrator.log 2>/dev/null | tail -5
echo

echo "4. Recent Agent Failures:"
echo "Last 10 agent spawns:"
grep -a "Spawned agent" ~/dev/voice-clone-pipeline/.skills/research-manager/state/orchestrator.log 2>/dev/null | tail -10 | while read line; do
  echo "  $line"
done
echo

echo "5. Codex Fallback Warnings:"
grep -ac "Codex required but unavailable" ~/dev/voice-clone-pipeline/.skills/research-manager/state/orchestrator.log 2>/dev/null
echo "times Codex was needed but missing"
echo

echo "6. Check Latest Agent Log for Errors:"
LATEST_LOG=$(ls -t ~/dev/voice-clone-pipeline/.skills/research-manager/state/outputs/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
  echo "Latest: $LATEST_LOG"
  echo "Errors:"
  strings "$LATEST_LOG" | grep -i "error\|failed\|exception" | head -5
else
  echo "No agent logs found"
fi
echo

echo "7. Recommended Fix:"
echo "  Install Codex on 4090:"
echo "    npm install -g @openrouter/codex-cli"
echo
echo "  Or try larger local model:"
echo "    ollama pull deepseek-coder:33b"
echo "    # Then update claude-free to use deepseek-coder:33b"
