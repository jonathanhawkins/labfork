#!/bin/bash
# Monitor the Ollama triage progress

echo "🦙 Monitoring Ollama Research Triage"
echo "===================================="
echo ""

OUTPUT_FILE="/private/tmp/claude/-Users-light-dev-web-apps-voice-clone-pipeline/tasks/bac3bd8.output"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ Triage not running"
    exit 1
fi

# Show live progress
echo "📊 Live Progress:"
echo ""
tail -f "$OUTPUT_FILE" | grep --line-buffered -E "^\[|Score:|TOP 10|TOTAL:|saved to"
