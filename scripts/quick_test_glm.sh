#!/bin/bash
# Quick test to see if GLM-4.7-Flash is working

echo "🧪 Quick GLM-4.7-Flash Test"
echo "============================="
echo ""

if ! ollama list | grep -q "glm-4.7-flash"; then
    echo "❌ GLM-4.7-Flash not installed yet"
    echo "   Run: ollama pull glm-4.7-flash"
    exit 1
fi

echo "✓ GLM-4.7-Flash is installed"
echo ""
echo "Testing basic response..."
echo ""

ollama run glm-4.7-flash "Say hello in exactly 3 words" 2>&1 | head -20

echo ""
echo ""
echo "============================="
echo "If you see a coherent 3-word response above, GLM is working!"
echo ""
echo "Try it with Claude Code:"
echo "  ./scripts/claude-free-glm"
