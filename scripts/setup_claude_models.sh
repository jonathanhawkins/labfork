#!/bin/bash
# Create Ollama models optimized for Claude Code

cd "$(dirname "$0")"

echo "🔧 Setting up Claude Code optimized models"
echo "==========================================="
echo ""

# Create Qwen variant
echo "Creating qwen-claude model..."
ollama create qwen-claude -f Modelfile.qwen-claude
if [ $? -eq 0 ]; then
    echo "✓ qwen-claude created"
else
    echo "✗ Failed to create qwen-claude"
fi
echo ""

# Create GLM variant (only if GLM is installed)
if ollama list | grep -q "glm-4.7-flash"; then
    echo "Creating glm-claude model..."
    ollama create glm-claude -f Modelfile.glm-claude
    if [ $? -eq 0 ]; then
        echo "✓ glm-claude created"
    else
        echo "✗ Failed to create glm-claude"
    fi
else
    echo "⏳ GLM-4.7-Flash not installed yet, skipping glm-claude"
    echo "   Run this script again after 'ollama pull glm-4.7-flash' completes"
fi
echo ""

echo "==========================================="
echo "Available models for Claude Code:"
ollama list | grep -E "(qwen-claude|glm-claude)"
echo ""
echo "Update claude-free scripts to use these optimized models"
