#!/bin/bash
# Setup Claude Code to use Ollama with local models

echo "🦙 Setting up Claude Code + Ollama integration"
echo ""

# Check Ollama version
OLLAMA_VERSION=$(ollama --version | head -1 | awk '{print $3}')
echo "✓ Ollama version: $OLLAMA_VERSION (need 0.14.0+)"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Starting it..."
    ollama serve &
    sleep 3
fi

echo "✓ Ollama is running on http://localhost:11434"
echo ""

# List available models
echo "📦 Available local models:"
ollama list
echo ""

# Create Claude settings directory if it doesn't exist
mkdir -p ~/.claude

# Check if settings file exists
if [ -f ~/.claude/settings.json ]; then
    echo "⚠️  ~/.claude/settings.json already exists"
    echo "   Manual action needed: Add this to your settings.json:"
    echo ""
    cat << 'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
EOF
else
    echo "✓ Creating ~/.claude/settings.json"
    cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
EOF
fi

echo ""
echo "🎯 To use local models with Claude Code:"
echo ""
echo "1. With inline env vars:"
echo "   ANTHROPIC_AUTH_TOKEN=ollama ANTHROPIC_BASE_URL=http://localhost:11434 claude"
echo ""
echo "2. Or add to your ~/.zshrc:"
echo "   export ANTHROPIC_BASE_URL='http://localhost:11434'"
echo "   export ANTHROPIC_AUTH_TOKEN='ollama'"
echo "   export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1'"
echo ""
echo "3. Then run:"
echo "   claude --model qwen2.5-coder:14b"
echo ""
echo "💡 For research triage, use qwen2.5-coder:14b (8GB RAM, GPT-4 class)"
