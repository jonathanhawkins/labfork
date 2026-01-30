#!/bin/bash
# Setup Ollama for research-manager skill (FREE alternative to Claude API)

echo "🦙 Setting up Ollama Research Manager (FREE!)"
echo "=============================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not installed. Install with: brew install ollama"
    exit 1
fi
echo "✓ Ollama installed"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama not running. Starting..."
    ollama serve &
    sleep 3
fi
echo "✓ Ollama running"

# Check for Qwen2.5-Coder
if ! ollama list | grep -q "qwen2.5-coder:14b"; then
    echo "📥 Downloading Qwen2.5-Coder:14b (this will take a few minutes)..."
    ollama pull qwen2.5-coder:14b
else
    echo "✓ Qwen2.5-Coder:14b ready"
fi

# Install MCP client for Ollama
echo ""
echo "📦 Installing MCP client for Ollama..."
pip3 install mcp-client-for-ollama langchain-ollama --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "💰 Cost Savings:"
echo "   - Code analysis: $0 (was $15-30/session)"
echo "   - Web research: $0 with free tier (was $10-20/session)"
echo "   - 10 research agents: $0 (was $100-200/day)"
echo ""
echo "🎯 Next Steps:"
echo "   1. Sign up for free Ollama account (for web search):"
echo "      https://ollama.com/signup"
echo ""
echo "   2. Get API key and set:"
echo "      export OLLAMA_API_KEY='your-key-here'"
echo ""
echo "   3. Update research-manager to use Ollama"
echo "      (I'll create the integration script next)"
