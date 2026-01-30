#!/bin/bash
# Test both models to see which works better with Claude Code

echo "🧪 Testing Claude Code with Ollama Models"
echo "=========================================="
echo ""

# Test 1: GLM-4.7-Flash
echo "Test 1: GLM-4.7-Flash"
echo "---------------------"
if ollama list | grep -q "glm-4.7-flash"; then
    echo "✓ Model installed"
    echo ""
    echo "Testing basic response..."
    timeout 30 ollama run glm-4.7-flash "Say 'hello' in one word" 2>&1 | head -10
    echo ""
    echo "Testing tool calling capability..."
    timeout 30 ollama run glm-4.7-flash "You have access to a calculator tool. Use it to compute 2+2" 2>&1 | head -20
else
    echo "✗ Not installed (downloading in background)"
fi

echo ""
echo ""

# Test 2: Qwen2.5-Coder
echo "Test 2: Qwen2.5-Coder:14b"
echo "-------------------------"
echo "✓ Model installed"
echo ""
echo "Testing basic response..."
timeout 30 ollama run qwen2.5-coder:14b "Say 'hello' in one word" 2>&1 | head -10
echo ""
echo "Testing code generation..."
timeout 30 ollama run qwen2.5-coder:14b "Write a Python hello world in one line" 2>&1 | head -20

echo ""
echo ""
echo "=========================================="
echo "RECOMMENDATION"
echo "=========================================="
echo ""
echo "Try claude-free now and see which model it uses:"
echo "  ./scripts/claude-free"
echo ""
echo "If GLM crashes or has issues:"
echo "  1. Edit scripts/claude-free"
echo "  2. Change MODEL=\"glm-4.7-flash\" to MODEL=\"qwen2.5-coder:14b\""
echo "  3. Or just uninstall: ollama rm glm-4.7-flash"
