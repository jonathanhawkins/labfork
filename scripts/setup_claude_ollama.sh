#!/bin/bash
# Setup Claude Code to work with FREE Ollama (Qwen2.5-Coder)

set -e

echo "🦙 Claude Code + Ollama Setup (FREE AI!)"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Ollama
echo "📋 Checking prerequisites..."
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama not installed${NC}"
    echo "Install with: brew install ollama"
    exit 1
fi
echo -e "${GREEN}✓ Ollama installed${NC}"

# Check Ollama version
OLLAMA_VERSION=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo -e "${GREEN}✓ Ollama version: $OLLAMA_VERSION${NC}"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama not running, starting...${NC}"
    ollama serve &
    sleep 3
fi
echo -e "${GREEN}✓ Ollama running on http://localhost:11434${NC}"

# Check for Qwen2.5-Coder
echo ""
echo "📦 Checking models..."
if ollama list | grep -q "qwen2.5-coder:14b"; then
    echo -e "${GREEN}✓ qwen2.5-coder:14b already installed (9GB)${NC}"
else
    echo -e "${YELLOW}📥 Pulling qwen2.5-coder:14b (this will take ~5 minutes)...${NC}"
    ollama pull qwen2.5-coder:14b
    echo -e "${GREEN}✓ qwen2.5-coder:14b installed${NC}"
fi

# Check Claude Code
echo ""
echo "📋 Checking Claude Code..."
if ! command -v claude &> /dev/null; then
    echo -e "${RED}❌ Claude Code not installed${NC}"
    echo "Install from: https://claude.ai/download"
    exit 1
fi
echo -e "${GREEN}✓ Claude Code CLI installed${NC}"

# Detect shell
SHELL_RC=""
if [ -f ~/.zshrc ]; then
    SHELL_RC=~/.zshrc
    SHELL_NAME="zsh"
elif [ -f ~/.bashrc ]; then
    SHELL_RC=~/.bashrc
    SHELL_NAME="bash"
else
    echo -e "${RED}❌ Could not find ~/.zshrc or ~/.bashrc${NC}"
    exit 1
fi

echo ""
echo "🔧 Configuring shell ($SHELL_NAME)..."

# Backup
cp "$SHELL_RC" "$SHELL_RC.backup.$(date +%s)"
echo -e "${GREEN}✓ Backed up $SHELL_RC${NC}"

# Check if already configured
if grep -q "ANTHROPIC_BASE_URL" "$SHELL_RC"; then
    echo -e "${YELLOW}⚠️  Configuration already exists in $SHELL_RC${NC}"
    echo "Would you like to:"
    echo "  1) Keep existing config"
    echo "  2) Replace with new config"
    read -p "Choice (1/2): " choice

    if [ "$choice" = "2" ]; then
        # Remove old config
        sed -i.bak '/# Claude Code - Ollama/,/alias claude-paid/d' "$SHELL_RC"
        echo -e "${GREEN}✓ Removed old configuration${NC}"
    else
        echo "Keeping existing configuration"
    fi
fi

# Add configuration if not present
if ! grep -q "alias claude-free" "$SHELL_RC"; then
    cat >> "$SHELL_RC" << 'EOF'

# ============================================
# Claude Code - Ollama Integration (FREE!)
# ============================================

# Default to FREE Ollama (comment out to use paid API)
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"

# Aliases for easy switching
alias claude-free='ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_AUTH_TOKEN=ollama claude --model qwen2.5-coder:14b'
alias claude-paid='ANTHROPIC_BASE_URL="" ANTHROPIC_AUTH_TOKEN="" claude'

# Quick toggle
function claude-mode() {
    if [ "$1" = "free" ]; then
        export ANTHROPIC_BASE_URL="http://localhost:11434"
        export ANTHROPIC_AUTH_TOKEN="ollama"
        echo "🦙 Switched to FREE mode (Ollama)"
    elif [ "$1" = "paid" ]; then
        unset ANTHROPIC_BASE_URL
        unset ANTHROPIC_AUTH_TOKEN
        echo "💳 Switched to PAID mode (Anthropic API)"
    else
        echo "Usage: claude-mode [free|paid]"
    fi
}

EOF
    echo -e "${GREEN}✓ Added configuration to $SHELL_RC${NC}"
fi

# Test connection
echo ""
echo "🧪 Testing connection..."

# Test Ollama
TEST_RESPONSE=$(curl -s http://localhost:11434/api/tags)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Ollama API responding${NC}"
else
    echo -e "${RED}❌ Ollama API not responding${NC}"
    exit 1
fi

# Test Qwen model
echo "Testing Qwen2.5-Coder..."
TEST_OUTPUT=$(ollama run qwen2.5-coder:14b "Say 'hello' in one word" 2>&1 | head -5)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Qwen2.5-Coder responding${NC}"
else
    echo -e "${RED}❌ Qwen2.5-Coder not responding${NC}"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Reload your shell:"
echo -e "   ${GREEN}source $SHELL_RC${NC}"
echo ""
echo "2. Test FREE mode:"
echo -e "   ${GREEN}claude-free${NC}"
echo "   > What model are you?"
echo ""
echo "3. Test PAID mode:"
echo -e "   ${GREEN}claude-paid${NC}"
echo "   > What model are you?"
echo ""
echo "4. Switch modes:"
echo -e "   ${GREEN}claude-mode free${NC}  # Use FREE Ollama"
echo -e "   ${GREEN}claude-mode paid${NC}  # Use PAID Anthropic"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💰 COST SAVINGS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Research/coding:    $50/day  → $0/day  (FREE)"
echo "Implementation:     $200     → $10     (95% savings)"
echo "Monthly estimate:   $1200    → $150    (88% savings)"
echo ""
echo -e "${YELLOW}💡 TIP: Use FREE by default, escalate to PAID only when stuck!${NC}"
echo ""
echo "📖 Full docs: docs/CLAUDE_CODE_OLLAMA_SETUP.md"
