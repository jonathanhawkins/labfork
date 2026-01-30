# Ollama Integration & AI Cost Strategy

This document covers the local Ollama setup for free Claude Code usage and the hybrid cost optimization strategy.

## FREE Local Claude Code with Ollama (2026 Update)

Run Claude Code for FREE using local Ollama models via the **Anthropic API compatibility** (added in Ollama 0.14.0, January 2026).

### How It Works

As of January 2026, Ollama supports the Anthropic Messages API, allowing Claude Code CLI to connect directly to local Ollama models by setting environment variables. No cloud API costs!

### Requirements

- **Ollama 0.14.0+** (on 4090: 0.15.2)
- **Claude Code CLI** or **Codex CLI** (compatible alternative)
- **Model with tool support**: qwen3-coder-32k, glm-4.7, etc.

### Setup (One-time)

```bash
# 1. Install Ollama (if not installed)
# Mac:
brew install ollama
# Linux (4090/WSL2):
curl https://ollama.ai/install.sh | sh

# 2. Download model with large context (required for tools)
ollama pull qwen3-coder:30b

# 3. Create 32k context version (CRITICAL - Claude Code needs 20k+ tokens)
cat > /tmp/Modelfile.qwen3-coder-32k << 'EOF'
FROM qwen3-coder:30b
PARAMETER num_ctx 32768
EOF
ollama create qwen3-coder-32k -f /tmp/Modelfile.qwen3-coder-32k

# 4. Verify
ollama list | grep qwen3-coder-32k
```

### Usage

**Method 1: Environment Variables (Recommended)**

```bash
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_API_KEY=""  # Required but ignored

# Then run claude/codex normally
codex --model qwen3-coder-32k "your prompt here"
```

**Method 2: Using ollama launch (Simplest)**

```bash
# Ollama 0.15+ includes this command
ollama launch claude --model qwen3-coder-32k
```

**Method 3: Scripts**

```bash
# Mac - Use the provided script
./scripts/claude-free

# 4090 - Configured in research orchestrator (automatic)
```

### Key Details

- **Model**: qwen3-coder-32k (30B MoE, ~3B active params)
- **Memory**: ~21GB GPU (fits on 48GB M4 Pro, 24GB 4090)
- **Context**: 32768 tokens (required for tool definitions)
- **Speed**: 40-50 tokens/sec on 4090, slower on Mac
- **Cost**: FREE (no API calls)
- **Privacy**: Code never leaves your machine

### WSL2/Windows 10 Setup

On the 4090 (Windows 10 + WSL2), enable localhost forwarding:

**File: `C:\Users\Doc Holiday\.wslconfig`**
```ini
[wsl2]
localhostForwarding=true
```

Then restart WSL: `wsl --shutdown`

See [WSL2_OLLAMA_TROUBLESHOOTING.md](./WSL2_OLLAMA_TROUBLESHOOTING.md) for complete troubleshooting guide.

### Troubleshooting

**Tools don't work**: Check context size
```bash
ollama ps  # Should show CONTEXT: 32768
```

**Connection refused**: Check Ollama is running
```bash
curl http://localhost:11434/api/version
# Should return: {"version":"0.15.2"}
```

### Sources

- [Ollama Anthropic API compatibility](https://ollama.com/blog/claude)
- [Run Claude Code with Local LLMs](https://medium.com/data-science-in-your-pocket/run-claude-code-with-local-llms-using-ollama-a97d2c2f2bd1)
- [Ollama API Documentation](https://docs.ollama.com/api/anthropic-compatibility)

---

## AI Cost Optimization Strategy

Use FREE local models for exploration/research, automatically route complex coding tasks to paid OpenAI Codex.

### What is Codex?

**Codex** = OpenAI's Codex CLI (https://github.com/openai/codex)
- **NOT a free tool** - Uses OpenAI API (paid)
- Terminal-based coding agent from OpenAI
- Much better at coding than local Ollama models
- Used automatically by orchestrator for reviews and complex tasks

### Installation

```bash
# Mac (via Homebrew)
brew install --cask codex

# Linux (4090 machine)
curl -fsSL https://github.com/openai/codex/releases/download/rust-v0.92.0/codex-x86_64-unknown-linux-musl.tar.gz | tar -xz
mv codex-x86_64-unknown-linux-musl ~/bin/codex
chmod +x ~/bin/codex

# Login with OpenAI account
codex login
```

### Task Routing (Automatic)

The orchestrator automatically routes tasks based on complexity:

| Task Type | Agent Type | Model | Cost |
|-----------|------------|-------|------|
| Simple exploration, research | Ollama | qwen3-coder-32k | FREE |
| Reviews, audits, validation | **Codex** | codex-mini-latest | Paid (~$0.50/task) |
| Complex multi-file work | **Codex** | codex-mini-latest | Paid (~$1-2/task) |
| Architecture, refactoring | **Codex** | codex-mini-latest | Paid (~$2-5/task) |

**No manual routing needed** - the orchestrator picks the right tool automatically.

### Available Tools

```bash
# Lab manager (Claude Code + Ollama) - FREE
./scripts/claude-free
tmux new-session -s claude-free "./scripts/claude-free"

# Research orchestrator on 4090 - Hybrid (FREE + Paid)
ssh doc@$REMOTE_GPU_HOST -t "tmux attach -t lab-manager"
```

---

## 4090 Research System (Hybrid: FREE + Paid)

The 4090 runs a **hybrid system** that automatically uses the right tool:
- **Ollama (FREE)** for simple exploration and web research
- **Codex (PAID)** for complex coding, reviews, and implementations

### How It Works

The orchestrator (`orchestrator.js`) automatically selects the right agent:

```javascript
// Simple task -> Ollama (FREE)
"Explore DiffStyleTTS approach" -> type: "ollama"

// Complex task -> Codex (PAID)
"Review prosody implementation" -> type: "codex" (keyword: review)
"Refactor multi-file architecture" -> type: "codex" (keyword: refactor)
```

### Setup on 4090

```bash
# SSH into 4090
ssh doc@$REMOTE_GPU_HOST

# Install Codex CLI (one-time)
cd /tmp
curl -fsSL https://github.com/openai/codex/releases/download/rust-v0.92.0/codex-x86_64-unknown-linux-musl.tar.gz -o codex.tar.gz
tar -xzf codex.tar.gz
mkdir -p ~/bin
mv codex-x86_64-unknown-linux-musl ~/bin/codex
chmod +x ~/bin/codex

# Login with OpenAI account
~/bin/codex login

# Start orchestrator
cd ~/dev/labfork
.skills/research-manager/rm orchestrator start

# Start Ollama
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

### Monitoring

```bash
# Check orchestrator status
ssh doc@$REMOTE_GPU_HOST "cd ~/dev/labfork && .skills/research-manager/rm orchestrator status"

# View logs
ssh doc@$REMOTE_GPU_HOST "cd ~/dev/labfork && .skills/research-manager/rm orchestrator logs"

# Check GPU usage
ssh doc@$REMOTE_GPU_HOST "/usr/lib/wsl/lib/nvidia-smi"

# Check what agents are running
ssh doc@$REMOTE_GPU_HOST "tmux list-sessions"
```

### Cost Estimate

With hybrid routing:
- ~5-10 Codex calls per day = $5-15/day
- Simple exploration = FREE (Ollama)
- **Much cheaper than all-paid** ($100+/day)
- **Much better than all-free** (8.9% completion rate -> 40-60%)
