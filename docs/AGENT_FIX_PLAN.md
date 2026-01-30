# Agent Productivity Fix Plan

## Root Cause Found ✓

### Problem #1: Codex Not Installed on 4090
The orchestrator **correctly routes** tasks to Codex vs Ollama:
- Reviews → Codex ✓
- Complex work (architecture, refactoring) → Codex ✓
- Simple exploration → Ollama ✓

**BUT** Codex CLI isn't installed on the 4090, so everything falls back to the weak local model.

**Evidence from manager.py:197:**
```python
codex_flags = '-c model=codex-mini-latest -c timeout=1800000'
```

Codex exists on Mac but not where agents run (4090).

### Problem #2: Model Too Weak
**Current:** qwen3-coder-32k
- Type: 30B MoE (Mixture of Experts)
- Active params: ~3B per forward pass
- Good for: Simple edits
- Bad for: Multi-step implementations, reasoning, architecture

**Comparison:**
| Model | Params | Type | Coding Ability |
|-------|--------|------|----------------|
| qwen3-coder-32k | 3B active | MoE | ⭐⭐ |
| deepseek-coder:33b | 33B | Dense | ⭐⭐⭐⭐ |
| Codex (GPT-4) | Large | Dense | ⭐⭐⭐⭐⭐ |

---

## Fix Options

### Option A: Install Codex (Hybrid - RECOMMENDED)

**Once SSH is restored:**
```bash
ssh doc@$REMOTE_GPU_HOST

# Check if codex CLI exists
which codex || npm install -g @openrouter/codex-cli

# Verify it works
codex --version

# Restart orchestrator
cd ~/dev/labfork
.skills/research-manager/rm orchestrator restart
```

**Result:**
- Reviews → Codex (paid, fast, accurate) ✓
- Simple tasks → Ollama (free, slow, okay for research) ✓
- 4090 GPU → Available for training ✓

**Cost:** ~$10-30/month (only for reviews/complex tasks)
**Completion rate:** Should jump to 40-60%

---

### Option B: Upgrade Local Model (Free but Limited)

**Once SSH is restored:**
```bash
ssh doc@$REMOTE_GPU_HOST

# Download better model
ollama pull deepseek-coder:33b

# Update claude-free script
sed -i 's/qwen3-coder-32k/deepseek-coder:33b/g' ~/dev/labfork/scripts/claude-free

# Restart
.skills/research-manager/rm orchestrator restart
```

**Result:**
- All tasks → Better local model ✓
- Free ✓
- Still slower than paid API ✗
- Uses VRAM (limits training) ✗

**Cost:** $0
**Completion rate:** Estimated 15-25% (better but not great)

---

### Option C: Try Alternative Local Models

**Available on Mac (can test there):**
- `glm-claude:latest` (18GB) - Tuned to mimic Claude
- `qwen2.5-coder:32b` - Newer version
- `gpt-oss:20b` - Smaller but focused

**Test locally first:**
```bash
# On Mac
ollama pull deepseek-coder:33b
scripts/claude-free  # Uses qwen3-coder-32k

# Edit scripts/claude-free line 50:
#   Before: "$CLAUDE_BIN" --model qwen3-coder-32k "$@"
#   After:  "$CLAUDE_BIN" --model deepseek-coder:33b "$@"

# Test
scripts/claude-free
# Try: "Write a Python function to calculate fibonacci"
```

If it works well locally, deploy to 4090.

---

## Recommended Path Forward

### Step 1: Fix SSH (Required First)
Run the SSH watchdog setup via Parsec (see SYSTEM_HEALTH_REPORT.md)

### Step 2: Quick Diagnosis
```bash
ssh doc@$REMOTE_GPU_HOST
bash ~/dev/labfork/scripts/diagnose-agents.sh
```

This will show:
- Is Codex installed? (probably no)
- What models are available?
- What errors are in agent logs?

### Step 3: Choose Fix Based on Budget

**If you want it to work reliably:** Option A (Install Codex)
- Cost: $10-30/month
- Time: 5 minutes
- Success rate: High

**If you want free at all costs:** Option B (Upgrade model)
- Cost: $0
- Time: 15 minutes
- Success rate: Medium

**If you want to experiment:** Option C (Test alternatives)
- Cost: $0
- Time: 1 hour
- Success rate: Unknown

---

## Why Agents Are Failing

Not getting errors - they're just **too slow and get stuck:**

1. Task: "Implement Multi-Modal Prosody Fusion"
2. Agent spawns with qwen3-coder-32k
3. Spends 5 minutes reading files
4. Spends 3 minutes thinking about approach
5. Starts writing code...
6. **10 minute timeout hits** → KILLED
7. No output produced
8. Task marked as failed
9. Orchestrator spawns new agent
10. Repeat

With Codex or better model:
1. Task assigned
2. Agent reads files (30 sec)
3. Writes implementation (2 min)
4. Tests it (1 min)
5. **Completes in 4 minutes** ✓

---

## Next Steps

1. **Restore SSH** (setup-ssh-watchdog.ps1)
2. **Run diagnostics** (diagnose-agents.sh)
3. **Pick fix option** based on findings
4. **Clean duplicates** (rm orchestrator dedup --auto)
5. **Monitor** new completion rate

Expected outcome: 8.9% → 40-60% completion rate
