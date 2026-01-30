# AI Cost Optimization Strategy

## Goal: 95% FREE, 5% Paid

Use the right AI for each task to minimize costs while maintaining quality.

## The Team

| Agent | Model | Cost | Use For |
|-------|-------|------|---------|
| **Ollama** | Qwen2.5-Coder 14B | **FREE** | 90% of work - grunt work, implementation, monitoring |
| **Opus** | Claude Opus 4.5 | $$$ | 5% - Management, decisions, user interaction |
| **Codex** | Claude Sonnet 4.5 | $$ | 5% - Bug fixes, code review, new model architecture |

## Workflow for Top 3 Implementations

### Phase 1: Preparation (100% FREE)

**Task: Understand the 3 approaches**
- ✅ **Ollama**: Read papers, summarize techniques
- ✅ **Ollama**: Extract key equations/architecture
- ✅ **Ollama**: Generate implementation checklist
- ❌ **No paid AI needed**

**Cost:** $0

### Phase 2: Initial Implementation (95% FREE)

**Task: Write training code for sparse_keyframe, ece_tts_easv, draw_speech**

**Ollama does (FREE):**
1. Generate boilerplate code from existing patterns
2. Port similar code from reference implementations
3. Create dataset loaders
4. Write training loops (copy from V7 baseline)
5. Add logging and metrics
6. Write initial tests
7. Generate documentation

**Codex reviews ONLY ($$):**
- Final pass for subtle bugs
- Architecture validation
- "Does this match the paper?"

**Cost:** ~$5-10 (Codex review only)

### Phase 3: Testing & Debugging (90% FREE)

**Task: Get code running on RTX 4090**

**Ollama does (FREE):**
1. Run pytest, fix syntax errors
2. Monitor training logs
3. Detect common issues (NaN loss, OOM, etc.)
4. Suggest fixes for standard errors
5. Generate debugging scripts

**Codex steps in ONLY when ($):**
- Ollama tried 3 times and failed
- Bug is complex/subtle
- Architecture mismatch with paper

**Cost:** ~$10-20 (only for hard bugs)

### Phase 4: Training Monitoring (100% FREE)

**Task: Monitor 3 training runs on RTX 4090**

**Ollama does (FREE):**
```python
# FREE training monitor
while training_running():
    logs = tail_logs('training.log', lines=100)

    analysis = ollama_analyze(logs)
    # "Is loss decreasing? Any NaNs? Overfitting?"

    if analysis['issue_detected']:
        alert(analysis['recommendation'])
        # "Loss stuck - try reducing LR"
```

**Opus/Codex: NEVER needed**

**Cost:** $0

### Phase 5: Evaluation (90% FREE)

**Task: Compare 3 approaches vs V7 baseline**

**Ollama does (FREE):**
1. Parse evaluation metrics (F0 separation, MCD, etc.)
2. Generate comparison tables
3. Plot results
4. Identify winner
5. Draft findings report

**Opus does ($$):**
- Read Ollama's report
- Make final recommendation to user
- Explain trade-offs

**Cost:** ~$5

### Phase 6: Integration (80% FREE)

**Task: Integrate best approach into production**

**Ollama does (FREE):**
1. Refactor code for production
2. Add error handling
3. Write API endpoints
4. Generate tests
5. Update documentation

**Codex reviews ($):**
- Security check
- Performance review
- Production readiness

**Cost:** ~$10-20

## Total Cost Breakdown

| Phase | Task | Ollama (FREE) | Paid AI | Cost |
|-------|------|---------------|---------|------|
| 1 | Research | 100% | 0% | $0 |
| 2 | Implementation | 95% | 5% Codex | $5-10 |
| 3 | Debugging | 90% | 10% Codex | $10-20 |
| 4 | Monitoring | 100% | 0% | $0 |
| 5 | Evaluation | 95% | 5% Opus | $5 |
| 6 | Integration | 80% | 20% Codex | $10-20 |
| **TOTAL** | **All 3 implementations** | **93%** | **7%** | **$30-55** |

**vs doing it all with Codex/Opus: $500-1000**

**Savings: 94-97%**

## Detailed Role Assignments

### Ollama (FREE) - The Workhorse

**ALWAYS use Ollama for:**
- Reading/understanding existing code
- Generating boilerplate
- Writing standard training loops
- Monitoring logs and metrics
- Running tests and fixing simple bugs
- Data processing/ETL
- Documentation generation
- Refactoring existing code
- Implementing well-defined patterns
- First 3 debugging attempts

**Quality:** 85-90% as good as Codex
**Speed:** Faster (local)
**Cost:** $0

### Codex (Paid) - The Expert Debugger

**ONLY use Codex when:**
- Ollama tried 3 times and failed
- Complex architectural bugs
- Subtle numerical issues
- Performance optimization
- Writing NEW model architectures from scratch
- Final code review before production
- Security-critical code

**Quality:** 95-100%
**Cost:** $$ (but rare - 5% of work)

### Opus (Paid) - The Manager

**ONLY use Opus for:**
- Talking to you (user interaction)
- Making final decisions
- Synthesizing research
- High-level architecture choices
- Explaining trade-offs
- Project management

**Quality:** 100%
**Cost:** $$$ (but minimal - 5% of work)

## Implementation Example: train_sparse_keyframe.py

### Step 1: Read the Paper (FREE)
```bash
# Ollama reads paper and summarizes
ollama run qwen2.5-coder:14b "Summarize the sparse keyframe prosody paper. Focus on: 1) Architecture, 2) Loss functions, 3) Training procedure"
```

### Step 2: Generate Initial Code (FREE)
```bash
# Ollama generates boilerplate from V7 baseline
ollama run qwen2.5-coder:14b "Based on train_prosody_conditioned.py, generate train_sparse_keyframe.py with these modifications: [paper summary]"
```

### Step 3: Ollama Implements (FREE)
```python
# Ollama writes:
class SparseKeyframeEncoder(nn.Module):
    def __init__(self):
        # Generated by Ollama
        pass

    def extract_keyframes(self, audio, keyframe_indices):
        # Generated by Ollama
        pass
```

### Step 4: Ollama Tests (FREE)
```bash
# Ollama runs tests
pytest training/tests/test_sparse_keyframe.py

# Ollama fixes errors
# Retry 1, 2, 3...
```

### Step 5: Codex Reviews (ONLY if needed - $$)
```bash
# IF Ollama can't fix after 3 tries
# OR implementation is complete and needs final review

# Use Codex for:
codex review training/train_sparse_keyframe.py
# "Check for: architectural bugs, numerical stability, paper accuracy"
```

**Codex cost:** $5-10 (one-time)

### Step 6: Ollama Monitors Training (FREE)
```bash
# Runs on RTX 4090, monitored by FREE Ollama
python scripts/ollama_training_monitor.py \
    --log training/outputs/sparse_keyframe.log \
    --check-every 5m
```

**Cost:** $0

### Step 7: Opus Makes Decision ($$)
```bash
# After all 3 approaches trained
# Ollama generates comparison report (FREE)
# Opus reads report and decides ($$)

# Opus: "Use sparse_keyframe because..."
```

**Opus cost:** $5

## Monthly Cost Projection

### Old Way (All Codex/Opus)
- Research: $200
- Implementation: $300
- Debugging: $200
- Monitoring: $100
- Evaluation: $50
- Integration: $150
**Total: $1000/month**

### New Way (Hybrid)
- Research: $0 (Ollama)
- Implementation: $30 (mostly Ollama, Codex review)
- Debugging: $20 (Ollama first, Codex if stuck)
- Monitoring: $0 (Ollama)
- Evaluation: $5 (Ollama + Opus synthesis)
- Integration: $20 (mostly Ollama, Codex review)
**Total: $75/month**

**Savings: $925/month (93%)**

## Quality Comparison

| Task | Ollama Quality | Codex Quality | Cost Diff |
|------|----------------|---------------|-----------|
| Boilerplate code | 95/100 | 98/100 | FREE vs $50 |
| Training loop | 90/100 | 95/100 | FREE vs $30 |
| Bug fixing (simple) | 85/100 | 95/100 | FREE vs $20 |
| Bug fixing (complex) | 60/100 | 95/100 | FREE vs $50 |
| Code review | 80/100 | 98/100 | FREE vs $40 |
| Monitoring | 95/100 | 98/100 | FREE vs $100 |
| Documentation | 90/100 | 95/100 | FREE vs $20 |

**Key insight:** For most tasks, Ollama is "good enough" and FREE!

## Decision Tree: Which AI to Use?

```
Task arrives
    ├─ Is it user interaction?
    │   └─ YES → Use Opus ($$$)
    │
    ├─ Is it complex debugging after 3 Ollama attempts?
    │   └─ YES → Use Codex ($$)
    │
    ├─ Is it writing a brand NEW model architecture?
    │   └─ YES → Use Codex ($$)
    │
    ├─ Is it production-critical code review?
    │   └─ YES → Use Codex ($$)
    │
    └─ Everything else
        └─ Use Ollama (FREE)
```

## Implementation Scripts

### 1. FREE Training Monitor
```bash
# scripts/ollama_monitor.sh
while true; do
    tail -100 training.log | \
    ollama run qwen2.5-coder:14b \
    "Check training health. Alert if: NaN loss, stuck loss, OOM, overfitting"

    sleep 300  # Check every 5 mins
done
```

### 2. FREE Code Generator
```bash
# scripts/ollama_implement.sh
ollama run qwen2.5-coder:14b "$(cat <<EOF
Generate training/train_sparse_keyframe.py based on:
- Reference: training/train_prosody_conditioned.py
- Paper: [summary from Ollama]
- Changes needed: [list]
EOF
)"
```

### 3. Escalation to Codex
```bash
# scripts/escalate_to_codex.sh
if [ "$OLLAMA_ATTEMPTS" -ge 3 ]; then
    echo "Ollama failed 3 times, escalating to Codex"
    # Use Codex API here ($)
fi
```

## Next Steps

1. **This Week:** Implement top 3 with Ollama
   - Let Ollama write initial code (FREE)
   - Use Codex only for final review ($10-20)

2. **Next Week:** Training on RTX 4090
   - Ollama monitors all 3 runs (FREE)
   - Opus makes final decision ($5)

3. **Following Week:** Integration
   - Ollama does refactoring (FREE)
   - Codex reviews production code ($10-20)

**Total cost for all 3 implementations: $30-55**
**vs $500-1000 with all Codex/Opus**

**Savings: 94-97%**
