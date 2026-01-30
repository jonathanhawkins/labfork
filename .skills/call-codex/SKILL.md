---
name: call-codex
description: Call paid Codex to analyze a task and return an execution plan
metadata:
  tags: codex, paid-api, analysis, planning, hybrid
---

# Call Codex Skill

**Get help from paid Codex API for complex code analysis.**

This skill is designed for the FREE Ollama-based lab-manager. When you encounter a task too complex for local analysis, call Codex to get a detailed execution plan.

## When to Use

Call Codex when you need:
- Deep code analysis (understanding 500+ line implementations)
- Multi-file changes that need coordination
- Creating new implementations following existing patterns
- Architecture decisions

Do NOT call Codex for:
- Reading files (just use Read tool)
- Simple edits
- Running scripts
- Git operations

## Quick Usage

```bash
# Basic call
~/bin/call-codex "Create train_ddgan_prosody.py for task #36"

# With file context (recommended - Codex reads files for you)
~/bin/call-codex --files "training/ddgan_prosody.py,training/train_prosody_hed.py" \
  "Create training script following the pattern of train_prosody_hed.py"
```

## Output

Codex returns a JSON execution plan:

```json
{
  "task": "Create train_ddgan_prosody.py",
  "analysis": "The ddgan_prosody.py has X components...",
  "steps": [
    {"id": 1, "type": "write_file", "path": "...", "content": "..."},
    {"id": 2, "type": "bash", "command": "..."}
  ],
  "verification": "Run python train_ddgan_prosody.py --help"
}
```

## Execute the Plan

After getting the plan:

```bash
# Execute all steps
~/bin/execute-plan ~/.codex-plans/plan_*.json

# Or dry-run first
~/bin/execute-plan ~/.codex-plans/plan_*.json --dry-run

# Or execute specific step
~/bin/execute-plan ~/.codex-plans/plan_*.json --step 2
```

## Cost

- Codex analysis: ~$0.50-$2.00 per call
- Execution by lab-manager: $0.00 (FREE Ollama)
- Total savings: ~80% compared to all-paid approach

## Example Workflow for Pending Tasks

### Task #36: DDGAN Prosody Training Script

```bash
# 1. Get file context
ls training/train_*.py | head -5

# 2. Call Codex
~/bin/call-codex --files "training/ddgan_prosody.py,training/train_prosody_hed.py,training/config/ddgan_prosody.yaml" \
  "Create train_ddgan_prosody.py following the pattern of train_prosody_hed.py. Include:
   - DataLoader setup for mel + text + speaker
   - Alternating generator/discriminator training
   - VQ-VAE pre-training phase
   - Logging to wandb"

# 3. Execute
~/bin/execute-plan ~/.codex-plans/plan_*.json

# 4. Test
cd training && python train_ddgan_prosody.py --help

# 5. Complete task
# Use TaskUpdate to mark #36 complete
```

## Prerequisites

1. Set API key on 4090 machine:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. Or add to ~/.bashrc for persistence:
   ```bash
   echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
   ```
