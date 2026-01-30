# Hybrid Lab System

A cost-effective approach where FREE local AI handles orchestration and execution, while paid Codex provides complex analysis when needed.

## The Problem

The FREE Ollama-based lab-manager (qwen3-coder-32k) struggles with:
- Deep code analysis of 500+ line implementations
- Multi-step planning for complex tasks
- Understanding patterns across multiple files

But it excels at:
- File operations (read, write, edit)
- Running bash commands
- Following step-by-step instructions
- Task tracking and status updates

## The Solution: Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID WORKFLOW                              │
│                                                                  │
│   ┌──────────────┐         ┌──────────────┐                     │
│   │ LAB-MANAGER  │  call   │    CODEX     │                     │
│   │ (FREE)       │ ──────▶ │   (PAID)     │                     │
│   │ Ollama       │  ◀───── │   CLI        │                     │
│   └──────┬───────┘  plan   └──────────────┘                     │
│          │                                                       │
│          │ execute                                               │
│          ▼                                                       │
│   ┌──────────────┐                                              │
│   │  FILE OPS    │  write_file, edit_file, bash                 │
│   │  (FREE)      │                                              │
│   └──────────────┘                                              │
│                                                                  │
│   Cost: Codex thinking (~$1) + FREE execution = ~90% savings   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. call-codex Script

Sends context to paid Codex API and receives a JSON execution plan.

```bash
# Usage
call-codex --files "file1.py,file2.py" "Task description"

# Output: JSON plan with steps
```

### 2. execute-plan Script

Executes a JSON plan step by step.

```bash
# Dry run first
execute-plan plan.json --dry-run

# Execute all steps
execute-plan plan.json

# Execute specific step
execute-plan plan.json --step 3
```

### 3. Lab-Manager Instructions

The lab-manager uses these tools in a workflow:

1. Read task with TaskGet
2. Assess complexity
3. For complex tasks: `call-codex`
4. Execute returned plan
5. Verify and complete task

## JSON Plan Format

```json
{
  "task": "Brief title",
  "analysis": "What needs to be done and why",
  "steps": [
    {
      "id": 1,
      "type": "write_file",
      "path": "relative/path.py",
      "description": "What this creates",
      "content": "#!/usr/bin/env python3\n..."
    },
    {
      "id": 2,
      "type": "bash",
      "command": "python script.py --help",
      "description": "Verify script runs"
    },
    {
      "id": 3,
      "type": "edit_file",
      "path": "existing/file.py",
      "find": "old text",
      "replace": "new text",
      "description": "Update import"
    }
  ],
  "verification": "How to verify completion"
}
```

## Step Types

| Type | Fields | Purpose |
|------|--------|---------|
| `write_file` | path, content | Create new file |
| `edit_file` | path, find, replace | Modify existing file |
| `bash` | command | Run shell command |
| `read_file` | path | Read and display file |

## Cost Comparison

| Task Type | All Paid | Hybrid | Savings |
|-----------|----------|--------|---------|
| Simple file edit | $2 | $0 | 100% |
| Create training script | $50 | $2 | 96% |
| Multi-file refactor | $100 | $5 | 95% |
| Architecture analysis | $30 | $3 | 90% |

For 7 pending tasks:
- All Paid: ~$400-500
- Hybrid: ~$20-30
- **Savings: ~$380-470 (95%)**

## Setup on RTX 4090

```bash
# SSH to machine
ssh doc@$REMOTE_GPU_HOST

# Run setup
cd ~/dev/labfork
./scripts/setup-4090-hybrid.sh

# Ensure Codex CLI is installed and authenticated
# codex --version

# Start hybrid lab
~/bin/start-hybrid-lab
```

## Example Session

```
# Attach to lab-manager
ssh doc@$REMOTE_GPU_HOST -t "tmux attach -t lab-manager"

# In lab-manager, user asks: "Work on task #36"

# Lab-manager response:
1. Let me get the task details...
   TaskGet 36

2. This is DDGAN prosody diffusion - complex task.
   I'll call Codex for analysis.

3. ~/bin/call-codex --files "training/ddgan_prosody.py,training/train_prosody_hed.py" \
     "Create train_ddgan_prosody.py following existing patterns"

4. Plan received. Executing...
   ~/bin/execute-plan ~/.codex-plans/plan_*.json

5. Verifying...
   python training/train_ddgan_prosody.py --help

6. Task complete!
   TaskUpdate 36 status=completed
```

## When Lab-Manager Should Call Codex

**DO call Codex for:**
- Creating new 100+ line implementations
- Understanding complex existing code
- Multi-file coordinated changes
- Architecture decisions

**DON'T call Codex for:**
- Reading files (use Read tool)
- Simple single-line edits
- Running existing scripts
- Git operations
- Task list management

## Troubleshooting

### Codex call returns error
```bash
# Check Codex CLI availability
which codex
codex --version

# If codex isn't on PATH, set CODEX_PATH
export CODEX_PATH="$HOME/.nvm/versions/node/<version>/bin/codex"
```

### Plan execution fails at step
```bash
# Run specific step
execute-plan plan.json --step 3

# Or manually execute the step
# Read the plan and do it yourself
```

### Lab-manager unresponsive
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Restart
tmux kill-session -t lab-manager
~/bin/lab-manager
```

## Files Created

| File | Purpose |
|------|---------|
| `scripts/call-codex` | Call paid Codex CLI for analysis |
| `scripts/execute-plan` | Execute JSON plans |
| `scripts/setup-4090-hybrid.sh` | Setup script for 4090 |
| `.claude/commands/call-codex.md` | Skill documentation |
| `.skills/call-codex/SKILL.md` | Skill metadata |
| `.codex-plans/` | Directory for saved plans |

## Integration with Existing System

This builds on the existing lab-manager workflow:
- Uses the same tmux session structure
- Compatible with existing lab-manager
- Adds Codex as "expert advisor" capability
- Plans saved for audit/review
