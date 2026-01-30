---
name: monitor-agents
description: Check status and read output from running agents
metadata:
  tags: agent, status, read, monitor, output
---

# Monitor Agents

Commands to check agent status and read their output.

## Status Command

Get the status of all agents or a specific agent.

```bash
# List all agents
python .skills/research-manager/manager.py status

# Get specific agent status
python .skills/research-manager/manager.py status --name "arch-analysis"

# Output as JSON
python .skills/research-manager/manager.py status --json
```

### Status Values

| Status | Meaning |
|--------|---------|
| `running` | Tmux session exists, agent active |
| `stopped` | Tmux session ended, agent finished |
| `killed` | Agent was manually terminated |

## Read Command

Read output from an agent's log file.

```bash
# Read last 100 lines (default)
python .skills/research-manager/manager.py read --name "arch-analysis"

# Read last 50 lines
python .skills/research-manager/manager.py read --name "arch-analysis" --tail 50

# Read entire log
python .skills/research-manager/manager.py read --name "arch-analysis" --tail 0
```

## Sessions Command

List all tmux sessions (managed and unmanaged).

```bash
python .skills/research-manager/manager.py sessions
```

Output shows:
- `rm-*` sessions are managed by Research Manager
- Other sessions are independent

## Direct Tmux Access

For interactive control, attach directly to a session:

```bash
# Attach to agent session
tmux attach -t rm-arch-analysis

# Detach: Ctrl+B, then D

# List all sessions
tmux list-sessions
```

## Output File Location

All agent output is logged to:
```
.skills/research-manager/state/outputs/<agent-name>.log
```

You can also read these directly:
```bash
tail -f .skills/research-manager/state/outputs/arch-analysis.log
```

## Checking Completion

An agent is considered complete when:
1. Its tmux session no longer exists
2. Its status shows as "stopped"

Check completion:
```bash
# Quick check
python .skills/research-manager/manager.py status --name "arch-analysis"

# Or use wait command (blocks until complete)
python .skills/research-manager/manager.py wait --agent "arch-analysis"
```
