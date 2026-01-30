---
name: kill-agent
description: Terminate running agents
metadata:
  tags: agent, kill, terminate, stop, cleanup
---

# Kill Agent

Terminate a running agent and its tmux session.

## Usage

```bash
# Kill specific agent
python .skills/research-manager/manager.py kill --name "agent-name"

# Kill ALL managed agents
python .skills/research-manager/manager.py kill
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--name`, `-n` | No | Agent to kill (omit to kill all) |

## What Happens

1. Tmux session is killed (terminates the agent process)
2. Agent status updated to "killed"
3. Timestamp recorded in agent registry
4. Output log preserved for review

## Examples

```bash
# Kill a completed or stuck agent
python .skills/research-manager/manager.py kill --name "old-analysis"

# Kill all agents to start fresh
python .skills/research-manager/manager.py kill

# Verify agent is gone
python .skills/research-manager/manager.py status
```

## When to Kill Agents

1. **Task complete**: Agent finished its work
2. **Wrong direction**: Agent is not producing useful results
3. **Superseded**: A better approach was found
4. **Resource cleanup**: Need to free tmux sessions
5. **Stuck/hung**: Agent is not making progress

## Preserving Work

Before killing, consider:

```bash
# Read final output
python .skills/research-manager/manager.py read --name "agent-name" --tail 0

# Output is preserved in:
# .skills/research-manager/state/outputs/<agent-name>.log
```

## Clearing Registry

To clear killed agents from the registry:

```bash
python .skills/research-manager/manager.py clear --agents
```

This removes entries from `agents.json` but doesn't affect log files.
