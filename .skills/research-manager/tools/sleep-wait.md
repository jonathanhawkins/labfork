---
name: sleep-wait
description: Put the manager to sleep or wait for conditions
metadata:
  tags: sleep, wait, pause, block, condition
---

# Sleep and Wait

Mechanisms for the Research Manager to pause and wait for conditions.

## Sleep Command

Sleep for a specific duration, checking reminders periodically.

```bash
python .skills/research-manager/manager.py sleep --seconds 300
```

### What Happens During Sleep

1. Manager enters sleep state
2. Every 5 seconds, checks for triggered reminders
3. Triggered reminders are printed
4. After duration completes, prints "Woke up!"

### Use Cases

- Wait for agents to make progress
- Allow time for external processes
- Pause between workflow phases

## Wait Commands

Wait for a specific condition to be met.

### Wait for Agent Completion

Block until an agent's tmux session ends:

```bash
python .skills/research-manager/manager.py wait \
  --agent "analysis-agent" \
  --timeout 3600
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--agent`, `-a` | - | Agent name to wait for |
| `--timeout`, `-t` | 3600 | Max seconds to wait |

### Wait for File

Block until a file exists:

```bash
python .skills/research-manager/manager.py wait \
  --file "output/results.json" \
  --timeout 1800
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--file`, `-f` | - | File path to wait for |
| `--timeout`, `-t` | 3600 | Max seconds to wait |

## Example Workflows

### Sequential Agent Workflow

```bash
# Start first agent
python .skills/research-manager/manager.py spawn \
  --type codex --name "phase1" --task "Analyze the problem"

# Wait for it to complete
python .skills/research-manager/manager.py wait --agent "phase1"

# Read results
python .skills/research-manager/manager.py read --name "phase1"

# Start second agent based on results
python .skills/research-manager/manager.py spawn \
  --type ollama --name "phase2" --task "Implement based on phase1 findings"
```

### Parallel Agents with Check-in

```bash
# Start multiple agents
python .skills/research-manager/manager.py spawn \
  --type codex --name "research-a" --task "Research approach A"
python .skills/research-manager/manager.py spawn \
  --type ollama --name "research-b" --task "Research approach B"

# Set reminder and sleep
python .skills/research-manager/manager.py remind \
  --in 5m --message "Check agent progress"
python .skills/research-manager/manager.py sleep --seconds 300

# Review both
python .skills/research-manager/manager.py read --name "research-a"
python .skills/research-manager/manager.py read --name "research-b"
```

### File-based Coordination

```bash
# Start agent that produces output file
python .skills/research-manager/manager.py spawn \
  --type ollama --name "generator" \
  --task "Generate results and write to output/results.json"

# Wait for the file
python .skills/research-manager/manager.py wait --file "output/results.json"

# Process the results
echo "Results ready!"
```

## Timeout Behavior

- If timeout is reached before condition is met, the command returns
- A message indicates whether condition was met or timed out
- Return code: 0 if condition met, non-zero if timed out
