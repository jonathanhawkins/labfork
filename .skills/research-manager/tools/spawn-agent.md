---
name: spawn-agent
description: Spawn a new AI agent in a tmux session
metadata:
tags: agent, spawn, codex, ollama, tmux
---

# Spawn Agent

Creates a new AI agent running in a tmux session. The agent will work on the given task autonomously.

## Usage

```bash
python .skills/research-manager/manager.py spawn \
  --type <codex|ollama> \
  --name "unique-name" \
  --task "Your task description here"
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--type`, `-t` | Yes | Agent type: `codex` (OpenAI) or `ollama` (local Claude Code) |
| `--name`, `-n` | Yes | Unique identifier for this agent |
| `--task` | Yes | The task/prompt to give the agent |
| `--dir`, `-d` | No | Working directory (defaults to project root) |

## Agent Types

### Codex (OpenAI)
- Best for: Deep analysis, complex reasoning, detailed code review
- CLI: `codex`
- Use when: You need careful, thorough thinking

### Opus (Claude)
- Best for: Implementation, code writing, broad exploration
- CLI: `./clauder` (runs `claude --dangerously-skip-permissions`)
- Use when: You need fast implementation or exploration
- Auto-runs `/rename RM:<name>` for easy identification

## Examples

```bash
# Spawn a Codex agent for deep analysis
python .skills/research-manager/manager.py spawn \
  --type codex \
  --name "arch-analysis" \
  --task "Analyze the prosody pipeline architecture and identify potential bottlenecks"

# Spawn an Opus agent for implementation
python .skills/research-manager/manager.py spawn \
  --type ollama \
  --name "feature-impl" \
  --task "Implement the new prosody conditioning feature based on the analysis"

# Spawn with custom working directory
python .skills/research-manager/manager.py spawn \
  --type ollama \
  --name "frontend-work" \
  --task "Fix the recording UI bug" \
  --dir "./frontend"
```

## Session Management

- Session name format: `rm-<name>` (e.g., `rm-arch-analysis`)
- Attach to session: `tmux attach -t rm-<name>`
- Output logged to: `.skills/research-manager/state/outputs/<name>.log`

## Best Practices

1. **Use descriptive names**: `prosody-analysis`, `lora-debug`, not `agent1`
2. **Be specific in tasks**: Give clear, actionable instructions
3. **Create tasks first**: Use TaskCreate before spawning agents
4. **Monitor output**: Regularly check agent output with the `read` command
5. **Clean up**: Kill completed agents to free tmux sessions
