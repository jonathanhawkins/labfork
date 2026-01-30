---
name: send-message
description: Send messages to running agents
metadata:
  tags: agent, send, message, interact, communicate
---

# Send Message to Agent

Send text or commands to a running agent via its tmux session.

## Usage

```bash
python .skills/research-manager/manager.py send \
  --name "agent-name" \
  --message "Your message here"
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--name`, `-n` | Yes | Agent name to send to |
| `--message`, `-m` | Yes | Text to send (Enter pressed after) |

## Examples

```bash
# Provide guidance to an agent
python .skills/research-manager/manager.py send \
  --name "impl-agent" \
  --message "Focus on the error handling in the prosody module first"

# Ask agent to clarify something
python .skills/research-manager/manager.py send \
  --name "analysis-agent" \
  --message "What are the memory implications of your proposed changes?"

# Send a follow-up task
python .skills/research-manager/manager.py send \
  --name "research-agent" \
  --message "Now investigate how this integrates with the existing LoRA training"
```

## Use Cases

1. **Provide feedback**: Steer the agent based on intermediate results
2. **Ask questions**: Get clarification on agent's findings
3. **Add context**: Provide information the agent might need
4. **Change direction**: Refocus the agent on different aspects
5. **Request summary**: Ask agent to summarize findings

## Important Notes

- The agent must be running (status = "running")
- Message is sent exactly as typed, followed by Enter
- Agent may take time to respond
- Check output with `read` command after sending

## Interactive Session

For real-time interaction, attach directly:

```bash
tmux attach -t rm-agent-name
```

This gives you a live terminal with the agent.
