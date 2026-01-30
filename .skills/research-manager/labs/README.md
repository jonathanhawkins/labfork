# Research Manager Labs

This directory contains lab configurations for the multi-lab research manager system.

## Quick Start

```bash
# List all labs
./rm lab list

# Create a new lab
./rm lab create my-research --name "My Research Lab" --description "Description here"

# Switch to a lab
./rm lab switch my-research

# Check status
./rm lab status
```

## Current Labs

| Lab ID | Name | Task List | Domain |
|--------|------|-----------|--------|
| voice-clone | Voice Clone Research | TASKS-ALIGNED | voice-clone |
| firefly-network | The Firefly Network | firefly-network | firefly-network |

## Directory Structure

```
labs/
  _schema.json              # JSON Schema for lab configs
  README.md                 # This file
  voice-clone/
    config.json             # Lab configuration
    state/                  # Lab-specific state
      agents.json           # Running agents
      proposals.json        # Research proposals
      progress.json         # Task progress history
      focus.json            # Focus mode state
      research-state.json   # Research session state
      outputs/              # Agent output logs
  firefly-network/
    config.json
    state/
      ...
```

## Lab Commands

### Create a Lab
```bash
./rm lab create <id> --name "Lab Name" --description "What this lab does" [--domain <slug>]
```

### Switch Active Lab
```bash
./rm lab switch <id>
```

### List Labs
```bash
./rm lab list
```

### Check Lab Status
```bash
./rm lab status              # Current lab
./rm lab status <id>         # Specific lab
```

### Get Lab Info
```bash
./rm lab info <id>           # Full config as JSON
```

### Delete a Lab
```bash
./rm lab delete <id>         # Prompts for confirmation
./rm lab delete <id> --force # Skips confirmation
```

### Migrate Global State
```bash
./rm lab migrate <id>        # Migrate global state to lab
./rm lab migrate <id> --overwrite  # Overwrite existing files
```

## Lab Override

You can temporarily override the active lab for any command:

```bash
# Using --lab flag
./rm --lab firefly-network status
./rm --lab voice-clone spawn research "Task here"

# Using environment variable
export CLAUDE_CODE_LAB_ID=firefly-network
./rm status

# Orchestrator respects lab context
./rm --lab firefly-network orchestrator start
```

## Configuration

Each lab has a `config.json` with:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (lowercase, hyphens only) |
| `name` | string | Display name |
| `description` | string | What this lab researches |
| `taskListId` | string | Claude Code task list ID |
| `active` | boolean | Whether the lab is active |
| `domain` | string | Optional link to `.domains/<slug>/` |
| `settings.maxAgents` | number | Max concurrent agents (default: 3) |
| `settings.autoSpawn` | boolean | Auto-spawn agents (default: true) |
| `settings.researchInterval` | number | Minutes between research (default: 30) |

## State Files

Each lab's `state/` directory contains:

| File | Description |
|------|-------------|
| `agents.json` | Currently registered agents and their status |
| `proposals.json` | Research proposals awaiting review |
| `progress.json` | Task completion history |
| `focus.json` | Focus mode state (halts other work) |
| `research-state.json` | Current research session state |
| `reminders.json` | Scheduled reminders |
| `outputs/` | Agent output log files |

## Integration with Domains

Labs can optionally link to a `.domains/<slug>/` directory for:

- Research configuration (arXiv categories, keywords)
- 3D scene configuration for the web frontend
- Evaluation metrics and baselines
- Prompt templates for agents

Set the `domain` field in config.json to the domain slug.

## Priority Resolution

The active lab is determined in this order:

1. `--lab` flag on command line
2. `CLAUDE_CODE_LAB_ID` environment variable
3. Active lab from `state/active-lab.json`
4. Default: `voice-clone`

## API for Python Scripts

```python
from labs import (
    get_effective_lab_id,
    get_effective_lab,
    get_task_list_id,
    set_active_lab,
    list_labs,
    create_lab,
)

# Get current lab context
lab_id = get_effective_lab_id()
lab_config = get_effective_lab()
task_list = get_task_list_id()

# Switch labs programmatically
set_active_lab("firefly-network")

# Create a new lab
create_lab(
    lab_id="new-lab",
    name="New Research Lab",
    description="Exploring new frontiers",
    domain="new-domain",
    settings={"maxAgents": 5}
)
```
