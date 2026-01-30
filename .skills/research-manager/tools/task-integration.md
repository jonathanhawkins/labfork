---
name: task-integration
description: Using the task system effectively with Research Manager
metadata:
  tags: task, tracking, workflow, dependencies, progress
---

# Task System Integration

The Research Manager MUST use the task system to track all work. This is not optional - it's core to the workflow.

## Why Tasks Matter

1. **Visibility**: See what's being worked on and what's complete
2. **Coordination**: Multiple agents can work on related tasks
3. **Dependencies**: Block tasks until prerequisites are done
4. **History**: Record findings and decisions persistently
5. **Recovery**: If an agent fails, task state helps resume

## Task Tools (Claude Built-ins)

These are Claude's native tools, NOT bash commands:

### TaskCreate
```
TaskCreate:
  subject: "Brief title (imperative form)"
  description: "Detailed description with context and acceptance criteria"
  activeForm: "Present continuous (shown while in_progress)"
```

### TaskUpdate
```
TaskUpdate:
  taskId: "1"
  status: "in_progress" | "completed" | "pending"
  description: "Updated description with findings"
  addBlockedBy: ["2", "3"]  # This task waits for 2 and 3
  addBlocks: ["4"]          # Task 4 waits for this one
  metadata: {"agent_name": "prosody-analysis"}
```

### TaskList
```
TaskList  # Returns all tasks with status, owner, blockedBy
```

### TaskGet
```
TaskGet:
  taskId: "1"  # Returns full task details
```

## Standard Patterns

### Pattern 1: Research Project

```
# 1. Create master task
TaskCreate:
  subject: "[PROSODY] Research conditioning improvements"
  description: "Investigate and recommend improvements to prosody conditioning"
  activeForm: "Researching prosody improvements"
# Returns: Task ID 1

# 2. Create analysis sub-task
TaskCreate:
  subject: "[PROSODY] Analyze current implementation"
  description: "Review train_prosody_conditioned.py and prosody_conditioning.py"
  activeForm: "Analyzing prosody code"
# Returns: Task ID 2

# 3. Create research sub-task
TaskCreate:
  subject: "[PROSODY] Research alternative approaches"
  description: "Find papers and repos with better prosody methods"
  activeForm: "Researching alternatives"
# Returns: Task ID 3

# 4. Create implementation task (blocked)
TaskCreate:
  subject: "[PROSODY] Implement recommended changes"
  description: "Apply improvements based on research findings"
  activeForm: "Implementing prosody changes"
# Returns: Task ID 4

# 5. Set dependencies
TaskUpdate:
  taskId: "4"
  addBlockedBy: ["2", "3"]  # Wait for analysis and research

# 6. Start work on task 2
TaskUpdate:
  taskId: "2"
  status: "in_progress"
  metadata: {"agent_name": "prosody-analyst"}

# 7. Spawn agent
.skills/research-manager/rm spawn --type codex --name "prosody-analyst" \
  --task "Task #2: Analyze train_prosody_conditioned.py..."

# 8. When agent completes, update task
TaskUpdate:
  taskId: "2"
  status: "completed"
  description: |
    Original: Review train_prosody_conditioned.py

    FINDINGS:
    - LoRA not targeting temporal encoder
    - Validation is a stub
    - Prosody cache keyed by index (risk of mismatch)

    RECOMMENDATIONS:
    - Add temporal encoder to LoRA targets
    - Implement proper validation loop
```

### Pattern 2: Bug Investigation

```
# 1. Create investigation task
TaskCreate:
  subject: "[BUG] Training loss spikes at epoch 10"
  description: "Investigate why loss suddenly increases around epoch 10"
  activeForm: "Investigating loss spike"

# 2. Mark in progress and spawn
TaskUpdate: taskId="1", status="in_progress"
.skills/research-manager/rm spawn --type codex --name "loss-debug" \
  --task "Task #1: Investigate loss spike..."

# 3. Based on findings, create fix task
TaskCreate:
  subject: "[BUG] Fix learning rate schedule"
  description: "Adjust LR decay to prevent epoch 10 spike"
  activeForm: "Fixing LR schedule"

TaskUpdate:
  taskId: "2"
  addBlockedBy: ["1"]  # Wait for investigation
```

### Pattern 3: Parallel Research

```
# Create parallel tasks (no blockedBy)
TaskCreate: subject="[RESEARCH] Approach A - attention pooling"
TaskCreate: subject="[RESEARCH] Approach B - cross-attention"
TaskCreate: subject="[RESEARCH] Approach C - concatenation"

# Spawn agents in parallel
.skills/research-manager/rm spawn --type codex --name "approach-a" --task "..."
.skills/research-manager/rm spawn --type codex --name "approach-b" --task "..."
.skills/research-manager/rm spawn --type ollama --name "approach-c" --task "..."

# Create synthesis task blocked by all three
TaskCreate: subject="[RESEARCH] Compare approaches and recommend"
TaskUpdate: taskId="4", addBlockedBy=["1", "2", "3"]
```

## Agent-Task Linking

Always include task ID in agent prompts:

```bash
.skills/research-manager/rm spawn --type codex --name "task-2-analysis" \
  --task "You are working on Task #2: Analyze current implementation.

Context: [paste task description]

When complete, summarize findings for task update.
Focus on actionable insights."
```

And record agent in task metadata:

```
TaskUpdate:
  taskId: "2"
  metadata: {
    "agent_name": "task-2-analysis",
    "agent_type": "codex",
    "output_file": ".skills/research-manager/state/outputs/task-2-analysis.log"
  }
```

## Completion Checklist

Before marking a task complete:

1. ✅ Agent finished (session ended or output indicates done)
2. ✅ Read full agent output
3. ✅ Extract key findings
4. ✅ Update task description with findings
5. ✅ Check if any blocked tasks can now proceed
6. ✅ Kill the agent to clean up

```
# Complete workflow for finishing a task
.skills/research-manager/rm read --name "my-agent" --tail 0  # Read all output
TaskUpdate: taskId="2", status="completed", description="... findings ..."
.skills/research-manager/rm kill --name "my-agent"
TaskList  # Check what's now unblocked
```

## Task Status Meanings

| Status | Meaning |
|--------|---------|
| `pending` | Not started, waiting or available |
| `in_progress` | Being actively worked on |
| `completed` | Done, findings recorded |

## Best Practices

1. **Always create task before spawning agent** - Don't spawn orphan agents
2. **Include task ID in agent prompt** - Agent knows what it's working on
3. **Record agent name in task metadata** - Easy to find agent for task
4. **Update description with findings** - Don't lose agent output
5. **Use blockedBy for dependencies** - Prevents premature work
6. **Check TaskList after completing** - See what's unblocked
7. **Keep tasks focused** - One clear objective per task
