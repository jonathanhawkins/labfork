---
description: Call paid Codex to analyze a task and return an execution plan
allowed-tools: Bash, Read, Write, Edit
---

# Call Codex for Complex Analysis

When you (the lab-manager running on FREE Ollama) encounter a complex task that requires deep code analysis, use this command to get help from paid Codex.

## How It Works

1. You provide context (task description, relevant files)
2. Codex analyzes and returns a JSON execution plan
3. You execute each step in the plan

## Usage

```bash
# Call Codex with a task description
~/bin/call-codex "Analyze training/ddgan_prosody.py and create train_ddgan_prosody.py"

# Call with file context (recommended)
~/bin/call-codex --files "training/ddgan_prosody.py,training/train_prosody_hed.py" \
  "Create a training script for DDGAN prosody following existing patterns"
```

## Output Format

Codex returns a JSON execution plan:

```json
{
  "task": "Create train_ddgan_prosody.py",
  "analysis": "Brief analysis of what needs to be done",
  "steps": [
    {
      "id": 1,
      "type": "write_file",
      "path": "training/train_ddgan_prosody.py",
      "description": "Create the training script",
      "content": "#!/usr/bin/env python3\n..."
    },
    {
      "id": 2,
      "type": "bash",
      "command": "python training/train_ddgan_prosody.py --help",
      "description": "Verify the script runs"
    },
    {
      "id": 3,
      "type": "edit_file",
      "path": "CLAUDE.md",
      "find": "# Training",
      "replace": "# Training\n\n- train_ddgan_prosody.py - DDGAN accelerated prosody",
      "description": "Update documentation"
    }
  ],
  "verification": "Run test and check output"
}
```

## Executing the Plan

After receiving the plan, execute each step:

```bash
# For write_file steps
# Read the content from the plan and write to the file

# For bash steps
cd ~/dev/labfork && <command>

# For edit_file steps
# Use Edit tool with find/replace
```

## When to Call Codex

Call Codex when you need:
- Deep code analysis (understanding complex implementations)
- Multi-file refactoring plans
- Architecture decisions
- Creating new implementations based on existing patterns

Do NOT call Codex for:
- Simple file reads
- Running existing scripts
- Basic git operations
- Single-file edits you understand

## Cost

Each Codex call costs approximately $0.50-$2.00 depending on context size.
Your FREE Ollama execution of the returned steps costs $0.00.

## Example Session

```
You (lab-manager): I need to create train_ddgan_prosody.py for task #36

1. First, check what files exist:
   ls training/train_*.py | head -5

2. Call Codex:
   ~/bin/call-codex --files "training/ddgan_prosody.py,training/train_prosody_hed.py" \
     "Create train_ddgan_prosody.py following the pattern of train_prosody_hed.py"

3. Execute returned steps...

4. Mark task complete:
   Update task #36 status to completed
```
