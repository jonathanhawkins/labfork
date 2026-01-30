---
name: reminders
description: Create and manage time-based reminders
metadata:
  tags: reminder, timer, schedule, notify, wake
---

# Reminders

Schedule reminders for future actions. Reminders are checked during sleep/wait operations.

## Create Reminder

```bash
# Relative time (from now)
python .skills/research-manager/manager.py remind \
  --message "Check codex agent progress" \
  --in 5m

# Absolute time
python .skills/research-manager/manager.py remind \
  --message "Review daily results" \
  --at "15:00"

# Full datetime
python .skills/research-manager/manager.py remind \
  --message "Follow up on research" \
  --at "2024-01-24 09:00"
```

## Duration Format

| Format | Meaning |
|--------|---------|
| `30s` | 30 seconds |
| `5m` | 5 minutes |
| `2h` | 2 hours |
| `1d` | 1 day |

## List Reminders

```bash
python .skills/research-manager/manager.py reminders
```

Output:
```
#1 [pending] @ 2024-01-23T15:00:00: Check codex agent progress
#2 [TRIGGERED] @ 2024-01-23T14:30:00: Review analysis results
```

## Check Reminders

Manually check and trigger due reminders:

```bash
python .skills/research-manager/manager.py check-reminders
```

This is automatically called during `sleep` and `wait` operations.

## Clear Reminders

```bash
python .skills/research-manager/manager.py clear --reminders
```

## How Reminders Work

1. Reminders are stored in `.skills/research-manager/state/reminders.json`
2. During `sleep` or `wait`, reminders are checked every 5 seconds
3. When trigger time passes, reminder is marked as triggered and printed
4. Triggered reminders remain in the list (for logging) but don't fire again

## Use Cases

1. **Agent check-in**: Set reminder to review agent progress
2. **Scheduled review**: Daily summary review times
3. **Timeout warnings**: Reminder if task takes too long
4. **Follow-up**: Remember to continue work after break

## Example Workflow

```bash
# Start a long-running analysis
python .skills/research-manager/manager.py spawn \
  --type codex --name "deep-analysis" \
  --task "Comprehensive architecture review"

# Set reminder to check in 10 minutes
python .skills/research-manager/manager.py remind \
  --message "Check deep-analysis progress, provide guidance if needed" \
  --in 10m

# Sleep (will wake on reminder)
python .skills/research-manager/manager.py sleep --seconds 600
```
