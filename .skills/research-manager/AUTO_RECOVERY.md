# Auto-Recovery Architecture for Research System

## Problem Statement

The current system has three critical failure modes:
1. WSL SSH daemon stops frequently (requires manual intervention)
2. Agents spawn but produce no output (qwen3-coder gets stuck)
3. Task explosion: 293 created, 26 completed (agents create more work than they finish)

## Solution: Multi-Layer Recovery System

### Layer 1: SSH Auto-Recovery (Windows Side)

**Create Windows Task Scheduler job to restart SSH every 5 minutes if down:**

```powershell
# File: C:\Users\doc\restart-ssh.ps1
wsl bash -c "sudo service ssh status || sudo service ssh start"
```

**Task Scheduler Setup:**
- Trigger: Every 5 minutes
- Action: `powershell.exe -File C:\Users\doc\restart-ssh.ps1`
- Run whether user is logged on or not
- Run with highest privileges

### Layer 2: Agent Timeout & Auto-Cleanup

**Problem:** Agents get stuck, consume resources, never complete.

**Solution:** Already implemented in orchestrator:
- ✓ 10-minute idle timeout (kills stuck agents)
- ✓ Cleanup runs every 30 minutes (orphaned processes)

**Enhancement needed:** Reduce spawn rate when completion rate is low.

### Layer 3: Task Creation Throttling

**Problem:** 293 tasks created vs 26 completed = unsustainable backlog.

**Solution:** Add task creation rate limiting:
- Max 5 new tasks per agent
- Only allow TaskCreate if completion rate > 10%
- Pause research agents when backlog > 200 tasks

### Layer 4: Completion Rate Monitoring

**Add to orchestrator:** Track and log completion metrics:
```javascript
const metrics = {
  spawns: totalSpawns,
  completions: completedTasks.length,
  completionRate: completedTasks.length / totalSpawns,
  taskBacklog: pendingTasks.length
};

// If completion rate < 0.1 (10%), switch to FOCUS MODE
if (metrics.completionRate < 0.1 && metrics.taskBacklog > 100) {
  log('warn', 'Low completion rate - entering FOCUS MODE');
  // Only work on top 3 priority tasks
}
```

### Layer 5: Fallback to Paid API for Critical Tasks

When local agents are unproductive:
- Detect: 10+ failed attempts on same task
- Action: Flag task for human review or call paid Codex
- Cost control: Max $5/day for critical tasks

## Implementation Priority

1. **SSH Auto-Restart** (15 min) - Fixes immediate access issue
2. **Completion Rate Monitor** (30 min) - Add metrics to orchestrator
3. **Task Throttling** (1 hour) - Prevent backlog explosion
4. **Focus Mode Enhancement** (1 hour) - Work on fewer tasks when stuck

## Success Metrics

- SSH uptime: > 95% (currently ~30%)
- Completion rate: > 25% (currently 8.9%)
- Task backlog: < 150 (currently 267)
- Agent churn: < 5 spawns per completion (currently infinite)
