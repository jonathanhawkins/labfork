# System Health Report - 2026-01-27

## Critical Issues Found

### 1. WSL SSH Daemon Failure ⛔ CRITICAL
**Status:** DOWN (3rd failure today)
**Impact:** Cannot access 4090, all research stopped
**Root Cause:** Windows WSL SSH daemon stops/crashes frequently

**Solution Implemented:**
- Created PowerShell watchdog script (`scripts/windows/restart-ssh.ps1`)
- Setup script for Windows Task Scheduler (`scripts/windows/setup-ssh-watchdog.ps1`)
- Will auto-restart SSH every 5 minutes if down

**Action Required:**
1. Via Parsec, open PowerShell as Administrator
2. Navigate to: `\\wsl.localhost\Ubuntu\home\doc\dev\labfork\scripts\windows\`
3. Run: `.\setup-ssh-watchdog.ps1`
4. This will create a Task Scheduler job to keep SSH alive

---

### 2. Agent Productivity Crisis ⚠️ HIGH
**Status:** Severe
**Metrics:**
- Total spawns: 293+
- Completions: 26
- **Completion rate: 8.9%** (should be >25%)
- Task backlog: 267 pending

**Root Cause:**
- Agents create tasks faster than they complete them
- Local qwen3-coder gets stuck on complex multi-step tasks
- 10-minute timeout kills agents before they produce output

**Solutions Implemented:**
1. **Productivity monitoring** - Orchestrator now tracks completion rate every 5 min
2. **Duplicate prevention** - Agents must check for dupes before creating tasks
3. **Dedup tool** - `./rm orchestrator dedup --auto` to clean up existing duplicates
4. **Auto-recovery doc** - `.skills/research-manager/AUTO_RECOVERY.md` with full strategy

**Recommendations:**
- Run dedup tool once SSH is back up
- Consider increasing timeout from 10min to 20min for complex tasks
- May need to switch critical tasks to paid Codex API

---

### 3. Windows Sleep/Power Management ⚠️ MEDIUM
**Status:** FIXED
**What Happened:** Machine went to sleep last night, killed all processes
**Fix Applied:** Disabled sleep/hibernate on AC power via `powercfg`

---

## System Architecture Issues

### Current Problems:
1. **SSH is single point of failure** - When SSH dies, entire system is unreachable
2. **No connection resilience** - Orchestrator can't recover from SSH loss
3. **Agent churn vs productivity** - Spawning agents that don't complete work wastes resources
4. **Task explosion** - Research agents create more work than they finish

### Proposed Enhancements:

#### Short-term (1-2 hours):
- ✅ SSH watchdog (implemented, needs Windows setup)
- ✅ Productivity tracking (implemented in orchestrator)
- ✅ Duplicate task prevention (implemented)
- ⏳ Run dedup cleanup (needs SSH access)

#### Medium-term (1 week):
- Add HTTP API fallback for when SSH is down
- Increase agent timeout to 20 minutes
- Add "focus mode" - work on top 3 tasks only when backlog > 200
- Throttle research agents when completion rate < 10%

#### Long-term (1 month):
- Hybrid architecture: Mac orchestrator + 4090 workers
- Paid API fallback for tasks with >5 retries
- Better agent success detection (file modifications, not just time)

---

## Files Created/Modified

### New Tools:
1. `scripts/windows/restart-ssh.ps1` - SSH watchdog
2. `scripts/windows/setup-ssh-watchdog.ps1` - One-time setup
3. `.skills/research-manager/dedup-tasks.js` - Remove duplicate tasks
4. `.skills/research-manager/check-duplicate.js` - Check before creating
5. `.skills/research-manager/AUTO_RECOVERY.md` - Full recovery strategy
6. `docs/SYSTEM_HEALTH_REPORT.md` - This report

### Enhanced:
1. `orchestrator.js` - Added productivity tracking & warnings
2. `orchestrator.js` - Added duplicate filtering at assignment
3. `orchestrator.js` - Updated research agent prompt to check dupes
4. `rm` script - Added `dedup` command

---

## Next Steps

### Immediate (now):
1. **Fix SSH** - Run setup-ssh-watchdog.ps1 via Parsec
2. **Clean duplicates** - `ssh doc@$REMOTE_GPU_HOST "cd ~/dev/labfork && ./rm orchestrator dedup --auto"`
3. **Restart orchestrator** - Will pick up new productivity tracking

### Short-term (this week):
1. Monitor completion rate logs
2. If still < 10%, increase timeout or reduce concurrent agents
3. Consider paid API for high-priority tasks

### Long-term (next month):
1. Evaluate if local Ollama is worth the complexity
2. May be better to use 4090 for training only
3. Run research on Mac with paid Claude (higher quality, fewer issues)

---

## Cost Analysis

### Current Setup (Free Local):
- **Cost:** $0/month
- **Reliability:** ~30% uptime (SSH failures)
- **Productivity:** 8.9% completion rate
- **Maintenance:** High (constant troubleshooting)

### Alternative (Paid API):
- **Cost:** ~$50-100/month for research
- **Reliability:** 99.9% uptime
- **Productivity:** ~60% completion rate (estimated)
- **Maintenance:** Low (just works)

### Recommendation:
**Hybrid approach** - Use free local for exploration/research, switch to paid API when a task needs to actually ship code.
