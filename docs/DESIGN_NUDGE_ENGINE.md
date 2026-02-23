# Nudge Engine: Technical Design

## Think Carmack

Strip everything. What's the actual machine here?

A database with tasks in it. A cron that looks at them every N minutes. Workers that poll for tasks and report results. That's three moving parts. Everything else is configuration.

Our Workers codebase already does this. It's buried under LabFork-specific stuff (papers, voice synthesis, agent personas), but the bones are clean. The job is surgery, not invention.

## The Protocol

Forget SDKs, plugins, frameworks. The product is an HTTP protocol. If your tool can `curl`, it can participate. Three endpoints. That's the entire client contract.

```
POST /register     → Here's who I am. Give me a token.
POST /poll         → Got any work? Here's my heartbeat.
POST /report       → Here's what I did.
```

That's it. Claude Code, Codex, Ollama, a shell script, a phone - anything that can make HTTP requests is a valid worker. The server doesn't care what you are. It cares that you took a task and brought back a result.

### POST /register

```json
// Request
{
  "name": "jonathan-mbp",
  "type": "claude-code",        // claude-code | codex | ollama | custom
  "capabilities": ["code", "research", "test", "review"]
}

// Response
{
  "id": "w_abc123",
  "token": "sk_...",           // Bearer token for all subsequent calls
  "registered": true
}
```

One call. Store the token. Done. No OAuth, no handshakes. A 32-byte random token over HTTPS is sufficient. If it leaks, revoke and re-register. Don't build a key management system.

### POST /poll

```json
// Request (just the auth header, body optional)
Authorization: Bearer sk_...
{
  "status": "idle",            // idle | busy | done
  "completedTaskId": "t_xyz"   // if reporting completion inline
}

// Response — no work
{ "task": null }

// Response — here's work
{
  "task": {
    "id": "t_789",
    "action": "fix-failing-test",
    "description": "pytest tests/test_auth.py::test_login_timeout is failing on main. The test expects a 5s timeout but the handler was changed to 10s. Either fix the handler or update the test.",
    "context": {
      "repo": "github.com/org/app",
      "branch": "main",
      "files": ["src/auth/handler.py", "tests/test_auth.py"],
      "relatedIssue": "#142"
    },
    "constraints": {
      "maxFiles": 3,            // Don't touch more than 3 files
      "mustPassTests": true,    // Run tests before reporting success
      "createPR": true,         // Don't push to main
      "timeoutMinutes": 30
    }
  }
}
```

The `action` field is a short verb-noun. The `description` is a plain English paragraph. The `context` gives the worker enough to start. The `constraints` tell it what NOT to do. This is the manager talking to a junior dev: specific task, clear boundaries, defined done.

### POST /report

```json
// Request
Authorization: Bearer sk_...
{
  "taskId": "t_789",
  "success": true,
  "result": {
    "summary": "Updated test_login_timeout to expect 10s. Root cause: handler timeout was intentionally increased in commit abc123.",
    "filesChanged": ["tests/test_auth.py"],
    "prUrl": "https://github.com/org/app/pull/155",
    "testsPass": true
  },
  "durationSeconds": 180
}
```

Success or failure, a summary, what changed. The manager logs this and moves on.

## The Cron — Where The Magic Is

The cron is a Cloudflare Worker scheduled trigger. Every 15 minutes it runs a function. Not a workflow engine. Not a DAG. A function.

```
scheduled() {
  1. Observe   — What's the current state of the world?
  2. Decide    — What 0-3 tasks should exist that don't?
  3. Nudge     — Are any assigned tasks stuck? Break them down or reassign.
  4. Cleanup   — Mark dead workers offline. Reset orphaned tasks.
}
```

### Step 1: Observe

The cron reads **observers**. An observer is a function that returns structured state. Ship with three:

```typescript
// Git observer — runs `git log`, `git status` via the repo's CI or a registered worker
interface GitObservation {
  recentCommits: { hash: string, message: string, author: string, age: string }[]
  openPRs: { number: number, title: string, age: string, status: string }[]
  failingCI: boolean
  uncommittedChanges: boolean
}

// Issues observer — reads GitHub/Linear/Jira issues
interface IssuesObservation {
  newIssues: { id: string, title: string, labels: string[], age: string }[]
  staleIssues: { id: string, title: string, staleDays: number }[]
}

// Tasks observer — reads own task table
interface TasksObservation {
  pending: number
  inProgress: number
  stuck: { id: string, title: string, assignedMinutesAgo: number }[]
  completed24h: number
  failed24h: number
}
```

Observers don't need to run inside the cron. They can be **cached**. A worker reports git state as part of its heartbeat. The cron just reads the latest snapshot. This is important: the cron itself should complete in < 1 second. It reads state and writes tasks. No network calls to external APIs during the cron.

### Step 2: Decide

Here's where an LLM earns its keep — but only if we have one available. The cron has two modes:

**Heuristic mode (no LLM needed):**
```
if (failingCI && no task exists for "fix CI"):
    create task "fix-ci"
if (newIssues.length > 0 && no task exists for "triage issues"):
    create task "triage-issues"
if (stalePRs.length > 0 && no task exists for "review PRs"):
    create task "review-prs"
if (pending == 0 && inProgress == 0):
    create task "check-project-health"
```

Rule-based. Fast. Free. No API calls. This is the default and it covers 80% of the value.

**LLM mode (optional, expensive):**
Dispatch an `assessment` task to an available worker. "Here's the project state. What are the 1-3 most important things to do next?" The worker runs it through Claude/Codex/Ollama and reports back. The NEXT cron cycle reads the assessment result and creates the suggested tasks. Two cycles to get LLM-informed task creation. This is fine. We're not in a hurry. That's the whole point.

### Step 3: Nudge

```
for each task where status == 'in_progress' AND assigned_minutes_ago > timeout:
    if worker is still alive (recent heartbeat):
        // Worker is slow, not dead. Leave it alone but log a warning.
        log("Task {id} running long: {minutes}min")
    else:
        // Worker died. Reset the task.
        task.status = 'pending'
        task.assigned_worker = null
        log("Task {id} orphaned, reset to pending")
```

For tasks that have been pending too long (nobody picked them up):
```
for each task where status == 'pending' AND created_minutes_ago > 60:
    if task.attempts >= 3:
        task.status = 'failed'
        log("Task {id} failed after 3 attempts, needs human attention")
    else:
        // Bump priority so it gets picked up next
        task.priority += 1
        log("Task {id} bumped priority to {priority}")
```

### Step 4: Cleanup

Same as what we have now. Mark offline workers, clear stale state, delete old completed tasks (keep 7 days). Boring but essential.

## The Database

One D1 database. Five tables. No joins needed in the hot path.

```sql
CREATE TABLE workers (
  id TEXT PRIMARY KEY,                    -- w_abc123
  name TEXT NOT NULL,                     -- jonathan-mbp
  type TEXT NOT NULL,                     -- claude-code | codex | ollama
  capabilities TEXT NOT NULL,             -- JSON ["code","test","review"]
  status TEXT DEFAULT 'online',           -- online | busy | offline
  auth_token TEXT NOT NULL,               -- sk_...
  current_task_id TEXT,
  last_heartbeat TEXT,
  tasks_completed INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tasks (
  id TEXT PRIMARY KEY,                    -- t_abc123
  action TEXT NOT NULL,                   -- fix-failing-test
  description TEXT NOT NULL,              -- Plain English, 1-3 paragraphs
  context TEXT,                           -- JSON: repo, branch, files, etc.
  constraints TEXT,                       -- JSON: maxFiles, mustPassTests, etc.
  status TEXT DEFAULT 'pending',          -- pending | assigned | completed | failed
  priority INTEGER DEFAULT 5,            -- 1-10
  assigned_worker_id TEXT,
  result TEXT,                            -- JSON: summary, filesChanged, etc.
  error TEXT,
  source TEXT,                            -- What created this: "cron", "human", "llm"
  attempts INTEGER DEFAULT 0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  assigned_at TEXT,
  completed_at TEXT
);

CREATE TABLE observations (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,                     -- git | issues | tests | custom
  data TEXT NOT NULL,                     -- JSON snapshot
  observed_by TEXT,                       -- worker_id or "cron"
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE work_log (
  id TEXT PRIMARY KEY,
  worker_id TEXT NOT NULL,
  task_id TEXT,
  action TEXT NOT NULL,                   -- assigned | completed | failed | nudged
  detail TEXT,                            -- JSON
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE config (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

That's it. No agent_state, no personas, no compute tiers, no research tables. Those are LabFork concerns. The Nudge Engine is a task queue with a cron and a protocol.

## What Ships

One Cloudflare Worker. One D1 database. One `wrangler.toml`. Deploy in 60 seconds.

```
nudge-engine/
├── wrangler.toml           # Cron trigger + D1 binding
├── schema.sql              # The 5 tables above
├── src/
│   ├── index.ts            # Hono app + scheduled handler
│   ├── api.ts              # /register, /poll, /report + admin endpoints
│   ├── cron.ts             # observe → decide → nudge → cleanup
│   ├── observers.ts        # Git, issues, tasks observers
│   └── rules.ts            # Heuristic task creation rules
├── package.json
└── README.md
```

Seven files. No `lib/`, no `utils/`, no `helpers/`. If it doesn't fit in 7 files, we're overbuilding.

## Client Integration

### Claude Code (via Hook)

A `SessionStart` hook that polls for tasks and injects them:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "curl -s -H 'Authorization: Bearer $NUDGE_TOKEN' https://nudge.example.com/poll | jq -r '.task.description // empty'"
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "curl -s -X POST -H 'Authorization: Bearer $NUDGE_TOKEN' https://nudge.example.com/poll -d '{\"status\":\"idle\"}'"
      }]
    }]
  }
}
```

Claude Code starts. The hook fires. If there's a task, it gets injected as context. Claude reads it and starts working. When the session ends, the Stop hook reports idle. Zero changes to Claude Code itself.

For continuous mode, a background process:

```bash
#!/bin/bash
# nudge-worker.sh — run in a tmux pane
while true; do
  TASK=$(curl -s -H "Authorization: Bearer $NUDGE_TOKEN" https://nudge.example.com/poll)
  if echo "$TASK" | jq -e '.task' > /dev/null 2>&1; then
    DESCRIPTION=$(echo "$TASK" | jq -r '.task.description')
    TASK_ID=$(echo "$TASK" | jq -r '.task.id')

    # Run Claude Code headless
    RESULT=$(claude --headless --print --prompt "$DESCRIPTION" 2>&1)

    # Report back
    curl -s -X POST -H "Authorization: Bearer $NUDGE_TOKEN" \
      https://nudge.example.com/report \
      -d "{\"taskId\":\"$TASK_ID\",\"success\":true,\"result\":{\"summary\":\"$RESULT\"}}"
  fi
  sleep 300  # Check every 5 minutes
done
```

30 lines of bash. That's the entire Claude Code integration. No plugin, no SDK, no framework.

### Codex

Same pattern. Codex has `codex --quiet --prompt "..."`. Swap the command.

### Ollama

Same pattern. `curl http://localhost:11434/api/generate -d '{"model":"qwen3","prompt":"..."}'`. Parse the response. Report back.

### Any future tool

If it takes a prompt and returns text, it's compatible. The protocol doesn't know or care what's behind the `/poll` → `/report` cycle.

## What This Replaces (Your Job)

Let's be honest about what the cron is doing. It's the part of your job that's pattern recognition on project state:

| What you do today | What the cron does |
|---|---|
| Check CI dashboard in the morning | Observe: read test results |
| Scan new issues, decide priority | Observe + Decide: triage issues |
| Notice a PR has been open 3 days | Observe: stalePRs detection |
| Tell a dev "hey, can you look at this?" | Decide: create task, assign to idle worker |
| Check if that bug fix actually shipped | Nudge: is the task still in progress? |
| Realize someone's blocked, reassign | Nudge: detect stuck tasks, reset |
| Weekly cleanup of stale branches | Cleanup: automated |

It doesn't replace the judgment calls. "Should we refactor auth or ship the feature?" — that's still you. But the toil of scanning, triaging, nudging, following up? That's the cron's job now.

## What We DON'T Build

- **No web dashboard.** `curl /stats` returns JSON. Pipe it to `jq`. If you need a dashboard, point Grafana at it later.
- **No user accounts.** Workers register with a name. That's identity enough. Auth is a bearer token.
- **No LLM in the cron.** The cron runs rules. If you want LLM-informed decisions, dispatch an assessment task to a worker and read the result next cycle. The cron stays fast and free.
- **No SDK.** The protocol is three HTTP endpoints. `curl` is the SDK.
- **No plugin system.** Observers are functions in `observers.ts`. Add yours with a PR. If we need a plugin system later, we'll know because people keep forking and patching `observers.ts`. That's the signal to extract.
- **No retry logic in the client.** The server handles retries. If a task fails, the cron resets it to pending. Another worker picks it up. The client's job is simple: poll, work, report.

## Migration Path From LabFork Workers

Our current codebase → Nudge Engine in three steps:

**Step 1: Fork and delete.**
Copy `workers/`. Delete: `agents/`, `workflows/worker.ts`, all LabFork-specific routes from `api/routes.ts`, research tables from `schema.sql`, paper/crowd task generation from `index.ts`. Keep: `compute.ts` (rename to `api.ts`), device registration (rename to worker registration), task CRUD, cleanup logic, cron handler.

**Step 2: Simplify the schema.**
Replace `compute_devices` → `workers`, `compute_tasks` → `tasks`. Drop `tier`, `platform`, `capabilities` (GPU-specific). Add `type`, `capabilities` as simple string/JSON. Drop `compute_tasks.min_tier` routing — the Nudge Engine doesn't route by hardware tier, it routes by capability tags.

**Step 3: Add the decision engine.**
Replace `ManagerWorkflow` (which dispatches assessment compute tasks to the 4090) with `cron.ts` that runs heuristic rules. The LLM assessment path becomes optional: if a worker with `["assessment"]` capability is online, dispatch to it. Otherwise, use rules.

Estimated effort: 1 day of focused work. We're deleting more than we're writing.

## Cost

- Cloudflare Workers free tier: 100k requests/day, 10ms CPU per request
- D1 free tier: 5M reads/day, 100k writes/day
- Cron running every 15 min = 96 invocations/day
- Each poll/report = 1 request + 1-2 D1 queries

A single-developer project with 10 workers polling every 5 minutes:
- 10 workers × 288 polls/day = 2,880 requests/day
- 96 cron runs × ~5 DB queries = 480 queries/day
- Task reports: maybe 50/day = 50 requests + 150 queries

**Total: ~3,000 requests/day, ~700 DB queries/day. Well within free tier.** The Nudge Engine costs $0/month to run for a small team. The LLM calls (via workers) are the only cost, and those scale with how much work you actually dispatch.

## The Carmack Principle

The right version of this is the smallest thing that works. Not the most general, not the most configurable, not the most future-proof. The version that a single developer can deploy in 5 minutes, connect their Claude Code to with a shell script, and wake up tomorrow to find their failing tests fixed and their issues triaged.

If that works, everything else follows. If it doesn't, no amount of architecture will save it.

Ship the protocol. Ship the cron. Ship the 30-line bash wrapper. See what happens.
