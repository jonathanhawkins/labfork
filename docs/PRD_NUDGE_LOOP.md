# PRD: The Nudge Loop

## A persistent autonomous manager for Claude Code Agent Teams

---

## The Problem

Claude Code Agent Teams (Feb 2026) let you spawn multiple Claude instances that coordinate on tasks in parallel. It's powerful. But it has a fundamental constraint: **someone has to start it**.

A human opens a terminal, describes the work, spawns teammates, watches them go. When the session ends, everything stops. The next day, the human has to remember where things left off, re-assess, re-spawn, re-direct. The agents are capable workers, but nobody's managing the shop overnight.

This is the same gap that exists in every AI coding tool today. They're reactive. You ask, they do. You stop asking, they stop doing.

## The Insight

We accidentally built the missing piece while working on LabFork's distributed research system. It's embarrassingly simple:

**A cron job that runs every 15 minutes and asks: "What needs to happen next?"**

That's it. No complex planning. No massive context windows. Just a persistent loop that:

1. Looks at the current state of things
2. Identifies what's stalled, what's ready, what's new
3. Creates small, simple tasks
4. Dispatches them to workers
5. Checks on previous work
6. Goes back to sleep

We call it the **Nudge Loop**. It doesn't try to be smart. It tries to be persistent. The intelligence lives in the workers (Claude). The loop just keeps nudging them.

## Why It Works

### Simplicity is resilience
Each task the loop creates is deliberately small. "Assess this project." "Plan next steps for this module." "Execute step 3." If a task fails, the next cron cycle picks it up. If a worker gets confused, the manager breaks the work down smaller. There's no catastrophic failure mode because there's no complex plan to derail.

### State lives in the database, not in a session
Claude Code Agent Teams store state in `~/.claude/tasks/{team-name}/` - ephemeral, session-scoped. Our state lives in a database. Projects, tasks, agent assignments, work logs - all persistent. The cron job is stateless. It reads the world, decides what to nudge, and writes tasks. If the cron process dies and restarts, nothing is lost.

### Small nudges compound
A single 15-minute cycle doesn't do much. But 96 cycles per day, each making small progress - assessing a project, filing an issue, fixing a test, updating docs - compounds into significant autonomous output. Like a Roomba: any single pass is unremarkable, but leave it running and the floor stays clean.

### It's not trying to be an AGI planner
The manager doesn't build a grand plan and execute it. It looks at what's in front of it right now and picks the most obvious next thing. This is intentionally dumb. Grand plans fail. Small, repeated observations and actions converge.

## How It Works Today (LabFork)

Our current implementation runs as a Cloudflare Worker with a cron trigger:

```
Every 15 minutes:
  ManagerWorkflow.run()
    ├── Scan active projects
    ├── For each project:
    │   ├── What's the current status?
    │   ├── Are there stalled tasks? → Re-assign or break down
    │   ├── Are there completed tasks? → Aggregate results, create next tasks
    │   ├── Is there new input? → Assess and create initial tasks
    │   └── Are agents idle? → Find work for them
    ├── Clean up stale state (offline devices, stuck tasks)
    └── Sleep until next cycle
```

Workers are simple executors. They receive a task, do the thing, report back. The manager never does work itself - it only creates and monitors tasks.

**What we learned building this:**
- Tasks must be atomic and idempotent (safe to retry)
- The manager needs to detect stuck states (task assigned but no progress for N cycles)
- Result validation matters (timing checks, content checks prevent garbage results)
- Rate limiting prevents runaway loops (max tasks per cycle, per agent)
- Work logs are essential for debugging (why did the manager create this task?)

## The Product: Nudge Loop for Claude Code

### What we're building

A Claude Code Plugin that adds a persistent autonomous manager layer on top of Agent Teams. Install it, point it at a repo, and it keeps your project moving forward between coding sessions.

### Core Components

**1. The Loop (Cron/Scheduler)**
Runs on a schedule. Could be:
- Cloudflare Worker cron (what we use today)
- GitHub Actions scheduled workflow
- Local `launchd`/`cron` on a dev machine
- A simple `while true; sleep 900; do ... done`

The loop itself does minimal work. It calls Claude with a small, focused prompt: "Here's the state of the project. What are the 1-3 most important small tasks to create or nudge?"

**2. The State Store**
Persistent task and project state. Options:
- Cloudflare D1 (our current approach - free tier, global edge)
- SQLite file in the repo (`.nudge/state.db`)
- GitHub Issues as task store (free, visible, collaborative)
- Simple JSON files (`.nudge/tasks/*.json`)

**3. The Dispatcher**
Creates and assigns work. Two modes:
- **Claude Code Agent Teams** - spawn teammates for complex work
- **Single Claude Code session** - for simple tasks, just run `claude --prompt "fix this test"`
- **Headless Claude** - `claude --headless --prompt "..."` for CI/background work

**4. The Observer**
Reads project state to inform the manager:
- `git status` / `git log` - what changed recently?
- Test results - anything failing?
- Open issues / PRs - anything stale?
- Build status - CI green?
- Custom signals - whatever matters to this project

### User Experience

**Install:**
```bash
claude plugin install nudge-loop
# or
claude --plugin-dir ./nudge-loop
```

**Configure (`.nudge/config.json`):**
```json
{
  "schedule": "*/15 * * * *",
  "scope": "development",
  "observers": ["git", "tests", "issues"],
  "maxTasksPerCycle": 3,
  "maxTeammates": 2,
  "rules": [
    "Never push to main directly",
    "Always run tests before marking a task complete",
    "Create a PR for any change over 50 lines"
  ],
  "focus": [
    "Keep test coverage above 80%",
    "Triage new issues within 1 cycle",
    "Fix failing CI within 2 cycles"
  ]
}
```

**What happens next (without you doing anything):**

```
Cycle 1 (8:00 AM):
  Observer: "3 new issues filed overnight. CI is green. No stale PRs."
  Manager: Creates task "Triage issue #47 - user reports timeout on upload"
  Manager: Creates task "Triage issue #48 - feature request for dark mode"
  Manager: Creates task "Triage issue #49 - typo in README"
  Dispatcher: Spawns Claude → fixes README typo, creates PR
  Dispatcher: Spawns Claude → adds labels + repro steps to #47, #48

Cycle 2 (8:15 AM):
  Observer: "Issue #47 triaged as bug, priority high. PR #50 open for README fix."
  Manager: Creates task "Investigate timeout bug from issue #47"
  Manager: Creates task "Review PR #50 (README fix)"
  Dispatcher: Spawns teammate → investigates #47, finds root cause, creates PR
  Dispatcher: Spawns teammate → reviews PR #50, approves

Cycle 3 (8:30 AM):
  Observer: "PR #51 open for timeout fix. Tests passing. PR #50 merged."
  Manager: Creates task "Review PR #51 (timeout fix)"
  Manager: No other urgent work. Idle.
  Dispatcher: Spawns Claude → reviews #51, requests one change

... you arrive at 9 AM, three issues triaged, typo fixed,
    timeout bug has a PR with one review comment. You just
    address the review comment and merge.
```

### Applied Beyond Development

The same pattern works for anything with observable state and decomposable tasks:

**Research (what we do today):**
- Observe: paper database, experiment results, compute device availability
- Nudge: "assess this paper", "run this simulation", "synthesize these results"

**DevOps:**
- Observe: monitoring alerts, deploy status, resource utilization
- Nudge: "investigate this alert", "scale down this idle service", "update this dependency"

**Content/Docs:**
- Observe: code changes without doc updates, stale examples, broken links
- Nudge: "update docs for this API change", "fix broken link in guide", "add example for new feature"

**QA/Testing:**
- Observe: code coverage, untested paths, flaky tests
- Nudge: "write tests for this uncovered function", "investigate flaky test X", "add edge case tests for new feature"

## Architecture

```
┌─────────────────────────────────────────────┐
│              The Nudge Loop                  │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Observer  │  │ Manager  │  │Dispatcher │  │
│  │          │──▶│          │──▶│           │  │
│  │ git,CI,  │  │ assess,  │  │ claude    │  │
│  │ issues,  │  │ decide,  │  │ --headless│  │
│  │ tests    │  │ create   │  │ or teams  │  │
│  └──────────┘  │ tasks    │  └───────────┘  │
│                └─────┬────┘        │         │
│                      │             │         │
│                ┌─────▼─────────────▼───┐     │
│                │     State Store       │     │
│                │  tasks, work log,     │     │
│                │  project state        │     │
│                └───────────────────────┘     │
│                                              │
│  Runs every N minutes. Stateless process.    │
│  All state lives in the store.               │
└─────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  ┌──────────────┐          ┌──────────────┐
  │ Claude Code  │          │ Claude Code  │
  │ (headless)   │          │ Agent Teams  │
  │              │          │              │
  │ Simple tasks │          │ Complex work │
  │ "fix typo"   │          │ "refactor    │
  │ "triage #47" │          │  auth system"│
  └──────────────┘          └──────────────┘
```

## What Makes This Different

| | Traditional CI/CD | Claude Code Teams | Nudge Loop |
|---|---|---|---|
| **Trigger** | Push/PR events | Human starts session | Persistent timer |
| **Intelligence** | Static rules | Full Claude | Claude for both manager + workers |
| **Scope** | Build/test/deploy | Whatever you ask | Continuous project health |
| **Persistence** | Per-pipeline | Per-session | Always-on |
| **Human role** | Configure pipelines | Direct the team | Set goals, review output |
| **Failure mode** | Pipeline fails, alerts | Session ends | Next cycle retries |

## Implementation Phases

### Phase 1: Extract the Loop
Pull our ManagerWorkflow pattern out of the LabFork-specific code. Make it generic:
- Configurable observers (git, issues, tests, custom)
- Configurable rules and focus areas
- Generic task creation (not research-specific)
- Pluggable state store (D1, SQLite, JSON files)

### Phase 2: Claude Code Integration
Wire the dispatcher to Claude Code:
- `claude --headless --prompt "..."` for simple tasks
- Agent Teams spawning for complex work
- Hook into `TaskCompleted` for result validation
- Use Claude Code's native task system as the state store

### Phase 3: Package as Plugin
- `.claude-plugin/plugin.json` manifest
- Skills: `/nudge:status`, `/nudge:configure`, `/nudge:pause`, `/nudge:history`
- Hooks: `TaskCompleted` validation, `SessionStart` state injection
- MCP server for the state store (so Claude can query task history)

### Phase 4: Scheduling Options
- Cloudflare Worker cron (free, global, our default)
- GitHub Actions (free for public repos, familiar)
- Local cron/launchd (offline, private)
- `claude --daemon` mode (if Anthropic ships it)

## Open Questions

1. **Cost control** - Each cycle spawns Claude sessions. Need budget caps, token tracking, and the ability to say "nothing urgent, skip this cycle."

2. **Scope creep** - The manager might keep finding things to do forever. Need clear boundaries: "only work on issues labeled 'nudge-ok'" or "only touch files in src/".

3. **Human override** - What happens when a human is actively coding and the nudge loop tries to change the same files? Need conflict detection and automatic deferral.

4. **Trust escalation** - Some tasks (delete a file, push to main, close an issue) need human approval. The loop should create a "needs-review" task instead of acting directly.

5. **Observability** - How does the human know what the loop did overnight? Daily digest? Slack notification? Dashboard?

## Prior Art

- **Dependabot / Renovate** - Automated dependency PRs. Same nudge pattern but limited to deps.
- **GitHub Copilot Workspace** - AI-assisted issue-to-PR. But reactive (triggered by human), not persistent.
- **Devin / SWE-Agent** - Autonomous coding agents. Session-scoped, not persistent loops.
- **Our LabFork ManagerWorkflow** - The direct ancestor. Proven pattern running in production on Cloudflare Workers since Feb 2026.

## Success Metrics

- **Mean time to triage**: New issues assessed within 1 cycle (15 min)
- **CI recovery time**: Failing tests fixed within 2-4 cycles (30-60 min)
- **Stale PR reduction**: No PR sits unreviewed for more than 1 day
- **Developer morning experience**: Arrive to triaged issues, draft PRs, and a status summary instead of a cold start
- **Token efficiency**: < $5/day for a typical active repo

---

*This document describes a product concept derived from LabFork's production infrastructure. The core pattern - a persistent, stateless cron loop that observes state, creates small tasks, and dispatches AI workers - is infrastructure-agnostic and applies to any domain with observable state and decomposable work.*
