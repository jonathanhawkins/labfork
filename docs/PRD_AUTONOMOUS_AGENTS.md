# PRD: Autonomous Agent System for LabFork

> **Version:** 1.0.0
> **Status:** Draft
> **Author:** LabFork CEO Agent
> **Date:** January 2026

## Executive Summary

LabFork currently displays mock/demo agents that show fake progress. This PRD defines the system to make agents **real and autonomous** - agents that wake up on their own, assess project state, make decisions, and do actual work until they hit physical barriers.

## Problem Statement

### Current State (Broken)
- Agents (Spark, Mesh, Lumen) are **static seed data** in `firefly-seed-data.ts`
- Progress percentages are **hardcoded strings** that never change
- No actual AI inference happens for project work
- `/watch` page shows a theater, not real work
- Humans must manually drive all progress

### Desired State
- Agents **wake up autonomously** on a schedule
- A **Manager Agent** assesses projects and creates tasks
- **Worker Agents** pull tasks and execute them with AI
- Progress is **real and persisted** to database
- Agents work until they hit **physical barriers** (ordering parts, testing hardware)
- `/watch` page shows **actual work happening**

## Technical Architecture

### Platform: Cloudflare Workflows

We use [Cloudflare Workflows](https://developers.cloudflare.com/workflows/) because:
- **Durable execution**: Run for minutes, hours, days
- **State persistence**: Remember where we left off (free: 3 days)
- **Auto-retries**: Built-in failure handling
- **Workers AI**: Free AI inference
- **Cron Triggers**: Free scheduled execution
- **Free tier**: Generous limits for our use case

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUDFLARE WORKERS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Cron      │───▶│  Manager    │───▶│  Task       │         │
│  │  Trigger    │    │  Workflow   │    │  Queue      │         │
│  │ (every 15m) │    │             │    │  (D1/KV)    │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                 │
│                     ┌──────────────────────────┼─────────┐      │
│                     │                          │         │      │
│                     ▼                          ▼         ▼      │
│              ┌─────────────┐           ┌─────────┐ ┌─────────┐ │
│              │   Worker    │           │ Worker  │ │ Worker  │ │
│              │  Workflow   │           │Workflow │ │Workflow │ │
│              │  (Spark)    │           │ (Mesh)  │ │(Lumen)  │ │
│              └──────┬──────┘           └────┬────┘ └────┬────┘ │
│                     │                       │           │      │
│                     └───────────┬───────────┴───────────┘      │
│                                 │                               │
│                                 ▼                               │
│                          ┌─────────────┐                       │
│                          │ Workers AI  │                       │
│                          │ (Inference) │                       │
│                          └─────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUDFLARE D1 (SQLite)                        │
├─────────────────────────────────────────────────────────────────┤
│  Tables:                                                         │
│  - projects        (project metadata)                           │
│  - tasks           (work items with status)                     │
│  - agent_state     (persistent agent memory)                    │
│  - work_log        (audit trail of actions)                     │
│  - artifacts       (generated code, research notes)             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VERCEL (Frontend)                             │
├─────────────────────────────────────────────────────────────────┤
│  - /watch          (live view of agent work)                    │
│  - /contribute     (join compute network)                       │
│  - /projects/*     (project dashboards)                         │
│  - /api/agents/*   (proxy to Cloudflare)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Models

#### Projects Table
```sql
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  status TEXT DEFAULT 'active',
  config JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Tasks Table
```sql
CREATE TABLE tasks (
  id TEXT PRIMARY KEY,
  project_id TEXT REFERENCES projects(id),
  title TEXT NOT NULL,
  description TEXT,
  status TEXT DEFAULT 'pending', -- pending, in_progress, completed, blocked
  priority INTEGER DEFAULT 5,
  assigned_agent TEXT,
  blocked_by JSON, -- array of task IDs
  requires_physical BOOLEAN DEFAULT FALSE,
  progress INTEGER DEFAULT 0,
  result JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Agent State Table
```sql
CREATE TABLE agent_state (
  agent_id TEXT PRIMARY KEY,
  project_id TEXT REFERENCES projects(id),
  persona JSON, -- name, role, specialization
  memory JSON, -- context, learnings, preferences
  current_task_id TEXT REFERENCES tasks(id),
  status TEXT DEFAULT 'idle',
  tokens_used INTEGER DEFAULT 0,
  last_active TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Work Log Table
```sql
CREATE TABLE work_log (
  id TEXT PRIMARY KEY,
  agent_id TEXT REFERENCES agent_state(agent_id),
  task_id TEXT REFERENCES tasks(id),
  action TEXT NOT NULL,
  input JSON,
  output JSON,
  duration_ms INTEGER,
  tokens_used INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Artifacts Table
```sql
CREATE TABLE artifacts (
  id TEXT PRIMARY KEY,
  task_id TEXT REFERENCES tasks(id),
  type TEXT NOT NULL, -- code, research_note, design, bom
  name TEXT NOT NULL,
  content TEXT,
  metadata JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Workflow Definitions

#### 1. Manager Workflow (Scheduled)

Runs every 15 minutes via Cron Trigger.

```typescript
// workers/src/workflows/manager.ts
export class ManagerWorkflow extends WorkflowEntrypoint {
  async run(event: WorkflowEvent, step: WorkflowStep) {
    // Step 1: Get all active projects
    const projects = await step.do('get-projects', async () => {
      return await this.env.DB.prepare(
        'SELECT * FROM projects WHERE status = ?'
      ).bind('active').all();
    });

    // Step 2: For each project, assess state
    for (const project of projects.results) {
      const assessment = await step.do(`assess-${project.id}`, async () => {
        return await this.assessProject(project);
      });

      // Step 3: Create/update tasks based on assessment
      if (assessment.needsWork) {
        await step.do(`create-tasks-${project.id}`, async () => {
          return await this.createTasks(project, assessment);
        });
      }

      // Step 4: Assign idle agents to tasks
      await step.do(`assign-agents-${project.id}`, async () => {
        return await this.assignAgentsToTasks(project);
      });
    }

    // Step 5: Trigger worker workflows for assigned tasks
    await step.do('trigger-workers', async () => {
      return await this.triggerWorkerWorkflows();
    });
  }

  async assessProject(project: Project): Promise<Assessment> {
    // Use Workers AI to assess project state
    const ai = new Ai(this.env.AI);
    const response = await ai.run('@cf/meta/llama-3.1-70b-instruct', {
      messages: [
        { role: 'system', content: PROJECT_ASSESSMENT_PROMPT },
        { role: 'user', content: JSON.stringify(project) }
      ]
    });
    return JSON.parse(response.response);
  }
}
```

#### 2. Worker Workflow (Task Execution)

Triggered by Manager or directly for specific tasks.

```typescript
// workers/src/workflows/worker.ts
export class WorkerWorkflow extends WorkflowEntrypoint {
  async run(event: WorkflowEvent<{ taskId: string, agentId: string }>, step: WorkflowStep) {
    const { taskId, agentId } = event.payload;

    // Step 1: Load task and agent state
    const context = await step.do('load-context', async () => {
      const task = await this.getTask(taskId);
      const agent = await this.getAgent(agentId);
      return { task, agent };
    });

    // Step 2: Check if task requires physical action
    if (context.task.requires_physical) {
      await step.do('mark-blocked', async () => {
        await this.markTaskBlocked(taskId, 'Requires physical action');
        await this.notifyHumans(context.task);
      });
      return { status: 'blocked', reason: 'physical_required' };
    }

    // Step 3: Plan the work
    const plan = await step.do('plan-work', async () => {
      return await this.planWork(context);
    });

    // Step 4: Execute each step of the plan
    for (const planStep of plan.steps) {
      const result = await step.do(`execute-${planStep.id}`, async () => {
        return await this.executeStep(planStep, context);
      });

      // Step 5: Update progress
      await step.do(`update-progress-${planStep.id}`, async () => {
        await this.updateTaskProgress(taskId, result);
        await this.logWork(agentId, taskId, planStep, result);
      });

      // Check if we hit a physical barrier mid-task
      if (result.requiresPhysical) {
        await step.do('mark-blocked-mid-task', async () => {
          await this.markTaskBlocked(taskId, result.blockReason);
        });
        return { status: 'blocked', reason: result.blockReason };
      }
    }

    // Step 6: Complete task
    await step.do('complete-task', async () => {
      await this.completeTask(taskId, plan.results);
    });

    return { status: 'completed', results: plan.results };
  }
}
```

### Agent Personas

Each agent has a defined persona that shapes their behavior:

#### Spark (Solar Specialist)
```json
{
  "id": "spark",
  "name": "Spark",
  "role": "Solar & Power Specialist",
  "model": "@cf/meta/llama-3.1-70b-instruct",
  "specializations": ["MPPT algorithms", "battery management", "power optimization"],
  "system_prompt": "You are Spark, a solar energy specialist working on the Firefly Network project..."
}
```

#### Mesh (Network Architect)
```json
{
  "id": "mesh",
  "name": "Mesh",
  "role": "Network Architect",
  "model": "@cf/meta/llama-3.1-70b-instruct",
  "specializations": ["Thread protocol", "mesh networking", "routing algorithms"],
  "system_prompt": "You are Mesh, a network architect working on the Firefly Network project..."
}
```

#### Lumen (Light Engineer)
```json
{
  "id": "lumen",
  "name": "Lumen",
  "role": "Light Engineer",
  "model": "@cf/meta/llama-3.1-70b-instruct",
  "specializations": ["LED optimization", "circadian lighting", "optics"],
  "system_prompt": "You are Lumen, a lighting engineer working on the Firefly Network project..."
}
```

### Physical Barrier Protocol

When an agent determines work requires physical action:

1. **Detection**: Agent recognizes task needs physical world (ordering, testing, assembly)
2. **Documentation**: Agent documents exactly what's needed
3. **Task Creation**: Creates human task with clear instructions
4. **Notification**: Sends notification (email, Slack, dashboard alert)
5. **Blocking**: Marks dependent tasks as blocked
6. **Waiting**: Workflow pauses, waiting for human input
7. **Resumption**: When human marks physical task complete, workflow continues

```typescript
interface PhysicalBarrier {
  type: 'order_parts' | 'assemble' | 'test' | 'measure' | 'photograph';
  description: string;
  items?: BOMItem[];
  instructions: string[];
  estimated_cost?: number;
  estimated_time?: string;
  blocking_tasks: string[];
}
```

### API Endpoints

#### Cloudflare Workers API

```
POST /api/workflows/manager/trigger    - Manually trigger manager
POST /api/workflows/worker/trigger     - Trigger specific worker
GET  /api/projects/:id/status          - Get project status
GET  /api/agents/:id/status            - Get agent status
GET  /api/tasks                        - List all tasks
POST /api/tasks/:id/complete           - Mark physical task complete
GET  /api/work-log                     - Get recent work log
WS   /api/live                         - WebSocket for live updates
```

#### Vercel API (Proxy)

```
GET  /api/agents/status                - Proxy to CF workers
GET  /api/agents/live                  - WebSocket proxy
POST /api/agents/trigger               - Trigger workflow
```

### Frontend Integration

#### /watch Page Updates

Replace mock data fetching with real API:

```typescript
// Current (fake)
const agents = FIREFLY_AGENTS; // Static seed data

// New (real)
const { data: agents } = useSWR('/api/agents/status', fetcher, {
  refreshInterval: 5000 // Poll every 5s
});

// Even better (WebSocket)
const agents = useAgentLiveUpdates(); // Real-time via WebSocket
```

#### Live Activity Feed

```typescript
const ActivityFeed = () => {
  const { logs } = useWorkLog({ limit: 50 });

  return (
    <div>
      {logs.map(log => (
        <ActivityItem
          key={log.id}
          agent={log.agent_id}
          action={log.action}
          timestamp={log.created_at}
        />
      ))}
    </div>
  );
};
```

## Implementation Phases

### Phase 1: Infrastructure (Week 1)
- [ ] Set up Cloudflare Workers project
- [ ] Create D1 database with schema
- [ ] Implement basic Manager Workflow
- [ ] Implement basic Worker Workflow
- [ ] Set up Cron Trigger (15 min)

### Phase 2: Agent Intelligence (Week 2)
- [ ] Define agent personas and prompts
- [ ] Implement project assessment logic
- [ ] Implement task creation from assessment
- [ ] Implement task execution with Workers AI
- [ ] Add work logging

### Phase 3: Physical Barriers (Week 3)
- [ ] Implement barrier detection
- [ ] Create human task workflow
- [ ] Add notification system
- [ ] Implement task resumption

### Phase 4: Frontend Integration (Week 4)
- [ ] Update /watch to use real API
- [ ] Add WebSocket live updates
- [ ] Update project dashboards
- [ ] Remove all seed/mock data

### Phase 5: Firefly Bootstrap (Week 5)
- [ ] Seed Firefly project in database
- [ ] Create initial tasks from architecture doc
- [ ] Let agents start working
- [ ] Monitor and iterate

## Success Metrics

| Metric | Target |
|--------|--------|
| Agent uptime | >99% |
| Tasks completed/week | >10 |
| Time to first physical barrier | <1 week |
| Real code artifacts generated | >5 files |
| Zero mock data in production | 100% |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| AI generates bad code | Human review before merge |
| Infinite loops | Max iterations per workflow |
| Cost overruns | Workers AI free tier, budget alerts |
| Agents get stuck | Timeout + human escalation |

## Open Questions

1. Should agents be able to create GitHub PRs directly?
2. How do we handle API keys for external services (component DBs)?
3. Should we allow agents to spend money (order parts)?
4. How do we prevent duplicate work across agents?

## Appendix

### A. Workers AI Models Available (Free Tier)

- `@cf/meta/llama-3.1-70b-instruct` - Best for reasoning
- `@cf/meta/llama-3.1-8b-instruct` - Fast for simple tasks
- `@cf/baai/bge-large-en-v1.5` - Embeddings
- `@cf/microsoft/phi-2` - Lightweight coding

### B. Cloudflare Free Tier Limits

- Workers: 100K requests/day
- D1: 5GB storage, 5M rows read/day
- Workflows: 1K executions/day
- AI: 10K neurons/day (plenty for text)

### C. Related Documents

- [FIREFLY_SOFTWARE_ARCH.md](./FIREFLY_SOFTWARE_ARCH.md)
- [firefly-network/domain.yaml](../.domains/firefly-network/domain.yaml)
