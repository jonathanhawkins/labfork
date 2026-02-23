-- Nudge Engine Schema
-- A persistent autonomous task manager for AI CLI tools.
-- 4 tables. No joins in the hot path.

-- Workers: anything that polls for tasks and reports results
CREATE TABLE IF NOT EXISTS workers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,                     -- claude-code | codex | ollama | custom
  capabilities TEXT NOT NULL DEFAULT '[]', -- JSON array: ["code","test","review","research"]
  status TEXT DEFAULT 'online' CHECK(status IN ('online', 'busy', 'offline')),
  auth_token TEXT NOT NULL UNIQUE,
  current_task_id TEXT,
  last_heartbeat TEXT,
  tasks_completed INTEGER DEFAULT 0,
  tasks_failed INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

-- Tasks: units of work
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  action TEXT NOT NULL,                   -- verb-noun: fix-test, triage-issue, review-pr
  description TEXT NOT NULL,              -- Plain English, what to do and what done looks like
  context TEXT,                           -- JSON: repo, branch, files, relatedIssue, etc.
  constraints TEXT,                       -- JSON: maxFiles, mustPassTests, createPR, timeoutMinutes
  status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'assigned', 'completed', 'failed')),
  priority INTEGER DEFAULT 5 CHECK(priority >= 1 AND priority <= 10),
  assigned_worker_id TEXT,
  result TEXT,                            -- JSON: summary, filesChanged, prUrl, testsPass
  error TEXT,
  required_capability TEXT,               -- NULL = any worker, or: code | monitor | advisory
  source TEXT DEFAULT 'human',            -- human | cron | llm | observer
  attempts INTEGER DEFAULT 0,
  timeout_minutes INTEGER DEFAULT 30,
  parent_task_id TEXT,                    -- NULL = top-level task, or: parent task ID for subtasks
  created_at TEXT DEFAULT (datetime('now')),
  assigned_at TEXT,
  completed_at TEXT,
  FOREIGN KEY (assigned_worker_id) REFERENCES workers(id) ON DELETE SET NULL,
  FOREIGN KEY (parent_task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

-- Observations: snapshots of external state fed into the cron
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,                     -- git | issues | tests | health | custom
  data TEXT NOT NULL,                     -- JSON snapshot
  observed_by TEXT,                       -- worker_id or 'cron'
  created_at TEXT DEFAULT (datetime('now'))
);

-- Work log: audit trail
CREATE TABLE IF NOT EXISTS work_log (
  id TEXT PRIMARY KEY,
  worker_id TEXT,
  task_id TEXT,
  event TEXT NOT NULL,                    -- registered | assigned | completed | failed | nudged | created | observation
  detail TEXT,                            -- JSON, event-specific
  created_at TEXT DEFAULT (datetime('now'))
);

-- Indexes for hot paths
CREATE INDEX IF NOT EXISTS idx_workers_status ON workers(status);
CREATE INDEX IF NOT EXISTS idx_workers_auth_token ON workers(auth_token);
CREATE INDEX IF NOT EXISTS idx_tasks_status_priority ON tasks(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_worker ON tasks(assigned_worker_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_observations_type_created ON observations(type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_work_log_created ON work_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_work_log_task ON work_log(task_id);
