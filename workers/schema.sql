-- D1 Database Schema for LabFork Voice Clone Pipeline
-- This schema manages projects, tasks, agents, work logs, and artifacts

-- Projects table: represents research labs or experiments
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  status TEXT DEFAULT 'active' CHECK(status IN ('active', 'archived', 'paused')),
  config TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table: represents work items within projects
CREATE TABLE tasks (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'blocked')),
  priority INTEGER DEFAULT 5 CHECK(priority >= 1 AND priority <= 10),
  assigned_agent TEXT,
  blocked_by TEXT,
  requires_physical INTEGER DEFAULT 0 CHECK(requires_physical IN (0, 1)),
  progress INTEGER DEFAULT 0 CHECK(progress >= 0 AND progress <= 100),
  result TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Agent state table: tracks agent status, memory, and context
CREATE TABLE agent_state (
  agent_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  persona TEXT,
  memory TEXT,
  current_task_id TEXT,
  status TEXT DEFAULT 'idle' CHECK(status IN ('idle', 'working', 'error', 'offline')),
  tokens_used INTEGER DEFAULT 0,
  last_active TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
  FOREIGN KEY (current_task_id) REFERENCES tasks(id) ON DELETE SET NULL
);

-- Work log table: audit trail of all agent actions
CREATE TABLE work_log (
  id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL,
  task_id TEXT,
  action TEXT NOT NULL,
  input TEXT,
  output TEXT,
  duration_ms INTEGER,
  tokens_used INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (agent_id) REFERENCES agent_state(agent_id) ON DELETE CASCADE,
  FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE SET NULL
);

-- Artifacts table: stores generated files, results, and outputs
CREATE TABLE artifacts (
  id TEXT PRIMARY KEY,
  task_id TEXT,
  type TEXT NOT NULL,
  name TEXT NOT NULL,
  content TEXT,
  metadata TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX idx_tasks_project_id_status ON tasks(project_id, status);
CREATE INDEX idx_tasks_assigned_agent ON tasks(assigned_agent);
CREATE INDEX idx_agent_state_project_id ON agent_state(project_id);
CREATE INDEX idx_work_log_agent_id_created_at ON work_log(agent_id, created_at);
CREATE INDEX idx_work_log_task_id ON work_log(task_id);
CREATE INDEX idx_artifacts_task_id ON artifacts(task_id);
CREATE INDEX idx_artifacts_type ON artifacts(type);
CREATE INDEX idx_projects_slug ON projects(slug);
