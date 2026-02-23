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

-- ============================================================================
-- DISTRIBUTED COMPUTE NETWORK TABLES
-- These tables support the compute network where inference runs on registered
-- devices (4090 GPU, contributors, WebGPU) instead of Workers AI
-- ============================================================================

-- Compute devices table: registered compute resources
CREATE TABLE compute_devices (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  tier TEXT NOT NULL CHECK(tier IN ('power', 'standard', 'crowd')),
  platform TEXT NOT NULL CHECK(platform IN ('cuda', 'metal', 'webgpu', 'cpu')),
  capabilities TEXT NOT NULL, -- JSON: {compute: TFLOPS, memory: GB, models: []}
  endpoint_url TEXT, -- Where to send tasks (null for polling devices)
  status TEXT DEFAULT 'offline' CHECK(status IN ('online', 'busy', 'offline', 'paused')),
  current_task_id TEXT,
  last_heartbeat TEXT,
  stats TEXT, -- JSON: {tasksCompleted, creditsEarned, totalComputeTime}
  auth_token TEXT, -- Bearer token for device authentication
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Compute tasks table: inference tasks dispatched to compute network
CREATE TABLE compute_tasks (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK(type IN (
    'inference', 'embedding', 'assessment', 'planning', 'execution',
    'draft_generation', 'draft_verification', 'summarization', 'classification',
    'simulation'
  )),
  input TEXT NOT NULL, -- JSON: prompt, model_id, config
  config TEXT NOT NULL, -- JSON: maxTokens, temperature, minTier
  status TEXT DEFAULT 'pending' CHECK(status IN (
    'pending', 'assigned', 'processing', 'completed', 'failed', 'timeout'
  )),
  priority INTEGER DEFAULT 5,
  min_tier TEXT CHECK(min_tier IN ('power', 'standard', 'crowd')),
  assigned_device_id TEXT,
  result TEXT, -- JSON: output, metrics
  error TEXT,
  parent_task_id TEXT, -- Links to tasks table for workflow integration
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  assigned_at TEXT,
  completed_at TEXT,
  FOREIGN KEY (assigned_device_id) REFERENCES compute_devices(id) ON DELETE SET NULL,
  FOREIGN KEY (parent_task_id) REFERENCES tasks(id) ON DELETE SET NULL
);

-- Indexes for compute tables
CREATE INDEX idx_compute_devices_tier_status ON compute_devices(tier, status);
CREATE INDEX idx_compute_devices_last_heartbeat ON compute_devices(last_heartbeat);
CREATE INDEX idx_compute_devices_auth_token ON compute_devices(auth_token);
CREATE INDEX idx_compute_tasks_status_priority ON compute_tasks(status, priority DESC);
CREATE INDEX idx_compute_tasks_assigned_device ON compute_tasks(assigned_device_id);
CREATE INDEX idx_compute_tasks_parent_task ON compute_tasks(parent_task_id);

-- ============================================================================
-- RESEARCH TABLES
-- These tables support independent research by 4090 and syncing findings
-- ============================================================================

-- Research objectives: queue of research topics for autonomous exploration
CREATE TABLE research_objectives (
  id TEXT PRIMARY KEY,
  lab_id TEXT NOT NULL DEFAULT 'voice-clone',
  title TEXT NOT NULL,
  description TEXT,
  priority INTEGER DEFAULT 5 CHECK(priority >= 1 AND priority <= 10),
  status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'archived')),
  tags TEXT, -- JSON array of tags
  assigned_device_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (assigned_device_id) REFERENCES compute_devices(id) ON DELETE SET NULL
);

-- Research results: findings synced from 4090's local research
CREATE TABLE research_results (
  id TEXT PRIMARY KEY,
  device_id TEXT NOT NULL,
  lab_id TEXT NOT NULL DEFAULT 'voice-clone',
  objective_id TEXT,
  objective_title TEXT,
  objective_description TEXT,
  success INTEGER DEFAULT 0 CHECK(success IN (0, 1)),
  output TEXT, -- Summary or full research output
  duration_minutes REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (device_id) REFERENCES compute_devices(id) ON DELETE CASCADE,
  FOREIGN KEY (objective_id) REFERENCES research_objectives(id) ON DELETE SET NULL
);

-- Indexes for research tables
CREATE INDEX idx_research_objectives_lab_status ON research_objectives(lab_id, status);
CREATE INDEX idx_research_objectives_priority ON research_objectives(priority DESC);
CREATE INDEX idx_research_results_lab_created ON research_results(lab_id, created_at DESC);
CREATE INDEX idx_research_results_device ON research_results(device_id);

-- ============================================================================
-- SEED PAPERS TABLE
-- Papers imported from arXiv/Semantic Scholar that feed the compute task queue.
-- Each paper generates 3-5 crowd-tier compute tasks (summarization, classification,
-- embedding, assessment).
-- ============================================================================

CREATE TABLE IF NOT EXISTS seed_papers (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  abstract TEXT NOT NULL,
  source_url TEXT,
  domain TEXT DEFAULT 'general',
  added_at TEXT DEFAULT (datetime('now')),
  tasks_generated INTEGER DEFAULT 0
);

CREATE INDEX idx_seed_papers_domain ON seed_papers(domain);
CREATE INDEX idx_seed_papers_tasks_generated ON seed_papers(tasks_generated);
