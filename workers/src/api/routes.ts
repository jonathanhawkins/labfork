import { Hono } from 'hono';
import { cors } from 'hono/cors';

// ============================================================================
// Types
// ============================================================================

interface Env {
  DB: D1Database;
  AI: Ai;
}

// Response types
interface HealthResponse {
  status: 'ok' | 'error';
  timestamp: string;
}

interface ProjectStatus {
  id: string;
  name: string;
  status: string;
  created_at: string;
  updated_at: string;
  task_counts: {
    total: number;
    pending: number;
    in_progress: number;
    completed: number;
    blocked: number;
  };
  agents: AgentSummary[];
}

interface AgentSummary {
  id: string;
  name: string;
  type: string;
  status: string;
  current_task_id: string | null;
}

interface AgentState {
  id: string;
  name: string;
  type: string;
  status: string;
  current_task_id: string | null;
  current_task: TaskSummary | null;
  capabilities: string[];
  created_at: string;
  updated_at: string;
}

interface AgentDetailedStatus extends AgentState {
  recent_work_log: WorkLogEntry[];
  stats: {
    tasks_completed_today: number;
    tasks_completed_total: number;
    average_task_duration_minutes: number;
  };
}

interface TaskSummary {
  id: string;
  title: string;
  status: string;
  priority: number;
}

interface Task {
  id: string;
  project_id: string;
  title: string;
  description: string | null;
  status: string;
  priority: number;
  assigned_agent_id: string | null;
  parent_task_id: string | null;
  blocked_by: string[];
  context: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

interface WorkLogEntry {
  id: string;
  agent_id: string;
  task_id: string | null;
  action: string;
  details: Record<string, unknown>;
  created_at: string;
}

interface ApiError {
  error: string;
  message: string;
  status: number;
}

// ============================================================================
// Helper Functions
// ============================================================================

function parseJsonField<T>(value: string | null, defaultValue: T): T {
  if (!value) return defaultValue;
  try {
    return JSON.parse(value) as T;
  } catch {
    return defaultValue;
  }
}

function createErrorResponse(
  message: string,
  status: number = 500
): ApiError {
  return {
    error: status >= 500 ? 'Internal Server Error' : 'Bad Request',
    message,
    status,
  };
}

async function getTaskCounts(
  db: D1Database,
  projectId: string
): Promise<ProjectStatus['task_counts']> {
  const result = await db
    .prepare(
      `SELECT
        COUNT(*) as total,
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) as blocked
      FROM tasks WHERE project_id = ?`
    )
    .bind(projectId)
    .first();

  return {
    total: (result?.total as number) || 0,
    pending: (result?.pending as number) || 0,
    in_progress: (result?.in_progress as number) || 0,
    completed: (result?.completed as number) || 0,
    blocked: (result?.blocked as number) || 0,
  };
}

async function getProjectAgents(
  db: D1Database,
  projectId: string
): Promise<AgentSummary[]> {
  const results = await db
    .prepare(
      `SELECT a.agent_id, a.persona, a.status, a.current_task_id
       FROM agent_state a
       WHERE a.project_id = ?`
    )
    .bind(projectId)
    .all();

  return (results.results || []).map((row) => {
    const persona = parseJsonField<{ name?: string; role?: string }>(row.persona as string, {});
    return {
      id: row.agent_id as string,
      name: persona.name || row.agent_id as string,
      type: persona.role || 'agent',
      status: row.status as string,
      current_task_id: row.current_task_id as string | null,
    };
  });
}

async function unblockDependentTasks(
  db: D1Database,
  completedTaskId: string
): Promise<number> {
  // Find tasks blocked by this task
  const blockedTasks = await db
    .prepare(
      `SELECT id, blocked_by FROM tasks
       WHERE status = 'blocked'
       AND blocked_by LIKE ?`
    )
    .bind(`%${completedTaskId}%`)
    .all();

  let unblockedCount = 0;

  for (const task of blockedTasks.results || []) {
    const blockedBy = parseJsonField<string[]>(task.blocked_by as string, []);
    const updatedBlockedBy = blockedBy.filter((id) => id !== completedTaskId);

    if (updatedBlockedBy.length === 0) {
      // No more blockers, set to pending
      await db
        .prepare(
          `UPDATE tasks SET status = 'pending', blocked_by = '[]', updated_at = datetime('now')
           WHERE id = ?`
        )
        .bind(task.id)
        .run();
      unblockedCount++;
    } else {
      // Still has blockers, just update the blocked_by array
      await db
        .prepare(
          `UPDATE tasks SET blocked_by = ?, updated_at = datetime('now')
           WHERE id = ?`
        )
        .bind(JSON.stringify(updatedBlockedBy), task.id)
        .run();
    }
  }

  return unblockedCount;
}

// ============================================================================
// API Routes
// ============================================================================

const api = new Hono<{ Bindings: Env }>();

// Enable CORS for frontend
api.use(
  '/*',
  cors({
    origin: ['http://localhost:3003', 'http://localhost:3000', '*'],
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization'],
  })
);

// GET /api/health - Health check
api.get('/health', (c) => {
  const response: HealthResponse = {
    status: 'ok',
    timestamp: new Date().toISOString(),
  };
  return c.json(response);
});

// GET /api/projects/:id/status - Get project status with stats
api.get('/projects/:id/status', async (c) => {
  try {
    const projectId = c.req.param('id');

    if (!projectId) {
      return c.json(createErrorResponse('Project ID is required', 400), 400);
    }

    const project = await c.env.DB.prepare(
      `SELECT * FROM projects WHERE id = ?`
    )
      .bind(projectId)
      .first();

    if (!project) {
      return c.json(createErrorResponse('Project not found', 404), 404);
    }

    const [taskCounts, agents] = await Promise.all([
      getTaskCounts(c.env.DB, projectId),
      getProjectAgents(c.env.DB, projectId),
    ]);

    const response: ProjectStatus = {
      id: project.id as string,
      name: project.name as string,
      status: project.status as string,
      created_at: project.created_at as string,
      updated_at: project.updated_at as string,
      task_counts: taskCounts,
      agents,
    };

    return c.json(response);
  } catch (error) {
    console.error('Error fetching project status:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch project status'
      ),
      500
    );
  }
});

// GET /api/agents - List all agents
api.get('/agents', async (c) => {
  try {
    const results = await c.env.DB.prepare(
      `SELECT a.*, t.id as task_id, t.title as task_title, t.status as task_status, t.priority as task_priority
       FROM agent_state a
       LEFT JOIN tasks t ON a.current_task_id = t.id
       ORDER BY a.agent_id`
    ).all();

    const agents: AgentState[] = (results.results || []).map((row) => {
      const persona = parseJsonField<{ name?: string; role?: string }>(row.persona as string, {});
      return {
        id: row.agent_id as string,
        name: persona.name || row.agent_id as string,
        type: persona.role || 'agent',
        status: row.status as string,
        current_task_id: row.current_task_id as string | null,
        current_task: row.task_id
          ? {
              id: row.task_id as string,
              title: row.task_title as string,
              status: row.task_status as string,
              priority: row.task_priority as number,
            }
          : null,
        capabilities: [],
        created_at: row.created_at as string,
        updated_at: row.last_active as string || row.created_at as string,
      };
    });

    return c.json({ agents, count: agents.length });
  } catch (error) {
    console.error('Error fetching agents:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch agents'
      ),
      500
    );
  }
});

// GET /api/agents/:id/status - Get specific agent status
api.get('/agents/:id/status', async (c) => {
  try {
    const agentId = c.req.param('id');

    if (!agentId) {
      return c.json(createErrorResponse('Agent ID is required', 400), 400);
    }

    const agent = await c.env.DB.prepare(
      `SELECT a.*, t.id as task_id, t.title as task_title, t.status as task_status, t.priority as task_priority
       FROM agent_state a
       LEFT JOIN tasks t ON a.current_task_id = t.id
       WHERE a.agent_id = ?`
    )
      .bind(agentId)
      .first();

    if (!agent) {
      return c.json(createErrorResponse('Agent not found', 404), 404);
    }

    // Get recent work log
    const workLogResults = await c.env.DB.prepare(
      `SELECT * FROM work_log WHERE agent_id = ? ORDER BY created_at DESC LIMIT 20`
    )
      .bind(agentId)
      .all();

    const recentWorkLog: WorkLogEntry[] = (workLogResults.results || []).map(
      (row) => ({
        id: row.id as string,
        agent_id: row.agent_id as string,
        task_id: row.task_id as string | null,
        action: row.action as string,
        details: parseJsonField<Record<string, unknown>>(
          row.details as string,
          {}
        ),
        created_at: row.created_at as string,
      })
    );

    // Get stats
    const today = new Date().toISOString().split('T')[0];
    const statsResult = await c.env.DB.prepare(
      `SELECT
        (SELECT COUNT(*) FROM work_log WHERE agent_id = ? AND action = 'task_completed' AND date(created_at) = ?) as completed_today,
        (SELECT COUNT(*) FROM work_log WHERE agent_id = ? AND action = 'task_completed') as completed_total`
    )
      .bind(agentId, today, agentId)
      .first();

    const persona = parseJsonField<{ name?: string; role?: string }>(agent.persona as string, {});
    const response: AgentDetailedStatus = {
      id: agent.agent_id as string,
      name: persona.name || agent.agent_id as string,
      type: persona.role || 'agent',
      status: agent.status as string,
      current_task_id: agent.current_task_id as string | null,
      current_task: agent.task_id
        ? {
            id: agent.task_id as string,
            title: agent.task_title as string,
            status: agent.task_status as string,
            priority: agent.task_priority as number,
          }
        : null,
      capabilities: [],
      created_at: agent.created_at as string,
      updated_at: agent.last_active as string || agent.created_at as string,
      recent_work_log: recentWorkLog,
      stats: {
        tasks_completed_today: (statsResult?.completed_today as number) || 0,
        tasks_completed_total: (statsResult?.completed_total as number) || 0,
        average_task_duration_minutes: 0, // Would need additional tracking to calculate
      },
    };

    return c.json(response);
  } catch (error) {
    console.error('Error fetching agent status:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch agent status'
      ),
      500
    );
  }
});

// GET /api/tasks - List tasks with filters
api.get('/tasks', async (c) => {
  try {
    const projectId = c.req.query('project_id');
    const status = c.req.query('status');
    const assignedAgent = c.req.query('assigned_agent');
    const limit = parseInt(c.req.query('limit') || '50', 10);
    const offset = parseInt(c.req.query('offset') || '0', 10);

    // Build query dynamically based on filters
    let query = 'SELECT * FROM tasks WHERE 1=1';
    const params: (string | number)[] = [];

    if (projectId) {
      query += ' AND project_id = ?';
      params.push(projectId);
    }

    if (status) {
      query += ' AND status = ?';
      params.push(status);
    }

    if (assignedAgent) {
      query += ' AND assigned_agent_id = ?';
      params.push(assignedAgent);
    }

    query += ' ORDER BY priority DESC, created_at ASC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    const stmt = c.env.DB.prepare(query);
    const results = await stmt.bind(...params).all();

    const tasks: Task[] = (results.results || []).map((row) => ({
      id: row.id as string,
      project_id: row.project_id as string,
      title: row.title as string,
      description: row.description as string | null,
      status: row.status as string,
      priority: row.priority as number,
      assigned_agent_id: row.assigned_agent_id as string | null,
      parent_task_id: row.parent_task_id as string | null,
      blocked_by: parseJsonField<string[]>(row.blocked_by as string, []),
      context: parseJsonField<Record<string, unknown>>(
        row.context as string,
        {}
      ),
      created_at: row.created_at as string,
      updated_at: row.updated_at as string,
    }));

    return c.json({ tasks, count: tasks.length, limit, offset });
  } catch (error) {
    console.error('Error fetching tasks:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch tasks'
      ),
      500
    );
  }
});

// POST /api/tasks/:id/complete - Mark task as complete (for physical tasks)
api.post('/tasks/:id/complete', async (c) => {
  try {
    const taskId = c.req.param('id');

    if (!taskId) {
      return c.json(createErrorResponse('Task ID is required', 400), 400);
    }

    // Check if task exists
    const task = await c.env.DB.prepare(`SELECT * FROM tasks WHERE id = ?`)
      .bind(taskId)
      .first();

    if (!task) {
      return c.json(createErrorResponse('Task not found', 404), 404);
    }

    if (task.status === 'completed') {
      return c.json(createErrorResponse('Task is already completed', 400), 400);
    }

    // Update task status to completed
    await c.env.DB.prepare(
      `UPDATE tasks SET status = 'completed', updated_at = datetime('now') WHERE id = ?`
    )
      .bind(taskId)
      .run();

    // Unblock dependent tasks
    const unblockedCount = await unblockDependentTasks(c.env.DB, taskId);

    // Log the completion
    await c.env.DB.prepare(
      `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
       VALUES (?, ?, ?, 'task_completed', ?, datetime('now'))`
    )
      .bind(
        crypto.randomUUID(),
        task.assigned_agent_id || 'manual',
        taskId,
        JSON.stringify({ completed_by: 'api', unblocked_tasks: unblockedCount })
      )
      .run();

    return c.json({
      success: true,
      task_id: taskId,
      unblocked_tasks: unblockedCount,
      message: `Task completed. ${unblockedCount} dependent task(s) unblocked.`,
    });
  } catch (error) {
    console.error('Error completing task:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to complete task'
      ),
      500
    );
  }
});

// GET /api/work-log - Get recent work log
api.get('/work-log', async (c) => {
  try {
    const limit = parseInt(c.req.query('limit') || '50', 10);
    const agentId = c.req.query('agent_id');
    const taskId = c.req.query('task_id');

    let query = 'SELECT * FROM work_log WHERE 1=1';
    const params: (string | number)[] = [];

    if (agentId) {
      query += ' AND agent_id = ?';
      params.push(agentId);
    }

    if (taskId) {
      query += ' AND task_id = ?';
      params.push(taskId);
    }

    query += ' ORDER BY created_at DESC LIMIT ?';
    params.push(limit);

    const stmt = c.env.DB.prepare(query);
    const results = await stmt.bind(...params).all();

    const entries: WorkLogEntry[] = (results.results || []).map((row) => ({
      id: row.id as string,
      agent_id: row.agent_id as string,
      task_id: row.task_id as string | null,
      action: row.action as string,
      details: parseJsonField<Record<string, unknown>>(
        row.details as string,
        {}
      ),
      created_at: row.created_at as string,
    }));

    return c.json({ entries, count: entries.length });
  } catch (error) {
    console.error('Error fetching work log:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch work log'
      ),
      500
    );
  }
});

// POST /api/workflows/manager/trigger - Manually trigger manager
api.post('/workflows/manager/trigger', async (c) => {
  try {
    // In a real implementation, this would trigger a Cloudflare Workflow
    // For now, we'll just log and return success
    const body = await c.req.json().catch(() => ({}));
    const projectId = body.project_id;

    if (!projectId) {
      return c.json(createErrorResponse('project_id is required in body', 400), 400);
    }

    // Log the trigger attempt
    await c.env.DB.prepare(
      `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
       VALUES (?, 'system', NULL, 'manager_triggered', ?, datetime('now'))`
    )
      .bind(
        crypto.randomUUID(),
        JSON.stringify({ project_id: projectId, triggered_by: 'api' })
      )
      .run();

    return c.json({
      success: true,
      message: 'Manager workflow trigger logged',
      project_id: projectId,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Error triggering manager workflow:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to trigger manager workflow'
      ),
      500
    );
  }
});

// POST /api/workflows/worker/trigger - Trigger specific worker
api.post('/workflows/worker/trigger', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}));
    const { taskId, agentId } = body as { taskId?: string; agentId?: string };

    if (!taskId) {
      return c.json(createErrorResponse('taskId is required in body', 400), 400);
    }

    if (!agentId) {
      return c.json(createErrorResponse('agentId is required in body', 400), 400);
    }

    // Verify task exists
    const task = await c.env.DB.prepare(`SELECT * FROM tasks WHERE id = ?`)
      .bind(taskId)
      .first();

    if (!task) {
      return c.json(createErrorResponse('Task not found', 404), 404);
    }

    // Verify agent exists
    const agent = await c.env.DB.prepare(`SELECT * FROM agent_state WHERE id = ?`)
      .bind(agentId)
      .first();

    if (!agent) {
      return c.json(createErrorResponse('Agent not found', 404), 404);
    }

    // Assign task to agent and set to in_progress
    await c.env.DB.prepare(
      `UPDATE tasks SET assigned_agent_id = ?, status = 'in_progress', updated_at = datetime('now') WHERE id = ?`
    )
      .bind(agentId, taskId)
      .run();

    // Update agent's current task
    await c.env.DB.prepare(
      `UPDATE agent_state SET current_task_id = ?, status = 'working', last_active = datetime('now') WHERE agent_id = ?`
    )
      .bind(taskId, agentId)
      .run();

    // Log the trigger
    await c.env.DB.prepare(
      `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
       VALUES (?, ?, ?, 'worker_triggered', ?, datetime('now'))`
    )
      .bind(
        crypto.randomUUID(),
        agentId,
        taskId,
        JSON.stringify({ triggered_by: 'api' })
      )
      .run();

    return c.json({
      success: true,
      message: 'Worker workflow triggered',
      task_id: taskId,
      agent_id: agentId,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Error triggering worker workflow:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to trigger worker workflow'
      ),
      500
    );
  }
});

export default api;
