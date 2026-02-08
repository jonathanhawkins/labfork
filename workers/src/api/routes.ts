import { Hono } from 'hono';
import { cors } from 'hono/cors';
import compute from './compute';

// ============================================================================
// Types
// ============================================================================

interface Env {
  DB: D1Database;
  // AI binding removed - using distributed compute network
  MANAGER_WORKFLOW: Workflow;
  WORKER_WORKFLOW: Workflow;
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
  assigned_agent: string | null;
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

// CORS is handled at app level in index.ts
// No additional CORS middleware needed here

// GET /api/health - Health check
api.get('/health', (c) => {
  const response: HealthResponse = {
    status: 'ok',
    timestamp: new Date().toISOString(),
  };
  return c.json(response);
});

// GET /api/projects - List all projects with basic stats
api.get('/projects', async (c) => {
  try {
    const status = c.req.query('status');
    const limit = parseInt(c.req.query('limit') || '50', 10);
    const offset = parseInt(c.req.query('offset') || '0', 10);

    // Build query
    let query = 'SELECT * FROM projects WHERE 1=1';
    const params: (string | number)[] = [];

    if (status) {
      query += ' AND status = ?';
      params.push(status);
    }

    query += ' ORDER BY updated_at DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    const stmt = c.env.DB.prepare(query);
    const results = await stmt.bind(...params).all();

    // Get task counts for each project
    const projects = await Promise.all(
      (results.results || []).map(async (row) => {
        const taskCounts = await getTaskCounts(c.env.DB, row.id as string);
        const config = parseJsonField<Record<string, unknown>>(row.config as string, {});

        return {
          id: row.id as string,
          name: row.name as string,
          slug: row.slug as string,
          status: row.status as string,
          config,
          created_at: row.created_at as string,
          updated_at: row.updated_at as string,
          task_counts: taskCounts,
        };
      })
    );

    return c.json({
      projects,
      count: projects.length,
      limit,
      offset,
    });
  } catch (error) {
    console.error('Error fetching projects:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch projects'
      ),
      500
    );
  }
});

// POST /api/projects - Create a new project (for syncing labs from frontend)
api.post('/projects', async (c) => {
  try {
    const body = await c.req.json();
    const { id, name, slug, status, config, domainSlug, domainName, description, tags } = body;

    if (!name || !slug) {
      return c.json(createErrorResponse('Missing required fields: name, slug', 400), 400);
    }

    // Check if project already exists by slug
    const existing = await c.env.DB.prepare(
      `SELECT id FROM projects WHERE slug = ?`
    ).bind(slug).first();

    const now = new Date().toISOString();
    const projectId = id || `proj_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;

    // Build config object
    const projectConfig = {
      domainSlug: domainSlug || 'general',
      domainName: domainName || 'General',
      description: description || '',
      tags: tags || [],
      ...(config || {})
    };

    if (existing) {
      // Update existing project
      await c.env.DB.prepare(`
        UPDATE projects
        SET name = ?, config = ?, status = COALESCE(?, status), updated_at = ?
        WHERE slug = ?
      `).bind(name, JSON.stringify(projectConfig), status || null, now, slug).run();

      console.log(`[Projects] Updated project: ${slug}`);

      return c.json({
        success: true,
        project: {
          id: existing.id,
          name,
          slug,
          status: status || 'active',
          config: projectConfig
        },
        action: 'updated'
      });
    }

    // Create new project
    await c.env.DB.prepare(`
      INSERT INTO projects (id, name, slug, status, config, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(
      projectId,
      name,
      slug,
      status || 'active',
      JSON.stringify(projectConfig),
      now,
      now
    ).run();

    console.log(`[Projects] Created project: ${slug} (${projectId})`);

    return c.json({
      success: true,
      project: {
        id: projectId,
        name,
        slug,
        status: status || 'active',
        config: projectConfig
      },
      action: 'created'
    });
  } catch (error) {
    console.error('Error creating project:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to create project'
      ),
      500
    );
  }
});

// POST /api/projects/sync - Bulk sync labs from frontend
api.post('/projects/sync', async (c) => {
  try {
    const body = await c.req.json();
    const { labs } = body;

    if (!labs || !Array.isArray(labs)) {
      return c.json(createErrorResponse('Missing required field: labs (array)', 400), 400);
    }

    const results = {
      created: 0,
      updated: 0,
      failed: 0,
      projects: [] as { slug: string; action: string }[]
    };

    const now = new Date().toISOString();

    for (const lab of labs) {
      try {
        const { id, slug, name, description, domainSlug, domainName, tags, status } = lab;

        if (!slug || !name) {
          results.failed++;
          continue;
        }

        // Check if exists
        const existing = await c.env.DB.prepare(
          `SELECT id FROM projects WHERE slug = ?`
        ).bind(slug).first();

        const projectConfig = {
          domainSlug: domainSlug || 'general',
          domainName: domainName || 'General',
          description: description || '',
          tags: tags || [],
          labId: id // Store the lab ID for reference
        };

        if (existing) {
          await c.env.DB.prepare(`
            UPDATE projects
            SET name = ?, config = ?, updated_at = ?
            WHERE slug = ?
          `).bind(name, JSON.stringify(projectConfig), now, slug).run();

          results.updated++;
          results.projects.push({ slug, action: 'updated' });
        } else {
          const projectId = `proj_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;

          await c.env.DB.prepare(`
            INSERT INTO projects (id, name, slug, status, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
          `).bind(
            projectId,
            name,
            slug,
            status || 'active',
            JSON.stringify(projectConfig),
            now,
            now
          ).run();

          results.created++;
          results.projects.push({ slug, action: 'created' });
        }
      } catch (labError) {
        console.error(`Failed to sync lab ${lab.slug}:`, labError);
        results.failed++;
      }
    }

    console.log(`[Projects] Sync complete: ${results.created} created, ${results.updated} updated, ${results.failed} failed`);

    return c.json({
      success: true,
      ...results,
      timestamp: now
    });
  } catch (error) {
    console.error('Error syncing projects:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to sync projects'
      ),
      500
    );
  }
});

// GET /api/projects/:id - Get project details with task summary and active agents
api.get('/projects/:id', async (c) => {
  try {
    const projectId = c.req.param('id');

    if (!projectId) {
      return c.json(createErrorResponse('Project ID is required', 400), 400);
    }

    // Check if this is actually the /status route (Hono routes are matched in order)
    // Since :id could match 'status', we need to handle this case
    // Actually, the /status route is separate and will be matched first if it comes before

    const project = await c.env.DB.prepare(
      `SELECT * FROM projects WHERE id = ?`
    )
      .bind(projectId)
      .first();

    if (!project) {
      return c.json(createErrorResponse('Project not found', 404), 404);
    }

    // Get task counts, agents, and recent completed tasks in parallel
    const [taskCounts, agents, recentCompletedResults] = await Promise.all([
      getTaskCounts(c.env.DB, projectId),
      getProjectAgents(c.env.DB, projectId),
      c.env.DB.prepare(
        `SELECT id, title, description, status, priority, assigned_agent, updated_at
         FROM tasks
         WHERE project_id = ? AND status = 'completed'
         ORDER BY updated_at DESC
         LIMIT 5`
      )
        .bind(projectId)
        .all(),
    ]);

    const recentCompletedTasks = (recentCompletedResults.results || []).map((row) => ({
      id: row.id as string,
      title: row.title as string,
      description: row.description as string | null,
      status: row.status as string,
      priority: row.priority as number,
      assigned_agent: row.assigned_agent as string | null,
      completed_at: row.updated_at as string,
    }));

    // Filter active agents (working status or has current task)
    const activeAgents = agents.filter(
      (agent) => agent.status === 'working' || agent.current_task_id !== null
    );

    const config = parseJsonField<Record<string, unknown>>(project.config as string, {});

    return c.json({
      id: project.id as string,
      name: project.name as string,
      slug: project.slug as string,
      status: project.status as string,
      config,
      created_at: project.created_at as string,
      updated_at: project.updated_at as string,
      task_summary: taskCounts,
      recent_completed_tasks: recentCompletedTasks,
      active_agents: activeAgents,
    });
  } catch (error) {
    console.error('Error fetching project details:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch project details'
      ),
      500
    );
  }
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
      query += ' AND assigned_agent = ?';
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
      assigned_agent: row.assigned_agent as string | null,
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

    // Log the completion (non-blocking - work_log has FK constraint to agent_state)
    try {
      await c.env.DB.prepare(
        `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
         VALUES (?, ?, ?, 'task_completed', ?, datetime('now'))`
      )
        .bind(
          crypto.randomUUID(),
          task.assigned_agent || 'manual',
          taskId,
          JSON.stringify({ completed_by: 'api', unblocked_tasks: unblockedCount })
        )
        .run();
    } catch (logError) {
      // work_log has FK constraint on agent_id -> agent_state
      // Compute devices don't have agent_state entries, so this may fail
      console.log(`[Tasks] work_log insert skipped for task completion: ${taskId}`);
    }

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

// ============================================================================
// Project Tasks API - For Compute Agents (Main Work Driver)
// ============================================================================

/**
 * GET /api/tasks/pending - Get pending project tasks for compute agents
 *
 * This endpoint returns tasks from the main tasks table that are available
 * for compute agents to claim. These are the PRIMARY work driver (Firefly Network tasks).
 *
 * Query params:
 *   - deviceId: The compute device requesting tasks
 *   - projectId: Filter by project (optional)
 *   - limit: Max tasks to return (default 5)
 *   - requiresPhysical: Filter by physical requirement (default false = only non-physical)
 */
api.get('/tasks/pending', async (c) => {
  try {
    const deviceId = c.req.query('deviceId');
    const projectId = c.req.query('projectId');
    const limit = parseInt(c.req.query('limit') || '5', 10);
    const requiresPhysical = c.req.query('requiresPhysical') === 'true';

    // Build query for available tasks
    // Pending or blocked tasks that are now unblocked
    // Handle requires_physical: NULL and 0 both mean non-physical
    let query = `
      SELECT t.*, p.name as project_name, p.slug as project_slug
      FROM tasks t
      LEFT JOIN projects p ON t.project_id = p.id
      WHERE t.status IN ('pending')
        AND (t.assigned_agent IS NULL OR t.assigned_agent = '')
        AND (t.blocked_by IS NULL OR t.blocked_by = '' OR t.blocked_by = '[]')
    `;
    const params: (string | number)[] = [];

    // Filter by physical requirement
    if (requiresPhysical) {
      query += ' AND t.requires_physical = 1';
    } else {
      query += ' AND (t.requires_physical IS NULL OR t.requires_physical = 0)';
    }

    if (projectId) {
      query += ' AND t.project_id = ?';
      params.push(projectId);
    }

    query += ' ORDER BY t.priority DESC, t.created_at ASC LIMIT ?';
    params.push(limit);

    const stmt = c.env.DB.prepare(query);
    const results = await stmt.bind(...params).all();

    const tasks = (results.results || []).map((row) => ({
      id: row.id as string,
      project_id: row.project_id as string,
      project_name: row.project_name as string,
      project_slug: row.project_slug as string,
      title: row.title as string,
      description: row.description as string,
      status: row.status as string,
      priority: row.priority as number,
      requires_physical: Boolean(row.requires_physical),
      progress: row.progress as number,
      created_at: row.created_at as string,
    }));

    return c.json({
      tasks,
      count: tasks.length,
      device_id: deviceId,
      message: tasks.length > 0 ? 'Tasks available' : 'No pending tasks',
    });
  } catch (error) {
    console.error('[Tasks] Get pending error:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to get pending tasks'
      ),
      500
    );
  }
});

/**
 * POST /api/tasks/:id/claim - Claim a project task for execution
 *
 * Request body:
 *   - deviceId: The compute device claiming the task
 *   - agentName: Optional agent name for display
 */
api.post('/tasks/:id/claim', async (c) => {
  try {
    const taskId = c.req.param('id');
    const body = await c.req.json().catch(() => ({}));
    const { deviceId, agentName } = body as { deviceId?: string; agentName?: string };

    if (!taskId) {
      return c.json(createErrorResponse('Task ID is required', 400), 400);
    }

    if (!deviceId) {
      return c.json(createErrorResponse('deviceId is required', 400), 400);
    }

    // Verify task exists and is available (join with projects to get project_name)
    const task = await c.env.DB.prepare(`
      SELECT t.*, p.name as project_name, p.slug as project_slug
      FROM tasks t
      LEFT JOIN projects p ON t.project_id = p.id
      WHERE t.id = ?
    `)
      .bind(taskId)
      .first();

    if (!task) {
      return c.json(createErrorResponse('Task not found', 404), 404);
    }

    if (task.status !== 'pending') {
      return c.json(createErrorResponse(`Task is not pending (status: ${task.status})`, 400), 400);
    }

    if (task.assigned_agent && task.assigned_agent !== '') {
      return c.json(createErrorResponse('Task is already claimed', 400), 400);
    }

    // Verify device exists
    const device = await c.env.DB.prepare(`SELECT * FROM compute_devices WHERE id = ?`)
      .bind(deviceId)
      .first();

    if (!device) {
      return c.json(createErrorResponse('Device not registered', 404), 404);
    }

    // Claim the task
    const assignedAgent = agentName || `device-${deviceId.substring(0, 8)}`;
    await c.env.DB.prepare(
      `UPDATE tasks SET
        assigned_agent = ?,
        status = 'in_progress',
        updated_at = datetime('now')
       WHERE id = ?`
    )
      .bind(assignedAgent, taskId)
      .run();

    // Log the claim (non-blocking - work_log has FK constraint to agent_state)
    try {
      await c.env.DB.prepare(
        `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
         VALUES (?, ?, ?, 'task_claimed', ?, datetime('now'))`
      )
        .bind(
          crypto.randomUUID(),
          assignedAgent,
          taskId,
          JSON.stringify({ device_id: deviceId, agent_name: agentName })
        )
        .run();
    } catch (logError) {
      // work_log has FK constraint on agent_id -> agent_state
      // Compute devices don't have agent_state entries, so this may fail
      console.log(`[Tasks] work_log insert skipped for compute device: ${deviceId}`);
    }

    return c.json({
      success: true,
      task: {
        id: task.id,
        title: task.title,
        description: task.description,
        priority: task.priority,
        project_id: task.project_id,
        project_name: task.project_name,
        project_slug: task.project_slug,
      },
      assigned_agent: assignedAgent,
      message: 'Task claimed successfully',
    });
  } catch (error) {
    console.error('[Tasks] Claim error:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to claim task'
      ),
      500
    );
  }
});

/**
 * POST /api/tasks/:id/progress - Update task progress
 *
 * Request body:
 *   - deviceId: The compute device updating progress
 *   - progress: Progress percentage (0-100)
 *   - status: Optional status update ('in_progress', 'blocked')
 *   - result: Optional intermediate result
 */
api.post('/tasks/:id/progress', async (c) => {
  try {
    const taskId = c.req.param('id');
    const body = await c.req.json().catch(() => ({}));
    const { deviceId, progress, status, result } = body as {
      deviceId?: string;
      progress?: number;
      status?: string;
      result?: string;
    };

    if (!taskId) {
      return c.json(createErrorResponse('Task ID is required', 400), 400);
    }

    // Verify task exists
    const task = await c.env.DB.prepare(`SELECT * FROM tasks WHERE id = ?`)
      .bind(taskId)
      .first();

    if (!task) {
      return c.json(createErrorResponse('Task not found', 404), 404);
    }

    // Build update query
    const updates: string[] = ['updated_at = datetime(\'now\')'];
    const params: (string | number)[] = [];

    if (progress !== undefined && progress >= 0 && progress <= 100) {
      updates.push('progress = ?');
      params.push(progress);
    }

    if (status && ['in_progress', 'blocked', 'pending'].includes(status)) {
      updates.push('status = ?');
      params.push(status);
    }

    if (result) {
      updates.push('result = ?');
      params.push(result);
    }

    params.push(taskId);

    await c.env.DB.prepare(
      `UPDATE tasks SET ${updates.join(', ')} WHERE id = ?`
    )
      .bind(...params)
      .run();

    // Log progress update (non-blocking - work_log has FK constraint)
    try {
      await c.env.DB.prepare(
        `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
         VALUES (?, ?, ?, 'task_progress', ?, datetime('now'))`
      )
        .bind(
          crypto.randomUUID(),
          task.assigned_agent || deviceId || 'unknown',
          taskId,
          JSON.stringify({ progress, status, device_id: deviceId })
        )
        .run();
    } catch (logError) {
      // work_log has FK constraint, may fail for compute devices
      console.log(`[Tasks] work_log insert skipped for progress update`);
    }

    return c.json({
      success: true,
      task_id: taskId,
      progress: progress ?? task.progress,
      status: status ?? task.status,
    });
  } catch (error) {
    console.error('[Tasks] Progress update error:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to update progress'
      ),
      500
    );
  }
});

/**
 * PATCH /api/tasks/:id - Update task properties
 *
 * Request body:
 *   - status: New status ('pending', 'in_progress', 'completed', 'blocked')
 *   - priority: New priority (1-10)
 *   - requires_physical: Whether task requires physical work
 *   - blocked_by: Array of blocking task IDs or JSON string
 *   - description: Updated description
 *   - result: Task result/output
 */
api.patch('/tasks/:id', async (c) => {
  try {
    const taskId = c.req.param('id');
    const body = await c.req.json().catch(() => ({}));
    const { status, priority, requires_physical, blocked_by, description, result } = body as {
      status?: string;
      priority?: number;
      requires_physical?: boolean;
      blocked_by?: string[] | string;
      description?: string;
      result?: string;
    };

    if (!taskId) {
      return c.json(createErrorResponse('Task ID is required', 400), 400);
    }

    // Verify task exists
    const task = await c.env.DB.prepare(`SELECT * FROM tasks WHERE id = ?`)
      .bind(taskId)
      .first();

    if (!task) {
      return c.json(createErrorResponse('Task not found', 404), 404);
    }

    // Build update query
    const updates: string[] = ['updated_at = datetime(\'now\')'];
    const params: (string | number)[] = [];

    if (status && ['pending', 'in_progress', 'completed', 'blocked'].includes(status)) {
      updates.push('status = ?');
      params.push(status);
    }

    if (priority !== undefined && priority >= 1 && priority <= 10) {
      updates.push('priority = ?');
      params.push(priority);
    }

    if (requires_physical !== undefined) {
      updates.push('requires_physical = ?');
      params.push(requires_physical ? 1 : 0);
    }

    if (blocked_by !== undefined) {
      const blockedByStr = Array.isArray(blocked_by) ? JSON.stringify(blocked_by) : blocked_by;
      updates.push('blocked_by = ?');
      params.push(blockedByStr || '');
    }

    if (description !== undefined) {
      updates.push('description = ?');
      params.push(description);
    }

    if (result !== undefined) {
      updates.push('result = ?');
      params.push(result);
    }

    if (updates.length === 1) {
      return c.json(createErrorResponse('No valid fields to update', 400), 400);
    }

    params.push(taskId);

    await c.env.DB.prepare(
      `UPDATE tasks SET ${updates.join(', ')} WHERE id = ?`
    )
      .bind(...params)
      .run();

    // Fetch updated task
    const updatedTask = await c.env.DB.prepare(`SELECT * FROM tasks WHERE id = ?`)
      .bind(taskId)
      .first();

    console.log(`[Tasks] Updated task ${taskId}:`, { status, priority, requires_physical });

    return c.json({
      success: true,
      task: updatedTask,
    });
  } catch (error) {
    console.error('[Tasks] Update error:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to update task'
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

// POST /api/workflows/manager/trigger - Manually trigger manager workflow
api.post('/workflows/manager/trigger', async (c) => {
  try {
    const body = await c.req.json().catch(() => ({}));
    const projectId = body.project_id;

    if (!projectId) {
      return c.json(createErrorResponse('project_id is required in body', 400), 400);
    }

    // Actually trigger the Manager Workflow
    const instance = await c.env.MANAGER_WORKFLOW.create();

    // Log the trigger
    await c.env.DB.prepare(
      `INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
       VALUES (?, 'system', NULL, 'manager_triggered', ?, datetime('now'))`
    )
      .bind(
        crypto.randomUUID(),
        JSON.stringify({ project_id: projectId, triggered_by: 'api', workflow_id: instance.id })
      )
      .run();

    return c.json({
      success: true,
      message: 'Manager workflow started',
      workflow_id: instance.id,
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
    const agent = await c.env.DB.prepare(`SELECT * FROM agent_state WHERE agent_id = ?`)
      .bind(agentId)
      .first();

    if (!agent) {
      return c.json(createErrorResponse('Agent not found', 404), 404);
    }

    // Assign task to agent and set to in_progress
    await c.env.DB.prepare(
      `UPDATE tasks SET assigned_agent = ?, status = 'in_progress', updated_at = datetime('now') WHERE id = ?`
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

// Mount compute network routes under /compute
api.route('/compute', compute);

// ============================================================================
// Research Sync Routes - For 4090 Local Research Integration
// ============================================================================

/**
 * POST /api/research/sync - Sync research results from 4090 to Workers
 *
 * This endpoint receives research findings from the hybrid agent daemon
 * and stores them in the database for display on the /watch page.
 *
 * Request body:
 * {
 *   deviceId: string,
 *   objective: { id, title, description },
 *   results: { success, output, duration_minutes },
 *   labId: string
 * }
 */
api.post('/research/sync', async (c) => {
  try {
    const body = await c.req.json();
    const { deviceId, objective, results, labId } = body;

    if (!deviceId || !objective || !results) {
      return c.json(createErrorResponse('Missing required fields: deviceId, objective, results', 400), 400);
    }

    const id = crypto.randomUUID();
    const now = new Date().toISOString();

    // Store research result in a dedicated table (create if not exists via schema)
    await c.env.DB.prepare(`
      INSERT INTO research_results (
        id, device_id, lab_id, objective_id, objective_title,
        objective_description, success, output, duration_minutes, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      id,
      deviceId,
      labId || 'voice-clone',
      objective.id || 'unknown',
      objective.title || 'Unknown Research',
      objective.description || '',
      results.success ? 1 : 0,
      results.output || '',
      results.duration_minutes || 0,
      now
    ).run();

    // Also log to work_log for visibility
    await c.env.DB.prepare(`
      INSERT INTO work_log (id, agent_id, task_id, action, details, created_at)
      VALUES (?, ?, NULL, 'research_synced', ?, datetime('now'))
    `).bind(
      crypto.randomUUID(),
      deviceId,
      JSON.stringify({
        objective_id: objective.id,
        objective_title: objective.title,
        success: results.success,
        duration_minutes: results.duration_minutes
      })
    ).run();

    console.log(`[Research] Synced result from ${deviceId}: ${objective.title} (${results.success ? 'success' : 'failed'})`);

    return c.json({
      success: true,
      id,
      message: 'Research result synced',
      timestamp: now
    });
  } catch (error) {
    console.error('Error syncing research:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to sync research'
      ),
      500
    );
  }
});

/**
 * GET /api/research/results - Get research results
 *
 * Query params:
 *   labId: string (optional, defaults to 'voice-clone')
 *   limit: number (optional, default 20)
 */
api.get('/research/results', async (c) => {
  try {
    const labId = c.req.query('labId') || 'voice-clone';
    const limit = parseInt(c.req.query('limit') || '20', 10);

    const results = await c.env.DB.prepare(`
      SELECT * FROM research_results
      WHERE lab_id = ?
      ORDER BY created_at DESC
      LIMIT ?
    `).bind(labId, limit).all();

    const items = (results.results || []).map((row) => ({
      id: row.id,
      deviceId: row.device_id,
      labId: row.lab_id,
      objective: {
        id: row.objective_id,
        title: row.objective_title,
        description: row.objective_description
      },
      success: row.success === 1,
      output: row.output,
      durationMinutes: row.duration_minutes,
      createdAt: row.created_at
    }));

    return c.json({
      results: items,
      count: items.length,
      labId
    });
  } catch (error) {
    console.error('Error fetching research results:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to fetch research results'
      ),
      500
    );
  }
});

/**
 * GET /api/research/objectives - Get research objectives queue
 *
 * Returns the current research objectives that devices can work on.
 * Query params:
 *   labId: string (optional, defaults to 'voice-clone')
 */
api.get('/research/objectives', async (c) => {
  try {
    const labId = c.req.query('labId') || 'voice-clone';

    // Try to get objectives from database, or return defaults
    const results = await c.env.DB.prepare(`
      SELECT * FROM research_objectives
      WHERE lab_id = ? AND status = 'pending'
      ORDER BY priority DESC
    `).bind(labId).all();

    const items = (results.results || []).map((row) => ({
      id: row.id,
      title: row.title,
      description: row.description,
      priority: row.priority,
      status: row.status,
      tags: parseJsonField<string[]>(row.tags as string, [])
    }));

    // If no objectives in DB, return default ones
    if (items.length === 0) {
      return c.json({
        objectives: [
          {
            id: 'prosody-conditioning-review',
            title: 'Review prosody conditioning approaches',
            description: 'Research and document different approaches to prosody conditioning in TTS models.',
            priority: 8,
            status: 'pending',
            tags: ['research', 'prosody', 'tts']
          },
          {
            id: 'csm-1b-finetuning-guide',
            title: 'Document CSM-1B fine-tuning process',
            description: 'Create a comprehensive guide for fine-tuning CSM-1B with emotional prosody.',
            priority: 7,
            status: 'pending',
            tags: ['documentation', 'csm-1b', 'training']
          }
        ],
        count: 2,
        labId,
        source: 'defaults'
      });
    }

    return c.json({
      objectives: items,
      count: items.length,
      labId,
      source: 'database'
    });
  } catch (error) {
    // If table doesn't exist yet, return defaults
    return c.json({
      objectives: [
        {
          id: 'prosody-conditioning-review',
          title: 'Review prosody conditioning approaches',
          description: 'Research and document different approaches to prosody conditioning in TTS models.',
          priority: 8,
          status: 'pending',
          tags: ['research', 'prosody', 'tts']
        }
      ],
      count: 1,
      labId: c.req.query('labId') || 'voice-clone',
      source: 'defaults'
    });
  }
});

/**
 * POST /api/research/objectives - Add a new research objective
 */
api.post('/research/objectives', async (c) => {
  try {
    const body = await c.req.json();
    const { id, title, description, priority, tags, labId } = body;

    if (!title || !description) {
      return c.json(createErrorResponse('Missing required fields: title, description', 400), 400);
    }

    const objectiveId = id || crypto.randomUUID();
    const now = new Date().toISOString();

    await c.env.DB.prepare(`
      INSERT INTO research_objectives (
        id, lab_id, title, description, priority, status, tags, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
    `).bind(
      objectiveId,
      labId || 'voice-clone',
      title,
      description,
      priority || 5,
      JSON.stringify(tags || []),
      now,
      now
    ).run();

    console.log(`[Research] Created objective: ${title}`);

    return c.json({
      success: true,
      id: objectiveId,
      message: 'Research objective created'
    });
  } catch (error) {
    console.error('Error creating research objective:', error);
    return c.json(
      createErrorResponse(
        error instanceof Error ? error.message : 'Failed to create research objective'
      ),
      500
    );
  }
});

export default api;
