/**
 * Nudge Engine API
 *
 * Three core endpoints: /register, /poll, /report
 * Plus admin endpoints for manual task creation and inspection.
 */

import { Hono, type Context } from 'hono';
import type { Env } from './index';
import { generateId, generateToken, now, parseJson } from './utils';

const api = new Hono<{ Bindings: Env }>();

// ============================================================================
// Auth helpers
// ============================================================================

/** Validate worker auth token. Returns worker ID or null. */
async function authenticate(c: Context<{ Bindings: Env }>): Promise<string | null> {
  const header = c.req.header('Authorization');
  if (!header?.startsWith('Bearer ')) return null;
  const token = header.slice(7);
  if (!token) return null;

  const row = await c.env.DB.prepare(
    'SELECT id FROM workers WHERE auth_token = ?'
  ).bind(token).first<{ id: string }>();
  return row?.id ?? null;
}

/** Validate admin API key. Pass-through in dev mode (no key configured). */
function isAdmin(c: Context<{ Bindings: Env }>): boolean {
  const adminKey = c.env.ADMIN_API_KEY;
  if (!adminKey) return true; // dev mode

  const header = c.req.header('Authorization');
  if (!header?.startsWith('Bearer ')) return false;
  const token = header.slice(7);
  if (token.length !== adminKey.length) return false;

  let mismatch = 0;
  for (let i = 0; i < token.length; i++) {
    mismatch |= token.charCodeAt(i) ^ adminKey.charCodeAt(i);
  }
  return mismatch === 0;
}

// ============================================================================
// Core Protocol: /register, /poll, /report
// ============================================================================

/**
 * POST /register — Register a new worker
 *
 * Body: { name: string, type: string, capabilities?: string[] }
 * Returns: { id, token, registered }
 */
api.post('/register', async (c) => {
  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid JSON body' }, 400);
  const { name, type, capabilities } = body;

  if (!name || !type) {
    return c.json({ error: 'name and type are required' }, 400);
  }

  const validTypes = ['claude-code', 'codex', 'ollama', 'custom'];
  if (!validTypes.includes(type)) {
    return c.json({ error: `type must be one of: ${validTypes.join(', ')}` }, 400);
  }

  const id = generateId('w');
  const token = generateToken();
  const ts = now();

  await c.env.DB.prepare(`
    INSERT INTO workers (id, name, type, capabilities, status, auth_token, last_heartbeat, created_at, updated_at)
    VALUES (?, ?, ?, ?, 'online', ?, ?, ?, ?)
  `).bind(
    id, name, type,
    JSON.stringify(capabilities || []),
    token, ts, ts, ts
  ).run();

  // Log registration
  await c.env.DB.prepare(`
    INSERT INTO work_log (id, worker_id, event, detail, created_at)
    VALUES (?, ?, 'registered', ?, ?)
  `).bind(generateId('log'), id, JSON.stringify({ name, type }), ts).run();

  return c.json({ id, token, registered: true });
});

/**
 * POST /poll — Heartbeat + request work
 *
 * Auth: Bearer <token>
 * Body (optional): { status?: string }
 * Returns: { task: null } or { task: { id, action, description, context, constraints } }
 */
api.post('/poll', async (c) => {
  const workerId = await authenticate(c);
  if (!workerId) return c.json({ error: 'Authorization required. Include Authorization: Bearer <token>' }, 401);

  const body = await c.req.json().catch(() => ({}));
  const ts = now();

  // Update heartbeat — preserve 'busy' status if worker has an active task
  await c.env.DB.prepare(`
    UPDATE workers SET last_heartbeat = ?,
      status = CASE WHEN current_task_id IS NOT NULL THEN 'busy' ELSE 'online' END,
      updated_at = ? WHERE id = ?
  `).bind(ts, ts, workerId).run();

  // Check if worker already has a task assigned
  const worker = await c.env.DB.prepare(
    'SELECT current_task_id FROM workers WHERE id = ?'
  ).bind(workerId).first<{ current_task_id: string | null }>();

  if (worker?.current_task_id) {
    const existing = await c.env.DB.prepare(
      'SELECT id, action, description, context, constraints, priority FROM tasks WHERE id = ? AND status = ?'
    ).bind(worker.current_task_id, 'assigned').first();

    if (existing) {
      return c.json({
        task: {
          id: existing.id,
          action: existing.action,
          description: existing.description,
          context: parseJson(existing.context as string | null, null),
          constraints: parseJson(existing.constraints as string | null, null),
          priority: existing.priority,
        },
      });
    }
    // Task no longer assigned (completed/failed elsewhere) — clear stale ref
    await c.env.DB.prepare(
      'UPDATE workers SET current_task_id = NULL WHERE id = ?'
    ).bind(workerId).run();
  }

  // Find highest-priority pending task, filtered by worker capabilities
  const workerCaps = await c.env.DB.prepare(
    'SELECT capabilities FROM workers WHERE id = ?'
  ).bind(workerId).first<{ capabilities: string }>();
  const capsStr = workerCaps?.capabilities || '[]';

  // Match tasks where:
  //   - no required_capability on task, OR
  //   - worker has empty capabilities '[]' (universal — backwards compat), OR
  //   - worker's capabilities contain the required one
  // Skip parent tasks (they have children and are just tracking containers)
  const task = await c.env.DB.prepare(`
    SELECT t.id, t.action, t.description, t.context, t.constraints, t.priority, t.parent_task_id
    FROM tasks t
    WHERE t.status = 'pending'
      AND (t.required_capability IS NULL OR ? = '[]' OR instr(?, '"' || t.required_capability || '"') > 0)
      AND NOT EXISTS (SELECT 1 FROM tasks c WHERE c.parent_task_id = t.id)
    ORDER BY t.priority DESC, t.created_at ASC
    LIMIT 1
  `).bind(capsStr, capsStr).first();

  if (!task) {
    return c.json({ task: null });
  }

  // Atomic assign: only succeeds if still pending
  const assign = await c.env.DB.prepare(`
    UPDATE tasks SET status = 'assigned', assigned_worker_id = ?, assigned_at = ?, attempts = attempts + 1
    WHERE id = ? AND status = 'pending'
  `).bind(workerId, ts, task.id).run();

  if (!assign.meta?.changes || assign.meta.changes === 0) {
    // Race condition — another worker grabbed it
    return c.json({ task: null });
  }

  // Mark worker busy
  await c.env.DB.prepare(`
    UPDATE workers SET status = 'busy', current_task_id = ?, updated_at = ? WHERE id = ?
  `).bind(task.id, ts, workerId).run();

  // Log assignment
  await c.env.DB.prepare(`
    INSERT INTO work_log (id, worker_id, task_id, event, created_at)
    VALUES (?, ?, ?, 'assigned', ?)
  `).bind(generateId('log'), workerId, task.id, ts).run();

  return c.json({
    task: {
      id: task.id,
      action: task.action,
      description: task.description,
      context: parseJson(task.context as string | null, null),
      constraints: parseJson(task.constraints as string | null, null),
      priority: task.priority,
      parent_task_id: (task as Record<string, unknown>).parent_task_id || null,
    },
  });
});

/**
 * POST /report — Report task result
 *
 * Auth: Bearer <token>
 * Body: { taskId, success, result?: { summary, filesChanged, prUrl, testsPass }, error?: string }
 */
api.post('/report', async (c) => {
  const workerId = await authenticate(c);
  if (!workerId) return c.json({ error: 'Authorization required. Include Authorization: Bearer <token>' }, 401);

  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid JSON body' }, 400);
  const { taskId, success, result, error } = body;

  if (!taskId || typeof success !== 'boolean') {
    return c.json({ error: 'taskId and success (boolean) are required' }, 400);
  }

  const ts = now();

  // Get task
  const task = await c.env.DB.prepare(
    'SELECT id, status, assigned_worker_id FROM tasks WHERE id = ?'
  ).bind(taskId).first<{ id: string; status: string; assigned_worker_id: string | null }>();

  if (!task) {
    return c.json({ error: 'Task not found' }, 404);
  }
  if (task.status === 'completed' || task.status === 'failed') {
    return c.json({ error: 'Task already resolved', status: task.status }, 409);
  }
  if (task.assigned_worker_id !== workerId) {
    return c.json({ error: 'Task not assigned to this worker' }, 403);
  }

  const newStatus = success ? 'completed' : 'failed';

  // Update task
  await c.env.DB.prepare(`
    UPDATE tasks SET status = ?, result = ?, error = ?, completed_at = ? WHERE id = ?
  `).bind(
    newStatus,
    result ? JSON.stringify(result) : null,
    error || null,
    ts,
    taskId
  ).run();

  // Free worker
  await c.env.DB.prepare(`
    UPDATE workers SET status = 'online', current_task_id = NULL, updated_at = ?,
      tasks_completed = tasks_completed + ?,
      tasks_failed = tasks_failed + ?
    WHERE id = ?
  `).bind(ts, success ? 1 : 0, success ? 0 : 1, workerId).run();

  // Log
  await c.env.DB.prepare(`
    INSERT INTO work_log (id, worker_id, task_id, event, detail, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).bind(
    generateId('log'), workerId, taskId, newStatus,
    JSON.stringify({ success, summary: result?.summary }),
    ts
  ).run();

  return c.json({ success: true, taskId, status: newStatus });
});

// ============================================================================
// Observations: workers report what they see
// ============================================================================

/**
 * POST /observe — Submit an observation
 *
 * Auth: Bearer <token>
 * Body: { type: string, data: object }
 */
api.post('/observe', async (c) => {
  // Accept both worker tokens and admin API key
  const workerId = await authenticate(c);
  const admin = isAdmin(c);
  if (!workerId && !admin) return c.json({ error: 'Authorization required. Include Authorization: Bearer <token>' }, 401);

  const observedBy = workerId || 'admin';

  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid JSON body' }, 400);
  const { type, data } = body;

  if (!type || !data) {
    return c.json({ error: 'type and data are required' }, 400);
  }

  const id = generateId('obs');
  const ts = now();

  await c.env.DB.prepare(`
    INSERT INTO observations (id, type, data, observed_by, created_at) VALUES (?, ?, ?, ?, ?)
  `).bind(id, type, JSON.stringify(data), observedBy, ts).run();

  // Log
  await c.env.DB.prepare(`
    INSERT INTO work_log (id, worker_id, event, detail, created_at) VALUES (?, ?, 'observation', ?, ?)
  `).bind(generateId('log'), observedBy, JSON.stringify({ type }), ts).run();

  return c.json({ success: true, observationId: id });
});

// ============================================================================
// Admin: manual task creation, inspection, stats
// ============================================================================

/**
 * POST /tasks — Create a task manually
 *
 * Auth: Admin API key
 * Body: { action, description, context?, constraints?, priority? }
 */
api.post('/tasks', async (c) => {
  const workerId = await authenticate(c);
  const admin = isAdmin(c);
  if (!workerId && !admin) return c.json({ error: 'Authorization required (admin key or worker token)' }, 401);
  const taskSource = admin ? 'human' : 'worker';

  const body = await c.req.json().catch(() => null);
  if (!body) return c.json({ error: 'Invalid JSON body' }, 400);
  const { action, description, context, constraints, priority, required_capability, parent_task_id } = body;

  if (!action || !description) {
    return c.json({ error: 'action and description are required' }, 400);
  }

  const validCapabilities = ['code', 'monitor', 'advisory'];
  if (required_capability && !validCapabilities.includes(required_capability)) {
    return c.json({ error: `required_capability must be one of: ${validCapabilities.join(', ')}` }, 400);
  }

  // Validate parent_task_id exists if provided
  if (parent_task_id) {
    const parent = await c.env.DB.prepare(
      'SELECT id FROM tasks WHERE id = ?'
    ).bind(parent_task_id).first();
    if (!parent) {
      return c.json({ error: 'parent_task_id not found' }, 400);
    }
  }

  const id = generateId('t');
  const ts = now();

  await c.env.DB.prepare(`
    INSERT INTO tasks (id, action, description, context, constraints, status, priority, required_capability, source, parent_task_id, created_at)
    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
  `).bind(
    id, action, description,
    context ? JSON.stringify(context) : null,
    constraints ? JSON.stringify(constraints) : null,
    priority || 5,
    required_capability || null,
    taskSource,
    parent_task_id || null,
    ts
  ).run();

  await c.env.DB.prepare(`
    INSERT INTO work_log (id, event, detail, created_at) VALUES (?, 'created', ?, ?)
  `).bind(generateId('log'), JSON.stringify({ taskId: id, action, source: taskSource, parentTaskId: parent_task_id || null }), ts).run();

  return c.json({ success: true, task: { id, action, status: 'pending', priority: priority || 5, parent_task_id: parent_task_id || null } });
});

/**
 * GET /tasks — List tasks
 *
 * Query: status?, limit?, offset?
 */
api.get('/tasks', async (c) => {
  const status = c.req.query('status');
  const parentId = c.req.query('parent_task_id');
  const limit = Math.min(parseInt(c.req.query('limit') || '50', 10), 200);
  const offset = parseInt(c.req.query('offset') || '0', 10);

  let query = 'SELECT id, action, description, status, priority, required_capability, assigned_worker_id, result, source, attempts, error, parent_task_id, created_at, assigned_at, completed_at FROM tasks';
  const conditions: string[] = [];
  const params: (string | number)[] = [];

  if (status) {
    conditions.push('status = ?');
    params.push(status);
  }
  if (parentId) {
    conditions.push('parent_task_id = ?');
    params.push(parentId);
  }

  if (conditions.length > 0) {
    query += ' WHERE ' + conditions.join(' AND ');
  }

  query += ' ORDER BY priority DESC, created_at DESC LIMIT ? OFFSET ?';
  params.push(limit, offset);

  const result = await c.env.DB.prepare(query).bind(...params).all();

  return c.json({ tasks: result.results || [], count: (result.results || []).length });
});

/**
 * GET /tasks/:id — Get a single task with full detail
 */
api.get('/tasks/:id', async (c) => {
  const taskId = c.req.param('id');
  const task = await c.env.DB.prepare(`
    SELECT id, action, description, context, constraints, status, priority,
           required_capability, assigned_worker_id, result, error, source, attempts,
           timeout_minutes, parent_task_id, created_at, assigned_at, completed_at
    FROM tasks WHERE id = ?
  `).bind(taskId).first();

  if (!task) return c.json({ error: 'Task not found' }, 404);

  // If this task has children, include a summary
  const children = await c.env.DB.prepare(`
    SELECT id, action, status FROM tasks WHERE parent_task_id = ? ORDER BY created_at ASC
  `).bind(taskId).all<{ id: string; action: string; status: string }>();

  const childrenList = children.results || [];

  return c.json({
    ...task,
    context: parseJson(task.context as string | null, null),
    constraints: parseJson(task.constraints as string | null, null),
    result: parseJson(task.result as string | null, null),
    children: childrenList.length > 0 ? childrenList : undefined,
    childrenSummary: childrenList.length > 0 ? {
      total: childrenList.length,
      completed: childrenList.filter(ch => ch.status === 'completed').length,
      failed: childrenList.filter(ch => ch.status === 'failed').length,
      pending: childrenList.filter(ch => ch.status === 'pending').length,
      assigned: childrenList.filter(ch => ch.status === 'assigned').length,
    } : undefined,
  });
});

/**
 * GET /tasks/:id/children — List subtasks of a parent task
 */
api.get('/tasks/:id/children', async (c) => {
  const parentId = c.req.param('id');

  const parent = await c.env.DB.prepare(
    'SELECT id FROM tasks WHERE id = ?'
  ).bind(parentId).first();
  if (!parent) return c.json({ error: 'Parent task not found' }, 404);

  const result = await c.env.DB.prepare(`
    SELECT id, action, description, status, priority, required_capability, assigned_worker_id,
           result, error, attempts, created_at, assigned_at, completed_at
    FROM tasks WHERE parent_task_id = ?
    ORDER BY created_at ASC
  `).bind(parentId).all();

  const children = result.results || [];

  return c.json({
    parentId,
    children,
    summary: {
      total: children.length,
      completed: children.filter(ch => ch.status === 'completed').length,
      failed: children.filter(ch => ch.status === 'failed').length,
      pending: children.filter(ch => ch.status === 'pending').length,
      assigned: children.filter(ch => ch.status === 'assigned').length,
    },
  });
});

/**
 * DELETE /tasks/:id — Delete a task (admin)
 */
api.delete('/tasks/:id', async (c) => {
  if (!isAdmin(c)) return c.json({ error: 'Admin API key required' }, 401);

  const taskId = c.req.param('id');

  // Free worker if task was assigned
  const task = await c.env.DB.prepare(
    'SELECT assigned_worker_id, status FROM tasks WHERE id = ?'
  ).bind(taskId).first<{ assigned_worker_id: string | null; status: string }>();

  if (!task) return c.json({ error: 'Task not found' }, 404);

  if (task.assigned_worker_id && task.status === 'assigned') {
    await c.env.DB.prepare(`
      UPDATE workers SET status = 'online', current_task_id = NULL, updated_at = ? WHERE id = ?
    `).bind(now(), task.assigned_worker_id).run();
  }

  await c.env.DB.prepare('DELETE FROM tasks WHERE id = ?').bind(taskId).run();

  return c.json({ success: true, taskId });
});

/**
 * GET /workers — List workers
 */
api.get('/workers', async (c) => {
  const result = await c.env.DB.prepare(`
    SELECT id, name, type, capabilities, status, current_task_id, last_heartbeat,
           tasks_completed, tasks_failed, created_at
    FROM workers ORDER BY last_heartbeat DESC
  `).all();

  const fiveMin = new Date(Date.now() - 5 * 60 * 1000).toISOString();
  const workers = (result.results || []).map((w) => ({
    ...w,
    capabilities: parseJson(w.capabilities as string, []),
    actuallyOnline: w.status !== 'offline' && !!w.last_heartbeat && (w.last_heartbeat as string) > fiveMin,
  }));

  return c.json({
    workers,
    count: workers.length,
    online: workers.filter((w) => w.actuallyOnline).length,
  });
});

/**
 * GET /stats — Network statistics
 */
api.get('/stats', async (c) => {
  const fiveMin = new Date(Date.now() - 5 * 60 * 1000).toISOString();

  const workers = await c.env.DB.prepare(`
    SELECT
      COUNT(*) as total,
      SUM(CASE WHEN status IN ('online','busy') AND last_heartbeat > ? THEN 1 ELSE 0 END) as online,
      SUM(CASE WHEN status = 'busy' AND last_heartbeat > ? THEN 1 ELSE 0 END) as busy
    FROM workers
  `).bind(fiveMin, fiveMin).first();

  const tasks = await c.env.DB.prepare(`
    SELECT
      COUNT(*) as total,
      SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
      SUM(CASE WHEN status = 'assigned' THEN 1 ELSE 0 END) as assigned,
      SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
      SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
    FROM tasks
  `).first();

  // Last 24h activity
  const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const recent = await c.env.DB.prepare(`
    SELECT COUNT(*) as completed_24h FROM tasks WHERE status = 'completed' AND completed_at > ?
  `).bind(oneDayAgo).first<{ completed_24h: number }>();

  return c.json({
    workers: {
      total: workers?.total || 0,
      online: workers?.online || 0,
      busy: workers?.busy || 0,
    },
    tasks: {
      total: tasks?.total || 0,
      pending: tasks?.pending || 0,
      assigned: tasks?.assigned || 0,
      completed: tasks?.completed || 0,
      failed: tasks?.failed || 0,
    },
    last24h: {
      tasksCompleted: recent?.completed_24h || 0,
    },
    timestamp: now(),
  });
});

/**
 * GET /log — Work log (recent events)
 *
 * Query: limit?, taskId?, workerId?
 */
api.get('/log', async (c) => {
  if (!isAdmin(c)) return c.json({ error: 'Admin API key required' }, 401);

  const limit = Math.min(parseInt(c.req.query('limit') || '50', 10), 200);
  const taskId = c.req.query('taskId');
  const workerId = c.req.query('workerId');

  let query = 'SELECT id, worker_id, task_id, event, detail, created_at FROM work_log WHERE 1=1';
  const params: (string | number)[] = [];

  if (taskId) {
    query += ' AND task_id = ?';
    params.push(taskId);
  }
  if (workerId) {
    query += ' AND worker_id = ?';
    params.push(workerId);
  }

  query += ' ORDER BY created_at DESC LIMIT ?';
  params.push(limit);

  const result = await c.env.DB.prepare(query).bind(...params).all();

  return c.json({
    events: (result.results || []).map((e) => ({
      ...e,
      detail: parseJson(e.detail as string | null, null),
    })),
  });
});

export default api;
