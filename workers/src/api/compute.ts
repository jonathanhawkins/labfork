/**
 * Compute Network API Routes
 *
 * These routes handle:
 * - Device registration (4090, contributor GPUs, WebGPU browsers)
 * - Task dispatch (create tasks for compute network)
 * - Task claiming (devices poll and claim work)
 * - Result reporting (devices report completed work)
 *
 * This is the core of the distributed compute system that replaces Workers AI.
 */

import { Hono } from 'hono';

// ============================================================================
// Types
// ============================================================================

interface Env {
  DB: D1Database;
}

/**
 * Device tiers match frontend/lib/compute/types.ts
 */
type DeviceTier = 'power' | 'standard' | 'crowd';
type DevicePlatform = 'cuda' | 'metal' | 'webgpu' | 'cpu';
type DeviceStatus = 'online' | 'busy' | 'offline' | 'paused';

/**
 * Compute task types
 */
type ComputeTaskType =
  | 'inference'
  | 'embedding'
  | 'assessment'
  | 'planning'
  | 'execution'
  | 'draft_generation'
  | 'draft_verification'
  | 'summarization'
  | 'classification';

type ComputeTaskStatus = 'pending' | 'assigned' | 'processing' | 'completed' | 'failed' | 'timeout';

/**
 * Device capabilities
 */
interface DeviceCapabilities {
  compute: number; // TFLOPS
  memory: number; // GB
  bandwidth?: number; // Mbps
  models: string[]; // Cached model IDs
  gpuName?: string;
}

/**
 * Compute device from DB
 */
interface ComputeDevice {
  id: string;
  name: string;
  tier: DeviceTier;
  platform: DevicePlatform;
  capabilities: string; // JSON
  endpoint_url: string | null;
  status: DeviceStatus;
  current_task_id: string | null;
  last_heartbeat: string | null;
  stats: string | null; // JSON
  created_at: string;
  updated_at: string;
}

/**
 * Compute task from DB
 */
interface ComputeTask {
  id: string;
  type: ComputeTaskType;
  input: string; // JSON
  config: string; // JSON
  status: ComputeTaskStatus;
  priority: number;
  min_tier: DeviceTier | null;
  assigned_device_id: string | null;
  result: string | null; // JSON
  error: string | null;
  parent_task_id: string | null;
  created_at: string;
  assigned_at: string | null;
  completed_at: string | null;
}

// ============================================================================
// Helpers
// ============================================================================

function generateId(prefix: string): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `${prefix}_${timestamp}_${random}`;
}

function parseJson<T>(value: string | null, defaultValue: T): T {
  if (!value) return defaultValue;
  try {
    return JSON.parse(value) as T;
  } catch {
    return defaultValue;
  }
}

/**
 * Classify device tier based on capabilities
 */
function classifyTier(capabilities: DeviceCapabilities, platform: DevicePlatform): DeviceTier {
  // Power tier: High-end GPUs (4090, A100, etc.) with CUDA
  if (capabilities.compute >= 40 && capabilities.memory >= 16 && platform === 'cuda') {
    return 'power';
  }

  // Standard tier: Mid-range GPUs, Apple Silicon
  if (capabilities.compute >= 5 && capabilities.memory >= 8) {
    return 'standard';
  }

  // Crowd tier: Everything else (browsers, phones, low-end)
  return 'crowd';
}

/**
 * Check if a device can handle a task based on tier
 */
function deviceCanHandleTask(deviceTier: DeviceTier, minTier: DeviceTier | null): boolean {
  if (!minTier) return true;

  const tierOrder: DeviceTier[] = ['crowd', 'standard', 'power'];
  const deviceIndex = tierOrder.indexOf(deviceTier);
  const minIndex = tierOrder.indexOf(minTier);

  return deviceIndex >= minIndex;
}

// ============================================================================
// API Routes
// ============================================================================

const compute = new Hono<{ Bindings: Env }>();

/**
 * POST /devices - Register a new compute device
 *
 * Request body:
 * {
 *   name: string,
 *   platform: 'cuda' | 'metal' | 'webgpu' | 'cpu',
 *   capabilities: { compute: number, memory: number, models: string[] },
 *   endpointUrl?: string  // Optional push endpoint for tasks
 * }
 */
compute.post('/devices', async (c) => {
  try {
    const body = await c.req.json();
    const { name, platform, capabilities, endpointUrl } = body;

    if (!name || !platform || !capabilities) {
      return c.json({ error: 'Missing required fields: name, platform, capabilities' }, 400);
    }

    const id = generateId('dev');
    const tier = classifyTier(capabilities, platform);
    const now = new Date().toISOString();

    await c.env.DB.prepare(`
      INSERT INTO compute_devices (
        id, name, tier, platform, capabilities, endpoint_url,
        status, last_heartbeat, stats, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, 'online', ?, ?, ?, ?)
    `).bind(
      id,
      name,
      tier,
      platform,
      JSON.stringify(capabilities),
      endpointUrl || null,
      now,
      JSON.stringify({ tasksCompleted: 0, creditsEarned: 0, totalComputeTime: 0 }),
      now,
      now
    ).run();

    console.log(`[Compute] Device registered: ${id} (${name}, ${tier} tier, ${platform})`);

    return c.json({
      success: true,
      device: {
        id,
        name,
        tier,
        platform,
        capabilities,
        status: 'online',
        registeredAt: now,
      },
    });
  } catch (error) {
    console.error('[Compute] Device registration error:', error);
    return c.json({ error: 'Failed to register device' }, 500);
  }
});

/**
 * PATCH /devices/:id - Update device status (heartbeat)
 *
 * Request body:
 * {
 *   status?: 'online' | 'busy' | 'offline' | 'paused',
 *   taskProgress?: number  // 0-100 for current task
 * }
 */
compute.patch('/devices/:id', async (c) => {
  try {
    const deviceId = c.req.param('id');
    const body = await c.req.json();
    const { status, taskProgress } = body;

    const now = new Date().toISOString();

    // Update device
    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET last_heartbeat = ?,
          status = COALESCE(?, status),
          updated_at = ?
      WHERE id = ?
    `).bind(now, status || null, now, deviceId).run();

    // Check for pending tasks that could be assigned
    const device = await c.env.DB.prepare(`
      SELECT * FROM compute_devices WHERE id = ?
    `).bind(deviceId).first<ComputeDevice>();

    if (!device) {
      return c.json({ error: 'Device not found' }, 404);
    }

    // If device is online and has no current task, look for work
    let assignedTask = null;
    if (device.status === 'online' && !device.current_task_id) {
      // Find highest priority pending task that this device can handle
      const pendingTask = await c.env.DB.prepare(`
        SELECT * FROM compute_tasks
        WHERE status = 'pending'
          AND (min_tier IS NULL OR min_tier = ? OR
               (min_tier = 'crowd') OR
               (min_tier = 'standard' AND ? IN ('standard', 'power')) OR
               (min_tier = 'power' AND ? = 'power'))
        ORDER BY priority DESC, created_at ASC
        LIMIT 1
      `).bind(device.tier, device.tier, device.tier).first<ComputeTask>();

      if (pendingTask) {
        // Assign task to device
        await c.env.DB.prepare(`
          UPDATE compute_tasks
          SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
          WHERE id = ?
        `).bind(deviceId, now, pendingTask.id).run();

        await c.env.DB.prepare(`
          UPDATE compute_devices
          SET status = 'busy', current_task_id = ?, updated_at = ?
          WHERE id = ?
        `).bind(pendingTask.id, now, deviceId).run();

        assignedTask = {
          id: pendingTask.id,
          type: pendingTask.type,
          input: parseJson(pendingTask.input, {}),
          config: parseJson(pendingTask.config, {}),
          priority: pendingTask.priority,
        };

        console.log(`[Compute] Task ${pendingTask.id} assigned to device ${deviceId}`);
      }
    }

    return c.json({
      success: true,
      deviceId,
      status: device.status,
      task: assignedTask,
    });
  } catch (error) {
    console.error('[Compute] Heartbeat error:', error);
    return c.json({ error: 'Failed to update device' }, 500);
  }
});

/**
 * GET /devices - List all compute devices
 */
compute.get('/devices', async (c) => {
  try {
    const tier = c.req.query('tier');
    const status = c.req.query('status');

    let query = 'SELECT * FROM compute_devices WHERE 1=1';
    const params: (string | null)[] = [];

    if (tier) {
      query += ' AND tier = ?';
      params.push(tier);
    }

    if (status) {
      query += ' AND status = ?';
      params.push(status);
    }

    query += ' ORDER BY tier DESC, last_heartbeat DESC';

    const result = await c.env.DB.prepare(query).bind(...params).all<ComputeDevice>();

    const devices = (result.results || []).map((d) => ({
      id: d.id,
      name: d.name,
      tier: d.tier,
      platform: d.platform,
      capabilities: parseJson<DeviceCapabilities>(d.capabilities, { compute: 0, memory: 0, models: [] }),
      status: d.status,
      currentTaskId: d.current_task_id,
      lastHeartbeat: d.last_heartbeat,
      stats: parseJson(d.stats, { tasksCompleted: 0, creditsEarned: 0, totalComputeTime: 0 }),
    }));

    // Calculate network stats
    const online = devices.filter((d) => d.status === 'online' || d.status === 'busy');
    const totalCompute = online.reduce((sum, d) => sum + d.capabilities.compute, 0);

    return c.json({
      devices,
      count: devices.length,
      online: online.length,
      totalCompute: Math.round(totalCompute * 10) / 10,
      byTier: {
        power: devices.filter((d) => d.tier === 'power').length,
        standard: devices.filter((d) => d.tier === 'standard').length,
        crowd: devices.filter((d) => d.tier === 'crowd').length,
      },
    });
  } catch (error) {
    console.error('[Compute] List devices error:', error);
    return c.json({ error: 'Failed to list devices' }, 500);
  }
});

/**
 * POST /tasks - Create a new compute task
 *
 * Request body:
 * {
 *   type: 'inference' | 'embedding' | 'assessment' | 'planning' | 'execution',
 *   input: { prompt?: string, messages?: array, model?: string },
 *   config: { maxTokens?: number, temperature?: number },
 *   priority?: number,
 *   minTier?: 'power' | 'standard' | 'crowd',
 *   parentTaskId?: string  // Links to workflow task
 * }
 */
compute.post('/tasks', async (c) => {
  try {
    const body = await c.req.json();
    const { type, input, config, priority, minTier, parentTaskId } = body;

    if (!type || !input) {
      return c.json({ error: 'Missing required fields: type, input' }, 400);
    }

    const id = generateId('ctask');
    const now = new Date().toISOString();

    await c.env.DB.prepare(`
      INSERT INTO compute_tasks (
        id, type, input, config, status, priority, min_tier,
        parent_task_id, created_at
      ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)
    `).bind(
      id,
      type,
      JSON.stringify(input),
      JSON.stringify(config || {}),
      priority || 5,
      minTier || null,
      parentTaskId || null,
      now
    ).run();

    console.log(`[Compute] Task created: ${id} (${type}, priority ${priority || 5})`);

    // Try to assign immediately to an available device
    const device = await c.env.DB.prepare(`
      SELECT * FROM compute_devices
      WHERE status = 'online'
        AND current_task_id IS NULL
        AND (? IS NULL OR tier = ? OR
             (? = 'crowd') OR
             (? = 'standard' AND tier IN ('standard', 'power')) OR
             (? = 'power' AND tier = 'power'))
      ORDER BY
        CASE tier WHEN 'power' THEN 3 WHEN 'standard' THEN 2 ELSE 1 END DESC,
        last_heartbeat DESC
      LIMIT 1
    `).bind(minTier || null, minTier || null, minTier || null, minTier || null, minTier || null).first<ComputeDevice>();

    if (device) {
      // Assign task to device immediately
      await c.env.DB.prepare(`
        UPDATE compute_tasks
        SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
        WHERE id = ?
      `).bind(device.id, now, id).run();

      await c.env.DB.prepare(`
        UPDATE compute_devices
        SET status = 'busy', current_task_id = ?, updated_at = ?
        WHERE id = ?
      `).bind(id, now, device.id).run();

      console.log(`[Compute] Task ${id} immediately assigned to ${device.id}`);
    }

    return c.json({
      success: true,
      task: {
        id,
        type,
        status: device ? 'assigned' : 'pending',
        assignedDeviceId: device?.id || null,
        createdAt: now,
      },
    });
  } catch (error) {
    console.error('[Compute] Create task error:', error);
    return c.json({ error: 'Failed to create task' }, 500);
  }
});

/**
 * GET /tasks/pending - Get pending tasks for a device to claim
 *
 * Query params:
 *   deviceId: string (required)
 *   limit: number (default 1)
 */
compute.get('/tasks/pending', async (c) => {
  try {
    const deviceId = c.req.query('deviceId');
    const limit = parseInt(c.req.query('limit') || '1', 10);

    if (!deviceId) {
      return c.json({ error: 'deviceId is required' }, 400);
    }

    // Get device info
    const device = await c.env.DB.prepare(`
      SELECT * FROM compute_devices WHERE id = ?
    `).bind(deviceId).first<ComputeDevice>();

    if (!device) {
      return c.json({ error: 'Device not found' }, 404);
    }

    // Find pending tasks that this device can handle
    const result = await c.env.DB.prepare(`
      SELECT * FROM compute_tasks
      WHERE status = 'pending'
        AND (min_tier IS NULL OR
             (min_tier = 'crowd') OR
             (min_tier = 'standard' AND ? IN ('standard', 'power')) OR
             (min_tier = 'power' AND ? = 'power'))
      ORDER BY priority DESC, created_at ASC
      LIMIT ?
    `).bind(device.tier, device.tier, limit).all<ComputeTask>();

    const tasks = (result.results || []).map((t) => ({
      id: t.id,
      type: t.type,
      input: parseJson(t.input, {}),
      config: parseJson(t.config, {}),
      priority: t.priority,
      minTier: t.min_tier,
      createdAt: t.created_at,
    }));

    return c.json({
      tasks,
      count: tasks.length,
    });
  } catch (error) {
    console.error('[Compute] Get pending tasks error:', error);
    return c.json({ error: 'Failed to get pending tasks' }, 500);
  }
});

/**
 * POST /tasks/:id/claim - Claim a task for execution
 *
 * Request body:
 * {
 *   deviceId: string
 * }
 */
compute.post('/tasks/:id/claim', async (c) => {
  try {
    const taskId = c.req.param('id');
    const body = await c.req.json();
    const { deviceId } = body;

    if (!deviceId) {
      return c.json({ error: 'deviceId is required' }, 400);
    }

    const now = new Date().toISOString();

    // Check task is still pending
    const task = await c.env.DB.prepare(`
      SELECT * FROM compute_tasks WHERE id = ?
    `).bind(taskId).first<ComputeTask>();

    if (!task) {
      return c.json({ error: 'Task not found' }, 404);
    }

    if (task.status !== 'pending') {
      return c.json({ error: 'Task is no longer available', status: task.status }, 409);
    }

    // Assign task
    await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
      WHERE id = ? AND status = 'pending'
    `).bind(deviceId, now, taskId).run();

    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET status = 'busy', current_task_id = ?, updated_at = ?
      WHERE id = ?
    `).bind(taskId, now, deviceId).run();

    console.log(`[Compute] Task ${taskId} claimed by device ${deviceId}`);

    return c.json({
      success: true,
      task: {
        id: task.id,
        type: task.type,
        input: parseJson(task.input, {}),
        config: parseJson(task.config, {}),
        priority: task.priority,
      },
    });
  } catch (error) {
    console.error('[Compute] Claim task error:', error);
    return c.json({ error: 'Failed to claim task' }, 500);
  }
});

/**
 * POST /tasks/:id/complete - Report task completion
 *
 * Request body:
 * {
 *   deviceId: string,
 *   success: boolean,
 *   result?: { output: any, metrics: { computeTime: number } },
 *   error?: string
 * }
 */
compute.post('/tasks/:id/complete', async (c) => {
  try {
    const taskId = c.req.param('id');
    const body = await c.req.json();
    const { deviceId, success, result, error } = body;

    if (!deviceId) {
      return c.json({ error: 'deviceId is required' }, 400);
    }

    const now = new Date().toISOString();

    // Get task
    const task = await c.env.DB.prepare(`
      SELECT * FROM compute_tasks WHERE id = ?
    `).bind(taskId).first<ComputeTask>();

    if (!task) {
      return c.json({ error: 'Task not found' }, 404);
    }

    if (task.assigned_device_id !== deviceId) {
      return c.json({ error: 'Task not assigned to this device' }, 403);
    }

    // Update task
    const newStatus = success ? 'completed' : 'failed';
    await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = ?, result = ?, error = ?, completed_at = ?
      WHERE id = ?
    `).bind(newStatus, result ? JSON.stringify(result) : null, error || null, now, taskId).run();

    // Free up device
    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET status = 'online', current_task_id = NULL, updated_at = ?
      WHERE id = ?
    `).bind(now, deviceId).run();

    // Update device stats if successful
    if (success && result?.metrics?.computeTime) {
      await c.env.DB.prepare(`
        UPDATE compute_devices
        SET stats = json_set(
          COALESCE(stats, '{}'),
          '$.tasksCompleted', COALESCE(json_extract(stats, '$.tasksCompleted'), 0) + 1,
          '$.totalComputeTime', COALESCE(json_extract(stats, '$.totalComputeTime'), 0) + ?
        )
        WHERE id = ?
      `).bind(result.metrics.computeTime / 1000, deviceId).run();
    }

    console.log(`[Compute] Task ${taskId} completed by ${deviceId}: ${newStatus}`);

    return c.json({
      success: true,
      taskId,
      status: newStatus,
    });
  } catch (error) {
    console.error('[Compute] Complete task error:', error);
    return c.json({ error: 'Failed to complete task' }, 500);
  }
});

/**
 * GET /tasks/:id - Get task status
 */
compute.get('/tasks/:id', async (c) => {
  try {
    const taskId = c.req.param('id');

    const task = await c.env.DB.prepare(`
      SELECT * FROM compute_tasks WHERE id = ?
    `).bind(taskId).first<ComputeTask>();

    if (!task) {
      return c.json({ error: 'Task not found' }, 404);
    }

    return c.json({
      id: task.id,
      type: task.type,
      status: task.status,
      input: parseJson(task.input, {}),
      config: parseJson(task.config, {}),
      result: parseJson(task.result, null),
      error: task.error,
      assignedDeviceId: task.assigned_device_id,
      createdAt: task.created_at,
      assignedAt: task.assigned_at,
      completedAt: task.completed_at,
    });
  } catch (error) {
    console.error('[Compute] Get task error:', error);
    return c.json({ error: 'Failed to get task' }, 500);
  }
});

/**
 * GET /tasks/:id/wait - Long-poll for task completion
 *
 * Query params:
 *   timeout: number (ms, default 30000)
 */
compute.get('/tasks/:id/wait', async (c) => {
  try {
    const taskId = c.req.param('id');
    const timeout = parseInt(c.req.query('timeout') || '30000', 10);

    const startTime = Date.now();
    const pollInterval = 500; // Poll every 500ms

    while (Date.now() - startTime < timeout) {
      const task = await c.env.DB.prepare(`
        SELECT * FROM compute_tasks WHERE id = ?
      `).bind(taskId).first<ComputeTask>();

      if (!task) {
        return c.json({ error: 'Task not found' }, 404);
      }

      if (task.status === 'completed' || task.status === 'failed') {
        return c.json({
          id: task.id,
          status: task.status,
          result: parseJson(task.result, null),
          error: task.error,
          completedAt: task.completed_at,
        });
      }

      // Wait before next poll
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    // Timeout - return current status
    const task = await c.env.DB.prepare(`
      SELECT status FROM compute_tasks WHERE id = ?
    `).bind(taskId).first<{ status: string }>();

    return c.json({
      id: taskId,
      status: task?.status || 'unknown',
      timeout: true,
    });
  } catch (error) {
    console.error('[Compute] Wait for task error:', error);
    return c.json({ error: 'Failed to wait for task' }, 500);
  }
});

/**
 * GET /stats - Get compute network statistics
 */
compute.get('/stats', async (c) => {
  try {
    // Device stats
    const deviceStats = await c.env.DB.prepare(`
      SELECT
        COUNT(*) as total,
        SUM(CASE WHEN status IN ('online', 'busy') THEN 1 ELSE 0 END) as online,
        SUM(CASE WHEN tier = 'power' THEN 1 ELSE 0 END) as power,
        SUM(CASE WHEN tier = 'standard' THEN 1 ELSE 0 END) as standard,
        SUM(CASE WHEN tier = 'crowd' THEN 1 ELSE 0 END) as crowd
      FROM compute_devices
    `).first();

    // Task stats
    const taskStats = await c.env.DB.prepare(`
      SELECT
        COUNT(*) as total,
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status IN ('assigned', 'processing') THEN 1 ELSE 0 END) as processing,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
      FROM compute_tasks
    `).first();

    // Total compute power (from online devices)
    const computeResult = await c.env.DB.prepare(`
      SELECT SUM(json_extract(capabilities, '$.compute')) as total_compute
      FROM compute_devices
      WHERE status IN ('online', 'busy')
    `).first<{ total_compute: number }>();

    return c.json({
      devices: {
        total: deviceStats?.total || 0,
        online: deviceStats?.online || 0,
        byTier: {
          power: deviceStats?.power || 0,
          standard: deviceStats?.standard || 0,
          crowd: deviceStats?.crowd || 0,
        },
      },
      tasks: {
        total: taskStats?.total || 0,
        pending: taskStats?.pending || 0,
        processing: taskStats?.processing || 0,
        completed: taskStats?.completed || 0,
        failed: taskStats?.failed || 0,
      },
      totalCompute: Math.round((computeResult?.total_compute || 0) * 10) / 10,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('[Compute] Stats error:', error);
    return c.json({ error: 'Failed to get stats' }, 500);
  }
});

export default compute;
