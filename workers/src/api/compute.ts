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

/**
 * Generate a secure auth token for device authentication
 */
function generateAuthToken(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Validate device auth token from Authorization header
 * Returns device ID if valid, null otherwise
 */
async function validateAuthToken(c: any): Promise<string | null> {
  const authHeader = c.req.header('Authorization');
  if (!authHeader?.startsWith('Bearer ')) {
    return null;
  }

  const token = authHeader.slice(7);
  if (!token) {
    return null;
  }

  try {
    const device = await c.env.DB.prepare(`
      SELECT id FROM compute_devices WHERE auth_token = ?
    `).bind(token).first<{ id: string }>();

    return device?.id || null;
  } catch {
    return null;
  }
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
    const authToken = generateAuthToken();
    const now = new Date().toISOString();

    await c.env.DB.prepare(`
      INSERT INTO compute_devices (
        id, name, tier, platform, capabilities, endpoint_url,
        status, last_heartbeat, stats, auth_token, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, 'online', ?, ?, ?, ?, ?)
    `).bind(
      id,
      name,
      tier,
      platform,
      JSON.stringify(capabilities),
      endpointUrl || null,
      now,
      JSON.stringify({ tasksCompleted: 0, creditsEarned: 0, totalComputeTime: 0 }),
      authToken,
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
      // Return auth token - client must store this securely
      authToken,
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
 * POST /tasks/assign - Request task assignment (poll endpoint for browser agents)
 *
 * This endpoint is optimized for browser-based device agents that poll for work.
 * It atomically finds and assigns a task in one operation.
 *
 * Request body:
 * {
 *   deviceId: string,
 *   capabilities?: { compute: number, memory: number, cachedModels: string[] }
 * }
 */
compute.post('/tasks/assign', async (c) => {
  try {
    const body = await c.req.json();
    const { deviceId, capabilities } = body;

    if (!deviceId) {
      return c.json({ error: 'deviceId is required' }, 400);
    }

    // Validate auth token if provided (recommended for security)
    const authDeviceId = await validateAuthToken(c);
    if (authDeviceId && authDeviceId !== deviceId) {
      return c.json({ error: 'Device ID does not match auth token' }, 403);
    }
    if (!authDeviceId) {
      console.log(`[Compute] Task assign request without auth token for device ${deviceId}`);
    }

    const now = new Date().toISOString();

    // Get device info
    const device = await c.env.DB.prepare(`
      SELECT * FROM compute_devices WHERE id = ?
    `).bind(deviceId).first<ComputeDevice>();

    if (!device) {
      return c.json({ error: 'Device not found. Please register first.' }, 404);
    }

    // Update device heartbeat
    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET last_heartbeat = ?, updated_at = ?, status = 'online'
      WHERE id = ?
    `).bind(now, now, deviceId).run();

    // If device already has a task, return that
    if (device.current_task_id) {
      const currentTask = await c.env.DB.prepare(`
        SELECT * FROM compute_tasks WHERE id = ?
      `).bind(device.current_task_id).first<ComputeTask>();

      if (currentTask && currentTask.status === 'assigned') {
        return c.json({
          hasWork: true,
          task: {
            id: currentTask.id,
            type: currentTask.type,
            input: parseJson(currentTask.input, {}),
            config: parseJson(currentTask.config, {}),
            priority: currentTask.priority,
            reward: 1, // Default reward
          },
        });
      }
    }

    // Find and assign a task atomically
    const task = await c.env.DB.prepare(`
      SELECT * FROM compute_tasks
      WHERE status = 'pending'
        AND (min_tier IS NULL OR
             (min_tier = 'crowd') OR
             (min_tier = 'standard' AND ? IN ('standard', 'power')) OR
             (min_tier = 'power' AND ? = 'power'))
      ORDER BY priority DESC, created_at ASC
      LIMIT 1
    `).bind(device.tier, device.tier).first<ComputeTask>();

    if (!task) {
      return c.json({
        hasWork: false,
        message: 'No tasks available for your device tier',
        deviceTier: device.tier,
      });
    }

    // Assign task to device
    await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
      WHERE id = ? AND status = 'pending'
    `).bind(deviceId, now, task.id).run();

    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET status = 'busy', current_task_id = ?, updated_at = ?
      WHERE id = ?
    `).bind(task.id, now, deviceId).run();

    console.log(`[Compute] Task ${task.id} assigned to device ${deviceId} via poll`);

    return c.json({
      hasWork: true,
      task: {
        id: task.id,
        type: task.type,
        input: parseJson(task.input, {}),
        config: parseJson(task.config, {}),
        priority: task.priority,
        reward: 1, // Default reward - could be calculated based on task complexity
      },
    });
  } catch (error) {
    console.error('[Compute] Task assign error:', error);
    return c.json({ error: 'Failed to assign task' }, 500);
  }
});

/**
 * POST /tasks/generate-crowd - Generate crowd-tier tasks from project work
 *
 * This endpoint breaks down larger work into bite-sized tasks suitable for
 * browser contributors. Called by the orchestrator or manually to seed work.
 *
 * Request body:
 * {
 *   projectId?: string,
 *   count?: number (default 10),
 *   types?: string[] (subset of crowd-compatible types)
 * }
 */
compute.post('/tasks/generate-crowd', async (c) => {
  try {
    const body = await c.req.json();
    const { projectId, count = 10, types } = body;

    const now = new Date().toISOString();
    const generatedTasks: { id: string; type: string }[] = [];

    // Crowd-compatible task types with sample inputs
    const crowdTaskTemplates = [
      {
        type: 'classification',
        templates: [
          { input: { text: 'Analyze the sentiment of user feedback about voice quality', categories: ['positive', 'negative', 'neutral', 'mixed'] }, priority: 5 },
          { input: { text: 'Classify this research finding by relevance to TTS', categories: ['highly_relevant', 'somewhat_relevant', 'not_relevant'] }, priority: 4 },
          { input: { text: 'Categorize prosody feature by type', categories: ['pitch', 'rhythm', 'stress', 'intonation', 'duration'] }, priority: 5 },
        ],
      },
      {
        type: 'summarization',
        templates: [
          { input: { text: 'Summarize the key findings from this voice clone experiment', maxLength: 150 }, priority: 4 },
          { input: { text: 'Create a brief summary of the prosody analysis results', maxLength: 100 }, priority: 4 },
          { input: { text: 'Summarize user feedback about voice naturalness', maxLength: 120 }, priority: 3 },
        ],
      },
      {
        type: 'embedding',
        templates: [
          { input: { text: 'Generate embedding for voice sample metadata', model: 'default' }, priority: 3 },
          { input: { text: 'Create semantic embedding for research note', model: 'default' }, priority: 3 },
          { input: { text: 'Embed prosody feature description', model: 'default' }, priority: 2 },
        ],
      },
      {
        type: 'assessment',
        templates: [
          { input: { prompt: 'Rate the quality of this transcription on a scale of 1-5', context: 'Voice clone quality assessment' }, priority: 5 },
          { input: { prompt: 'Evaluate if this voice sample has clear prosody markers', context: 'Prosody detection assessment' }, priority: 4 },
          { input: { prompt: 'Assess the naturalness of this synthesized speech description', context: 'TTS quality assessment' }, priority: 5 },
        ],
      },
    ];

    // Filter by requested types if specified
    const taskPool = types
      ? crowdTaskTemplates.filter((t) => types.includes(t.type))
      : crowdTaskTemplates;

    if (taskPool.length === 0) {
      return c.json({ error: 'No valid crowd task types specified' }, 400);
    }

    // Generate tasks
    for (let i = 0; i < count; i++) {
      const taskType = taskPool[Math.floor(Math.random() * taskPool.length)];
      const template = taskType.templates[Math.floor(Math.random() * taskType.templates.length)];

      const taskId = generateId('ctask');

      await c.env.DB.prepare(`
        INSERT INTO compute_tasks (
          id, type, input, config, status, priority, min_tier,
          parent_task_id, created_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, 'crowd', ?, ?)
      `).bind(
        taskId,
        taskType.type,
        JSON.stringify(template.input),
        JSON.stringify({ maxTokens: 256, temperature: 0.7 }),
        template.priority,
        projectId || null,
        now
      ).run();

      generatedTasks.push({ id: taskId, type: taskType.type });
    }

    console.log(`[Compute] Generated ${generatedTasks.length} crowd tasks`);

    return c.json({
      success: true,
      generated: generatedTasks.length,
      tasks: generatedTasks,
      message: `Generated ${generatedTasks.length} crowd-tier tasks for browser contributors`,
    });
  } catch (error) {
    console.error('[Compute] Generate crowd tasks error:', error);
    return c.json({ error: 'Failed to generate crowd tasks' }, 500);
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
 * POST /tasks/:id - Complete a task (alias for browser device-agent compatibility)
 *
 * Request body:
 * {
 *   deviceId: string,
 *   success: boolean,
 *   result?: { text?: string, metrics: { computeTime: number } },
 *   error?: string
 * }
 */
compute.post('/tasks/:id', async (c) => {
  // Route to the complete handler
  const taskId = c.req.param('id');

  // Skip if this looks like a different route pattern
  if (taskId === 'assign' || taskId === 'pending' || taskId === 'generate-crowd') {
    return c.json({ error: 'Not found' }, 404);
  }

  try {
    const body = await c.req.json();
    const { deviceId, success, result, error } = body;

    if (!deviceId) {
      return c.json({ error: 'deviceId is required' }, 400);
    }

    // Validate auth token if provided
    const authDeviceId = await validateAuthToken(c);
    if (authDeviceId && authDeviceId !== deviceId) {
      return c.json({ error: 'Device ID does not match auth token' }, 403);
    }
    if (!authDeviceId) {
      console.log(`[Compute] Task complete request without auth token for device ${deviceId}`);
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
    const computeTime = result?.metrics?.computeTime || 0;
    if (success) {
      await c.env.DB.prepare(`
        UPDATE compute_devices
        SET stats = json_set(
          COALESCE(stats, '{}'),
          '$.tasksCompleted', COALESCE(json_extract(stats, '$.tasksCompleted'), 0) + 1,
          '$.creditsEarned', COALESCE(json_extract(stats, '$.creditsEarned'), 0) + 1,
          '$.totalComputeTime', COALESCE(json_extract(stats, '$.totalComputeTime'), 0) + ?
        )
        WHERE id = ?
      `).bind(computeTime / 1000, deviceId).run();
    }

    console.log(`[Compute] Task ${taskId} completed by ${deviceId}: ${newStatus}`);

    return c.json({
      success: true,
      taskId,
      status: newStatus,
      creditsAwarded: success ? 1 : 0,
    });
  } catch (error) {
    console.error('[Compute] Complete task error:', error);
    return c.json({ error: 'Failed to complete task' }, 500);
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

    // Update device stats if successful (always award credits on success)
    if (success) {
      const computeTime = result?.metrics?.computeTime || 0;
      await c.env.DB.prepare(`
        UPDATE compute_devices
        SET stats = json_set(
          COALESCE(stats, '{}'),
          '$.tasksCompleted', COALESCE(json_extract(stats, '$.tasksCompleted'), 0) + 1,
          '$.creditsEarned', COALESCE(json_extract(stats, '$.creditsEarned'), 0) + 1,
          '$.totalComputeTime', COALESCE(json_extract(stats, '$.totalComputeTime'), 0) + ?
        )
        WHERE id = ?
      `).bind(computeTime / 1000, deviceId).run();
    }

    console.log(`[Compute] Task ${taskId} completed by ${deviceId}: ${newStatus}`);

    return c.json({
      success: true,
      taskId,
      status: newStatus,
      creditsAwarded: success ? 1 : 0,
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

/**
 * GET /leaderboard - Get top contributors ranked by credits earned
 *
 * Query params:
 *   limit: number (default 20, max 100)
 *   tier: 'power' | 'standard' | 'crowd' (optional filter)
 */
compute.get('/leaderboard', async (c) => {
  try {
    const limit = Math.min(parseInt(c.req.query('limit') || '20', 10), 100);
    const tier = c.req.query('tier');

    let query = `
      SELECT
        id,
        name,
        tier,
        platform,
        json_extract(stats, '$.tasksCompleted') as tasks_completed,
        json_extract(stats, '$.creditsEarned') as credits_earned,
        json_extract(stats, '$.totalComputeTime') as total_compute_time,
        created_at
      FROM compute_devices
      WHERE json_extract(stats, '$.tasksCompleted') > 0
    `;
    const params: (string | number)[] = [];

    if (tier) {
      query += ' AND tier = ?';
      params.push(tier);
    }

    query += ' ORDER BY json_extract(stats, \'$.creditsEarned\') DESC LIMIT ?';
    params.push(limit);

    const result = await c.env.DB.prepare(query).bind(...params).all();

    // Helper to calculate rank based on tasks
    const calculateRank = (tasks: number): 'novice' | 'contributor' | 'expert' | 'legend' => {
      if (tasks >= 1000) return 'legend';
      if (tasks >= 100) return 'expert';
      if (tasks >= 10) return 'contributor';
      return 'novice';
    };

    // Helper to check badges
    const checkBadges = (tasks: number, credits: number, tier: string) => {
      const badges: { id: string; name: string; description: string; icon: string; earnedAt: string }[] = [];
      const now = new Date().toISOString();

      if (tasks >= 1) badges.push({ id: 'first_task', name: 'First Contribution', description: 'Completed your first task', icon: '🌟', earnedAt: now });
      if (tasks >= 100) badges.push({ id: 'hundred_tasks', name: 'Century', description: 'Completed 100 tasks', icon: '💯', earnedAt: now });
      if (tasks >= 1000) badges.push({ id: 'thousand_tasks', name: 'Legend', description: 'Completed 1,000 tasks', icon: '🏆', earnedAt: now });
      if (credits >= 1000) badges.push({ id: 'thousand_credits', name: 'Millionaire', description: 'Earned 1,000 credits', icon: '💰', earnedAt: now });
      if (tier === 'power') badges.push({ id: 'power_contributor', name: 'Power Contributor', description: 'Registered a power-tier device', icon: '⚡', earnedAt: now });

      return badges;
    };

    // Transform to ContributorProfile format expected by frontend
    const leaderboard = (result.results || []).map((d: any) => {
      const tasksCompleted = d.tasks_completed || 0;
      const creditsEarned = d.credits_earned || 0;
      const computeTime = d.total_compute_time || 0;

      return {
        userId: d.id,
        displayName: d.name || `Contributor ${d.id.slice(-6)}`,
        rank: calculateRank(tasksCompleted),
        totalCreditsEarned: creditsEarned,
        totalTasksCompleted: tasksCompleted,
        totalComputeTime: Math.round(computeTime * 10) / 10,
        devices: [d.id],
        badges: checkBadges(tasksCompleted, creditsEarned, d.tier),
        joinedAt: d.created_at,
        // Extra fields for debugging
        tier: d.tier,
        platform: d.platform,
      };
    });

    // Get total contributor count
    const totalResult = await c.env.DB.prepare(`
      SELECT COUNT(*) as count FROM compute_devices
      WHERE json_extract(stats, '$.tasksCompleted') > 0
    `).first<{ count: number }>();

    // Return as array for frontend compatibility
    return c.json(leaderboard);
  } catch (error) {
    console.error('[Compute] Leaderboard error:', error);
    return c.json({ error: 'Failed to get leaderboard' }, 500);
  }
});

/**
 * GET /devices/:id/stats - Get stats for a specific device
 */
compute.get('/devices/:id/stats', async (c) => {
  try {
    const deviceId = c.req.param('id');

    const device = await c.env.DB.prepare(`
      SELECT
        id,
        name,
        tier,
        platform,
        status,
        stats,
        created_at,
        last_heartbeat
      FROM compute_devices
      WHERE id = ?
    `).bind(deviceId).first();

    if (!device) {
      return c.json({ error: 'Device not found' }, 404);
    }

    const stats = parseJson<{
      tasksCompleted: number;
      creditsEarned: number;
      totalComputeTime: number;
    }>(device.stats as string, { tasksCompleted: 0, creditsEarned: 0, totalComputeTime: 0 });

    // Get rank
    const rankResult = await c.env.DB.prepare(`
      SELECT COUNT(*) + 1 as rank
      FROM compute_devices
      WHERE json_extract(stats, '$.creditsEarned') > ?
    `).bind(stats.creditsEarned).first<{ rank: number }>();

    return c.json({
      deviceId: device.id,
      name: device.name,
      tier: device.tier,
      platform: device.platform,
      status: device.status,
      stats: {
        tasksCompleted: stats.tasksCompleted,
        creditsEarned: stats.creditsEarned,
        totalComputeTime: Math.round(stats.totalComputeTime * 10) / 10,
      },
      rank: rankResult?.rank || 1,
      memberSince: device.created_at,
      lastActive: device.last_heartbeat,
    });
  } catch (error) {
    console.error('[Compute] Device stats error:', error);
    return c.json({ error: 'Failed to get device stats' }, 500);
  }
});

export default compute;
