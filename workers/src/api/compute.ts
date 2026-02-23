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

import { Hono, type Context } from 'hono';

// ============================================================================
// Types
// ============================================================================

interface Env {
  DB: D1Database;
  ADMIN_API_KEY?: string;
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
async function validateAuthToken(c: Context<{ Bindings: Env }>): Promise<string | null> {
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
  } catch (err) {
    console.warn('[Compute] Auth token validation failed:', err);
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
 * Validate admin API key from Authorization header.
 * Returns true if the request has a valid admin key.
 */
function validateAdminKey(c: Context<{ Bindings: Env }>): boolean {
  const adminKey = c.env.ADMIN_API_KEY;
  if (!adminKey) {
    // No admin key configured — allow (dev mode)
    return true;
  }

  const authHeader = c.req.header('Authorization');
  if (!authHeader?.startsWith('Bearer ')) {
    return false;
  }

  const token = authHeader.slice(7);
  // Constant-time comparison to prevent timing attacks
  if (token.length !== adminKey.length) return false;
  let mismatch = 0;
  for (let i = 0; i < token.length; i++) {
    mismatch |= token.charCodeAt(i) ^ adminKey.charCodeAt(i);
  }
  return mismatch === 0;
}

/**
 * Admin auth middleware — returns 401 if admin key is required and missing/invalid.
 * If ADMIN_API_KEY is not set in env, all requests pass (dev mode).
 */
function requireAdmin(c: Context<{ Bindings: Env }>): Response | null {
  if (!validateAdminKey(c)) {
    return c.json({ error: 'Admin API key required. Include Authorization: Bearer <key> header.' }, 401);
  }
  return null;
}

// ============================================================================
// Result Validation Helpers
// ============================================================================

interface ValidationResult {
  valid: boolean;
  reason?: string;
  adjustedComputeMode?: string;
}

/**
 * Validate task result timing. Catches impossibly fast completions.
 * Uses client-reported computeTime as the primary signal since server-side
 * assigned_at/completed_at can have clock skew or same-tick resolution in tests.
 */
function validateTiming(
  taskType: string,
  assignedAt: string | null,
  _completedAt: string, // kept for future server-side delta checks
  clientComputeTime: number,
  computeMode: string
): ValidationResult {
  if (!assignedAt) return { valid: true };

  // Use client-reported compute time as primary validation signal
  // If computeTime is 0 or missing, that itself is suspicious for non-mock modes
  const checkTime = clientComputeTime;

  if (computeMode === 'mock') {
    // For mock mode, 0/missing computeTime is acceptable (mock may not track time)
    // but suspiciously fast positive values are rejected
    if (checkTime > 0 && checkTime < 200) {
      return { valid: false, reason: `Task completed too fast: ${checkTime}ms client compute time (min 200ms for mock)` };
    }
  } else if (computeMode === 'webllm') {
    // Missing/zero computeTime for webllm is suspicious — real inference always takes time
    if (!checkTime || checkTime <= 0) {
      return { valid: false, reason: `WebLLM task reported no compute time (computeTime=${checkTime})` };
    }
    const minTime: Record<string, number> = {
      summarization: 1000,
      classification: 500,
      embedding: 500,
      assessment: 1000,
      inference: 1000,
    };
    const min = minTime[taskType] || 500;
    if (checkTime < min) {
      return { valid: false, reason: `WebLLM task completed too fast: ${checkTime}ms (min ${min}ms for ${taskType})` };
    }
  }

  return { valid: true };
}

/**
 * Validate task result content based on task type.
 */
function validateContent(
  taskType: string,
  input: Record<string, unknown>,
  result: Record<string, unknown>
): ValidationResult {
  switch (taskType) {
    case 'summarization': {
      const text = result.text as string | undefined;
      if (!text || typeof text !== 'string') {
        return { valid: false, reason: 'Summarization result must contain text string' };
      }
      if (text.length < 20 || text.length > 5000) {
        return { valid: false, reason: `Summarization text length out of range: ${text.length} chars (20-5000)` };
      }
      // Check it's not identical to input
      const inputText = (input.text as string) || '';
      if (text === inputText) {
        return { valid: false, reason: 'Summarization result is identical to input' };
      }
      return { valid: true };
    }

    case 'classification': {
      const text = result.text as string | undefined;
      const categories = (input.categories as string[]) || [];
      if (!text || typeof text !== 'string') {
        return { valid: false, reason: 'Classification result must contain text string' };
      }
      // Check result is one of the valid categories (case-insensitive)
      if (categories.length > 0) {
        const lowerResult = text.toLowerCase().trim();
        const validCategory = categories.some((cat) => lowerResult.includes(cat.toLowerCase()));
        if (!validCategory) {
          return { valid: false, reason: `Classification result "${text}" not in valid categories: ${categories.join(', ')}` };
        }
      }
      return { valid: true };
    }

    case 'embedding': {
      const embedding = result.embedding as number[] | undefined;
      if (!embedding || !Array.isArray(embedding)) {
        // Embeddings can be null if no model available (honest mock)
        if (result.computeMode === 'mock') return { valid: true };
        return { valid: false, reason: 'Embedding result must contain embedding array' };
      }
      if (embedding.length !== 768 && embedding.length !== 384) {
        return { valid: false, reason: `Embedding dimension ${embedding.length} not valid (expected 384 or 768)` };
      }
      // Check values are in valid range
      const outOfRange = embedding.some((v) => typeof v !== 'number' || v < -10 || v > 10);
      if (outOfRange) {
        return { valid: false, reason: 'Embedding values out of range [-10, 10]' };
      }
      return { valid: true };
    }

    case 'assessment': {
      const text = result.text as string | undefined;
      if (!text || typeof text !== 'string' || text.length < 10) {
        return { valid: false, reason: 'Assessment result must contain text with at least 10 chars' };
      }
      return { valid: true };
    }

    default:
      return { valid: true };
  }
}

/**
 * Determine server-side compute mode based on device capabilities.
 * Don't blindly trust client-reported computeMode.
 */
function determineComputeMode(
  device: ComputeDevice,
  clientComputeMode: string
): string {
  const capabilities = parseJson<DeviceCapabilities>(device.capabilities, {
    compute: 0,
    memory: 0,
    models: [],
  });

  // If device is CPU-only with no WebGPU, force mock pricing
  if (device.platform === 'cpu' && !capabilities.models?.length) {
    return 'mock';
  }

  // If device claims webllm but has no models loaded, downgrade
  if (clientComputeMode === 'webllm' && (!capabilities.models || capabilities.models.length === 0)) {
    return 'mock';
  }

  // Trust client for webgpu/cuda/metal platforms with models
  return clientComputeMode;
}

/**
 * Check per-device rate limits. Returns null if OK, error message if exceeded.
 */
async function checkRateLimit(
  db: D1Database,
  deviceId: string,
  deviceTier: DeviceTier
): Promise<string | null> {
  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();

  const countResult = await db.prepare(`
    SELECT COUNT(*) as cnt FROM compute_tasks
    WHERE assigned_device_id = ? AND completed_at > ?
  `).bind(deviceId, oneHourAgo).first<{ cnt: number }>();

  const completedLastHour = countResult?.cnt || 0;
  const maxPerHour = deviceTier === 'crowd' ? 60 : 200;

  if (completedLastHour >= maxPerHour) {
    return `Rate limit exceeded: ${completedLastHour}/${maxPerHour} tasks per hour for ${deviceTier} tier`;
  }

  return null;
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

const VALID_PLATFORMS: readonly string[] = ['cuda', 'metal', 'webgpu', 'cpu'] as const;
const VALID_TASK_TYPES: readonly string[] = [
  'inference', 'embedding', 'assessment', 'planning', 'execution',
  'draft_generation', 'draft_verification', 'summarization', 'classification',
] as const;
const VALID_TIERS: readonly string[] = ['power', 'standard', 'crowd'] as const;

// ============================================================================
// API Routes
// ============================================================================

const compute = new Hono<{ Bindings: Env }>();

// CORS is handled by the parent router (routes.ts)
// Sub-routers don't need their own CORS middleware

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

    if (!VALID_PLATFORMS.includes(platform)) {
      return c.json({ error: `Invalid platform: ${platform}. Must be one of: ${VALID_PLATFORMS.join(', ')}` }, 400);
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

    // Validate device auth token
    const authDeviceId = await validateAuthToken(c);
    if (!authDeviceId) {
      return c.json({ error: 'Valid auth token required. Include Authorization: Bearer <token> header.' }, 401);
    }
    if (authDeviceId !== deviceId) {
      return c.json({ error: 'Device ID does not match auth token' }, 403);
    }

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
      SELECT id, name, tier, platform, capabilities, status, current_task_id, last_heartbeat, stats FROM compute_devices WHERE id = ?
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
        // Assign task to device (with race condition check)
        const assignResult = await c.env.DB.prepare(`
          UPDATE compute_tasks
          SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
          WHERE id = ? AND status = 'pending'
        `).bind(deviceId, now, pendingTask.id).run();

        if (assignResult.meta?.changes && assignResult.meta.changes > 0) {
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
        } else {
          console.log(`[Compute] Task ${pendingTask.id} already claimed by another device`);
        }
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
 *
 * Query params:
 *   tier: 'power' | 'standard' | 'crowd' (optional filter)
 *   status: 'online' | 'busy' | 'offline' | 'paused' (optional filter)
 *   activeOnly: 'true' to only show devices with recent heartbeat (5 min)
 */
compute.get('/devices', async (c) => {
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const tier = c.req.query('tier');
    const status = c.req.query('status');
    const activeOnly = c.req.query('activeOnly') === 'true';

    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();

    let query = 'SELECT id, name, tier, platform, capabilities, endpoint_url, status, current_task_id, last_heartbeat, stats, created_at, updated_at FROM compute_devices WHERE 1=1';
    const params: (string | null)[] = [];

    if (tier) {
      query += ' AND tier = ?';
      params.push(tier);
    }

    if (status) {
      query += ' AND status = ?';
      params.push(status);
    }

    if (activeOnly) {
      query += ' AND last_heartbeat > ?';
      params.push(fiveMinutesAgo);
    }

    query += ' ORDER BY tier DESC, last_heartbeat DESC';

    const result = await c.env.DB.prepare(query).bind(...params).all<ComputeDevice>();

    const devices = (result.results || []).map((d) => {
      const lastHeartbeat = d.last_heartbeat ? new Date(d.last_heartbeat) : null;
      const isRecentlyActive = lastHeartbeat && lastHeartbeat > new Date(fiveMinutesAgo);

      return {
        id: d.id,
        name: d.name,
        tier: d.tier,
        platform: d.platform,
        capabilities: parseJson<DeviceCapabilities>(d.capabilities, { compute: 0, memory: 0, models: [] }),
        status: d.status,
        // Add computed "actuallyOnline" field based on heartbeat
        actuallyOnline: isRecentlyActive && (d.status === 'online' || d.status === 'busy'),
        currentTaskId: d.current_task_id,
        lastHeartbeat: d.last_heartbeat,
        stats: parseJson(d.stats, { tasksCompleted: 0, creditsEarned: 0, totalComputeTime: 0 }),
      };
    });

    // Calculate network stats - only count truly active devices
    const actuallyOnline = devices.filter((d) => d.actuallyOnline);
    const totalCompute = actuallyOnline.reduce((sum, d) => sum + d.capabilities.compute, 0);

    return c.json({
      devices,
      count: devices.length,
      online: actuallyOnline.length,
      totalCompute: Math.round(totalCompute * 10) / 10,
      byTier: {
        power: actuallyOnline.filter((d) => d.tier === 'power').length,
        standard: actuallyOnline.filter((d) => d.tier === 'standard').length,
        crowd: actuallyOnline.filter((d) => d.tier === 'crowd').length,
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
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const body = await c.req.json();
    const { type, input, config, priority, minTier, parentTaskId } = body;

    if (!type || !input) {
      return c.json({ error: 'Missing required fields: type, input' }, 400);
    }

    if (!VALID_TASK_TYPES.includes(type)) {
      return c.json({ error: `Invalid task type: ${type}. Must be one of: ${VALID_TASK_TYPES.join(', ')}` }, 400);
    }

    if (minTier && !VALID_TIERS.includes(minTier)) {
      return c.json({ error: `Invalid tier: ${minTier}. Must be one of: ${VALID_TIERS.join(', ')}` }, 400);
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
      SELECT id, name, tier, platform, capabilities, status, current_task_id, last_heartbeat
      FROM compute_devices
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

    let assignedToDevice: string | null = null;
    if (device) {
      // Assign task to device immediately (with race condition check)
      const assignResult = await c.env.DB.prepare(`
        UPDATE compute_tasks
        SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
        WHERE id = ? AND status = 'pending'
      `).bind(device.id, now, id).run();

      if (assignResult.meta?.changes && assignResult.meta.changes > 0) {
        await c.env.DB.prepare(`
          UPDATE compute_devices
          SET status = 'busy', current_task_id = ?, updated_at = ?
          WHERE id = ?
        `).bind(id, now, device.id).run();

        assignedToDevice = device.id;
        console.log(`[Compute] Task ${id} immediately assigned to ${device.id}`);
      }
    }

    return c.json({
      success: true,
      task: {
        id,
        type,
        status: assignedToDevice ? 'assigned' : 'pending',
        assignedDeviceId: assignedToDevice,
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
    const limit = Math.min(parseInt(c.req.query('limit') || '1', 10), 50);

    if (!deviceId) {
      return c.json({ error: 'deviceId is required' }, 400);
    }

    // Validate device auth token
    const authDeviceId = await validateAuthToken(c);
    if (!authDeviceId) {
      return c.json({ error: 'Valid auth token required. Include Authorization: Bearer <token> header.' }, 401);
    }
    if (authDeviceId !== deviceId) {
      return c.json({ error: 'Device ID does not match auth token' }, 403);
    }

    // Get device info
    const device = await c.env.DB.prepare(`
      SELECT id, tier, platform, capabilities, status FROM compute_devices WHERE id = ?
    `).bind(deviceId).first<ComputeDevice>();

    if (!device) {
      return c.json({ error: 'Device not found' }, 404);
    }

    // Find pending tasks that this device can handle
    const result = await c.env.DB.prepare(`
      SELECT id, type, input, config, priority, min_tier, created_at FROM compute_tasks
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

    // Validate auth token (required)
    const authDeviceId = await validateAuthToken(c);
    if (!authDeviceId) {
      return c.json({ error: 'Valid auth token required. Include Authorization: Bearer <token> header.' }, 401);
    }
    if (authDeviceId !== deviceId) {
      return c.json({ error: 'Device ID does not match auth token' }, 403);
    }

    const now = new Date().toISOString();

    // Get device info (exclude auth_token)
    const device = await c.env.DB.prepare(`
      SELECT id, name, tier, platform, capabilities, endpoint_url, status,
             current_task_id, last_heartbeat, stats, created_at, updated_at
      FROM compute_devices WHERE id = ?
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
        SELECT id, type, input, config, status, priority, min_tier, assigned_device_id,
               result, error, parent_task_id, created_at, assigned_at, completed_at
        FROM compute_tasks WHERE id = ?
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
      SELECT id, type, input, config, status, priority, min_tier, assigned_device_id,
             created_at, assigned_at
      FROM compute_tasks
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

    // Assign task to device (with race condition check)
    const assignResult = await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
      WHERE id = ? AND status = 'pending'
    `).bind(deviceId, now, task.id).run();

    if (!assignResult.meta?.changes || assignResult.meta.changes === 0) {
      // Another device claimed it between SELECT and UPDATE
      return c.json({
        hasWork: false,
        message: 'Task was claimed by another device, try again',
        deviceTier: device.tier,
      });
    }

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
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const body = await c.req.json();
    const { projectId, types } = body;
    // Cap count to prevent resource exhaustion (max 100 tasks per call)
    const count = Math.min(Math.max(body.count || 10, 1), 100);

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

// ============================================================================
// Phase 1: Real Task Generation from Papers
// ============================================================================

/**
 * POST /tasks/from-paper - Generate compute tasks from a research paper
 *
 * Request body:
 * {
 *   paperId: string,
 *   title: string,
 *   abstract: string,
 *   sections?: { title: string, text: string }[]
 * }
 *
 * Generates 3-5 crowd-tier tasks per paper:
 * - summarization of abstract + sections
 * - classification by domain relevance
 * - embedding for semantic search
 * - assessment of paper quality
 */
compute.post('/tasks/from-paper', async (c) => {
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const body = await c.req.json();
    const { paperId, title, abstract, sections } = body;

    if (!paperId || !title || !abstract) {
      return c.json({ error: 'Missing required fields: paperId, title, abstract' }, 400);
    }

    const now = new Date().toISOString();
    const generatedTasks: { id: string; type: string; section?: string }[] = [];

    // Calculate priority based on recency (higher for newer papers)
    const basePriority = 5;

    // 1. Summarization of abstract
    const abstractTaskId = generateId('ctask');
    await c.env.DB.prepare(`
      INSERT INTO compute_tasks (
        id, type, input, config, status, priority, min_tier, created_at
      ) VALUES (?, 'summarization', ?, ?, 'pending', ?, 'crowd', ?)
    `).bind(
      abstractTaskId,
      JSON.stringify({
        text: abstract,
        maxLength: 150,
        source: { type: 'paper', paperId, section: 'abstract' },
        paperTitle: title,
      }),
      JSON.stringify({ maxTokens: 256, temperature: 0.5 }),
      basePriority + 1,
      now
    ).run();
    generatedTasks.push({ id: abstractTaskId, type: 'summarization', section: 'abstract' });

    // 2. Classification by domain
    const classTaskId = generateId('ctask');
    await c.env.DB.prepare(`
      INSERT INTO compute_tasks (
        id, type, input, config, status, priority, min_tier, created_at
      ) VALUES (?, 'classification', ?, ?, 'pending', ?, 'crowd', ?)
    `).bind(
      classTaskId,
      JSON.stringify({
        text: `${title}\n\n${abstract.slice(0, 500)}`,
        categories: ['voice_synthesis', 'speech_recognition', 'nlp', 'audio_processing', 'machine_learning', 'other'],
        source: { type: 'paper', paperId, section: 'classification' },
        paperTitle: title,
      }),
      JSON.stringify({ maxTokens: 64, temperature: 0.3 }),
      basePriority,
      now
    ).run();
    generatedTasks.push({ id: classTaskId, type: 'classification', section: 'classification' });

    // 3. Embedding for semantic search
    const embedTaskId = generateId('ctask');
    await c.env.DB.prepare(`
      INSERT INTO compute_tasks (
        id, type, input, config, status, priority, min_tier, created_at
      ) VALUES (?, 'embedding', ?, ?, 'pending', ?, 'crowd', ?)
    `).bind(
      embedTaskId,
      JSON.stringify({
        text: `${title}. ${abstract.slice(0, 300)}`,
        model: 'default',
        source: { type: 'paper', paperId, section: 'embedding' },
        paperTitle: title,
      }),
      JSON.stringify({ maxTokens: 32, temperature: 0 }),
      basePriority - 1,
      now
    ).run();
    generatedTasks.push({ id: embedTaskId, type: 'embedding', section: 'embedding' });

    // 4. Quality/relevance assessment
    const assessTaskId = generateId('ctask');
    await c.env.DB.prepare(`
      INSERT INTO compute_tasks (
        id, type, input, config, status, priority, min_tier, created_at
      ) VALUES (?, 'assessment', ?, ?, 'pending', ?, 'crowd', ?)
    `).bind(
      assessTaskId,
      JSON.stringify({
        prompt: `Rate the relevance and quality of this paper for voice synthesis research. Paper: "${title}". Abstract: ${abstract.slice(0, 400)}. Respond with a JSON object: { "relevanceScore": 1-10, "qualityScore": 1-10, "reasoning": "..." }`,
        context: 'Paper quality assessment for voice synthesis lab',
        source: { type: 'paper', paperId, section: 'assessment' },
        paperTitle: title,
      }),
      JSON.stringify({ maxTokens: 256, temperature: 0.3 }),
      basePriority + 1,
      now
    ).run();
    generatedTasks.push({ id: assessTaskId, type: 'assessment', section: 'assessment' });

    // 5. Summarize individual sections if provided
    if (sections && Array.isArray(sections)) {
      for (const section of sections.slice(0, 3)) { // Max 3 section summaries
        const sectionTaskId = generateId('ctask');
        await c.env.DB.prepare(`
          INSERT INTO compute_tasks (
            id, type, input, config, status, priority, min_tier, created_at
          ) VALUES (?, 'summarization', ?, ?, 'pending', ?, 'crowd', ?)
        `).bind(
          sectionTaskId,
          JSON.stringify({
            text: section.text?.slice(0, 2000) || '',
            maxLength: 120,
            source: { type: 'paper', paperId, section: section.title || 'section' },
            paperTitle: title,
          }),
          JSON.stringify({ maxTokens: 200, temperature: 0.5 }),
          basePriority - 1,
          now
        ).run();
        generatedTasks.push({ id: sectionTaskId, type: 'summarization', section: section.title });
      }
    }

    // Upsert into seed_papers table
    await c.env.DB.prepare(`
      INSERT INTO seed_papers (id, title, abstract, source_url, domain, tasks_generated)
      VALUES (?, ?, ?, ?, 'research', ?)
      ON CONFLICT(id) DO UPDATE SET tasks_generated = tasks_generated + ?
    `).bind(
      paperId,
      title,
      abstract.slice(0, 5000),
      body.sourceUrl || null,
      generatedTasks.length,
      generatedTasks.length
    ).run();

    console.log(`[Compute] Generated ${generatedTasks.length} tasks from paper: ${title}`);

    return c.json({
      success: true,
      paperId,
      generated: generatedTasks.length,
      tasks: generatedTasks,
    });
  } catch (error) {
    console.error('[Compute] Generate tasks from paper error:', error);
    return c.json({ error: 'Failed to generate tasks from paper' }, 500);
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

    // Validate device auth token
    const authDeviceId = await validateAuthToken(c);
    if (!authDeviceId) {
      return c.json({ error: 'Valid auth token required. Include Authorization: Bearer <token> header.' }, 401);
    }
    if (authDeviceId !== deviceId) {
      return c.json({ error: 'Device ID does not match auth token' }, 403);
    }

    const now = new Date().toISOString();

    // Check task is still pending
    const task = await c.env.DB.prepare(`
      SELECT id, type, input, config, status, priority, min_tier, assigned_device_id,
             created_at, assigned_at
      FROM compute_tasks WHERE id = ?
    `).bind(taskId).first<ComputeTask>();

    if (!task) {
      return c.json({ error: 'Task not found' }, 404);
    }

    if (task.status !== 'pending') {
      return c.json({ error: 'Task is no longer available', status: task.status }, 409);
    }

    // Assign task (with race condition check)
    const assignResult = await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'assigned', assigned_device_id = ?, assigned_at = ?
      WHERE id = ? AND status = 'pending'
    `).bind(deviceId, now, taskId).run();

    if (!assignResult.meta?.changes || assignResult.meta.changes === 0) {
      return c.json({ error: 'Task was claimed by another device', status: 'conflict' }, 409);
    }

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
 * Shared task completion logic — validates, updates task/device, awards credits.
 * Used by both POST /tasks/:id and POST /tasks/:id/complete to avoid duplication.
 */
async function handleTaskCompletion(
  c: Context<{ Bindings: Env }>,
  taskId: string,
  body: { deviceId: string; success: boolean; result?: Record<string, unknown>; error?: string }
): Promise<Response> {
  const { deviceId, success, result, error } = body;

  if (!deviceId) {
    return c.json({ error: 'deviceId is required' }, 400);
  }

  // Validate auth token (required)
  const authDeviceId = await validateAuthToken(c);
  if (!authDeviceId) {
    return c.json({ error: 'Valid auth token required. Include Authorization: Bearer <token> header.' }, 401);
  }
  if (authDeviceId !== deviceId) {
    return c.json({ error: 'Device ID does not match auth token' }, 403);
  }

  const now = new Date().toISOString();

  // Get task
  const task = await c.env.DB.prepare(`
    SELECT id, type, input, config, status, priority, min_tier, assigned_device_id,
           result, error, parent_task_id, created_at, assigned_at, completed_at
    FROM compute_tasks WHERE id = ?
  `).bind(taskId).first<ComputeTask>();

  if (!task) {
    return c.json({ error: 'Task not found' }, 404);
  }

  // Prevent double-completion (credits would be awarded twice)
  if (task.status === 'completed' || task.status === 'failed') {
    return c.json({ error: 'Task already completed', status: task.status }, 409);
  }

  if (task.assigned_device_id !== deviceId) {
    return c.json({ error: 'Task not assigned to this device' }, 403);
  }

  // Get device for server-side compute mode detection (exclude auth_token)
  const device = await c.env.DB.prepare(`
    SELECT id, name, tier, platform, capabilities, endpoint_url, status,
           current_task_id, last_heartbeat, stats, created_at, updated_at
    FROM compute_devices WHERE id = ?
  `).bind(deviceId).first<ComputeDevice>();

  if (!device) {
    return c.json({ error: 'Device not found' }, 404);
  }

  // Server-side compute mode detection — don't blindly trust client
  const clientComputeMode = (result?.computeMode as string) || 'mock';
  const computeMode = determineComputeMode(device, clientComputeMode);

  if (success && result) {
    // Timing validation
    const metrics = result.metrics as Record<string, unknown> | undefined;
    const computeTime = (metrics?.computeTime as number) || 0;
    const timingCheck = validateTiming(task.type, task.assigned_at, now, computeTime, computeMode);
    if (!timingCheck.valid) {
      console.warn(`[Compute] Timing validation failed for task ${taskId}: ${timingCheck.reason}`);
      return c.json({ error: `Result rejected: ${timingCheck.reason}` }, 422);
    }

    // Content validation
    const taskInput = parseJson<Record<string, unknown>>(task.input, {});
    const contentCheck = validateContent(task.type, taskInput, result);
    if (!contentCheck.valid) {
      console.warn(`[Compute] Content validation failed for task ${taskId}: ${contentCheck.reason}`);
      return c.json({ error: `Result rejected: ${contentCheck.reason}` }, 422);
    }

    // Rate limiting
    const rateLimitError = await checkRateLimit(c.env.DB, deviceId, device.tier as DeviceTier);
    if (rateLimitError) {
      console.warn(`[Compute] Rate limit exceeded for device ${deviceId}: ${rateLimitError}`);
      return c.json({ error: rateLimitError }, 429);
    }
  }

  const creditsToAward = computeMode === 'webllm' ? 1.0 : 0.1;

  // Update task (store server-validated computeMode in result)
  const newStatus = success ? 'completed' : 'failed';
  const validatedResult = result ? { ...result, computeMode, serverValidated: true } : null;
  await c.env.DB.prepare(`
    UPDATE compute_tasks
    SET status = ?, result = ?, error = ?, completed_at = ?
    WHERE id = ?
  `).bind(newStatus, validatedResult ? JSON.stringify(validatedResult) : null, error || null, now, taskId).run();

  // Free up device
  await c.env.DB.prepare(`
    UPDATE compute_devices
    SET status = 'online', current_task_id = NULL, updated_at = ?
    WHERE id = ?
  `).bind(now, deviceId).run();

  // Update device stats if successful
  if (success) {
    const metrics = result?.metrics as Record<string, unknown> | undefined;
    const computeTime = (metrics?.computeTime as number) || 0;
    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET stats = json_set(
        COALESCE(stats, '{}'),
        '$.tasksCompleted', COALESCE(json_extract(stats, '$.tasksCompleted'), 0) + 1,
        '$.creditsEarned', COALESCE(json_extract(stats, '$.creditsEarned'), 0) + ?,
        '$.totalComputeTime', COALESCE(json_extract(stats, '$.totalComputeTime'), 0) + ?
      )
      WHERE id = ?
    `).bind(creditsToAward, computeTime / 1000, deviceId).run();
  }

  console.log(`[Compute] Task ${taskId} completed by ${deviceId}: ${newStatus} (client=${clientComputeMode}, server=${computeMode}, ${creditsToAward} credits)`);

  return c.json({
    success: true,
    taskId,
    status: newStatus,
    creditsAwarded: success ? creditsToAward : 0,
    computeMode,
  });
}

/**
 * POST /tasks/:id - Complete a task (alias for browser device-agent compatibility)
 */
compute.post('/tasks/:id', async (c) => {
  const taskId = c.req.param('id');

  // Skip if this looks like a different route pattern
  if (taskId === 'assign' || taskId === 'pending' || taskId === 'generate-crowd') {
    return c.json({ error: 'Not found' }, 404);
  }

  try {
    const body = await c.req.json();
    return handleTaskCompletion(c, taskId, body);
  } catch (error) {
    console.error('[Compute] Complete task error:', error);
    return c.json({ error: 'Failed to complete task' }, 500);
  }
});

/**
 * POST /tasks/:id/complete - Report task completion
 */
compute.post('/tasks/:id/complete', async (c) => {
  try {
    const taskId = c.req.param('id');
    const body = await c.req.json();
    return handleTaskCompletion(c, taskId, body);
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
      SELECT id, type, input, config, status, priority, min_tier, assigned_device_id,
             result, error, parent_task_id, created_at, assigned_at, completed_at
      FROM compute_tasks WHERE id = ?
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
    // Cap timeout at 30 seconds to prevent resource exhaustion
    const rawTimeout = parseInt(c.req.query('timeout') || '30000', 10);
    const timeout = Math.min(Math.max(rawTimeout, 1000), 30000);

    const startTime = Date.now();
    const pollInterval = 2000; // Poll every 2s (was 500ms — too aggressive for D1)

    while (Date.now() - startTime < timeout) {
      const task = await c.env.DB.prepare(`
        SELECT id, status, result, error, completed_at FROM compute_tasks WHERE id = ?
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
 * POST /cleanup - Mark stale devices as offline and clean up
 *
 * This should be called periodically (e.g., every 5 minutes) to:
 * - Mark devices offline if no heartbeat in 5+ minutes
 * - Reset stuck tasks that were assigned but never completed
 * - Delete very old offline devices (optional)
 */
compute.post('/cleanup', async (c) => {
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const now = new Date();
    const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000).toISOString();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000).toISOString();

    // Mark devices offline if no heartbeat in 5 minutes
    const offlineResult = await c.env.DB.prepare(`
      UPDATE compute_devices
      SET status = 'offline', updated_at = ?
      WHERE status IN ('online', 'busy')
        AND last_heartbeat < ?
    `).bind(now.toISOString(), fiveMinutesAgo).run();

    // Reset tasks that were assigned but device went offline
    const resetResult = await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'pending', assigned_device_id = NULL, assigned_at = NULL
      WHERE status = 'assigned'
        AND assigned_at < ?
    `).bind(oneHourAgo).run();

    // Clear current_task_id for offline devices
    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET current_task_id = NULL
      WHERE status = 'offline' AND current_task_id IS NOT NULL
    `).run();

    console.log(`[Compute] Cleanup: ${offlineResult.meta?.changes || 0} devices marked offline, ${resetResult.meta?.changes || 0} tasks reset`);

    return c.json({
      success: true,
      devicesMarkedOffline: offlineResult.meta?.changes || 0,
      tasksReset: resetResult.meta?.changes || 0,
      timestamp: now.toISOString(),
    });
  } catch (error) {
    console.error('[Compute] Cleanup error:', error);
    return c.json({ error: 'Failed to run cleanup' }, 500);
  }
});

/**
 * DELETE /devices/stale - Delete devices that have been offline for 24+ hours
 */
compute.delete('/devices/stale', async (c) => {
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const now = new Date();
    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString();

    const result = await c.env.DB.prepare(`
      DELETE FROM compute_devices
      WHERE status = 'offline'
        AND last_heartbeat < ?
        AND json_extract(stats, '$.tasksCompleted') = 0
    `).bind(oneDayAgo).run();

    console.log(`[Compute] Deleted ${result.meta?.changes || 0} stale devices`);

    return c.json({
      success: true,
      devicesDeleted: result.meta?.changes || 0,
      timestamp: now.toISOString(),
    });
  } catch (error) {
    console.error('[Compute] Delete stale error:', error);
    return c.json({ error: 'Failed to delete stale devices' }, 500);
  }
});

/**
 * POST /devices/cleanup - Force cleanup: mark stale devices offline and delete old ones
 * This runs the same cleanup as the cron job but on-demand
 */
compute.post('/devices/cleanup', async (c) => {
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const now = new Date().toISOString();
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

    // Step 1: Mark devices offline if no heartbeat in 5 minutes
    const offlineResult = await c.env.DB.prepare(`
      UPDATE compute_devices
      SET status = 'offline', updated_at = ?
      WHERE status IN ('online', 'busy')
        AND last_heartbeat < ?
    `).bind(now, fiveMinutesAgo).run();

    // Step 2: Reset tasks assigned to offline devices
    await c.env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'pending', assigned_device_id = NULL, assigned_at = NULL
      WHERE status = 'assigned'
        AND assigned_device_id IN (SELECT id FROM compute_devices WHERE status = 'offline')
    `).run();

    // Step 3: Clear current_task_id for offline devices
    await c.env.DB.prepare(`
      UPDATE compute_devices
      SET current_task_id = NULL
      WHERE status = 'offline' AND current_task_id IS NOT NULL
    `).run();

    // Step 4: Delete devices offline for 24+ hours with no completed tasks
    const deleteResult = await c.env.DB.prepare(`
      DELETE FROM compute_devices
      WHERE status = 'offline'
        AND last_heartbeat < ?
        AND json_extract(stats, '$.tasksCompleted') = 0
    `).bind(oneDayAgo).run();

    console.log(`[Compute] Cleanup: ${offlineResult.meta?.changes || 0} marked offline, ${deleteResult.meta?.changes || 0} deleted`);

    return c.json({
      success: true,
      markedOffline: offlineResult.meta?.changes || 0,
      devicesDeleted: deleteResult.meta?.changes || 0,
      timestamp: now,
    });
  } catch (error) {
    console.error('[Compute] Cleanup error:', error);
    return c.json({ error: 'Failed to cleanup devices' }, 500);
  }
});

/**
 * GET /stats - Get compute network statistics
 */
compute.get('/stats', async (c) => {
  try {
    // Only count devices with recent heartbeat as "online"
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();

    // Device stats - only count actually online devices
    const deviceStats = await c.env.DB.prepare(`
      SELECT
        COUNT(*) as total,
        SUM(CASE WHEN status IN ('online', 'busy') AND last_heartbeat > ? THEN 1 ELSE 0 END) as online,
        SUM(CASE WHEN tier = 'power' THEN 1 ELSE 0 END) as power,
        SUM(CASE WHEN tier = 'standard' THEN 1 ELSE 0 END) as standard,
        SUM(CASE WHEN tier = 'crowd' THEN 1 ELSE 0 END) as crowd
      FROM compute_devices
    `).bind(fiveMinutesAgo).first();

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

    // Total compute power (from actually online devices with recent heartbeat)
    const computeResult = await c.env.DB.prepare(`
      SELECT SUM(json_extract(capabilities, '$.compute')) as total_compute
      FROM compute_devices
      WHERE status IN ('online', 'busy')
        AND last_heartbeat > ?
    `).bind(fiveMinutesAgo).first<{ total_compute: number }>();

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
    const checkBadges = (tasks: number, credits: number, tier: unknown) => {
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
    const leaderboard = (result.results || []).map((d: Record<string, unknown>) => {
      const tasksCompleted = Number(d.tasks_completed) || 0;
      const creditsEarned = Number(d.credits_earned) || 0;
      const computeTime = Number(d.total_compute_time) || 0;

      const deviceId = String(d.id || '');

      return {
        userId: deviceId,
        displayName: d.name || `Contributor ${deviceId.slice(-6)}`,
        rank: calculateRank(tasksCompleted),
        totalCreditsEarned: creditsEarned,
        totalTasksCompleted: tasksCompleted,
        totalComputeTime: Math.round(computeTime * 10) / 10,
        devices: [deviceId],
        badges: checkBadges(tasksCompleted, creditsEarned, d.tier),
        joinedAt: d.created_at,
        // Extra fields for debugging
        tier: d.tier,
        platform: d.platform,
      };
    });

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

// ============================================================================
// Phase 4: Results API
// ============================================================================

/**
 * GET /results - Get aggregated compute results
 *
 * Query params:
 *   type: task type filter
 *   paperId: filter by paper ID (from task input source)
 *   status: filter by status (default: completed)
 *   limit: max results (default 50, max 200)
 *   offset: pagination offset
 *   aggregated: 'true' for aggregated stats only (public), 'false' for full results (admin)
 */
compute.get('/results', async (c) => {
  try {
    const type = c.req.query('type');
    const paperId = c.req.query('paperId');
    const status = c.req.query('status') || 'completed';
    const limit = Math.min(parseInt(c.req.query('limit') || '50', 10), 200);
    const offset = parseInt(c.req.query('offset') || '0', 10);
    const aggregatedOnly = c.req.query('aggregated') !== 'false';

    if (!aggregatedOnly) {
      // Full results require admin auth
      const adminError = requireAdmin(c);
      if (adminError) return adminError;
    }

    let query = `SELECT id, type, input, config, status, priority, min_tier, assigned_device_id, result, error, created_at, assigned_at, completed_at FROM compute_tasks WHERE status = ?`;
    const params: (string | number)[] = [status];

    if (type) {
      query += ' AND type = ?';
      params.push(type);
    }

    if (paperId) {
      query += ` AND json_extract(input, '$.source.paperId') = ?`;
      params.push(paperId);
    }

    query += ' ORDER BY completed_at DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    const result = await c.env.DB.prepare(query).bind(...params).all<ComputeTask>();
    const tasks = result.results || [];

    if (aggregatedOnly) {
      // Return aggregated stats (public)
      const summaryCount = tasks.filter((t) => t.type === 'summarization').length;
      const classCount = tasks.filter((t) => t.type === 'classification').length;
      const embedCount = tasks.filter((t) => t.type === 'embedding').length;
      const assessCount = tasks.filter((t) => t.type === 'assessment').length;

      // Extract unique paper IDs
      const paperIds = new Set<string>();
      for (const t of tasks) {
        const input = parseJson<Record<string, unknown>>(t.input, {});
        const source = input.source as Record<string, unknown> | undefined;
        if (source?.paperId) {
          paperIds.add(source.paperId as string);
        }
      }

      return c.json({
        totalResults: tasks.length,
        byType: {
          summarization: summaryCount,
          classification: classCount,
          embedding: embedCount,
          assessment: assessCount,
        },
        uniquePapers: paperIds.size,
        offset,
        limit,
      });
    }

    // Full results (admin only)
    const fullResults = tasks.map((t) => ({
      id: t.id,
      type: t.type,
      input: parseJson(t.input, {}),
      result: parseJson(t.result, null),
      status: t.status,
      assignedDeviceId: t.assigned_device_id,
      completedAt: t.completed_at,
    }));

    return c.json({
      results: fullResults,
      count: fullResults.length,
      offset,
      limit,
    });
  } catch (error) {
    console.error('[Compute] Results error:', error);
    return c.json({ error: 'Failed to get results' }, 500);
  }
});

/**
 * DELETE /tasks/:id - Delete a task (admin only)
 */
compute.delete('/tasks/:id', async (c) => {
  const adminError = requireAdmin(c);
  if (adminError) return adminError;

  try {
    const taskId = c.req.param('id');

    // Check if the task is assigned to a device, and clear the device's current_task_id
    const task = await c.env.DB.prepare(`
      SELECT assigned_device_id, status FROM compute_tasks WHERE id = ?
    `).bind(taskId).first<{ assigned_device_id: string | null; status: string }>();

    if (!task) {
      return c.json({ error: 'Task not found' }, 404);
    }

    // If task is assigned/processing, free the device first
    if (task.assigned_device_id && (task.status === 'assigned' || task.status === 'processing')) {
      await c.env.DB.prepare(`
        UPDATE compute_devices
        SET status = 'online', current_task_id = NULL, updated_at = ?
        WHERE id = ? AND current_task_id = ?
      `).bind(new Date().toISOString(), task.assigned_device_id, taskId).run();
    }

    await c.env.DB.prepare(`
      DELETE FROM compute_tasks WHERE id = ?
    `).bind(taskId).run();

    return c.json({ success: true, taskId });
  } catch (error) {
    console.error('[Compute] Delete task error:', error);
    return c.json({ error: 'Failed to delete task' }, 500);
  }
});

export default compute;
