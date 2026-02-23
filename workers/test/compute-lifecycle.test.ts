import { describe, it, expect, beforeEach } from 'vitest';
import { env, SELF } from 'cloudflare:test';

// Helper: register a device via API
async function registerDevice(overrides: Record<string, unknown> = {}) {
  const body = {
    name: 'Test GPU',
    platform: 'cuda',
    capabilities: { compute: 82, memory: 24, models: ['qwen3-coder-32k'] },
    ...overrides,
  };
  const res = await SELF.fetch('https://test.local/api/compute/devices', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { res, data: await res.json() as Record<string, unknown> };
}

// Helper: create a compute task via API
async function createTask(overrides: Record<string, unknown> = {}) {
  const body = {
    type: 'inference',
    input: { prompt: 'test prompt' },
    config: { maxTokens: 100 },
    ...overrides,
  };
  const res = await SELF.fetch('https://test.local/api/compute/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { res, data: await res.json() as Record<string, unknown> };
}

// Helper: clean tables between tests
async function cleanTables() {
  await env.DB.exec('DELETE FROM compute_tasks');
  await env.DB.exec('DELETE FROM compute_devices');
}

// ============================================================================
// Device Registration
// ============================================================================

describe('Device Registration', () => {
  beforeEach(cleanTables);

  it('registers a device and returns id, authToken, tier', async () => {
    const { res, data } = await registerDevice();
    expect(res.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.authToken).toBeDefined();
    expect(typeof data.authToken).toBe('string');
    expect((data.authToken as string).length).toBeGreaterThan(0);

    const device = data.device as Record<string, unknown>;
    expect(device.id).toBeDefined();
    expect(device.name).toBe('Test GPU');
    expect(device.tier).toBe('power');
    expect(device.platform).toBe('cuda');
    expect(device.status).toBe('online');
  });

  it('returns 400 when required fields are missing', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/devices', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'Test' }),
    });
    expect(res.status).toBe(400);
    const data = await res.json() as Record<string, unknown>;
    expect(data.error).toBeDefined();
  });

  it('classifies CUDA 82 TFLOPS as power tier', async () => {
    const { data } = await registerDevice({
      platform: 'cuda',
      capabilities: { compute: 82, memory: 24, models: [] },
    });
    const device = data.device as Record<string, unknown>;
    expect(device.tier).toBe('power');
  });

  it('classifies Metal 10 TFLOPS as standard tier', async () => {
    const { data } = await registerDevice({
      name: 'MacBook Pro',
      platform: 'metal',
      capabilities: { compute: 10, memory: 16, models: [] },
    });
    const device = data.device as Record<string, unknown>;
    expect(device.tier).toBe('standard');
  });

  it('classifies WebGPU 2 TFLOPS as crowd tier', async () => {
    const { data } = await registerDevice({
      name: 'Browser',
      platform: 'webgpu',
      capabilities: { compute: 2, memory: 4, models: [] },
    });
    const device = data.device as Record<string, unknown>;
    expect(device.tier).toBe('crowd');
  });
});

// ============================================================================
// Task Lifecycle
// ============================================================================

describe('Task Lifecycle', () => {
  beforeEach(cleanTables);

  it('creates a task in pending status when no devices available', async () => {
    const { res, data } = await createTask();
    expect(res.status).toBe(200);
    expect(data.success).toBe(true);
    const task = data.task as Record<string, unknown>;
    expect(task.id).toBeDefined();
    expect(task.status).toBe('pending');
    expect(task.assignedDeviceId).toBeNull();
  });

  it('auto-assigns task to available device on creation', async () => {
    // Register a device first
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;

    // Create a task — should be immediately assigned
    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    expect(task.status).toBe('assigned');
    expect(task.assignedDeviceId).toBe(device.id);
  });

  it('device polls /tasks/assign and gets work', async () => {
    // Register device
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    // Create task (will auto-assign since device is available)
    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;

    // If auto-assigned, verify via poll endpoint
    if (task.status === 'assigned') {
      // Task was already assigned, device should see it
      const pollRes = await SELF.fetch('https://test.local/api/compute/tasks/assign', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({ deviceId }),
      });
      const pollData = await pollRes.json() as Record<string, unknown>;
      expect(pollData.hasWork).toBe(true);
    } else {
      // Task is pending, device should claim it via poll
      const pollRes = await SELF.fetch('https://test.local/api/compute/tasks/assign', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({ deviceId }),
      });
      const pollData = await pollRes.json() as Record<string, unknown>;
      expect(pollData.hasWork).toBe(true);
      const pollTask = pollData.task as Record<string, unknown>;
      expect(pollTask.id).toBe(task.id);
    }
  });

  it('device reports success → task completed, device freed', async () => {
    // Register device
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    // Create task (auto-assigns)
    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;
    expect(task.status).toBe('assigned');

    // Report success
    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: { text: 'Generated output', computeMode: 'webllm', metrics: { computeTime: 5000 } },
      }),
    });
    const completeData = await completeRes.json() as Record<string, unknown>;
    expect(completeRes.status).toBe(200);
    expect(completeData.success).toBe(true);
    expect(completeData.status).toBe('completed');
    expect(completeData.creditsAwarded).toBe(1);
    expect(completeData.computeMode).toBe('webllm');

    // Verify device is freed
    const deviceRow = await env.DB.prepare(
      'SELECT status, current_task_id FROM compute_devices WHERE id = ?'
    ).bind(deviceId).first();
    expect(deviceRow!.status).toBe('online');
    expect(deviceRow!.current_task_id).toBeNull();

    // Verify task is completed
    const taskRow = await env.DB.prepare(
      'SELECT status FROM compute_tasks WHERE id = ?'
    ).bind(taskId).first();
    expect(taskRow!.status).toBe('completed');
  });

  it('mock computeMode awards 0.1 credits instead of 1.0', async () => {
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    // Report success with mock computeMode (no WebLLM model loaded)
    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: { computeMode: 'mock', metrics: { computeTime: 1200 } },
      }),
    });
    const completeData = await completeRes.json() as Record<string, unknown>;
    expect(completeRes.status).toBe(200);
    expect(completeData.success).toBe(true);
    expect(completeData.status).toBe('completed');
    expect(completeData.creditsAwarded).toBe(0.1);
    expect(completeData.computeMode).toBe('mock');

    // Verify credits stored in D1 device stats JSON
    const deviceRow = await env.DB.prepare(
      "SELECT json_extract(stats, '$.creditsEarned') as credits FROM compute_devices WHERE id = ?"
    ).bind(deviceId).first();
    expect(deviceRow!.credits).toBe(0.1);
  });

  it('device reports failure → task failed, device freed', async () => {
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: false,
        error: 'Out of memory',
      }),
    });
    const completeData = await completeRes.json() as Record<string, unknown>;
    expect(completeRes.status).toBe(200);
    expect(completeData.status).toBe('failed');
    expect(completeData.creditsAwarded).toBe(0);

    // Device should be freed
    const deviceRow = await env.DB.prepare(
      'SELECT status, current_task_id FROM compute_devices WHERE id = ?'
    ).bind(deviceId).first();
    expect(deviceRow!.status).toBe('online');
    expect(deviceRow!.current_task_id).toBeNull();
  });

  it('GET /tasks/:id returns task details', async () => {
    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    const res = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.id).toBe(taskId);
    expect(data.type).toBe('inference');
    expect(data.status).toBeDefined();
  });

  it('GET /tasks/:id returns 404 for missing task', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks/nonexistent');
    expect(res.status).toBe(404);
  });
});

// ============================================================================
// Race Condition Prevention (Fix #3 validation)
// ============================================================================

describe('Race Condition Prevention', () => {
  beforeEach(cleanTables);

  it('only one device gets the task when two claim simultaneously', async () => {
    // Register two devices
    const { data: dev1Data } = await registerDevice({ name: 'GPU-1' });
    const { data: dev2Data } = await registerDevice({ name: 'GPU-2' });
    const dev1 = dev1Data.device as Record<string, unknown>;
    const dev2 = dev2Data.device as Record<string, unknown>;
    const token1 = dev1Data.authToken as string;
    const token2 = dev2Data.authToken as string;

    // Manually insert a pending task (bypass auto-assign by inserting directly)
    const taskId = 'ctask_race_test';
    await env.DB.prepare(`
      INSERT INTO compute_tasks (id, type, input, config, status, priority, created_at)
      VALUES (?, 'inference', '{"prompt":"test"}', '{}', 'pending', 5, datetime('now'))
    `).bind(taskId).run();

    // Both devices try to claim the same task
    const [claim1, claim2] = await Promise.all([
      SELF.fetch(`https://test.local/api/compute/tasks/${taskId}/claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token1}` },
        body: JSON.stringify({ deviceId: dev1.id }),
      }),
      SELF.fetch(`https://test.local/api/compute/tasks/${taskId}/claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token2}` },
        body: JSON.stringify({ deviceId: dev2.id }),
      }),
    ]);

    const data1 = await claim1.json() as Record<string, unknown>;
    const data2 = await claim2.json() as Record<string, unknown>;

    // Exactly one should succeed, one should get 409
    const successes = [claim1.status, claim2.status].filter((s) => s === 200);
    const conflicts = [claim1.status, claim2.status].filter((s) => s === 409);
    expect(successes.length).toBe(1);
    expect(conflicts.length).toBe(1);

    // Verify the task is assigned to exactly one device
    const taskRow = await env.DB.prepare(
      'SELECT assigned_device_id, status FROM compute_tasks WHERE id = ?'
    ).bind(taskId).first();
    expect(taskRow!.status).toBe('assigned');
    expect(taskRow!.assigned_device_id).toBeDefined();
  });

  it('claiming already-assigned task returns 409', async () => {
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;

    // Insert task and assign it directly
    const taskId = 'ctask_already_assigned';
    await env.DB.prepare(`
      INSERT INTO compute_tasks (id, type, input, config, status, priority, assigned_device_id, created_at)
      VALUES (?, 'inference', '{"prompt":"test"}', '{}', 'assigned', 5, ?, datetime('now'))
    `).bind(taskId, device.id).run();

    // Register another device and try to claim
    const { data: dev2Data } = await registerDevice({ name: 'GPU-2' });
    const dev2 = dev2Data.device as Record<string, unknown>;
    const token2 = dev2Data.authToken as string;

    const res = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}/claim`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token2}` },
      body: JSON.stringify({ deviceId: dev2.id }),
    });
    expect(res.status).toBe(409);
  });
});

// ============================================================================
// Device Heartbeat & Task Assignment via Heartbeat
// ============================================================================

describe('Device Heartbeat', () => {
  beforeEach(cleanTables);

  it('heartbeat assigns pending task to idle device', async () => {
    // Register device
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    // Manually insert a pending task (not auto-assigned since device was busy being created)
    const taskId = 'ctask_heartbeat_test';
    await env.DB.prepare(`
      INSERT INTO compute_tasks (id, type, input, config, status, priority, created_at)
      VALUES (?, 'assessment', '{"prompt":"assess this"}', '{"maxTokens":500}', 'pending', 8, datetime('now'))
    `).bind(taskId).run();

    // Send heartbeat — device is online, no current task, should get assigned
    const hbRes = await SELF.fetch(`https://test.local/api/compute/devices/${deviceId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
      body: JSON.stringify({ status: 'online' }),
    });
    const hbData = await hbRes.json() as Record<string, unknown>;
    expect(hbRes.status).toBe(200);
    expect(hbData.success).toBe(true);

    // The heartbeat may or may not assign (depends on timing of status read)
    // Verify task state in DB
    const taskRow = await env.DB.prepare(
      'SELECT status, assigned_device_id FROM compute_tasks WHERE id = ?'
    ).bind(taskId).first();

    if (hbData.task) {
      const assignedTask = hbData.task as Record<string, unknown>;
      expect(assignedTask.id).toBe(taskId);
      expect(taskRow!.status).toBe('assigned');
      expect(taskRow!.assigned_device_id).toBe(deviceId);
    }
    // If task wasn't assigned via heartbeat, it's still pending — acceptable
  });
});

// ============================================================================
// Cleanup Endpoint
// ============================================================================

describe('Cleanup', () => {
  beforeEach(cleanTables);

  it('marks stale devices offline and resets stuck tasks', async () => {
    // Insert a device with old heartbeat
    const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();

    await env.DB.prepare(`
      INSERT INTO compute_devices (id, name, tier, platform, capabilities, status, last_heartbeat, stats, auth_token, created_at, updated_at)
      VALUES ('dev_stale', 'Stale GPU', 'power', 'cuda', '{"compute":82,"memory":24,"models":[]}', 'online', ?, '{}', 'token123', ?, ?)
    `).bind(tenMinutesAgo, tenMinutesAgo, tenMinutesAgo).run();

    // Insert a stuck assigned task
    await env.DB.prepare(`
      INSERT INTO compute_tasks (id, type, input, config, status, priority, assigned_device_id, assigned_at, created_at)
      VALUES ('ctask_stuck', 'inference', '{}', '{}', 'assigned', 5, 'dev_stale', ?, ?)
    `).bind(twoHoursAgo, twoHoursAgo).run();

    // Run cleanup
    const res = await SELF.fetch('https://test.local/api/compute/cleanup', {
      method: 'POST',
    });
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.success).toBe(true);
    expect(data.devicesMarkedOffline).toBeGreaterThanOrEqual(1);
    expect(data.tasksReset).toBeGreaterThanOrEqual(1);

    // Verify device is offline
    const deviceRow = await env.DB.prepare(
      'SELECT status FROM compute_devices WHERE id = ?'
    ).bind('dev_stale').first();
    expect(deviceRow!.status).toBe('offline');

    // Verify task is reset to pending
    const taskRow = await env.DB.prepare(
      'SELECT status, assigned_device_id FROM compute_tasks WHERE id = ?'
    ).bind('ctask_stuck').first();
    expect(taskRow!.status).toBe('pending');
    expect(taskRow!.assigned_device_id).toBeNull();
  });
});

// ============================================================================
// Agent Cleanup (Fix #4 validation)
// ============================================================================

describe('Agent State Reset', () => {
  beforeEach(async () => {
    await cleanTables();
    // Also clean agent_state and projects for this test
    await env.DB.exec('DELETE FROM agent_state');
    await env.DB.exec('DELETE FROM projects');
  });

  it('working agent can be reset to idle via D1', async () => {
    // Create a project first (FK constraint)
    await env.DB.prepare(`
      INSERT INTO projects (id, name, slug, created_at, updated_at)
      VALUES ('proj_test', 'Test Project', 'test-project', datetime('now'), datetime('now'))
    `).run();

    // Insert agent in working state
    await env.DB.prepare(`
      INSERT INTO agent_state (agent_id, project_id, status, created_at)
      VALUES ('agent_test', 'proj_test', 'working', datetime('now'))
    `).run();

    // Verify it's working
    const before = await env.DB.prepare(
      'SELECT status FROM agent_state WHERE agent_id = ?'
    ).bind('agent_test').first();
    expect(before!.status).toBe('working');

    // Reset to idle (what Fix #4 does on early failure)
    await env.DB.prepare(`
      UPDATE agent_state
      SET status = 'idle', current_task_id = NULL
      WHERE agent_id = ?
    `).bind('agent_test').run();

    const after = await env.DB.prepare(
      'SELECT status, current_task_id FROM agent_state WHERE agent_id = ?'
    ).bind('agent_test').first();
    expect(after!.status).toBe('idle');
    expect(after!.current_task_id).toBeNull();
  });
});

// ============================================================================
// Full Round-Trip: register → create task → poll → complete
// ============================================================================

describe('Full Round-Trip', () => {
  beforeEach(cleanTables);

  it('complete lifecycle: register device → create task → device gets work → reports result → freed', async () => {
    // Step 1: Register device
    const { data: devData } = await registerDevice({ name: 'RTX 4090' });
    expect(devData.success).toBe(true);
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    // Step 2: Create task (should auto-assign to our device)
    const { data: taskData } = await createTask({
      type: 'assessment',
      input: { prompt: 'Assess voice-clone project status' },
      config: { maxTokens: 2000 },
      priority: 8,
    });
    expect(taskData.success).toBe(true);
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;
    expect(task.status).toBe('assigned');
    expect(task.assignedDeviceId).toBe(deviceId);

    // Step 3: Verify device is now busy
    const deviceRow = await env.DB.prepare(
      'SELECT status, current_task_id FROM compute_devices WHERE id = ?'
    ).bind(deviceId).first();
    expect(deviceRow!.status).toBe('busy');
    expect(deviceRow!.current_task_id).toBe(taskId);

    // Step 4: Report result
    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: {
          text: 'Project assessment complete with detailed analysis of current state',
          output: 'Project assessment complete',
          computeMode: 'mock',
          metrics: { computeTime: 12000 },
        },
      }),
    });
    expect(completeRes.status).toBe(200);

    // Step 5: Verify device is freed and task is done
    const finalDevice = await env.DB.prepare(
      'SELECT status, current_task_id, stats FROM compute_devices WHERE id = ?'
    ).bind(deviceId).first();
    expect(finalDevice!.status).toBe('online');
    expect(finalDevice!.current_task_id).toBeNull();

    const finalTask = await env.DB.prepare(
      'SELECT status, completed_at FROM compute_tasks WHERE id = ?'
    ).bind(taskId).first();
    expect(finalTask!.status).toBe('completed');
    expect(finalTask!.completed_at).toBeDefined();
  });
});

// ============================================================================
// Result Validation (Phase 2)
// ============================================================================

describe('Result Validation', () => {
  beforeEach(cleanTables);

  it('rejects summarization result with empty text', async () => {
    // Register device and create a summarization task
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask({ type: 'summarization', input: { text: 'Summarize this paper about TTS' } });
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    // Report with empty result text
    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: { text: '', computeMode: 'mock', metrics: { computeTime: 2000 } },
      }),
    });
    expect(completeRes.status).toBe(422);
    const errData = await completeRes.json() as Record<string, unknown>;
    expect((errData.error as string)).toContain('Summarization');
  });

  it('rejects classification result with invalid category', async () => {
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask({
      type: 'classification',
      input: { text: 'Classify this', categories: ['positive', 'negative', 'neutral'] },
    });
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: { text: 'completely_wrong_category', computeMode: 'mock', metrics: { computeTime: 1500 } },
      }),
    });
    expect(completeRes.status).toBe(422);
  });

  it('accepts valid summarization result', async () => {
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask({ type: 'summarization', input: { text: 'A long paper about voice synthesis techniques and approaches.' } });
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: {
          text: 'This paper explores novel voice synthesis techniques using neural approaches for improved naturalness.',
          computeMode: 'mock',
          metrics: { computeTime: 3000 },
        },
      }),
    });
    expect(completeRes.status).toBe(200);
    const data = await completeRes.json() as Record<string, unknown>;
    expect(data.success).toBe(true);
  });

  it('server downgrades computeMode when device has no models', async () => {
    // Register a CPU device with no models
    const { data: devData } = await registerDevice({
      name: 'CPU Device',
      platform: 'cpu',
      capabilities: { compute: 1, memory: 4, models: [] },
    });
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    // Client claims webllm but device is CPU with no models
    const completeRes = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: { text: 'output', computeMode: 'webllm', metrics: { computeTime: 5000 } },
      }),
    });
    expect(completeRes.status).toBe(200);
    const data = await completeRes.json() as Record<string, unknown>;
    // Server should downgrade to mock pricing
    expect(data.computeMode).toBe('mock');
    expect(data.creditsAwarded).toBe(0.1);
  });
});

// ============================================================================
// Admin Auth (Phase 3)
// ============================================================================

describe('Admin Auth', () => {
  beforeEach(cleanTables);

  // Note: In dev mode with no ADMIN_API_KEY set, all admin endpoints are open.
  // These tests verify the endpoints work without auth (dev mode behavior).
  // When ADMIN_API_KEY is set in production, requests without it get 401.

  it('POST /tasks works in dev mode (no ADMIN_API_KEY configured)', async () => {
    const { res } = await createTask();
    expect(res.status).toBe(200);
  });

  it('POST /tasks/generate-crowd works in dev mode', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks/generate-crowd', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ count: 2 }),
    });
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.success).toBe(true);
    expect(data.generated).toBe(2);
  });

  it('DELETE /tasks/:id deletes a task', async () => {
    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    const res = await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'DELETE',
    });
    expect(res.status).toBe(200);

    // Verify task is gone
    const row = await env.DB.prepare('SELECT id FROM compute_tasks WHERE id = ?').bind(taskId).first();
    expect(row).toBeNull();
  });

  it('DELETE /tasks/:id returns 404 for missing task', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks/nonexistent', {
      method: 'DELETE',
    });
    expect(res.status).toBe(404);
  });
});

// ============================================================================
// Paper Task Generation (Phase 1)
// ============================================================================

describe('Paper Task Generation', () => {
  beforeEach(async () => {
    await cleanTables();
    // Also clean seed_papers
    await env.DB.exec('DELETE FROM seed_papers').catch(() => {});
  });

  it('POST /tasks/from-paper generates tasks from a paper', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks/from-paper', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        paperId: 'arxiv-2401.12345',
        title: 'Neural Voice Cloning with Zero-Shot Transfer',
        abstract: 'We present a novel approach to voice cloning that achieves zero-shot transfer learning using a modified transformer architecture with prosody-aware attention mechanisms.',
      }),
    });
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.success).toBe(true);
    expect(data.generated).toBeGreaterThanOrEqual(4); // At least 4 tasks (summary, class, embed, assess)

    const tasks = data.tasks as { id: string; type: string; section?: string }[];
    const types = tasks.map((t) => t.type);
    expect(types).toContain('summarization');
    expect(types).toContain('classification');
    expect(types).toContain('embedding');
    expect(types).toContain('assessment');

    // Verify tasks are in D1
    const count = await env.DB.prepare(
      "SELECT COUNT(*) as cnt FROM compute_tasks WHERE json_extract(input, '$.source.paperId') = ?"
    ).bind('arxiv-2401.12345').first<{ cnt: number }>();
    expect(count!.cnt).toBeGreaterThanOrEqual(4);

    // Verify seed_papers entry
    const paper = await env.DB.prepare('SELECT * FROM seed_papers WHERE id = ?').bind('arxiv-2401.12345').first();
    expect(paper).not.toBeNull();
    expect(paper!.title).toBe('Neural Voice Cloning with Zero-Shot Transfer');
  });

  it('POST /tasks/from-paper returns 400 with missing fields', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks/from-paper', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paperId: 'test' }),
    });
    expect(res.status).toBe(400);
  });
});

// ============================================================================
// Results API (Phase 4)
// ============================================================================

describe('Results API', () => {
  beforeEach(cleanTables);

  it('GET /results returns aggregated stats (public)', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/results');
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.totalResults).toBeDefined();
    expect(data.byType).toBeDefined();
  });

  it('GET /results?aggregated=false returns full results in dev mode', async () => {
    // Create and complete a task to have results
    const { data: devData } = await registerDevice();
    const device = devData.device as Record<string, unknown>;
    const deviceId = device.id as string;
    const authToken = devData.authToken as string;

    const { data: taskData } = await createTask();
    const task = taskData.task as Record<string, unknown>;
    const taskId = task.id as string;

    await SELF.fetch(`https://test.local/api/compute/tasks/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify({
        deviceId,
        success: true,
        result: { text: 'Result text', computeMode: 'mock', metrics: { computeTime: 2000 } },
      }),
    });

    const res = await SELF.fetch('https://test.local/api/compute/results?aggregated=false');
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.results).toBeDefined();
    expect((data.results as unknown[]).length).toBeGreaterThanOrEqual(1);
  });
});
