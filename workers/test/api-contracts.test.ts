import { describe, it, expect, beforeEach } from 'vitest';
import { env, SELF } from 'cloudflare:test';

// Helper: clean tables between tests
async function cleanTables() {
  await env.DB.exec('DELETE FROM compute_tasks');
  await env.DB.exec('DELETE FROM compute_devices');
}

// ============================================================================
// Health Endpoint
// ============================================================================

describe('GET /api/health', () => {
  it('returns ok status', async () => {
    const res = await SELF.fetch('https://test.local/api/health');
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.status).toBe('ok');
    expect(data.timestamp).toBeDefined();
  });
});

// ============================================================================
// Compute Stats Endpoint
// ============================================================================

describe('GET /api/compute/stats', () => {
  beforeEach(cleanTables);

  it('returns expected shape with devices, tasks, totalCompute', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/stats');
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;

    // Verify top-level shape
    expect(data.devices).toBeDefined();
    expect(data.tasks).toBeDefined();
    expect(typeof data.totalCompute).toBe('number');
    expect(data.timestamp).toBeDefined();

    // Verify devices shape
    const devices = data.devices as Record<string, unknown>;
    expect(typeof devices.total).toBe('number');
    expect(typeof devices.online).toBe('number');
    expect(devices.byTier).toBeDefined();

    // Verify tasks shape
    const tasks = data.tasks as Record<string, unknown>;
    expect(typeof tasks.total).toBe('number');
    expect(typeof tasks.pending).toBe('number');
    expect(typeof tasks.completed).toBe('number');
    expect(typeof tasks.failed).toBe('number');
  });
});

// ============================================================================
// Devices List Endpoint
// ============================================================================

describe('GET /api/compute/devices', () => {
  beforeEach(cleanTables);

  it('returns expected shape with devices array, count, byTier', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/devices');
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;

    expect(Array.isArray(data.devices)).toBe(true);
    expect(typeof data.count).toBe('number');
    expect(typeof data.online).toBe('number');
    expect(typeof data.totalCompute).toBe('number');
    expect(data.byTier).toBeDefined();

    const byTier = data.byTier as Record<string, unknown>;
    expect(typeof byTier.power).toBe('number');
    expect(typeof byTier.standard).toBe('number');
    expect(typeof byTier.crowd).toBe('number');
  });

  it('lists registered devices', async () => {
    // Register a device
    await SELF.fetch('https://test.local/api/compute/devices', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'Test Device',
        platform: 'cuda',
        capabilities: { compute: 82, memory: 24, models: [] },
      }),
    });

    const res = await SELF.fetch('https://test.local/api/compute/devices');
    const data = await res.json() as Record<string, unknown>;
    expect(data.count).toBe(1);

    const devices = data.devices as Record<string, unknown>[];
    expect(devices[0].name).toBe('Test Device');
    expect(devices[0].tier).toBe('power');
  });
});

// ============================================================================
// Device Registration Validation
// ============================================================================

describe('POST /api/compute/devices validation', () => {
  beforeEach(cleanTables);

  it('validates required fields: name, platform, capabilities', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/devices', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(400);
    const data = await res.json() as Record<string, unknown>;
    expect(data.error).toContain('Missing required fields');
  });
});

// ============================================================================
// Task Creation Validation
// ============================================================================

describe('POST /api/compute/tasks validation', () => {
  beforeEach(cleanTables);

  it('validates required fields: type, input', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: {} }),
    });
    expect(res.status).toBe(400);
    const data = await res.json() as Record<string, unknown>;
    expect(data.error).toContain('Missing required fields');
  });

  it('creates task with default priority 5', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'embedding',
        input: { text: 'test' },
      }),
    });
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    const task = data.task as Record<string, unknown>;
    expect(task.id).toBeDefined();

    // Check in DB that priority defaulted to 5
    const dbTask = await env.DB.prepare(
      'SELECT priority FROM compute_tasks WHERE id = ?'
    ).bind(task.id).first();
    expect(dbTask!.priority).toBe(5);
  });
});

// ============================================================================
// Task Status Endpoint
// ============================================================================

describe('GET /api/compute/tasks/:id', () => {
  beforeEach(cleanTables);

  it('returns task details for existing task', async () => {
    // Create a task
    const createRes = await SELF.fetch('https://test.local/api/compute/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'inference',
        input: { prompt: 'test' },
        config: { maxTokens: 100 },
      }),
    });
    const createData = await createRes.json() as Record<string, unknown>;
    const task = createData.task as Record<string, unknown>;

    const res = await SELF.fetch(`https://test.local/api/compute/tasks/${task.id}`);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.id).toBe(task.id);
    expect(data.type).toBe('inference');
    expect(data.config).toBeDefined();
    expect(data.createdAt).toBeDefined();
  });

  it('returns 404 for non-existent task', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/tasks/ctask_doesnotexist');
    expect(res.status).toBe(404);
  });
});

// ============================================================================
// Cleanup Endpoint Shape
// ============================================================================

describe('POST /api/compute/cleanup', () => {
  beforeEach(cleanTables);

  it('returns expected shape with devicesMarkedOffline and tasksReset', async () => {
    const res = await SELF.fetch('https://test.local/api/compute/cleanup', {
      method: 'POST',
    });
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.success).toBe(true);
    expect(typeof data.devicesMarkedOffline).toBe('number');
    expect(typeof data.tasksReset).toBe('number');
    expect(data.timestamp).toBeDefined();
  });
});

// ============================================================================
// Root Health Check
// ============================================================================

describe('GET /health (root)', () => {
  it('returns service-level health check', async () => {
    const res = await SELF.fetch('https://test.local/health');
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data.status).toBe('ok');
    expect(data.service).toBe('labfork-agents');
  });
});
