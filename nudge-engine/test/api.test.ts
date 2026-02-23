/**
 * Nudge Engine API Tests
 *
 * Tests the full register → poll → report cycle.
 */

import { env, createExecutionContext, waitOnExecutionContext } from 'cloudflare:test';
import { describe, it, expect, beforeEach } from 'vitest';
import app from '../src/index';
import { evaluate } from '../src/rules';
import { runCronCycle } from '../src/cron';

// Helper to make requests
async function request(method: string, path: string, body?: unknown, headers?: Record<string, string>) {
  const init: RequestInit = { method, headers: { 'Content-Type': 'application/json', ...headers } };
  if (body) init.body = JSON.stringify(body);
  const ctx = createExecutionContext();
  const res = await app.fetch(new Request(`http://localhost${path}`, init), env, ctx);
  await waitOnExecutionContext(ctx);
  return res;
}

async function json(res: Response) {
  return res.json() as Promise<Record<string, unknown>>;
}

const adminAuth = { Authorization: 'Bearer test-admin-key-for-vitest' };

describe('Nudge Engine API', () => {
  // Run schema before tests
  beforeEach(async () => {
    // Reset tables
    await env.DB.exec('DELETE FROM work_log');
    await env.DB.exec('DELETE FROM observations');
    await env.DB.exec('DELETE FROM tasks');
    await env.DB.exec('DELETE FROM workers');
  });

  describe('GET /', () => {
    it('returns health check', async () => {
      const res = await request('GET', '/');
      expect(res.status).toBe(200);
      const data = await json(res);
      expect(data.name).toBe('nudge-engine');
      expect(data.endpoints).toBeTruthy();
    });
  });

  describe('POST /register', () => {
    it('registers a new worker', async () => {
      const res = await request('POST', '/register', {
        name: 'test-worker',
        type: 'claude-code',
        capabilities: ['code', 'test'],
      });
      expect(res.status).toBe(200);
      const data = await json(res);
      expect(data.registered).toBe(true);
      expect(data.id).toBeTruthy();
      expect(data.token).toBeTruthy();
      expect((data.token as string).length).toBe(64); // 32 bytes hex
    });

    it('rejects invalid type', async () => {
      const res = await request('POST', '/register', {
        name: 'test',
        type: 'invalid',
      });
      expect(res.status).toBe(400);
    });

    it('rejects missing fields', async () => {
      const res = await request('POST', '/register', { name: 'test' });
      expect(res.status).toBe(400);
    });
  });

  describe('Full lifecycle: register → poll → report', () => {
    it('completes the full task cycle', async () => {
      // 1. Register a worker
      const regRes = await request('POST', '/register', {
        name: 'lifecycle-worker',
        type: 'claude-code',
      });
      const reg = await json(regRes);
      const token = reg.token as string;
      const auth = { Authorization: `Bearer ${token}` };

      // 2. Create a task
      const taskRes = await request('POST', '/tasks', {
        action: 'fix-test',
        description: 'Fix the failing test in auth module',
        priority: 8,
      }, adminAuth);
      expect(taskRes.status).toBe(200);
      const taskData = await json(taskRes);
      const taskId = (taskData.task as Record<string, unknown>).id as string;

      // 3. Poll for work — should get the task
      const pollRes = await request('POST', '/poll', {}, auth);
      expect(pollRes.status).toBe(200);
      const pollData = await json(pollRes);
      expect(pollData.task).toBeTruthy();
      const task = pollData.task as Record<string, unknown>;
      expect(task.id).toBe(taskId);
      expect(task.action).toBe('fix-test');

      // 4. Poll again — should return the same task (already assigned)
      const poll2Res = await request('POST', '/poll', {}, auth);
      const poll2Data = await json(poll2Res);
      expect((poll2Data.task as Record<string, unknown>).id).toBe(taskId);

      // 5. Report success
      const reportRes = await request('POST', '/report', {
        taskId,
        success: true,
        result: {
          summary: 'Fixed timeout assertion in test_login_timeout',
          filesChanged: ['tests/test_auth.py'],
          testsPass: true,
        },
      }, auth);
      expect(reportRes.status).toBe(200);
      const reportData = await json(reportRes);
      expect(reportData.status).toBe('completed');

      // 6. Poll again — no more tasks
      const poll3Res = await request('POST', '/poll', {}, auth);
      const poll3Data = await json(poll3Res);
      expect(poll3Data.task).toBeNull();

      // 7. Verify task state
      const getRes = await request('GET', `/tasks/${taskId}`);
      const getTask = await json(getRes);
      expect(getTask.status).toBe('completed');
      expect(getTask.result).toBeTruthy();
    });

    it('handles task failure', async () => {
      // Register
      const reg = await json(await request('POST', '/register', { name: 'fail-worker', type: 'ollama' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      // Create task
      const taskRes = await json(await request('POST', '/tasks', {
        action: 'impossible-task',
        description: 'This will fail',
      }, adminAuth));
      const taskId = (taskRes.task as Record<string, unknown>).id as string;

      // Poll
      await request('POST', '/poll', {}, auth);

      // Report failure
      const reportRes = await request('POST', '/report', {
        taskId,
        success: false,
        error: 'Could not find the file',
      }, auth);
      expect(reportRes.status).toBe(200);
      const data = await json(reportRes);
      expect(data.status).toBe('failed');

      // Verify
      const getTask = await json(await request('GET', `/tasks/${taskId}`));
      expect(getTask.status).toBe('failed');
      expect(getTask.error).toBe('Could not find the file');
    });
  });

  describe('POST /poll — auth', () => {
    it('rejects unauthenticated requests', async () => {
      const res = await request('POST', '/poll');
      expect(res.status).toBe(401);
    });

    it('rejects invalid token', async () => {
      const res = await request('POST', '/poll', {}, { Authorization: 'Bearer invalid' });
      expect(res.status).toBe(401);
    });
  });

  describe('POST /report — guards', () => {
    it('rejects double completion', async () => {
      const reg = await json(await request('POST', '/register', { name: 'dupe-worker', type: 'custom' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      await request('POST', '/tasks', { action: 'dupe-test', description: 'Test' }, adminAuth);
      const poll = await json(await request('POST', '/poll', {}, auth));
      const taskId = (poll.task as Record<string, unknown>).id as string;

      // First report
      await request('POST', '/report', { taskId, success: true, result: { summary: 'Done' } }, auth);

      // Second report — should fail
      const res = await request('POST', '/report', { taskId, success: true, result: { summary: 'Again' } }, auth);
      expect(res.status).toBe(409);
    });
  });

  describe('POST /observe', () => {
    it('accepts observations from workers', async () => {
      const reg = await json(await request('POST', '/register', { name: 'observer', type: 'custom' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      const res = await request('POST', '/observe', {
        type: 'git',
        data: { failingCI: true, recentCommits: [] },
      }, auth);
      expect(res.status).toBe(200);
      const data = await json(res);
      expect(data.observationId).toBeTruthy();
    });
  });

  describe('GET /stats', () => {
    it('returns network statistics', async () => {
      const res = await request('GET', '/stats');
      expect(res.status).toBe(200);
      const data = await json(res);
      expect(data.workers).toBeTruthy();
      expect(data.tasks).toBeTruthy();
    });
  });

  describe('Task priority ordering', () => {
    it('assigns highest priority task first', async () => {
      const reg = await json(await request('POST', '/register', { name: 'prio-worker', type: 'claude-code' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      // Create low priority task first
      await request('POST', '/tasks', { action: 'low-prio', description: 'Low', priority: 2 }, adminAuth);
      // Create high priority task second
      await request('POST', '/tasks', { action: 'high-prio', description: 'High', priority: 9 }, adminAuth);

      // Poll — should get high priority
      const poll = await json(await request('POST', '/poll', {}, auth));
      expect((poll.task as Record<string, unknown>).action).toBe('high-prio');
    });
  });

  // ===========================================================================
  // Goals rule tests
  // ===========================================================================

  describe('pursueGoals rule (unit)', () => {
    it('creates a task from the first objective', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Fix the login bug', 'Add dark mode'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      // Only the goals task — emptyQueue skips because goals already suggested work
      expect(suggestions.length).toBe(1);
      expect(suggestions[0].action).toBe('goal-fix-the-login-bug');
      expect(suggestions[0].description).toContain('Fix the login bug');
      expect(suggestions[0].priority).toBe(5);
    });

    it('falls through to emptyQueue when no goals exist', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [],
        suggestedThisCycle: 0,
      });

      // No goals, no other observations — emptyQueue fires
      expect(suggestions.length).toBe(1);
      expect(suggestions[0].action).toBe('check-health');
    });

    it('backs off when queue already has 2+ tasks', () => {
      const suggestions = evaluate({
        tasks: { pending: 2, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 2 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Build something'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions.length).toBe(0);
    });

    it('backs off when no idle workers', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 1, completed24h: 0, failed24h: 0, stuck: [], total: 1 },
        workers: { online: 1, busy: 1, offline: 0, idle: 0, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Build something'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions.length).toBe(0);
    });

    it('slugifies long objectives and truncates to 40 chars', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Implement a really long objective name that should be truncated for the action slug'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions[0].action.startsWith('goal-')).toBe(true);
      // 'goal-' is 5 chars + max 40 chars of slug
      expect(suggestions[0].action.length).toBeLessThanOrEqual(45);
    });

    it('urgent rules take priority over goals', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [
          { type: 'goals', data: { objectives: ['Add dark mode'] }, observedAt: new Date().toISOString() },
          { type: 'git', data: { failingCI: true }, observedAt: new Date().toISOString() },
        ],
        suggestedThisCycle: 0,
      });

      // CI fix should come first (priority 9 vs goals priority 5)
      expect(suggestions[0].action).toBe('fix-ci');
    });
  });

  describe('Goals integration (cron cycle)', () => {
    it('cron creates goal task when goals observation exists', async () => {
      // Register a worker so there's an idle worker
      const reg = await json(await request('POST', '/register', { name: 'goal-worker', type: 'ollama' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      // Heartbeat to keep worker online
      await request('POST', '/poll', {}, auth);

      // Post a goals observation
      await request('POST', '/observe', {
        type: 'goals',
        data: { objectives: ['Set up CI pipeline', 'Write integration tests'] },
      }, auth);

      // Run the cron cycle
      const result = await runCronCycle(env.DB);
      expect(result.tasksCreated).toBeGreaterThanOrEqual(1);

      // Verify the task was created
      const tasksRes = await request('GET', '/tasks');
      const tasksData = await json(tasksRes);
      const tasks = tasksData.tasks as Array<Record<string, unknown>>;
      const goalTask = tasks.find((t) => (t.action as string).startsWith('goal-'));

      expect(goalTask).toBeTruthy();
      expect(goalTask!.action).toBe('goal-set-up-ci-pipeline');
      expect(goalTask!.status).toBe('pending');
      expect(goalTask!.description).toContain('Set up CI pipeline');
    });

    it('cron does not duplicate goal tasks', async () => {
      const reg = await json(await request('POST', '/register', { name: 'dedup-worker', type: 'ollama' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth);

      await request('POST', '/observe', {
        type: 'goals',
        data: { objectives: ['Add logging'] },
      }, auth);

      // Run cron twice
      await runCronCycle(env.DB);
      await runCronCycle(env.DB);

      // Should only have one goal task (dedup by action name)
      const tasksData = await json(await request('GET', '/tasks'));
      const tasks = tasksData.tasks as Array<Record<string, unknown>>;
      const goalTasks = tasks.filter((t) => (t.action as string).startsWith('goal-'));

      expect(goalTasks.length).toBe(1);
    });

    it('full goals lifecycle: observe → cron → poll → report', async () => {
      // 1. Register worker
      const reg = await json(await request('POST', '/register', { name: 'lifecycle-goal', type: 'ollama' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth); // heartbeat

      // 2. Post goals
      await request('POST', '/observe', {
        type: 'goals',
        data: { objectives: ['Implement feature X', 'Implement feature Y'] },
      }, auth);

      // 3. Cron creates goal task
      await runCronCycle(env.DB);

      // 4. Worker polls and gets the goal task
      const poll = await json(await request('POST', '/poll', {}, auth));
      expect(poll.task).toBeTruthy();
      const task = poll.task as Record<string, unknown>;
      expect(task.action).toBe('goal-implement-feature-x');

      // 5. Worker completes the task
      await request('POST', '/report', {
        taskId: task.id,
        success: true,
        result: { summary: 'Feature X implemented successfully' },
      }, auth);

      // 6. Verify task completed
      const getTask = await json(await request('GET', `/tasks/${task.id}`));
      expect(getTask.status).toBe('completed');
    });
  });

  // ===========================================================================
  // Capability matching
  // ===========================================================================

  describe('Capability matching', () => {
    it('worker with matching capability gets the task', async () => {
      const reg = await json(await request('POST', '/register', {
        name: 'code-worker',
        type: 'codex',
        capabilities: ['code', 'monitor'],
      }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      await request('POST', '/tasks', {
        action: 'goal-implement-foo',
        description: 'Build foo',
        required_capability: 'code',
      }, adminAuth);

      const poll = await json(await request('POST', '/poll', {}, auth));
      expect(poll.task).toBeTruthy();
      expect((poll.task as Record<string, unknown>).action).toBe('goal-implement-foo');
    });

    it('worker without matching capability skips the task', async () => {
      const reg = await json(await request('POST', '/register', {
        name: 'monitor-only',
        type: 'ollama',
        capabilities: ['monitor', 'advisory'],
      }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      await request('POST', '/tasks', {
        action: 'goal-build-bar',
        description: 'Build bar',
        required_capability: 'code',
      }, adminAuth);

      const poll = await json(await request('POST', '/poll', {}, auth));
      expect(poll.task).toBeNull();
    });

    it('worker with empty capabilities gets any task (backwards compat)', async () => {
      const reg = await json(await request('POST', '/register', {
        name: 'legacy-worker',
        type: 'custom',
      }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      await request('POST', '/tasks', {
        action: 'goal-fix-thing',
        description: 'Fix it',
        required_capability: 'code',
      }, adminAuth);

      const poll = await json(await request('POST', '/poll', {}, auth));
      expect(poll.task).toBeTruthy();
    });

    it('tasks without required_capability go to any worker', async () => {
      const reg = await json(await request('POST', '/register', {
        name: 'limited-worker',
        type: 'ollama',
        capabilities: ['advisory'],
      }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      await request('POST', '/tasks', {
        action: 'check-health',
        description: 'Health check',
      }, adminAuth);

      const poll = await json(await request('POST', '/poll', {}, auth));
      expect(poll.task).toBeTruthy();
    });
  });

  // ===========================================================================
  // Worker task creation
  // ===========================================================================

  describe('POST /tasks — worker auth', () => {
    it('workers can create tasks (not just admins)', async () => {
      const reg = await json(await request('POST', '/register', {
        name: 'subtask-creator',
        type: 'codex',
      }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      const res = await request('POST', '/tasks', {
        action: 'goal-implement-subtask',
        description: 'A subtask created by a worker',
        priority: 6,
        required_capability: 'code',
      }, auth);

      expect(res.status).toBe(200);
      const data = await json(res);
      expect(data.success).toBe(true);

      // Verify the task exists and has source=worker
      const taskId = (data.task as Record<string, unknown>).id as string;
      const taskRes = await json(await request('GET', `/tasks/${taskId}`));
      expect(taskRes.source).toBe('worker');
      expect(taskRes.required_capability).toBe('code');
    });

    it('rejects unauthenticated task creation', async () => {
      const res = await request('POST', '/tasks', {
        action: 'sneaky-task',
        description: 'No auth',
      });
      expect(res.status).toBe(401);
    });

    it('validates required_capability values', async () => {
      const res = await request('POST', '/tasks', {
        action: 'bad-cap',
        description: 'Invalid capability',
        required_capability: 'superpower',
      }, adminAuth);
      expect(res.status).toBe(400);
    });
  });

  // ===========================================================================
  // Rules engine — required_capability on generated tasks
  // ===========================================================================

  describe('Rules engine capability assignment', () => {
    it('sets code capability for implementation goals', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Implement user auth'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions[0].action).toBe('goal-implement-user-auth');
      expect(suggestions[0].required_capability).toBe('code');
    });

    it('sets monitor capability for review goals', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Review API security'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions[0].action).toBe('goal-review-api-security');
      expect(suggestions[0].required_capability).toBe('monitor');
    });

    it('defaults to code capability for all non-monitor goals', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'goals', data: { objectives: ['Optimize database queries'] }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions[0].action).toBe('goal-optimize-database-queries');
      expect(suggestions[0].required_capability).toBe('code');
    });

    it('sets code capability for fix-ci rule', () => {
      const suggestions = evaluate({
        tasks: { pending: 0, assigned: 0, completed24h: 0, failed24h: 0, stuck: [], total: 0 },
        workers: { online: 1, busy: 0, offline: 0, idle: 1, stale: [] },
        external: [{ type: 'git', data: { failingCI: true }, observedAt: new Date().toISOString() }],
        suggestedThisCycle: 0,
      });

      expect(suggestions[0].action).toBe('fix-ci');
      expect(suggestions[0].required_capability).toBe('code');
    });

    it('cron stores required_capability in DB', async () => {
      const reg = await json(await request('POST', '/register', { name: 'cap-test', type: 'ollama' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth);

      await request('POST', '/observe', {
        type: 'goals',
        data: { objectives: ['Build a dashboard component'] },
      }, auth);

      await runCronCycle(env.DB);

      const tasksRes = await json(await request('GET', '/tasks'));
      const tasks = tasksRes.tasks as Array<Record<string, unknown>>;
      const goalTask = tasks.find((t) => (t.action as string).startsWith('goal-build'));

      expect(goalTask).toBeTruthy();
      expect(goalTask!.required_capability).toBe('code');
    });
  });

  // ===========================================================================
  // Task Decomposition — parent/child relationships
  // ===========================================================================

  describe('Task decomposition (parent/child)', () => {
    it('creates a subtask with parent_task_id', async () => {
      // Create parent task
      const parentRes = await request('POST', '/tasks', {
        action: 'goal-build-auth-system',
        description: 'Build a complete authentication system',
        priority: 5,
      }, adminAuth);
      const parentData = await json(parentRes);
      const parentId = (parentData.task as Record<string, unknown>).id as string;

      // Create subtask with parent_task_id
      const childRes = await request('POST', '/tasks', {
        action: 'goal-implement-password-hash',
        description: 'Implement password hashing function',
        priority: 6,
        required_capability: 'code',
        parent_task_id: parentId,
      }, adminAuth);
      expect(childRes.status).toBe(200);
      const childData = await json(childRes);
      expect(childData.success).toBe(true);
      expect((childData.task as Record<string, unknown>).parent_task_id).toBe(parentId);

      // Verify child appears in parent detail
      const getParent = await json(await request('GET', `/tasks/${parentId}`));
      expect(getParent.children).toBeTruthy();
      expect((getParent.children as Array<Record<string, unknown>>).length).toBe(1);
      expect(getParent.childrenSummary).toBeTruthy();
      expect((getParent.childrenSummary as Record<string, unknown>).total).toBe(1);
      expect((getParent.childrenSummary as Record<string, unknown>).pending).toBe(1);
    });

    it('rejects subtask with invalid parent_task_id', async () => {
      const res = await request('POST', '/tasks', {
        action: 'goal-orphan-task',
        description: 'Task with non-existent parent',
        parent_task_id: 'nonexistent_id',
      }, adminAuth);
      expect(res.status).toBe(400);
      const data = await json(res);
      expect(data.error).toContain('parent_task_id not found');
    });

    it('GET /tasks/:id/children returns subtasks', async () => {
      // Create parent
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-build-api',
        description: 'Build REST API',
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      // Create 3 subtasks
      for (const action of ['goal-implement-get', 'goal-implement-post', 'goal-implement-delete']) {
        await request('POST', '/tasks', {
          action,
          description: `Implement ${action}`,
          parent_task_id: parentId,
          required_capability: 'code',
        }, adminAuth);
      }

      // Fetch children
      const res = await request('GET', `/tasks/${parentId}/children`);
      expect(res.status).toBe(200);
      const data = await json(res);
      expect(data.parentId).toBe(parentId);
      expect((data.children as unknown[]).length).toBe(3);
      expect((data.summary as Record<string, unknown>).total).toBe(3);
      expect((data.summary as Record<string, unknown>).pending).toBe(3);
    });

    it('GET /tasks filters by parent_task_id', async () => {
      // Create parent + child
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-filter-parent',
        description: 'Parent for filter test',
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      await request('POST', '/tasks', {
        action: 'goal-filter-child',
        description: 'Child for filter test',
        parent_task_id: parentId,
      }, adminAuth);

      // Also create an unrelated task
      await request('POST', '/tasks', {
        action: 'unrelated-task',
        description: 'Not a child',
      }, adminAuth);

      // Filter by parent
      const res = await json(await request('GET', `/tasks?parent_task_id=${parentId}`));
      const tasks = res.tasks as Array<Record<string, unknown>>;
      expect(tasks.length).toBe(1);
      expect(tasks[0].action).toBe('goal-filter-child');
    });

    it('poll skips parent tasks that have children', async () => {
      // Register worker
      const reg = await json(await request('POST', '/register', { name: 'decomp-worker', type: 'codex' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      // Create parent task
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-complex-parent',
        description: 'A complex goal that has been decomposed',
        priority: 9,
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      // Create child tasks
      await request('POST', '/tasks', {
        action: 'goal-simple-child-1',
        description: 'First subtask',
        parent_task_id: parentId,
        priority: 6,
      }, adminAuth);

      // Poll — should get the child task, NOT the parent (even though parent has higher priority)
      const poll = await json(await request('POST', '/poll', {}, auth));
      expect(poll.task).toBeTruthy();
      expect((poll.task as Record<string, unknown>).action).toBe('goal-simple-child-1');
    });

    it('cron completes parent when all children complete', async () => {
      // Register worker
      const reg = await json(await request('POST', '/register', { name: 'parent-complete-worker', type: 'codex' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth); // heartbeat

      // Create parent task
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-decomposed-parent',
        description: 'Parent that should auto-complete',
        priority: 5,
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      // Create 2 child tasks
      const child1Res = await json(await request('POST', '/tasks', {
        action: 'goal-child-a',
        description: 'Subtask A',
        parent_task_id: parentId,
      }, adminAuth));
      const child1Id = (child1Res.task as Record<string, unknown>).id as string;

      const child2Res = await json(await request('POST', '/tasks', {
        action: 'goal-child-b',
        description: 'Subtask B',
        parent_task_id: parentId,
      }, adminAuth));
      const child2Id = (child2Res.task as Record<string, unknown>).id as string;

      // Complete both children via direct DB update (simulating worker reports)
      const ts = new Date().toISOString();
      await env.DB.prepare('UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?').bind('completed', ts, child1Id).run();
      await env.DB.prepare('UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?').bind('completed', ts, child2Id).run();

      // Run cron — should resolve parent
      const result = await runCronCycle(env.DB);
      expect(result.parentsResolved).toBe(1);

      // Verify parent is completed
      const parentTask = await json(await request('GET', `/tasks/${parentId}`));
      expect(parentTask.status).toBe('completed');
      expect(parentTask.result).toBeTruthy();
      const parentResult = parentTask.result as Record<string, unknown>;
      expect(parentResult.decomposed).toBe(true);
      expect(parentResult.childrenCompleted).toBe(2);
    });

    it('cron fails parent when a child fails and none are pending', async () => {
      const reg = await json(await request('POST', '/register', { name: 'parent-fail-worker', type: 'codex' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth);

      // Create parent
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-failing-parent',
        description: 'Parent with a failing child',
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      // Create 2 children
      const child1Res = await json(await request('POST', '/tasks', {
        action: 'goal-succeed-child',
        description: 'This one succeeds',
        parent_task_id: parentId,
      }, adminAuth));
      const child1Id = (child1Res.task as Record<string, unknown>).id as string;

      const child2Res = await json(await request('POST', '/tasks', {
        action: 'goal-fail-child',
        description: 'This one fails',
        parent_task_id: parentId,
      }, adminAuth));
      const child2Id = (child2Res.task as Record<string, unknown>).id as string;

      // Complete one, fail the other
      const ts = new Date().toISOString();
      await env.DB.prepare('UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?').bind('completed', ts, child1Id).run();
      await env.DB.prepare('UPDATE tasks SET status = ?, error = ?, completed_at = ? WHERE id = ?').bind('failed', 'Test failure', ts, child2Id).run();

      // Run cron
      const result = await runCronCycle(env.DB);
      expect(result.parentsResolved).toBe(1);

      // Verify parent is failed
      const parentTask = await json(await request('GET', `/tasks/${parentId}`));
      expect(parentTask.status).toBe('failed');
      expect(parentTask.error).toContain('subtasks failed');
    });

    it('cron does NOT resolve parent when children are still pending', async () => {
      const reg = await json(await request('POST', '/register', { name: 'parent-wait-worker', type: 'codex' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth);

      // Create parent + 2 children, only complete one
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-waiting-parent',
        description: 'Parent waiting for children',
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      const child1Res = await json(await request('POST', '/tasks', {
        action: 'goal-done-child',
        description: 'Done',
        parent_task_id: parentId,
      }, adminAuth));
      const child1Id = (child1Res.task as Record<string, unknown>).id as string;

      await request('POST', '/tasks', {
        action: 'goal-pending-child',
        description: 'Still pending',
        parent_task_id: parentId,
      }, adminAuth);

      // Complete only one child
      await env.DB.prepare('UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?')
        .bind('completed', new Date().toISOString(), child1Id).run();

      // Run cron
      const result = await runCronCycle(env.DB);
      expect(result.parentsResolved).toBe(0);

      // Parent should still be pending
      const parentTask = await json(await request('GET', `/tasks/${parentId}`));
      expect(parentTask.status).toBe('pending');
    });

    it('worker creates subtasks with parent_task_id', async () => {
      // Register worker
      const reg = await json(await request('POST', '/register', { name: 'decomposer', type: 'codex' }));
      const auth = { Authorization: `Bearer ${reg.token}` };

      // Create parent task
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-big-feature',
        description: 'Build a big feature',
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      // Worker creates subtasks (simulating decompose mode)
      const sub1Res = await request('POST', '/tasks', {
        action: 'goal-implement-helper',
        description: 'Create helper function',
        parent_task_id: parentId,
        required_capability: 'code',
        priority: 6,
      }, auth);
      expect(sub1Res.status).toBe(200);
      const sub1Data = await json(sub1Res);
      expect(sub1Data.success).toBe(true);
      expect((sub1Data.task as Record<string, unknown>).parent_task_id).toBe(parentId);

      // Verify parent shows children
      const childrenRes = await json(await request('GET', `/tasks/${parentId}/children`));
      expect((childrenRes.children as unknown[]).length).toBe(1);
      expect((childrenRes.summary as Record<string, unknown>).total).toBe(1);
    });

    it('full decomposition lifecycle: parent → children → auto-complete', async () => {
      // 1. Register worker
      const reg = await json(await request('POST', '/register', { name: 'full-decomp', type: 'codex' }));
      const auth = { Authorization: `Bearer ${reg.token}` };
      await request('POST', '/poll', {}, auth); // heartbeat

      // 2. Create parent task (simulating what cron does for a goal)
      const parentRes = await json(await request('POST', '/tasks', {
        action: 'goal-build-calculator',
        description: 'Build a calculator with add, subtract, multiply functions',
        priority: 5,
      }, adminAuth));
      const parentId = (parentRes.task as Record<string, unknown>).id as string;

      // 3. Worker decomposes: creates 3 subtasks
      const subtasks = [
        { action: 'goal-implement-add', description: 'Create add(a, b) -> a + b. Test: add(2,3)==5, add(-1,1)==0' },
        { action: 'goal-implement-subtract', description: 'Create subtract(a, b) -> a - b. Test: subtract(5,3)==2' },
        { action: 'goal-implement-multiply', description: 'Create multiply(a, b) -> a * b. Test: multiply(3,4)==12' },
      ];

      const childIds: string[] = [];
      for (const sub of subtasks) {
        const res = await json(await request('POST', '/tasks', {
          ...sub,
          parent_task_id: parentId,
          required_capability: 'code',
          priority: 6,
        }, auth));
        childIds.push((res.task as Record<string, unknown>).id as string);
      }

      // 4. Verify parent has 3 children
      const parentDetail = await json(await request('GET', `/tasks/${parentId}`));
      expect((parentDetail.childrenSummary as Record<string, unknown>).total).toBe(3);

      // 5. Complete all children
      const ts = new Date().toISOString();
      for (const childId of childIds) {
        await env.DB.prepare('UPDATE tasks SET status = ?, result = ?, completed_at = ? WHERE id = ?')
          .bind('completed', JSON.stringify({ summary: 'Done' }), ts, childId).run();
      }

      // 6. Run cron — parent should auto-complete
      const cronResult = await runCronCycle(env.DB);
      expect(cronResult.parentsResolved).toBe(1);

      // 7. Verify parent completed
      const final = await json(await request('GET', `/tasks/${parentId}`));
      expect(final.status).toBe('completed');
      const result = final.result as Record<string, unknown>;
      expect(result.decomposed).toBe(true);
      expect(result.childrenCompleted).toBe(3);
    });
  });
});
