/**
 * Cron — the nudge loop.
 *
 * Runs every 15 minutes. Four steps:
 * 1. Observe  — What's the current state?
 * 2. Decide   — What 0-3 tasks should exist that don't?
 * 3. Nudge    — Are any assigned tasks stuck?
 * 4. Cleanup  — Mark dead workers offline. Reset orphaned tasks.
 */

import { observeTasks, observeWorkers, getLatestObservations } from './observers';
import { evaluate } from './rules';
import { generateId } from './utils';

interface CronResult {
  observed: { tasks: unknown; workers: unknown; external: number };
  tasksCreated: number;
  tasksNudged: number;
  parentsResolved: number;
  workersOfflined: number;
  tasksReset: number;
  staleDeleted: number;
  tasksPruned: number;
}

export async function runCronCycle(db: D1Database, options?: { force?: boolean }): Promise<CronResult> {
  const ts = new Date().toISOString();
  let tasksCreated = 0;
  let tasksNudged = 0;

  // Idempotency guard: skip if a cron ran within the last 5 minutes (Workers can double-fire)
  // Can be bypassed with force=true for manual admin triggers.
  if (!options?.force) {
    const fiveMinAgoGuard = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const recentCron = await db.prepare(
      `SELECT id FROM work_log WHERE event = 'cron_complete' AND created_at > ? LIMIT 1`
    ).bind(fiveMinAgoGuard).first();
    if (recentCron) {
      console.log('[Cron] Skipping — already ran within last 5 minutes');
      return { observed: { tasks: {} as unknown, workers: {} as unknown, external: 0 }, tasksCreated: 0, tasksNudged: 0, parentsResolved: 0, workersOfflined: 0, tasksReset: 0, staleDeleted: 0, tasksPruned: 0 };
    }
  }

  // =========================================================================
  // Step 1: Observe
  // =========================================================================
  const [taskObs, workerObs, externalObs] = await Promise.all([
    observeTasks(db),
    observeWorkers(db),
    getLatestObservations(db),
  ]);

  console.log(`[Cron] Observed: ${taskObs.pending} pending, ${taskObs.assigned} assigned, ${workerObs.online} workers online, ${externalObs.length} external observations`);

  // =========================================================================
  // Step 2: Decide — run heuristic rules to create tasks
  // =========================================================================
  // Check completed goals so the rules engine can skip them
  const completedGoalsRows = await db.prepare(`
    SELECT DISTINCT action FROM tasks WHERE status = 'completed' AND action LIKE 'goal-%'
  `).all<{ action: string }>();
  const completedGoals = new Set((completedGoalsRows.results || []).map((r) => r.action));

  const suggestions = evaluate({
    tasks: taskObs,
    workers: workerObs,
    external: externalObs,
    suggestedThisCycle: 0,
    completedGoals,
  });

  if (suggestions.length > 0) {
    // Check for duplicate actions already pending/assigned
    const existingActions = await db.prepare(`
      SELECT action FROM tasks WHERE status IN ('pending', 'assigned')
    `).all<{ action: string }>();
    const activeActions = new Set((existingActions.results || []).map((r) => r.action));

    // Cooldown for check-health: skip if one was completed in the last 2 hours
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString();
    const recentHealthCheck = await db.prepare(`
      SELECT id FROM tasks WHERE action = 'check-health' AND status = 'completed' AND completed_at > ? LIMIT 1
    `).bind(twoHoursAgo).first();
    if (recentHealthCheck) {
      activeActions.add('check-health');
    }

    // Cooldown for investigate-failures: skip if one was completed in the last 4 hours
    const fourHoursAgo = new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString();
    const recentInvestigation = await db.prepare(`
      SELECT id FROM tasks WHERE action = 'investigate-failures' AND status IN ('completed', 'failed') AND completed_at > ? LIMIT 1
    `).bind(fourHoursAgo).first();
    if (recentInvestigation) {
      activeActions.add('investigate-failures');
    }

    for (const task of suggestions) {
      if (activeActions.has(task.action)) {
        console.log(`[Cron] Skipping duplicate task: ${task.action}`);
        continue;
      }

      const taskId = generateId('t');
      await db.prepare(`
        INSERT INTO tasks (id, action, description, context, constraints, status, priority, required_capability, source, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, 'cron', ?)
      `).bind(
        taskId,
        task.action,
        task.description,
        task.context ? JSON.stringify(task.context) : null,
        task.constraints ? JSON.stringify(task.constraints) : null,
        task.priority,
        task.required_capability || null,
        ts
      ).run();

      // Log creation
      await db.prepare(`
        INSERT INTO work_log (id, event, detail, created_at)
        VALUES (?, 'created', ?, ?)
      `).bind(
        generateId('log'),
        JSON.stringify({ taskId, action: task.action, source: 'cron', reason: 'rule-based' }),
        ts
      ).run();

      tasksCreated++;
      activeActions.add(task.action);
      console.log(`[Cron] Created task: ${task.action} (priority ${task.priority})`);
    }
  }

  // =========================================================================
  // Step 2.5: Parent task resolution — complete parents when children finish
  // =========================================================================
  let parentsResolved = 0;

  // Find parent tasks that are pending/assigned and have children
  const parentRows = await db.prepare(`
    SELECT DISTINCT t.id, t.status
    FROM tasks t
    WHERE t.id IN (SELECT DISTINCT parent_task_id FROM tasks WHERE parent_task_id IS NOT NULL)
      AND t.status IN ('pending', 'assigned')
  `).all<{ id: string; status: string }>();

  for (const parent of (parentRows.results || [])) {
    const children = await db.prepare(`
      SELECT status FROM tasks WHERE parent_task_id = ?
    `).bind(parent.id).all<{ status: string }>();

    const childList = children.results || [];
    if (childList.length === 0) continue;

    const allDone = childList.every(ch => ch.status === 'completed' || ch.status === 'failed');
    if (!allDone) continue;

    const allCompleted = childList.every(ch => ch.status === 'completed');
    const anyFailed = childList.some(ch => ch.status === 'failed');

    if (allCompleted) {
      // All children completed successfully — mark parent completed
      await db.prepare(`
        UPDATE tasks SET status = 'completed',
          result = ?,
          completed_at = ?,
          assigned_worker_id = NULL
        WHERE id = ?
      `).bind(
        JSON.stringify({
          summary: `All ${childList.length} subtasks completed successfully`,
          decomposed: true,
          childrenCompleted: childList.length,
        }),
        ts,
        parent.id
      ).run();

      // Free worker if assigned
      await db.prepare(`
        UPDATE workers SET current_task_id = NULL, status = 'online'
        WHERE current_task_id = ?
      `).bind(parent.id).run();

      parentsResolved++;
      console.log(`[Cron] Parent ${parent.id} completed (all ${childList.length} children done)`);
    } else if (anyFailed) {
      const failedCount = childList.filter(ch => ch.status === 'failed').length;
      const completedCount = childList.filter(ch => ch.status === 'completed').length;

      await db.prepare(`
        UPDATE tasks SET status = 'failed',
          error = ?,
          result = ?,
          completed_at = ?,
          assigned_worker_id = NULL
        WHERE id = ?
      `).bind(
        `${failedCount}/${childList.length} subtasks failed`,
        JSON.stringify({
          decomposed: true,
          childrenCompleted: completedCount,
          childrenFailed: failedCount,
          childrenTotal: childList.length,
        }),
        ts,
        parent.id
      ).run();

      await db.prepare(`
        UPDATE workers SET current_task_id = NULL, status = 'online'
        WHERE current_task_id = ?
      `).bind(parent.id).run();

      parentsResolved++;
      console.log(`[Cron] Parent ${parent.id} failed (${failedCount}/${childList.length} children failed)`);
    }

    if (allDone) {
      await db.prepare(`
        INSERT INTO work_log (id, task_id, event, detail, created_at) VALUES (?, ?, 'parent_resolved', ?, ?)
      `).bind(
        generateId('log'), parent.id,
        JSON.stringify({ childrenTotal: childList.length, allCompleted, anyFailed }),
        ts
      ).run();
    }
  }

  // =========================================================================
  // Step 3: Nudge — handle stuck tasks
  // =========================================================================
  for (const stuck of taskObs.stuck) {
    // Check if the assigned worker is still alive
    const task = await db.prepare(`
      SELECT assigned_worker_id, attempts FROM tasks WHERE id = ?
    `).bind(stuck.id).first<{ assigned_worker_id: string | null; attempts: number }>();

    if (!task?.assigned_worker_id) continue;

    const worker = await db.prepare(`
      SELECT status, last_heartbeat FROM workers WHERE id = ?
    `).bind(task.assigned_worker_id).first<{ status: string; last_heartbeat: string | null }>();

    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const workerDead = !worker || worker.status === 'offline' ||
      (worker.last_heartbeat && worker.last_heartbeat < fiveMinAgo);

    if (workerDead) {
      // Worker is dead — reset task to pending
      if (task.attempts >= 3) {
        // Too many retries — mark as failed
        await db.prepare(`
          UPDATE tasks SET status = 'failed', error = 'Exceeded max attempts (3)', completed_at = ? WHERE id = ?
        `).bind(ts, stuck.id).run();
        console.log(`[Cron] Task ${stuck.id} failed after ${task.attempts} attempts`);
      } else {
        await db.prepare(`
          UPDATE tasks SET status = 'pending', assigned_worker_id = NULL, assigned_at = NULL WHERE id = ?
        `).bind(stuck.id).run();
        console.log(`[Cron] Reset stuck task ${stuck.id} (worker dead, attempt ${task.attempts})`);
      }

      // Clear worker's current task ref
      await db.prepare(`
        UPDATE workers SET current_task_id = NULL, status = 'offline' WHERE id = ?
      `).bind(task.assigned_worker_id).run();

      tasksNudged++;
    } else {
      // Worker is alive but slow — just log it
      console.log(`[Cron] Task ${stuck.id} running long: ${stuck.assignedMinutesAgo}min (worker alive)`);
    }

    // Log the nudge
    await db.prepare(`
      INSERT INTO work_log (id, task_id, worker_id, event, detail, created_at) VALUES (?, ?, ?, 'nudged', ?, ?)
    `).bind(
      generateId('log'), stuck.id, task.assigned_worker_id,
      JSON.stringify({ assignedMinutesAgo: stuck.assignedMinutesAgo, workerDead, attempts: task.attempts }),
      ts
    ).run();
  }

  // =========================================================================
  // Step 4: Cleanup
  // =========================================================================
  const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
  const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();

  // Mark stale workers offline
  const offlineResult = await db.prepare(`
    UPDATE workers SET status = 'offline', updated_at = ?
    WHERE status IN ('online', 'busy') AND last_heartbeat < ?
  `).bind(ts, fiveMinAgo).run();
  const workersOfflined = offlineResult.meta?.changes || 0;

  // Reset tasks assigned to offline workers (if assigned > 1 hour ago)
  const resetResult = await db.prepare(`
    UPDATE tasks SET status = 'pending', assigned_worker_id = NULL, assigned_at = NULL
    WHERE status = 'assigned' AND assigned_at < ?
  `).bind(oneHourAgo).run();
  const tasksReset = resetResult.meta?.changes || 0;

  // Clear current_task_id for offline workers
  await db.prepare(`
    UPDATE workers SET current_task_id = NULL WHERE status = 'offline' AND current_task_id IS NOT NULL
  `).run();

  // Delete workers offline for 24h+ with no completed tasks
  const deleteResult = await db.prepare(`
    DELETE FROM workers WHERE status = 'offline' AND last_heartbeat < ? AND tasks_completed = 0
  `).bind(oneDayAgo).run();
  const staleDeleted = deleteResult.meta?.changes || 0;

  // Prune completed/failed tasks older than 24h (keeps the board clean automatically)
  // Goal tasks are kept for 7 days so we can track objective progress
  const tasksPruned = await db.prepare(`
    DELETE FROM tasks WHERE status IN ('completed', 'failed')
      AND completed_at < ?
      AND action NOT LIKE 'goal-%'
  `).bind(oneDayAgo).run();
  const tasksAutoDeleted = tasksPruned.meta?.changes || 0;

  // Prune old goal tasks (keep 7 days)
  const goalsPruned = await db.prepare(`
    DELETE FROM tasks WHERE status IN ('completed', 'failed')
      AND completed_at < ?
      AND action LIKE 'goal-%'
  `).bind(sevenDaysAgo).run();
  const goalsAutoDeleted = goalsPruned.meta?.changes || 0;

  // Prune old observations (keep 7 days — goals and other long-lived observations need time)
  await db.prepare('DELETE FROM observations WHERE created_at < ?').bind(sevenDaysAgo).run();

  // Prune old work_log entries (keep 7 days)
  await db.prepare('DELETE FROM work_log WHERE created_at < ?').bind(sevenDaysAgo).run();

  if (workersOfflined > 0 || tasksReset > 0 || staleDeleted > 0 || tasksAutoDeleted > 0) {
    console.log(`[Cron] Cleanup: ${workersOfflined} workers offlined, ${tasksReset} tasks reset, ${staleDeleted} stale deleted, ${tasksAutoDeleted + goalsAutoDeleted} tasks pruned`);
  }

  // Log cron completion
  await db.prepare(`
    INSERT INTO work_log (id, event, detail, created_at)
    VALUES (?, 'cron_complete', ?, ?)
  `).bind(
    generateId('log'),
    JSON.stringify({ tasksCreated, tasksNudged, parentsResolved, workersOfflined, tasksReset, staleDeleted, tasksPruned: tasksAutoDeleted + goalsAutoDeleted }),
    ts
  ).run();

  return {
    observed: { tasks: taskObs, workers: workerObs, external: externalObs.length },
    tasksCreated,
    tasksNudged,
    parentsResolved,
    workersOfflined,
    tasksReset,
    staleDeleted,
    tasksPruned: tasksAutoDeleted + goalsAutoDeleted,
  };
}
