import { Hono } from 'hono';
import { cors } from 'hono/cors';
import api from './api/routes';

// Types for Cloudflare bindings
export interface Env {
  DB: D1Database;
  // AI binding removed - using distributed compute network instead
  MANAGER_WORKFLOW: Workflow;
  WORKER_WORKFLOW: Workflow;
  ENVIRONMENT: string;
}

// Create Hono app
const app = new Hono<{ Bindings: Env }>();

// Enable CORS for all origins (public API)
app.use('*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));

// Root health check endpoint (service-level)
app.get('/health', (c) => {
  return c.json({
    status: 'ok',
    service: 'labfork-agents',
    timestamp: new Date().toISOString(),
    environment: c.env.ENVIRONMENT || 'unknown',
  });
});

// Mount API routes under /api
app.route('/api', api);

// Default handler
export default app;

// Export workflows for Cloudflare Workers runtime
export { ManagerWorkflow } from './workflows/manager';
export { WorkerWorkflow } from './workflows/worker';

// Scheduled handler for cron triggers
export const scheduled: ExportedHandlerScheduledHandler<Env> = async (event, env, ctx) => {
  console.log(`Cron trigger fired at ${new Date().toISOString()}`);

  // 1. Run compute network cleanup (mark stale devices offline)
  try {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const now = new Date().toISOString();

    // Mark devices offline if no heartbeat in 5 minutes
    const offlineResult = await env.DB.prepare(`
      UPDATE compute_devices
      SET status = 'offline', updated_at = ?
      WHERE status IN ('online', 'busy')
        AND last_heartbeat < ?
    `).bind(now, fiveMinutesAgo).run();

    // Reset tasks that were assigned but device went offline
    const resetResult = await env.DB.prepare(`
      UPDATE compute_tasks
      SET status = 'pending', assigned_device_id = NULL, assigned_at = NULL
      WHERE status = 'assigned'
        AND assigned_at < ?
    `).bind(oneHourAgo).run();

    // Clear current_task_id for offline devices
    await env.DB.prepare(`
      UPDATE compute_devices
      SET current_task_id = NULL
      WHERE status = 'offline' AND current_task_id IS NOT NULL
    `).run();

    console.log(`[Cleanup] ${offlineResult.meta?.changes || 0} devices offline, ${resetResult.meta?.changes || 0} tasks reset`);
  } catch (error) {
    console.error('[Cleanup] Failed:', error);
  }

  // 2. Trigger manager workflow
  try {
    const instance = await env.MANAGER_WORKFLOW.create();
    console.log(`Started manager workflow: ${instance.id}`);
  } catch (error) {
    console.error('Failed to start manager workflow:', error);
  }
};
