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

  try {
    // Trigger manager workflow on schedule
    const instance = await env.MANAGER_WORKFLOW.create();
    console.log(`Started manager workflow: ${instance.id}`);
  } catch (error) {
    console.error('Failed to start manager workflow:', error);
  }
};
