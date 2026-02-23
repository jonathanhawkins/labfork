/**
 * Nudge Engine — Persistent autonomous task manager for AI CLI tools.
 *
 * A Cloudflare Worker with a cron trigger and three HTTP endpoints.
 * Any tool that can `curl` can participate.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import api from './api';
import { runCronCycle } from './cron';

export interface Env {
  DB: D1Database;
  ADMIN_API_KEY?: string;
  ENVIRONMENT?: string;
}

const app = new Hono<{ Bindings: Env }>();

// CORS for all origins (workers poll from anywhere)
app.use('*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));

// Health check
app.get('/', (c) => {
  return c.json({
    name: 'nudge-engine',
    version: '0.1.0',
    description: 'Persistent autonomous task manager for AI CLI tools',
    endpoints: {
      register: 'POST /register',
      poll: 'POST /poll',
      report: 'POST /report',
      observe: 'POST /observe',
      tasks: 'GET /tasks',
      workers: 'GET /workers',
      stats: 'GET /stats',
    },
  });
});

// Admin-only manual cron trigger (for debugging, avoids waiting 15 min)
// Always uses force=true to bypass the idempotency guard since manual triggers are intentional.
app.post('/cron', async (c) => {
  const authHeader = c.req.header('Authorization');
  const token = authHeader?.replace('Bearer ', '');
  if (token !== c.env.ADMIN_API_KEY) {
    return c.json({ error: 'Unauthorized' }, 401);
  }
  const result = await runCronCycle(c.env.DB, { force: true });
  return c.json({ success: true, result });
});

// Mount all API routes at root level (clean protocol)
app.route('/', api);

// Default export must include both fetch and scheduled for Cloudflare Workers
export default {
  fetch: app.fetch,
  async scheduled(event: ScheduledEvent, env: Env) {
    console.log(`[Nudge] Cron fired at ${new Date().toISOString()}`);

    try {
      const result = await runCronCycle(env.DB);
      console.log(`[Nudge] Cron complete:`, JSON.stringify(result));
    } catch (error) {
      console.error('[Nudge] Cron failed:', error);
    }
  },
};
