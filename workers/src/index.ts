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
  ADMIN_API_KEY?: string;
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

  // 2. Generate crowd tasks from seed papers and research objectives if queue is low
  try {
    const countResult = await env.DB.prepare(`
      SELECT COUNT(*) as cnt FROM compute_tasks WHERE status = 'pending' AND min_tier = 'crowd'
    `).first<{ cnt: number }>();

    const pendingCrowdTasks = countResult?.cnt || 0;
    console.log(`[CrowdTasks] ${pendingCrowdTasks} pending crowd tasks`);

    if (pendingCrowdTasks < 20) {
      const now = new Date().toISOString();
      let generated = 0;

      // Strategy 1: Generate tasks from seed papers that haven't been fully processed
      const papers = await env.DB.prepare(`
        SELECT * FROM seed_papers
        WHERE tasks_generated < 7
        ORDER BY added_at DESC
        LIMIT 5
      `).all<{ id: string; title: string; abstract: string; domain: string; tasks_generated: number }>();

      if (papers.results && papers.results.length > 0) {
        for (const paper of papers.results) {
          if (generated >= (20 - pendingCrowdTasks)) break;

          // Generate paper-specific tasks
          const taskTypes = [
            {
              type: 'summarization' as const,
              input: {
                text: paper.abstract.slice(0, 2000),
                maxLength: 150,
                source: { type: 'paper', paperId: paper.id, section: 'abstract' },
                paperTitle: paper.title,
              },
              priority: 5,
            },
            {
              type: 'classification' as const,
              input: {
                text: `${paper.title}\n\n${paper.abstract.slice(0, 500)}`,
                categories: ['voice_synthesis', 'speech_recognition', 'nlp', 'audio_processing', 'machine_learning', 'other'],
                source: { type: 'paper', paperId: paper.id, section: 'classification' },
                paperTitle: paper.title,
              },
              priority: 4,
            },
            {
              type: 'embedding' as const,
              input: {
                text: `${paper.title}. ${paper.abstract.slice(0, 300)}`,
                model: 'default',
                source: { type: 'paper', paperId: paper.id, section: 'embedding' },
                paperTitle: paper.title,
              },
              priority: 3,
            },
            {
              type: 'assessment' as const,
              input: {
                prompt: `Rate this paper's relevance to voice synthesis research: "${paper.title}". Abstract: ${paper.abstract.slice(0, 400)}. Respond with JSON: { "relevanceScore": 1-10, "qualityScore": 1-10, "reasoning": "..." }`,
                context: 'Paper quality assessment',
                source: { type: 'paper', paperId: paper.id, section: 'assessment' },
                paperTitle: paper.title,
              },
              priority: 5,
            },
          ];

          let paperTasksGenerated = 0;
          for (const taskDef of taskTypes) {
            if (generated >= (20 - pendingCrowdTasks)) break;
            const taskId = `ctask_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
            try {
              await env.DB.prepare(`
                INSERT INTO compute_tasks (
                  id, type, input, config, status, priority, min_tier, created_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, 'crowd', ?)
              `).bind(
                taskId,
                taskDef.type,
                JSON.stringify(taskDef.input),
                JSON.stringify({ maxTokens: 256, temperature: 0.5 }),
                taskDef.priority,
                now
              ).run();
              generated++;
              paperTasksGenerated++;
            } catch (insertErr) {
              console.warn(`[CrowdTasks] Failed to insert ${taskDef.type} task for paper ${paper.id}:`, insertErr);
            }
          }

          // Update seed paper tasks_generated count with actual count
          if (paperTasksGenerated > 0) {
            await env.DB.prepare(`
              UPDATE seed_papers SET tasks_generated = tasks_generated + ? WHERE id = ?
            `).bind(paperTasksGenerated, paper.id).run();
          }
        }
      }

      // Strategy 2: Generate from research objectives if we still need more tasks
      if (generated < (20 - pendingCrowdTasks)) {
        const objectives = await env.DB.prepare(`
          SELECT * FROM research_objectives
          WHERE status IN ('pending', 'in_progress')
          ORDER BY priority DESC
          LIMIT 3
        `).all<{ id: string; title: string; description: string | null }>();

        if (objectives.results && objectives.results.length > 0) {
          for (const obj of objectives.results) {
            if (generated >= (20 - pendingCrowdTasks)) break;

            // Check if a pending task already exists for this objective (dedup)
            const existingTask = await env.DB.prepare(`
              SELECT id FROM compute_tasks
              WHERE status = 'pending'
                AND json_extract(input, '$.source.objectiveId') = ?
              LIMIT 1
            `).bind(obj.id).first();
            if (existingTask) continue; // Already has a pending task

            const taskId = `ctask_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
            try {
              await env.DB.prepare(`
                INSERT INTO compute_tasks (
                  id, type, input, config, status, priority, min_tier, created_at
                ) VALUES (?, 'summarization', ?, ?, 'pending', 5, 'crowd', ?)
              `).bind(
                taskId,
                JSON.stringify({
                  text: `Research objective: ${obj.title}. ${obj.description || ''}`,
                  maxLength: 150,
                  source: { type: 'objective', objectiveId: obj.id },
                }),
                JSON.stringify({ maxTokens: 256, temperature: 0.5 }),
                now
              ).run();
              generated++;
            } catch (insertErr) {
              console.warn(`[CrowdTasks] Failed to insert objective task for ${obj.id}:`, insertErr);
            }
          }
        }
      }

      console.log(`[CrowdTasks] Generated ${generated} tasks from papers/objectives`);
    }
  } catch (error) {
    console.error('[CrowdTasks] Failed:', error);
  }

  // 2b. Aggregate completed task results
  try {
    // Find papers with all tasks completed but not yet aggregated
    const completedPaperTasks = await env.DB.prepare(`
      SELECT
        json_extract(input, '$.source.paperId') as paper_id,
        json_extract(input, '$.paperTitle') as paper_title,
        COUNT(*) as total,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN json_extract(result, '$.serverValidated') = 1 THEN 1 ELSE 0 END) as validated
      FROM compute_tasks
      WHERE json_extract(input, '$.source.type') = 'paper'
        AND json_extract(input, '$.source.paperId') IS NOT NULL
      GROUP BY json_extract(input, '$.source.paperId')
      HAVING completed >= 3 AND validated > 0
      LIMIT 10
    `).all<{ paper_id: string; paper_title: string; total: number; completed: number; validated: number }>();

    if (completedPaperTasks.results) {
      for (const paper of completedPaperTasks.results) {
        // Check if we already have a research_result for this paper
        const existing = await env.DB.prepare(`
          SELECT id FROM research_results WHERE objective_title = ?
        `).bind(`Paper: ${paper.paper_id}`).first();

        if (existing) continue; // Already aggregated

        // Gather all completed results for this paper
        const results = await env.DB.prepare(`
          SELECT type, result FROM compute_tasks
          WHERE json_extract(input, '$.source.paperId') = ?
            AND status = 'completed'
        `).bind(paper.paper_id).all<{ type: string; result: string | null }>();

        const aggregated: Record<string, unknown> = { paperId: paper.paper_id, title: paper.paper_title };
        for (const r of results.results || []) {
          let parsed: unknown = null;
          try {
            parsed = r.result ? JSON.parse(r.result) : null;
          } catch {
            console.warn(`[Aggregation] Failed to parse result for paper ${paper.paper_id}, type ${r.type}`);
            parsed = { raw: r.result };
          }
          if (!aggregated[r.type]) aggregated[r.type] = [];
          (aggregated[r.type] as unknown[]).push(parsed);
        }

        // Store aggregated result
        const resultId = `res_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
        await env.DB.prepare(`
          INSERT INTO research_results (id, device_id, lab_id, objective_title, objective_description, success, output, created_at)
          VALUES (?, 'crowd', 'voice-clone', ?, ?, 1, ?, datetime('now'))
        `).bind(
          resultId,
          `Paper: ${paper.paper_id}`,
          `Aggregated ${paper.completed} compute results for: ${paper.paper_title || 'Unknown Paper'}`,
          JSON.stringify(aggregated)
        ).run();

        console.log(`[Aggregation] Aggregated results for paper ${paper.paper_id}`);
      }
    }
  } catch (error) {
    console.error('[Aggregation] Failed:', error);
  }

  // 3. Trigger manager workflow
  try {
    const instance = await env.MANAGER_WORKFLOW.create();
    console.log(`Started manager workflow: ${instance.id}`);
  } catch (error) {
    console.error('Failed to start manager workflow:', error);
  }
};
