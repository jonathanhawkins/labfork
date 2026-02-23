/**
 * Observers — read the world and return structured state.
 *
 * Observers don't make decisions. They report facts.
 * The cron reads observations and feeds them to rules.
 */

export interface TasksObservation {
  pending: number;
  assigned: number;
  completed24h: number;
  failed24h: number;
  stuck: { id: string; action: string; assignedMinutesAgo: number }[];
  total: number;
}

export interface WorkersObservation {
  online: number;
  busy: number;
  offline: number;
  idle: number; // online but not working
  stale: { id: string; name: string; lastHeartbeatMinutesAgo: number }[];
}

export interface ExternalObservation {
  type: string;
  data: Record<string, unknown>;
  observedAt: string;
}

/**
 * Observe task queue state from the database.
 */
export async function observeTasks(db: D1Database): Promise<TasksObservation> {
  const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const tenMinAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();

  // Single query for all counts including 24h completed/failed
  const counts = await db.prepare(`
    SELECT
      SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
      SUM(CASE WHEN status = 'assigned' THEN 1 ELSE 0 END) as assigned,
      SUM(CASE WHEN status = 'completed' AND completed_at > ? THEN 1 ELSE 0 END) as completed24h,
      SUM(CASE WHEN status = 'failed' AND completed_at > ? THEN 1 ELSE 0 END) as failed24h,
      COUNT(*) as total
    FROM tasks
  `).bind(oneDayAgo, oneDayAgo).first<{
    pending: number; assigned: number; completed24h: number; failed24h: number; total: number;
  }>();

  // Stuck tasks: assigned but no progress for > 10 minutes
  const stuckRows = await db.prepare(`
    SELECT id, action, assigned_at FROM tasks
    WHERE status = 'assigned' AND assigned_at < ?
    ORDER BY assigned_at ASC
    LIMIT 10
  `).bind(tenMinAgo).all<{ id: string; action: string; assigned_at: string }>();

  const stuck = (stuckRows.results || []).map((t) => ({
    id: t.id,
    action: t.action,
    assignedMinutesAgo: Math.round((Date.now() - new Date(t.assigned_at).getTime()) / 60000),
  }));

  return {
    pending: counts?.pending || 0,
    assigned: counts?.assigned || 0,
    completed24h: counts?.completed24h || 0,
    failed24h: counts?.failed24h || 0,
    stuck,
    total: counts?.total || 0,
  };
}

/**
 * Observe worker fleet state.
 */
export async function observeWorkers(db: D1Database): Promise<WorkersObservation> {
  const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();

  const counts = await db.prepare(`
    SELECT
      SUM(CASE WHEN status IN ('online','busy') AND last_heartbeat > ? THEN 1 ELSE 0 END) as online_total,
      SUM(CASE WHEN status = 'busy' AND last_heartbeat > ? THEN 1 ELSE 0 END) as busy,
      SUM(CASE WHEN status = 'offline' OR last_heartbeat <= ? THEN 1 ELSE 0 END) as offline
    FROM workers
  `).bind(fiveMinAgo, fiveMinAgo, fiveMinAgo).first<{
    online_total: number;
    busy: number;
    offline: number;
  }>();

  const onlineTotal = counts?.online_total || 0;
  const busy = counts?.busy || 0;

  // Find workers with stale heartbeats that are still marked online/busy
  const staleRows = await db.prepare(`
    SELECT id, name, last_heartbeat FROM workers
    WHERE status IN ('online', 'busy') AND last_heartbeat < ?
    LIMIT 10
  `).bind(fiveMinAgo).all<{ id: string; name: string; last_heartbeat: string }>();

  const stale = (staleRows.results || []).map((w) => ({
    id: w.id,
    name: w.name,
    lastHeartbeatMinutesAgo: Math.round((Date.now() - new Date(w.last_heartbeat).getTime()) / 60000),
  }));

  return {
    online: onlineTotal,
    busy,
    offline: counts?.offline || 0,
    idle: onlineTotal - busy,
    stale,
  };
}

/**
 * Get the latest external observations (submitted by workers via /observe).
 */
export async function getLatestObservations(db: D1Database): Promise<ExternalObservation[]> {
  // 7-day window so long-lived observations (like goals) don't vanish
  const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();

  // Subquery to get the latest observation per type (avoids GROUP BY data ambiguity)
  const rows = await db.prepare(`
    SELECT o.type, o.data, o.created_at
    FROM observations o
    INNER JOIN (
      SELECT type, MAX(created_at) as max_created
      FROM observations
      WHERE created_at > ?
      GROUP BY type
    ) latest ON o.type = latest.type AND o.created_at = latest.max_created
    ORDER BY o.created_at DESC
    LIMIT 20
  `).bind(sevenDaysAgo).all<{ type: string; data: string; created_at: string }>();

  return (rows.results || []).map((r) => {
    let data: Record<string, unknown> = {};
    try { data = JSON.parse(r.data); } catch { /* ignore */ }
    return { type: r.type, data, observedAt: r.created_at };
  });
}
