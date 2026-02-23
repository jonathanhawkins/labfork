/**
 * Meta-Agent Dashboard
 *
 * Real-time status tracking for all 5 meta-agents with live activity log,
 * performance metrics, discovery counts, health monitoring, and admin controls.
 */

import {
  MetaAgentStatus,
  MetaAgentName,
  AgentStatus,
  HealthStatus,
  AgentMetrics,
  Discovery,
  ActivityLogEntry,
  MetaAgentDashboard,
  SystemHealth,
  DashboardSummary,
  SignificanceLevel,
} from "./types";

// ============================================================================
// Agent Dashboard Interface
// ============================================================================

export interface AgentDashboardState {
  agents: Map<MetaAgentName, MetaAgentStatus>;
  systemHealth: SystemHealth;
  activityBuffer: ActivityLogEntry[];
  lastUpdate: string;
}

export interface AgentConfig {
  maxActivityLogEntries: number;
  maxRecentDiscoveries: number;
  healthCheckIntervalMs: number;
  activityRetentionHours: number;
}

const DEFAULT_CONFIG: AgentConfig = {
  maxActivityLogEntries: 100,
  maxRecentDiscoveries: 20,
  healthCheckIntervalMs: 30000,
  activityRetentionHours: 24,
};

// ============================================================================
// Agent Definitions
// ============================================================================

export const AGENT_DEFINITIONS: Record<
  MetaAgentName,
  { displayName: string; description: string }
> = {
  "synergy-detector": {
    displayName: "Synergy Detector",
    description: "Discovers beneficial technique combinations across labs",
  },
  "pattern-recognizer": {
    displayName: "Pattern Recognizer",
    description: "Identifies recurring successful patterns in research",
  },
  "gap-analyzer": {
    displayName: "Gap Analyzer",
    description: "Finds unexplored areas and missing techniques",
  },
  "evolution-engine": {
    displayName: "Evolution Engine",
    description: "Evolves and optimizes techniques using genetic algorithms",
  },
  "transfer-agent": {
    displayName: "Transfer Agent",
    description: "Adapts techniques across different domains",
  },
};

// ============================================================================
// Factory Functions
// ============================================================================

export function createAgentDashboard(): AgentDashboardState {
  const agents = new Map<MetaAgentName, MetaAgentStatus>();

  // Initialize all agents
  const agentNames: MetaAgentName[] = [
    "synergy-detector",
    "pattern-recognizer",
    "gap-analyzer",
    "evolution-engine",
    "transfer-agent",
  ];

  for (const name of agentNames) {
    agents.set(name, createAgentStatus(name));
  }

  return {
    agents,
    systemHealth: createInitialSystemHealth(),
    activityBuffer: [],
    lastUpdate: new Date().toISOString(),
  };
}

function createAgentStatus(name: MetaAgentName): MetaAgentStatus {
  const def = AGENT_DEFINITIONS[name];
  const now = new Date().toISOString();

  return {
    id: `agent-${name}`,
    name,
    displayName: def.displayName,
    status: "idle",
    health: "healthy",
    lastActivity: now,
    queuedTasks: 0,
    metrics: createInitialMetrics(),
    recentDiscoveries: [],
    activityLog: [],
    isEnabled: true,
  };
}

function createInitialMetrics(): AgentMetrics {
  return {
    totalRuns: 0,
    successfulRuns: 0,
    failedRuns: 0,
    avgExecutionTimeMs: 0,
    discoveriesCount: 0,
    lastHourActivity: 0,
    last24HourActivity: 0,
    uptime: 100,
    errorRate: 0,
  };
}

function createInitialSystemHealth(): SystemHealth {
  return {
    overall: "healthy",
    cpu: 0,
    memory: 0,
    activeConnections: 0,
    queueDepth: 0,
    errorRate: 0,
    latency: 0,
  };
}

// ============================================================================
// Agent Status Management
// ============================================================================

export function getAgentStatus(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName
): MetaAgentStatus | null {
  return dashboard.agents.get(agentName) || null;
}

export function getAllAgentStatuses(
  dashboard: AgentDashboardState
): MetaAgentStatus[] {
  return Array.from(dashboard.agents.values());
}

export function updateAgentStatus(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName,
  status: AgentStatus,
  currentTask?: string
): MetaAgentStatus | null {
  const agent = dashboard.agents.get(agentName);
  if (!agent) return null;

  const updated: MetaAgentStatus = {
    ...agent,
    status,
    currentTask,
    lastActivity: new Date().toISOString(),
  };

  dashboard.agents.set(agentName, updated);
  dashboard.lastUpdate = new Date().toISOString();

  return updated;
}

export function setAgentHealth(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName,
  health: HealthStatus
): MetaAgentStatus | null {
  const agent = dashboard.agents.get(agentName);
  if (!agent) return null;

  const updated: MetaAgentStatus = {
    ...agent,
    health,
    lastActivity: new Date().toISOString(),
  };

  dashboard.agents.set(agentName, updated);
  updateSystemHealth(dashboard);

  return updated;
}

// ============================================================================
// Agent Control
// ============================================================================

export function enableAgent(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName
): MetaAgentStatus | null {
  const agent = dashboard.agents.get(agentName);
  if (!agent) return null;

  const updated: MetaAgentStatus = {
    ...agent,
    isEnabled: true,
    status: "idle",
    pausedAt: undefined,
    pauseReason: undefined,
    lastActivity: new Date().toISOString(),
  };

  dashboard.agents.set(agentName, updated);
  logActivity(dashboard, agentName, "enabled", "Agent enabled", "success");

  return updated;
}

export function disableAgent(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName,
  reason?: string
): MetaAgentStatus | null {
  const agent = dashboard.agents.get(agentName);
  if (!agent) return null;

  const updated: MetaAgentStatus = {
    ...agent,
    isEnabled: false,
    status: "paused",
    pausedAt: new Date().toISOString(),
    pauseReason: reason,
    currentTask: undefined,
    lastActivity: new Date().toISOString(),
  };

  dashboard.agents.set(agentName, updated);
  logActivity(
    dashboard,
    agentName,
    "disabled",
    `Agent disabled: ${reason || "No reason provided"}`,
    "success"
  );

  return updated;
}

export function pauseAgent(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName,
  reason?: string
): MetaAgentStatus | null {
  return disableAgent(dashboard, agentName, reason);
}

export function resumeAgent(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName
): MetaAgentStatus | null {
  return enableAgent(dashboard, agentName);
}

// ============================================================================
// Activity Logging
// ============================================================================

export function logActivity(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName,
  action: string,
  details: string,
  result: "success" | "failure" | "pending",
  duration?: number,
  config: AgentConfig = DEFAULT_CONFIG
): void {
  const agent = dashboard.agents.get(agentName);
  if (!agent) return;

  const entry: ActivityLogEntry = {
    id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date().toISOString(),
    action,
    details,
    result,
    duration,
  };

  // Add to agent log
  const updatedLog = [entry, ...agent.activityLog].slice(
    0,
    config.maxActivityLogEntries
  );

  // Update metrics
  const updatedMetrics = updateMetricsFromActivity(
    agent.metrics,
    result,
    duration
  );

  const updated: MetaAgentStatus = {
    ...agent,
    activityLog: updatedLog,
    metrics: updatedMetrics,
    lastActivity: entry.timestamp,
  };

  dashboard.agents.set(agentName, updated);

  // Add to global buffer
  dashboard.activityBuffer.unshift(entry);
  if (dashboard.activityBuffer.length > config.maxActivityLogEntries * 5) {
    dashboard.activityBuffer = dashboard.activityBuffer.slice(
      0,
      config.maxActivityLogEntries * 5
    );
  }
}

function updateMetricsFromActivity(
  metrics: AgentMetrics,
  result: "success" | "failure" | "pending",
  duration?: number
): AgentMetrics {
  const totalRuns = metrics.totalRuns + 1;
  const successfulRuns =
    metrics.successfulRuns + (result === "success" ? 1 : 0);
  const failedRuns = metrics.failedRuns + (result === "failure" ? 1 : 0);

  // Update average execution time
  let avgExecutionTimeMs = metrics.avgExecutionTimeMs;
  if (duration !== undefined) {
    avgExecutionTimeMs =
      (metrics.avgExecutionTimeMs * metrics.totalRuns + duration) / totalRuns;
  }

  // Calculate error rate
  const errorRate = totalRuns > 0 ? (failedRuns / totalRuns) * 100 : 0;

  return {
    ...metrics,
    totalRuns,
    successfulRuns,
    failedRuns,
    avgExecutionTimeMs,
    errorRate,
    lastHourActivity: metrics.lastHourActivity + 1,
    last24HourActivity: metrics.last24HourActivity + 1,
  };
}

// ============================================================================
// Discovery Tracking
// ============================================================================

export function recordDiscovery(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName,
  type: string,
  description: string,
  significance: SignificanceLevel,
  metadata: Record<string, unknown> = {},
  config: AgentConfig = DEFAULT_CONFIG
): Discovery {
  const agent = dashboard.agents.get(agentName);

  const discovery: Discovery = {
    id: `disc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    type,
    description,
    significance,
    timestamp: new Date().toISOString(),
    metadata,
  };

  if (agent) {
    const updatedDiscoveries = [discovery, ...agent.recentDiscoveries].slice(
      0,
      config.maxRecentDiscoveries
    );

    const updated: MetaAgentStatus = {
      ...agent,
      recentDiscoveries: updatedDiscoveries,
      metrics: {
        ...agent.metrics,
        discoveriesCount: agent.metrics.discoveriesCount + 1,
      },
      lastActivity: discovery.timestamp,
    };

    dashboard.agents.set(agentName, updated);

    logActivity(
      dashboard,
      agentName,
      "discovery",
      `${type}: ${description}`,
      "success"
    );
  }

  return discovery;
}

export function getRecentDiscoveries(
  dashboard: AgentDashboardState,
  limit: number = 20
): Discovery[] {
  const allDiscoveries: Discovery[] = [];

  for (const agent of Array.from(dashboard.agents.values())) {
    allDiscoveries.push(...agent.recentDiscoveries);
  }

  return allDiscoveries
    .sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    )
    .slice(0, limit);
}

// ============================================================================
// System Health
// ============================================================================

export function updateSystemHealth(dashboard: AgentDashboardState): void {
  const agents = Array.from(dashboard.agents.values());

  let healthyCount = 0;
  let degradedCount = 0;
  let unhealthyCount = 0;
  let totalErrorRate = 0;
  let totalLatency = 0;
  let totalQueueDepth = 0;

  for (const agent of agents) {
    if (agent.health === "healthy") healthyCount++;
    else if (agent.health === "degraded") degradedCount++;
    else if (agent.health === "unhealthy") unhealthyCount++;

    totalErrorRate += agent.metrics.errorRate;
    totalLatency += agent.metrics.avgExecutionTimeMs;
    totalQueueDepth += agent.queuedTasks;
  }

  const agentCount = agents.length;
  const overall: HealthStatus =
    unhealthyCount > 0
      ? "unhealthy"
      : degradedCount > agentCount / 2
      ? "degraded"
      : "healthy";

  dashboard.systemHealth = {
    overall,
    cpu: Math.random() * 30 + 10, // Simulated
    memory: Math.random() * 40 + 20, // Simulated
    activeConnections: agents.filter((a) => a.status === "running").length,
    queueDepth: totalQueueDepth,
    errorRate: agentCount > 0 ? totalErrorRate / agentCount : 0,
    latency: agentCount > 0 ? totalLatency / agentCount : 0,
  };

  dashboard.lastUpdate = new Date().toISOString();
}

export function setSystemMetrics(
  dashboard: AgentDashboardState,
  metrics: Partial<SystemHealth>
): void {
  dashboard.systemHealth = {
    ...dashboard.systemHealth,
    ...metrics,
  };
  dashboard.lastUpdate = new Date().toISOString();
}

// ============================================================================
// Task Queue Management
// ============================================================================

export function queueTask(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName
): MetaAgentStatus | null {
  const agent = dashboard.agents.get(agentName);
  if (!agent) return null;

  const updated: MetaAgentStatus = {
    ...agent,
    queuedTasks: agent.queuedTasks + 1,
  };

  dashboard.agents.set(agentName, updated);
  updateSystemHealth(dashboard);

  return updated;
}

export function dequeueTask(
  dashboard: AgentDashboardState,
  agentName: MetaAgentName
): MetaAgentStatus | null {
  const agent = dashboard.agents.get(agentName);
  if (!agent || agent.queuedTasks <= 0) return null;

  const updated: MetaAgentStatus = {
    ...agent,
    queuedTasks: agent.queuedTasks - 1,
  };

  dashboard.agents.set(agentName, updated);
  updateSystemHealth(dashboard);

  return updated;
}

// ============================================================================
// Dashboard Generation
// ============================================================================

export function generateDashboard(
  state: AgentDashboardState
): MetaAgentDashboard {
  const agents = getAllAgentStatuses(state);
  const discoveries = getRecentDiscoveries(state, 10);

  // Count discoveries today
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const discoveriesToday = discoveries.filter(
    (d) => new Date(d.timestamp) >= today
  ).length;

  // Count alerts (simulated)
  const alertsToday = Math.floor(Math.random() * 10);

  const summary: DashboardSummary = {
    totalDiscoveries: agents.reduce(
      (sum, a) => sum + a.metrics.discoveriesCount,
      0
    ),
    discoveriesToday,
    activeAgents: agents.filter((a) => a.status === "running").length,
    pausedAgents: agents.filter((a) => a.status === "paused").length,
    pendingTasks: agents.reduce((sum, a) => sum + a.queuedTasks, 0),
    alertsToday,
  };

  // Generate recent alerts from discoveries
  const recentAlerts = discoveries.slice(0, 5).map((d) => ({
    id: `alert-${d.id}`,
    type: "breakthrough" as const,
    significance: d.significance,
    title: d.type,
    description: d.description,
    relatedIds: [],
    metadata: d.metadata as Record<string, unknown>,
    createdAt: d.timestamp,
    read: false,
    dismissed: false,
  }));

  return {
    agents,
    systemHealth: state.systemHealth,
    recentAlerts,
    summary,
    updatedAt: state.lastUpdate,
  };
}

// ============================================================================
// Activity Cleanup
// ============================================================================

export function cleanupOldActivity(
  dashboard: AgentDashboardState,
  config: AgentConfig = DEFAULT_CONFIG
): number {
  const cutoff = new Date();
  cutoff.setHours(cutoff.getHours() - config.activityRetentionHours);
  let count = 0;

  // Clean global buffer
  const originalLength = dashboard.activityBuffer.length;
  dashboard.activityBuffer = dashboard.activityBuffer.filter(
    (entry) => new Date(entry.timestamp) > cutoff
  );
  count += originalLength - dashboard.activityBuffer.length;

  // Clean agent logs
  for (const [name, agent] of Array.from(dashboard.agents.entries())) {
    const originalLogLength = agent.activityLog.length;
    const filteredLog = agent.activityLog.filter(
      (entry) => new Date(entry.timestamp) > cutoff
    );
    count += originalLogLength - filteredLog.length;

    dashboard.agents.set(name, {
      ...agent,
      activityLog: filteredLog,
    });
  }

  // Reset hourly activity counter if needed
  const now = new Date();
  const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

  for (const [name, agent] of Array.from(dashboard.agents.entries())) {
    const hourlyActivity = agent.activityLog.filter(
      (entry) => new Date(entry.timestamp) > oneHourAgo
    ).length;

    const dailyActivity = agent.activityLog.filter(
      (entry) => new Date(entry.timestamp) > cutoff
    ).length;

    dashboard.agents.set(name, {
      ...agent,
      metrics: {
        ...agent.metrics,
        lastHourActivity: hourlyActivity,
        last24HourActivity: dailyActivity,
      },
    });
  }

  return count;
}

// ============================================================================
// Export
// ============================================================================

export const agentConfig = DEFAULT_CONFIG;
