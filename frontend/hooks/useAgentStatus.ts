/**
 * Agent Status Hooks
 *
 * Custom hooks for fetching agent status and work log data
 * with automatic polling and error handling.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface TaskSummary {
  id: string;
  title: string;
  status: string;
  priority: number;
}

export interface AgentStatus {
  id: string;
  name: string;
  type: string;
  status: 'idle' | 'working' | 'blocked';
  current_task_id: string | null;
  current_task: TaskSummary | null;
  capabilities: string[];
  created_at: string;
  updated_at: string;
}

export interface AgentDetailedStatus extends AgentStatus {
  recent_work_log: WorkLogEntry[];
  stats: {
    tasks_completed_today: number;
    tasks_completed_total: number;
    average_task_duration_minutes: number;
  };
}

export interface WorkLogEntry {
  id: string;
  agent_id: string;
  task_id: string | null;
  action: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface AgentStatusResponse {
  agents: AgentStatus[];
  count: number;
  source: 'workers' | 'demo';
  timestamp: string;
  fallback?: boolean;
}

export interface WorkLogResponse {
  entries: WorkLogEntry[];
  count: number;
  source: 'workers' | 'demo';
  timestamp: string;
  fallback?: boolean;
}

// ============================================================================
// useAgentStatus Hook
// ============================================================================

export interface UseAgentStatusOptions {
  pollInterval?: number; // ms, 0 to disable polling (default: 5000)
  enabled?: boolean;
}

export interface UseAgentStatusResult {
  agents: AgentStatus[];
  isLoading: boolean;
  error: string | null;
  isDemo: boolean;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

/**
 * Hook for fetching all agent statuses with automatic polling
 */
export function useAgentStatus(
  options: UseAgentStatusOptions = {}
): UseAgentStatusResult {
  const { pollInterval = 5000, enabled = true } = options;

  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const mountedRef = useRef(true);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const refresh = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      setIsLoading(true);
      const response = await fetch('/api/agents/status', {
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: AgentStatusResponse = await response.json();

      if (!mountedRef.current) return;

      setAgents(data.agents || []);
      setIsDemo(data.source === 'demo' || data.fallback === true);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    mountedRef.current = true;

    if (enabled) {
      refresh();
    }

    return () => {
      mountedRef.current = false;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [enabled, refresh]);

  // Set up polling
  useEffect(() => {
    if (!enabled || pollInterval <= 0) return;

    const poll = () => {
      pollTimeoutRef.current = setTimeout(async () => {
        await refresh();
        if (mountedRef.current && enabled) {
          poll();
        }
      }, pollInterval);
    };

    poll();

    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [enabled, pollInterval, refresh]);

  return {
    agents,
    isLoading,
    error,
    isDemo,
    lastUpdated,
    refresh,
  };
}

// ============================================================================
// useAgentWorkLog Hook
// ============================================================================

export interface UseAgentWorkLogOptions {
  limit?: number;
  agentId?: string;
  taskId?: string;
  pollInterval?: number; // ms, 0 to disable polling (default: 5000)
  enabled?: boolean;
}

export interface UseAgentWorkLogResult {
  entries: WorkLogEntry[];
  isLoading: boolean;
  error: string | null;
  isDemo: boolean;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

/**
 * Hook for fetching work log entries with automatic polling
 */
export function useAgentWorkLog(
  options: UseAgentWorkLogOptions = {}
): UseAgentWorkLogResult {
  const {
    limit = 20,
    agentId,
    taskId,
    pollInterval = 5000,
    enabled = true,
  } = options;

  const [entries, setEntries] = useState<WorkLogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const mountedRef = useRef(true);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const refresh = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      setIsLoading(true);

      // Build query params
      const params = new URLSearchParams({ limit: String(limit) });
      if (agentId) params.set('agent_id', agentId);
      if (taskId) params.set('task_id', taskId);

      const response = await fetch(`/api/agents/work-log?${params.toString()}`, {
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: WorkLogResponse = await response.json();

      if (!mountedRef.current) return;

      setEntries(data.entries || []);
      setIsDemo(data.source === 'demo' || data.fallback === true);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [limit, agentId, taskId]);

  // Initial fetch
  useEffect(() => {
    mountedRef.current = true;

    if (enabled) {
      refresh();
    }

    return () => {
      mountedRef.current = false;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [enabled, refresh]);

  // Set up polling
  useEffect(() => {
    if (!enabled || pollInterval <= 0) return;

    const poll = () => {
      pollTimeoutRef.current = setTimeout(async () => {
        await refresh();
        if (mountedRef.current && enabled) {
          poll();
        }
      }, pollInterval);
    };

    poll();

    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [enabled, pollInterval, refresh]);

  return {
    entries,
    isLoading,
    error,
    isDemo,
    lastUpdated,
    refresh,
  };
}

// ============================================================================
// useSingleAgentStatus Hook
// ============================================================================

export interface UseSingleAgentStatusOptions {
  pollInterval?: number;
  enabled?: boolean;
}

export interface UseSingleAgentStatusResult {
  agent: AgentDetailedStatus | null;
  isLoading: boolean;
  error: string | null;
  isDemo: boolean;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

/**
 * Hook for fetching a single agent's detailed status
 */
export function useSingleAgentStatus(
  agentId: string | null,
  options: UseSingleAgentStatusOptions = {}
): UseSingleAgentStatusResult {
  const { pollInterval = 5000, enabled = true } = options;

  const [agent, setAgent] = useState<AgentDetailedStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const mountedRef = useRef(true);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const refresh = useCallback(async () => {
    if (!mountedRef.current || !agentId) return;

    try {
      setIsLoading(true);

      const response = await fetch(`/api/agents/${agentId}/status`, {
        cache: 'no-store',
      });

      if (!response.ok) {
        if (response.status === 404) {
          setAgent(null);
          setError('Agent not found');
          return;
        }
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      if (!mountedRef.current) return;

      setAgent(data);
      setIsDemo(data.source === 'demo' || data.fallback === true);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, [agentId]);

  // Initial fetch
  useEffect(() => {
    mountedRef.current = true;

    if (enabled && agentId) {
      refresh();
    } else {
      setAgent(null);
      setIsLoading(false);
    }

    return () => {
      mountedRef.current = false;
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [enabled, agentId, refresh]);

  // Set up polling
  useEffect(() => {
    if (!enabled || !agentId || pollInterval <= 0) return;

    const poll = () => {
      pollTimeoutRef.current = setTimeout(async () => {
        await refresh();
        if (mountedRef.current && enabled) {
          poll();
        }
      }, pollInterval);
    };

    poll();

    return () => {
      if (pollTimeoutRef.current) {
        clearTimeout(pollTimeoutRef.current);
      }
    };
  }, [enabled, agentId, pollInterval, refresh]);

  return {
    agent,
    isLoading,
    error,
    isDemo,
    lastUpdated,
    refresh,
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert work log entries to activity log format for display
 */
export function workLogToActivityLog(
  entries: WorkLogEntry[],
  agents: AgentStatus[]
): { agent: string; action: string; time: Date }[] {
  const agentMap = new Map(agents.map((a) => [a.id, a.name]));

  return entries.map((entry) => ({
    agent: agentMap.get(entry.agent_id) || entry.agent_id,
    action: formatWorkLogAction(entry),
    time: new Date(entry.created_at),
  }));
}

/**
 * Format a work log entry's action for display
 */
function formatWorkLogAction(entry: WorkLogEntry): string {
  const { action, details } = entry;

  // Map common actions to readable descriptions
  const actionMap: Record<string, (details: Record<string, unknown>) => string> = {
    task_started: (d) => `Started: ${d.task || 'new task'}`,
    task_completed: (d) => `Completed: ${d.task || 'task'}`,
    research_completed: (d) => `Analyzed ${d.papers_analyzed || 'papers'}`,
    code_implemented: (d) => `Implemented ${d.component || 'code'} (${d.files_changed || '?'} files)`,
    synergy_detected: (d) => `Found synergy: ${d.synergy || 'unknown'}`,
    schedule_created: (d) => `Scheduled ${d.training_jobs || '?'} training jobs`,
    error_occurred: (d) => `Error: ${d.message || 'unknown error'}`,
    worker_triggered: () => 'Started working on task',
    manager_triggered: () => 'Manager workflow activated',
  };

  const formatter = actionMap[action];
  if (formatter) {
    return formatter(details);
  }

  // Default: capitalize and format action name
  return action.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Get agent status color for UI
 */
export function getAgentStatusColor(status: string): string {
  switch (status) {
    case 'working':
      return 'text-green-500';
    case 'blocked':
      return 'text-amber-500';
    case 'idle':
    default:
      return 'text-muted-foreground';
  }
}

/**
 * Get working agents count
 */
export function getWorkingAgentsCount(agents: AgentStatus[]): number {
  return agents.filter((a) => a.status === 'working').length;
}
