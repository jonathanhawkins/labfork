// useLabActivities - Hook for fetching real activity data
// Aggregates data from multiple sources into unified activity state

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  ActivityState,
  ActivityWithConfig,
  getActivityConfig,
  getDefaultActivityConfig,
} from './index';

export interface LabActivitiesData {
  activities: ActivityWithConfig[];
  activeCount: number;
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

export interface UseLabActivitiesOptions {
  pollInterval?: number;  // ms, 0 to disable polling
  enabled?: boolean;
}

/**
 * Fetch activities from the aggregated API endpoint
 */
async function fetchActivities(): Promise<ActivityState[]> {
  try {
    const response = await fetch('/api/lab/activities');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    return data.activities || [];
  } catch (error) {
    console.error('[useLabActivities] Failed to fetch activities:', error);
    throw error;
  }
}

/**
 * Hook for managing lab activities state
 */
export function useLabActivities(
  options: UseLabActivitiesOptions = {}
): LabActivitiesData {
  const { pollInterval = 3000, enabled = true } = options;

  const [activities, setActivities] = useState<ActivityWithConfig[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const mountedRef = useRef(true);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Map activity state to activity with config
  const mapActivityToWithConfig = useCallback(
    (activity: ActivityState): ActivityWithConfig => {
      const config =
        getActivityConfig(activity.type) ||
        getDefaultActivityConfig(activity.type);
      return {
        ...activity,
        config,
      };
    },
    []
  );

  // Fetch and update activities
  const refresh = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      setIsLoading(true);
      const rawActivities = await fetchActivities();

      if (!mountedRef.current) return;

      const mappedActivities = rawActivities.map(mapActivityToWithConfig);

      // Sort by priority (higher first)
      mappedActivities.sort((a, b) => b.config.priority - a.config.priority);

      setActivities(mappedActivities);
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
  }, [mapActivityToWithConfig]);

  // Initial fetch and polling
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

  const activeCount = activities.filter((a) => a.active).length;

  return {
    activities,
    activeCount,
    isLoading,
    error,
    lastUpdated,
    refresh,
  };
}

/**
 * Get the primary (highest priority) active activity
 */
export function getPrimaryActivity(
  activities: ActivityWithConfig[]
): ActivityWithConfig | null {
  const active = activities.filter((a) => a.active);
  if (active.length === 0) return null;
  return active[0]; // Already sorted by priority
}

/**
 * Get activities grouped by their assigned agent
 */
export function getActivitiesByAgent(
  activities: ActivityWithConfig[]
): Map<string, ActivityWithConfig[]> {
  const byAgent = new Map<string, ActivityWithConfig[]>();

  for (const activity of activities) {
    if (!activity.active) continue;
    const agent = activity.assignedAgent || 'unassigned';
    const existing = byAgent.get(agent) || [];
    existing.push(activity);
    byAgent.set(agent, existing);
  }

  return byAgent;
}

/**
 * Convert activities to log entries for the activity log
 */
export function activitiesToLog(
  activities: ActivityWithConfig[]
): { agent: string; action: string; time: Date }[] {
  return activities
    .filter((a) => a.active)
    .map((a) => ({
      agent: a.assignedAgent || 'System',
      action: a.message || a.config.name,
      time: a.startedAt ? new Date(a.startedAt) : new Date(),
    }));
}
