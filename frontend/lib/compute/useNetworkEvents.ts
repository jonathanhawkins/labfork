/**
 * React Hook for Network Events (Polling)
 *
 * Polls the compute network stats endpoint for real-time updates
 */

import { useState, useEffect, useCallback, useRef } from "react";
import type { NetworkStats, DeviceTier } from "./types";

/**
 * Completed task event
 */
export interface CompletedTask {
  count: number;
  totalToday: number;
  creditsAwarded: number;
}

/**
 * Hook state
 */
export interface NetworkEventsState {
  /** Connection status */
  isConnected: boolean;
  /** Current network statistics */
  networkStats: NetworkStats | null;
  /** Recent task completion event */
  recentCompletion: CompletedTask | null;
  /** Last error */
  error: Error | null;
  /** Whether polling is supported */
  isSupported: boolean;
}

/**
 * Default network stats
 */
const DEFAULT_STATS: NetworkStats = {
  totalDevices: 0,
  onlineDevices: 0,
  devicesByTier: {
    power: 0,
    standard: 0,
    crowd: 0,
  },
  totalCompute: 0,
  pendingTasks: 0,
  processingTasks: 0,
  completedToday: 0,
  creditsToday: 0,
};

/** Polling interval in ms */
const POLL_INTERVAL = 5000;

/**
 * Hook for real-time network events via polling
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const {
 *     isConnected,
 *     networkStats,
 *     recentCompletion,
 *     error,
 *   } = useNetworkEvents();
 *
 *   return (
 *     <div>
 *       <p>Status: {isConnected ? 'Connected' : 'Disconnected'}</p>
 *       <p>Devices: {networkStats?.onlineDevices}</p>
 *     </div>
 *   );
 * }
 * ```
 */
export function useNetworkEvents(): NetworkEventsState {
  const [isConnected, setIsConnected] = useState(false);
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [recentCompletion, setRecentCompletion] = useState<CompletedTask | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const previousCompletedRef = useRef<number>(0);

  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch("/api/compute/stats");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Transform Workers API response to NetworkStats format
      const stats: NetworkStats = {
        totalDevices: data.devices?.total || 0,
        onlineDevices: data.devices?.online || 0,
        devicesByTier: {
          power: data.devices?.byTier?.power || 0,
          standard: data.devices?.byTier?.standard || 0,
          crowd: data.devices?.byTier?.crowd || 0,
        },
        totalCompute: data.totalCompute || 0,
        pendingTasks: data.tasks?.pending || 0,
        processingTasks: data.tasks?.processing || 0,
        completedToday: data.tasks?.completed || 0,
        creditsToday: data.tasks?.completed || 0, // Assume 1 credit per task
      };

      setNetworkStats(stats);
      setIsConnected(true);
      setError(null);

      // Check for new completions
      const completed = stats.completedToday;
      if (previousCompletedRef.current > 0 && completed > previousCompletedRef.current) {
        const newCompletions = completed - previousCompletedRef.current;
        setRecentCompletion({
          count: newCompletions,
          totalToday: completed,
          creditsAwarded: newCompletions, // 1 credit per task
        });

        // Clear completion after 5 seconds
        setTimeout(() => {
          setRecentCompletion(null);
        }, 5000);
      }
      previousCompletedRef.current = completed;
    } catch (err) {
      console.error("Failed to fetch network stats:", err);
      setError(err instanceof Error ? err : new Error("Failed to fetch stats"));
      setIsConnected(false);
    }
  }, []);

  // Start polling on mount
  useEffect(() => {
    // Initial fetch
    fetchStats();

    // Set up polling interval
    intervalRef.current = setInterval(fetchStats, POLL_INTERVAL);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchStats]);

  return {
    isConnected,
    networkStats,
    recentCompletion,
    error,
    isSupported: true, // Polling is always supported
  };
}

/**
 * Format device tier breakdown for display
 */
export function formatTierBreakdown(
  devicesByTier: Record<DeviceTier, number>
): Array<{ tier: DeviceTier; count: number; label: string; color: string }> {
  return [
    {
      tier: "power" as const,
      count: devicesByTier.power,
      label: "Power",
      color: "bg-purple-500",
    },
    {
      tier: "standard" as const,
      count: devicesByTier.standard,
      label: "Standard",
      color: "bg-blue-500",
    },
    {
      tier: "crowd" as const,
      count: devicesByTier.crowd,
      label: "Crowd",
      color: "bg-green-500",
    },
  ];
}

/**
 * Calculate tasks per hour from today's completions
 */
export function calculateTasksPerHour(completedToday: number): number {
  const now = new Date();
  const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const hoursElapsed = (now.getTime() - startOfDay.getTime()) / (1000 * 60 * 60);

  if (hoursElapsed === 0) return 0;

  return Math.round((completedToday / hoursElapsed) * 10) / 10;
}

/**
 * Calculate network health score (0-100)
 */
export function calculateNetworkHealth(stats: NetworkStats | null): number {
  if (!stats) return 0;

  let score = 0;

  // Online devices contribute to health
  if (stats.onlineDevices > 0) score += 30;
  if (stats.onlineDevices > 5) score += 20;
  if (stats.onlineDevices > 10) score += 10;

  // Processing tasks shows active network
  if (stats.processingTasks > 0) score += 20;

  // Completed tasks show productivity
  if (stats.completedToday > 0) score += 10;
  if (stats.completedToday > 10) score += 10;

  return Math.min(100, score);
}
