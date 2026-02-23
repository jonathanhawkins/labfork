/**
 * Contributor Hooks
 *
 * React hooks for managing contributor profiles and statistics.
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import type { ContributorProfile } from "./user-types";

interface UseContributorReturn {
  profile: ContributorProfile | null;
  isLoading: boolean;
  error: Error | null;
  refresh: () => void;
}

/**
 * Get contributor profile by user ID
 */
export function useContributor(userId?: string): UseContributorReturn {
  const [profile, setProfile] = useState<ContributorProfile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchProfile = useCallback(async () => {
    if (!userId) {
      setProfile(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/contributor/${userId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch profile: ${response.statusText}`);
      }

      const data = await response.json();
      setProfile(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
      setProfile(null);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  return {
    profile,
    isLoading,
    error,
    refresh: fetchProfile,
  };
}

interface UseLeaderboardReturn {
  leaders: ContributorProfile[];
  isLoading: boolean;
  error: Error | null;
  refresh: () => void;
}

/**
 * Get top contributors leaderboard
 */
export function useLeaderboard(limit = 10): UseLeaderboardReturn {
  const [leaders, setLeaders] = useState<ContributorProfile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchLeaderboard = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/contributor/leaderboard?limit=${limit}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch leaderboard: ${response.statusText}`);
      }

      const data = await response.json();
      setLeaders(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
      setLeaders([]);
    } finally {
      setIsLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchLeaderboard();
  }, [fetchLeaderboard]);

  return {
    leaders,
    isLoading,
    error,
    refresh: fetchLeaderboard,
  };
}

interface UseContributionStatsReturn {
  totalContributors: number;
  totalCreditsEarned: number;
  totalTasksCompleted: number;
  isLoading: boolean;
  error: Error | null;
  refresh: () => void;
}

/**
 * Get global contribution statistics
 */
export function useContributionStats(): UseContributionStatsReturn {
  const [stats, setStats] = useState({
    totalContributors: 0,
    totalCreditsEarned: 0,
    totalTasksCompleted: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchStats = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/contributor/stats");
      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.statusText}`);
      }

      const data = await response.json();
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
      setStats({
        totalContributors: 0,
        totalCreditsEarned: 0,
        totalTasksCompleted: 0,
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  return {
    ...stats,
    isLoading,
    error,
    refresh: fetchStats,
  };
}
