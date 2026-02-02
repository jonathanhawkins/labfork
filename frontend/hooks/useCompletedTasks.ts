"use client";

import { useState, useEffect, useCallback } from "react";

// Workers API base URL
const WORKERS_API_BASE = process.env.NEXT_PUBLIC_WORKERS_API_URL ||
  "https://labfork-agents.jonathan-hawkins.workers.dev/api";

export interface CompletedTask {
  id: string;
  title: string;
  description: string | null;
  status: string;
  priority: number;
  assigned_agent: string | null;
  completed_at: string;
}

export interface ProjectTaskSummary {
  total: number;
  pending: number;
  in_progress: number;
  completed: number;
  blocked: number;
}

export interface ProjectDetails {
  id: string;
  name: string;
  slug: string;
  status: string;
  task_summary: ProjectTaskSummary;
  recent_completed_tasks: CompletedTask[];
  active_agents: Array<{
    id: string;
    name: string;
    status: string;
  }>;
}

/**
 * Hook to fetch completed tasks from the Firefly Network project
 */
export function useCompletedTasks(options?: {
  projectId?: string;
  pollInterval?: number;
}) {
  const projectId = options?.projectId || "firefly-network";
  const pollInterval = options?.pollInterval || 30000; // 30 seconds

  const [projectDetails, setProjectDetails] = useState<ProjectDetails | null>(null);
  const [completedTasks, setCompletedTasks] = useState<CompletedTask[]>([]);
  const [taskSummary, setTaskSummary] = useState<ProjectTaskSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchProjectDetails = useCallback(async () => {
    try {
      const response = await fetch(`${WORKERS_API_BASE}/projects/${projectId}`, {
        headers: {
          "Accept": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch project: ${response.status}`);
      }

      const data: ProjectDetails = await response.json();

      setProjectDetails(data);
      setCompletedTasks(data.recent_completed_tasks || []);
      setTaskSummary(data.task_summary);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      console.error("Error fetching project details:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch");
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  // Initial fetch
  useEffect(() => {
    fetchProjectDetails();
  }, [fetchProjectDetails]);

  // Polling
  useEffect(() => {
    if (pollInterval <= 0) return;

    const interval = setInterval(fetchProjectDetails, pollInterval);
    return () => clearInterval(interval);
  }, [fetchProjectDetails, pollInterval]);

  return {
    projectDetails,
    completedTasks,
    taskSummary,
    isLoading,
    error,
    lastUpdated,
    refresh: fetchProjectDetails,
  };
}

export default useCompletedTasks;
