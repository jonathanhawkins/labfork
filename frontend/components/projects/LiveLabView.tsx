/**
 * LiveLabView - Real-time display of Firefly Lab #1 activity
 *
 * Shows:
 * - Active agents and their current tasks
 * - Recent activities
 * - Published results with comments
 * - Task progress
 */

"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

// Types matching the seed data
interface LabAgent {
  id: string;
  name: string;
  displayName: string;
  type: "ollama" | "codex" | "claude";
  model: string;
  status: "idle" | "working" | "thinking" | "paused";
  currentTask?: string;
  progress: number;
  tokensGenerated: number;
  costEstimate: number;
  lastActivityAt: string;
  color: number;
}

interface LabTask {
  id: string;
  subject: string;
  type: string;
  status: "pending" | "in_progress" | "completed" | "blocked";
  priority: string;
  progress: number;
  assignedAgent?: string;
}

interface LabPaper {
  id: string;
  title: string;
  status: "to-read" | "reading" | "implementing" | "implemented";
  progress: number;
  assignedAgent?: string;
}

interface LabActivity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
}

interface LabResult {
  id: string;
  title: string;
  type: string;
  description: string;
  likes: number;
  comments: { id: string; username: string; content: string }[];
  createdAt: string;
}

interface FireflyLabData {
  lab?: {
    name: string;
    stats: {
      stars: number;
      forks: number;
      views: number;
    };
  };
  agents?: LabAgent[];
  tasks?: LabTask[];
  papers?: LabPaper[];
  activities?: LabActivity[];
  results?: LabResult[];
  paperStats?: {
    total: number;
    implemented: number;
    implementing: number;
    reading: number;
  };
  taskStats?: {
    total: number;
    completed: number;
    inProgress: number;
    pending: number;
  };
  agentStats?: {
    total: number;
    working: number;
    totalCost: number;
    totalTokens: number;
  };
}

export function LiveLabView({ className = "" }: { className?: string }) {
  const [data, setData] = useState<FireflyLabData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [seeding, setSeeding] = useState(false);

  // Fetch lab data
  const fetchData = async () => {
    try {
      const response = await fetch("/api/labs/firefly");
      if (!response.ok) throw new Error("Failed to fetch");
      const json = await response.json();
      if (json.success) {
        setData(json);
        setError(null);
      }
    } catch (err) {
      console.error("Error fetching Firefly data:", err);
      setError("Could not load lab data");
    } finally {
      setLoading(false);
    }
  };

  // Seed the database
  const seedDatabase = async () => {
    setSeeding(true);
    try {
      const response = await fetch("/api/labs/firefly", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false }),
      });
      if (response.ok) {
        await fetchData();
      }
    } catch (err) {
      console.error("Error seeding:", err);
    } finally {
      setSeeding(false);
    }
  };

  useEffect(() => {
    fetchData();
    // Poll for updates every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto-seed on first load if no data
  useEffect(() => {
    if (!loading && !data?.lab && !seeding) {
      seedDatabase();
    }
  }, [loading, data, seeding]);

  if (loading) {
    return (
      <div className={`${className} animate-pulse`}>
        <div className="h-64 bg-white/5 rounded-2xl" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className={`${className} p-8 text-center`}>
        <p className="text-gray-400 mb-4">Lab data not available</p>
        <button
          onClick={seedDatabase}
          disabled={seeding}
          className="px-4 py-2 bg-amber-500/20 text-amber-400 rounded-lg hover:bg-amber-500/30 disabled:opacity-50"
        >
          {seeding ? "Initializing Lab..." : "Initialize Firefly Lab"}
        </button>
      </div>
    );
  }

  const { agents, tasks, papers, activities, results, taskStats, paperStats, agentStats } = data;

  return (
    <div className={`${className} space-y-8`}>
      {/* Lab Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Active Agents", value: agentStats?.working || 0, color: "amber" },
          { label: "Tasks Completed", value: taskStats?.completed || 0, color: "green" },
          { label: "Papers Implemented", value: paperStats?.implemented || 0, color: "blue" },
          { label: "In Progress", value: taskStats?.inProgress || 0, color: "purple" },
        ].map((stat) => (
          <div
            key={stat.label}
            className={`p-4 rounded-xl bg-${stat.color}-500/10 border border-${stat.color}-500/20`}
          >
            <div className={`text-2xl font-bold text-${stat.color}-400`}>
              {stat.value}
            </div>
            <div className="text-sm text-gray-400">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Active Agents */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          Active Agents
        </h3>
        <div className="grid md:grid-cols-3 gap-4">
          {(agents || [])
            .filter((a) => a.status === "working")
            .map((agent) => (
              <div
                key={agent.id}
                className="p-4 rounded-xl border border-white/10 bg-white/5"
                style={{ borderLeftColor: `#${agent.color.toString(16).padStart(6, "0")}`, borderLeftWidth: "4px" }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-white">{agent.name}</span>
                  <span className="text-xs px-2 py-1 rounded-full bg-green-500/20 text-green-400">
                    {agent.status}
                  </span>
                </div>
                <p className="text-sm text-gray-400 mb-3 line-clamp-2">
                  {agent.currentTask}
                </p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-500"
                      style={{ width: `${agent.progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500">{agent.progress}%</span>
                </div>
                <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                  <span>{agent.model}</span>
                  <span>
                    {agent.costEstimate > 0 ? `$${agent.costEstimate.toFixed(2)}` : "FREE"}
                  </span>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Recent Tasks */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Task Progress</h3>
        <div className="space-y-3">
          {(tasks || [])
            .filter((t) => t.status === "in_progress" || t.status === "completed")
            .slice(0, 5)
            .map((task) => (
              <div
                key={task.id}
                className="p-4 rounded-xl border border-white/10 bg-white/5 flex items-center gap-4"
              >
                <div
                  className={`w-3 h-3 rounded-full ${
                    task.status === "completed"
                      ? "bg-green-500"
                      : "bg-amber-500 animate-pulse"
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-white font-medium truncate">
                      {task.subject}
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        task.priority === "critical"
                          ? "bg-red-500/20 text-red-400"
                          : task.priority === "high"
                          ? "bg-orange-500/20 text-orange-400"
                          : "bg-gray-500/20 text-gray-400"
                      }`}
                    >
                      {task.priority}
                    </span>
                  </div>
                  {task.assignedAgent && (
                    <span className="text-xs text-gray-500">
                      Agent: {task.assignedAgent}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        task.status === "completed"
                          ? "bg-green-500"
                          : "bg-amber-500"
                      }`}
                      style={{ width: `${task.progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 w-8">
                    {task.progress}%
                  </span>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Recent Results with Comments */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Published Results</h3>
        <div className="space-y-4">
          {(results || []).slice(0, 3).map((result) => (
            <div
              key={result.id}
              className="p-5 rounded-xl border border-white/10 bg-white/5"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <span
                    className={`text-xs px-2 py-1 rounded-full mr-2 ${
                      result.type === "paper"
                        ? "bg-blue-500/20 text-blue-400"
                        : result.type === "code"
                        ? "bg-green-500/20 text-green-400"
                        : "bg-purple-500/20 text-purple-400"
                    }`}
                  >
                    {result.type}
                  </span>
                  <span className="text-white font-semibold">{result.title}</span>
                </div>
                <div className="flex items-center gap-3 text-sm text-gray-500">
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" />
                    </svg>
                    {result.likes}
                  </span>
                  <span className="flex items-center gap-1">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7z" />
                    </svg>
                    {result.comments.length}
                  </span>
                </div>
              </div>
              <p className="text-gray-400 text-sm mb-4">{result.description}</p>

              {/* Top comment */}
              {result.comments.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/5">
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-xs font-bold text-white">
                      {result.comments[0].username[0].toUpperCase()}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium text-white">
                          @{result.comments[0].username}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">
                        {result.comments[0].content}
                      </p>
                    </div>
                  </div>
                  {result.comments.length > 1 && (
                    <button className="mt-3 text-sm text-amber-400 hover:text-amber-300">
                      View {result.comments.length - 1} more comments
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4">Activity Feed</h3>
        <div className="space-y-2">
          {(activities || []).slice(0, 6).map((activity, i) => (
            <div
              key={activity.id}
              className="flex items-center gap-3 p-3 rounded-lg hover:bg-white/5 transition-colors"
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  activity.type === "task_completed"
                    ? "bg-green-500"
                    : activity.type === "result_posted"
                    ? "bg-blue-500"
                    : activity.type === "star"
                    ? "bg-yellow-500"
                    : activity.type === "fork"
                    ? "bg-purple-500"
                    : "bg-gray-500"
                }`}
              />
              <span className="flex-1 text-sm text-gray-300">
                {activity.description}
              </span>
              <span className="text-xs text-gray-500">
                {formatTimeAgo(activity.timestamp)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* CTA to full lab view */}
      <div className="text-center pt-4">
        <Link
          href="/lab?id=lab_firefly001"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors"
        >
          <span>Open Full Lab View</span>
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </Link>
      </div>
    </div>
  );
}

function formatTimeAgo(timestamp: string): string {
  const now = new Date();
  const then = new Date(timestamp);
  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return then.toLocaleDateString();
}

export default LiveLabView;
