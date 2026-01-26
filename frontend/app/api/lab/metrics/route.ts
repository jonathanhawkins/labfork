import { NextResponse } from "next/server";
import { readFileSync, existsSync, writeFileSync, statSync } from "fs";
import { join } from "path";

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;

// Paths
const projectRoot = join(process.cwd(), "..");
const PROGRESS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "progress.json"
);
const METRICS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "metrics.json"
);
const AGENTS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "agents.json"
);

interface ProgressHistory {
  agentName: string;
  taskId?: string;
  outcome: "completed" | "stuck" | "error" | "timeout";
  duration: number;
  progressScore: number;
  timestamp: string;
  retryCount: number;
}

interface DailyMetrics {
  date: string;
  tasksCompleted: number;
  tasksFailed: number;
  totalDuration: number;
  avgDuration: number;
  avgProgressScore: number;
  researchAgents: number;
  taskAgents: number;
  costEstimate: number; // rough estimate based on runtime
}

interface Metrics {
  // Overall stats
  totalTasksCompleted: number;
  totalTasksFailed: number;
  successRate: number;
  avgCompletionTime: number;
  avgProgressScore: number;

  // Recent activity (last 24 hours)
  last24h: {
    completed: number;
    failed: number;
    avgDuration: number;
  };

  // By outcome
  byOutcome: {
    completed: number;
    stuck: number;
    error: number;
    timeout: number;
  };

  // Current state
  currentAgents: number;
  currentPendingTasks: number;

  // Daily history (last 7 days)
  dailyHistory: DailyMetrics[];

  // Task type breakdown
  byTaskType: {
    research: { count: number; avgDuration: number; successRate: number };
    task: { count: number; avgDuration: number; successRate: number };
  };

  // Cost estimate
  estimatedCostToday: number;
  estimatedCostTotal: number;

  // Uptime
  orchestratorUptime: number | null;
}

/**
 * Get progress history
 */
function getProgressHistory(): ProgressHistory[] {
  try {
    if (existsSync(PROGRESS_FILE)) {
      return JSON.parse(readFileSync(PROGRESS_FILE, "utf-8")).history || [];
    }
  } catch {}
  return [];
}

/**
 * Get running agents count
 */
function getRunningAgentsCount(): number {
  try {
    if (existsSync(AGENTS_FILE)) {
      const agents = JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
      return Object.values(agents).filter((a: any) => a.status === "running").length;
    }
  } catch {}
  return 0;
}

/**
 * Estimate cost based on runtime (rough estimate)
 * Opus: ~$15/hour, Sonnet: ~$3/hour, Haiku: ~$0.25/hour
 * Using average of $10/hour for opus agents
 */
function estimateCost(durationMinutes: number, agentType: string = "opus"): number {
  const hourlyRates: Record<string, number> = {
    opus: 10,
    sonnet: 3,
    haiku: 0.25,
  };
  const rate = hourlyRates[agentType] || hourlyRates.opus;
  return (durationMinutes / 60) * rate;
}

/**
 * Calculate metrics from progress history
 */
function calculateMetrics(): Metrics {
  const history = getProgressHistory();
  const now = Date.now();
  const oneDayAgo = now - 24 * 60 * 60 * 1000;
  const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;

  // Filter for different time periods
  const last24hHistory = history.filter(
    (h) => new Date(h.timestamp).getTime() > oneDayAgo
  );
  const last7dHistory = history.filter(
    (h) => new Date(h.timestamp).getTime() > sevenDaysAgo
  );

  // Overall stats
  const completed = history.filter((h) => h.outcome === "completed");
  const failed = history.filter((h) => h.outcome !== "completed");

  const totalTasksCompleted = completed.length;
  const totalTasksFailed = failed.length;
  const successRate =
    history.length > 0 ? (totalTasksCompleted / history.length) * 100 : 0;
  const avgCompletionTime =
    completed.length > 0
      ? completed.reduce((sum, h) => sum + h.duration, 0) / completed.length
      : 0;
  const avgProgressScore =
    history.length > 0
      ? history.reduce((sum, h) => sum + h.progressScore, 0) / history.length
      : 0;

  // Last 24h
  const last24hCompleted = last24hHistory.filter((h) => h.outcome === "completed");
  const last24hFailed = last24hHistory.filter((h) => h.outcome !== "completed");

  // By outcome
  const byOutcome = {
    completed: history.filter((h) => h.outcome === "completed").length,
    stuck: history.filter((h) => h.outcome === "stuck").length,
    error: history.filter((h) => h.outcome === "error").length,
    timeout: history.filter((h) => h.outcome === "timeout").length,
  };

  // By task type
  const researchHistory = history.filter((h) => h.agentName.includes("researcher"));
  const taskHistory = history.filter((h) => h.agentName.includes("task-"));

  const byTaskType = {
    research: {
      count: researchHistory.length,
      avgDuration:
        researchHistory.length > 0
          ? researchHistory.reduce((sum, h) => sum + h.duration, 0) /
            researchHistory.length
          : 0,
      successRate:
        researchHistory.length > 0
          ? (researchHistory.filter((h) => h.outcome === "completed").length /
              researchHistory.length) *
            100
          : 0,
    },
    task: {
      count: taskHistory.length,
      avgDuration:
        taskHistory.length > 0
          ? taskHistory.reduce((sum, h) => sum + h.duration, 0) / taskHistory.length
          : 0,
      successRate:
        taskHistory.length > 0
          ? (taskHistory.filter((h) => h.outcome === "completed").length /
              taskHistory.length) *
            100
          : 0,
    },
  };

  // Daily history
  const dailyMap = new Map<string, DailyMetrics>();
  for (let i = 0; i < 7; i++) {
    const date = new Date(now - i * 24 * 60 * 60 * 1000)
      .toISOString()
      .split("T")[0];
    dailyMap.set(date, {
      date,
      tasksCompleted: 0,
      tasksFailed: 0,
      totalDuration: 0,
      avgDuration: 0,
      avgProgressScore: 0,
      researchAgents: 0,
      taskAgents: 0,
      costEstimate: 0,
    });
  }

  for (const h of last7dHistory) {
    const date = h.timestamp.split("T")[0];
    const daily = dailyMap.get(date);
    if (daily) {
      if (h.outcome === "completed") {
        daily.tasksCompleted++;
      } else {
        daily.tasksFailed++;
      }
      daily.totalDuration += h.duration;
      daily.costEstimate += estimateCost(h.duration);
      if (h.agentName.includes("researcher")) {
        daily.researchAgents++;
      } else {
        daily.taskAgents++;
      }
    }
  }

  // Calculate averages for daily
  for (const daily of Array.from(dailyMap.values())) {
    const total = daily.tasksCompleted + daily.tasksFailed;
    if (total > 0) {
      daily.avgDuration = daily.totalDuration / total;
    }
  }

  const dailyHistory = Array.from(dailyMap.values()).sort(
    (a, b) => a.date.localeCompare(b.date)
  );

  // Cost estimates
  const todayDate = new Date().toISOString().split("T")[0];
  const todayMetrics = dailyMap.get(todayDate);
  const estimatedCostToday = todayMetrics?.costEstimate || 0;
  const estimatedCostTotal = history.reduce(
    (sum, h) => sum + estimateCost(h.duration),
    0
  );

  // Orchestrator uptime
  let orchestratorUptime: number | null = null;
  try {
    const pidFile = join(
      projectRoot,
      ".skills",
      "research-manager",
      "state",
      "orchestrator.pid"
    );
    if (existsSync(pidFile)) {
      const stats = statSync(pidFile);
      orchestratorUptime = (now - stats.mtime.getTime()) / 60000; // minutes
    }
  } catch {}

  return {
    totalTasksCompleted,
    totalTasksFailed,
    successRate: Math.round(successRate * 10) / 10,
    avgCompletionTime: Math.round(avgCompletionTime * 10) / 10,
    avgProgressScore: Math.round(avgProgressScore),

    last24h: {
      completed: last24hCompleted.length,
      failed: last24hFailed.length,
      avgDuration:
        last24hHistory.length > 0
          ? Math.round(
              (last24hHistory.reduce((sum, h) => sum + h.duration, 0) /
                last24hHistory.length) *
                10
            ) / 10
          : 0,
    },

    byOutcome,

    currentAgents: getRunningAgentsCount(),
    currentPendingTasks: 0, // Will be filled by caller if needed

    dailyHistory,
    byTaskType,

    estimatedCostToday: Math.round(estimatedCostToday * 100) / 100,
    estimatedCostTotal: Math.round(estimatedCostTotal * 100) / 100,

    orchestratorUptime:
      orchestratorUptime !== null ? Math.round(orchestratorUptime) : null,
  };
}

/**
 * GET /api/lab/metrics
 * Returns orchestrator metrics
 */
export async function GET(request: Request) {
  try {
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/metrics${url.search}`, {
          cache: "no-store",
        });
        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }
        const text = await response.text();
        return NextResponse.json(
          { error: text || "Backend error" },
          { status: response.status }
        );
      } catch (error) {
        console.error("[Metrics] Backend fetch failed, falling back to local:", error);
      }
    }

    const metrics = calculateMetrics();

    return NextResponse.json(metrics);
  } catch (error) {
    console.error("[Metrics] Error:", error);
    return NextResponse.json({ error: "Failed to get metrics" }, { status: 500 });
  }
}
