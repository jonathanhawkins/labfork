import { NextResponse } from "next/server";
import { readFileSync, existsSync, statSync, writeFileSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
const AGENT_STATE_URL = process.env.AGENT_STATE_URL || '';

// Paths
const projectRoot = join(process.cwd(), "..");
const AGENTS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "agents.json"
);
const PROGRESS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "progress.json"
);

interface AgentProgress {
  name: string;
  taskId?: string;
  startedAt: string;
  lastActivity: number; // minutes ago

  // Progress indicators
  filesRead: number;
  filesWritten: number;
  filesEdited: number;
  toolCalls: number;
  tasksCreated: number;
  tasksCompleted: number;
  webSearches: number;
  errors: number;

  // Computed
  progressScore: number; // 0-100
  status: "active" | "slow" | "stuck" | "error" | "completed";
  statusReason: string;
}

interface ProgressHistory {
  agentName: string;
  taskId?: string;
  outcome: "completed" | "stuck" | "error" | "timeout";
  duration: number; // minutes
  progressScore: number;
  timestamp: string;
  retryCount: number;
}

/**
 * Strip ANSI codes from log content
 */
function stripAnsi(str: string): string {
  return str
    .replace(/\x1b\[[0-9;?]*[a-zA-Z]/g, "")
    .replace(/\u001b\[[0-9;?]*[a-zA-Z]/g, "")
    .replace(/[\x00-\x09\x0b-\x1f]/g, "");
}

/**
 * Analyze agent log to extract progress indicators
 */
function analyzeAgentProgress(logPath: string, agentName: string, startedAt: string): AgentProgress {
  const progress: AgentProgress = {
    name: agentName,
    startedAt,
    lastActivity: 0,
    filesRead: 0,
    filesWritten: 0,
    filesEdited: 0,
    toolCalls: 0,
    tasksCreated: 0,
    tasksCompleted: 0,
    webSearches: 0,
    errors: 0,
    progressScore: 0,
    status: "active",
    statusReason: "Running normally",
  };

  // Extract task ID from agent name
  const taskMatch = agentName.match(/^task-(\d+)-/);
  if (taskMatch) {
    progress.taskId = taskMatch[1];
  }

  if (!existsSync(logPath)) {
    progress.status = "error";
    progress.statusReason = "No log file found";
    return progress;
  }

  const stats = statSync(logPath);
  progress.lastActivity = Math.round((Date.now() - stats.mtime.getTime()) / 60000);

  try {
    const content = readFileSync(logPath, "utf-8");
    const cleanContent = stripAnsi(content);

    // Count tool usages
    progress.filesRead = (cleanContent.match(/Read\(/g) || []).length;
    progress.filesWritten = (cleanContent.match(/Write\(/g) || []).length;
    progress.filesEdited = (cleanContent.match(/Edit\(/g) || []).length;
    progress.webSearches = (cleanContent.match(/WebSearch\(/g) || []).length;
    progress.tasksCreated = (cleanContent.match(/TaskCreate/g) || []).length;
    progress.tasksCompleted = (cleanContent.match(/TaskUpdate.*completed/gi) || []).length;

    // Count total tool calls
    const toolPatterns = [
      /Read\(/g, /Write\(/g, /Edit\(/g, /Bash\(/g, /Grep\(/g, /Glob\(/g,
      /WebSearch\(/g, /WebFetch\(/g, /Task\(/g, /TaskCreate/g, /TaskUpdate/g,
    ];
    for (const pattern of toolPatterns) {
      progress.toolCalls += (cleanContent.match(pattern) || []).length;
    }

    // Count errors - be smarter about what counts as a real error
    // Minor errors (recoverable, don't affect status)
    const minorErrorPatterns = [
      /Exit code 1/gi,  // Common for failed commands, usually handled
      /error: Exit/gi,
      /No such file/gi,  // Often expected when checking if file exists
    ];
    let minorErrors = 0;
    for (const pattern of minorErrorPatterns) {
      minorErrors += (cleanContent.match(pattern) || []).length;
    }

    // Critical errors (should affect status)
    const criticalErrorPatterns = [
      /CUDA out of memory/gi,
      /RuntimeError/gi,
      /PermissionError/gi,
      /rate limit/gi,
      /API error/gi,
      /Connection refused/gi,
      /Traceback.*most recent/gi,
    ];
    let criticalErrors = 0;
    for (const pattern of criticalErrorPatterns) {
      criticalErrors += (cleanContent.match(pattern) || []).length;
    }

    progress.errors = criticalErrors; // Only count critical errors

    // Calculate progress score (0-100)
    // Weights: file ops = high, tool calls = medium, time = factor
    const fileScore = Math.min(40, (progress.filesWritten * 10) + (progress.filesEdited * 5) + (progress.filesRead * 2));
    const toolScore = Math.min(30, progress.toolCalls * 0.5);
    const taskScore = Math.min(20, (progress.tasksCreated * 5) + (progress.tasksCompleted * 10));
    const errorPenalty = Math.min(20, progress.errors * 2);

    progress.progressScore = Math.max(0, Math.min(100, fileScore + toolScore + taskScore - errorPenalty));

    // Determine status
    const runningMinutes = (Date.now() - new Date(startedAt).getTime()) / 60000;

    if (progress.errors >= 3) {
      progress.status = "error";
      progress.statusReason = `Critical errors detected (${progress.errors})`;
    } else if (progress.lastActivity > 10) {
      progress.status = "stuck";
      progress.statusReason = `No activity for ${progress.lastActivity} minutes`;
    } else if (runningMinutes > 30 && progress.progressScore < 20) {
      progress.status = "slow";
      progress.statusReason = `Low progress after ${Math.round(runningMinutes)} minutes`;
    } else if (progress.progressScore > 80) {
      progress.status = "active";
      progress.statusReason = "High progress, nearing completion";
    } else {
      progress.status = "active";
      progress.statusReason = `Progress: ${progress.progressScore}%, ${progress.toolCalls} tool calls`;
    }

  } catch (e) {
    progress.status = "error";
    progress.statusReason = "Failed to read log file";
  }

  return progress;
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
 * Save progress history
 */
function saveProgressHistory(history: ProgressHistory[]) {
  // Keep last 100 entries
  const trimmed = history.slice(-100);
  writeFileSync(PROGRESS_FILE, JSON.stringify({ history: trimmed }, null, 2));
}

/**
 * Record agent outcome
 */
function recordOutcome(
  agentName: string,
  taskId: string | undefined,
  outcome: ProgressHistory["outcome"],
  duration: number,
  progressScore: number
) {
  const history = getProgressHistory();

  // Count retries for this task
  const retryCount = taskId
    ? history.filter(h => h.taskId === taskId && h.outcome !== "completed").length
    : 0;

  history.push({
    agentName,
    taskId,
    outcome,
    duration,
    progressScore,
    timestamp: new Date().toISOString(),
    retryCount,
  });

  saveProgressHistory(history);
  return retryCount;
}

/**
 * Get retry recommendation for a task
 */
function getRetryRecommendation(taskId: string): { shouldRetry: boolean; reason: string; strategy?: string } {
  const history = getProgressHistory();
  const taskHistory = history.filter(h => h.taskId === taskId);

  if (taskHistory.length === 0) {
    return { shouldRetry: true, reason: "No history, first attempt" };
  }

  const failures = taskHistory.filter(h => h.outcome !== "completed");
  const successes = taskHistory.filter(h => h.outcome === "completed");

  if (successes.length > 0) {
    return { shouldRetry: false, reason: "Task already completed successfully" };
  }

  if (failures.length >= 3) {
    return {
      shouldRetry: false,
      reason: `Task failed ${failures.length} times, needs manual review`
    };
  }

  // Analyze failure patterns
  const stuckCount = failures.filter(h => h.outcome === "stuck").length;
  const errorCount = failures.filter(h => h.outcome === "error").length;
  const timeoutCount = failures.filter(h => h.outcome === "timeout").length;

  let strategy = "standard";
  if (stuckCount > 0) {
    strategy = "break_into_subtasks";
  } else if (errorCount > 0) {
    strategy = "add_error_handling";
  } else if (timeoutCount > 0) {
    strategy = "simplify_scope";
  }

  return {
    shouldRetry: true,
    reason: `Attempt ${failures.length + 1} of 3`,
    strategy,
  };
}

/**
 * Fetch data from remote agent state API (for Vercel)
 */
async function fetchRemoteProgressData(): Promise<{
  progress: { history: ProgressHistory[] };
  agents: Record<string, any>;
} | null> {
  if (!AGENT_STATE_URL) return null;

  try {
    const [progressRes, agentsRes] = await Promise.all([
      fetch(`${AGENT_STATE_URL}/progress`, {
        cache: 'no-store',
        signal: AbortSignal.timeout(5000),
      }),
      fetch(`${AGENT_STATE_URL}/agents`, {
        cache: 'no-store',
        signal: AbortSignal.timeout(5000),
      }),
    ]);

    if (!progressRes.ok || !agentsRes.ok) {
      console.error('[Progress] Remote fetch failed:', progressRes.status, agentsRes.status);
      return null;
    }

    const progress = await progressRes.json();
    const agents = await agentsRes.json();

    return { progress, agents };
  } catch (e) {
    console.error('[Progress] Remote fetch error:', e);
    return null;
  }
}

/**
 * Build progress response from provided data
 */
function buildProgressResponse(history: ProgressHistory[], agents: Record<string, any>) {
  const recentHistory = history.slice(-20);

  const stats = {
    totalAttempts: history.length,
    completed: history.filter(h => h.outcome === "completed").length,
    stuck: history.filter(h => h.outcome === "stuck").length,
    errors: history.filter(h => h.outcome === "error").length,
    timeouts: history.filter(h => h.outcome === "timeout").length,
    avgDuration: history.length > 0
      ? Math.round(history.reduce((sum, h) => sum + h.duration, 0) / history.length)
      : 0,
    avgProgressScore: history.length > 0
      ? Math.round(history.reduce((sum, h) => sum + h.progressScore, 0) / history.length)
      : 0,
  };

  // For remote agents, we can't analyze log files, so just return basic info
  const progressReports: AgentProgress[] = [];
  for (const [name, agent] of Object.entries(agents)) {
    if ((agent as any).status === "running") {
      const taskMatch = name.match(/^task-(\d+)-/);
      progressReports.push({
        name,
        taskId: taskMatch ? taskMatch[1] : undefined,
        startedAt: (agent as any).started_at,
        lastActivity: 0,
        filesRead: 0,
        filesWritten: 0,
        filesEdited: 0,
        toolCalls: 0,
        tasksCreated: 0,
        tasksCompleted: 0,
        webSearches: 0,
        errors: 0,
        progressScore: 50, // Default for remote agents
        status: "active",
        statusReason: "Running (remote)",
      });
    }
  }

  // Find tasks needing retry
  const pendingRetries: { taskId: string; recommendation: { shouldRetry: boolean; reason: string; strategy?: string } }[] = [];
  const failedTaskIds = new Set(
    history
      .filter(h => h.taskId && h.outcome !== "completed")
      .map(h => h.taskId!)
  );

  for (const taskId of Array.from(failedTaskIds)) {
    const taskHistory = history.filter(h => h.taskId === taskId);
    const failures = taskHistory.filter(h => h.outcome !== "completed");
    const successes = taskHistory.filter(h => h.outcome === "completed");

    if (successes.length === 0 && failures.length < 3) {
      pendingRetries.push({
        taskId,
        recommendation: {
          shouldRetry: true,
          reason: `Attempt ${failures.length + 1} of 3`,
        },
      });
    }
  }

  return {
    agents: progressReports,
    history: recentHistory,
    stats,
    pendingRetries,
  };
}

/**
 * GET /api/lab/progress
 * Returns progress for all running agents and history
 */
export async function GET(request: Request) {
  try {
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/progress${url.search}`, {
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
        console.error("[Progress] Backend fetch failed, falling back to local:", error);
      }
    }

    // Check if we're on Vercel and should use remote data
    const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;

    if (isVercel && AGENT_STATE_URL) {
      const remoteData = await fetchRemoteProgressData();
      if (remoteData) {
        const history = remoteData.progress?.history || [];
        return NextResponse.json(buildProgressResponse(history, remoteData.agents));
      }
      return NextResponse.json(buildProgressResponse([], {}));
    }

    // Get running agents
    let agents: Record<string, any> = {};
    if (existsSync(AGENTS_FILE)) {
      agents = JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
    }

    const progressReports: AgentProgress[] = [];

    for (const [name, agent] of Object.entries(agents)) {
      if ((agent as any).status === "running") {
        const progress = analyzeAgentProgress(
          (agent as any).output_file,
          name,
          (agent as any).started_at
        );
        progressReports.push(progress);
      }
    }

    // Get history stats
    const history = getProgressHistory();
    const recentHistory = history.slice(-20);

    const stats = {
      totalAttempts: history.length,
      completed: history.filter(h => h.outcome === "completed").length,
      stuck: history.filter(h => h.outcome === "stuck").length,
      errors: history.filter(h => h.outcome === "error").length,
      timeouts: history.filter(h => h.outcome === "timeout").length,
      avgDuration: history.length > 0
        ? Math.round(history.reduce((sum, h) => sum + h.duration, 0) / history.length)
        : 0,
      avgProgressScore: history.length > 0
        ? Math.round(history.reduce((sum, h) => sum + h.progressScore, 0) / history.length)
        : 0,
    };

    // Find tasks needing retry
    const pendingRetries: { taskId: string; recommendation: ReturnType<typeof getRetryRecommendation> }[] = [];
    const failedTaskIds = new Set(
      history
        .filter(h => h.taskId && h.outcome !== "completed")
        .map(h => h.taskId!)
    );

    for (const taskId of Array.from(failedTaskIds)) {
      const rec = getRetryRecommendation(taskId);
      if (rec.shouldRetry) {
        pendingRetries.push({ taskId, recommendation: rec });
      }
    }

    return NextResponse.json({
      agents: progressReports,
      history: recentHistory,
      stats,
      pendingRetries,
    });
  } catch (error) {
    console.error("[Progress] Error:", error);
    return NextResponse.json({ error: "Failed to get progress" }, { status: 500 });
  }
}

/**
 * POST /api/lab/progress
 * Record an outcome or get retry recommendation
 */
export async function POST(request: Request) {
  try {
    const rawBody = await request.text();
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/progress${url.search}`, {
          method: "POST",
          headers: {
            "Content-Type": request.headers.get("content-type") || "application/json",
          },
          body: rawBody || undefined,
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
        console.error("[Progress] Backend update failed, falling back to local:", error);
      }
    }

    const body = rawBody ? JSON.parse(rawBody) : {};
    const { action, agentName, taskId, outcome, duration, progressScore } = body;

    switch (action) {
      case "record": {
        const retryCount = recordOutcome(agentName, taskId, outcome, duration, progressScore);
        return NextResponse.json({ success: true, retryCount });
      }

      case "should-retry": {
        if (!taskId) {
          return NextResponse.json({ error: "taskId required" }, { status: 400 });
        }
        const recommendation = getRetryRecommendation(taskId);
        return NextResponse.json(recommendation);
      }

      default:
        return NextResponse.json({ error: "Unknown action" }, { status: 400 });
    }
  } catch (error) {
    console.error("[Progress] Error:", error);
    return NextResponse.json({ error: "Failed to process" }, { status: 500 });
  }
}
