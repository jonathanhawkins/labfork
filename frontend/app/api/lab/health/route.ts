import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { join } from "path";
import {
  readFileSync,
  existsSync,
  statSync,
  writeFileSync,
} from "fs";

export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;
const AGENT_STATE_URL = process.env.AGENT_STATE_URL || '';

// Paths
const projectRoot = join(process.cwd(), "..");
const RM_SCRIPT = join(projectRoot, ".skills", "research-manager", "rm");
const AGENTS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "agents.json"
);
const OUTPUTS_DIR = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "outputs"
);

interface AgentInfo {
  name: string;
  type: string;
  task: string;
  status: string;
  started_at: string;
  output_file: string;
  killed_at?: string;
}

interface AgentHealth {
  name: string;
  status: string;
  runningFor: number; // minutes
  lastActivity: number; // minutes ago
  logSize: number;
  isStuck: boolean;
  stuckReason?: string;
  hasErrors: boolean;
  errors: string[];
}

/**
 * Get all agents from state file
 */
function getAllAgents(): Record<string, AgentInfo> {
  if (!existsSync(AGENTS_FILE)) return {};
  try {
    return JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
  } catch {
    return {};
  }
}

/**
 * Save agents to state file
 */
function saveAgents(agents: Record<string, AgentInfo>) {
  writeFileSync(AGENTS_FILE, JSON.stringify(agents, null, 2));
}

/**
 * Check agent health based on log activity and patterns
 */
function checkAgentHealth(agent: AgentInfo): AgentHealth {
  const health: AgentHealth = {
    name: agent.name,
    status: agent.status,
    runningFor: 0,
    lastActivity: 0,
    logSize: 0,
    isStuck: false,
    hasErrors: false,
    errors: [],
  };

  // Calculate running time
  const startTime = new Date(agent.started_at).getTime();
  health.runningFor = Math.round((Date.now() - startTime) / 60000);

  // Check log file
  const logPath = agent.output_file;
  if (existsSync(logPath)) {
    const stats = statSync(logPath);
    health.logSize = stats.size;
    health.lastActivity = Math.round((Date.now() - stats.mtime.getTime()) / 60000);

    // Read last part of log for error detection
    try {
      const content = readFileSync(logPath, "utf-8");
      const lastChunk = content.slice(-10000);

      // Check for critical error patterns (not minor ones like "Exit code 1")
      const errorPatterns = [
        /CUDA out of memory/i,
        /RuntimeError/i,
        /PermissionError/i,
        /rate limit/i,
        /API error/i,
        /Connection refused/i,
        /Traceback.*most recent/i,
        /OutOfMemoryError/i,
        /torch\.cuda\.OutOfMemoryError/i,
      ];

      for (const pattern of errorPatterns) {
        const matches = lastChunk.match(new RegExp(pattern.source, "gi"));
        if (matches && matches.length > 0) {
          health.hasErrors = true;
          health.errors.push(`${pattern.source}: ${matches.length} occurrences`);
        }
      }

      // Check for stuck patterns
      if (agent.status === "running") {
        // Stuck if no log activity for 10+ minutes
        if (health.lastActivity > 10) {
          health.isStuck = true;
          health.stuckReason = `No log activity for ${health.lastActivity} minutes`;
        }

        // Stuck if running for 60+ minutes (most tasks shouldn't take that long)
        if (health.runningFor > 60 && health.lastActivity > 5) {
          health.isStuck = true;
          health.stuckReason = `Running for ${health.runningFor} minutes with minimal recent activity`;
        }

        // Check for repeated content (spinner stuck)
        const lines = lastChunk.split("\n").filter(l => l.trim().length > 0);
        const lastLines = lines.slice(-50);
        const uniqueLines = new Set(lastLines.map(l => l.substring(0, 50)));
        if (lastLines.length > 30 && uniqueLines.size < 5) {
          health.isStuck = true;
          health.stuckReason = "Repeated log output detected (possible spinner stuck)";
        }
      }
    } catch (e) {
      // Can't read log
    }
  } else {
    // No log file - might be stuck at startup
    if (agent.status === "running" && health.runningFor > 2) {
      health.isStuck = true;
      health.stuckReason = "No log file created after 2 minutes";
    }
  }

  return health;
}

/**
 * Kill an agent using the rm script
 */
async function killAgent(agentName: string): Promise<boolean> {
  return new Promise((resolve) => {
    const child = spawn(RM_SCRIPT, ["kill", "--name", agentName], {
      cwd: projectRoot,
      stdio: "pipe",
    });

    child.on("close", (code) => {
      resolve(code === 0);
    });

    child.on("error", () => resolve(false));
    setTimeout(() => resolve(false), 10000);
  });
}

/**
 * GET /api/lab/health
 * Returns health status of all agents
 */
/**
 * Fetch agents from remote agent state API (for Vercel)
 */
async function fetchRemoteAgents(): Promise<Record<string, AgentInfo> | null> {
  if (!AGENT_STATE_URL) return null;

  try {
    const response = await fetch(`${AGENT_STATE_URL}/agents`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      console.error('[Health] Remote fetch failed:', response.status);
      return null;
    }

    return await response.json();
  } catch (e) {
    console.error('[Health] Remote fetch error:', e);
    return null;
  }
}

/**
 * Build health response from agents data
 */
function buildHealthResponse(agents: Record<string, AgentInfo>) {
  const healthReports: AgentHealth[] = [];
  const stuckAgents: string[] = [];
  const errorAgents: string[] = [];

  for (const [name, agent] of Object.entries(agents)) {
    if (agent.status === "running") {
      // For remote agents, create basic health info
      const startTime = new Date(agent.started_at).getTime();
      const runningFor = Math.round((Date.now() - startTime) / 60000);

      const health: AgentHealth = {
        name,
        status: agent.status,
        runningFor,
        lastActivity: 0, // Can't determine from remote
        logSize: 0,
        isStuck: runningFor > 60, // Consider stuck after 1 hour
        stuckReason: runningFor > 60 ? `Running for ${runningFor} minutes` : undefined,
        hasErrors: false,
        errors: [],
      };

      healthReports.push(health);
      if (health.isStuck) stuckAgents.push(name);
    }
  }

  const killedAgents = Object.entries(agents)
    .filter(([_, a]) => a.status === "killed")
    .map(([name]) => name);

  return {
    healthy: stuckAgents.length === 0 && errorAgents.length === 0,
    runningCount: healthReports.length,
    stuckCount: stuckAgents.length,
    errorCount: errorAgents.length,
    killedCount: killedAgents.length,
    agents: healthReports,
    stuckAgents,
    errorAgents,
    killedAgents: killedAgents.slice(0, 10),
    recommendations: healthReports.length === 0
      ? ["No agents currently running. Use POST /api/lab/auto-spawn to start work."]
      : [],
  };
}

export async function GET(request: Request) {
  try {
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/health${url.search}`, {
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
        console.error("[Health] Backend fetch failed, falling back to local:", error);
      }
    }

    // Check if we're on Vercel and should use remote data
    const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;

    if (isVercel && AGENT_STATE_URL) {
      const remoteAgents = await fetchRemoteAgents();
      if (remoteAgents) {
        return NextResponse.json(buildHealthResponse(remoteAgents));
      }
      return NextResponse.json(buildHealthResponse({}));
    }

    const agents = getAllAgents();
    const healthReports: AgentHealth[] = [];
    const stuckAgents: string[] = [];
    const errorAgents: string[] = [];

    for (const [name, agent] of Object.entries(agents)) {
      if (agent.status === "running") {
        const health = checkAgentHealth(agent);
        healthReports.push(health);

        if (health.isStuck) {
          stuckAgents.push(name);
        }
        if (health.hasErrors) {
          errorAgents.push(name);
        }
      }
    }

    // Count killed agents for cleanup info
    const killedAgents = Object.entries(agents)
      .filter(([_, a]) => a.status === "killed")
      .map(([name]) => name);

    return NextResponse.json({
      healthy: stuckAgents.length === 0,
      runningCount: healthReports.length,
      stuckCount: stuckAgents.length,
      errorCount: errorAgents.length,
      killedCount: killedAgents.length,
      agents: healthReports,
      stuckAgents,
      errorAgents,
      killedAgents: killedAgents.slice(0, 10), // Only show first 10
      recommendations: generateRecommendations(healthReports, killedAgents),
    });
  } catch (error) {
    console.error("[Health] Error:", error);
    return NextResponse.json(
      { error: "Failed to check health" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/lab/health
 * Actions: kill-stuck, cleanup-killed, restart-agent
 */
export async function POST(request: Request) {
  try {
    const rawBody = await request.text();
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/health${url.search}`, {
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
        console.error("[Health] Backend update failed, falling back to local:", error);
      }
    }

    const body = rawBody ? JSON.parse(rawBody) : {};
    const { action, agentName } = body;

    const agents = getAllAgents();

    switch (action) {
      case "kill-stuck": {
        // Kill all stuck agents
        const killed: string[] = [];
        for (const [name, agent] of Object.entries(agents)) {
          if (agent.status === "running") {
            const health = checkAgentHealth(agent);
            if (health.isStuck) {
              const success = await killAgent(name);
              if (success) {
                killed.push(name);
                agents[name].status = "killed";
                agents[name].killed_at = new Date().toISOString();
              }
            }
          }
        }
        saveAgents(agents);
        return NextResponse.json({ success: true, killed });
      }

      case "kill-agent": {
        // Kill specific agent
        if (!agentName || !agents[agentName]) {
          return NextResponse.json({ error: "Agent not found" }, { status: 404 });
        }
        const success = await killAgent(agentName);
        if (success) {
          agents[agentName].status = "killed";
          agents[agentName].killed_at = new Date().toISOString();
          saveAgents(agents);
        }
        return NextResponse.json({ success });
      }

      case "cleanup-killed": {
        // Remove old killed agents (killed > 1 hour ago)
        const oneHourAgo = Date.now() - 60 * 60 * 1000;
        const removed: string[] = [];

        for (const [name, agent] of Object.entries(agents)) {
          if (agent.status === "killed" && agent.killed_at) {
            const killedTime = new Date(agent.killed_at).getTime();
            if (killedTime < oneHourAgo) {
              delete agents[name];
              removed.push(name);
            }
          }
        }

        saveAgents(agents);
        return NextResponse.json({ success: true, removed });
      }

      default:
        return NextResponse.json({ error: "Unknown action" }, { status: 400 });
    }
  } catch (error) {
    console.error("[Health] Error:", error);
    return NextResponse.json(
      { error: "Failed to perform action" },
      { status: 500 }
    );
  }
}

/**
 * Generate recommendations based on health state
 */
function generateRecommendations(
  healthReports: AgentHealth[],
  killedAgents: string[]
): string[] {
  const recommendations: string[] = [];

  // Stuck agents
  const stuckAgents = healthReports.filter(h => h.isStuck);
  if (stuckAgents.length > 0) {
    recommendations.push(
      `${stuckAgents.length} agent(s) appear stuck. Consider killing them with POST /api/lab/health { action: "kill-stuck" }`
    );
  }

  // Error agents
  const errorAgents = healthReports.filter(h => h.hasErrors);
  if (errorAgents.length > 0) {
    recommendations.push(
      `${errorAgents.length} agent(s) have errors in logs. Check their output for issues.`
    );
  }

  // Long-running agents
  const longRunning = healthReports.filter(h => h.runningFor > 30 && !h.isStuck);
  if (longRunning.length > 0) {
    recommendations.push(
      `${longRunning.length} agent(s) running for 30+ minutes. Monitor for completion.`
    );
  }

  // Cleanup killed
  if (killedAgents.length > 5) {
    recommendations.push(
      `${killedAgents.length} killed agents in state file. Run cleanup with POST /api/lab/health { action: "cleanup-killed" }`
    );
  }

  // No agents running
  if (healthReports.length === 0) {
    recommendations.push(
      "No agents currently running. Use POST /api/lab/auto-spawn to start work."
    );
  }

  return recommendations;
}
