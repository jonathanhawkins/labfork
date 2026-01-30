import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export const dynamic = "force-dynamic";

// Public agent state API (for Vercel deployment - can't read local files)
const AGENT_STATE_URL = process.env.AGENT_STATE_URL || '';

// Paths to research-manager state files (for local development)
const projectRoot = join(process.cwd(), "..");
const STATE_DIR = join(projectRoot, ".skills", "research-manager", "state");
const PROGRESS_FILE = join(STATE_DIR, "progress.json");
const AGENTS_FILE = join(STATE_DIR, "agents.json");

// Pricing configuration (per million tokens) - January 2026
// Sources:
// - OpenAI Codex: https://platform.openai.com/docs/pricing
// - Claude Opus 4.5: https://platform.claude.com/docs/en/about-claude/pricing
const PRICING = {
  ollama: {
    input: 0,      // FREE - local inference
    output: 0,     // FREE - local inference
    name: "Ollama (4090)",
    description: "Local qwen3-coder-32k",
  },
  codex: {
    input: 1.50,   // codex-mini-latest: $1.50/M input
    output: 6.00,  // codex-mini-latest: $6.00/M output
    name: "OpenAI Codex",
    description: "codex-mini-latest",
  },
  opus: {
    input: 5.00,   // Claude Opus 4.5: $5/M input
    output: 25.00, // Claude Opus 4.5: $25/M output
    name: "Claude Opus",
    description: "Claude Opus 4.5",
  },
};

// Estimated tokens per minute of agent runtime
// Based on typical Claude Code usage patterns
const TOKENS_PER_MINUTE = {
  ollama: { input: 2000, output: 800 },  // Local inference on 4090
};

interface ProgressEntry {
  agentName: string;
  taskId?: string;
  outcome: string;
  duration: number; // minutes
  progressScore: number;
  timestamp: string;
  retryCount?: number;
}

interface AgentEntry {
  name: string;
  type: string;
  status: string;
  task?: string;
  started_at: string;
}

interface TokenUsage {
  service: string;
  name: string;
  description: string;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
  sessions: number;
  totalMinutes: number;
}

interface HypotheticalCost {
  codex: number;  // What it would cost with Codex pricing
  opus: number;   // What it would cost with Opus 4.5 pricing
}

interface TokenUsageResponse {
  services: {
    ollama: TokenUsage;
  };
  totals: {
    totalTokens: number;
    totalCost: number;        // Always FREE since using Ollama
    hypotheticalCost: HypotheticalCost;  // What it WOULD have cost
  };
  timeRange: {
    start: string;
    end: string;
    days: number;
  };
  lastUpdated: string;
}

function readJSON<T>(filePath: string, defaultValue: T): T {
  try {
    if (existsSync(filePath)) {
      return JSON.parse(readFileSync(filePath, "utf-8"));
    }
  } catch (e) {
    console.error(`[TokenUsage] Failed to read ${filePath}:`, e);
  }
  return defaultValue;
}

function calculateTokens(
  durationMinutes: number
): { input: number; output: number } {
  // Use ollama rates for all token estimation
  const rates = TOKENS_PER_MINUTE.ollama;
  return {
    input: Math.round(durationMinutes * rates.input),
    output: Math.round(durationMinutes * rates.output),
  };
}

function calculateHypotheticalCost(
  inputTokens: number,
  outputTokens: number,
  service: "codex" | "opus"
): number {
  const pricing = PRICING[service];
  const inputCost = (inputTokens / 1_000_000) * pricing.input;
  const outputCost = (outputTokens / 1_000_000) * pricing.output;
  return inputCost + outputCost;
}

function calculateTokenUsage(): TokenUsageResponse {
  // Read progress history
  const progressData = readJSON<{ history: ProgressEntry[] }>(PROGRESS_FILE, { history: [] });
  const history = progressData.history || [];

  // Read running agents for real-time data
  const agentsData = readJSON<Record<string, AgentEntry>>(AGENTS_FILE, {});
  const runningAgents = Object.values(agentsData).filter(a => a.status === "running");

  // Initialize usage tracking - ALL usage goes through Ollama (FREE)
  const usage: TokenUsage = {
    service: "ollama",
    name: PRICING.ollama.name,
    description: PRICING.ollama.description,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    cost: 0,  // Always FREE
    sessions: 0,
    totalMinutes: 0,
  };

  // Process historical sessions
  let earliestTimestamp = new Date().toISOString();
  let latestTimestamp = new Date(0).toISOString();

  for (const entry of history) {
    const tokens = calculateTokens(entry.duration);

    usage.inputTokens += tokens.input;
    usage.outputTokens += tokens.output;
    usage.totalTokens += tokens.input + tokens.output;
    usage.sessions += 1;
    usage.totalMinutes += entry.duration;

    if (entry.timestamp < earliestTimestamp) {
      earliestTimestamp = entry.timestamp;
    }
    if (entry.timestamp > latestTimestamp) {
      latestTimestamp = entry.timestamp;
    }
  }

  // Add currently running agents (estimate based on elapsed time)
  const now = Date.now();
  for (const agent of runningAgents) {
    const startTime = new Date(agent.started_at).getTime();
    const elapsedMinutes = (now - startTime) / 60000;

    if (elapsedMinutes > 0 && elapsedMinutes < 120) { // Cap at 2 hours
      const tokens = calculateTokens(elapsedMinutes);

      usage.inputTokens += tokens.input;
      usage.outputTokens += tokens.output;
      usage.totalTokens += tokens.input + tokens.output;
      // Don't count as session (it's ongoing)
      usage.totalMinutes += elapsedMinutes;
    }
  }

  // Calculate hypothetical costs - what it WOULD have cost with paid services
  const hypotheticalCodexCost = calculateHypotheticalCost(
    usage.inputTokens,
    usage.outputTokens,
    "codex"
  );
  const hypotheticalOpusCost = calculateHypotheticalCost(
    usage.inputTokens,
    usage.outputTokens,
    "opus"
  );

  // Calculate time range
  const startDate = new Date(earliestTimestamp);
  const endDate = new Date(latestTimestamp);
  const days = Math.max(1, Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)));

  return {
    services: {
      ollama: usage,
    },
    totals: {
      totalTokens: usage.totalTokens,
      totalCost: 0, // Always FREE - using local Ollama
      hypotheticalCost: {
        codex: Math.round(hypotheticalCodexCost * 100) / 100,
        opus: Math.round(hypotheticalOpusCost * 100) / 100,
      },
    },
    timeRange: {
      start: earliestTimestamp,
      end: latestTimestamp,
      days,
    },
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * Fetch data from remote agent state API (for Vercel)
 */
async function fetchRemoteData(): Promise<{
  progress: { history: ProgressEntry[] };
  agents: Record<string, AgentEntry>;
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
      console.error('[TokenUsage] Remote fetch failed:', progressRes.status, agentsRes.status);
      return null;
    }

    const progress = await progressRes.json();
    const agents = await agentsRes.json();

    return { progress, agents };
  } catch (e) {
    console.error('[TokenUsage] Remote fetch error:', e);
    return null;
  }
}

/**
 * Calculate token usage from provided data (for remote or local)
 */
function calculateTokenUsageFromData(
  history: ProgressEntry[],
  agents: Record<string, AgentEntry>
): TokenUsageResponse {
  const runningAgents = Object.values(agents).filter(a => a.status === "running");

  // Initialize usage tracking - ALL usage goes through Ollama (FREE)
  const usage: TokenUsage = {
    service: "ollama",
    name: PRICING.ollama.name,
    description: PRICING.ollama.description,
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    cost: 0,  // Always FREE
    sessions: 0,
    totalMinutes: 0,
  };

  // Process historical sessions
  let earliestTimestamp = new Date().toISOString();
  let latestTimestamp = new Date(0).toISOString();

  for (const entry of history) {
    const tokens = calculateTokens(entry.duration);

    usage.inputTokens += tokens.input;
    usage.outputTokens += tokens.output;
    usage.totalTokens += tokens.input + tokens.output;
    usage.sessions += 1;
    usage.totalMinutes += entry.duration;

    if (entry.timestamp < earliestTimestamp) {
      earliestTimestamp = entry.timestamp;
    }
    if (entry.timestamp > latestTimestamp) {
      latestTimestamp = entry.timestamp;
    }
  }

  // Add currently running agents (estimate based on elapsed time)
  const now = Date.now();
  for (const agent of runningAgents) {
    const startTime = new Date(agent.started_at).getTime();
    const elapsedMinutes = (now - startTime) / 60000;

    if (elapsedMinutes > 0 && elapsedMinutes < 120) { // Cap at 2 hours
      const tokens = calculateTokens(elapsedMinutes);

      usage.inputTokens += tokens.input;
      usage.outputTokens += tokens.output;
      usage.totalTokens += tokens.input + tokens.output;
      // Don't count as session (it's ongoing)
      usage.totalMinutes += elapsedMinutes;
    }
  }

  // Calculate hypothetical costs - what it WOULD have cost with paid services
  const hypotheticalCodexCost = calculateHypotheticalCost(
    usage.inputTokens,
    usage.outputTokens,
    "codex"
  );
  const hypotheticalOpusCost = calculateHypotheticalCost(
    usage.inputTokens,
    usage.outputTokens,
    "opus"
  );

  // Calculate time range
  const startDate = new Date(earliestTimestamp);
  const endDate = new Date(latestTimestamp);
  const days = Math.max(1, Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)));

  return {
    services: {
      ollama: usage,
    },
    totals: {
      totalTokens: usage.totalTokens,
      totalCost: 0, // Always FREE - using local Ollama
      hypotheticalCost: {
        codex: Math.round(hypotheticalCodexCost * 100) / 100,
        opus: Math.round(hypotheticalOpusCost * 100) / 100,
      },
    },
    timeRange: {
      start: earliestTimestamp,
      end: latestTimestamp,
      days,
    },
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * GET /api/lab/token-usage
 * Returns token usage statistics for all AI services
 */
export async function GET() {
  try {
    // Check if we're on Vercel and should use remote data
    const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;

    if (isVercel && AGENT_STATE_URL) {
      // Try to fetch from remote agent state API
      const remoteData = await fetchRemoteData();
      if (remoteData) {
        const history = remoteData.progress?.history || [];
        const usage = calculateTokenUsageFromData(history, remoteData.agents);
        return NextResponse.json(usage);
      }
      // Fall through to return empty data if remote fetch failed
      return NextResponse.json(calculateTokenUsageFromData([], {}));
    }

    // Local development - read from files
    const usage = calculateTokenUsage();
    return NextResponse.json(usage);
  } catch (error) {
    console.error("[TokenUsage] Error:", error);
    return NextResponse.json(
      { error: "Failed to get token usage" },
      { status: 500 }
    );
  }
}
