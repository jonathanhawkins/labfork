/**
 * Research Manager Lab Detail API
 *
 * GET /api/labs/research/[id] - Get detailed research state for a lab
 *
 * Returns:
 * - Lab config
 * - Active agents with current tasks
 * - Proposals with approval status
 * - Task list from Claude Code
 * - Research outputs from docs/<domain>/
 * - Progress history
 */

import { NextRequest, NextResponse } from "next/server";
import * as fs from "fs/promises";
import * as path from "path";

// Research manager paths - go up one level from frontend to project root
const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const RESEARCH_MANAGER_DIR = path.join(
  PROJECT_ROOT,
  ".skills",
  "research-manager"
);
const LABS_DIR = path.join(RESEARCH_MANAGER_DIR, "labs");
const DOCS_DIR = path.join(PROJECT_ROOT, "docs");
const CLAUDE_TASKS_DIR = path.join(
  process.env.HOME || "/Users/light",
  ".claude",
  "tasks"
);

interface RouteParams {
  params: Promise<{ id: string }>;
}

interface LabConfig {
  id: string;
  name: string;
  description: string;
  taskListId: string;
  active: boolean;
  domain?: string;
  settings?: {
    maxAgents?: number;
    autoSpawn?: boolean;
    researchInterval?: number;
  };
  createdAt: string;
  updatedAt: string;
}

interface Agent {
  id?: string;
  name?: string;
  displayName?: string;
  type?: string;
  model?: string;
  status: string;
  currentTask?: string;
  currentTaskId?: string;
  progress?: number;
  tokensGenerated?: number;
  costEstimate?: number;
  startedAt?: string;
  lastActivityAt?: string;
  pid?: number;
}

interface AgentState {
  [agentId: string]: Agent;
}

interface Proposal {
  id: string;
  title?: string;
  storyId?: string;
  status: string;
  summary?: string;
  recommendation?: string;
  filePath?: string;
  createdAt?: string;
  updatedAt?: string;
}

interface ProposalsState {
  proposals: { [id: string]: Proposal };
}

interface ProgressEntry {
  timestamp: string;
  type: string;
  taskId?: string;
  subject?: string;
  agent?: string;
  output?: string;
}

interface ProgressState {
  history: ProgressEntry[];
}

interface ClaudeTask {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  priority?: string;
  owner?: string;
  activeForm?: string;
  blockedBy?: string[];
  blocks?: string[];
  createdAt?: string;
  updatedAt?: string;
  completedAt?: string;
}

interface ResearchDocument {
  name: string;
  path: string;
  content: string;
  size: number;
  modified: string;
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFile<T>(filePath: string, defaultValue: T): Promise<T> {
  try {
    const content = await fs.readFile(filePath, "utf-8");
    return JSON.parse(content) as T;
  } catch {
    return defaultValue;
  }
}

async function getLabConfig(labId: string): Promise<LabConfig | null> {
  const configPath = path.join(LABS_DIR, labId, "config.json");
  if (!(await fileExists(configPath))) {
    return null;
  }
  return readJsonFile<LabConfig | null>(configPath, null);
}

async function getLabAgents(labId: string): Promise<Agent[]> {
  const agentsPath = path.join(LABS_DIR, labId, "state", "agents.json");
  const agents = await readJsonFile<AgentState>(agentsPath, {});
  return Object.entries(agents).map(([id, agent]) => ({
    id,
    ...agent,
  }));
}

async function getLabProposals(labId: string): Promise<Proposal[]> {
  const proposalsPath = path.join(LABS_DIR, labId, "state", "proposals.json");
  const state = await readJsonFile<ProposalsState>(proposalsPath, {
    proposals: {},
  });
  return Object.entries(state.proposals || {}).map(([id, proposal]) => ({
    id,
    ...proposal,
  }));
}

async function getLabProgress(labId: string): Promise<ProgressEntry[]> {
  const progressPath = path.join(LABS_DIR, labId, "state", "progress.json");
  const state = await readJsonFile<ProgressState>(progressPath, { history: [] });
  return state.history || [];
}

async function getClaudeTasks(taskListId: string): Promise<ClaudeTask[]> {
  const tasksDir = path.join(CLAUDE_TASKS_DIR, taskListId);
  if (!(await fileExists(tasksDir))) {
    return [];
  }

  try {
    const files = await fs.readdir(tasksDir);
    const tasks: ClaudeTask[] = [];

    for (const file of files) {
      if (file.endsWith(".json") && !file.startsWith(".")) {
        const taskPath = path.join(tasksDir, file);
        const task = await readJsonFile<ClaudeTask | null>(taskPath, null);
        if (task && task.id) {
          tasks.push(task);
        }
      }
    }

    // Sort by status (in_progress first, then pending, then completed)
    const statusOrder: Record<string, number> = {
      in_progress: 0,
      pending: 1,
      completed: 2,
    };
    tasks.sort(
      (a, b) =>
        (statusOrder[a.status] ?? 3) - (statusOrder[b.status] ?? 3)
    );

    return tasks;
  } catch {
    return [];
  }
}

async function getResearchDocuments(
  domain: string
): Promise<ResearchDocument[]> {
  // Check multiple possible locations for research docs
  const possiblePaths = [
    path.join(DOCS_DIR, domain),
    path.join(DOCS_DIR, `${domain}-case-study`),
    path.join(DOCS_DIR, domain.replace(/-/g, "_")),
  ];

  const documents: ResearchDocument[] = [];

  for (const docsPath of possiblePaths) {
    if (await fileExists(docsPath)) {
      try {
        const files = await fs.readdir(docsPath, { withFileTypes: true });
        for (const file of files) {
          if (file.isFile() && file.name.endsWith(".md")) {
            const filePath = path.join(docsPath, file.name);
            const stat = await fs.stat(filePath);
            const content = await fs.readFile(filePath, "utf-8");
            documents.push({
              name: file.name,
              path: filePath,
              content,
              size: stat.size,
              modified: stat.mtime.toISOString(),
            });
          }
        }
      } catch {
        // Continue to next path
      }
    }
  }

  // Also check for PROPOSAL.md in the domain folder
  const proposalPath = path.join(DOCS_DIR, domain, "PROPOSAL.md");
  if (await fileExists(proposalPath)) {
    try {
      const stat = await fs.stat(proposalPath);
      const content = await fs.readFile(proposalPath, "utf-8");
      // Only add if not already in documents
      if (!documents.find((d) => d.name === "PROPOSAL.md")) {
        documents.push({
          name: "PROPOSAL.md",
          path: proposalPath,
          content,
          size: stat.size,
          modified: stat.mtime.toISOString(),
        });
      }
    } catch {
      // Skip
    }
  }

  // Sort by modified date (most recent first)
  documents.sort(
    (a, b) =>
      new Date(b.modified).getTime() - new Date(a.modified).getTime()
  );

  return documents;
}

async function getAgentOutputs(labId: string): Promise<string[]> {
  const outputsDir = path.join(LABS_DIR, labId, "state", "outputs");
  if (!(await fileExists(outputsDir))) {
    return [];
  }

  try {
    const files = await fs.readdir(outputsDir);
    return files.filter((f) => f.endsWith(".log") || f.endsWith(".task"));
  } catch {
    return [];
  }
}

/**
 * GET /api/labs/research/[id]
 * Get detailed research state for a specific lab
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id: labId } = await params;

    // Get lab config
    const config = await getLabConfig(labId);
    if (!config) {
      return NextResponse.json(
        { success: false, error: `Lab '${labId}' not found` },
        { status: 404 }
      );
    }

    // Get all lab data in parallel
    const [agents, proposals, progress, tasks, outputs] = await Promise.all([
      getLabAgents(labId),
      getLabProposals(labId),
      getLabProgress(labId),
      getClaudeTasks(config.taskListId),
      getAgentOutputs(labId),
    ]);

    // Get research documents if domain is set
    let documents: ResearchDocument[] = [];
    if (config.domain) {
      documents = await getResearchDocuments(config.domain);
    }

    // Calculate stats
    const runningAgents = agents.filter(
      (a) => a.status === "running" || a.status === "working"
    );
    const pendingProposals = proposals.filter(
      (p) => p.status === "pending" || p.status === "pending_review"
    );
    const taskStats = {
      total: tasks.length,
      pending: tasks.filter((t) => t.status === "pending").length,
      in_progress: tasks.filter((t) => t.status === "in_progress").length,
      completed: tasks.filter((t) => t.status === "completed").length,
    };

    return NextResponse.json({
      success: true,
      lab: {
        id: config.id,
        name: config.name,
        description: config.description,
        active: config.active,
        domain: config.domain,
        taskListId: config.taskListId,
        settings: config.settings,
        createdAt: config.createdAt,
        updatedAt: config.updatedAt,
      },
      agents: {
        list: agents,
        running: runningAgents.length,
        total: agents.length,
      },
      proposals: {
        list: proposals,
        pending: pendingProposals.length,
        total: proposals.length,
      },
      tasks: {
        list: tasks,
        stats: taskStats,
      },
      progress: {
        recent: progress.slice(-50), // Last 50 entries
        total: progress.length,
      },
      documents,
      outputs,
    });
  } catch (error) {
    console.error("Failed to get lab details:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to get lab details",
      },
      { status: 500 }
    );
  }
}
