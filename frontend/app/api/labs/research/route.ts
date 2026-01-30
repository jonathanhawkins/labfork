/**
 * Research Manager Labs API
 *
 * GET /api/labs/research - List labs from research manager with live status
 *
 * This endpoint reads from the research manager's lab configs and state files
 * to provide real-time status of research labs.
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
const CLAUDE_TASKS_DIR = path.join(
  process.env.HOME || "/Users/light",
  ".claude",
  "tasks"
);

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

interface AgentState {
  [agentId: string]: {
    name?: string;
    displayName?: string;
    type?: string;
    status: string;
    currentTask?: string;
    progress?: number;
    startedAt?: string;
    lastActivityAt?: string;
  };
}

interface Proposal {
  id: string;
  title?: string;
  status: string;
  createdAt?: string;
}

interface ProposalsState {
  proposals: { [id: string]: Proposal };
}

interface ClaudeTask {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  priority?: string;
  owner?: string;
  createdAt?: string;
  updatedAt?: string;
}

interface LabStatus {
  id: string;
  name: string;
  description: string;
  active: boolean;
  domain?: string;
  settings?: LabConfig["settings"];
  agents: {
    running: number;
    total: number;
    list: AgentState[string][];
  };
  tasks: {
    total: number;
    pending: number;
    in_progress: number;
    completed: number;
  };
  proposals: {
    total: number;
    pending: number;
  };
  createdAt: string;
  updatedAt: string;
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

async function getLabAgents(labId: string): Promise<AgentState> {
  const agentsPath = path.join(LABS_DIR, labId, "state", "agents.json");
  return readJsonFile<AgentState>(agentsPath, {});
}

async function getLabProposals(labId: string): Promise<ProposalsState> {
  const proposalsPath = path.join(LABS_DIR, labId, "state", "proposals.json");
  return readJsonFile<ProposalsState>(proposalsPath, { proposals: {} });
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

    return tasks;
  } catch {
    return [];
  }
}

async function getLabStatus(labId: string): Promise<LabStatus | null> {
  const config = await getLabConfig(labId);
  if (!config) {
    return null;
  }

  const [agents, proposals, tasks] = await Promise.all([
    getLabAgents(labId),
    getLabProposals(labId),
    getClaudeTasks(config.taskListId),
  ]);

  // Count agents
  const agentList = Object.values(agents);
  const runningAgents = agentList.filter(
    (a) => a.status === "running" || a.status === "working"
  );

  // Count proposals
  const proposalList = Object.values(proposals.proposals || {});
  const pendingProposals = proposalList.filter(
    (p) => p.status === "pending" || p.status === "pending_review"
  );

  // Count tasks
  const taskCounts = {
    total: tasks.length,
    pending: tasks.filter((t) => t.status === "pending").length,
    in_progress: tasks.filter((t) => t.status === "in_progress").length,
    completed: tasks.filter((t) => t.status === "completed").length,
  };

  return {
    id: config.id,
    name: config.name,
    description: config.description,
    active: config.active,
    domain: config.domain,
    settings: config.settings,
    agents: {
      running: runningAgents.length,
      total: agentList.length,
      list: agentList,
    },
    tasks: taskCounts,
    proposals: {
      total: proposalList.length,
      pending: pendingProposals.length,
    },
    createdAt: config.createdAt,
    updatedAt: config.updatedAt,
  };
}

/**
 * GET /api/labs/research
 * List all research manager labs with status
 */
export async function GET(request: NextRequest) {
  try {
    // Check if labs directory exists
    if (!(await fileExists(LABS_DIR))) {
      return NextResponse.json({
        success: true,
        labs: [],
        message: "No research manager labs found",
      });
    }

    // List all lab directories
    const entries = await fs.readdir(LABS_DIR, { withFileTypes: true });
    const labDirs = entries.filter(
      (e) => e.isDirectory() && !e.name.startsWith("_") && !e.name.startsWith(".")
    );

    // Get status for each lab
    const labs: LabStatus[] = [];
    for (const dir of labDirs) {
      const status = await getLabStatus(dir.name);
      if (status) {
        labs.push(status);
      }
    }

    // Sort by updated time (most recent first)
    labs.sort(
      (a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );

    return NextResponse.json({
      success: true,
      labs,
      total: labs.length,
      labsDir: LABS_DIR, // For debugging
    });
  } catch (error) {
    console.error("Failed to list research labs:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to list labs",
      },
      { status: 500 }
    );
  }
}
