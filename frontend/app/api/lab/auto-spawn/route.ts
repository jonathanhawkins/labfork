import { NextResponse } from "next/server";
import { spawn } from "child_process";
import { join } from "path";
import {
  readFileSync,
  readdirSync,
  existsSync,
  statSync,
  writeFileSync,
} from "fs";
import { homedir } from "os";

// Research Manager paths
const projectRoot = join(process.cwd(), "..");
const RM_SCRIPT = join(projectRoot, ".skills", "research-manager", "rm");
const AGENTS_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "agents.json"
);
const RESEARCH_STATE_FILE = join(
  projectRoot,
  ".skills",
  "research-manager",
  "state",
  "research-state.json"
);

// Research topics to cycle through
const RESEARCH_TOPICS = [
  "prosody conditioning TTS 2024 2025 emotion neural speech synthesis",
  "disentangled speech synthesis prosody content separation",
  "emotion transfer voice cloning zero-shot",
  "pitch contour prediction neural TTS F0 modeling",
  "DeepSeek techniques for speech synthesis",
  "variational autoencoder prosody disentanglement",
];

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL;

interface Task {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  owner?: string;
  blockedBy?: string[];  // Tasks that must complete before this one can start
}

interface AgentInfo {
  name: string;
  type: string;
  status: string;
  task: string;
}

/**
 * Get all tasks from Claude Code task list
 */
function getAllTasks(): Task[] {
  const tasksDir = join(homedir(), ".claude", "tasks");
  if (!existsSync(tasksDir)) return [];

  const sessions = readdirSync(tasksDir)
    .filter((name) => {
      const fullPath = join(tasksDir, name);
      return statSync(fullPath).isDirectory() && !name.startsWith(".");
    })
    .map((name) => ({
      name,
      mtime: statSync(join(tasksDir, name)).mtime,
    }))
    .sort((a, b) => b.mtime.getTime() - a.mtime.getTime());

  if (sessions.length === 0) return [];

  const sessionDir = join(tasksDir, sessions[0].name);
  const taskFiles = readdirSync(sessionDir).filter(
    (f) => f.endsWith(".json") && !f.startsWith(".")
  );

  const tasks: Task[] = [];
  for (const file of taskFiles) {
    try {
      const content = readFileSync(join(sessionDir, file), "utf-8");
      const task = JSON.parse(content) as Task;
      tasks.push(task);
    } catch (e) {
      // Skip invalid files
    }
  }

  return tasks;
}

/**
 * Get pending tasks from Claude Code task list
 * Filters out tasks that are blocked by incomplete tasks
 */
function getPendingTasks(): Task[] {
  const allTasks = getAllTasks();
  const completedIds = new Set(allTasks.filter(t => t.status === "completed").map(t => t.id));

  return allTasks.filter(t => {
    // Must be pending and unowned
    if (t.status !== "pending" || t.owner) return false;

    // Check if blocked by any incomplete task
    if (t.blockedBy && t.blockedBy.length > 0) {
      const hasIncompleteBlocker = t.blockedBy.some(blockerId => !completedIds.has(blockerId));
      if (hasIncompleteBlocker) return false;
    }

    return true;
  });
}

/**
 * Get completed task IDs
 */
function getCompletedTaskIds(): Set<string> {
  return new Set(getAllTasks().filter(t => t.status === "completed").map(t => t.id));
}

/**
 * Get running research agents with full info
 */
interface FullAgentInfo extends AgentInfo {
  started_at?: string;
}

function getRunningAgents(): FullAgentInfo[] {
  if (!existsSync(AGENTS_FILE)) return [];

  try {
    const agents = JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
    return Object.values(agents).filter(
      (a: any) => a.status === "running"
    ) as FullAgentInfo[];
  } catch {
    return [];
  }
}

/**
 * Check if an agent's log shows completion signals
 */
function checkLogForCompletion(agentName: string): { completed: boolean; reason?: string } {
  try {
    const agents = JSON.parse(readFileSync(AGENTS_FILE, "utf-8"));
    const agent = agents[agentName];
    if (!agent?.output_file || !existsSync(agent.output_file)) {
      return { completed: false };
    }

    const content = readFileSync(agent.output_file, "utf-8");
    const lastChunk = content.slice(-20000); // Check last 20KB

    // Completion signals in agent logs
    const completionPatterns = [
      /TaskUpdate.*status.*completed/i,
      /marked.*as.*completed/i,
      /task.*completed.*successfully/i,
      /Successfully completed/i,
      /All tasks.*completed/i,
      /No more.*pending.*tasks/i,
    ];

    for (const pattern of completionPatterns) {
      if (pattern.test(lastChunk)) {
        return { completed: true, reason: `Log contains: ${pattern.source}` };
      }
    }

    return { completed: false };
  } catch {
    return { completed: false };
  }
}

/**
 * Kill an agent by name
 */
function killAgent(name: string): boolean {
  try {
    const result = require("child_process").spawnSync(
      RM_SCRIPT,
      ["kill", "--name", name],
      { cwd: projectRoot, encoding: "utf-8" }
    );
    console.log(`[Auto-spawn] Killed agent ${name}:`, result.stdout);
    return result.status === 0;
  } catch (e) {
    console.error(`[Auto-spawn] Failed to kill agent ${name}:`, e);
    return false;
  }
}

/**
 * Record agent outcome to progress tracker
 */
async function recordAgentOutcome(
  agentName: string,
  taskId: string | undefined,
  outcome: "completed" | "stuck" | "error" | "timeout",
  duration: number,
  progressScore: number
) {
  try {
    await fetch(`http://localhost:3003/api/lab/progress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "record",
        agentName,
        taskId,
        outcome,
        duration,
        progressScore,
      }),
    });
  } catch (e) {
    console.error("[Auto-spawn] Failed to record outcome:", e);
  }
}

/**
 * Get retry recommendation for a task
 */
async function getRetryRecommendation(taskId: string): Promise<{
  shouldRetry: boolean;
  reason: string;
  strategy?: string;
}> {
  try {
    const response = await fetch(`http://localhost:3003/api/lab/progress`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "should-retry", taskId }),
    });
    return await response.json();
  } catch (e) {
    return { shouldRetry: true, reason: "Could not check history" };
  }
}

/**
 * Clean up finished agents:
 * 1. Task agents whose tasks are completed
 * 2. Research agents running > 20 minutes
 * 3. Any agent running > 60 minutes (stuck)
 * Now also records outcomes for progress tracking
 */
function cleanupFinishedAgents(): { killed: string[]; reason: Record<string, string> } {
  const runningAgents = getRunningAgents();
  const completedTaskIds = getCompletedTaskIds();
  const killed: string[] = [];
  const reason: Record<string, string> = {};
  const now = Date.now();

  for (const agent of runningAgents) {
    const startedAt = agent.started_at ? new Date(agent.started_at).getTime() : now;
    const runningMinutes = (now - startedAt) / 60000;

    // Check if this is a task agent with completed task
    const taskMatch = agent.name.match(/^task-(\d+)-/);
    if (taskMatch) {
      const taskId = taskMatch[1];
      if (completedTaskIds.has(taskId)) {
        console.log(`[Auto-spawn] Killing ${agent.name}: task #${taskId} completed`);
        if (killAgent(agent.name)) {
          killed.push(agent.name);
          reason[agent.name] = `Task #${taskId} completed`;
          // Record successful completion
          recordAgentOutcome(agent.name, taskId, "completed", runningMinutes, 100);
        }
        continue;
      }

      // Also check log for completion signals (task might not be marked yet)
      const logCompletion = checkLogForCompletion(agent.name);
      if (logCompletion.completed && runningMinutes > 5) {
        console.log(`[Auto-spawn] Killing ${agent.name}: log shows completion - ${logCompletion.reason}`);
        if (killAgent(agent.name)) {
          killed.push(agent.name);
          reason[agent.name] = `Log completion: ${logCompletion.reason}`;
          recordAgentOutcome(agent.name, taskId, "completed", runningMinutes, 90);
        }
        continue;
      }
    }

    // Check if research agent running too long (20 min)
    if ((agent.name.includes("researcher") || agent.name.includes("web-research")) && runningMinutes > 20) {
      console.log(`[Auto-spawn] Killing ${agent.name}: research timeout (${Math.round(runningMinutes)}min)`);
      if (killAgent(agent.name)) {
        killed.push(agent.name);
        reason[agent.name] = `Research timeout (${Math.round(runningMinutes)}min)`;
        // Record timeout
        recordAgentOutcome(agent.name, undefined, "timeout", runningMinutes, 50);
      }
      continue;
    }

    // Check if any agent running too long (60 min = probably stuck)
    if (runningMinutes > 60) {
      console.log(`[Auto-spawn] Killing ${agent.name}: stuck timeout (${Math.round(runningMinutes)}min)`);
      if (killAgent(agent.name)) {
        killed.push(agent.name);
        reason[agent.name] = `Stuck timeout (${Math.round(runningMinutes)}min)`;
        // Record stuck outcome
        const stuckTaskId = taskMatch ? taskMatch[1] : undefined;
        recordAgentOutcome(agent.name, stuckTaskId, "stuck", runningMinutes, 20);
      }
    }
  }

  return { killed, reason };
}

/**
 * Get research state (last research time, topic index)
 */
function getResearchState(): { lastResearchTime: number; topicIndex: number } {
  try {
    if (existsSync(RESEARCH_STATE_FILE)) {
      return JSON.parse(readFileSync(RESEARCH_STATE_FILE, "utf-8"));
    }
  } catch {}
  return { lastResearchTime: 0, topicIndex: 0 };
}

/**
 * Save research state
 */
function saveResearchState(state: { lastResearchTime: number; topicIndex: number }) {
  try {
    writeFileSync(RESEARCH_STATE_FILE, JSON.stringify(state, null, 2));
  } catch (e) {
    console.error("[Auto-spawn] Failed to save research state:", e);
  }
}

/**
 * Check if a web research agent is already running
 */
function hasRunningResearchAgent(agents: AgentInfo[]): boolean {
  return agents.some(a =>
    a.name.includes("web-research") ||
    a.name.includes("researcher") ||
    a.task?.includes("WebSearch")
  );
}

/**
 * Spawn a web research agent
 */
async function spawnWebResearchAgent(): Promise<boolean> {
  const state = getResearchState();
  const topic = RESEARCH_TOPICS[state.topicIndex % RESEARCH_TOPICS.length];

  return new Promise((resolve) => {
    const agentName = `web-researcher-${Date.now()}`;
    const taskPrompt = `You are a web research agent. Your job is to find NEW ideas for improving prosody and emotion conditioning in TTS systems.

USE WebSearch TOOL (Claude built-in) to search for recent papers and techniques.

YOUR RESEARCH TOPIC: "${topic}"

CURRENT STATUS:
- V7 baseline training complete (Happy > Sad pitch verified)
- Task #6 "V7 Baseline Verification" must complete before new features
- All new feature tasks should be BLOCKED BY #6

INSTRUCTIONS:
1. First call TaskList to see current tasks and their status
2. Use WebSearch to find recent papers/repos on this topic
3. For each promising finding, create a task with TaskCreate that includes:
   - Subject: What to implement
   - Description MUST include:
     a) Key technique summary (2-3 sentences)
     b) How to integrate with our codebase
     c) SUCCESS CRITERIA: Specific metric improvement expected
     d) VERIFICATION: How to test it works
     e) DEPENDENCIES: What must be done first (usually "Requires V7 baseline (#6)")
4. After creating tasks, use TaskUpdate to set blockedBy: ["6"] for each new task
5. DO NOT create tasks that are just "implement X" - every task needs measurable success criteria

EXAMPLE GOOD TASK:
  Subject: "Add emotion-aware attention to prosody encoder"
  Description: "...SUCCESS CRITERIA: Happy pitch > Sad pitch after training. VERIFICATION: Run inference/evaluate.py. DEPENDENCIES: Requires V7 baseline (#6) verified first."
  Then: TaskUpdate with blockedBy: ["6"]

You have WebSearch access - USE IT! Start searching now.`;

    const child = spawn(RM_SCRIPT, [
      "spawn",
      "--type",
      "opus",
      "--name",
      agentName,
      "--task",
      taskPrompt,
    ], {
      cwd: projectRoot,
      stdio: "pipe",
    });

    child.on("close", (code) => {
      if (code === 0) {
        // Update research state - move to next topic
        saveResearchState({
          lastResearchTime: Date.now(),
          topicIndex: state.topicIndex + 1,
        });
      }
      resolve(code === 0);
    });

    child.on("error", () => resolve(false));
    setTimeout(() => resolve(false), 30000);
  });
}

/**
 * Spawn a research agent to work on a task
 * Now includes retry strategy if this is a retry attempt
 */
async function spawnAgentForTask(task: Task): Promise<{ success: boolean; isRetry: boolean; strategy?: string }> {
  // Check if this is a retry
  const retryRec = await getRetryRecommendation(task.id);

  if (!retryRec.shouldRetry && retryRec.reason.includes("failed")) {
    console.log(`[Auto-spawn] Skipping task #${task.id}: ${retryRec.reason}`);
    return { success: false, isRetry: false };
  }

  const isRetry = retryRec.reason.includes("Attempt") && !retryRec.reason.includes("first");
  const strategy = retryRec.strategy;

  // Build strategy hint for retry attempts
  let strategyHint = "";
  if (isRetry && strategy) {
    switch (strategy) {
      case "break_into_subtasks":
        strategyHint = `
RETRY STRATEGY: Previous attempt got stuck. Break this into smaller subtasks:
1. First, analyze what specific subtasks are needed
2. Create separate tasks for each subtask using TaskCreate
3. Complete the subtasks one by one
4. Don't try to do everything at once`;
        break;
      case "add_error_handling":
        strategyHint = `
RETRY STRATEGY: Previous attempt had errors. Be more careful:
1. Read existing code thoroughly before making changes
2. Test changes incrementally
3. If you encounter an error, stop and analyze before retrying
4. Use smaller, safer edits`;
        break;
      case "simplify_scope":
        strategyHint = `
RETRY STRATEGY: Previous attempt timed out. Simplify:
1. Focus on the minimum viable implementation
2. Skip optional features
3. Leave TODOs for future improvements
4. Aim for a working version, not perfection`;
        break;
    }
  }

  return new Promise((resolve) => {
    const agentName = `task-${task.id}-${Date.now()}`;
    const taskPrompt = `Work on this task from the shared task list (CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline"):

TASK #${task.id}: ${task.subject}
${task.description ? `\nDescription: ${task.description}` : ""}
${strategyHint}
CRITICAL PROJECT CONTEXT:
- Current F0 correlation: 0.051 (target: > 0.3)
- Current problem: Happy pitch < Sad pitch (INVERTED)
- Goal: Fix prosody/emotion conditioning in TTS

INSTRUCTIONS:
1. Call TaskUpdate to mark task #${task.id} as in_progress with your name as owner
2. Implement the feature/fix described
3. VERIFY IT WORKS:
   - If the task has SUCCESS CRITERIA in description, check those
   - Run any relevant tests or evaluation scripts
   - Check that your changes don't break existing functionality
4. ONLY mark completed if you have EVIDENCE it works:
   - Show test output or evaluation metrics
   - If you can't verify (e.g., needs GPU training), note this and mark completed with caveat
5. If verification FAILS, do NOT mark complete - fix the issue or document what went wrong

DO NOT just write code and mark complete. VERIFY your changes work.

You have access to all Claude Code tools. Be autonomous and thorough.`;

    const child = spawn(RM_SCRIPT, [
      "spawn",
      "--type",
      "opus",
      "--name",
      agentName,
      "--task",
      taskPrompt,
    ], {
      cwd: projectRoot,
      stdio: "pipe",
    });

    let output = "";
    child.stdout?.on("data", (data) => {
      output += data.toString();
    });
    child.stderr?.on("data", (data) => {
      output += data.toString();
    });

    child.on("close", (code) => {
      console.log(`[Auto-spawn] Agent ${agentName} spawn result:`, output);
      resolve({ success: code === 0, isRetry, strategy });
    });

    child.on("error", (err) => {
      console.error(`[Auto-spawn] Failed to spawn agent:`, err);
      resolve({ success: false, isRetry, strategy });
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      resolve({ success: false, isRetry, strategy });
    }, 30000);
  });
}

/**
 * POST /api/lab/auto-spawn
 * Triggers auto-spawn of research agents for pending tasks
 * Also auto-spawns web research agents periodically
 * Now includes auto-cleanup of stuck/finished agents
 */
export async function POST(request: Request) {
  try {
    // Check for force-research parameter
    const url = new URL(request.url);
    if (BACKEND_URL) {
      try {
        const response = await fetch(`${BACKEND_URL}/api/lab/auto-spawn${url.search}`, {
          method: "POST",
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
        console.error("[Auto-spawn] Backend request failed, falling back to local:", error);
      }
    }

    const forceResearch = url.searchParams.get("research") === "true";
    const cleanupOnly = url.searchParams.get("cleanup") === "true";

    // FIRST: Clean up finished/stuck agents to free slots
    const cleanupResult = cleanupFinishedAgents();
    if (cleanupResult.killed.length > 0) {
      console.log(`[Auto-spawn] Cleaned up ${cleanupResult.killed.length} agents:`, cleanupResult.reason);
    }

    // If cleanup-only mode, return results
    if (cleanupOnly) {
      return NextResponse.json({
        success: true,
        type: "cleanup",
        killed: cleanupResult.killed,
        reasons: cleanupResult.reason,
      });
    }

    // Get current state (after cleanup)
    const pendingTasks = getPendingTasks();
    const runningAgents = getRunningAgents();
    const researchState = getResearchState();

    console.log(
      `[Auto-spawn] Pending tasks: ${pendingTasks.length}, Running agents: ${runningAgents.length}`
    );

    // Check if we should spawn a research agent
    const RESEARCH_INTERVAL = 30 * 60 * 1000; // 30 minutes
    const timeSinceLastResearch = Date.now() - researchState.lastResearchTime;
    const needsResearch = timeSinceLastResearch > RESEARCH_INTERVAL;
    const hasResearcher = hasRunningResearchAgent(runningAgents);

    // Don't spawn if already have max agents running (max 3 concurrent)
    if (runningAgents.length >= 3) {
      return NextResponse.json({
        success: false,
        reason: "Max concurrent agents reached",
        runningAgents: runningAgents.length,
        pendingTasks: pendingTasks.length,
        researchStatus: hasResearcher ? "running" : "idle",
        cleanup: cleanupResult.killed.length > 0 ? cleanupResult : undefined,
      });
    }

    // Priority 1: Spawn research agent if needed (or forced)
    if ((needsResearch || forceResearch) && !hasResearcher && runningAgents.length < 3) {
      console.log(`[Auto-spawn] Spawning web research agent (last research: ${Math.round(timeSinceLastResearch / 60000)}min ago)`);
      const spawned = await spawnWebResearchAgent();

      if (spawned) {
        return NextResponse.json({
          success: true,
          type: "research",
          topic: RESEARCH_TOPICS[researchState.topicIndex % RESEARCH_TOPICS.length],
          runningAgents: runningAgents.length + 1,
          pendingTasks: pendingTasks.length,
        });
      }
    }

    // Priority 2: Spawn task agent if pending tasks exist
    if (pendingTasks.length > 0 && runningAgents.length < 3) {
      const taskToAssign = pendingTasks[0];
      console.log(`[Auto-spawn] Spawning agent for task: ${taskToAssign.subject}`);
      const spawnResult = await spawnAgentForTask(taskToAssign);

      return NextResponse.json({
        success: spawnResult.success,
        type: "task",
        assignedTask: taskToAssign,
        runningAgents: runningAgents.length + (spawnResult.success ? 1 : 0),
        pendingTasks: pendingTasks.length - (spawnResult.success ? 1 : 0),
        isRetry: spawnResult.isRetry,
        retryStrategy: spawnResult.strategy,
      });
    }

    // Nothing to spawn
    return NextResponse.json({
      success: false,
      reason: "No work to assign",
      runningAgents: runningAgents.length,
      pendingTasks: pendingTasks.length,
      researchStatus: hasResearcher ? "running" : (needsResearch ? "due" : "recent"),
      nextResearchIn: Math.max(0, RESEARCH_INTERVAL - timeSinceLastResearch),
      cleanup: cleanupResult.killed.length > 0 ? cleanupResult : undefined,
    });
  } catch (error) {
    console.error("[Auto-spawn] Error:", error);
    return NextResponse.json(
      { error: "Failed to auto-spawn agent" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/lab/auto-spawn
 * Returns current state (pending tasks, running agents, research status)
 */
export async function GET(request: Request) {
  try {
    if (BACKEND_URL) {
      try {
        const url = new URL(request.url);
        const response = await fetch(`${BACKEND_URL}/api/lab/auto-spawn${url.search}`, {
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
        console.error("[Auto-spawn] Backend fetch failed, falling back to local:", error);
      }
    }

    const pendingTasks = getPendingTasks();
    const runningAgents = getRunningAgents();
    const researchState = getResearchState();

    const RESEARCH_INTERVAL = 30 * 60 * 1000; // 30 minutes
    const timeSinceLastResearch = Date.now() - researchState.lastResearchTime;
    const hasResearcher = hasRunningResearchAgent(runningAgents);
    const nextTopic = RESEARCH_TOPICS[researchState.topicIndex % RESEARCH_TOPICS.length];

    return NextResponse.json({
      pendingTasks: pendingTasks.length,
      runningAgents: runningAgents.length,
      tasks: pendingTasks.map((t) => ({
        id: t.id,
        subject: t.subject,
      })),
      agents: runningAgents.map((a) => ({
        name: a.name,
        type: a.type,
        status: a.status,
      })),
      canSpawn: pendingTasks.length > 0 && runningAgents.length < 3,
      // Research status
      research: {
        hasResearcher,
        lastResearchTime: researchState.lastResearchTime,
        timeSinceLastResearch,
        isDue: timeSinceLastResearch > RESEARCH_INTERVAL,
        nextTopic,
        topicIndex: researchState.topicIndex,
        totalTopics: RESEARCH_TOPICS.length,
      },
    });
  } catch (error) {
    console.error("[Auto-spawn] Error:", error);
    return NextResponse.json(
      { error: "Failed to get auto-spawn status" },
      { status: 500 }
    );
  }
}
