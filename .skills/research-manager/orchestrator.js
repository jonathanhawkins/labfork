#!/usr/bin/env node
/**
 * Research Manager Orchestrator Daemon
 *
 * Runs continuously in the background to:
 * 1. Monitor agent health and kill stuck agents
 * 2. Clean up completed/timed-out agents
 * 3. Auto-spawn agents for pending tasks
 * 4. Track progress and handle retries
 *
 * Usage: node orchestrator.js [--interval 30] [--max-agents 3]
 */

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Notification system
const notifications = require('./notifications');

// Configuration
const CONFIG = {
  checkInterval: parseInt(process.argv.find((a, i) => process.argv[i-1] === '--interval') || '30') * 1000,
  maxAgents: parseInt(process.argv.find((a, i) => process.argv[i-1] === '--max-agents') || '3'),
  researchTimeout: 60, // minutes (increased from 20 - research takes time)
  stuckTimeout: 30, // minutes no activity (increased from 10 - agents can be thinking)
  maxRuntime: 240, // minutes (4 hours - safety net only, activity-based detection is primary)
  projectRoot: path.join(__dirname, '..', '..'),

  // Task prioritization settings
  prioritization: {
    maxRetries: 5,            // Stop retrying after this many failures
    retryCooldownMinutes: 10, // Wait this long after failure before retry
    ageBonus: 1.25,           // Score bonus per hour of age
    maxAgeBonus: 30,          // Cap on age bonus
    blockerPenalty: 50,       // Penalty per pending blocker
    inProgressBlockerPenalty: 20, // Penalty per in-progress blocker
    unblockBonus: 15,         // Bonus for each task this would unblock
  }
};

// Lab-aware state paths
const LABS_DIR = path.join(__dirname, 'labs');
const GLOBAL_STATE_DIR = path.join(__dirname, 'state');
const ACTIVE_LAB_FILE = path.join(GLOBAL_STATE_DIR, 'active-lab.json');
const RM_SCRIPT = path.join(__dirname, 'rm');
const CODEX_CMD = 'codex';

// Get current lab ID from env, flag, or active lab file
function getLabId() {
  // 1. Command line override
  const labArg = process.argv.find((a, i) => process.argv[i-1] === '--lab');
  if (labArg) return labArg;

  // 2. Environment variable
  if (process.env.CLAUDE_CODE_LAB_ID) return process.env.CLAUDE_CODE_LAB_ID;

  // 3. Active lab from state file
  try {
    if (fs.existsSync(ACTIVE_LAB_FILE)) {
      const data = JSON.parse(fs.readFileSync(ACTIVE_LAB_FILE, 'utf-8'));
      if (data.active_lab_id) return data.active_lab_id;
    }
  } catch (e) {
    // Ignore errors
  }

  // 4. Default
  return 'voice-clone';
}

// Get lab configuration
function getLabConfig(labId) {
  const configPath = path.join(LABS_DIR, labId, 'config.json');
  try {
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    }
  } catch (e) {
    // Ignore errors
  }
  // Return virtual config for backwards compatibility
  return {
    id: labId,
    name: labId,
    taskListId: labId,
    settings: { maxAgents: 3, autoSpawn: true }
  };
}

// Current lab context
const CURRENT_LAB_ID = getLabId();
const LAB_CONFIG = getLabConfig(CURRENT_LAB_ID);

// Get state directory for current lab
function getStateDir() {
  const labDir = path.join(LABS_DIR, CURRENT_LAB_ID);
  const labStateDir = path.join(labDir, 'state');
  if (fs.existsSync(path.join(labDir, 'config.json'))) {
    if (!fs.existsSync(labStateDir)) {
      fs.mkdirSync(labStateDir, { recursive: true });
    }
    return labStateDir;
  }
  // Fallback to global state
  return GLOBAL_STATE_DIR;
}

// Dynamic state paths based on lab
const STATE_DIR = getStateDir();
const AGENTS_FILE = path.join(STATE_DIR, 'agents.json');
const PROGRESS_FILE = path.join(STATE_DIR, 'progress.json');
const COST_FILE = path.join(STATE_DIR, 'cost-tracking.json');
const RESEARCH_STATE_FILE = path.join(STATE_DIR, 'research-state.json');
const PROPOSALS_FILE = path.join(STATE_DIR, 'proposals.json');
const ORCHESTRATOR_LOG = path.join(STATE_DIR, 'orchestrator.log');
const TASK_LIST_ID = process.env.CLAUDE_CODE_TASK_LIST_ID || LAB_CONFIG.taskListId || CURRENT_LAB_ID;

// Cost configuration - estimated costs per hour by model type
// We default to local Ollama (free) and use Codex only for hard analysis.
const COST_RATES = {
  codex: 2.0,     // Estimate for Codex CLI usage
  ollama: 0.0,    // Local Ollama is free
  opus: 0.0,      // Legacy alias (should not be used)
};

// Cost alert thresholds (in USD)
const COST_THRESHOLDS = {
  dailyWarning: 50,    // $50 daily warning
  dailyAlert: 100,     // $100 daily alert
  dailyHardCap: 150,   // $150 daily HARD CAP - stops all spawning
  weeklyWarning: 300,  // $300 weekly warning
  weeklyAlert: 500,    // $500 weekly alert
  weeklyHardCap: 750,  // $750 weekly HARD CAP - stops all spawning
};

// Task limits
const TASK_LIMITS = {
  maxTasksPerDay: 20,        // Max tasks completed per day
  maxResearchPerDay: 5,      // Max research sessions per day (web searches)
  maxAgentMinutesPerDay: 180, // Max total agent runtime per day (3 hours)
};

// Ensure state directory exists
if (!fs.existsSync(STATE_DIR)) {
  fs.mkdirSync(STATE_DIR, { recursive: true });
}

// Log rotation - keep log under 1MB
function rotateLogIfNeeded() {
  try {
    if (fs.existsSync(ORCHESTRATOR_LOG)) {
      const stats = fs.statSync(ORCHESTRATOR_LOG);
      if (stats.size > 1024 * 1024) { // 1MB
        const backupLog = ORCHESTRATOR_LOG + '.old';
        if (fs.existsSync(backupLog)) {
          fs.unlinkSync(backupLog);
        }
        fs.renameSync(ORCHESTRATOR_LOG, backupLog);
      }
    }
  } catch (e) {
    // Ignore rotation errors
  }
}

// Logging (only to file, not console - nohup redirects stdout to log anyway)
function log(level, message, data = {}) {
  rotateLogIfNeeded();
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] [${level.toUpperCase()}] ${message} ${Object.keys(data).length ? JSON.stringify(data) : ''}`;
  fs.appendFileSync(ORCHESTRATOR_LOG, logLine + '\n');
}

// Read JSON file safely
function readJSON(filePath, defaultValue = {}) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
  } catch (e) {
    log('warn', `Failed to read ${filePath}`, { error: e.message });
  }
  return defaultValue;
}

// Write JSON file
function writeJSON(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

function resolveCodexPath() {
  const fromEnv = process.env.CODEX_PATH;
  if (fromEnv && fs.existsSync(fromEnv)) return fromEnv;

  const whichResult = spawnSync('which', [CODEX_CMD], { encoding: 'utf-8' });
  if (whichResult.status === 0) {
    const found = (whichResult.stdout || '').trim();
    if (found) return found;
  }

  const home = process.env.HOME || os.homedir();
  const nvmRoot = path.join(home, '.nvm', 'versions', 'node');
  if (fs.existsSync(nvmRoot)) {
    const versions = fs.readdirSync(nvmRoot)
      .map((v) => path.join(nvmRoot, v, 'bin', CODEX_CMD))
      .filter((p) => fs.existsSync(p));
    if (versions.length > 0) {
      return versions.sort().slice(-1)[0];
    }
  }

  return null;
}

function commandExists(cmd) {
  if (cmd === CODEX_CMD) {
    return Boolean(resolveCodexPath());
  }
  const result = spawnSync('which', [cmd], { encoding: 'utf-8' });
  return result.status === 0;
}

// Get all agents
function getAgents() {
  return readJSON(AGENTS_FILE, {});
}

// Save agents
function saveAgents(agents) {
  writeJSON(AGENTS_FILE, agents);
}

// Get running agents
function getRunningAgents() {
  const agents = getAgents();
  return Object.entries(agents)
    .filter(([_, a]) => a.status === 'running')
    .map(([name, a]) => ({ name, ...a }));
}

// Kill an agent
function killAgent(name) {
  const result = spawnSync(RM_SCRIPT, ['kill', '--name', name], {
    cwd: CONFIG.projectRoot,
    encoding: 'utf-8',
  });
  return result.status === 0;
}

// Clean up stale agents (run periodically)
let lastCleanupTime = 0;
const CLEANUP_INTERVAL_MS = 30 * 60 * 1000; // 30 minutes

function runCleanupIfNeeded() {
  const now = Date.now();
  if (now - lastCleanupTime < CLEANUP_INTERVAL_MS) {
    return { skipped: true, reason: 'too soon' };
  }

  lastCleanupTime = now;
  const cleanupScript = path.join(CONFIG.projectRoot, '.skills', 'research-manager', 'cleanup.py');

  try {
    const result = spawnSync('python3', [cleanupScript, '--execute', '--quiet'], {
      cwd: CONFIG.projectRoot,
      encoding: 'utf-8',
      timeout: 30000,
    });

    if (result.status === 0) {
      const cleaned = parseInt(result.stdout.trim()) || 0;
      if (cleaned > 0) {
        log('info', 'Cleanup completed', { cleaned });
      }
      return { success: true, cleaned };
    } else {
      log('warn', 'Cleanup failed', { stderr: result.stderr });
      return { success: false, error: result.stderr };
    }
  } catch (e) {
    log('error', 'Cleanup error', { error: e.message });
    return { success: false, error: e.message };
  }
}

// Get tasks from Claude Code task list
function resolveTaskSessionDir() {
  const tasksDir = path.join(os.homedir(), '.claude', 'tasks');
  if (!fs.existsSync(tasksDir)) return null;

  const preferredDir = path.join(tasksDir, TASK_LIST_ID);
  if (fs.existsSync(preferredDir) && fs.statSync(preferredDir).isDirectory()) {
    return preferredDir;
  }

  const sessions = fs.readdirSync(tasksDir)
    .filter(name => {
      const fullPath = path.join(tasksDir, name);
      return fs.statSync(fullPath).isDirectory() && !name.startsWith('.');
    })
    .map(name => ({
      name,
      mtime: fs.statSync(path.join(tasksDir, name)).mtime,
    }))
    .sort((a, b) => b.mtime - a.mtime);

  if (sessions.length === 0) return null;
  return path.join(tasksDir, sessions[0].name);
}

function getTasks() {
  const sessionDir = resolveTaskSessionDir();
  if (!sessionDir) return [];
  const taskFiles = fs.readdirSync(sessionDir).filter(f => f.endsWith('.json') && !f.startsWith('.'));

  const tasks = [];
  for (const file of taskFiles) {
    try {
      const content = fs.readFileSync(path.join(sessionDir, file), 'utf-8');
      tasks.push(JSON.parse(content));
    } catch (e) {}
  }
  return tasks;
}

function getLatestTaskSessionDir() {
  return resolveTaskSessionDir();
}

function updateTaskFile(taskId, updates) {
  const sessionDir = getLatestTaskSessionDir();
  if (!sessionDir) return false;

  const taskFile = path.join(sessionDir, `${taskId}.json`);
  if (!fs.existsSync(taskFile)) return false;

  try {
    const task = JSON.parse(fs.readFileSync(taskFile, 'utf-8'));
    const updated = {
      ...task,
      ...updates,
      updatedAt: new Date().toISOString(),
    };
    fs.writeFileSync(taskFile, JSON.stringify(updated, null, 2));
    return true;
  } catch (e) {
    log('warn', 'Failed to update task file', { taskId, error: e.message });
    return false;
  }
}

// Get task IDs that already have agents running
function getTasksWithRunningAgents() {
  const runningAgents = getRunningAgents();
  const taskIds = new Set();
  for (const agent of runningAgents) {
    const match = agent.name.match(/^task-(\d+)-/);
    if (match) {
      taskIds.add(match[1]);
    }
  }
  return taskIds;
}

// Check if an implementation task is blocked by proposal approval
function isBlockedByProposalApproval(task) {
  const storyId = getStoryFromTask(task);
  if (!storyId) return false;

  // Only check implementation tasks (not research tasks)
  if (isResearchTask(task)) return false;

  // Check if this story has research tasks
  const stories = getTasksByStory();
  const story = stories[storyId];
  if (!story || story.researchTasks.length === 0) return false;

  // Check if research is complete
  const researchStatus = isStoryResearchComplete(storyId);
  if (!researchStatus.complete) {
    // Research not complete - task is blocked by research, not proposal
    return false;
  }

  // Research is complete - check proposal status
  const proposals = getProposals();
  const proposal = proposals.proposals?.[storyId];

  if (!proposal) {
    // No proposal yet - blocked pending proposal generation
    return true;
  }

  // Check proposal status
  if (proposal.status === 'approved') {
    return false; // Approved - can proceed
  }

  // Any other status (generating, pending_review, rejected, needs_revision) blocks
  return true;
}

// Get pending tasks (no owner, status pending, no agent already running)
function getPendingTasks() {
  const tasks = getTasks();
  const tasksWithAgents = getTasksWithRunningAgents();
  const completedTaskIds = new Set(tasks.filter(t => t.status === 'completed').map(t => String(t.id)));
  return tasks.filter(t =>
    t.status === 'pending' &&
    !t.owner &&
    !tasksWithAgents.has(String(t.id)) &&
    !(t.blockedBy || []).some(id => !completedTaskIds.has(String(id))) &&
    !isBlockedByProposalApproval(t)  // NEW: Check proposal approval gate
  );
}

// Smart task prioritization
// Returns tasks sorted by priority score (higher = more urgent)
function prioritizeTasks(pendingTasks, allTasks) {
  const completedTaskIds = new Set(allTasks.filter(t => t.status === 'completed').map(t => t.id));
  const inProgressTaskIds = new Set(allTasks.filter(t => t.status === 'in_progress').map(t => t.id));
  const progress = readJSON(PROGRESS_FILE, { history: [] });
  const now = Date.now();

  // Calculate retry counts per task
  const retryCountsPerTask = {};
  for (const entry of progress.history) {
    if (entry.taskId && entry.outcome !== 'completed') {
      retryCountsPerTask[entry.taskId] = (retryCountsPerTask[entry.taskId] || 0) + 1;
    }
  }

  // Calculate last failure time per task
  const lastFailureTime = {};
  for (const entry of progress.history) {
    if (entry.taskId && entry.outcome !== 'completed') {
      const ts = new Date(entry.timestamp).getTime();
      if (!lastFailureTime[entry.taskId] || ts > lastFailureTime[entry.taskId]) {
        lastFailureTime[entry.taskId] = ts;
      }
    }
  }

  const scoredTasks = pendingTasks.map(task => {
    let score = 100; // Base score
    const reasons = [];

    // 1. Manual priority override (highest impact: -50 to +50)
    // metadata.priority: "high" (+50), "low" (-30), "critical" (+100), "blocked" (-100)
    const priority = task.metadata?.priority;
    if (priority === 'critical') {
      score += 100;
      reasons.push('critical priority (+100)');
    } else if (priority === 'high') {
      score += 50;
      reasons.push('high priority (+50)');
    } else if (priority === 'low') {
      score -= 30;
      reasons.push('low priority (-30)');
    } else if (priority === 'blocked') {
      score -= 100;
      reasons.push('manually blocked (-100)');
    }

    // 2. Task dependencies - check blockedBy field
    // If blocked by incomplete tasks, heavily penalize
    const p = CONFIG.prioritization;
    if (task.blockedBy && task.blockedBy.length > 0) {
      const unresolvedBlockers = task.blockedBy.filter(
        blockerId => !completedTaskIds.has(blockerId)
      );
      if (unresolvedBlockers.length > 0) {
        // Check if blockers are in progress (might complete soon)
        const inProgressBlockers = unresolvedBlockers.filter(
          blockerId => inProgressTaskIds.has(blockerId)
        );
        const pendingBlockers = unresolvedBlockers.length - inProgressBlockers.length;

        // Configurable penalties per blocker type
        const blockerPenalty = (pendingBlockers * p.blockerPenalty) + (inProgressBlockers.length * p.inProgressBlockerPenalty);
        score -= blockerPenalty;
        reasons.push(`blocked by ${unresolvedBlockers.length} tasks (-${blockerPenalty})`);
      }
    }

    // 3. Task age - older tasks get priority
    // Use createdAt if available, otherwise use a default recent time
    const createdAt = task.createdAt ? new Date(task.createdAt).getTime() : now;
    const ageMs = now - createdAt;
    const ageHours = ageMs / (1000 * 60 * 60);
    const ageBonus = Math.min(p.maxAgeBonus, Math.floor(ageHours * p.ageBonus));
    if (ageBonus > 0) {
      score += ageBonus;
      reasons.push(`age ${Math.round(ageHours)}h (+${ageBonus})`);
    }

    // 4. Retry history - penalize tasks that keep failing
    const retries = retryCountsPerTask[task.id] || 0;
    if (retries > 0) {
      // Progressive backoff: -10 for 1st retry, -25 for 2nd, -45 for 3rd, etc
      const retryPenalty = Math.min(100, retries * retries * 5 + retries * 5);
      score -= retryPenalty;
      reasons.push(`${retries} retries (-${retryPenalty})`);

      // Cooldown after recent failure - extra penalty if failed recently
      const lastFail = lastFailureTime[task.id];
      if (lastFail) {
        const minutesSinceFailure = (now - lastFail) / (1000 * 60);
        // If failed within cooldown period, add penalty
        if (minutesSinceFailure < p.retryCooldownMinutes) {
          const cooldownPenalty = Math.floor((p.retryCooldownMinutes - minutesSinceFailure) * 5);
          score -= cooldownPenalty;
          reasons.push(`recent failure ${Math.round(minutesSinceFailure)}m ago (-${cooldownPenalty})`);
        }
      }
    }

    // 5. Bonus for tasks that unblock others
    const blocksCount = allTasks.filter(t =>
      t.status === 'pending' &&
      t.blockedBy &&
      t.blockedBy.includes(task.id)
    ).length;
    if (blocksCount > 0) {
      const unblockBonus = blocksCount * p.unblockBonus;
      score += unblockBonus;
      reasons.push(`unblocks ${blocksCount} tasks (+${unblockBonus})`);
    }

    // 6. Max retries circuit breaker - if task exceeded max retries, skip it
    if (retries >= p.maxRetries) {
      score = -999;
      reasons.push(`MAX RETRIES EXCEEDED (${retries}/${p.maxRetries})`);
    }

    return {
      ...task,
      _priority: {
        score,
        reasons,
        retries,
        ageHours: Math.round(ageHours * 10) / 10,
      }
    };
  });

  // Sort by score (descending) - highest priority first
  scoredTasks.sort((a, b) => b._priority.score - a._priority.score);

  return scoredTasks;
}

// Get completed task IDs
function getCompletedTaskIds() {
  return new Set(getTasks().filter(t => t.status === 'completed').map(t => t.id));
}

// Check if agent log shows completion
function checkLogForCompletion(agent) {
  if (!agent.output_file || !fs.existsSync(agent.output_file)) return false;

  try {
    const content = fs.readFileSync(agent.output_file, 'utf-8');
    const lastChunk = content.slice(-20000);

    const completionPatterns = [
      /TaskUpdate.*status.*completed/i,
      /marked.*as.*completed/i,
      /task.*completed.*successfully/i,
      /Successfully completed/i,
    ];

    return completionPatterns.some(p => p.test(lastChunk));
  } catch (e) {
    return false;
  }
}

// Extract potential deliverable paths from task description
function extractDeliverablePaths(taskDescription) {
  if (!taskDescription) return [];

  const paths = [];
  const projectRoot = CONFIG.projectRoot;

  // Pattern 1: Explicit file paths like "docs/firefly/analysis.md" or "frontend/lib/types.ts"
  // Note: tsx/jsx must come before ts/js in the alternation to match correctly
  const filePathPattern = /(?:^|[\s`'"(])([a-zA-Z0-9_\-./]+\.(?:md|tsx|jsx|ts|js|py|yaml|yml|json|sql))/g;
  let match;
  while ((match = filePathPattern.exec(taskDescription)) !== null) {
    const filePath = match[1];
    // Skip common false positives
    if (!filePath.includes('/') || filePath.startsWith('http')) continue;
    paths.push(path.join(projectRoot, filePath));
  }

  // Pattern 2: Create file instructions like "Create `frontend/lib/activities/types.ts`"
  const createFilePattern = /[Cc]reate\s+(?:file:?\s*)?[`'"]*([a-zA-Z0-9_\-./]+\.(?:md|ts|tsx|js|jsx|py|yaml|yml|json|sql))[`'"]*/g;
  while ((match = createFilePattern.exec(taskDescription)) !== null) {
    paths.push(path.join(projectRoot, match[1]));
  }

  // Pattern 3: Output paths like "Write to docs/analysis.md"
  const writePattern = /[Ww]rite\s+(?:to\s+)?[`'"]*([a-zA-Z0-9_\-./]+\.(?:md|ts|tsx|js|jsx|py|yaml|yml|json|sql))[`'"]*/g;
  while ((match = writePattern.exec(taskDescription)) !== null) {
    paths.push(path.join(projectRoot, match[1]));
  }

  // Deduplicate
  return [...new Set(paths)];
}

// Check if deliverables mentioned in task description exist
function checkDeliverablesExist(task) {
  const description = task?.description || '';
  const subject = task?.subject || '';
  const fullText = `${subject}\n${description}`;

  const deliverablePaths = extractDeliverablePaths(fullText);

  if (deliverablePaths.length === 0) {
    return { hasDeliverables: false, checked: 0, found: 0, paths: [] };
  }

  const foundPaths = [];
  for (const filePath of deliverablePaths) {
    if (fs.existsSync(filePath)) {
      // Check if file was recently modified (within last 24 hours)
      try {
        const stats = fs.statSync(filePath);
        const hoursSinceModified = (Date.now() - stats.mtime.getTime()) / (1000 * 60 * 60);
        if (hoursSinceModified < 24) {
          foundPaths.push(filePath);
        }
      } catch (e) {
        // If we can't stat, still count it as found
        foundPaths.push(filePath);
      }
    }
  }

  return {
    hasDeliverables: foundPaths.length > 0,
    checked: deliverablePaths.length,
    found: foundPaths.length,
    paths: foundPaths,
  };
}

// Check if agent is making progress (activity-based heartbeat)
function isAgentMakingProgress(agent) {
  if (!agent.output_file || !fs.existsSync(agent.output_file)) {
    return { making_progress: false, reason: 'no output file' };
  }

  try {
    const stats = fs.statSync(agent.output_file);
    const minutesSinceActivity = (Date.now() - stats.mtime.getTime()) / (1000 * 60);

    // If output file was modified in the last 5 minutes, agent is active
    if (minutesSinceActivity < 5) {
      return { making_progress: true, reason: `output modified ${Math.round(minutesSinceActivity)}m ago` };
    }

    // Check file size - if it's growing, agent is working
    const size = stats.size;
    if (size > 50000) {
      // Large output file suggests substantial work
      return { making_progress: minutesSinceActivity < 15, reason: `large output (${Math.round(size/1024)}KB), last activity ${Math.round(minutesSinceActivity)}m ago` };
    }

    return { making_progress: false, reason: `no activity for ${Math.round(minutesSinceActivity)}m` };
  } catch (e) {
    return { making_progress: false, reason: `error checking output: ${e.message}` };
  }
}

// Check agent health
function checkAgentHealth(agent) {
  const now = Date.now();
  const startedAt = new Date(agent.started_at).getTime();
  const runningMinutes = (now - startedAt) / 60000;

  let lastActivity = 999;
  if (agent.output_file && fs.existsSync(agent.output_file)) {
    const stats = fs.statSync(agent.output_file);
    lastActivity = (now - stats.mtime.getTime()) / 60000;
  }

  return {
    runningMinutes,
    lastActivity,
    isStuck: lastActivity > CONFIG.stuckTimeout,
    isTimedOut: runningMinutes > CONFIG.maxRuntime,
    isResearchTimeout: agent.name.includes('researcher') && runningMinutes > CONFIG.researchTimeout,
  };
}

// Check if a research-lead agent completed and update proposal status
function checkResearchLeadCompletion(agent, outcome) {
  // Check if this is a research-lead agent
  const match = agent.name.match(/^research-lead-(s\d+)-/i);
  if (!match) return;

  const storyId = match[1].toUpperCase();
  const proposals = getProposals();

  if (!proposals.proposals[storyId]) {
    proposals.proposals[storyId] = {};
  }

  if (outcome === 'completed') {
    // Check if proposal document was created
    const projectRoot = path.join(__dirname, '..', '..');
    const proposalPaths = [
      path.join(projectRoot, 'docs', storyId.toLowerCase(), 'PROPOSAL.md'),
      path.join(projectRoot, 'docs', `story_${storyId.toLowerCase()}`, 'PROPOSAL.md'),
      path.join(projectRoot, 'docs', 'firefly', 'PROPOSAL.md'),
    ];

    const documentExists = proposalPaths.some(p => fs.existsSync(p));

    if (documentExists) {
      proposals.proposals[storyId].status = 'pending_review';
      proposals.proposals[storyId].document_created_at = new Date().toISOString();

      // Notify user that proposal is ready
      notifications.notifyProposalReady(
        storyId,
        proposals.proposals[storyId].story_title
      ).catch(() => {});

      log('info', `Proposal for ${storyId} ready for review`);
    } else {
      proposals.proposals[storyId].status = 'generation_failed';
      proposals.proposals[storyId].error = 'Document not created';
      log('warn', `Research lead completed but proposal document not found for ${storyId}`);
    }
  } else {
    proposals.proposals[storyId].status = 'generation_failed';
    proposals.proposals[storyId].error = `Agent outcome: ${outcome}`;
  }

  proposals.proposals[storyId].updated_at = new Date().toISOString();
  saveProposals(proposals);
}

// Record outcome to progress history
function recordOutcome(agentName, taskId, outcome, duration, progressScore) {
  const progress = readJSON(PROGRESS_FILE, { history: [] });

  const retryCount = taskId
    ? progress.history.filter(h => h.taskId === taskId && h.outcome !== 'completed').length
    : 0;

  progress.history.push({
    agentName,
    taskId,
    outcome,
    duration,
    progressScore,
    timestamp: new Date().toISOString(),
    retryCount,
  });

  // Keep last 100
  progress.history = progress.history.slice(-100);
  writeJSON(PROGRESS_FILE, progress);

  return retryCount;
}

// Cleanup old killed agents from state file
function cleanupKilledAgents() {
  const agents = getAgents();
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  let removed = 0;

  for (const [name, agent] of Object.entries(agents)) {
    if (agent.status === 'killed' && agent.killed_at) {
      const killedTime = new Date(agent.killed_at).getTime();
      if (killedTime < oneHourAgo) {
        delete agents[name];
        removed++;
      }
    }
  }

  if (removed > 0) {
    saveAgents(agents);
    log('info', `Cleaned up ${removed} old killed agents`);
  }

  return removed;
}

// Cleanup finished/stuck agents
function cleanupAgents() {
  const agents = getAgents();
  const runningAgents = getRunningAgents();
  const completedTaskIds = getCompletedTaskIds();
  const allTasks = getTasks();
  const killed = [];

  for (const agent of runningAgents) {
    const health = checkAgentHealth(agent);
    const taskMatch = agent.name.match(/^task-(\d+)-/);
    const taskId = taskMatch ? taskMatch[1] : undefined;

    // Get the task for deliverable checking
    const task = taskId ? allTasks.find(t => String(t.id) === String(taskId)) : null;

    let shouldKill = false;
    let reason = '';
    let outcome = 'completed';

    // Check if task is completed
    if (taskId && completedTaskIds.has(taskId)) {
      shouldKill = true;
      reason = `Task #${taskId} completed`;
      outcome = 'completed';
    }
    // Check log for completion signals
    else if (taskId && health.runningMinutes > 5 && checkLogForCompletion(agent)) {
      shouldKill = true;
      reason = 'Log shows completion';
      outcome = 'completed';
    }
    // Check if deliverables exist (task completed but agent didn't mark it)
    else if (task && health.runningMinutes > 10) {
      const deliverables = checkDeliverablesExist(task);
      if (deliverables.hasDeliverables) {
        shouldKill = true;
        reason = `Deliverables found (${deliverables.found}/${deliverables.checked} files exist)`;
        outcome = 'completed';
        log('info', `Marking task complete due to deliverables`, {
          taskId,
          paths: deliverables.paths,
        });
      }
    }

    // Only check for stuck/timeout if we haven't already decided to kill
    if (!shouldKill) {
      // Check if agent is making progress (activity-based heartbeat)
      const progress = isAgentMakingProgress(agent);

      // Research agents get more lenient timeout
      const isResearcher = agent.name.includes('researcher') || agent.name.includes('research');

      // Stuck detection - only if truly no activity
      if (health.isStuck && !progress.making_progress) {
        // Before killing as stuck, check deliverables one more time
        if (task) {
          const deliverables = checkDeliverablesExist(task);
          if (deliverables.hasDeliverables) {
            shouldKill = true;
            reason = `Stuck but deliverables found (${deliverables.found} files)`;
            outcome = 'completed';
          } else {
            shouldKill = true;
            reason = `No activity for ${Math.round(health.lastActivity)}min, no progress detected`;
            outcome = 'stuck';
            notifications.notifyAgentStuck(agent.name, health.lastActivity).catch(() => {});
          }
        } else {
          shouldKill = true;
          reason = `No activity for ${Math.round(health.lastActivity)}min`;
          outcome = 'stuck';
          notifications.notifyAgentStuck(agent.name, health.lastActivity).catch(() => {});
        }
      }
      // Research timeout - but respect activity
      else if (health.isResearchTimeout && isResearcher && !progress.making_progress) {
        shouldKill = true;
        reason = `Research timeout (${Math.round(health.runningMinutes)}min), no recent progress`;
        outcome = 'timeout';
      }
      // Max runtime exceeded - this is a hard safety limit
      else if (health.isTimedOut) {
        // Even at max runtime, if making progress, log but still kill (safety)
        if (progress.making_progress) {
          log('warn', `Killing active agent at max runtime`, {
            name: agent.name,
            runtime: Math.round(health.runningMinutes),
            progress: progress.reason,
          });
        }
        // Check deliverables before killing
        if (task) {
          const deliverables = checkDeliverablesExist(task);
          if (deliverables.hasDeliverables) {
            shouldKill = true;
            reason = `Max runtime but deliverables found (${deliverables.found} files)`;
            outcome = 'completed';
          } else {
            shouldKill = true;
            reason = `Max runtime exceeded (${Math.round(health.runningMinutes)}min) - safety limit`;
            outcome = 'timeout';
          }
        } else {
          shouldKill = true;
          reason = `Max runtime exceeded (${Math.round(health.runningMinutes)}min) - safety limit`;
          outcome = 'timeout';
        }
      }
    }

    if (shouldKill) {
      log('info', `Killing agent: ${agent.name}`, { reason, outcome });
      agents[agent.name].kill_reason = reason;

      if (killAgent(agent.name)) {
        killed.push(agent.name);
        agents[agent.name].status = 'killed';
        agents[agent.name].killed_at = new Date().toISOString();
        const retryCount = recordOutcome(agent.name, taskId, outcome, health.runningMinutes, outcome === 'completed' ? 100 : 30);

        // Check if this was a research-lead agent and update proposal status
        checkResearchLeadCompletion(agent, outcome);

        // Notify on task completion
        if (outcome === 'completed' && taskId) {
          const task = allTasks.find(t => String(t.id) === String(taskId));
          notifications.notifyTaskCompleted(taskId, task?.subject || 'Unknown task', health.runningMinutes).catch(() => {});

          // Mark the task as completed since deliverables exist
          updateTaskFile(taskId, { status: 'completed', owner: `rm:${agent.name}` });
        }

        // Notify on repeated errors
        if (outcome !== 'completed' && taskId && retryCount > 0) {
          const task = allTasks.find(t => String(t.id) === String(taskId));
          notifications.notifyAgentErrorRepeated(taskId, task?.subject || 'Unknown task', retryCount + 1).catch(() => {});
        }

        if (outcome !== 'completed' && taskId) {
          const task = allTasks.find(t => String(t.id) === String(taskId));
          if (!task || !task.owner || task.owner === `rm:${agent.name}`) {
            updateTaskFile(taskId, { status: 'pending', owner: null });
          }
        }
      }
    }
  }

  if (killed.length > 0) {
    saveAgents(agents);
  }

  return killed;
}

// Clean up orphaned in_progress tasks (tasks with no active agent)
function cleanupOrphanedTasks() {
  const allTasks = getTasks();
  const agents = readJSON(AGENTS_FILE, {});
  const activeAgentTaskIds = new Set();

  // Build set of task IDs that have active agents
  for (const [agentId, agent] of Object.entries(agents)) {
    if (agent.status === 'running') {
      // Extract task ID from agent name (e.g., "task-371-1234567890" -> "371")
      const match = agentId.match(/^task-(\d+)-/);
      if (match) {
        activeAgentTaskIds.add(match[1]);
      }
    }
  }

  const orphaned = [];
  const taskDir = path.join(os.homedir(), '.claude', 'tasks', TASK_LIST_ID);

  for (const task of allTasks) {
    if (task.status !== 'in_progress') continue;
    if (activeAgentTaskIds.has(task.id)) continue; // Has active agent

    // This task is in_progress but has no active agent - it's orphaned
    // Check if it has deliverables
    const deliverableCheck = checkDeliverablesExist(task);

    const taskFile = path.join(taskDir, `${task.id}.json`);
    if (!fs.existsSync(taskFile)) continue;

    try {
      const taskData = JSON.parse(fs.readFileSync(taskFile, 'utf-8'));

      if (deliverableCheck.found) {
        // Deliverables exist - mark as completed
        taskData.status = 'completed';
        taskData.completedAt = new Date().toISOString();
        taskData.completedBy = 'orchestrator:deliverable-detection';
        fs.writeFileSync(taskFile, JSON.stringify(taskData, null, 2));
        orphaned.push({ id: task.id, action: 'completed', reason: 'deliverables exist' });
        log('info', `Orphaned task ${task.id} completed (deliverables found)`, {
          deliverables: deliverableCheck.paths,
        });
      } else {
        // No deliverables - check how long it's been stuck
        const updatedAt = taskData.updatedAt ? new Date(taskData.updatedAt).getTime() : 0;
        const stuckMinutes = (Date.now() - updatedAt) / (1000 * 60);

        if (stuckMinutes > 60) { // Stuck for more than 1 hour
          // Reset to pending so it can be picked up again
          taskData.status = 'pending';
          delete taskData.owner;
          taskData.resetAt = new Date().toISOString();
          taskData.resetReason = 'orphaned:no-agent';
          fs.writeFileSync(taskFile, JSON.stringify(taskData, null, 2));
          orphaned.push({ id: task.id, action: 'reset', reason: `stuck ${Math.round(stuckMinutes)} min` });
          log('info', `Orphaned task ${task.id} reset to pending (stuck ${Math.round(stuckMinutes)} min)`);
        }
      }
    } catch (e) {
      log('warn', `Failed to process orphaned task ${task.id}: ${e.message}`);
    }
  }

  return orphaned;
}

// Spawn agent for a task
function selectAgentType(task) {
  const subject = (task.subject || '').toLowerCase();
  const description = (task.description || '').toLowerCase();
  const text = `${subject} ${description}`;

  // Explicit override via tag in subject/description
  if (text.includes('[codex]') || text.includes('codex:') || text.includes('use codex')) {
    return { type: 'codex', reason: 'explicit', requiresCodex: true };
  }

  // Reviews should go to Codex for speed and depth
  const reviewKeywords = new Set([
    'review',
    'reviewer',
    'audit',
    'auditing',
    'critique',
    'assessment',
    'assess',
    'validate',
    'validation',
    'verify',
    'verification',
    'postmortem',
    'retrospective',
  ]);
  const words = text.split(/[^a-z0-9]+/).filter(Boolean);
  if (words.some((word) => reviewKeywords.has(word))) {
    return { type: 'codex', reason: 'review', requiresCodex: true };
  }

  // Heuristics for "hard work" that benefits from Codex
  const codexHints = [
    'architecture',
    'multi-file',
    'deep analysis',
    'root cause',
    'refactor',
    'benchmark',
    'design doc',
    'tradeoff',
    'performance investigation',
  ];
  if (codexHints.some((hint) => text.includes(hint))) {
    return { type: 'codex', reason: 'complexity', requiresCodex: false };
  }

  // Default: local Ollama agent
  return { type: 'ollama', reason: 'default', requiresCodex: false };
}

function spawnTaskAgent(task) {
  const agentName = `task-${task.id}-${Date.now()}`;
  const selection = selectAgentType(task);
  let agentType = selection.type;
  if (agentType === 'codex' && !commandExists(CODEX_CMD)) {
    if (selection.requiresCodex) {
      log('warn', 'Codex required but unavailable; skipping spawn', {
        taskId: task.id,
        reason: selection.reason,
      });
      return { spawned: false, reason: 'codex-unavailable', selection };
    }
    log('warn', 'Codex not found; falling back to ollama', { taskId: task.id });
    agentType = 'ollama';
  }
  const taskPrompt = `Work on this task from the shared task list (CLAUDE_CODE_TASK_LIST_ID="${TASK_LIST_ID}"):

TASK #${task.id}: ${task.subject}
${task.description ? `\nDescription: ${task.description}` : ''}

INSTRUCTIONS:
1. Read .skills/research-manager/MISSION.md and align work to current priority
2. First call TaskList to see all tasks
3. Call TaskUpdate to mark task #${task.id} as in_progress with your name as owner
4. Work on the task autonomously
5. When done, call TaskUpdate to mark it completed
6. Check TaskList for more pending tasks

You have access to all Claude Code tools. Be autonomous and thorough.`;

  const runPolicy = `\n\nRESEARCH OUTPUT POLICY:\nIf this task produces experimental results (training/eval/technique tests), you MUST:\n1) Create a run record:\n   python scripts/research/run_registry.py new --title \"${task.subject}\" --task-id ${task.id}\n2) Write metrics (use quick_eval if applicable):\n   python inference/quick_eval.py --checkpoint <ckpt> --run-dir <run_dir>\n3) Update the run with metrics:\n   python scripts/research/run_registry.py update --run-dir <run_dir> --metrics-file <run_dir>/metrics.json\n4) Leave review status as pending for the manager.\n`;

  const fullPrompt = taskPrompt + runPolicy;

  const result = spawnSync(RM_SCRIPT, [
    'spawn', '--type', agentType, '--name', agentName, '--task', fullPrompt
  ], {
    cwd: CONFIG.projectRoot,
    encoding: 'utf-8',
  });

  if (result.status === 0) {
    if (task.status === 'pending' && !task.owner) {
      updateTaskFile(task.id, {
        status: 'in_progress',
        owner: `rm:${agentName}`,
      });
    }
    return { spawned: true, name: agentName, type: agentType, selection };
  }
  return { spawned: false, reason: 'spawn-failed', selection };
}

// Spawn research agent
function spawnResearchAgent() {
  const state = readJSON(RESEARCH_STATE_FILE, { lastResearchTime: 0, topicIndex: 0 });
  const topics = [
    'prosody conditioning TTS 2024 2025 emotion neural speech synthesis',
    'disentangled speech synthesis prosody content separation',
    'emotion transfer voice cloning zero-shot',
    'pitch contour prediction neural TTS F0 modeling',
  ];

  const topic = topics[state.topicIndex % topics.length];
  const agentName = `web-researcher-${Date.now()}`;

  const taskPrompt = `You are a web research agent. Your job is to find NEW ideas for improving prosody and emotion conditioning in TTS systems.

USE WebSearch TOOL (Claude built-in) to search for recent papers and techniques.

YOUR RESEARCH TOPIC: "${topic}"

INSTRUCTIONS:
1. Use WebSearch to find recent papers/repos on this topic
2. For each promising finding:
   - Summarize the key technique (2-3 sentences)
   - Note how it could help our prosody models
   - Use TaskCreate to add promising ideas to implement
3. After researching, update the task list with findings

You have WebSearch access - USE IT! Start searching now.`;

  const result = spawnSync(RM_SCRIPT, [
    'spawn', '--type', 'ollama', '--name', agentName, '--task', taskPrompt
  ], {
    cwd: CONFIG.projectRoot,
    encoding: 'utf-8',
  });

  if (result.status === 0) {
    writeJSON(RESEARCH_STATE_FILE, {
      lastResearchTime: Date.now(),
      topicIndex: state.topicIndex + 1,
    });
    return agentName;
  }
  return null;
}

// Check if research is due
function isResearchDue() {
  const state = readJSON(RESEARCH_STATE_FILE, { lastResearchTime: 0 });
  const RESEARCH_INTERVAL = 30 * 60 * 1000; // 30 minutes
  return Date.now() - state.lastResearchTime > RESEARCH_INTERVAL;
}

// Check if we have a research agent running
function hasResearchAgent() {
  return getRunningAgents().some(a => a.name.includes('researcher'));
}

// Check if focus mode is blocking agent spawning
function checkFocusMode() {
  try {
    const focusScript = path.join(__dirname, 'focus.py');
    const result = spawnSync('python3', [focusScript, 'block', '--json'], {
      encoding: 'utf-8',
      timeout: 5000,
    });

    if (result.status === 0) {
      // Not blocked
      return { blocked: false };
    }

    // Parse JSON output to get focus details
    try {
      const data = JSON.parse(result.stdout);
      return {
        blocked: data.blocked,
        focus: data.focus,
        description: data.focus?.description || 'Priority task in focus',
        taskId: data.focus?.task_id,
      };
    } catch (e) {
      // Fallback if JSON parsing fails
      return {
        blocked: true,
        description: result.stdout.trim() || 'Focus mode active',
      };
    }
  } catch (e) {
    log('error', 'Failed to check focus mode', { error: e.message });
    // On error, don't block - fail open
    return { blocked: false };
  }
}

// Auto-spawn agents
function autoSpawn() {
  // Check hard limits FIRST
  const limitCheck = checkLimits();
  if (limitCheck.exceeded) {
    log('warn', 'HARD LIMIT EXCEEDED - spawning paused', {
      limits: limitCheck.limits,
      stats: limitCheck.stats
    });
    return { spawned: null, reason: `Limits exceeded: ${limitCheck.limits.join(', ')}` };
  }

  // Check focus mode - blocks spawning EXCEPT for the focused task
  const focusCheck = checkFocusMode();
  const focusTaskId = focusCheck.taskId ? String(focusCheck.taskId) : null;

  const runningAgents = getRunningAgents();
  const pendingTasks = getPendingTasks();
  const allTasks = getTasks();

  if (runningAgents.length >= CONFIG.maxAgents) {
    return { spawned: null, reason: 'Max agents reached' };
  }

  // Priority 1: Research agent if due (blocked by focus mode)
  if (isResearchDue() && !hasResearchAgent()) {
    if (focusCheck.blocked) {
      log('info', 'FOCUS MODE - skipping research agent', { focus: focusCheck.description });
    } else {
      const agentName = spawnResearchAgent();
      if (agentName) {
        log('info', 'Spawned research agent', { name: agentName });

        // Notify about spawned research agent
        notifications.notifyAgentSpawned(agentName, 'Web research for TTS/prosody papers').catch(() => {});

        return { spawned: agentName, type: 'research' };
      }
    }
  }

  // Priority 2: Task agent (with smart prioritization)
  if (pendingTasks.length > 0) {
    // Apply smart prioritization
    const prioritizedTasks = prioritizeTasks(pendingTasks, allTasks);

    // Filter out tasks with negative scores (max retries exceeded, heavily blocked)
    let viableTasks = prioritizedTasks.filter(t => t._priority.score > 0);

    // Focus mode: ONLY allow the focused task to be spawned
    if (focusCheck.blocked && focusTaskId) {
      const focusedTask = viableTasks.find(t => String(t.id) === focusTaskId);
      if (focusedTask) {
        log('info', 'FOCUS MODE - allowing focused task only', {
          taskId: focusTaskId,
          subject: focusedTask.subject,
        });
        viableTasks = [focusedTask];
      } else {
        log('info', 'FOCUS MODE - focused task not in viable list', {
          focusTaskId,
          reason: 'Task may be completed, blocked, or not pending',
        });
        return {
          spawned: null,
          reason: `Focus mode: waiting for task #${focusTaskId}`,
          focusMode: true,
        };
      }
    }

    if (viableTasks.length === 0) {
      log('info', 'No viable tasks (all blocked or max retries)', {
        skipped: prioritizedTasks.map(t => ({
          id: t.id,
          score: t._priority.score,
          reasons: t._priority.reasons.join(', '),
        }))
      });
      return { spawned: null, reason: 'No viable tasks' };
    }

    const task = viableTasks[0];
    log('info', 'Selected task by priority', {
      taskId: task.id,
      subject: task.subject,
      score: task._priority.score,
      reasons: task._priority.reasons.join(', '),
      alternatives: viableTasks.slice(1, 4).map(t => ({
        id: t.id,
        score: t._priority.score,
      }))
    });

    const spawnResult = spawnTaskAgent(task);
    if (spawnResult && spawnResult.spawned) {
      log('info', 'Spawned task agent', {
        name: spawnResult.name,
        task: task.subject,
        priority: task._priority.score,
        type: spawnResult.type,
        selection: spawnResult.selection?.reason,
      });

      // Notify about spawned agent
      notifications.notifyAgentSpawned(spawnResult.name, `#${task.id}: ${task.subject}`).catch(() => {});

      return {
        spawned: spawnResult.name,
        type: 'task',
        task: task.subject,
        priority: task._priority,
      };
    }
    if (spawnResult && !spawnResult.spawned && spawnResult.reason) {
      log('info', 'Skipped task spawn', {
        taskId: task.id,
        reason: spawnResult.reason,
        selection: spawnResult.selection?.reason,
      });
      return { spawned: null, reason: spawnResult.reason };
    }
  }

  return { spawned: null, reason: 'Nothing to spawn' };
}

// Get current cost and task stats
function getDailyStats() {
  try {
    const costData = readJSON(COST_FILE, { sessions: [], totals: {} });
    const progressData = readJSON(PROGRESS_FILE, { tasks: {} });

    const today = new Date().toISOString().split('T')[0];
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];

    // Calculate costs
    const todaySessions = costData.sessions.filter(s => s.date === today);
    const dailyCost = todaySessions.reduce((sum, s) => sum + (s.estimatedCost || 0), 0);
    const weeklyCost = costData.sessions
      .filter(s => s.date >= weekAgo)
      .reduce((sum, s) => sum + (s.estimatedCost || 0), 0);

    // Calculate task completions today
    const tasksCompletedToday = Object.values(progressData.tasks || {})
      .filter(t => t.completedAt && t.completedAt.startsWith(today))
      .length;

    // Calculate research sessions today
    const researchToday = todaySessions.filter(s => s.type === 'research').length;

    // Calculate agent minutes today
    const agentMinutesToday = todaySessions.reduce((sum, s) => sum + (s.durationMinutes || 0), 0);

    return { dailyCost, weeklyCost, tasksCompletedToday, researchToday, agentMinutesToday };
  } catch (e) {
    log('warn', 'Failed to get daily stats', { error: e.message });
    return { dailyCost: 0, weeklyCost: 0, tasksCompletedToday: 0, researchToday: 0, agentMinutesToday: 0 };
  }
}

// Check if we've hit any hard limits
function checkLimits() {
  const stats = getDailyStats();
  const limits = [];

  // Cost hard caps
  if (stats.dailyCost >= COST_THRESHOLDS.dailyHardCap) {
    limits.push(`Daily cost cap exceeded ($${stats.dailyCost.toFixed(2)} >= $${COST_THRESHOLDS.dailyHardCap})`);
  }
  if (stats.weeklyCost >= COST_THRESHOLDS.weeklyHardCap) {
    limits.push(`Weekly cost cap exceeded ($${stats.weeklyCost.toFixed(2)} >= $${COST_THRESHOLDS.weeklyHardCap})`);
  }

  // Task limits
  if (stats.tasksCompletedToday >= TASK_LIMITS.maxTasksPerDay) {
    limits.push(`Daily task limit reached (${stats.tasksCompletedToday}/${TASK_LIMITS.maxTasksPerDay})`);
  }
  if (stats.researchToday >= TASK_LIMITS.maxResearchPerDay) {
    limits.push(`Daily research limit reached (${stats.researchToday}/${TASK_LIMITS.maxResearchPerDay})`);
  }
  if (stats.agentMinutesToday >= TASK_LIMITS.maxAgentMinutesPerDay) {
    limits.push(`Daily agent time limit reached (${stats.agentMinutesToday}/${TASK_LIMITS.maxAgentMinutesPerDay} min)`);
  }

  return { exceeded: limits.length > 0, limits, stats };
}

// Check cost thresholds and notify if exceeded
function checkCostThresholds() {
  try {
    const costData = readJSON(COST_FILE, { sessions: [], totals: {} });
    const config = notifications.loadConfig();
    const thresholds = config.thresholds;

    // Calculate daily cost
    const today = new Date().toISOString().split('T')[0];
    const dailyCost = costData.sessions
      .filter(s => s.date === today)
      .reduce((sum, s) => sum + (s.estimatedCost || 0), 0);

    // Calculate weekly cost
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
    const weeklyCost = costData.sessions
      .filter(s => s.date >= weekAgo)
      .reduce((sum, s) => sum + (s.estimatedCost || 0), 0);

    // Check thresholds (alert takes precedence over warning)
    if (dailyCost >= thresholds.dailyCostAlert) {
      notifications.notifyCostThreshold('dailyAlert', dailyCost, thresholds.dailyCostAlert).catch(() => {});
    } else if (dailyCost >= thresholds.dailyCostWarning) {
      notifications.notifyCostThreshold('dailyWarning', dailyCost, thresholds.dailyCostWarning).catch(() => {});
    }

    if (weeklyCost >= thresholds.weeklyCostAlert) {
      notifications.notifyCostThreshold('weeklyAlert', weeklyCost, thresholds.weeklyCostAlert).catch(() => {});
    } else if (weeklyCost >= thresholds.weeklyCostWarning) {
      notifications.notifyCostThreshold('weeklyWarning', weeklyCost, thresholds.weeklyCostWarning).catch(() => {});
    }

    return { dailyCost, weeklyCost };
  } catch (e) {
    log('warn', 'Failed to check cost thresholds', { error: e.message });
    return { dailyCost: 0, weeklyCost: 0 };
  }
}

// Main loop
async function mainLoop() {
  if (mainLoop.running) {
    log('warn', 'Previous loop still running; skipping tick');
    return;
  }
  mainLoop.running = true;
  log('info', '=== Orchestrator Check ===');

  try {
    // 1. Cleanup running agents (stuck, completed, timed out)
    const killed = cleanupAgents();
    if (killed.length > 0) {
      log('info', 'Cleaned up agents', { killed });
    }

    // 1b. Cleanup old killed agents from state file
    cleanupKilledAgents();

    // 1c. Cleanup orphaned in_progress tasks (no active agent)
    const orphaned = cleanupOrphanedTasks();
    if (orphaned.length > 0) {
      log('info', `Cleaned up ${orphaned.length} orphaned tasks`, { orphaned });
    }

    // 2. Status with prioritization details
    const running = getRunningAgents();
    const pending = getPendingTasks();
    const allTasks = getTasks();

    // Get prioritized view of pending tasks
    const prioritized = prioritizeTasks(pending, allTasks);
    const prioritySummary = prioritized.slice(0, 5).map(t => ({
      id: t.id,
      subject: t.subject.substring(0, 40),
      score: t._priority.score,
      retries: t._priority.retries,
      blocked: t.blockedBy?.length || 0,
    }));

    log('info', 'Status', {
      running: running.length,
      pending: pending.length,
      viable: prioritized.filter(t => t._priority.score > 0).length,
      blocked: prioritized.filter(t => t._priority.score <= 0).length,
      agents: running.map(a => a.name),
      topTasks: prioritySummary,
    });

    // 3. Check for completion/idle states and notify
    const completedTasks = allTasks.filter(t => t.status === 'completed');
    const inProgressTasks = allTasks.filter(t => t.status === 'in_progress');

    // Check if all tasks are completed
    if (allTasks.length > 0 && pending.length === 0 && inProgressTasks.length === 0) {
      notifications.notifyAllTasksCompleted(completedTasks.length).catch(() => {});
    }
    // Check for idle state (no pending tasks, no running agents)
    else if (pending.length === 0 && running.length === 0 && allTasks.length > 0) {
      const viablePending = prioritized.filter(t => t._priority.score > 0);
      if (viablePending.length === 0 && prioritized.length > 0) {
        notifications.notifyIdleState('No viable tasks (all blocked or max retries exceeded)').catch(() => {});
      } else if (viablePending.length === 0) {
        notifications.notifyIdleState('No pending tasks available').catch(() => {});
      }
    }

    // 4. Check cost thresholds
    checkCostThresholds();

    // 5. Cleanup stale agents (every 30 min)
    runCleanupIfNeeded();

    // 6. Check for story research completion and trigger proposals
    // DISABLED: Auto-proposal causes runaway agent spawning. Use manual: rm proposal generate <story>
    // const proposalResults = checkStoryCompletionAndProposals();
    // if (proposalResults.length > 0) {
    //   log('info', 'Proposal actions taken', { results: proposalResults });
    // }
    const proposalResults = [];

    // 7. Auto-spawn (skip if we just triggered a proposal)
    if (proposalResults.some(r => r.success)) {
      log('info', 'Skipping auto-spawn - proposal generation in progress');
      return;
    }

    const spawnResult = autoSpawn();
    if (spawnResult.spawned) {
      log('info', 'Spawned agent', spawnResult);
    }
  } finally {
    mainLoop.running = false;
  }
}

// Utility: Reset retry count for a task (useful after fixing issues)
function resetTaskRetries(taskId) {
  const progress = readJSON(PROGRESS_FILE, { history: [] });
  const before = progress.history.length;

  // Remove failed entries for this task
  progress.history = progress.history.filter(h =>
    !(h.taskId === taskId && h.outcome !== 'completed')
  );

  const removed = before - progress.history.length;
  if (removed > 0) {
    writeJSON(PROGRESS_FILE, progress);
    log('info', `Reset retry count for task ${taskId}`, { removed });
  }
  return removed;
}

// Utility: Get task priorities (for debugging/CLI)
function getTaskPriorities() {
  const pending = getPendingTasks();
  const allTasks = getTasks();
  return prioritizeTasks(pending, allTasks);
}

// =============================================================================
// Story Completion Detection & Proposal Generation
// =============================================================================

// Extract story ID from task subject (e.g., "[S1][P1] Do something" -> "S1")
function getStoryFromTask(task) {
  const subject = task.subject || '';
  const match = subject.match(/\[S(\d+)\]/);
  return match ? `S${match[1]}` : null;
}

// Check if task is a research task (not implementation)
function isResearchTask(task) {
  const subject = (task.subject || '').toLowerCase();
  const researchKeywords = [
    'research', 'explore', 'analyze', 'investigate', 'study',
    'compare', 'evaluate', 'survey', 'deep dive', 'assessment'
  ];
  return researchKeywords.some(kw => subject.includes(kw));
}

// Get all tasks grouped by story
function getTasksByStory() {
  const tasks = getTasks();
  const stories = {};

  for (const task of tasks) {
    const storyId = getStoryFromTask(task);
    if (!storyId) continue;

    if (!stories[storyId]) {
      stories[storyId] = {
        storyId,
        allTasks: [],
        researchTasks: [],
        implementationTasks: [],
      };
    }

    stories[storyId].allTasks.push(task);

    if (isResearchTask(task)) {
      stories[storyId].researchTasks.push(task);
    } else {
      stories[storyId].implementationTasks.push(task);
    }
  }

  return stories;
}

// Check if a story's research is complete
function isStoryResearchComplete(storyId) {
  const stories = getTasksByStory();
  const story = stories[storyId];

  if (!story || story.researchTasks.length === 0) {
    return { complete: false, reason: 'no-research-tasks' };
  }

  const completed = story.researchTasks.filter(t => t.status === 'completed');
  const pending = story.researchTasks.filter(t => t.status === 'pending');
  const inProgress = story.researchTasks.filter(t => t.status === 'in_progress');

  if (pending.length > 0 || inProgress.length > 0) {
    return {
      complete: false,
      reason: 'tasks-remaining',
      completed: completed.length,
      pending: pending.length,
      inProgress: inProgress.length,
      total: story.researchTasks.length,
    };
  }

  return {
    complete: true,
    completed: completed.length,
    total: story.researchTasks.length,
    tasks: completed.map(t => ({ id: t.id, subject: t.subject })),
  };
}

// Load proposals state
function getProposals() {
  return readJSON(PROPOSALS_FILE, { proposals: {} });
}

// Save proposals state
function saveProposals(proposals) {
  writeJSON(PROPOSALS_FILE, proposals);
}

// Check if proposal already exists or is being generated for a story
function hasProposal(storyId) {
  const proposals = getProposals();
  const proposal = proposals.proposals?.[storyId];

  if (!proposal) return { exists: false };

  // If proposal is being generated, treat as exists to prevent duplicate spawning
  if (proposal.status === 'generating') {
    return { exists: true, status: 'generating', documentExists: false };
  }

  // Check if proposal document exists on disk
  const projectRoot = path.join(__dirname, '..', '..');
  const proposalPaths = [
    path.join(projectRoot, 'docs', storyId.toLowerCase(), 'PROPOSAL.md'),
    path.join(projectRoot, 'docs', `story_${storyId.toLowerCase()}`, 'PROPOSAL.md'),
    path.join(projectRoot, 'docs', 'firefly', 'PROPOSAL.md'),
  ];

  const documentExists = proposalPaths.some(p => fs.existsSync(p));

  return {
    exists: true,
    status: proposal.status,
    documentExists,
    agent: proposal.agent,
    updatedAt: proposal.updated_at || proposal.started_at,
  };
}

// Trigger proposal generation for a story
function triggerProposalGeneration(storyId, storyTitle = null) {
  log('info', `Triggering proposal generation for ${storyId}`, { storyTitle });

  // Use the rm script to spawn research lead
  const args = ['proposal', 'generate', storyId];
  if (storyTitle) {
    args.push('--title', storyTitle);
  }

  const result = spawnSync(RM_SCRIPT, args, {
    cwd: CONFIG.projectRoot,
    encoding: 'utf-8',
  });

  if (result.status === 0) {
    log('info', `Proposal generation started for ${storyId}`);

    // Notify about proposal generation
    notifications.notify('proposal_generating', {
      title: `Generating Proposal for ${storyId}`,
      message: `Research complete. Synthesizing findings into a proposal.`,
      eventKey: storyId,
    }).catch(() => {});

    return { success: true, output: result.stdout };
  } else {
    log('warn', `Failed to start proposal generation for ${storyId}`, {
      stderr: result.stderr,
    });
    return { success: false, error: result.stderr };
  }
}

// Track recent proposal generation attempts to prevent rapid re-triggering
const proposalGenerationCooldowns = {};
const PROPOSAL_COOLDOWN_MS = 5 * 60 * 1000; // 5 minutes cooldown

// Check all stories for research completion and trigger proposals
function checkStoryCompletionAndProposals() {
  const stories = getTasksByStory();
  const results = [];

  for (const [storyId, story] of Object.entries(stories)) {
    // Skip if no research tasks
    if (story.researchTasks.length === 0) continue;

    const researchStatus = isStoryResearchComplete(storyId);

    if (!researchStatus.complete) continue;

    // Check if proposal already exists
    const proposalStatus = hasProposal(storyId);

    if (proposalStatus.exists) {
      // Proposal exists - check status
      if (proposalStatus.status === 'pending_review' && proposalStatus.documentExists) {
        // Notify user that proposal is ready for review
        notifications.notify('proposal_ready', {
          title: `Proposal Ready: ${storyId}`,
          message: `Research proposal for ${storyId} is ready for your review.`,
          eventKey: `${storyId}-review`,
        }).catch(() => {});
      }
      continue;
    }

    // Check if a research-lead agent is already running for this story
    const agents = readJSON(AGENTS_FILE, {});
    const runningResearchLead = Object.entries(agents).some(([id, agent]) => {
      return agent.status === 'running' &&
             id.toLowerCase().includes(`research-lead-${storyId.toLowerCase()}`);
    });

    if (runningResearchLead) {
      log('debug', `Research lead already running for ${storyId}, skipping`);
      continue;
    }

    // Check cooldown - don't retry proposal generation too quickly
    const lastAttempt = proposalGenerationCooldowns[storyId];
    if (lastAttempt && (Date.now() - lastAttempt) < PROPOSAL_COOLDOWN_MS) {
      log('debug', `Proposal generation for ${storyId} on cooldown, skipping`);
      continue;
    }

    // Research complete, no proposal - trigger generation
    log('info', `Story ${storyId} research complete, generating proposal`, {
      completedTasks: researchStatus.completed,
    });

    // Set cooldown BEFORE attempting to prevent rapid retries
    proposalGenerationCooldowns[storyId] = Date.now();

    const genResult = triggerProposalGeneration(storyId);
    results.push({
      storyId,
      action: 'proposal-generation-triggered',
      success: genResult.success,
    });

    // Only trigger one proposal per cycle to prevent overload
    break;
  }

  return results;
}

// Start daemon
log('info', 'Orchestrator daemon starting', {
  lab: CURRENT_LAB_ID,
  labName: LAB_CONFIG.name,
  taskListId: TASK_LIST_ID,
  stateDir: STATE_DIR,
  interval: CONFIG.checkInterval / 1000 + 's',
  maxAgents: CONFIG.maxAgents,
});

// Initialize notifications config
notifications.ensureConfig();

// Run immediately
mainLoop();

// Then run on interval
setInterval(mainLoop, CONFIG.checkInterval);

// Handle shutdown
process.on('SIGINT', () => {
  log('info', 'Orchestrator shutting down');
  process.exit(0);
});

process.on('SIGTERM', () => {
  log('info', 'Orchestrator shutting down');
  process.exit(0);
});

log('info', `Orchestrator running for lab "${CURRENT_LAB_ID}" (checking every ${CONFIG.checkInterval/1000}s)`);
