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
  researchTimeout: 20, // minutes
  stuckTimeout: 10, // minutes no activity
  maxRuntime: 60, // minutes
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

const STATE_DIR = path.join(__dirname, 'state');
const AGENTS_FILE = path.join(STATE_DIR, 'agents.json');
const PROGRESS_FILE = path.join(STATE_DIR, 'progress.json');
const COST_FILE = path.join(STATE_DIR, 'cost-tracking.json');
const RESEARCH_STATE_FILE = path.join(STATE_DIR, 'research-state.json');
const RM_SCRIPT = path.join(__dirname, 'rm');
const ORCHESTRATOR_LOG = path.join(STATE_DIR, 'orchestrator.log');
const CODEX_CMD = 'codex';
const TASK_LIST_ID = process.env.CLAUDE_CODE_TASK_LIST_ID || 'voice-clone-pipeline';

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

// Get pending tasks (no owner, status pending, no agent already running)
function getPendingTasks() {
  const tasks = getTasks();
  const tasksWithAgents = getTasksWithRunningAgents();
  const completedTaskIds = new Set(tasks.filter(t => t.status === 'completed').map(t => String(t.id)));
  return tasks.filter(t =>
    t.status === 'pending' &&
    !t.owner &&
    !tasksWithAgents.has(String(t.id)) &&
    !(t.blockedBy || []).some(id => !completedTaskIds.has(String(id)))
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
    // Research timeout
    else if (health.isResearchTimeout) {
      shouldKill = true;
      reason = `Research timeout (${Math.round(health.runningMinutes)}min)`;
      outcome = 'timeout';
    }
    // Stuck (no activity)
    else if (health.isStuck) {
      shouldKill = true;
      reason = `No activity for ${Math.round(health.lastActivity)}min`;
      outcome = 'stuck';

      // Notify about stuck agent
      notifications.notifyAgentStuck(agent.name, health.lastActivity).catch(() => {});
    }
    // Max runtime exceeded
    else if (health.isTimedOut) {
      shouldKill = true;
      reason = `Max runtime exceeded (${Math.round(health.runningMinutes)}min)`;
      outcome = 'timeout';
    }

    if (shouldKill) {
      log('info', `Killing agent: ${agent.name}`, { reason });
      if (killAgent(agent.name)) {
        killed.push(agent.name);
        agents[agent.name].status = 'killed';
        agents[agent.name].killed_at = new Date().toISOString();
        const retryCount = recordOutcome(agent.name, taskId, outcome, health.runningMinutes, outcome === 'completed' ? 100 : 30);

        // Notify on task completion
        if (outcome === 'completed' && taskId) {
          const task = allTasks.find(t => t.id === taskId);
          notifications.notifyTaskCompleted(taskId, task?.subject || 'Unknown task', health.runningMinutes).catch(() => {});
        }

        // Notify on repeated errors
        if (outcome !== 'completed' && taskId && retryCount > 0) {
          const task = allTasks.find(t => t.id === taskId);
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

// Spawn agent for a task
function selectAgentType(task) {
  const subject = (task.subject || '').toLowerCase();
  const description = (task.description || '').toLowerCase();
  const text = `${subject} ${description}`;

  // Explicit override via tag in subject/description
  if (text.includes('[codex]') || text.includes('codex:') || text.includes('use codex')) {
    return 'codex';
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
    return 'codex';
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
    return 'codex';
  }

  // Default: local Ollama agent
  return 'ollama';
}

function spawnTaskAgent(task) {
  const agentName = `task-${task.id}-${Date.now()}`;
  let agentType = selectAgentType(task);
  if (agentType === 'codex' && !commandExists(CODEX_CMD)) {
    log('warn', 'Codex not found; falling back to ollama', { taskId: task.id });
    agentType = 'ollama';
  }
  const taskPrompt = `Work on this task from the shared task list (CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline"):

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
    return agentName;
  }
  return null;
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

    const agentName = spawnTaskAgent(task);
    if (agentName) {
      log('info', 'Spawned task agent', {
        name: agentName,
        task: task.subject,
        priority: task._priority.score,
      });

      // Notify about spawned agent
      notifications.notifyAgentSpawned(agentName, `#${task.id}: ${task.subject}`).catch(() => {});

      return {
        spawned: agentName,
        type: 'task',
        task: task.subject,
        priority: task._priority,
      };
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

    // 6. Auto-spawn
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

// Start daemon
log('info', 'Orchestrator daemon starting', {
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

log('info', `Orchestrator running (checking every ${CONFIG.checkInterval/1000}s)`);
