#!/usr/bin/env node
/**
 * Show task priorities for debugging
 * Usage: node show-priorities.js
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const STATE_DIR = path.join(__dirname, 'state');
const PROGRESS_FILE = path.join(STATE_DIR, 'progress.json');
const AGENTS_FILE = path.join(STATE_DIR, 'agents.json');

const CONFIG = {
  prioritization: {
    maxRetries: 5,
    retryCooldownMinutes: 10,
    ageBonus: 1.25,
    maxAgeBonus: 30,
    blockerPenalty: 50,
    inProgressBlockerPenalty: 20,
    unblockBonus: 15,
  }
};

// Read JSON file safely
function readJSON(filePath, defaultValue = {}) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
  } catch (e) {}
  return defaultValue;
}

// Get tasks from Claude Code task list
function getTasks() {
  const tasksDir = path.join(os.homedir(), '.claude', 'tasks');
  if (!fs.existsSync(tasksDir)) return [];

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

  if (sessions.length === 0) return [];

  const sessionDir = path.join(tasksDir, sessions[0].name);
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

// Get running agents
function getRunningAgents() {
  const agents = readJSON(AGENTS_FILE, {});
  return Object.entries(agents)
    .filter(([_, a]) => a.status === 'running')
    .map(([name, a]) => ({ name, ...a }));
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

// Get pending tasks
function getPendingTasks() {
  const tasksWithAgents = getTasksWithRunningAgents();
  return getTasks().filter(t =>
    t.status === 'pending' &&
    !t.owner &&
    !tasksWithAgents.has(t.id)
  );
}

// Prioritize tasks (same logic as orchestrator.js)
function prioritizeTasks(pendingTasks, allTasks) {
  const completedTaskIds = new Set(allTasks.filter(t => t.status === 'completed').map(t => t.id));
  const inProgressTaskIds = new Set(allTasks.filter(t => t.status === 'in_progress').map(t => t.id));
  const progress = readJSON(PROGRESS_FILE, { history: [] });
  const now = Date.now();

  const retryCountsPerTask = {};
  for (const entry of progress.history) {
    if (entry.taskId && entry.outcome !== 'completed') {
      retryCountsPerTask[entry.taskId] = (retryCountsPerTask[entry.taskId] || 0) + 1;
    }
  }

  const lastFailureTime = {};
  for (const entry of progress.history) {
    if (entry.taskId && entry.outcome !== 'completed') {
      const ts = new Date(entry.timestamp).getTime();
      if (!lastFailureTime[entry.taskId] || ts > lastFailureTime[entry.taskId]) {
        lastFailureTime[entry.taskId] = ts;
      }
    }
  }

  const p = CONFIG.prioritization;

  const scoredTasks = pendingTasks.map(task => {
    let score = 100;
    const reasons = [];

    // 1. Manual priority override
    const priority = task.metadata?.priority;
    if (priority === 'critical') {
      score += 100;
      reasons.push('critical (+100)');
    } else if (priority === 'high') {
      score += 50;
      reasons.push('high (+50)');
    } else if (priority === 'low') {
      score -= 30;
      reasons.push('low (-30)');
    } else if (priority === 'blocked') {
      score -= 100;
      reasons.push('blocked (-100)');
    }

    // 2. Task dependencies
    if (task.blockedBy && task.blockedBy.length > 0) {
      const unresolvedBlockers = task.blockedBy.filter(
        blockerId => !completedTaskIds.has(blockerId)
      );
      if (unresolvedBlockers.length > 0) {
        const inProgressBlockers = unresolvedBlockers.filter(
          blockerId => inProgressTaskIds.has(blockerId)
        );
        const pendingBlockers = unresolvedBlockers.length - inProgressBlockers.length;
        const blockerPenalty = (pendingBlockers * p.blockerPenalty) + (inProgressBlockers.length * p.inProgressBlockerPenalty);
        score -= blockerPenalty;
        reasons.push(`blocked:${unresolvedBlockers.length} (-${blockerPenalty})`);
      }
    }

    // 3. Task age
    const createdAt = task.createdAt ? new Date(task.createdAt).getTime() : now;
    const ageMs = now - createdAt;
    const ageHours = ageMs / (1000 * 60 * 60);
    const ageBonus = Math.min(p.maxAgeBonus, Math.floor(ageHours * p.ageBonus));
    if (ageBonus > 0) {
      score += ageBonus;
      reasons.push(`age:${Math.round(ageHours)}h (+${ageBonus})`);
    }

    // 4. Retry history
    const retries = retryCountsPerTask[task.id] || 0;
    if (retries > 0) {
      const retryPenalty = Math.min(100, retries * retries * 5 + retries * 5);
      score -= retryPenalty;
      reasons.push(`retry:${retries} (-${retryPenalty})`);

      const lastFail = lastFailureTime[task.id];
      if (lastFail) {
        const minutesSinceFailure = (now - lastFail) / (1000 * 60);
        if (minutesSinceFailure < p.retryCooldownMinutes) {
          const cooldownPenalty = Math.floor((p.retryCooldownMinutes - minutesSinceFailure) * 5);
          score -= cooldownPenalty;
          reasons.push(`cooldown (-${cooldownPenalty})`);
        }
      }
    }

    // 5. Unblock bonus
    const blocksCount = allTasks.filter(t =>
      t.status === 'pending' &&
      t.blockedBy &&
      t.blockedBy.includes(task.id)
    ).length;
    if (blocksCount > 0) {
      const unblockBonus = blocksCount * p.unblockBonus;
      score += unblockBonus;
      reasons.push(`unblocks:${blocksCount} (+${unblockBonus})`);
    }

    // 6. Max retries circuit breaker
    if (retries >= p.maxRetries) {
      score = -999;
      reasons.push('MAX_RETRIES');
    }

    return {
      id: task.id,
      subject: task.subject,
      score,
      reasons,
      retries,
      blockedBy: task.blockedBy || [],
      priority: task.metadata?.priority,
    };
  });

  scoredTasks.sort((a, b) => b.score - a.score);
  return scoredTasks;
}

// Main
const allTasks = getTasks();
const pending = getPendingTasks();
const prioritized = prioritizeTasks(pending, allTasks);

console.log('\n=== Task Priorities ===\n');

if (prioritized.length === 0) {
  console.log('No pending tasks available for auto-spawn.\n');
  process.exit(0);
}

const viable = prioritized.filter(t => t.score > 0);
const blocked = prioritized.filter(t => t.score <= 0);

console.log(`Viable tasks (${viable.length}):`);
console.log('─'.repeat(80));

for (const task of viable) {
  const status = task.score > 100 ? '\x1b[32m[HIGH]\x1b[0m' :
                 task.score < 50 ? '\x1b[33m[LOW]\x1b[0m' : '[NORM]';
  console.log(`${status} #${task.id} (score: ${task.score})`);
  console.log(`    ${task.subject.substring(0, 70)}${task.subject.length > 70 ? '...' : ''}`);
  console.log(`    Factors: ${task.reasons.join(', ') || 'base score'}`);
  console.log('');
}

if (blocked.length > 0) {
  console.log(`\nBlocked/Max-Retries tasks (${blocked.length}):`);
  console.log('─'.repeat(80));

  for (const task of blocked) {
    console.log(`\x1b[31m[SKIP]\x1b[0m #${task.id} (score: ${task.score})`);
    console.log(`    ${task.subject.substring(0, 70)}${task.subject.length > 70 ? '...' : ''}`);
    console.log(`    Reason: ${task.reasons.join(', ')}`);
    console.log('');
  }
}

console.log('─'.repeat(80));
console.log(`Next task to spawn: #${viable[0]?.id || 'none'}`);
console.log('');
