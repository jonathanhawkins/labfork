#!/usr/bin/env node
/**
 * Reset retry count for a specific task
 * Usage: node reset-retries.js <task-id>
 */

const fs = require('fs');
const path = require('path');

const STATE_DIR = path.join(__dirname, 'state');
const PROGRESS_FILE = path.join(STATE_DIR, 'progress.json');

// Read JSON file safely
function readJSON(filePath, defaultValue = {}) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    }
  } catch (e) {}
  return defaultValue;
}

// Write JSON file
function writeJSON(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

// Main
const taskId = process.argv[2];

if (!taskId) {
  console.error('Usage: node reset-retries.js <task-id>');
  process.exit(1);
}

const progress = readJSON(PROGRESS_FILE, { history: [] });
const before = progress.history.length;

// Count failures for this task before reset
const failuresBefore = progress.history.filter(h =>
  h.taskId === taskId && h.outcome !== 'completed'
).length;

if (failuresBefore === 0) {
  console.log(`Task #${taskId} has no retry history to reset.`);
  process.exit(0);
}

// Remove failed entries for this task (keep completed ones)
progress.history = progress.history.filter(h =>
  !(h.taskId === taskId && h.outcome !== 'completed')
);

const removed = before - progress.history.length;
writeJSON(PROGRESS_FILE, progress);

console.log(`Reset retry history for task #${taskId}:`);
console.log(`  - Removed ${removed} failure record(s)`);
console.log(`  - Task can now be retried by the orchestrator`);
