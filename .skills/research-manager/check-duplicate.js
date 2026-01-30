#!/usr/bin/env node
// Check if a task subject is a duplicate before creating it
// Usage: node check-duplicate.js "Task subject to check"
// Exits with code 0 if unique, 1 if duplicate

const { execSync } = require('child_process');

// Normalize a task subject for similarity comparison
function normalizeSubject(subject) {
  return (subject || '').toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\s+/g, ' ').trim();
}

// Check if two subjects are duplicates (>80% word overlap)
function areSubjectsSimilar(a, b) {
  const wordsA = new Set(normalizeSubject(a).split(' ').filter(w => w.length > 2));
  const wordsB = new Set(normalizeSubject(b).split(' ').filter(w => w.length > 2));
  if (wordsA.size === 0 || wordsB.size === 0) return false;
  const intersection = [...wordsA].filter(w => wordsB.has(w)).length;
  const smaller = Math.min(wordsA.size, wordsB.size);
  return intersection / smaller >= 0.8;
}

// Get all tasks
function getTasks() {
  try {
    const output = execSync('claude task list --format json', { encoding: 'utf-8' });
    return JSON.parse(output);
  } catch (err) {
    console.error('Failed to get tasks:', err.message);
    process.exit(2);
  }
}

function main() {
  const subjectToCheck = process.argv[2];

  if (!subjectToCheck) {
    console.error('Usage: check-duplicate.js "Task subject to check"');
    process.exit(2);
  }

  const tasks = getTasks();
  const activeTasks = tasks.filter(t => t.status !== 'completed' && t.status !== 'deleted');

  for (const task of activeTasks) {
    if (areSubjectsSimilar(subjectToCheck, task.subject)) {
      console.log(`DUPLICATE: Task #${task.id} - "${task.subject}" (${task.status})`);
      process.exit(1);
    }
  }

  console.log('OK: No duplicate found');
  process.exit(0);
}

if (require.main === module) {
  main();
}

module.exports = { areSubjectsSimilar, normalizeSubject };
