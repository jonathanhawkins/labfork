#!/usr/bin/env node
// Task deduplication utility for Claude Code task list
// Usage: node dedup-tasks.js [--dry-run] [--auto]

const fs = require('fs');
const path = require('path');
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

// Get all tasks from Claude Code task list
function getTasks() {
  try {
    const output = execSync('claude task list --format json', { encoding: 'utf-8' });
    return JSON.parse(output);
  } catch (err) {
    console.error('Failed to get tasks:', err.message);
    console.error('Make sure you are in a Claude Code session with CLAUDE_CODE_TASK_LIST_ID set');
    process.exit(1);
  }
}

// Delete a task by ID
function deleteTask(taskId, dryRun = false) {
  if (dryRun) {
    console.log(`  [DRY RUN] Would delete task #${taskId}`);
    return true;
  }
  try {
    execSync(`claude task update ${taskId} --status deleted`, { encoding: 'utf-8' });
    return true;
  } catch (err) {
    console.error(`  Failed to delete task #${taskId}:`, err.message);
    return false;
  }
}

// Find and group duplicate tasks
function findDuplicates(tasks) {
  const groups = [];
  const seen = new Map();

  // Skip completed tasks - only dedup pending/in_progress
  const activeTasks = tasks.filter(t => t.status !== 'completed');

  for (const task of activeTasks) {
    let foundGroup = false;

    // Check against existing groups
    for (const group of groups) {
      if (areSubjectsSimilar(task.subject, group[0].subject)) {
        group.push(task);
        foundGroup = true;
        break;
      }
    }

    // Create new group if no match
    if (!foundGroup) {
      groups.push([task]);
    }
  }

  // Return only groups with duplicates
  return groups.filter(g => g.length > 1);
}

// Main function
function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const auto = args.includes('--auto');

  console.log('Task Deduplication Utility');
  console.log('==========================\n');

  const tasks = getTasks();
  console.log(`Total tasks: ${tasks.length}`);

  const duplicateGroups = findDuplicates(tasks);

  if (duplicateGroups.length === 0) {
    console.log('\n✓ No duplicates found!');
    return;
  }

  console.log(`\nFound ${duplicateGroups.length} groups with duplicates:\n`);

  let totalDupes = 0;
  for (const group of duplicateGroups) {
    totalDupes += group.length - 1;

    console.log(`Group: "${group[0].subject.substring(0, 60)}..."`);
    console.log(`  Total: ${group.length} tasks`);

    // Sort by ID (oldest first)
    group.sort((a, b) => parseInt(a.id) - parseInt(b.id));

    const keep = group[0];
    const toDelete = group.slice(1);

    console.log(`  Keep: #${keep.id} (${keep.status})`);
    toDelete.forEach(t => {
      console.log(`  Delete: #${t.id} (${t.status})`);
    });
    console.log();

    if (auto || dryRun) {
      toDelete.forEach(t => deleteTask(t.id, dryRun));
    }
  }

  console.log(`\nSummary: ${totalDupes} duplicate tasks to remove`);

  if (dryRun) {
    console.log('\n[DRY RUN] No tasks were actually deleted.');
    console.log('Run without --dry-run to delete duplicates.');
  } else if (auto) {
    console.log('\n✓ Duplicates removed!');
  } else {
    console.log('\nRun with --auto to delete duplicates automatically.');
    console.log('Or run with --dry-run to preview changes first.');
  }
}

if (require.main === module) {
  main();
}
