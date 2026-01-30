#!/usr/bin/env node
/**
 * Demo Page Sync System
 *
 * Scans all inference scripts and creates missing demo pages.
 * Can be run manually or as a periodic check.
 *
 * Usage:
 *   node sync-demos.js              # Dry run - show what would be created
 *   node sync-demos.js --execute    # Actually create missing pages
 *   node sync-demos.js --watch      # Run periodically (every 5 min)
 */

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

// Configuration
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const INFERENCE_DIR = path.join(PROJECT_ROOT, 'inference');
const DEMOS_DIR = path.join(PROJECT_ROOT, 'frontend', 'app', 'demos');
const API_DIR = path.join(PROJECT_ROOT, 'frontend', 'app', 'api', 'demos');
const GENERATOR_SCRIPT = path.join(__dirname, 'generate.py');
const STATE_FILE = path.join(__dirname, 'sync-state.json');

// Logging
function log(level, message) {
  const timestamp = new Date().toISOString().slice(11, 19);
  const prefix = {
    info: '\x1b[36m[INFO]\x1b[0m',
    warn: '\x1b[33m[WARN]\x1b[0m',
    error: '\x1b[31m[ERROR]\x1b[0m',
    success: '\x1b[32m[SUCCESS]\x1b[0m',
  }[level] || '[LOG]';
  console.log(`${timestamp} ${prefix} ${message}`);
}

// Load/save state
function loadState() {
  try {
    if (fs.existsSync(STATE_FILE)) {
      return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
    }
  } catch (e) {}
  return { lastSync: null, created: [], failed: [] };
}

function saveState(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

// Get all inference scripts
function getInferenceScripts() {
  if (!fs.existsSync(INFERENCE_DIR)) {
    log('warn', `Inference directory not found: ${INFERENCE_DIR}`);
    return [];
  }

  return fs.readdirSync(INFERENCE_DIR)
    .filter(f => f.startsWith('generate_with_') && f.endsWith('.py'))
    .map(f => {
      const technique = f.replace('generate_with_', '').replace('.py', '');
      return {
        name: technique,
        script: path.join(INFERENCE_DIR, f),
        demoPage: path.join(DEMOS_DIR, technique, 'page.tsx'),
        apiRoute: path.join(API_DIR, technique, 'route.ts'),
      };
    });
}

// Extract description from inference script
function getDescription(scriptPath) {
  try {
    const content = fs.readFileSync(scriptPath, 'utf8');
    // Look for docstring
    const docMatch = content.match(/"""([^"]+)"""/s);
    if (docMatch) {
      const firstLine = docMatch[1].trim().split('\n')[0];
      if (firstLine.length > 10 && firstLine.length < 200) {
        return firstLine;
      }
    }
    // Look for argparse description
    const argMatch = content.match(/description=['""]([^'"]+)['"]/);
    if (argMatch) {
      return argMatch[1];
    }
  } catch (e) {}
  return `Demo for ${path.basename(scriptPath, '.py').replace('generate_with_', '')}`;
}

// Check what's missing
function findMissingDemos() {
  const scripts = getInferenceScripts();
  const missing = [];

  for (const script of scripts) {
    const hasDemoPage = fs.existsSync(script.demoPage);
    const hasApiRoute = fs.existsSync(script.apiRoute);

    if (!hasDemoPage || !hasApiRoute) {
      missing.push({
        ...script,
        hasDemoPage,
        hasApiRoute,
        description: getDescription(script.script),
      });
    }
  }

  return { scripts, missing };
}

// Create a demo page
function createDemoPage(technique, description) {
  if (!fs.existsSync(GENERATOR_SCRIPT)) {
    log('error', `Generator script not found: ${GENERATOR_SCRIPT}`);
    return false;
  }

  const result = spawnSync('python3', [
    GENERATOR_SCRIPT,
    '--technique', technique,
    '--inference', `inference/generate_with_${technique}.py`,
    '--description', description,
  ], {
    cwd: PROJECT_ROOT,
    encoding: 'utf8',
    timeout: 120000, // 2 minute timeout
  });

  if (result.status === 0) {
    log('success', `Created demo page for: ${technique}`);
    return true;
  } else {
    log('error', `Failed to create ${technique}: ${result.stderr || result.stdout}`);
    return false;
  }
}

// Main sync function
function syncDemos(execute = false) {
  log('info', `Scanning for missing demo pages...`);

  const { scripts, missing } = findMissingDemos();
  const state = loadState();

  log('info', `Found ${scripts.length} inference scripts, ${missing.length} missing demos`);

  if (missing.length === 0) {
    log('success', 'All demo pages are up to date!');
    state.lastSync = new Date().toISOString();
    saveState(state);
    return { total: scripts.length, missing: 0, created: 0, failed: 0 };
  }

  console.log('\nMissing demo pages:');
  console.log('─'.repeat(60));
  for (const m of missing) {
    const status = [];
    if (!m.hasDemoPage) status.push('page');
    if (!m.hasApiRoute) status.push('api');
    console.log(`  ${m.name.padEnd(20)} missing: ${status.join(', ')}`);
  }
  console.log('─'.repeat(60));

  if (!execute) {
    log('info', 'Dry run - use --execute to create missing pages');
    return { total: scripts.length, missing: missing.length, created: 0, failed: 0 };
  }

  // Create missing pages
  let created = 0;
  let failed = 0;

  for (const m of missing) {
    log('info', `Creating demo for: ${m.name}`);

    if (createDemoPage(m.name, m.description)) {
      created++;
      state.created.push({ technique: m.name, date: new Date().toISOString() });
    } else {
      failed++;
      state.failed.push({ technique: m.name, date: new Date().toISOString() });
    }

    // Small delay between creations
    if (missing.indexOf(m) < missing.length - 1) {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 500);
    }
  }

  state.lastSync = new Date().toISOString();
  saveState(state);

  console.log('\n' + '═'.repeat(60));
  log('info', `Sync complete: ${created} created, ${failed} failed`);

  return { total: scripts.length, missing: missing.length, created, failed };
}

// Watch mode
function watchMode() {
  const INTERVAL = 5 * 60 * 1000; // 5 minutes

  log('info', 'Starting watch mode - checking every 5 minutes');
  log('info', 'Press Ctrl+C to stop');

  // Initial sync
  syncDemos(true);

  // Periodic check
  setInterval(() => {
    log('info', 'Periodic check...');
    syncDemos(true);
  }, INTERVAL);
}

// CLI
const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h')) {
  console.log(`
Demo Page Sync System

Usage:
  node sync-demos.js              Dry run - show what would be created
  node sync-demos.js --execute    Actually create missing pages
  node sync-demos.js --watch      Run periodically (every 5 min)
  node sync-demos.js --status     Show current sync status

Options:
  --execute    Create missing demo pages
  --watch      Continuous monitoring mode
  --status     Show sync state and history
  --help       Show this help
`);
  process.exit(0);
}

if (args.includes('--status')) {
  const state = loadState();
  console.log('\nSync Status:');
  console.log('─'.repeat(40));
  console.log(`Last sync: ${state.lastSync || 'Never'}`);
  console.log(`Created: ${state.created?.length || 0} pages`);
  console.log(`Failed: ${state.failed?.length || 0} attempts`);

  if (state.failed?.length > 0) {
    console.log('\nRecent failures:');
    state.failed.slice(-5).forEach(f => {
      console.log(`  - ${f.technique} (${f.date})`);
    });
  }
  process.exit(0);
}

if (args.includes('--watch')) {
  watchMode();
} else {
  const result = syncDemos(args.includes('--execute'));
  process.exit(result.failed > 0 ? 1 : 0);
}
