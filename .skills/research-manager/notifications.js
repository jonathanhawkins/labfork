/**
 * Notification System for Orchestrator Events
 *
 * Supports:
 * - Desktop notifications (macOS/Linux)
 * - Sound alerts (macOS/Linux)
 * - Webhook notifications (Slack/Discord)
 *
 * Events:
 * - all_tasks_completed: All tasks in the list are done
 * - agent_stuck: Agent has been stuck (no activity) for too long
 * - agent_error_repeated: Same task has failed multiple times
 * - cost_threshold_exceeded: Daily/weekly cost threshold exceeded
 * - idle_state: No pending tasks and no running agents
 * - task_completed: Individual task completed
 * - agent_spawned: New agent started
 * - proposal_generating: Research lead is synthesizing findings
 * - proposal_ready: Proposal is ready for user review
 */

const { exec, execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

const STATE_DIR = path.join(__dirname, 'state');
const CONFIG_FILE = path.join(STATE_DIR, 'notifications-config.json');
const NOTIFICATION_LOG = path.join(STATE_DIR, 'notifications.log');
const COOLDOWN_FILE = path.join(STATE_DIR, 'notification-cooldowns.json');

// Default configuration
const DEFAULT_CONFIG = {
  enabled: true,

  // Desktop notifications (macOS terminal-notifier or Linux notify-send)
  desktop: {
    enabled: true,
    // Which events trigger desktop notifications
    events: ['all_tasks_completed', 'agent_stuck', 'agent_error_repeated', 'cost_threshold_exceeded', 'idle_state', 'proposal_ready'],
  },

  // Sound alerts
  sound: {
    enabled: true,
    // macOS: use system sounds or say command
    // Linux: use paplay, aplay, or speaker-test
    events: ['all_tasks_completed', 'cost_threshold_exceeded'],
    // Custom sound file path (optional)
    customSound: null,
  },

  // Webhook notifications (Slack/Discord)
  webhook: {
    enabled: false,
    // Webhook URL (Slack incoming webhook or Discord webhook)
    url: null,
    // Type: 'slack' or 'discord'
    type: 'slack',
    // Which events trigger webhook notifications
    events: ['all_tasks_completed', 'agent_error_repeated', 'cost_threshold_exceeded'],
    // Mention users on certain events (Slack: <@USER_ID>, Discord: <@USER_ID>)
    mentions: {
      cost_threshold_exceeded: null, // e.g., '<@U123ABC>'
    },
  },

  // Cooldown periods (in minutes) to prevent notification spam
  cooldowns: {
    idle_state: 30,           // Only notify about idle state every 30 mins
    agent_stuck: 10,          // Notify about same stuck agent every 10 mins
    agent_error_repeated: 15, // Notify about repeated errors every 15 mins
    cost_threshold_exceeded: 60, // Cost alerts every hour
    all_tasks_completed: 5,   // Task completion every 5 mins
    task_completed: 0,        // No cooldown for individual tasks
    agent_spawned: 0,         // No cooldown for spawned agents
  },

  // Thresholds for events
  thresholds: {
    // Notify after this many consecutive failures for same task
    errorRepeatCount: 3,
    // Cost thresholds (USD)
    dailyCostWarning: 50,
    dailyCostAlert: 100,
    weeklyCostWarning: 300,
    weeklyCostAlert: 500,
  },
};

// Load configuration
function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      const userConfig = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf-8'));
      // Deep merge with defaults
      return deepMerge(DEFAULT_CONFIG, userConfig);
    }
  } catch (e) {
    logNotification('warn', 'Failed to load config, using defaults', { error: e.message });
  }
  return DEFAULT_CONFIG;
}

// Save configuration (creates default if doesn't exist)
function saveConfig(config) {
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
}

// Deep merge objects (preserves arrays)
function deepMerge(target, source) {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    // Arrays should replace, not merge
    if (Array.isArray(source[key])) {
      result[key] = [...source[key]];
    } else if (source[key] instanceof Object && !Array.isArray(source[key]) &&
               key in target && target[key] instanceof Object && !Array.isArray(target[key])) {
      result[key] = deepMerge(target[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

// Log notification events
function logNotification(level, message, data = {}) {
  const timestamp = new Date().toISOString();
  const logLine = `[${timestamp}] [${level.toUpperCase()}] ${message} ${Object.keys(data).length ? JSON.stringify(data) : ''}\n`;
  fs.appendFileSync(NOTIFICATION_LOG, logLine);
}

// Check and update cooldown
function checkCooldown(eventType, eventKey = null) {
  const cooldowns = loadCooldowns();
  const config = loadConfig();
  const cooldownMinutes = config.cooldowns[eventType] || 0;

  if (cooldownMinutes === 0) return true; // No cooldown

  const key = eventKey ? `${eventType}:${eventKey}` : eventType;
  const now = Date.now();
  const lastNotified = cooldowns[key] || 0;
  const cooldownMs = cooldownMinutes * 60 * 1000;

  if (now - lastNotified < cooldownMs) {
    return false; // Still in cooldown
  }

  // Update cooldown
  cooldowns[key] = now;
  saveCooldowns(cooldowns);
  return true;
}

function loadCooldowns() {
  try {
    if (fs.existsSync(COOLDOWN_FILE)) {
      return JSON.parse(fs.readFileSync(COOLDOWN_FILE, 'utf-8'));
    }
  } catch (e) {}
  return {};
}

function saveCooldowns(cooldowns) {
  // Clean up old cooldowns (older than 24 hours)
  const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
  for (const key of Object.keys(cooldowns)) {
    if (cooldowns[key] < oneDayAgo) {
      delete cooldowns[key];
    }
  }
  fs.writeFileSync(COOLDOWN_FILE, JSON.stringify(cooldowns, null, 2));
}

// Detect platform
function getPlatform() {
  const platform = process.platform;
  if (platform === 'darwin') return 'macos';
  if (platform === 'linux') return 'linux';
  if (platform === 'win32') return 'windows';
  return 'unknown';
}

// Send desktop notification
async function sendDesktopNotification(title, message, options = {}) {
  const platform = getPlatform();

  try {
    if (platform === 'macos') {
      // Use osascript for macOS notifications
      const subtitle = options.subtitle || '';
      const sound = options.sound !== false ? '-sound default' : '';
      const script = `display notification "${message.replace(/"/g, '\\"')}" with title "${title.replace(/"/g, '\\"')}"${subtitle ? ` subtitle "${subtitle.replace(/"/g, '\\"')}"` : ''}`;
      execSync(`osascript -e '${script}'`);
      logNotification('info', 'Desktop notification sent (macOS)', { title, message });
    } else if (platform === 'linux') {
      // Use notify-send for Linux
      const urgency = options.urgency || 'normal'; // low, normal, critical
      execSync(`notify-send -u ${urgency} "${title}" "${message}"`);
      logNotification('info', 'Desktop notification sent (Linux)', { title, message });
    } else {
      logNotification('warn', 'Desktop notifications not supported on this platform', { platform });
    }
  } catch (e) {
    logNotification('error', 'Failed to send desktop notification', { error: e.message });
  }
}

// Play sound alert
async function playSound(options = {}) {
  const platform = getPlatform();
  const config = loadConfig();

  try {
    if (platform === 'macos') {
      if (config.sound.customSound && fs.existsSync(config.sound.customSound)) {
        execSync(`afplay "${config.sound.customSound}"`);
      } else {
        // Use system sound
        execSync(`afplay /System/Library/Sounds/Glass.aiff`);
      }
      logNotification('info', 'Sound alert played (macOS)');
    } else if (platform === 'linux') {
      if (config.sound.customSound && fs.existsSync(config.sound.customSound)) {
        execSync(`paplay "${config.sound.customSound}" 2>/dev/null || aplay "${config.sound.customSound}" 2>/dev/null`);
      } else {
        // Try to play a system sound or use speaker-test
        execSync(`paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null || speaker-test -t sine -f 1000 -l 1 2>/dev/null &`);
      }
      logNotification('info', 'Sound alert played (Linux)');
    }
  } catch (e) {
    logNotification('warn', 'Failed to play sound', { error: e.message });
  }
}

// Send webhook notification (Slack/Discord)
async function sendWebhook(eventType, title, message, options = {}) {
  const config = loadConfig();

  if (!config.webhook.enabled || !config.webhook.url) {
    return;
  }

  const mention = config.webhook.mentions[eventType] || '';

  try {
    let payload;

    if (config.webhook.type === 'discord') {
      // Discord webhook format
      payload = JSON.stringify({
        content: mention ? `${mention} ` : '',
        embeds: [{
          title: title,
          description: message,
          color: getColorForEvent(eventType),
          timestamp: new Date().toISOString(),
          footer: {
            text: 'Research Manager Orchestrator',
          },
          fields: options.fields || [],
        }],
      });
    } else {
      // Slack webhook format (default)
      payload = JSON.stringify({
        text: mention ? `${mention} ${title}` : title,
        blocks: [
          {
            type: 'header',
            text: {
              type: 'plain_text',
              text: title,
              emoji: true,
            },
          },
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: message,
            },
          },
          ...(options.fields || []).map(field => ({
            type: 'section',
            fields: [{
              type: 'mrkdwn',
              text: `*${field.name}*\n${field.value}`,
            }],
          })),
        ],
      });
    }

    const url = new URL(config.webhook.url);
    const requestOptions = {
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname + url.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
      },
    };

    const lib = url.protocol === 'https:' ? https : http;

    await new Promise((resolve, reject) => {
      const req = lib.request(requestOptions, (res) => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          logNotification('info', 'Webhook notification sent', { eventType, title });
          resolve();
        } else {
          reject(new Error(`Webhook returned status ${res.statusCode}`));
        }
      });

      req.on('error', reject);
      req.write(payload);
      req.end();
    });
  } catch (e) {
    logNotification('error', 'Failed to send webhook', { error: e.message });
  }
}

// Get color code for event type (Discord embeds)
function getColorForEvent(eventType) {
  const colors = {
    all_tasks_completed: 0x00FF00, // Green
    task_completed: 0x00CC00,       // Light green
    agent_spawned: 0x0099FF,        // Blue
    idle_state: 0xFFAA00,           // Orange
    agent_stuck: 0xFF6600,          // Dark orange
    agent_error_repeated: 0xFF3300, // Red-orange
    cost_threshold_exceeded: 0xFF0000, // Red
    proposal_generating: 0x9966FF,  // Purple
    proposal_ready: 0x00FFAA,       // Teal
  };
  return colors[eventType] || 0x808080; // Default gray
}

// Get emoji for event type
function getEmoji(eventType) {
  const emojis = {
    all_tasks_completed: '🎉',
    task_completed: '✅',
    agent_spawned: '🚀',
    idle_state: '😴',
    agent_stuck: '⚠️',
    agent_error_repeated: '🔴',
    cost_threshold_exceeded: '💰',
    proposal_generating: '📝',
    proposal_ready: '📋',
  };
  return emojis[eventType] || '📢';
}

/**
 * Main notification function
 *
 * @param {string} eventType - Type of event
 * @param {object} data - Event data
 * @param {string} data.title - Notification title
 * @param {string} data.message - Notification message
 * @param {string} [data.eventKey] - Unique key for cooldown (e.g., task ID)
 * @param {object} [data.fields] - Additional fields for webhook
 */
async function notify(eventType, data) {
  const config = loadConfig();

  if (!config.enabled) {
    return;
  }

  // Check cooldown
  if (!checkCooldown(eventType, data.eventKey)) {
    logNotification('debug', 'Notification skipped (cooldown)', { eventType, eventKey: data.eventKey });
    return;
  }

  const emoji = getEmoji(eventType);
  const title = `${emoji} ${data.title}`;

  // Desktop notification
  if (config.desktop.enabled && config.desktop.events.includes(eventType)) {
    await sendDesktopNotification(title, data.message, data);
  }

  // Sound alert
  if (config.sound.enabled && config.sound.events.includes(eventType)) {
    await playSound(data);
  }

  // Webhook
  if (config.webhook.enabled && config.webhook.events.includes(eventType)) {
    await sendWebhook(eventType, title, data.message, data);
  }

  logNotification('info', 'Notification processed', { eventType, title: data.title });
}

// Convenience functions for specific events

async function notifyAllTasksCompleted(taskCount) {
  await notify('all_tasks_completed', {
    title: 'All Tasks Completed!',
    message: `All ${taskCount} tasks have been completed. The orchestrator is now idle.`,
    fields: [
      { name: 'Total Tasks', value: String(taskCount) },
    ],
  });
}

async function notifyTaskCompleted(taskId, taskSubject, duration) {
  await notify('task_completed', {
    title: `Task #${taskId} Completed`,
    message: `${taskSubject}`,
    eventKey: taskId,
    fields: [
      { name: 'Task', value: `#${taskId}: ${taskSubject}` },
      { name: 'Duration', value: `${Math.round(duration)} minutes` },
    ],
  });
}

async function notifyAgentSpawned(agentName, taskInfo) {
  await notify('agent_spawned', {
    title: 'Agent Spawned',
    message: `${agentName} started${taskInfo ? ` for: ${taskInfo}` : ''}`,
    eventKey: agentName,
    fields: taskInfo ? [{ name: 'Task', value: taskInfo }] : [],
  });
}

async function notifyAgentStuck(agentName, inactiveMinutes) {
  await notify('agent_stuck', {
    title: 'Agent Stuck',
    message: `${agentName} has been inactive for ${Math.round(inactiveMinutes)} minutes`,
    eventKey: agentName,
    urgency: 'critical',
    fields: [
      { name: 'Agent', value: agentName },
      { name: 'Inactive', value: `${Math.round(inactiveMinutes)} minutes` },
    ],
  });
}

async function notifyAgentErrorRepeated(taskId, taskSubject, retryCount) {
  const config = loadConfig();
  if (retryCount < config.thresholds.errorRepeatCount) {
    return; // Don't notify until threshold reached
  }

  await notify('agent_error_repeated', {
    title: `Task #${taskId} Failing Repeatedly`,
    message: `${taskSubject} has failed ${retryCount} times`,
    eventKey: taskId,
    urgency: 'critical',
    fields: [
      { name: 'Task', value: `#${taskId}: ${taskSubject}` },
      { name: 'Failures', value: String(retryCount) },
    ],
  });
}

async function notifyCostThreshold(thresholdType, currentCost, threshold) {
  const isAlert = thresholdType.includes('Alert');
  await notify('cost_threshold_exceeded', {
    title: `Cost ${isAlert ? 'Alert' : 'Warning'}: ${thresholdType}`,
    message: `Current cost: $${currentCost.toFixed(2)} (threshold: $${threshold})`,
    eventKey: thresholdType,
    urgency: isAlert ? 'critical' : 'normal',
    fields: [
      { name: 'Current Cost', value: `$${currentCost.toFixed(2)}` },
      { name: 'Threshold', value: `$${threshold}` },
      { name: 'Type', value: thresholdType },
    ],
  });
}

async function notifyIdleState(reason) {
  await notify('idle_state', {
    title: 'Orchestrator Idle',
    message: reason || 'No pending tasks and no running agents',
    fields: reason ? [{ name: 'Reason', value: reason }] : [],
  });
}

async function notifyProposalReady(storyId, storyTitle) {
  await notify('proposal_ready', {
    title: `Proposal Ready: ${storyId}`,
    message: `Research proposal for "${storyTitle || storyId}" is ready for your review.`,
    eventKey: storyId,
    fields: [
      { name: 'Story', value: storyId },
      { name: 'Action', value: 'Review with: .skills/research-manager/rm proposal status ' + storyId },
    ],
  });
}

async function notifyProposalGenerating(storyId, taskCount) {
  await notify('proposal_generating', {
    title: `Generating Proposal: ${storyId}`,
    message: `Synthesizing ${taskCount} research findings into a proposal.`,
    eventKey: storyId,
    fields: [
      { name: 'Story', value: storyId },
      { name: 'Research Tasks', value: String(taskCount) },
    ],
  });
}

// Initialize config file if it doesn't exist
function ensureConfig() {
  if (!fs.existsSync(CONFIG_FILE)) {
    saveConfig(DEFAULT_CONFIG);
    logNotification('info', 'Created default notification config', { path: CONFIG_FILE });
  }
}

// Ensure state directory exists
if (!fs.existsSync(STATE_DIR)) {
  fs.mkdirSync(STATE_DIR, { recursive: true });
}

// Export functions
module.exports = {
  notify,
  notifyAllTasksCompleted,
  notifyTaskCompleted,
  notifyAgentSpawned,
  notifyAgentStuck,
  notifyAgentErrorRepeated,
  notifyCostThreshold,
  notifyIdleState,
  notifyProposalReady,
  notifyProposalGenerating,
  loadConfig,
  saveConfig,
  ensureConfig,
  DEFAULT_CONFIG,
};

// CLI interface for testing
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  ensureConfig();

  if (command === 'test') {
    console.log('Testing notifications...');
    (async () => {
      await sendDesktopNotification('Test Notification', 'This is a test message from the orchestrator');
      await playSound();
      console.log('Test complete. Check for desktop notification and sound.');
    })();
  } else if (command === 'config') {
    console.log('Current configuration:');
    console.log(JSON.stringify(loadConfig(), null, 2));
  } else if (command === 'reset') {
    saveConfig(DEFAULT_CONFIG);
    console.log('Configuration reset to defaults');
  } else {
    console.log('Notification System for Orchestrator');
    console.log('');
    console.log('Usage:');
    console.log('  node notifications.js test    - Test desktop notification and sound');
    console.log('  node notifications.js config  - Show current configuration');
    console.log('  node notifications.js reset   - Reset to default configuration');
    console.log('');
    console.log('Configuration file:', CONFIG_FILE);
  }
}
