import { NextResponse } from 'next/server';
import { readFileSync, readdirSync, existsSync, statSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';
import { execSync } from 'child_process';

export const dynamic = "force-dynamic";

// Activity state from various sources
interface ActivityState {
  id: string;
  type: string;
  active: boolean;
  progress?: number;
  metrics?: Record<string, number | string>;
  message?: string;
  startedAt?: string;
  assignedAgent?: string;
}

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8003';
const HAS_BACKEND = Boolean(process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL);

// Public agent state API (for Vercel deployment - can't SSH from cloud)
const AGENT_STATE_URL = process.env.AGENT_STATE_URL || '';

function normalizeAgentId(raw?: string | null): string | undefined {
  if (!raw) return undefined;
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  return trimmed.startsWith('rm:') ? trimmed.slice(3) : trimmed;
}

function getRunningAgentIdsFromAgents(agents: Record<string, any>): Set<string> {
  const running = new Set<string>();
  Object.entries(agents).forEach(([name, agent]: [string, any]) => {
    if (agent?.status === 'running') {
      const id = normalizeAgentId(name);
      if (id) running.add(id);
    }
  });
  return running;
}

function getLocalRunningAgentIds(): Set<string> {
  const running = new Set<string>();
  try {
    const projectRoot = join(process.cwd(), '..');
    const agentStatePath = join(
      projectRoot,
      '.skills',
      'research-manager',
      'state',
      'agents.json'
    );
    if (!existsSync(agentStatePath)) return running;
    const agents = JSON.parse(readFileSync(agentStatePath, 'utf-8'));
    return getRunningAgentIdsFromAgents(agents);
  } catch {
    return running;
  }
  return running;
}

/**
 * Fetch training status from backend
 */
async function fetchTrainingStatus(): Promise<ActivityState | null> {
  try {
    const response = await fetch(`${BACKEND_URL}/training/status`, {
      next: { revalidate: 0 },
    });
    if (!response.ok) return null;

    const data = await response.json();

    if (data.status === 'running' || data.status === 'training') {
      return {
        id: 'training-active',
        type: 'training',
        active: true,
        progress: data.progress ?? 0,
        metrics: {
          epoch: data.epoch ?? 0,
          loss: data.loss ?? 0,
          learningRate: data.learning_rate ?? 0,
        },
        message: `Training epoch ${data.epoch ?? 0} - Loss: ${(data.loss ?? 0).toFixed(4)}`,
        startedAt: data.started_at,
        assignedAgent: 'codex',  // Assign training to Codex agent
      };
    }

    return null;
  } catch (error) {
    console.error('[Activities API] Training status fetch failed:', error);
    return null;
  }
}

/**
 * Fetch recording sessions from backend
 */
async function fetchRecordingStatus(): Promise<ActivityState | null> {
  try {
    const response = await fetch(`${BACKEND_URL}/voice-recording/sessions`, {
      next: { revalidate: 0 },
    });
    if (!response.ok) return null;

    const data = await response.json();

    // Find any recent active sessions (within last 5 minutes)
    const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
    const activeSessions = (data.sessions || []).filter((session: any) => {
      const sessionTime = new Date(session.created_at || 0).getTime();
      return sessionTime > fiveMinutesAgo;
    });

    if (activeSessions.length > 0) {
      const latestSession = activeSessions[0];
      const progress =
        ((latestSession.recorded_count || 0) / (latestSession.total_lines || 1)) * 100;

      return {
        id: `recording-${latestSession.session_id}`,
        type: 'recording',
        active: true,
        progress: Math.round(progress),
        metrics: {
          recorded: latestSession.recorded_count || 0,
          total: latestSession.total_lines || 0,
        },
        message: `Recording ${latestSession.recorded_count}/${latestSession.total_lines} samples`,
        startedAt: latestSession.created_at,
        assignedAgent: 'opus',  // Assign recording to Opus agent
      };
    }

    return null;
  } catch (error) {
    console.error('[Activities API] Recording status fetch failed:', error);
    return null;
  }
}

/**
 * Fetch live transform status from backend
 */
async function fetchLiveTransformStatus(): Promise<ActivityState | null> {
  try {
    const response = await fetch(`${BACKEND_URL}/live-transform/status`, {
      next: { revalidate: 0 },
    });
    if (!response.ok) return null;

    const data = await response.json();

    if (data.active || data.initialized) {
      return {
        id: 'live-transform-active',
        type: 'live-transform',
        active: data.active || false,
        metrics: {
          emotion: data.emotion || 'neutral',
          intensity: data.intensity || 0,
        },
        message: `Live transform: ${data.emotion || 'neutral'} (${Math.round((data.intensity || 0) * 100)}%)`,
        assignedAgent: 'explorer',  // Assign live transform to Explorer agent
      };
    }

    return null;
  } catch (error) {
    console.error('[Activities API] Live transform status fetch failed:', error);
    return null;
  }
}

async function fetchRemoteTasks(): Promise<any[] | null> {
  if (!HAS_BACKEND) return null;
  try {
    const response = await fetch(`${BACKEND_URL}/api/lab/tasks`, {
      cache: 'no-store',
    });
    if (!response.ok) return null;
    const data = await response.json();
    return data.tasks || null;
  } catch {
    return null;
  }
}

async function fetchRemoteAgents(): Promise<Record<string, any> | null> {
  if (!HAS_BACKEND) return null;
  try {
    const response = await fetch(`${BACKEND_URL}/api/lab/research-agents`, {
      cache: 'no-store',
    });
    if (!response.ok) return null;
    const data = await response.json();
    return data.agents || null;
  } catch {
    return null;
  }
}

/**
 * Fetch Claude Code tasks
 */
function buildTaskActivities(
  tasks: any[],
  runningAgentIds?: Set<string>
): ActivityState[] {
  const activities: ActivityState[] = [];

  if (!tasks || tasks.length === 0) return activities;

  const inProgressTasks = tasks.filter(
    (t: any) => t.status === 'in_progress'
  );

  inProgressTasks.forEach((task: any, index: number) => {
    const owner = normalizeAgentId(task.owner);
    if (!owner) {
      return;
    }
    if (runningAgentIds && runningAgentIds.size > 0 && !runningAgentIds.has(owner)) {
      return;
    }
    activities.push({
      id: `task-${task.id}`,
      type: 'task',
      active: true,
      message: task.activeForm || task.subject || 'Working on task',
      assignedAgent: owner || (index === 0 ? 'planner' : 'explorer'),
    });
  });

  return activities;
}

async function fetchTaskActivities(): Promise<ActivityState[]> {
  try {
    const runningAgentIds = getLocalRunningAgentIds();
    if (runningAgentIds.size === 0) return [];
    const tasksDir = join(homedir(), '.claude', 'tasks');
    if (!existsSync(tasksDir)) return [];

    const sessions = readdirSync(tasksDir).filter((name) => {
      const fullPath = join(tasksDir, name);
      return statSync(fullPath).isDirectory() && !name.startsWith('.');
    });

    const sessionTasks = sessions
      .map((sessionId) => {
        const sessionDir = join(tasksDir, sessionId);
        try {
          const taskFiles = readdirSync(sessionDir).filter(
            (f) => f.endsWith('.json') && !f.startsWith('.')
          );

          const tasks = taskFiles.map((file) => {
            const content = readFileSync(join(sessionDir, file), 'utf-8');
            return JSON.parse(content);
          });

          const stat = statSync(sessionDir);
          return {
            sessionId,
            tasks,
            modifiedAt: stat.mtime,
          };
        } catch {
          return { sessionId, tasks: [], modifiedAt: new Date(0) };
        }
      })
      .filter((s) => s.tasks.length > 0)
      .sort((a, b) => b.modifiedAt.getTime() - a.modifiedAt.getTime());

    if (sessionTasks.length === 0) return [];
    return buildTaskActivities(sessionTasks[0].tasks, runningAgentIds);
  } catch (error) {
    console.error('[Activities API] Task fetch failed:', error);
  }

  return [];
}

/**
 * Fetch Research Manager agent state
 */
function buildResearchAgentActivities(agents: Record<string, any>): ActivityState[] {
  const activities: ActivityState[] = [];

  if (!agents) return activities;

  Object.entries(agents).forEach(([name, agent]: [string, any]) => {
    if (agent.status === 'running') {
      const agentId = normalizeAgentId(name);
      // Determine which 3D agent this maps to based on agent type/name
      let assignedAgent = 'explorer';
      let activityType = 'task';
      const taskMatch = agent.task?.match(/TASK #\d+:\s*(.+?)(?:\\n|\n|$)/);
      let message = 'Researching...';
      if (taskMatch) {
        message = taskMatch[1].trim().substring(0, 60);
      } else if (agent.task && !agent.task.startsWith('⚠') && !agent.task.includes('CRITICAL')) {
        message = agent.task.substring(0, 50);
      }

      // The auto-improver is the Manager - it orchestrates everything
      if (name === 'auto-improver' || name.includes('manager') || name.includes('loop')) {
        assignedAgent = 'manager';  // Special - the Manager agent
        activityType = 'processing';
        // Extract a short meaningful message from the task
        const taskText = agent.task || '';
        if (taskText.includes('research')) {
          message = 'Researching improvements...';
        } else if (taskText.includes('training') || taskText.includes('train')) {
          message = 'Managing training run...';
        } else {
          message = 'Orchestrating the lab...';
        }
      } else if (agentId) {
        assignedAgent = agentId;
      } else if (agent.type === 'codex' || name.includes('codex') || name.includes('analyzer')) {
        assignedAgent = 'codex';
        activityType = 'inference';
      } else if (agent.type === 'opus' || name.includes('opus') || name.includes('impl')) {
        assignedAgent = 'opus';
        activityType = 'task';
      }

      activities.push({
        id: `research-${name}`,
        type: activityType,
        active: true,
        message,
        startedAt: agent.started_at,
        assignedAgent,
      });
    }
  });

  return activities;
}

async function fetchResearchAgentActivities(): Promise<ActivityState[]> {
  const REMOTE_HOST = 'doc@100.83.78.111';
  const REMOTE_AGENTS_FILE = '~/dev/voice-clone-pipeline/.skills/research-manager/state/agents.json';

  // Try local first
  try {
    const projectRoot = join(process.cwd(), '..');
    const agentStatePath = join(
      projectRoot,
      '.skills',
      'research-manager',
      'state',
      'agents.json'
    );

    if (existsSync(agentStatePath)) {
      const agents = JSON.parse(readFileSync(agentStatePath, 'utf-8'));
      const activities = buildResearchAgentActivities(agents);
      if (activities.length > 0) return activities;
    }
  } catch { /* fall through to remote */ }

  // Fall back to remote SSH
  try {
    const raw = execSync(
      `ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_HOST} "cat ${REMOTE_AGENTS_FILE} 2>/dev/null"`,
      { encoding: 'utf-8', timeout: 5000 }
    );
    const agents = JSON.parse(raw);
    return buildResearchAgentActivities(agents);
  } catch (error) {
    console.error('[Activities API] Research agent fetch failed:', error);
  }

  return [];
}

export async function GET() {
  // Check if we have a real backend URL configured (ngrok/Tailscale tunnel)
  const hasRealBackend = BACKEND_URL && !BACKEND_URL.includes('localhost');

  // On Vercel without a real backend, try AGENT_STATE_URL for activities
  const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;
  if (isVercel && !hasRealBackend) {
    // Try to fetch from public agent state API
    if (AGENT_STATE_URL) {
      try {
        const response = await fetch(`${AGENT_STATE_URL}/agents`, {
          cache: 'no-store',
          signal: AbortSignal.timeout(5000),
        });
        if (response.ok) {
          const agents = await response.json();
          const researchActivities = buildResearchAgentActivities(agents);
          return NextResponse.json({
            activities: researchActivities,
            timestamp: new Date().toISOString(),
            connected: true,
          });
        }
      } catch (e) {
        console.error('[Activities API] Failed to fetch from AGENT_STATE_URL:', e);
      }
    }

    return NextResponse.json({
      activities: [],
      timestamp: new Date().toISOString(),
      connected: false,
    });
  }

  try {
    // Fetch all activity sources in parallel
    const [
      trainingStatus,
      recordingStatus,
      liveTransformStatus,
      remoteTasks,
      remoteAgents,
    ] = await Promise.all([
      fetchTrainingStatus(),
      fetchRecordingStatus(),
      fetchLiveTransformStatus(),
      fetchRemoteTasks(),
      fetchRemoteAgents(),
    ]);

    let taskActivities: ActivityState[] = [];
    let researchActivities: ActivityState[] = [];

    if (remoteTasks && remoteAgents) {
      const runningAgentIds = getRunningAgentIdsFromAgents(remoteAgents);
      taskActivities = buildTaskActivities(remoteTasks, runningAgentIds);
      researchActivities = buildResearchAgentActivities(remoteAgents);
    } else {
      [taskActivities, researchActivities] = await Promise.all([
        fetchTaskActivities(),
        fetchResearchAgentActivities(),
      ]);
    }

    // Combine all activities
    const activities: ActivityState[] = [
      ...(trainingStatus ? [trainingStatus] : []),
      ...(recordingStatus ? [recordingStatus] : []),
      ...(liveTransformStatus ? [liveTransformStatus] : []),
      ...taskActivities,
      ...researchActivities.filter(
        (activity) =>
          !taskActivities.some(
            (taskActivity) =>
              taskActivity.assignedAgent &&
              taskActivity.assignedAgent === activity.assignedAgent
          )
      ),
    ];

    // Deduplicate by id
    const activityMap = new Map(activities.map((a) => [a.id, a]));
    const uniqueActivities = Array.from(activityMap.values());

    return NextResponse.json({
      activities: uniqueActivities,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('[Activities API] Error:', error);
    return NextResponse.json(
      {
        activities: [],
        error: 'Failed to fetch activities',
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
