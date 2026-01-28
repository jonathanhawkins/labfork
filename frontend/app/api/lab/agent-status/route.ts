import { NextResponse } from 'next/server';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8003';

// Paths for local agents.json (works when frontend runs on same machine as orchestrator)
const projectRoot = join(process.cwd(), '..');
const AGENTS_FILE = join(projectRoot, '.skills', 'research-manager', 'state', 'agents.json');

// Remote 4090 config
const REMOTE_HOST = 'doc@100.83.78.111';
const REMOTE_AGENTS_FILE = '~/dev/voice-clone-pipeline/.skills/research-manager/state/agents.json';

interface AgentInfo {
  name: string;
  type: string;
  task: string;
  status: string;
  started_at: string;
  output_file: string;
  killed_at?: string;
}

/**
 * Read agents from local file or remote 4090 via SSH
 */
function getRealAgents(): AgentInfo[] | null {
  // Try local file first
  if (existsSync(AGENTS_FILE)) {
    try {
      const data = JSON.parse(readFileSync(AGENTS_FILE, 'utf-8'));
      const agents: AgentInfo[] = Object.values(data);
      return agents;
    } catch { /* fall through */ }
  }

  // Try SSH to 4090
  try {
    const raw = execSync(
      `ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_HOST} "cat ${REMOTE_AGENTS_FILE} 2>/dev/null"`,
      { encoding: 'utf-8', timeout: 5000 }
    );
    const data = JSON.parse(raw);
    const agents: AgentInfo[] = Object.values(data);
    return agents;
  } catch { /* fall through */ }

  return null;
}

/**
 * Get all live tmux session names in one call
 */
function getLiveSessions(isRemote: boolean): Set<string> {
  try {
    const cmd = 'tmux list-sessions -F "#{session_name}" 2>/dev/null';
    const raw = isRemote
      ? execSync(`ssh -o ConnectTimeout=3 ${REMOTE_HOST} '${cmd}'`, { encoding: 'utf-8', timeout: 5000 })
      : execSync(cmd, { encoding: 'utf-8' });
    return new Set(raw.trim().split('\n').filter(Boolean));
  } catch {
    return new Set();
  }
}

export async function GET() {
  // Try real backend first
  const hasRealBackend = BACKEND_URL && !BACKEND_URL.includes('localhost');
  if (hasRealBackend) {
    try {
      const response = await fetch(`${BACKEND_URL}/api/lab/agent-status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
      });
      if (response.ok) {
        return NextResponse.json(await response.json());
      }
    } catch { /* fall through */ }
  }

  // Read real agent data from agents.json
  const allAgents = getRealAgents();
  if (!allAgents) {
    return NextResponse.json({
      connected: false,
      timestamp: new Date().toISOString(),
      agents: [],
    });
  }

  const isRemote = !existsSync(AGENTS_FILE);
  const liveSessions = getLiveSessions(isRemote);

  // Filter to only agents that are actually running (not killed, session alive)
  const running = allAgents.filter(a => {
    if (a.killed_at || a.status === 'killed') return false;
    return liveSessions.has(`rm-${a.name}`);
  });

  const agents = running.map(a => {
    // Extract task subject from the prompt (after "TASK #N: ")
    const taskMatch = a.task?.match(/TASK #\d+:\s*(.+?)(?:\\n|\n|$)/);
    const taskSubject = taskMatch ? taskMatch[1].trim() : a.task?.substring(0, 60) || 'Working...';

    return {
      id: a.name,
      name: a.name,
      status: 'working',
      task: taskSubject,
      type: a.type,
      startedAt: a.started_at,
    };
  });

  return NextResponse.json({
    connected: true,
    timestamp: new Date().toISOString(),
    agents,
  });
}
