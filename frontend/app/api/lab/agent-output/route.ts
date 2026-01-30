import { NextResponse } from 'next/server';
import { execSync } from 'child_process';

export const dynamic = "force-dynamic";

const REMOTE_GPU_HOST = process.env.REMOTE_GPU_HOST || '';
const REMOTE_GPU_USER = process.env.REMOTE_GPU_USER || 'doc';
const REMOTE_HOST = REMOTE_GPU_HOST ? `${REMOTE_GPU_USER}@${REMOTE_GPU_HOST}` : '';
const REMOTE_AGENTS_FILE = '~/dev/labfork/.skills/research-manager/state/agents.json';

// Sanitize path to prevent command injection
function sanitizePath(path: string): string | null {
  if (!path || typeof path !== 'string') return null;
  // Only allow paths under expected directories, no shell metacharacters
  if (!/^[a-zA-Z0-9_\-/.~]+$/.test(path)) return null;
  // Block obvious injection attempts
  if (path.includes('..') || path.includes(';') || path.includes('|') || path.includes('&') || path.includes('$') || path.includes('`')) {
    return null;
  }
  return path;
}

export async function GET() {
  try {
    const agentsRaw = execSync(
      `ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_HOST} "cat ${REMOTE_AGENTS_FILE} 2>/dev/null"`,
      { encoding: 'utf-8', timeout: 5000 }
    );

    let agents: Record<string, any>;
    try {
      agents = JSON.parse(agentsRaw);
    } catch {
      return NextResponse.json({ outputs: {}, error: 'Invalid agents JSON', timestamp: new Date().toISOString() });
    }

    if (!agents || typeof agents !== 'object') {
      return NextResponse.json({ outputs: {}, error: 'Invalid agents data', timestamp: new Date().toISOString() });
    }

    const outputs: Record<string, { lines: string[], file: string }> = {};

    for (const [name, agent] of Object.entries(agents) as [string, any][]) {
      if (agent.status !== 'running' || !agent.output_file) continue;

      // Sanitize the output file path to prevent command injection
      const safePath = sanitizePath(agent.output_file);
      if (!safePath) {
        console.warn(`[agent-output] Skipped unsafe path for agent ${name}: ${agent.output_file}`);
        continue;
      }

      try {
        const tail = execSync(
          `ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_HOST} "tail -20 '${safePath}' 2>/dev/null"`,
          { encoding: 'utf-8', timeout: 5000 }
        );
        outputs[name] = {
          lines: tail.split('\n').filter(Boolean).slice(-15),
          file: safePath,
        };
      } catch {
        // Skip agents with inaccessible output - don't log to avoid noise
      }
    }

    return NextResponse.json({ outputs, timestamp: new Date().toISOString() });
  } catch (error) {
    console.error('[agent-output] SSH error:', error instanceof Error ? error.message : String(error));
    return NextResponse.json({ outputs: {}, error: 'Failed to fetch', timestamp: new Date().toISOString() });
  }
}
