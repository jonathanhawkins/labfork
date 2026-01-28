import { NextResponse } from 'next/server';
import { execSync } from 'child_process';

export const dynamic = "force-dynamic";

const REMOTE_HOST = 'doc@100.83.78.111';
const REMOTE_AGENTS_FILE = '~/dev/voice-clone-pipeline/.skills/research-manager/state/agents.json';

export async function GET() {
  try {
    const agentsRaw = execSync(
      `ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_HOST} "cat ${REMOTE_AGENTS_FILE} 2>/dev/null"`,
      { encoding: 'utf-8', timeout: 5000 }
    );
    const agents = JSON.parse(agentsRaw);

    const outputs: Record<string, { lines: string[], file: string }> = {};

    for (const [name, agent] of Object.entries(agents) as [string, any][]) {
      if (agent.status !== 'running' || !agent.output_file) continue;
      try {
        const tail = execSync(
          `ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no ${REMOTE_HOST} "tail -20 '${agent.output_file}' 2>/dev/null"`,
          { encoding: 'utf-8', timeout: 5000 }
        );
        outputs[name] = {
          lines: tail.split('\n').filter(Boolean).slice(-15),
          file: agent.output_file,
        };
      } catch { /* skip agents with inaccessible output */ }
    }

    return NextResponse.json({ outputs, timestamp: new Date().toISOString() });
  } catch (error) {
    return NextResponse.json({ outputs: {}, error: 'Failed to fetch', timestamp: new Date().toISOString() });
  }
}
