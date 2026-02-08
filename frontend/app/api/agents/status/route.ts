/**
 * Agent Status API Proxy
 *
 * GET /api/agents/status - Fetch all agent statuses from Cloudflare Workers
 *
 * Proxies to the autonomous agent system running on Cloudflare Workers.
 * Falls back to demo data if the Workers API is unavailable.
 */

import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const AGENTS_API_URL =
  process.env.NEXT_PUBLIC_AGENTS_API_URL ||
  'https://labfork-agents.jonathan-hawkins.workers.dev';

// Demo agents for fallback when API is unavailable
const DEMO_AGENTS = [
  {
    id: 'opus-1',
    name: 'Opus',
    type: 'researcher',
    status: 'working',
    current_task_id: 'task-demo-1',
    current_task: {
      id: 'task-demo-1',
      title: 'Analyzing prosody patterns in speech synthesis papers',
      status: 'in_progress',
      priority: 3,
    },
    capabilities: ['research', 'analysis', 'documentation'],
    created_at: new Date(Date.now() - 3600000).toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: 'codex-1',
    name: 'Codex',
    type: 'developer',
    status: 'working',
    current_task_id: 'task-demo-2',
    current_task: {
      id: 'task-demo-2',
      title: 'Implementing MaskGCT attention module',
      status: 'in_progress',
      priority: 2,
    },
    capabilities: ['coding', 'testing', 'refactoring'],
    created_at: new Date(Date.now() - 7200000).toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: 'scout-1',
    name: 'Scout',
    type: 'explorer',
    status: 'idle',
    current_task_id: null,
    current_task: null,
    capabilities: ['search', 'discovery', 'synergy-detection'],
    created_at: new Date(Date.now() - 10800000).toISOString(),
    updated_at: new Date(Date.now() - 600000).toISOString(),
  },
  {
    id: 'planner-1',
    name: 'Planner',
    type: 'coordinator',
    status: 'idle',
    current_task_id: null,
    current_task: null,
    capabilities: ['planning', 'scheduling', 'prioritization'],
    created_at: new Date(Date.now() - 14400000).toISOString(),
    updated_at: new Date(Date.now() - 1800000).toISOString(),
  },
];

export async function GET() {
  try {
    // Attempt to fetch from Cloudflare Workers API
    const response = await fetch(`${AGENTS_API_URL}/api/agents`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      throw new Error(`Workers API returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      agents: data.agents || [],
      count: data.count || 0,
      source: 'workers',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.warn('[Agents Status API] Workers API unavailable, using demo data:', error);

    // Return demo data as fallback
    return NextResponse.json({
      agents: DEMO_AGENTS,
      count: DEMO_AGENTS.length,
      source: 'demo',
      timestamp: new Date().toISOString(),
      fallback: true,
    });
  }
}
