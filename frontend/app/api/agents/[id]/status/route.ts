/**
 * Individual Agent Status API Proxy
 *
 * GET /api/agents/[id]/status - Fetch detailed status for a specific agent
 *
 * Returns agent info with recent work log and statistics.
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const AGENTS_API_URL =
  process.env.NEXT_PUBLIC_AGENTS_API_URL ||
  'https://labfork-agents.jonathan-hawkins.workers.dev';

// Demo agent details for fallback
const DEMO_AGENT_DETAILS: Record<string, object> = {
  'opus-1': {
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
    recent_work_log: [
      {
        id: 'log-1',
        agent_id: 'opus-1',
        task_id: 'task-demo-1',
        action: 'research_completed',
        details: { papers_analyzed: 5 },
        created_at: new Date(Date.now() - 30000).toISOString(),
      },
    ],
    stats: {
      tasks_completed_today: 3,
      tasks_completed_total: 47,
      average_task_duration_minutes: 12,
    },
  },
  'codex-1': {
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
    recent_work_log: [
      {
        id: 'log-2',
        agent_id: 'codex-1',
        task_id: 'task-demo-2',
        action: 'code_implemented',
        details: { files_changed: 3 },
        created_at: new Date(Date.now() - 60000).toISOString(),
      },
    ],
    stats: {
      tasks_completed_today: 5,
      tasks_completed_total: 89,
      average_task_duration_minutes: 8,
    },
  },
  'scout-1': {
    id: 'scout-1',
    name: 'Scout',
    type: 'explorer',
    status: 'idle',
    current_task_id: null,
    current_task: null,
    capabilities: ['search', 'discovery', 'synergy-detection'],
    created_at: new Date(Date.now() - 10800000).toISOString(),
    updated_at: new Date(Date.now() - 600000).toISOString(),
    recent_work_log: [
      {
        id: 'log-3',
        agent_id: 'scout-1',
        task_id: null,
        action: 'synergy_detected',
        details: { synergy: 'EmoProsody + StyleTransfer' },
        created_at: new Date(Date.now() - 120000).toISOString(),
      },
    ],
    stats: {
      tasks_completed_today: 2,
      tasks_completed_total: 31,
      average_task_duration_minutes: 5,
    },
  },
  'planner-1': {
    id: 'planner-1',
    name: 'Planner',
    type: 'coordinator',
    status: 'idle',
    current_task_id: null,
    current_task: null,
    capabilities: ['planning', 'scheduling', 'prioritization'],
    created_at: new Date(Date.now() - 14400000).toISOString(),
    updated_at: new Date(Date.now() - 1800000).toISOString(),
    recent_work_log: [
      {
        id: 'log-4',
        agent_id: 'planner-1',
        task_id: null,
        action: 'schedule_created',
        details: { training_jobs: 3 },
        created_at: new Date(Date.now() - 180000).toISOString(),
      },
    ],
    stats: {
      tasks_completed_today: 1,
      tasks_completed_total: 24,
      average_task_duration_minutes: 3,
    },
  },
};

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: agentId } = await params;

  if (!agentId) {
    return NextResponse.json(
      { error: 'Agent ID is required' },
      { status: 400 }
    );
  }

  try {
    // Attempt to fetch from Cloudflare Workers API
    const response = await fetch(`${AGENTS_API_URL}/api/agents/${agentId}/status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      if (response.status === 404) {
        // Check if we have demo data for this agent
        const demoAgent = DEMO_AGENT_DETAILS[agentId];
        if (demoAgent) {
          return NextResponse.json({
            ...demoAgent,
            source: 'demo',
            fallback: true,
          });
        }
        return NextResponse.json(
          { error: 'Agent not found' },
          { status: 404 }
        );
      }
      throw new Error(`Workers API returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      ...data,
      source: 'workers',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.warn(`[Agent Status API] Workers API unavailable for agent ${agentId}:`, error);

    // Return demo data as fallback
    const demoAgent = DEMO_AGENT_DETAILS[agentId];
    if (demoAgent) {
      return NextResponse.json({
        ...demoAgent,
        source: 'demo',
        timestamp: new Date().toISOString(),
        fallback: true,
      });
    }

    return NextResponse.json(
      { error: 'Agent not found and API unavailable' },
      { status: 404 }
    );
  }
}
