/**
 * Agent Work Log API Proxy
 *
 * GET /api/agents/work-log - Fetch recent work log entries from Cloudflare Workers
 *
 * Query parameters:
 * - limit: number (default 50) - Maximum entries to return
 * - agent_id: string (optional) - Filter by specific agent
 * - task_id: string (optional) - Filter by specific task
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const AGENTS_API_URL =
  process.env.NEXT_PUBLIC_AGENTS_API_URL || 'http://localhost:8787';

// Demo work log entries for fallback
function generateDemoWorkLog() {
  const now = Date.now();
  return [
    {
      id: 'log-demo-1',
      agent_id: 'opus-1',
      task_id: 'task-demo-1',
      action: 'research_completed',
      details: {
        topic: 'Prosody patterns in neural TTS',
        papers_analyzed: 5,
        key_findings: ['Duration modeling crucial', 'F0 contour prediction improves naturalness'],
      },
      created_at: new Date(now - 30000).toISOString(),
    },
    {
      id: 'log-demo-2',
      agent_id: 'codex-1',
      task_id: 'task-demo-2',
      action: 'code_implemented',
      details: {
        component: 'MaskGCT attention module',
        files_changed: 3,
        lines_added: 156,
      },
      created_at: new Date(now - 60000).toISOString(),
    },
    {
      id: 'log-demo-3',
      agent_id: 'scout-1',
      task_id: null,
      action: 'synergy_detected',
      details: {
        synergy: 'EmoProsody + StyleTransfer',
        confidence: 0.87,
        potential_impact: 'high',
      },
      created_at: new Date(now - 120000).toISOString(),
    },
    {
      id: 'log-demo-4',
      agent_id: 'planner-1',
      task_id: null,
      action: 'schedule_created',
      details: {
        training_jobs: 3,
        estimated_duration: '4 hours',
        priority: 'overnight batch',
      },
      created_at: new Date(now - 180000).toISOString(),
    },
    {
      id: 'log-demo-5',
      agent_id: 'opus-1',
      task_id: 'task-demo-1',
      action: 'task_started',
      details: {
        task: 'Analyzing prosody patterns',
        approach: 'Literature review + code analysis',
      },
      created_at: new Date(now - 240000).toISOString(),
    },
  ];
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const limit = searchParams.get('limit') || '50';
  const agentId = searchParams.get('agent_id');
  const taskId = searchParams.get('task_id');

  try {
    // Build query string for Workers API
    const params = new URLSearchParams({ limit });
    if (agentId) params.set('agent_id', agentId);
    if (taskId) params.set('task_id', taskId);

    // Attempt to fetch from Cloudflare Workers API
    const response = await fetch(`${AGENTS_API_URL}/api/work-log?${params.toString()}`, {
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
      entries: data.entries || [],
      count: data.count || 0,
      source: 'workers',
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.warn('[Work Log API] Workers API unavailable, using demo data:', error);

    // Return demo data as fallback
    const demoLog = generateDemoWorkLog();
    const limitNum = parseInt(limit, 10) || 50;

    // Filter if needed
    let filteredLog = demoLog;
    if (agentId) {
      filteredLog = filteredLog.filter((log) => log.agent_id === agentId);
    }
    if (taskId) {
      filteredLog = filteredLog.filter((log) => log.task_id === taskId);
    }

    return NextResponse.json({
      entries: filteredLog.slice(0, limitNum),
      count: filteredLog.length,
      source: 'demo',
      timestamp: new Date().toISOString(),
      fallback: true,
    });
  }
}
