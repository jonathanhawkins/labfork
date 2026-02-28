/**
 * Research Tasks API
 *
 * GET /api/labs/[id]/research - Get research tasks for a lab (proxied from Nudge Engine)
 * POST /api/labs/[id]/research - Create a research task for a lab
 */

import { NextRequest, NextResponse } from "next/server";

const NUDGE_ENGINE_URL =
  process.env.NUDGE_ENGINE_URL ||
  "https://nudge-engine.jonathan-hawkins.workers.dev";
const NUDGE_ENGINE_API_KEY = process.env.NUDGE_ENGINE_API_KEY || "";

interface RouteParams {
  params: Promise<{ id: string }>;
}

interface NudgeTask {
  id: string;
  action: string;
  description: string;
  status: "pending" | "assigned" | "completed" | "failed";
  priority: number;
  assigned_worker_id: string | null;
  result: string | null;
  error: string | null;
  parent_task_id: string | null;
  lab_id: string | null;
  created_at: string;
  assigned_at: string | null;
  completed_at: string | null;
}

/** Humanize action: strip goal-research-{slug}- prefix, replace hyphens, title case */
function humanizeAction(action: string): string {
  // Strip common prefixes
  let cleaned = action
    .replace(/^goal-research-[a-z0-9-]+-/, "")
    .replace(/^goal-research-/, "")
    .replace(/^goal-/, "");

  // Replace hyphens with spaces and title case
  return cleaned
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Map Nudge Engine task to frontend format */
function mapTask(task: NudgeTask) {
  const isFailed = task.status === "failed";
  let resultSummary: string | undefined;

  if (task.result) {
    try {
      const parsed =
        typeof task.result === "string" ? JSON.parse(task.result) : task.result;
      resultSummary = parsed?.summary;
    } catch {
      // result is not JSON
    }
  }

  return {
    id: task.id,
    nudgeTaskId: task.id,
    subject: humanizeAction(task.action),
    description: task.description,
    status:
      task.status === "assigned"
        ? ("in_progress" as const)
        : task.status === "failed"
          ? ("completed" as const)
          : (task.status as "pending" | "completed"),
    isFailed,
    owner: task.assigned_worker_id ? "AI Agent" : undefined,
    parentTaskId: task.parent_task_id,
    resultSummary,
    createdAt: task.created_at,
    completedAt: task.completed_at,
  };
}

/** Actions that are infrastructure noise, not research tasks */
const NOISE_ACTIONS = new Set([
  "check-health",
  "investigate-failures",
]);

/** Check if a task is a research task (goal-based, not infra noise) */
function isResearchTask(task: NudgeTask): boolean {
  if (NOISE_ACTIONS.has(task.action)) return false;
  return true;
}

/**
 * GET /api/labs/[id]/research
 * Fetch research tasks from Nudge Engine.
 * First tries lab-specific tasks; falls back to all research tasks
 * (global tasks created by the cron/goals pipeline).
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { id: labId } = await params;
    const headers = NUDGE_ENGINE_API_KEY
      ? { Authorization: `Bearer ${NUDGE_ENGINE_API_KEY}` }
      : {};

    // Try lab-specific tasks first
    const labUrl = `${NUDGE_ENGINE_URL}/tasks?lab_id=${encodeURIComponent(labId)}&limit=50`;
    const labRes = await fetch(labUrl, { headers, next: { revalidate: 0 } });

    let allTasks: NudgeTask[] = [];

    if (labRes.ok) {
      const labData = await labRes.json();
      allTasks = (labData.tasks || []) as NudgeTask[];
    }

    // Fallback: if no lab-specific tasks, fetch all tasks from the engine.
    // Research tasks created via cron/goals don't have lab_id but are still
    // relevant to the primary research lab.
    if (allTasks.length === 0) {
      const globalUrl = `${NUDGE_ENGINE_URL}/tasks?limit=50`;
      const globalRes = await fetch(globalUrl, {
        headers,
        next: { revalidate: 0 },
      });

      if (globalRes.ok) {
        const globalData = await globalRes.json();
        allTasks = (globalData.tasks || []) as NudgeTask[];
      }
    }

    // Filter out infrastructure noise (health checks, failure investigations)
    allTasks = allTasks.filter(isResearchTask);

    // Separate parent tasks and leaf tasks
    const parentIds = new Set(
      allTasks
        .filter((t) => t.parent_task_id)
        .map((t) => t.parent_task_id as string)
    );

    // Show leaf tasks only (filter out parents that have children)
    const leafTasks = allTasks.filter((t) => !parentIds.has(t.id));

    // Find latest activity timestamp
    const latestActivity = allTasks.reduce((latest, t) => {
      const ts = t.completed_at || t.assigned_at || t.created_at;
      return ts > latest ? ts : latest;
    }, "");

    return NextResponse.json({
      success: true,
      tasks: leafTasks.map(mapTask),
      total: leafTasks.length,
      latestActivity: latestActivity || undefined,
    });
  } catch (error) {
    console.error("Failed to fetch research tasks:", error);
    return NextResponse.json(
      { success: false, error: "Internal server error" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/labs/[id]/research
 * Create a research task in the Nudge Engine with lab_id
 */
export async function POST(request: NextRequest, { params }: RouteParams) {
  try {
    const { id: labId } = await params;
    const body = await request.json();

    const res = await fetch(`${NUDGE_ENGINE_URL}/tasks`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(NUDGE_ENGINE_API_KEY
          ? { Authorization: `Bearer ${NUDGE_ENGINE_API_KEY}` }
          : {}),
      },
      body: JSON.stringify({
        ...body,
        lab_id: labId,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      return NextResponse.json(
        {
          success: false,
          error: errData.error || "Failed to create research task",
        },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json({ success: true, ...data });
  } catch (error) {
    console.error("Failed to create research task:", error);
    return NextResponse.json(
      { success: false, error: "Internal server error" },
      { status: 500 }
    );
  }
}
