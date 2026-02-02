/**
 * Task-specific API
 *
 * GET /api/compute/tasks/[taskId] - Get task info (proxies to Workers)
 * POST /api/compute/tasks/[taskId] - Complete a task (proxies to Workers)
 */

import { NextRequest, NextResponse } from "next/server";

const WORKERS_API = "https://labfork-agents.jonathan-hawkins.workers.dev/api/compute";

interface RouteParams {
  params: {
    taskId: string;
  };
}

/**
 * GET /api/compute/tasks/[taskId]
 * Get task information
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { taskId } = params;

    const response = await fetch(`${WORKERS_API}/tasks/${taskId}`, {
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Get task error:", error);
    return NextResponse.json(
      { error: "Failed to get task" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/compute/tasks/[taskId]
 * Complete a task (expects body with deviceId, success, result/error)
 */
export async function POST(request: NextRequest, { params }: RouteParams) {
  try {
    const { taskId } = params;
    const body = await request.json();
    const authHeader = request.headers.get("Authorization");

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    // Forward auth token if present
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }

    const response = await fetch(`${WORKERS_API}/tasks/${taskId}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Complete task error:", error);
    return NextResponse.json(
      { error: "Failed to complete task" },
      { status: 500 }
    );
  }
}
