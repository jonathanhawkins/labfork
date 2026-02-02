/**
 * Task Assignment API
 *
 * POST /api/compute/tasks/assign - Request work assignment for a device (proxies to Workers)
 */

import { NextRequest, NextResponse } from "next/server";

const WORKERS_API = "https://labfork-agents.jonathan-hawkins.workers.dev/api/compute";

/**
 * POST /api/compute/tasks/assign
 * Assign task to requesting device
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const authHeader = request.headers.get("Authorization");

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    // Forward auth token if present
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }

    const response = await fetch(`${WORKERS_API}/tasks/assign`, {
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
    console.error("Task assignment error:", error);
    return NextResponse.json(
      { error: "Failed to assign task" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/tasks/assign
 * Get assignment statistics (for monitoring)
 */
export async function GET() {
  try {
    // This endpoint doesn't exist in Workers, return basic stats
    const statsResponse = await fetch(`${WORKERS_API}/stats`, {
      cache: "no-store",
    });

    const stats = await statsResponse.json();

    return NextResponse.json({
      success: true,
      stats: {
        pendingTasks: stats.tasks?.pending || 0,
        assignedTasks: stats.tasks?.processing || 0,
        totalTasks: stats.tasks?.total || 0,
        onlineDevices: stats.devices?.online || 0,
      },
    });
  } catch (error) {
    console.error("Get assignment stats error:", error);
    return NextResponse.json(
      { error: "Failed to get assignment statistics" },
      { status: 500 }
    );
  }
}
