/**
 * Compute Leaderboard API
 *
 * GET /api/compute/leaderboard - Get top contributors (proxies to Workers)
 */

import { NextRequest, NextResponse } from "next/server";

const WORKERS_API = "https://labfork-agents.jonathan-hawkins.workers.dev/api/compute";

/**
 * GET /api/compute/leaderboard
 * Get top contributors by credits earned
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get("limit") || "20";

    const response = await fetch(`${WORKERS_API}/leaderboard?limit=${limit}`, {
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    // Workers returns array directly, wrap for frontend compatibility
    return NextResponse.json(data);
  } catch (error) {
    console.error("Get leaderboard error:", error);
    return NextResponse.json(
      { error: "Failed to get leaderboard" },
      { status: 500 }
    );
  }
}
