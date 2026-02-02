/**
 * Leaderboard API
 *
 * GET /api/contributor/leaderboard - Get top contributors (proxies to Workers)
 */

import { NextRequest, NextResponse } from "next/server";

const WORKERS_API = "https://labfork-agents.jonathan-hawkins.workers.dev/api/compute";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = searchParams.get("limit") || "10";

    const response = await fetch(`${WORKERS_API}/leaderboard?limit=${limit}`, {
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Workers API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching leaderboard:", error);
    return NextResponse.json(
      { error: "Failed to fetch leaderboard" },
      { status: 500 }
    );
  }
}
