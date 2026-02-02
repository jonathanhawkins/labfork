/**
 * Compute Network Stats API
 *
 * GET /api/compute/stats - Get network statistics (proxies to Workers backend)
 */

import { NextResponse } from "next/server";

const WORKERS_API = "https://labfork-agents.jonathan-hawkins.workers.dev/api/compute";

/**
 * GET /api/compute/stats
 * Get network statistics from Workers backend
 */
export async function GET() {
  try {
    const response = await fetch(`${WORKERS_API}/stats`, {
      headers: {
        "Content-Type": "application/json",
      },
      // Don't cache stats
      cache: "no-store",
    });

    if (!response.ok) {
      throw new Error(`Workers API error: ${response.status}`);
    }

    const data = await response.json();

    // Return Workers format directly for frontend compatibility
    return NextResponse.json(data);
  } catch (error) {
    console.error("Get stats error:", error);
    return NextResponse.json(
      { error: "Failed to get network stats" },
      { status: 500 }
    );
  }
}
