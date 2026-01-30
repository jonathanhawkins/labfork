/**
 * Compute Leaderboard API
 *
 * GET /api/compute/leaderboard - Get top contributors
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";

/**
 * GET /api/compute/leaderboard
 * Get top contributors by credits earned
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limitParam = searchParams.get("limit");
    const limit = limitParam ? parseInt(limitParam, 10) : 10;

    const orchestrator = getOrchestrator();
    const leaderboard = orchestrator.getLeaderboard(limit);

    return NextResponse.json({
      success: true,
      leaderboard,
      count: leaderboard.length,
    });
  } catch (error) {
    console.error("Get leaderboard error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to get leaderboard" },
      { status: 500 }
    );
  }
}
