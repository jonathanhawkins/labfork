/**
 * Compute Network Stats API
 *
 * GET /api/compute/stats - Get network statistics
 */

import { NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";

/**
 * GET /api/compute/stats
 * Get network statistics
 */
export async function GET() {
  try {
    const orchestrator = getOrchestrator();
    const stats = orchestrator.getNetworkStats();

    return NextResponse.json({
      success: true,
      stats,
    });
  } catch (error) {
    console.error("Get stats error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to get network stats" },
      { status: 500 }
    );
  }
}
