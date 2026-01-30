/**
 * Compute Network Analytics API
 *
 * GET /api/compute/analytics - Get comprehensive network analytics
 *
 * Returns:
 * - Total network TFLOPS
 * - Active contributors by tier
 * - Task completion stats
 * - Latency percentiles
 * - Network health score
 * - Throughput metrics
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import {
  generateNetworkAnalytics,
  type TimePeriod,
} from "@/lib/compute/analytics";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/compute/analytics
 * Get comprehensive network analytics
 *
 * Query params:
 * - period: Time period (1h, 24h, 7d, 30d) - defaults to 24h
 */
export async function GET(req: NextRequest) {
  try {
    const orchestrator = getOrchestrator();

    // Get query params
    const { searchParams } = new URL(req.url);
    const period = (searchParams.get("period") as TimePeriod) || "24h";

    // Validate period
    const validPeriods: TimePeriod[] = ["1h", "24h", "7d", "30d"];
    if (!validPeriods.includes(period)) {
      return NextResponse.json(
        { error: "Invalid period. Must be one of: 1h, 24h, 7d, 30d" },
        { status: 400 }
      );
    }

    // Get current network stats
    const stats = orchestrator.getNetworkStats();

    // Get recent tasks for latency analysis
    // In production, this would query a database with time-based filtering
    // For MVP, we'll use in-memory data
    const tasks: any[] = []; // orchestrator.getRecentTasks(period) would be ideal

    // Generate analytics
    const analytics = generateNetworkAnalytics(stats, tasks, period);

    return NextResponse.json(analytics);
  } catch (error) {
    console.error("Analytics API error:", error);
    return NextResponse.json(
      { error: "Failed to generate analytics" },
      { status: 500 }
    );
  }
}
