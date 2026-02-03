/**
 * Contribution Statistics API
 *
 * GET /api/contributor/stats - Get global contribution statistics
 */

import { NextResponse } from "next/server";
import { isSupabaseConfigured } from "@/lib/supabase";
import { getContributionStats } from "@/lib/supabase/contributors";

export async function GET() {
  try {
    // Check if Supabase is configured
    if (!isSupabaseConfigured) {
      return NextResponse.json(
        {
          totalContributors: 0,
          totalCreditsEarned: 0,
          totalTasksCompleted: 0,
          message: "Credits system not configured",
        },
        { status: 200 }
      );
    }

    const stats = await getContributionStats();

    return NextResponse.json(stats);
  } catch (error) {
    console.error("Error fetching contribution stats:", error);
    return NextResponse.json(
      { error: "Failed to fetch contribution stats" },
      { status: 500 }
    );
  }
}
