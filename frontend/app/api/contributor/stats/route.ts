/**
 * Contribution Statistics API
 *
 * GET /api/contributor/stats - Get global contribution statistics
 */

import { NextResponse } from "next/server";
import { getContributionStats } from "@/lib/supabase/contributors";

export async function GET() {
  try {
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
