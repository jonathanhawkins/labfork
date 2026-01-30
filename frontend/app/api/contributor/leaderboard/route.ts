/**
 * Leaderboard API
 *
 * GET /api/contributor/leaderboard - Get top contributors
 */

import { NextRequest, NextResponse } from "next/server";
import { getLeaderboard } from "@/lib/supabase/contributors";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limitParam = searchParams.get("limit");
    const limit = limitParam ? parseInt(limitParam, 10) : 10;

    // Validate limit
    if (limit < 1 || limit > 100) {
      return NextResponse.json(
        { error: "Limit must be between 1 and 100" },
        { status: 400 }
      );
    }

    const leaders = await getLeaderboard(limit);

    return NextResponse.json(leaders);
  } catch (error) {
    console.error("Error fetching leaderboard:", error);
    return NextResponse.json(
      { error: "Failed to fetch leaderboard" },
      { status: 500 }
    );
  }
}
