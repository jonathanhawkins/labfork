/**
 * Results API - Trending Results
 *
 * GET /api/results/trending - Get trending results
 */

import { NextRequest, NextResponse } from "next/server";
import { getTrendingResults } from "@/lib/social/results";

/**
 * GET /api/results/trending
 * Get trending results based on engagement and recency
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "10", 10);

    const results = await getTrendingResults(Math.min(limit, 50));

    return NextResponse.json({ results });
  } catch (error) {
    console.error("Error getting trending results:", error);
    return NextResponse.json(
      { error: "Failed to get trending results" },
      { status: 500 }
    );
  }
}
