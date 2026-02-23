/**
 * Results API - Featured Results
 *
 * GET /api/results/featured - Get featured results
 */

import { NextRequest, NextResponse } from "next/server";
import { getFeaturedResults } from "@/lib/social/results";

/**
 * GET /api/results/featured
 * Get featured results
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "10", 10);

    const results = await getFeaturedResults(Math.min(limit, 50));

    return NextResponse.json({ results });
  } catch (error) {
    console.error("Error getting featured results:", error);
    return NextResponse.json(
      { error: "Failed to get featured results" },
      { status: 500 }
    );
  }
}
