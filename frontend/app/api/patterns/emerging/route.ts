/**
 * Emerging Patterns API
 *
 * GET /api/patterns/emerging - Get newly emerging patterns and trends
 */

import { NextRequest, NextResponse } from "next/server";
import { getEmergingTrends, getDetectedPatterns } from "@/lib/meta/patterns";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const limit = parseInt(searchParams.get("limit") || "10");
    const includePatterns = searchParams.get("includePatterns") !== "false";

    // Get emerging trends (high momentum)
    const emergingTrends = getEmergingTrends(limit);

    // Get recently detected patterns
    let emergingPatterns = includePatterns ? getDetectedPatterns() : [];
    if (emergingPatterns.length > 0) {
      // Sort by first detected (most recent first) and limit
      emergingPatterns = emergingPatterns
        .sort(
          (a, b) =>
            b.firstDetected.getTime() - a.firstDetected.getTime()
        )
        .slice(0, limit);
    }

    return NextResponse.json({
      success: true,
      data: {
        trends: emergingTrends.map((t) => ({
          id: t.id,
          name: t.name,
          description: t.description,
          category: t.category,
          keywords: t.keywords,
          strength: t.strength,
          momentum: t.momentum,
          confidence: t.confidence,
          firstDetected: t.firstDetected,
        })),
        patterns: emergingPatterns.map((p) => ({
          id: p.id,
          name: p.name,
          description: p.description,
          frequency: p.frequency,
          domains: p.domains,
          confidence: p.confidence,
          firstDetected: p.firstDetected,
        })),
      },
      meta: {
        trendCount: emergingTrends.length,
        patternCount: emergingPatterns.length,
      },
    });
  } catch (error) {
    console.error("Error fetching emerging patterns:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch emerging patterns",
      },
      { status: 500 }
    );
  }
}
