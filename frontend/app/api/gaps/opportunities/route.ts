/**
 * Gap Opportunities API
 *
 * GET /api/gaps/opportunities - List research opportunities
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";
import { getGlobalGapAnalyzer, GapOpportunity, OpportunityType, EffortLevel } from "@/lib/meta/gaps";

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const typeFilter = searchParams.get("type") as OpportunityType | null;
    const effortFilter = searchParams.get("effort") as EffortLevel | null;
    const minImpact = parseFloat(searchParams.get("minImpact") || "0");
    const page = parseInt(searchParams.get("page") || "1", 10);
    const limit = parseInt(searchParams.get("limit") || "20", 10);

    const graph = getGlobalGraph();
    const analyzer = getGlobalGapAnalyzer();

    const { opportunities } = analyzer.analyze(graph);

    // Apply filters
    let filtered = opportunities;

    if (typeFilter) {
      filtered = filtered.filter((o) => o.type === typeFilter);
    }

    if (effortFilter) {
      filtered = filtered.filter((o) => o.effort.level === effortFilter);
    }

    if (minImpact > 0) {
      filtered = filtered.filter((o) => o.impactScore >= minImpact);
    }

    // Sort by priority score
    filtered.sort((a, b) => b.priorityScore - a.priorityScore);

    // Paginate
    const total = filtered.length;
    const start = (page - 1) * limit;
    const paged = filtered.slice(start, start + limit);

    // Serialize for response
    const serialized = paged.map((o) => ({
      ...o,
      identifiedAt: o.identifiedAt.toISOString(),
    }));

    return NextResponse.json({
      success: true,
      data: serialized,
      pagination: {
        page,
        limit,
        total,
        hasMore: start + limit < total,
      },
      meta: {
        executionTimeMs: Date.now() - startTime,
      },
    });
  } catch (error) {
    console.error("Failed to get opportunities:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to get research opportunities",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
