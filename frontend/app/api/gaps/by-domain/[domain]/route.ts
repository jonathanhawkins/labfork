/**
 * Gap by Domain API
 *
 * GET /api/gaps/by-domain/[domain] - Get gaps for a specific domain
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";
import { getGlobalGapAnalyzer, GapSeverity } from "@/lib/meta/gaps";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ domain: string }> }
) {
  const startTime = Date.now();

  try {
    const { domain } = await params;
    const searchParams = request.nextUrl.searchParams;
    const severity = searchParams.get("severity") as GapSeverity | null;

    const graph = getGlobalGraph();
    const analyzer = getGlobalGapAnalyzer();

    const { gaps, opportunities } = analyzer.analyze(graph);

    // Filter by domain
    let domainGaps = gaps.filter((g) => g.domains.includes(domain));
    const domainOpportunities = opportunities.filter((o) =>
      gaps.find((g) => g.id === o.gapId && g.domains.includes(domain))
    );

    // Apply severity filter
    if (severity) {
      domainGaps = domainGaps.filter((g) => g.severity === severity);
    }

    // Serialize for response
    const serializedGaps = domainGaps.map((g) => ({
      ...g,
      detectedAt: g.detectedAt.toISOString(),
    }));

    const serializedOpportunities = domainOpportunities.map((o) => ({
      ...o,
      identifiedAt: o.identifiedAt.toISOString(),
    }));

    return NextResponse.json({
      success: true,
      data: {
        domain,
        gaps: serializedGaps,
        opportunities: serializedOpportunities,
        summary: {
          totalGaps: domainGaps.length,
          criticalGaps: domainGaps.filter((g) => g.severity === "critical").length,
          highGaps: domainGaps.filter((g) => g.severity === "high").length,
          mediumGaps: domainGaps.filter((g) => g.severity === "medium").length,
          totalOpportunities: domainOpportunities.length,
        },
      },
      meta: {
        executionTimeMs: Date.now() - startTime,
      },
    });
  } catch (error) {
    console.error("Failed to get gaps by domain:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to get gaps for domain",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
