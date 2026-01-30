/**
 * Gap Landscape API
 *
 * GET /api/gaps/landscape - Get research landscape visualization data
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";
import { getGlobalGapAnalyzer } from "@/lib/meta/gaps";

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const domain = searchParams.get("domain") || undefined;

    const graph = getGlobalGraph();
    const analyzer = getGlobalGapAnalyzer();

    const landscape = analyzer.generateLandscape(graph, domain);

    return NextResponse.json({
      success: true,
      data: {
        id: landscape.id,
        domain: landscape.domain,
        nodes: landscape.nodes,
        edges: landscape.edges,
        clusters: landscape.clusters,
        gapCount: landscape.gaps.length,
        opportunityCount: landscape.opportunities.length,
        coverageScore: landscape.coverageScore,
        densityScore: landscape.densityScore,
        generatedAt: landscape.generatedAt.toISOString(),
      },
      meta: {
        executionTimeMs: Date.now() - startTime,
      },
    });
  } catch (error) {
    console.error("Failed to generate landscape:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to generate research landscape",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
