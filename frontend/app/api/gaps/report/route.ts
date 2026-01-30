/**
 * Gap Report API
 *
 * GET /api/gaps/report - Generate comprehensive gap analysis report
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";
import { getGlobalGapAnalyzer } from "@/lib/meta/gaps";

// Cache for reports
let cachedReport: {
  data: ReturnType<ReturnType<typeof getGlobalGapAnalyzer>["generateReport"]>;
  timestamp: number;
} | null = null;

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;
    const domain = searchParams.get("domain") || undefined;
    const refresh = searchParams.get("refresh") === "true";

    // Check cache
    if (
      !refresh &&
      cachedReport &&
      !domain &&
      Date.now() - cachedReport.timestamp < CACHE_TTL_MS
    ) {
      return NextResponse.json({
        success: true,
        data: {
          ...cachedReport.data,
          generatedAt: cachedReport.data.generatedAt.toISOString(),
          topOpportunities: cachedReport.data.topOpportunities.map((o) => ({
            ...o,
            identifiedAt: o.identifiedAt.toISOString(),
          })),
        },
        meta: {
          executionTimeMs: Date.now() - startTime,
          cached: true,
        },
      });
    }

    const graph = getGlobalGraph();
    const analyzer = getGlobalGapAnalyzer();

    const report = analyzer.generateReport(graph, domain);

    // Cache if no domain filter
    if (!domain) {
      cachedReport = {
        data: report,
        timestamp: Date.now(),
      };
    }

    return NextResponse.json({
      success: true,
      data: {
        ...report,
        generatedAt: report.generatedAt.toISOString(),
        topOpportunities: report.topOpportunities.map((o) => ({
          ...o,
          identifiedAt: o.identifiedAt.toISOString(),
        })),
      },
      meta: {
        executionTimeMs: Date.now() - startTime,
        cached: false,
      },
    });
  } catch (error) {
    console.error("Failed to generate gap report:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to generate gap report",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
