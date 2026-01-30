/**
 * Pattern Report API
 *
 * GET /api/patterns/report - Generate comprehensive pattern report
 */

import { NextRequest, NextResponse } from "next/server";
import { analyzePatterns, getLastReport } from "@/lib/meta/patterns";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const forceRefresh = searchParams.get("refresh") === "true";

    const startTime = performance.now();

    // Check if we have a recent report
    const lastReport = getLastReport();
    const reportAge = lastReport
      ? Date.now() - lastReport.generatedAt.getTime()
      : Infinity;
    const maxAge = 5 * 60 * 1000; // 5 minutes

    // Use cached report if available and not too old
    const report =
      !forceRefresh && lastReport && reportAge < maxAge
        ? lastReport
        : analyzePatterns();

    const executionTimeMs = performance.now() - startTime;

    return NextResponse.json({
      success: true,
      data: {
        id: report.id,
        generatedAt: report.generatedAt,
        periodStart: report.periodStart,
        periodEnd: report.periodEnd,
        trends: report.trends.slice(0, 20).map((t) => ({
          id: t.id,
          name: t.name,
          category: t.category,
          strength: t.strength,
          momentum: t.momentum,
          confidence: t.confidence,
        })),
        emergingPatterns: report.emergingPatterns.slice(0, 10).map((p) => ({
          id: p.id,
          name: p.name,
          frequency: p.frequency,
          domains: p.domains,
          confidence: p.confidence,
        })),
        adoptionMetrics: report.adoptionMetrics.slice(0, 15).map((a) => ({
          techniqueId: a.techniqueId,
          techniqueName: a.techniqueName,
          adoptionScore: a.adoptionScore,
          stage: a.stage,
        })),
        crossDomainTransfers: report.crossDomainTransfers.slice(0, 10).map((t) => ({
          id: t.id,
          techniqueName: t.techniqueName,
          sourceDomain: t.sourceDomain,
          targetDomain: t.targetDomain,
          successScore: t.successScore,
        })),
        summary: report.summary,
      },
      meta: {
        executionTimeMs: Math.round(executionTimeMs * 100) / 100,
        cached: !forceRefresh && lastReport !== null && reportAge < maxAge,
        reportAge: Math.round(reportAge / 1000), // seconds
      },
    });
  } catch (error) {
    console.error("Error generating pattern report:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to generate pattern report",
      },
      { status: 500 }
    );
  }
}
