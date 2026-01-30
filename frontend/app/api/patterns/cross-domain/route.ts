/**
 * Cross-Domain Transfer API
 *
 * GET /api/patterns/cross-domain - Get cross-domain transfer patterns
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalPatternRecognition, getCrossDomainTransfers } from "@/lib/meta/patterns";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get filter parameters
    const sourceDomain = searchParams.get("sourceDomain");
    const targetDomain = searchParams.get("targetDomain");
    const minSuccess = searchParams.get("minSuccess");
    const sortBy = searchParams.get("sortBy") || "successScore";
    const order = searchParams.get("order") || "desc";
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");

    // Ensure cross-domain transfers are detected
    const recognition = getGlobalPatternRecognition();
    if (recognition.getCrossDomainTransfers().length === 0) {
      recognition.detectCrossDomainTransfers();
    }

    // Get transfers
    let transfers = getCrossDomainTransfers();

    // Filter by source domain
    if (sourceDomain) {
      transfers = transfers.filter((t) => t.sourceDomain === sourceDomain);
    }

    // Filter by target domain
    if (targetDomain) {
      transfers = transfers.filter((t) => t.targetDomain === targetDomain);
    }

    // Filter by minimum success
    if (minSuccess) {
      const threshold = parseFloat(minSuccess);
      transfers = transfers.filter((t) => t.successScore >= threshold);
    }

    // Sort
    transfers.sort((a, b) => {
      let comparison: number;

      switch (sortBy) {
        case "successScore":
          comparison = a.successScore - b.successScore;
          break;
        case "confidence":
          comparison = a.confidence - b.confidence;
          break;
        case "detectedAt":
          comparison = a.detectedAt.getTime() - b.detectedAt.getTime();
          break;
        case "techniqueName":
          comparison = a.techniqueName.localeCompare(b.techniqueName);
          break;
        default:
          comparison = a.successScore - b.successScore;
      }

      return order === "desc" ? -comparison : comparison;
    });

    // Paginate
    const total = transfers.length;
    const offset = (page - 1) * limit;
    transfers = transfers.slice(offset, offset + limit);

    // Collect unique domains for summary
    const allTransfers = getCrossDomainTransfers();
    const sourceDomains = Array.from(new Set(allTransfers.map((t) => t.sourceDomain)));
    const targetDomains = Array.from(new Set(allTransfers.map((t) => t.targetDomain)));

    return NextResponse.json({
      success: true,
      data: transfers.map((t) => ({
        id: t.id,
        techniqueId: t.techniqueId,
        techniqueName: t.techniqueName,
        sourceDomain: t.sourceDomain,
        targetDomain: t.targetDomain,
        successScore: t.successScore,
        adaptations: t.adaptations,
        detectedAt: t.detectedAt,
        evidencePapers: t.evidencePapers,
        confidence: t.confidence,
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + transfers.length < total,
      },
      summary: {
        totalTransfers: allTransfers.length,
        sourceDomains,
        targetDomains,
        avgSuccessScore:
          allTransfers.length > 0
            ? allTransfers.reduce((sum, t) => sum + t.successScore, 0) /
              allTransfers.length
            : 0,
      },
    });
  } catch (error) {
    console.error("Error fetching cross-domain transfers:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch cross-domain transfers",
      },
      { status: 500 }
    );
  }
}
