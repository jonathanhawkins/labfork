/**
 * Synergy Proposals API
 *
 * GET /api/synergy/proposals - List synergy proposals with filtering
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getAllProposals,
  getProposalsByStatus,
  isProposalStatus,
} from "@/lib/meta/synergy";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get filter parameters
    const status = searchParams.get("status");
    const minScore = searchParams.get("minScore");
    const sortBy = searchParams.get("sortBy") || "score";
    const order = searchParams.get("order") || "desc";
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");

    // Get proposals
    let proposals = status && isProposalStatus(status)
      ? getProposalsByStatus(status)
      : getAllProposals();

    // Filter by minimum score
    if (minScore) {
      const threshold = parseFloat(minScore);
      proposals = proposals.filter((p) => p.score.overall >= threshold);
    }

    // Sort
    proposals.sort((a, b) => {
      let aValue: number | Date;
      let bValue: number | Date;

      switch (sortBy) {
        case "score":
          aValue = a.score.overall;
          bValue = b.score.overall;
          break;
        case "confidence":
          aValue = a.score.confidence;
          bValue = b.score.confidence;
          break;
        case "novelty":
          aValue = a.score.components.novelty;
          bValue = b.score.components.novelty;
          break;
        case "impact":
          aValue = a.score.components.impact;
          bValue = b.score.components.impact;
          break;
        case "createdAt":
          aValue = a.createdAt;
          bValue = b.createdAt;
          break;
        default:
          aValue = a.score.overall;
          bValue = b.score.overall;
      }

      const comparison =
        aValue instanceof Date
          ? aValue.getTime() - (bValue as Date).getTime()
          : (aValue as number) - (bValue as number);

      return order === "desc" ? -comparison : comparison;
    });

    // Paginate
    const total = proposals.length;
    const offset = (page - 1) * limit;
    proposals = proposals.slice(offset, offset + limit);

    return NextResponse.json({
      success: true,
      data: proposals.map((p) => ({
        id: p.id,
        techniqueA: {
          id: p.techniqueA.id,
          name: p.techniqueA.name,
          domains: p.techniqueA.domains,
        },
        techniqueB: {
          id: p.techniqueB.id,
          name: p.techniqueB.name,
          domains: p.techniqueB.domains,
        },
        score: p.score,
        justification: p.justification,
        combinationAspects: p.combinationAspects,
        expectedOutcomes: p.expectedOutcomes,
        status: p.status,
        createdAt: p.createdAt,
        updatedAt: p.updatedAt,
        notes: p.notes,
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + proposals.length < total,
      },
    });
  } catch (error) {
    console.error("Error fetching proposals:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to fetch proposals",
      },
      { status: 500 }
    );
  }
}
