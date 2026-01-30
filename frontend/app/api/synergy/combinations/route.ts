/**
 * Explored Combinations API
 *
 * GET /api/synergy/combinations - List explored combinations
 */

import { NextRequest, NextResponse } from "next/server";
import { getExploredCombinations } from "@/lib/meta/synergy";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get filter parameters
    const status = searchParams.get("status");
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");

    // Get combinations
    let combinations = getExploredCombinations();

    // Filter by status
    if (status) {
      combinations = combinations.filter((c) => c.status === status);
    }

    // Sort by start date (most recent first)
    combinations.sort(
      (a, b) => b.startedAt.getTime() - a.startedAt.getTime()
    );

    // Paginate
    const total = combinations.length;
    const offset = (page - 1) * limit;
    combinations = combinations.slice(offset, offset + limit);

    return NextResponse.json({
      success: true,
      data: combinations.map((c) => ({
        proposal: {
          id: c.proposal.id,
          techniqueA: {
            id: c.proposal.techniqueA.id,
            name: c.proposal.techniqueA.name,
          },
          techniqueB: {
            id: c.proposal.techniqueB.id,
            name: c.proposal.techniqueB.name,
          },
          score: c.proposal.score,
        },
        startedAt: c.startedAt,
        completedAt: c.completedAt,
        status: c.status,
        results: c.results,
        implementationNotes: c.implementationNotes,
        resultingTechniqueId: c.resultingTechniqueId,
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + combinations.length < total,
      },
    });
  } catch (error) {
    console.error("Error fetching combinations:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch combinations",
      },
      { status: 500 }
    );
  }
}
