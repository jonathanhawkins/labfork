/**
 * Technique Adoption API
 *
 * GET /api/patterns/adoption - Get technique adoption rates and trends
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalPatternRecognition,
  getAdoptionMetrics,
  isAdoptionStage,
} from "@/lib/meta/patterns";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get filter parameters
    const stage = searchParams.get("stage");
    const minScore = searchParams.get("minScore");
    const sortBy = searchParams.get("sortBy") || "adoptionScore";
    const order = searchParams.get("order") || "desc";
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");

    // Ensure adoption metrics are computed
    const recognition = getGlobalPatternRecognition();
    if (recognition.getAdoptionMetrics().length === 0) {
      recognition.trackTechniqueAdoption();
    }

    // Get adoption metrics
    let metrics = getAdoptionMetrics();

    // Filter by stage
    if (stage && isAdoptionStage(stage)) {
      metrics = metrics.filter((m) => m.stage === stage);
    }

    // Filter by minimum score
    if (minScore) {
      const threshold = parseFloat(minScore);
      metrics = metrics.filter((m) => m.adoptionScore >= threshold);
    }

    // Sort
    metrics.sort((a, b) => {
      let comparison: number;

      switch (sortBy) {
        case "adoptionScore":
          comparison = a.adoptionScore - b.adoptionScore;
          break;
        case "adoptionTrend":
          comparison = a.adoptionTrend - b.adoptionTrend;
          break;
        case "citations":
          comparison = a.citationCount - b.citationCount;
          break;
        case "implementations":
          comparison = a.implementationCount - b.implementationCount;
          break;
        case "name":
          comparison = a.techniqueName.localeCompare(b.techniqueName);
          break;
        default:
          comparison = a.adoptionScore - b.adoptionScore;
      }

      return order === "desc" ? -comparison : comparison;
    });

    // Paginate
    const total = metrics.length;
    const offset = (page - 1) * limit;
    metrics = metrics.slice(offset, offset + limit);

    return NextResponse.json({
      success: true,
      data: metrics.map((m) => ({
        techniqueId: m.techniqueId,
        techniqueName: m.techniqueName,
        adoptionScore: m.adoptionScore,
        adoptionTrend: m.adoptionTrend,
        stage: m.stage,
        timeToMainstream: m.timeToMainstream,
        citationCount: m.citationCount,
        implementationCount: m.implementationCount,
        timeSeries: m.timeSeries,
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + metrics.length < total,
      },
      summary: {
        emerging: metrics.filter((m) => m.stage === "emerging").length,
        growing: metrics.filter((m) => m.stage === "growing").length,
        mainstream: metrics.filter((m) => m.stage === "mainstream").length,
        mature: metrics.filter((m) => m.stage === "mature").length,
        declining: metrics.filter((m) => m.stage === "declining").length,
      },
    });
  } catch (error) {
    console.error("Error fetching adoption metrics:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch adoption metrics",
      },
      { status: 500 }
    );
  }
}
