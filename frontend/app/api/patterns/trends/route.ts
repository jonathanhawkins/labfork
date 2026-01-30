/**
 * Research Trends API
 *
 * GET /api/patterns/trends - Get current research trends
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalPatternRecognition,
  getCurrentTrends,
  isTrendCategory,
} from "@/lib/meta/patterns";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get filter parameters
    const category = searchParams.get("category");
    const minStrength = searchParams.get("minStrength");
    const domain = searchParams.get("domain");
    const sortBy = searchParams.get("sortBy") || "strength";
    const order = searchParams.get("order") || "desc";
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "20");

    // Ensure trends are detected
    const recognition = getGlobalPatternRecognition();
    if (recognition.getTrends().length === 0) {
      recognition.detectTrends();
    }

    // Get trends
    let trends = getCurrentTrends();

    // Filter by category
    if (category && isTrendCategory(category)) {
      trends = trends.filter((t) => t.category === category);
    }

    // Filter by minimum strength
    if (minStrength) {
      const threshold = parseFloat(minStrength);
      trends = trends.filter((t) => t.strength >= threshold);
    }

    // Filter by domain
    if (domain) {
      trends = trends.filter((t) => t.domains.includes(domain));
    }

    // Sort
    trends.sort((a, b) => {
      let comparison: number;

      switch (sortBy) {
        case "strength":
          comparison = a.strength - b.strength;
          break;
        case "momentum":
          comparison = a.momentum - b.momentum;
          break;
        case "confidence":
          comparison = a.confidence - b.confidence;
          break;
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        default:
          comparison = a.strength - b.strength;
      }

      return order === "desc" ? -comparison : comparison;
    });

    // Paginate
    const total = trends.length;
    const offset = (page - 1) * limit;
    trends = trends.slice(offset, offset + limit);

    return NextResponse.json({
      success: true,
      data: trends.map((t) => ({
        id: t.id,
        name: t.name,
        description: t.description,
        category: t.category,
        keywords: t.keywords,
        domains: t.domains,
        strength: t.strength,
        momentum: t.momentum,
        confidence: t.confidence,
        firstDetected: t.firstDetected,
        lastUpdated: t.lastUpdated,
        timeSeries: t.timeSeries,
        relatedTechniques: t.relatedTechniques.slice(0, 5),
        relatedPapers: t.relatedPapers.slice(0, 5),
      })),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + trends.length < total,
      },
    });
  } catch (error) {
    console.error("Error fetching trends:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to fetch trends",
      },
      { status: 500 }
    );
  }
}
