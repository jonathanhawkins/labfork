/**
 * Custom Research Goal Analysis API
 *
 * POST - Analyze a natural language research goal
 */

import { NextRequest, NextResponse } from "next/server";
import { analyzeGoal, GoalAnalysis } from "@/lib/research/goal-analyzer";

export const dynamic = "force-dynamic";

/**
 * POST /api/research/goal - Analyze research goal
 *
 * Body: { goal: string }
 * Returns: { success: boolean, analysis?: GoalAnalysis, papers?: Paper[], error?: string }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { goal } = body;

    if (!goal) {
      return NextResponse.json(
        { success: false, error: "Research goal is required" },
        { status: 400 }
      );
    }

    // Validate goal length
    const trimmed = goal.trim();
    if (trimmed.length < 20) {
      return NextResponse.json(
        {
          success: false,
          error: "Goal must be at least 20 characters long",
        },
        { status: 400 }
      );
    }

    if (trimmed.split(" ").length < 4) {
      return NextResponse.json(
        {
          success: false,
          error: "Goal must contain at least 4 words",
        },
        { status: 400 }
      );
    }

    // Analyze the goal
    const analysis: GoalAnalysis = analyzeGoal(trimmed);

    // Generate paper-like objects for suggestions
    const papers = analysis.paperSuggestions.map((suggestion, idx) => ({
      id: `goal-suggestion:${idx}`,
      metadata: {
        id: `goal-suggestion:${idx}`,
        title: suggestion.title,
        authors: [],
        abstract: suggestion.reason,
        source: "goal" as const,
        categories: [suggestion.category],
        sourceMetadata: {
          searchQuery: suggestion.searchQuery,
          relevance: suggestion.relevance,
        },
      },
      status: "suggested" as const,
      addedAt: new Date().toISOString(),
    }));

    return NextResponse.json({
      success: true,
      analysis,
      papers,
    });
  } catch (error) {
    console.error("Error analyzing research goal:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to analyze research goal",
      },
      { status: 500 }
    );
  }
}
