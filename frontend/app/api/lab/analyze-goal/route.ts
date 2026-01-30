import { NextRequest, NextResponse } from "next/server";
import {
  GOAL_ANALYSIS_SYSTEM_PROMPT,
  generateGoalPrompt,
  parseGoalAnalysisResponse,
  MOCK_GOAL_ANALYSIS,
} from "@/lib/lab-wizard/goal-analyzer";

/**
 * Goal Analysis API
 *
 * POST /api/lab/analyze-goal - Analyze research goal with AI
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { goalText, preferredDomain, hardwareVram, generateTasks, domainSlug, papers } = body;

    if (!goalText?.trim()) {
      return NextResponse.json(
        { success: false, error: "Goal text is required" },
        { status: 400 }
      );
    }

    // Check for Claude API key
    const apiKey = process.env.ANTHROPIC_API_KEY;

    if (!apiKey) {
      // Return mock analysis if no API key
      console.log("No ANTHROPIC_API_KEY, returning mock goal analysis");

      // Customize mock based on goal text
      const mockAnalysis = { ...MOCK_GOAL_ANALYSIS };

      // Simple domain matching
      const goalLower = goalText.toLowerCase();
      if (goalLower.includes("voice") || goalLower.includes("speech") || goalLower.includes("audio")) {
        mockAnalysis.suggestedDomain = "voice-clone";
        mockAnalysis.domainName = "Voice Cloning Research";
      } else if (goalLower.includes("trading") || goalLower.includes("finance") || goalLower.includes("stock")) {
        mockAnalysis.suggestedDomain = "quant-trading";
        mockAnalysis.domainName = "Quantitative Trading";
        mockAnalysis.arxivCategories = ["q-fin.ST", "cs.LG"];
      } else if (goalLower.includes("robot") || goalLower.includes("control") || goalLower.includes("motion")) {
        mockAnalysis.suggestedDomain = "robotics";
        mockAnalysis.domainName = "Robotics Research";
        mockAnalysis.arxivCategories = ["cs.RO", "cs.AI"];
      } else if (goalLower.includes("drug") || goalLower.includes("molecule") || goalLower.includes("protein")) {
        mockAnalysis.suggestedDomain = "biotech";
        mockAnalysis.domainName = "Biotech & Drug Discovery";
        mockAnalysis.arxivCategories = ["q-bio.BM", "cs.LG"];
      }

      return NextResponse.json({
        success: true,
        analysis: mockAnalysis,
        mock: true,
      });
    }

    // Call Claude API for analysis
    try {
      const prompt = generateGoalPrompt(goalText, { preferredDomain, hardwareVram });

      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 2000,
          system: GOAL_ANALYSIS_SYSTEM_PROMPT,
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error("Claude API error:", errorData);

        // Fall back to mock on API error
        return NextResponse.json({
          success: true,
          analysis: MOCK_GOAL_ANALYSIS,
          mock: true,
          warning: "Using mock analysis due to API error",
        });
      }

      const data = await response.json();
      const content = data.content?.[0]?.text;

      if (!content) {
        return NextResponse.json({
          success: true,
          analysis: MOCK_GOAL_ANALYSIS,
          mock: true,
          warning: "No response from API",
        });
      }

      // Parse the response
      const result = parseGoalAnalysisResponse(content);

      if (!result.success) {
        return NextResponse.json({
          success: true,
          analysis: MOCK_GOAL_ANALYSIS,
          mock: true,
          warning: result.error,
        });
      }

      return NextResponse.json({
        success: true,
        analysis: result.analysis,
      });
    } catch (apiError) {
      console.error("Goal analysis API error:", apiError);

      // Return mock on any error
      return NextResponse.json({
        success: true,
        analysis: MOCK_GOAL_ANALYSIS,
        mock: true,
        warning: "API error, using mock analysis",
      });
    }
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Goal analysis failed",
      },
      { status: 500 }
    );
  }
}
