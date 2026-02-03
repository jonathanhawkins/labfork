/**
 * Paper Analysis API
 *
 * POST - Analyze a paper using AI
 */

import { NextRequest, NextResponse } from "next/server";
import type { Paper, PaperAnalysis } from "@/lib/papers/types";
import {
  generateAnalysisPrompt,
  parseAnalysisResponse,
  PAPER_ANALYSIS_SYSTEM_PROMPT,
} from "@/lib/papers/prompts";
import { loadDomainConfig } from "@/lib/domain/loader";
import { getPaperById, updatePaper } from "@/lib/papers/repository";

export const dynamic = "force-dynamic";

/**
 * Call Claude API for analysis
 *
 * Note: In production, this should use the actual Claude API.
 * For now, we'll simulate or use a backend proxy.
 */
async function callAnalysisAPI(
  systemPrompt: string,
  userPrompt: string
): Promise<{ success: boolean; response?: string; error?: string }> {
  // Check for Anthropic API key
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    // Return a mock analysis for development/testing
    console.warn("No ANTHROPIC_API_KEY set, returning mock analysis");
    return {
      success: true,
      response: JSON.stringify({
        relevanceScore: 75,
        relevanceReason:
          "This paper presents techniques that could be applicable to the domain's research goals.",
        techniques: [
          {
            name: "Main Technique",
            description: "The primary contribution of the paper",
            isMainContribution: true,
            relatedTo: [],
          },
        ],
        novelty:
          "The paper introduces a novel approach to the problem domain.",
        complexity: "moderate",
        complexityReason:
          "Implementation requires moderate effort with some dependencies.",
        resources: [
          {
            type: "gpu",
            name: "NVIDIA GPU",
            required: true,
            estimate: "8GB+ VRAM",
            notes: "Required for training and inference",
          },
        ],
        taskBreakdown: {
          research: {
            title: "Research paper techniques",
            description: "Study the paper and understand the key contributions",
            estimatedHours: 4,
          },
          implementation: {
            title: "Implement core algorithm",
            description:
              "Build the main algorithm based on paper specifications",
            estimatedHours: 16,
            dependencies: ["Research completion"],
          },
          evaluation: {
            title: "Evaluate implementation",
            description: "Test and benchmark against paper results",
            estimatedHours: 8,
            metrics: ["Accuracy", "Performance"],
          },
        },
        risks: ["May require dataset not publicly available"],
        relatedWork: [],
      }),
    };
  }

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 4096,
        system: systemPrompt,
        messages: [
          {
            role: "user",
            content: userPrompt,
          },
        ],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Claude API error:", errorText);
      return {
        success: false,
        error: `API error: ${response.status}`,
      };
    }

    const data = await response.json();
    const textContent = data.content?.find(
      (c: { type: string }) => c.type === "text"
    );

    if (!textContent?.text) {
      return {
        success: false,
        error: "No text response from API",
      };
    }

    return {
      success: true,
      response: textContent.text,
    };
  } catch (error) {
    console.error("Error calling Claude API:", error);
    return {
      success: false,
      error:
        error instanceof Error ? error.message : "Unknown API error",
    };
  }
}

/**
 * POST /api/papers/analyze - Analyze a paper
 *
 * Body: { paperId: string, domainSlug?: string, force?: boolean }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { paperId, domainSlug, force = false } = body;

    if (!paperId) {
      return NextResponse.json(
        { success: false, error: "Paper ID is required" },
        { status: 400 }
      );
    }

    // Get paper from repository
    const paper = await getPaperById(paperId);

    if (!paper) {
      return NextResponse.json(
        { success: false, error: "Paper not found" },
        { status: 404 }
      );
    }

    // Check if already analyzed (unless force)
    if (paper.analysis && !force) {
      return NextResponse.json({
        success: true,
        paper,
        message: "Paper already analyzed",
        fromCache: true,
      });
    }

    // Update status to analyzing
    await updatePaper(paperId, { status: "analyzing" });

    // Load domain config for context
    let domainConfig = null;
    const effectiveDomain = domainSlug || paper.domainSlug;
    if (effectiveDomain) {
      try {
        domainConfig = await loadDomainConfig(effectiveDomain);
      } catch {
        console.warn(`Could not load domain config: ${effectiveDomain}`);
      }
    }

    // Generate analysis prompt
    const userPrompt = generateAnalysisPrompt(
      {
        title: paper.metadata.title,
        authors: paper.metadata.authors,
        abstract: paper.metadata.abstract,
        categories: paper.metadata.categories,
        citationCount: paper.metadata.citationCount,
        venue: paper.metadata.venue,
      },
      domainConfig || undefined
    );

    // Call AI for analysis
    const apiResult = await callAnalysisAPI(
      PAPER_ANALYSIS_SYSTEM_PROMPT,
      userPrompt
    );

    if (!apiResult.success || !apiResult.response) {
      const errorPaper = await updatePaper(paperId, {
        status: "error",
        error: apiResult.error || "Analysis failed",
      });

      return NextResponse.json(
        {
          success: false,
          error: apiResult.error || "Analysis failed",
          paper: errorPaper,
        },
        { status: 500 }
      );
    }

    // Parse analysis response
    const parsed = parseAnalysisResponse(apiResult.response);

    if (!parsed.success || !parsed.analysis) {
      const errorPaper = await updatePaper(paperId, {
        status: "error",
        error: parsed.error || "Failed to parse analysis",
      });

      return NextResponse.json(
        {
          success: false,
          error: parsed.error || "Failed to parse analysis",
          paper: errorPaper,
          rawResponse: apiResult.response,
        },
        { status: 500 }
      );
    }

    // Create analysis object
    const analysis: PaperAnalysis = {
      ...parsed.analysis,
      analyzedAt: new Date().toISOString(),
      domainSlug: effectiveDomain,
      rawResponse: apiResult.response,
    };

    // Update paper with analysis
    const updates: Partial<Paper> = {
      analysis,
      status: "analyzed",
    };
    if (effectiveDomain) {
      updates.domainSlug = effectiveDomain;
    }

    const updatedPaper = await updatePaper(paperId, updates);

    return NextResponse.json({
      success: true,
      paper: updatedPaper,
      fromCache: false,
    });
  } catch (error) {
    console.error("Error analyzing paper:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to analyze paper",
      },
      { status: 500 }
    );
  }
}
