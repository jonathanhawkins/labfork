/**
 * Paper Analysis API
 *
 * POST - Analyze a paper using AI
 */

import { NextRequest, NextResponse } from "next/server";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import type { Paper, PaperAnalysis } from "@/lib/papers/types";
import {
  generateAnalysisPrompt,
  parseAnalysisResponse,
  PAPER_ANALYSIS_SYSTEM_PROMPT,
} from "@/lib/papers/prompts";
import { loadDomainConfig } from "@/lib/domain/loader";

export const dynamic = "force-dynamic";

// Storage path for papers
const getStoragePath = () => {
  const projectRoot = join(process.cwd(), "..");
  const papersDir = join(projectRoot, "data", "papers");

  if (!existsSync(papersDir)) {
    mkdirSync(papersDir, { recursive: true });
  }

  return join(papersDir, "papers.json");
};

// Load papers from storage
function loadPapers(): Paper[] {
  const path = getStoragePath();
  if (!existsSync(path)) {
    return [];
  }
  try {
    const content = readFileSync(path, "utf-8");
    return JSON.parse(content);
  } catch {
    return [];
  }
}

// Save papers to storage
function savePapers(papers: Paper[]): void {
  const path = getStoragePath();
  writeFileSync(path, JSON.stringify(papers, null, 2));
}

// Find paper by ID
function findPaper(papers: Paper[], id: string): Paper | undefined {
  return papers.find((p) => p.id === id);
}

// Update paper in list
function updatePaper(papers: Paper[], updatedPaper: Paper): Paper[] {
  const index = papers.findIndex((p) => p.id === updatedPaper.id);
  if (index !== -1) {
    papers[index] = updatedPaper;
  }
  return papers;
}

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

    // Load papers
    const papers = loadPapers();
    const paper = findPaper(papers, paperId);

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
    paper.status = "analyzing";
    paper.updatedAt = new Date().toISOString();
    savePapers(updatePaper(papers, paper));

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
      paper.status = "error";
      paper.error = apiResult.error || "Analysis failed";
      paper.updatedAt = new Date().toISOString();
      savePapers(updatePaper(papers, paper));

      return NextResponse.json(
        {
          success: false,
          error: paper.error,
          paper,
        },
        { status: 500 }
      );
    }

    // Parse analysis response
    const parsed = parseAnalysisResponse(apiResult.response);

    if (!parsed.success || !parsed.analysis) {
      paper.status = "error";
      paper.error = parsed.error || "Failed to parse analysis";
      paper.updatedAt = new Date().toISOString();
      savePapers(updatePaper(papers, paper));

      return NextResponse.json(
        {
          success: false,
          error: paper.error,
          paper,
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
    paper.analysis = analysis;
    paper.status = "analyzed";
    paper.updatedAt = new Date().toISOString();
    if (effectiveDomain) {
      paper.domainSlug = effectiveDomain;
    }

    savePapers(updatePaper(papers, paper));

    return NextResponse.json({
      success: true,
      paper,
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
