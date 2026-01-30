/**
 * Opportunity Board API
 *
 * GET /api/opportunities/board - Get opportunity board with filtering
 * POST /api/opportunities/board - Create a new opportunity
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createOpportunityBoard,
  createOpportunity,
  addOpportunity,
  getOpportunities,
  getOpportunitiesWithBounty,
  searchOpportunities,
  claimOpportunity,
  unclaimOpportunity,
  updateProgress,
  completeOpportunity,
  getLeaderboard,
  getBoardStats,
  DifficultyLevel,
  SignificanceLevel,
  GapType,
  BountyInfo,
} from "@/lib/meta/community";

// Singleton opportunity board
let opportunityBoard = createOpportunityBoard();

// Initialize with some demo opportunities
initializeDemoOpportunities();

function initializeDemoOpportunities() {
  const opportunities = [
    createOpportunity(
      "Multi-speaker prosody adaptation",
      "Develop a technique for adapting prosody features across multiple speakers while maintaining natural speech patterns.",
      "missing-technique",
      "advanced",
      "high",
      "Voice Cloning",
      {
        tags: ["prosody", "multi-speaker", "adaptation"],
        requiredExpertise: ["Deep Learning", "Speech Processing"],
        suggestedApproaches: [
          "Transfer learning from pre-trained models",
          "Disentangled representation learning",
        ],
      }
    ),
    createOpportunity(
      "Real-time emotion detection in TTS",
      "Create a fast emotion detection module that can be integrated into TTS pipelines with minimal latency.",
      "performance-optimization",
      "intermediate",
      "medium",
      "TTS",
      {
        tags: ["emotion", "real-time", "optimization"],
        requiredExpertise: ["Emotion Recognition", "TTS"],
        bounty: {
          amount: 500,
          currency: "USD",
          sponsor: "Voice AI Foundation",
          conditions: ["Must achieve <50ms latency", "Open source release"],
        },
      }
    ),
    createOpportunity(
      "Cross-lingual voice cloning documentation",
      "Write comprehensive documentation for cross-lingual voice cloning techniques and best practices.",
      "documentation",
      "beginner",
      "low",
      "Documentation",
      {
        tags: ["documentation", "cross-lingual", "tutorial"],
        requiredExpertise: ["Technical Writing"],
      }
    ),
  ];

  for (const opp of opportunities) {
    addOpportunity(opportunityBoard, opp);
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const page = parseInt(searchParams.get("page") || "0", 10);
    const pageSize = parseInt(searchParams.get("pageSize") || "20", 10);
    const domain = searchParams.get("domain") || undefined;
    const difficulty = searchParams.get("difficulty") as DifficultyLevel | undefined;
    const impact = searchParams.get("impact") as SignificanceLevel | undefined;
    const status = searchParams.get("status") || undefined;
    const search = searchParams.get("search") || undefined;
    const withBounty = searchParams.get("withBounty") === "true";
    const includeLeaderboard = searchParams.get("includeLeaderboard") === "true";
    const includeStats = searchParams.get("includeStats") === "true";

    let result;

    if (search) {
      const items = searchOpportunities(opportunityBoard, search, pageSize);
      result = {
        items,
        total: items.length,
        page: 0,
        pageSize,
        hasMore: false,
      };
    } else if (withBounty) {
      const items = getOpportunitiesWithBounty(opportunityBoard);
      result = {
        items: items.slice(page * pageSize, (page + 1) * pageSize),
        total: items.length,
        page,
        pageSize,
        hasMore: (page + 1) * pageSize < items.length,
      };
    } else {
      result = getOpportunities(
        opportunityBoard,
        { domain, difficulty, impact, status },
        page,
        pageSize
      );
    }

    const response: {
      opportunities: typeof result;
      leaderboard?: ReturnType<typeof getLeaderboard>;
      stats?: ReturnType<typeof getBoardStats>;
    } = { opportunities: result };

    if (includeLeaderboard) {
      response.leaderboard = getLeaderboard(opportunityBoard, "all-time", 10);
    }

    if (includeStats) {
      response.stats = getBoardStats(opportunityBoard);
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error("Error fetching opportunities:", error);
    return NextResponse.json(
      { error: "Failed to fetch opportunities" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      title,
      description,
      gapType,
      difficulty,
      impact,
      domain,
      tags,
      requiredExpertise,
      suggestedApproaches,
      bounty,
      deadline,
    } = body as {
      title: string;
      description: string;
      gapType: GapType;
      difficulty: DifficultyLevel;
      impact: SignificanceLevel;
      domain: string;
      tags?: string[];
      requiredExpertise?: string[];
      suggestedApproaches?: string[];
      bounty?: BountyInfo;
      deadline?: string;
    };

    if (!title || !description || !gapType || !difficulty || !impact || !domain) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    const opportunity = createOpportunity(
      title,
      description,
      gapType,
      difficulty,
      impact,
      domain,
      { tags, requiredExpertise, suggestedApproaches, bounty, deadline }
    );

    addOpportunity(opportunityBoard, opportunity);

    return NextResponse.json(opportunity, { status: 201 });
  } catch (error) {
    console.error("Error creating opportunity:", error);
    return NextResponse.json(
      { error: "Failed to create opportunity" },
      { status: 500 }
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const { opportunityId, action, labId, labName, progress } = body as {
      opportunityId: string;
      action: "claim" | "unclaim" | "progress" | "complete";
      labId?: string;
      labName?: string;
      progress?: number;
    };

    if (!opportunityId || !action) {
      return NextResponse.json(
        { error: "opportunityId and action are required" },
        { status: 400 }
      );
    }

    let result;

    switch (action) {
      case "claim":
        if (!labId || !labName) {
          return NextResponse.json(
            { error: "labId and labName are required for claiming" },
            { status: 400 }
          );
        }
        result = claimOpportunity(opportunityBoard, opportunityId, labId, labName);
        break;

      case "unclaim":
        if (!labId) {
          return NextResponse.json(
            { error: "labId is required for unclaiming" },
            { status: 400 }
          );
        }
        result = unclaimOpportunity(opportunityBoard, opportunityId, labId);
        break;

      case "progress":
        if (!labId || progress === undefined) {
          return NextResponse.json(
            { error: "labId and progress are required" },
            { status: 400 }
          );
        }
        result = updateProgress(opportunityBoard, opportunityId, labId, progress);
        break;

      case "complete":
        result = completeOpportunity(opportunityBoard, opportunityId);
        break;

      default:
        return NextResponse.json(
          { error: "Invalid action" },
          { status: 400 }
        );
    }

    if (!result) {
      return NextResponse.json(
        { error: "Operation failed. Check opportunity status and permissions." },
        { status: 400 }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error updating opportunity:", error);
    return NextResponse.json(
      { error: "Failed to update opportunity" },
      { status: 500 }
    );
  }
}
