/**
 * Social Suggestions API - Voting
 *
 * POST /api/social/suggestions/[id]/vote - Vote on a suggestion
 * GET /api/social/suggestions/[id]/vote - Get user's vote
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getSuggestionById,
  voteSuggestion,
  getUserVote,
} from "@/lib/social/suggestions";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/social/suggestions/[id]/vote
 * Get user's vote on a suggestion
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");

    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    const suggestion = await getSuggestionById(id);
    if (!suggestion) {
      return NextResponse.json(
        { error: "Suggestion not found" },
        { status: 404 }
      );
    }

    const vote = await getUserVote(id, userId);

    return NextResponse.json({ vote });
  } catch (error) {
    console.error("Error getting vote:", error);
    return NextResponse.json(
      { error: "Failed to get vote" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/social/suggestions/[id]/vote
 * Vote on a suggestion (upvote or downvote)
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { userId, vote } = body as { userId?: string; vote?: number };

    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    if (vote !== 1 && vote !== -1) {
      return NextResponse.json(
        { error: "vote must be 1 (upvote) or -1 (downvote)" },
        { status: 400 }
      );
    }

    const suggestion = await getSuggestionById(id);
    if (!suggestion) {
      return NextResponse.json(
        { error: "Suggestion not found" },
        { status: 404 }
      );
    }

    const result = await voteSuggestion(id, userId, vote);

    if (!result) {
      return NextResponse.json(
        { error: "Failed to vote" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: result.success,
      stats: result.stats,
    });
  } catch (error) {
    console.error("Error voting:", error);
    return NextResponse.json(
      { error: "Failed to vote" },
      { status: 500 }
    );
  }
}
