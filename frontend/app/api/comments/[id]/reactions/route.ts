/**
 * Comments API - Reactions
 *
 * POST /api/comments/[id]/reactions - Add a reaction
 * DELETE /api/comments/[id]/reactions - Remove a reaction
 * GET /api/comments/[id]/reactions - Get reactions
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getCommentById,
  addReaction,
  removeReaction,
  getUserReaction,
  getReactors,
  ReactionType,
} from "@/lib/social/comments";

interface RouteParams {
  params: Promise<{ id: string }>;
}

const VALID_REACTIONS: ReactionType[] = [
  "like",
  "love",
  "insightful",
  "celebrate",
  "curious",
  "disagree",
];

/**
 * GET /api/comments/[id]/reactions
 * Get reactions for a comment
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");
    const type = searchParams.get("type") as ReactionType | null;

    const comment = await getCommentById(id);

    if (!comment) {
      return NextResponse.json(
        { error: "Comment not found" },
        { status: 404 }
      );
    }

    // Get user's reaction if userId provided
    let userReaction: ReactionType | null = null;
    if (userId) {
      userReaction = await getUserReaction(id, userId);
    }

    // Get list of reactors
    const reactors = await getReactors(id, type || undefined);

    return NextResponse.json({
      counts: comment.reactionCounts,
      userReaction,
      reactors: reactors.slice(0, 50), // Limit to 50
    });
  } catch (error) {
    console.error("Error getting reactions:", error);
    return NextResponse.json(
      { error: "Failed to get reactions" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/comments/[id]/reactions
 * Add a reaction to a comment
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { userId, type } = body as { userId?: string; type?: ReactionType };

    if (!userId || !type) {
      return NextResponse.json(
        { error: "userId and type are required" },
        { status: 400 }
      );
    }

    if (!VALID_REACTIONS.includes(type)) {
      return NextResponse.json(
        { error: `Invalid reaction type. Must be one of: ${VALID_REACTIONS.join(", ")}` },
        { status: 400 }
      );
    }

    const comment = await getCommentById(id);

    if (!comment) {
      return NextResponse.json(
        { error: "Comment not found" },
        { status: 404 }
      );
    }

    if (comment.status !== "active") {
      return NextResponse.json(
        { error: "Cannot react to this comment" },
        { status: 400 }
      );
    }

    // Remove existing reaction if different type
    const existingReaction = await getUserReaction(id, userId);
    if (existingReaction && existingReaction !== type) {
      await removeReaction(id, userId, existingReaction);
    }

    const success = await addReaction(id, userId, type);

    if (!success) {
      return NextResponse.json(
        { error: "Already reacted with this type" },
        { status: 409 }
      );
    }

    // Get updated comment
    const updated = await getCommentById(id);

    return NextResponse.json({
      success: true,
      counts: updated?.reactionCounts || comment.reactionCounts,
    });
  } catch (error) {
    console.error("Error adding reaction:", error);
    return NextResponse.json(
      { error: "Failed to add reaction" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/comments/[id]/reactions
 * Remove a reaction from a comment
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");
    const type = searchParams.get("type") as ReactionType | null;

    if (!userId || !type) {
      return NextResponse.json(
        { error: "userId and type are required" },
        { status: 400 }
      );
    }

    if (!VALID_REACTIONS.includes(type)) {
      return NextResponse.json(
        { error: `Invalid reaction type` },
        { status: 400 }
      );
    }

    const comment = await getCommentById(id);

    if (!comment) {
      return NextResponse.json(
        { error: "Comment not found" },
        { status: 404 }
      );
    }

    const success = await removeReaction(id, userId, type);

    if (!success) {
      return NextResponse.json(
        { error: "Reaction not found" },
        { status: 404 }
      );
    }

    // Get updated comment
    const updated = await getCommentById(id);

    return NextResponse.json({
      success: true,
      counts: updated?.reactionCounts || comment.reactionCounts,
    });
  } catch (error) {
    console.error("Error removing reaction:", error);
    return NextResponse.json(
      { error: "Failed to remove reaction" },
      { status: 500 }
    );
  }
}
