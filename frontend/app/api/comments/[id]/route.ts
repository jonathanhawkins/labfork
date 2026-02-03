/**
 * Comments API - Single Comment Operations
 *
 * GET /api/comments/[id] - Get a comment by ID
 * PATCH /api/comments/[id] - Update a comment
 * DELETE /api/comments/[id] - Delete a comment
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getCommentById,
  updateComment,
  deleteComment,
  getReplies,
  canEditComment,
  canDeleteComment,
} from "@/lib/social/comments";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/comments/[id]
 * Get a single comment by ID
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const includeReplies = searchParams.get("includeReplies") === "true";

    const comment = await getCommentById(id);

    if (!comment) {
      return NextResponse.json(
        { error: "Comment not found" },
        { status: 404 }
      );
    }

    if (comment.status !== "active") {
      return NextResponse.json(
        { error: "Comment not available" },
        { status: 410 }
      );
    }

    // Get replies if requested
    let replies: Awaited<ReturnType<typeof getReplies>> = [];
    if (includeReplies) {
      replies = await getReplies(id);
    }

    return NextResponse.json({
      ...comment,
      replies,
    });
  } catch (error) {
    console.error("Error getting comment:", error);
    return NextResponse.json(
      { error: "Failed to get comment" },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/comments/[id]
 * Update a comment
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();

    const comment = await getCommentById(id);

    if (!comment) {
      return NextResponse.json(
        { error: "Comment not found" },
        { status: 404 }
      );
    }

    if (comment.status !== "active") {
      return NextResponse.json(
        { error: "Cannot edit this comment" },
        { status: 400 }
      );
    }

    // Verify user has permission to edit
    const userId = request.headers.get("x-user-id");
    if (!canEditComment(comment, userId || undefined)) {
      return NextResponse.json(
        { error: "You do not have permission to edit this comment" },
        { status: 403 }
      );
    }

    const { content } = body;

    if (!content) {
      return NextResponse.json(
        { error: "content is required" },
        { status: 400 }
      );
    }

    if (content.length > 10000) {
      return NextResponse.json(
        { error: "Comment content exceeds maximum length" },
        { status: 400 }
      );
    }

    const updated = await updateComment(id, { content });

    if (!updated) {
      return NextResponse.json(
        { error: "Failed to update comment" },
        { status: 500 }
      );
    }

    return NextResponse.json(updated);
  } catch (error) {
    console.error("Error updating comment:", error);
    return NextResponse.json(
      { error: "Failed to update comment" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/comments/[id]
 * Delete a comment
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    const comment = await getCommentById(id);

    if (!comment) {
      return NextResponse.json(
        { error: "Comment not found" },
        { status: 404 }
      );
    }

    if (comment.status === "deleted") {
      return NextResponse.json(
        { error: "Comment already deleted" },
        { status: 410 }
      );
    }

    // Verify user has permission to delete
    const userId = request.headers.get("x-user-id");
    if (!canDeleteComment(comment, userId || undefined)) {
      return NextResponse.json(
        { error: "You do not have permission to delete this comment" },
        { status: 403 }
      );
    }

    const deleted = await deleteComment(id);

    if (!deleted) {
      return NextResponse.json(
        { error: "Failed to delete comment" },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting comment:", error);
    return NextResponse.json(
      { error: "Failed to delete comment" },
      { status: 500 }
    );
  }
}
