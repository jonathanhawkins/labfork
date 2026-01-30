/**
 * Comments API - List and Create
 *
 * GET /api/comments - List comments for an entity
 * POST /api/comments - Create a new comment
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listComments,
  createComment,
  CommentableEntity,
  CreateCommentInput,
  CommentAuthor,
} from "@/lib/social/comments";

/**
 * GET /api/comments
 * List comments for an entity
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const entityType = searchParams.get("entityType") as CommentableEntity;
    const entityId = searchParams.get("entityId");

    if (!entityType || !entityId) {
      return NextResponse.json(
        { error: "entityType and entityId are required" },
        { status: 400 }
      );
    }

    const options = {
      entityType,
      entityId,
      topLevelOnly: searchParams.get("topLevelOnly") !== "false",
      parentId: searchParams.get("parentId") || undefined,
      includeReplies: searchParams.get("includeReplies") !== "false",
      maxDepth: parseInt(searchParams.get("maxDepth") || "3", 10),
      sortBy: (searchParams.get("sortBy") as "newest" | "oldest" | "popular") || "newest",
      page: parseInt(searchParams.get("page") || "1", 10),
      limit: parseInt(searchParams.get("limit") || "20", 10),
    };

    const result = await listComments(options);

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error listing comments:", error);
    return NextResponse.json(
      { error: "Failed to list comments" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/comments
 * Create a new comment
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const { entityType, entityId, content, parentId, author } = body as {
      entityType?: CommentableEntity;
      entityId?: string;
      content?: string;
      parentId?: string;
      author?: CommentAuthor;
    };

    if (!entityType || !entityId || !content || !author) {
      return NextResponse.json(
        { error: "entityType, entityId, content, and author are required" },
        { status: 400 }
      );
    }

    // Validate entity type
    const validTypes: CommentableEntity[] = ["result", "lab", "task", "paper"];
    if (!validTypes.includes(entityType)) {
      return NextResponse.json(
        { error: `Invalid entityType. Must be one of: ${validTypes.join(", ")}` },
        { status: 400 }
      );
    }

    // Validate content length
    if (content.length > 10000) {
      return NextResponse.json(
        { error: "Comment content exceeds maximum length of 10000 characters" },
        { status: 400 }
      );
    }

    const input: CreateCommentInput = {
      entityType,
      entityId,
      content,
      parentId,
    };

    const comment = await createComment(input, author);

    return NextResponse.json(comment, { status: 201 });
  } catch (error) {
    console.error("Error creating comment:", error);
    return NextResponse.json(
      { error: "Failed to create comment" },
      { status: 500 }
    );
  }
}
