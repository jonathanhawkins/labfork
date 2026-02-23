/**
 * Results API - List and Create
 *
 * GET /api/results - List results with filtering
 * POST /api/results - Create a new result
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listResults,
  createResult,
  ResultType,
  ResultVisibility,
  ResultStatus,
  CreateResultInput,
  ResultAuthor,
} from "@/lib/social/results";

/**
 * GET /api/results
 * List results with optional filters
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const options = {
      labId: searchParams.get("labId") || undefined,
      authorId: searchParams.get("authorId") || undefined,
      type: (searchParams.get("type") as ResultType) || undefined,
      visibility: (searchParams.get("visibility") as ResultVisibility) || undefined,
      status: (searchParams.get("status") as ResultStatus) || undefined,
      tags: searchParams.get("tags")?.split(",").filter(Boolean) || undefined,
      search: searchParams.get("search") || undefined,
      sortBy: (searchParams.get("sortBy") as "likes" | "comments" | "views" | "created" | "updated") || undefined,
      sortDir: (searchParams.get("sortDir") as "asc" | "desc") || undefined,
      page: parseInt(searchParams.get("page") || "1", 10),
      limit: parseInt(searchParams.get("limit") || "20", 10),
    };

    const results = await listResults(options);

    return NextResponse.json(results);
  } catch (error) {
    console.error("Error listing results:", error);
    return NextResponse.json(
      { error: "Failed to list results" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/results
 * Create a new result
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    const { type, title, description, labId, author } = body as {
      type?: ResultType;
      title?: string;
      description?: string;
      labId?: string;
      author?: ResultAuthor;
    } & Partial<CreateResultInput>;

    if (!type || !title || !description || !labId || !author) {
      return NextResponse.json(
        { error: "Missing required fields: type, title, description, labId, author" },
        { status: 400 }
      );
    }

    // Validate type
    const validTypes: ResultType[] = ["model", "demo", "finding", "comparison", "dataset", "paper"];
    if (!validTypes.includes(type)) {
      return NextResponse.json(
        { error: `Invalid type. Must be one of: ${validTypes.join(", ")}` },
        { status: 400 }
      );
    }

    const input: CreateResultInput = {
      type,
      title,
      description,
      labId,
      content: body.content,
      taskId: body.taskId,
      visibility: body.visibility,
      tags: body.tags,
      metadata: body.metadata,
    };

    const result = await createResult(input, author);

    return NextResponse.json(result, { status: 201 });
  } catch (error) {
    console.error("Error creating result:", error);
    return NextResponse.json(
      { error: "Failed to create result" },
      { status: 500 }
    );
  }
}
