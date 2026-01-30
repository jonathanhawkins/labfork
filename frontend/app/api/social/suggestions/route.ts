/**
 * Social Suggestions API - List and Create
 *
 * GET /api/social/suggestions - List suggestions
 * POST /api/social/suggestions - Create a suggestion
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listSuggestions,
  createSuggestion,
  SuggestionCategory,
  SuggestionStatus,
  SuggestionPriority,
  CreateSuggestionInput,
  SuggestionAuthor,
} from "@/lib/social/suggestions";

/**
 * GET /api/social/suggestions
 * List suggestions with optional filters
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const statusParam = searchParams.get("status");
    let status: SuggestionStatus | SuggestionStatus[] | undefined;
    if (statusParam) {
      status = statusParam.includes(",")
        ? (statusParam.split(",") as SuggestionStatus[])
        : (statusParam as SuggestionStatus);
    }

    const options = {
      labId: searchParams.get("labId") || undefined,
      authorId: searchParams.get("authorId") || undefined,
      category: (searchParams.get("category") as SuggestionCategory) || undefined,
      status,
      priority: (searchParams.get("priority") as SuggestionPriority) || undefined,
      search: searchParams.get("search") || undefined,
      sortBy: (searchParams.get("sortBy") as
        | "votes"
        | "comments"
        | "newest"
        | "oldest"
        | "priority") || undefined,
      page: parseInt(searchParams.get("page") || "1", 10),
      limit: parseInt(searchParams.get("limit") || "20", 10),
    };

    const result = await listSuggestions(options);

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error listing suggestions:", error);
    return NextResponse.json(
      { error: "Failed to list suggestions" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/social/suggestions
 * Create a new suggestion
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const { labId, title, description, category, author } = body as {
      labId?: string;
      title?: string;
      description?: string;
      category?: SuggestionCategory;
      author?: SuggestionAuthor;
    } & Partial<CreateSuggestionInput>;

    if (!labId || !title || !description || !category || !author) {
      return NextResponse.json(
        { error: "labId, title, description, category, and author are required" },
        { status: 400 }
      );
    }

    // Validate category
    const validCategories: SuggestionCategory[] = [
      "research_direction",
      "improvement",
      "bug_report",
      "feature_request",
      "question",
      "collaboration",
    ];
    if (!validCategories.includes(category)) {
      return NextResponse.json(
        { error: `Invalid category. Must be one of: ${validCategories.join(", ")}` },
        { status: 400 }
      );
    }

    const input: CreateSuggestionInput = {
      labId,
      title,
      description,
      category,
      taskId: body.taskId,
      resultId: body.resultId,
      priority: body.priority,
      tags: body.tags,
    };

    const suggestion = await createSuggestion(input, author);

    return NextResponse.json(suggestion, { status: 201 });
  } catch (error) {
    console.error("Error creating suggestion:", error);
    return NextResponse.json(
      { error: "Failed to create suggestion" },
      { status: 500 }
    );
  }
}
